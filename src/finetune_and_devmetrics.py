#!/usr/bin/env python3
"""
finetune_and_devmetrics.py
-------------------------
Fine‑tune a pretrained auto‑encoder on each machine’s **ADD‑TRAIN** normals,
run inference on the **DEV** test clips, compute AUC / pAUC / PRF metrics, and
print a ready‑to‑paste YAML snippet for the meta file.

✔ **Bug‑fixed**
  • Path/array mismatch removed – `extract_logmel` now gets a filepath.
  • Guaranteed `float32` tensors throughout to avoid Float/Double errors.
  • Robust threshold lookup with a fallback if `train.gamma_percentile` is
    missing (uses `threshold.percentile`).
  • CSVs now include headers for convenience.

⚙️ **Config mapping**
  - Uses `cfg["audio"]`, `cfg["train"]`, and `cfg["model"]` exactly as your
    YAML defines them (no more hard‑coded defaults).
  - The script will raise a clear error if the pretrained weights are absent.

📈 **Metrics**: source vs target AUC, pAUC@0.1 FPR, and per‑domain
precision/recall/F1. Everything is collected into `dev_results/` and then
summarised in YAML.
"""

import os
import glob
import math
import pathlib
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from scipy.stats import gamma

from utils import extract_logmel, make_windows

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

class AE(nn.Module):
    """Symmetric 3‑layer encoder/decoder MLP auto‑encoder."""

    def __init__(self, dim: int, hidden: int, bottleneck: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.BatchNorm1d(bottleneck), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) → (B, D)
        return self.dec(self.enc(x))


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load(open(path, "r"))


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def windows_from_wavs(files: list[str], audio_cfg: dict) -> np.ndarray:
    """Load every *.wav file, convert to context‑windowed log‑mels, concatenate."""
    all_windows = []
    for f in files:
        mel = extract_logmel(
            f,
            sr=audio_cfg["sample_rate"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            n_mels=audio_cfg["n_mels"],
        )
        all_windows.append(make_windows(mel, audio_cfg["context"]))
    return np.concatenate(all_windows, axis=0).astype(np.float32)


def gamma_threshold(errors: np.ndarray, percentile: float) -> float:
    """Return the Gamma‑fit percentile threshold."""
    shape, loc, scale = gamma.fit(errors, floc=0)
    return float(gamma.ppf(percentile / 100.0, shape, loc=loc, scale=scale))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_cfg()
    audio_cfg = cfg["audio"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    D = audio_cfg["n_mels"] * audio_cfg["context"]

    dev_root = pathlib.Path(cfg["paths"]["dev_root"])
    eval_root = pathlib.Path(cfg["paths"]["eval_root"])
    out_root = pathlib.Path("dev_results")
    out_root.mkdir(parents=True, exist_ok=True)

    # ── pretrained weights ──────────────────────────────────────────────
    weights_path = pathlib.Path(cfg["paths"]["pretrained_dir"]) / "ae_dev.pt"
    if not weights_path.exists():
        raise FileNotFoundError("Pretrained weights not found – run pretrain_dev.py first.")

    device = get_device(cfg.get("device", "auto"))
    print("Device:", device)

    yaml_lines = ["  development_dataset:"]

    machines = sorted([d.name for d in dev_root.iterdir() if d.is_dir()])
    for machine in machines:
        print(f"\n=== {machine} ===")

        # ---------------- model init ----------------
        ae = AE(D, model_cfg["hidden"], model_cfg["bottleneck"]).to(device)
        ae.load_state_dict(torch.load(weights_path, map_location=device))

        # ---------------- fine‑tune ------------------
        train_wavs = glob.glob(str(eval_root / machine / "train" / "**" / "*.wav"), recursive=True)
        X = windows_from_wavs(train_wavs, audio_cfg)
        loader = torch.utils.data.DataLoader(
            torch.from_numpy(X),
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        opt = optim.Adam(ae.parameters(), lr=train_cfg["lr"])
        ae.train()
        for epoch in range(train_cfg["epochs_finetune"]):
            pbar = tqdm(loader, desc=f"FT {machine} {epoch+1}/{train_cfg['epochs_finetune']}", leave=False)
            for batch in pbar:
                batch = batch.to(device, dtype=torch.float32)
                loss = ((ae(batch) - batch) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---------------- threshold ------------------
        with torch.no_grad():
            ae.eval()
            errs = []
            for batch in loader:
                batch = batch.to(device, dtype=torch.float32)
                errs.append(((ae(batch) - batch) ** 2).mean(1).cpu().numpy())
        gamma_pct = train_cfg.get("gamma_percentile", cfg.get("threshold", {}).get("percentile", 90))
        thr = gamma_threshold(np.concatenate(errs), gamma_pct)
        print(f"threshold={thr:.3f}")

        # ---------------- inference ------------------
        test_wavs = sorted(glob.glob(str(dev_root / machine / "test" / "*.wav")))
        y_true, y_score, domains = [], [], []

        score_csv = out_root / f"anomaly_score_{machine}_section_00_test.csv"
        decide_csv = out_root / f"decision_result_{machine}_section_00_test.csv"
        with score_csv.open("w") as sf, decide_csv.open("w") as df:
            sf.write("file,anomaly_score\n")
            df.write("file,is_anomaly\n")
            for wf in tqdm(test_wavs, desc="Infer"):
                fname = os.path.basename(wf)
                label = 0 if "normal" in fname else 1
                domain = "source" if "_source_" in fname else "target"

                mel = extract_logmel(
                    wf,
                    sr=audio_cfg["sample_rate"],
                    n_fft=audio_cfg["n_fft"],
                    hop_length=audio_cfg["hop_length"],
                    n_mels=audio_cfg["n_mels"],
                )
                W = make_windows(mel, audio_cfg["context"])
                W_t = torch.from_numpy(W).to(device, dtype=torch.float32)
                with torch.no_grad():
                    err = ((ae(W_t) - W_t) ** 2).mean(1).mean().item()

                sf.write(f"{fname},{err:.6f}\n")
                df.write(f"{fname},{int(err > thr)}\n")

                y_true.append(label)
                y_score.append(err)
                domains.append(domain)

        # ---------------- metrics --------------------
        y_true = np.array(y_true);
        y_score = np.array(y_score);
        domains = np.array(domains)

        def auc_domain(dom):
            mask = domains == dom
            return roc_auc_score(y_true[mask], y_score[mask]) * 100 if mask.any() and y_true[mask].sum() else math.nan

        auc_src = auc_domain("source")
        auc_tgt = auc_domain("target")

        fpr, tpr, _ = roc_curve(y_true, y_score)
        mask = fpr <= 0.1
        pauc = (np.trapz(tpr[mask], fpr[mask]) / 0.1) * 100

        y_pred = (y_score > thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary")

        def prf(dom):
            m = domains == dom
            return precision_recall_fscore_support(y_true[m], y_pred[m], pos_label=1, average="binary", zero_division=0)

        pr_src = prf("source")
        pr_tgt = prf("target")

        # ---------------- YAML lines -----------------
        yaml_lines.extend([
            f"    {machine}:",
            f"      auc_source: {auc_src:.2f}",
            f"      auc_target: {auc_tgt:.2f}",
            f"      pauc: {pauc:.2f}",
            f"      precision_source: {pr_src[0]:.3f}",
            f"      precision_target: {pr_tgt[0]:.3f}",
            f"      recall_source: {pr_src[1]:.3f}",
            f"      recall_target: {pr_tgt[1]:.3f}",
            f"      f1_source: {pr_src[2]:.3f}",
            f"      f1_target: {pr_tgt[2]:.3f}",
        ])

        print(f"→ completed {machine}")

    # -----------------------------------------------------------------
    print("\n\n==== YAML block for meta file ====")
    print("\n".join(yaml_lines))
    print("==================================")


if __name__ == "__main__":
    main()
