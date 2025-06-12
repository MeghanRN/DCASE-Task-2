#!/usr/bin/env python3
"""
finetune_and_devmetrics.py
--------------------------
• Fine-tune a pretrained auto-encoder on each machine’s *ADD-TRAIN* normals.
• Run inference on the *DEV* test clips.
• Compute AUC (source/target), pAUC@10 % FPR, and precision/recall/F1.
• Emit a YAML snippet ready for the DCASE meta file.
"""

from __future__ import annotations

import os
import glob
import math
import pathlib
import yaml
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gamma
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)

from utils import extract_logmel, make_windows

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class AE(nn.Module):
    """Symmetric 2-hidden-layer encoder/decoder MLP auto-encoder."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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


def windows_from_wavs(files: List[str], audio_cfg: dict) -> np.ndarray:
    """Load WAVs, convert to mel-windows, concatenate (float32)."""
    if not files:
        return np.empty((0, audio_cfg["n_mels"] * audio_cfg["context"]), dtype=np.float32)

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
    """Fit Gamma and return percentile threshold (fallback to raw pct if fit fails)."""
    try:
        shape, loc, scale = gamma.fit(errors, floc=0)
        return float(gamma.ppf(percentile / 100.0, shape, loc=loc, scale=scale))
    except Exception:  # pragma: no cover
        print("⚠️  Gamma fit failed; using empirical percentile.")
        return float(np.percentile(errors, percentile))


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for the given slice (mask=None → global)."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    return p, r, f


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


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

    # pretrained weights
    weights_path = pathlib.Path(cfg["paths"]["pretrained_dir"]) / "ae_dev.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            "Pretrained weights not found – run pretrain_dev.py first."
        )

    device = get_device(cfg.get("device", "auto"))
    print("Device:", device)

    yaml_lines = ["  development_dataset:"]

    machines = sorted([d.name for d in dev_root.iterdir() if d.is_dir()])
    for machine in machines:
        print(f"\n=== {machine} ===")

        # ── Collect training WAVs (eval-train first, else dev-train) ──
        train_wavs = glob.glob(
            str(eval_root / machine / "train" / "**" / "*.wav"), recursive=True
        )
        if not train_wavs:
            train_wavs = glob.glob(
                str(dev_root / machine / "train" / "*.wav"), recursive=True
            )
        if not train_wavs:
            print(f"⚠️  No training WAVs for {machine}; skipping.")
            continue

        # ── Prepare data ─────────────────────────────────────────────
        X = windows_from_wavs(train_wavs, audio_cfg)
        if X.size == 0:
            print(f"⚠️  Training data empty for {machine}; skipping.")
            continue
        loader = torch.utils.data.DataLoader(
            torch.from_numpy(X),
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        # ── Model ────────────────────────────────────────────────────
        ae = AE(D, model_cfg["hidden"], model_cfg["bottleneck"]).to(device)
        ae.load_state_dict(torch.load(weights_path, map_location=device))

        opt = optim.Adam(ae.parameters(), lr=train_cfg["lr"])
        ae.train()
        for epoch in range(train_cfg["epochs_finetune"]):
            pbar = tqdm(
                loader,
                desc=f"FT {machine} {epoch+1}/{train_cfg['epochs_finetune']}",
                leave=False,
            )
            for batch in pbar:
                batch = batch.to(device, dtype=torch.float32)
                loss = ((ae(batch) - batch) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Threshold ───────────────────────────────────────────────
        ae.eval()
        with torch.no_grad():
            errs = []
            for batch in loader:
                batch = batch.to(device, dtype=torch.float32)
                errs.append(((ae(batch) - batch) ** 2).mean(1).cpu().numpy())
        gamma_pct = train_cfg.get(
            "gamma_percentile", cfg.get("threshold", {}).get("percentile", 90)
        )
        thr = gamma_threshold(np.concatenate(errs), gamma_pct)
        print(f"threshold={thr:.3f}")

        # ── Inference ───────────────────────────────────────────────
        test_wavs = sorted(glob.glob(str(dev_root / machine / "test" / "*.wav")))

        y_true, y_score, domain_tags = [], [], []

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
                domain_tags.append(domain)

        # ── Metrics ─────────────────────────────────────────────────
        y_true_np = np.array(y_true, dtype=int)
        y_score_np = np.array(y_score, dtype=float)
        domain_np = np.array(domain_tags, dtype="<U6")

        def auc_for(domain: str) -> float:
            mask = domain_np == domain
            if mask.sum() == 0 or y_true_np[mask].sum() == 0:
                return math.nan
            return 100.0 * roc_auc_score(y_true_np[mask], y_score_np[mask])

        auc_src = auc_for("source")
        auc_tgt = auc_for("target")

        # pAUC up to 10 % FPR
        fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
        mask = fpr <= 0.10
        pauc = 100.0 * np.trapz(tpr[mask], fpr[mask]) / 0.10

        # Global PRF
        y_pred_np = (y_score_np > thr).astype(int)
        prec, rec, f1 = precision_recall_f1(y_true_np, y_pred_np)

        # Per-domain PRF
        pr_src = precision_recall_f1(y_true_np, y_pred_np, domain_np == "source")
        pr_tgt = precision_recall_f1(y_true_np, y_pred_np, domain_np == "target")

        # ── YAML output lines ───────────────────────────────────────
        yaml_lines.extend(
            [
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
            ]
        )

        print(f"→ completed {machine}")

    # ────────────────────────────────────────────────────────────────
    print("\n\n==== YAML block for meta file ====")
    print("\n".join(yaml_lines))
    print("==================================")


if __name__ == "__main__":
    main()
