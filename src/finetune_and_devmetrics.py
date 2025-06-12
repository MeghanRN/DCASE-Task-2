#!/usr/bin/env python3
"""
finetune_and_devmetrics.py
-------------------------
Fine‑tune a pretrained auto‑encoder on each machine’s **ADD‑TRAIN** normals
(or fall back to the DEV‑train normals if no ADD‑TRAIN exists), run inference
on the DEV test clips, compute AUC / pAUC / precision/recall/F1 metrics, and
print a ready‑to‑paste YAML snippet for your meta file.

Key features
~~~~~~~~~~~~
* Graceful skips when a machine has no training or test data.
* Guards against Float↔Double dtype mismatches.
* Robust percentile threshold with fallback to `threshold.percentile`.
* CSVs include headers for quick spreadsheet inspection.
"""

from __future__ import annotations

import glob
import math
import os
import pathlib
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.stats import gamma
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from utils import extract_logmel, make_windows

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class AE(nn.Module):
    """Simple symmetric MLP auto‑encoder."""

    def __init__(self, dim: int, hidden: int, bottleneck: int) -> None:
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
# Helper functions
# -----------------------------------------------------------------------------

def load_cfg(path: str = "config.yaml") -> dict:
    return yaml.safe_load(open(path, "r"))


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def windows_from_wavs(files: List[str], audio_cfg: dict) -> np.ndarray:
    """Load WAVs, build context‑window log‑mel feature matrix."""
    if not files:
        raise ValueError("windows_from_wavs called with an empty file list")
    all_windows: List[np.ndarray] = []
    for wav_path in files:
        mel = extract_logmel(
            wav_path,
            sr=audio_cfg["sample_rate"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            n_mels=audio_cfg["n_mels"],
        )
        all_windows.append(make_windows(mel, audio_cfg["context"]))
    return np.concatenate(all_windows, axis=0).astype(np.float32)


def gamma_threshold(errors: np.ndarray, percentile: float) -> float:
    """Gamma‑fit percentile threshold."""
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

    dim = audio_cfg["n_mels"] * audio_cfg["context"]

    dev_root = pathlib.Path(cfg["paths"]["dev_root"])
    eval_root = pathlib.Path(cfg["paths"]["eval_root"])
    out_root = pathlib.Path("dev_results")
    out_root.mkdir(parents=True, exist_ok=True)

    # ── pretrained weights ─────────────────────────────────────────────
    weights_path = pathlib.Path(cfg["paths"]["pretrained_dir"]) / "ae_dev.pt"
    if not weights_path.exists():
        raise FileNotFoundError("Pretrained weights not found – run pretrain_dev.py first.")

    device = get_device(cfg.get("device", "auto"))
    print("Device:", device)

    yaml_lines: List[str] = ["  development_dataset:"]

    machines = sorted([d.name for d in dev_root.iterdir() if d.is_dir()])
    for machine in machines:
        print(f"\n=== {machine} ===")

        # ---------------- gather training WAVs ----------------
        train_wavs = glob.glob(str(eval_root / machine / "train" / "**" / "*.wav"), recursive=True)
        if not train_wavs:
            train_wavs = glob.glob(str(dev_root / machine / "train" / "**" / "*.wav"), recursive=True)
            if train_wavs:
                print("  ↳ no ADD‑TRAIN found, using DEV train normals instead")
        if not train_wavs:
            print(f"  ↳ no training data for {machine}; skipping machine.")
            continue

        # ---------------- model init --------------------------
        ae = AE(dim, model_cfg["hidden"], model_cfg["bottleneck"]).to(device)
        ae.load_state_dict(torch.load(weights_path, map_location=device))

        # ---------------- fine‑tune ---------------------------
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
            pbar = tqdm(loader, desc=f"FT {machine} {epoch + 1}/{train_cfg['epochs_finetune']}", leave=False)
            for batch in pbar:
                batch = batch.to(device, dtype=torch.float32)
                loss = ((ae(batch) - batch) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---------------- threshold --------------------------
        with torch.no_grad():
            ae.eval()
            errs = []
            for batch in loader:
                batch = batch.to(device, dtype=torch.float32)
                errs.append(((ae(batch) - batch) ** 2).mean(1).cpu().numpy())
        gamma_pct = train_cfg.get("gamma_percentile", cfg.get("threshold", {}).get("percentile", 90))
        thr = gamma_threshold(np.concatenate(errs), gamma_pct)
        print(f"threshold={thr:.3f}")

        # ---------------- inference --------------------------
        test_wavs = sorted(glob.glob(str(dev_root / machine / "test" / "*.wav")))
        if not test_wavs:
            print(f"  ↳ no DEV test clips for {machine}; skipping.")
            continue

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

        # ---------------- metrics ----------------------------
        y_true_np = np.array(y_true)
        y_score_np = np.array(y_score)
        domains_np = np.array(domains)

        def auc_for(dom: str) -> float:
            mask = domains_np == dom
            if not mask.any() or y_true_np[mask].sum() == 0:
                return float("nan")
            return roc_auc_score(y_true_np[mask], y_score_np[mask]) * 100

        auc_src = auc_for("source")
        auc_tgt = auc_for("target")

        fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
        pauc_mask = fpr <= 0.1
        pauc = (np.trapz(tpr[p
