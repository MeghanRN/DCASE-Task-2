#!/usr/bin/env python3
"""
finetune_and_infer.py ─ Fine‑tune a pretrained auto‑encoder per machine and
generate anomaly‑score / binary‑decision CSVs for DCASE Task 2.

Key fixes vs. the original draft
--------------------------------
1. **Path vs waveform bug**: we now pass the audio *filepath* to
   `utils.extract_logmel` instead of a NumPy array.
2. **dtype alignment**: all tensors are promoted to `float32` before they hit
   the network to avoid `Float vs Double` runtime errors.
3. **Config hygiene**: handles the case where the Gamma percentile is stored
   under either `train.gamma_percentile` *or* `threshold.percentile`.
4. **BatchNorm stability**: BatchNorm layers are put in *eval* mode during
   fine‑tuning; this avoids noisy statistics when each machine has only a few
   minutes of data.

Everything else (hyper‑parameters, directory layout) is read from the same
`config.yaml` used by pre‑training.
"""

from __future__ import annotations

import glob
import os
import pathlib
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gamma as sps_gamma
from tqdm import tqdm

import utils as ut

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def gamma_threshold(errors: np.ndarray, percentile: float) -> float:
    """Fit a 3‑parameter Gamma distribution to *errors* and return the
    *percentile*‑th cutoff.  `floc` is fixed at 0 for numerical stability."""
    shape, loc, scale = sps_gamma.fit(errors, floc=0)
    return float(sps_gamma.ppf(percentile / 100.0, shape, loc=loc, scale=scale))


def dataset_windows(wav_files: List[str], audio_cfg: dict) -> np.ndarray:
    """Return a 2‑D (n_windows × D) float32 matrix with all context windows from
    *wav_files*.  Each window is flatten(log‑Mel[frame:frame+context])."""
    windows: List[np.ndarray] = []
    for wf in wav_files:
        M = ut.extract_logmel(
            wf,
            sr=audio_cfg["sample_rate"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            n_mels=audio_cfg["n_mels"],
        )
        windows.append(ut.make_windows(M, audio_cfg["context"]))
    return np.concatenate(windows, axis=0).astype(np.float32)


class AE(nn.Module):
    """A simple fully connected auto‑encoder used for every machine."""

    def __init__(self, D: int, H: int, B: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(D, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Linear(H, B),
            nn.BatchNorm1d(B),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(B, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Linear(H, D),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.dec(self.enc(x))


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    cfg = ut.load_config()
    AUDIO, TRAIN = cfg["audio"], cfg["train"]
    D = AUDIO["n_mels"] * AUDIO["context"]

    gamma_pct = TRAIN.get(
        "gamma_percentile",
        cfg.get("threshold", {}).get("percentile", 90),
    )

    # --- device selection ----------------------------------------------------
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device:", device)

    # --- where results will be written --------------------------------------
    out_root = pathlib.Path(cfg["paths"]["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    # --- pretrained weights (shared across machines) ------------------------
    w_pre = pathlib.Path(cfg["paths"]["pretrained_dir"]) / "ae_dev.pt"
    if not w_pre.exists():
        raise RuntimeError("Pre‑train weights missing. Run pretrain_dev.py first!")

    eval_root = pathlib.Path(cfg["paths"]["eval_root"])

    # --- iterate over machines ---------------------------------------------
    for machine in sorted(os.listdir(eval_root)):
        mdir = eval_root / machine
        print(f"\n=== {machine} ===")

        # Build model and load shared weights
        ae = AE(D, H=cfg["model"]["hidden"], B=cfg["model"]["bottleneck"]).to(device)
        ae.load_state_dict(torch.load(w_pre, map_location=device), strict=True)

        # Freeze BatchNorm running stats for tiny datasets
        ae.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm1d) else None)

        # ------------------------------------------------ Fine‑tune ----------
        train_wavs = glob.glob(str(mdir / "train/**/*.wav"), recursive=True)
        X = dataset_windows(train_wavs, AUDIO)  # float32
        loader = torch.utils.data.DataLoader(
            torch.from_numpy(X),
            batch_size=TRAIN["batch_size"],
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        opt = optim.Adam(ae.parameters(), lr=TRAIN["lr"])
        ae.train()
        epochs_ft = TRAIN.get("epochs_finetune", TRAIN.get("epochs", 1))
        for ep in range(epochs_ft):
            bar = tqdm(loader, desc=f"FT {machine} {ep + 1}/{epochs_ft}")
            for batch in bar:
                batch = batch.to(device, dtype=torch.float32)
                loss = ((ae(batch) - batch) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                bar.set_postfix(loss=f"{loss.item():.4f}")

        # ------------------------------------------- Threshold estimation ----
        ae.eval()
        with torch.no_grad():
            errs = []
            for batch in loader:
                batch = batch.to(device, dtype=torch.float32)
                errs.append(((ae(batch) - batch) ** 2).mean(1).cpu().numpy())
            thr = gamma_threshold(np.concatenate(errs), gamma_pct)
        print(f"threshold={thr:.3f}")

        # ------------------------------------------------ Inference ----------
        test_wavs = sorted(glob.glob(str(mdir / "test/*.wav")))
        score_csv = out_root / f"anomaly_score_{machine}_section_00_test.csv"
        decision_csv = out_root / f"decision_result_{machine}_section_00_test.csv"

        with score_csv.open("w") as sf, decision_csv.open("w") as df:
            sf.write("file,anomaly_score\n")
            df.write("file,is_anomaly\n")

            for wf in tqdm(test_wavs, desc="Infer"):
                M = ut.extract_logmel(
                    wf,
                    sr=AUDIO["sample_rate"],
                    n_fft=AUDIO["n_fft"],
                    hop_length=AUDIO["hop_length"],
                    n_mels=AUDIO["n_mels"],
                )
                W = ut.make_windows(M, AUDIO["context"])
                W = torch.from_numpy(W).to(device, dtype=torch.float32)

                with torch.no_grad():
                    e = ((ae(W) - W) ** 2).mean(1).mean().item()

                sf.write(f"{os.path.basename(wf)},{e:.6f}\n")
                df.write(f"{os.path.basename(wf)},{int(e > thr)}\n")

        print("✓ wrote CSVs for", machine)

    print("\nAll machines done → results in", out_root)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
