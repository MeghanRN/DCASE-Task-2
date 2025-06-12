#!/usr/bin/env python3
# ------------------------------------------------------------
# pretrain_dev.py
# Pre-trains an AE on *all normal* dev-set windows and
# stores the weights + reconstruction-error histogram +
# a Gamma-percentile threshold.
# ------------------------------------------------------------
import os, glob, yaml, pathlib, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from utils import extract_logmel, make_windows, fit_gamma_threshold   # your utils.py

# ----------------- config helpers ---------------------------
def load_cfg(path="config.yaml"):
    return yaml.safe_load(open(path, "r"))

def get_device(cfg):
    if cfg.get("device", "auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available()
                            else "cpu")
    return torch.device(cfg["device"])

# ----------------- dataset ----------------------------------
class DevWindows(IterableDataset):
    """
    Streams *all* context-windows (640-D) from every
    <machine>/train/*.wav in dev_root.
    """
    def __init__(self, dev_root, cfg):
        self.wavs = glob.glob(f"{dev_root}/*/train/*.wav")
        self.cfg  = cfg

    def __iter__(self):
        for wav_f in self.wavs:
            y, sr = _read_wav(wav_f)
            M = extract_logmel(
                    y, sr,
                    n_fft      = self.cfg["feature"]["n_fft"],
                    hop_length = self.cfg["feature"]["hop_length"],
                    n_mels     = self.cfg["feature"]["n_mels"])
            W = make_windows(M, self.cfg["feature"]["context"])
            for w in W:
                yield torch.from_numpy(w).float()

def _read_wav(path):
    import scipy.io.wavfile as wav
    sr, y = wav.read(path)
    return (y.astype(np.float32)/32768.0, sr)

# ----------------- model ------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, D, h=128, b=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, b), nn.BatchNorm1d(b), nn.ReLU(),
            nn.Linear(b, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, D)
        )
    def forward(self, x): return self.net(x)

# ----------------- main -------------------------------------
def main():
    cfg      = load_cfg()
    dev_root = cfg["paths"]["dev_root"]     # <-- SAME KEY AS finetune_and_infer.py
    out_dir  = pathlib.Path(cfg["paths"]["pretrained_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    D = cfg["feature"]["n_mels"] * cfg["feature"]["context"]
    device = get_device(cfg)
    print("Device:", device)

    # dataset & loader -------------------------------------------------
    ds      = DevWindows(dev_root, cfg)
    loader  = DataLoader(ds,
                         batch_size = cfg["train"]["batch_size"],
                         shuffle    = False,
                         num_workers=2)
    ae      = AutoEncoder(D,
                          h = cfg["model"]["hidden"],
                          b = cfg["model"]["bottleneck"]).to(device)
    opt     = torch.optim.Adam(ae.parameters(), lr=cfg["train"]["lr"])
    mse     = nn.MSELoss()

    # training ---------------------------------------------------------
    n_epochs = cfg["train"]["epochs_pretrain"]
    print(f"Pre-training for {n_epochs} epoch(s) …")
    for ep in range(n_epochs):
        bar = tqdm(loader, desc=f"Epoch {ep+1}/{n_epochs}")
        for batch in bar:
            batch = batch.to(device)
            opt.zero_grad()
            loss = mse(ae(batch), batch)
            loss.backward()
            opt.step()
            bar.set_postfix(loss=loss.item())

    # collect reconstruction errors -----------------------------------
    print("Collecting reconstruction errors …")
    errs = []
    ae.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            e = torch.mean((ae(batch)-batch)**2, dim=1).cpu().numpy()
            errs.append(e)
    errs = np.concatenate(errs)

    thr = fit_gamma_threshold(errs, cfg["threshold"]["percentile"])
    print(f"Gamma {cfg['threshold']['percentile']}-th percentile threshold = {thr:.6f}")

    # save -------------------------------------------------------------
    torch.save(ae.state_dict(), out_dir / "ae_dev.pt")
    np.save(out_dir / "dev_errors.npy", errs)
    with open(out_dir / "threshold.txt", "w") as f:
        f.write(f"{thr:.9f}")
    print("✔  Saved weights + errors + threshold to", out_dir)

if __name__ == "__main__":
    main()
