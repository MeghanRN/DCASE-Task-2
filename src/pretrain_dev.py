#!/usr/bin/env python3
# ------------------------------------------------------------
# Pre-train one AE on *all* dev-set normals (domain-agnostic)
# ------------------------------------------------------------
import os, glob, torch, numpy as np
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import utils as ut

cfg = ut.load_config()
AUDIO, TRAIN = cfg["audio"], cfg["train"]

# -------- dataset -------------------------------------------------
class DevWindows(IterableDataset):
    def __init__(self, root):
        self.wavs = glob.glob(f"{root}/*/train/*.wav")

    def __iter__(self):
        for wf in self.wavs:
            y, sr = ut.librosa.load(wf, sr=AUDIO["sample_rate"])
            M = ut.extract_logmel(y, sr,
                                  AUDIO["n_fft"],
                                  AUDIO["hop_length"],
                                  AUDIO["n_mels"])
            for w in ut.make_windows(M, AUDIO["context"]):
                yield torch.from_numpy(w)

# -------- model ---------------------------------------------------
class AE(nn.Module):
    def __init__(self, D, H=cfg["model"]["hidden"], B=cfg["model"]["bottleneck"]):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(D, H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H, H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H, B), nn.BatchNorm1d(B), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(B, H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H, H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H, D))
    def forward(self,x): return self.dec(self.enc(x))

# -------- training ------------------------------------------------
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print("Device:", device)

dataset = DevWindows(cfg["paths"]["dev_root"])
loader  = DataLoader(dataset, batch_size=TRAIN["batch_size"],
                     shuffle=False, num_workers=2)

D = AUDIO["n_mels"] * AUDIO["context"]
model = AE(D).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
lossf = nn.MSELoss()

print("Pre-training …")
for ep in range(TRAIN["epochs_pretrain"]):
    bar = tqdm(loader, desc=f"Epoch {ep+1}/{TRAIN['epochs_pretrain']}")
    for batch in bar:
        batch = batch.to(device, dtype=torch.float32)
        loss  = lossf(model(batch), batch)
        opt.zero_grad(); loss.backward(); opt.step()
        bar.set_postfix(loss=f"{loss.item():.4f}")

os.makedirs(cfg["paths"]["pretrained_dir"], exist_ok=True)
torch.save(model.state_dict(), f"{cfg['paths']['pretrained_dir']}/ae_dev.pt")
print("✓ saved pretrained/ae_dev.pt")
