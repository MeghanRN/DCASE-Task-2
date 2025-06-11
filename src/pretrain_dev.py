#!/usr/bin/env python3
import os, glob, yaml, numpy as np, torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from utils import load_config, extract_logmel, make_windows, fit_gamma_threshold

class DevIterableDataset(IterableDataset):
    def __init__(self, dev_root, cfg):
        self.machine_dirs = [os.path.join(dev_root, m, "train")
                             for m in os.listdir(dev_root)]
        self.cfg = cfg

    def __iter__(self):
        for mdir in self.machine_dirs:
            for wav in glob.glob(os.path.join(mdir, "*.wav")):
                logmel = extract_logmel(
                    wav,
                    sr=self.cfg["audio"]["sample_rate"],
                    n_fft=self.cfg["audio"]["n_fft"],
                    hop_length=self.cfg["audio"]["hop_length"],
                    n_mels=self.cfg["audio"]["n_mels"]
                )
                windows = make_windows(logmel, self.cfg["audio"]["context"])
                for w in windows:
                    yield torch.from_numpy(w).float()

class AutoEncoder(nn.Module):
    def __init__(self, D, h=[128,64], b=8):
        super().__init__()
        dims = [D] + h + [b]
        self.enc = nn.Sequential(*[
            nn.Sequential(nn.Linear(dims[i], dims[i+1]),
                          nn.BatchNorm1d(dims[i+1]), nn.ReLU())
            for i in range(len(dims)-1)
        ])
        dims2 = [b] + h[::-1] + [D]
        self.dec = nn.Sequential(*[
            nn.Sequential(nn.Linear(dims2[i], dims2[i+1]),
                          nn.BatchNorm1d(dims2[i+1]),
                          nn.ReLU() if i < len(dims2)-2 else nn.Identity())
            for i in range(len(dims2)-1)
        ])

    def forward(self, x):
        return self.dec(self.enc(x))

def main():
    cfg = load_config()
    DEV_ROOT = cfg["data"]["dev_dir"]
    os.makedirs("pretrained", exist_ok=True)

    # dataset & loader
    ds = DevIterableDataset(DEV_ROOT, cfg)
    loader = DataLoader(ds,
                        batch_size=cfg["train"]["batch_size"],
                        shuffle=False,
                        num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    D = cfg["audio"]["n_mels"] * cfg["audio"]["context"]
    model = AutoEncoder(D).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = nn.MSELoss()

    print("Pre-training on development data...")
    for ep in range(cfg["train"]["epochs"]):
        model.train()
        bar = tqdm(loader, desc=f"Pretrain Epoch {ep+1}")
        for batch in bar:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            bar.set_postfix(loss=loss.item())

    # compute errors & threshold
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in DataLoader(ds,
                    batch_size=cfg["train"]["batch_size"],
                    num_workers=2):
            batch = batch.to(device)
            errs.append(torch.mean((model(batch)-batch)**2, dim=1).cpu().numpy())
    errs = np.concatenate(errs)
    thr = fit_gamma_threshold(errs, cfg["threshold"]["percentile"])

    # save
    torch.save(model.state_dict(), "pretrained/pretrained_dev.pth")
    np.save("pretrained/pretrained_dev_errors.npy", errs)
    with open("pretrained/pretrained_dev_threshold.txt", "w") as f:
        f.write(str(thr))
    print(f"Saved pre-train model + threshold={thr:.6f}")

if __name__ == "__main__":
    main()
