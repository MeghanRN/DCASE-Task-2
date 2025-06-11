#!/usr/bin/env python3
import os, glob, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from utils import load_config, extract_logmel, make_windows, write_csv
from scipy.stats import gamma

class EvalTrainDataset(Dataset):
    def __init__(self, mdir, cfg):
        self.windows = []
        for wav in glob.glob(os.path.join(mdir, "train", "*.wav")):
            logmel = extract_logmel(wav,
                sr=cfg["audio"]["sample_rate"],
                n_fft=cfg["audio"]["n_fft"],
                hop_length=cfg["audio"]["hop_length"],
                n_mels=cfg["audio"]["n_mels"])
            self.windows.append(make_windows(logmel, cfg["audio"]["context"]))
        self.windows = np.concatenate(self.windows, axis=0)

    def __len__(self): return len(self.windows)
    def __getitem__(self, i): return torch.from_numpy(self.windows[i]).float()

class AutoEncoder(nn.Module):
    def __init__(self, D, h=[128,64], b=8):
        super().__init__()
        dims=[D]+h+[b]
        self.enc=nn.Sequential(*[
            nn.Sequential(nn.Linear(dims[i],dims[i+1]),
                          nn.BatchNorm1d(dims[i+1]),nn.ReLU())
            for i in range(len(dims)-1)
        ])
        dims2=[b]+h[::-1]+[D]
        self.dec=nn.Sequential(*[
            nn.Sequential(nn.Linear(dims2[i],dims2[i+1]),
                          nn.BatchNorm1d(dims2[i+1]),
                          nn.ReLU() if i<len(dims2)-2 else nn.Identity())
            for i in range(len(dims2)-1)
        ])

    def forward(self,x): return self.dec(self.enc(x))

def gamma_threshold(errs,pct):
    a,loc,scale = gamma.fit(errs, floc=0)
    return float(gamma.ppf(pct/100.0, a, loc=loc, scale=scale))

def main():
    cfg = load_config()
    eval_root = cfg["data"]["eval_dir"]
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    D = cfg["audio"]["n_mels"] * cfg["audio"]["context"]

    pretrained = torch.load("pretrained/pretrained_dev.pth", map_location=device)

    for machine in os.listdir(eval_root):
        mdir = os.path.join(eval_root, machine)
        out = os.path.join("outputs/task2/MeghanKret_Cooper_task2_1", machine)
        os.makedirs(out, exist_ok=True)

        # fine-tune
        ds = EvalTrainDataset(mdir, cfg)
        loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
        model = AutoEncoder(D).to(device)
        model.load_state_dict(pretrained, strict=False)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
        loss_fn = nn.MSELoss()

        for ep in range(cfg["train"]["epochs"]):
            model.train()
            bar = tqdm(loader, desc=f"{machine} FT ep{ep+1}")
            for batch in bar:
                batch = batch.to(device)
                pred = model(batch)
                loss = loss_fn(pred, batch)
                opt.zero_grad(); loss.backward(); opt.step()
                bar.set_postfix(loss=loss.item())

        # threshold
        model.eval()
        errs=[]
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=cfg["train"]["batch_size"]):
                b = batch.to(device)
                errs.append(torch.mean((model(b)-b)**2, dim=1).cpu().numpy())
        errs = np.concatenate(errs)
        thr = gamma_threshold(errs, cfg["threshold"]["percentile"])
        torch.save(model.state_dict(), os.path.join(out,"model.pth"))
        np.save(os.path.join(out,"train_errs.npy"), errs)
        open(os.path.join(out,"threshold.txt"), "w").write(str(thr))

        # inference
        rows_score, rows_dec = [],[]
        for wav in sorted(glob.glob(os.path.join(mdir,"test","*.wav"))):
            logmel = extract_logmel(wav,
                sr=cfg["audio"]["sample_rate"],
                n_fft=cfg["audio"]["n_fft"],
                hop_length=cfg["audio"]["hop_length"],
                n_mels=cfg["audio"]["n_mels"])
            wins = make_windows(logmel, cfg["audio"]["context"])
            x = torch.from_numpy(wins).float().to(device)
            with torch.no_grad():
                e = torch.mean((model(x)-x)**2, dim=1).cpu().numpy()
            sc = float(e.mean()); label = 1 if sc>thr else 0
            fn = os.path.basename(wav)
            rows_score.append([fn, sc])
            rows_dec.append([fn, label])

        write_csv(os.path.join(out,
                   f"anomaly_score_{machine}_section_00_test.csv"), rows_score)
        write_csv(os.path.join(out,
                   f"decision_result_{machine}_section_00_test.csv"), rows_dec)

    print("Done fine‐tune & inference.")

if __name__=="__main__":
    main()
