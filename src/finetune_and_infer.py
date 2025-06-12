#!/usr/bin/env python
# ------------------------------------------------------------
# finetune_and_infer.py
# Fine-tunes on additional-train normals and infers on eval-test.
# Writes CSVs + decision CSVs, builds DCASE submission ZIP.
# ------------------------------------------------------------
import os, yaml, glob, zipfile, shutil, pathlib
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import scipy.io.wavfile as wav
from tqdm import tqdm
from datetime import datetime
from utils import extract_logmel, make_windows  # from your existing utils

# --------------- simple AE -----------------
class AutoEncoder(nn.Module):
    def __init__(self, D, h=128, bottleneck=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, bottleneck), nn.BatchNorm1d(bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, D)
        )
    def forward(self, x): return self.net(x)

# --------------- helpers -------------------
def load_cfg():
    return yaml.safe_load(open("config.yaml","r"))

def device_from_cfg(cfg):
    if cfg["device"]=="auto":
        return torch.device("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available()
                            else "cpu")
    return torch.device(cfg["device"])

def gamma_threshold(errs, p):
    from scipy.stats import gamma
    shape, loc, scale = gamma.fit(errs, floc=0)
    return gamma.ppf(p, shape, loc=loc, scale=scale)

def dataset_windows(wav_files, cfg):
    xs = []
    for wf in wav_files:
        sr, y = wav.read(wf); y = y.astype(np.float32)/32768
        M = extract_logmel(y, sr, cfg)
        xs.append(make_windows(M, cfg))
    return np.concatenate(xs,0)

# --------------- main ----------------------
def main():
    cfg  = load_cfg()
    dev  = cfg["paths"]["dev_root"]
    eva  = cfg["paths"]["eval_root"]
    out_root = cfg["paths"]["out_root"]
    os.makedirs(out_root, exist_ok=True)
    D = cfg["feature"]["n_mels"]*cfg["feature"]["context"]
    dev_pretrain_weights = os.path.join(cfg["paths"]["pretrained_dir"], "ae_dev.pt")

    device = device_from_cfg(cfg)
    print("Using device:", device)

    # -------------------------------------------------
    # 1) (Optional) pre-train on dev normals once
    # -------------------------------------------------
    if not os.path.exists(dev_pretrain_weights):
        print("Pre-training on dev normals …")
        wavs = glob.glob(f"{dev}/*/train/**/*.wav", recursive=True)
        X = dataset_windows(wavs, cfg)
        loader = torch.utils.data.DataLoader(
            torch.from_numpy(X), batch_size=cfg["train"]["batch_size"], shuffle=True)
        ae = AutoEncoder(D, cfg["model"]["hidden"], cfg["model"]["bottleneck"]).to(device)
        opt = optim.Adam(ae.parameters(), lr=cfg["train"]["lr"])
        for ep in range(cfg["train"]["epochs_pretrain"]):
            pbar = tqdm(loader, desc=f"Pre-train {ep+1}/{cfg['train']['epochs_pretrain']}")
            for batch in pbar:
                batch = batch.to(device)
                opt.zero_grad()
                loss = ((ae(batch)-batch)**2).mean()
                loss.backward(); opt.step()
                pbar.set_postfix(loss=loss.item())
        os.makedirs(os.path.dirname(dev_pretrain_weights), exist_ok=True)
        torch.save(ae.state_dict(), dev_pretrain_weights)
        print("Saved", dev_pretrain_weights)
    else:
        print("Found", dev_pretrain_weights)

    # -------------------------------------------------
    # 2) Fine-tune per machine + infer
    # -------------------------------------------------
    label = pathlib.Path(out_root).name    # used for submission dir
    for machine in sorted(os.listdir(eva)):
        m_root = os.path.join(eva, machine)
        print(f"\n=== {machine} ===")
        ae = AutoEncoder(D, cfg["model"]["hidden"], cfg["model"]["bottleneck"]).to(device)
        ae.load_state_dict(torch.load(dev_pretrain_weights, map_location=device))

        # ---- fine-tune ----------------------------------
        train_wavs = glob.glob(f"{m_root}/train/**/*.wav", recursive=True)
        X = dataset_windows(train_wavs, cfg)
        loader = torch.utils.data.DataLoader(torch.from_numpy(X),
                    batch_size=cfg["train"]["batch_size"], shuffle=True)
        opt = optim.Adam(ae.parameters(), lr=cfg["train"]["lr"])
        ae.train()
        for ep in range(cfg["train"]["epochs_finetune"]):
            pbar=tqdm(loader, desc=f"Fine-tune {machine} {ep+1}/{cfg['train']['epochs_finetune']}")
            for batch in pbar:
                batch=batch.to(device); opt.zero_grad()
                loss=((ae(batch)-batch)**2).mean()
                loss.backward(); opt.step(); pbar.set_postfix(loss=loss.item())

        # determine threshold
        with torch.no_grad():
            errs=[]
            for batch in loader:
                batch=batch.to(device)
                e=((ae(batch)-batch)**2).mean(1).cpu().numpy()
                errs.append(e)
        errs=np.concatenate(errs)
        thr=gamma_threshold(errs, cfg["train"]["gamma_percentile"])
        print(f"  threshold={thr:.3f}")

        # ---- inference ----------------------------------
        test_wavs=sorted(glob.glob(f"{m_root}/test/*.wav"))
        score_csv  = os.path.join(out_root,
                       f"anomaly_score_{machine}_section_00_test.csv")
        decision_csv=os.path.join(out_root,
                       f"decision_result_{machine}_section_00_test.csv")
        with open(score_csv,"w") as sf, open(decision_csv,"w") as df:
            for wf in tqdm(test_wavs, desc="Infer"):
                sr,y=wav.read(wf); y=y.astype(np.float32)/32768
                M=extract_logmel(y,sr,cfg)
                W=make_windows(M,cfg)
                with torch.no_grad():
                    e=((ae(torch.from_numpy(W).to(device))-torch.from_numpy(W).to(device))**2
                       ).mean(1).cpu().numpy().mean()
                sf.write(f"{os.path.basename(wf)},{e:.6f}\n")
                df.write(f"{os.path.basename(wf)},{int(e>thr)}\n")
        print("  ↪ wrote", os.path.basename(score_csv))

    # -------------------------------------------------
    # 3) Package submission
    # -------------------------------------------------
    meta   = f"meta/{label}.meta.yaml"               # you create manually
    report = f"reports/{label}.technical_report.pdf" # you create manually
    assert os.path.exists(meta) and os.path.exists(report), \
        "Place meta YAML in meta/ and PDF in reports/ before zipping."

    sub_dir = f"task2/{label}"
    with zipfile.ZipFile("submission.zip","w",zipfile.ZIP_DEFLATED) as zf:
        zf.write(report,               arcname=f"task2/{pathlib.Path(report).name}")
        zf.write(meta,                 arcname=f"{sub_dir}/{pathlib.Path(meta).name}")
        for csv in glob.glob(f"{out_root}/*.csv"):
            zf.write(csv, arcname=f"{sub_dir}/{os.path.basename(csv)}")
    print("\nBuilt submission.zip ✅  ->  ready for upload.")

if __name__=="__main__":
    main()
