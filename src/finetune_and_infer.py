#!/usr/bin/env python3
# ------------------------------------------------------------
# Fine-tune per eval-machine + inference + package submission
# ------------------------------------------------------------
import os, glob, yaml, zipfile, pathlib, shutil, numpy as np, torch
import torch.nn as nn, torch.optim as optim
import scipy.io.wavfile as wav
from tqdm import tqdm
import utils as ut

cfg  = ut.load_config()
AUDIO, TRAIN = cfg["audio"], cfg["train"]
D = AUDIO["n_mels"] * AUDIO["context"]

# ------------ helper -------------------
def gamma_threshold(errs, p):
    import scipy.stats as st
    shape, loc, scale = st.gamma.fit(errs, floc=0)
    return st.gamma.ppf(p, shape, loc=loc, scale=scale)

def dataset_windows(wav_files):
    xs = []
    for wf in wav_files:
        # Let extract_logmel load the .wav itself
        M = ut.extract_logmel(
            wf,                            # filepath
            sr=AUDIO["sample_rate"],       # sampling rate
            n_fft=AUDIO["n_fft"],
            hop_length=AUDIO["hop_length"],
            n_mels=AUDIO["n_mels"]
        )
        xs.append(ut.make_windows(M, AUDIO["context"]))
    # concatenate all windows into one array
    return np.concatenate(xs, axis=0)

# ------------ model --------------------
class AE(nn.Module):
    def __init__(self, D, H=cfg["model"]["hidden"], B=cfg["model"]["bottleneck"]):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(D,H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H,H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H,B), nn.BatchNorm1d(B), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(B,H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H,H), nn.BatchNorm1d(H), nn.ReLU(),
            nn.Linear(H,D))
    def forward(self,x): return self.dec(self.enc(x))

# ------------ main ---------------------
def main():
    dev_root  = cfg["paths"]["dev_root"]
    eval_root = cfg["paths"]["eval_root"]
    out_root  = pathlib.Path(cfg["paths"]["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    # device
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print("Device:", device)

    # ---- load / create dev-pretrain weights --------------
    w_pre = pathlib.Path(cfg["paths"]["pretrained_dir"])/"ae_dev.pt"
    if not w_pre.exists():
        raise RuntimeError("Pre-train weights missing. Run pretrain_dev.py first!")

    # ---- iterate over machines ---------------------------
    for machine in sorted(os.listdir(eval_root)):
        mdir = pathlib.Path(eval_root)/machine
        print(f"\n=== {machine} ===")
        ae = AE(D).to(device)
        ae.load_state_dict(torch.load(w_pre, map_location=device))

        # ---- fine-tune -----------------------------------
        train_wavs = glob.glob(str(mdir/"train/**/*.wav"), recursive=True)
        X = dataset_windows(train_wavs)
        loader = torch.utils.data.DataLoader(torch.from_numpy(X),
                    batch_size=TRAIN["batch_size"], shuffle=True)
        opt = optim.Adam(ae.parameters(), lr=TRAIN["lr"])
        ae.train()
        for ep in range(TRAIN["epochs_finetune"]):
            bar=tqdm(loader, desc=f"FT {machine} {ep+1}/{TRAIN['epochs_finetune']}")
            for b in bar:
                b=b.to(device); loss=((ae(b)-b)**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                bar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- threshold from fine-tune data ---------------
        with torch.no_grad():
            errs=[]
            for b in loader:
                b=b.to(device)
                errs.append(((ae(b)-b)**2).mean(1).cpu().numpy())
            thr = gamma_threshold(np.concatenate(errs), TRAIN["gamma_percentile"])
        print(f"threshold={thr:.3f}")

        # ---- inference on eval-test ----------------------
        test_wavs = sorted(glob.glob(str(mdir/"test/*.wav")))
        score_csv    = out_root/f"anomaly_score_{machine}_section_00_test.csv"
        decision_csv = out_root/f"decision_result_{machine}_section_00_test.csv"
        with score_csv.open("w") as sf, decision_csv.open("w") as df:
            for wf in tqdm(test_wavs, desc="Infer"):
                sr,y = wav.read(wf); y=y.astype(np.float32)/32768
                M = ut.extract_logmel(y, sr,
                                      AUDIO["n_fft"], AUDIO["hop_length"], AUDIO["n_mels"])
                W = ut.make_windows(M, AUDIO["context"])
                with torch.no_grad():
                    e = ((ae(torch.from_numpy(W).to(device))-torch.from_numpy(W).to(device))**2
                         ).mean(1).mean().item()
                sf.write(f"{os.path.basename(wf)},{e:.6f}\n")
                df.write(f"{os.path.basename(wf)},{int(e>thr)}\n")
        print("✓ wrote CSVs for", machine)

    print("\nAll machines done → results in", out_root)

if __name__ == "__main__":
    main()
