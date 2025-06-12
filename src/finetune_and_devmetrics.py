#!/usr/bin/env python
# ------------------------------------------------------------
# finetune_and_devmetrics.py
# Fine-tunes on additional-train normals, infers on *DEV*-set
# test clips, computes AUC / pAUC / PRF, and prints a YAML
# block you can paste into your meta file.
# ------------------------------------------------------------
import os, yaml, glob, pathlib, zipfile, math
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import scipy.io.wavfile as wav
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.stats import gamma
from utils import extract_logmel, make_windows         

# ---------------- model -----------------
class AE(nn.Module):
    def __init__(self, D, h=128, bottleneck=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,bottleneck),nn.BatchNorm1d(bottleneck),nn.ReLU(),
            nn.Linear(bottleneck,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Linear(h,D)
        )
    def forward(self,x): return self.net(x)

# -------------- helpers -----------------
def cfg(): return yaml.safe_load(open("config.yaml"))
def device(cfg):
    if cfg["device"]=="auto":
        return torch.device("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(cfg["device"])

def win_array(wavs,cfg_):
    xs=[]
    for wf in wavs:
        sr,y=wav.read(wf); y=y.astype(np.float32)/32768
        M=extract_logmel(y,sr,cfg_)
        xs.append(make_windows(M,cfg_))
    return np.concatenate(xs,0)

def thr_gamma(errors,p):
    a,loc,s=gamma.fit(errors,floc=0)
    return gamma.ppf(p,a,loc=loc,scale=s)

# -------------- main --------------------
def main():
    cfg_   = cfg()
    D      = cfg_["audio"]["n_mels"]*cfg_["audio"]["context"]
    dev_root = cfg_["paths"]["dev_root"]
    eva_root = cfg_["paths"]["eval_root"]
    out_root = pathlib.Path("dev_results"); out_root.mkdir(exist_ok=True,parents=True)

    # Load once-trained dev weights if present
    base_w = pathlib.Path(cfg_["paths"]["pretrained_dir"]) / "ae_dev.pt"
    if not base_w.exists():
        raise RuntimeError("Pre-train weights not found; run pretrain_dev.py first.")

    dev   = device(cfg_); print("Device:",dev)

    metrics_yaml = ["  development_dataset:"]

    # ------------------------------------------------------------------
    for machine in sorted(os.listdir(dev_root)):
        print(f"\n=== {machine} ===")
        ae=AE(D,cfg_["model"]["hidden"],cfg_["model"]["bottleneck"]).to(dev)
        ae.load_state_dict(torch.load(base_w,map_location=dev))

        # ---------- fine-tune on ADD-TRAIN normals --------------------
        tr_wavs=glob.glob(f"{eva_root}/{machine}/train/**/*.wav",recursive=True)
        X=win_array(tr_wavs,cfg_); ld=torch.utils.data.DataLoader(
              torch.from_numpy(X),batch_size=cfg_["train"]["batch_size"],shuffle=True)
        opt=optim.Adam(ae.parameters(),lr=cfg_["train"]["lr"])
        ae.train()
        for ep in range(cfg_["train"]["epochs_finetune"]):
            for b in ld:
                b=b.to(dev); opt.zero_grad()
                loss=((ae(b)-b)**2).mean(); loss.backward(); opt.step()

        # ---------- threshold from training errors --------------------
        with torch.no_grad():
            errs=[]
            for b in ld:
                b=b.to(dev)
                errs.append(((ae(b)-b)**2).mean(1).cpu().numpy())
        thr=thr_gamma(np.concatenate(errs),cfg_["train"]["gamma_percentile"])

        # ---------- inference on DEV test -----------------------------
        test_wavs=sorted(glob.glob(f"{dev_root}/{machine}/test/*.wav"))
        y_true=[]; y_score=[]
        out_scores = out_root/f"anomaly_score_{machine}_section_00_test.csv"
        out_decide = out_root/f"decision_result_{machine}_section_00_test.csv"
        with open(out_scores,"w") as sf, open(out_decide,"w") as df:
            ae.eval()
            for wf in tqdm(test_wavs,desc="Infer"):
                fname=os.path.basename(wf)
                label=0 if "normal" in fname else 1
                domain="source" if "_source_" in fname else "target"
                sr,y=wav.read(wf); y=y.astype(np.float32)/32768
                M=extract_logmel(y,sr,cfg_); W=make_windows(M,cfg_)
                with torch.no_grad():
                    e=((ae(torch.from_numpy(W).to(dev))-torch.from_numpy(W).to(dev))**2
                       ).mean(1).cpu().numpy().mean()
                sf.write(f"{fname},{e:.6f}\n")
                df.write(f"{fname},{int(e>thr)}\n")
                y_true.append((label,domain)); y_score.append(e)

        # ---------- metrics ------------------------------------------
        y_score=np.array(y_score)
        y_lab  = np.array([l for l,_ in y_true])
        dom    = np.array([d for _,d in y_true])

        def auc_dom(d):
            idx=(dom==d); 
            return roc_auc_score(y_lab[idx],y_score[idx]) if idx.sum() and y_lab[idx].sum() else math.nan
        auc_src, auc_tgt = auc_dom("source")*100, auc_dom("target")*100

        # pAUC over BOTH domains up to FPR 0.1
        from sklearn.metrics import roc_curve
        fpr,tpr,_=roc_curve(y_lab,y_score)
        p_mask=fpr<=0.1
        pauc = np.trapz(tpr[p_mask],fpr[p_mask])/0.1*100

        # precision/recall/F1 per domain
        y_pred=(y_score>thr).astype(int)
        prec,rec,f1,_ = precision_recall_fscore_support(
            y_lab,y_pred,labels=[1],average=None)  # overall
        # per domain
        pr_src=precision_recall_fscore_support(
            y_lab[dom=="source"],y_pred[dom=="source"],labels=[1],zero_division=0)
        pr_tgt=precision_recall_fscore_support(
            y_lab[dom=="target"],y_pred[dom=="target"],labels=[1],zero_division=0)

        # ----- print YAML snippet ------------------------------------
        metrics_yaml.append(f"    {machine}:")
        metrics_yaml.append(f"      auc_source: {auc_src:.2f}")
        metrics_yaml.append(f"      auc_target: {auc_tgt:.2f}")
        metrics_yaml.append(f"      pauc: {pauc:.2f}")
        metrics_yaml.append(f"      precision_source: {pr_src[0][0]:.3f}")
        metrics_yaml.append(f"      precision_target: {pr_tgt[0][0]:.3f}")
        metrics_yaml.append(f"      recall_source: {pr_src[1][0]:.3f}")
        metrics_yaml.append(f"      recall_target: {pr_tgt[1][0]:.3f}")
        metrics_yaml.append(f"      f1_source: {pr_src[2][0]:.3f}")
        metrics_yaml.append(f"      f1_target: {pr_tgt[2][0]:.3f}")

        print(f"→ completed {machine}")

    # ----------------------------------------------------------------
    print("\n\n==== YAML block for meta file ====")
    print("\n".join(metrics_yaml))
    print("==================================")

if __name__=="__main__":
    main()
