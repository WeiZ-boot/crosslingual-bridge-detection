"""FUSE-C Stage-1 training: fused stance+temporal+similarity student with calibration.

Steps:
 1. Train XLM-R cross-encoder on silver (non-gold events) with warmup + grad-clip (stability fix).
 2. Extract 4-class logits for train (out-of-fold via a calibration holdout) and test.
 3. Fit fusion MLP over [stance_logits ; temporal(7) ; sim(5)].
 4. Temperature-scale on a grouped holdout.
Outputs: fusec_artifacts.npz (student probs on test, CE probs, all feature blocks),
         student_vs_ce.json (per-class F1 comparison).
Runs CPU. Deterministic seeds. ~ tens of minutes.
"""
from __future__ import annotations
import json, math, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

HERE = Path(__file__).resolve().parent
WS = Path.cwd()  # workspace holds features.npz / pair_texts.json
CLASSES = ["no","A","B","C"]; C2I={c:i for i,c in enumerate(CLASSES)}
BACKBONE = "FacebookAI/xlm-roberta-base"
MAXLEN, EPOCHS, BS, LR = 256, 4, 16, 2e-5
import os
SEED = int(os.environ.get("FUSEC_SEED", "0"))
SUFFIX = f"_seed{SEED}" if SEED != 0 else ""  # seed-0 filenames unchanged
DEV = "cpu"

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def load():
    F = np.load(WS/"features.npz", allow_pickle=True)
    T = json.load(open(WS/"pair_texts.json"))
    return F, T

class PairDS(Dataset):
    def __init__(self, texts, y, tok):
        self.texts, self.y, self.tok = texts, y, tok
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        e,z = self.texts[i]
        enc = self.tok(e, z, truncation=True, max_length=MAXLEN, padding="max_length",
                       return_tensors="pt")
        return {k:v.squeeze(0) for k,v in enc.items()}, int(self.y[i])

def train_ce(texts, y, tok):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=4).to(DEV)
    cnt = Counter(int(v) for v in y)
    # tempered class weights (temperature 0.5 -> sqrt, softer than raw inverse-freq)
    raw = np.array([len(y)/(4*cnt.get(i,1)) for i in range(4)])
    w = torch.tensor((raw**0.5), dtype=torch.float, device=DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    dl = DataLoader(PairDS(texts, y, tok), batch_size=BS, shuffle=True)
    total = len(dl)*EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1*total), total)  # WARMUP fix
    for ep in range(EPOCHS):
        model.train(); running=0.0
        for bi,(batch,yy) in enumerate(dl):
            batch={k:v.to(DEV) for k,v in batch.items()}; yy=torch.tensor(yy,device=DEV)
            opt.zero_grad()
            loss=lossf(model(**batch).logits, yy); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # GRAD-CLIP fix
            opt.step(); sched.step(); running+=loss.item()
            if bi%80==0: log(f"  ep{ep} step{bi}/{len(dl)} loss={loss.item():.3f}")
        log(f" epoch {ep} mean loss {running/len(dl):.3f}")
    return model

@torch.no_grad()
def ce_logits(model, texts, tok):
    model.eval(); out=[]
    dl = DataLoader(PairDS(texts,[0]*len(texts),tok), batch_size=BS)
    for batch,_ in dl:
        batch={k:v.to(DEV) for k,v in batch.items()}
        out.append(model(**batch).logits.cpu().numpy())
    return np.concatenate(out,0)

class Fusion(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,4))
    def forward(self,x): return self.net(x)

def fit_fusion(Xtr, ytr, Xva, yva, cnt):
    torch.manual_seed(SEED)
    raw=np.array([len(ytr)/(4*cnt.get(i,1)) for i in range(4)])
    w=torch.tensor(raw**0.5,dtype=torch.float)
    m=Fusion(Xtr.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.CrossEntropyLoss(weight=w)
    Xt=torch.tensor(Xtr); yt=torch.tensor(ytr)
    best=None; best_f1=-1
    for ep in range(200):
        m.train(); opt.zero_grad(); loss=lossf(m(Xt),yt); loss.backward(); opt.step()
        if ep%10==0:
            m.eval()
            with torch.no_grad():
                pv=m(torch.tensor(Xva)).argmax(1).numpy()
            f1=f1_score(yva,pv,average="macro")
            if f1>best_f1: best_f1=f1; best={k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best); return m

def temp_scale(logits, y):
    logits=torch.tensor(logits); y=torch.tensor(y)
    T=torch.nn.Parameter(torch.ones(1))
    opt=torch.optim.LBFGS([T],lr=0.01,max_iter=200)
    lossf=nn.CrossEntropyLoss()
    def clo():
        opt.zero_grad(); l=lossf(logits/T, y); l.backward(); return l
    opt.step(clo); return float(T.detach())

def main():
    F,T = load()
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    txt_tr=[tuple(t) for t in [(r["en"],r["zh"]) for r in T["train"]]]
    txt_te=[(r["en"],r["zh"]) for r in T["test"]]
    ytr=F["ytr"]; yte=F["yte"]; evtr=F["evtr"]
    # FUSE-C+ Signal 2: recover pair_id per TEST row so student probs join to gold.
    try:
        pair_ids_te = [r["pair_id"] for r in T["test"]]
    except (KeyError, TypeError):
        raise SystemExit(
            "train_fusec.py: T['test'] rows lack 'pair_id'. Add pair_id to the test rows "
            "in the feature builder that writes pair_texts.json (same order as yte), then rerun.")
    Xtmp_tr,Xsim_tr=F["Xtmp_tr"],F["Xsim_tr"]
    Xtmp_te,Xsim_te=F["Xtmp_te"],F["Xsim_te"]

    # grouped holdout of TRAIN events for fusion-val + temperature
    gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=SEED)
    tr_idx,va_idx=next(gss.split(np.zeros(len(ytr)),ytr,groups=evtr))
    log(f"CE train on {len(tr_idx)} fit-fold pairs, {len(va_idx)} cal-holdout")

    ce=train_ce([txt_tr[i] for i in tr_idx], ytr[tr_idx], tok)
    log("extracting CE logits...")
    L_fit=ce_logits(ce,[txt_tr[i] for i in tr_idx],tok)
    L_va =ce_logits(ce,[txt_tr[i] for i in va_idx],tok)
    L_te =ce_logits(ce,txt_te,tok)

    def block(L,tmp,sim): return np.concatenate([L,tmp,sim],1).astype(np.float32)
    Xf_fit=block(L_fit,Xtmp_tr[tr_idx],Xsim_tr[tr_idx])
    Xf_va =block(L_va ,Xtmp_tr[va_idx],Xsim_tr[va_idx])
    Xf_te =block(L_te ,Xtmp_te,Xsim_te)
    cnt=Counter(int(v) for v in ytr[tr_idx])
    fus=fit_fusion(Xf_fit,ytr[tr_idx],Xf_va,ytr[va_idx],cnt)

    with torch.no_grad():
        fus_va_logits=fus(torch.tensor(Xf_va)).numpy()
        fus_te_logits=fus(torch.tensor(Xf_te)).numpy()
    Temp=temp_scale(fus_va_logits, ytr[va_idx])
    log(f"temperature T={Temp:.3f}")
    def sm(x): 
        x=x-x.max(1,keepdims=True); e=np.exp(x); return e/e.sum(1,keepdims=True)
    student_te=sm(fus_te_logits/Temp)
    ce_te=sm(L_te)

    # metrics
    yhat_student=student_te.argmax(1); yhat_ce=ce_te.argmax(1)
    def per(y,yh):
        rep=classification_report(y,yh,target_names=CLASSES,output_dict=True,zero_division=0)
        return {c:round(rep[c]["f1-score"],4) for c in CLASSES}|{"macro":round(rep["macro avg"]["f1-score"],4)}
    res={"ce_only":per(yte,yhat_ce),"student_fused_calibrated":per(yte,yhat_student),
         "temperature":round(Temp,4),"n_test":int(len(yte))}
    json.dump(res, open(WS/f"student_vs_ce{SUFFIX}.json","w"), indent=2)
    np.savez_compressed(WS/f"fusec_artifacts{SUFFIX}.npz",
        student_te=student_te, ce_te=ce_te, L_te=L_te, yte=yte,
        Xtmp_te=Xtmp_te, Xsim_te=Xsim_te, evte=F["evte"], Temp=np.array([Temp]),
        fus_te_logits=fus_te_logits, pair_ids_te=np.array(pair_ids_te))
    # FUSE-C+ Signal 2: pair_id-keyed calibrated student probabilities for the arbiter.
    json.dump({pid: student_te[i].tolist() for i, pid in enumerate(pair_ids_te)},
              open(WS/f"student_probs{SUFFIX}.json", "w"))
    torch.save({"fusion":fus.state_dict(),"temperature":Temp}, WS/f"fuse_student{SUFFIX}.pt")
    log("DONE"); print(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
