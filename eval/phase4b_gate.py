"""Threshold-calibrated bridge-vs-no gate on top of the 4-class cross-encoder.

Targets the residual no<->B confusion (F1[no]~0.25) that class-reweighting could not fix. Instead of
plain argmax, predict "no" when P(no) >= tau, else argmax over {A,B,C}. tau is tuned on a grouped
DEV split carved from the training events (GPT-5.5 silver) — NEVER on the gold test.

Design:
  train events = 62 non-Claude events, split grouped into TRAIN (~85%) + DEV (~15%).
  TEST-GOLD    = Claude<->GPT-5.5 consensus from the 8 Claude events (unchanged, non-circular).
  Model/recipe = XLM-R-base, sqrt tempered class weights (the adopted config), 4 epochs.
  Report on GOLD: argmax (baseline) vs gated (tau* from DEV). Saves probs for free re-sweeps.

Run:  ~/opt/anaconda3/envs/dify/bin/python phase4b_gate.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CLASSES = ["no", "A", "B", "C"]
C2I = {c: i for i, c in enumerate(CLASSES)}
BACKBONE = "FacebookAI/xlm-roberta-base"
MAXLEN, EPOCHS, BS = 256, 4, 16


def load(p):
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(HERE / p) if l.strip())}


def lab4(r):
    return r["bridge_type"] if r["is_bridge"] == "yes" else ("no" if r["is_bridge"] == "no" else None)


class PairDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["en"], r["zh"], truncation=True, max_length=MAXLEN,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, r["y"]


def probs(model, rows, tok, device):
    dl = DataLoader(PairDS(rows, tok), batch_size=BS)
    model.eval(); ps, gts = [], []
    with torch.no_grad():
        for batch, y in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            ps.append(torch.softmax(model(**batch).logits, -1).cpu().numpy())
            gts += list(y)
    return np.concatenate(ps), np.array(gts)


def scored(P, yt, tau):
    """gated prediction: 'no' if P[no]>=tau else argmax over A/B/C."""
    if tau is None:
        pred = P.argmax(1)
    else:
        pred = np.where(P[:, 0] >= tau, 0, 1 + P[:, 1:].argmax(1))
    yp = np.array([CLASSES[i] for i in pred]); yg = np.array([CLASSES[i] for i in yt])
    macro = f1_score(yg, yp, average="macro", labels=CLASSES, zero_division=0)
    perc = f1_score(yg, yp, average=None, labels=CLASSES, zero_division=0)
    return macro, {c: float(f) for c, f in zip(CLASSES, perc)}, confusion_matrix(yg, yp, labels=CLASSES), (yp == yg).mean()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device, flush=True)
    cl, gp = load("llm_labels_claude.jsonl"), load("llm_labels_gpt55.jsonl")
    cev = set(r["event_id"] for r in cl.values())

    pool = [{"event": r["event_id"], "en": r["en_text"], "zh": r["zh_text"], "y": C2I[lab4(r)]}
            for r in gp.values() if r["event_id"] not in cev and lab4(r) in CLASSES]
    groups = np.array([r["event"] for r in pool])
    gtr, gde = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(pool, [r["y"] for r in pool], groups))
    train_rows = [pool[i] for i in gtr]; dev_rows = [pool[i] for i in gde]
    gold = [{"en": gp[k]["en_text"], "zh": gp[k]["zh_text"], "y": C2I[lab4(cl[k])]}
            for k in cl if k in gp and lab4(cl[k]) and lab4(cl[k]) == lab4(gp[k])]
    print(f"train={len(train_rows)} ({len(set(groups[gtr]))} ev)  dev={len(dev_rows)} ({len(set(groups[gde]))} ev)  gold={len(gold)}")
    print(f"train dist={dict(Counter(CLASSES[r['y']] for r in train_rows))}  gold dist={dict(Counter(CLASSES[r['y']] for r in gold))}", flush=True)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    torch.manual_seed(0); np.random.seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=4).to(device)
    cnt = Counter(r["y"] for r in train_rows)
    raw = np.sqrt(np.array([len(train_rows) / max(cnt.get(i, 1), 1) for i in range(4)], float)); raw /= raw.mean()
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(raw, dtype=torch.float, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    tr = DataLoader(PairDS(train_rows, tok), batch_size=BS, shuffle=True)
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for batch, y in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = torch.tensor(y).to(device)
            opt.zero_grad(); loss = lossf(model(**batch).logits, y); loss.backward(); opt.step(); tot += loss.item()
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(tr):.3f}", flush=True)

    Pdev, ydev = probs(model, dev_rows, tok, device)
    Pgold, ygold = probs(model, gold, tok, device)
    np.savez_compressed(HERE / "phase4b_gate_probs.npz", Pdev=Pdev, ydev=ydev, Pgold=Pgold, ygold=ygold)

    # sweep tau on DEV
    taus = [None] + list(np.round(np.arange(0.10, 0.65, 0.05), 2))
    print("\n--- DEV tau sweep (macro-F1, F1[no]) ---")
    best_tau, best_dev = None, -1
    for t in taus:
        m, per, _, _ = scored(Pdev, ydev, t)
        tag = "argmax" if t is None else f"tau={t}"
        print(f"  {tag:>10}: dev macroF1={m:.3f}  F1[no]={per['no']:.3f}")
        if t is not None and m > best_dev:
            best_dev, best_tau = m, t
    print(f"  -> best tau on DEV = {best_tau} (dev macroF1={best_dev:.3f})")

    print("\n==================== GOLD (non-circular): argmax vs gated ====================")
    res = {}
    for name, t in [("argmax", None), (f"gated(tau={best_tau})", best_tau)]:
        m, per, cm, acc = scored(Pgold, ygold, t)
        res[name] = {"tau": t, "acc": float(acc), "macroF1": float(m), "perclass": per, "confusion": cm.tolist()}
        print(f"[{name}] acc={acc:.3f} macroF1={m:.3f}  "
              f"F1 no={per['no']:.3f} A={per['A']:.3f} B={per['B']:.3f} C={per['C']:.3f}")
        print("  confusion(no/A/B/C):", cm.tolist())
    (HERE / "phase4b_gate_result.json").write_text(json.dumps({"best_tau": best_tau, "gold": res}, indent=2))
    print("\nsaved phase4b_gate_result.json + phase4b_gate_probs.npz")


if __name__ == "__main__":
    main()
