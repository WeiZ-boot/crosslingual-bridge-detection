"""NLI / stance-transfer zero-shot baseline on the frozen consensus gold.

Question: does off-the-shelf cross-lingual entailment/contradiction transfer to the bridge taxonomy?
Uses mDeBERTa-v3-xnli (cached). Premise=EN, hypothesis=ZH -> P(entail/neutral/contra). Zero-shot map:
  contradiction -> C (ZH rebuts EN)   entailment -> A (faithful restatement)   neutral -> B (reframe)
(no training; pure transfer. "no" is not producible by NLI, so it's a 3-way A/B/C proxy.)
Also a trained variant: LogisticRegression on the 3 NLI probs -> 4-class, trained on GPT-5.5 silver
(events not in gold), which CAN produce "no". Appends both rows to gold_baselines_result.json.
Run (CPU, ~few min):  ~/opt/anaconda3/envs/dify/bin/python nli_baseline.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CLASSES = ["no", "A", "B", "C"]
NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEV = "cpu"


def load(p):
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(HERE / p) if l.strip())}


def lab4(r):
    return r["bridge_type"] if r["is_bridge"] == "yes" else ("no" if r["is_bridge"] == "no" else None)


def nli_probs(rows, tok, model):
    """rows: list of (en, zh). returns [N,3] in order [entail, neutral, contra]."""
    id2l = {i: l.lower() for i, l in model.config.id2label.items()}
    order = [next(i for i, l in id2l.items() if "entail" in l),
             next(i for i, l in id2l.items() if "neutral" in l),
             next(i for i, l in id2l.items() if "contra" in l)]
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), 32):
            batch = rows[s:s + 32]
            enc = tok([r[0] for r in batch], [r[1] for r in batch], truncation=True, max_length=256,
                      padding=True, return_tensors="pt").to(DEV)
            p = torch.softmax(model(**enc).logits, -1).cpu().numpy()
            out.append(p[:, order])
            print(f"  nli {s+len(batch)}/{len(rows)}", end="\r", flush=True)
    print()
    return np.concatenate(out)


def report(name, yt, yp, out):
    macro = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)
    per = dict(zip(CLASSES, f1_score(yt, yp, average=None, labels=CLASSES, zero_division=0)))
    acc = float(np.mean([a == b for a, b in zip(yt, yp)]))
    out[name] = {"acc": acc, "macroF1": float(macro), "perclass": {c: float(per[c]) for c in CLASSES}}
    print(f"[{name:<18}] acc={acc:.3f} macroF1={macro:.3f}  F1 no={per['no']:.3f} A={per['A']:.3f} B={per['B']:.3f} C={per['C']:.3f}")


def main():
    gp = load("llm_labels_gpt55.jsonl")
    gold_events = set(json.loads((HERE / "gold_events.json").read_text()))
    gold = [json.loads(l) for l in (HERE / "gold_consensus.jsonl").read_text().splitlines() if l.strip()]
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI).to(DEV)
    print("id2label:", model.config.id2label, flush=True)

    yg = [r["label"] for r in gold]
    Pg = nli_probs([(r["en_text"], r["zh_text"]) for r in gold], tok, model)
    out = json.loads((HERE / "gold_baselines_result.json").read_text()) if (HERE / "gold_baselines_result.json").exists() else {}

    # zero-shot map: entail->A, neutral->B, contra->C
    m = {0: "A", 1: "B", 2: "C"}
    report("nli-zeroshot", yg, [m[i] for i in Pg.argmax(1)], out)

    # trained: LR on NLI probs -> 4-class, train on GPT-5.5 silver (non-gold events)
    train = [r for r in gp.values() if r["event_id"] not in gold_events and lab4(r) in CLASSES]
    # subsample train for speed (NLI over 11k is slow on CPU); stratified ~3000
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(train))[:3000]
    train = [train[i] for i in idx]
    ytr = [lab4(r) for r in train]
    Ptr = nli_probs([(r["en_text"], r["zh_text"]) for r in train], tok, model)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Ptr, ytr)
    report("nli-LR", yg, list(lr.predict(Pg)), out)

    (HERE / "gold_baselines_result.json").write_text(json.dumps(out, indent=2))
    print("appended nli-zeroshot + nli-LR to gold_baselines_result.json")


if __name__ == "__main__":
    main()
