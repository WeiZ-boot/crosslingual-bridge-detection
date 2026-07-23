"""Baseline benchmark on the Claude<->GPT-5.5 consensus GOLD (1,042 pairs, 4-class {no,A,B,C}).

All methods share ONE honest test set (the consensus gold from the 8 Claude events). Trained
baselines learn from GPT-5.5 silver on the 62 non-Claude events (event-disjoint from gold), reusing
cached BGE-M3 embeddings (pair_emb_v2.npz) aligned by pair_id -- no GPU, ~1 min.

Baselines:
  majority       -- predict the most frequent train class
  sim-only LR    -- logistic regression on BGE-M3 cosine similarity (1-D)
  bi-encoder MLP -- MLP on [u,v,|u-v|,u*v]  (the Phase-4 content model)
  zero-shot LLM  -- gpt-5.4-nano labels (llm_labels_v2.jsonl) scored vs gold  (NOTE: GPT-5.5 and
                    Claude are excluded -- each is an annotator OF the gold, so trivially 100%)
Cross-encoder (XLM-R) and NLI-zero-shot are reported from their own scripts (phase4b_*, nli_*).
Run:  python3 gold_baselines.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
CLASSES = ["no", "A", "B", "C"]


def load(p):
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(HERE / p) if l.strip())}


def lab4(r):
    return r["bridge_type"] if r["is_bridge"] == "yes" else ("no" if r["is_bridge"] == "no" else None)


def report(name, yt, yp, out):
    macro = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)
    per = dict(zip(CLASSES, f1_score(yt, yp, average=None, labels=CLASSES, zero_division=0)))
    acc = np.mean([a == b for a, b in zip(yt, yp)])
    out[name] = {"acc": float(acc), "macroF1": float(macro), "perclass": {c: float(per[c]) for c in CLASSES}}
    print(f"[{name:<16}] acc={acc:.3f} macroF1={macro:.3f}  "
          f"F1 no={per['no']:.3f} A={per['A']:.3f} B={per['B']:.3f} C={per['C']:.3f}")


def main():
    gp, nano = load("llm_labels_gpt55.jsonl"), load("llm_labels_v2.jsonl")
    # FROZEN gold snapshot (gold_consensus.jsonl) + excluded events (gold_events.json)
    gold_rows = [json.loads(l) for l in (HERE / "gold_consensus.jsonl").read_text().splitlines() if l.strip()]
    gold_events = set(json.loads((HERE / "gold_events.json").read_text()))

    # embeddings aligned to llm_labels_v2 line order -> pair_id map
    v2_ids = [json.loads(l)["pair_id"] for l in (HERE / "llm_labels_v2.jsonl").read_text().splitlines() if l.strip()]
    z = np.load(HERE / "pair_emb_v2.npz"); EU, ZV = z["eu"], z["zv"]
    idx = {pid: i for i, pid in enumerate(v2_ids)}

    def feats(pids):
        i = np.array([idx[p] for p in pids]); eu, zv = EU[i], ZV[i]
        return np.concatenate([eu, zv, np.abs(eu - zv), eu * zv], 1), np.sum(eu * zv, 1)

    # TRAIN = GPT-5.5 silver on events NOT in the frozen gold
    train_ids = [k for k, r in gp.items() if r["event_id"] not in gold_events and lab4(r) in CLASSES]
    ytr = np.array([lab4(gp[k]) for k in train_ids])
    Xtr, str_ = feats(train_ids)
    # GOLD = frozen consensus
    gold_ids = [r["pair_id"] for r in gold_rows]
    yg = [r["label"] for r in gold_rows]
    Xg, sg = feats(gold_ids)
    print(f"train={len(train_ids)} gold={len(gold_ids)}  gold dist={dict(Counter(yg))}\n")

    out = {}
    # majority
    maj = Counter(ytr).most_common(1)[0][0]
    report("majority", yg, [maj] * len(yg), out)
    # sim-only LR
    lr = LogisticRegression(max_iter=1000).fit(str_.reshape(-1, 1), ytr)
    report("sim-only-LR", yg, list(lr.predict(sg.reshape(-1, 1))), out)
    # bi-encoder MLP
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=0).fit(Xtr, ytr)
    report("bi-encoder-MLP", yg, list(mlp.predict(Xg)), out)
    # zero-shot LLM = nano labels vs gold
    ynano = [lab4(nano[k]) if k in nano and lab4(nano[k]) else "no" for k in gold_ids]
    report("zeroshot-nano", yg, ynano, out)

    (HERE / "gold_baselines_result.json").write_text(json.dumps(out, indent=2))
    print("\nsaved gold_baselines_result.json  (cross-encoder + NLI reported separately)")


if __name__ == "__main__":
    main()
