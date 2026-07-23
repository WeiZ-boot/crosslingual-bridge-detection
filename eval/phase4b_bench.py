"""Cross-encoder (XLM-R, sqrt weights) on the FROZEN consensus gold — for the baseline benchmark.

Train = GPT-5.5 silver on events NOT in gold_events.json; test = gold_consensus.jsonl (same frozen
1,305-pair gold all baselines use). Appends its row to gold_baselines_result.json under 'cross-encoder'.
Run:  ~/opt/anaconda3/envs/dify/bin/python phase4b_bench.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
CLASSES = ["no", "A", "B", "C"]; C2I = {c: i for i, c in enumerate(CLASSES)}
BACKBONE = "FacebookAI/xlm-roberta-base"; MAXLEN, EPOCHS, BS = 256, 4, 16


def load(p):
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(HERE / p) if l.strip())}


def lab4(r):
    return r["bridge_type"] if r["is_bridge"] == "yes" else ("no" if r["is_bridge"] == "no" else None)


class DS(Dataset):
    def __init__(s, rows, tok): s.rows, s.tok = rows, tok
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = s.rows[i]
        e = s.tok(r["en"], r["zh"], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, r["y"]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; print("device:", dev, flush=True)
    gp = load("llm_labels_gpt55.jsonl")
    gold_events = set(json.loads((HERE / "gold_events.json").read_text()))
    gold = [json.loads(l) for l in (HERE / "gold_consensus.jsonl").read_text().splitlines() if l.strip()]
    train = [{"en": r["en_text"], "zh": r["zh_text"], "y": C2I[lab4(r)]}
             for r in gp.values() if r["event_id"] not in gold_events and lab4(r) in CLASSES]
    test = [{"en": r["en_text"], "zh": r["zh_text"], "y": C2I[r["label"]]} for r in gold]
    print(f"train={len(train)} gold={len(test)} dist={dict(Counter(CLASSES[r['y']] for r in test))}", flush=True)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    torch.manual_seed(0); np.random.seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=4).to(dev)
    cnt = Counter(r["y"] for r in train)
    raw = np.sqrt(np.array([len(train) / max(cnt.get(i, 1), 1) for i in range(4)], float)); raw /= raw.mean()
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(raw, dtype=torch.float, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    tr = DataLoader(DS(train, tok), batch_size=BS, shuffle=True)
    total_steps = EPOCHS * len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)  # 10% warmup
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for b, y in tr:
            b = {k: v.to(dev) for k, v in b.items()}; y = torch.tensor(y).to(dev)
            opt.zero_grad(); loss = lossf(model(**b).logits, y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                    # grad clip
            opt.step(); sched.step(); tot += loss.item()
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(tr):.3f}", flush=True)

    te = DataLoader(DS(test, tok), batch_size=BS); model.eval(); pr, gt = [], []
    with torch.no_grad():
        for b, y in te:
            b = {k: v.to(dev) for k, v in b.items()}; pr += model(**b).logits.argmax(-1).cpu().tolist(); gt += list(y)
    # save per-pair gold predictions (keyed by pair_id) for CI computation
    (HERE / "gold_ce_preds.json").write_text(json.dumps(
        {gold[i]["pair_id"]: CLASSES[pr[i]] for i in range(len(gold))}))
    yp = np.array([CLASSES[i] for i in pr]); yt = np.array([CLASSES[i] for i in gt])
    per = dict(zip(CLASSES, f1_score(yt, yp, average=None, labels=CLASSES, zero_division=0)))
    row = {"acc": float((yp == yt).mean()), "macroF1": float(f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)),
           "perclass": {c: float(per[c]) for c in CLASSES}}
    print(f"\n[cross-encoder] acc={row['acc']:.3f} macroF1={row['macroF1']:.3f} "
          f"F1 no={per['no']:.3f} A={per['A']:.3f} B={per['B']:.3f} C={per['C']:.3f}")
    print("confusion:", confusion_matrix(yt, yp, labels=CLASSES).tolist())
    f = HERE / "gold_baselines_result.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d["cross-encoder"] = row; f.write_text(json.dumps(d, indent=2))
    print("appended 'cross-encoder' to gold_baselines_result.json")


if __name__ == "__main__":
    main()
