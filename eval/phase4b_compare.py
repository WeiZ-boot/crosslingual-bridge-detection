"""Phase 4b cross-encoder (XLM-R), nano vs GPT-5.5 silver, SAME event-grouped split.

Tests whether the silver source changes the cross-encoder's ability to recover Type-C
(counter-narrative) on unseen events. Mirrors phase4b_cross_encoder_v2.py exactly; loops over both
label files. Run with the torch env:
  ~/opt/anaconda3/envs/dify/bin/python phase4b_compare.py
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


def load_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("is_bridge") == "no":
            y = "no"
        elif r.get("bridge_type") in ("A", "B", "C"):
            y = r["bridge_type"]
        else:
            continue
        rows.append({"event": r["event_id"], "en": r["en_text"], "zh": r["zh_text"], "y": C2I[y]})
    return rows


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


def run(train_rows, test_rows, tag, device):
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=4).to(device)
    cnt = Counter(r["y"] for r in train_rows)
    w = torch.tensor([len(train_rows) / (4 * cnt.get(i, 1)) for i in range(4)], dtype=torch.float, device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    tr = DataLoader(PairDS(train_rows, tok), batch_size=BS, shuffle=True)
    te = DataLoader(PairDS(test_rows, tok), batch_size=BS)
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for batch, y in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = torch.tensor(y).to(device)
            opt.zero_grad()
            loss = lossf(model(**batch).logits, y)
            loss.backward(); opt.step(); tot += loss.item()
        print(f"  [{tag}] epoch {ep+1}/{EPOCHS} loss={tot/len(tr):.3f}", flush=True)
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for batch, y in te:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds += model(**batch).logits.argmax(-1).cpu().tolist()
            gts += list(y)
    yp = np.array([CLASSES[i] for i in preds]); yt = np.array([CLASSES[i] for i in gts])
    macro = f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0)
    perc = f1_score(yt, yp, average=None, labels=CLASSES, zero_division=0)
    print(f"\n[{tag}] acc={(yp==yt).mean():.3f}  macro-F1={macro:.3f}")
    print("  per-class F1:", {c: round(f, 3) for c, f in zip(CLASSES, perc)})
    print("  confusion (true rows / pred cols no/A/B/C):"); print(confusion_matrix(yt, yp, labels=CLASSES))
    return {"acc": float((yp == yt).mean()), "macroF1": float(macro),
            "perclass": {c: float(f) for c, f in zip(CLASSES, perc)}}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device, flush=True)
    summary = {}
    for tag, path in [("nano", "llm_labels_v2.jsonl"), ("gpt55", "llm_labels_gpt55.jsonl")]:
        rows = load_rows(path)
        groups = np.array([r["event"] for r in rows]); y = np.array([r["y"] for r in rows])
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        gtr, gte = next(gss.split(rows, y, groups))
        train_rows = [rows[i] for i in gtr]; test_rows = [rows[i] for i in gte]
        print(f"\n########## {tag}: {path} ##########")
        print(f"rows={len(rows)} dist={dict(Counter(CLASSES[r['y']] for r in rows))}")
        print(f"train={len(train_rows)} test={len(test_rows)} "
              f"test-C={sum(1 for r in test_rows if r['y']==C2I['C'])} train-C={sum(1 for r in train_rows if r['y']==C2I['C'])}", flush=True)
        summary[tag] = run(train_rows, test_rows, f"cross-enc {tag}", device)

    print("\n\n==================== CROSS-ENCODER: NANO vs GPT-5.5 ====================")
    for c in CLASSES:
        print(f"  F1[{c}]   nano={summary['nano']['perclass'][c]:.3f}   gpt55={summary['gpt55']['perclass'][c]:.3f}")
    print(f"  macro-F1 nano={summary['nano']['macroF1']:.3f}   gpt55={summary['gpt55']['macroF1']:.3f}")
    print(f"  acc      nano={summary['nano']['acc']:.3f}   gpt55={summary['gpt55']['acc']:.3f}")
    (HERE / "phase4b_compare_result.json").write_text(json.dumps(summary, indent=2))
    print("\nsaved phase4b_compare_result.json")


if __name__ == "__main__":
    main()
