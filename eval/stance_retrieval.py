"""Stance-aware retrieval for counter-narrative (C) — recover what similarity misses.

Finding (Issue 2): BGE-M3 similarity (gated OR ranked) under-retrieves counter-narrative, because C
is a STANCE/opposition relation, not high semantic overlap. Hypothesis: a multilingual NLI
CONTRADICTION signal ranks C pairs higher than similarity does.

Pools all labeled pairs (main + recall-probe + event-scoped sample), so C appears across the whole
similarity range. Compares retrieval of C by: BGE-M3 similarity vs NLI-contradiction vs fusion.
Metric: Average Precision + Recall@k for retrieving bridge_C.

Model: multilingual NLI (mDeBERTa-v3 xnli). ⚠️ labels are LLM silver.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_pool():
    seen, pool = set(), []
    for fn in ("llm_labels_v3.jsonl", "recall_probe_labels.jsonl", "es_sample_labels.jsonl"):
        f = HERE / fn
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            key = (r.get("en_text", "")[:90], r.get("zh_text", "")[:90])
            if key in seen:
                continue
            seen.add(key)
            if "sim" not in r or not r.get("en_text") or not r.get("zh_text"):
                continue
            is_c = 1 if (r.get("is_bridge") == "yes" and r.get("bridge_type") == "C") else 0
            is_bridge = 1 if r.get("is_bridge") == "yes" else 0
            pool.append({"en": r["en_text"], "zh": r["zh_text"], "sim": float(r["sim"]),
                         "is_c": is_c, "is_bridge": is_bridge})
    return pool


def contradiction_scores(pool):
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI).to(DEV).eval()
    # label order for this model: 0 entailment, 1 neutral, 2 contradiction
    lab = model.config.id2label
    contra_idx = [i for i, v in lab.items() if "contrad" in v.lower()][0]
    scores = []
    B = 32
    for s in range(0, len(pool), B):
        chunk = pool[s:s + B]
        # both directions, take max contradiction
        out = []
        for prem, hyp in (("en", "zh"), ("zh", "en")):
            enc = tok([c[prem] for c in chunk], [c[hyp] for c in chunk], truncation=True,
                      max_length=256, padding=True, return_tensors="pt").to(DEV)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits, -1)[:, contra_idx].cpu().numpy()
            out.append(p)
        scores.extend(np.maximum(out[0], out[1]).tolist())
    return np.array(scores)


def recall_at_k(score, gold, ks=(20, 50, 100, 200)):
    order = np.argsort(-score)
    g = np.array(gold)[order]
    tot = g.sum()
    return {f"R@{k}": (g[:k].sum() / tot if tot else 0) for k in ks}


def main():
    pool = load_pool()
    sim = np.array([p["sim"] for p in pool])
    isc = np.array([p["is_c"] for p in pool])
    print(f"pool={len(pool)}  C bridges={isc.sum()} ({isc.mean():.1%})")
    print("loading NLI ...")
    contra = contradiction_scores(pool)

    # standardize for fusion
    zs = (sim - sim.mean()) / sim.std()
    zc = (contra - contra.mean()) / contra.std()
    fusion = zc - zs            # high contradiction, low similarity = counter-narrative
    fusion2 = zc + zs           # additive

    print("\n=== RETRIEVE counter-narrative (C) — Average Precision ===")
    print(f"  prevalence (random)        AP={isc.mean():.3f}")
    print(f"  BGE-M3 similarity          AP={average_precision_score(isc, sim):.3f}")
    print(f"  NLI contradiction          AP={average_precision_score(isc, contra):.3f}")
    print(f"  fusion (contra - sim)      AP={average_precision_score(isc, fusion):.3f}")
    print(f"  fusion (contra + sim)      AP={average_precision_score(isc, fusion2):.3f}")

    print("\n=== Recall@k for C ===")
    for name, sc in [("similarity", sim), ("contradiction", contra), ("contra-sim", fusion)]:
        rk = recall_at_k(sc, isc)
        print(f"  {name:14} " + "  ".join(f"{k}={v:.3f}" for k, v in rk.items()))

    # correlation: is C anti-correlated with similarity?
    print(f"\n  corr(similarity, is_C) = {np.corrcoef(sim, isc)[0,1]:+.3f}")
    print(f"  corr(contradiction, is_C) = {np.corrcoef(contra, isc)[0,1]:+.3f}")
    print(f"  mean sim:   C={sim[isc==1].mean():.3f}  non-C={sim[isc==0].mean():.3f}")
    print(f"  mean contra: C={contra[isc==1].mean():.3f}  non-C={contra[isc==0].mean():.3f}")


if __name__ == "__main__":
    main()
