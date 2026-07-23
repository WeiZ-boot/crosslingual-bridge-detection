"""Candidate-recall probe (Issue 2) — does the sim>=0.60 gate miss genuine bridges?

The dataset only labeled pairs with cosine >= 0.60. Counter-narrative (C) is a STANCE relation that
often uses different vocabulary -> LOW similarity -> may sit BELOW 0.60 and never be a candidate.
This generates a sample of sub-threshold pairs [0.50, 0.60) for a few events (incl. C-rich ones) to
be LLM-labeled; comparing their bridge/type rate to the >=0.60 set estimates the recall loss.

Writes recall_probe_input.jsonl (full-field, ready for classify_pairs.py).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SAMPLE_EVENTS = ["xinjiang_cotton", "eileen_gu_identity", "china_spy_balloon",
                 "titan_submersible", "sam_altman_ousting", "us_svb_collapse"]
LO, HI = 0.50, 0.60        # just-below-gate band
PER_EVENT = 30
SEED = 11


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    rng = random.Random(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading BGE-M3 on {dev} ...")
    m = SentenceTransformer("BAAI/bge-m3", device=dev)
    m.max_seq_length = 512

    out = []
    for eid in SAMPLE_EVENTS:
        en = read(OUT / eid / "corpus_en.jsonl")
        zh = read(OUT / eid / "corpus_zh.jsonl")
        if not en or not zh:
            print(f"  {eid}: missing corpus"); continue
        eu = m.encode([r["text"] for r in en], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        zv = m.encode([r["text"] for r in zh], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        sims = eu @ zv.T
        # collect pairs in [LO, HI)
        band = []
        for i in range(len(en)):
            for j in np.where((sims[i] >= LO) & (sims[i] < HI))[0]:
                band.append((i, int(j), float(sims[i][j])))
        rng.shuffle(band)
        picked = band[:PER_EVENT]
        for i, j, s in picked:
            e, z = en[i], zh[j]
            h = hashlib.md5((e["id"] + "|" + z["id"]).encode()).hexdigest()[:10]
            out.append({
                "pair_id": f"{eid}_sub_{h}", "event_id": eid, "sim": round(s, 4),
                "band": "sub0.60",
                "en_platform": e["platform"], "en_text": e["text"][:300],
                "zh_platform": z["platform"], "zh_text": z["text"][:300],
            })
        print(f"  {eid:24} band[{LO},{HI}) pairs={len(band):6} sampled={len(picked)}")
    (HERE / "recall_probe_input.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out))
    print(f"\nwrote recall_probe_input.jsonl ({len(out)} sub-threshold pairs)")


if __name__ == "__main__":
    main()
