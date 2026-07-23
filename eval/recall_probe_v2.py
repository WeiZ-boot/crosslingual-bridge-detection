"""Low-floor recall probe (R8 redo). The v2 candidate set is high-sim by construction (top-k),
so it can't test what a 0.60 similarity GATE would drop. Here we deliberately sample pairs in the
[0.45, 0.60) band for C-rich events, to measure the bridge / A-B-C rate that a 0.60 floor discards.
Writes recall_probe_input.jsonl (for nano labeling).
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from unified_candidates import load_en, load_zh

random.seed(0)
HERE = Path(__file__).resolve().parent
C_RICH = ["gaza_oct7_2023", "fukushima_release", "philippines_scs_collision", "scs_philippines_2024",
          "eileen_gu_identity", "india_pakistan_2025", "nvidia_h20_china", "china_spy_balloon",
          "pelosi_taiwan", "syria_assad_fall"]
LO, HI = 0.45, 0.60
PER_EVENT = 40
CAP = 700


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer("BAAI/bge-m3", device=dev); m.max_seq_length = 512
    out = []
    for eid in C_RICH:
        en = load_en(eid)[:CAP]; zh = load_zh(eid)[:CAP]
        if len(en) < 5 or len(zh) < 5:
            continue
        ev = m.encode([r["text"] for r in en], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        zv = m.encode([r["text"] for r in zh], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        sims = ev @ zv.T
        band = [(i, j) for i in range(len(en)) for j in range(len(zh))
                if LO <= sims[i, j] < HI]
        random.shuffle(band)
        for i, j in band[:PER_EVENT]:
            out.append({"pair_id": f"{eid}_lo_{i}_{j}", "event_id": eid,
                        "sim": round(float(sims[i, j]), 4), "band": "lowfloor",
                        "en_platform": en[i]["platform"], "en_text": en[i]["text"][:400],
                        "zh_platform": zh[j]["platform"], "zh_text": zh[j]["text"][:400]})
        print(f"[{eid}] band-pairs sampled: {min(len(band), PER_EVENT)}", flush=True)
    (HERE / "recall_probe_input.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out))
    print(f"=== {len(out)} low-band pairs -> recall_probe_input.jsonl ===")


if __name__ == "__main__":
    main()
