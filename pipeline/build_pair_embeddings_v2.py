"""BGE-M3 pair embeddings for the 70-event labeled set, aligned to llm_labels_v2.jsonl line order.
Saves pair_emb_v2.npz: eu [N,1024] (EN), zv [N,1024] (ZH), both L2-normalized. Cached for the
phase4/5/6-style experiments (pair features = [u, v, |u-v|, u*v]).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
LABELS = HERE / "llm_labels_v2.jsonl"
OUT = HERE / "pair_emb_v2.npz"


def main():
    rows = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()]
    en = [r["en_text"] for r in rows]
    zh = [r["zh_text"] for r in rows]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"encoding {len(rows)} pairs on {dev}", flush=True)
    m = SentenceTransformer("BAAI/bge-m3", device=dev); m.max_seq_length = 512
    eu = m.encode(en, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=True).astype(np.float32)
    zv = m.encode(zh, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=True).astype(np.float32)
    np.savez_compressed(OUT, eu=eu, zv=zv)
    print(f"saved {OUT.name}  eu={eu.shape} zv={zv.shape}", flush=True)


if __name__ == "__main__":
    main()
