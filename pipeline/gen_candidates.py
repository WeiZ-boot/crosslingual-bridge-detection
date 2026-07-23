"""Phase 1 — cross-lingual candidate pair generation with BGE-M3.

For each event: encode EN anchors + ZH cphot posts with BGE-M3 (multilingual), compute
cosine similarity, and emit top-k ZH candidates per EN anchor above a threshold.

IMPORTANT (CHARTER): similarity is ONLY for candidate generation to make human annotation
tractable. It is NOT the label. The bridge label + A/B/C type come from human annotators.

Output per event: out/<event>/candidates.jsonl  {en_id,en_text,zh_id,zh_text,sim,event_id}
Plus a yield report: how many anchors have >=1 candidate above threshold (= does the event
actually contain cross-lingual bridges?).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"loading BAAI/bge-m3 on {dev} ...")
        _MODEL = SentenceTransformer("BAAI/bge-m3", device=dev)
        # BGE-M3 defaults to 8192 tokens -> huge attention buffer (MPS OOM).
        # Social posts are short; 512 is ample and keeps attention small.
        _MODEL.max_seq_length = 512
    return _MODEL


def _read(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text().strip():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def encode(texts: list[str]) -> np.ndarray:
    m = get_model()
    emb = m.encode(texts, batch_size=64, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True)
    return emb


def process_event(eid: str, topk: int, thresh: float) -> dict:
    en = _read(OUT / eid / "corpus_en.jsonl")
    zh = _read(OUT / eid / "corpus_zh.jsonl")
    if not en or not zh:
        return {"event": eid, "en": len(en), "zh": len(zh), "candidates": 0,
                "anchors_with_bridge": 0, "yield": 0.0, "mean_top1": 0.0}

    en_emb = encode([r["text"] for r in en])
    zh_emb = encode([r["text"] for r in zh])
    sims = en_emb @ zh_emb.T                      # cosine (normalized) [n_en, n_zh]

    cands = []
    top1s = []
    anchors_with_bridge = 0
    k = min(topk, len(zh))
    for i, row in enumerate(sims):
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]
        top1s.append(float(row[idx[0]]))
        kept = [j for j in idx if row[j] >= thresh]
        if kept:
            anchors_with_bridge += 1
        for j in kept:
            cands.append({
                "en_id": en[i]["id"], "en_text": en[i]["text"][:300],
                "en_platform": en[i]["platform"],
                "zh_id": zh[j]["id"], "zh_text": zh[j]["text"][:300],
                "zh_platform": zh[j]["platform"],
                "sim": round(float(row[j]), 4),
                "event_id": eid,
            })
    (OUT / eid / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cands))
    return {"event": eid, "en": len(en), "zh": len(zh), "candidates": len(cands),
            "anchors_with_bridge": anchors_with_bridge,
            "yield": round(anchors_with_bridge / len(en), 2),
            "mean_top1": round(float(np.mean(top1s)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--only", nargs="*", help="limit to these event ids")
    args = ap.parse_args()

    events = sorted(d.name for d in OUT.iterdir()
                    if d.is_dir() and (d / "corpus_en.jsonl").exists())
    if args.only:
        events = [e for e in events if e in args.only]

    print(f"Phase 1: {len(events)} events, topk={args.topk}, thresh={args.thresh}\n")
    rows = []
    for eid in events:
        r = process_event(eid, args.topk, args.thresh)
        rows.append(r)
        print(f"  {r['event']:28} en={r['en']:4} zh={r['zh']:4} "
              f"cands={r['candidates']:5} yield={r['yield']:.2f} top1={r['mean_top1']:.3f}")
    rows.sort(key=lambda r: -r["yield"])
    print("\n=== ranked by yield (anchors with >=1 candidate >= thresh) ===")
    for r in rows:
        flag = "" if r["yield"] >= 0.3 else "  <-- LOW yield"
        print(f"  {r['event']:28} yield={r['yield']:.2f} cands={r['candidates']:5} "
              f"top1={r['mean_top1']:.3f}{flag}")
    tot = sum(r["candidates"] for r in rows)
    print(f"\nTOTAL candidate pairs: {tot}  across {len(rows)} events")
    (HERE / "phase1_yield.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
