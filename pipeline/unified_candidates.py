"""Unified candidate generation across all 70 events. Pools EN (reddit+x+youtube+hackernews+
bluesky) x ZH (cphot archive for cphot events, else weibo+bilibili+zhihu+douyin), BGE-M3
encodes, and emits top-k cross-lingual candidate pairs (no similarity floor; sim is a ranking
feature). Output: out/<event>/candidates_v2.jsonl + annotation_set_v2.jsonl (classify_pairs schema).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CPHOT = HERE / "../../../07-code/M-BridgeNet/data/cphot/raw"
EN_PLATS = ["reddit", "x", "youtube", "hackernews", "bluesky", "telegram"]
ZH_PLATS = ["weibo", "bilibili", "zhihu", "douyin"]
EN_CAP, ZH_CAP = 1000, 1000     # per-event encode cap (longest texts kept)
TOPK = 5                         # top-k ZH per EN anchor
ZH_MAX_MATCH = 3                 # a ZH post matches at most this many EN anchors
PAIR_CAP = 200                   # pairs persisted per event (highest sim)
MINLEN = 8


def _clean(t):
    return re.sub(r"\s+", " ", (t or "").strip())


def load_en(eid):
    seen, rows = set(), []
    for plat in EN_PLATS:
        f = OUT / eid / f"{plat}.jsonl"
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            t = _clean(r.get("text"))
            if len(t) < MINLEN or t[:80] in seen:
                continue
            seen.add(t[:80])
            rows.append({"text": t, "platform": plat, "ts": r.get("timestamp")})
    rows.sort(key=lambda r: -len(r["text"]))
    return rows[:EN_CAP]


def load_zh(eid):
    f = CPHOT / f"{eid}_001.json"
    seen, rows = set(), []
    if f.is_file():
        for p in json.loads(f.read_text()).get("posts", []):
            t = _clean(p.get("text"))
            if len(t) < MINLEN or t[:60] in seen:
                continue
            seen.add(t[:60]); rows.append({"text": t, "platform": p.get("platform", "weibo"),
                                           "ts": p.get("timestamp")})
    else:
        for plat in ZH_PLATS:
            pf = OUT / eid / f"{plat}.jsonl"
            if not pf.exists():
                continue
            for l in pf.read_text().splitlines():
                if not l.strip():
                    continue
                r = json.loads(l); t = _clean(r.get("text"))
                if len(t) < MINLEN or t[:60] in seen:
                    continue
                seen.add(t[:60]); rows.append({"text": t, "platform": plat, "ts": r.get("timestamp")})
    rows.sort(key=lambda r: -len(r["text"]))
    return rows[:ZH_CAP]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer("BAAI/bge-m3", device=dev); m.max_seq_length = 512
    events = sorted([d.name for d in OUT.iterdir() if d.is_dir()])
    combined, summ = [], []
    for eid in events:
        en, zh = load_en(eid), load_zh(eid)
        if len(en) < 3 or len(zh) < 3:
            print(f"[{eid}] en={len(en)} zh={len(zh)} -> skip", flush=True); continue
        ev = m.encode([r["text"] for r in en], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        zv = m.encode([r["text"] for r in zh], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        sims = ev @ zv.T
        pairs = []
        for i in range(len(en)):
            for j in np.argsort(-sims[i])[:TOPK]:
                pairs.append((float(sims[i][int(j)]), i, int(j)))
        pairs.sort(reverse=True)
        zc, kept = {}, []
        for s, i, j in pairs:
            if zc.get(j, 0) >= ZH_MAX_MATCH:
                continue
            zc[j] = zc.get(j, 0) + 1
            kept.append((s, i, j))
            if len(kept) >= PAIR_CAP:
                break
        rows = [{"pair_id": f"{eid}_{i}_{j}", "event_id": eid, "sim": round(s, 4), "band": "v2",
                 "en_platform": en[i]["platform"], "en_text": en[i]["text"][:400],
                 "zh_platform": zh[j]["platform"], "zh_text": zh[j]["text"][:400]} for s, i, j in kept]
        (OUT / eid / "candidates_v2.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        combined.extend(rows)
        lo = min((r["sim"] for r in rows), default=0); hi = max((r["sim"] for r in rows), default=0)
        summ.append((eid, len(en), len(zh), len(rows)))
        print(f"[{eid}] en={len(en)} zh={len(zh)} pairs={len(rows)} sim={lo:.2f}-{hi:.2f}", flush=True)
    (HERE / "annotation_set_v2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in combined))
    print(f"\n=== {len(combined)} pairs across {len(summ)} events -> annotation_set_v2.jsonl ===", flush=True)


if __name__ == "__main__":
    main()
