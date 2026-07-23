"""Expansion candidate generation: pair the NEW English anchors (out/<event>/reddit.jsonl)
with the already-rich cphot CHINESE posts for the same event, and persist an annotation set.

Why no temporal gate: cphot ZH timestamps are unreliable (many placeholder 2020-01-01), so we
pair on EVENT MEMBERSHIP + BGE-M3 similarity RANKING (top-k ZH per EN anchor, no floor). The
temporal axis (direction/parallel) is recovered later only where ZH timestamps are trustworthy.

Output:
  out/<event>/candidates_expand.jsonl  (per event)
  annotation_set_expand2.jsonl          (combined, schema = classify_pairs input)
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

EVENTS = ["evergrande_liquidation","openai_o1","nvidia_market_cap","scs_philippines_2024",
          "houthi_redsea","shanghai_lockdown_2022","black_myth_wukong","tiktok_ban",
          "pla_rocket_force_purge","unitree_gala_2025","spacex_launch_opinion","nezha2_movie"]

TOPK = 5          # top-k ZH per EN anchor
ZH_CAP = 1200     # cap ZH posts encoded per event (highest-quality subset)
PAIR_CAP = 140    # cap pairs persisted per event (highest sim) -> balanced annotation set


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_zh(event_id):
    f = CPHOT / f"{event_id}_001.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    posts = d.get("posts", []) if isinstance(d, dict) else []
    seen, out = set(), []
    for p in posts:
        t = (p.get("text") or "").strip()
        t = re.sub(r"\s+", " ", t)
        if len(t) < 10:
            continue
        key = t[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append({"post_id": p.get("post_id", ""), "text": t,
                    "platform": p.get("platform", "weibo"), "timestamp": p.get("timestamp")})
    return out[:ZH_CAP]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"BGE-M3 on {dev}; topk={TOPK} zh_cap={ZH_CAP} pair_cap={PAIR_CAP}\n")
    m = SentenceTransformer("BAAI/bge-m3", device=dev)
    m.max_seq_length = 512

    combined = []
    summary = []
    for eid in EVENTS:
        enf = OUT / eid / "reddit.jsonl"
        if not enf.exists():
            print(f"[{eid}] no EN file, skip"); continue
        en = read_jsonl(enf)
        zh = load_zh(eid)
        if not en or not zh:
            print(f"[{eid}] en={len(en)} zh={len(zh)} -> skip"); continue
        ev = m.encode([r["text"] for r in en], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        zv = m.encode([r["text"] for r in zh], batch_size=64, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)
        sims = ev @ zv.T  # [en, zh]
        pairs = []
        for i in range(len(en)):
            idx = np.argsort(-sims[i])[:TOPK]
            for j in idx:
                pairs.append((float(sims[i][j]), i, int(j)))
        # dedup by (i,j) already unique; cap by highest sim
        pairs.sort(reverse=True)
        # de-dup repeated zh so the set isn't dominated by one viral ZH post
        zseen, kept = {}, []
        for s, i, j in pairs:
            if zseen.get(j, 0) >= 3:      # a ZH post matched to at most 3 EN anchors
                continue
            zseen[j] = zseen.get(j, 0) + 1
            kept.append((s, i, j))
            if len(kept) >= PAIR_CAP:
                break
        rows = []
        for s, i, j in kept:
            rows.append({
                "pair_id": f"{eid}_{i}_{j}", "event_id": eid,
                "sim": round(s, 4), "band": "expand",
                "en_platform": "reddit", "en_text": en[i]["text"][:400],
                "zh_platform": zh[j]["platform"], "zh_text": zh[j]["text"][:400],
            })
        (OUT / eid / "candidates_expand.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        combined.extend(rows)
        slo = min((r["sim"] for r in rows), default=0)
        shi = max((r["sim"] for r in rows), default=0)
        summary.append((eid, len(en), len(zh), len(rows), slo, shi))
        print(f"[{eid}] en={len(en)} zh={len(zh)} pairs={len(rows)} sim={slo:.2f}-{shi:.2f}", flush=True)

    (HERE / "annotation_set_expand2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in combined))
    print(f"\n=== persisted {len(combined)} pairs across {len(summary)} events "
          f"-> annotation_set_expand2.jsonl ===")


if __name__ == "__main__":
    main()
