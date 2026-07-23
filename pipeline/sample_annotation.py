"""Phase 2 — stratified sampling of candidate pairs for human A/B/C annotation.

Draws a balanced ~2-3k-pair annotation set from the 29k candidate pool:
  - 22 final events (drops low-yield / weak events)
  - 3 similarity bands per event (low/mid/high) so annotators see translations,
    reframings, AND hard near-threshold negatives.
  - dedup exact (en_text, zh_text); cap each EN anchor to <=2 distinct ZH partners
    so one viral anchor's near-duplicates don't dominate.

Output: annotation_set.jsonl + annotation_set.csv with empty label columns.
Similarity is ONLY the sampling signal — annotators assign the bridge label + A/B/C type.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

DROP = {"taiwan_strait_drill_2025", "us_china_sanctions", "us_china_reckoning"}

BANDS = [("low", 0.60, 0.64), ("mid", 0.64, 0.70), ("high", 0.70, 1.01)]
PER_BAND = 36                     # target pairs per band per event (cap)
MAX_PER_ANCHOR = 2               # distinct ZH partners kept per EN anchor
SEED = 20260611


def _band(sim: float) -> str | None:
    for name, lo, hi in BANDS:
        if lo <= sim < hi:
            return name
    return None


def _pid(c: dict) -> str:
    h = hashlib.md5((c["en_id"] + "|" + c["zh_id"]).encode()).hexdigest()[:10]
    return f"{c['event_id']}_{h}"


def main():
    rng = random.Random(SEED)
    events = sorted(d.name for d in OUT.iterdir()
                    if d.is_dir() and (d / "candidates.jsonl").exists()
                    and d.name not in DROP)
    sampled = []
    for eid in events:
        cands = [json.loads(l) for l in (OUT / eid / "candidates.jsonl").read_text().splitlines() if l.strip()]
        # dedup exact pair text; cap per anchor
        seen_pair = set()
        per_anchor = defaultdict(int)
        pool = []
        rng.shuffle(cands)
        for c in cands:
            key = (c["en_text"].strip(), c["zh_text"].strip())
            if key in seen_pair:
                continue
            if per_anchor[c["en_id"]] >= MAX_PER_ANCHOR:
                continue
            seen_pair.add(key)
            per_anchor[c["en_id"]] += 1
            pool.append(c)
        # stratify by band
        by_band = defaultdict(list)
        for c in pool:
            b = _band(c["sim"])
            if b:
                by_band[b].append(c)
        picked = []
        for name, _, _ in BANDS:
            rng.shuffle(by_band[name])
            picked.extend(by_band[name][:PER_BAND])
        for c in picked:
            sampled.append({
                "pair_id": _pid(c),
                "event_id": c["event_id"],
                "sim": c["sim"],
                "band": _band(c["sim"]),
                "en_platform": c["en_platform"],
                "en_text": c["en_text"],
                "zh_platform": c["zh_platform"],
                "zh_text": c["zh_text"],
                # ---- annotator fills these ----
                "is_bridge": "",          # yes / no
                "bridge_type": "",        # A / B / C  (blank if not a bridge)
                "confidence": "",         # 1-3
                "note": "",
            })
        print(f"  {eid:28} pool={len(pool):5} sampled={len(picked):3}")

    rng.shuffle(sampled)
    (HERE / "annotation_set.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sampled))
    with (HERE / "annotation_set.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sampled[0].keys()))
        w.writeheader()
        w.writerows(sampled)

    from collections import Counter
    print(f"\nTOTAL sampled: {len(sampled)} pairs across {len(events)} events")
    print("by band:", dict(Counter(r["band"] for r in sampled)))
    print("wrote annotation_set.jsonl + annotation_set.csv")


if __name__ == "__main__":
    main()
