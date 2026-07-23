"""Phase 2/3 — assemble a balanced, C-enriched GOLD annotation set for humans.

Pools silver labels (original 2,157 + C-rich expansion), then builds a target-balanced set
across A / B / C / not-a-bridge. Type C is the binding constraint, so we take ALL silver-C and
match the other classes to it. Each row carries the nano silver guess + rationale as a PRE-FILL
reference (annotator confirms/corrects) plus empty gold columns.

Output: gold_annotation_set.{jsonl,csv}
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 99

# target gold-set composition (C maximized; others matched for a usable eval split)
TARGET = {"C": 250, "B": 250, "A": 250, "no": 150}


def load_silver() -> dict[str, dict]:
    pool = {}
    # v3 relabel = single combined file; fall back to v1 split files if absent
    for fn in ("llm_labels_v3.jsonl", "llm_labels.jsonl", "llm_labels_expansion.jsonl"):
        f = HERE / fn
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pool[r["pair_id"]] = r          # later file wins on dup (same label anyway)
    return pool


def main():
    rng = random.Random(SEED)
    silver = list(load_silver().values())

    buckets = defaultdict(list)
    for r in silver:
        if r.get("is_bridge") == "no":
            buckets["no"].append(r)
        elif r.get("bridge_type") in ("A", "B", "C"):
            buckets[r["bridge_type"]].append(r)

    print("silver pool:", {k: len(v) for k, v in buckets.items()})

    gold = []
    for cls, n in TARGET.items():
        rng.shuffle(buckets[cls])
        take = buckets[cls][:n]
        gold.extend(take)
        print(f"  {cls}: target {n}, took {len(take)}")

    rng.shuffle(gold)
    rows = []
    for r in gold:
        rows.append({
            "pair_id": r["pair_id"], "event_id": r["event_id"],
            "sim": r["sim"], "band": r["band"],
            "en_text": r["en_text"], "zh_text": r["zh_text"],
            # nano silver guess as pre-fill reference (NOT gold)
            "silver_is_bridge": r.get("is_bridge", ""),
            "silver_type": r.get("bridge_type", ""),
            "silver_rationale": r.get("rationale", ""),
            # ---- human annotator fills these (gold) ----
            "gold_is_bridge": "", "gold_type": "", "confidence": "", "note": "",
        })

    (HERE / "gold_annotation_set.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    with (HERE / "gold_annotation_set.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"\nGOLD set: {len(rows)} pairs")
    print("silver-class mix:", dict(Counter(
        (r['silver_type'] if r['silver_is_bridge'] == 'yes' else 'no') for r in rows)))
    print("events:", len(Counter(r['event_id'] for r in rows)))
    print("wrote gold_annotation_set.{jsonl,csv}")


if __name__ == "__main__":
    main()
