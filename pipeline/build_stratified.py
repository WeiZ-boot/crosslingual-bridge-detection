"""Design a STRATIFIED gold: balanced across events and oversampling rare classes (C, no).

The current consensus gold is alphabetically biased (13 early events) and misses the flagship
high-C dispute events. This builds a stratified TARGET over all 70 events -- per event, quota by
GPT-5.5-predicted class {C:up to 8, no:5, A:4, B:6} -- then reports, for each target pair, whether
Claude has already labeled it (=> usable consensus now) or it's a GAP needing annotation.

Outputs:
  stratified_targets.jsonl   -- the selected pairs (pair_id,event_id,gpt_label,en,zh) to annotate
  stratified_gap.jsonl       -- targets NOT yet Claude-labeled (the annotation to-do, C-first)
  gold_stratified.jsonl      -- targets that ARE Claude<->GPT-5.5 consensus (usable gold right now)
Run:  python3 build_stratified.py
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUOTA = {"C": 8, "no": 5, "A": 4, "B": 6}   # per-event target (oversamples C & no vs natural ~12%/15%)


def load(p):
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(HERE / p) if l.strip())}


def l4(r):
    return r["bridge_type"] if r["is_bridge"] == "yes" else ("no" if r["is_bridge"] == "no" else None)


def main():
    gp, cl = load("llm_labels_gpt55.jsonl"), load("llm_labels_claude.jsonl")
    def cons(k):  # Claude<->GPT-5.5 consensus label, or None
        return l4(cl[k]) if (k in cl and l4(cl[k]) and l4(cl[k]) == l4(gp[k])) else None

    by_ev = defaultdict(lambda: defaultdict(list))
    for k, r in gp.items():
        c = l4(r)
        if c:
            by_ev[r["event_id"]][c].append(k)

    targets = []
    for ev in sorted(by_ev):
        for cls, cap in QUOTA.items():
            pool = by_ev[ev][cls]
            # prefer pairs Claude already labeled (free consensus), then fill from the rest
            pool = sorted(pool, key=lambda k: (k not in cl,))   # already-labeled first
            for k in pool[:cap]:
                targets.append(k)

    rows = [{"pair_id": k, "event_id": gp[k]["event_id"], "gpt_label": l4(gp[k]),
             "en_text": gp[k]["en_text"], "zh_text": gp[k]["zh_text"]} for k in targets]
    (HERE / "stratified_targets.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    gap = [r for r in rows if r["pair_id"] not in cl]
    # C-first ordering so annotating the gap maximally tightens the C CI
    gap.sort(key=lambda r: {"C": 0, "no": 1, "A": 2, "B": 3}[r["gpt_label"]])
    (HERE / "stratified_gap.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gap))

    goldrows = [{"pair_id": r["pair_id"], "event_id": r["event_id"], "en_text": r["en_text"],
                 "zh_text": r["zh_text"], "label": cons(r["pair_id"])}
                for r in rows if cons(r["pair_id"])]
    (HERE / "gold_stratified.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in goldrows))

    print(f"stratified TARGET: {len(rows)} pairs across {len(by_ev)} events")
    print(f"  target class mix: {dict(Counter(r['gpt_label'] for r in rows))}")
    print(f"usable gold NOW (Claude consensus): {len(goldrows)} pairs, "
          f"C={sum(1 for r in goldrows if r['label']=='C')} no={sum(1 for r in goldrows if r['label']=='no')}, "
          f"events={len(set(r['event_id'] for r in goldrows))}")
    print(f"ANNOTATION GAP (need Claude labels): {len(gap)} pairs, "
          f"class mix {dict(Counter(r['gpt_label'] for r in gap))}, events={len(set(r['event_id'] for r in gap))}")
    # gap C by event (the flagship disputes to label first)
    gapC = Counter(r["event_id"] for r in gap if r["gpt_label"] == "C")
    print("\ngap C-target pairs by event (label these first, top 20):")
    for e, n in gapC.most_common(20):
        print(f"  {e:26s} C-gap={n}")


if __name__ == "__main__":
    main()
