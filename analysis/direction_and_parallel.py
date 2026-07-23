"""Framework adjustment — direction tagging + parallel-coverage probe.

(1) DIRECTION: tag every labeled bridge EN->ZH / ZH->EN / concurrent from timestamps.
(2) PARALLEL-COVERAGE PROBE: distinguish genuine bridging (content flowed) from parallel coverage
    (both platforms independently report the same fact). Two signals:
      - temporal: concurrent (|lead|<24h) = parallel-coverage suspect; clear lead = bridge candidate.
      - specificity: a genuine migration matches ONE specific counterpart (peaked top-1 similarity);
        parallel coverage of a viral fact matches MANY posts diffusely (flat similarity).

Writes llm_labels_directed.jsonl (adds direction) + prints the probe.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import temporal_analysis as ta   # reuse timestamp builders

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LABELS = HERE / "llm_labels_v3.jsonl"
CONC_H = 24.0   # |lead| < this = concurrent


def build_pid_sim_and_anchor_sims():
    """pid -> sim; and en_id -> sorted list of its candidate sims (for specificity)."""
    pid_sim, anchor_sims = {}, defaultdict(list)
    import hashlib
    for d in OUT.iterdir():
        f = d / "candidates.jsonl" if d.is_dir() else None
        if not f or not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            h = hashlib.md5((c["en_id"] + "|" + c["zh_id"]).encode()).hexdigest()[:10]
            pid_sim[f"{c['event_id']}_{h}"] = c["sim"]
            anchor_sims[c["en_id"]].append(c["sim"])
    for k in anchor_sims:
        anchor_sims[k].sort(reverse=True)
    return pid_sim, anchor_sims


def main():
    en_ts, zh_ts = ta.build_en_ts(), ta.build_zh_ts()
    pid2ids = ta.build_pid_to_ids()
    pid_sim, anchor_sims = build_pid_sim_and_anchor_sims()

    rows = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()]
    recs, out_lines = [], []
    for r in rows:
        ids = pid2ids.get(r["pair_id"])
        if not ids:
            continue
        et, zt = en_ts.get(ids[0]), zh_ts.get(ids[1])
        if et is None or zt is None:
            continue
        lead_h = (zt - et).total_seconds() / 3600.0    # >0 => EN precedes
        if abs(lead_h) < CONC_H:
            direction = "concurrent"
        elif lead_h > 0:
            direction = "EN->ZH"
        else:
            direction = "ZH->EN"
        y = "no" if r.get("is_bridge") == "no" else r.get("bridge_type")
        sims = anchor_sims.get(ids[0], [])
        peaked = (sims[0] - np.mean(sims[1:5])) if len(sims) >= 2 else 0.0
        recs.append({"event": r["event_id"], "y": y, "dir": direction, "lead_h": lead_h,
                     "sim": pid_sim.get(r["pair_id"], np.nan), "peaked": peaked,
                     "n_cand": len(sims)})
        out_lines.append(json.dumps({**r, "direction": direction,
                                     "lead_hours": round(lead_h, 1)}, ensure_ascii=False))
    (HERE / "llm_labels_directed.jsonl").write_text("\n".join(out_lines))

    bridges = [x for x in recs if x["y"] in ("A", "B", "C")]
    n = len(bridges)
    print(f"labeled bridges with timestamps: {n}\n")

    print("=== (1) DIRECTION distribution ===")
    from collections import Counter
    dc = Counter(x["dir"] for x in bridges)
    for k in ("EN->ZH", "ZH->EN", "concurrent"):
        print(f"  {k:11}: {dc[k]:5} ({dc[k]/n:.1%})")

    print("\n=== direction BY TYPE (row=type) ===")
    print(f"  {'type':4} {'EN->ZH':>8} {'ZH->EN':>8} {'concur':>8}  n")
    for t in ("A", "B", "C"):
        sub = [x for x in bridges if x["y"] == t]
        c = Counter(x["dir"] for x in sub)
        m = len(sub) or 1
        print(f"  {t:4} {c['EN->ZH']/m:8.0%} {c['ZH->EN']/m:8.0%} {c['concurrent']/m:8.0%}  {len(sub)}")

    print("\n=== (2a) PARALLEL-COVERAGE suspects (concurrent share) ===")
    conc = [x for x in bridges if x["dir"] == "concurrent"]
    print(f"  concurrent bridges: {len(conc)} ({len(conc)/n:.1%}) <- parallel-coverage suspects")
    print(f"  directional bridges: {n-len(conc)} ({(n-len(conc))/n:.1%}) <- genuine-bridge candidates")

    print("\n=== (2b) mean similarity & specificity (peaked top-1) by direction ===")
    for k in ("EN->ZH", "ZH->EN", "concurrent"):
        sub = [x for x in bridges if x["dir"] == k]
        if sub:
            print(f"  {k:11}: sim={np.nanmean([x['sim'] for x in sub]):.3f}  "
                  f"peaked={np.mean([x['peaked'] for x in sub]):.3f}  n={len(sub)}")
    print("  (higher peaked = anchor has ONE specific match = migration-like;")
    print("   low peaked + concurrent = matches many posts = parallel-coverage-like)")

    # combined suspect: concurrent AND diffuse (low peakedness)
    med_peak = np.median([x["peaked"] for x in bridges])
    susp = [x for x in bridges if x["dir"] == "concurrent" and x["peaked"] < med_peak]
    print(f"\n=== combined PARALLEL-COVERAGE suspect (concurrent & below-median peakedness) ===")
    print(f"  {len(susp)} of {n} bridges ({len(susp)/n:.1%}) are strong parallel-coverage suspects")
    print(f"  by type: " + ", ".join(
        f"{t}={sum(1 for x in susp if x['y']==t)}" for t in ('A','B','C')))
    print("\nwrote llm_labels_directed.jsonl")


if __name__ == "__main__":
    main()
