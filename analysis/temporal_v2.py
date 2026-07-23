"""Temporal layer for the 70-event labeled set (llm_labels_v2.jsonl).

Re-derives each pair's EN/ZH timestamps by index (load_en/load_zh are deterministic, so en[i]/zh[j]
align with the pair_id eid_i_j used at candidate-gen time), then tags:
  - direction: EN->ZH / ZH->EN / concurrent  (|lead| < CONC_H hours = concurrent)
  - reframed_label: not_related | parallel_coverage | bridge_A/B/C (+ direction)
A genuine bridge = related & directional & typed; parallel_coverage = both sides ~simultaneous.

Writes llm_labels_v2_directed.jsonl + llm_labels_v2_reframed.jsonl, prints the breakdown.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from unified_candidates import load_en, load_zh

HERE = Path(__file__).resolve().parent
LABELS = HERE / "llm_labels_v2.jsonl"
CONC_H = 24.0


def parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        try:
            return float(ts) / (1000.0 if ts > 1e12 else 1.0)
        except (TypeError, ValueError):
            return None
    s = str(ts).strip()
    # ISO 8601
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except ValueError:
        pass
    # Twitter "Wed Dec 15 23:58:26 +0000 2024"
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp()
    except ValueError:
        return None


def plausible(ep, lo=1.4e9, hi=1.9e9):
    # 2014-05 .. 2030-03 — drops placeholder/garbage timestamps
    return ep is not None and lo < ep < hi


def main():
    labels = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()]
    by_ev = defaultdict(list)
    for r in labels:
        by_ev[r["event_id"]].append(r)

    directed = []
    for eid, rows in by_ev.items():
        en = load_en(eid)
        zh = load_zh(eid)
        en_ts = [parse_ts(r.get("ts")) for r in en]
        zh_ts = [parse_ts(r.get("ts")) for r in zh]
        for r in rows:
            try:
                _, i, j = r["pair_id"].rsplit("_", 2)
                i, j = int(i), int(j)
                ets = en_ts[i] if i < len(en_ts) else None
                zts = zh_ts[j] if j < len(zh_ts) else None
            except (ValueError, IndexError):
                ets = zts = None
            lead = None
            if plausible(ets) and plausible(zts):
                lead = (zts - ets) / 3600.0   # hours ZH after EN
            if lead is None:
                direction = "na"
            elif lead > CONC_H:
                direction = "EN->ZH"
            elif lead < -CONC_H:
                direction = "ZH->EN"
            else:
                direction = "concurrent"
            directed.append({**r, "lead_h": (round(lead, 1) if lead is not None else None),
                             "direction": direction})

    (HERE / "llm_labels_v2_directed.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in directed))

    # reframed
    reframed = []
    for r in directed:
        if r.get("is_bridge") == "no":
            lab, d = "not_related", "na"
        elif r["direction"] == "na":
            lab, d = f"bridge_{r.get('bridge_type')}_untimed", "na"   # typed but no usable time
        elif r["direction"] == "concurrent":
            lab, d = "parallel_coverage", "concurrent"
        else:
            lab, d = f"bridge_{r.get('bridge_type')}", r["direction"]
        reframed.append({**r, "reframed_label": lab, "reframed_direction": d})
    (HERE / "llm_labels_v2_reframed.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in reframed))

    # report
    bridges = [r for r in directed if r.get("is_bridge") == "yes"]
    timed = [r for r in bridges if r["direction"] != "na"]
    print(f"pairs={len(directed)}  bridges={len(bridges)}  bridges_with_usable_time={len(timed)} "
          f"({len(timed)/max(len(bridges),1):.0%})")
    dirc = Counter(r["direction"] for r in timed)
    print("direction among timed bridges:", dict(dirc))
    n = max(len(timed), 1)
    en2zh, zh2en, conc = dirc.get("EN->ZH", 0), dirc.get("ZH->EN", 0), dirc.get("concurrent", 0)
    print(f"  EN->ZH {en2zh/n:.1%}  ZH->EN {zh2en/n:.1%}  concurrent(parallel) {conc/n:.1%}")
    print("reframed_label dist:", dict(Counter(r["reframed_label"] for r in reframed)))
    # C among genuine directional bridges vs parallel
    gen = [r for r in reframed if r["reframed_label"].startswith("bridge_") and "untimed" not in r["reframed_label"]]
    print(f"\ngenuine directional bridges={len(gen)}; among them: "
          f"{dict(Counter(r['bridge_type'] for r in gen))}")
    print("=== temporal layer done -> llm_labels_v2_directed.jsonl, llm_labels_v2_reframed.jsonl ===")


if __name__ == "__main__":
    main()
