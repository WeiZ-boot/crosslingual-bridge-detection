"""Framework adjustment — Issue 3: temporal precedence of LLM-labeled bridges.

"Migration" (EN->ZH) implies the English post PRECEDES the Chinese one. We never checked this.
This joins each labeled pair to full-precision timestamps (EN from out/<event>/{reddit,x}.jsonl,
ZH from cphot raw) and asks:
  - Of pairs the LLM calls bridges, what fraction have EN before ZH (consistent with EN->ZH)?
  - Lead-time distribution; per-type (A/B/C) patterns; per-event direction (find ZH->EN events).
  - How many "bridges" violate precedence (ZH clearly first) -> spurious / reverse-direction.

Day-granularity is too coarse (21-day windows); we parse full timestamps to hours.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CPHOT = HERE.parents[2] / "07-code/M-BridgeNet/data/cphot/raw"
LABELS = HERE / "llm_labels_v3.jsonl"


def parse_ts(ts):
    if not ts:
        return None
    try:  # twitter "Wed Jun 10 03:38:42 +0000 2026"
        return datetime.strptime(ts, "%a %b %d %H:%M:%S %z %Y").astimezone(timezone.utc)
    except (ValueError, TypeError):
        pass
    try:  # ISO (reddit, cphot)
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def build_en_ts():
    """corpus_en id = '<platform>_<post_id>'  -> full timestamp from raw out files."""
    m = {}
    for d in OUT.iterdir():
        if not d.is_dir():
            continue
        for plat in ("reddit", "x"):
            f = d / f"{plat}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                p = json.loads(line)
                m[f"{plat}_{p['post_id']}"] = parse_ts(p.get("timestamp"))
    return m


def build_zh_ts():
    m = {}
    for f in CPHOT.glob("*.json"):
        path = f
        if f.is_dir():
            path = f / f.name
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        for p in d.get("posts", []):
            m[f"zh_{p['post_id']}"] = parse_ts(p.get("timestamp"))
    return m


def build_pid_to_ids():
    """Recompute pair_id -> (en_id, zh_id) from candidate files (hash matches sampler)."""
    m = {}
    for d in OUT.iterdir():
        f = d / "candidates.jsonl" if d.is_dir() else None
        if not f or not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            h = hashlib.md5((c["en_id"] + "|" + c["zh_id"]).encode()).hexdigest()[:10]
            m[f"{c['event_id']}_{h}"] = (c["en_id"], c["zh_id"])
    return m


def main():
    en_ts, zh_ts = build_en_ts(), build_zh_ts()
    pid2ids = build_pid_to_ids()
    print(f"en_ts={len(en_ts)} zh_ts={len(zh_ts)} pid_map={len(pid2ids)}")

    rows = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()]
    recs = []
    for r in rows:
        ids = pid2ids.get(r["pair_id"])
        if not ids:
            continue
        et, zt = en_ts.get(ids[0]), zh_ts.get(ids[1])
        if et is None or zt is None:
            continue
        delta_h = (zt - et).total_seconds() / 3600.0   # >0 => EN precedes ZH (EN->ZH)
        y = "no" if r.get("is_bridge") == "no" else r.get("bridge_type")
        recs.append({"event": r["event_id"], "y": y, "delta_h": delta_h})
    print(f"pairs with both timestamps: {len(recs)} / {len(rows)}")

    bridges = [x for x in recs if x["y"] in ("A", "B", "C")]
    d = np.array([x["delta_h"] for x in bridges])
    print(f"\n=== BRIDGES (n={len(bridges)}) temporal precedence ===")
    print(f"  EN precedes ZH (delta>0):   {(d > 0).mean():.1%}")
    print(f"  ZH precedes EN (delta<0):   {(d < 0).mean():.1%}")
    print(f"  |delta|<24h (≈concurrent):  {(np.abs(d) < 24).mean():.1%}")
    print(f"  median lead (EN->ZH) hours: {np.median(d):.1f}  ({np.median(d)/24:.1f} days)")

    print("\n=== by type: EN-precedes rate | median lead (days) ===")
    for t in ("A", "B", "C"):
        dt = np.array([x["delta_h"] for x in bridges if x["y"] == t])
        if len(dt):
            print(f"  {t}: n={len(dt):4}  EN-first={ (dt>0).mean():.1%}  med_lead={np.median(dt)/24:+.1f}d")

    print("\n=== per-event direction (median lead days; negative = ZH leads = reverse) ===")
    evs = sorted({x["event"] for x in bridges})
    rows_e = []
    for e in evs:
        de = np.array([x["delta_h"] for x in bridges if x["event"] == e])
        rows_e.append((e, len(de), np.median(de) / 24, (de > 0).mean()))
    for e, n, med, rate in sorted(rows_e, key=lambda z: z[2]):
        flag = "  <-- ZH-leads (reverse/bidir)" if med < 0 else ""
        print(f"  {e:28} n={n:3} med={med:+.1f}d EN-first={rate:.0%}{flag}")

    # spurious check: bridges where ZH precedes EN by >3 days (cannot be EN->ZH migration)
    viol = [x for x in bridges if x["delta_h"] < -72]
    print(f"\nbridges with ZH preceding EN by >3d (NOT EN->ZH): {len(viol)} ({len(viol)/len(bridges):.1%})")


if __name__ == "__main__":
    main()
