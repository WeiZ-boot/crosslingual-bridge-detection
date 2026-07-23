"""Phase 0 — build a unified per-event corpus: English anchors + Chinese cphot posts.

For each clean event:
  EN side = in-window Reddit + X posts (YouTube optional; no reliable timestamp)
  ZH side = all cphot posts for that event (already event-scoped)
Writes out/<event>/corpus_en.jsonl and corpus_zh.jsonl with a normalized schema:
  {id, text, platform, lang, timestamp, source, event_id}
These feed Phase 1 (BGE-M3 candidate pairing).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CPHOT = HERE.parents[2] / "07-code/M-BridgeNet/data/cphot/raw"
OUT = HERE / "out"

# events too weak to use (decided in DATASET.md)
DROP = {"taiwan_strait_drill_2025", "us_china_sanctions"}


def _in_window(ts: str | None, start: str, end: str) -> bool:
    if not ts:
        return False
    d = _norm_date(ts)
    return bool(d) and start <= d <= end


def _norm_date(ts: str | None) -> str | None:
    if not ts:
        return None
    try:  # twitter format
        return datetime.strptime(ts, "%a %b %d %H:%M:%S %z %Y").date().isoformat()
    except ValueError:
        return ts[:10]


def _load_en(event_id: str, start: str, end: str, include_youtube: bool) -> list[dict]:
    rows = []
    plats = ["reddit", "x"] + (["youtube"] if include_youtube else [])
    for plat in plats:
        f = OUT / event_id / f"{plat}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            p = json.loads(line)
            # youtube has no reliable timestamp -> keep all; others must be in-window
            if plat != "youtube" and not _in_window(p.get("timestamp"), start, end):
                continue
            rows.append({
                "id": f"{plat}_{p['post_id']}",
                "text": p["text"],
                "platform": plat,
                "lang": "en",
                "timestamp": _norm_date(p.get("timestamp")),
                "source": "anchor",
                "event_id": event_id,
            })
    return rows


def _load_zh(event_id: str) -> list[dict]:
    f = CPHOT / f"{event_id}_001.json"
    if f.is_dir():
        f = f / f"{event_id}_001.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    rows = []
    for p in d.get("posts", []):
        if not p.get("text"):
            continue
        rows.append({
            "id": f"zh_{p['post_id']}",
            "text": p["text"],
            "platform": p.get("platform", ""),
            "lang": "zh",
            "timestamp": (p.get("timestamp") or "")[:10] or None,
            "source": "cphot",
            "event_id": event_id,
        })
    return rows


def main():
    cfg = yaml.safe_load((HERE / "events_clean.yaml").read_text())
    rows_summary = []
    for ev in cfg["events"]:
        eid = ev["id"]
        if eid in DROP:
            continue
        en = _load_en(eid, ev["start"], ev["end"], include_youtube=False)
        zh = _load_zh(eid)
        d = OUT / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "corpus_en.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in en))
        (d / "corpus_zh.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in zh))
        rows_summary.append((eid, len(en), len(zh)))
        print(f"  {eid:28} en={len(en):4}  zh={len(zh):4}")
    tot_en = sum(r[1] for r in rows_summary)
    tot_zh = sum(r[2] for r in rows_summary)
    print(f"\n{len(rows_summary)} events  |  EN anchors={tot_en}  ZH posts={tot_zh}")


if __name__ == "__main__":
    main()
