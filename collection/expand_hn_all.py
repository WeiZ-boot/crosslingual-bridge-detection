"""Expand Hacker News across ALL existing events (the original 22 have zero HN).
Keyless, date-bounded, HTTP. Targets events under TARGET; appends + dedups into
out/<event>/hackernews.jsonl. HN is a tech/finance/geopolitics audience, so non-tech events
(swimming, cotton boycott) will stay near zero — that's correct platform-fit, not a failure.

Configs unified from events_clean.yaml (25) + expand_collect{,2} (24) + collect_top3 (3).
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

from collect_hackernews import collect_event
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2
from collect_top3 import EVENTS as E3

HERE = Path(__file__).resolve().parent
TARGET = 100
MAX_PAGES = 8


def _configs():
    cfg = {}
    for e in yaml.safe_load(open(HERE / "events_clean.yaml"))["events"]:
        cfg[e["id"]] = (e.get("keywords") or [], e["start"], e["end"])
    for e in (E1 + E2):
        cfg.setdefault(e["id"], (e["kw"], e["start"], e["end"]))
    for e in E3:
        cfg.setdefault(e["id"], (e["en_kw"], e["start"], e["end"]))
    return cfg


def run_one(eid, kws, start, end) -> tuple[int, int]:
    out = HERE / "out" / eid / "hackernews.jsonl"
    before = sum(1 for _ in open(out)) if out.exists() else 0
    posts = collect_event(kws, start, end, event_id=eid, with_comments=True, max_pages=MAX_PAGES)
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:
        for p in posts:
            w.write(p)
    return before, sum(1 for _ in open(out))


if __name__ == "__main__":
    cfg = _configs()
    only = set(sys.argv[1:]) or None
    todo = []
    for eid, (kws, start, end) in cfg.items():
        if only and eid not in only:
            continue
        f = HERE / "out" / eid / "hackernews.jsonl"
        cur = sum(1 for _ in open(f)) if f.exists() else 0
        if cur >= TARGET:
            print(f"[{eid}] skip (already {cur})", flush=True); continue
        todo.append((eid, kws, start, end))
    print(f"=== expanding HN for {len(todo)} events ===", flush=True)
    grand = 0
    for eid, kws, start, end in todo:
        try:
            b, a = run_one(eid, kws, start, end)
        except Exception as e:
            print(f"[{eid}] ERROR {e}", flush=True); continue
        grand += (a - b)
        print(f"[{eid}] hn {b} -> {a}  (+{a-b})", flush=True)
    print(f"=== HN expand-all done, +{grand} ===", flush=True)
