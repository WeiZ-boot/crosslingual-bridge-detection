"""Expand YouTube across ALL existing events with the deep browser+scroll collector.
Targets events whose current YouTube count is under TARGET (the original 22 were collected
with the old shallow static method). Appends + dedups into out/<event>/youtube.jsonl.

Configs unified from events_clean.yaml (25 originals) + expand_collect{,2} (24) + collect_top3 (3).
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

from collect_youtube import collect_search_deep
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2
from collect_top3 import EVENTS as E3

HERE = Path(__file__).resolve().parent
TARGET = 100
KW = 2


def _configs():
    cfg = {}
    for e in yaml.safe_load(open(HERE / "events_clean.yaml"))["events"]:
        kws = list(dict.fromkeys((e.get("youtube") or []) + (e.get("keywords") or [])))
        cfg[e["id"]] = kws
    for e in (E1 + E2):
        cfg.setdefault(e["id"], e["kw"])
    for e in E3:
        cfg.setdefault(e["id"], e["en_kw"])
    return cfg


def run_one(eid, kws) -> tuple[int, int]:
    out = HERE / "out" / eid / "youtube.jsonl"
    before = sum(1 for _ in open(out)) if out.exists() else 0
    seen, posts = set(), []
    for kw in kws[:KW]:
        try:
            got = collect_search_deep(kw, keywords=kws, event_id=eid, scroll_rounds=8)
        except Exception as e:
            print(f"    [{eid}|{kw}] ERR {e}", flush=True); got = []
        for p in got:
            if p.post_id not in seen:
                seen.add(p.post_id); posts.append(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:
        for p in posts:
            w.write(p)
    return before, sum(1 for _ in open(out))


if __name__ == "__main__":
    cfg = _configs()
    only = set(sys.argv[1:]) or None
    todo = []
    for eid, kws in cfg.items():
        if only and eid not in only:
            continue
        f = HERE / "out" / eid / "youtube.jsonl"
        cur = sum(1 for _ in open(f)) if f.exists() else 0
        if cur >= TARGET:
            print(f"[{eid}] skip (already {cur})", flush=True); continue
        todo.append((eid, kws))
    print(f"=== deep-expanding YouTube for {len(todo)} events ===", flush=True)
    grand = 0
    for eid, kws in todo:
        b, a = run_one(eid, kws)
        grand += (a - b)
        print(f"[{eid}] youtube {b} -> {a}  (+{a-b})", flush=True)
    print(f"=== YouTube expand-all done, +{grand} ===", flush=True)
