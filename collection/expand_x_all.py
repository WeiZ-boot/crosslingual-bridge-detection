"""Expand X across ALL existing events, rotating the two X accounts to spread load.
Account 1 = x_profile (user_data_dir); Account 2 = x2_cookies.json (Chrome-imported).
Targets events whose current X count is under TARGET; appends + dedups into out/<event>/x.jsonl.

Configs unified from events_clean.yaml (25 originals) + expand_collect{,2} (24) + collect_top3 (3).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import yaml

from collect_x import collect_search
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2
from collect_top3 import EVENTS as E3

HERE = Path(__file__).resolve().parent
PROFILE = str(HERE / "x_profile")
COOKIES = json.load(open(HERE / "x2_cookies.json"))

TARGET = 100        # skip events already at/above this many X posts
KW = 3
SCROLL = 6


def _configs():
    cfg = {}
    yml = yaml.safe_load(open(HERE / "events_clean.yaml"))["events"]
    for e in yml:
        kws = list(dict.fromkeys((e.get("x_search") or []) + (e.get("keywords") or [])))
        cfg[e["id"]] = (kws, e["start"], e["end"])
    for e in (E1 + E2):
        cfg.setdefault(e["id"], (e["kw"], e["start"], e["end"]))
    for e in E3:
        cfg.setdefault(e["id"], (e["en_kw"], e["start"], e["end"]))
    return cfg


def run_one(eid, kws, start, end, use_cookies: bool) -> tuple[int, int]:
    out = HERE / "out" / eid / "x.jsonl"
    before = sum(1 for _ in open(out)) if out.exists() else 0
    auth = {"cookies": COOKIES} if use_cookies else {"user_data_dir": PROFILE}
    seen, posts = set(), []
    for i, kw in enumerate(kws[:KW]):
        try:
            got = collect_search(kw, keywords=kws, event_id=eid, scroll_rounds=SCROLL,
                                 since=start, until=end, **auth)
        except Exception as e:
            print(f"    [{eid}|{kw}] ERR {e}", flush=True); got = []
        for p in got:
            if p.post_id not in seen:
                seen.add(p.post_id); posts.append(p)
        if i < KW - 1:
            time.sleep(8)
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
        cur = sum(1 for _ in open(HERE / "out" / eid / "x.jsonl")) if (HERE / "out" / eid / "x.jsonl").exists() else 0
        if cur >= TARGET:
            print(f"[{eid}] skip (already {cur})", flush=True); continue
        todo.append((eid, kws, start, end))
    print(f"=== expanding X for {len(todo)} events (rotating 2 accounts) ===", flush=True)
    for j, (eid, kws, start, end) in enumerate(todo):
        use_cookies = (j % 2 == 0)           # alternate accounts per event
        acct = "acct2" if use_cookies else "acct1"
        b, a = run_one(eid, kws, start, end, use_cookies)
        print(f"[{eid}] ({acct}) X {b} -> {a}  (+{a-b})", flush=True)
        if j < len(todo) - 1:
            time.sleep(25)
    print("=== X expand-all done ===", flush=True)
