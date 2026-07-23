"""Back-fill the X (Twitter) platform for the 24 new events (Reddit-only so far).
Uses the logged-in x_profile + date-bounded since/until search. Output: out/<event>/x.jsonl.
Reuses the event windows/keywords defined in expand_collect.py and expand_collect2.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_x import collect_search
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2

HERE = Path(__file__).resolve().parent
PROFILE = str(HERE / "x_profile")
ALL = E1 + E2
KW_PER_EVENT = 2          # number of keyword queries per event (rate/time budget)
SCROLL = 5


def run_one(ev: dict) -> int:
    seen, posts = set(), []
    for kw in ev["kw"][:KW_PER_EVENT]:
        try:
            got = collect_search(kw, PROFILE, keywords=ev["kw"], event_id=ev["id"],
                                 scroll_rounds=SCROLL, since=ev["start"], until=ev["end"])
        except Exception as e:
            print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True); continue
        for p in got:
            if p.post_id in seen:
                continue
            seen.add(p.post_id)
            # since/until already bounds the window server-side; X created_at is
            # Twitter format (not ISO), so no client-side date filter here.
            posts.append(p)
    out = HERE / "out" / ev["id"] / "x.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:
        for p in posts:
            w.write(p)
    return len(posts)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    grand = 0
    for ev in ALL:
        if only and ev["id"] not in only:
            continue
        n = run_one(ev)
        grand += n
        print(f"[{ev['id']}] X posts = {n}", flush=True)
    print(f"=== TOTAL new X posts = {grand} ===", flush=True)
