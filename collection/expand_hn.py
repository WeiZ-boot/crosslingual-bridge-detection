"""Back-fill Hacker News (keyless, date-bounded) across the 24 new events.
Output: out/<event>/hackernews.jsonl. Reuses event windows/keywords from expand_collect*.
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_hackernews import collect_event
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2

HERE = Path(__file__).resolve().parent
ALL = E1 + E2


def run_one(ev: dict) -> int:
    posts = collect_event(ev["kw"], ev["start"], ev["end"], event_id=ev["id"],
                          with_comments=True, max_pages=5)
    out = HERE / "out" / ev["id"] / "hackernews.jsonl"
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
        try:
            n = run_one(ev)
        except Exception as e:
            print(f"[{ev['id']}] ERROR {e}", flush=True); continue
        grand += n
        print(f"[{ev['id']}] HN posts = {n}", flush=True)
    print(f"=== TOTAL new HN posts = {grand} ===", flush=True)
