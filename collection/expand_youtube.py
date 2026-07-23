"""Back-fill YouTube for the 24 new events (HTTP search, no login).
Caveat: search page gives only relative publish time -> recency-biased, no hard date window.
Output: out/<event>/youtube.jsonl. Reuses event keywords from the expand_collect* modules.
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_youtube import collect_search
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2

HERE = Path(__file__).resolve().parent
ALL = E1 + E2
KW_PER_EVENT = 2


def run_one(ev: dict) -> int:
    seen, posts = set(), []
    for kw in ev["kw"][:KW_PER_EVENT]:
        try:
            got = collect_search(kw, keywords=ev["kw"], event_id=ev["id"], max_videos=30)
        except Exception as e:
            print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True); continue
        for p in got:
            if p.post_id in seen:
                continue
            seen.add(p.post_id)
            posts.append(p)
    out = HERE / "out" / ev["id"] / "youtube.jsonl"
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
        print(f"[{ev['id']}] YouTube videos = {n}", flush=True)
    print(f"=== TOTAL new YouTube videos = {grand} ===", flush=True)
