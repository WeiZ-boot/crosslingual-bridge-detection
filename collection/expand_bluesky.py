"""Back-fill Bluesky (keyless, date-bounded) across the 24 expansion + 3 top events.
Output: out/<event>/bluesky.jsonl. Bluesky is active from 2024 on, so windows here are well-covered.
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_bluesky import collect_event
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2
from collect_top3 import EVENTS as E3

HERE = Path(__file__).resolve().parent
TARGETS = [(e["id"], e["kw"], e["start"], e["end"]) for e in (E1 + E2)] + \
          [(e["id"], e["en_kw"], e["start"], e["end"]) for e in E3]


def run_one(eid, kws, start, end) -> int:
    posts = collect_event(kws, start, end, event_id=eid, max_pages=4)
    out = HERE / "out" / eid / "bluesky.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:
        for p in posts:
            w.write(p)
    return sum(1 for _ in open(out))


if __name__ == "__main__":
    only = set(sys.argv[1:]) or None
    grand = 0
    for eid, kws, start, end in TARGETS:
        if only and eid not in only:
            continue
        try:
            n = run_one(eid, kws, start, end)
        except Exception as e:
            print(f"[{eid}] ERROR {e}", flush=True); continue
        grand += n
        print(f"[{eid}] bluesky = {n}", flush=True)
    print(f"=== TOTAL bluesky = {grand} ===", flush=True)
