"""Re-collect YouTube with the deep browser+scroll method (captures search continuations)
for the 24 expansion events + 3 top events. Appends + dedups into out/<event>/youtube.jsonl.
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_youtube import collect_search_deep
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2
from collect_top3 import EVENTS as E3

HERE = Path(__file__).resolve().parent

# unify: (id, keywords)
TARGETS = [(e["id"], e["kw"]) for e in (E1 + E2)] + [(e["id"], e["en_kw"]) for e in E3]
KW = 2


def run_one(eid: str, kws: list[str]) -> tuple[int, int]:
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
    only = set(sys.argv[1:]) or None
    grand = 0
    for eid, kws in TARGETS:
        if only and eid not in only:
            continue
        b, a = run_one(eid, kws)
        grand += (a - b)
        print(f"[{eid}] youtube {b} -> {a}  (+{a-b})", flush=True)
    print(f"=== deep YouTube done, +{grand} total ===", flush=True)
