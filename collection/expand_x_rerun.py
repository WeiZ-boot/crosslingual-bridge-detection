"""Re-run X for the events throttled during the 24-event batch — one event at a time with
pauses to dodge rate limits. Appends + dedups into the existing out/<event>/x.jsonl, so we
keep whatever was already collected and only add new tweets. More keywords + more scroll.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

from collect_x import collect_search
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2

HERE = Path(__file__).resolve().parent
PROFILE = str(HERE / "x_profile")
BYID = {e["id"]: e for e in (E1 + E2)}

# events whose X count was depressed by mid-batch rate limiting
THIN = ["syria_assad_fall", "nvidia_market_cap", "taiwan_lai_twostate", "nvidia_h20_china",
        "nk_troops_russia", "fujian_carrier_trial", "pla_rocket_force_purge", "dji_us_ban",
        "shanghai_lockdown_2022", "philippines_scs_collision"]

KW_PER_EVENT = 3
SCROLL = 6
QUERY_GAP = 10        # seconds between keyword queries
EVENT_GAP = 50        # seconds between events (let X cool down)


def run_one(ev: dict) -> tuple[int, int]:
    out = HERE / "out" / ev["id"] / "x.jsonl"
    before = sum(1 for _ in open(out)) if out.exists() else 0
    seen, posts = set(), []
    for i, kw in enumerate(ev["kw"][:KW_PER_EVENT]):
        try:
            got = collect_search(kw, PROFILE, keywords=ev["kw"], event_id=ev["id"],
                                 scroll_rounds=SCROLL, since=ev["start"], until=ev["end"])
        except Exception as e:
            print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True)
            got = []
        for p in got:
            if p.post_id in seen:
                continue
            seen.add(p.post_id)
            posts.append(p)
        if i < KW_PER_EVENT - 1:
            time.sleep(QUERY_GAP)
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:      # append + dedup vs existing
        for p in posts:
            w.write(p)
    after = sum(1 for _ in open(out))
    return before, after


if __name__ == "__main__":
    ids = sys.argv[1:] or THIN
    for j, eid in enumerate(ids):
        ev = BYID.get(eid)
        if not ev:
            print(f"[{eid}] not found", flush=True); continue
        b, a = run_one(ev)
        print(f"[{eid}] X {b} -> {a}  (+{a-b})", flush=True)
        if j < len(ids) - 1:
            time.sleep(EVENT_GAP)
    print("=== X re-run done ===", flush=True)
