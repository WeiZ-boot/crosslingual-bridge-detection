"""Boost thin X cells using the SECOND X account (Chrome-imported cookies in x2_cookies.json).
Fresh account => not throttled. Appends + dedups into out/<event>/x.jsonl.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

from collect_x import collect_search
from common import JsonlWriter
from expand_collect import EVENTS as E1
from expand_collect2 import EVENTS as E2

HERE = Path(__file__).resolve().parent
COOKIES = json.load(open(HERE / "x2_cookies.json"))
BYID = {e["id"]: e for e in (E1 + E2)}

# expansion events whose X count is still thin (<~50)
THIN = ["taiwan_lai_twostate", "syria_assad_fall", "nvidia_market_cap", "pla_rocket_force_purge",
        "unitree_gala_2025", "sixthgen_fighter_2024", "nk_troops_russia", "openai_o1",
        "tiktok_ban", "dji_us_ban", "joint_sword_2024b"]

KW = 3
SCROLL = 6


def run_one(ev: dict) -> tuple[int, int]:
    out = HERE / "out" / ev["id"] / "x.jsonl"
    before = sum(1 for _ in open(out)) if out.exists() else 0
    seen, posts = set(), []
    for i, kw in enumerate(ev["kw"][:KW]):
        try:
            got = collect_search(kw, keywords=ev["kw"], event_id=ev["id"], scroll_rounds=SCROLL,
                                 since=ev["start"], until=ev["end"], cookies=COOKIES)
        except Exception as e:
            print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True); got = []
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
    ids = sys.argv[1:] or THIN
    for j, eid in enumerate(ids):
        ev = BYID.get(eid)
        if not ev:
            print(f"[{eid}] not found", flush=True); continue
        b, a = run_one(ev)
        print(f"[{eid}] X(acct2) {b} -> {a}  (+{a-b})", flush=True)
        if j < len(ids) - 1:
            time.sleep(20)
    print("=== acct2 X boost done ===", flush=True)
