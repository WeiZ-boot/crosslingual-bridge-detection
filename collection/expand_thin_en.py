"""Boost EN for thin original events that never got the Reddit wide-boost or Bluesky.
Reddit (wide subreddits + keywords) + Bluesky (keyless). Windows pulled from events_clean.yaml.
Idempotent (JsonlWriter now dedups vs existing file). Appends into out/<event>/{reddit,bluesky}.jsonl.
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

from collect_reddit_archive import collect_event as reddit_event
from collect_bluesky import collect_event as bsky_event
from common import JsonlWriter

HERE = Path(__file__).resolve().parent
WIN = {e["id"]: (e["start"], e["end"]) for e in yaml.safe_load(open(HERE / "events_clean.yaml"))["events"]}

# enriched Reddit configs for thin original events (wide nets + more keywords)
CFG = {
    "eu_china_ev_tariffs": dict(
        subs=["worldnews", "geopolitics", "economics", "europe", "electricvehicles", "china", "news"],
        kw=["China EV tariff", "EU tariff China", "electric vehicle tariff", "China car tariff", "EV trade war"]),
    "huawei_mate60_launch": dict(
        subs=["technology", "geopolitics", "hardware", "China", "Android", "news", "chips"],
        kw=["Huawei Mate 60", "Kirin 9000s", "Huawei chip", "SMIC 7nm", "Huawei sanction"]),
    "xinjiang_cotton": dict(
        subs=["worldnews", "geopolitics", "china", "news", "fashion", "europe"],
        kw=["Xinjiang cotton", "H&M China", "Nike Xinjiang", "Xinjiang boycott", "BCI cotton", "Uyghur cotton"]),
    "likeqiang_death": dict(
        subs=["worldnews", "geopolitics", "China", "news"],
        kw=["Li Keqiang", "Li Keqiang death", "Chinese premier", "China former premier"]),
    "saudi_iran_china_mediation": dict(
        subs=["worldnews", "geopolitics", "news", "MiddleEastNews"],
        kw=["Saudi Iran China", "Iran Saudi deal", "China Middle East", "Saudi Iran agreement", "Beijing Saudi Iran"]),
    "us_china_sanctions": dict(
        subs=["worldnews", "geopolitics", "economics", "news", "technology"],
        kw=["US China sanctions", "China sanctions", "export controls China", "chip sanctions China", "China tech ban"]),
    "us_china_reckoning": dict(
        subs=["worldnews", "geopolitics", "news", "economics"],
        kw=["US China decoupling", "US China tensions", "China cold war", "US China trade", "China containment"]),
    "taiwan_strait_drill_2025": dict(
        subs=["worldnews", "geopolitics", "taiwan", "China", "news", "LessCredibleDefence"],
        kw=["Taiwan strait drill", "PLA Taiwan drill", "China Taiwan military", "Taiwan blockade", "China encircle Taiwan"]),
    "eileen_gu_identity": dict(
        subs=["worldnews", "olympics", "china", "sports", "news", "FreeSkiing"],
        kw=["Eileen Gu", "Gu Ailing", "Eileen Gu China", "Eileen Gu citizenship", "Eileen Gu Olympics"]),
    "pan_zhanle_100m": dict(
        subs=["Swimming", "olympics", "sports", "worldnews", "news"],
        kw=["Pan Zhanle", "100m freestyle", "swimming world record", "Pan Zhanle Olympics", "Pan Zhanle doping"]),
}


def run_one(eid: str) -> tuple[int, int, int, int]:
    start, end = WIN.get(eid, (None, None))
    c = CFG[eid]
    rf = HERE / "out" / eid / "reddit.jsonl"
    bf = HERE / "out" / eid / "bluesky.jsonl"
    rb = sum(1 for _ in open(rf)) if rf.exists() else 0
    bb = sum(1 for _ in open(bf)) if bf.exists() else 0
    # Reddit
    rposts = reddit_event(c["subs"], c["kw"], start, end, event_id=eid, with_comments=True, limit=100)
    rposts = [p for p in rposts if p.timestamp and start <= p.timestamp[:10] <= end]
    with JsonlWriter(str(rf)) as w:
        for p in rposts:
            w.write(p)
    # Bluesky (only meaningful for 2024+ windows, but harmless otherwise)
    bposts = bsky_event(c["kw"], start, end, event_id=eid, max_pages=4)
    with JsonlWriter(str(bf)) as w:
        for p in bposts:
            w.write(p)
    ra = sum(1 for _ in open(rf)); ba = sum(1 for _ in open(bf))
    return rb, ra, bb, ba


if __name__ == "__main__":
    only = set(sys.argv[1:]) or None
    for eid in CFG:
        if only and eid not in only:
            continue
        rb, ra, bb, ba = run_one(eid)
        print(f"[{eid}] reddit {rb}->{ra} (+{ra-rb})  bluesky {bb}->{ba} (+{ba-bb})", flush=True)
    print("=== thin EN boost done ===", flush=True)
