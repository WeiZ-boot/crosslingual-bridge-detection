"""Scale-up batch 2: English side (Reddit Arctic Shift archive) for 12 MORE bridgeable
China-West / globally-covered events whose Chinese side already exists richly in cphot.
Mirrors expand_collect.py. Output: out/<event>/reddit.jsonl (id matches cphot slug).
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_reddit_archive import collect_event
from common import JsonlWriter

HERE = Path(__file__).resolve().parent

EVENTS = [
    dict(id="evergrande_liquidation", start="2024-01-26", end="2024-02-08",
         subs=["worldnews", "economics", "geopolitics", "investing"],
         kw=["Evergrande", "Evergrande liquidation", "China property crisis", "China real estate"]),
    dict(id="openai_o1", start="2024-09-12", end="2024-09-26",
         subs=["technology", "singularity", "OpenAI", "LocalLLaMA", "artificial"],
         kw=["OpenAI o1", "o1 model", "o1 preview", "Strawberry OpenAI"]),
    dict(id="nvidia_market_cap", start="2024-06-14", end="2024-06-26",
         subs=["stocks", "wallstreetbets", "technology", "investing"],
         kw=["Nvidia most valuable", "Nvidia market cap", "Nvidia largest company", "Nvidia trillion"]),
    dict(id="scs_philippines_2024", start="2024-06-14", end="2024-06-27",
         subs=["worldnews", "geopolitics", "Philippines"],
         kw=["South China Sea", "Philippines China", "Second Thomas Shoal", "Sierra Madre"]),
    dict(id="houthi_redsea", start="2024-01-09", end="2024-01-23",
         subs=["worldnews", "geopolitics", "CredibleDefense"],
         kw=["Houthi", "Red Sea attack", "US strikes Houthi", "Houthi shipping"]),
    dict(id="shanghai_lockdown_2022", start="2022-04-01", end="2022-04-22",
         subs=["worldnews", "china", "China_irl", "Coronavirus"],
         kw=["Shanghai lockdown", "Shanghai COVID", "zero COVID Shanghai", "Shanghai quarantine"]),
    dict(id="black_myth_wukong", start="2024-08-19", end="2024-09-02",
         subs=["Games", "pcgaming", "gaming", "JRPG"],
         kw=["Black Myth Wukong", "Wukong game", "Black Myth"]),
    dict(id="tiktok_ban", start="2024-04-22", end="2024-05-04",
         subs=["technology", "worldnews", "politics", "tiktok"],
         kw=["TikTok ban", "TikTok divest", "ByteDance ban", "TikTok law"]),
    dict(id="pla_rocket_force_purge", start="2023-12-26", end="2024-01-12",
         subs=["worldnews", "geopolitics", "LessCredibleDefence"],
         kw=["Rocket Force purge", "PLA purge", "China defense minister", "Li Shangfu"]),
    dict(id="unitree_gala_2025", start="2025-01-28", end="2025-02-10",
         subs=["technology", "robotics", "interestingasfuck", "Damnthatsinteresting"],
         kw=["Unitree robot", "China robot gala", "Spring Festival robot", "dancing robots China"]),
    dict(id="spacex_launch_opinion", start="2024-06-04", end="2024-06-16",
         subs=["space", "SpaceXLounge", "spacex", "technology"],
         kw=["Starship", "SpaceX Starship", "Starship flight", "Starship splashdown"]),
    dict(id="nezha2_movie", start="2025-02-08", end="2025-02-26",
         subs=["movies", "boxoffice", "China_irl"],
         kw=["Ne Zha 2", "Nezha 2", "Chinese box office", "Nezha movie"]),
]


def run_one(ev: dict) -> int:
    posts = collect_event(ev["subs"], ev["kw"], ev["start"], ev["end"],
                          event_id=ev["id"], with_comments=True, limit=100)
    posts = [p for p in posts if p.timestamp and ev["start"] <= p.timestamp[:10] <= ev["end"]]
    out = HERE / "out" / ev["id"] / "reddit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:
        for p in posts:
            w.write(p)
    return len(posts)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    grand = 0
    for ev in EVENTS:
        if only and ev["id"] not in only:
            continue
        try:
            n = run_one(ev)
        except Exception as e:
            print(f"[{ev['id']}] ERROR {e}", flush=True); continue
        grand += n
        print(f"[{ev['id']}] reddit in-window posts = {n}", flush=True)
    print(f"=== TOTAL new EN posts (batch 2) = {grand} ===", flush=True)
