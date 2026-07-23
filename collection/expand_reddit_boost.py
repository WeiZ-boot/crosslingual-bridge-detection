"""Boost thin Reddit cells (<100) with WIDER subreddit nets + more keywords + wider windows.
Appends to the existing out/<event>/reddit.jsonl (JsonlWriter dedups), so re-running only adds
new unique posts. Targets the expansion events whose Reddit count came in under ~100.
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_reddit_archive import collect_event
from common import Post, JsonlWriter

HERE = Path(__file__).resolve().parent

# enriched configs: broad general + topical subreddits, more keywords, slightly wider windows
BOOST = [
    dict(id="dji_us_ban", start="2024-08-28", end="2024-09-25",
         subs=["technology", "drones", "geopolitics", "gadgets", "hardware", "worldnews",
               "Multicopter", "dronefishing", "news"],
         kw=["DJI", "DJI ban", "DJI drone", "Countering CCP Drones", "drone ban", "DJI Mavic", "Chinese drone"]),
    dict(id="evergrande_liquidation", start="2024-01-18", end="2024-02-14",
         subs=["worldnews", "economics", "geopolitics", "investing", "stocks", "China",
               "finance", "REBubble", "news", "business"],
         kw=["Evergrande", "China property", "China real estate", "China debt", "property developer",
             "China economy", "China housing", "liquidation"]),
    dict(id="fujian_carrier_trial", start="2024-04-22", end="2024-05-16",
         subs=["worldnews", "geopolitics", "WarshipPorn", "LessCredibleDefence", "China", "navy",
               "warcollege", "CredibleDefense", "NonCredibleDefense", "Military"],
         kw=["Fujian carrier", "China carrier", "Type 003", "aircraft carrier China", "PLAN carrier",
             "China navy", "China warship", "carrier sea trial"]),
    dict(id="joint_sword_2024b", start="2024-10-06", end="2024-10-26",
         subs=["worldnews", "geopolitics", "taiwan", "China", "LessCredibleDefence",
               "CredibleDefense", "NonCredibleDefense", "Military", "news"],
         kw=["Taiwan drills", "Joint Sword", "PLA Taiwan", "China Taiwan military", "Taiwan blockade",
             "China encircle Taiwan", "Taiwan strait", "China military exercise"]),
    dict(id="nvidia_h20_china", start="2025-04-06", end="2025-04-30",
         subs=["technology", "geopolitics", "hardware", "stocks", "nvidia", "semiconductors",
               "wallstreetbets", "investing", "news"],
         kw=["Nvidia H20", "Nvidia China", "H20 chip", "chip export China", "Nvidia ban", "AI chip China",
             "export control", "Nvidia restriction"]),
    dict(id="nvidia_market_cap", start="2024-06-08", end="2024-06-30",
         subs=["stocks", "wallstreetbets", "technology", "investing", "StockMarket",
               "ValueInvesting", "news", "business"],
         kw=["Nvidia", "Nvidia market cap", "Nvidia valuation", "Nvidia most valuable", "Nvidia stock",
             "NVDA", "Nvidia trillion", "most valuable company"]),
    dict(id="philippines_scs_collision", start="2024-06-08", end="2024-06-30",
         subs=["worldnews", "geopolitics", "Philippines", "China", "navy", "CredibleDefense",
               "NonCredibleDefense", "Military", "news"],
         kw=["South China Sea", "Philippines China", "Second Thomas Shoal", "Sabina Shoal",
             "Sierra Madre", "Philippine coast guard", "Scarborough", "China coast guard"]),
    dict(id="pla_rocket_force_purge", start="2023-12-18", end="2024-01-16",
         subs=["worldnews", "geopolitics", "LessCredibleDefence", "China", "CredibleDefense",
               "NonCredibleDefense", "news"],
         kw=["Rocket Force", "PLA purge", "China defense minister", "Li Shangfu", "China military purge",
             "Li Yuchao", "China generals", "China corruption military"]),
    dict(id="sixthgen_fighter_2024", start="2024-12-18", end="2025-01-12",
         subs=["worldnews", "geopolitics", "LessCredibleDefence", "FighterJets", "China",
               "NonCredibleDefense", "CredibleDefense", "aviation", "Military"],
         kw=["China sixth generation", "J-36", "Chengdu fighter", "China stealth", "white emperor",
             "J-50", "China jet", "sixth gen fighter", "China warplane"]),
    dict(id="taiwan_lai_twostate", start="2024-09-28", end="2024-10-20",
         subs=["worldnews", "geopolitics", "taiwan", "China", "news", "internationalpolitics"],
         kw=["Lai Ching-te", "Taiwan sovereignty", "two-state", "Taiwan national day",
             "Taiwan president", "cross-strait", "Taiwan independence", "Taiwan China"]),
    dict(id="unitree_gala_2025", start="2025-01-22", end="2025-02-14",
         subs=["technology", "robotics", "interestingasfuck", "Damnthatsinteresting", "Futurology",
               "China", "singularity", "gadgets", "videos"],
         kw=["Unitree", "China robot", "Spring Festival robot", "dancing robots", "humanoid robot",
             "robot gala", "Chinese robot", "robot dance"]),
    dict(id="zhuhai_airshow_2024", start="2024-11-06", end="2024-11-22",
         subs=["geopolitics", "LessCredibleDefence", "China", "aviation", "FighterJets",
               "NonCredibleDefense", "worldnews", "CredibleDefense"],
         kw=["Zhuhai Airshow", "Airshow China", "J-35", "China air show", "J-20",
             "China military aircraft", "China stealth", "China fighter"]),
]


def run_one(ev: dict) -> tuple[int, int]:
    out_path = HERE / "out" / ev["id"] / "reddit.jsonl"
    before = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    posts = collect_event(ev["subs"], ev["kw"], ev["start"], ev["end"],
                          event_id=ev["id"], with_comments=True, limit=100)
    posts = [p for p in posts if p.timestamp and ev["start"] <= p.timestamp[:10] <= ev["end"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out_path)) as w:   # appends + dedups against existing
        for p in posts:
            w.write(p)
    after = sum(1 for _ in open(out_path))
    return before, after


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for ev in BOOST:
        if only and ev["id"] not in only:
            continue
        try:
            b, a = run_one(ev)
        except Exception as e:
            print(f"[{ev['id']}] ERROR {e}", flush=True); continue
        print(f"[{ev['id']}] reddit {b} -> {a}  (+{a-b})", flush=True)
    print("=== reddit boost done ===", flush=True)
