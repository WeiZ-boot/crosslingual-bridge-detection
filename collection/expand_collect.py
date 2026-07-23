"""Scale-up: collect the ENGLISH side (Reddit Arctic Shift archive, no login) for the
bridgeable China-West dispute events whose CHINESE side already exists richly in cphot
(~2-3k posts each). These become new EN×ZH event pairs for the enlarged annotation set.

Each event: English keywords + subreddits + a tight date window around the real event date.
Output: out/<event>/reddit.jsonl  (unified Post schema), ready for candidate generation.
"""
from __future__ import annotations
import sys
from pathlib import Path

from collect_reddit_archive import collect_event
from common import JsonlWriter

HERE = Path(__file__).resolve().parent

# id must match cphot raw slug (minus _001) so candidate gen can pair EN with cphot ZH.
EVENTS = [
    dict(id="taiwan_election_2024", start="2024-01-08", end="2024-01-20",
         subs=["worldnews", "geopolitics", "taiwan", "China_irl"],
         kw=["Taiwan election", "Lai Ching-te", "Taiwan president", "DPP Taiwan"]),
    dict(id="joint_sword_2024b", start="2024-10-10", end="2024-10-22",
         subs=["worldnews", "geopolitics", "taiwan", "LessCredibleDefence"],
         kw=["Taiwan drills", "Joint Sword", "PLA Taiwan", "China military Taiwan"]),
    dict(id="taiwan_lai_twostate", start="2024-10-05", end="2024-10-16",
         subs=["worldnews", "geopolitics", "taiwan"],
         kw=["Lai two-state", "Taiwan sovereignty", "Lai Ching-te speech", "Taiwan national day"]),
    dict(id="fujian_carrier_trial", start="2024-04-28", end="2024-05-12",
         subs=["worldnews", "geopolitics", "WarshipPorn", "LessCredibleDefence"],
         kw=["Fujian carrier", "China aircraft carrier", "Type 003", "carrier sea trial"]),
    dict(id="sixthgen_fighter_2024", start="2024-12-23", end="2025-01-08",
         subs=["worldnews", "geopolitics", "LessCredibleDefence", "FighterJets"],
         kw=["China sixth generation fighter", "J-36", "Chengdu fighter", "China stealth jet"]),
    dict(id="zhuhai_airshow_2024", start="2024-11-10", end="2024-11-19",
         subs=["geopolitics", "LessCredibleDefence", "China_irl"],
         kw=["Zhuhai Airshow", "Airshow China", "J-35", "China air show"]),
    dict(id="nvidia_h20_china", start="2025-04-12", end="2025-04-26",
         subs=["technology", "geopolitics", "hardware", "stocks"],
         kw=["Nvidia H20", "Nvidia China ban", "H20 export", "chip export China"]),
    dict(id="dji_us_ban", start="2024-09-05", end="2024-09-22",
         subs=["technology", "drones", "geopolitics"],
         kw=["DJI ban", "DJI drone ban", "Countering CCP Drones", "DJI United States"]),
    dict(id="nk_troops_russia", start="2024-10-18", end="2024-11-08",
         subs=["worldnews", "geopolitics", "UkraineWarVideoReport"],
         kw=["North Korea troops Russia", "DPRK soldiers Ukraine", "Kim troops Russia", "North Korean soldiers"]),
    dict(id="syria_assad_fall", start="2024-12-05", end="2024-12-16",
         subs=["worldnews", "geopolitics", "syriancivilwar"],
         kw=["Assad fall", "Syria Assad", "Damascus rebels", "HTS Syria"]),
    dict(id="lebanon_pager_2024", start="2024-09-16", end="2024-09-26",
         subs=["worldnews", "geopolitics"],
         kw=["Hezbollah pager", "pager explosion", "Lebanon pagers", "Israel pager attack"]),
    dict(id="philippines_scs_collision", start="2024-06-14", end="2024-06-26",
         subs=["worldnews", "geopolitics", "Philippines"],
         kw=["South China Sea collision", "Philippines China coast guard", "Second Thomas Shoal", "Sabina Shoal"]),
]


def run_one(ev: dict) -> int:
    posts = collect_event(ev["subs"], ev["kw"], ev["start"], ev["end"],
                          event_id=ev["id"], with_comments=True, limit=100)
    # keep in-window only
    posts = [p for p in posts if p.timestamp and ev["start"] <= p.timestamp[:10] <= ev["end"]]
    out = HERE / "out" / ev["id"] / "reddit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:
        for p in posts:
            w.write(p)
    return len([p for p in posts])


if __name__ == "__main__":
    only = sys.argv[1:] or None
    grand = 0
    for ev in EVENTS:
        if only and ev["id"] not in only:
            continue
        try:
            n = run_one(ev)
        except Exception as e:
            print(f"[{ev['id']}] ERROR {e}", flush=True)
            continue
        grand += n
        print(f"[{ev['id']}] reddit in-window posts = {n}", flush=True)
    print(f"=== TOTAL new EN posts = {grand} ===", flush=True)
