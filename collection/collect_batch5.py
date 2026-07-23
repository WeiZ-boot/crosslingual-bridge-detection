"""5 more globally-viral, China-relevant hot-topic events (-> 70-event dataset).
EN: Reddit + HN + Bluesky + X(rotate 2 accts) + YouTube(deep). ZH done separately via MediaCrawler.
Run a stage at a time: python collect_batch5.py <reddit|hn|bluesky|x|youtube|all>
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

from common import JsonlWriter

HERE = Path(__file__).resolve().parent

EVENTS = [
    dict(id="charlie_kirk_2025", start="2025-09-10", end="2025-09-21",
         en_kw=["Charlie Kirk", "Charlie Kirk shooting", "Charlie Kirk assassination", "Turning Point USA", "Kirk killed"],
         subs=["news", "worldnews", "politics", "Conservative"],
         zh_kw=["查理柯克", "柯克 枪击", "美国 保守派 遇刺", "查理·柯克", "柯克 遇刺"]),
    dict(id="us_govt_shutdown_2025", start="2025-10-01", end="2025-10-22",
         en_kw=["government shutdown", "US shutdown", "federal shutdown", "shutdown 2025", "SNAP shutdown"],
         subs=["politics", "news", "worldnews", "economics"],
         zh_kw=["美国 政府停摆", "美国 停摆", "联邦政府 关门", "美国 政府关门"]),
    dict(id="openai_gpt5_2025", start="2025-08-07", end="2025-08-18",
         en_kw=["GPT-5", "OpenAI GPT-5", "GPT5 launch", "ChatGPT 5"],
         subs=["technology", "singularity", "OpenAI", "artificial", "LocalLLaMA"],
         zh_kw=["GPT-5", "GPT5", "OpenAI GPT5", "ChatGPT5"]),
    dict(id="greenland_trump_2025", start="2025-01-06", end="2025-01-20",
         en_kw=["Greenland", "Trump Greenland", "buy Greenland", "Greenland annexation", "Denmark Greenland"],
         subs=["worldnews", "geopolitics", "politics", "europe"],
         zh_kw=["格陵兰", "特朗普 格陵兰", "美国 格陵兰", "格陵兰岛"]),
    dict(id="la_ice_protests_2025", start="2025-06-06", end="2025-06-18",
         en_kw=["LA protests", "ICE protests", "Los Angeles ICE", "National Guard Los Angeles", "immigration protests"],
         subs=["news", "worldnews", "politics", "LosAngeles"],
         zh_kw=["洛杉矶 抗议", "美国 移民 抗议", "洛杉矶 国民警卫队", "美国 骚乱"]),
]


def _write(eid, plat, posts):
    out = HERE / "out" / eid / f"{plat}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(str(out)) as w:
        for p in posts:
            w.write(p)
    return sum(1 for _ in open(out))


def stage_reddit():
    from collect_reddit_archive import collect_event
    for ev in EVENTS:
        posts = collect_event(ev["subs"], ev["en_kw"], ev["start"], ev["end"],
                              event_id=ev["id"], with_comments=True, limit=100)
        posts = [p for p in posts if p.timestamp and ev["start"] <= p.timestamp[:10] <= ev["end"]]
        print(f"[{ev['id']}] reddit = {_write(ev['id'],'reddit',posts)}", flush=True)


def stage_hn():
    from collect_hackernews import collect_event
    for ev in EVENTS:
        posts = collect_event(ev["en_kw"], ev["start"], ev["end"], event_id=ev["id"],
                              with_comments=True, max_pages=8)
        print(f"[{ev['id']}] hackernews = {_write(ev['id'],'hackernews',posts)}", flush=True)


def stage_bluesky():
    from collect_bluesky import collect_event
    for ev in EVENTS:
        posts = collect_event(ev["en_kw"], ev["start"], ev["end"], event_id=ev["id"], max_pages=4)
        print(f"[{ev['id']}] bluesky = {_write(ev['id'],'bluesky',posts)}", flush=True)


def stage_x():
    from collect_x import collect_search
    prof = str(HERE / "x_profile")
    cookies = json.load(open(HERE / "x2_cookies.json"))
    for j, ev in enumerate(EVENTS):
        auth = {"cookies": cookies} if j % 2 == 0 else {"user_data_dir": prof}
        seen, posts = set(), []
        for i, kw in enumerate(ev["en_kw"][:3]):
            try:
                got = collect_search(kw, keywords=ev["en_kw"], event_id=ev["id"], scroll_rounds=6,
                                     since=ev["start"], until=ev["end"], **auth)
            except Exception as e:
                print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True); got = []
            for p in got:
                if p.post_id not in seen:
                    seen.add(p.post_id); posts.append(p)
            time.sleep(8)
        print(f"[{ev['id']}] x = {_write(ev['id'],'x',posts)}", flush=True)
        if j < len(EVENTS) - 1:
            time.sleep(25)


def stage_youtube():
    from collect_youtube import collect_search_deep
    for ev in EVENTS:
        seen, posts = set(), []
        for kw in ev["en_kw"][:2]:
            try:
                for p in collect_search_deep(kw, keywords=ev["en_kw"], event_id=ev["id"], scroll_rounds=8):
                    if p.post_id not in seen:
                        seen.add(p.post_id); posts.append(p)
            except Exception as e:
                print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True)
        print(f"[{ev['id']}] youtube = {_write(ev['id'],'youtube',posts)}", flush=True)


STAGES = {"reddit": stage_reddit, "hn": stage_hn, "bluesky": stage_bluesky,
          "x": stage_x, "youtube": stage_youtube}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["reddit", "hn", "bluesky", "youtube", "x"] if arg == "all" else [arg]
    for s in order:
        print(f"\n### STAGE {s} ###", flush=True)
        STAGES[s]()
    print("\n=== batch5 EN done ===", flush=True)
