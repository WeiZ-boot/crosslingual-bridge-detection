"""End-to-end collection for the 3 X-rich, bridgeable hot-topic events (no cphot ZH side).
EN: Reddit archive + Hacker News + X (paced) + YouTube.  ZH: Weibo historical + Bilibili.
Run a single stage at a time:  python collect_top3.py <stage>  where stage in
  reddit | hn | x | youtube | weibo | bilibili | all
Writes out/<event>/<platform>.jsonl (unified Post schema). EN/ZH keywords per event below.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

from common import JsonlWriter

HERE = Path(__file__).resolve().parent

EVENTS = [
    dict(
        id="rednote_migration_2025", start="2025-01-13", end="2025-01-23",
        en_kw=["RedNote", "Xiaohongshu", "TikTok refugee", "Little Red Book", "TikTok ban migration"],
        subs=["technology", "TikTok", "RedNote", "tiktokrefugees", "China", "socialmedia", "news"],
        zh_kw=["小红书 TikTok", "TikTok难民", "洋抖", "美国网友 小红书", "外国人 小红书", "TikTok 小红书"],
    ),
    dict(
        id="la_wildfires_2025", start="2025-01-07", end="2025-01-19",
        en_kw=["Los Angeles wildfire", "LA fires", "Palisades fire", "Eaton fire", "California wildfire"],
        subs=["news", "worldnews", "California", "LosAngeles", "politics", "pics"],
        zh_kw=["洛杉矶 山火", "加州 大火", "洛杉矶大火", "美国 山火", "加州山火", "洛杉矶 火灾"],
    ),
    dict(
        id="india_pakistan_2025", start="2025-05-07", end="2025-05-17",
        en_kw=["India Pakistan", "Operation Sindoor", "Rafale shot down", "J-10C", "Kashmir strike"],
        subs=["worldnews", "geopolitics", "india", "pakistan", "CredibleDefense",
              "LessCredibleDefence", "NonCredibleDefense"],
        zh_kw=["印巴 冲突", "歼10", "枭龙", "印度 巴基斯坦", "阵风 击落", "J10C"],
    ),
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
        posts = collect_event(ev["en_kw"], ev["start"], ev["end"], event_id=ev["id"], with_comments=True)
        print(f"[{ev['id']}] hackernews = {_write(ev['id'],'hackernews',posts)}", flush=True)


def stage_x():
    from collect_x import collect_search
    prof = str(HERE / "x_profile")
    for j, ev in enumerate(EVENTS):
        seen, posts = set(), []
        for i, kw in enumerate(ev["en_kw"][:3]):
            try:
                got = collect_search(kw, prof, keywords=ev["en_kw"], event_id=ev["id"],
                                     scroll_rounds=6, since=ev["start"], until=ev["end"])
            except Exception as e:
                print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True); got = []
            for p in got:
                if p.post_id not in seen:
                    seen.add(p.post_id); posts.append(p)
            time.sleep(10)
        print(f"[{ev['id']}] x = {_write(ev['id'],'x',posts)}", flush=True)
        if j < len(EVENTS) - 1:
            time.sleep(50)


def stage_youtube():
    from collect_youtube import collect_search
    for ev in EVENTS:
        seen, posts = set(), []
        for kw in ev["en_kw"][:2]:
            try:
                for p in collect_search(kw, keywords=ev["en_kw"], event_id=ev["id"], max_videos=30):
                    if p.post_id not in seen:
                        seen.add(p.post_id); posts.append(p)
            except Exception as e:
                print(f"    [{ev['id']}|{kw}] ERR {e}", flush=True)
        print(f"[{ev['id']}] youtube = {_write(ev['id'],'youtube',posts)}", flush=True)


def stage_weibo():
    from collect_weibo_historical import collect_event
    for ev in EVENTS:
        try:
            posts = collect_event(ev["zh_kw"], ev["start"], ev["end"], event_id=ev["id"], max_pages=12)
        except Exception as e:
            print(f"[{ev['id']}] weibo ERR {e}", flush=True); continue
        print(f"[{ev['id']}] weibo = {_write(ev['id'],'weibo',posts)}", flush=True)


def stage_bilibili():
    from collect_bilibili import collect_event
    for ev in EVENTS:
        try:
            posts = collect_event(ev["zh_kw"], event_id=ev["id"], start=ev["start"], end=ev["end"])
        except Exception as e:
            print(f"[{ev['id']}] bilibili ERR {e}", flush=True); continue
        print(f"[{ev['id']}] bilibili = {_write(ev['id'],'bilibili',posts)}", flush=True)


STAGES = {"reddit": stage_reddit, "hn": stage_hn, "x": stage_x, "youtube": stage_youtube,
          "weibo": stage_weibo, "bilibili": stage_bilibili}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["reddit", "hn", "youtube", "x", "weibo", "bilibili"] if arg == "all" else [arg]
    for s in order:
        print(f"\n### STAGE {s} ###", flush=True)
        STAGES[s]()
    print("\n=== top3 stage(s) done ===", flush=True)
