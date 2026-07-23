"""End-to-end collection for 10 NEW globally-viral, China-relevant hot-topic events
(no cphot ZH side). EN: Reddit + HN + Bluesky + X (rotating 2 accounts) + YouTube(deep).
ZH: Weibo historical + Bilibili.  Run one stage at a time:
    python collect_batch10.py <stage>   stage in reddit|hn|bluesky|x|youtube|weibo|bilibili
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

from common import JsonlWriter

HERE = Path(__file__).resolve().parent

EVENTS = [
    dict(id="venezuela_us_2026", start="2026-05-18", end="2026-06-18",
         en_kw=["Venezuela", "Maduro", "US Venezuela", "Venezuela military", "Venezuela intervention"],
         subs=["worldnews", "geopolitics", "politics", "venezuela", "anime_titties"],
         zh_kw=["委内瑞拉", "马杜罗", "美国 委内瑞拉", "委内瑞拉 美军", "委内瑞拉 战争"]),
    dict(id="gaza_oct7_2023", start="2023-10-07", end="2023-10-21",
         en_kw=["Hamas attack", "Israel Gaza", "October 7", "Gaza hostages", "Hamas Israel war"],
         subs=["worldnews", "geopolitics", "politics", "IsraelPalestine"],
         zh_kw=["哈马斯", "以色列 加沙", "巴以冲突", "加沙 人质", "哈马斯 以色列"]),
    dict(id="epstein_case", start="2024-01-03", end="2024-01-17",
         en_kw=["Epstein", "Jeffrey Epstein", "Epstein list", "Epstein documents", "Epstein files"],
         subs=["news", "worldnews", "conspiracy", "politics"],
         zh_kw=["爱泼斯坦", "爱泼斯坦 名单", "萝莉岛", "爱泼斯坦 文件"]),
    dict(id="brics_summit_2023", start="2023-08-20", end="2023-08-28",
         en_kw=["BRICS", "BRICS summit", "BRICS expansion", "de-dollarization", "BRICS currency"],
         subs=["worldnews", "geopolitics", "economics", "investing"],
         zh_kw=["金砖", "金砖峰会", "金砖扩员", "去美元化", "金砖国家"]),
    dict(id="gaza_ceasefire_2025", start="2025-01-15", end="2025-01-25",
         en_kw=["Gaza ceasefire", "Israel Hamas ceasefire", "Gaza truce", "hostage deal Gaza"],
         subs=["worldnews", "geopolitics", "politics"],
         zh_kw=["加沙 停火", "以色列 哈马斯 停火", "加沙停火协议", "加沙 人质 协议"]),
    dict(id="luigi_mangione_2024", start="2024-12-04", end="2024-12-18",
         en_kw=["Luigi Mangione", "UnitedHealthcare CEO", "Brian Thompson", "CEO shooting", "health insurance CEO"],
         subs=["news", "politics", "worldnews", "LateStageCapitalism"],
         zh_kw=["联合健康 CEO", "美国 医保", "枪杀 CEO", "曼吉奥尼", "美国 医疗"]),
    dict(id="trump_zelensky_oval_2025", start="2025-02-28", end="2025-03-08",
         en_kw=["Trump Zelensky", "Oval Office", "Zelensky meeting", "Vance Zelensky", "Zelensky White House"],
         subs=["worldnews", "geopolitics", "politics", "ukraine"],
         zh_kw=["特朗普 泽连斯基", "泽连斯基 白宫", "泽连斯基 特朗普", "椭圆形办公室"]),
    dict(id="pope_leo_2025", start="2025-05-08", end="2025-05-18",
         en_kw=["Pope Leo", "new pope", "Pope Leo XIV", "Robert Prevost", "American pope"],
         subs=["worldnews", "news", "Catholicism", "politics"],
         zh_kw=["新教皇", "良十四世", "美国 教皇", "教皇 良十四"]),
    dict(id="signalgate_2025", start="2025-03-24", end="2025-04-03",
         en_kw=["Signalgate", "Signal chat", "Hegseth Signal", "war plans Signal", "Atlantic group chat"],
         subs=["politics", "worldnews", "news"],
         zh_kw=["signal 泄密", "美国 群聊 泄密", "赫格塞斯", "美国 军事 泄密"]),
    dict(id="doge_musk_2025", start="2025-02-01", end="2025-02-22",
         en_kw=["DOGE Musk", "Department of Government Efficiency", "Musk USAID", "federal layoffs", "DOGE cuts"],
         subs=["politics", "worldnews", "news", "technology"],
         zh_kw=["马斯克 政府效率部", "DOGE 马斯克", "美国 裁员", "USAID 马斯克"]),
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


STAGES = {"reddit": stage_reddit, "hn": stage_hn, "bluesky": stage_bluesky, "x": stage_x,
          "youtube": stage_youtube, "weibo": stage_weibo, "bilibili": stage_bilibili}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["reddit", "hn", "bluesky", "youtube", "x", "weibo", "bilibili"] if arg == "all" else [arg]
    for s in order:
        print(f"\n### STAGE {s} ###", flush=True)
        STAGES[s]()
    print("\n=== batch10 stage(s) done ===", flush=True)
