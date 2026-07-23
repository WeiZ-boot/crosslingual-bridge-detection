"""Orchestrator — run all English-side collectors per event from events.yaml.

Output layout:
    out/<event_id>/telegram.jsonl
    out/<event_id>/youtube.jsonl
    out/<event_id>/reddit.jsonl
    out/<event_id>/x.jsonl
    out/<event_id>/_summary.json

Usage:
    python run_collection.py                      # all events, all platforms
    python run_collection.py --event us-iran-strikes-2026
    python run_collection.py --platforms telegram youtube   # skip browser ones
    python run_collection.py --x-user-data-dir ./x_profile  # enable X keyword search
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import yaml

from common import JsonlWriter
import collect_telegram
import collect_youtube
import collect_reddit
import collect_reddit_archive
import collect_x

ALL_PLATFORMS = ["telegram", "youtube", "reddit", "x"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_platform(name, fn, out_dir: Path, event_id: str) -> dict:
    """Run one collector fn -> list[Post], write jsonl, return stats."""
    t0 = time.time()
    try:
        posts = fn()
    except Exception as e:
        traceback.print_exc()
        return {"platform": name, "status": "error", "error": str(e)[:200],
                "collected": 0, "written": 0, "seconds": round(time.time() - t0, 1)}
    with JsonlWriter(out_dir / f"{name}.jsonl") as w:
        for p in posts:
            w.write(p)
        written = w.written
    return {"platform": name, "status": "ok", "collected": len(posts),
            "written": written, "seconds": round(time.time() - t0, 1)}


def collect_event(ev: dict, defaults: dict, platforms: list[str],
                  out_root: Path, x_user_data_dir: str | None) -> dict:
    eid = ev["id"]
    out_dir = out_root / eid
    kw = ev.get("keywords", [])
    start, end = ev.get("start"), ev.get("end")
    print(f"\n{'='*60}\nEVENT: {eid} — {ev.get('name','')}")
    stats = []

    if "telegram" in platforms:
        def _tg():
            out = []
            for ch in ev.get("telegram", []):
                out += collect_telegram.collect_channel(
                    ch, kw, event_id=eid, start=start, end=end,
                    max_pages=defaults.get("telegram_max_pages", 3))
            return out
        stats.append(_run_platform("telegram", _tg, out_dir, eid))

    if "youtube" in platforms:
        def _yt():
            out = []
            for q in ev.get("youtube", []):
                out += collect_youtube.collect_search(
                    q, kw, event_id=eid,
                    max_videos=defaults.get("youtube_max_videos", 30))
            return out
        stats.append(_run_platform("youtube", _yt, out_dir, eid))

    if "reddit" in platforms:
        def _rd():
            # historical events -> Arctic Shift archive (date-bounded). Live www.reddit.com
            # search only returns recent posts, so it is useless for past windows.
            if defaults.get("reddit_live"):
                out = []
                sort = defaults.get("reddit_sort", "new")
                for sub in ev.get("subreddits", []):
                    out += collect_reddit.collect_subreddit(sub, kw, event_id=eid, sort=sort)
                for q in ev.get("reddit_search", []):
                    out += collect_reddit.collect_search(q, kw, event_id=eid, sort=sort)
                return out
            return collect_reddit_archive.collect_event(
                ev.get("subreddits", []), kw, start, end, event_id=eid,
                with_comments=defaults.get("reddit_comments", False))
        stats.append(_run_platform("reddit", _rd, out_dir, eid))

    if "x" in platforms:
        def _x():
            out = []
            rounds = defaults.get("x_scroll_rounds", 0)
            for acct in ev.get("x_accounts", []):
                out += collect_x.collect_profile(acct, kw, event_id=eid, scroll_rounds=rounds)
            if x_user_data_dir:
                for q in ev.get("x_search", []):
                    out += collect_x.collect_search(q, x_user_data_dir, kw, event_id=eid,
                                                    since=start, until=end)
            return out
        stats.append(_run_platform("x", _x, out_dir, eid))

    for s in stats:
        print(f"  {s['platform']:9} {s['status']:5} "
              f"collected={s['collected']:4} written={s['written']:4} ({s['seconds']}s)"
              + (f"  ERR: {s.get('error')}" if s['status'] == 'error' else ""))

    summary = {"event_id": eid, "name": ev.get("name", ""), "platforms": stats,
               "total_written": sum(s["written"] for s in stats)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="events.yaml")
    ap.add_argument("--event", help="run a single event id")
    ap.add_argument("--platforms", nargs="+", default=ALL_PLATFORMS,
                    choices=ALL_PLATFORMS)
    ap.add_argument("--out", default="out")
    ap.add_argument("--x-user-data-dir", default=None,
                    help="logged-in X profile dir to enable x_search")
    args = ap.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    events = cfg.get("events", [])
    if args.event:
        events = [e for e in events if e["id"] == args.event]
        if not events:
            raise SystemExit(f"event '{args.event}' not found in {args.config}")

    out_root = Path(args.out)
    grand = []
    for ev in events:
        grand.append(collect_event(ev, defaults, args.platforms, out_root,
                                   args.x_user_data_dir))

    print(f"\n{'='*60}\nGRAND TOTAL")
    for g in grand:
        print(f"  {g['event_id']:28} {g['total_written']:5} posts")
    print(f"  {'TOTAL':28} {sum(g['total_written'] for g in grand):5} posts")


if __name__ == "__main__":
    main()
