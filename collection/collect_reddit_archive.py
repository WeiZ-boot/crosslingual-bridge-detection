"""Historical Reddit collector via Arctic Shift archive API (free, date-ranged).

Live www.reddit.com search returns only recent posts — useless for historical events.
Arctic Shift (arctic-shift.photon-reddit.com) serves Reddit's historical archive with
date filtering. Note: full-text search REQUIRES a subreddit (or author) scope — there is
no global query. So we search each event's seed subreddits by title keyword within window.

Endpoints used:
  /api/posts/search?subreddit=<s>&title=<kw>&after=<ISO>&before=<ISO>&limit=100
  /api/comments/search?subreddit=<s>&body=<kw>&after=<ISO>&before=<ISO>&limit=100
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import urllib.parse
import urllib.request

from common import Post, JsonlWriter, matches_keywords, clean_text

API = "https://arctic-shift.photon-reddit.com/api"


def _get(path: str, params: dict, retries: int = 3) -> list[dict]:
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "paper2-research/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                import json
                payload = json.loads(r.read().decode("utf-8", "ignore"))
            data = payload.get("data")
            if data is None:                      # error envelope
                return []
            return data
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(1.5)
    return []


def _epoch_to_iso(ts) -> str | None:
    try:
        return datetime.fromtimestamp(int(ts), timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def collect_event(subreddits: list[str], keywords: list[str], start: str, end: str,
                  event_id: str = "", with_comments: bool = False,
                  limit: int = 100) -> list[Post]:
    """For each subreddit × keyword, fetch date-bounded historical posts (+optional comments)."""
    out: list[Post] = []
    for sub in subreddits:
        for kw in keywords:
            for row in _get("posts/search", {
                "subreddit": sub, "title": kw,
                "after": start, "before": end, "limit": limit,
            }):
                title = clean_text(row.get("title", ""))
                body = clean_text(row.get("selftext", ""))
                text = f"{title}. {body}".strip(". ")
                if not title:
                    continue
                out.append(Post(
                    platform="reddit",
                    post_id=row.get("id", row.get("permalink", title[:40])),
                    url="https://www.reddit.com" + row.get("permalink", ""),
                    text=text,
                    author=row.get("author", ""),
                    timestamp=_epoch_to_iso(row.get("created_utc")),
                    event_id=event_id,
                    query=f"{sub}:{kw}",
                    metrics={"subreddit": sub, "score": row.get("score"),
                             "comments": row.get("num_comments")},
                ))
            if with_comments:
                for row in _get("comments/search", {
                    "subreddit": sub, "body": kw,
                    "after": start, "before": end, "limit": limit,
                }):
                    text = clean_text(row.get("body", ""))
                    if not text or not matches_keywords(text, keywords):
                        continue
                    out.append(Post(
                        platform="reddit",
                        post_id="c_" + row.get("id", text[:30]),
                        url="https://www.reddit.com" + row.get("permalink", ""),
                        text=text,
                        author=row.get("author", ""),
                        timestamp=_epoch_to_iso(row.get("created_utc")),
                        event_id=event_id,
                        query=f"{sub}:{kw}:comment",
                        metrics={"subreddit": sub, "score": row.get("score"),
                                 "kind": "comment"},
                    ))
    return out


if __name__ == "__main__":
    # smoke: deepseek_viral window
    posts = collect_event(
        ["technology", "LocalLLaMA", "singularity"],
        ["DeepSeek", "DeepSeek R1"],
        "2025-01-24", "2025-02-16", event_id="smoke")
    inwin = sum(1 for p in posts if p.timestamp and "2025-01-24" <= p.timestamp[:10] <= "2025-02-16")
    print(f"collected {len(posts)} posts, in_window={inwin}")
    for p in posts[:6]:
        print(f"  [{(p.timestamp or '?')[:10]}] r/{p.metrics.get('subreddit')}: {p.text[:60]}")
    with JsonlWriter("out/reddit_archive_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
