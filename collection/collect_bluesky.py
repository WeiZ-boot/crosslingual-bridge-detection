"""English-side collector — Bluesky via the public AppView search API (no key, no login).

Endpoint public.api.bsky.app/xrpc/app.bsky.feed.searchPosts supports keyword search + a hard
date window (since/until, RFC3339) and cursor pagination. Bluesky is English-heavy opinion
discourse — a clean, keyless, exactly-date-bounded EN source like Hacker News. Returns the
unified Post schema (platform=bluesky, lang from record langs).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from common import Post, JsonlWriter, matches_keywords, clean_text

_BASE = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
_UA = "Mozilla/5.0 (research; bridge-detection) bsky-collector/1.0"


def _rfc3339(d: str, end: bool = False) -> str:
    y, m, dd = (int(x) for x in d.split("-"))
    t = datetime(y, m, dd, 23, 59, 59 if end else 0, 0, tzinfo=timezone.utc) if end \
        else datetime(y, m, dd, 0, 0, 0, 0, tzinfo=timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(params: dict) -> dict:
    url = _BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _search(query: str, start: str, end: str, max_pages: int = 6) -> list[dict]:
    out, cursor, page = [], None, 0
    while page < max_pages:
        params = {"q": query, "limit": 100, "sort": "latest",
                  "since": _rfc3339(start), "until": _rfc3339(end, end=True)}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _get(params)
        except Exception as e:
            print(f"    bsky err {query!r} p{page}: {e}", flush=True)
            break
        batch = data.get("posts", [])
        out.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
        page += 1
        time.sleep(0.3)
    return out


def collect_event(keywords: list[str], start: str, end: str, event_id: str = "",
                  max_pages: int = 6) -> list[Post]:
    seen: set[str] = set()
    out: list[Post] = []
    for kw in keywords:
        for p in _search(kw, start, end, max_pages):
            uri = p.get("uri", "")
            if uri in seen:
                continue
            rec = p.get("record", {}) or {}
            text = clean_text(rec.get("text", "") or "")
            if not text or not matches_keywords(text, keywords):
                continue
            seen.add(uri)
            author = p.get("author", {}) or {}
            handle = author.get("handle", "")
            rkey = uri.rsplit("/", 1)[-1]
            langs = rec.get("langs") or []
            out.append(Post(
                platform="bluesky", post_id=uri,
                url=f"https://bsky.app/profile/{handle}/post/{rkey}",
                text=text, author=handle, lang=(langs[0] if langs else "en"),
                timestamp=rec.get("createdAt"), event_id=event_id, query=kw,
                metrics={"likes": p.get("likeCount"), "reposts": p.get("repostCount"),
                         "replies": p.get("replyCount")}))
    return out


if __name__ == "__main__":
    posts = collect_event(["RedNote", "Xiaohongshu"], "2025-01-13", "2025-01-23", event_id="smoke")
    inwin = sum(1 for p in posts if p.timestamp and "2025-01-13" <= p.timestamp[:10] <= "2025-01-23")
    print(f"bluesky collected {len(posts)} (in_window={inwin})")
    for p in posts[:6]:
        print(f"  [{(p.timestamp or '')[:10]}] @{p.author}: {p.text[:70]}")
    with JsonlWriter("out/bluesky_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
