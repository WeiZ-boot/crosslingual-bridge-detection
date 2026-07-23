"""English-side collector — Hacker News via the Algolia search API (no key, no login).

Endpoint hn.algolia.com/api/v1/search_by_date supports keyword search + a hard date window
(numericFilters on created_at_i, a unix ts) and paginates. We pull both STORIES (title +
self-text) and COMMENTS (the opinion-rich part) within [start,end]. Returns the unified Post
schema (platform=hackernews, lang=en). This is the cleanest historical EN source we have:
keyless, exactly date-bounded, reliable — ideal for the tech-heavy events.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

from common import Post, JsonlWriter, matches_keywords, clean_text

_BASE = "http://hn.algolia.com/api/v1/search_by_date"
_UA = "Mozilla/5.0 (research; bridge-detection) hn-collector/1.0"


def _epoch(d: str, end: bool = False) -> int:
    y, m, dd = (int(x) for x in d.split("-"))
    t = datetime(y, m, dd, 23, 59, 59 if end else 0, tzinfo=timezone.utc) if end \
        else datetime(y, m, dd, 0, 0, 0, tzinfo=timezone.utc)
    return int(t.timestamp())


def _ts_iso(created_at_i) -> str | None:
    try:
        return datetime.fromtimestamp(int(created_at_i), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _get(params: dict) -> dict:
    url = _BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _search(query: str, tags: str, start: str, end: str, max_pages: int = 5) -> list[dict]:
    nf = f"created_at_i>{_epoch(start)},created_at_i<{_epoch(end, end=True)}"
    hits, page = [], 0
    while page < max_pages:
        try:
            data = _get({"query": query, "tags": tags, "numericFilters": nf,
                         "hitsPerPage": 100, "page": page})
        except Exception as e:
            print(f"    HN err {query!r} p{page}: {e}", flush=True)
            break
        batch = data.get("hits", [])
        hits.extend(batch)
        if page >= data.get("nbPages", 0) - 1 or not batch:
            break
        page += 1
        time.sleep(0.3)
    return hits


def collect_event(keywords: list[str], start: str, end: str, event_id: str = "",
                  with_comments: bool = True, max_pages: int = 5) -> list[Post]:
    seen: set[str] = set()
    out: list[Post] = []
    for kw in keywords:
        for tag in (["story", "comment"] if with_comments else ["story"]):
            for h in _search(kw, tag, start, end, max_pages):
                oid = h.get("objectID", "")
                if oid in seen:
                    continue
                if tag == "comment":
                    text = clean_text(h.get("comment_text", "") or "")
                else:
                    text = clean_text(f"{h.get('title','')}. {h.get('story_text','') or ''}".strip(". "))
                if not text or not matches_keywords(text, keywords):
                    continue
                seen.add(oid)
                out.append(Post(
                    platform="hackernews", post_id=oid,
                    url=f"https://news.ycombinator.com/item?id={oid}",
                    text=text, author=h.get("author", ""), lang="en",
                    timestamp=_ts_iso(h.get("created_at_i")),
                    event_id=event_id, query=f"{kw}:{tag}",
                    metrics={"points": h.get("points"), "num_comments": h.get("num_comments"),
                             "kind": tag}))
    return out


if __name__ == "__main__":
    posts = collect_event(["OpenAI o1", "o1 model"], "2024-09-12", "2024-09-26",
                          event_id="smoke")
    inwin = sum(1 for p in posts if p.timestamp and "2024-09-12" <= p.timestamp[:10] <= "2024-09-26")
    print(f"HN collected {len(posts)} (in_window={inwin})")
    for p in posts[:6]:
        print(f"  [{p.timestamp[:10]}] ({p.metrics['kind']}) {p.text[:75]}")
    with JsonlWriter("out/hn_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
