"""Chinese-side collector — Weibo (the ZH counterpart to X).

No API / no login for current search: StealthyFetcher browser loads m.weibo.cn search; the frontend
fires the getIndex API; capture_xhr intercepts the JSON (same trick as X's GraphQL). Scrolling
triggers more getIndex pages. Returns the unified Post schema (platform=weibo, lang=zh).

Verified: search '新疆棉花' returns counter-narrative content (rebutting Western forced-labor framing).

HISTORICAL NOTE: m.weibo.cn search = recent posts only. For a past event window, use Weibo advanced
search s.weibo.com with timescope=custom:START:END + a logged-in cookie (see collect_weibo_historical
stub) — the ZH analog of X since/until.
"""
from __future__ import annotations

import json
import re
import urllib.parse

from scrapling.fetchers import StealthyFetcher

from common import Post, JsonlWriter, matches_keywords, clean_text

_TAG = re.compile(r"<[^>]+>")


def _captured(page):
    return getattr(page, "xhr_responses", None) or getattr(page, "captured_xhr", None) or []


def _body(r):
    raw = r.get("body") if isinstance(r, dict) else getattr(r, "body", None)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    return raw


def _scroll(page, rounds):
    for _ in range(rounds):
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(1500)
    return page


def _mblog_to_post(mb, event_id, query):
    text = clean_text(_TAG.sub("", mb.get("text", "")))
    if not text:
        return None
    user = mb.get("user") or {}
    return Post(
        platform="weibo", post_id=str(mb.get("id", "")),
        url=f"https://m.weibo.cn/detail/{mb.get('id','')}",
        text=text, author=str(user.get("screen_name", "") or user.get("id", "")),
        lang="zh", timestamp=mb.get("created_at"), event_id=event_id, query=query,
        metrics={"reposts": mb.get("reposts_count"), "comments": mb.get("comments_count"),
                 "likes": mb.get("attitudes_count")},
    )


def _extract(page, event_id, query):
    out = []
    for r in _captured(page):
        try:
            d = json.loads(_body(r))
        except (TypeError, ValueError):
            continue
        for card in d.get("data", {}).get("cards", []):
            if card.get("mblog"):
                p = _mblog_to_post(card["mblog"], event_id, query)
                if p:
                    out.append(p)
            for c2 in card.get("card_group", []) or []:
                if c2.get("mblog"):
                    p = _mblog_to_post(c2["mblog"], event_id, query)
                    if p:
                        out.append(p)
    return out


def collect_search(keyword: str, keywords: list[str] | None = None, event_id: str = "",
                   scroll_rounds: int = 4) -> list[Post]:
    """Current Weibo search via browser + captured getIndex JSON (no login)."""
    keywords = keywords or []
    cid = urllib.parse.quote(f"100103type=1&q={keyword}")
    url = f"https://m.weibo.cn/search?containerid={cid}"
    page = StealthyFetcher.fetch(
        url, headless=True, timeout=90000, wait=4000,
        capture_xhr=r"getIndex",
        page_action=lambda p: _scroll(p, scroll_rounds))
    posts = _extract(page, event_id, keyword)
    # dedupe by id, keyword-filter
    seen, out = set(), []
    for p in posts:
        if p.post_id in seen:
            continue
        seen.add(p.post_id)
        if matches_keywords(p.text, keywords):
            out.append(p)
    return out


if __name__ == "__main__":
    posts = collect_search("新疆棉花", event_id="smoke", scroll_rounds=4)
    print(f"weibo posts: {len(posts)}")
    for p in posts[:6]:
        print(f"  [{p.timestamp}] {p.text[:60]}  (赞{p.metrics.get('likes')})")
    with JsonlWriter("out/weibo_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
