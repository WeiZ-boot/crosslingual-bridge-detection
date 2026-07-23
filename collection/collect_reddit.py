"""Reddit collector — full stealth browser on www.reddit.com.

No API/login. Works from datacenter IPs because the JS-rendered site serves a
properly-fingerprinted stealth browser (the lightweight Fetcher / .json / old.reddit
endpoints get 403 from datacenter IPs, so we use StealthyFetcher).

Two entry points:
  - collect_search(query): site search across all subreddits for a hot-topic event
  - collect_subreddit(sub): newest posts from a specific subreddit
"""
from __future__ import annotations

from urllib.parse import quote_plus

from scrapling.fetchers import StealthyFetcher

from common import Post, JsonlWriter, matches_keywords, clean_text


def _scroll(page, rounds=4):
    """page_action: scroll to trigger lazy-loaded posts."""
    for _ in range(rounds):
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(1200)
    return page


def _parse_posts(page, event_id: str, query: str) -> list[Post]:
    posts: list[Post] = []
    # modern reddit uses <shreddit-post> custom elements with rich attributes
    for el in page.css("shreddit-post"):
        a = el.attrib
        pid = a.get("id", "") or a.get("permalink", "")
        title = clean_text(a.get("post-title", ""))
        if not title:
            # fall back to slotted title link text
            t = el.css("a[slot=title]")
            title = clean_text(t[0].text) if t else ""
        if not title:
            continue
        permalink = a.get("permalink", "")
        posts.append(Post(
            platform="reddit",
            post_id=a.get("id", permalink or title[:40]),
            url=f"https://www.reddit.com{permalink}" if permalink.startswith("/") else permalink,
            text=title,
            author=a.get("author", ""),
            timestamp=a.get("created-timestamp"),
            event_id=event_id,
            query=query,
            metrics={
                "subreddit": a.get("subreddit-prefixed-name", ""),
                "score": a.get("score", ""),
                "comments": a.get("comment-count", ""),
            },
        ))
    return posts


def _fetch(url: str, scroll_rounds: int = 4):
    return StealthyFetcher.fetch(
        url, headless=True, network_idle=True, timeout=90000, wait=2500,
        page_action=lambda p: _scroll(p, scroll_rounds),
    )


def collect_search(query: str, keywords: list[str] | None = None,
                   event_id: str = "", sort: str = "new") -> list[Post]:
    keywords = keywords or []
    url = f"https://www.reddit.com/search/?q={quote_plus(query)}&sort={sort}"
    page = _fetch(url)
    if page.status != 200:
        return []
    return [p for p in _parse_posts(page, event_id, query)
            if matches_keywords(p.text, keywords)]


def collect_subreddit(sub: str, keywords: list[str] | None = None,
                      event_id: str = "", sort: str = "new") -> list[Post]:
    keywords = keywords or []
    url = f"https://www.reddit.com/r/{sub}/{sort}/"
    page = _fetch(url)
    if page.status != 200:
        return []
    return [p for p in _parse_posts(page, event_id, sub)
            if matches_keywords(p.text, keywords)]


if __name__ == "__main__":
    print("--- subreddit r/worldnews ---")
    posts = collect_subreddit("worldnews", event_id="smoke")
    print(f"collected {len(posts)} posts")
    for p in posts[:5]:
        print(f"  [{p.metrics.get('subreddit')}] {p.text[:70]} (score={p.metrics.get('score')})")
    with JsonlWriter("out/reddit_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
