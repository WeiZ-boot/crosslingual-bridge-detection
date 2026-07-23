"""X (Twitter) collector — capture GraphQL XHR JSON via stealth browser.

Guest mode (no login) works for PROFILE timelines: X's frontend fires the
`UserTweets` GraphQL call before the login wall engages, and capture_xhr
intercepts the JSON. Verified: 99 tweets from one Reuters profile load.

Keyword SEARCH requires login -> pass user_data_dir of a logged-in profile
(see collect_search). Profile timelines are the reliable no-login path.
"""
from __future__ import annotations

import json

from scrapling.fetchers import StealthyFetcher

from common import Post, JsonlWriter, matches_keywords, clean_text


def _captured(page) -> list:
    return getattr(page, "xhr_responses", None) or getattr(page, "captured_xhr", None) or []


def _resp_json(resp):
    raw = resp.get("body") if isinstance(resp, dict) else getattr(resp, "body", None)
    if raw is None and isinstance(resp, dict):
        raw = resp.get("text")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _walk_tweets(obj, out: list[dict]):
    """Collect tweet 'legacy' result dicts (have full_text + id_str)."""
    if isinstance(obj, dict):
        if "full_text" in obj and "id_str" in obj:
            out.append(obj)
        for v in obj.values():
            _walk_tweets(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_tweets(v, out)


def _scroll(page, rounds=3):
    for _ in range(rounds):
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(1500)
    return page


def _to_posts(legacy_tweets: list[dict], event_id: str, query: str,
              keywords: list[str]) -> list[Post]:
    posts: list[Post] = []
    for t in legacy_tweets:
        text = clean_text(t.get("full_text", ""))
        if not text or not matches_keywords(text, keywords):
            continue
        posts.append(Post(
            platform="x",
            post_id=t.get("id_str", ""),
            url=f"https://x.com/i/status/{t.get('id_str','')}",
            text=text,
            author=t.get("screen_name", "") or str(t.get("user_id_str", "")),
            lang=t.get("lang", "en"),
            timestamp=t.get("created_at"),         # twitter format, e.g. "Wed Nov 11 ..."
            event_id=event_id,
            query=query,
            metrics={
                "likes": t.get("favorite_count"),
                "retweets": t.get("retweet_count"),
                "replies": t.get("reply_count"),
                "views": (t.get("ext_views") or {}).get("count") if isinstance(t.get("ext_views"), dict) else None,
            },
        ))
    return posts


def collect_profile(screen_name: str, keywords: list[str] | None = None,
                    event_id: str = "", scroll_rounds: int = 0) -> list[Post]:
    """Guest mode — no login. Profile timeline via captured UserTweets JSON.

    scroll_rounds=0 (default): single load, ~99 tweets from one UserTweets call.
    scroll_rounds>0: scroll for more, but X's persistent connections mean
    network_idle never settles, so we drop network_idle when scrolling.
    """
    keywords = keywords or []
    kwargs = dict(
        headless=True, timeout=90000, wait=4000,
        capture_xhr=r"(UserTweets|UserMedia)",
    )
    if scroll_rounds > 0:
        kwargs["page_action"] = lambda p: _scroll(p, scroll_rounds)
        kwargs["wait"] = 6000          # no network_idle; rely on fixed waits
    else:
        kwargs["network_idle"] = True
    page = StealthyFetcher.fetch(f"https://x.com/{screen_name}", **kwargs)
    tweets: list[dict] = []
    for resp in _captured(page):
        data = _resp_json(resp)
        if data:
            _walk_tweets(data, tweets)
    return _to_posts(tweets, event_id, screen_name, keywords)


def collect_search(query: str, user_data_dir: str | None = None, keywords: list[str] | None = None,
                   event_id: str = "", scroll_rounds: int = 4,
                   since: str | None = None, until: str | None = None,
                   cookies: list[dict] | None = None) -> list[Post]:
    """Keyword search — authenticated via EITHER a logged-in profile dir OR a cookies list.

    Pass `user_data_dir` (from x_login.py) for account 1, or `cookies` (a Playwright-format
    list, e.g. from import_chrome_x_cookies.py) for a second account — lets us rotate accounts
    to spread load and avoid rate limits.

    For HISTORICAL events, pass since/until (ISO dates); they are appended as X's
    `since:`/`until:` search operators, which return date-bounded historical tweets.
    """
    from urllib.parse import quote_plus
    keywords = keywords or []
    q = query
    if since:
        q += f" since:{since}"
    if until:
        q += f" until:{until}"
    url = f"https://x.com/search?q={quote_plus(q)}&f=live"
    auth = {"cookies": cookies} if cookies else {"user_data_dir": user_data_dir}
    page = StealthyFetcher.fetch(
        url, headless=True, network_idle=True, timeout=90000, wait=4000,
        capture_xhr=r"SearchTimeline",
        page_action=lambda p: _scroll(p, scroll_rounds),
        **auth,
    )
    tweets: list[dict] = []
    for resp in _captured(page):
        data = _resp_json(resp)
        if data:
            _walk_tweets(data, tweets)
    return _to_posts(tweets, event_id, query, keywords)


if __name__ == "__main__":
    print("--- guest profile @Reuters ---")
    posts = collect_profile("Reuters", event_id="smoke", scroll_rounds=2)
    print(f"collected {len(posts)} tweets")
    for p in posts[:5]:
        print(f"  [{p.timestamp}] {p.text[:75]} (likes={p.metrics.get('likes')})")
    with JsonlWriter("out/x_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
