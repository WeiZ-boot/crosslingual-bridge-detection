"""Telegram collector — public channel web previews via t.me/s/<channel>.

No account / API needed. Pulls recent messages from given public channels,
filters by keyword + date window. Pagination via ?before=<message_id>.
"""
from __future__ import annotations

from scrapling.fetchers import Fetcher

from common import Post, JsonlWriter, matches_keywords, in_window, clean_text


def _first(el, sel):
    found = el.css(sel)
    return found[0] if found else None


def _parse_messages(page, channel: str, event_id: str, query: str) -> list[Post]:
    posts: list[Post] = []
    for wrap in page.css(".tgme_widget_message_wrap"):
        msg = _first(wrap, ".tgme_widget_message")
        if not msg:
            continue
        data_post = msg.attrib.get("data-post", "")          # e.g. "channel/12345"
        mid = data_post.split("/")[-1] if data_post else ""
        text_node = _first(wrap, ".tgme_widget_message_text")
        text = clean_text(text_node.text) if text_node else ""
        if not text:
            continue
        time_node = _first(wrap, "time")
        ts = time_node.attrib.get("datetime") if time_node else None
        views_node = _first(wrap, ".tgme_widget_message_views")
        views = views_node.text if views_node else ""
        posts.append(Post(
            platform="telegram",
            post_id=f"{channel}/{mid}",
            url=f"https://t.me/{data_post}" if data_post else f"https://t.me/s/{channel}",
            text=text,
            author=channel,
            timestamp=ts,
            event_id=event_id,
            query=query,
            metrics={"views": views},
        ))
    return posts


def collect_channel(channel: str, keywords: list[str], event_id: str = "",
                    start: str | None = None, end: str | None = None,
                    max_pages: int = 3) -> list[Post]:
    """Walk back up to `max_pages` pages of a public channel."""
    out: list[Post] = []
    before = None
    for _ in range(max_pages):
        url = f"https://t.me/s/{channel}"
        if before:
            url += f"?before={before}"
        page = Fetcher.get(url, impersonate="chrome")
        if page.status != 200:
            break
        batch = _parse_messages(page, channel, event_id, channel)
        if not batch:
            break
        out.extend(batch)
        # oldest message id on this page → paginate further back
        try:
            before = min(int(p.post_id.split("/")[-1]) for p in batch if p.post_id.split("/")[-1].isdigit())
        except ValueError:
            break
    # filter
    return [p for p in out
            if matches_keywords(p.text, keywords) and in_window(p.timestamp, start, end)]


if __name__ == "__main__":
    # smoke test: a high-volume English news channel
    posts = collect_channel("disclosetv", keywords=["Iran", "US", "inflation"],
                            event_id="smoke", max_pages=2)
    print(f"collected {len(posts)} matching messages")
    for p in posts[:5]:
        print(f"  [{p.timestamp}] {p.text[:80]}")
    with JsonlWriter("out/telegram_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
