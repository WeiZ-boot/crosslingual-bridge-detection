"""Chinese-side collector — Bilibili video search (no API signing needed).

The search API needs wbi signing, but search.bilibili.com renders result cards in HTML. StealthyFetcher
loads the page (browser handles the anti-bot), we parse .bili-video-card cards for title/author/date/
link. Returns the unified Post schema (platform=bilibili, lang=zh). Title+description as the bridge
text. (Comments are richer for counter-narrative — a follow-on: bvid -> x/v2/reply, signed/browser.)

Date filtering: parse the card's publish date; keep those in [start,end] when given. Bilibili search
is relevance-ordered (recent-biased); &order=pubdate helps for historical.
"""
from __future__ import annotations

import re
import urllib.parse
from datetime import date

from scrapling.fetchers import StealthyFetcher

from common import Post, JsonlWriter, matches_keywords, clean_text

_DATE = re.compile(r"(20\d\d)-(\d{1,2})-(\d{1,2})")


def _first(el, sel):
    f = el.css(sel)
    return f[0] if f else None


def _scroll(page, rounds):
    for _ in range(rounds):
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(1500)
    return page


def _card_date(c):
    el = _first(c, ".bili-video-card__info--date")
    if not el:
        return None
    m = _DATE.search(el.text.clean())
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()


def collect_search(keyword: str, keywords: list[str] | None = None, event_id: str = "",
                   start: str | None = None, end: str | None = None,
                   order: str = "pubdate", scroll_rounds: int = 3) -> list[Post]:
    keywords = keywords or []
    url = (f"https://search.bilibili.com/video?keyword={urllib.parse.quote(keyword)}"
           f"&order={order}")
    page = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=80000, wait=4000,
                                 page_action=lambda p: _scroll(p, scroll_rounds))
    out = []
    for c in page.css(".bili-video-card"):
        tit = _first(c, "h3.bili-video-card__info--tit")
        title = clean_text(tit.attrib.get("title", "") or tit.text) if tit else ""
        if not title:
            continue
        link = _first(c, "a[href*=video]")
        href = link.attrib.get("href", "") if link else ""
        auth = _first(c, ".bili-video-card__info--author")
        d = _card_date(c)
        if start and end and d and not (start <= d <= end):
            continue
        bvid = ""
        mb = re.search(r"(BV[0-9A-Za-z]+)", href)
        if mb:
            bvid = mb.group(1)
        out.append(Post(
            platform="bilibili", post_id=bvid or title[:32],
            url=("https:" + href) if href.startswith("//") else href,
            text=title, author=auth.text.clean() if auth else "",
            lang="zh", timestamp=d, event_id=event_id, query=keyword,
            metrics={}))
    # dedup + keyword filter
    seen, res = set(), []
    for p in out:
        if p.post_id in seen:
            continue
        seen.add(p.post_id)
        if matches_keywords(p.text, keywords):
            res.append(p)
    return res


def collect_event(keywords: list[str], event_id: str = "", start: str | None = None,
                  end: str | None = None) -> list[Post]:
    seen, out = set(), []
    for kw in keywords:
        for p in collect_search(kw, event_id=event_id, start=start, end=end):
            if p.post_id in seen:
                continue
            seen.add(p.post_id)
            out.append(p)
        print(f"  [{kw}] cumulative={len(out)}")
    return out


if __name__ == "__main__":
    posts = collect_search("福岛核污水", event_id="smoke", scroll_rounds=3)
    print(f"bilibili videos: {len(posts)}")
    for p in posts[:6]:
        print(f"  [{p.timestamp}] {p.text[:50]}  ({p.author})")
    with JsonlWriter("out/bilibili_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
