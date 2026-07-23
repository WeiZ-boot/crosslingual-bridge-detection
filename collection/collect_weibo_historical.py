"""Weibo HISTORICAL collector — s.weibo.com advanced search with date window (needs login).

The ZH analog of X since/until. Uses the logged-in weibo_profile (weibo_login.py) to query
s.weibo.com advanced search with timescope=custom:START:END, paginating &page=1..N, parsing the
HTML result cards. Returns the unified Post schema (platform=weibo, lang=zh) within the event window.
"""
from __future__ import annotations

import urllib.parse
from datetime import date

from scrapling.fetchers import StealthyFetcher

from common import Post, JsonlWriter, matches_keywords, clean_text


def _first(el, sel):
    f = el.css(sel)
    return f[0] if f else None


def _parse_cards(page, event_id: str, query: str) -> list[Post]:
    out = []
    for c in page.css("div.card-wrap"):
        mid = c.attrib.get("mid", "")
        if not mid:                       # skip non-post cards (ads, topics)
            continue
        # post text: prefer the expanded full content, else the short content
        node = _first(c, "p[node-type=feed_list_content_full]") or \
            _first(c, "p[node-type=feed_list_content]")
        if not node:
            continue
        text = clean_text(node.text)
        if not text or len(text.strip("/ @")) < 4:    # skip empty/bare-repost stubs
            continue
        froms = c.css("div.from a")
        ts = froms[0].text.clean() if froms else None        # "2021年03月25日 10:24"
        href = froms[0].attrib.get("href", "") if froms else ""
        name = _first(c, "a.name")
        acts = [a.text.clean() for a in c.css("div.card-act li")]
        out.append(Post(
            platform="weibo", post_id=mid,
            url=("https:" + href) if href.startswith("//") else href,
            text=text, author=name.text.clean() if name else "",
            lang="zh", timestamp=ts, event_id=event_id, query=query,
            metrics={"act": " ".join(acts)[:80]}))
    return out


def collect(keyword: str, start: str, end: str, keywords: list[str] | None = None,
            event_id: str = "", profile: str = "weibo_profile", max_pages: int = 10) -> list[Post]:
    keywords = keywords or []
    s = date.fromisoformat(start); e = date.fromisoformat(end)
    ts = f"custom:{s.isoformat()}-0:{e.isoformat()}-23"
    out, seen = [], set()
    for pg in range(1, max_pages + 1):
        url = (f"https://s.weibo.com/weibo?q={urllib.parse.quote(keyword)}"
               f"&timescope={urllib.parse.quote(ts)}&Refer=g&page={pg}")
        page = StealthyFetcher.fetch(url, headless=True, user_data_dir=profile,
                                     network_idle=True, timeout=60000, wait=2500)
        if "登录" in (page.css("title::text").get() or ""):
            print("  LOGIN REQUIRED — run weibo_login.py first"); break
        cards = _parse_cards(page, event_id, keyword)
        if not cards:
            break
        new = [c for c in cards if c.post_id not in seen]
        for c in new:
            seen.add(c.post_id)
        out += new
        if len(new) < 5:           # near end of results
            break
    return [p for p in out if matches_keywords(p.text, keywords)]


def collect_event(keywords: list[str], start: str, end: str, event_id: str = "",
                  profile: str = "weibo_profile", max_pages: int = 12) -> list[Post]:
    """Run several ZH keyword queries for one event, aggregate + dedup by post_id."""
    seen, out = set(), []
    for kw in keywords:
        for p in collect(kw, start, end, event_id=event_id, profile=profile, max_pages=max_pages):
            if p.post_id in seen:
                continue
            seen.add(p.post_id)
            out.append(p)
        print(f"  [{kw}] cumulative={len(out)}")
    return out


if __name__ == "__main__":
    import sys
    posts = collect("新疆棉花", "2021-03-24", "2021-04-14", event_id="smoke", max_pages=5)
    print(f"historical weibo posts: {len(posts)}")
    for p in posts[:6]:
        print(f"  [{p.timestamp}] {p.text[:60]}")
    with JsonlWriter("out/weibo_hist_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
