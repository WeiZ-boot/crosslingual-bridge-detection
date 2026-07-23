"""YouTube collector — search results via embedded ytInitialData JSON.

No API key needed. Parses the videoRenderer entries from the search page.
Captures title + channel + view/publish metadata. (Comments would need a
second innertube call; left as a TODO since titles+descriptions already give
strong English-anchor signal for bridge pairing.)
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote_plus

from scrapling.fetchers import Fetcher

from common import Post, JsonlWriter, matches_keywords, clean_text

_YTDATA = re.compile(r"var ytInitialData = (\{.*?\});</script>", re.DOTALL)


def _extract_initial_data(body: str) -> dict | None:
    m = _YTDATA.search(body)
    if not m:
        # fallback pattern
        m = re.search(r'ytInitialData"\]\s*=\s*(\{.*?\});', body, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _walk_video_renderers(obj, out: list[dict]):
    if isinstance(obj, dict):
        if "videoRenderer" in obj:
            out.append(obj["videoRenderer"])
        for v in obj.values():
            _walk_video_renderers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_video_renderers(v, out)


def _runs_text(node) -> str:
    if not node:
        return ""
    if "simpleText" in node:
        return node["simpleText"]
    return "".join(r.get("text", "") for r in node.get("runs", []))


def collect_search(query: str, keywords: list[str] | None = None,
                   event_id: str = "", max_videos: int = 30) -> list[Post]:
    keywords = keywords or []
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    page = Fetcher.get(url, impersonate="chrome")
    body = page.body.decode("utf-8", "ignore") if isinstance(page.body, bytes) else page.body
    data = _extract_initial_data(body)
    if not data:
        return []
    renderers: list[dict] = []
    _walk_video_renderers(data, renderers)

    posts: list[Post] = []
    for r in renderers[:max_videos]:
        vid = r.get("videoId", "")
        if not vid:
            continue
        title = _runs_text(r.get("title"))
        channel = _runs_text(r.get("ownerText")) or _runs_text(r.get("longBylineText"))
        published = _runs_text(r.get("publishedTimeText"))   # relative, e.g. "2 days ago"
        views = _runs_text(r.get("viewCountText"))
        snippet = ""
        for s in r.get("detailedMetadataSnippets", []):
            snippet += _runs_text(s.get("snippetText"))
        text = clean_text(f"{title}. {snippet}")
        if not matches_keywords(f"{title} {snippet}", keywords):
            continue
        posts.append(Post(
            platform="youtube",
            post_id=vid,
            url=f"https://www.youtube.com/watch?v={vid}",
            text=text,
            author=channel,
            timestamp=None,                # only relative time on search page
            event_id=event_id,
            query=query,
            metrics={"views": views, "published_relative": published},
        ))
    return posts


def _renderer_to_post(r: dict, keywords: list[str], event_id: str, query: str) -> Post | None:
    vid = r.get("videoId", "")
    if not vid:
        return None
    title = _runs_text(r.get("title"))
    channel = _runs_text(r.get("ownerText")) or _runs_text(r.get("longBylineText"))
    published = _runs_text(r.get("publishedTimeText"))
    views = _runs_text(r.get("viewCountText"))
    snippet = "".join(_runs_text(s.get("snippetText")) for s in r.get("detailedMetadataSnippets", []))
    if not matches_keywords(f"{title} {snippet}", keywords):
        return None
    return Post(platform="youtube", post_id=vid,
                url=f"https://www.youtube.com/watch?v={vid}",
                text=clean_text(f"{title}. {snippet}"), author=channel, lang="en",
                timestamp=None, event_id=event_id, query=query,
                metrics={"views": views, "published_relative": published})


def collect_search_deep(query: str, keywords: list[str] | None = None, event_id: str = "",
                        scroll_rounds: int = 8, max_videos: int = 200) -> list[Post]:
    """Browser + scroll: loads the search page and scrolls to trigger youtubei/v1/search
    continuation XHRs, parsing videoRenderers from BOTH the initial data and every captured
    continuation response. Yields far more than the static ~20-result first page.
    """
    from scrapling.fetchers import StealthyFetcher

    keywords = keywords or []
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"

    def _scroll(page):
        for _ in range(scroll_rounds):
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(1400)
        return page

    page = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=90000, wait=4000,
                                 capture_xhr=r"/youtubei/v1/search", page_action=_scroll)
    renderers: list[dict] = []
    body = getattr(page, "html_content", None) or str(page)
    data = _extract_initial_data(body)
    if data:
        _walk_video_renderers(data, renderers)
    for resp in (getattr(page, "xhr_responses", None) or getattr(page, "captured_xhr", None) or []):
        raw = resp.get("body") if isinstance(resp, dict) else getattr(resp, "body", None)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "ignore")
        try:
            _walk_video_renderers(json.loads(raw), renderers)
        except (TypeError, ValueError):
            continue
    seen, posts = set(), []
    for r in renderers:
        vid = r.get("videoId", "")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        p = _renderer_to_post(r, keywords, event_id, query)
        if p:
            posts.append(p)
        if len(posts) >= max_videos:
            break
    return posts


if __name__ == "__main__":
    posts = collect_search_deep("Black Myth Wukong", event_id="smoke", scroll_rounds=8)
    print(f"deep collected {len(posts)} videos")
    for p in posts[:5]:
        print(f"  [{p.author}] {p.text[:75]}  ({p.metrics.get('views')})")
    with JsonlWriter("out/youtube_smoke.jsonl") as w:
        for p in posts:
            w.write(p)
        print("written:", w.written)
