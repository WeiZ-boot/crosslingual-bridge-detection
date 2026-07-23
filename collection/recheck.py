from scrapling.fetchers import StealthyFetcher

print("=== REDDIT: StealthyFetcher, solve_cloudflare=True, real waits ===")
try:
    p = StealthyFetcher.fetch(
        "https://www.reddit.com/r/worldnews/new/",
        headless=True, solve_cloudflare=True, network_idle=True,
        timeout=90000, wait=3000, google_search=True,
    )
    print("status:", p.status)
    body = p.body.decode("utf-8","ignore") if isinstance(p.body,bytes) else p.body
    print("len(body):", len(body))
    print("title:", (p.css("title::text").get() or "")[:80])
    print("'blocked/whoa':", any(k in body.lower() for k in ["whoa there","blocked","network policy"]))
    titles = p.css("a[slot=title]::text").getall() or p.css("h3::text").getall()
    print("post titles:", len(titles))
    for t in titles[:5]: print("   -", t[:70])
except Exception as e:
    print("ERR:", type(e).__name__, str(e)[:200])
