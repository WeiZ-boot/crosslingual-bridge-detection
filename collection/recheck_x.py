from scrapling.fetchers import StealthyFetcher

# Nitter-free guest test: X search/profile w/ full stealth, capture GraphQL XHR JSON
print("=== X: profile via StealthyFetcher, capture_xhr GraphQL ===")
captured = {}
try:
    p = StealthyFetcher.fetch(
        "https://x.com/Reuters",
        headless=True, network_idle=True, timeout=90000, wait=5000,
        capture_xhr=r"(UserTweets|TweetResultByRestId|SearchTimeline)",
    )
    print("status:", p.status)
    body = p.body.decode("utf-8","ignore") if isinstance(p.body,bytes) else p.body
    print("title:", (p.css("title::text").get() or "")[:80])
    tweets = p.css("[data-testid=tweetText]")
    print("rendered tweets:", len(tweets))
    for t in tweets[:3]: print("   -", t.text.clean()[:80])
    # captured XHR JSON (X's own API responses)
    xhr = getattr(p, "xhr_responses", None) or getattr(p, "captured_xhr", None)
    print("captured_xhr attr present:", xhr is not None, "| count:", len(xhr) if xhr else 0)
    print("login-wall in body:", any(k in body for k in ["Sign in to X","Log in","Sign in"]))
except Exception as e:
    print("ERR:", type(e).__name__, str(e)[:200])
