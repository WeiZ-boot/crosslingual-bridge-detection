from scrapling.fetchers import StealthyFetcher
import json

p = StealthyFetcher.fetch(
    "https://x.com/Reuters",
    headless=True, network_idle=True, timeout=90000, wait=5000,
    capture_xhr=r"UserTweets",
)
xhr = getattr(p, "xhr_responses", None) or getattr(p, "captured_xhr", None) or []
print("captured responses:", len(xhr))
for resp in xhr:
    # resp may be dict-like with body/text
    raw = resp.get("body") if isinstance(resp, dict) else getattr(resp, "body", None)
    if raw is None and isinstance(resp, dict): raw = resp.get("text")
    if isinstance(raw, bytes): raw = raw.decode("utf-8","ignore")
    try:
        data = json.loads(raw)
    except Exception as e:
        print("  (non-json or empty:", str(e)[:60], ")"); 
        print("  attrs:", resp.keys() if isinstance(resp,dict) else dir(resp)[:10])
        continue
    # walk for full_text
    texts = []
    def walk(o):
        if isinstance(o, dict):
            if "full_text" in o: texts.append(o["full_text"])
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(data)
    print("  tweets in JSON:", len(texts))
    for t in texts[:6]: print("   -", t.replace("\n"," ")[:90])
