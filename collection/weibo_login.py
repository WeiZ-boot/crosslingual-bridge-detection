"""One-time Weibo login -> persistent profile for historical (s.weibo.com) date-range search.

Opens a VISIBLE browser at the Weibo login page. Log in by hand — usually scan the QR code with the
Weibo mobile app (or phone + password). The script polls for the `SUB` auth cookie and saves the
session to ./weibo_profile. Then collect_weibo_historical reuses that dir.

Run:  .venv/bin/python weibo_login.py --minutes 8
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from scrapling.fetchers import StealthyFetcher


def _clear_locks(profile: str):
    for n in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        try:
            os.remove(os.path.join(profile, n))
        except OSError:
            pass


def _has_auth(profile: str) -> bool:
    db = os.path.join(profile, "Default", "Cookies")
    if not os.path.exists(db):
        return False
    tmp = tempfile.mktemp()
    try:
        shutil.copy(db, tmp)
        con = sqlite3.connect(tmp)
        rows = con.execute("SELECT name FROM cookies WHERE host_key LIKE '%weibo%'").fetchall()
        con.close()
        return "SUB" in {r[0] for r in rows}
    except sqlite3.Error:
        return False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _wait_for_login(max_seconds: int):
    def action(page):
        print(f"\n>>> Scan the QR with the Weibo app (or log in). Waiting up to {max_seconds}s...\n")
        for i in range(max_seconds):
            try:
                cookies = page.context.cookies()
            except Exception:
                cookies = []
            if any(c.get("name") == "SUB" and c.get("value") for c in cookies):
                print(">>> SUB cookie detected — login successful. Saving session.")
                page.wait_for_timeout(1500)
                return page
            if i % 15 == 0 and i:
                print(f"    ...still waiting ({i}s)")
            page.wait_for_timeout(1000)
        print(">>> Timed out (no SUB cookie).")
        return page
    return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="weibo_profile")
    ap.add_argument("--minutes", type=float, default=8.0)
    args = ap.parse_args()
    profile = str(Path(args.dir).resolve())
    Path(profile).mkdir(parents=True, exist_ok=True)
    _clear_locks(profile)
    print(f"profile dir: {profile}")

    StealthyFetcher.fetch(
        "https://passport.weibo.com/sso/signin?entry=miniblog&source=miniblog",
        headless=False, user_data_dir=profile,
        timeout=int(args.minutes * 60 * 1000) + 30000,
        page_action=_wait_for_login(int(args.minutes * 60)))

    _clear_locks(profile)
    if _has_auth(profile):
        print("\n✅ SUB cookie saved. Use it with collect_weibo_historical (--profile weibo_profile)")
    else:
        print("\n⚠️  No SUB cookie — login not completed. Re-run and finish login in the window.")


if __name__ == "__main__":
    main()
