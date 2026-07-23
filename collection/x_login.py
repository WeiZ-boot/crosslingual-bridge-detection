"""One-time X login -> persistent profile dir for keyword search.

Opens a VISIBLE browser window with a persistent user_data_dir. You log in by
hand (username/password, plus any 2FA/captcha). The script polls for the
`auth_token` cookie and exits once login is detected, persisting the session.

Run:
    .venv/bin/python x_login.py                 # default dir ./x_profile
    .venv/bin/python x_login.py --dir ./x_profile --minutes 5

Then collection:
    .venv/bin/python run_collection.py --event <id> --x-user-data-dir ./x_profile
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
    """Remove stale Singleton* locks so a re-run doesn't fail to launch."""
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        try:
            os.remove(os.path.join(profile, name))
        except OSError:
            pass


def _has_auth_cookie(profile: str) -> bool:
    """Read the on-disk cookie DB for a real auth_token (definitive login check)."""
    db = os.path.join(profile, "Default", "Cookies")
    if not os.path.exists(db):
        return False
    tmp = tempfile.mktemp()
    try:
        shutil.copy(db, tmp)
        con = sqlite3.connect(tmp)
        rows = con.execute(
            "SELECT name FROM cookies WHERE (host_key LIKE '%x.com' OR host_key LIKE '%twitter.com')"
        ).fetchall()
        con.close()
        return "auth_token" in {r[0] for r in rows}
    except sqlite3.Error:
        return False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _wait_for_login(max_seconds: int):
    """page_action: poll the browser's cookies until auth_token appears."""
    def action(page):
        print(f"\n>>> Log in to X in the opened window. Waiting up to {max_seconds}s...\n")
        for i in range(max_seconds):
            try:
                cookies = page.context.cookies()
            except Exception:
                cookies = []
            if any(c.get("name") == "auth_token" and c.get("value") for c in cookies):
                print(">>> auth_token detected — login successful. Saving session.")
                page.wait_for_timeout(1500)   # let cookies flush to disk
                return page
            if i % 15 == 0 and i:
                print(f"    ...still waiting ({i}s)")
            page.wait_for_timeout(1000)
        print(">>> Timed out waiting for login (no auth_token cookie).")
        return page
    return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="x_profile", help="persistent profile dir")
    ap.add_argument("--minutes", type=float, default=5.0, help="max wait for manual login")
    args = ap.parse_args()

    profile = str(Path(args.dir).resolve())
    Path(profile).mkdir(parents=True, exist_ok=True)
    _clear_locks(profile)
    print(f"profile dir: {profile}")

    StealthyFetcher.fetch(
        "https://x.com/login",
        headless=False,                       # visible window for manual login
        user_data_dir=profile,
        timeout=int(args.minutes * 60 * 1000) + 30000,
        page_action=_wait_for_login(int(args.minutes * 60)),
    )

    # definitive verify: is auth_token actually in the saved cookie DB?
    _clear_locks(profile)
    if _has_auth_cookie(profile):
        print("\n✅ auth_token saved. Session is logged in.")
        print("   Use it with: --x-user-data-dir", profile)
    else:
        print("\n⚠️  No auth_token in saved profile — login was NOT completed.")
        print("   Re-run and finish login (until you see your home feed) in the window.")


if __name__ == "__main__":
    main()
