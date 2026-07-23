"""Shared schema + helpers for Paper 2 English-side collection.

Unified Post record across X / Reddit / Telegram / YouTube so that downstream
cross-lingual bridge annotation has a consistent format. One JSONL line = one post.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Post:
    platform: str                 # "x" | "reddit" | "telegram" | "youtube"
    post_id: str                  # platform-native id
    url: str
    text: str
    author: str = ""
    lang: str = "en"              # English-side anchor; filled best-effort
    timestamp: Optional[str] = None   # ISO-8601 UTC if known
    event_id: str = ""            # which hot-topic event this was collected for
    query: str = ""               # the keyword/channel that surfaced it
    metrics: dict = field(default_factory=dict)   # likes/forwards/score/views...
    collected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def matches_keywords(text: str, keywords: list[str]) -> bool:
    """Case-insensitive OR match. Empty keyword list = match everything."""
    if not keywords:
        return True
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def in_window(ts_iso: Optional[str], start: Optional[str], end: Optional[str]) -> bool:
    """Inclusive date-window filter on ISO timestamps. Missing ts => keep (can't exclude)."""
    if not ts_iso or (not start and not end):
        return True
    try:
        t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except ValueError:
        return True
    if start and t < datetime.fromisoformat(start).replace(tzinfo=t.tzinfo):
        return False
    if end and t > datetime.fromisoformat(end).replace(tzinfo=t.tzinfo):
        return False
    return True


class JsonlWriter:
    """Append JSONL writer that dedupes by (platform, post_id) against BOTH the current run
    AND any rows already in the file — so re-running a collector never appends duplicates."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[tuple[str, str]] = set()
        # preload existing keys so re-runs are idempotent (the file is append-mode)
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    self._seen.add((r.get("platform", ""), r.get("post_id", "")))
                except (ValueError, TypeError):
                    pass
        self._fh = self.path.open("a", encoding="utf-8")
        self.written = 0

    def write(self, post: Post) -> bool:
        key = (post.platform, post.post_id)
        if key in self._seen:
            return False
        self._seen.add(key)
        self._fh.write(post.to_json() + "\n")
        self.written += 1
        return True

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()
