"""Dataset-wide dedup. JsonlWriter only deduped within a single run (append mode), so re-running
a collector on an event that already had data appended duplicates. This rewrites every
out/<event>/<platform>.jsonl keeping the FIRST occurrence per post_id (then per exact text),
and reports total -> unique. Idempotent.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
PLATS = ["reddit", "x", "youtube", "hackernews", "bluesky", "telegram",
         "weibo", "bilibili", "zhihu", "douyin"]


def dedup_file(p: Path) -> tuple[int, int]:
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    seen_id, seen_text, out = set(), set(), []
    for r in rows:
        pid = r.get("post_id") or ""
        txt = (r.get("text") or "").strip()
        if pid and pid in seen_id:
            continue
        if txt and txt in seen_text:
            continue
        if pid:
            seen_id.add(pid)
        if txt:
            seen_text.add(txt)
        out.append(r)
    if len(out) != len(rows):
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    return len(rows), len(out)


def main():
    tot_before = tot_after = 0
    changed = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        for plat in PLATS:
            f = d / f"{plat}.jsonl"
            if not f.exists():
                continue
            b, a = dedup_file(f)
            tot_before += b
            tot_after += a
            if b != a:
                changed.append((d.name, plat, b, a))
    print(f"files with duplicates removed: {len(changed)}")
    for name, plat, b, a in sorted(changed, key=lambda x: x[2] - x[3], reverse=True)[:25]:
        print(f"  {name:<26} {plat:<11} {b:>6} -> {a:>6}  (-{b-a})")
    print(f"\nTOTAL posts {tot_before} -> {tot_after}  (removed {tot_before - tot_after} duplicates)")


if __name__ == "__main__":
    main()
