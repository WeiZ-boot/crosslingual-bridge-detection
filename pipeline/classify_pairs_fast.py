"""Concurrent LLM labeling — reuses classify()/make_client() from classify_pairs but runs a
thread pool so the full candidate set is feasible (~30-45 min vs ~7h sequential). Resumes by
skipping pair_ids already in --out. Same output schema as classify_pairs.
"""
from __future__ import annotations
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from classify_pairs import make_client, classify

_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="annotation_set_v2.jsonl")
    ap.add_argument("--out", default="llm_labels_v2.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["pair_id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["pair_id"] not in done]
    print(f"to label: {len(todo)} (skipping {len(done)} done)  workers={args.workers}", flush=True)

    client = make_client()
    fh = out_path.open("a", encoding="utf-8")
    n = [0]

    def work(r):
        lab = classify(client, r)
        rec = {**{k: r[k] for k in ("pair_id", "event_id", "sim", "band")},
               "en_text": r["en_text"], "zh_text": r["zh_text"], **lab}
        with _lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n[0] += 1
            if n[0] % 200 == 0:
                print(f"  {n[0]}/{len(todo)}", flush=True)
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  err: {str(e)[:100]}", flush=True)
    fh.close()
    print(f"done -> {out_path}  (total labeled now {len(done)+n[0]})", flush=True)


if __name__ == "__main__":
    main()
