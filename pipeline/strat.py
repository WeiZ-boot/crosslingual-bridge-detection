"""Tiny harness to label the stratified-gap first pass, BLIND (gpt_label hidden), into a SEPARATE
file (llm_labels_claude_strat.jsonl) so it never races the main annotation process.

  python3 strat.py show 50 [offset]   # print next unlabeled pairs (no gpt hint)
  python3 strat.py add /tmp/s.jsonl   # append labels {pair_id,is_bridge,bridge_type,rationale,confidence}
  python3 strat.py progress
Input: /tmp/strat_work.jsonl (has pair_id,event_id,gpt_label,en_text,zh_text). Output schema matches
llm_labels_claude.jsonl (minus gpt_label) so build_stratified.py can read it.
"""
import json, sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
IN = Path("/tmp/strat_work.jsonl")
OUT = HERE / "llm_labels_claude_strat.jsonl"
VALID = {"A", "B", "C", ""}


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def done_ids():
    return {json.loads(l)["pair_id"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "progress"
    rows = load(IN)
    if cmd == "show":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        done = done_ids()
        todo = [r for r in rows if r["pair_id"] not in done][:n]
        print(f"=== {len(todo)} pairs to label (BLIND) ===\n")
        for r in todo:
            print(f"PAIR_ID: {r['pair_id']}   [{r['event_id']}]")
            print(f"  EN: {r['en_text']}")
            print(f"  ZH: {r['zh_text']}\n")
    elif cmd == "add":
        meta = {r["pair_id"]: r for r in rows}
        done = done_ids()
        added = bad = 0
        with OUT.open("a", encoding="utf-8") as fh:
            for line in Path(sys.argv[2]).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    pid, ib, bt = r["pair_id"], r["is_bridge"], r.get("bridge_type", "")
                    assert ib in ("yes", "no") and bt in VALID and pid in meta
                    if ib == "no":
                        bt = ""
                    if ib == "yes" and bt == "":
                        raise ValueError("yes needs A/B/C")
                except Exception as e:
                    print(f"  SKIP ({str(e)[:40]}): {line[:70]}"); bad += 1; continue
                if pid in done:
                    continue
                done.add(pid); m = meta[pid]
                fh.write(json.dumps({"pair_id": pid, "event_id": m["event_id"], "en_text": m["en_text"],
                                     "zh_text": m["zh_text"], "is_bridge": ib, "bridge_type": bt,
                                     "rationale": str(r.get("rationale", "")), "confidence": int(r.get("confidence", 2))},
                                    ensure_ascii=False) + "\n")
                added += 1
        print(f"added {added} (skipped {bad})")
    rows_done = load(OUT) if OUT.exists() else []
    print(f"progress: {len(rows_done)}/{len(rows)} strat-gap labeled  "
          f"types={dict(Counter((r['bridge_type'] if r['is_bridge']=='yes' else 'no') for r in rows_done))}")


if __name__ == "__main__":
    main()
