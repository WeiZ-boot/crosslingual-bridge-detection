"""FUSE-C+ Signal 1 runner: produce llm_probs.jsonl (per-class LLM probabilities).

Apply fewshot_eval_logprobs.patch first (adds classify_with_probs to fewshot_eval.py).
Checkpointed + resumable, same threading pattern as fewshot_v3_run.py.

Run:  python run_llm_probs.py --model gpt-5.4-nano --gold gold_consensus_v23.jsonl
      python run_llm_probs.py --model gpt-5.4      --gold gold_consensus_v23.jsonl
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "eval"))
from openai import OpenAI
from fewshot_eval import classify_with_probs, CLASSES

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--gold", default="gold_consensus_v23.jsonl")
    ap.add_argument("--out", default="llm_probs.jsonl")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(HERE/args.gold) if l.strip()]
    ckpt = HERE / (args.out + ".ckpt")
    have = set()
    if ckpt.exists():
        for l in open(ckpt):
            if l.strip(): have.add(json.loads(l)["pair_id"])
    todo = [r for r in gold if r["pair_id"] not in have]
    print(f"gold={len(gold)} have={len(have)} todo={len(todo)}", flush=True)

    client = OpenAI(timeout=60.0, max_retries=2)
    rows = [{"pair_id": r["pair_id"], "event_id": r["event_id"],
             "en_text": r["en_text"], "zh_text": r["zh_text"]} for r in todo]
    lock = Lock(); done = 0; errs = 0; first_err = None
    with open(ckpt, "a") as ck, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(classify_with_probs, client, args.model, r) for r in rows]
        for f in as_completed(futs):
            pid, lab, llm_p = f.result()
            with lock:
                done += 1
                if isinstance(lab, str) and lab.startswith("ERR"):
                    # do NOT checkpoint failures — a resume run retries them
                    errs += 1
                    if first_err is None: first_err = lab
                else:
                    ck.write(json.dumps({"pair_id": pid, "llm_label": lab, "llm_p": llm_p}) + "\n"); ck.flush()
                if done % 250 == 0: print(f"  {done}/{len(rows)} (errs={errs})", flush=True)
    if errs:
        print(f"WARNING: {errs}/{len(rows)} calls failed and were NOT checkpointed; first error: {first_err}")
        print("Re-run to retry the missing pairs.")

    # consolidate ckpt -> out (dedup, last wins)
    merged = {}
    for l in open(ckpt):
        if l.strip():
            r = json.loads(l); merged[r["pair_id"]] = r
    with open(HERE/args.out, "w") as w:
        for r in merged.values():
            w.write(json.dumps(r) + "\n")
    print(f"wrote {args.out} ({len(merged)} pairs)")

if __name__ == "__main__":
    main()
