"""FUSE-C+ V2 signal: self-consistency vote distribution from the STRONG few-shot LLM.

Samples the original reasoning few-shot elicitation (fewshot_eval.classify, default temperature)
k times per pair; the vote fractions are a calibrated confidence for the strong arm — the signal
the fight-zone diagnostic said was missing (logprobs are unavailable at reasoning effort).

Checkpointed per (pair_id, sample_k); resumable. Output llm_votes.jsonl:
  {"pair_id", "votes": {no,A,B,C}, "k"}
"""
import argparse, json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "eval"))
from openai import OpenAI
from fewshot_eval import classify, CLASSES

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--gold", default="gold_consensus.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="llm_votes.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(HERE/args.gold) if l.strip()]
    ckpt = HERE / (args.out + ".ckpt")
    have = Counter()
    if ckpt.exists():
        for l in open(ckpt):
            if l.strip(): have[json.loads(l)["pair_id"]] += 1
    jobs = []
    for r in gold:
        for _ in range(args.k - have.get(r["pair_id"], 0)):
            jobs.append(r)
    print(f"gold={len(gold)} k={args.k} calls_done={sum(have.values())} todo={len(jobs)}", flush=True)

    client = OpenAI(timeout=90.0, max_retries=2)
    lock = Lock(); done = 0; errs = 0; first_err = None
    with open(ckpt, "a") as ck, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(classify, client, args.model, r) for r in jobs]
        for f in as_completed(futs):
            pid, lab = f.result()
            with lock:
                done += 1
                if isinstance(lab, str) and lab.startswith("ERR"):
                    errs += 1
                    if first_err is None: first_err = lab
                else:
                    ck.write(json.dumps({"pair_id": pid, "vote": lab}) + "\n"); ck.flush()
                if done % 500 == 0: print(f"  {done}/{len(jobs)} (errs={errs})", flush=True)
    if errs:
        print(f"WARNING: {errs}/{len(jobs)} calls failed (not checkpointed; rerun to fill). First: {first_err}")

    votes = {}
    for l in open(ckpt):
        if l.strip():
            r = json.loads(l); votes.setdefault(r["pair_id"], Counter())[r["vote"]] += 1
    with open(HERE/args.out, "w") as w:
        for pid, c in votes.items():
            w.write(json.dumps({"pair_id": pid, "votes": {cl: c.get(cl, 0) for cl in CLASSES},
                                "k": sum(c.values())}) + "\n")
    ks = Counter(sum(c.values()) for c in votes.values())
    print(f"wrote {args.out} ({len(votes)} pairs; k distribution {dict(ks)})")

if __name__ == "__main__":
    main()
