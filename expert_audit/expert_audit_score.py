"""Score the expert audit: join the filled expert_audit_sheet.csv against expert_audit_key.jsonl.

Reports expert<->gold agreement overall, per class, per source (agree vs adjudicated), Cohen's kappa,
and the confusion matrix. Rows with an empty 'your_label' column are skipped (partial audits OK).

Run:  python3 expert_audit_score.py
"""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = ["no", "A", "B", "C"]


def main():
    key = {r["audit_id"]: r for r in (json.loads(l) for l in open(HERE / "expert_audit_key.jsonl") if l.strip())}
    rows = []
    with open(HERE / "expert_audit_sheet.csv", newline="") as f:
        for r in csv.DictReader(f):
            lab = (r.get("your_label (no/A/B/C)") or "").strip()
            if lab in CLASSES:
                rows.append((int(r["audit_id"]), lab))
    if not rows:
        print("no filled rows found in expert_audit_sheet.csv"); return

    # Batches 1-2: stratified VALIDATION of gold v2.2 (headline numbers).
    # Batch 3: targeted cleanup (adjudicated-C + Bno), biased — reported separately.
    # Batch 4: stratified VALIDATION of the extended-consensus tier (Route A) — its own headline.
    # Batch 5: exhaustive C cleanup (extended-consensus C + original-consensus unaudited C), biased.
    val_rows = [(i, l) for i, l in rows if key[i].get("batch", 1) in (1, 2)]
    for bn, desc in ((3, "targeted cleanup (adjudicated-C + Bno), biased"),
                     (5, "exhaustive C cleanup (ext-C + orig-consensus-C), biased")):
        b = [(i, l) for i, l in rows if key[i].get("batch", 1) == bn]
        if b:
            a = sum(key[i]["label"] == l for i, l in b)
            print(f"batch {bn} ({desc}): {len(b)} filled, confirms current label {a}/{len(b)} = {a/len(b):.3f}")
            bc = [(i, l) for i, l in b if key[i]["label"] == "C"]
            if bc:
                ca = sum(l == "C" for _, l in bc)
                print(f"  of which C rows: expert confirms C {ca}/{len(bc)} = {ca/len(bc):.3f}")
    b4 = [(i, l) for i, l in rows if key[i].get("batch", 1) == 4]
    if b4:
        a = sum(key[i]["label"] == l for i, l in b4)
        print(f"\nbatch 4 (VALIDATION of extended-consensus tier, stratified): {len(b4)} filled, "
              f"agreement {a}/{len(b4)} = {a/len(b4):.3f}   <- compare vs 0.848 (original consensus tier)")
        for c in CLASSES:
            idx = [(i, l) for i, l in b4 if key[i]["label"] == c]
            if idx:
                ca = sum(l == c for _, l in idx)
                print(f"    {c:3s} n={len(idx):3d} agree={ca/len(idx):.3f}")
    if any(key[i].get("batch", 1) >= 3 for i, _ in rows):
        print("--- gold v2.2 headline validation below uses batches 1-2 only ---\n")
    rows = val_rows
    if not rows:
        print("no batch 1-2 rows filled"); return

    yt = [key[i]["label"] for i, _ in rows]          # gold
    yp = [lab for _, lab in rows]                    # expert
    src = [key[i]["source"] for i, _ in rows]
    n = len(rows)
    agree = sum(a == b for a, b in zip(yt, yp))
    print(f"audited (batches 1-2): {n}/500   expert<->gold agreement: {agree}/{n} = {agree/n:.3f}")

    # Cohen's kappa
    po = agree / n
    ct, cp = Counter(yt), Counter(yp)
    pe = sum(ct[c] * cp[c] for c in CLASSES) / (n * n)
    print(f"Cohen's kappa: {(po - pe) / (1 - pe):.3f}")

    print("\nper-class (gold class -> agreement):")
    for c in CLASSES:
        idx = [i for i, t in enumerate(yt) if t == c]
        if idx:
            a = sum(yp[i] == c for i in idx)
            print(f"  {c:3s} n={len(idx):3d}  agree={a/len(idx):.3f}")

    print("\nper-source:")
    for s in ("agree", "adjudicated"):
        idx = [i for i, x in enumerate(src) if x == s]
        if idx:
            a = sum(yt[i] == yp[i] for i in idx)
            print(f"  {s:12s} n={len(idx):3d}  agree={a/len(idx):.3f}")

    print("\nconfusion (rows=gold, cols=expert, order no/A/B/C):")
    m = {(a, b): 0 for a in CLASSES for b in CLASSES}
    for a, b in zip(yt, yp):
        m[(a, b)] += 1
    for a in CLASSES:
        print(f"  {a:3s} " + " ".join(f"{m[(a,b)]:4d}" for b in CLASSES))


if __name__ == "__main__":
    main()
