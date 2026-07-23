"""Inter-annotator reliability: Claude (hand, partial) vs GPT-5.5 (full set), on the overlap.

Claude hand-labeled the first 848 pairs (alphabetical by event) as an independent annotator;
GPT-5.5 labeled the full 13,992. This computes raw agreement + Cohen's kappa on the overlap for:
binary is_bridge, 4-class {no,A,B,C}, A/B/C among mutual bridges, and C-vs-rest (the hard class),
plus the 4-class confusion matrix. Run:  python3 agreement_claude_gpt55.py
"""
import json, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load(p):
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(HERE / p) if l.strip())}

def lab4(r):
    return r["bridge_type"] if r["is_bridge"] == "yes" else "no"

def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0

def main():
    cl, gp = load("llm_labels_claude.jsonl"), load("llm_labels_gpt55.jsonl")
    ov = [k for k in cl if k in gp]
    print(f"Claude={len(cl)}  GPT-5.5={len(gp)}  overlap={len(ov)}\n")

    for name, fa, fb in [
        ("is_bridge yes/no", lambda k: cl[k]["is_bridge"], lambda k: gp[k]["is_bridge"]),
        ("4-class no/A/B/C", lambda k: lab4(cl[k]), lambda k: lab4(gp[k])),
        ("C vs rest", lambda k: "C" if lab4(cl[k]) == "C" else "x",
                      lambda k: "C" if lab4(gp[k]) == "C" else "x"),
    ]:
        po, k = kappa([fa(k) for k in ov], [fb(k) for k in ov])
        print(f"[{name:>16}]  raw={po:.3f}  kappa={k:.3f}")

    both = [k for k in ov if cl[k]["is_bridge"] == "yes" and gp[k]["is_bridge"] == "yes"]
    po, k = kappa([cl[k]["bridge_type"] for k in both], [gp[k]["bridge_type"] for k in both])
    print(f"[A/B/C|both bridge]  raw={po:.3f}  kappa={k:.3f}  (n={len(both)})")

    cats = ["no", "A", "B", "C"]
    m = collections.Counter((lab4(cl[k]), lab4(gp[k])) for k in ov)
    print("\nConfusion 4-class (row=Claude, col=GPT-5.5):")
    print("        " + "".join(f"{c:>7}" for c in cats) + "   | total")
    for r in cats:
        row = [m[(r, c)] for c in cats]
        print(f"  {r:>5} " + "".join(f"{v:>7}" for v in row) + f"   | {sum(row)}")
    print("  total " + "".join(f"{sum(m[(r,c)] for r in cats):>7}" for c in cats))
    agree = sum(m[(c, c)] for c in cats)
    print(f"\noverlap={len(ov)}  4-class agree={agree} ({agree/len(ov):.1%})  disagree={len(ov)-agree}")

if __name__ == "__main__":
    main()
