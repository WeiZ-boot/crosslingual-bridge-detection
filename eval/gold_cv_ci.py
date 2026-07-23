"""Grouped k-fold + cluster-bootstrap confidence intervals on the frozen consensus gold.

The training side is fixed (GPT-5.5 silver on the 59 non-gold events), so uncertainty is measured on
the TEST side: how much do macro-F1 / C-F1 vary across the 11 gold EVENTS? Two views:
  - grouped 5-fold: partition the 11 events into 5 folds; per-fold metric -> mean +/- 95% CI (t, df=4)
  - cluster bootstrap: resample the 11 events with replacement 5000x -> 2.5/97.5 percentile CI
    (robust when C is sparse/uneven across events -- the recommended CI for C-F1).

Per-pair predictions: cross-encoder from gold_ce_preds.json; baselines recomputed here (deterministic).
Run (anaconda base, after phase4b_bench.py has written gold_ce_preds.json):  python3 gold_cv_ci.py
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
CLASSES = ["no", "A", "B", "C"]


def load(p):
    return {r["pair_id"]: r for r in (json.loads(l) for l in open(HERE / p) if l.strip())}


def lab4(r):
    return r["bridge_type"] if r["is_bridge"] == "yes" else ("no" if r["is_bridge"] == "no" else None)


def macro_and_C(yt, yp):
    return (f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0),
            f1_score(yt, yp, average=None, labels=CLASSES, zero_division=0)[CLASSES.index("C")])


def main():
    gp, nano = load("llm_labels_gpt55.jsonl"), load("llm_labels_v2.jsonl")
    gold = [json.loads(l) for l in (HERE / "gold_consensus.jsonl").read_text().splitlines() if l.strip()]
    gold_events = set(json.loads((HERE / "gold_events.json").read_text()))
    ce_pred = json.loads((HERE / "gold_ce_preds.json").read_text())

    # embeddings for the trained baselines
    v2_ids = [json.loads(l)["pair_id"] for l in (HERE / "llm_labels_v2.jsonl").read_text().splitlines() if l.strip()]
    z = np.load(HERE / "pair_emb_v2.npz"); EU, ZV = z["eu"], z["zv"]; idx = {p: i for i, p in enumerate(v2_ids)}

    def feats(pids):
        i = np.array([idx[p] for p in pids]); eu, zv = EU[i], ZV[i]
        return np.concatenate([eu, zv, np.abs(eu - zv), eu * zv], 1), np.sum(eu * zv, 1)

    train_ids = [k for k, r in gp.items() if r["event_id"] not in gold_events and lab4(r) in CLASSES]
    ytr = np.array([lab4(gp[k]) for k in train_ids]); Xtr, str_ = feats(train_ids)
    gids = [r["pair_id"] for r in gold]; yt = np.array([r["label"] for r in gold])
    ev = np.array([r["event_id"] for r in gold]); Xg, sg = feats(gids)

    # per-pair predictions per method
    maj = Counter(ytr).most_common(1)[0][0]
    preds = {
        "majority": np.array([maj] * len(gids)),
        "sim-only-LR": LogisticRegression(max_iter=1000).fit(str_.reshape(-1, 1), ytr).predict(sg.reshape(-1, 1)),
        "bi-encoder-MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=0).fit(Xtr, ytr).predict(Xg),
        "zeroshot-nano": np.array([lab4(nano[k]) if k in nano and lab4(nano[k]) else "no" for k in gids]),
        "cross-encoder": np.array([ce_pred[k] for k in gids]),
    }
    events = sorted(set(ev))
    print(f"gold={len(gids)} events={len(events)}  C per event:",
          {e: int(((ev == e) & (yt == 'C')).sum()) for e in events})

    # ----- grouped 5-fold (partition events) -----
    rng = np.random.default_rng(0)
    ev_shuf = list(events); rng.shuffle(ev_shuf)
    folds = [ev_shuf[i::5] for i in range(5)]
    print("\n=== grouped 5-fold (mean +/- 95% CI, t df=4) ===")
    t95 = 2.776  # t_0.975, df=4
    kfold = {}
    for name, yp in preds.items():
        mf, cf = [], []
        for fo in folds:
            m = np.isin(ev, fo)
            a, c = macro_and_C(yt[m], yp[m]); mf.append(a); cf.append(c)
        mf, cf = np.array(mf), np.array(cf)
        kfold[name] = {"macro_mean": mf.mean(), "macro_ci": t95 * mf.std(ddof=1) / np.sqrt(5),
                       "C_mean": cf.mean(), "C_ci": t95 * cf.std(ddof=1) / np.sqrt(5)}
        print(f"  {name:<16} macroF1={mf.mean():.3f}±{kfold[name]['macro_ci']:.3f}   "
              f"C-F1={cf.mean():.3f}±{kfold[name]['C_ci']:.3f}")

    # ----- cluster bootstrap over events (5000x) -----
    print(f"\n=== cluster bootstrap over {len(events)} events (95% CI, 5000 resamples) ===")
    B = 5000
    pair_by_ev = {e: np.where(ev == e)[0] for e in events}
    boot = {}
    for name, yp in preds.items():
        ms, cs = np.empty(B), np.empty(B)
        for b in range(B):
            samp = rng.choice(events, size=len(events), replace=True)
            ii = np.concatenate([pair_by_ev[e] for e in samp])
            ms[b], cs[b] = macro_and_C(yt[ii], yp[ii])
        boot[name] = {"macro": (np.percentile(ms, 2.5), np.percentile(ms, 97.5)),
                      "C": (np.percentile(cs, 2.5), np.percentile(cs, 97.5)),
                      "macro_med": np.median(ms), "C_med": np.median(cs)}
        print(f"  {name:<16} macroF1 95%CI=[{boot[name]['macro'][0]:.3f},{boot[name]['macro'][1]:.3f}]   "
              f"C-F1 95%CI=[{boot[name]['C'][0]:.3f},{boot[name]['C'][1]:.3f}]")

    (HERE / "gold_cv_ci_result.json").write_text(json.dumps(
        {"kfold": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in kfold.items()},
         "bootstrap": {k: {"macro_CI": [float(x) for x in v["macro"]], "C_CI": [float(x) for x in v["C"]],
                           "macro_med": float(v["macro_med"]), "C_med": float(v["C_med"])} for k, v in boot.items()}},
        indent=2))
    print("\nsaved gold_cv_ci_result.json")


if __name__ == "__main__":
    main()
