"""Re-run the v2 downstream bi-encoder pipeline on a silver-label file and compare nano vs GPT-5.5.

Mirrors temporal_v2.py (direction/reframed_label) + experiments_v2.py (EXP A/B/C) exactly, but
parametrized by labels file. Reuses pair_emb_v2.npz (BGE-M3 embeddings depend only on EN/ZH text,
identical per pair_id across label sets) by re-aligning rows via pair_id -- no recompute.

Run:  python3 downstream_compare.py
Writes llm_labels_<tag>_reframed.jsonl for each tag; prints a side-by-side metrics table.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import re

HERE = Path(__file__).resolve().parent
CONC_H = 24.0
RNG = 0

# ---- inlined load_en/load_zh from unified_candidates (avoids its module-level torch import) ----
OUT = HERE / "out"
CPHOT = HERE / "../../../07-code/M-BridgeNet/data/cphot/raw"
EN_PLATS = ["reddit", "x", "youtube", "hackernews", "bluesky", "telegram"]
ZH_PLATS = ["weibo", "bilibili", "zhihu", "douyin"]
EN_CAP = ZH_CAP = 1000
MINLEN = 8


def _clean(t):
    return re.sub(r"\s+", " ", (t or "").strip())


def load_en(eid):
    seen, rows = set(), []
    for plat in EN_PLATS:
        f = OUT / eid / f"{plat}.jsonl"
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            t = _clean(r.get("text"))
            if len(t) < MINLEN or t[:80] in seen:
                continue
            seen.add(t[:80])
            rows.append({"text": t, "platform": plat, "ts": r.get("timestamp")})
    rows.sort(key=lambda r: -len(r["text"]))
    return rows[:EN_CAP]


def load_zh(eid):
    f = CPHOT / f"{eid}_001.json"
    seen, rows = set(), []
    if f.is_file():
        for p in json.loads(f.read_text()).get("posts", []):
            t = _clean(p.get("text"))
            if len(t) < MINLEN or t[:60] in seen:
                continue
            seen.add(t[:60])
            rows.append({"text": t, "platform": p.get("platform", "weibo"), "ts": p.get("timestamp")})
    else:
        for plat in ZH_PLATS:
            pf = OUT / eid / f"{plat}.jsonl"
            if not pf.exists():
                continue
            for l in pf.read_text().splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                t = _clean(r.get("text"))
                if len(t) < MINLEN or t[:60] in seen:
                    continue
                seen.add(t[:60])
                rows.append({"text": t, "platform": plat, "ts": r.get("timestamp")})
    rows.sort(key=lambda r: -len(r["text"]))
    return rows[:ZH_CAP]

# ---- embedding reuse: map pair_id -> row in pair_emb_v2.npz (aligned to llm_labels_v2.jsonl) ----
_V2_IDS = [json.loads(l)["pair_id"] for l in (HERE / "llm_labels_v2.jsonl").read_text().splitlines() if l.strip()]
_Z = np.load(HERE / "pair_emb_v2.npz")
_EU, _ZV = _Z["eu"], _Z["zv"]
_IDX = {pid: i for i, pid in enumerate(_V2_IDS)}


def parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        try:
            return float(ts) / (1000.0 if ts > 1e12 else 1.0)
        except (TypeError, ValueError):
            return None
    s = str(ts).strip()
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp()
    except ValueError:
        return None


def plausible(ep, lo=1.4e9, hi=1.9e9):
    return ep is not None and lo < ep < hi


def temporal(labels_path):
    """Return reframed rows (temporal_v2 logic). Caches load_en/load_zh per event."""
    labels = [json.loads(l) for l in Path(labels_path).read_text().splitlines() if l.strip()]
    by_ev = defaultdict(list)
    for r in labels:
        by_ev[r["event_id"]].append(r)
    directed = []
    for eid, rows in by_ev.items():
        en_ts = [parse_ts(r.get("ts")) for r in load_en(eid)]
        zh_ts = [parse_ts(r.get("ts")) for r in load_zh(eid)]
        for r in rows:
            try:
                _, i, j = r["pair_id"].rsplit("_", 2)
                i, j = int(i), int(j)
                ets = en_ts[i] if i < len(en_ts) else None
                zts = zh_ts[j] if j < len(zh_ts) else None
            except (ValueError, IndexError):
                ets = zts = None
            lead = (zts - ets) / 3600.0 if (plausible(ets) and plausible(zts)) else None
            if lead is None:
                direction = "na"
            elif lead > CONC_H:
                direction = "EN->ZH"
            elif lead < -CONC_H:
                direction = "ZH->EN"
            else:
                direction = "concurrent"
            directed.append({**r, "lead_h": (round(lead, 1) if lead is not None else None),
                             "direction": direction})
    reframed = []
    for r in directed:
        if r.get("is_bridge") == "no":
            lab = "not_related"
        elif r["direction"] == "na":
            lab = f"bridge_{r.get('bridge_type')}_untimed"
        elif r["direction"] == "concurrent":
            lab = "parallel_coverage"
        else:
            lab = f"bridge_{r.get('bridge_type')}"
        reframed.append({**r, "reframed_label": lab})
    return reframed


def align_emb(rows):
    idx = np.array([_IDX[r["pair_id"]] for r in rows])
    eu, zv = _EU[idx], _ZV[idx]
    feats = np.concatenate([eu, zv, np.abs(eu - zv), eu * zv], axis=1)
    sims = np.sum(eu * zv, axis=1)
    return feats, sims


def experiments(rows, feats, sims):
    groups = np.array([r["event_id"] for r in rows])
    out = {}

    # EXP A: 4-class type, event-grouped 5-fold
    lab = np.array([(r["bridge_type"] if r["is_bridge"] == "yes" else "no") for r in rows])
    lab = np.array([x if x in ("no", "A", "B", "C") else "no" for x in lab])
    gkf = GroupKFold(n_splits=5)
    mlp_f1, sim_f1, maj_f1 = [], [], []
    perclass = {c: [] for c in ["no", "A", "B", "C"]}
    for tr, te in gkf.split(feats, lab, groups):
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=RNG)
        clf.fit(feats[tr], lab[tr])
        pred = clf.predict(feats[te])
        mlp_f1.append(f1_score(lab[te], pred, average="macro", labels=["no", "A", "B", "C"], zero_division=0))
        for c in ["no", "A", "B", "C"]:
            perclass[c].append(f1_score(lab[te] == c, pred == c, zero_division=0))
        maj = Counter(lab[tr]).most_common(1)[0][0]
        maj_f1.append(f1_score(lab[te], np.array([maj] * len(te)), average="macro", labels=["no", "A", "B", "C"], zero_division=0))
        lr = LogisticRegression(max_iter=500).fit(sims[tr].reshape(-1, 1), lab[tr])
        sim_f1.append(f1_score(lab[te], lr.predict(sims[te].reshape(-1, 1)), average="macro", labels=["no", "A", "B", "C"], zero_division=0))
    out["expA"] = {"dist": dict(Counter(lab)), "mlp_macroF1": np.mean(mlp_f1), "mlp_std": np.std(mlp_f1),
                   "sim_macroF1": np.mean(sim_f1), "maj_macroF1": np.mean(maj_f1),
                   "perclass": {c: np.mean(perclass[c]) for c in ["no", "A", "B", "C"]}}

    # EXP B: genuine-bridge detection, content vs content x time
    keep = np.array(["untimed" not in r["reframed_label"] for r in rows])
    y = np.array([1 if (r["reframed_label"].startswith("bridge_") and "untimed" not in r["reframed_label"]) else 0 for r in rows])
    leads = np.array([abs(r["lead_h"]) if r.get("lead_h") is not None else 0.0 for r in rows])
    bridge_lab = np.array([(r["bridge_type"] if r["is_bridge"] == "yes" else "no") for r in rows])
    gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=RNG)
    ap_c, ap_ct = [], []
    for tr, te in gss.split(feats, y, groups):
        tr = tr[keep[tr]]; te2 = te[keep[te]]
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=RNG)
        clf.fit(feats[tr], (bridge_lab[tr] != "no").astype(int))
        score = clf.predict_proba(feats[te2])[:, 1]
        gate = (leads[te2] >= 24.0).astype(float)
        ap_c.append(average_precision_score(y[te2], score))
        ap_ct.append(average_precision_score(y[te2], score * gate))
    out["expB"] = {"content_AP": np.mean(ap_c), "content_std": np.std(ap_c),
                   "content_time_AP": np.mean(ap_ct), "pos": int(y[keep].sum()), "n": int(keep.sum())}

    # EXP C: 0.60 candidate-gate recall on genuine bridges
    gen = np.array([r["reframed_label"].startswith("bridge_") for r in rows])
    gsims = sims[gen]
    out["expC"] = {"below_060": float((gsims < 0.60).mean()), "median_sim": float(np.median(gsims))}
    return out


def main():
    sets = [("nano", "llm_labels_v2.jsonl"), ("gpt55", "llm_labels_gpt55.jsonl")]
    results = {}
    for tag, path in sets:
        print(f"\n########## {tag}: {path} ##########", flush=True)
        rows = temporal(path)
        (HERE / f"llm_labels_{tag}_reframed.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        feats, sims = align_emb(rows)
        br = [r for r in rows if r["is_bridge"] == "yes"]
        timed = [r for r in br if r["direction"] != "na"]
        dirc = Counter(r["direction"] for r in timed)
        gen = [r for r in rows if r["reframed_label"].startswith("bridge_") and "untimed" not in r["reframed_label"]]
        print(f"pairs={len(rows)} bridges={len(br)} timed={len(timed)} "
              f"dir={dict(dirc)} genuine_typed={dict(Counter(r['bridge_type'] for r in gen))}", flush=True)
        results[tag] = {"exp": experiments(rows, feats, sims),
                        "n_bridges": len(br), "n_timed": len(timed),
                        "dir": {k: dirc.get(k, 0) / max(len(timed), 1) for k in ("EN->ZH", "ZH->EN", "concurrent")}}

    # side-by-side
    print("\n\n==================== NANO vs GPT-5.5 (silver) ====================")
    a = results["nano"]["exp"]; b = results["gpt55"]["exp"]
    print(f"\nEXP A  type {{no,A,B,C}}, event-grouped 5-fold (bi-encoder MLP):")
    print(f"  class dist      nano={a['expA']['dist']}")
    print(f"                  gpt55={b['expA']['dist']}")
    print(f"  MLP macro-F1    nano={a['expA']['mlp_macroF1']:.3f}±{a['expA']['mlp_std']:.3f}   gpt55={b['expA']['mlp_macroF1']:.3f}±{b['expA']['mlp_std']:.3f}")
    print(f"  sim-only / maj  nano={a['expA']['sim_macroF1']:.3f}/{a['expA']['maj_macroF1']:.3f}   gpt55={b['expA']['sim_macroF1']:.3f}/{b['expA']['maj_macroF1']:.3f}")
    for c in ["no", "A", "B", "C"]:
        print(f"    F1[{c}]        nano={a['expA']['perclass'][c]:.3f}        gpt55={b['expA']['perclass'][c]:.3f}")
    print(f"\nEXP B  genuine-bridge detection (Average Precision, grouped holdout):")
    print(f"  content_only    nano={a['expB']['content_AP']:.3f}±{a['expB']['content_std']:.3f}   gpt55={b['expB']['content_AP']:.3f}±{b['expB']['content_std']:.3f}")
    print(f"  content x time  nano={a['expB']['content_time_AP']:.3f}        gpt55={b['expB']['content_time_AP']:.3f}")
    print(f"  positives       nano={a['expB']['pos']}/{a['expB']['n']}   gpt55={b['expB']['pos']}/{b['expB']['n']}")
    print(f"\nEXP C  0.60 sim-floor recall loss on genuine bridges:")
    print(f"  below 0.60      nano={a['expC']['below_060']:.1%}   gpt55={b['expC']['below_060']:.1%}")
    print(f"\nDirectionality among timed bridges (EN->ZH / ZH->EN / parallel):")
    for tag in ("nano", "gpt55"):
        d = results[tag]["dir"]
        print(f"  {tag:>5}: {d['EN->ZH']:.1%} / {d['ZH->EN']:.1%} / {d['concurrent']:.1%}  (timed n={results[tag]['n_timed']})")
    print("\n==================== done ====================")


if __name__ == "__main__":
    main()
