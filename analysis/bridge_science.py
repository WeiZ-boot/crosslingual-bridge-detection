"""Bridge science: substantive analyses of the bridges themselves (GPT-5.5 labels, 13,992 pairs).

E1  Counter-narrative directional asymmetry — direction shares by type (is C overwhelmingly EN->ZH?)
E2  Bridge latency by type — |lead| and signed lead distributions (is rebuttal slower than translation?)
E3  Platform pathways — bridge volume and C-share by (en_platform, zh_platform)
E4  Within-event lifecycle — does C-share among bridges rise over an event's lifetime?
E5  Event bridge signatures — per-event [A,B,C mix, %EN->ZH, median |lag|] and dispute/non-dispute split

Reuses the pair_id -> timestamp join from downstream_compare.temporal().
Run:  ~/opt/anaconda3/bin/python3 bridge_science.py   (numpy/scipy; no torch)
Saves bridge_science_result.json.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import downstream_compare as dc

HERE = Path(__file__).resolve().parent
TYPES = ["A", "B", "C"]
RES = {}


def boot_ci(vals, stat=np.median, n=5000, seed=0):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    ss = [stat(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return (float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5)))


def main():
    rows = dc.temporal(HERE / "llm_labels_gpt55.jsonl")  # adds lead_h, direction
    plat = {r["pair_id"]: (r["en_platform"], r["zh_platform"])
            for r in (json.loads(l) for l in (HERE / "annotation_set_v2.jsonl").read_text().splitlines() if l.strip())}
    bridges = [r for r in rows if r.get("is_bridge") == "yes" and r.get("bridge_type") in TYPES]
    timed = [r for r in bridges if r["direction"] != "na"]
    print(f"pairs={len(rows)}  bridges={len(bridges)}  timed={len(timed)}")

    # ---------------- E1 direction x type ----------------
    print("\n== E1: direction shares by type (timed bridges) ==")
    e1 = {}
    for t in TYPES:
        sub = [r for r in timed if r["bridge_type"] == t]
        c = Counter(r["direction"] for r in sub)
        n = len(sub)
        e1[t] = {d: round(c[d] / n, 3) for d in ("EN->ZH", "ZH->EN", "concurrent")} | {"n": n}
        print(f"  {t}: n={n:5d}  EN->ZH {c['EN->ZH']/n:.1%}  ZH->EN {c['ZH->EN']/n:.1%}  concurrent {c['concurrent']/n:.1%}")
    # asymmetry among DIRECTIONAL bridges only
    for t in TYPES:
        sub = [r for r in timed if r["bridge_type"] == t and r["direction"] != "concurrent"]
        k = sum(r["direction"] == "EN->ZH" for r in sub)
        e1[t]["enzh_share_directional"] = round(k / len(sub), 3) if sub else None
        print(f"  {t}: among directional, EN->ZH = {k}/{len(sub)} = {k/len(sub):.1%}")
    RES["E1_direction_by_type"] = e1

    # ---------------- E2 latency by type ----------------
    print("\n== E2: latency by type (hours; signed lead = ZH - EN) ==")
    e2 = {}
    for t in TYPES:
        leads = np.array([r["lead_h"] for r in timed if r["bridge_type"] == t and r["lead_h"] is not None])
        med_abs = float(np.median(np.abs(leads)))
        ci = boot_ci(np.abs(leads))
        e2[t] = {"n": len(leads), "median_abs_h": round(med_abs, 1), "abs_ci": [round(ci[0], 1), round(ci[1], 1)],
                 "median_signed_h": round(float(np.median(leads)), 1),
                 "frac_within_24h": round(float(np.mean(np.abs(leads) <= 24)), 3)}
        print(f"  {t}: n={len(leads):5d}  median|lag|={med_abs:7.1f}h  CI[{ci[0]:.0f},{ci[1]:.0f}]  "
              f"median signed={np.median(leads):7.1f}h  within24h={np.mean(np.abs(leads)<=24):.1%}")
    for a, b in (("A", "C"), ("B", "C"), ("A", "B")):
        la = np.abs([r["lead_h"] for r in timed if r["bridge_type"] == a and r["lead_h"] is not None])
        lb = np.abs([r["lead_h"] for r in timed if r["bridge_type"] == b and r["lead_h"] is not None])
        u, p = mannwhitneyu(la, lb, alternative="two-sided")
        e2[f"MWU_{a}_vs_{b}_p"] = float(p)
        print(f"  Mann-Whitney |lag| {a} vs {b}: p={p:.2e}")
    RES["E2_latency_by_type"] = e2

    # EN->ZH response lag by type (positive lead only: ZH responds to EN)
    print("  -- EN->ZH response lag (ZH after EN) --")
    e2r = {}
    for t in TYPES:
        leads = np.array([r["lead_h"] for r in timed if r["bridge_type"] == t and r["direction"] == "EN->ZH"])
        e2r[t] = {"n": len(leads), "median_h": round(float(np.median(leads)), 1)}
        print(f"  {t}: n={len(leads):4d}  median response lag = {np.median(leads):.1f}h")
    RES["E2_enzh_response_lag"] = e2r

    # ---------------- E3 platform pathways ----------------
    print("\n== E3: platform pathways (bridges; C-share among bridges) ==")
    path = defaultdict(Counter)
    for r in bridges:
        ep, zp = plat.get(r["pair_id"], ("?", "?"))
        path[(ep, zp)][r["bridge_type"]] += 1
    e3 = []
    for (ep, zp), c in sorted(path.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        if n < 30:
            continue
        e3.append({"en": ep, "zh": zp, "n": n, "C_share": round(c["C"] / n, 3),
                   "A_share": round(c["A"] / n, 3), "B_share": round(c["B"] / n, 3)})
        print(f"  {ep:10s} -> {zp:8s}  n={n:5d}  A {c['A']/n:.1%}  B {c['B']/n:.1%}  C {c['C']/n:.1%}")
    # marginals
    for side, idx in (("en", 0), ("zh", 1)):
        marg = defaultdict(Counter)
        for (ep, zp), c in path.items():
            marg[(ep, zp)[idx]].update(c)
        print(f"  -- {side} marginals --")
        for p, c in sorted(marg.items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(c.values())
            print(f"    {p:10s} n={n:5d}  C {c['C']/n:.1%}")
        RES[f"E3_marginal_{side}"] = {p: {"n": sum(c.values()), "C_share": round(c["C"] / sum(c.values()), 3)}
                                      for p, c in marg.items()}
    RES["E3_pathways"] = e3

    # ---------------- E4 within-event lifecycle ----------------
    print("\n== E4: within-event lifecycle (C-share among bridges by event-time tercile) ==")
    by_ev = defaultdict(list)
    for r in timed:
        # bridge 'completion' time = the later post; robust to direction
        by_ev[r["event_id"]].append(r)
    terc = [Counter(), Counter(), Counter()]  # per tercile: counts by type
    used_events = 0
    for eid, rs in by_ev.items():
        # completion timestamp proxy: within-event rank of (max of the pair) ~ use lead-adjusted; we only
        # have lead_h, so reconstruct: rank events by ZH time for EN->ZH, EN time for ZH->EN. Simplest
        # robust proxy: order pairs by their pair index-independent completion = we lack raw ts here, so
        # recompute from dc loaders once per event.
        en_ts = [dc.parse_ts(x.get("ts")) for x in dc.load_en(eid)]
        zh_ts = [dc.parse_ts(x.get("ts")) for x in dc.load_zh(eid)]
        comp = []
        for r in rs:
            try:
                _, i, j = r["pair_id"].rsplit("_", 2)
                e, z = en_ts[int(i)], zh_ts[int(j)]
            except (ValueError, IndexError):
                continue
            if dc.plausible(e) and dc.plausible(z):
                comp.append((max(e, z), r["bridge_type"]))
        if len(comp) < 15:
            continue
        used_events += 1
        comp.sort()
        n = len(comp)
        for k, (_, t) in enumerate(comp):
            terc[min(2, 3 * k // n)][t] += 1
    e4 = {}
    for k, name in enumerate(("early", "mid", "late")):
        n = sum(terc[k].values())
        e4[name] = {"n": n, "C_share": round(terc[k]["C"] / n, 3), "A_share": round(terc[k]["A"] / n, 3)}
        print(f"  {name:5s}: n={n:5d}  A {terc[k]['A']/n:.1%}  B {terc[k]['B']/n:.1%}  C {terc[k]['C']/n:.1%}")
    e4["events_used"] = used_events
    RES["E4_lifecycle"] = e4

    # ---------------- E5 event bridge signatures ----------------
    print("\n== E5: event bridge signatures (dispute vs non-dispute) ==")
    crate = {e: v["c_rate"] for e, v in json.loads((HERE / "genre_crate_gpt55.json").read_text()).items()}
    top = set(sorted(crate, key=crate.get, reverse=True)[:15])   # dispute-genre proxy
    bot = set(sorted(crate, key=crate.get)[:15])
    sig = {}
    for eid, rs in by_ev.items():
        n = len(rs)
        if n < 15:
            continue
        d = Counter(r["direction"] for r in rs)
        tps = Counter(r["bridge_type"] for r in rs)
        leads = np.abs([r["lead_h"] for r in rs if r["lead_h"] is not None])
        ndir = d["EN->ZH"] + d["ZH->EN"]
        sig[eid] = {"n": n, "C": tps["C"] / n, "A": tps["A"] / n,
                    "enzh_dir": (d["EN->ZH"] / ndir if ndir else None),
                    "conc": d["concurrent"] / n,
                    "med_lag_h": float(np.median(leads)) if len(leads) else None}
    for name, grp in (("dispute(top15 C)", top), ("non-dispute(bot15 C)", bot)):
        g = [sig[e] for e in grp if e in sig]
        if not g:
            continue
        agg = {k: round(float(np.median([x[k] for x in g if x[k] is not None])), 3)
               for k in ("C", "A", "enzh_dir", "conc", "med_lag_h")}
        RES[f"E5_{name}"] = agg
        print(f"  {name:22s} (n_events={len(g)}):  C {agg['C']:.1%}  A {agg['A']:.1%}  "
              f"EN->ZH|dir {agg['enzh_dir']:.1%}  concurrent {agg['conc']:.1%}  med|lag| {agg['med_lag_h']:.0f}h")
    RES["E5_signatures"] = sig

    (HERE / "bridge_science_result.json").write_text(json.dumps(RES, indent=1, default=float))
    print("\nsaved bridge_science_result.json")


if __name__ == "__main__":
    main()
