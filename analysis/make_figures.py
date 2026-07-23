"""Publication figures for PAPER_DRAFT.md, generated from frozen result artifacts.

fig4  genre concentration — per-event C-share among bridges (GPT-5.5 silver)
fig5  benchmark — macro-F1 + C-F1 per system with bootstrap CI whiskers + annotator ceiling
fig6  anatomy — (a) direction/concurrency mix by type, (b) median |lag| by type (log scale)
fig7  event bridge signatures — C-share vs concurrency scatter, dispute/non-dispute poles

Run:  ~/opt/anaconda3/bin/python3 make_figures.py   → ../figures/fig{4..7}_*.{pdf,png}
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
# colorblind-safe
C_A, C_B, C_C, C_NO = "#0173B2", "#DE8F05", "#D55E00", "#949494"
C_DISP, C_NON = "#D55E00", "#0173B2"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


# ---------------- fig4: genre concentration ----------------
rows = [json.loads(l) for l in (HERE / "llm_labels_gpt55.jsonl").read_text().splitlines() if l.strip()]
by = defaultdict(Counter)
for r in rows:
    lab = r.get("bridge_type", "") if r.get("is_bridge") == "yes" else "no"
    by[r["event_id"]][lab or "?"] += 1
ev = []
for e, c in by.items():
    br = c["A"] + c["B"] + c["C"]
    if br >= 20:
        ev.append((c["C"] / br, e))
ev.sort()
rates = [x[0] for x in ev]
names = [x[1] for x in ev]
n = len(ev)
colors = [C_DISP if r >= sorted(rates)[-15] else (C_NON if r <= sorted(rates)[14] else "#BBBBBB") for r in rates]
fig, ax = plt.subplots(figsize=(6.5, 8))
ax.barh(range(n), [100 * r for r in rates], color=colors, height=0.75)
ax.set_yticks(range(n))
ax.set_yticklabels([nm.replace("_", " ") for nm in names], fontsize=6)
ax.set_xlabel("counter-narrative share among bridges (%)")
ax.set_title("Counter-narrative is genre-concentrated (70 events, GPT-5.5 silver)", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_DISP, label="top-15 by C-share (contested China–West framing)"),
                   Patch(color=C_NON, label="bottom-15 (uncontested: product launches / US-internal / disasters)"),
                   Patch(color="#BBBBBB", label="middle")], loc="lower right", fontsize=7, frameon=False)
save(fig, "fig4_genre_concentration")

# ---------------- fig5: benchmark (gold v2.2 + expert-verified core) ----------------
v22 = json.loads((HERE / "v22_benchmark.json").read_text())
sysrows, core = v22["systems"], v22["expert_core"]
fig, (ax, axc) = plt.subplots(1, 2, figsize=(8.6, 3.1), width_ratios=[7, 4])
x = np.arange(len(sysrows))
ax.bar(x - 0.19, [r["macro"] for r in sysrows], width=0.36, color="#949494", label="macro-F1")
ax.bar(x + 0.19, [r["C"] for r in sysrows], width=0.36, color=C_C, label="C-F1 (counter-narrative)")
for i, r in enumerate(sysrows):
    if "C_CI" in r:
        lo, hi = r["C_CI"]
        ax.errorbar(i + 0.19, r["C"], yerr=[[r["C"] - lo], [hi - r["C"]]], fmt="none",
                    ecolor="black", elinewidth=1, capsize=2.5)
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in sysrows], fontsize=6.5)
ax.set_ylabel(f"F1 on gold v2.3 (n={v22['full_gold_n']:,})")
ax.set_ylim(0, 0.9)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set_title("(a) only stance-aware models detect counter-narrative", fontsize=9)
xc = np.arange(len(core))
for i, r in enumerate(core):
    kw = dict(hatch="//", edgecolor="white") if r["ref"] else {}
    axc.bar(i - 0.19, r["macro"], width=0.36, color="#BBBBBB" if r["ref"] else "#949494", **kw)
    axc.bar(i + 0.19, r["C"], width=0.36, color="#E8A87C" if r["ref"] else C_C, **kw)
axc.set_xticks(xc)
axc.set_xticklabels([r["label"] for r in core], fontsize=6.5)
axc.set_ylabel(f"F1 vs expert labels (n={v22['expert_core_n']})")
axc.set_ylim(0, 0.9)
axc.set_title("(b) expert-verified core:\ngap widens; annotators optimistic refs", fontsize=8.5)
save(fig, "fig5_benchmark")

# ---------------- fig6: anatomy ----------------
bs = json.loads((HERE / "bridge_science_result.json").read_text())
e1, e2 = bs["E1_direction_by_type"], bs["E2_latency_by_type"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
types = ["A", "B", "C"]
tl = ["A\ntranslation", "B\nreframing", "C\ncounter-narr."]
bot = np.zeros(3)
for key, col, lab in (("EN->ZH", "#0173B2", "EN→ZH"), ("ZH->EN", "#56B4E9", "ZH→EN"),
                      ("concurrent", "#CCCCCC", "concurrent (<24h)\n= parallel coverage")):
    vals = np.array([e1[t][key] for t in types]) * 100
    a1.bar(tl, vals, bottom=bot, color=col, label=lab, width=0.6)
    bot += vals
for i, t in enumerate(types):
    a1.text(i, 101, f"{e1[t]['concurrent']*100:.0f}% conc.", ha="center", fontsize=7)
a1.set_ylabel("share of timed bridges (%)")
a1.set_ylim(0, 112)
a1.legend(fontsize=6, frameon=False, loc="lower left")
a1.set_title("(a) concurrency gradient", fontsize=9)
med = [e2[t]["median_abs_h"] for t in types]
lo = [e2[t]["abs_ci"][0] for t in types]
hi = [e2[t]["abs_ci"][1] for t in types]
a2.bar(tl, med, color=[C_A, C_B, C_C], width=0.6)
a2.errorbar(range(3), med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt="none",
            ecolor="black", elinewidth=1, capsize=3)
a2.set_yscale("log")
a2.set_ylabel("median |EN–ZH lag| (hours, log)")
for i, m in enumerate(med):
    a2.text(i, m * 1.35, f"{m:.0f}h" if m < 100 else f"{m/24:.0f}d", ha="center", fontsize=7.5)
a2.set_title("(b) fidelity is fast, transformation is slow", fontsize=9)
save(fig, "fig6_anatomy")

# ---------------- fig7: signatures ----------------
sig = bs["E5_signatures"]
crate = json.loads((HERE / "genre_crate_gpt55.json").read_text())
top = set(sorted(crate, key=lambda e: crate[e]["c_rate"], reverse=True)[:15])
bot15 = set(sorted(crate, key=lambda e: crate[e]["c_rate"])[:15])
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for e, s in sig.items():
    if s["med_lag_h"] is None:
        continue
    col = C_DISP if e in top else (C_NON if e in bot15 else "#BBBBBB")
    z = 3 if e in top | bot15 else 2
    ax.scatter(100 * s["conc"], 100 * s["C"], s=np.sqrt(s["n"]) * 4, c=col, alpha=0.75, zorder=z,
               edgecolors="white", linewidths=0.4)
ax.set_xlabel("parallel coverage: concurrent share of timed bridges (%)")
ax.set_ylabel("counter-narrative share (%)")
ax.set_title("Event bridge signatures separate narrative contests", fontsize=9)
ax.scatter([], [], c=C_DISP, label="dispute (top-15 C)")
ax.scatter([], [], c=C_NON, label="non-dispute (bottom-15 C)")
ax.scatter([], [], c="#BBBBBB", label="middle")
ax.legend(fontsize=7, frameon=False)
save(fig, "fig7_signatures")
print("all figures written to", FIG)
