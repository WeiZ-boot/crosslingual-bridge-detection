"""Cascade budget sweep for the README: confidence-margin vs random routing.

Reads the frozen sweep artifact results/pareto_fine_v23.json and asserts every
published number before drawing, so the figure cannot drift from the artifact:
routing AUC 0.420 vs 0.409 (+0.011), the five checkpoint gaps, the shared
endpoints, and the 95%-of-pure-LLM knee at r=0.38 (interpolated on this grid;
the stored b95=0.40 is the first grid point at/above 95%).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
p1 = json.load(open(ROOT / "results" / "pareto_fine_v23.json"))

rs = np.array(p1["rs"])
conf = np.array(p1["conf_macro"])
rand = np.array(p1["rand_macro"])

auc_conf = np.trapz(conf, rs)
auc_rand = np.trapz(rand, rs)
auc_gap = auc_conf - auc_rand
assert f"{auc_conf:.3f}" == "0.420" and f"{auc_rand:.3f}" == "0.409"
assert f"{auc_gap:.3f}" == "0.011"

checkpoints = [0.10, 0.20, 0.30, 0.40, 0.50]
idxs = [int(np.argmin(np.abs(rs - c))) for c in checkpoints]
gaps = conf[idxs] - rand[idxs]
assert [f"+{g:.3f}" for g in gaps] == \
    ["+0.010", "+0.014", "+0.021", "+0.027", "+0.015"]
assert conf[0] == rand[0] and abs(conf[-1] - rand[-1]) < 1e-9

# the paper's knee: 95% of the pure-LLM macro-F1, interpolated on the sweep
target = 0.95 * p1["pure_nano_macro"]
r95 = float(np.interp(target, conf, rs))
assert f"{r95:.2f}" == "0.38"
assert f"{p1['b95']:.2f}" == "0.40"

C_CONF = "#c0392b"
C_RAND = "#8a8a8a"

fig, ax = plt.subplots(figsize=(3.6, 3.2))

ax.fill_between(rs, rand, conf, color=C_CONF, alpha=0.12, lw=0, zorder=1)
ax.plot(rs, rand, color=C_RAND, lw=1.2, ls=(0, (4, 2)), zorder=2)
ax.plot(rs, conf, color=C_CONF, lw=1.5, zorder=3)

ax.plot(rs[idxs], conf[idxs], "o", ms=3.2, color=C_CONF, mec="white",
        mew=0.5, zorder=4)
ax.plot(rs[idxs], rand[idxs], "o", ms=2.6, color=C_RAND, mec="white",
        mew=0.5, zorder=4)
gap_offsets = [(0, 5), (0, 5), (0, 5), (-9, 6), (11, 6)]
for (i, g), (dx, dy) in zip(zip(idxs, gaps), gap_offsets):
    ax.annotate(f"+{g:.3f}", (rs[i], conf[i]), xytext=(dx, dy),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.0, color="#333333")

for x_e, y_e, lab, dx, dy, ha, va in (
        (0.0, conf[0], "pure student", 4, -8, "left", "top"),
        (1.0, conf[-1], "pure LLM", -1, 7, "right", "bottom")):
    ax.plot(x_e, y_e, "s", ms=3.4, color="#4a4a4a", mec="white", mew=0.5,
            zorder=4)
    ax.annotate(lab, (x_e, y_e), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=6.0, color="#4a4a4a")

# the knee: 95% of the pure-LLM macro-F1 at r=0.38
ax.plot([0.0, r95], [target, target], ls=":", lw=0.8, color="#4a4a4a",
        zorder=2)
ax.plot([r95, r95], [0.340, target], ls=":", lw=0.8, color="#4a4a4a",
        zorder=2)
ax.text(0.02, target + 0.0025, "0.95 × pure LLM", fontsize=6.0,
        ha="left", va="bottom", color="#4a4a4a")
ax.text(r95, 0.3415, "r=0.38", fontsize=6.0, ha="center", va="bottom",
        color="#4a4a4a")

ax.text(0.06, 0.93, "confidence-margin", transform=ax.transAxes,
        ha="left", va="top", fontsize=6.5, fontweight="bold", color=C_CONF)
ax.annotate("random", (0.62, np.interp(0.62, rs, rand)), xytext=(6, -8),
            textcoords="offset points", ha="left", va="top", fontsize=6.5,
            color=C_RAND)

ax.text(0.98, 0.06,
        f"routing AUC: {auc_conf:.3f} vs. {auc_rand:.3f} (+{auc_gap:.3f})",
        transform=ax.transAxes, fontsize=6.3, ha="right", va="bottom",
        color="#333333")

ax.set_xlim(-0.02, 1.02)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim(0.340, 0.475)
ax.set_yticks([0.36, 0.39, 0.42, 0.45])
ax.set_xlabel("LLM budget $r$", fontsize=8)
ax.set_ylabel("macro-F1", fontsize=8)
ax.tick_params(labelsize=7.5)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig8_cascade_sweep.png", dpi=300)
fig.savefig(ROOT / "figures" / "fig8_cascade_sweep.pdf")
print("saved fig8_cascade_sweep with artifact-asserted numbers")
