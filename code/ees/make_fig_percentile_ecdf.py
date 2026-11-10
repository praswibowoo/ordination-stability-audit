#!/usr/bin/env python3
"""make_fig_percentile_ecdf.py — Figure 3: block-percentile ECDF vs uniform.

Uses the curated coverage (117 studies) and value (65 studies) event tables.
The percentile is the mid-rank percentile of the block drift within its 100
matched draws, matching clean_primary_audit.percentile_summary.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "wp3_clean")
FIGS = os.path.join(ROOT, "figures")


def percentiles(path):
    d = pd.read_csv(path)
    block = d[d["arm"] == "block"].set_index("paper")["med_drift"]
    matched = d[d["arm"] == "matched"].groupby("paper")["med_drift"].apply(list)
    pct = {}
    above = below = 0
    for p, b in block.items():
        m = np.asarray(matched.loc[p], float)
        pct[p] = 100.0 * (np.sum(m < b) + 0.5 * np.sum(m == b)) / len(m)
        if b > np.percentile(m, 95):
            above += 1
        if b < np.percentile(m, 5):
            below += 1
    return pd.Series(pct).sort_values(), above, below


def ecdf(ax, pct, color, title, mark, exceed, below):
    x = np.sort(pct.to_numpy(float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.r_[0, x], np.r_[0, y], where="post", color=color, lw=1.6,
            label=f"Observed (n={len(x)})")
    ax.plot([0, 100], [0, 1], color="0.55", lw=0.9, ls="--", label="Uniform null")
    ax.axvline(5, color="0.35", lw=0.8, ls=":")
    ax.axvline(95, color="0.35", lw=0.8, ls=":")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Block percentile within matched draws", fontsize=10.5)
    ax.set_ylabel("Cumulative fraction of studies", fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9.0, loc="upper left")
    # Tail annotations.
    ax.text(96.5, 0.06, f"{exceed:.1f}% >95th", ha="right", va="bottom",
            fontsize=9.0, color=color)
    ax.text(3.5, 0.30, f"{below:.1f}% <5th", ha="left", va="bottom",
            fontsize=9.0, color=color)
    # Mark named events.
    for i, (label, paper) in enumerate(mark):
        if paper in pct.index:
            yy = 0.90 - 0.07 * i
            ax.annotate(label, xy=(100, 1.0), xytext=(63, yy),
                        fontsize=8.5, color=color,
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color=color, alpha=0.7))


def main():
    cov, above_cov, below_cov = percentiles(os.path.join(RESULTS, "coverage_events.csv"))
    val, above_val, below_val = percentiles(os.path.join(RESULTS, "value_events.csv"))
    n_cov, n_val = len(cov), len(val)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    ecdf(axes[0], cov, "#1f4e79", f"(a) Coverage PCA (n={n_cov})",
         [("B128 (water-only)", "B128"), ("B063 (out-of-scope)", "B063")],
         100 * above_cov / n_cov, 100 * below_cov / n_cov)
    ecdf(axes[1], val, "#c55a11", f"(b) Value PCA (n={n_val})",
         [("B156 (derived row)", "B156")],
         100 * above_val / n_val, 100 * below_val / n_val)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_percentile_ecdf.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("wrote figures/fig_percentile_ecdf.pdf/.png")
    print(f"coverage: n={n_cov}, >95 {above_cov} ({100*above_cov/n_cov:.1f}%), "
          f"<5 {below_cov} ({100*below_cov/n_cov:.1f}%)")
    print(f"value: n={n_val}, >95 {above_val} ({100*above_val/n_val:.1f}%), "
          f"<5 {below_val} ({100*below_val/n_val:.1f}%)")


if __name__ == "__main__":
    main()
