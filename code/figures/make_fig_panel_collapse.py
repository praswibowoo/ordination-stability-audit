#!/usr/bin/env python3
"""Figure: full-set versus panel-collapsed per-study ratios."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "post_audit")
FIGS = os.path.join(ROOT, "figures")


def ratios(tag: str):
    b = pd.read_csv(os.path.join(R, f"{tag}_block.csv")).set_index("paper")
    m = (pd.read_csv(os.path.join(R, f"{tag}_matched.csv"))
         .groupby("paper")["med_drift"].median())
    return (b["med_drift"] / m).dropna()


def main():
    full = ratios("baseline")
    coll = ratios("collapsed")
    common = sorted(set(full.index) & set(coll.index))
    full = full.loc[common]
    coll = coll.loc[common]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)

    ax = axes[0]
    ax.scatter(full, coll, s=14, alpha=0.7, color="#1f4e79", edgecolor="none")
    lim = [0.18, 10.5]
    ax.plot(lim, lim, color="black", lw=0.9, ls="--", label="1:1")
    ax.axhline(1.0, color="#b0b0b0", lw=0.8, ls=":")
    ax.axvline(1.0, color="#b0b0b0", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Full-matrix ratio")
    ax.set_ylabel("Panel-collapsed ratio")
    ax.set_title("(a) Per-study ratio before and after collapse", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    ax = axes[1]
    data = [full.to_numpy(float), coll.to_numpy(float)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(color="#555555", lw=0.8),
                    capprops=dict(color="#555555", lw=0.8),
                    boxprops=dict(lw=0.8))
    for patch, color in zip(bp["boxes"], ("#1f4e79", "#c55a11")):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Full\n(716 rows)", "Collapsed\n(151 rows)"])
    ax.set_ylabel("Block / matched ratio")
    ax.set_title("(b) Distribution before and after collapse", loc="left")
    ax.grid(alpha=0.2, axis="y", which="both")

    for ax in axes:
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_panel_collapse.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"full median {np.median(full):.3f}; collapsed median {np.median(coll):.3f}")


if __name__ == "__main__":
    main()
