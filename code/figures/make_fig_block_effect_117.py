#!/usr/bin/env python3
"""Figure 1: full-set volume-matched block effect (117 curated studies)."""
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


def main():
    b = pd.read_csv(os.path.join(R, "baseline_block.csv")).set_index("paper")
    m = (pd.read_csv(os.path.join(R, "baseline_matched.csv"))
         .groupby("paper")["med_drift"].median())
    df = pd.DataFrame({"block": b["med_drift"], "matched": m}).dropna()
    df["ratio"] = df["block"] / df["matched"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    ax = axes[0]
    r = np.sort(df["ratio"].to_numpy(float))
    ax.plot(np.arange(1, len(r) + 1), r, color="#1f4e79", lw=1.0)
    ax.axhline(np.median(r), color="#1f4e79", ls=":", lw=1.0,
               label=f"median {np.median(r):.2f}")
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Studies, sorted by ratio")
    ax.set_ylabel("Block / matched drift")
    ax.set_title("(a) Per-study ratio", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.2, which="both")

    ax = axes[1]
    ax.scatter(df["matched"], df["block"], s=12, alpha=0.65,
               color="#1f4e79", edgecolor="none")
    lo = min(df["matched"].min(), df["block"].min()) * 0.7
    hi = max(df["matched"].max(), df["block"].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.9, ls="--", label="1:1")
    ax.scatter([df["matched"].median()], [df["block"].median()],
               s=55, marker="D", color="#1f4e79", edgecolor="black", lw=0.5,
               zorder=5, label="median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Matched scattered median drift")
    ax.set_ylabel("Block median drift")
    ax.set_title("(b) Block versus matched", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    for ax in axes:
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_block_effect_117.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"n={len(df)} median={np.median(df['ratio']):.3f}")


if __name__ == "__main__":
    main()
