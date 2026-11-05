#!/usr/bin/env python3
"""make_fig8_block_effect.py — per-study volume-matched block effect figure.

Reads results/exH_volumematch.csv and results/exH_value_volumematch.csv.
Writes figures/fig8_block_effect.pdf and .png.

Run: python3 code/make_fig8_block_effect.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")


def load_pair(path: str):
    d = pd.read_csv(path)
    lopo = d[d["arm"] == "lopo"][["paper", "med_drift", "n_rows_deleted"]].rename(
        columns={"med_drift": "block"})
    matched = d[d["arm"] == "matched_scattered"][["paper", "med_drift"]].rename(
        columns={"med_drift": "matched"})
    out = lopo.merge(matched, on="paper", how="inner")
    out["ratio"] = out["block"] / out["matched"]
    return out


def main() -> None:
    pres = load_pair(os.path.join(RESULTS, "exH_volumematch.csv"))
    val = load_pair(os.path.join(RESULTS, "exH_value_volumematch.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

    # (a) sorted per-study paired ratios
    ax = axes[0]
    for df, label, color in ((pres, f"Presence PCA (n={len(pres)})", "#1f4e79"),
                             (val, f"Std-value PCA (n={len(val)})", "#c55a11")):
        r = np.sort(df["ratio"].to_numpy(float))
        x = np.arange(1, len(r) + 1)
        ax.plot(x, r, marker="o", ms=2.6, lw=1.0, color=color, label=label, alpha=0.9)
        ax.axhline(np.median(r), color=color, ls=":", lw=1.0, alpha=0.8)
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Studies, sorted by ratio")
    ax.set_ylabel("Block / matched scattered drift")
    ax.set_title("(a) Per-study paired ratio", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    # (b) block vs matched drift
    ax = axes[1]
    for df, label, color in ((pres, "Presence PCA", "#1f4e79"),
                             (val, "Std-value PCA", "#c55a11")):
        ax.scatter(df["matched"], df["block"], s=12, alpha=0.65,
                   color=color, edgecolor="none", label=label)
        ax.scatter([np.median(df["matched"])], [np.median(df["block"])],
                   s=55, marker="D", color=color, edgecolor="black", lw=0.5, zorder=5)
    lo = min(pres["matched"].min(), pres["block"].min(),
             val["matched"].min(), val["block"].min()) * 0.7
    hi = max(pres["matched"].max(), pres["block"].max(),
             val["matched"].max(), val["block"].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.9, ls="--", label="1:1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Matched scattered median drift")
    ax.set_ylabel("Block median drift")
    ax.set_title("(b) Block versus matched drift", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig8_block_effect.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    print("wrote figures/fig8_block_effect.pdf and .png")
    print(f"presence median ratio {np.median(pres['ratio']):.3f}; "
          f"value median ratio {np.median(val['ratio']):.3f}")


if __name__ == "__main__":
    main()
