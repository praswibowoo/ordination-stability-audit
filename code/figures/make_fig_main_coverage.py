#!/usr/bin/env python3
"""make_fig_main_coverage.py — main-text coverage figures + SI value twins.

The narrowed manuscript's main figures are coverage-only:
  fig8_block_effect.pdf        (two panels: sorted ratios; block vs matched)
  fig_percentile_ecdf.pdf      (one ECDF panel)

Value-layer twins are written for the SI:
  fig8_block_effect_value.pdf
  fig_percentile_ecdf_value.pdf
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


def load_events(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    rows = []
    for paper, g in d.groupby("paper"):
        b = g[g["arm"] == "block"]["med_drift"]
        m = g[g["arm"] == "matched"]["med_drift"]
        if len(b) and len(m):
            rows.append(dict(paper=paper, block=float(b.median()),
                             matched=float(m.median()),
                             n_rows_deleted=int(g["n_rows_deleted"].iloc[0])))
    out = pd.DataFrame(rows)
    out["ratio"] = out["block"] / out["matched"]
    return out


def block_figure(df: pd.DataFrame, path: str, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    ax = axes[0]
    r = np.sort(df["ratio"].to_numpy(float))
    x = np.arange(1, len(r) + 1)
    ax.plot(x, r, marker="o", ms=2.6, lw=1.0, color="#1f4e79",
            label=f"{label} (n={len(r)})", alpha=0.9)
    ax.axhline(np.median(r), color="#1f4e79", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Studies, sorted by ratio")
    ax.set_ylabel("Block / matched scattered drift")
    ax.set_title("(a) Per-study paired ratio", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    ax = axes[1]
    ax.scatter(df["matched"], df["block"], s=12, alpha=0.65,
               color="#1f4e79", edgecolor="none", label=label)
    ax.scatter([np.median(df["matched"])], [np.median(df["block"])],
               s=55, marker="D", color="#1f4e79", edgecolor="black", lw=0.5,
               zorder=5, label="median")
    lo = min(df["matched"].min(), df["block"].min()) * 0.7
    hi = max(df["matched"].max(), df["block"].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.9, ls="--", label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Matched scattered median drift")
    ax.set_ylabel("Block median drift")
    ax.set_title("(b) Block versus matched drift", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    for ext in ("pdf", "png"):
        fig.savefig(path.replace(".pdf", f".{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def percentiles(path: str):
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


def ecdf_figure(pct: pd.Series, above: int, below: int, path: str,
                title: str, marks, color: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    x = np.sort(pct.to_numpy(float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.r_[0, x], np.r_[0, y], where="post", color=color, lw=1.6,
            label=f"Observed (n={len(x)})")
    ax.plot([0, 100], [0, 1], color="0.55", lw=0.9, ls="--", label="Uniform null")
    ax.axvline(5, color="0.35", lw=0.8, ls=":")
    ax.axvline(95, color="0.35", lw=0.8, ls=":")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.03)
    ax.set_xlabel("Block percentile within matched draws", fontsize=10.5)
    ax.set_ylabel("Cumulative fraction of studies", fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9.0, loc="upper left")
    ax.text(96.5, 0.06, f"{100*above/len(x):.1f}% >95th", ha="right", va="bottom",
            fontsize=9.0, color=color)
    ax.text(3.5, 0.30, f"{100*below/len(x):.1f}% <5th", ha="left", va="bottom",
            fontsize=9.0, color=color)
    for i, (label, paper) in enumerate(marks):
        if paper in pct.index:
            yy = 0.90 - 0.07 * i
            ax.annotate(label, xy=(100, 1.0), xytext=(63, yy),
                        fontsize=8.5, color=color,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=color, alpha=0.7))
    for ext in ("pdf", "png"):
        fig.savefig(path.replace(".pdf", f".{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pres = load_events(os.path.join(RESULTS, "wp3_clean", "coverage_events.csv"))
    val = load_events(os.path.join(RESULTS, "wp3_clean", "value_events.csv"))
    block_figure(pres, os.path.join(FIGS, "fig8_block_effect.pdf"), "Coverage PCA")
    block_figure(val, os.path.join(FIGS, "fig8_block_effect_value.pdf"), "Std-value PCA")

    cov, a_cov, b_cov = percentiles(os.path.join(RESULTS, "wp3_clean", "coverage_events.csv"))
    value, a_val, b_val = percentiles(os.path.join(RESULTS, "wp3_clean", "value_events.csv"))
    ecdf_figure(cov, a_cov, b_cov, os.path.join(FIGS, "fig_percentile_ecdf.pdf"),
                f"Coverage PCA (n={len(cov)})",
                [("B128 (water-only)", "B128"), ("B063", "B063")], "#1f4e79")
    ecdf_figure(value, a_val, b_val, os.path.join(FIGS, "fig_percentile_ecdf_value.pdf"),
                f"Value PCA (n={len(value)})", [("B156", "B156")], "#c55a11")
    print("wrote coverage main figures and value SI twins")


if __name__ == "__main__":
    main()
