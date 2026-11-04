#!/usr/bin/env python3
"""make_fig5_print.py — Figure 5 (family stability gain & absolute drift ratio) at print size.

Two-panel figure:
  (a) Ratio Gain G (same-rows paired LOPO, 1,000 bootstrap resamples, Holm correction across families)
  (b) Absolute Drift Reduction Ratio (delta_all / delta_fam, log scale, from Table 3 / exK_absratio_summary.csv)

Outputs figures/fig5.png and fig5.pdf (600 dpi).
Run: python3 code/make_fig5_print.py
"""
from __future__ import annotations

import csv
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117/D132)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
DPI = 600

DISPLAY = {
    "F1_physical": "F1 physical", "F1_physicala": "F1a physical\u2013anion",
    "F1_clarity": "F1 clarity", "F2_metals": "F2 metals",
    "F3_nutrients": "F3 nutrients", "F4_organic": "F4 organic",
    "F6_aeration": "F6 aeration",
}


def main() -> None:
    with open(os.path.join(RESULTS, "exC2_summary.csv"), encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    for r in recs:
        r["G_median"] = float(r["G_median"])
        lo, hi = r["G_CI95"].strip("()").split(",")
        r["lo"], r["hi"] = float(lo), float(hi)
        r["holm"] = float(r.get("P_G_gt0_holm") or r["P_G_gt0"])
    recs = sorted(recs, key=lambda r: -r["G_median"])

    labels_a = [f"{DISPLAY.get(r['family'], r['family'])} [{r['variant'].replace('_pres', '').upper()}]"
                for r in recs]
    g = np.asarray([r["G_median"] for r in recs])
    lo_a = np.asarray([r["lo"] for r in recs])
    hi_a = np.asarray([r["hi"] for r in recs])
    holm = [r["holm"] for r in recs]

    with open(os.path.join(RESULTS, "exG_absratio.csv"), encoding="utf-8") as fh:
        abs_recs = list(csv.DictReader(fh))
    for r in abs_recs:
        r["ratio"] = float(r["ratio_of_medians"])
        r["med_ratio"] = float(r["median_of_ratios"])
        r["ci_lo"] = float(r["ratio_ci_lo"])
        r["ci_hi"] = float(r["ratio_ci_hi"])
        r["fam_med"] = float(r["fam_med"])
        r["at_floor"] = str(r.get("at_floor", "")).lower() == "true"
    abs_recs = sorted(abs_recs, key=lambda r: -r["ratio"])

    labels_b = [DISPLAY.get(r["family"], r["family"]) for r in abs_recs]
    fam_med_b = np.asarray([r["fam_med"] for r in abs_recs])
    floor_b = np.asarray([r["at_floor"] for r in abs_recs])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.4), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.15, 0.85]})

    # --- Panel (a): Ratio Gain G ---
    y_a = np.arange(len(recs))[::-1]
    cols_a = ["#4c72b0" if v > 0 else "#c62828" for v in g]
    ax1.barh(y_a, g, color=cols_a, height=0.68)
    ax1.errorbar(g, y_a, xerr=[np.clip(g - lo_a, 0, None), np.clip(hi_a - g, 0, None)],
                 fmt="none", ecolor="#37474f", capsize=2.2, lw=0.85)
    for i, yy in enumerate(y_a):
        s = f"G={g[i]:+.2f}, P={holm[i]:.3f}"
        if g[i] >= 0:
            _xx = min(max(hi_a[i], g[i]) + 0.18, 2.80)
            ax1.annotate(s, (_xx, yy), fontsize=7.4,
                         ha="right" if _xx >= 2.80 else "left", va="center")
        else:
            ax1.annotate(s, (0.05, yy), fontsize=7.4,
                         ha="left", va="center", color="#b71c1c")
    ax1.axvline(0, c="k", lw=0.6)
    ax1.set_yticks(y_a)
    ax1.set_yticklabels(labels_a, fontsize=8.8)
    ax1.set_xlabel("G (paired LOPO)", fontsize=8.8)
    ax1.set_xlim(-1.4, 3.6)
    ax1.tick_params(axis="x", labelsize=7.6)
    ax1.set_title("(a) Ratio gain", fontsize=9.0, pad=5)

    # --- Panel (b): Family drift against the noise floor ---
    y_b = np.arange(len(abs_recs))[::-1]
    cols_b = ["#c62828" if f else "#1b4f72" for f in floor_b]
    ax2.scatter(fam_med_b, y_b, c=cols_b, s=20, zorder=3)
    ax2.axvline(1e-6, ls="--", c="crimson", lw=0.8)
    ax2.set_ylim(-0.7, 6.9)
    ax2.annotate(r"$\tau=10^{-6}$", (1.15e-6, 6.62), fontsize=7.2, color="crimson",
                 fontstyle="italic", va="center", ha="left")
    ax2.set_xscale("log")
    ax2.set_xlim(1e-10, 1e-2)
    ax2.set_yticks(y_b)
    ax2.set_yticklabels(labels_b, fontsize=8.8)
    ax2.set_xlabel(r"Family drift $\delta_{\mathrm{fam}}$ (log scale)", fontsize=8.8)
    ax2.tick_params(axis="x", labelsize=7.6)
    ax2.set_title(r"(b) Drift vs noise floor $\tau$", fontsize=9.0, pad=5)

    fig.savefig(os.path.join(FIGS, "fig5.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig5.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[fig5-print] rebuilt figures/fig5.png/pdf (two-panel, print size, 600 dpi)")


if __name__ == "__main__":
    main()
