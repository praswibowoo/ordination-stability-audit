#!/usr/bin/env python3
"""make_fig2_print.py — re-render Figure 2 at print size with legible type.

The previous composite was drawn for screen proportions and shrank to ~2.6 in
on the page (~3 pt effective type). This script redraws all three panels at
their final printed size (panel a full width, panels b and c side by side
below) with absolute font sizes, so no downscaling occurs at \\linewidth.

Data: frozen C1 event layer (results/exC1_rows.csv.gz) + corrections outputs
(results/exC1_blockperm.csv). Writes figures/fig2_a/b/c and fig2_composite
(PNG + PDF, 600 dpi).

Run: python3 code/figures/make_fig2_print.py
"""
from __future__ import annotations

import csv
import gzip
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117/D132)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
DPI = 600

COLORS = {"pca_pres": "#1f77b4", "pca_vraw": "#ff7f0e", "pca_vstd": "#2ca02c",
          "mds_pres": "#9467bd", "tsne_pres": "#8c564b"}
ORDER = ["pca_pres", "pca_vraw", "pca_vstd", "tsne_pres"]


def _open_rows():
    """Read the per-row drift layer, gzip or plain (both forms exist)."""
    plain = os.path.join(RESULTS, "exC1_rows.csv")
    if os.path.exists(plain):
        return open(plain, encoding="utf-8")
    return gzip.open(plain + ".gz", "rt", encoding="utf-8")


def load_rows():
    data = {v: ([], []) for v in ORDER}
    with _open_rows() as fh:
        for r in csv.DictReader(fh):
            if r["mode"] == "lopo" and r["variant"] in data:
                data[r["variant"]][0].append(float(r["rho"]))
                data[r["variant"]][1].append(float(r["drift"]))
    return {v: (np.asarray(a), np.asarray(b)) for v, (a, b) in data.items()}


def trend(xs, ys, nbins=70):
    edges = np.linspace(xs.min(), xs.max(), nbins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbins - 1)
    cx, cy = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 3:
            cx.append(float(np.median(xs[m])))
            cy.append(float(np.median(ys[m])))
    return np.asarray(cx), np.asarray(cy)


NAMES = {"pca_pres": "Presence PCA", "pca_vraw": "Raw-value PCA", "pca_vstd": "Std-value PCA",
         "mds_pres": "Presence MDS", "tsne_pres": "Presence t-SNE"}


def panel_a():
    data = load_rows()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    pr = data["pca_pres"][0]
    q1, q3 = np.quantile(pr, [0.25, 0.75])
    for v in ORDER:
        xs, ys = data[v]
        nz = ys > 0
        if v == "pca_vraw":
            continue
        cx, cy = trend(xs[nz], ys[nz])
        ax.plot(cx, cy, c=COLORS[v], lw=1.6, alpha=0.95,
                label=f"{NAMES[v]} ({len(xs):,})")
    ax.axvline(q1, ls=":", c="k", lw=1)
    ax.axvline(q3, ls=":", c="k", lw=1)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_ylim(1e-5, 3e-2)
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2])
    ax.set_xlabel("row density $\\rho$", fontsize=8)
    ax.set_ylabel("Procrustes drift $\\delta$ (LOPO)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.annotate(f"Q1={q1:.2f}", (q1, 0.05), xycoords=("data", "axes fraction"),
                fontsize=7, xytext=(3, 2), textcoords="offset points", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
    ax.annotate(f"Q3={q3:.2f}", (q3, 0.05), xycoords=("data", "axes fraction"),
                fontsize=7, xytext=(3, 2), textcoords="offset points", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
    # legend below the axes: five series cannot be keyed inside the cloud
    ax.legend(fontsize=6.8, markerscale=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              handletextpad=0.4, columnspacing=1.0)
    ax.set_title("(a) drift vs row density — plotted variants (LOPO; binned medians; dotted = $\\rho$ IQR)",
                 fontsize=8.2, pad=5)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "fig2_a.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig2_a.pdf"), bbox_inches="tight")
    plt.close(fig)


def panel_b():
    recs = {r["variant"]: r for r in
            csv.DictReader(open(os.path.join(RESULTS, "sdr_weighting.csv"), encoding="utf-8"))}
    si = [np.nan if v == "pca_vraw" else float(recs[v]["event_paper_sdr"]) for v in ORDER]
    ci_lo = [np.nan if v == "pca_vraw" else float(recs[v]["event_paper_ci_lo"]) for v in ORDER]
    ci_hi = [np.nan if v == "pca_vraw" else float(recs[v]["event_paper_ci_hi"]) for v in ORDER]
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    ax.bar(range(len(ORDER)), si,
           color=["#b0bec5" if v == "pca_vraw" else "#1b4f72" for v in ORDER],
           width=0.65)
    err_lo = [s - l if not np.isnan(s) else 0 for s, l in zip(si, ci_lo)]
    err_hi = [h - s if not np.isnan(s) else 0 for s, h in zip(si, ci_hi)]
    ax.errorbar(range(len(ORDER)), si, yerr=[err_lo, err_hi], fmt="none",
                ecolor="#263238", elinewidth=1.0, capsize=2.5, capthick=0.9)
    # The per-event test does not control size, so the figure reports the
    # descriptive estimate and interval without a p-value.
    ax.axhline(1.0, ls="--", c="crimson", lw=0.9, zorder=0)
    ax.set_xticks(range(len(ORDER)))
    tick = ["PCA\npres", "PCA\nraw\nn/a", "PCA\nstd", "t-SNE\npres"]
    ax.set_xticklabels(tick, fontsize=6.6)
    ax.set_ylabel("Sparsity drift ratio SDR (LOPO)", fontsize=8)
    ax.set_ylim(0, 1.95)
    ax.tick_params(labelsize=7)
    ax.set_title("(b) per-event paper-weighted SDR by variant\n(95% CI; dashed line: neutral SDR = 1.0)", fontsize=7.8, pad=4)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "fig2_b.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig2_b.pdf"), bbox_inches="tight")
    plt.close(fig)


def panel_c():
    rho, ch = [], []
    with _open_rows() as fh:
        for r in csv.DictReader(fh):
            if r["variant"] == "pca_pres" and r["mode"] == "lopo":
                rho.append(float(r["rho"]))
                ch.append(float(r["churn"]) / 2.0)
    rho = np.asarray(rho)
    ch = np.asarray(ch)
    q = np.quantile(rho, [0.25, 0.5, 0.75])
    masks = [rho <= q[0], (rho > q[0]) & (rho <= q[1]),
             (rho > q[1]) & (rho <= q[2]), rho > q[2]]
    groups = [ch[m] for m in masks]
    # Panel (c) data, written out so the quartiles quoted in §3.2 are pinned to
    # an artifact: the released layer stores |A △ B| / k, the reported fraction
    # chi = |A △ B| / 2k is half of it (both are recorded here).
    out_csv = os.path.join(RESULTS, "fig2_churn_quartiles.csv")
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quartile", "n_rows", "median_churn_reported",
                    "median_churn_raw", "rho_lo", "rho_hi"])
        cuts = [-np.inf, q[0], q[1], q[2], np.inf]
        for i, (g, m) in enumerate(zip(groups, masks), start=1):
            raw = [2.0 * x for x in g]
            w.writerow([f"Q{i}", int(m.sum()), f"{np.median(g):.2f}",
                        f"{np.median(raw):.2f}", f"{cuts[i-1]:.4f}", f"{cuts[i]:.4f}"])
    os.replace(tmp, out_csv)
    print(f"[fig2-print] wrote {os.path.relpath(out_csv, ROOT)}")
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    ax.boxplot(groups, tick_labels=["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("density quartile (sparsest → densest)", fontsize=7.6)
    ax.set_ylabel("fraction of 5-NN replaced", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(c) index-level churn (tie artifact)\n(Presence PCA, LOPO)", fontsize=7.6, pad=4)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "fig2_c.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, "fig2_c.pdf"), bbox_inches="tight")
    plt.close(fig)


def compose():
    from PIL import Image
    a = Image.open(os.path.join(FIGS, "fig2_a.png"))
    b = Image.open(os.path.join(FIGS, "fig2_b.png"))
    c = Image.open(os.path.join(FIGS, "fig2_c.png"))
    gap = 110  # room for the panel-(a) legend between the rows
    w = max(a.width, b.width + c.width + gap)
    h = a.height + gap + max(b.height, c.height)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(a, ((w - a.width) // 2, 0))
    y = a.height + gap
    x0 = (w - (b.width + c.width + gap)) // 2
    canvas.paste(b, (x0, y))
    canvas.paste(c, (x0 + b.width + gap, y))
    out = os.path.join(FIGS, "fig2_composite.png")
    canvas.save(out + ".tmp.png", format="PNG", dpi=(DPI, DPI))
    os.replace(out + ".tmp.png", out)
    canvas.save(os.path.join(FIGS, "fig2_composite.pdf"), format="PDF",
                resolution=float(DPI), creationDate="D:19700101000000",
                modDate="D:19700101000000")
    print(f"[fig2-print] composite {w}x{h}px at {DPI} dpi "
          f"({w/DPI:.2f} x {h/DPI:.2f} in)")


def main():
    panel_a()
    panel_b()
    panel_c()
    compose()


if __name__ == "__main__":
    main()
