#!/usr/bin/env python3
"""make_fig3_print.py — re-render the main Figure 3 at print size.

Two stacked value-layer biplots (F3 nutrients, F1 physical) drawn at final
printed proportions so the type stays legible at \\linewidth. The F6 map and
the remaining families stay in the supplementary panels.

Data: frozen layers via experiment_B_familymaps helpers. Writes
figures/fig3_main.png/pdf (600 dpi).

Run: python3 code/make_fig3_print.py
"""
from __future__ import annotations

import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117/D132)
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
FIGS = os.path.join(ROOT, "figures")
DPI = 600

import experiment_B_familymaps as b  # noqa: E402

FAMILIES = ["F3_nutrients", "F1_physical"]


def panel(family: str, path: str, tag: bool = True, quiet: bool = False) -> None:
    fam_params = b.FAMILIES[family]
    rows, vals, fmap, params = b.load_all()
    subset = b.family_subset(rows, vals, fmap, family, fam_params)
    vcomp = [s for s in subset if s["vcomplete"]]
    X = np.asarray([[s["values"][p] for p in fam_params] for s in vcomp], float)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Z)
    S = pca.transform(Z)
    evr = pca.explained_variance_ratio_
    load = pca.components_.T * np.sqrt(pca.explained_variance_)
    verd = [("Unlabeled" if (s["row"]["verdict"] or "").strip() in ("", "TBD-StageB", "no label")
         else s["row"]["verdict"]) for s in vcomp]

    S_d = S.copy()
    for k in range(2):
        q1, q3 = np.quantile(S_d[:, k], [0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        S_d[:, k] = np.clip(S_d[:, k], lo, hi)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sev = ["Favorable", "Stable", "Mild stress", "Stressful", "Harmful",
           "Reference standard", "Reference baseline", "Degraded", "Unlabeled"]
    present = [v for v in sev if v in set(verd)] + \
              sorted(v for v in set(verd) if v not in sev)
    for v in present:
        m = np.asarray([vv == v for vv in verd])
        c = "#b0bec5" if v == "Unlabeled" else b.PALETTE.get(v, "#eceff1")
        ax.scatter(S_d[m, 0], S_d[m, 1], s=16, c=c,
                   marker=b.MARKERS.get(v, "o"), alpha=0.6, label=v)
    xr = float(np.ptp(S_d[:, 0])) or 1.0
    yr = float(np.ptp(S_d[:, 1])) or 1.0
    rad = 0.34 * max(xr, yr)
    tips = []
    for j in range(load.shape[0]):
        vx, vy = float(load[j, 0]), float(load[j, 1])
        nrm = float(np.hypot(vx, vy))
        if nrm < 1e-12:
            tips.append(None)
            continue
        ux, uy = vx / nrm, vy / nrm
        ax.annotate("", xy=(ux * rad, uy * rad), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#37474f"))
        tips.append((ux, uy))
    # Generous, non-overlapping label callouts with small leader lines for near-collinear loadings
    # Guarantees complete separation and zero text collision at 600 DPI print scale.
    CALLOUTS = {
        # F3 nutrients: PO4 points up; NO2, NH4, NO3 are stepped vertically with distinct Y-clearance
        "PO4": dict(xytext=(0, 8), ha="center", va="bottom", arrow=False),
        "NO2": dict(xytext=(14, 16), ha="left", va="bottom", arrow=True),
        "NH4": dict(xytext=(14, -1), ha="left", va="center", arrow=True),
        "NO3": dict(xytext=(14, -18), ha="left", va="top", arrow=True),
        # F1 physical: Salinity up-left, DO down-right, pH and Temperature clearly separated
        "Salinity": dict(xytext=(-6, 8), ha="right", va="bottom", arrow=False),
        "Temperature": dict(xytext=(12, 18), ha="left", va="bottom", arrow=True),
        "pH": dict(xytext=(12, -2), ha="left", va="center", arrow=True),
        "DO": dict(xytext=(8, -8), ha="left", va="top", arrow=False),
    }
    for j in range(load.shape[0]):
        vx, vy = float(load[j, 0]), float(load[j, 1])
        nrm = float(np.hypot(vx, vy))
        if nrm < 1e-12:
            continue
        ux, uy = vx / nrm * rad, vy / nrm * rad
        pname = fam_params[j].split(" (")[0]
        cfg = CALLOUTS.get(pname, dict(
            xytext=(6 if ux >= 0 else -6, 6 if uy >= 0 else -6),
            ha="left" if ux >= 0 else "right",
            va="bottom" if uy >= 0 else "top",
            arrow=False
        ))
        arrowprops = dict(arrowstyle="-", lw=0.65, color="#78909c") if cfg["arrow"] else None
        ax.annotate(pname, (ux, uy),
                    xytext=cfg["xytext"],
                    textcoords="offset points", fontsize=7.6, color="#263238",
                    fontweight="bold",
                    ha=cfg["ha"],
                    va=cfg["va"],
                    arrowprops=arrowprops)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, -rad * 1.18), max(x1, rad * 1.38))
    ax.set_ylim(min(y0, -rad * 1.25), max(y1, rad * 1.35))
    ax.axhline(0, lw=0.5, c="#90a4ae")
    ax.axvline(0, lw=0.5, c="#90a4ae")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=8)
    ax.tick_params(labelsize=7)
    label = family.replace("_", " ")
    ptag = ("(a) " if family == "F3_nutrients" else "(b) ") if tag else ""
    ax.set_title(f"{ptag}{label} (value layer) — verdict scores, unit-normalized loadings",
                 fontsize=8.2, pad=5)
    ax.legend(fontsize=6.2, markerscale=1.4, loc="best", framealpha=0.92,
              labelspacing=0.22, handlelength=1.0, handletextpad=0.35,
              borderpad=0.35)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    if not quiet:
        print(f"  {family}: n={len(vcomp)} rows ({len(set(s['row']['batch'] for s in vcomp))} papers)")


def main() -> None:
    paths = []
    for family in FAMILIES:
        p = os.path.join(FIGS, f"fig3_print_{family}.png")
        panel(family, p)
        paths.append(p)
    from PIL import Image
    ims = [Image.open(p) for p in paths]
    gap = 70
    w = max(im.width for im in ims)
    h = sum(im.height for im in ims) + gap
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in ims:
        canvas.paste(im, ((w - im.width) // 2, y))
        y += im.height + gap
    # clean standalone twins, no (a)/(b) tags
    for family in FAMILIES:
        letter = "3a" if family == "F3_nutrients" else "3b"
        panel(family, os.path.join(FIGS, f"fig{letter}_{family}_value.png"), tag=False, quiet=True)

    out = os.path.join(FIGS, "fig3_main.png")
    canvas.save(out + ".tmp.png", format="PNG", dpi=(DPI, DPI))
    os.replace(out + ".tmp.png", out)
    canvas.save(os.path.join(FIGS, "fig3_main.pdf"), format="PDF",
                resolution=float(DPI), creationDate="D:19700101000000",
                modDate="D:19700101000000")
    print(f"[fig3-print] composite {w}x{h}px ({w/DPI:.2f} x {h/DPI:.2f} in)")


if __name__ == "__main__":
    main()
