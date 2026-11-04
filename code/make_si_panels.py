#!/usr/bin/env python3
"""make_si_panels.py — tidy rebuild of the supplementary family-map panels.

Fixes over the original experiment_B biplots:
  * display family names in titles ("F1a physical–anion", not "F1_physicala");
  * legend group "TBD-StageB" renamed to "no label";
  * arrow labels for near-collinear loadings (BOD/COD, TDS/EC) offset so they
    no longer overprint each other;
  * panel subtitle shortened to one line.

Panels rebuilt (11): presence and value variants for F1 clarity, F1a
physical–anion, F2 metals, F3 nutrients, F4 organic, F6 aeration, plus the
F1 physical presence variant (its value map is main-text Fig. 3). The two
value panels shown in main-text Fig. 3 are left untouched.

Writes figures/fig3_<family>_<layer>.png/pdf (600 dpi).
Run: python3 code/make_si_panels.py
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

DISPLAY = {
    "F1_physical": "F1 physical",
    "F1_physicala": "F1a physical\u2013anion",
    "F1_clarity": "F1 clarity",
    "F2_metals": "F2 metals",
    "F3_nutrients": "F3 nutrients",
    "F4_organic": "F4 organic",
    "F6_aeration": "F6 aeration",
}
LEGEND = {"TBD-StageB": "Unlabeled"}
PANELS = [
    ("F1_physical", "presence"),
    ("F1_clarity", "presence"), ("F1_clarity", "value"),
    ("F1_physicala", "presence"), ("F1_physicala", "value"),
    ("F2_metals", "presence"),
    ("F3_nutrients", "presence"),
    ("F4_organic", "presence"), ("F4_organic", "value"),
    ("F6_aeration", "presence"), ("F6_aeration", "value"),
]


def draw(family: str, layer: str, path: str) -> None:
    fam_params = b.FAMILIES[family]
    rows, vals, fmap, params = b.load_all()
    subset = b.family_subset(rows, vals, fmap, family, fam_params)
    if layer == "presence":
        sub = [s for s in subset if s["cov"]]
        X = np.asarray([s["X_fam"] for s in sub], float)
        pca = PCA(n_components=2, random_state=42).fit(X)
        S = pca.transform(X)
        load = pca.components_.T * np.sqrt(pca.explained_variance_)
    else:
        sub = [s for s in subset if s["vcomplete"]]
        X = np.asarray([[s["values"][p] for p in fam_params] for s in sub], float)
        Z = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42).fit(Z)
        S = pca.transform(Z)
        load = pca.components_.T * np.sqrt(pca.explained_variance_)
    verd = [s["row"]["verdict"] or "TBD-StageB" for s in sub]

    S_d = S.copy()
    n_out = 0
    for k in range(2):
        q1, q3 = np.quantile(S_d[:, k], [0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_out += int(((S_d[:, k] < lo) | (S_d[:, k] > hi)).sum())
        S_d[:, k] = np.clip(S_d[:, k], lo, hi)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sev = ["Favorable", "Stable", "Mild stress", "Stressful", "Harmful",
           "Reference standard", "Reference baseline", "Degraded", "TBD-StageB"]
    present = [v for v in sev if v in set(verd)] + \
              sorted(v for v in set(verd) if v not in sev)
    for v in present:
        m = np.asarray([vv == v for vv in verd])
        ax.scatter(S_d[m, 0], S_d[m, 1], s=14, c=b.PALETTE.get(v, "#eceff1"),
                   marker=b.MARKERS.get(v, "o"), alpha=0.6, label=LEGEND.get(v, v))

    xr = float(np.ptp(S_d[:, 0])) or 1.0
    yr = float(np.ptp(S_d[:, 1])) or 1.0
    rad = 0.34 * max(xr, yr)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x0, x1 = min(x0, -rad * 1.15), max(x1, rad * 1.15)
    y0, y1 = min(y0, -rad * 1.15), max(y1, rad * 1.15)
    placed = []  # angles of already-placed arrow labels (collinear-group stacking)
    for j in range(load.shape[0]):
        vx, vy = float(load[j, 0]), float(load[j, 1])
        nrm = float(np.hypot(vx, vy))
        if nrm < 1e-12:
            continue
        ux, uy = vx / nrm * rad, vy / nrm * rad
        ang = float(np.arctan2(uy, ux))
        k = sum(1 for a in placed if abs(a - ang) < 0.12)  # index within group
        placed.append(ang)
        ax.annotate("", xy=(ux, uy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#37474f"))
        # base label offset, then deterministic stagger along the perpendicular
        # for collinear loadings (11 pt steps: 0, +11, -11, +22, -22, ...)
        base = (3 if ux >= 0 else -3, 3 if uy >= 0 else -3)
        perp = np.asarray([-uy, ux]) / max(np.hypot(uy, ux), 1e-9)
        if k == 0:
            shift = np.zeros(2)
        else:
            step = 11.0 * ((k + 1) // 2)
            shift = perp * step * (1.0 if k % 2 == 1 else -1.0)
        if family == "F1_physical" and layer == "presence":
            fx = (ux - x0) / (x1 - x0)
            fy = (uy - y0) / (y1 - y0)
            if fx > 0.85:
                ha, dx = "right", -4.0
            elif fx < 0.15:
                ha, dx = "left", 4.0
            else:
                ha, dx = ("left", 3.0) if ux >= 0 else ("right", -3.0)
            if fy > 0.88:
                va, dy = "top", -4.0
            elif fy < 0.12:
                va, dy = "bottom", 4.0
            else:
                va, dy = ("bottom", 3.0) if uy >= 0 else ("top", -3.0)
            ax.annotate(fam_params[j].split(" (")[0], (ux, uy),
                        xytext=(dx + float(shift[0]), dy + float(shift[1])),
                        textcoords="offset points", fontsize=6.6, color="#263238",
                        ha=ha, va=va)
        else:
            ax.annotate(fam_params[j].split(" (")[0], (ux, uy),
                        xytext=(base[0] + float(shift[0]), base[1] + float(shift[1])),
                        textcoords="offset points", fontsize=6.6, color="#263238",
                        ha="left" if ux >= 0 else "right",
                        va="bottom" if uy >= 0 else "top")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axhline(0, lw=0.5, c="#90a4ae")
    ax.axvline(0, lw=0.5, c="#90a4ae")
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{DISPLAY[family]} ({layer}) \u2014 verdict scores, "
                 f"unit-normalized loadings",
                 fontsize=8.5)
    if n_out:
        ax.annotate(f"{n_out} score(s) beyond 1.5\u00d7IQR display fence "
                    "(clipped for readability)",
                    (0.01, 0.01), xycoords="axes fraction", fontsize=7.0,
                    color="#37474f")
    ax.legend(fontsize=6.4, markerscale=1.4, loc="best", framealpha=0.9)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {family:14s} {layer:8s} n={len(sub)}")


def main() -> None:
    for family, layer in PANELS:
        draw(family, layer, os.path.join(FIGS, f"fig3_{family}_{layer}.png"))
    print(f"[si-panels] rebuilt {len(PANELS)} tidied panels")


if __name__ == "__main__":
    main()
