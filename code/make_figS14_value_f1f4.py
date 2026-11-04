#!/usr/bin/env python3
"""make_figS14_value_f1f4.py — value-layer cross-ordination, F1 physical + F4 organic.

The value layer has no all-parameter complete case: 0 rows are complete for the
28 assigned parameters, and the largest complete multi-family set spans three
families. This figure is the second two-family case the data support with a
usable n: DO, pH and Temperature (F1 physical) plus BOD and COD (F4 organic).

Selection
---------
* every cell must be present in docs/extraction/stageB/matrix_values_v2.csv;
* the row must be a data row (docs/family_map.csv row_kind == "data");
* rows whose verdict is "Reference standard" are excluded (they are regulatory
  rows, not samples).

Result: n = 73 rows from 12 papers (74 complete cases minus 1 reference row).
PCA on the correlation matrix (z-scored columns, ddof=1). Signs are pinned so
that BOD loads positively on PC1 and Temperature positively on PC2, which keeps
the figure reproducible across LAPACK builds.

Writes
------
figures/figS14_value_f1f4.png / .pdf        (600 dpi twins)
results/exS14_value_f1f4_scores.csv         (row-level scores)
results/exS14_value_f1f4_summary.csv        (n, papers, variance, loadings)

Run: python3 code/make_figS14_value_f1f4.py
"""
from __future__ import annotations

import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twin

import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures", "figS14_value_f1f4")
RESULTS = os.path.join(ROOT, "results")
DPI = 600
PARAMS = ["DO (mg/L)", "pH", "Temperature (°C)", "BOD (mg/L)", "COD (mg/L)"]
SHORT = ["DO", "pH", "Temp", "BOD", "COD"]
F1_PARAMS = {"DO", "pH", "Temp"}
VERDICT_COLORS = {"Favorable": "#2e7d32", "Stable": "#66bb6a", "Mild stress": "#f9a825",
                  "Stressful": "#ef6c00", "Harmful": "#c62828", "Unlabeled": "#b0bec5",
                  "Reference standard": "#9e9e9e"}


def load_data() -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(open(os.path.join(
        ROOT, "docs/extraction/stageB/matrix_values_v2.csv"), encoding="utf-8")))
    values: dict[tuple[str, str], dict[str, str]] = {}
    scenario: dict[tuple[str, str], str] = {}
    for r in rows:
        key = (r["batch_id"], r["row_id"])
        scenario[key] = r["scenario"]
        if (r["value"] or "").strip():
            values.setdefault(key, {})[r["param"]] = r["value"].strip()

    # verdicts: author labels first, derived labels second
    verdict: dict[tuple[str, str], str] = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "release/p1_family_pca_data_v1.0/verdicts.csv"), encoding="utf-8")):
        verdict[(r["batch_id"], r["row_id"])] = r["verdict"]
    for r in csv.DictReader(open(os.path.join(
            ROOT, "results/verdict_derivation_log.csv"), encoding="utf-8")):
        verdict.setdefault((r["batch_id"], r["row_id"]), r["verdict"])

    data_keys = {(r["batch_id_filled"], r["row_id"]) for r in csv.DictReader(open(
        os.path.join(ROOT, "docs", "family_map.csv"), encoding="utf-8"))
        if r["row_kind"] == "data"}

    out = []
    for key, d in values.items():
        if not set(PARAMS) <= set(d):
            continue
        if key not in data_keys:
            continue
        v = verdict.get(key, "")
        if not v or v == "TBD-StageB":
            v = "Unlabeled"
        if v == "Reference standard":
            continue
        out.append({"key": key, "batch": key[0], "row": key[1],
                    "scenario": scenario.get(key, ""), "verdict": v,
                    "values": [float(d[p]) for p in PARAMS]})

    assert len(out) == 73, f"expected 73 analytical rows, got {len(out)}"
    papers = sorted({r["batch"] for r in out})
    assert len(papers) == 12, papers
    return out, {"papers": papers}


def pca(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray([r["values"] for r in rows], float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S ** 2) / np.sum(S ** 2)
    # pin signs for reproducibility: NH4 positive on PC1, Temp positive on PC2
    if Vt[0, PARAMS.index("BOD (mg/L)")] < 0:
        Vt[0] *= -1.0
        U[:, 0] *= -1.0
    if Vt[1, PARAMS.index("Temperature (°C)")] < 0:
        Vt[1] *= -1.0
        U[:, 1] *= -1.0
    scores = U[:, :2] * S[:2]
    return scores, Vt[:2], var[:2]


def figure(rows: list[dict], scores: np.ndarray, load: np.ndarray, var: np.ndarray,
           papers: list[str]) -> None:
    verdicts = [r["verdict"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharex=True, sharey=True)

    # Panel A: verdict class
    ax = axes[0]
    for v in ["Favorable", "Stable", "Mild stress", "Stressful", "Harmful", "Unlabeled"]:
        m = np.asarray([x == v for x in verdicts])
        if m.any():
            ax.scatter(scores[m, 0], scores[m, 1], s=26, c=VERDICT_COLORS[v],
                       alpha=0.85, edgecolor="white", linewidth=0.5, label=v)
    ax.set_title("A  coloured by verdict class", loc="left", fontsize=9)
    ax.legend(fontsize=6.0, frameon=False, loc="lower right", handletextpad=0.2,
              borderpad=0.2, labelspacing=0.25)

    # Panel B: paper identity
    ax = axes[1]
    cmap = plt.get_cmap("tab20")
    for i, paper in enumerate(papers):
        m = np.asarray([r["batch"] == paper for r in rows])
        ax.scatter(scores[m, 0], scores[m, 1], s=26, c=[cmap(i % 20)],
                   alpha=0.85, edgecolor="white", linewidth=0.5, label=paper)
    ax.set_title("B  coloured by paper", loc="left", fontsize=9)

    for ax in axes:
        ax.axhline(0, lw=0.5, c="#90a4ae")
        ax.axvline(0, lw=0.5, c="#90a4ae")
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)", fontsize=8.6)
    axes[0].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)", fontsize=8.6)
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8)
    axes[0].text(0.02, 0.07, "F1 physical: DO, pH, Temp", transform=axes[0].transAxes,
                 fontsize=5.8, color="#263238", va="bottom", bbox=box)
    axes[0].text(0.02, 0.02, "F4 organic: BOD, COD", transform=axes[0].transAxes,
                 fontsize=5.8, color="#263238", va="bottom", bbox=box)
    axes[1].set_ylabel("")

    # Unit-normalised loading arrows (direction only), same on both panels
    for ax in axes:
        rad = 0.30 * min(np.ptp(scores[:, 0]), np.ptp(scores[:, 1]))
        for j, name in enumerate(SHORT):
            vx, vy = float(load[0, j]), float(load[1, j])
            nrm = float(np.hypot(vx, vy)) or 1.0
            ux, uy = vx / nrm * rad, vy / nrm * rad
            ax.annotate("", xy=(ux, uy), xytext=(0, 0), zorder=6, annotation_clip=False,
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="#37474f"))
            off = {"DO": (8, -6, "left", "top"),
                   "pH": (8, 4, "left", "bottom"),
                   "Temp": (-8, 2, "right", "bottom"),
                   "BOD": (9, 6, "left", "bottom"),
                   "COD": (9, -6, "left", "top")}[name]
            ax.annotate(name, (ux, uy), xytext=off[:2], textcoords="offset points",
                        fontsize=6.8, color="#263238", ha=off[2], va=off[3], zorder=7,
                        annotation_clip=False)
        ax.margins(0.08)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=6.0,
               frameon=False, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3,
               columnspacing=0.9)
    fig.tight_layout(pad=0.6, rect=(0, 0.07, 1, 1))
    fig.savefig(FIG + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG + ".pdf", bbox_inches="tight")
    plt.close(fig)


def write_tables(rows: list[dict], scores: np.ndarray, load: np.ndarray,
                 var: np.ndarray) -> None:
    with open(os.path.join(RESULTS, "exS14_value_f1f4_scores.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["batch_id", "row_id", "scenario", "verdict", "PC1", "PC2"])
        for r, (x, y) in zip(rows, scores):
            w.writerow([r["batch"], r["row"], r["scenario"], r["verdict"],
                        f"{x:.6f}", f"{y:.6f}"])
    with open(os.path.join(RESULTS, "exS14_value_f1f4_summary.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["n_rows", len(rows)])
        w.writerow(["n_papers", len({r['batch'] for r in rows})])
        w.writerow(["variance_pc1", f"{var[0]:.6f}"])
        w.writerow(["variance_pc2", f"{var[1]:.6f}"])
        for j, p in enumerate(PARAMS):
            w.writerow([f"loading_pc1_{SHORT[j]}", f"{load[0, j]:.6f}"])
        for j, p in enumerate(PARAMS):
            w.writerow([f"loading_pc2_{SHORT[j]}", f"{load[1, j]:.6f}"])


def main() -> int:
    rows, info = load_data()
    scores, load, var = pca(rows)
    figure(rows, scores, load, var, info["papers"])
    write_tables(rows, scores, load, var)
    print(f"[figS14] n={len(rows)} rows, {len(info['papers'])} papers, "
          f"PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%")
    print(f"[figS14] wrote figures/figS14_value_f1f4.png/.pdf and results/exS14_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
