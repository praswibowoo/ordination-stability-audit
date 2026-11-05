#!/usr/bin/env python3
"""r1_anchor.py — R1: put Procrustes drift on an interpretable scale.

Two anchors, both computed on the real corpus:

1. Drift as a percentage of the median inter-row distance in the full
   ordination (same normalized coordinate system as the drift).
2. Rank preservation: Spearman correlation between the full-data first-axis
   scores and the leave-one-paper-out first-axis scores, over the rows both
   fits share (after Procrustes alignment).

Variants: presence PCA (764 rows) and standardized value PCA (421 rows,
70 papers). Writes results/r1_anchor.csv.

Run: python3 code/core/r1_anchor.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import load, ordinate  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def normalise(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    S = S - S.mean(axis=0, keepdims=True)
    n = np.linalg.norm(S)
    return S / n if n else S


def run(variant: str, X, Xv, paper, label: str):
    if variant == "pca_vstd":
        S = ordinate(variant, X, Xv)
        papers = paper
    else:
        S = ordinate(variant, X)
        papers = paper
    d_med = float(np.median(pdist(normalise(S))))

    rows = []
    all_drift = []
    for p in sorted(set(papers)):
        keep = np.where(papers != p)[0]
        if variant == "pca_vstd":
            S_red = ordinate(variant, X, Xv[keep])
        else:
            S_red = ordinate(variant, X[keep])
        a, b, _ = procrustes(S[keep], S_red)
        drift = np.linalg.norm(b - a, axis=1)
        all_drift.append(drift)
        rho = float(spearmanr(a[:, 0], b[:, 0]).statistic)
        rows.append(dict(paper=str(p), n_rows=int(len(keep)),
                         med_drift=float(np.median(drift)),
                         rho_pc1=rho))

    med_drift = float(np.median([r["med_drift"] for r in rows]))
    rhos = np.asarray([r["rho_pc1"] for r in rows], float)
    flat = np.concatenate(all_drift) if all_drift else np.asarray([np.nan])
    out = dict(
        variant=label,
        n_events=len(rows),
        n_rows=int(S.shape[0]),
        n_papers=int(len(set(papers))),
        median_interrow_distance=d_med,
        median_drift=med_drift,
        drift_pct_of_interrow=100.0 * med_drift / d_med if d_med else float("nan"),
        p95_drift_pct=100.0 * float(np.quantile(flat, 0.95)) / d_med if d_med else float("nan"),
        max_drift_pct=100.0 * float(np.max(flat)) / d_med if d_med else float("nan"),
        max_event_median_pct=100.0 * float(np.max([r["med_drift"] for r in rows])) / d_med if d_med else float("nan"),
        pct_rows_above_5pct=100.0 * float(np.mean(flat > 0.05 * d_med)) if d_med else float("nan"),
        median_rho_pc1=float(np.median(rhos)),
        q1_rho_pc1=float(np.quantile(rhos, 0.25)),
        q3_rho_pc1=float(np.quantile(rhos, 0.75)),
        min_rho_pc1=float(np.min(rhos)),
        pct_events_rho_below_0_90=100.0 * float(np.mean(rhos < 0.90)),
    )
    return out, rows


def main() -> None:
    X, rho, paper, params, Xv, keys, rho_v = load()
    paper_v = np.asarray([k[0] for k in keys])

    out = []
    detail = []
    for variant, label, pap in (
        ("pca_pres", "presence PCA (2-D)", paper),
        ("pca_vstd", "standardized value PCA (2-D)", paper_v),
    ):
        row, rows = run(variant, X, Xv, pap, label)
        out.append(row)
        for r in rows:
            r["variant"] = label
            detail.append(r)

    path = os.path.join(RESULTS, "r1_anchor.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    dpath = os.path.join(RESULTS, "r1_anchor_events.csv")
    with open(dpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "paper", "n_rows",
                                           "med_drift", "rho_pc1"])
        w.writeheader()
        w.writerows(detail)

    for r in out:
        print(f"[r1] {r['variant']}: events={r['n_events']} "
              f"drift={r['median_drift']:.3e} "
              f"({r['drift_pct_of_interrow']:.2f}% of median inter-row) "
              f"rho_PC1 median={r['median_rho_pc1']:.4f} "
              f"[{r['q1_rho_pc1']:.4f}, {r['q3_rho_pc1']:.4f}] "
              f"min={r['min_rho_pc1']:.4f} "
              f"events<0.90={r['pct_events_rho_below_0_90']:.1f}%")
    print(f"[r1] wrote {os.path.relpath(path, ROOT)} and {os.path.relpath(dpath, ROOT)}")


if __name__ == "__main__":
    main()
