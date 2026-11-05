#!/usr/bin/env python3
"""tie_order_sensitivity.py — does churn depend on how the query breaks ties?

The release computes k-nearest-neighbour churn with numpy's default argsort,
which does not guarantee index order among equidistant rows. This script repeats
the same leave-one-paper-out analysis twice per presence variant:

  default  — np.argsort(D, axis=1)          (the released pipeline)
  stable   — np.argsort(D, axis=1, kind="stable")  (ties resolve by row position)

If index-level churn is tie-resolution dependent, the two rules diverge; if the
tie-aware (configuration-level) measure is robust, it should not move.

Outputs: results/tie_order_sensitivity.csv
Run:    python3 code/core/tie_order_sensitivity.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist, squareform

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "tie_order_sensitivity.csv")

from experiment_C1 import load, ordinate  # noqa: E402

K = 5
VARIANTS = ("pca_pres", "mds_pres", "tsne_pres")


def knn(S: np.ndarray, stable: bool) -> np.ndarray:
    D = squareform(pdist(S))
    np.fill_diagonal(D, np.inf)
    if stable:
        return np.argsort(D, axis=1, kind="stable")[:, :K]
    return np.argsort(D, axis=1)[:, :K]


def main() -> None:
    X, _rho, paper, _params, *_ = load()
    B = (X > 0).astype(int)
    _pats, inv, _cnt = np.unique(B, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()

    records = []
    for variant in VARIANTS:
        S_full = ordinate(variant, X, None)
        full = {False: knn(S_full, False), True: knn(S_full, True)}
        acc = {False: ([], [], []), True: ([], [], [])}

        for p in sorted(set(paper)):
            keep = paper != p
            keep_idx = np.where(keep)[0]
            S_red = ordinate(variant, X[keep], None)
            red = {False: knn(S_red, False), True: knn(S_red, True)}

            for stable in (False, True):
                A = full[stable][keep_idx]
                Bn = keep_idx[red[stable]]
                idx_churn, pat_churn, tie_vals = [], [], []
                for a, b in zip(A, Bn):
                    idx_churn.append(len(set(a) ^ set(b)) / (2.0 * K))
                    pa, pb = Counter(inv[a]), Counter(inv[b])
                    diff = sum((pa - pb).values()) + sum((pb - pa).values())
                    pat_churn.append(diff / (2.0 * len(a)))
                    if len(pa) > 1 and len(pb) > 1:
                        tie_vals.append(diff / (2.0 * min(len(pa), len(pb))))
                acc[stable][0].append(float(np.median(idx_churn)))
                acc[stable][1].append(float(np.median(pat_churn)))
                acc[stable][2].append(
                    float(np.mean(np.asarray(tie_vals) > 0)) if tie_vals else float("nan")
                )

        for stable, label in ((False, "default"), (True, "index-stable")):
            records.append({
                "variant": variant,
                "tie_rule": label,
                "mode": "lopo",
                "events_evaluated": len(acc[stable][0]),
                "median_idx_churn": float(np.median(acc[stable][0])),
                "median_pat_churn": float(np.median(acc[stable][1])),
                "frac_tie_aware_gt0": float(np.nanmedian(acc[stable][2])),
            })

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"[tie-order] wrote {OUT} ({len(records)} rows)")


if __name__ == "__main__":
    main()
