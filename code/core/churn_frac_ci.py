#!/usr/bin/env python3
"""churn_frac_ci.py — paper-level interval for the tie-aware churn fraction.

`pattern_churn.py` writes the point estimate only: `frac_rows_tie_aware_churn_gt0`
is the median over the 122 leave-one-paper-out events of the per-event fraction of
multi-configuration rows whose tie-aware churn is positive. This script keeps the
per-event values and puts a percentile interval around that median by resampling
whole papers, which is the same unit the manuscript resamples elsewhere (SDR
interval, Section 2.1).

Estimand: median over events of (rows with tie-aware churn > 0) / (rows whose
neighbourhood spans more than one configuration). Intervals come from 1,000
bootstrap resamples of the 122 event values, seed 42.

Outputs:
  results/ex_pattern_churn_ci.csv

Run: python3 code/core/churn_frac_ci.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter

import numpy as np

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")

from experiment_C1 import load, ordinate, knn_idx

VARIANTS = ("pca_pres", "tsne_pres", "mds_pres")
RESAMPLES = 1000
SEED = 42


def per_event_frac(variant: str, X: np.ndarray, paper: np.ndarray,
                   inv: np.ndarray) -> np.ndarray:
    """Fraction of multi-configuration rows with positive tie-aware churn, per event."""
    knn_full = knn_idx(ordinate(variant, X, None))
    papers = sorted(set(paper))
    out = []
    for p in papers:
        keep = (paper != p)
        keep_idx = np.where(keep)[0]
        nn_red = keep_idx[knn_idx(ordinate(variant, X[keep], None))]
        frac = []
        for a, b in zip(knn_full[keep_idx], nn_red):
            pats_a, pats_b = Counter(inv[a]), Counter(inv[b])
            if len(pats_a) > 1 and len(pats_b) > 1:
                diff = (sum((pats_a - pats_b).values())
                        + sum((pats_b - pats_a).values()))
                frac.append(diff / (2.0 * min(len(pats_a), len(pats_b))) > 0)
        out.append(float(np.mean(frac)) if frac else np.nan)
    return np.asarray(out, float)


def main() -> None:
    X, _rho, paper, _params, *_ = load()
    B = (X > 0).astype(int)
    _pats, inv, _cnt = np.unique(B, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()

    rows = []
    for variant in VARIANTS:
        events = per_event_frac(variant, X, paper, inv)
        rng = np.random.default_rng(SEED)
        medians = [float(np.median(rng.choice(events, size=events.size, replace=True)))
                   for _ in range(RESAMPLES)]
        point, lo, hi = (float(np.median(events)),
                         float(np.percentile(medians, 2.5)),
                         float(np.percentile(medians, 97.5)))
        rows.append({
            "variant": variant,
            "mode": "lopo",
            "n_events": int(events.size),
            "median_pct": round(point * 100.0, 2),
            "ci_lo_pct": round(lo * 100.0, 2),
            "ci_hi_pct": round(hi * 100.0, 2),
            "resamples": RESAMPLES,
            "seed": SEED,
        })
        print(f"[churn-ci] {variant:10s} median {point * 100:5.2f}%  "
              f"95% CI {lo * 100:5.2f} to {hi * 100:5.2f}  (n_events={events.size})")

    path = os.path.join(RESULTS, "ex_pattern_churn_ci.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[churn-ci] wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
