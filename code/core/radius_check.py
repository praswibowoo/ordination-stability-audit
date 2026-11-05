#!/usr/bin/env python3
"""radius_check.py — is the reverse density association a geometry artifact?

Presence-PCA and t-SNE show denser rows drifting more (two-sided paper-level
p = 0.002 and 0.010; Table S3). A natural artifact explanation is that denser
rows sit farther from the ordination centroid and therefore move more under any
rotation. This script tests that explanation directly: row radius in the full
presence-PCA fit is correlated with drift and with density.

Writes results/exC1_radius_check.csv.
Run: python3 code/core/radius_check.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import load  # noqa: E402


def main() -> None:
    X, rho, paper, *_ = load()
    S = PCA(n_components=2, random_state=42).fit_transform(X)
    radius = np.linalg.norm(S, axis=1)

    drift = {}
    with gzip.open(os.path.join(ROOT, "results", "exC1_rows.csv.gz"),
                   "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] == "pca_pres" and r["mode"] == "lopo":
                drift.setdefault(int(r["row_id"]), []).append(float(r["drift"]))
    md = np.asarray([np.median(drift[i]) if i in drift else np.nan
                     for i in range(len(X))])
    ok = ~np.isnan(md)

    rows = [
        dict(association="radius vs density rho",
             spearman=round(float(spearmanr(radius, rho).statistic), 4),
             n=int(len(rho))),
        dict(association="radius vs per-row median drift",
             spearman=round(float(spearmanr(radius[ok], md[ok]).statistic), 4),
             n=int(ok.sum())),
        dict(association="density rho vs per-row median drift",
             spearman=round(float(spearmanr(rho[ok], md[ok]).statistic), 4),
             n=int(ok.sum())),
    ]
    path = os.path.join(ROOT, "results", "exC1_radius_check.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["association", "spearman", "n"])
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)
    for r in rows:
        print(f"[radius] {r['association']:38s} rho = {r['spearman']:+.4f} (n={r['n']})")


if __name__ == "__main__":
    main()
