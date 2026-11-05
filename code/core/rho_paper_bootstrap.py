#!/usr/bin/env python3
"""rho_paper_bootstrap.py — paper-level intervals for the presence-PCA density and radius diagnostics

Presence PCA puts denser rows further from their leave-one-paper-out positions
(Spearman rho, per-row median drift = 0.537 over 764 rows; code/core/radius_check.py).
Those rows are nested in 122 studies, and the manuscript treats the study as the
inference unit, so a row-level coefficient leaves open whether the association
survives at the level the paper claims. This script resamples papers with
replacement and recomputes each coefficient in every resample.

It also records where row density sits in the fit: the component correlations
(rho vs PC1, rho vs PC2 scores) are written here because the manuscript's account
of the reverse density association depends on which component carries it.

Point estimates are recomputed here rather than copied from
exC1_radius_check.csv, so each interval and its estimate come from one pass.

Writes results/exC1_rho_paper_ci.csv.
Run: make rho-bootstrap
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

N_BOOT = 2000
SEED = 42
ALPHA = 0.05


def load_drift(n_rows: int) -> np.ndarray:
    """Per-row median LOPO drift for the presence-PCA variant."""
    drift = {}
    with gzip.open(os.path.join(ROOT, "results", "exC1_rows.csv.gz"),
                   "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] == "pca_pres" and r["mode"] == "lopo":
                drift.setdefault(int(r["row_id"]), []).append(float(r["drift"]))
    return np.asarray([np.median(drift[i]) if i in drift else np.nan
                       for i in range(n_rows)])


def main() -> None:
    X, rho, paper, *_ = load()
    # Same fit as radius_check.py: two-component PCA on the full presence matrix.
    S = PCA(n_components=2, random_state=42).fit_transform(X)
    radius = np.linalg.norm(S, axis=1)

    md = load_drift(len(X))
    ok = ~np.isnan(md)
    rho, radius, paper, md = rho[ok], radius[ok], paper[ok], md[ok]
    pc1, pc2 = S[ok, 0], S[ok, 1]

    pairs = {
        "rho_drift": (rho, md),
        "radius_drift": (radius, md),
    }

    papers = np.array(sorted(set(paper)))
    idx = {p: np.where(paper == p)[0] for p in papers}

    rng = np.random.default_rng(SEED)
    boots = {k: [] for k in pairs}
    for _ in range(N_BOOT):
        pick = rng.choice(len(papers), len(papers), replace=True)
        sel = np.concatenate([idx[papers[k]] for k in pick])
        if np.std(md[sel]) == 0.0:
            continue
        for k, (a, b) in pairs.items():
            if np.std(a[sel]) == 0.0:
                continue
            boots[k].append(float(spearmanr(a[sel], b[sel]).statistic))

    rows = []
    for k, (a, b) in pairs.items():
        arr = np.asarray(boots[k])
        lo, hi = np.percentile(arr, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        rows.append(dict(metric=k, value=round(float(spearmanr(a, b).statistic), 4),
                         abs_value="",
                         ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                         n_papers=len(papers), n_rows=int(len(a)), n_boot=len(arr)))

    for name, (a, b) in (("rho_pc1", (rho, pc1)), ("rho_pc2", (rho, pc2))):
        v = float(spearmanr(a, b).statistic)
        rows.append(dict(metric=name, value=round(v, 4),
                         abs_value=round(abs(v), 4),
                         ci_lo="", ci_hi="", n_papers=len(papers),
                         n_rows=int(len(a)), n_boot=""))

    path = os.path.join(ROOT, "results", "exC1_rho_paper_ci.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)

    for r in rows:
        ci = (f"  95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
              if r["ci_lo"] != "" else "")
        print(f"[rho-boot] {r['metric']:12s} = {r['value']:+.4f}{ci}")


if __name__ == "__main__":
    main()
