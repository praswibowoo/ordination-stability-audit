#!/usr/bin/env python3
"""experiment_F_absdrift.py — absolute drift for family-restricted vs all-parameter fits.

Computes absolute Procrustes drift for each family under the paired LOPO design
(same rows, family columns vs all 44 columns) using pure Procrustes drift from
scipy.spatial.procrustes (which is already scale-free on unit Frobenius norm clouds).
Records per-event medians, means, and unique-configuration drift across events.

Reads frozen layers only. Writes results/exC2_absolute_drift.csv.
Run: python3 code/experiment_F_absdrift.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes as sp_procrustes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import load  # noqa: E402
from experiment_B_familymaps import FAMILIES  # noqa: E402


def fit(X: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=SEED).fit_transform(X)


def family_absdrift(X, rho, paper, params, family, fam_params) -> list[dict]:
    cols = [params.index(pt) for pt in fam_params]
    cover = X[:, cols].sum(axis=1) >= 1
    idxs = np.where(cover)[0]
    rho_f = rho[idxs]
    strata = paper[idxs]
    X_all = X[np.ix_(idxs, range(len(params)))]
    X_fam = X[np.ix_(idxs, cols)]
    q1, q3 = np.quantile(rho_f, [0.25, 0.75])

    rows = []
    for p in sorted(set(strata)):
        keep = strata != p
        if keep.sum() < 8:
            continue
        Sa = fit(X_all)
        Sfa = fit(X_fam)
        Sa_r = fit(X_all[keep])
        Sf_r = fit(X_fam[keep])
        a1, b1, _ = sp_procrustes(Sa[keep], Sa_r)
        a2, b2, _ = sp_procrustes(Sfa[keep], Sf_r)
        da = np.linalg.norm(b1 - a1, axis=1)
        df = np.linalg.norm(b2 - a2, axis=1)
        r = rho_f[keep]
        da_med = float(np.median(da))
        df_med = float(np.median(df))
        da_mean = float(np.mean(da))
        df_mean = float(np.mean(df))

        keep_cfgs, keep_inv = np.unique(X_fam[keep], axis=0, return_inverse=True)
        cfg_drifts = [float(np.median(df[keep_inv == c])) for c in range(len(keep_cfgs))]
        df_cfg_med = float(np.median(cfg_drifts))

        rows.append(dict(
            family=family, deleted=p, n_keep=int(keep.sum()),
            all_med=da_med,
            all_mean=da_mean,
            all_sparse=float(np.median(da[r <= q1])),
            all_dense=float(np.median(da[r >= q3])),
            fam_med=df_med,
            fam_mean=df_mean,
            fam_cfg_med=df_cfg_med,
            fam_sparse=float(np.median(df[r <= q1])),
            fam_dense=float(np.median(df[r >= q3]))))
    return rows


def main() -> None:
    X, rho, paper, params, *_ = load()
    all_rows = []
    for family, fam_params in FAMILIES.items():
        rows = family_absdrift(X, rho, paper, params, family, fam_params)
        agg = {k: float(np.median([r[k] for r in rows]))
               for k in ("all_med", "all_sparse", "all_dense",
                         "fam_med", "fam_sparse", "fam_dense")}
        ratio = agg["all_med"] / agg["fam_med"] if agg["fam_med"] > 0 else float("inf")
        print(f"[absdrift] {family:14s} events={len(rows):3d} | all {agg['all_med']:.2e} "
              f"| family {agg['fam_med']:.2e} | ratio {ratio:.0f}x")
        all_rows += rows

    out = os.path.join(RESULTS, "exC2_absolute_drift.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    os.replace(out + ".tmp", out)
    print(f"[absdrift] wrote {os.path.relpath(out, ROOT)} ({len(all_rows)} event rows, 7 families)")


if __name__ == "__main__":
    main()
