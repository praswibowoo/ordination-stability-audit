#!/usr/bin/env python3
"""block_model_diagnostics.py — block x volume interaction and residual checks.

The manuscript's primary volume-matched model is
    log(drift) = b0 + bV log(V) + bblock * block + e.
This script asks whether the block factor is constant across deleted volumes,
and records standard residual diagnostics.

Outputs
  results/block_model_diagnostics.csv

Run: python3 code/core/block_model_diagnostics.py
"""
from __future__ import annotations

import csv
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import numpy as np
import pandas as pd
from scipy.stats import chi2

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
SEED = 42
N_BOOT = 2000


def load_events():
    lopo = pd.read_csv(os.path.join(RESULTS, "exH_volumematch.csv"))
    lopo = lopo[lopo["arm"] == "lopo"]
    draws = pd.read_csv(os.path.join(RESULTS, "exH_volumematch_draws.csv"))
    ev = pd.concat([lopo, draws], ignore_index=True)
    y = np.log(ev["med_drift"].to_numpy(float))
    v = np.log(ev["n_rows_deleted"].to_numpy(float))
    block = (ev["arm"] == "lopo").to_numpy(float)
    groups = ev["paper"].to_numpy()
    return y, v, block, groups


def fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def cluster_boot(X, y, groups, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.asarray(sorted(set(groups)), dtype=object)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[j]] for j in pick])
        try:
            out.append(fit(X[sel], y[sel]))
        except np.linalg.LinAlgError:
            continue
    return np.asarray(out)


def main():
    y, v, block, groups = load_events()
    X = np.column_stack([np.ones(len(y)), v, block, v * block])
    beta = fit(X, y)
    boots = cluster_boot(X, y, groups)

    def summarise(j, name):
        col = boots[:, j]
        lo, hi = np.quantile(col, [0.025, 0.975])
        p = 2.0 * min(np.mean(col <= 0), np.mean(col >= 0))
        return dict(term=name, coef=f"{beta[j]:.6f}", ci_lo=f"{lo:.6f}",
                    ci_hi=f"{hi:.6f}", boot_p=f"{p:.4f}")

    # no-interaction reference
    X0 = np.column_stack([np.ones(len(y)), v, block])
    b0 = fit(X0, y)
    boots0 = cluster_boot(X0, y, groups)
    block_lo, block_hi = np.quantile(boots0[:, 2], [0.025, 0.975])

    fitted = X @ beta
    resid = y - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    n, k = X.shape
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    xtx_inv = np.linalg.inv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    mse = ss_res / (n - k)
    cooks = (resid ** 2 / (k * mse)) * (leverage / (1.0 - leverage) ** 2)
    # Breusch-Pagan LM: regress squared residuals on the design matrix.
    aux_beta, *_ = np.linalg.lstsq(X, resid ** 2, rcond=None)
    aux_resid = resid ** 2 - X @ aux_beta
    aux_r2 = 1.0 - np.sum(aux_resid ** 2) / np.sum((resid ** 2 - np.mean(resid ** 2)) ** 2)
    bp_p = float(chi2.sf(n * aux_r2, df=k - 1))
    resid_fit = np.corrcoef(fitted, resid)[0, 1]

    rows = [
        summarise(0, "intercept"),
        summarise(1, "log_volume"),
        summarise(2, "block"),
        summarise(3, "log_volume_x_block"),
        dict(term="block_no_interaction_factor",
             coef=f"{np.exp(b0[2]):.4f}",
             ci_lo=f"{np.exp(block_lo):.4f}", ci_hi=f"{np.exp(block_hi):.4f}",
             boot_p=""),
        dict(term="block_factor_median5",
             coef=f"{np.exp(beta[2] + beta[3] * np.log(5)):.4f}",
             ci_lo="", ci_hi="", boot_p=""),
        dict(term="block_factor_q95",
             coef=f"{np.exp(beta[2] + beta[3] * np.log(14)):.4f}",
             ci_lo="", ci_hi="", boot_p=""),
        dict(term="r2", coef=f"{r2:.4f}", ci_lo="", ci_hi="", boot_p=""),
        dict(term="adj_r2", coef=f"{adj_r2:.4f}", ci_lo="", ci_hi="", boot_p=""),
        dict(term="resid_sd", coef=f"{np.std(resid, ddof=4):.4f}", ci_lo="", ci_hi="", boot_p=""),
        dict(term="breusch_pagan_p", coef=f"{bp_p:.4f}", ci_lo="", ci_hi="", boot_p=""),
        dict(term="max_cooks_d", coef=f"{np.max(cooks):.4f}", ci_lo="", ci_hi="", boot_p=""),
        dict(term="resid_fitted_corr", coef=f"{resid_fit:.4f}", ci_lo="", ci_hi="", boot_p=""),
    ]
    out = os.path.join(RESULTS, "block_model_diagnostics.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["term", "coef", "ci_lo", "ci_hi", "boot_p"])
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".tmp", out)
    print(f"[block-diag] interaction coef {beta[3]:.4f} "
          f"({rows[3]['ci_lo']}, {rows[3]['ci_hi']}); "
          f"block factor (no interaction) {np.exp(b0[2]):.3f}; "
          f"R2 {r2:.3f}")


if __name__ == "__main__":
    main()
