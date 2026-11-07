#!/usr/bin/env python3
"""study_level_adjusted.py — study-size-adjusted own-deletion density check.

The own-deletion SDR is a marginal association.  This script reports a
study-level regression of log(median remaining-row drift) on study mean
coverage density, log(rows deleted), and configuration rarity, plus the SDR
within small/large study strata.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "core"))
RESULTS = os.path.join(ROOT, "results")

import experiment_C1 as c1  # noqa: E402


def sdr(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    hi = np.median(drift[rho >= q3])
    return float(np.median(drift[rho <= q1]) / hi) if hi > 0 else float("nan")


def main() -> None:
    X, rho, paper, params, *_ = c1.load_primary()
    paper = np.asarray(paper)
    cfg = [bytes(X[i].astype(np.uint8)) for i in range(len(X))]
    freq = Counter(cfg)

    block = {}
    with open(os.path.join(RESULTS, "exH_volumematch.csv"), newline="",
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] == "lopo":
                block[r["paper"]] = float(r["med_drift"])

    rows = []
    for p in sorted(set(paper)):
        idx = np.where(paper == p)[0]
        rows.append(dict(
            paper=p,
            density=float(np.mean(rho[idx])),
            log_drift=math.log(block[p]),
            n_rows=len(idx),
            n_configs=len({cfg[i] for i in idx}),
            rarity=float(np.mean([-math.log(freq[cfg[i]] / len(X)) for i in idx])),
        ))

    y = np.array([r["log_drift"] for r in rows])
    density = np.array([r["density"] for r in rows])
    log_rows = np.log(np.array([r["n_rows"] for r in rows], float))
    rarity = np.array([r["rarity"] for r in rows])
    Xd = np.column_stack([np.ones(len(y)), density, log_rows, rarity])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    meat = (Xd * resid[:, None]).T @ (Xd * resid[:, None])
    n, k = Xd.shape
    cov = xtx_inv @ meat @ xtx_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    names = ["intercept", "density", "log_n_rows", "config_rarity"]
    out = []
    for j, name in enumerate(names):
        t = beta[j] / se[j]
        p = 2 * stats.t.sf(abs(t), n - k)
        out.append(dict(term=name, coef=beta[j], se=se[j], p=p, n=n))

    med_rows = float(np.median(log_rows))
    small = [r for r in rows if r["n_rows"] <= np.median([x["n_rows"] for x in rows])]
    large = [r for r in rows if r["n_rows"] > np.median([x["n_rows"] for x in rows])]
    strata = []
    for label, sub in (("small_study", small), ("large_study", large)):
        strata.append(dict(
            term=f"sdr_{label}",
            coef=sdr(np.array([r["density"] for r in sub]),
                     np.array([math.exp(r["log_drift"]) for r in sub])),
            se="", p="", n=len(sub),
        ))

    path = os.path.join(RESULTS, "study_level_adjusted.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term", "coef", "se", "p", "n"])
        w.writeheader()
        w.writerows(out + strata)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    for r in out + strata:
        print(r)


if __name__ == "__main__":
    main()
