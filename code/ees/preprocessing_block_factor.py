#!/usr/bin/env python3
"""preprocessing_block_factor.py — fixed-preprocessing block factor.

The per-paper file results/preprocessing_sensitivity.csv contains block and
matched-first drift under fixed full-data centring (and z-scoring for the value
layer). This script fits the same log-linear block model to those per-paper
events and reports the fixed-preprocessing block factor with a paper-level
cluster bootstrap.

Writes results/preprocessing_block_factor.csv.

Run: python3 code/ees/preprocessing_block_factor.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
SEED = 42


def fit(df: pd.DataFrame, seed: int = SEED, n_boot: int = 2000):
    ev = []
    for _, r in df.iterrows():
        ev.append(dict(paper=r["paper"], y=np.log(r["block_drift"]),
                       v=np.log(r["n_rows"]), block=1))
        ev.append(dict(paper=r["paper"], y=np.log(r["matched_drift"]),
                       v=np.log(r["n_rows"]), block=0))
    ev = pd.DataFrame(ev)
    X = np.column_stack([np.ones(len(ev)), ev["v"].to_numpy(float),
                         ev["block"].to_numpy(float)])
    y = ev["y"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    uniq = np.asarray(sorted(set(ev["paper"])), dtype=object)
    gid = pd.Categorical(ev["paper"], categories=uniq).codes
    idx = {j: np.where(gid == j)[0] for j in range(len(uniq))}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[j] for j in pick])
        boots.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0])
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots[:, 2], [0.025, 0.975])
    return dict(n_papers=len(df), n_events=len(ev),
                block_factor=float(np.exp(beta[2])),
                block_ci_lo=float(np.exp(lo)),
                block_ci_hi=float(np.exp(hi)),
                beta_block=float(beta[2]))


def main() -> None:
    d = pd.read_csv(os.path.join(RESULTS, "preprocessing_sensitivity.csv"))
    out = []
    for v in ("presence_pca", "std_value_pca"):
        r = fit(d[d["variant"] == v])
        r["layer"] = v
        out.append(r)
    df = pd.DataFrame(out)[["layer", "n_papers", "n_events", "block_factor",
                            "block_ci_lo", "block_ci_hi", "beta_block"]]
    df.to_csv(os.path.join(RESULTS, "preprocessing_block_factor.csv"),
              index=False, float_format="%.6g")
    print(df.to_string(index=False))
    print("\nwrote preprocessing_block_factor.csv")


if __name__ == "__main__":
    main()
