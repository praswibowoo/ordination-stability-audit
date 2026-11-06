#!/usr/bin/env python3
"""joint_2d3d_ratio.py — paired 2-D vs 3-D block-factor contrast.

Fits the same log-linear deletion model in 2-D and 3-D,

    log(drift) ~ 1 + log(n_rows_deleted) + block,

and bootstraps whole studies jointly so that the 2-D and 3-D estimates are
paired on the same resampled studies.  The reported quantity is
exp(beta_block_2d - beta_block_3d), i.e. how much larger the 2-D block factor
is than the 3-D block factor.

Reads:
  results/exH_volumematch.csv          (2-D block + first matched draw)
  results/exH_volumematch_draws.csv    (2-D all matched draws)
  results/exH_3d_presence.csv          (3-D block + matched draws)

Writes:
  results/joint_2d3d_ratio.csv

Run: python3 code/joint_2d3d_ratio.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
SEED = 42
N_BOOT = 2000


def load_2d() -> pd.DataFrame:
    block = pd.read_csv(os.path.join(RESULTS, "exH_volumematch.csv"))
    block = block[block["arm"] == "lopo"][["paper", "med_drift", "n_rows_deleted"]]
    block = block.rename(columns={"med_drift": "drift"}).assign(block=1)
    draws = pd.read_csv(os.path.join(RESULTS, "exH_volumematch_draws.csv"))
    draws = draws[["paper", "med_drift", "n_rows_deleted"]]
    draws = draws.rename(columns={"med_drift": "drift"}).assign(block=0)
    return pd.concat([block, draws], ignore_index=True)


def load_3d() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS, "exH_3d_presence.csv"))
    df = df.rename(columns={"n_rows": "n_rows_deleted"})
    df["block"] = (df["arm"] == "block").astype(int)
    return df[["paper", "drift", "n_rows_deleted", "block"]]


def fit_block(d: pd.DataFrame) -> float:
    y = np.log(d["drift"].to_numpy(float))
    X = np.column_stack([
        np.ones(len(d)),
        np.log(d["n_rows_deleted"].to_numpy(float)),
        d["block"].to_numpy(float),
    ])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[2])


def main() -> None:
    d2 = load_2d()
    d3 = load_3d()
    papers = np.array(sorted(set(d2["paper"]) & set(d3["paper"])))
    idx2 = {p: d2.index[d2["paper"] == p].to_numpy() for p in papers}
    idx3 = {p: d3.index[d3["paper"] == p].to_numpy() for p in papers}

    b2 = fit_block(d2)
    b3 = fit_block(d3)
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(papers, size=len(papers), replace=True)
        dd2 = pd.concat([d2.loc[idx2[p]] for p in pick], ignore_index=True)
        dd3 = pd.concat([d3.loc[idx3[p]] for p in pick], ignore_index=True)
        boot[b] = fit_block(dd2) - fit_block(dd3)

    lo, hi = np.quantile(boot, [0.025, 0.975])
    p2 = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    out = pd.DataFrame([dict(
        n_papers=len(papers),
        block_factor_2d=float(np.exp(b2)),
        block_factor_3d=float(np.exp(b3)),
        ratio_2d_3d=float(np.exp(b2 - b3)),
        ratio_ci_lo=float(np.exp(lo)),
        ratio_ci_hi=float(np.exp(hi)),
        boot_p_two_sided=float(min(1.0, p2)),
        n_boot=N_BOOT,
    )])
    path = os.path.join(RESULTS, "joint_2d3d_ratio.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    print(out.to_string(index=False))
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
