#!/usr/bin/env python3
"""curated_composition_extra.py — direct contrast and exact subset on 716/117."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results", "wp3_clean")
SEED = 42
from composition_matched_control import cluster_boot_block  # noqa: E402


def main() -> None:
    comp = pd.read_csv(os.path.join(RESULTS, "composition_matched_control.csv"))
    a = (comp[comp["arm"] == "composition_matched"]
         .groupby("paper")["med_drift"].median())
    b = (comp[comp["arm"] == "density_only"]
         .groupby("paper")["med_drift"].median())
    papers = sorted(set(a.index) & set(b.index))
    a = a.loc[papers].to_numpy(float)
    b = b.loc[papers].to_numpy(float)
    d = np.log(a) - np.log(b)
    rng = np.random.default_rng(SEED)
    boot = np.empty(10000)
    for i in range(10000):
        idx = rng.integers(0, len(d), len(d))
        boot[i] = np.median(d[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    out = pd.DataFrame([dict(
        n=len(d), median_log_diff=float(np.median(d)),
        ci_lo=float(lo), ci_hi=float(hi),
        wilcoxon_p=float(stats.wilcoxon(d).pvalue),
        n_comp_gt_density=int(np.sum(d > 0)),
        estimand="log(composition_matched) - log(density_only)",
        sign="positive = composition arm drifts more than density-only",
    )])
    out.to_csv(os.path.join(RESULTS, "composition_vs_density.csv"),
               index=False, float_format="%.6g")
    print(out.to_string(index=False))

    exact = (comp[comp["arm"] == "composition_matched"]
             .groupby("paper")["exact_fraction"].mean())
    exact_papers = sorted(exact[exact > 0].index)
    block = pd.read_csv(os.path.join(RESULTS, "coverage_events.csv"))
    block = block[(block["arm"] == "block")
                  & (block["paper"].isin(exact_papers))]
    sub = comp[(comp["arm"] == "composition_matched")
               & (comp["paper"].isin(exact_papers))]
    ev = pd.concat([
        block.assign(arm="block")[["paper", "med_drift", "n_rows_deleted", "arm"]],
        sub[["paper", "med_drift", "n_rows_deleted", "arm"]],
    ], ignore_index=True)
    bf, lo, hi = cluster_boot_block(ev, n_boot=2000)
    sub_out = pd.DataFrame([dict(
        n_papers=len(exact_papers), n_events=len(ev),
        block_factor=bf, block_ci_lo=lo, block_ci_hi=hi,
        mean_exact_fraction=float(exact.loc[exact_papers].mean()),
        definition="studies with at least one exact configuration match",
    )])
    sub_out.to_csv(os.path.join(RESULTS, "composition_exact_subset.csv"),
                   index=False, float_format="%.6g")
    print(sub_out.to_string(index=False))


if __name__ == "__main__":
    main()
