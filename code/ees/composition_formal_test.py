#!/usr/bin/env python3
"""composition_formal_test.py — one reproducible composition-vs-volume test.

Primary estimand
----------------
Median paired log difference

    d_p = log(composition_matched_drift_p) - log(volume_matched_drift_p)

with the first matched draw for each arm (draw 0). Positive values mean the
composition-matched control drifts more than the volume-matched control, i.e.
composition matching attenuates the block factor. The 95% interval is a
paper-level bootstrap; the Wilcoxon and paired t tests are reported for
completeness.

Secondary (reported, not the primary claim):
  * median log ratio = median(log(composition)) - median(log(volume));
  * log ratio of medians = log(median(composition) / median(volume));
  * the same primary estimand using each paper's median across draws.

Reads: results/exH_volumematch_draws.csv
       results/composition_matched_control.csv
Writes: results/composition_formal_test.csv

Run: python3 code/composition_formal_test.py [--draw 0] [--median-draws]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
SEED = 42


def estimate(pairs: pd.DataFrame, rng, n_boot: int = 10000):
    d = (np.log(pairs["comp_drift"].to_numpy(float))
         - np.log(pairs["vol_drift"].to_numpy(float)))
    med = float(np.median(d))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        boot[b] = np.median(d[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(
        n=len(d),
        median_log_diff=med,
        ci_lo=float(lo),
        ci_hi=float(hi),
        wilcoxon_p=float(stats.wilcoxon(d).pvalue),
        paired_t_p=float(stats.ttest_rel(
            np.log(pairs["comp_drift"].to_numpy(float)),
            np.log(pairs["vol_drift"].to_numpy(float))).pvalue),
        n_comp_gt_vol=int(np.sum(d > 0)),
        median_log_ratio=float(np.median(np.log(pairs["comp_drift"]))
                               - np.median(np.log(pairs["vol_drift"]))),
        log_ratio_of_medians=float(np.log(np.median(pairs["comp_drift"])
                                          / np.median(pairs["vol_drift"]))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-csv", default=os.path.join(
        RESULTS, "exH_volumematch_draws.csv"))
    ap.add_argument("--comp-csv", default=os.path.join(
        RESULTS, "composition_matched_control.csv"))
    ap.add_argument("--draw", type=int, default=0,
                    help="matched draw used for the primary estimate")
    ap.add_argument("--median-draws", action="store_true",
                    help="use each paper's median across draws instead of one draw")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(
        RESULTS, "composition_formal_test.csv"))
    args = ap.parse_args()

    vol = pd.read_csv(args.vol_csv)
    vol = vol[vol["arm"] == "matched_scattered"]
    comp = pd.read_csv(args.comp_csv)
    comp = comp[comp["arm"] == "composition_matched"]

    if args.median_draws:
        vol = vol.groupby("paper", as_index=False).agg(
            vol_drift=("med_drift", "median"))
        comp = comp.groupby("paper", as_index=False).agg(
            comp_drift=("med_drift", "median"))
    else:
        vol = vol[vol["draw"] == args.draw][["paper", "med_drift"]].rename(
            columns={"med_drift": "vol_drift"})
        comp = comp[comp["draw"] == args.draw][["paper", "med_drift"]].rename(
            columns={"med_drift": "comp_drift"})

    pairs = vol.merge(comp, on="paper", how="inner").dropna()
    if len(pairs) != vol["paper"].nunique() or len(pairs) != comp["paper"].nunique():
        raise ValueError("volume and composition arms do not cover the same papers")
    if (pairs[["vol_drift", "comp_drift"]] <= 0).any().any():
        raise ValueError("drifts must be positive for the log contrast")

    rng = np.random.default_rng(SEED)
    result = estimate(pairs, rng, n_boot=args.bootstrap)
    result.update(
        estimand="median_paired_log_difference",
        contrast="log(composition_matched_drift) - log(volume_matched_drift)",
        draw=("median" if args.median_draws else args.draw),
        sign="positive = composition matching attenuates the block factor",
    )
    out = pd.DataFrame([result])
    path = args.out
    out.to_csv(path, index=False, float_format="%.6g")
    print(out.to_string(index=False))
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
