#!/usr/bin/env python3
"""experiment_H_volumematch_value.py — value-layer replication of the volume-matched control.

The presence-layer block model is reported in Section 4.2 of the EES manuscript.
This script runs the same paired design on the standardized value PCA (four F1
physical parameters, 421 rows, 70 papers), so the block effect is not a
presence-layer/reporting artifact.

Reads the frozen layers through experiment_C1.load(); writes:
  results/exH_value_volumematch.csv
  results/exH_value_loglinear_model.csv
  results/exH_value_summary.csv

Run: python3 code/experiment_H_volumematch_value.py [--draws 100]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import (load, ordinate, knn_idx, knn_churn,
                           drift_after_procrustes)  # noqa: E402
from experiment_H_volumematch import ols_cluster_boot  # noqa: E402


def event_stats(s_full, knn_full, X, rho, keep):
    """Median drift and churn for one deletion event."""
    S_red = ordinate("pca_vstd", X[keep], X[keep])
    drift, _ = drift_after_procrustes(s_full[keep], S_red)
    ki = np.where(keep)[0]
    ch = knn_churn(knn_full[keep], ki[knn_idx(S_red)])
    return dict(med_drift=float(np.median(drift)),
                med_churn=float(np.median(ch))), drift


def summarise(events, fc, mc):
    return (float(np.median([e[fc] for e in events])),
            float(np.median([e[mc] for e in events])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    n_draws = max(1, args.draws)

    _X, _rho, _paper, _params, Xv, keysv, rho_v = load()
    paper = np.asarray([k[0] for k in keysv], dtype=object)
    n = len(Xv)
    rho = np.asarray(rho_v, float)
    uniq = sorted(set(paper))
    print(f"value {Xv.shape} | {len(uniq)} papers | {n} rows | "
          f"rho NaN {int(np.isnan(rho).sum())}")

    # Rows with a missing presence-density link are retained in the ordination
    # but cannot enter the density quartiles; the block model does not use rho.
    S_full = ordinate("pca_vstd", Xv, Xv)
    knn_full = knn_idx(S_full)

    lopo, matched = [], []
    for p in uniq:
        mask_p = paper == p
        n_p = int(mask_p.sum())
        keep = ~mask_p
        st, _ = event_stats(S_full, knn_full, Xv, rho, keep)
        st.update(paper=p, arm="lopo", n_rows_deleted=n_p, overlap_with_paper=n_p)
        lopo.append(st)

        rng = np.random.default_rng(SEED + zlib.crc32(str(p).encode()) % 100000)
        drop = rng.choice(n, size=n_p, replace=False)
        keep2 = np.ones(n, bool)
        keep2[drop] = False
        st2, _ = event_stats(S_full, knn_full, Xv, rho, keep2)
        st2.update(paper=p, arm="matched_scattered", n_rows_deleted=n_p,
                   overlap_with_paper=int(np.sum(mask_p[drop])), draw=0)
        matched.append(st2)

        for dr in range(n_draws - 1):
            rng_d = np.random.default_rng(
                SEED + zlib.crc32(str(p).encode()) % 100000 + 7919 * (dr + 1))
            drop_d = rng_d.choice(n, size=n_p, replace=False)
            keep_d = np.ones(n, bool)
            keep_d[drop_d] = False
            st_d, _ = event_stats(S_full, knn_full, Xv, rho, keep_d)
            st_d.update(paper=p, arm="matched_scattered", n_rows_deleted=n_p,
                        overlap_with_paper=int(np.sum(mask_p[drop_d])),
                        draw=dr + 1)
            matched.append(st_d)

    matched_first = [e for e in matched if e.get("draw") == 0]
    md_l, mc_l = summarise(lopo, "med_drift", "med_churn")
    md_m, mc_m = summarise(matched_first, "med_drift", "med_churn")

    path = os.path.join(RESULTS, "exH_value_volumematch.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lopo[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(lopo)
        w.writerows(matched_first)
    os.replace(path + ".tmp", path)

    path_d = os.path.join(RESULTS, "exH_value_volumematch_draws.csv")
    with open(path_d + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matched[0].keys()))
        w.writeheader()
        w.writerows(matched)
    os.replace(path_d + ".tmp", path_d)

    # log-linear model on all events, paper-clustered bootstrap
    ev_all = lopo + matched
    yv = np.log(np.asarray([e["med_drift"] for e in ev_all], float))
    vv = np.log(np.asarray([e["n_rows_deleted"] for e in ev_all], float))
    blk = np.asarray([1.0 if e["arm"] == "lopo" else 0.0 for e in ev_all])
    grp = np.asarray([e["paper"] for e in ev_all], dtype=object)
    Xd = np.column_stack([np.ones(len(ev_all)), vv, blk])
    beta, lo, hi, n_ok, boots = ols_cluster_boot(Xd, yv, grp,
                                                 n_boot=args.bootstrap)
    with open(os.path.join(RESULTS, "exH_value_loglinear_model.csv") + ".tmp",
              "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["term", "coef", "ci_lo", "ci_hi", "mult_factor",
                    "mult_lo", "mult_hi", "boot_p_two_sided", "n_boot", "n_obs"])
        for j, name in enumerate(("intercept", "log_volume", "block")):
            b, l, h = beta[j], lo[j], hi[j]
            col = boots[:, j]
            p_boot = 2.0 * min(float(np.mean(col <= 0.0)),
                               float(np.mean(col >= 0.0)))
            w.writerow([name, round(float(b), 4), round(float(l), 4),
                        round(float(h), 4), round(float(np.exp(b)), 4),
                        round(float(np.exp(l)), 4), round(float(np.exp(h)), 4),
                        round(p_boot, 6), n_ok, len(ev_all)])
    os.replace(os.path.join(RESULTS, "exH_value_loglinear_model.csv") + ".tmp",
               os.path.join(RESULTS, "exH_value_loglinear_model.csv"))

    d_l = np.array([e["med_drift"] for e in lopo])
    d_m = np.array([e["med_drift"] for e in matched_first])
    paired = d_l / d_m
    ratio_of_meds = float(np.median(d_l) / np.median(d_m))
    med_paired = float(np.median(paired))
    boot_rng = np.random.default_rng(SEED)
    n_ev = len(d_l)
    boot_r = np.empty(10000)
    boot_p = np.empty(10000)
    for b in range(10000):
        idx = boot_rng.choice(n_ev, size=n_ev, replace=True)
        boot_r[b] = np.median(d_l[idx]) / np.median(d_m[idx])
        boot_p[b] = np.median(paired[idx])
    lo_r, hi_r = np.quantile(boot_r, [0.025, 0.975])
    lo_p, hi_p = np.quantile(boot_p, [0.025, 0.975])
    w_res = stats.wilcoxon(d_l, d_m)
    greater = int(np.sum(d_l > d_m))

    summary = [dict(
        d_lopo_med=md_l, d_matched_med=md_m,
        ratio_of_medians=ratio_of_meds,
        ratio_of_medians_ci_lo=float(lo_r),
        ratio_of_medians_ci_hi=float(hi_r),
        median_paired_ratio=med_paired,
        median_paired_ratio_ci_lo=float(lo_p),
        median_paired_ratio_ci_hi=float(hi_p),
        wilcoxon_stat=float(w_res.statistic), wilcoxon_p=float(w_res.pvalue),
        n_block_gt_scattered=greater, n_total=n_ev,
        pct_block_gt_scattered=100.0 * greater / n_ev,
        churn_lopo_med=mc_l, churn_matched_med=mc_m,
        n_draws=n_draws,
        n_rows_deleted_lopo=int(np.median([e["n_rows_deleted"] for e in lopo])),
        n_rows_deleted_matched=int(np.median([e["n_rows_deleted"] for e in matched_first])),
    )]
    with open(os.path.join(RESULTS, "exH_value_summary.csv") + ".tmp", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    os.replace(os.path.join(RESULTS, "exH_value_summary.csv") + ".tmp",
               os.path.join(RESULTS, "exH_value_summary.csv"))

    print("\nvalue-layer per-event medians")
    print(f"  LOPO block deletion      : drift {md_l:.3e}  churn {mc_l:.2f}")
    print(f"  volume-matched scattered : drift {md_m:.3e}  churn {mc_m:.2f}")
    print(f"  ratio of medians         : {ratio_of_meds:.2f} "
          f"(95% CI {lo_r:.2f}-{hi_r:.2f})")
    print(f"  median paired ratio      : {med_paired:.2f} "
          f"(95% CI {lo_p:.2f}-{hi_p:.2f})")
    print(f"  Wilcoxon                 : p = {w_res.pvalue:.2e} "
          f"({greater}/{n_ev} block > scattered)")
    print("\nvalue-layer log-linear model")
    for name, b, l, h in (("intercept", beta[0], lo[0], hi[0]),
                          ("log_volume", beta[1], lo[1], hi[1]),
                          ("block", beta[2], lo[2], hi[2])):
        print(f"  {name:12s} beta={b:+.4f} CI [{l:+.4f}, {h:+.4f}] "
              f"x{np.exp(b):.3f} [{np.exp(l):.3f}, {np.exp(h):.3f}]")


if __name__ == "__main__":
    main()
