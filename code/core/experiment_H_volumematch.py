#!/usr/bin/env python3
"""experiment_H_volumematch.py — volume-matched row-deletion control (reviewer R2; Phase F2).

The LOPO-vs-LORO contrast is not volume-matched: deleting a paper removes many rows,
deleting one row removes one. This control deletes, for every paper p, a uniformly
random subset of rows of the SAME SIZE as p's row block, then recomputes presence-PCA
drift and kNN churn under the identical Procrustes protocol.

Both arms are summarised with the IDENTICAL estimator: per-event median drift and
per-event median churn fraction (chi/2), then the median across the 121/122 events.
This removes the pooled-vs-per-event metric mismatch in the released summary.

Reads the frozen presence layer via experiment_C1.load() (no writes to frozen outputs).
Writes: results/exH_volumematch.csv

Phase F2 adds:
  * N_DRAWS scattered control draws per paper (was a single draw), so the scattered
    arm's uncertainty is quantified rather than sampled once;
  * a fitted log-linear model  log(delta) ~ log(V) + block  with PAPER-LEVEL
    cluster bootstrap CIs, written to results/exH_loglinear_model.csv. Table S10's
    note previously carried beta_block = 0.333, beta_logV = 0.844 and p = 6.63e-6
    as literal strings with no script behind them.

Run: python3 code/core/experiment_H_volumematch.py [--draws 100]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib

import numpy as np

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import (load, ordinate, knn_idx, knn_churn,
                           drift_after_procrustes)  # noqa: E402


def swing(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    lo = float(np.median(drift[rho <= q1]))
    hi = float(np.median(drift[rho >= q3]))
    return (lo / hi if hi > 0 else float("nan")), lo, hi


def event_stats(s_full, knn_full, X, rho, keep):
    """Identical estimator for both arms: median drift, median churn fraction, SI."""
    S_red = ordinate("pca_pres", X[keep])
    drift, _ = drift_after_procrustes(s_full[keep], S_red)
    ki = np.where(keep)[0]
    ch = knn_churn(knn_full[keep], ki[knn_idx(S_red)])
    si, lo, hi = swing(rho[keep], drift)
    return dict(med_drift=float(np.median(drift)), med_churn=float(np.median(ch)),
                SI=si, med_lo=lo, med_hi=hi,
                n_rows=int(keep.sum()), n_kept=int(keep.sum())), rho[keep], drift


def ols_cluster_boot(Xd, y, groups, n_boot=2000, seed=SEED):
    """OLS with PAPER-LEVEL cluster bootstrap CIs.

    Xd     (n, k) design matrix including an intercept column
    y      (n,)   response
    groups (n,)   paper identifier of each observation

    Returns (coef, lo, hi) with percentile CIs from resampling whole papers,
    which is the correct unit here because each paper contributes one LOPO event
    and N_DRAWS scattered events.
    """
    def fit(Xm, ym):
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        return beta
    beta = fit(Xd, y)
    uniq = np.asarray(sorted(set(groups)), dtype=object)
    gid = np.searchsorted(uniq, groups)
    idx_by = {j: np.where(gid == j)[0] for j in range(len(uniq))}
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, Xd.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        # Draw whole papers WITH replacement: concatenate each sampled paper's
        # rows, including duplicates. Do not use np.isin here: it de-duplicates
        # the draw and turns the cluster bootstrap into a subsample, which does
        # not reproduce the deposited interval.
        sel = np.concatenate([idx_by[j] for j in pick])
        try:
            boots[b] = fit(Xd[sel], y[sel])
        except np.linalg.LinAlgError:
            boots[b] = np.nan
    ok = ~np.isnan(boots).any(axis=1)
    lo = np.nanpercentile(boots[ok], 2.5, axis=0)
    hi = np.nanpercentile(boots[ok], 97.5, axis=0)
    return beta, lo, hi, int(ok.sum()), boots[ok]


def summarise(events, fc, mc, sc):
    md = float(np.median([e[fc] for e in events]))
    mc_ = float(np.median([e[mc] for e in events]))
    si = float(np.median([e[sc] for e in events if e[sc] == e[sc]]))
    return md, mc_, si


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100,
                    help="scattered control draws per paper (Phase F2; was 1)")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="paper-level cluster bootstrap resamples for the log-linear CIs")
    args = ap.parse_args()
    n_draws = max(1, args.draws)

    X, rho, paper, params, _Xv, _keysv, _rho_v = load()
    n = len(X)
    uniq = sorted(set(paper))
    print(f"presence {X.shape} | {len(uniq)} papers | {n} rows")

    S_full = ordinate("pca_pres", X)
    knn_full = knn_idx(S_full)

    lopo, matched = [], []
    pool = {"lopo": ([], []), "matched_scattered": ([], [])}
    for p in uniq:
        mask_p = paper == p
        n_p = int(mask_p.sum())
        # arm 1 — block deletion (LOPO)
        keep = ~mask_p
        st, r_, d_ = event_stats(S_full, knn_full, X, rho, keep)
        st.update(paper=p, arm="lopo", n_rows_deleted=n_p, overlap_with_paper=n_p)
        lopo.append(st)
        pool["lopo"][0].append(r_); pool["lopo"][1].append(d_)
        # arm 2 — volume-matched scattered deletion
        rng = np.random.default_rng(SEED + zlib.crc32(p.encode()) % 100000)
        drop = rng.choice(n, size=n_p, replace=False)
        keep2 = np.ones(n, bool)
        keep2[drop] = False
        st2, r2_, d2_ = event_stats(S_full, knn_full, X, rho, keep2)
        st2.update(paper=p, arm="matched_scattered", n_rows_deleted=n_p,
                   overlap_with_paper=int(np.sum(mask_p[drop])), draw=0)
        matched.append(st2)
        pool["matched_scattered"][0].append(r2_); pool["matched_scattered"][1].append(d2_)
        # Phase F2: N_DRAWS independent scattered draws for the same paper so the
        # scattered arm's own variability is quantified rather than sampled once.
        for dr in range(n_draws - 1):
            rng_d = np.random.default_rng(
                SEED + zlib.crc32(p.encode()) % 100000 + 7919 * (dr + 1))
            drop_d = rng_d.choice(n, size=n_p, replace=False)
            keep_d = np.ones(n, bool)
            keep_d[drop_d] = False
            st_d, r_d, d_d = event_stats(S_full, knn_full, X, rho, keep_d)
            st_d.update(paper=p, arm="matched_scattered", n_rows_deleted=n_p,
                        overlap_with_paper=int(np.sum(mask_p[drop_d])), draw=dr + 1)
            matched.append(st_d)

    md_l, mc_l, si_l = summarise(lopo, "med_drift", "med_churn", "SI")
    # the paired comparison uses the FIRST scattered draw per paper; the
    # remaining draws enter the log-linear model only.
    matched_first = [e for e in matched if e.get("draw") == 0]
    md_m, mc_m, si_m = summarise(matched_first, "med_drift", "med_churn", "SI")
    pool_si = {k: swing(np.concatenate(v[0]), np.concatenate(v[1]))[0]
               for k, v in pool.items()}

    path = os.path.join(RESULTS, "exH_volumematch.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lopo[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(lopo); w.writerows(matched_first)
    os.replace(path + ".tmp", path)
    # the full draw set goes to its own layer so the paired CSV keeps its schema
    path_d = os.path.join(RESULTS, "exH_volumematch_draws.csv")
    with open(path_d + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matched[0].keys()))
        w.writeheader(); w.writerows(matched)
    os.replace(path_d + ".tmp", path_d)

    # ---- Phase F2: fitted log-linear model with paper-level cluster bootstrap ----
    # log(delta) ~ 1 + log(V) + block ; V = number of rows deleted in the event.
    ev_all = lopo + matched
    yv = np.log(np.asarray([e["med_drift"] for e in ev_all], float))
    vv = np.log(np.asarray([e["n_rows_deleted"] for e in ev_all], float))
    blk = np.asarray([1.0 if e["arm"] == "lopo" else 0.0 for e in ev_all])
    grp = np.asarray([e["paper"] for e in ev_all], dtype=object)
    Xd = np.column_stack([np.ones(len(ev_all)), vv, blk])
    beta, lo, hi, n_ok, boots = ols_cluster_boot(Xd, yv, grp,
                                                 n_boot=args.bootstrap)
    with open(os.path.join(RESULTS, "exH_loglinear_model.csv") + ".tmp", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["term", "coef", "ci_lo", "ci_hi", "mult_factor",
                    "mult_lo", "mult_hi", "boot_p_two_sided", "n_boot", "n_obs"])
        for j, name in enumerate(("intercept", "log_volume", "block")):
            b, l, h = beta[j], lo[j], hi[j]
            # two-sided bootstrap p: twice the smaller tail
            col = boots[:, j]
            p_boot = 2.0 * min(float(np.mean(col <= 0.0)), float(np.mean(col >= 0.0)))
            w.writerow([name, round(float(b), 4), round(float(l), 4),
                        round(float(h), 4),
                        round(float(np.exp(b)), 4), round(float(np.exp(l)), 4),
                        round(float(np.exp(h)), 4), round(p_boot, 6),
                        n_ok, len(ev_all)])
    os.replace(os.path.join(RESULTS, "exH_loglinear_model.csv") + ".tmp",
               os.path.join(RESULTS, "exH_loglinear_model.csv"))
    print(f"[F2] log-linear model (n={len(ev_all)} events, {n_ok} bootstrap resamples):")
    for name, b, l, h in (("intercept", beta[0], lo[0], hi[0]),
                          ("log_volume", beta[1], lo[1], hi[1]),
                          ("block", beta[2], lo[2], hi[2])):
        print(f"     {name:12s} beta={b:+.4f}  95% CI [{l:+.4f}, {h:+.4f}]  "
              f"x{np.exp(b):.3f} [{np.exp(l):.3f}, {np.exp(h):.3f}]")

    # Paired and dose-response statistics
    from scipy import stats
    d_l = np.array([e["med_drift"] for e in lopo])
    d_m = np.array([e["med_drift"] for e in matched_first])
    c_l = np.array([e["med_churn"] / 2 for e in lopo])
    c_m = np.array([e["med_churn"] / 2 for e in matched_first])
    n_del = np.array([e["n_rows_deleted"] for e in lopo])

    paired_ratio = d_l / d_m
    ratio_of_meds = float(np.median(d_l) / np.median(d_m))
    med_paired = float(np.median(paired_ratio))

    boot_rng = np.random.default_rng(SEED)
    n_ev = len(d_l)
    boot_r_meds = np.empty(10000)
    boot_med_paired = np.empty(10000)
    for b in range(10000):
        idx = boot_rng.choice(n_ev, size=n_ev, replace=True)
        boot_r_meds[b] = np.median(d_l[idx]) / np.median(d_m[idx])
        boot_med_paired[b] = np.median(paired_ratio[idx])

    lo_r_meds, hi_r_meds = np.quantile(boot_r_meds, [0.025, 0.975])
    lo_med_paired, hi_med_paired = np.quantile(boot_med_paired, [0.025, 0.975])

    w_res = stats.wilcoxon(d_l, d_m)
    greater = int(np.sum(d_l > d_m))

    ch_diff = c_l - c_m
    ch_gt = int(np.sum(ch_diff > 0))
    ch_lt = int(np.sum(ch_diff < 0))
    ch_eq = int(np.sum(ch_diff == 0))

    sp_l = stats.spearmanr(n_del, d_l)
    sp_m = stats.spearmanr(n_del, d_m)

    summary_row = [{
        "d_lopo_med": float(np.median(d_l)),
        "d_matched_med": float(np.median(d_m)),
        "ratio_of_medians": round(ratio_of_meds, 4),
        "ratio_of_medians_ci_lo": round(float(lo_r_meds), 4),
        "ratio_of_medians_ci_hi": round(float(hi_r_meds), 4),
        "median_paired_ratio": round(med_paired, 4),
        "median_paired_ratio_ci_lo": round(float(lo_med_paired), 4),
        "median_paired_ratio_ci_hi": round(float(hi_med_paired), 4),
        "wilcoxon_stat": float(w_res.statistic),
        "wilcoxon_p": float(w_res.pvalue),
        "n_block_gt_scattered": greater,
        "n_total": n_ev,
        "pct_block_gt_scattered": round(100 * greater / n_ev, 2),
        # c_l / c_m are already the chi/2 fraction
        "churn_lopo_med": float(np.median(c_l)),
        "churn_matched_med": float(np.median(c_m)),
        "churn_higher_block": ch_gt,
        "churn_lower_block": ch_lt,
        "churn_equal": ch_eq,
        "spearman_rho_block": round(float(sp_l.statistic), 4),
        "spearman_p_block": float(sp_l.pvalue),
        "spearman_rho_scattered": round(float(sp_m.statistic), 4),
        "spearman_p_scattered": float(sp_m.pvalue),
        # Phase F2: fields Table S10 reads (previously hand-entered in its note)
        "n_draws": int(n_draws),
        "n_rows_deleted_lopo": int(np.median([e["n_rows_deleted"] for e in lopo])),
        "n_rows_deleted_matched": int(np.median([e["n_rows_deleted"] for e in matched_first])),
        # pooled SI on all events of each arm (the paper metric), not the median
        # per-event SI
        "pooled_si_lopo": round(pool_si["lopo"], 3),
        "pooled_si_matched": round(pool_si["matched_scattered"], 3),
    }]

    sum_path = os.path.join(RESULTS, "exH_volumematch_summary.csv")
    with open(sum_path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_row[0].keys()))
        w.writeheader()
        w.writerows(summary_row)
    os.replace(sum_path + ".tmp", sum_path)

    print("\nper-event median drift / churn fraction (chi/2) / SI, then median across events")
    print(f"  LOPO block deletion        : drift {md_l:.3e}  churn {mc_l/2:.2f}  SI {si_l:.3f}")
    print(f"  volume-matched scattered   : drift {md_m:.3e}  churn {mc_m/2:.2f}  SI {si_m:.3f}")
    print(f"  pooled SI (paper metric)   : block {pool_si['lopo']:.3f}  "
          f"matched {pool_si['matched_scattered']:.3f}")
    print(f"  ratio of medians           : {ratio_of_meds:.2f} (95% CI [{lo_r_meds:.2f}, {hi_r_meds:.2f}])")
    print(f"  median of paired ratios    : {med_paired:.2f} (95% CI [{lo_med_paired:.2f}, {hi_med_paired:.2f}])")
    print(f"  Wilcoxon signed-rank test  : p = {w_res.pvalue:.2e} ({greater}/{n_ev} papers block > scattered)")
    print(f"  churn comparison           : higher={ch_gt}, lower={ch_lt}, equal={ch_eq}")
    print(f"  volume Spearman rho        : block {sp_l.statistic:.2f} (p={sp_l.pvalue:.2e}), scattered {sp_m.statistic:.2f} (p={sp_m.pvalue:.2e})")
    print(f"\nwrote {os.path.relpath(path, ROOT)} and {os.path.relpath(sum_path, ROOT)}")


if __name__ == "__main__":
    main()

