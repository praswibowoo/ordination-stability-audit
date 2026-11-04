#!/usr/bin/env python3
"""f6_tail_analysis.py — detailed tail distribution and mechanism for F6 aeration drift.

Analyzes the leave-one-paper-out drift distribution of F6 aeration (683 cover rows,
112 papers) to distinguish between numerical floor behavior and structured configuration-weight
rebalancing. Computes per-event medians, means, maxima, and row exceedance shares
above tau = 10^-6 (and tau = 10^-7), tracks realized binary configurations (5 -> 5),
and measures L1 frequency vector displacement.

Outputs:
  results/ex_f6_tail_analysis.csv (event-level table)
  results/ex_f6_tail_summary.csv (corpus summary metrics)

Run: python3 code/f6_tail_analysis.py
"""
from __future__ import annotations

import csv
import os
import sys
import numpy as np
from scipy.spatial import procrustes as sp_procrustes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")

from experiment_C1 import load
from experiment_B_familymaps import FAMILIES
from experiment_F_absdrift import fit

TAU_6 = 1e-6
TAU_7 = 1e-7


def main() -> None:
    X, rho, paper, params, *_ = load()
    family = "F6_aeration"
    fam_params = FAMILIES[family]
    cols = [params.index(pt) for pt in fam_params]
    cover = X[:, cols].sum(axis=1) >= 1
    idxs = np.where(cover)[0]
    strata = paper[idxs]
    X_all = X[np.ix_(idxs, range(len(params)))]
    X_fam = X[np.ix_(idxs, cols)]

    pats_full, inv_full = np.unique(X_fam, axis=0, return_inverse=True)
    n_pats_full = len(pats_full)
    freq_full = np.bincount(inv_full, minlength=n_pats_full) / len(X_fam)

    events = []
    pooled_fam_drifts = []

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

        pats_red = np.unique(X_fam[keep], axis=0)
        n_pats_red = len(pats_red)

        # map keep rows to canonical full pattern IDs
        keep_in_full = [np.where((pats_full == row).all(axis=1))[0][0] for row in X_fam[keep]]
        freq_keep = np.bincount(keep_in_full, minlength=n_pats_full) / len(keep_in_full)
        l1_shift = float(np.sum(np.abs(freq_keep - freq_full)))

        pooled_fam_drifts.extend(df)
        share_tau6 = float(np.mean(df > TAU_6))
        share_tau7 = float(np.mean(df > TAU_7))

        events.append({
            "deleted": p,
            "n_del": int((~keep).sum()),
            "n_keep": int(keep.sum()),
            "all_med": float(np.median(da)),
            "all_mean": float(np.mean(da)),
            "fam_med": float(np.median(df)),
            "fam_mean": float(np.mean(df)),
            "fam_max": float(np.max(df)),
            "share_above_tau6": share_tau6,
            "share_above_tau7": share_tau7,
            "cfgs_before": n_pats_full,
            "cfgs_after": n_pats_red,
            "l1_shift": l1_shift
        })

    event_file = os.path.join(RESULTS, "ex_f6_tail_analysis.csv")
    with open(event_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)
    print(f"[f6-tail] wrote {os.path.relpath(event_file, ROOT)}")

    pooled_drifts = np.asarray(pooled_fam_drifts)
    med_fam_med = float(np.median([e["fam_med"] for e in events]))
    med_all_med = float(np.median([e["all_med"] for e in events]))
    raw_ratio = float(med_all_med / med_fam_med) if med_fam_med > 0 else float("inf")

    med_fam_max = float(np.median([e["fam_max"] for e in events]))
    pooled_share_tau6 = float(np.mean(pooled_drifts > TAU_6))
    pooled_share_tau7 = float(np.mean(pooled_drifts > TAU_7))

    per_event_share_med = float(np.median([e["share_above_tau6"] for e in events]))
    per_event_share_mean = float(np.mean([e["share_above_tau6"] for e in events]))
    per_event_share_max = float(np.max([e["share_above_tau6"] for e in events]))

    n_events_gt_10pct = int(sum(e["share_above_tau6"] > 0.10 for e in events))
    n_events_gt_50pct = int(sum(e["share_above_tau6"] > 0.50 for e in events))

    all_cfgs_preserved = all(e["cfgs_before"] == e["cfgs_after"] for e in events)

    # Tail attribution: is the drift above tau driven by configuration-weight
    # shifts rather than configuration loss? Correlate the per-event L1 shift of
    # the configuration weights with the per-event maximum drift.
    from scipy.stats import pearsonr, spearmanr
    _l1 = np.asarray([e["l1_shift"] for e in events], float)
    _fmax = np.asarray([e["fam_max"] for e in events], float)
    _sp_rho, _sp_p = spearmanr(_l1, _fmax)
    _pe_r, _pe_p = pearsonr(_l1, _fmax)

    summary_file = os.path.join(RESULTS, "ex_f6_tail_summary.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["n_events", len(events)])
        w.writerow(["total_evaluations", len(pooled_drifts)])
        w.writerow(["median_fam_med", med_fam_med])
        w.writerow(["median_all_med", med_all_med])
        w.writerow(["raw_drift_ratio", round(raw_ratio, 1)])
        w.writerow(["median_fam_max", med_fam_max])
        w.writerow(["pooled_share_above_tau6", round(pooled_share_tau6, 4)])
        w.writerow(["pooled_share_above_tau7", round(pooled_share_tau7, 4)])
        w.writerow(["per_event_share_median", round(per_event_share_med, 4)])
        w.writerow(["per_event_share_mean", round(per_event_share_mean, 4)])
        w.writerow(["per_event_share_max", round(per_event_share_max, 4)])
        w.writerow(["events_gt_10pct_above_tau", n_events_gt_10pct])
        w.writerow(["events_gt_50pct_above_tau", n_events_gt_50pct])
        w.writerow(["all_configurations_preserved", int(all_cfgs_preserved)])
        w.writerow(["spearman_l1shift_vs_fammax", round(float(_sp_rho), 3)])
        w.writerow(["spearman_l1shift_vs_fammax_p", float(f"{_sp_p:.3e}")])
        w.writerow(["pearson_l1shift_vs_fammax", round(float(_pe_r), 3)])
        w.writerow(["pearson_l1shift_vs_fammax_p", round(float(_pe_p), 4)])

    print(f"[f6-tail] wrote {os.path.relpath(summary_file, ROOT)}")
    print(f"  median_fam_med={med_fam_med:.3e} raw_ratio={raw_ratio:,.0f} median_max={med_fam_max:.3e}")
    print(f"  pooled_share_tau6={pooled_share_tau6*100:.1f}% per_event_median={per_event_share_med*100:.1f}%")
    print(f"  events_gt_10pct={n_events_gt_10pct}/112 events_gt_50pct={n_events_gt_50pct}/112 cfgs_preserved={all_cfgs_preserved}")


if __name__ == "__main__":
    main()

