#!/usr/bin/env python3
"""pattern_churn.py — tie structure and configuration-level neighbour churn.

Computes exact binary presence configuration tie statistics (155 unique patterns,
719 duplicated rows, 508 in groups > 5) and evaluates neighbour churn at both the
row-index level and the unique-configuration (pattern) multiset level across
presence PCA, MDS, and t-SNE.

Outputs:
  results/ex_pattern_churn.csv (summary for Table S4)
  results/ex_pattern_structure.csv (tie structure counts)

Run: python3 code/core/pattern_churn.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter
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

from experiment_C1 import load, ordinate, knn_idx, knn_churn


def compute_pattern_churn_for_variant(variant: str, X: np.ndarray, paper: np.ndarray,
                                      inv: np.ndarray, modes=("lopo",), max_events=None):
    S_full = ordinate(variant, X, None)
    knn_full = knn_idx(S_full)
    papers = sorted(set(paper))
    if max_events is not None:
        papers = papers[:max_events]

    out_records = []
    if "lopo" in modes:
        idx_churns = []
        pat_churns = []
        tie_churns, tie_frac, tie_n, tie_drop = [], [], [], []
        tv_churns, tv_frac = [], []
        for p in papers:
            keep = (paper != p)
            keep_idx = np.where(keep)[0]
            S_red = ordinate(variant, X[keep], None)
            nn_full = knn_full[keep_idx]
            nn_red = keep_idx[knn_idx(S_red)]

            # Index churn fraction (|A ^ B| / (2 * k))
            ch_idx = knn_churn(nn_full, nn_red) / 2.0
            idx_churns.append(np.median(ch_idx))

            # Pattern multiset churn
            ch_pat = []
            ch_tie = []          # Phase G1: tie-aware (configuration-level) churn
            tv_multi = []        # bounded total-variation version (C6)
            for a, b in zip(nn_full, nn_red):
                pats_a = Counter(inv[a])
                pats_b = Counter(inv[b])
                diff = sum((pats_a - pats_b).values()) + sum((pats_b - pats_a).values())
                ch_pat.append(diff / (2.0 * len(a)))
                # tie-aware: count distinct CONFIGURATIONS, not distinct rows. A
                # row whose k neighbours all share one configuration has nothing
                # to churn (single tie group), so it is dropped, not scored 0.
                n_a, n_b = len(pats_a), len(pats_b)
                if n_a > 1 and n_b > 1:
                    ch_tie.append(diff / (2.0 * min(n_a, n_b)))
                    keys = set(pats_a) | set(pats_b)
                    tv_multi.append(0.5 * sum(
                        abs(pats_a.get(k, 0) / len(a) - pats_b.get(k, 0) / len(b))
                        for k in keys))
            ch_tie_arr = np.asarray(ch_tie, float)
            tv_arr = np.asarray(tv_multi, float)
            pat_churns.append(np.median(ch_pat))
            tie_churns.append(float(np.median(ch_tie_arr)) if ch_tie_arr.size else float("nan"))
            tie_frac.append(float(np.mean(ch_tie_arr > 0)) if ch_tie_arr.size else float("nan"))
            tv_churns.append(float(np.median(tv_arr)) if tv_arr.size else float("nan"))
            tv_frac.append(float(np.mean(tv_arr > 0)) if tv_arr.size else float("nan"))
            tie_n.append(int(ch_tie_arr.size))
            tie_drop.append(float(np.mean(ch_tie_arr <= 0)) if ch_tie_arr.size else float("nan"))

        out_records.append({
            "variant": variant,
            "mode": "lopo",
            "events_evaluated": len(papers),
            "median_idx_churn": float(np.median(idx_churns)),
            "median_pat_churn": float(np.median(pat_churns)),
            "pct_events_zero_pat_churn": float(np.mean([x == 0.0 for x in pat_churns]) * 100.0),
            # Phase G1: tie-aware neighbour measure
            "median_tie_aware_churn": (float(np.nanmedian(tie_churns))
                                       if tie_churns else float("nan")),
            "frac_rows_tie_aware_churn_gt0": (float(np.nanmedian(tie_frac))
                                              if tie_frac else float("nan")),
            "median_tv_churn": (float(np.nanmedian(tv_churns))
                                if tv_churns else float("nan")),
            "frac_rows_tv_churn_gt0": (float(np.nanmedian(tv_frac))
                                       if tv_frac else float("nan")),
            "n_rows_in_multiconfig_neigh": (int(np.nansum(tie_n)) if tie_n else 0),
            "n_rows_single_tie_dropped": (int(np.nansum(tie_drop) * len(pat_churns))
                                          if tie_drop else 0),
        })

    if "loro" in modes:
        n = len(X)
        idx_churns = []
        pat_churns = []
        tie_churns, tie_frac, tie_n, tie_drop = [], [], [], []
        tv_churns, tv_frac = [], []
        for i in range(n):
            keep = np.ones(n, bool)
            keep[i] = False
            keep_idx = np.where(keep)[0]
            S_red = ordinate(variant, X[keep], None)
            nn_full = knn_full[keep_idx]
            nn_red = keep_idx[knn_idx(S_red)]

            ch_idx = knn_churn(nn_full, nn_red) / 2.0
            idx_churns.append(np.median(ch_idx))

            ch_pat = []
            ch_tie = []          # Phase G1: tie-aware (configuration-level) churn
            tv_multi = []        # bounded total-variation version (C6)
            for a, b in zip(nn_full, nn_red):
                pats_a = Counter(inv[a])
                pats_b = Counter(inv[b])
                diff = sum((pats_a - pats_b).values()) + sum((pats_b - pats_a).values())
                ch_pat.append(diff / (2.0 * len(a)))
                # tie-aware: count distinct CONFIGURATIONS, not distinct rows. A
                # row whose k neighbours all share one configuration has nothing
                # to churn (single tie group), so it is dropped, not scored 0.
                n_a, n_b = len(pats_a), len(pats_b)
                if n_a > 1 and n_b > 1:
                    ch_tie.append(diff / (2.0 * min(n_a, n_b)))
                    keys = set(pats_a) | set(pats_b)
                    tv_multi.append(0.5 * sum(
                        abs(pats_a.get(k, 0) / len(a) - pats_b.get(k, 0) / len(b))
                        for k in keys))
            ch_tie_arr = np.asarray(ch_tie, float)
            tv_arr = np.asarray(tv_multi, float)
            pat_churns.append(np.median(ch_pat))
            tie_churns.append(float(np.median(ch_tie_arr)) if ch_tie_arr.size else float("nan"))
            tie_frac.append(float(np.mean(ch_tie_arr > 0)) if ch_tie_arr.size else float("nan"))
            tv_churns.append(float(np.median(tv_arr)) if tv_arr.size else float("nan"))
            tv_frac.append(float(np.mean(tv_arr > 0)) if tv_arr.size else float("nan"))
            tie_n.append(int(ch_tie_arr.size))
            tie_drop.append(float(np.mean(ch_tie_arr <= 0)) if ch_tie_arr.size else float("nan"))

        out_records.append({
            "variant": variant,
            "mode": "loro",
            "events_evaluated": n,
            "median_idx_churn": float(np.median(idx_churns)),
            "median_pat_churn": float(np.median(pat_churns)),
            "pct_events_zero_pat_churn": float(np.mean([x == 0.0 for x in pat_churns]) * 100.0),
            # Phase G1: tie-aware neighbour measure
            "median_tie_aware_churn": (float(np.nanmedian(tie_churns))
                                       if tie_churns else float("nan")),
            "frac_rows_tie_aware_churn_gt0": (float(np.nanmedian(tie_frac))
                                              if tie_frac else float("nan")),
            "median_tv_churn": (float(np.nanmedian(tv_churns))
                                if tv_churns else float("nan")),
            "frac_rows_tv_churn_gt0": (float(np.nanmedian(tv_frac))
                                       if tv_frac else float("nan")),
            "n_rows_in_multiconfig_neigh": (int(np.nansum(tie_n)) if tie_n else 0),
            "n_rows_single_tie_dropped": (int(np.nansum(tie_drop) * len(pat_churns))
                                          if tie_drop else 0),
        })

    return out_records


def main() -> None:
    X, rho, paper, params, *_ = load()
    B = (X > 0).astype(int)
    pats, inv, cnt = np.unique(B, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()

    n_total = len(B)
    n_pats = len(pats)
    n_shared = int((cnt[inv] > 1).sum())
    pct_shared = float(n_shared / n_total * 100.0)
    n_large = int((cnt[inv] > 5).sum())
    pct_large = float(n_large / n_total * 100.0)

    print(f"[tie-structure] total_rows={n_total} patterns={n_pats} shared={n_shared} ({pct_shared:.1f}%) in_gt_5={n_large} ({pct_large:.1f}%)")

    struct_file = os.path.join(RESULTS, "ex_pattern_structure.csv")
    with open(struct_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["total_rows", n_total])
        w.writerow(["unique_patterns", n_pats])
        w.writerow(["shared_rows", n_shared])
        w.writerow(["pct_shared_rows", round(pct_shared, 2)])
        w.writerow(["rows_in_groups_gt_5", n_large])
        w.writerow(["pct_rows_in_groups_gt_5", round(pct_large, 2)])
    print(f"[tie-structure] wrote {os.path.relpath(struct_file, ROOT)}")

    results = []
    # PCA LOPO & LORO
    print("[churn] running pca_pres lopo & loro...")
    results.extend(compute_pattern_churn_for_variant("pca_pres", X, paper, inv, modes=("lopo", "loro")))

    # MDS LOPO — all 122 papers.
    # R4-11 / R5: the earlier 20-event subset (papers[:20], not a random draw) was
    # only a compute shortcut; the abstract states pattern churn unconditionally,
    # so the reported coverage must be the full corpus.
    print("[churn] running mds_pres lopo (all 122 events)...")
    results.extend(compute_pattern_churn_for_variant("mds_pres", X, paper, inv,
                                                     modes=("lopo",)))

    # t-SNE LOPO — all 122 papers (same rationale).
    print("[churn] running tsne_pres lopo (all 122 events)...")
    results.extend(compute_pattern_churn_for_variant("tsne_pres", X, paper, inv,
                                                     modes=("lopo",)))

    churn_file = os.path.join(RESULTS, "ex_pattern_churn.csv")
    with open(churn_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[churn] wrote {os.path.relpath(churn_file, ROOT)}")
    for r in results:
        print(f"  {r['variant']:10s} {r['mode']:4s} n_events={r['events_evaluated']:>3d} "
              f"median_idx={r['median_idx_churn']:.2f} median_pat={r['median_pat_churn']:.2f} "
              f"zero_pat_events={r['pct_events_zero_pat_churn']:.1f}%")


if __name__ == "__main__":
    main()

