#!/usr/bin/env python3
"""experiment_M_familyexpansion.py — Family-suite expansion audit (Phase M; D-row pending).

PURPOSE
  Evaluate the 7 frozen parameter families plus 9 exploratory candidate suites under
  the Table 5 diagnostic battery in three spaces: 2-D presence PCA, 2-D Jaccard MDS,
  3-D presence PCA; plus coordinate-tie (SI) statistics for ALL 16 suites, a 100-suite
  random 4-parameter benchmark control against F6_aeration_pH, and a continuous
  value-layer audit for F6_aeration_pH.

  RAW METRICS ONLY. No stability tier ("ready" / "provisional" / "incidental") is
  assigned here. Tier assignment is deferred to the extended-battery registry
  (pre-registered before tiering; see docs/family_group_investigation_findings.md
  for the un-ratified draft battery). The scratch implementation assigned "READY"
  from an INCOMPLETE rubric (it never consulted the random-suite benchmark nor the
  tie-collapse rate); that verdict logic is deliberately NOT reproduced.

PROVENANCE
  Faithful promotion of three scratch runs of 2026-10-13 (all preserved in `docs/archive/phase_m_scratch_2026-10-13/`):
    scratch/investigate_families.py                  -> exM_suite_metrics.csv (metrics)
    scratch/test_si_and_benchmark.py                 -> exM_suite_metrics.csv (SI cols;
                                                        EXTENDED here from 9 to all 16
                                                        suites to close provenance gaps)
    scratch/run_immediate_computational_actions.py   -> exM_benchmark_4param*.csv +
                                                        exM_continuous_audit_*.csv
  Differences from scratch (deliberate):
    * verify_baseline() hard gate is called first (scratch scripts had none).
    * Continuous audit reads the repo layer docs/extraction/stageB/matrix_values_v2.csv
      (scratch read release/p1_family_pca_data_v1.0/values.csv; layers are asserted
      byte-identical at release build, D139).
    * Outputs are CSVs under results/ (atomic writes), not JSON (the scratch-era originals are archived).

ESTIMATORS (fixed; do not tune without a registry amendment)
  * LOPO = leave-one-paper-out; an event is skipped when fewer than MIN_ROWS_KEEP
    rows would remain.
  * MDS uses classical-MDS initialization (sklearn >= 1.8); T2 is deterministic.
    The T7 benchmark stays 2-D presence PCA and does not use this fit.
  * Event drift = median per-row Procrustes-aligned displacement (all-space refit).
  * Ratio, paired = median over events of (event all-drift / event fam-drift);
    ratio, raw = ratio of medians (kept as computed).
  * 95% CI = percentile bootstrap over event ratios, n=BOOT_N, seed 42.
  * tau = 1e-6 noise floor; "share_above_tau_2d" = fraction of EVENTS whose median
    family drift exceeds tau.
  * MDS = precomputed Jaccard, n_init=1, normalized_stress='auto', seed 42.
  * SI (sparsity drift ratio) = med(drift | rho<=Q1) / med(drift | rho>=Q3) within each event;
    tie = |SI - 1| < 1e-6.
  NOTE: the frozen exK CIs (experiment_K / Table 3) use a different bootstrap run.
  The CI columns here are an independent re-computation and are labeled as such;
  they do NOT reproduce exK exactly.

Baseline guard: verifies the P1-4.5 freeze (results/phase4_baseline.json) first.
Outputs (atomic tmp->replace; row counts asserted):
  results/exM_suite_metrics.csv                  (16 rows)
  results/exM_benchmark_4param.csv               (100 rows)
  results/exM_benchmark_4param_summary.csv       (1 row)
  results/exM_continuous_audit_summary.csv       (1 row)
  results/exM_continuous_audit_events.csv        (<= 12 rows)

Run:
  python3 code/experiment_M_familyexpansion.py                # full metrics (~MDS-bound)
  python3 code/experiment_M_familyexpansion.py --pca-only     # skip per-suite MDS (smoke)
  python3 code/experiment_M_familyexpansion.py --benchmark-matched   # canonical T7 (registry §8)
  python3 code/experiment_M_familyexpansion.py --benchmark-matched --suite F6_aeration_pH  # T7 smoke
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import procrustes as sp_procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import classical_mds_init, load, jac_dist  # noqa: E402
from phase4_freeze import verify_baseline  # noqa: E402

SEED = 42
TAU = 1e-6
MIN_ROWS_KEEP = 10          # skip a deletion event if fewer rows would remain
BOOT_N = 1000
RESULTS = os.path.join(ROOT, "results")

CANDIDATES = {
    # Frozen 7 (order fixed; names must match results/phase4_baseline.json)
    "F1_physical": ["DO (mg/L)", "pH", "Salinity (ppt/\u2030)", "Temperature (\u00b0C)"],
    "F1_clarity": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)"],
    "F1_physicala": ["TDS", "EC (\u00b5S/cm)", "Cl- (ppt)", "Hardness (mg/L)", "Alkalinity (mg/L)"],
    "F2_metals": ["As (\u00b5g/L)", "Cd (\u00b5g/L)", "Cr (\u00b5g/L)", "Cu (\u00b5g/L)",
                  "Fe (\u00b5g/L)", "Ni (\u00b5g/L)", "Pb (\u00b5g/L)", "Zn (\u00b5g/L)", "Co (\u00b5g/L)"],
    "F3_nutrients": ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)", "PO4 (mg/L)"],
    "F4_organic": ["COD (mg/L)", "BOD (mg/L)", "Total Coliform (CFU/100mL)"],
    "F6_aeration": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"],
    # Exploratory candidates (post-hoc; not pre-registered)
    "F6_aeration_pH": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "pH"],
    "F6_aeration_temp": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "Temperature (\u00b0C)"],
    "F6_aeration_TSS": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "TSS (mg/L)"],
    "F6_aeration_PO4": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "PO4 (mg/L)"],
    "F6_core_5param": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "pH", "Temperature (\u00b0C)"],
    "W5C4_6param": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "pH", "Temperature (\u00b0C)", "TSS (mg/L)"],
    "F1_clarity_chla": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)", "Chlorophyll-a (mg/m3)"],
    "F1_clarity_core": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)", "DO (mg/L)", "pH"],
    "F1_phys_nutr": ["DO (mg/L)", "pH", "Salinity (ppt/\u2030)", "Temperature (\u00b0C)", "NO3 (mg/L)", "PO4 (mg/L)"],
}
FROZEN = [n for n in list(CANDIDATES)[:7]]


def write_csv(path, fieldnames, rows, expect_n=None):
    """Atomic write; assert exact row count when given."""
    if expect_n is not None:
        assert len(rows) == expect_n, f"{os.path.basename(path)}: {len(rows)} rows != {expect_n}"
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)
    print(f"  wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


def fit_pca(M, k):
    k = min(k, M.shape[1], M.shape[0] - 1)
    return PCA(n_components=k, random_state=SEED).fit_transform(M)


def fit_mds(D, k):
    return MDS(n_components=k, dissimilarity="precomputed", random_state=SEED,
               **classical_mds_init(), n_init=1,
               normalized_stress="auto").fit_transform(D)


def bootstrap_ci(paired_ratios, n_boot=BOOT_N):
    rng = np.random.default_rng(SEED)
    meds = [np.median(rng.choice(paired_ratios, size=len(paired_ratios), replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def run_suite(name, fam_params, X, rho, paper, params, pca_only):
    cols = [params.index(p) for p in fam_params]
    m = len(fam_params)

    has_any = X[:, cols].sum(axis=1) >= 1
    has_all = X[:, cols].sum(axis=1) == m
    cover_rows, full_rows = int(has_any.sum()), int(has_all.sum())
    cover_papers = int(len(set(paper[has_any])))
    full_papers = int(len(set(paper[has_all])))

    idxs = np.where(has_any)[0]
    X_all = X[idxs]
    X_fam = X[idxs][:, cols]
    pmask = paper[idxs]
    rho_f = rho[idxs]

    n_cfgs = int(len(np.unique(X_fam, axis=0)))
    max_cfgs = 2 ** m - 1

    # Reference fits (2-D and 3-D presence PCA)
    S2_all, S2_fam = fit_pca(X_all, 2), fit_pca(X_fam, 2)
    S3_all, S3_fam = fit_pca(X_all, 3), fit_pca(X_fam, 3)
    events_2d, events_3d = [], []

    # 2-D MDS reference fits (skipped in --pca-only mode)
    events_mds = []
    if not pca_only:
        S_mds_all = fit_mds(jac_dist(X_all), 2)
        S_mds_fam = fit_mds(jac_dist(X_fam), 2)

    # SI reference fit (2-D presence PCA family space; all suites)
    events_si = []

    for p in sorted(set(pmask)):
        keep = pmask != p
        if keep.sum() < MIN_ROWS_KEEP:
            continue

        a1, b1, _ = sp_procrustes(S2_all[keep], fit_pca(X_all[keep], 2))
        a2, b2, _ = sp_procrustes(S2_fam[keep], fit_pca(X_fam[keep], 2))
        events_2d.append((float(np.median(np.linalg.norm(b1 - a1, axis=1))),
                          float(np.median(np.linalg.norm(b2 - a2, axis=1)))))

        a3, b3, _ = sp_procrustes(S3_all[keep], fit_pca(X_all[keep], 3))
        a4, b4, _ = sp_procrustes(S3_fam[keep], fit_pca(X_fam[keep], 3))
        events_3d.append((float(np.median(np.linalg.norm(b3 - a3, axis=1))),
                          float(np.median(np.linalg.norm(b4 - a4, axis=1)))))

        if not pca_only:
            am1, bm1, _ = sp_procrustes(S_mds_all[keep], fit_mds(jac_dist(X_all[keep]), 2))
            am2, bm2, _ = sp_procrustes(S_mds_fam[keep], fit_mds(jac_dist(X_fam[keep]), 2))
            events_mds.append((float(np.median(np.linalg.norm(bm1 - am1, axis=1))),
                               float(np.median(np.linalg.norm(bm2 - am2, axis=1)))))

        # SI for this event (tie-aware convention, same as frozen C1/pattern_churn)
        a5, b5, _ = sp_procrustes(S2_fam[keep], fit_pca(X_fam[keep], 2))
        df = np.linalg.norm(b5 - a5, axis=1)
        q1, q3 = np.quantile(rho_f[keep], [0.25, 0.75])
        sp = df[rho_f[keep] <= q1]
        de = df[rho_f[keep] >= q3]
        med_sp = float(np.median(sp)) if len(sp) else 0.0
        med_de = float(np.median(de)) if len(de) else 0.0
        if med_de > 1e-15:
            events_si.append(med_sp / med_de)
        else:
            events_si.append(1.0 if abs(med_sp - med_de) < 1e-12 else np.nan)

    all2 = np.array([e[0] for e in events_2d]); fam2 = np.array([e[1] for e in events_2d])
    all3 = np.array([e[0] for e in events_3d]); fam3 = np.array([e[1] for e in events_3d])
    med_all2, med_fam2 = float(np.median(all2)), float(np.median(fam2))
    med_all3, med_fam3 = float(np.median(all3)), float(np.median(fam3))
    paired2 = all2 / np.maximum(fam2, 1e-15)
    ci2 = bootstrap_ci(paired2)
    reportable_3d = bool((med_fam3 >= TAU) and (m >= 4))

    row = {
        "name": name,
        "m_params": m,
        "cover_rows": cover_rows, "cover_papers": cover_papers,
        "full_rows": full_rows, "full_papers": full_papers,
        "n_events": len(events_2d),
        "n_cfgs": n_cfgs, "max_cfgs": max_cfgs,
        "med_all_2d_pca": med_all2, "med_fam_2d_pca": med_fam2,
        "raw_ratio_2d_pca": float(med_all2 / (med_fam2 + 1e-15)),
        "paired_ratio_2d_pca": float(np.median(paired2)),
        "ci_low_2d_pca": ci2[0], "ci_high_2d_pca": ci2[1],
        "share_above_tau_2d": float(np.mean(fam2 > TAU)),
        "med_all_mds": "", "med_fam_mds": "", "raw_ratio_mds": "",
        "paired_ratio_mds": "", "ci_low_mds": "", "ci_high_mds": "",
        "med_fam_3d_pca": med_fam3,
        "reportable_3d": reportable_3d,
        "raw_ratio_3d_pca": float(med_all3 / (med_fam3 + 1e-15)) if reportable_3d else "",
        "paired_ratio_3d_pca": float(np.median(all3 / np.maximum(fam3, 1e-15))) if reportable_3d else "",
        "median_si": float(np.nanmedian(events_si)) if events_si else "",
        "tie_rate": float(np.mean([abs(si - 1.0) < 1e-6 for si in events_si])) if events_si else "",
    }
    if not pca_only and events_mds:
        allm = np.array([e[0] for e in events_mds]); famm = np.array([e[1] for e in events_mds])
        med_allm, med_famm = float(np.median(allm)), float(np.median(famm))
        cim = bootstrap_ci(allm / np.maximum(famm, 1e-15))
        row.update({
            "med_all_mds": med_allm, "med_fam_mds": med_famm,
            "raw_ratio_mds": float(med_allm / (med_famm + 1e-15)),
            "paired_ratio_mds": float(np.median(allm / np.maximum(famm, 1e-15))),
            "ci_low_mds": cim[0], "ci_high_mds": cim[1],
        })
    return row


def lopo_paired(X, paper, idxs, cols):
    """Event-median paired drift stats for one suite on a fixed row subset.

    Returns (med_all, med_fam, paired_ratio, n_events). Shared by the sensitivity
    benchmark (fixed rule) and the canonical T7 matched benchmark (registry §8).
    """
    Xa, Xf = X[idxs], X[idxs][:, cols]
    pmask = paper[idxs]
    S_all, S_fam = fit_pca(Xa, 2), fit_pca(Xf, 2)
    da, df, pr = [], [], []
    for p in sorted(set(pmask)):
        keep = pmask != p
        if keep.sum() < MIN_ROWS_KEEP:
            continue
        a1, b1, _ = sp_procrustes(S_all[keep], fit_pca(Xa[keep], 2))
        a2, b2, _ = sp_procrustes(S_fam[keep], fit_pca(Xf[keep], 2))
        ma = float(np.median(np.linalg.norm(b1 - a1, axis=1)))
        mf = float(np.median(np.linalg.norm(b2 - a2, axis=1)))
        da.append(ma); df.append(mf); pr.append(ma / max(mf, 1e-15))
    return (float(np.median(da)), float(np.median(df)), float(np.median(pr)), len(da)) \
        if da else (0.0, 0.0, 0.0, 0)


def run_benchmark(X, paper, params):
    """100 random 4-parameter suites vs F6_aeration_pH, identical 2-D PCA LOPO protocol.
    SENSITIVITY VARIANT (fixed acceptance rule); the canonical T7 is span-matched
    (run_benchmark_matched). Kept unchanged for continuity (D158 receipts)."""
    target_params = ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "pH"]
    tcols = [params.index(p) for p in target_params]
    tcover = X[:, tcols].sum(axis=1) >= 1
    tidxs = np.where(tcover)[0]

    tgt_ma, tgt_mf, tgt_paired, _ = lopo_paired(X, paper, tidxs, tcols)
    tgt_raw = tgt_ma / max(tgt_mf, 1e-15)

    rng = np.random.default_rng(SEED)
    suites, attempts = [], 0
    while len(suites) < 100 and attempts < 1000:
        attempts += 1
        cols = sorted(rng.choice(len(params), size=4, replace=False).tolist())
        cover = X[:, cols].sum(axis=1) >= 1
        if cover.sum() < 50 or len(set(paper[cover])) < 10:
            continue
        if cols not in suites:
            suites.append(cols)

    rows = []
    t0 = time.time()
    for i, cols in enumerate(suites):
        idxs = np.where(X[:, cols].sum(axis=1) >= 1)[0]
        ma, mf, paired, nev = lopo_paired(X, paper, idxs, cols)
        rows.append({
            "suite_id": i + 1, "params": " | ".join(params[c] for c in cols),
            "n_cover": len(idxs), "n_events": nev,
            "med_all": ma, "med_fam": mf,
            "raw_ratio": ma / max(mf, 1e-15), "paired_ratio": paired,
            "is_floor": mf <= TAU,
        })
    print(f"  benchmark: {len(rows)} suites in {time.time()-t0:.0f}s")

    non_deg = [b for b in rows if not b["is_floor"]]
    ex = lambda pool, k: sum(1 for b in pool if b[k] >= (tgt_raw if k == "raw_ratio" else tgt_paired))
    summary = {
        "target_name": "F6_aeration_pH",
        "target_raw_ratio": tgt_raw, "target_paired_ratio": tgt_paired,
        "target_med_fam": tgt_mf,
        "n_total": len(rows), "n_non_deg": len(non_deg),
        "exceed_raw_all": ex(rows, "raw_ratio"), "exceed_paired_all": ex(rows, "paired_ratio"),
        "exceed_raw_nondeg": ex(non_deg, "raw_ratio"), "exceed_paired_nondeg": ex(non_deg, "paired_ratio"),
    }
    summary.update({
        "p_raw_all": summary["exceed_raw_all"] / len(rows),
        "p_paired_all": summary["exceed_paired_all"] / len(rows),
        "p_raw_nondeg": summary["exceed_raw_nondeg"] / len(non_deg) if non_deg else 1.0,
        "p_paired_nondeg": summary["exceed_paired_nondeg"] / len(non_deg) if non_deg else 1.0,
    })
    print(f"  target paired {tgt_paired:.3f} | exceeded (paired, non-deg) "
          f"{summary['exceed_paired_nondeg']}/{summary['n_non_deg']} (p={summary['p_paired_nondeg']:.3f})")
    return rows, summary


def run_benchmark_matched(X, paper, params, only=None):
    """Canonical T7 (registry §8): m-matched, cover-span-matched random-suite benchmark.

    Pool of 2000 accepted draws (cover >= 50 rows, >= 10 papers, unique, never the
    candidate itself); the 100 with minimal |log(cover / candidate cover)| are
    evaluated with the identical LOPO paired-ratio protocol. Non-degenerate :=
    median family drift > TAU; the decision statistic is the paired ratio.
    Writes results/exM_benchmark_matched_<suite>.csv per suite + a summary CSV.
    """
    summary = []
    for name, fam in CANDIDATES.items():
        if only and name != only:
            continue
        cols0 = [params.index(p) for p in fam]
        m = len(fam)
        tcover = int((X[:, cols0].sum(axis=1) >= 1).sum())
        tidxs = np.where(X[:, cols0].sum(axis=1) >= 1)[0]
        t_ma, t_mf, t_paired, _ = lopo_paired(X, paper, tidxs, cols0)

        rng = np.random.default_rng(SEED)
        seen, pool, attempts = set(), [], 0
        while len(pool) < 2000 and attempts < 400000:
            attempts += 1
            cols = tuple(sorted(rng.choice(len(params), size=m, replace=False).tolist()))
            if cols in seen or list(cols) == sorted(cols0):
                continue
            seen.add(cols)
            cover = X[:, list(cols)].sum(axis=1) >= 1
            ncov = int(cover.sum())
            if ncov < 50 or len(set(paper[cover])) < 10:
                continue
            pool.append((abs(math.log(ncov / max(tcover, 1))), list(cols), ncov))
        pool.sort(key=lambda t: t[0])  # stable: ties broken by draw order
        sel = pool[:100]

        rows = []
        t0 = time.time()
        for i, (span, cols, ncov) in enumerate(sel):
            idxs = np.where(X[:, list(cols)].sum(axis=1) >= 1)[0]
            ma, mf, paired, nev = lopo_paired(X, paper, idxs, cols)
            rows.append({
                "suite_id": i + 1, "params": " | ".join(params[c] for c in cols),
                "n_cover": ncov, "n_events": nev, "med_all": ma, "med_fam": mf,
                "raw_ratio": ma / max(mf, 1e-15), "paired_ratio": paired,
                "is_floor": mf <= TAU, "span_ratio": math.exp(span),
            })
        nd = [r for r in rows if not r["is_floor"]]
        exceed = sum(1 for r in nd if r["paired_ratio"] >= t_paired)
        spans = sorted(r["span_ratio"] for r in rows)
        row = {
            "candidate": name, "m_params": m, "target_cover_rows": tcover,
            "target_paired_ratio": t_paired, "target_raw_ratio": t_ma / max(t_mf, 1e-15),
            "pool_size": len(pool), "n_selected": len(rows), "n_non_deg": len(nd),
            "exceed_paired_nondeg": exceed,
            "p_paired_nondeg": (exceed / len(nd)) if nd else "",
            "span_ratio_median": spans[len(spans) // 2] if spans else "",
            "span_ratio_max": spans[-1] if spans else "",
        }
        summary.append(row)
        write_csv(os.path.join(RESULTS, f"exM_benchmark_matched_{name}.csv"),
                  ["suite_id", "params", "n_cover", "n_events", "med_all", "med_fam",
                   "raw_ratio", "paired_ratio", "is_floor", "span_ratio"],
                  rows, expect_n=len(rows))
        print(f"  [T7] {name:18s} cover={tcover:3d} paired={t_paired:10.2f} | "
              f"non-deg {len(nd):3d} | exceeded {exceed:3d}/{len(nd):3d} | "
              f"span p50 {row['span_ratio_median']:.3f} max {row['span_ratio_max']:.3f} | {time.time()-t0:.0f}s")
    if summary:
        write_csv(os.path.join(RESULTS, "exM_benchmark_matched_summary.csv"),
                  list(summary[0].keys()), summary, expect_n=len(summary))
    return summary


def run_continuous_audit():
    """Value-layer audit for F6_aeration_pH on the repo layer (not the release copy)."""
    vals = pd.read_csv(os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv"))
    target = ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)", "pH"]
    sub = vals[vals["param"].isin(target) & (vals["is_real_value"].astype(str).str.upper() == "TRUE")].copy()
    sub["val"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["val"])
    piv = sub.pivot_table(index=["batch_id", "row_id"], columns="param", values="val")
    complete = piv.dropna(subset=target)

    V = complete[target].values
    papers = complete.index.get_level_values("batch_id").to_numpy()
    uniq = len(np.unique(V, axis=0))

    from sklearn.preprocessing import StandardScaler
    V_std = StandardScaler().fit_transform(V)
    S_val = PCA(n_components=2, random_state=SEED).fit_transform(V_std)
    events = []
    for p in sorted(set(papers)):
        keep = papers != p
        if keep.sum() < 5:
            continue
        a, b, _ = sp_procrustes(S_val[keep], PCA(n_components=2, random_state=SEED).fit_transform(V_std[keep]))
        drift = np.linalg.norm(b - a, axis=1)
        events.append({"paper": p, "n_keep": int(keep.sum()),
                       "med_drift": float(np.median(drift)),
                       "mean_drift": float(np.mean(drift)),
                       "max_drift": float(np.max(drift))})
    med = float(np.median([e["med_drift"] for e in events]))
    summary = {
        "n_complete_rows": len(complete), "n_papers": len(set(papers)),
        "unique_vectors": uniq, "uniqueness_pct": float(uniq / len(V) * 100.0),
        "n_lopo_events": len(events),
        "med_val_drift": med,
        # scratch definition: mean of per-event MEDIANS (not mean of means — kept
        # faithful to the evidence chain; see D158 verification receipt)
        "mean_val_drift": float(np.mean([e["med_drift"] for e in events])),
        "ratio_to_tau": float(med / TAU),  # pins the digest's "4,077-fold above the floor"
        "above_tau": bool(med > TAU),
    }
    print(f"  continuous audit: {summary['n_complete_rows']} rows / {summary['n_papers']} papers | "
          f"unique {uniq} ({summary['uniqueness_pct']:.1f}%) | med drift {med:.3e}")
    return events, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca-only", action="store_true",
                    help="skip per-suite MDS (fast smoke); metric columns left empty")
    ap.add_argument("--benchmark-matched", action="store_true",
                    help="canonical T7 span-matched random-suite benchmark only (registry §8)")
    ap.add_argument("--suite", default="",
                    help="limit --benchmark-matched to one suite (testing)")
    args = ap.parse_args()

    lock = verify_baseline()
    print(f"baseline OK: {lock['data_version']} | seed {lock['seed']}")
    X, rho, paper, params, *_ = load()
    print(f"loaded {len(X)} data rows x {len(params)} params\n")

    if args.benchmark_matched:
        print("Canonical T7 matched benchmark (registry §8)\n")
        run_benchmark_matched(X, paper, params, only=(args.suite or None))
        print("\nT7 matched benchmark complete.")
        return

    rows = []
    for name, fam in CANDIDATES.items():
        t0 = time.time()
        r = run_suite(name, fam, X, rho, paper, params, args.pca_only)
        rows.append(r)
        mds = f"{r['paired_ratio_mds']:8.2f}" if r["paired_ratio_mds"] != "" else "    n/a"
        print(f"[{time.time()-t0:5.1f}s] {name:18s} m={r['m_params']} | "
              f"2D {r['paired_ratio_2d_pca']:9.2f} [{r['ci_low_2d_pca']:.2f},{r['ci_high_2d_pca']:.2f}] | "
              f"MDS {mds} | fam-d {r['med_fam_2d_pca']:.2e} | "
              f"tie {r['tie_rate']*100 if r['tie_rate'] != '' else float('nan'):5.1f}%")

    # ---- hard cross-checks (reproduction of the scratch runs + frozen baseline) ----
    assert len(rows) == 16, len(rows)
    by = {r["name"]: r for r in rows}
    base = json.load(open(os.path.join(RESULTS, "phase4_baseline.json"), encoding="utf-8"))["baselines"]["families"]
    for n in FROZEN:
        assert by[n]["full_rows"] == base[n], (n, by[n]["full_rows"], base[n])
    assert by["F6_aeration"]["cover_rows"] == 683 and by["F6_aeration"]["full_rows"] == 159
    assert abs(by["F6_aeration"]["paired_ratio_2d_pca"] - 76712.265) < 1.0
    assert abs(by["F6_aeration_pH"]["paired_ratio_2d_pca"] - 6.985368316569549) < 1e-6
    assert abs(by["F1_phys_nutr"]["paired_ratio_2d_pca"] - 48.887) < 1e-3
    # 1e-7: the current cross-process float path differs by ~2e-9 from the
    # scratch value; the estimate is stable well inside this tolerance.
    assert abs(by["F6_aeration"]["median_si"] - 0.017016637077189367) < 1e-7
    assert abs(by["F1_physical"]["tie_rate"] - 1.0) < 1e-12
    print("suite metrics cross-checks PASS (frozen baseline + scratch reproduction)")

    fields = ["name", "m_params", "cover_rows", "cover_papers", "full_rows", "full_papers",
              "n_events", "n_cfgs", "max_cfgs", "med_all_2d_pca", "med_fam_2d_pca",
              "raw_ratio_2d_pca", "paired_ratio_2d_pca", "ci_low_2d_pca", "ci_high_2d_pca",
              "share_above_tau_2d", "med_all_mds", "med_fam_mds", "raw_ratio_mds",
              "paired_ratio_mds", "ci_low_mds", "ci_high_mds", "med_fam_3d_pca",
              "reportable_3d", "raw_ratio_3d_pca", "paired_ratio_3d_pca", "median_si", "tie_rate"]
    write_csv(os.path.join(RESULTS, "exM_suite_metrics.csv"), fields, rows, expect_n=16)

    bench_rows, bench_summary = run_benchmark(X, paper, params)
    assert bench_summary["n_total"] == 100, bench_summary["n_total"]
    assert bench_summary["n_non_deg"] == 77, bench_summary["n_non_deg"]
    assert bench_summary["exceed_paired_all"] == 83, bench_summary["exceed_paired_all"]
    assert bench_summary["exceed_paired_nondeg"] == 60, bench_summary["exceed_paired_nondeg"]
    write_csv(os.path.join(RESULTS, "exM_benchmark_4param.csv"),
              ["suite_id", "params", "n_cover", "n_events", "med_all", "med_fam",
               "raw_ratio", "paired_ratio", "is_floor"], bench_rows, expect_n=100)
    write_csv(os.path.join(RESULTS, "exM_benchmark_4param_summary.csv"),
              list(bench_summary.keys()), [bench_summary], expect_n=1)

    events, cont = run_continuous_audit()
    assert cont["n_complete_rows"] == 73, cont["n_complete_rows"]
    assert cont["n_papers"] == 12, cont["n_papers"]
    assert cont["unique_vectors"] == 62, cont["unique_vectors"]
    write_csv(os.path.join(RESULTS, "exM_continuous_audit_summary.csv"),
              list(cont.keys()), [cont], expect_n=1)
    write_csv(os.path.join(RESULTS, "exM_continuous_audit_events.csv"),
              ["paper", "n_keep", "med_drift", "mean_drift", "max_drift"], events,
              expect_n=cont["n_lopo_events"])

    print("\nAll cross-checks PASS — metrics are frozen reproductions of the scratch runs.")


if __name__ == "__main__":
    main()
