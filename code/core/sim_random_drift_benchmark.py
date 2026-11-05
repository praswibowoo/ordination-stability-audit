#!/usr/bin/env python3
"""sim_random_drift_benchmark.py — deterministic 100-suite random drift benchmark.

Evaluates the 100 random 3-parameter suites from results/sim_random_families.csv
under the identical LOPO Procrustes protocol as F6 aeration (presence PCA, 2-D).
Computes:
  - Stability gain G (from sim_random_families.csv)
  - All-parameter median drift
  - Family-restricted median drift
  - Raw Procrustes drift ratio (ratio of medians)
  - Median of paired drift ratios

Outputs results/sim_random_drift_benchmark.csv and calculates empirical exceedance p-values
against F6 aeration:
  - G (+0.9634): 1/100 (p = 0.020)
  - Raw Procrustes drift ratio (60,900): 19/100 (p = 0.198)
  - Median paired ratio (76,712): 19/100 (p = 0.198)

Run: python3 code/core/sim_random_drift_benchmark.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import procrustes as sp_procrustes

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

from experiment_C1 import load  # noqa: E402


def fit(M: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=SEED).fit_transform(M)


def run_benchmark() -> list[dict]:
    X, rho, paper, params, *_ = load()
    short = {p.split(" (")[0]: i for i, p in enumerate(params)}

    in_csv = os.path.join(RESULTS, "sim_random_families.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing input suite file: {in_csv}")

    with open(in_csv, encoding="utf-8") as f:
        rand_rows = list(csv.DictReader(f))

    print(f"[benchmark] Loaded {len(rand_rows)} random families from {os.path.relpath(in_csv, ROOT)}")

    bench_results = []
    for r in rand_rows:
        fam_params = [m.strip() for m in r["members"].split("|")]
        cols = [short[pt] for pt in fam_params]
        cover = X[:, cols].sum(axis=1) >= 1
        idxs = np.where(cover)[0]
        strata = paper[idxs]
        X_all = X[np.ix_(idxs, range(len(params)))]
        X_fam = X[np.ix_(idxs, cols)]

        pats = np.unique((X_fam > 0).astype(int), axis=0)

        da_meds, df_meds, pratios = [], [], []
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
            m_a = float(np.median(da))
            m_f = float(np.median(df))
            da_meds.append(m_a)
            df_meds.append(m_f)
            if m_f > 0:
                pratios.append(m_a / m_f)

        if not da_meds:
            continue
        m_all = float(np.median(da_meds))
        m_fam = float(np.median(df_meds))
        ratio_raw = m_all / m_fam if m_fam > 0 else float("inf")
        med_paired = float(np.median(pratios)) if pratios else float("inf")

        bench_results.append({
            "members": r["members"],
            "n_cover": len(idxs),
            "n_events": len(da_meds),
            "n_configs": len(pats),
            "G": float(r["G"]),
            "all_med": m_all,
            "fam_med": m_fam,
            "ratio_raw": ratio_raw,
            "median_paired_ratio": med_paired,
        })

    out_csv = os.path.join(RESULTS, "sim_random_drift_benchmark.csv")
    with open(out_csv + ".tmp", "w", newline="", encoding="utf-8") as f:
        fields = [
            "members", "n_cover", "n_events", "n_configs", "G",
            "all_med", "fam_med", "ratio_raw", "median_paired_ratio"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(bench_results)
    os.replace(out_csv + ".tmp", out_csv)
    print(f"[benchmark] Wrote {len(bench_results)} rows to {os.path.relpath(out_csv, ROOT)}")
    return bench_results


def evaluate_exceedance(bench_results: list[dict]):
    # Dynamically load F6 reference values from exC2_absolute_drift.csv and exG_absratio.csv
    f6_ratio_med = 60900.28
    f6_paired_ratio = 76712.26
    f6_g = 0.9634

    c2_csv = os.path.join(RESULTS, "exC2_absolute_drift.csv")
    if os.path.exists(c2_csv):
        with open(c2_csv, encoding="utf-8") as f:
            f6_all_meds = []
            f6_fam_meds = []
            for r in csv.DictReader(f):
                if r.get("family") == "F6_aeration":
                    f6_all_meds.append(float(r["all_med"]))
                    f6_fam_meds.append(float(r["fam_med"]))
            if f6_all_meds and f6_fam_meds:
                med_fam = float(np.median(f6_fam_meds))
                if med_fam > 0:
                    f6_ratio_med = float(np.median(f6_all_meds)) / med_fam

    g_csv = os.path.join(RESULTS, "exG_absratio.csv")
    if os.path.exists(g_csv):
        with open(g_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("family") == "F6_aeration":
                    f6_g = float(r.get("G", f6_g))
                    f6_paired_ratio = float(r.get("median_paired_ratio", f6_paired_ratio))

    ratios = [b["ratio_raw"] for b in bench_results]
    pratios = [b["median_paired_ratio"] for b in bench_results]
    g_vals = [b["G"] for b in bench_results]

    exc_raw = sum(x >= f6_ratio_med for x in ratios)
    exc_paired = sum(x >= f6_paired_ratio for x in pratios)
    exc_g = sum(x >= f6_g for x in g_vals)

    p_raw = (exc_raw + 1) / (len(ratios) + 1)
    p_paired = (exc_paired + 1) / (len(pratios) + 1)
    p_g = (exc_g + 1) / (len(g_vals) + 1)

    print("\nBenchmark Evaluation vs F6 Aeration:")
    print(f"  Exceedances on G (+{f6_g:.3f}): {exc_g}/{len(g_vals)} (p = {p_g:.3f})")
    print(f"  Exceedances on raw drift ratio ({f6_ratio_med:.0f}): {exc_raw}/{len(ratios)} (p = {p_raw:.3f})")
    print(f"  Exceedances on median paired ratio ({f6_paired_ratio:.0f}): {exc_paired}/{len(pratios)} (p = {p_paired:.3f})")

    summary_file = os.path.join(RESULTS, "sim_random_drift_benchmark_summary.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "exceedances", "total_suites", "p_value", "f6_anchor"])
        w.writerow(["raw_ratio", exc_raw, len(ratios), round(p_raw, 3), round(f6_ratio_med, 1)])
        w.writerow(["paired_ratio", exc_paired, len(pratios), round(p_paired, 3), round(f6_paired_ratio, 1)])
        w.writerow(["G", exc_g, len(g_vals), round(p_g, 3), round(f6_g, 4)])
        # Density-restricted subset (mean slice density >= 0.45) and the
        # five-configuration subset, both benchmarked on the G metric.
        dens = {r["members"]: float(r["mean_slice_density"])
                for r in csv.DictReader(open(os.path.join(RESULTS, "sim_random_families.csv"),
                                             encoding="utf-8"))}
        hi = [(float(b["G"]), dens.get(b["members"], 0.0)) for b in bench_results
              if dens.get(b["members"], 0.0) >= 0.45]
        exc_hi = sum(g >= f6_g for g, _ in hi)
        w.writerow(["G_density_restricted", exc_hi, len(hi),
                    round((exc_hi + 1) / (len(hi) + 1), 3), round(f6_g, 4)])
        five = [(float(b["G"]), b["n_configs"]) for b in bench_results
                if int(b["n_configs"]) == 5]
        exc_five = sum(g >= f6_g for g, _ in five)
        w.writerow(["G_five_configuration", exc_five, len(five),
                    round((exc_five + 1) / (len(five) + 1), 3), round(f6_g, 4)])
        # Does the configuration count of a random suite predict its ratio gain G?
        from scipy.stats import spearmanr
        _rho, _rho_p = spearmanr([int(b["n_configs"]) for b in bench_results], g_vals)
        w.writerow(["config_count_vs_G_spearman", round(float(_rho), 3), len(g_vals),
                    round(float(f"{_rho_p:.3e}"), 12), round(f6_g, 4)])
        # Extreme G values among the random suites (Fig. 6c truncates the tail).
        w.writerow(["G_min", round(float(min(g_vals)), 3), len(g_vals), "", round(f6_g, 4)])
        w.writerow(["G_le_minus3_count", sum(1 for g in g_vals if g <= -3.0), len(g_vals),
                    "", round(f6_g, 4)])
    print(f"[benchmark] Wrote summary to {os.path.relpath(summary_file, ROOT)}")

    # non-degenerate refinement (P8F, D173): drop machine-precision denominators
    # (fam_med < 1e-16 = coordinate-floor collapse) before counting exceedances.
    nondeg = [b for b in bench_results if float(b["fam_med"]) >= 1e-16]
    exc_nd = sum(float(b["ratio_raw"]) >= f6_ratio_med for b in nondeg)
    n_nd = len(nondeg)
    p_nd = (exc_nd + 1) / (n_nd + 1)
    nd_file = os.path.join(RESULTS, "sim_random_drift_benchmark_nondeg.csv")
    with open(nd_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "exceedances", "total_suites", "p_value", "degenerate_excluded"])
        w.writerow(["raw_ratio_nondeg", exc_nd, n_nd, round(p_nd, 3), len(bench_results) - n_nd])
        # Range of the non-degenerate exceeders: configuration count and median
        # family drift (the manuscript quotes "4 to 6 configurations" and
        # "median drift 1e-12 to 1e-9").
        _nd_exc = [b for b in bench_results
                   if float(b["fam_med"]) >= 1e-16 and float(b["ratio_raw"]) >= f6_ratio_med]
        w.writerow(["nondeg_exceeders", len(_nd_exc), n_nd, round(p_nd, 3),
                    len(bench_results) - n_nd])
        w.writerow(["nondeg_exceeders_min_configs",
                    min((int(b["n_configs"]) for b in _nd_exc), default=-1), n_nd,
                    round(p_nd, 3), len(bench_results) - n_nd])
        w.writerow(["nondeg_exceeders_max_configs",
                    max((int(b["n_configs"]) for b in _nd_exc), default=-1), n_nd,
                    round(p_nd, 3), len(bench_results) - n_nd])
        w.writerow(["nondeg_exceeders_min_fam_med",
                    f"{min((float(b['fam_med']) for b in _nd_exc), default=float('nan')):.3e}",
                    n_nd, round(p_nd, 3), len(bench_results) - n_nd])
        w.writerow(["nondeg_exceeders_max_fam_med",
                    f"{max((float(b['fam_med']) for b in _nd_exc), default=float('nan')):.3e}",
                    n_nd, round(p_nd, 3), len(bench_results) - n_nd])
        w.writerow(["degenerate_count", len(bench_results) - n_nd, len(bench_results),
                    round(p_nd, 3), len(bench_results) - n_nd])
    print(f"[benchmark] Non-degenerate refinement: {exc_nd}/{n_nd} (p = {p_nd:.3f}; "
          f"{len(bench_results) - n_nd} degenerate excluded) -> {os.path.relpath(nd_file, ROOT)}")

    top5 = sorted(bench_results, key=lambda x: float(x["ratio_raw"]), reverse=True)[:5]
    print("\nTop 5 random suites by raw drift ratio:")
    for t in top5:
        print(f"  {t['members']:35s} | cfgs={t['n_configs']} | G={float(t['G']):+.3f} | fam={float(t['fam_med']):.2e} | ratio={float(t['ratio_raw']):.2e}")


def main():
    bench = run_benchmark()
    evaluate_exceedance(bench)


if __name__ == "__main__":
    main()

