#!/usr/bin/env python3
"""audit_review_findings.py — reproduce the review's central checks.

This is an audit script, not a manuscript analysis. It verifies four claims
against the deposited curated event tables and the production density layer:

  1. The exact-match subset comparison (volume vs composition), including the
     within-subset baseline that the manuscript does not report.
  2. The study-density association and density quartiles.
  3. The <=5 vs >=6 row-count contrast.
  4. The Table 4 layer values (coverage vs water-only).

It uses the production ``cluster_boot_block`` and ``load_layers`` code.

Run with a Python that has networkx installed:
    python3 code/audit_review_findings.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _sub in ("code", "code/core", "code/ees"):
    sys.path.insert(0, os.path.join(ROOT, _sub))

from composition_matched_control import cluster_boot_block  # noqa: E402
from revision_layers import load_layers  # noqa: E402
EXCLUDE = ("B175", "B111", "B095", "B182", "B239")

RESULTS = os.path.join(ROOT, "code", "results", "wp3_clean")
if not os.path.exists(os.path.join(RESULTS, "coverage_events.csv")):
    RESULTS = os.path.join(ROOT, "results", "wp3_clean")
OUT = os.path.join(ROOT, "results", "review_audit")
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_BOOT = 2000
N_PAIRED = 10000


def load_curated():
    vol = pd.read_csv(os.path.join(RESULTS, "coverage_events.csv"))
    comp = pd.read_csv(os.path.join(RESULTS, "composition_matched_control.csv"))
    return vol, comp


def block_and_matched(vol, comp=None, papers=None):
    """Return per-paper block and matched medians."""
    b = vol[vol["arm"] == "block"].groupby("paper")["med_drift"].median()
    m = vol[vol["arm"] == "matched"].groupby("paper")["med_drift"].median()
    out = pd.DataFrame({"block": b, "volume_matched": m})
    if comp is not None:
        c = (comp[comp["arm"] == "composition_matched"]
             .groupby("paper")["med_drift"].median())
        out["composition_matched"] = c
    if papers is not None:
        out = out.loc[papers]
    return out.dropna()


def bootstrap_median_diff(values, n_boot=N_PAIRED, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, float)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.median(vals[rng.integers(0, len(vals), len(vals))])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def bootstrap_ols(y, X, groups, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap OLS coefficients, resampling whole papers."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups)))
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[j]] for j in pick])
        boots.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0])
    boots = np.asarray(boots)
    return np.quantile(boots, [0.025, 0.975], axis=0)


def exact_subset(vol, comp):
    exact = (comp[comp["arm"] == "composition_matched"]
             .groupby("paper")["exact_fraction"].mean())
    papers = sorted(exact[exact > 0].index)
    block = vol[(vol["arm"] == "block") & (vol["paper"].isin(papers))][
        ["paper", "med_drift", "n_rows_deleted", "arm"]].copy()
    vmatch = vol[(vol["arm"] == "matched") & (vol["paper"].isin(papers))][
        ["paper", "med_drift", "n_rows_deleted", "arm"]].copy()
    cmatch = comp[(comp["arm"] == "composition_matched") & (comp["paper"].isin(papers))][
        ["paper", "med_drift", "n_rows_deleted", "arm"]].copy()

    bf_vol, lo_vol, hi_vol = cluster_boot_block(
        pd.concat([block, vmatch], ignore_index=True), n_boot=N_BOOT)
    bf_comp, lo_comp, hi_comp = cluster_boot_block(
        pd.concat([block, cmatch], ignore_index=True), n_boot=N_BOOT)

    ratios = block_and_matched(vol, comp, papers)
    rv = ratios["block"] / ratios["volume_matched"]
    rc = ratios["block"] / ratios["composition_matched"]
    diff = rc - rv
    d_lo, d_hi = bootstrap_median_diff(diff)
    p = stats.wilcoxon(rc, rv).pvalue
    return dict(
        n=len(papers),
        mean_exact_fraction=float(exact.loc[papers].mean()),
        volume_block_factor=float(bf_vol),
        volume_ci=[float(lo_vol), float(hi_vol)],
        composition_block_factor=float(bf_comp),
        composition_ci=[float(lo_comp), float(hi_comp)],
        volume_median_ratio=float(np.median(rv)),
        composition_median_ratio=float(np.median(rc)),
        paired_diff=float(np.median(diff)),
        paired_diff_ci=[float(d_lo), float(d_hi)],
        paired_wilcoxon_p=float(p),
        n_increase=int(np.sum(rc > rv)),
    )


def density_analysis(vol):
    L = load_layers()
    keep = ~np.isin(L.paper, EXCLUDE)
    rho = pd.Series(L.rho[keep], index=L.paper[keep]).groupby(level=0).mean()
    ratios = block_and_matched(vol)
    ratios["rho"] = rho.reindex(ratios.index)
    ratios = ratios.dropna()
    ratios["rows"] = vol[vol["arm"] == "block"].groupby("paper")["n_rows_deleted"].first()
    ratios = ratios.dropna()
    r = ratios["block"] / ratios["volume_matched"]
    rho_s = stats.spearmanr(ratios["rho"], r)
    pear = stats.pearsonr(np.log(r), ratios["rho"])
    ratios["q"] = pd.qcut(ratios["rho"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    q = ratios.groupby("q", observed=True).agg(
        n=("block", "size"),
        median_ratio=("block", lambda s: float(np.median(s / ratios.loc[s.index, "volume_matched"]))),
        median_rows=("rows", "median"),
        median_rho=("rho", "median"),
    )
    y = np.log(r.to_numpy(float))
    X = np.column_stack([np.ones(len(ratios)), np.log(ratios["rows"].to_numpy(float)),
                         ratios["rho"].to_numpy(float)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ci = bootstrap_ols(y, X, ratios.index.to_numpy())
    return dict(
        n=len(ratios),
        spearman_rho=float(rho_s.statistic),
        spearman_p=float(rho_s.pvalue),
        pearson_log_rho=float(pear.statistic),
        pearson_log_p=float(pear.pvalue),
        quartiles={k: dict(v) for k, v in q.iterrows()},
        rows_coef=float(beta[1]),
        rows_ci=[float(ci[0, 1]), float(ci[1, 1])],
        rho_coef=float(beta[2]),
        rho_ci=[float(ci[0, 2]), float(ci[1, 2])],
        rows_vs_rho=float(stats.spearmanr(ratios["rows"], ratios["rho"]).statistic),
        rows_vs_rho_p=float(stats.spearmanr(ratios["rows"], ratios["rho"]).pvalue),
    ), ratios


def row_count_analysis(ratios):
    r = ratios["block"] / ratios["volume_matched"]
    small = r[ratios["rows"] <= 5].to_numpy(float)
    large = r[ratios["rows"] >= 6].to_numpy(float)
    diff = float(np.median(large) - np.median(small))
    # Paired-free median difference CI: resample each group.
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_PAIRED):
        a = rng.choice(small, len(small), replace=True)
        b = rng.choice(large, len(large), replace=True)
        boot.append(np.median(b) - np.median(a))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(
        n_small=int(len(small)), n_large=int(len(large)),
        median_small=float(np.median(small)),
        median_large=float(np.median(large)),
        median_diff=diff, median_diff_ci=[float(lo), float(hi)],
        mannwhitney_p=float(stats.mannwhitneyu(small, large, alternative="two-sided").pvalue),
        spearman_rows=float(stats.spearmanr(ratios["rows"], r).statistic),
        spearman_rows_p=float(stats.spearmanr(ratios["rows"], r).pvalue),
    )


def table4_layers():
    cov = pd.read_csv(os.path.join(RESULTS, "coverage_events.csv"))
    wat = pd.read_csv(os.path.join(RESULTS, "coverage_water_events.csv"))
    rows = {}
    for p in ["B063", "B020", "B128", "B084", "B013"]:
        rows[p] = {}
        for label, d in (("coverage", cov), ("water", wat)):
            g = d[(d["paper"] == p) & (d["arm"] == "block")]
            if len(g):
                g = g.iloc[0]
                rows[p][label] = dict(
                    rows=int(g["n_rows_deleted"]),
                    drift=float(g["med_drift"] * 1e3),
                    grade=float(g["quartile_flip_pct"]),
                )
            else:
                rows[p][label] = None
    return rows


def main():
    vol, comp = load_curated()
    exact = exact_subset(vol, comp)
    dens, ratios = density_analysis(vol)
    rows = row_count_analysis(ratios)
    t4 = table4_layers()

    # Density of the exact subset (uses the same production rho).
    L = load_layers()
    keep = ~np.isin(L.paper, EXCLUDE)
    rho = pd.Series(L.rho[keep], index=L.paper[keep]).groupby(level=0).mean()
    ex = comp[comp["arm"] == "composition_matched"].groupby("paper")["exact_fraction"].mean()
    ex_papers = sorted(ex[ex > 0].index)
    exact_density = dict(
        exact_mean=float(rho.loc[ex_papers].mean()),
        other_mean=float(rho.drop(ex_papers).mean()),
        exact_median=float(rho.loc[ex_papers].median()),
        other_median=float(rho.drop(ex_papers).median()),
    )

    out = dict(exact=exact, density=dens, row_count=rows,
               exact_density=exact_density, table4=t4)
    with open(os.path.join(OUT, "review_findings.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print("== Exact-match subset ==")
    print(f"n={exact['n']}, mean exact fraction={exact['mean_exact_fraction']:.3f}")
    print(f"volume block factor = {exact['volume_block_factor']:.3f} "
          f"({exact['volume_ci'][0]:.3f}-{exact['volume_ci'][1]:.3f})")
    print(f"composition block factor = {exact['composition_block_factor']:.3f} "
          f"({exact['composition_ci'][0]:.3f}-{exact['composition_ci'][1]:.3f})")
    print(f"median ratio: volume={exact['volume_median_ratio']:.3f}, "
          f"composition={exact['composition_median_ratio']:.3f}, "
          f"diff={exact['paired_diff']:.3f} "
          f"({exact['paired_diff_ci'][0]:.3f}-{exact['paired_diff_ci'][1]:.3f}), "
          f"p={exact['paired_wilcoxon_p']:.3f}, "
          f"increase={exact['n_increase']}/{exact['n']}")
    print("\n== Density ==")
    print(f"Spearman rho={dens['spearman_rho']:.3f} (p={dens['spearman_p']:.2g}); "
          f"Pearson log r={dens['pearson_log_rho']:.3f} (p={dens['pearson_log_p']:.2g})")
    for k, v in dens["quartiles"].items():
        print(f"{k}: n={int(v['n'])}, median ratio={v['median_ratio']:.3f}, "
              f"median rows={v['median_rows']:.1f}, median rho={v['median_rho']:.3f}")
    print(f"joint model: rows={dens['rows_coef']:.3f} "
          f"({dens['rows_ci'][0]:.3f}-{dens['rows_ci'][1]:.3f}); "
          f"rho={dens['rho_coef']:.3f} "
          f"({dens['rho_ci'][0]:.3f}-{dens['rho_ci'][1]:.3f}); "
          f"rows-vs-rho={dens['rows_vs_rho']:.3f} (p={dens['rows_vs_rho_p']:.3f})")
    print(f"exact density: mean {exact_density['exact_mean']:.3f} vs "
          f"{exact_density['other_mean']:.3f}; median "
          f"{exact_density['exact_median']:.3f} vs {exact_density['other_median']:.3f}")
    print("\n== Row count ==")
    print(f"<=5: n={rows['n_small']}, median={rows['median_small']:.3f}; "
          f">=6: n={rows['n_large']}, median={rows['median_large']:.3f}; "
          f"diff={rows['median_diff']:.3f} "
          f"({rows['median_diff_ci'][0]:.3f}-{rows['median_diff_ci'][1]:.3f}), "
          f"MWU p={rows['mannwhitney_p']:.2g}; "
          f"Spearman rows={rows['spearman_rows']:.3f} (p={rows['spearman_rows_p']:.2g})")
    print("\n== Table 4 layer check ==")
    for p, v in t4.items():
        print(f"{p}: coverage={v['coverage']}; water={v['water']}")
    print(f"\nwrote {os.path.relpath(os.path.join(OUT, 'review_findings.json'), ROOT)}")

    # Assertions against the numbers used in the review.
    checks = [
        ("exact n=21", exact["n"] == 21),
        ("exact mean 0.613", abs(exact["mean_exact_fraction"] - 0.613) < 0.01),
        ("volume block 0.879", abs(exact["volume_block_factor"] - 0.879) < 0.01),
        ("composition block 1.071", abs(exact["composition_block_factor"] - 1.071) < 0.01),
        ("volume ratio 0.806", abs(exact["volume_median_ratio"] - 0.806) < 0.01),
        ("composition ratio 1.000", abs(exact["composition_median_ratio"] - 1.000) < 0.01),
        ("paired diff 0.194", abs(exact["paired_diff"] - 0.194) < 0.01),
        ("paired p < 0.02", exact["paired_wilcoxon_p"] < 0.02),
        ("density rho 0.442", abs(dens["spearman_rho"] - 0.442) < 0.02),
        ("Q4 ratio 2.606", abs(dens["quartiles"]["Q4"]["median_ratio"] - 2.606) < 0.05),
        ("rows coef 0.294", abs(dens["rows_coef"] - 0.294) < 0.05),
        ("rho coef 3.87", abs(dens["rho_coef"] - 3.87) < 0.1),
        ("<=5 median 1.165", abs(rows["median_small"] - 1.165) < 0.01),
        (">=6 median 1.899", abs(rows["median_large"] - 1.899) < 0.02),
        ("MWU p < 0.001", rows["mannwhitney_p"] < 0.001),
    ]
    print("\n== Checks ==")
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        ok = ok and passed
    if not ok:
        sys.exit("[audit] one or more checks failed")


if __name__ == "__main__":
    main()
