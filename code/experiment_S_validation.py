#!/usr/bin/env python3
"""experiment_S_validation.py — method-paper validation (corrected pass, Phase F1 power sweep).

Two validations of the stability framework:

S1  Resampling-unit calibration. Synthetic paper-structured corpora built on the
    real corpus geometry (122 papers, per-paper row counts and densities). Drift
    is generated with paper-level density effects of increasing size. For each
    corpus the SI is tested with (a) the row-level permutation used in the
    original C1 implementation and (b) a paper-level permutation. Reports
    rejection rates at alpha = 0.05 to show the unit-of-analysis effect.

S2  Random-family negative control on the real corpus. The C2 paired stability
    gain G is recomputed for random 3-parameter families sampled to match the F6
    aeration regime (cover-row count 400-900 of 764). The observed F6 gain is
    located in this null distribution (percentile + one-sided p).

Reads frozen presence layer only. Writes:
  results/sim_calibration.csv     per-replicate rejection outcomes
  results/sim_random_families.csv per-random-family G
  figures/fig7_simulation.png/pdf

Run: python3 code/experiment_S_validation.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
SEED = 42

from experiment_C1 import load  # noqa: E402
from experiment_C2_stability import (bootstrap_G, paired_event_si_same_rows,
                                     event_si)  # noqa: E402


def si_fast(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    return float(np.median(drift[rho <= q1]) / np.median(drift[rho >= q3]))


def drift_contrast_pct(gamma, delta_rho):
    """Exact drift contrast across the density IQR induced by a planted effect.

    lambda_p = exp(gamma * (med(rho) - rho_p)), so a paper at the top of the
    density IQR and one at the bottom differ in mean drift by
    exp(gamma * delta_rho) - 1. Phase F1 replaces the previous linearised
    parenthetical (the computed value is contrast_pct in
    results/sim_calibration_summary.csv, 1.94% at gamma = 0.20; an earlier
    "approximate 3%" note was stale)
    with this closed form, evaluated on the empirical delta_rho.
    """
    return 100.0 * (np.exp(gamma * delta_rho) - 1.0)


def calibration(X, rho, paper, scenarios, n_rep=1000, n_perm=199):
    """S1 — rejection rates for row-level vs paper-level permutation tests.

    scenarios: list of (sigma_paper, gamma);
      sigma_paper = sd of a paper-level random effect in log drift (nuisance
      clustering present in real corpora), gamma = planted paper-density effect.
    """
    uniq = np.asarray(sorted(set(paper)))
    code = np.searchsorted(uniq, paper)
    prho = np.array([rho[code == j].mean() for j in range(len(uniq))])
    med = np.median(prho)
    # empirical density IQR of papers -> exact gamma -> % contrast conversion
    delta_rho = float(np.quantile(prho, 0.75) - np.quantile(prho, 0.25))
    # Monte Carlo SE of a rejection rate: ~0.015 at n_rep=1000 versus ~0.05 at the
    # previous n_rep=100, which is what made 0.010/0.050/0.030 indistinguishable
    # from nominal (Round-4 R4-9).
    mc_se = lambda p: float(np.sqrt(p * (1.0 - p) / n_rep))
    print(f"  paper-density IQR delta_rho = {delta_rho:.4f} -> "
          f"contrast(gamma) = exp(gamma*delta_rho) - 1; "
          f"contrast(0.20) = {drift_contrast_pct(0.20, delta_rho):.2f}%")
    rows = []
    for sigma, gamma in scenarios:
        for rep in range(n_rep):
            rng = np.random.default_rng(SEED + 1000 * rep + int(1000 * gamma)
                                        + int(100 * sigma))
            lam = np.exp(gamma * (med - prho))[code]
            u = rng.standard_normal(len(uniq))
            drift = lam * np.exp(sigma * u[code] + 0.4 * rng.standard_normal(len(rho)))
            obs_row = si_fast(rho, drift)
            obs_paper = si_fast(prho[code], drift)
            # row-level: rows exchangeable (rho permuted across rows)
            null_row = np.empty(n_perm)
            for b in range(n_perm):
                null_row[b] = si_fast(rng.permutation(rho), drift)
            p_row = float((np.sum(null_row > obs_row) + 1) / (n_perm + 1))
            # paper-level proper: observed and null both use paper mean density.
            # The legacy hybrid (row-weighted observed vs paper-weighted null)
            # is retained for traceability but is not the inferential test.
            null_blk = np.empty(n_perm)
            for b in range(n_perm):
                null_blk[b] = si_fast(rng.permutation(prho)[code], drift)
            p_blk = float((np.sum(null_blk > obs_paper) + 1) / (n_perm + 1))
            p_blk_hybrid = float((np.sum(null_blk > obs_row) + 1) / (n_perm + 1))
            rows.append(dict(sigma_paper=sigma, gamma=gamma, rep=rep,
                             SI=round(obs_row, 4), SI_paper=round(obs_paper, 4),
                             p_row=round(p_row, 4),
                             p_block=round(p_blk, 4),
                             p_block_hybrid=round(p_blk_hybrid, 4),
                             rej_row=int(p_row < 0.05),
                             rej_block=int(p_blk < 0.05),
                             rej_block_hybrid=int(p_blk_hybrid < 0.05)))
        sub = [r for r in rows if r["gamma"] == gamma and r["sigma_paper"] == sigma]
        p_row = float(np.mean([r["rej_row"] for r in sub]))
        p_blk = float(np.mean([r["rej_block"] for r in sub]))
        print(f"  sigma_paper={sigma:.1f} gamma={gamma:.2f} | row {p_row:.3f}"
              f"(+-{mc_se(p_row):.3f}) | block {p_blk:.3f}(+-{mc_se(p_blk):.3f}) "
              f"| contrast {drift_contrast_pct(gamma, delta_rho):5.2f}% "
              f"| n={len(sub)}")
    cal_meta = dict(delta_rho=delta_rho,
                    contrast_pct_at_gamma020=drift_contrast_pct(0.20, delta_rho),
                    mc_se_formula="sqrt(p*(1-p)/n_rep), n_rep=1000")
    return rows, cal_meta


def random_family_control(X, rho, paper, params, n_valid=100, seed=SEED):
    """S2 — random 3-parameter families matched to the F6 cover-row regime."""
    f6 = ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"]
    f6_cols = [params.index(p) for p in f6]
    f6_cover = X[:, f6_cols].sum(axis=1) >= 1
    f6_dens = float(X[f6_cover][:, f6_cols].mean())
    print(f"  F6 reference: cover={int(f6_cover.sum())} rows | mean slice density={f6_dens:.3f}")

    rng = np.random.default_rng(seed)
    seen = set()
    cand = []
    tries = 0
    while len(cand) < n_valid and tries < 60000:
        tries += 1
        cols = tuple(sorted(rng.choice(len(params), size=3, replace=False)))
        if cols in seen:
            continue
        seen.add(cols)
        cover = X[:, list(cols)].sum(axis=1) >= 1
        n_cov = int(cover.sum())
        if 400 <= n_cov <= 900:
            dens = float(X[cover][:, list(cols)].mean())
            cand.append((cols, n_cov, dens))
    print(f"  sampled {len(cand)} random-family candidates in {tries} draws")

    rows = []
    t0 = time.time()
    for i, (cols, n_cov, dens) in enumerate(cand):
        cover = X[:, list(cols)].sum(axis=1) >= 1
        idxs = np.where(cover)[0]
        pairs = paired_event_si_same_rows(
            X[np.ix_(idxs, list(cols))], X[np.ix_(idxs, range(len(params)))],
            rho[idxs], paper[idxs], model="pca", dim=2)
        if len(pairs) < 20:
            continue
        fam = np.asarray([p["SI_family"] for p in pairs])
        allp = np.asarray([p["SI_all"] for p in pairs])
        g = 1.0 - np.median(fam) / np.median(allp)
        boot = bootstrap_G(pairs, np.random.default_rng(SEED), n_boot=400)
        rows.append(dict(members="|".join(params[c].split(" (")[0] for c in cols),
                         n_cover=n_cov, n_papers=len(set(paper[idxs])),
                         mean_slice_density=round(dens, 4), n_pairs=len(pairs),
                         G=round(float(g), 4), P_G_gt0=round(boot["P_G_gt0"], 4)))
        if (i + 1) % 10 == 0:
            print(f"    {len(rows)}/{len(cand)} families done "
                  f"({time.time()-t0:.0f}s)")

    # F6 under the identical function
    f6_idxs = np.where(f6_cover)[0]
    f6_pairs = paired_event_si_same_rows(
        X[np.ix_(f6_idxs, f6_cols)], X[np.ix_(f6_idxs, range(len(params)))],
        rho[f6_idxs], paper[f6_idxs], model="pca", dim=2)
    f6_G = 1.0 - np.median([p["SI_family"] for p in f6_pairs]) / \
        np.median([p["SI_all"] for p in f6_pairs])
    gs = np.asarray([r["G"] for r in rows])
    p_full = float((1 + np.sum(gs >= f6_G)) / (1 + len(gs)))
    matched = np.asarray([r["G"] for r in rows
                          if r["mean_slice_density"] >= f6_dens - 0.10])
    p_match = (float((1 + np.sum(matched >= f6_G)) / (1 + len(matched)))
               if len(matched) else float("nan"))
    print(f"  F6 G = {f6_G:.4f} | random-family null n={len(gs)} "
          f"median={np.median(gs):.3f} | percentile={100*(1-p_full):.1f} "
          f"| one-sided p={p_full:.3f} | matched-n={len(matched)} p={p_match:.3f}")
    return rows, f6_G, f6_dens, p_full, p_match


def fig7(cal, rand, f6_G, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # print width (13.7 cm text block): (a) spans the top row, (b) and (c) sit below,
    # so every panel is wide enough for legible tick and axis labels.
    _nrep = len({r["rep"] for r in cal})
    fig = plt.figure(figsize=(5.6, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.72, wspace=0.42)
    def _bars(ax, keys, title, show_legend=False):
        labels, rr, rb = [], [], []
        for s_, g in keys:
            s_, g = float(s_), float(g)
            if g == 0 and s_ == 0:
                labels.append("null")
            elif s_ > 0 and g == 0:
                labels.append(f"$\\sigma$={s_:g}")
            else:
                labels.append(f"$\\gamma$={g:g}")
            sub = [r for r in cal
                   if float(r["sigma_paper"]) == s_ and float(r["gamma"]) == g]
            rr.append(np.mean([float(r["rej_row"]) for r in sub]))
            rb.append(np.mean([float(r["rej_block"]) for r in sub]))
        x = np.arange(len(keys))
        ax.bar(x - 0.2, rr, 0.4, color="#c62828", label="row-level test")
        ax.bar(x + 0.2, rb, 0.4, color="#1b4f72", label="paper-level test")
        ax.axhline(0.05, ls="--", c="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.2, rotation=35, ha="right")
        ax.set_ylabel("rejection rate", fontsize=8.2, labelpad=6)
        ax.tick_params(axis="y", labelsize=7.8)
        ax.set_ylim(0, 0.45)
        ax.set_title(title, fontsize=8.2, pad=7)
        if show_legend:
            ax.legend(fontsize=7.2, loc="upper left", framealpha=0.92, handlelength=1.2)

    keys = sorted({(r["sigma_paper"], r["gamma"]) for r in cal})
    size_keys = [(s, g) for (s, g) in keys if float(g) == 0]
    power_keys = [(s, g) for (s, g) in keys if float(g) > 0]
    ax = fig.add_subplot(gs[0, 0])
    _bars(ax, size_keys, "(a) size at $\\gamma$ = 0", True)
    ax = fig.add_subplot(gs[0, 1])
    _bars(ax, power_keys, "(b) power at $\\sigma$ = 0.8")
    ax = fig.add_subplot(gs[1, 0])
    gs_ = np.asarray([float(r["G"]) for r in rand])
    bins = np.linspace(-3.0, 1.2, 22)
    ax.hist(np.clip(gs_, -3.0, 1.2), bins=bins, color="#7fa6c9", edgecolor="white", lw=0.5)
    ax.axvline(f6_G, c="#c62828", lw=1.6, label=f"F6 G = {f6_G:.3f}")
    ax.set_xlabel("paired stability gain G", fontsize=8.2)
    ax.set_ylabel("random families", fontsize=8.2)
    ax.set_xlim(-3.2, 1.3)
    ax.tick_params(labelsize=7.5)
    n_exceed = int(np.sum(gs_ >= f6_G))
    p_val = (1 + n_exceed) / (1 + len(gs_))
    ax.set_title(f"(c) random-family null\n({n_exceed}/{len(gs_)} exceed; "
                 f"p = {p_val:.3f})",
                 fontsize=8, pad=4)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.92, handlelength=1.3)
    n_out = int(np.sum(gs_ < -3.0))

    ax = fig.add_subplot(gs[1, 1])
    n_cov = np.asarray([float(r["n_cover"]) for r in rand])
    in_range = gs_ >= -3.2
    ax.scatter(n_cov[in_range], gs_[in_range], s=10, c="#4c72b0", alpha=0.75,
               label="random families")
    ax.scatter([683], [f6_G], s=48, c="#c62828", marker="*", zorder=4, label="F6 aeration")
    ax.set_xlabel("cover rows", fontsize=8.2)
    ax.set_ylabel("G", fontsize=8.2)
    ax.set_ylim(-3.2, 1.3)
    ax.tick_params(labelsize=7.5)
    ax.set_title("(d) G vs family cover size", fontsize=8, pad=4)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.92, handlelength=1.3,
              markerscale=1.4)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import matplotlib  # noqa: F401  (ensure backend set before figure helpers)
    X, rho, paper, params, Xv, keysv, rho_v = load()
    print(f"presence {X.shape} | {len(set(paper))} papers")

    print("[S1] calibration ...")
    # Phase F1 grid: a sigma sweep at gamma=0 (type-I error under clustering),
    # then a gamma sweep at sigma=0.8 (the power curve). 11 settings x 1000.
    cal, cal_meta = calibration(
        X, rho, paper,
        scenarios=((0.0, 0.0), (0.2, 0.0), (0.5, 0.0), (0.8, 0.0),
                   (0.8, 0.02), (0.8, 0.05), (0.8, 0.10), (0.8, 0.20),
                   (0.8, 0.30), (0.8, 0.45), (0.8, 0.60)))
    with open(os.path.join(RESULTS, "sim_calibration.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal[0].keys()))
        w.writeheader(); w.writerows(cal)

    # per-setting summary with Monte Carlo SEs + the minimum detectable effect at 80% power
    summ = []
    for (sg, gm) in sorted({(float(r["sigma_paper"]), float(r["gamma"])) for r in cal}):
        sub = [r for r in cal if float(r["sigma_paper"]) == sg and float(r["gamma"]) == gm]
        p_row = float(np.mean([r["rej_row"] for r in sub]))
        p_blk = float(np.mean([r["rej_block"] for r in sub]))
        summ.append(dict(sigma_paper=sg, gamma=gm, n_rep=len(sub),
                         rej_row=p_row, rej_row_se=float(np.sqrt(p_row * (1 - p_row) / len(sub))),
                         rej_block=p_blk,
                         rej_block_se=float(np.sqrt(p_blk * (1 - p_blk) / len(sub))),
                         contrast_pct=drift_contrast_pct(gm, cal_meta["delta_rho"])))
    pw = [s for s in summ if s["sigma_paper"] == 0.8 and s["gamma"] > 0]
    mde80 = next((s for s in pw if s["rej_block"] >= 0.80), None)
    summ.append(dict(sigma_paper=0.8, gamma=-1, n_rep=0, rej_row=-1,
                     rej_row_se=-1, rej_block=-1, rej_block_se=-1,
                     contrast_pct=-1))
    with open(os.path.join(RESULTS, "sim_calibration_summary.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    if mde80:
        print(f"  minimum detectable effect at 80% paper-level power: "
              f"gamma = {mde80['gamma']} ({mde80['contrast_pct']:.2f}% drift contrast)")
    else:
        print("  no setting reached 80% paper-level power within the swept gamma range")

    print("[S2] random-family negative control ...")
    rand, f6_G, f6_dens, p_full, p_match = random_family_control(X, rho, paper, params)
    with open(os.path.join(RESULTS, "sim_random_families.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rand[0].keys()))
        w.writeheader(); w.writerows(rand)

    fig7(cal, rand, f6_G, os.path.join(FIGS, "fig7_simulation.png"))
    print("[fig7] written figures/fig7_simulation.png (+pdf)")


if __name__ == "__main__":
    main()
