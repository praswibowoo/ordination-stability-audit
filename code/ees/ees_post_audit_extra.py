#!/usr/bin/env python3
"""Extra inferential checks for the EES panel-collapse revision.

1. Paired full-versus-collapsed ratio difference.
2. Paired full-versus-collapsed block-factor difference (cluster bootstrap).
3. Minimum detectable effect for the collapsed null.
4. Row-weighted (1/panel multiplicity) audit.
5. Collapsed extraction-error perturbation.

Writes results/post_audit/extra.json.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from core_revision_checks import load_curated  # noqa: E402
from ees_post_audit import (  # noqa: E402
    block_model, collapse_panels, event_metrics, ratio_summary, summarise,
)

SEED = 42
R = os.path.join(ROOT, "results", "post_audit")
OUT = os.path.join(R, "extra.json")
N_BOOT = 10000


def median_ci(values, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.median(x[rng.integers(0, len(x), len(x))])
    return {
        "median": float(np.median(x)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "n": int(len(x)),
    }


def paired_diff(a, b, n_boot=N_BOOT, seed=SEED):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.median(d[rng.integers(0, len(d), len(d))])
    from scipy import stats
    return {
        "median_diff": float(np.median(d)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
        "n": int(len(d)),
    }


def weighted_pca(X, w, dims=2):
    """Weighted PCA via weighted centring and row-scaled SVD."""
    w = np.asarray(w, float)
    w = w / w.sum()
    mu = (X * w[:, None]).sum(0)
    Xc = X - mu
    Xw = Xc * np.sqrt(w[:, None])
    U, S, _ = np.linalg.svd(Xw, full_matrices=False)
    return U[:, :dims] * S[:dims]


def row_weights(X, paper):
    """Weight each row by 1 / within-paper configuration multiplicity."""
    counts = Counter()
    for p, row in zip(paper, X):
        counts[(p, tuple(row.astype(int)))] += 1
    w = np.asarray([1.0 / counts[(p, tuple(row.astype(int)))]
                    for p, row in zip(paper, X)], float)
    return w


def run_weighted(X, rho, paper, draws=100, dims=2):
    w = row_weights(X, paper)
    S_full = weighted_pca(X, w, dims)
    n = len(X)
    uniq = sorted(set(paper))
    block_rows, matched_rows = [], []
    for p in uniq:
        idx = np.where(paper == p)[0]
        n_p = int(len(idx))
        keep = np.ones(n, bool)
        keep[idx] = False
        keep_idx = np.where(keep)[0]
        mb = event_metrics(S_full, weighted_pca(X[keep], w[keep], dims),
                           keep_idx, dims=dims, align="sign")
        mb.update(paper=p, arm="block", n_rows_deleted=n_p, draw=-1)
        block_rows.append(mb)
        donors = np.where(paper != p)[0]
        base = SEED + __import__("zlib").crc32(str(p).encode()) % 100000
        for dr in range(draws):
            rng = np.random.default_rng(base + 7919 * dr)
            drop = rng.choice(donors, size=n_p, replace=False)
            keep2 = np.ones(n, bool)
            keep2[drop] = False
            idx2 = np.where(keep2)[0]
            mm = event_metrics(S_full, weighted_pca(X[keep2], w[keep2], dims),
                               idx2, dims=dims, align="sign")
            mm.update(paper=p, arm="matched", n_rows_deleted=n_p, draw=dr)
            matched_rows.append(mm)
    return pd.DataFrame(block_rows), pd.DataFrame(matched_rows)



def draw_weight_matched(donors, w, target, rng):
    """Choose donor rows with total weight close to target."""
    order = rng.permutation(donors)
    chosen = []
    total = 0.0
    for j in order:
        if total >= target - 1e-9:
            break
        if total + w[j] <= target + 1e-9:
            chosen.append(int(j))
            total += w[j]
    if total < target - 1e-9:
        # allow a small overshoot with the smallest remaining row
        remaining = [j for j in order if int(j) not in set(chosen)]
        if remaining:
            j = min(remaining, key=lambda k: w[k])
            chosen.append(int(j))
            total += w[j]
    return np.asarray(chosen, int), float(total)


def run_weight_matched(X, rho, paper, draws=100, dims=2):
    w = row_weights(X, paper)
    S_full = weighted_pca(X, w, dims)
    n = len(X)
    uniq = sorted(set(paper))
    block_rows, matched_rows = [], []
    for p in uniq:
        idx = np.where(paper == p)[0]
        target = float(w[idx].sum())
        keep = np.ones(n, bool)
        keep[idx] = False
        keep_idx = np.where(keep)[0]
        mb = event_metrics(S_full, weighted_pca(X[keep], w[keep], dims),
                           keep_idx, dims=dims, align="sign")
        mb.update(paper=p, arm="block", n_rows_deleted=len(idx),
                  weight_deleted=target, draw=-1)
        block_rows.append(mb)
        donors = np.where(paper != p)[0]
        base = SEED + __import__("zlib").crc32(str(p).encode()) % 100000
        for dr in range(draws):
            rng = np.random.default_rng(base + 7919 * dr)
            drop, actual = draw_weight_matched(donors, w, target, rng)
            keep2 = np.ones(n, bool)
            keep2[drop] = False
            idx2 = np.where(keep2)[0]
            mm = event_metrics(S_full, weighted_pca(X[keep2], w[keep2], dims),
                               idx2, dims=dims, align="sign")
            mm.update(paper=p, arm="matched", n_rows_deleted=len(drop),
                      weight_deleted=actual, draw=dr)
            matched_rows.append(mm)
    return pd.DataFrame(block_rows), pd.DataFrame(matched_rows)

def design(block, matched):
    ev = pd.concat([block, matched], ignore_index=True)
    y = np.log(ev["med_drift"].to_numpy(float))
    lv = np.log(ev["n_rows_deleted"].to_numpy(float))
    lv = lv - lv.mean()
    blk = (ev["arm"] == "block").to_numpy(float)
    Xd = np.column_stack([np.ones(len(ev)), lv, blk, blk * lv])
    return Xd, y, ev["paper"].to_numpy(dtype=object)


def fit_beta(Xd, y):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta[2]


def boot_block_diff(full_block, full_matched, coll_block, coll_matched, n_boot=1000):
    papers = sorted(set(full_block["paper"]) & set(coll_block["paper"]))
    rng = np.random.default_rng(SEED)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(papers, len(papers), replace=True)
        # build event frames for the sampled papers (duplicates allowed)
        fb = pd.concat([full_block[full_block.paper == p] for p in pick],
                       ignore_index=True)
        fm = pd.concat([full_matched[full_matched.paper == p] for p in pick],
                       ignore_index=True)
        cb = pd.concat([coll_block[coll_block.paper == p] for p in pick],
                       ignore_index=True)
        cm = pd.concat([coll_matched[coll_matched.paper == p] for p in pick],
                       ignore_index=True)
        Xf, yf, _ = design(fb, fm)
        Xc, yc, _ = design(cb, cm)
        diffs[b] = fit_beta(Xf, yf) - fit_beta(Xc, yc)
    return {
        "median_diff_log": float(np.median(diffs)),
        "ci_lo": float(np.quantile(diffs, 0.025)),
        "ci_hi": float(np.quantile(diffs, 0.975)),
        "ratio_of_factors": float(np.exp(np.median(diffs))),
        "ratio_lo": float(np.exp(np.quantile(diffs, 0.025))),
        "ratio_hi": float(np.exp(np.quantile(diffs, 0.975))),
        "n_boot": n_boot,
    }


def mde():
    b = pd.read_csv(os.path.join(R, "collapsed_block.csv")).set_index("paper")
    m = (pd.read_csv(os.path.join(R, "collapsed_matched.csv"))
         .groupby("paper")["med_drift"].median())
    ratio = (b["med_drift"] / m).dropna()
    logr = np.log(ratio.to_numpy(float))
    n = len(logr)
    sd = float(np.std(logr, ddof=1))
    # two-sided alpha 0.05, 80% power, paired t-test approximation
    mde_log = (1.959964 + 0.841621) * sd / np.sqrt(n)
    return {
        "n": n,
        "sd_log_ratio": sd,
        "mde_log": float(mde_log),
        "mde_ratio": float(np.exp(mde_log)),
        "observed_median_ratio": float(np.exp(np.median(logr))),
        "observed_iqr_log": [float(np.quantile(logr, 0.25)),
                             float(np.quantile(logr, 0.75))],
    }


def perturbed_collapse(X, rho, paper, rate, rng):
    Xp = X.copy()
    present = Xp == 1
    flip = rng.random(Xp.shape) < rate
    Xp[present & flip] = 0.0
    rho_p = Xp.mean(1)
    Xc, rhoc, paperc, _ = collapse_panels(Xp, rho_p, paper)
    return Xc, rhoc, paperc


def main():
    X, rho, paper = load_curated()
    full_b = pd.read_csv(os.path.join(R, "baseline_block.csv"))
    full_m = pd.read_csv(os.path.join(R, "baseline_matched.csv"))
    coll_b = pd.read_csv(os.path.join(R, "collapsed_block.csv"))
    coll_m = pd.read_csv(os.path.join(R, "collapsed_matched.csv"))

    fb = full_b.set_index("paper")["med_drift"]
    fm = full_m.groupby("paper")["med_drift"].median()
    cb = coll_b.set_index("paper")["med_drift"]
    cm = coll_m.groupby("paper")["med_drift"].median()
    common = sorted(set(fb.index) & set(cb.index))
    full_ratio = (fb / fm).loc[common]
    coll_ratio = (cb / cm).loc[common]

    out = {
        "paired_ratio_difference": paired_diff(full_ratio, coll_ratio),
        "full_ratio": ratio_summary(full_ratio),
        "collapsed_ratio": ratio_summary(coll_ratio),
        "block_factor_difference": boot_block_diff(full_b, full_m, coll_b, coll_m),
        "mde": mde(),
    }

    # PC1/PC2/joint difference CIs are already in core.json.
    core = json.load(open(os.path.join(R, "core.json")))
    out["pc_differences"] = {
        "full": {
            "pc1": core["baseline"]["audit"]["baseline_pc1_flip"],
            "pc2": core["baseline"]["audit"]["baseline_pc2_flip"],
            "joint": core["baseline"]["audit"]["baseline_joint_flip"],
        },
        "collapsed": {
            "pc1": core["collapsed"]["audit"]["collapsed_pc1_flip"],
            "pc2": core["collapsed"]["audit"]["collapsed_pc2_flip"],
            "joint": core["collapsed"]["audit"]["collapsed_joint_flip"],
        },
    }

    # Row-weighted audit.
    wb, wm = run_weighted(X, rho, paper, draws=100, dims=2)
    out["weighted_naive"] = summarise(wb, wm, "weighted_naive_")
    wb.to_csv(os.path.join(R, "weighted_block.csv"), index=False)
    wm.to_csv(os.path.join(R, "weighted_matched.csv"), index=False)

    wmb, wmm = run_weight_matched(X, rho, paper, draws=100, dims=2)
    out["weighted_matched"] = summarise(wmb, wmm, "weighted_matched_")
    out["weighted_matched"]["weight_balance"] = {
        "mean_block_weight": float(wmb["weight_deleted"].mean()),
        "mean_matched_weight": float(wmm.groupby("paper")["weight_deleted"].median().mean()),
    }
    wmb.to_csv(os.path.join(R, "weighted_matched_block.csv"), index=False)
    wmm.to_csv(os.path.join(R, "weighted_matched_matched.csv"), index=False)

    # Collapsed extraction-error perturbation (11.1% present -> NR).
    reps = []
    for r in range(3):
        rng = np.random.default_rng(SEED + 1000 + r)
        Xc, rhoc, paperc = perturbed_collapse(X, rho, paper, 0.111, rng)
        b, m = __import__("ees_post_audit").run_audit(
            Xc, rhoc, paperc,
            lambda Z: __import__("core_revision_checks").fit(Z),
            dims=2, draws=20, align="sign", label="perturbed")
        reps.append(summarise(b, m, f"rep{r}_"))
    out["collapsed_perturbation"] = reps

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
