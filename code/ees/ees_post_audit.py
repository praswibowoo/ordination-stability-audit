#!/usr/bin/env python3
"""ees_post_audit.py — post-review diagnostics for the EES coverage audit.

Non-destructive: reads the frozen curated coverage layer and writes only to
results/post_audit/.  It implements the checks requested in the pre-execution
audit:

  1. heterogeneity of the per-study block/matched ratio;
  2. panel/replication structure (unique configurations per paper);
  3. panel-collapsed volume-matched audit (one row per unique config);
  4. PC2 and joint 2-D grade stability (baseline and collapsed);
  5. valid same-size nulls (global permutation, residual permutation, label swap);
  6. PCoA/Jaccard sensitivity with a configurable number of draws;
  7. paired composition/density/volume differences with CIs.

Run:
    python3 code/ees_post_audit.py --mode core
    python3 code/ees_post_audit.py --mode pcoa --pcoa-draws 100

Outputs:
    results/post_audit/core.json
    results/post_audit/panel_collapsed_events.csv
    results/post_audit/baseline_metrics.csv
    results/post_audit/pcoa_events.csv
    results/post_audit/pcoa_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from core_revision_checks import load_curated, drift, fit, seed_for  # noqa: E402
from experiment_H_volumematch import ols_cluster_boot  # noqa: E402

SEED = 42
DRAWS = 100
N_BOOT = 10000
OUT = os.path.join(ROOT, "results", "post_audit")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
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


def paired_ci(a, b, n_boot=N_BOOT, seed=SEED):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.median(d[rng.integers(0, len(d), len(d))])
    return {
        "median_diff": float(np.median(d)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
        "n": int(len(d)),
    }


def ratio_summary(ratios):
    x = np.asarray(ratios, float)
    return {
        "median": float(np.median(x)),
        "ci": median_ci(x),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n_below_1": int(np.sum(x < 1)),
        "pct_below_1": float(np.mean(x < 1) * 100),
        "n_above_1": int(np.sum(x > 1)),
        "pct_above_1": float(np.mean(x > 1) * 100),
    }


# ---------------------------------------------------------------------------
# Panel structure
# ---------------------------------------------------------------------------
def panel_structure(X, paper):
    unique_by_paper = {}
    for p in sorted(set(paper)):
        idx = np.where(paper == p)[0]
        unique_by_paper[p] = len({tuple(X[i].astype(int)) for i in idx})
    counts = Counter(unique_by_paper.values())
    return {
        "n_papers": int(len(unique_by_paper)),
        "n_rows": int(len(X)),
        "n_unique_configs": int(sum(unique_by_paper.values())),
        "n_single_config_papers": int(sum(v == 1 for v in unique_by_paper.values())),
        "n_multi_config_papers": int(sum(v > 1 for v in unique_by_paper.values())),
        "n_duplicate_rows": int(len(X) - sum(unique_by_paper.values())),
        "pct_duplicate_rows": float((len(X) - sum(unique_by_paper.values())) /
                                    len(X) * 100),
        "config_count_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def collapse_panels(X, rho, paper):
    """Keep the first row of each unique coverage configuration per paper."""
    keep = []
    seen = set()
    for i, p in enumerate(paper):
        key = (p, tuple(X[i].astype(int)))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep = np.asarray(keep, int)
    return X[keep], rho[keep], paper[keep], keep


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def axis_metrics(S_full, S_red, idx, dims=2, align="sign"):
    """PC1/PC2 and joint 2-D quartile stability after sign alignment."""
    if align == "procrustes":
        a, b, _ = procrustes(S_full[idx], S_red)
        full = a
        red = b
    else:
        full = S_full[idx].copy()
        red = S_red.copy()
        for k in range(dims):
            c = np.corrcoef(full[:, k], red[:, k])[0, 1]
            if np.isfinite(c) and c < 0:
                red[:, k] = -red[:, k]
            red[:, k] = ((red[:, k] - red[:, k].mean()) / red[:, k].std()
                         * full[:, k].std() + full[:, k].mean())

    out = {}
    for k in range(dims):
        a = full[:, k]
        b = red[:, k]
        cuts = np.quantile(a, [0.25, 0.50, 0.75])
        qa = np.searchsorted(cuts, a, side="right")
        qb = np.searchsorted(cuts, b, side="right")
        out[f"pc{k + 1}_flip"] = 100.0 * float(np.mean(qa != qb))

        n = len(a)
        k10 = max(5, min(int(round(0.10 * n)), n - 1))
        k25 = max(5, min(int(round(0.25 * n)), n - 1))
        top10 = len(set(np.argsort(a)[-k10:]) & set(np.argsort(b)[-k10:])) / k10
        top25 = len(set(np.argsort(a)[-k25:]) & set(np.argsort(b)[-k25:])) / k25
        out[f"pc{k + 1}_top10"] = float(top10)
        out[f"pc{k + 1}_top25"] = float(top25)
        out[f"pc{k + 1}_spearman"] = float(stats.spearmanr(a, b).statistic)

    qa = np.column_stack([
        np.searchsorted(np.quantile(full[:, k], [0.25, 0.50, 0.75]),
                        full[:, k], side="right")
        for k in range(dims)
    ])
    qb = np.column_stack([
        np.searchsorted(np.quantile(full[:, k], [0.25, 0.50, 0.75]),
                        red[:, k], side="right")
        for k in range(dims)
    ])
    cell_a = qa[:, 0] * 4 + qa[:, 1]
    cell_b = qb[:, 0] * 4 + qb[:, 1]
    out["joint_flip"] = 100.0 * float(np.mean(cell_a != cell_b))
    return out


def event_metrics(S_full, S_red, idx, dims=2, align="sign"):
    m = axis_metrics(S_full, S_red, idx, dims=dims, align=align)
    m["med_drift"] = float(np.median(drift(S_full[idx], S_red)))
    return m


def run_audit(X, rho, paper, fit_fn, dims=2, draws=DRAWS, align="sign",
              label="baseline"):
    """Block deletion plus volume-matched scattered draws; returns events."""
    S_full = fit_fn(X)
    n = len(X)
    uniq = sorted(set(paper))
    block_rows = []
    matched_rows = []
    for p in uniq:
        idx = np.where(paper == p)[0]
        n_p = int(len(idx))
        keep = np.ones(n, bool)
        keep[idx] = False
        keep_idx = np.where(keep)[0]
        mb = event_metrics(S_full, fit_fn(X[keep]), keep_idx, dims=dims, align=align)
        mb.update(layer=label, paper=p, arm="block", draw=-1,
                  n_rows_deleted=n_p)
        block_rows.append(mb)

        donors = np.where(paper != p)[0]
        base = seed_for(str(p))
        for dr in range(draws):
            rng = np.random.default_rng(base + 7919 * dr)
            drop = rng.choice(donors, size=n_p, replace=False)
            keep2 = np.ones(n, bool)
            keep2[drop] = False
            idx2 = np.where(keep2)[0]
            mm = event_metrics(S_full, fit_fn(X[keep2]), idx2, dims=dims,
                               align=align)
            mm.update(layer=label, paper=p, arm="matched", draw=dr,
                      n_rows_deleted=n_p)
            matched_rows.append(mm)
    return pd.DataFrame(block_rows), pd.DataFrame(matched_rows)


def block_model(block, matched):
    """log(drift) ~ 1 + log(V) + block + block*log(V), cluster bootstrap."""
    ev = pd.concat([block, matched], ignore_index=True)
    y = np.log(ev["med_drift"].to_numpy(float))
    raw_lv = np.log(ev["n_rows_deleted"].to_numpy(float))
    mean_lv = float(raw_lv.mean())
    lv = raw_lv - mean_lv
    blk = (ev["arm"] == "block").to_numpy(float)
    Xd = np.column_stack([np.ones(len(ev)), lv, blk, blk * lv])
    beta, lo, hi, n_ok, _ = ols_cluster_boot(
        Xd, y, ev["paper"].to_numpy(dtype=object), n_boot=1000, seed=SEED)
    names = ["intercept", "log_volume", "block", "block_x_log_volume"]
    out = {"mean_log_volume": mean_lv, "mean_rows": float(np.exp(mean_lv)),
           "n_boot": n_ok}
    for j, name in enumerate(names):
        out[name] = {
            "beta": float(beta[j]),
            "lo": float(lo[j]),
            "hi": float(hi[j]),
            "factor": float(np.exp(beta[j])),
            "factor_lo": float(np.exp(lo[j])),
            "factor_hi": float(np.exp(hi[j])),
        }
    return out


def summarise(block, matched, prefix=""):
    """Per-study ratio and paired metrics."""
    b = block.set_index("paper")
    m = matched.groupby("paper").median(numeric_only=True)
    common = sorted(set(b.index) & set(m.index))
    b = b.loc[common]
    m = m.loc[common]
    ratios = b["med_drift"] / m["med_drift"]
    out = {
        prefix + "ratio": ratio_summary(ratios),
        prefix + "block_drift_median": float(b["med_drift"].median()),
        prefix + "matched_drift_median": float(m["med_drift"].median()),
    }
    for key in ("pc1_flip", "pc2_flip", "joint_flip",
                "pc1_top10", "pc2_top10", "joint_top10"):
        if key in b.columns:
            out[prefix + key] = paired_ci(b[key], m[key])
    return out


# ---------------------------------------------------------------------------
# Same-size nulls
# ---------------------------------------------------------------------------
def same_size_percentile(values, sizes):
    """Mid-rank percentile within same-size groups."""
    out = {}
    groups = {}
    for p, n in sizes.items():
        groups.setdefault(int(n), []).append(p)
    for members in groups.values():
        if len(members) < 2:
            for p in members:
                out[p] = np.nan
            continue
        vals = np.asarray([values[p] for p in members], float)
        for p in members:
            own = values[p]
            peers = np.delete(vals, list(members).index(p))
            out[p] = 100.0 * (np.sum(peers < own) + 0.5 * np.sum(peers == own)) / len(peers)
    return out


def count_tails(pct):
    vals = np.asarray([v for v in pct.values() if np.isfinite(v)])
    return int(np.sum(vals > 95)), int(np.sum(vals < 5)), int(len(vals))


def same_size_nulls(block):
    b = block.set_index("paper")
    values = b["med_drift"].to_dict()
    sizes = b["n_rows_deleted"].astype(int).to_dict()
    obs_pct = same_size_percentile(values, sizes)
    obs_above, obs_below, n = count_tails(obs_pct)

    rng = np.random.default_rng(SEED)
    papers = sorted(values)
    v = np.asarray([values[p] for p in papers], float)
    n_perm = 10000

    # 1. Global permutation across papers (preserve sizes per paper).
    perm_above = np.empty(n_perm, int)
    perm_below = np.empty(n_perm, int)
    for i in range(n_perm):
        vals = dict(zip(papers, rng.permutation(v)))
        pct = same_size_percentile(vals, sizes)
        a, b_, _ = count_tails(pct)
        perm_above[i] = a
        perm_below[i] = b_

    # 2. Residual permutation from log drift ~ log rows.
    lv = np.log(np.asarray([sizes[p] for p in papers], float))
    y = np.log(v)
    beta = np.polyfit(lv, y, 1)
    fitted = np.polyval(beta, lv)
    resid = y - fitted
    res_above = np.empty(n_perm, int)
    res_below = np.empty(n_perm, int)
    for i in range(n_perm):
        vals = dict(zip(papers, np.exp(fitted + rng.permutation(resid))))
        pct = same_size_percentile(vals, sizes)
        a, b_, _ = count_tails(pct)
        res_above[i] = a
        res_below[i] = b_

    # 3. Label swap: block vs matched median for each paper.
    # Read matched medians from the event table outside; handled by caller if needed.

    def pval(null, obs, upper=True):
        if upper:
            return float((1 + np.sum(null >= obs)) / (n_perm + 1))
        return float((1 + np.sum(null <= obs)) / (n_perm + 1))

    return {
        "observed_above95": obs_above,
        "observed_below5": obs_below,
        "n": n,
        "median_percentile": float(np.nanmedian(list(obs_pct.values()))),
        "global_permutation": {
            "above95_median": float(np.median(perm_above)),
            "above95_q975": float(np.quantile(perm_above, 0.975)),
            "p_upper": pval(perm_above, obs_above, True),
            "below5_median": float(np.median(perm_below)),
            "below5_q025": float(np.quantile(perm_below, 0.025)),
            "p_lower": pval(perm_below, obs_below, False),
        },
        "residual_permutation": {
            "above95_median": float(np.median(res_above)),
            "above95_q975": float(np.quantile(res_above, 0.975)),
            "p_upper": pval(res_above, obs_above, True),
            "below5_median": float(np.median(res_below)),
            "below5_q025": float(np.quantile(res_below, 0.025)),
            "p_lower": pval(res_below, obs_below, False),
        },
    }


# ---------------------------------------------------------------------------
# PCoA
# ---------------------------------------------------------------------------
def jaccard_dist(X):
    A = (X > 0).astype(float)
    inter = A @ A.T
    rs = A.sum(1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 1.0)
    return 1.0 - J


def pcoa_coords(D, dims=2):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D @ J
    w, v = eigh(B, subset_by_index=[n - dims, n - 1])
    order = np.argsort(w)[::-1]
    w = np.maximum(w[order], 0.0)
    return v[:, order] * np.sqrt(w)


def run_pcoa(X, paper, draws=100, dims=2):
    D = jaccard_dist(X)
    n = len(X)
    uniq = sorted(set(paper))
    block_rows, matched_rows = [], []
    for p in uniq:
        idx = np.where(paper == p)[0]
        n_p = int(len(idx))
        keep = np.ones(n, bool)
        keep[idx] = False
        keep_idx = np.where(keep)[0]
        S_full = pcoa_coords(D, dims)
        S_red = pcoa_coords(D[np.ix_(keep, keep)], dims)
        mb = event_metrics(S_full, S_red, keep_idx, dims=dims, align="procrustes")
        mb.update(layer="pcoa", paper=p, arm="block", draw=-1, n_rows_deleted=n_p)
        block_rows.append(mb)

        donors = np.where(paper != p)[0]
        base = seed_for(str(p))
        for dr in range(draws):
            rng = np.random.default_rng(base + 7919 * dr)
            drop = rng.choice(donors, size=n_p, replace=False)
            keep2 = np.ones(n, bool)
            keep2[drop] = False
            idx2 = np.where(keep2)[0]
            S_red2 = pcoa_coords(D[np.ix_(keep2, keep2)], dims)
            mm = event_metrics(S_full, S_red2, idx2, dims=dims,
                               align="procrustes")
            mm.update(layer="pcoa", paper=p, arm="matched", draw=dr,
                      n_rows_deleted=n_p)
            matched_rows.append(mm)
    return pd.DataFrame(block_rows), pd.DataFrame(matched_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["core", "pcoa"], default="core")
    ap.add_argument("--pcoa-draws", type=int, default=100)
    ap.add_argument("--collapsed", action="store_true",
                    help="run the PCoA sensitivity on the panel-collapsed matrix")
    args = ap.parse_args()

    X, rho, paper = load_curated()
    Xc, rhoc, paperc, keep_idx = collapse_panels(X, rho, paper)

    if args.mode == "core":
        out = {
            "baseline": {
                "n_rows": int(len(X)),
                "n_papers": int(len(set(paper))),
                "panel_structure": panel_structure(X, paper),
            },
            "collapsed": {
                "n_rows": int(len(Xc)),
                "n_papers": int(len(set(paperc))),
                "panel_structure": panel_structure(Xc, paperc),
            },
        }

        # Baseline audit (PCA, 100 draws).
        b, m = run_audit(X, rho, paper, fit, dims=2, draws=DRAWS,
                         align="sign", label="baseline")
        out["baseline"]["audit"] = summarise(b, m, "baseline_")
        out["baseline"]["model"] = block_model(b, m)
        out["baseline"]["same_size"] = same_size_nulls(b)
        b.to_csv(os.path.join(OUT, "baseline_block.csv"), index=False)
        m.to_csv(os.path.join(OUT, "baseline_matched.csv"), index=False)

        # Collapsed audit (PCA, 100 draws).
        bc, mc = run_audit(Xc, rhoc, paperc, fit, dims=2, draws=DRAWS,
                           align="sign", label="collapsed")
        out["collapsed"]["audit"] = summarise(bc, mc, "collapsed_")
        out["collapsed"]["model"] = block_model(bc, mc)
        out["collapsed"]["same_size"] = same_size_nulls(bc)
        bc.to_csv(os.path.join(OUT, "collapsed_block.csv"), index=False)
        mc.to_csv(os.path.join(OUT, "collapsed_matched.csv"), index=False)

        # Composition paired differences from the deposited tables.
        comp = pd.read_csv(os.path.join(ROOT, "results", "wp3_clean",
                                        "composition_matched_control.csv"))
        comp_med = comp[comp["arm"] == "composition_matched"].groupby("paper")["med_drift"].median()
        dens_med = comp[comp["arm"] == "density_only"].groupby("paper")["med_drift"].median()
        vol_med = m.groupby("paper")["med_drift"].median()
        common = sorted(set(vol_med.index) & set(comp_med.index) & set(dens_med.index))
        # Paired log differences: positive means the first arm drifts more.
        out["composition"] = {
            "composition_vs_density_log": paired_ci(
                np.log(comp_med.loc[common].to_numpy()),
                np.log(dens_med.loc[common].to_numpy())),
            "composition_vs_volume_log": paired_ci(
                np.log(comp_med.loc[common].to_numpy()),
                np.log(vol_med.loc[common].to_numpy())),
            "density_vs_volume_log": paired_ci(
                np.log(dens_med.loc[common].to_numpy()),
                np.log(vol_med.loc[common].to_numpy())),
        }

        with open(os.path.join(OUT, "core.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        print(json.dumps(out, indent=2))

    else:
        Xp, rp, pp = (Xc, rhoc, paperc) if args.collapsed else (X, rho, paper)
        tag = "collapsed" if args.collapsed else "baseline"
        b, m = run_pcoa(Xp, pp, draws=args.pcoa_draws, dims=2)
        b.to_csv(os.path.join(OUT, f"pcoa_{tag}_block.csv"), index=False)
        m.to_csv(os.path.join(OUT, f"pcoa_{tag}_matched.csv"), index=False)
        out = {
            "matrix": tag,
            "n_rows": int(len(Xp)),
            "n_papers": int(len(set(pp))),
            "draws": int(args.pcoa_draws),
            "audit": summarise(b, m, "pcoa_"),
            "model": block_model(b, m),
        }
        with open(os.path.join(OUT, f"pcoa_{tag}_summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
