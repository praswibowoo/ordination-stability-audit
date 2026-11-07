#!/usr/bin/env python3
"""block_interaction_primary.py — primary block x log-volume interaction.

The manuscript's primary block model is

    log(delta) = b0 + bV log(V) + bB B.

That model reports one block factor.  The primary event files show a
significant block x log-volume interaction, so this script fits the
volume-centred version

    log(delta) = b0 + bV (log V - mean log V) + bB B
                       + bI B (log V - mean log V),

where exp(bB) is the block factor at the mean deleted volume (about 5 rows).
Inference uses a study-level cluster-robust (CR1) covariance and a 2,000-draw
study-level cluster bootstrap.  The bootstrap tail probability is not used as
a null-centred p-value; the CR1 p-value is the test and the percentile
interval is the interval.

Reads:  results/wp3_primary/exH_volumematch.csv
        results/wp3_primary/exH_volumematch_draws.csv
Writes: results/wp3_primary/exH_interaction_model.csv
        results/wp3_primary/exH_interaction_volume.csv

Run: python3 code/block_interaction_primary.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
SEED = 42


def read_events(lopo_path: Path, draws_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with lopo_path.open(newline="", encoding="utf-8") as f:
        rows.extend(r for r in csv.DictReader(f) if r["arm"] == "lopo")
    with draws_path.open(newline="", encoding="utf-8") as f:
        rows.extend(r for r in csv.DictReader(f) if r["arm"] == "matched_scattered")
    return rows


def fit(rows: list[dict[str, str]], n_boot: int, seed: int) -> tuple:
    y = np.array([math.log(float(r["med_drift"])) for r in rows], float)
    volume = np.array([float(r["n_rows_deleted"]) for r in rows], float)
    block = np.array([1.0 if r["arm"] == "lopo" else 0.0 for r in rows], float)
    paper = np.array([r["paper"] for r in rows], dtype=object)

    log_v = np.log(volume)
    centre = float(log_v.mean())
    lv = log_v - centre
    design = np.column_stack([np.ones(len(y)), lv, block, block * lv])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)

    # Study-level cluster-robust (CR1) covariance.
    resid = y - design @ beta
    xtx_inv = np.linalg.inv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for p in np.unique(paper):
        idx = np.where(paper == p)[0]
        score = design[idx].T @ resid[idx]
        meat += np.outer(score, score)
    n = len(y)
    k = design.shape[1]
    g = len(np.unique(paper))
    cov = xtx_inv @ meat @ xtx_inv * (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pval = 2.0 * stats.t.sf(np.abs(tstat), g - 1.0)

    # Study-level cluster bootstrap, whole studies with replacement.
    rng = np.random.default_rng(seed)
    papers = np.unique(paper)
    index_by_paper = {p: np.where(paper == p)[0] for p in papers}
    boot = np.empty((n_boot, k))
    for b in range(n_boot):
        pick = rng.choice(papers, size=len(papers), replace=True)
        idx = np.concatenate([index_by_paper[p] for p in pick])
        boot[b], *_ = np.linalg.lstsq(design[idx], y[idx], rcond=None)

    return y, volume, block, paper, centre, beta, se, tstat, pval, boot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lopo", type=Path,
                    default=RESULTS / "exH_volumematch.csv")
    ap.add_argument("--draws", type=Path,
                    default=RESULTS / "exH_volumematch_draws.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = read_events(args.lopo, args.draws)
    y, volume, block, paper, centre, beta, se, tstat, pval, boot = fit(
        rows, args.n_boot, args.seed)

    names = ["intercept", "log_volume_centred", "block_at_mean_volume",
             "block_x_log_volume"]
    out_rows = []
    for j, name in enumerate(names):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        out_rows.append(dict(
            term=name,
            coef=beta[j],
            se_cr1=se[j],
            t_cr1=tstat[j],
            p_cr1=pval[j],
            ci_lo=lo,
            ci_hi=hi,
            n_events=len(y),
            n_papers=len(np.unique(paper)),
            mean_log_volume=centre,
            n_boot=args.n_boot,
            seed=args.seed,
        ))

    # Volume-specific block factors, exp(bB + bI * (log V - mean log V)).
    volume_rows = []
    for v in (1, 2, 3, 5, 7, 10, 14):
        point = beta[2] + beta[3] * (math.log(v) - centre)
        draws = boot[:, 2] + boot[:, 3] * (math.log(v) - centre)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        volume_rows.append(dict(
            n_rows_deleted=v,
            block_factor=math.exp(point),
            factor_lo=math.exp(lo),
            factor_hi=math.exp(hi),
            n_events=len(y),
            n_papers=len(np.unique(paper)),
        ))

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "exH_interaction_model.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    with (RESULTS / "exH_interaction_volume.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(volume_rows[0]))
        w.writeheader()
        w.writerows(volume_rows)

    print("events:", len(y), "papers:", len(np.unique(paper)))
    print("mean log volume:", round(centre, 4))
    for r in out_rows:
        print(f"{r['term']}: coef={r['coef']:.4f} "
              f"CR1 p={r['p_cr1']:.5f} CI=({r['ci_lo']:.4f}, {r['ci_hi']:.4f})")
    print("volume-specific block factors:")
    for r in volume_rows:
        print(f"  V={r['n_rows_deleted']}: "
              f"{r['block_factor']:.3f} ({r['factor_lo']:.3f}, {r['factor_hi']:.3f})")


if __name__ == "__main__":
    main()
