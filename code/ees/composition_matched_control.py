#!/usr/bin/env python3
"""composition_matched_control.py — composition-matched deletion control.

The volume-matched control in experiment_H_volumematch.py matches the NUMBER of
rows deleted but not their composition. This script adds a composition-matched
control: for each study, it deletes a set of rows from OTHER studies whose
presence configurations match the study's rows (exact configuration where a
donor exists; nearest-density fallback otherwise). It then compares block
deletion drift with this composition-matched deletion drift.

If the block effect survives composition matching, it is not merely a
consequence of which presence patterns were removed. If it attenuates toward 1,
the volume-matched block effect is partly a composition effect.

Reads: frozen presence layer via experiment_C1.load(); results/exH_volumematch.csv
Writes: results/composition_matched_control.csv
        results/composition_matched_summary.csv

Run: python3 code/ees/composition_matched_control.py [--draws 100]
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import stats

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


def event_drift(S_full, knn_full, X, rho, keep):
    S_red = ordinate("pca_pres", X[keep])
    drift, _ = drift_after_procrustes(S_full[keep], S_red)
    ki = np.where(keep)[0]
    ch = knn_churn(knn_full[keep], ki[knn_idx(S_red)])
    return float(np.median(drift)), float(np.median(ch))


def cluster_boot_block(ev: pd.DataFrame, n_boot: int = 2000, seed: int = SEED):
    y = np.log(ev["med_drift"].to_numpy(float))
    v = np.log(ev["n_rows_deleted"].to_numpy(float))
    b = (ev["arm"] == "block").to_numpy(float)
    Xd = np.column_stack([np.ones(len(y)), v, b])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    uniq = np.asarray(sorted(set(ev["paper"])), dtype=object)
    gid = pd.Categorical(ev["paper"], categories=uniq).codes
    idx = {j: np.where(gid == j)[0] for j in range(len(uniq))}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[j] for j in pick])
        boots.append(np.linalg.lstsq(Xd[sel], y[sel], rcond=None)[0])
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots[:, 2], [0.025, 0.975])
    return float(np.exp(beta[2])), float(np.exp(lo)), float(np.exp(hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    X, rho, paper, params, *_ = load()
    paper = np.asarray(paper)
    n = len(X)
    uniq = sorted(set(paper))
    S_full = ordinate("pca_pres", X)
    knn_full = knn_idx(S_full)

    # Presence configuration per row, as a hashable bytes key.
    cfg = [bytes(X[i].astype(np.uint8)) for i in range(n)]

    lopo = pd.read_csv(os.path.join(RESULTS, "exH_volumematch.csv"))
    lopo = lopo[lopo["arm"] == "lopo"][["paper", "med_drift", "n_rows_deleted"]]
    block = dict(zip(lopo["paper"], zip(lopo["med_drift"], lopo["n_rows_deleted"])))

    rows = []
    for p in uniq:
        mask_p = paper == p
        n_p = int(mask_p.sum())
        idx_p = np.where(mask_p)[0]
        donors = np.where(~mask_p)[0]
        donor_cfg = {}
        for j in donors:
            donor_cfg.setdefault(cfg[j], []).append(j)

        for dr in range(args.draws):
            rng = np.random.default_rng(SEED + zlib.crc32(str(p).encode()) % 100000 + 7919 * dr)
            used = set()
            chosen = []
            exact = 0
            for i in idx_p:
                cand = [j for j in donor_cfg.get(cfg[i], []) if j not in used]
                if cand:
                    j = int(rng.choice(cand))
                    exact += 1
                else:
                    # nearest-density donor not already used
                    rem = np.array([j for j in donors if j not in used])
                    order = np.argsort(np.abs(rho[rem] - rho[i]))
                    j = int(rem[order[0]])
                used.add(j)
                chosen.append(j)
            keep = np.ones(n, bool)
            keep[np.asarray(chosen)] = False
            d, ch = event_drift(S_full, knn_full, X, rho, keep)
            rows.append(dict(paper=p, draw=dr, arm="composition_matched",
                             med_drift=d, med_churn=ch,
                             n_rows_deleted=n_p, exact_fraction=exact / n_p))

    matched = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "composition_matched_control.csv")
    matched.to_csv(out, index=False, float_format="%.6g")

    # Block + composition-matched first draw for the paired summary.
    first = matched[matched["draw"] == 0][["paper", "med_drift"]].rename(
        columns={"med_drift": "matched_drift"})
    bdf = pd.DataFrame([dict(paper=p, block_drift=v[0], n_rows_deleted=v[1])
                        for p, v in block.items()])
    pair = bdf.merge(first, on="paper", how="inner")
    pair["ratio"] = pair["block_drift"] / pair["matched_drift"]
    rng = np.random.default_rng(SEED)
    boot = np.empty(10000)
    for i in range(10000):
        idx = rng.integers(0, len(pair), len(pair))
        boot[i] = np.median(pair["ratio"].to_numpy()[idx])
    ci = np.quantile(boot, [0.025, 0.975])
    p_w = stats.wilcoxon(pair["block_drift"], pair["matched_drift"]).pvalue

    # Log-linear block factor: block events + all composition-matched draws.
    ev = pd.concat([
        pd.DataFrame([dict(paper=p, med_drift=v[0], n_rows_deleted=v[1], arm="block")
                      for p, v in block.items()]),
        matched[["paper", "med_drift", "n_rows_deleted", "arm"]],
    ], ignore_index=True)
    bf, lo, hi = cluster_boot_block(ev, n_boot=args.bootstrap)

    summary = dict(
        n_papers=len(pair),
        block_med=float(np.median(pair["block_drift"])),
        matched_med=float(np.median(pair["matched_drift"])),
        ratio_of_medians=float(np.median(pair["block_drift"]) / np.median(pair["matched_drift"])),
        median_paired_ratio=float(np.median(pair["ratio"])),
        paired_ci_lo=float(ci[0]), paired_ci_hi=float(ci[1]),
        wilcoxon_p=float(p_w),
        block_factor=bf, block_ci_lo=lo, block_ci_hi=hi,
        mean_exact_fraction=float(matched["exact_fraction"].mean()),
        min_exact_fraction=float(matched["exact_fraction"].min()),
        n_events=len(ev),
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS, "composition_matched_summary.csv"),
        index=False, float_format="%.6g")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nwrote {os.path.relpath(out, ROOT)} and composition_matched_summary.csv")


if __name__ == "__main__":
    main()
