#!/usr/bin/env python3
"""robustness_duplicates.py — clean three-treatment duplicate-handling table.

The one known re-publication pair is B111/B175. This script recomputes the four
headline statistics under all three treatments (full set, drop B111, drop B175)
with one consistent methodology: leave-one-paper-out Swing Index for three C1
variants plus the F6 paired stability gain, each with the paper-level
permutation test and cluster bootstrap used in the corrections pass.

Writes results/duplicate_handling.csv.
Run: python3 code/core/robustness_duplicates.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))

from experiment_C1 import load, run_variant              # noqa: E402
from experiment_C2_stability import (FAMILIES, bootstrap_G,   # noqa: E402
                                     paired_event_si_same_rows)
from stability_blockperm import si, N_PERM, SEED         # noqa: E402


def block_test(rho, drift, codes, n_papers, rng):
    """Paper-level permutation + cluster bootstrap (matches exC1_blockperm).

    Codes are compacted to the papers actually present in the row set, so that
    value variants (which cover only the family-complete papers) do not create
    empty-slice means for papers without value rows.
    """
    obs = si(rho, drift)
    uniq = np.unique(codes)
    codes = np.searchsorted(uniq, codes)
    n_papers = len(uniq)
    # paper-level densities
    prho = np.array([rho[codes == j].mean() for j in range(n_papers)])
    idx_by_paper = [np.where(codes == j)[0] for j in range(n_papers)]
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        rr = rng.permutation(prho)[codes]
        null[b] = si(rr, drift)
    p_gt = float((np.sum(null > obs) + 1) / (N_PERM + 1))
    p_lt = float((np.sum(null < obs) + 1) / (N_PERM + 1))
    boots = np.empty(1000)
    for b in range(1000):
        draws = rng.integers(0, n_papers, n_papers)
        idx = np.concatenate([idx_by_paper[j] for j in draws])
        boots[b] = si(rho[idx], drift[idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(SI=float(obs), p_one=round(p_gt, 4),
                p_two=round(min(1.0, 2 * min(p_gt, p_lt)), 4),
                ci_lo=float(lo), ci_hi=float(hi))


def main() -> None:
    X, rho, paper, params, Xv, keysv, rho_v = load()
    out = []
    for treatment, drop in (("full set", None), ("drop B111", "B111"),
                            ("drop B175", "B175")):
        t0 = time.time()
        keep = np.ones(len(X), bool) if drop is None else (paper != drop)
        keepv = np.ones(len(Xv), bool) if drop is None else \
            np.asarray([k[0] != drop for k in keysv])
        Xk, rhok, pk = X[keep], rho[keep], paper[keep]
        Xvk = Xv[keepv]
        keysk = [k for k in keysv if drop is None or k[0] != drop]
        rhovk = rho_v[keepv] if drop is None else np.asarray(
            [r for r, k in zip(rho_v, keysv) if k[0] != drop])
        uniq = sorted(set(pk)); code_of = {p: i for i, p in enumerate(uniq)}

        for variant in ("pca_pres", "mds_pres", "pca_vstd"):
            rows = run_variant(variant, Xk, Xvk,
                               rhovk if variant.startswith("pca_v") else rhok,
                               pk, [str(i) for i in range(len(Xk))], keysk,
                               modes=("lopo",))
            rrv = np.asarray([r["rho"] for r in rows])
            ddv = np.asarray([r["drift"] for r in rows])
            if variant.startswith("pca_v"):
                codes = np.asarray([code_of[r["row_id"].split("#")[0]] for r in rows])
            else:
                codes = np.asarray([code_of[pk[int(r["row_id"])]] for r in rows])
            st = block_test(rrv, ddv, codes, len(uniq),
                            np.random.default_rng(SEED))
            out.append(dict(treatment=treatment, test=f"C1 {variant} LOPO SI",
                            value=round(st["SI"], 4), p_one=st["p_one"],
                            p_two=st["p_two"], ci_lo=round(st["ci_lo"], 4),
                            ci_hi=round(st["ci_hi"], 4),
                            n_events=len({r["deleted"] for r in rows})))
            print(f"  {treatment:9s} {variant:9s} SI={st['SI']:.4f} "
                  f"p2={st['p_two']:.3f} CI=({st['ci_lo']:.3f},{st['ci_hi']:.3f}) "
                  f"[{time.time()-t0:.0f}s]")

        fam = "F6_aeration"
        fam_cols = [params.index(p) for p in FAMILIES[fam]]
        mask = Xk[:, fam_cols].sum(axis=1) >= 1
        idxs = np.where(mask)[0]
        pairs = paired_event_si_same_rows(
            Xk[np.ix_(idxs, fam_cols)], Xk[np.ix_(idxs, range(len(params)))],
            rhok[idxs], pk[idxs], model="pca", dim=2)
        boot = bootstrap_G(pairs, np.random.default_rng(SEED))
        out.append(dict(treatment=treatment, test="C2 F6_aeration paired G",
                        value=round(boot["G_median"], 4),
                        p_one=round(boot["P_G_gt0"], 4), p_two="",
                        ci_lo=round(boot["G_CI95"][0], 4),
                        ci_hi=round(boot["G_CI95"][1], 4),
                        n_events=len(pairs)))
        print(f"  {treatment:9s} F6 G={boot['G_median']:.4f} "
              f"P={boot['P_G_gt0']:.4f} n={len(pairs)} [{time.time()-t0:.0f}s]")

    path = os.path.join(ROOT, "results", "duplicate_handling.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    os.replace(path + ".tmp", path)
    print(f"[duplicates] wrote {os.path.relpath(path, ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
