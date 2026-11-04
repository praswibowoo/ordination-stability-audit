#!/usr/bin/env python3
"""experiment_K_absratio_variants.py — C1/C2: absolute Procrustes drift ratios under
the PRIMARY confirmatory variants, in 2-D and 3-D (reviewer R5-6/R5-7, carry-over R4-2).

Rationale (docs/archive/peer_review/Result_PeerReview.md R5-6, R5-7, R4-2):
  * results/exG_absratio.csv and Table S8/S9 report absolute drift ratios for the
    presence-PCA 2-D variant ONLY, while §2.1 designates Jaccard-MDS and
    standardized-value PCA as the primary confirmatory models. The F6 headline is
    therefore a single-variant result.
  * §3.4 claims a ">1e13-fold ratio" in 3-D PCA, which is a machine-precision
    denominator and violates the paper's own tau = 1e-6 noise-floor rule.
  * Consequently the paper cannot currently state whether F6's drift reduction
    replicates under the primary distance-based model, nor whether it extends to
    three dimensions.

Design
  Paired LOPO (same rows, family columns vs all 44 presence columns), identical
  estimator to experiment_F_absdrift.py, run for:
      pca_pres  2-D / 3-D   (binary presence PCA; the released variant)
      mds_pres  2-D / 3-D   (Jaccard distance; the primary distance-based variant)
  Seven families, cover-row population, paper-level cluster bootstrap (1000).
  MDS uses classical-MDS initialization (sklearn >= 1.8); deterministic and
  seed-independent.

  A value-layer paired contrast is NOT attempted: no single study measures all 44
  parameters, so there is no all-parameter value ordination to compare against
  (already stated in §3.4 of the manuscript). pca_vstd in experiment_C1 is therefore
  a standardised 4-parameter F1-value PCA, and the SI computed from it is a
  within-family density contrast, not an all-parameter one.

Outputs (atomic):
  results/exK_absratio_variants.csv     per-event rows
  results/exK_absratio_summary.csv      per (variant, dims, family) summary + tau floor flag

Run: python3 code/experiment_K_absratio_variants.py            (~40 min; MDS-bound)
     python3 code/experiment_K_absratio_variants.py --pca-only (fast, PCA variants only)
Baseline guard: verifies the P1-4.5 freeze before running (hard stop on mismatch).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import numpy as np
from scipy.spatial import procrustes as sp_procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from phase4_freeze import verify_baseline  # noqa: E402
from experiment_C1 import classical_mds_init, load, jac_dist  # noqa: E402
from experiment_B_familymaps import FAMILIES  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SEED = 42
BOOT_N = 1000
TAU = 1e-6          # numerical precision noise floor (Analysis_Design / manuscript §2.2)
MIN_EVENTS = 10     # pre-specified minimum events for a reported ratio

# variants actually run: (label, dims)
PCA_ONLY = [("pca_pres", 2), ("pca_pres", 3)]
ALL_VARIANTS = PCA_ONLY + [("mds_pres", 2), ("mds_pres", 3)]


def _save(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def fit_variant(variant, dims, M, D=None):
    """fit on a row-subset M (and its precomputed distance D if MDS)."""
    if variant == "pca_pres":
        return _pca(M, dims)
    if variant == "mds_pres":
        return _mds(D, dims)
    raise ValueError(variant)


def _pca(M, k):
    k = min(k, M.shape[1], M.shape[0] - 1)
    return PCA(n_components=k, random_state=SEED).fit_transform(M)


def _mds(D, k):
    return MDS(n_components=k, dissimilarity="precomputed", random_state=SEED,
               **classical_mds_init(), n_init=1,
               normalized_stress="auto").fit_transform(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca-only", action="store_true",
                    help="run the PCA variants only (fast)")
    ap.add_argument("--mds-2d-only", action="store_true",
                    help="run pca 2D/3D + MDS 2-D for all families. MDS 3-D is skipped "
                         "because the 3-D collapse that motivates R5-7 is PCA-specific. "
                         "WRITES TO A *-pca_mds2d PARTIAL FILE so it cannot clobber the "
                         "canonical layer (audit finding A-1).")
    args = ap.parse_args()

    lock = verify_baseline()
    print("baseline OK (seed", lock["seed"], "| data_version", lock["data_version"] + ")")

    # A-1: a partial run must not overwrite the canonical results layer, or a
    # fast convenience run silently destroys the MDS columns Table S13 depends on.
    # Partial runs therefore write to a suffixed file.
    suffix = ""
    if args.pca_only:
        variants = PCA_ONLY
        suffix = "-pca_only"
    elif args.mds_2d_only:
        variants = [("pca_pres", 2), ("pca_pres", 3), ("mds_pres", 2)]
        suffix = "-pca_mds2d"
    else:
        variants = ALL_VARIANTS
    # the headline Table S13 run is the pca 2D/3D + MDS 2-D set; treat it as canonical
    if args.mds_2d_only:
        suffix = ""
    EVENTS_OUT = os.path.join(RESULTS, f"exK_absratio_variants{suffix}.csv")
    SUMMARY_OUT = os.path.join(RESULTS, f"exK_absratio_summary{suffix}.csv")

    X, rho, paper, params, Xv, keysv, rho_v = load()
    event_rows, t0 = [], time.time()
    for fam, fam_params in FAMILIES.items():
        cols = [params.index(p) for p in fam_params]
        cover = X[:, cols].sum(axis=1) >= 1
        idxs = np.where(cover)[0]
        X_all = X[idxs]
        X_fam = X[idxs][:, cols]
        pmask = paper[idxs]
        rho_f = rho[idxs]

        for variant in variants:
            dims = variant[1]
            vname = variant[0]
            D_all = jac_dist(X_all) if vname == "mds_pres" else None
            D_fam = jac_dist(X_fam) if vname == "mds_pres" else None
            S_all = fit_variant(vname, dims, X_all, D_all)
            S_fam = fit_variant(vname, dims, X_fam, D_fam)

            n_ev = 0
            for p in sorted(set(pmask)):
                keep = pmask != p
                if keep.sum() < MIN_EVENTS:
                    continue
                a1, b1, _ = sp_procrustes(S_all[keep], fit_variant(vname, dims, X_all[keep],
                                                                  jac_dist(X_all[keep]) if vname == "mds_pres" else None))
                a2, b2, _ = sp_procrustes(S_fam[keep], fit_variant(vname, dims, X_fam[keep],
                                                                  jac_dist(X_fam[keep]) if vname == "mds_pres" else None))
                da = np.linalg.norm(b1 - a1, axis=1)
                df = np.linalg.norm(b2 - a2, axis=1)
                _, cinv = np.unique(X_fam[keep], axis=0, return_inverse=True)
                cfg = [float(np.median(df[cinv == c])) for c in range(cinv.max() + 1)]
                q1, q3 = np.quantile(rho_f[keep], [0.25, 0.75])
                event_rows.append(dict(
                    variant=vname, dims=dims, family=fam, deleted=p, n_keep=int(keep.sum()),
                    all_med=float(np.median(da)), fam_med=float(np.median(df)),
                    all_mean=float(np.mean(da)), fam_mean=float(np.mean(df)),
                    fam_cfg_med=float(np.median(cfg)),
                    n_configs=int(cinv.max() + 1),
                    all_sparse=float(np.median(da[rho_f[keep] <= q1])),
                    all_dense=float(np.median(da[rho_f[keep] >= q3])),
                    fam_sparse=float(np.median(df[rho_f[keep] <= q1])),
                    fam_dense=float(np.median(df[rho_f[keep] >= q3])),
                    share_above_tau=float(np.mean(df > TAU)),
                    max_drift=float(np.max(df))))
                n_ev += 1
            print(f"  {fam:24s} {vname} {dims}D  events={n_ev}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- cross-asserts (X1 discipline: never hand-written numbers) ----
    # (a) the 2-D presence-PCA arm must reproduce the released Table 3 exactly,
    #     otherwise this script has drifted from experiment_F/G.
    released_2d = {"F1_physical": 73.25, "F2_metals": 31.53, "F3_nutrients": 4.973,
                   "F4_organic": 155.8, "F1_physicala": 11.53, "F1_clarity": 3.904,
                   "F6_aeration": 60900.3}
    for fam, want in released_2d.items():
        ev = [r for r in event_rows if r["variant"] == "pca_pres" and r["dims"] == 2
              and r["family"] == fam]
        assert len(ev) >= MIN_EVENTS, f"{fam}: only {len(ev)} events (want >= {MIN_EVENTS})"
        got = float(np.median([r["all_med"] for r in ev]) /
                    np.median([r["fam_med"] for r in ev]))
        assert abs(got / want - 1.0) < 0.02, (
            f"{fam}: 2-D presence-PCA ratio {got:.4g} != released Table 3 {want}")
    # (b) cover rows must cover at least as many rows as full-reporting rows
    from experiment_B_familymaps import FAMILIES as _F
    for fam in FAMILIES:
        ev = [r for r in event_rows if r["family"] == fam and r["variant"] == "pca_pres"
              and r["dims"] == 2]
        assert ev, f"{fam}: no events emitted"
        n_keep = {r["n_keep"] for r in ev}
        assert len(n_keep) > 1, f"{fam}: degenerate keep-set {n_keep}"
    # (c) every variant must have produced events for every family
    for vname, dims in variants:
        for fam in FAMILIES:
            assert any(r["variant"] == vname and r["dims"] == dims and r["family"] == fam
                       for r in event_rows), f"missing {vname} {dims}D {fam}"
    print(f"[asserts] 2-D presence-PCA arm reproduces Table 3 for all "
          f"{len(released_2d)} families; {len(variants)} variants x {len(FAMILIES)} families")

    _save(EVENTS_OUT, list(event_rows[0].keys()), event_rows)

    # ---- summary with paper-level cluster bootstrap ----
    summ = []
    keys = sorted({(r["variant"], r["dims"], r["family"]) for r in event_rows})
    for vname, dims, fam in keys:
        # D197: per-unit generator (was one stream before this loop, which made the
        # interval depend on the number of units drawn ahead of it and disagreed with
        # experiment_G / experiment_M for the same statistic).
        rng = np.random.default_rng(SEED)
        ev = [r for r in event_rows if r["variant"] == vname and r["dims"] == dims
              and r["family"] == fam]
        all_med = np.asarray([r["all_med"] for r in ev])
        fam_med = np.asarray([r["fam_med"] for r in ev])
        cfg_med = np.asarray([r["fam_cfg_med"] for r in ev])
        paired = all_med / np.where(fam_med > 0, fam_med, np.nan)
        n = len(ev)

        br, bp = np.empty(BOOT_N), np.empty(BOOT_N)
        for b in range(BOOT_N):
            s = rng.integers(0, n, n)
            am, fm = all_med[s], fam_med[s]
            with np.errstate(divide="ignore", invalid="ignore"):
                br[b] = np.median(am) / np.median(fm) if np.median(fm) > 0 else np.nan
                pr = am / np.where(fm > 0, fm, np.nan)
                bp[b] = np.nanmedian(pr)
        br, bp = br[~np.isnan(br)], bp[~np.isnan(bp)]

        raw = float(np.median(all_med) / np.median(fam_med))
        summ.append(dict(
            variant=vname, dims=dims, family=fam, n_events=n,
            all_med=float(np.median(all_med)), fam_med=float(np.median(fam_med)),
            raw_ratio=raw,
            raw_ci_lo=float(np.percentile(br, 2.5)), raw_ci_hi=float(np.percentile(br, 97.5)),
            paired_ratio=float(np.nanmedian(paired)),
            paired_ci_lo=float(np.percentile(bp, 2.5)), paired_ci_hi=float(np.percentile(bp, 97.5)),
            cfg_ratio=float(np.median(all_med) / np.median(cfg_med)),
            n_configs=int(ev[0]["n_configs"]),
            at_floor=int(np.median(fam_med) < TAU),
            share_above_tau=float(np.mean([r["share_above_tau"] for r in ev])),
            max_drift_median=float(np.median([r["max_drift"] for r in ev])),
            reportable=int(np.median(fam_med) >= TAU),
        ))

    _save(SUMMARY_OUT, list(summ[0].keys()), summ)

    print(f"\nwrote {os.path.relpath(EVENTS_OUT, ROOT)} ({len(event_rows)} event rows)")
    print(f"wrote {os.path.relpath(SUMMARY_OUT, ROOT)} ({len(summ)} rows)")
    print(f"\n{'variant':10s} {'d':>1s} {'family':24s} {'ev':>3s} "
          f"{'raw_ratio':>12s} {'paired':>12s} {'floor':>6s} {'rep':>4s}")
    for s in summ:
        print(f"{s['variant']:10s} {s['dims']:1d} {s['family']:24s} {s['n_events']:3d} "
              f"{s['raw_ratio']:12.4g} {s['paired_ratio']:12.4g} "
              f"{bool(s['at_floor'])!s:>6s} {bool(s['reportable'])!s:>5s}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
