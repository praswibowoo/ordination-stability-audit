#!/usr/bin/env python3
"""experiment_G_absratio.py — pure Procrustes family/all drift reduction with paper-level bootstrap.

Reviewer-driven follow-up (R1 & R2): quantifies the absolute drift reduction of
conditional family restriction under the paired LOPO design (same rows, family
columns vs all 44 columns). Procrustes drift from scipy.spatial.procrustes is
already scale-free (Frobenius-norm 1). Reports pure Procrustes drift ratios:
  - ratio_of_medians (ratio_raw): median(all_med) / median(fam_med)
  - median_of_ratios (median_paired_ratio): median(all_med / fam_med)
  - mean_ratio: mean(all_mean) / mean(fam_mean)
  - ratio_unique_cfg: median(all_med) / median(fam_cfg_med)
along with paper-level cluster bootstrap 95% confidence intervals.

Reads the frozen per-event layer: results/exC2_absolute_drift.csv
Writes: results/exG_absratio.csv

Run: python3 code/experiment_G_absratio.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEED = 42
BOOT_N = 1000
TAU_FLOOR = 1e-6  # numerical precision noise floor


def main() -> None:
    in_path = os.path.join(RESULTS, "exC2_absolute_drift.csv")
    with open(in_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)

    out = []

    for fam in sorted(by_fam):
        # D197: per-unit generator. A single stream created before this loop made the
        # interval depend on how many families had been drawn ahead of the current one,
        # so the same statistic carried different CIs in exG / exK / exM. A fresh
        # generator per family makes the CI a property of the statistic.
        rng = np.random.default_rng(SEED)
        ev = by_fam[fam]
        all_med = np.asarray([float(r["all_med"]) for r in ev])
        fam_med = np.asarray([float(r["fam_med"]) for r in ev])
        all_mean = np.asarray([float(r["all_mean"]) for r in ev])
        fam_mean = np.asarray([float(r["fam_mean"]) for r in ev])
        fam_cfg = np.asarray([float(r["fam_cfg_med"]) for r in ev])

        all_sp = np.asarray([float(r["all_sparse"]) for r in ev])
        all_dn = np.asarray([float(r["all_dense"]) for r in ev])
        fam_sp = np.asarray([float(r["fam_sparse"]) for r in ev])
        fam_dn = np.asarray([float(r["fam_dense"]) for r in ev])

        paired = all_med / np.where(fam_med > 0, fam_med, np.nan)

        ratio_raw = float(np.median(all_med) / np.median(fam_med))
        med_paired = float(np.nanmedian(paired))
        mean_ratio = float(np.mean(all_mean) / np.mean(fam_mean))
        ratio_cfg = float(np.median(all_med) / np.median(fam_cfg))

        # paper-level cluster bootstrap over deleted papers
        n = len(ev)
        boot_raw = np.empty(BOOT_N)
        boot_paired = np.empty(BOOT_N)
        for b in range(BOOT_N):
            idx = rng.integers(0, n, n)
            boot_raw[b] = np.median(all_med[idx]) / np.median(fam_med[idx])
            boot_paired[b] = np.nanmedian(paired[idx])
        lo_raw, hi_raw = np.quantile(boot_raw, [0.025, 0.975])
        lo_paired, hi_paired = np.quantile(boot_paired, [0.025, 0.975])

        at_floor = bool(np.median(fam_med) < TAU_FLOOR)

        out.append(dict(
            family=fam, variant="pca_pres", layer="presence", subspace="2-D",
            n_events=n,
            all_med=float(np.median(all_med)), fam_med=float(np.median(fam_med)),
            # Primary ratio metrics
            ratio_raw=round(ratio_raw, 2),
            ratio_raw_ci_lo=round(float(lo_raw), 2), ratio_raw_ci_hi=round(float(hi_raw), 2),
            median_paired_ratio=round(med_paired, 2),
            median_paired_ratio_ci_lo=round(float(lo_paired), 2), median_paired_ratio_ci_hi=round(float(hi_paired), 2),
            # Compatibility aliases for figures and tables
            ratio_of_medians=round(ratio_raw, 2),
            median_of_ratios=round(med_paired, 2),
            ratio_ci_lo=round(float(lo_raw), 2), ratio_ci_hi=round(float(hi_raw), 2),
            mean_ratio=round(mean_ratio, 2),
            fam_cfg_med=float(np.median(fam_cfg)),
            ratio_unique_cfg=round(ratio_cfg, 2),
            all_sparse_med=float(np.median(all_sp)), all_dense_med=float(np.median(all_dn)),
            fam_sparse_med=float(np.median(fam_sp)), fam_dense_med=float(np.median(fam_dn)),
            fam_sparse_eq_dense_events=int(np.sum(fam_sp == fam_dn)),
            at_floor=at_floor,
        ))

    path = os.path.join(RESULTS, "exG_absratio.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    os.replace(path + ".tmp", path)

    print(f"{'family':14s} {'events':>6s} {'all_med':>10s} {'fam_med':>10s} "
          f"{'ratio_raw':>10s} {'paired_ratio':>12s} {'cfg_ratio':>10s} {'at_floor':>9s}")
    for r in out:
        print(f"{r['family']:14s} {r['n_events']:6d} {r['all_med']:10.3e} {r['fam_med']:10.3e} "
              f"{r['ratio_raw']:10.1f} {r['median_paired_ratio']:12.1f} {r['ratio_unique_cfg']:10.1f} {str(r['at_floor']):>9s}")
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
