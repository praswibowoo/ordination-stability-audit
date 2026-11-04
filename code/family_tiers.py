#!/usr/bin/env python3
"""family_tiers.py — apply the Table 5 evaluation rubric to every family.

The rubric in Table 5 assigns a family one of three statuses (ready,
provisional, incidental) from six criteria. A seventh both-interval check is
redundant with criteria 1 and 2; it is kept as a guard. The paper states the resulting
tier for all seven families, and this script produces that classification
criterion by criterion from the committed result files, so every tier traces to
a criterion rather than to prose.

This script applies the rubric to all seven families from the committed result
files and writes results/family_tiers.csv.

Criterion 2 (Jaccard MDS, 2-D) is DEMOTED to a sensitivity (D258): the released
classical-init MDS arm changed the family ordering, put F6 at the noise floor,
and reversed the sparsity-drift direction, so it can no longer set a tier. The
column is still computed and reported for transparency, but the tier comes from
the five tier-setting criteria (1, 4, 5, 6, 7). Every MDS ratio is > 1, so the
seven tiers are unchanged by the demotion.

Criteria and sources
--------------------
1 2-D presence-PCA raw ratio, interval excluding 1        exK_absratio_summary.csv
2 Jaccard MDS, 2-D (sensitivity; not tier-setting)       exK_absratio_summary.csv
3 2-D presence-PCA interval excludes 1 (guard only)       exK_absratio_summary.csv
4 median family drift above the noise floor tau           exK (fam_med, tau = 1e-6)
5 3-D result reportable and > 1                           exK (dims = 3, reportable)
6 no configuration-class collapse, or collapse above tau  exG_absratio.csv + fam_med
7 random-family benchmark: one-sided p < 0.05 against
  non-degenerate suites                                   exM_benchmark_matched_<family>.csv

Criterion 6 is read as the table writes it: "no collapse (or collapse above
tau)". A family that collapses onto one configuration class, but whose median
family drift stays above tau, therefore does not fail this criterion; the
collapse matters for the interpretation of ratio gain G, not for the tier. The
criterion is applied per family, not per event: a collapse at or below tau fails
it, a collapse above tau does not.

The rubric's incidental column is applied per criterion. Criterion 1 is
incidental when the ratio is <= 1 or the interval is indistinguishable from 1;
criterion 2 is reported but not applied (D258); criteria 3, 5, 6 and 7 have no
incidental cell, and criterion 4 is incidental on either reading. Criterion 1's
cells overlap: the provisional cell reads "> 1 (interval need not exclude 1)" while the
incidental cell reads "<= 1, or interval indistinguishable from 1". This script
applies the provisional reading, so a ratio above 1 whose interval contains 1 is
provisional rather than incidental. None of the seven families is affected (all
seven intervals exclude 1), and the overlap is flagged for the authors.

Criterion 7 uses the per-suite family-matched random-suite benchmark, which
carries a raw ratio for every suite, so the criterion is evaluated on the
metric it names. Degenerate suites are those with med_fam < 1e-16, the
definition used for the F6 benchmark in section 3.4. For F6 the manuscript
quotes a different benchmark design (100 random three-parameter control suites,
tau-aligned to the 13 families above the noise floor); this script uses the
matched design for all seven so that the comparison is uniform, and both
designs return the same status.

Writes results/family_tiers.csv.
Run: make family-tiers
"""
from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU = 1e-6        # noise floor, section 2.1
DEGENERATE = 1e-16  # coordinate-floor collapse, section 3.4

FAMILIES = ["F1_physical", "F1_physicala", "F1_clarity", "F2_metals",
            "F3_nutrients", "F4_organic", "F6_aeration"]


def read(name: str) -> list[dict]:
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rows: list[dict], **sel) -> dict | None:
    for r in rows:
        if all(r.get(k) == v for k, v in sel.items()):
            return r
    return None


def main() -> None:
    exK = read("exK_absratio_summary.csv")
    exG = read("exG_absratio.csv")

    out = []
    for fam in FAMILIES:
        p2 = pick(exK, variant="pca_pres", dims="2", family=fam)
        m2 = pick(exK, variant="mds_pres", dims="2", family=fam)
        p3 = pick(exK, variant="pca_pres", dims="3", family=fam)
        g = pick(exG, family=fam, variant="pca_pres", subspace="2-D")

        raw = float(p2["raw_ratio"])
        lo, hi = float(p2["raw_ci_lo"]), float(p2["raw_ci_hi"])
        mds = float(m2["raw_ratio"])
        mlo, mhi = float(m2["raw_ci_lo"]), float(m2["raw_ci_hi"])
        fam_med = float(p2["fam_med"])
        n_events = int(g["n_events"])
        collapse = int(g["fam_sparse_eq_dense_events"])
        rep3 = int(p3["reportable"]) if p3 else 0
        ratio3 = float(p3["raw_ratio"]) if p3 else float("nan")

        # criterion 7: family-matched random-suite benchmark, raw ratio
        bench = read(f"exM_benchmark_matched_{fam}.csv")
        nondeg = [b for b in bench if float(b["med_fam"]) >= DEGENERATE]
        exceed = [b for b in nondeg if float(b["raw_ratio"]) >= raw]

        c1 = raw > 1.0 and lo > 1.0
        c2_ready = mds > 1.0 and mlo > 1.0
        # Guard only: with MDS demoted (D258), the tier-setting intervals are
        # the 2-D presence-PCA ones; criterion 3 must not reintroduce the MDS
        # sensitivity through the back door.
        c3 = lo > 1.0
        c4_ready = fam_med > TAU
        c4_prov = not c4_ready
        c5_ready = rep3 == 1 and ratio3 > 1.0
        c5_prov = not c5_ready
        # "no collapse (or collapse above tau)": a collapse at or below tau
        # fails the criterion, a collapse above tau does not.
        c6_ready = (collapse == 0) or (fam_med > TAU)
        c7_p = (1 + len(exceed)) / (1 + len(nondeg))
        c7_ready = c7_p < 0.05

        # incidental, per the rubric's own incidental column. Criterion 2 (MDS)
        # was demoted to a sensitivity in D258 and no longer sets a tier.
        incidental = (raw <= 1.0)
        if c1 and c3 and c4_ready and c5_ready and c6_ready and c7_ready:
            tier = "ready"
        elif not incidental and (c4_prov or c5_prov or not c7_ready):
            tier = "provisional"
        else:
            tier = "incidental"

        triggers = [n for n, ok in (
            ("at_or_below_floor", c4_prov),
            ("three_d_unreportable_or_low", c5_prov),
            ("benchmark_exceeded", not c7_ready)) if ok]

        out.append(dict(
            family=fam, raw_pca2d=round(raw, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
            mds_ratio=round(mds, 3), mds_ci_lo=round(mlo, 3), mds_ci_hi=round(mhi, 3),
            fam_med=f"{fam_med:.6g}", at_floor=int(fam_med <= TAU),
            ratio_3d="" if not rep3 else round(ratio3, 2), reportable_3d=rep3,
            collapse_events=collapse, n_events=n_events,
            collapse_fraction=round(collapse / n_events, 4),
            benchmark_exceed=len(exceed), benchmark_nondeg=len(nondeg),
            benchmark="family-matched, raw ratio; non-degenerate if med_fam >= 1e-16",
            criterion1_pca2d_ok=int(c1), criterion2_mds_ok=int(c2_ready),
            criterion3_interval_excludes_1=int(c3), criterion4_above_floor=int(c4_ready),
            criterion5_3d_ok=int(c5_ready), criterion6_no_collapse_or_above_floor=int(c6_ready),
            criterion7_benchmark_ok=int(c7_ready),
            provisional_triggers="|".join(triggers) if triggers else "",
            tier=tier))

    path = os.path.join(ROOT, "results", "family_tiers.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    os.replace(path + ".tmp", path)

    for r in out:
        print(f"[tiers] {r['family']:14s} {r['tier']:12s} "
              f"benchmark {r['benchmark_exceed']}/{r['benchmark_nondeg']} | "
              f"triggers: {r['provisional_triggers'] or 'none'}")
    print(f"[tiers] {len(out)} families | "
          f"ready {sum(r['tier'] == 'ready' for r in out)} | "
          f"provisional {sum(r['tier'] == 'provisional' for r in out)} | "
          f"incidental {sum(r['tier'] == 'incidental' for r in out)}")


if __name__ == "__main__":
    main()
