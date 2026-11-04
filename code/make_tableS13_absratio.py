#!/usr/bin/env python3
"""make_tableS13_absratio.py — build Table S13 from results/exK_absratio_summary.csv.

Table S13 (Phase C1/C2) reports the absolute Procrustes drift reduction under
BOTH primary confirmatory variants and in 2-D and 3-D, closing reviewer issues
R5-6 / R5-7 / R4-2: the released absolute-ratio analysis existed for the
presence-PCA 2-D variant only, so the F6 headline was a single-variant result.

The table is generated from the results layer (never hand-written — X1 lesson).
Two-tier reporting policy (D133, matching Section 3.4 and Table 5): the results
layer flags every sub-tau median at-floor (reportable = 0), but a ratio is
withheld here only when the median family drift is double-precision residue
(delta_fam < 1e-15 — the three 3-parameter families in 3-D PCA, whose ~1e13
ratios are machine-precision denominators). A small-but-real median (F6 2-D,
4.9e-9, tail analyzed in Table S9) is reported with an at-floor flag, exactly
as Table 3 reports it; tau gates family status, not quotation.

Run: python3 code/make_tableS13_absratio.py
"""
from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "manuscript", "tables", "tableS13_absratio_variants.md")

TAU = 1e-6
DISPLAY = {
    "F1_physical": "F1 physical",
    "F2_metals": "F2 metals",
    "F3_nutrients": "F3 nutrients",
    "F4_organic": "F4 organic",
    "F1_physicala": "F1a physical--anion",
    "F1_clarity": "F1 clarity",
    "F6_aeration": "F6 aeration",
}
# short labels: long variant names overflow an 8-column tabular (the note defines them)
VMAP = {("pca_pres", 2): "PCA 2-D", ("pca_pres", 3): "PCA 3-D",
        ("mds_pres", 2): "MDS 2-D", ("mds_pres", 3): "MDS 3-D"}


def fmt(x, sig=3):
    """Plain decimal for mid-range values, LaTeX scientific notation for extremes."""
    if x != x:
        return "--"
    if x >= 1e4 or (x != 0 and x < 1e-3):
        mant, exp = f"{x:.2e}".split("e")
        exp = int(exp)
        sign = "-" if exp < 0 else ""
        return f"${mant}\\times 10^{{{sign}{abs(exp)}}}$"
    return f"{x:.4g}"


def main():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exK_absratio_summary.csv"),
                                    encoding="utf-8")))
    assert rows, "results/exK_absratio_summary.csv is empty"
    best = {}
    for r in rows:
        key = (r["variant"], int(r["dims"]))
        best.setdefault(key, []).append(r)
    assert best, "no variants found"

    # D197 scope guard. The canonical layer is the HEADLINE variant set: presence
    # PCA 2-D/3-D + Jaccard MDS 2-D, produced by
    # `experiment_K_absratio_variants.py --mds-2d-only`. A FULL run of that script
    # writes the same canonical paths with an extra MDS 3-D arm, which would
    # silently add rows to this table and contradict its own note ("MDS 3-D is
    # omitted: ... the 3-D comparison that motivates the sensitivity question is
    # PCA-specific"). Stop loudly instead of widening a publication artifact by
    # accident; an intentional scope change updates the note in the same commit.
    headline = {("pca_pres", 2), ("pca_pres", 3), ("mds_pres", 2)}
    extra, missing = set(best) - headline, headline - set(best)
    assert not extra, (
        f"exK layer carries units outside the headline set: {sorted(extra)} — "
        "Table S13's scope note would no longer hold. Regenerate the canonical "
        "layer with `--mds-2d-only`, or update the note deliberately")
    assert not missing, f"exK layer is missing headline units: {sorted(missing)}"

    # sanity: the released presence-PCA 2-D numbers must reproduce Table 3 exactly
    released = {"F1_physical": 73.25, "F2_metals": 31.53, "F3_nutrients": 4.973,
                "F4_organic": 155.8, "F1_physicala": 11.53, "F1_clarity": 3.904,
                "F6_aeration": 60900.3}
    for r in best[("pca_pres", 2)]:
        want = released[r["family"]]
        assert abs(float(r["raw_ratio"]) / want - 1.0) < 0.02, (
            f"Table S13 drift on {r['family']}: {r['raw_ratio']} vs released {want}")

    # D197: the intervals must agree with the Table 3 layer as well. Both scripts
    # now seed the paper-level bootstrap per (variant, dims, family) rather than
    # consuming ONE shared stream across the whole loop, which previously let the
    # same statistic carry different CIs in exG / exK / exM. This assertion is the
    # guard: it fails loudly if the seeding scheme ever diverges again.
    exg = {row["family"]: row for row in csv.DictReader(
        open(os.path.join(RESULTS, "exG_absratio.csv"), encoding="utf-8"))}
    for r in best[("pca_pres", 2)]:
        g = exg[r["family"]]
        for k13, k3 in (("raw_ci_lo", "ratio_raw_ci_lo"),
                        ("raw_ci_hi", "ratio_raw_ci_hi"),
                        ("paired_ci_lo", "median_paired_ratio_ci_lo"),
                        ("paired_ci_hi", "median_paired_ratio_ci_hi")):
            a, b = float(r[k13]), float(g[k3])
            assert abs(a - b) <= max(abs(b) * 0.005, 1.0), (
                f"Table S13 vs Table 3 interval mismatch on {r['family']} {k13}: "
                f"{a} (exK) vs {b} (exG) — the two bootstrap seeds have diverged")

    out = ["# Table S13 — Absolute Procrustes drift reduction by variant and dimensionality",
           "",
           "| Variant | Family | Events | All $\\delta$ | Family $\\delta$ | Raw ratio | Paired ratio | Reportable |",
           "|---|---|---:|---:|---:|---:|---:|:--:|"]
    for key in [("pca_pres", 2), ("mds_pres", 2), ("pca_pres", 3), ("mds_pres", 3)]:
        if key not in best:
            continue
        for r in sorted(best[key], key=lambda z: -float(z["raw_ratio"])):
            fam = DISPLAY[r["family"]]
            at_floor = str(r["at_floor"]).strip().lower() in ("true", "1")
            rep = str(r.get("reportable", "")).strip()
            assert (rep == "0") == at_floor, (
                f"inconsistent floor flags for {r['family']}: "
                f"at_floor={r['at_floor']} reportable={rep}")
            fam_med = float(r["fam_med"])
            # Two-tier reporting policy (D133). The results layer flags EVERY
            # sub-tau median at_floor/reportable=0, but the paper withholds a
            # ratio only when the denominator is double-precision residue: the
            # three 3-parameter families in 3-D PCA (delta ~ 1e-17, ratio
            # ~ 1e13 -- the same degeneracy criterion the random-family
            # benchmark applies at 1e-16). A small-but-real median (F6 2-D,
            # 4.9e-9, with the tail analysis of Table S9) is REPORTED with the
            # floor caveat, exactly as Table 3 and Section 3.4 do; the tau rule
            # gates family STATUS (Table 5), not quotation.
            RESIDUE = 1e-15
            residue = fam_med < RESIDUE
            assert not residue or at_floor, (
                f"residue row not flagged at_floor: {r['family']}")
            raw = "--" if residue else fmt(float(r["raw_ratio"]))
            paired = "--" if residue else fmt(float(r["paired_ratio"]))
            if residue:
                flag = "no ($\\delta<\\tau$: residue)"
            elif at_floor:
                flag = "yes (at floor)"
            else:
                flag = "yes"
            out.append(f"| {VMAP[key]} | {fam} | {r['n_events']} | "
                       f"{fmt(float(r['all_med']))} | {fmt(float(r['fam_med']))} | "
                       f"{raw} | {paired} | {flag} |")

    out += ["",
            "Note. PCA = binary presence PCA; MDS = Jaccard distance MDS; 2-D and 3-D are the leading subspace dimensions. Paired leave-one-paper-out design, same rows, family columns versus all 44 "
            "presence columns, evaluated over each family's cover rows. Raw ratio = "
            "$\\mathrm{median}(\\delta_{\\mathrm{all}})/\\mathrm{median}(\\delta_{\\mathrm{fam}})$; "
            "paired ratio = $\\mathrm{median}(\\delta_{\\mathrm{all}}/\\delta_{\\mathrm{fam}})$; "
            "95% intervals are paper-level cluster bootstrap resamples (1,000 draws, seed 42). "
            "Procrustes drift is scale-free on unit-Frobenius-norm clouds. "
            "Reporting follows the two-tier rule stated in Section 3.4 and Table 5: a ratio is "
            "\\textbf{withheld} only when the family median is double-precision residue "
            "($\\delta_{\\mathrm{fam}} < 10^{-15}$ — the three 3-parameter families in 3-D PCA, "
            "whose $\\sim 10^{13}$ ratios are machine-precision denominators, the same degeneracy "
            "criterion the random-family benchmark applies at $10^{-16}$). A median between "
            "$10^{-15}$ and the noise floor $\\tau = 10^{-6}$ (F6 aeration in 2-D presence PCA, "
            "$4.90 \\times 10^{-9}$, and in 2-D MDS, $5.34 \\times 10^{-7}$) is "
            "\\textbf{reported with an at-floor flag}, because the tail "
            "analysis (Table S9) shows genuine configuration-weight movement; the results layer "
            "marks every sub-$\\tau$ median at-floor/reportable = 0, and that strict flag gates "
            "family \\emph{status} in Table 5 rather than quotation. "
            "Families with exactly $m$ parameters collapse below $\\tau$ in an $m$-dimensional PCA, "
            "because a $p$-component PCA of $p$ parameters spans the family space exactly; this is why "
            "all three 3-parameter families (F6 aeration, F4 organic, F1 clarity) are unreportable in "
            "the 3-D PCA column, while the four families with more than three parameters return "
            "substantive ratios there. MDS 3-D is omitted: MDS embeddings retain stress and do not "
            "undergo the exact collapse, so the 3-D comparison that motivates the sensitivity question "
            "is PCA-specific. In the 2-D presence-PCA column this table reproduces Table 3: the point "
            "ratios and both interval endpoints of the shared variant are asserted against the released "
            "results layer at build time.",
            ""]

    # ---- Reading paragraph (Phase 4, D258): derived from exK, never hand-written.
    # The retired MDS arm made this paragraph stale (F6 5.0, clarity 92.9, F4/F1
    # physical 6.0). Every number below comes from the current results layer, and
    # the rank/floor assumptions are asserted so a future re-fit fails loudly
    # instead of silently leaving a wrong sentence in a publication artifact.
    def _row(variant, dims, family):
        for r in best[(variant, dims)]:
            if r["family"] == family:
                return r
        raise KeyError((variant, dims, family))

    def _ratio(variant, dims, family):
        return float(_row(variant, dims, family)["raw_ratio"])

    def _fmed(variant, dims, family):
        return float(_row(variant, dims, family)["fam_med"])

    def _f1(x):
        return f"{x:.1f}"

    def _f2(x):
        return f"{x:.2f}"

    f6_pca = _ratio("pca_pres", 2, "F6_aeration")
    f6_mds = _ratio("mds_pres", 2, "F6_aeration")
    f6_mds_med = _fmed("mds_pres", 2, "F6_aeration")
    assert f6_mds_med < TAU, "F6 MDS reading assumes a sub-tau family median"
    above = sorted([r for r in best[("mds_pres", 2)] if float(r["fam_med"]) > TAU],
                   key=lambda z: -float(z["raw_ratio"]))
    assert [r["family"] for r in above[:3]] == [
        "F1_physical", "F4_organic", "F1_clarity"], "MDS family order changed"
    assert [r["family"] for r in above[3:]] == [
        "F3_nutrients", "F1_physicala", "F2_metals"], "MDS family order changed"
    p3 = {r["family"]: r for r in best[("pca_pres", 3)]}
    for fam in ("F1_physical", "F3_nutrients", "F1_physicala", "F2_metals"):
        assert p3[fam]["raw_ratio"] not in ("", "nan"), f"3-D ratio missing: {fam}"

    reading = (
        f"Reading. The {f6_pca:,.0f}-fold F6 reduction is specific to 2-D presence PCA. "
        f"Under the Jaccard-MDS sensitivity (2-D) the F6 ratio is {_f1(f6_mds)}, but its "
        f"median family drift ({fmt(f6_mds_med)}) lies below $\\tau$, so the ratio sits at "
        "the noise floor and is not interpretable as a reduction. Among MDS families whose "
        f"median family drift is above $\\tau$, F1 physical leads ({_f1(float(above[0]['raw_ratio']))}), "
        f"followed by F4 organic ({_f1(float(above[1]['raw_ratio']))}) and F1 clarity "
        f"({_f1(float(above[2]['raw_ratio']))}); F3 nutrients, F1a and F2 metals lie between "
        f"{_f1(float(above[5]['raw_ratio']))} and {_f1(float(above[3]['raw_ratio']))}. "
        f"In 3-D presence PCA the reportable families give F1 physical "
        f"{_f1(float(p3['F1_physical']['raw_ratio']))} and F3 nutrients "
        f"{_f1(float(p3['F3_nutrients']['raw_ratio']))}, but F1a "
        f"({_f2(float(p3['F1_physicala']['raw_ratio']))}) and F2 metals "
        f"({_f2(float(p3['F2_metals']['raw_ratio']))}) show essentially no reduction, so the "
        "leading-plane result does not generalize across families.")
    out.append(reading)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    os.replace(tmp, OUT)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    print(f"  variants: {len(best)} | rows: {sum(len(v) for v in best.values())}")


if __name__ == "__main__":
    main()
