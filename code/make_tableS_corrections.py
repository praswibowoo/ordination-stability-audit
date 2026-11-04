#!/usr/bin/env python3
"""make_tableS_corrections.py — corrections pass: assemble supplementary tables S1-S10.

Reads frozen result CSVs and writes markdown tables consumed by
code/md2latex.py into the Supplementary material section:

  tableS1_sensitivity.md  alternative-assignment and subspace sensitivities
  tableS2_duplicate.md    duplicate-handling checks across three treatments
  tableS3_blockperm.md    paper-level permutation inference for the sparsity drift ratio
  tableS4_churn.md        corrected churn units (fraction = chi/2), LOPO vs LORO
  tableS5_verdict.md      full verdict-statistics matrix (all families x layers)
  tableS6_simulation.md   S1 calibration + S2 random-family + S3 config-matched control
  tableS7_d91.md          D91 verdict-restatement sensitivity
  tableS8_absdrift.md     absolute drift of paired design for all seven families
  tableS9_f6_absdrift.md  F6 paired design: absolute drift by density quartile

Run: python3 code/make_tableS_corrections.py
"""
from __future__ import annotations

import csv
import os
import numpy as np
import scipy.stats as stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "manuscript", "tables")

VARIANT_ORDER = ["pca_pres", "pca_vraw", "pca_vstd", "mds_pres", "tsne_pres"]

# frozen family definitions, imported so table builders never hard-code membership
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_B_familymaps import FAMILIES  # noqa: E402

FAM_DISPLAY = {
    "F1_physical": "F1 physical",
    "F1_physicala": "F1a physical--anion",
    "F1_clarity": "F1 clarity",
    "F2_metals": "F2 metals",
    "F3_nutrients": "F3 nutrients",
    "F4_organic": "F4 organic",
    "F6_aeration": "F6 aeration",
}


def read_csv(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_md(name, title, header, rows, note):
    path = os.path.join(TABLES, name)
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    if note:
        lines += ["", f"_{note}_"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {name}")


def _fmt(v):
    if v in (None, "", "nan"):
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def table_s1():
    recs = read_csv("exC2_sensitivity.csv")
    sens_map = {
        "pca3d": "PCA (3D)",
        "ward_k3": "Ward ($k=3$)",
        "ward_k5": "Ward ($k=5$)",
    }
    # R5-18: the pca3d rows carry no members string in exC2_sensitivity.csv; supply
    # them from the frozen FAMILIES definition rather than printing an em dash.
    pca3d_members = {f: ", ".join(x.split(" (")[0] for x in ps)
                     for f, ps in FAMILIES.items()}
    rows = []
    for r in recs:
        sens = sens_map.get(r["sensitivity"], r["sensitivity"])
        fam_raw = r["family"] or "—"
        name = FAM_DISPLAY.get(fam_raw, fam_raw.replace("_", " "))
        members = (r["members"] or "—").replace("|", ", ")
        if members == "—" and r["sensitivity"] == "pca3d" and r["family"] in pca3d_members:
            members = pca3d_members[r["family"]]
        rows.append([sens, name, members,
                     r["n_pairs"], f"{float(r['G_median']):+.3f}",
                     f"{float(r['P_G_gt0']):.3f}"])
    write_md("tableS1_sensitivity.md",
             "Table S1 — Alternative-assignment and subspace sensitivities for the ratio gain",
             ["Sensitivity", "Cluster / Family", "Members", "Pairs", "Gain $G$", "$P(G \\le 0)$"],
             rows,
             "pca3d = drift in the leading three-component subspace; ward_k3/k5 = raw Ward/Jaccard cluster families at cuts $k = 3$ and $k = 5$ under the same-rows paired design; $P$ = one-sided deletion-event bootstrap probability for $G \\le 0$. The F6 pca3d row is the top-3-PC sensitivity that does not reproduce the gain. Clusters with fewer than 10 complete rows or 3 members (e.g. W3C2, W5C2) are excluded by the size filter (main paper's family-derivation subsection). W5C4 is a distinct 6-parameter grouping (DO, pH, Temperature, COD, BOD, TSS), not a direct replication of the 3-parameter F6 suite, and pca3d membership comes from the family definitions. W5C3 shares parameters with F1 physical (DO, pH, Salinity, Temperature) yet reports the opposite sign of $G$ (+0.174 over 22 deletion events versus $-0.834$ over 121) because Ward-cluster rows condition on complete-cluster rows and family rows on cover rows, so the two must not be read against each other.")


def table_s2():
    recs = read_csv("duplicate_handling.csv")
    treat_map = {
        "full set": "Full set",
        "drop B111": "Drop B111",
        "drop B175": "Drop B175",
    }
    test_map = {
        "C1 pca_pres LOPO SI": "Presence PCA (LOPO SDR)",
        "C1 mds_pres LOPO SI": "Presence MDS (LOPO SDR)",
        "C1 pca_vstd LOPO SI": "Std-value PCA (LOPO SDR)",
        "C2 F6_aeration paired G": "F6 aeration (paired $G$)",
    }
    rows = []
    for r in recs:
        ci = (f"({float(r['ci_lo']):.3f}, {float(r['ci_hi']):.3f})"
              if r["ci_lo"] not in ("", None) else "—")
        treat = treat_map.get(r["treatment"], r["treatment"])
        test = test_map.get(r["test"], r["test"])
        rows.append([treat, test, f"{float(r['value']):.3f}",
                     f"{float(r['p_one']):.3f}", ci, r["n_events"]])
    mds = [float(r["value"]) for r in recs if r["test"].startswith("C1 mds_pres")]
    vstd = [float(r["value"]) for r in recs if r["test"].startswith("C1 pca_vstd")]
    pv = [float(r["p_two"]) for r in recs if r["test"].startswith("C1 pca_vstd")
          and r["p_two"] not in ("", None)]
    f6 = [float(r["value"]) for r in recs if r["test"].startswith("C2")]
    write_md("tableS2_duplicate.md",
             "Table S2 — Duplicate-handling checks for the headline statistics (three treatments)",
             ["Treatment", "Test", "Statistic", "$p$", "95% CI", "Events"],
             rows,
             "Treatments of the one known re-publication pair (B111/B175): full set, drop B111, "
             "drop B175. For the SDR rows $p$ is the one-sided paper-level permutation $p$ and for the F6 "
             "row $p$ is the one-sided deletion-event bootstrap probability for $G \\le 0$. The MDS SDR is "
             f"the sensitivity carrier (range across treatments {min(mds):.3f} to {max(mds):.3f}). "
             f"The standardized-value PCA is stable at SDR = {min(vstd):.3f} to {max(vstd):.3f} "
             f"(two-sided $p$ = {min(pv):.3f} to {max(pv):.3f}, and the table lists the one-sided values). The F6 gain is invariant at "
             f"$G$ = {min(f6):.3f} to {max(f6):.3f}.")


def table_s3():
    vraw_share = _vraw_scale()
    recs = {r["variant"]: r for r in read_csv("exC1_blockperm.csv")}
    var_map = {
        "pca_pres": "Presence PCA",
        "pca_vraw": "Raw-value PCA",
        "pca_vstd": "Std-value PCA",
        "mds_pres": "Presence MDS",
        "tsne_pres": "Presence $t$-SNE",
    }
    rows = []
    for v in VARIANT_ORDER:
        r = recs[v]
        v_name = var_map.get(v, v)
        rows.append([v_name, r["n_rows"], r["n_papers"], f"{float(r['SI']):.3f}",
                     f"{float(r['p_block']):.3f}",
                     f"{float(r['p_block_two']):.3f}",
                     f"({float(r['ci_lo']):.3f}, {float(r['ci_hi']):.3f})",
                     r["p_rowlevel_carried"]])
    # The manuscript S3 is the two-panel SDR-weighting table maintained by hand
    # (manuscript/tables/tableS3_blockperm.md). This generator produces the older
    # block-permutation view; write it under a distinct name so `make tables`
    # cannot overwrite the authoritative file.
    write_md("tableS3_blockperm_generated.md",
             "Table S3 — Sparsity drift ratio inference with the paper as the resampling unit",
             ["Variant", "Row evaluations", "Papers", "SDR", "$p$ (1-sided)",
              "$p$ (2-sided)", "95% CI", "$p_{\\mathrm{row}}$"],
             rows,
             "1000 paper-level permutations and 1000 cluster-bootstrap resamples, seed 42. "
             "$p$ (1-sided) is one-sided for the hypothesized direction and $p$ (2-sided) is twice "
             "the smaller tail. The 'Row evaluations' column counts row-level observations across "
             "deletion events ($N \\times (S-1)$), not distinct rows: presence variants have 764 "
             "distinct rows over 122 papers ($764 \\times 121 = 92{,}444$) and value variants 421 "
             "rows over 70 papers ($421 \\times 69 = 29{,}049$). Holm adjustment across the "
             "primary variant (standardized-value PCA) and the four sensitivity variants "
             "(presence PCA, presence MDS, raw-value PCA and presence $t$-SNE) leaves no variant "
             "significant: every Holm-adjusted $p = 1.000$, "
             "against unadjusted one-sided $p$ ranging from 0.437 to 1.000 (0.437 is the unadjusted "
             "minimum and Holm can only raise it). In unstandardized raw-value PCA (SDR = 0.623, 95% CI "
             "[0.218, 1.072], "
             "$p_{\\mathrm{block}} = 0.920$), salinity accounts for "
             f"{vraw_share:.2f}% of sample variance due to "
             "unscaled measurement units, pinning the leading axis so that deletion produces near-zero "
             "rotational drift (mean drift $1.3 \\times 10^{-6}$).")


def table_s4():
    recs = read_csv("exC1_churn_corrected.csv")
    pat_recs = read_csv("ex_pattern_churn.csv")
    pat_map = {(r["variant"], r["mode"]): r for r in pat_recs}
    keep = [r for r in recs if r["variant"] in ("pca_pres", "mds_pres", "tsne_pres")]
    var_map = {
        "pca_pres": "Presence PCA",
        "mds_pres": "Presence MDS",
        "tsne_pres": "Presence $t$-SNE",
    }
    rows = []
    for r in keep:
        v_name = var_map.get(r["variant"], r["variant"])
        mode = r["mode"].upper()
        pv = pat_map.get((r["variant"], r["mode"]))
        pat_frac = f"{float(pv['median_pat_churn']):.2f}" if pv is not None else "—"
        # Phase G1 tie-aware columns
        if pv is not None and pv.get("median_tie_aware_churn"):
            tie_med = f"{float(pv['median_tie_aware_churn']):.2f}"
            tie_frac = f"{float(pv['frac_rows_tie_aware_churn_gt0']) * 100:.1f}" + chr(37)
            tie_n = pv["n_rows_in_multiconfig_neigh"]
        else:
            tie_med = tie_frac = tie_n = "—"
        rows.append([v_name, mode, r["n"],
                     f"{float(r['median_chi']):.2f}",
                     f"{float(r['median_frac']):.2f}",
                     pat_frac, tie_med, tie_frac, tie_n,
                     f"{float(r['median_frac_Q1']):.2f} / {float(r['median_frac_Q2']):.2f} / "
                     f"{float(r['median_frac_Q3']):.2f} / {float(r['median_frac_Q4']):.2f}"])

    # Phase 4 (D258): the note's tie-aware percentages are derived, not literals.
    # The retired random-start MDS arm left 77.3% and 35.4% hard-coded here.
    lopo_tie = {v: pat_map.get((v, "lopo"))
                for v in ("pca_pres", "mds_pres", "tsne_pres")}
    assert all(v is not None for v in lopo_tie.values()), "missing LOPO tie-aware rows"

    def _tie_pct(variant):
        return f"{float(lopo_tie[variant]['frac_rows_tie_aware_churn_gt0']) * 100:.1f}"
    write_md("tableS4_churn.md",
             "Table S4 — Neighbour churn with corrected units, split by resampling mode",
             ["Variant", "Resampling", "Records", "Median $\\chi_{\\mathrm{raw}}$",
              "Median $\\chi$ (reported)", "Pattern", "Tie-aware median",
              "Tie-aware \\%>0", "Multi-cfg", "Idx. quantiles Q1--Q4"],
             rows,
             "The released layer stores raw churn $\\chi_{\\mathrm{raw}} \\in [0, 2]$ and the reported fraction $\\chi \\in [0, 1]$ (Section 2.1). Tie-aware churn counts replacements over distinct configurations, not row identifiers, and covers only rows whose $k=5$ neighbourhood spans more than one configuration. Its median is 0.00 for all three LOPO presence variants. The share of eligible rows with positive tie-aware churn is a median "
             + _tie_pct("pca_pres") + "\\% for presence PCA, " + _tie_pct("tsne_pres")
             + "\\% for t-SNE and " + _tie_pct("mds_pres") + "\\% for MDS."
             " This exposes neighbourhood turnover the index-level and pattern-level measures miss. Across all 122 papers and "
             f"{_pattern_events():,} deletion events multiset pattern churn is identically 0.00."
             " In the presence layer 94.1% of rows (719/764 across 155 unique patterns) share their binary configuration with $\\ge 1$ other row and 66.5% (508/764) sit in groups $> 5$, so index-level churn (median 0.20 for presence PCA) is permutation among identical-coordinate replicates, not movement across parameter states."
             " For submission this table ships as Table S4 in supplementary_main.tex, panels A to C, which adds the tie-rule sensitivity and the paper-level interval and lists the variants in the order the text uses, PCA, t-SNE, MDS.")


def table_s5():
    per = {(r["family"], r["layer"]): r for r in read_csv("exB_permanova.csv")}
    an = {(r["family"], r["layer"]): r for r in read_csv("exB_anosim.csv")}
    si = {(r["family"], r["layer"]): r for r in read_csv("exB_silhouette.csv")}
    keys = sorted(set(per) | set(an) | set(si))
    layer_map = {"presence": "Presence", "value": "Value"}
    rows = []
    for k in keys:
        p = per.get(k, {})
        a = an.get(k, {})
        s = si.get(k, {})
        fam = FAM_DISPLAY.get(k[0], k[0].replace("_", " "))
        lay = layer_map.get(k[1], k[1].capitalize())
        npap = p.get("papers", "")
        if k[1] == "value" and npap and int(npap) < 10:
            npap_str = f"{npap}$^\\dagger$"
        else:
            npap_str = str(npap)
        f_val = _fmt(p.get("pseudo_F"))
        if k == ("F1_physicala", "value") and p.get("pseudo_F") and float(p["pseudo_F"]) > 1e4:
            f_val = "—$^\\ddagger$"
        rows.append([
            fam, lay, p.get("n", s.get("n", "")), npap_str,
            f_val, _fmt(p.get("p_perm")),
            _fmt(p.get("pseudo_F_rho_residualized")), _fmt(p.get("p_perm_rho_residualized")),
            _fmt(a.get("R")), _fmt(a.get("p_perm")),
            _fmt(s.get("sil")), _fmt(s.get("sil_p"))])
    write_md("tableS5_verdict.md",
             "Table S5 — Full verdict-statistics matrix (all families x layers)",
             ["Family", "Layer", "$n$", "Papers", "$F$", "$p$", "$F_\\rho$", "$p_\\rho$",
              "$R$", "$p$", "$\\mathrm{sil}$", "$p$"],
             rows,
             "PERMANOVA and ANOSIM use 999 permutations restricted within papers and LABELED rows only; rows with no verdict (blank or TBD-StageB) are excluded. Silhouettes use 1000 shuffled-label draws. $F$ = PERMANOVA pseudo-$F$. $F_\\rho/p_\\rho$ = density-residualized PERMANOVA (presence layer only). $R$ = ANOSIM statistic. $\\mathrm{sil}$ = silhouette. $n$ is the labeled PERMANOVA denominator; a silhouette width drops classes with fewer than three rows, so its denominator can be 1--2 rows smaller. $^\\dagger$Exploratory configuration ($<10$ papers). $^\\ddagger$Not estimable (10 labeled rows, 3 papers, near-singular within-group variance). Confirmatory stability testing requires $\\ge 10$ complete rows and $\\ge 10$ papers. Negative silhouettes qualify the corresponding PERMANOVA result, since verdict classes are not compact in that space. Table S5 and Tables S12a/S12b share denominators because both condition on labeled rows; the stratified tables additionally split by label source. Full label-provenance detail for the F3 value layer and its B128 sensitivity is given in Table S12b and the Extended analyses.")


def _dag(p_str, f_str):
    """Append the degenerate-permutation dagger used by Table S12a: p >= 0.90 with
    F > 2 means within-paper permutations could not shuffle verdicts because most
    papers contain a single class, so the high p-value is structural."""
    try:
        if p_str in ("", "—", None, "nan") or f_str in ("", "—", None, "nan"):
            return "—" if p_str in ("", None, "nan") else p_str
        p, f = float(p_str), float(f_str)
        return f"{p_str}$^\\dagger$" if (p >= 0.90 and f > 2) else p_str
    except (TypeError, ValueError):
        return p_str


def table_s12a():
    """Phase H1: Table S12 split. S12a covers F1 physical and F6 aeration; S12b
    covers F3 nutrients, F4 organic and the B128 leave-one-out sensitivity. The
    combined table was 12 columns at \\tiny, which production flags."""
    strat = read_csv("exB_stratified.csv")
    f3_lopo = read_csv("exB_f3_lopo_sensitivity.csv")
    # author-labeled multi-class subset re-run (P8F, D173) — persisted artifact;
    # replaces the previously hardcoded values (found not reproducible from frozen layers)
    mc = read_csv("exB_multiclass.csv")[0]
    mc_F, mc_p = float(mc["pseudo_F"]), float(mc["p_perm"])
    mc_R, mc_Rp = float(mc["anosim_R"]), float(mc["p_anosim"])
    mc_sil, mc_silp = float(mc["silhouette"]), float(mc["p_silhouette"])
    layer_map = {"presence": "Presence", "value": "Value"}
    stratum_map = {"all": "Combined", "author": "Author-only", "rubric": "Rubric-only"}

    rows = []
    # Key families to display: F1 physical, F6 aeration, F3 nutrients, F4 organic
    for fam_key in ["F1_physical", "F6_aeration", "F3_nutrients", "F4_organic"]:
        fam_name = FAM_DISPLAY.get(fam_key, fam_key.replace("_", " "))
        for lay in ["presence", "value"]:
            sub = [r for r in strat if r["family"] == fam_key and r["layer"] == lay]
            for r in sub:
                if int(r["n"]) == 0 or r["pseudo_F"] == "" or r["pseudo_F"] == "nan":
                    continue
                # Flag low-sample degree-of-freedom artifact where SS_W -> 0
                if float(r["pseudo_F"]) > 1000:
                    f_str = f"{float(r['pseudo_F']):.1f}$^*$"
                else:
                    f_str = _fmt(r["pseudo_F"])

                # Flag within-paper permutation degeneracy: p >= 0.90 with F > 2
                is_deg = float(r["p_perm"]) >= 0.90 and float(r["pseudo_F"]) > 2.0
                p_str = f"{_fmt(r['p_perm'])}$^\\dagger$" if is_deg else _fmt(r["p_perm"])

                if lay == "presence":
                    f_res_str = _fmt(r["pseudo_F_res"])
                    if r["p_perm_res"] not in ("", None, "nan"):
                        is_deg_res = float(r["p_perm_res"]) >= 0.90 and float(r["pseudo_F_res"]) > 2.0
                        p_res_str = f"{_fmt(r['p_perm_res'])}$^\\dagger$" if is_deg_res else _fmt(r["p_perm_res"])
                    else:
                        p_res_str = "—"
                else:
                    f_res_str = "—"
                    p_res_str = "—"

                r_str = _fmt(r["anosim_R"])
                r_p_str = _fmt(r["p_anosim"])
                sil_str = _fmt(r["silhouette"])
                sil_p_str = _fmt(r["p_silhouette"])
                rows.append([
                    fam_name, layer_map.get(lay, lay), stratum_map.get(r["stratum"], r["stratum"]),
                    r["n"], r["papers"], r["groups"],
                    f_str, p_str, f_res_str, p_res_str,
                    f"{r_str} ({r_p_str})", f"{sil_str} ({sil_p_str})"
                ])
                if fam_key == "F1_physical" and lay == "value" and r["stratum"] == "author":
                    rows.append([
                        fam_name, "Value", "Author multi-class",
                        mc["n"], mc["papers"], mc["groups"],
                        f"{mc_F:.2f}", f"{mc_p:.2f}", "—", "—",
                        f"{mc_R:.2f} ({mc_Rp:.2f})", f"{mc_sil:.2f} ({mc_silp:.2f})"
                    ])
    write_md("tableS12a_verdict_stratified.md",
             "Table S12a — Provenance-stratified verdict separation, F1 physical and F6 aeration",
             ["Family", "Layer", "Stratum", "$n$", "Papers", "Grps", "$F$", "$p$", "$F_\\rho$", "$p_\\rho$", "$R$ ($p$)", "sil ($p$)"],
             rows,
             "All tests use 999 permutations restricted within papers. Silhouettes use 1000 shuffled draws. "
             "$^*$Low-sample residual degree-of-freedom artifact where within-group variance approaches zero ($n=6$, 2 papers, $p=0.381$). $^\\dagger$Degenerate permutation cell ($p \\ge 0.90$ with $F > 2$) where within-paper permutations cannot shuffle verdicts because most papers contain a single verdict class (F1 physical value author stratum: 17 of 30 papers and 78 of 135 rows single-class, despite $F = 4.43$). Re-running on the 13 multi-class papers alone ($n=57$) yields pseudo-$F = 0.81$, $p = 1.00$ (ANOSIM $R = 0.16$, $p = 0.73$; silhouette $-0.03$, $p = 0.40$). Provenance: 605 labeled data rows (267 author-provided, 338 rubric-derived) and 159 unlabeled (94 placeholder-label, 65 blank). The pre-specified rubric agrees with the authors on 23.0% of the 178 decidable rows (Cohen's $\\kappa = 0.126$, 95% CI 0.046 to 0.209). Full stratified detail and the F3 and B128 sensitivity are given in the Extended analyses.")


def table_s12b():
    """Phase H1: the second half of the split Table S12 — F3 nutrients, F4
    organic, and the F3 leave-one-paper-out sensitivity on study B128."""
    strat = read_csv("exB_stratified.csv")
    f3_lopo = read_csv("exB_f3_lopo_sensitivity.csv")
    fam_disp = {"F3_nutrients": "F3 nutrients", "F4_organic": "F4 organic"}
    lay_disp = {"presence": "Presence", "value": "Value"}
    strat_disp = {"all": "Combined", "author": "Author-only", "rubric": "Rubric-only"}

    rows = []
    for r in strat:
        if r["family"] not in fam_disp:
            continue
        rows.append([
            fam_disp[r["family"]], lay_disp.get(r["layer"], r["layer"]),
            strat_disp.get(r["stratum"], r["stratum"]),
            r["n"], r["papers"], r["groups"],
            _fmt(r.get("pseudo_F")), _fmt(r.get("p_perm")),
            _fmt(r.get("pseudo_F_res")), _dag(r.get("p_perm_res"), r.get("pseudo_F_res")),
            f"{_fmt(r.get('anosim_R'))} ({_fmt(r.get('p_anosim'))})",
            f"{_fmt(r.get('silhouette'))} ({_fmt(r.get('p_silhouette'))})"])

    b128 = next((r for r in f3_lopo if r["omitted_paper"] == "B128"), None)
    if b128:
        rows.append([
            "F3 nutrients", "Value", "Omit B128",
            b128["n"], b128["papers"], b128.get("n_groups", "4"),
            _fmt(b128["pseudo_F"]), _fmt(b128["p_perm"]), "—", "—",
            f"{_fmt(b128['anosim_R'])} ({_fmt(b128['p_anosim'])})",
            f"{_fmt(b128['silhouette'])} ({_fmt(b128['p_silhouette'])})"])
    assert rows, "no rows for Table S12b"

    write_md("tableS12b_f3_f4_sensitivity.md",
             "Table S12b — Provenance-stratified verdict separation, F3 nutrients and F4 organic, "
             "with the F3 leave-one-paper-out sensitivity",
             ["Family", "Layer", "Stratum", "$n$", "Papers", "Grps", "$F$", "$p$",
              r"$F_\rho$", r"$p_\rho$", "ANOSIM $R$ ($p$)", r"$\mathrm{sil}$ ($p$)"],
             rows,
             "Between-group tests permute 999 times within papers and silhouettes use 1000 "
             r"shuffled draws. $^\dagger$Single-class paper freezing: cells with ($p \ge 0.90$ with "
             "$F > 2$) arise where permutations cannot shuffle verdicts because most "
             "papers carry only one verdict class. The F3 leave-one-out row "
             "omits study B128 (18 rows dropped, $n = 17$ remaining): the PERMANOVA becomes non-significant "
             "($F = 4.88$, $p = 0.130$) while ANOSIM ($R = 0.593$, $p = 0.002$) and the "
             "silhouette ($0.260$, $p = 0.037$) remain significant, so dependence on a single "
             "study is statistic-dependent rather than uniform. Both F3 (7 papers) and F4 "
             "organic (3 papers) fall below the 10-paper confirmatory threshold set out in the main paper's "
             "Methods, so every value-layer row here is exploratory.")


def _paper_density_iqr():
    """Empirical IQR of per-paper mean row density (Phase F1 exact gamma -> % contrast)."""
    pres = list(csv.DictReader(open(os.path.join(
        ROOT, "docs", "extraction", "stageA", "matrix_presence_v2.csv"),
        encoding="utf-8-sig")))
    params = list(pres[0].keys())[7:51]
    own = ""
    per_paper = {}
    for r in pres:
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        if r["row_kind"] != "data" or not own:
            continue
        dens = sum(1 for p in params if (r[p] or "").strip() in ("Y", "NE")) / len(params)
        per_paper.setdefault(own, []).append(dens)
    prho = np.asarray([sum(v) / len(v) for v in per_paper.values()])
    return float(np.quantile(prho, 0.75) - np.quantile(prho, 0.25))


def _contrast_pct(gamma, delta_rho):
    """Mean-drift contrast across the density IQR induced by a planted effect gamma."""
    return 100.0 * (np.exp(gamma * delta_rho) - 1.0)


def table_s6():
    cal = read_csv("sim_calibration.csv")
    # aggregate replications per (sigma, gamma) and attach Monte Carlo SEs
    agg = {}
    for r in cal:
        k = (float(r["sigma_paper"]), float(r["gamma"]))
        d = agg.setdefault(k, [0, 0, 0])
        d[0] += int(r["rej_row"])
        d[1] += int(r["rej_block"])
        d[2] += 1
    n_rep = next(iter(agg.values()))[2]

    def _rate(num_den):
        num, den = num_den[0], num_den[1], num_den[2] if False else (num_den[0], num_den[1], num_den[2])
        return num / den

    rows = []
    for (sig, gam), (rr, rb, nn) in sorted(agg.items()):
        pr, pb = rr / nn, rb / nn
        rows.append(["S1", f"$\\sigma$={sig:g}, $\\gamma$={gam:g}", nn, "rejection",
                     f"{pr:.3f} $\\pm$ {(pr*(1-pr)/nn)**0.5:.3f} / "
                     f"{pb:.3f} $\\pm$ {(pb*(1-pb)/nn)**0.5:.3f}"])
    # Extended power sweep: the three largest gamma settings, 500 replicates each.
    ext = read_csv("sim_calibration_extended.csv")
    agg_ext = {}
    for r in ext:
        k = (float(r["sigma_paper"]), float(r["gamma"]))
        d = agg_ext.setdefault(k, [0, 0, 0])
        d[0] += int(r["rej_row"])
        d[1] += int(r["rej_block"])
        d[2] += 1
    for (sig, gam), (rr, rb, nn) in sorted(agg_ext.items()):
        pr, pb = rr / nn, rb / nn
        rows.append(["S1", f"$\\sigma$={sig:g}, $\\gamma$={gam:g}", nn, "rejection",
                     f"{pr:.3f} $\\pm$ {(pr*(1-pr)/nn)**0.5:.3f} / "
                     f"{pb:.3f} $\\pm$ {(pb*(1-pb)/nn)**0.5:.3f}"])
    p020 = agg[(0.8, 0.2)][1] / agg[(0.8, 0.2)][2]
    p060 = agg[(0.8, 0.6)][1] / agg[(0.8, 0.6)][2]
    delta_rho = _paper_density_iqr()
    c20, c60 = _contrast_pct(0.20, delta_rho), _contrast_pct(0.60, delta_rho)

    rand = read_csv("sim_random_families.csv")
    gs = [float(r["G"]) for r in rand]
    f6 = 0.9634
    exc_s2 = sum(1 for g in gs if g >= f6)
    p_s2 = (1 + exc_s2) / (1 + len(gs))
    d45 = [float(r["G"]) for r in rand if float(r["mean_slice_density"]) >= 0.45]
    exc_d45 = sum(1 for g in d45 if g >= f6)
    p_d45 = (1 + exc_d45) / (1 + len(d45))

    rows.append(["S2", "all sampled", len(rand), "median $G$", f"{np.median(gs):.3f}"])
    rows.append(["S2", "all sampled", len(rand), "exceedances", f"{exc_s2} ($p = {p_s2:.3f}$)"])
    rows.append(["S2", "density $\\bar{\\rho} \\ge 0.45$", len(d45), "exceedances",
                 f"{exc_d45} ($p = {p_d45:.3f}$)"])

    cfg = read_csv("exI_configmatch.csv")
    k5 = [r for r in cfg if int(r["n_configs"]) == 5]
    exc_k5 = sum(1 for r in k5 if float(r["G"]) >= f6)
    best_k5 = max(float(r["G"]) for r in k5)
    from scipy.stats import spearmanr as _sr
    r_s, p_rs = _sr([float(r["n_configs"]) for r in cfg],
                    [float(r["G"]) for r in cfg])

    rows.append(["S3", "exactly 5 configurations", len(k5), "exceedances",
                 f"{exc_k5} ($G = {best_k5:+.3f}$)"])
    rows.append(["S3", "configurations vs gain", len(cfg), "Spearman $r_s$",
                 f"{r_s:+.3f} ($p = {p_rs:.3f}$)"])

    note = (
        "Note: S1 = resampling-unit calibration; result = rejection rate at $\\alpha = 0.05$ for the "
        f"row-level / paper-level permutation test over {n_rep} synthetic replications per setting "
        "(500 for the three largest-$\\gamma$ settings) "
        "($\\sigma$ = sd of the paper drift effect, $\\gamma$ = planted density effect); "
        "$\\pm$ is the Monte Carlo standard error $\\sqrt{p(1-p)/n}$. The planted multiplier is "
        "$\\lambda_p = \\exp(\\gamma (\\mathrm{med}(\\rho) - \\rho_p))$, so a paper at the top of the "
        f"paper-density IQR and one at the bottom differ in mean drift by "
        f"$\\exp(\\gamma\\Delta\\rho) - 1$ with $\\Delta\\rho = {delta_rho:.4f}$ measured from the "
        f"corpus. $\\gamma = 0.20$ is therefore a {c20:.2f}% contrast. "
        "Paper-level power is low across the whole swept range: rejection is "
        f"{p020:.3f} at $\\gamma = 0.20$ and {p060:.3f} at $\\gamma = 0.60$ ({c60:.2f}% contrast), and "
        "no swept setting attains 80% power. At $\\gamma$ = 1.0, 1.5 and 2.0 (10.1%, 15.5% and 21.2%), "
        "paper-level power is 0.178, 0.226 and 0.396. Row-level inflation is graded in clustering strength "
        "(0.049, 0.118, 0.196, 0.256 at $\\sigma = 0, 0.2, 0.5, 0.8$) while the paper-level test stays "
        "at nominal (0.054, 0.037, 0.028, 0.053). S2 = random-family control; S3 = "
        "configuration-matched view (100 random 3-parameter families matching the F6 400--900-row "
        "span, counting exceedances of the observed F6 gain +0.963). In S2 and S3 the exceeding family is "
        "pH + TDS + Oil and Grease ($G = +0.989$, five presence configurations). One-sided $p$ uses "
        "$(1+k)/(1+n)$.")

    write_md("tableS6_simulation.md",
             "Table S6 — Validation experiments (S1 resampling-unit calibration, "
             "S2 random-family control, S3 configuration-matched view)",
             ["Test", "Setting", "$n$", "Statistic", "Result"],
             rows, note)


def table_s7():
    recs = read_csv("exB_d91_sensitivity.csv")
    frozen = {}
    for r in read_csv("exB_permanova.csv"):
        if r["layer"] == "value":
            frozen.setdefault(r["family"], {}).update(F=r["pseudo_F"], p=r["p_perm"])
    for r in read_csv("exB_anosim.csv"):
        if r["layer"] == "value":
            frozen.setdefault(r["family"], {}).update(R=r["R"], Rp=r["p_perm"])
    for r in read_csv("exB_silhouette.csv"):
        if r["layer"] == "value":
            frozen.setdefault(r["family"], {}).update(sil=r["sil"], silp=r["sil_p"])
    rows = []
    for r in recs:
        fam = FAM_DISPLAY.get(r["family"], r["family"].replace("_", " "))
        f = frozen.get(r["family"], {})
        rows.append([fam, "PERMANOVA $p$",
                     f.get("p", "—"), f"{float(r['p_perm']):.3f}"])
        rows.append([fam, "ANOSIM $R$ ($p$)",
                     f"{_fmt(f.get('R'))} ({_fmt(f.get('Rp'))})",
                     f"{_fmt(r['anosim_R'])} ({_fmt(r['anosim_p'])})"])
        rows.append([fam, "silhouette ($p$)",
                     f"{_fmt(f.get('sil'))} ({_fmt(f.get('silp'))})",
                     f"{_fmt(r['sil'])} ({_fmt(r['sil_p'])})"])
        rows.append(["—", "Rows analyzed (dropped)", f.get("n", "—"), f"{r['n']} ({r['dropped']} dropped)"])
    write_md("tableS7_d91.md",
             "Table S7 — Sensitivity of value-layer verdict tests to the verdict restatements",
             ["Family", "Statistic", "All rows", "Drop 9 restated"],
             rows,
             "Nine verdicts were restated to Harmful by the worst-case range-banding correction "
             "(B031 $\\times$5, B084 $\\times$3, B127 $\\times$1). Dropping those rows leaves overall statistical "
             "conclusions qualitatively consistent (e.g. F6 $p$ moves from 0.064 to 0.052, F1 $p$ moves from "
             "0.013 to 0.010, both preserving their respective significance tiers).")


def table_s8():
    recs = read_csv("exG_absratio.csv")
    def sci(v):
        if not v or float(v) == 0:
            return "0"
        mant, exp = f"{float(v):.2e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"

    rows = []
    for r in recs:
        fam_name = FAM_DISPLAY.get(r["family"], r["family"].replace("_", " "))
        rows.append([fam_name, sci(r["all_med"]), sci(r["fam_med"]),
                     f"{float(r['ratio_raw']):.0f}",
                     f"{float(r['median_paired_ratio']):.0f}",
                     f"{float(r['ratio_unique_cfg']):.0f}"])
    write_md("tableS8_absdrift.md",
             "Table S8 — Absolute drift and drift reduction ratios for all seven families (presence PCA, 2-D)",
             ["Family", "All $\\delta$", "Family $\\delta$", "Raw ratio", "Paired ratio (med)", "Unique-cfg ratio"],
             rows,
             "Medians across leave-one-paper-out events for presence PCA (2-D subspace). Raw ratio = median($\\delta_{\\mathrm{all}}$) / median($\\delta_{\\mathrm{fam}}$); paired ratio = median of per-event $\\delta_{\\mathrm{all}} / \\delta_{\\mathrm{fam}}$; unique-cfg ratio = median($\\delta_{\\mathrm{all}}$) / median($\\delta_{\\mathrm{fam,cfg}}$) with unique binary configurations weighted equally. Drift is scale-free on unit Frobenius norm clouds. Against 100 random 3-parameter suites (Table S6), F6's raw drift ratio (60,900) is exceeded by 19 of 100 suites ($p = 0.198$), but 87 of the 100 have median family drift below $\\tau = 10^{-6}$ and sit at the same coordinate floor. Above $\\tau$, 0 of 13 exceed ($p = 0.071$); at $10^{-9}$, 2 of 83 exceed ($p = 0.036$), so the benchmark is threshold-sensitive. On ratio gain $G$, only 1 of 100 random suites exceeds F6 ($p = 0.020$; $p = 0.043$ density-restricted). Extreme drift reduction is an arithmetic consequence of the discrete binary configuration space, not unique ecological coupling.")


def table_s9():
    recs = read_csv("exC2_absolute_drift.csv")
    sub = [r for r in recs if r["family"] == "F6_aeration"]
    agg = {k: float(np.median([float(r[k]) for r in sub]))
           for k in ("all_med", "all_sparse", "all_dense",
                     "fam_med", "fam_sparse", "fam_dense")}
    tail_summary = {r["metric"]: float(r["value"]) for r in read_csv("ex_f6_tail_summary.csv")}

    def sci(v):
        mant, exp = f"{v:.2e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"

    rows = [["All-parameter (44 parameters)", sci(agg["all_med"]), sci(agg["all_sparse"]),
             sci(agg["all_dense"])],
            ["Family-restricted (3 parameters)", sci(agg["fam_med"]), sci(agg["fam_sparse"]),
             sci(agg["fam_dense"])]]
    med_fam = tail_summary.get("median_fam_med", 4.903e-09)
    med_max = tail_summary.get("median_fam_max", 2.853e-06)
    pct_tau = tail_summary.get("pooled_share_above_tau6", 0.178) * 100
    n_gt10 = int(tail_summary.get("events_gt_10pct_above_tau", 49))
    n_gt50 = int(tail_summary.get("events_gt_50pct_above_tau", 5))
    n_events = int(tail_summary.get("n_events", 112))
    def sci_tex(v):
        mant, exp = f"{v:.2e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"

    med_fam_str = sci_tex(med_fam)
    med_max_str = sci_tex(med_max)
    write_md("tableS9_f6_absdrift.md",
             "Table S9 — F6 paired design: absolute drift by density quartile",
             ["Space", "Median $\\delta$", "Sparse quartile ($Q_1$)", "Dense quartile ($Q_4$)"],
             rows,
             f"Medians across the {n_events} F6 deletion events in 2-D presence PCA. While the median family-restricted drift sits below the numerical "
             f"precision threshold ({med_fam_str}), full row-level tail evaluation reveals that {pct_tau:.1f}% of pooled row evaluations "
             f"exceed $\\tau = 10^{{-6}}$ with an event-median maximum drift of {med_max_str} ({n_gt10}/{n_events} events have $>10\\%$ and {n_gt50}/{n_events} "
             f"have $>50\\%$ of rows above $\\tau$). All {n_events} deletion events strictly preserve all 5 unique presence configurations ($5 \\to 5$), "
             f"demonstrating that drift in the tail is driven by configuration-weight $L1$ frequency shifts under deletion rather than discrete configuration loss "
             f"(Spearman $r = 0.635$, $p = 5.72 \\times 10^{{-14}}$ and Pearson $r = 0.293$, $p = 0.0017$ between per-event configuration-weight $L1$ shift and maximum drift).")


def _sci(v, digits=1):
    """Format a float as LaTeX scientific notation, or plain decimal if mid-range."""
    t = f"{v:.{digits}e}"
    if "e" in t:
        mant, exp = t.split("e")
        sign = "-" if int(exp) < 0 else ""
        return f"${mant}\\times 10^{{{sign}{abs(int(exp))}}}$"
    return f"${v:.2f}$"


def table_s10():
    """Volume-matched deletion control. Phase F2: every figure is read from the
    results layer. The previous version carried beta_block = 0.333,
    beta_logV = 0.844 and p = 6.63e-6 as literal strings with no script behind
    them, and the LORO row was hand-entered from the C1 layer."""
    summ = read_csv("exH_volumematch_summary.csv")[0]      # wide single row
    md = {r["term"]: r for r in read_csv("exH_loglinear_model.csv")}
    c1s = {r["mode"]: r for r in read_csv("exC1_summary.csv")
           if r["variant"] == "pca_pres"}
    c1c = {(r["variant"], r["mode"]): r for r in read_csv("exC1_churn_corrected.csv")}
    loro_drift = float(c1s["loro"]["med_drift"])
    loro_churn = float(c1c[("pca_pres", "loro")]["median_frac"])
    loro_si = round(float(c1s["loro"]["SI"]), 3)

    # Control-overlap audit. The scattered arm draws uniformly from all rows, so a
    # draw can contain rows of the paper it is meant to control for. Record the
    # frequency and direction here rather than leaving it implicit in the CSV.
    draws = read_csv("exH_volumematch_draws.csv")
    first = [r for r in draws if r["draw"] == "0"]
    assert len(first) == int(summ["n_total"]), "first-draw count != paper count"
    ov = [int(r["overlap_with_paper"]) for r in draws]
    fov = [int(r["overlap_with_paper"]) for r in first]
    n_over, n_first_over = sum(1 for x in ov if x > 0), sum(1 for x in fov if x > 0)
    assert n_over and n_first_over, "no control overlap; delete the overlap sentence"
    assert max(fov) == 1, "first-draw overlap exceeds one row; reword the note"
    med_ov = float(np.median([float(r["med_drift"]) for r in first
                              if int(r["overlap_with_paper"]) > 0]))
    med_dis = float(np.median([float(r["med_drift"]) for r in first
                               if int(r["overlap_with_paper"]) == 0]))

    blk, lv = md["block"], md["log_volume"]
    assert int(blk["n_obs"]) == 122 * (1 + int(summ["n_draws"])), (
        f"log-linear n_obs {blk['n_obs']} != 122 x (1 + {summ['n_draws']})")

    rows = [
        ["Block (LOPO)", summ["n_rows_deleted_lopo"], _sci(float(summ["d_lopo_med"]), 2),
         f"{float(summ['churn_lopo_med']):.2f}", summ["pooled_si_lopo"]],
        ["Volume-matched", summ["n_rows_deleted_matched"],
         _sci(float(summ["d_matched_med"]), 2),
         f"{float(summ['churn_matched_med']):.2f}", summ["pooled_si_matched"]],
        ["Single row (LORO)", "1", _sci(loro_drift, 1), f"{loro_churn:.2f}",
         f"{loro_si:.3f}"],
    ]
    wilcox_p = float(summ["wilcoxon_p"])
    note = (
        "Median per-event Procrustes drift $\\delta$, neighbour churn fraction ($k=5$), and "
        "pooled sparsity drift ratio SDR across 122 leave-one-paper-out (LOPO) deletion blocks, "
        f"{summ['n_draws']} volume-matched scattered draws per paper, and 764 "
        "leave-one-row-out (LORO) events. Median ratio: "
        f"{summ['ratio_of_medians']} (95% CI {summ['ratio_of_medians_ci_lo']} to "
        f"{summ['ratio_of_medians_ci_hi']}, spanning unity). Paired-ratio median: "
        f"{summ['median_paired_ratio']} (95% CI {summ['median_paired_ratio_ci_lo']} to "
        f"{summ['median_paired_ratio_ci_hi']}, excluding unity). Paired Wilcoxon "
        f"$p = {wilcox_p:.2e}$, with {summ['n_block_gt_scattered']} of {summ['n_total']} "
        "papers showing higher drift under block deletion. Median neighbour churn is identical "
        "between the two procedures at the median because churn takes discrete multiples of 0.20, but paired "
        f"comparison shows block churn is higher in {summ['churn_higher_block']} papers, lower "
        f"in {summ['churn_lower_block']}, and equal in {summ['churn_equal']}. Deletion "
        f"fragility scales with deleted data volume (Spearman $\\rho = "
        f"{summ['spearman_rho_block']}$ block, $\\rho = "
        f"{summ['spearman_rho_scattered']}$ scattered). In a fitted log-linear model "
        "$\\log\\delta \\sim \\log V + \\mathrm{block}$ over all "
        f"{blk['n_obs']} events with paper-level cluster-bootstrap CIs, study block structure "
        f"contributes $\\beta_{{\\mathrm{{block}}}} = {blk['coef']}$ (multiplicative factor "
        f"{blk['mult_factor']}, 95% CI {blk['mult_lo']} to {blk['mult_hi']}; bootstrap "
        f"$p < 5 \\times 10^{{-4}}$, the resolution limit of {blk['n_boot']} resamples), while "
        f"the volume elasticity is $\\beta_{{\\log V}} = {lv['coef']}$ "
        f"({lv['mult_factor']}$\\times$ per e-fold of deleted volume, 95% CI "
        f"{lv['mult_lo']} to {lv['mult_hi']}). Block structure therefore adds roughly "
        f"{100 * (float(blk['mult_factor']) - 1):.0f}% drift on top of the "
        "dominant volume scaling. The scattered control draws uniformly from all rows, so a "
        f"draw can include rows of the deleted paper: {n_over} of {len(draws):,} draws "
        f"({100 * n_over / len(draws):.1f}%; at most {max(ov)} rows) do, and {n_first_over} "
        f"of the {len(first)} paired first draws share a row with the deleted paper. Among first "
        f"draws, overlapping controls drift slightly more than disjoint controls (median {_sci(med_ov, 2)} vs "
        f"{_sci(med_dis, 2)}), so the block-versus-scattered contrast is conservative.")

    write_md("tableS10_volumematch.md",
             "Table S10 — Volume-matched deletion control (presence PCA)",
             ["Procedure", "Rows deleted", "Drift $\\delta$", "Churn", "Pooled SDR"],
             rows, note)


def table_3():
    recs = read_csv("exG_absratio.csv")
    recs.sort(key=lambda r: float(r["ratio_raw"]), reverse=True)
    def sci(v):
        if not v or float(v) == 0:
            return "0"
        mant, exp = f"{float(v):.2e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"

    rows = []
    for r in recs:
        fam_name = FAM_DISPLAY.get(r["family"], r["family"].replace("_", " "))
        raw_ci = f"({int(round(float(r['ratio_raw_ci_lo']))):d}, {int(round(float(r['ratio_raw_ci_hi']))):d})"
        paired_ci = f"({int(round(float(r['median_paired_ratio_ci_lo']))):d}, {int(round(float(r['median_paired_ratio_ci_hi']))):d})"
        if r["family"] == "F6_aeration":
            status = "Median at floor"
        elif r["at_floor"] == "True" or r["at_floor"] is True:
            status = "At floor ($\\tau = 10^{-6}$)"
        elif int(r["fam_sparse_eq_dense_events"]) > 0:
            status = f"$\\mathrm{{SDR}}_{{\\mathrm{{fam}}}}=1$ ({r['fam_sparse_eq_dense_events']}/{r['n_events']})"
        else:
            status = "Above floor"
        rows.append([fam_name, r["n_events"], sci(r["all_med"]), sci(r["fam_med"]),
                     f"{float(r['ratio_raw']):.0f} {raw_ci}",
                     f"{float(r['median_paired_ratio']):.0f} {paired_ci}",
                     status])
    write_md("table3_c2.md",
             "Table 3 — Absolute Procrustes drift reduction across co-reporting families (presence PCA, 2-D)",
             ["Family", "Events", "All $\\delta$", "Family $\\delta$", "Raw (CI)", "Paired (CI)", "Floor status"],
             rows,
             "")


def table_s11():
    recs = read_csv("exC2_summary.csv")
    recs.sort(key=lambda r: -float(r["G_median"]))
    rows = []
    for r in recs:
        fam_name = FAM_DISPLAY.get(r["family"], r["family"].replace("_", " "))
        g = float(r["G_median"])
        lo, hi = (float(x) for x in r["G_CI95"].strip("[]()").replace(" ", "").split(","))
        h = float(r["P_G_gt0_holm"])
        rows.append([
            fam_name,
            r["variant"],
            r["n_pairs"],
            f"{float(r['SI_family_median']):.3f}",
            f"{float(r['SI_all_median']):.3f}",
            f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            f"{float(r['P_G_gt0']):.3f}",
            f"{h:.3f}",
        ])
    write_md("tableS11_absratio.md",
             "Table S11 — Family-restricted ratio gain G across co-reporting families (same-rows paired design)",
             ["Family", "Variant", "Pairs", "$\\mathrm{SDR}_{\\mathrm{fam}}$", "$\\mathrm{SDR}_{\\mathrm{all}}$", "$G$ [95% CI]", "$P(G \\le 0)$", "Holm"],
             rows,
             "Ratio gain $G = 1 - \\mathrm{median}(\\mathrm{SDR}_{\\mathrm{fam}}) / \\mathrm{median}(\\mathrm{SDR}_{\\mathrm{all}})$ is reported as a secondary summary. The primary stability assessment is reported via absolute Procrustes drift ratios in Table 3. Negative gains in this table reflect configuration-class coordinate collapse ($\\mathrm{SDR}_{\\mathrm{fam}} = 1.000$) rather than genuine geometric instability. Holm adjustment is evaluated within each pairing variant ($m = 7$ for pca_pres, $m = 3$ for mds_pres). We use 1000 cluster bootstrap resamples over paper-deletion events, seed 42.")



def _pattern_events():
    """Total deletion events across every variant/mode in the churn layer (D199).

    The S4 note states "122 papers and 1,130 deletion events"; 1,130 is the sum of
    ex_pattern_churn.csv's events_evaluated column (122+764+122+122), previously
    typed into the note by hand.
    """
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "ex_pattern_churn.csv"),
                                    encoding="utf-8-sig")))
    n = sum(int(r["events_evaluated"]) for r in rows)
    assert n == 1130, f"pattern-churn event total drifted: {n}"
    return n


def _vraw_scale():
    """Share of total variance held by the dominant unstandardized parameter (D199)."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "exC1_vraw_scale.csv"),
                                    encoding="utf-8")))
    top = [r for r in rows if r["rank"] == "1"][0]
    assert top["param"] == "Salinity (ppt/‰)", top
    return float(top["share_pct"])


def main():
    for fn in (table_s1, table_s2, table_s3, table_s4, table_s5, table_s6, table_s7,
               table_s8, table_s9, table_s10, table_s11, table_s12a, table_s12b,
               table_3):
        fn()


if __name__ == "__main__":
    main()
