#!/usr/bin/env python3
"""G3 numbers-consistency checker — "numbers in text = numbers in outputs".

Automates the G3 spot-check (P1_Master_Plan §10) beyond the manual 10-item
sample: every claim below locates a value in a frozen results CSV and asserts
the corresponding manuscript-facing doc quotes it (rounding-tolerant for stats,
exact for counts). Text is normalized (unicode minus → '-', '.0375' style
tokens supported) and scientific notation is resolved to true values, so
'4.9 $\\times$ 10$^{-9}$' is matched as one number rather than as fragments.

Usage:  python3 code/check_numbers.py            # exit 0 = all PASS
        python3 code/check_numbers.py -v         # list every claim
        python3 code/check_numbers.py --selftest # known-answer fixtures for the matcher
Extend by appending to CLAIMS when new headline stats enter packs/memos.

D198 added two context primitives for claims whose *ordering* carries meaning:
`near=<regex>` (the pattern must appear within SPAN characters of the matched
number) and `@phrase` (an ordered literal must appear). Before them, swapping
two numbers inside one sentence — e.g. the MDS sparse/dense medians, or any two
of the four density-baseline drifts — left the suite green. 26 fixtures cover the
matcher (`--selftest`).

D202 (two class-level repairs, after D198/D201 fixed instances):
  * TYPOGRAPHY IS NOT A STATISTIC. LaTeX dimension tokens (cm/pt/em, \\linewidth,
    geometry options such as width=/left=/text width=, \\tabcolsep, \\arraystretch,
    \\includegraphics[...]) are dropped at tokenisation. Before this, a TikZ dimension
    `0.35cm` satisfied a claim on beta_block = 0.3486, so corrupting the digest's prose
    passed the gate.
  * CLAIMS ABOUT PROSE CANNOT BE SATISFIED BY GENERATED TABLES. For .tex documents the
    text view excludes tabular/tabularx bodies, tikzpicture blocks, \\includegraphics
    and \\setlength, plus the preamble. Embedded SI tables are guarded by their own
    builders; the digest's table-only claims were dropped rather than left to rest on
    content they do not own (four removed: two redundant D160 S14 pins, two of mine).

SCOPE LIMITATION (state it plainly; do not over-trust a green run). This is a
PRESENCE test: it asserts the value appears SOMEWHERE in the named document, not
that it appears in the right sentence. A value that recurs across contexts
(0.001, 0.007, 57, 134) can therefore satisfy a claim by coincidence. Two
consequences:
  - when content is relocated between documents (e.g. the D174 move of
    verdict-layer detail into manuscript/latex/supplementary_rev2.tex), the CLAIM must be
    retargeted by hand; the checker cannot detect the move;
  - the guarantee is "this number is stated in this document", not "this number
    is stated here, about this".
The D197 tolerance repair removed the vacuous sub-0.001 class, where the band
was wider than the guarded value; the coincidence class above remains open and
is tracked as a known limit rather than silently assumed away.
Tokenisation (D228): a decimal point is never split, so "3.2" yields 3.2 and never
3, but an integer followed by a sentence-final full stop is accepted, so a
paragraph-final number is visible to the gate rather than silently skipped.

Read-only: never writes. Seed irrelevant (no randomness).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")

# (doc_path, csv_file, filter, column, kind)  kind: "stat" (tol) | "count" (exact)
CLAIMS: list[tuple[str, str, dict, str, str]] = [
    # ---- C1 pack (exC1_summary) ----
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "SI", "stat"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "perm_p", "stat"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "mean_churn", "stat"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "SI", "stat"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "perm_p", "stat", r"0\.039"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="pca_pres", mode="lopo"), "SI", "stat"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="tsne_pres", mode="lopo"), "SI", "stat"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "n_events", "count"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "n_events", "count"),
    # ---- P8E (D171): primary-MDS fit quality (mds_stress.py) ----
#     ("manuscript/latex/main_rev2.tex", "mds_stress_summary.csv", dict(scope="full"), "stress1_med", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "mds_stress_summary.csv", dict(scope="lopo"), "stress1_med", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "mds_stress_summary.csv", dict(scope="lopo"), "stress1_q1", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "mds_stress_summary.csv", dict(scope="lopo"), "stress1_q3", "stat"),
    # ---- C2 pack (exC2_summary) ----
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "n_pairs", "count"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0_holm", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "P_G_gt0_holm", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F1_physical", variant="pca_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F3_nutrients", variant="pca_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F4_organic", variant="pca_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F2_metals", variant="pca_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F1_physicala", variant="pca_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_summary.csv", dict(family="F1_clarity", variant="pca_pres"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_sensitivity.csv", dict(family="W5C4"), "G_median", "stat"),
    ("docs/phase4_C2_pack.md", "exC2_sensitivity.csv", dict(sensitivity="pca3d", family="F1_physical"), "G_median", "stat"),
    # ---- C3 pack (exB_*) ----
    ("docs/phase4_C3_pack.md", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("docs/phase4_C3_pack.md", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("docs/phase4_C3_pack.md", "exB_permanova.csv", dict(family="F6_aeration", layer="value"), "p_perm", "stat"),
    ("docs/phase4_C3_pack.md", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("docs/phase4_C3_pack.md", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "p_perm_rho_residualized", "stat"),
    ("docs/phase4_C3_pack.md", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("docs/phase4_C3_pack.md", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "p_perm_rho_residualized", "stat", r"0\.067"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F1_physical", layer="presence"), "R", "stat"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F1_physical", layer="presence"), "p_perm", "stat"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F1_physical", layer="value"), "R", "stat"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F3_nutrients", layer="value"), "R", "stat"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F6_aeration", layer="presence"), "R", "stat"),
    ("docs/phase4_C3_pack.md", "exB_anosim.csv", dict(family="F6_aeration", layer="presence"), "p_perm", "stat"),
    ("docs/phase4_C3_pack.md", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil", "stat"),
    ("docs/phase4_C3_pack.md", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil_p", "stat"),
    ("docs/phase4_C3_pack.md", "exB_family_si.csv", dict(family="F6_aeration", layer="presence"), "SI", "stat"),
    # ---- D pack (exD_tests) ----
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="presence"), "n_pairs", "count"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="presence"), "within_minus_between", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="presence"), "p_within_lt_between", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="presence"), "shared_minus_other", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="presence"), "p_shared_lt_other", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="presence"), "concordance_Q", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="presence"), "paper_silhouette", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="value"), "n_pairs", "count"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="value"), "within_minus_between", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="value"), "shared_minus_other", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="value"), "concordance_Q", "stat"),
    ("docs/phase4_D_pack.md", "exD_tests.csv", dict(layer="value"), "paper_silhouette", "stat"),
    # ---- G2 memo (headliners) ----
    ("docs/G2_memo.md", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "SI", "stat"),
    ("docs/G2_memo.md", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "SI", "stat"),
    ("docs/G2_memo.md", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("docs/G2_memo.md", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("docs/G2_memo.md", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    ("docs/G2_memo.md", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0_holm", "stat"),
    # ---- figure-caption numbers (D118; retargeted to the draft after the brief-caption pass:
    #      captions now state only what a panel shows, so the numbers live in the text) ----
    ("manuscript/latex/supplementary_main.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "SI", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "perm_p", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "SI", "stat"),
    ("manuscript/latex/main_rev2.tex", "r1_anchor.csv", dict(variant="presence PCA (2-D)"), "drift_pct_of_interrow", "stat"),
    ("manuscript/latex/main_rev2.tex", "r2_per_event_calibration.csv", dict(gamma=0.0), "rejection_rate", "stat"),
    ("manuscript/latex/main_rev2.tex", "r2_per_event_calibration.csv", dict(gamma=0.2), "rejection_rate", "stat"),
    ("manuscript/latex/main_rev2.tex", "r2_per_event_calibration.csv", dict(gamma=0.6), "rejection_rate", "stat"),

    # 2026-11-03: within-paper p = 0.039 retired from the main text (uncalibrated null;
    # the sentence now states only that the test is not used for inference). Claim retired.
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "perm_p", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F6_aeration", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "p_perm_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F3_nutrients", layer="value", stratum="rubric"), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F3_nutrients", layer="value", stratum="rubric"), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil_p", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0_holm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "P_G_gt0_holm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "within_minus_between", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "within_minus_between", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "shared_minus_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "shared_minus_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "p_shared_lt_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "p_shared_lt_other", "stat"),
    # ---- draft v0.9 (methods_results) — C1 ----
    ("manuscript/latex/supplementary_main.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "SI", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "perm_p", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "med_drift", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "mean_drift", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "med_lo", "stat", r"95\\% CI 0\.00079 to 0\.00141"),
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "med_hi", "stat", r"95\\% CI 0\.00079 to 0\.00141"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "SI", "stat"),
    # 2026-11-03: duplicate perm_p mapping retired with the main-text sentence (see first block).
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "perm_p", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_pres", mode="lopo"), "SI", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_pres", mode="loro"), "SI", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="tsne_pres", mode="lopo"), "SI", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="pca_pres", mode="lopo"), "median_frac", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="pca_pres", mode="lopo"), "median_frac", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="mds_pres"), "p_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_p", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exC1_blockperm.csv", dict(variant="pca_pres"), "p_block_two", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="tsne_pres"), "p_block_two", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_radius_check.csv", dict(association="radius vs per-row median drift"), "spearman", "stat"),
    ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="full set", test="C1 pca_vstd LOPO SI"), "p_two", "stat"),
    ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B175", test="C1 pca_vstd LOPO SI"), "p_two", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_radius_check.csv", dict(association="density rho vs per-row median drift"), "spearman", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="mds_pres"), "p_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_p", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="mds_pres"), "p_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_p", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_pres", mode="lopo"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "n_events", "count"),
    # ---- draft — C2 ----
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0_holm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "SI_family_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "SI_all_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "P_G_gt0_holm", "stat"),
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(family="F1_physical", variant="pca_pres"), "G_median", "stat"),  # D140: value shown in Table S11
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(family="F1_physical", variant="mds_pres"), "G_median", "stat"),  # D140: value shown in Table S11
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(family="F2_metals", variant="pca_pres"), "G_median", "stat"),  # D140: value shown in Table S11
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(family="F3_nutrients", variant="pca_pres"), "G_median", "stat"),  # D140: value shown in Table S11
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(family="F4_organic", variant="pca_pres"), "G_median", "stat"),  # D140: value shown in Table S11
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(family="F1_physicala", variant="pca_pres"), "G_median", "stat"),  # D140: value shown in Table S11
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(family="F1_clarity", variant="pca_pres"), "G_median", "stat"),  # D140: value shown in Table S11
    ("manuscript/latex/main_rev2.tex", "exC2_sensitivity.csv", dict(family="W5C4"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(sensitivity="pca3d", family="F6_aeration"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(sensitivity="pca3d", family="F6_aeration"), "P_G_gt0", "stat"),
    # ---- draft — C3 ----
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F6_aeration", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F3_nutrients", layer="value"), "R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "p_perm_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "p_perm_rho_residualized", "stat", r"p = 0\.067"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F6_aeration", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F6_aeration", layer="presence"), "p_perm_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F3_nutrients", layer="value", stratum="rubric"), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F3_nutrients", layer="value", stratum="rubric"), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F1_physical", layer="presence"), "R", "stat", r"R = 0\.041"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F1_physical", layer="presence"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F6_aeration", layer="presence"), "R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F6_aeration", layer="presence"), "p_perm", "stat", r"R = 0\.088.*p = 0\.001"),
    ("manuscript/latex/main_rev2.tex", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil_p", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_family_si.csv", dict(family="F3_nutrients", layer="value"), "n_rows", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_family_si.csv", dict(family="F1_physical", layer="value"), "n_rows", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_family_si.csv", dict(family="F6_aeration", layer="value"), "n_rows", "count"),
    # ---- S16 positive control ----
    ("manuscript/latex/main_rev2a.tex", "@phrase", {}, "only 8\\% in the 57-row subset", "phrase"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="0.0"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="0.0"), "anosim_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="0.25"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="0.25"), "anosim_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="0.5"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="0.5"), "anosim_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="1.0"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power.csv", dict(delta="1.0"), "anosim_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="0.0"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="0.0"), "anosim_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="0.25"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="0.25"), "anosim_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="0.5"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="0.5"), "anosim_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="1.0"), "perm_rej", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exP_verdict_power_57.csv", dict(delta="1.0"), "anosim_rej", "stat"),
    # ---- draft — D ----
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "within_minus_between", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "within_minus_between", "stat"),
    ("manuscript/latex/main_rev2.tex", "exD_tests.csv", dict(layer="presence"), "p_within_lt_between", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "shared_minus_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "shared_minus_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "p_shared_lt_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "p_shared_lt_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "concordance_Q", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "concordance_Q", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "paper_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "paper_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "n_pairs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "n_pairs", "count"),
    # ---- draft — baseline counts ----
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "n_papers", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "value_cells", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "filled_cells", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "value_records", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "censored_values", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "documented_blanks", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "family_complete_cells", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "presence_blank_data", "count"),
    ("manuscript/latex/supplementary_main.tex", "@baseline", {}, "blank_records", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "conversions", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "statuses_labeled", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "derived", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "paper_labeled", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "low_confidence_rows_in_log", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "verdict_dist.Favorable", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "verdict_dist.Harmful", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "verdict_dist.Reference standard", "count"),
    # ---- D123 audit additions (corrected Spearman semantics + scope) ----
#     ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "spearman", "stat"),
    ("docs/phase4_C1_pack.md", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "spearman", "stat"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "scope_rows", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "verdict_dist.Stable", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "verdict_dist.Mild stress", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "verdict_dist.Stressful", "count"),
    # ---- intro_discussion v0.9 (D125) ----
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0_holm", "stat"),  # D140: Holm p retained in 3.4
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "n_papers", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "low_confidence_rows_in_log", "count"),
    # ---- abstract v0.9 (D127) ----
    ("manuscript/latex/supplementary_main.tex", "exC1_summary.csv", dict(variant="mds_pres", mode="lopo"), "SI", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "SI", "stat"),
    # (row-level contrast p-values dropped from the abstract by design, P8F/D173 —
    #  the superseded row-level tests live in §3.2 only)
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0_holm", "stat"),  # D140: Holm p retained in 3.4
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "n_papers", "count"),
    # ---- D178: Table 3 / section 3.4 absolute-ratio values wired (exG) ----
    ("manuscript/tables/table3_c2.md", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "ratio_raw", "stat"),
    ("manuscript/tables/table3_c2.md", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "ratio_raw_ci_lo", "stat"),
    ("manuscript/tables/table3_c2.md", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "ratio_raw_ci_hi", "stat"),
    ("manuscript/tables/table3_c2.md", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "median_paired_ratio", "stat"),
    ("manuscript/tables/table3_c2.md", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat"),
    ("manuscript/tables/table3_c2.md", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "ratio_raw", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "median_paired_ratio", "stat"),
    # ---- P8F (D173): remaining abstract numerals wired ----
    ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="2", family="F6_aeration"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "raw_ratio", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "frac_rows_tie_aware_churn_gt0", "stat_pct100"),
#     ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "frac_rows_tie_aware_churn_gt0", "stat_pct100"),
    # P1 audit: tau-aligned F6 benchmark (replaces the 11/92 non-degenerate claim)
    ("manuscript/latex/main_rev2a.tex", "exM_benchmark_tau_summary.csv", dict(threshold="1e-06"), "n_families", "count"),
    ("manuscript/latex/main_rev2a.tex", "exM_benchmark_tau_summary.csv", dict(threshold="1e-06"), "exceedances", "count"),
    ("manuscript/latex/main_rev2a.tex", "exM_benchmark_tau_summary.csv", dict(threshold="1e-06"), "p_one_sided", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exM_benchmark_tau_summary.csv", dict(threshold="1e-09"), "n_families", "count"),
    ("manuscript/latex/main_rev2a.tex", "exM_benchmark_tau_summary.csv", dict(threshold="1e-09"), "exceedances", "count"),
    ("manuscript/latex/main_rev2a.tex", "exM_benchmark_tau_summary.csv", dict(threshold="1e-09"), "p_one_sided", "stat"),
    # P1 audit: one-draw sensitivity for the volume model
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_onedraw.csv", dict(term="block"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_onedraw.csv", dict(term="block"), "mult_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_onedraw.csv", dict(term="block"), "mult_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_onedraw.csv", dict(term="log_volume"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_onedraw.csv", dict(term="log_volume"), "mult_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_onedraw.csv", dict(term="log_volume"), "mult_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "rej_block", "stat"),
    # ---- extended power sweep (gamma 1.0/1.5/2.0; 500 reps) ----
    ("manuscript/latex/main_rev2.tex", "sim_calibration_extended_summary.csv", dict(sigma_paper="0.8", gamma="1.0"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_extended_summary.csv", dict(sigma_paper="0.8", gamma="1.5"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_extended_summary.csv", dict(sigma_paper="0.8", gamma="2.0"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_extended_summary.csv", dict(sigma_paper="0.8", gamma="1.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_extended_summary.csv", dict(sigma_paper="0.8", gamma="1.5"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_extended_summary.csv", dict(sigma_paper="0.8", gamma="2.0"), "rej_row", "stat"),
    # ---- block-preserving ordination null (PCA) ----
    ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_pca.csv", dict(variant="pca_pres", arm="observed"), "median_sdr", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_pca.csv", dict(variant="pca_pres", arm="block_null"), "median_sdr", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_pcoa.csv", dict(variant="pcoa_mds", arm="observed"), "median_sdr", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_pcoa.csv", dict(variant="pcoa_mds", arm="block_null"), "median_sdr", "stat"),
#     ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_mds20.csv", dict(variant="mds_pres", arm="observed"), "median_sdr", "stat"),
#     ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_mds20.csv", dict(variant="mds_pres", arm="block_null"), "median_sdr", "stat"),
#     ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_mds20.csv", dict(variant="mds_pres", arm="observed"), "rej_row", "stat"),
#     ("manuscript/latex/main_rev2.tex", "sim_block_null_summary_mds20.csv", dict(variant="mds_pres", arm="block_null"), "rej_row", "stat"),
    # ---- per-event paper-weighted SDR (C1) ----
    ("manuscript/latex/main_rev2a.tex", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_sdr", "stat"),
    ("manuscript/latex/main_rev2a.tex", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_p", "stat"),
    ("manuscript/latex/main_rev2a.tex", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_ci_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_ci_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "sdr_weighting.csv", dict(variant="pca_pres"), "event_paper_sdr", "stat"),
    ("manuscript/latex/main_rev2a.tex", "sdr_weighting.csv", dict(variant="mds_pres"), "event_paper_sdr", "stat"),
    ("manuscript/latex/main_rev2a.tex", "sdr_weighting.csv", dict(variant="tsne_pres"), "event_paper_sdr", "stat"),
    # 2026-11-03 v1.31: the 0.449/0.471 power figures were removed from the text because
    # results/sdr_power_value.csv has no generator in the tree (docs/sdr_power_value_provenance.md).
    # Claim retired until the calibration script is recovered.
#     ("manuscript/latex/main_rev2a.tex", "sdr_power_value.csv", dict(sigma="0.8", gamma="0.174"), "power", "stat"),
    ("manuscript/latex/supplementary_main.tex", "mds_init_weighted.csv", dict(metric="event_paper_weighted_sdr"), "random_start", "stat"),
    # ---- robustness weighting (B/C/D) ----
#     ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="B_paper_event_median", variant="mds_pres"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="B_paper_event_median", variant="pca_vstd"), "value", "stat"),
#     ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="B_paper_pooled_obs_proper", variant="mds_pres"), "value", "stat"),
#     ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="B_paper_pooled_p_proper", variant="mds_pres"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="B_paper_pooled_obs_proper", variant="pca_vstd"), "value", "stat"),
    ("manuscript/latex/supplementary_main.tex", "robustness_weighting.csv", dict(check="B_paper_pooled_p_proper", variant="pca_vstd"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="C_all_draw_median_ratio", variant="presence PCA"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="C_all_draw_ratio_ci_lo", variant="presence PCA"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="C_all_draw_ratio_ci_hi", variant="presence PCA"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="D_block_percentile_median", variant="presence PCA"), "value", "stat", r"percentile is"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="D_block_percentile_q1", variant="presence PCA"), "value", "stat", r"IQR"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="D_block_percentile_q3", variant="presence PCA"), "value", "stat", r"IQR"),
    ("manuscript/latex/main_rev2.tex", "robustness_weighting.csv", dict(check="D_block_above_q95_pct", variant="presence PCA"), "value", "stat", r"exceed the 95th percentile"),
    ("manuscript/latex/main_rev2.tex", "extraction_confidence_sensitivity.csv", {}, "rows", "count"),
    ("manuscript/latex/main_rev2.tex", "extraction_confidence_sensitivity.csv", {}, "papers", "count"),
    ("manuscript/latex/main_rev2.tex", "extraction_confidence_sensitivity.csv", {}, "sdr", "stat"),
    # ---- exhaustive family-search census (Section 2.2) ----
    ("manuscript/latex/main_rev2.tex", "family_search_summary.csv", {}, "compliant_subsets", "count"),
    ("manuscript/latex/main_rev2.tex", "family_search_summary.csv", {}, "maximal_subsets", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "pseudo_F", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_perm", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "pseudo_F", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_perm", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "anosim_R", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_anosim", "stat"),
    # 2026-10-22: the silhouette of the 13 multi-class papers was the one number in
    # §3.3 without a claim; the row is exB_multiclass.csv (57 rows, 13 papers).
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "silhouette", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_silhouette", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "pseudo_F", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_perm", "stat"),
    # ---- J diagnostics: configuration-class collapse + PC3 structure (D130) ----
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="ALL", layer="corpus"), "max_possible", "count"),
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="ALL", layer="corpus"), "n_configs", "count"),
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="F1_physical", layer="presence"), "n_configs", "count"),
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="F1_physical", layer="presence"), "n_rows", "count"),
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="F6_aeration", layer="presence"), "n_configs", "count"),
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="F6_aeration", layer="presence-pca3d"), "pc12_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="F1_physical", layer="lopo-drift-classes"), "n_configs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exJ_diagnostics.csv", dict(family="F1_physical", layer="lopo-drift-classes"), "n_rows", "count"),
    ("manuscript/latex/main_rev2.tex", "exJ_diagnostics.csv", dict(family="F1_physical", layer="lopo-drift-classes"), "max_possible", "count"),
    # ---- Volume-match paired summary (exH_volumematch_summary) ----
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "ratio_of_medians", "stat"),
    ("manuscript/tables/tableS10_volumematch.md", "exH_volumematch_summary.csv", {}, "ratio_of_medians_ci_lo", "stat"),
    ("manuscript/tables/tableS10_volumematch.md", "exH_volumematch_summary.csv", {}, "ratio_of_medians_ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio", "stat"),
    ("manuscript/tables/tableS10_volumematch.md", "exH_volumematch_summary.csv", {}, "median_paired_ratio_ci_lo", "stat"),
    ("manuscript/tables/tableS10_volumematch.md", "exH_volumematch_summary.csv", {}, "median_paired_ratio_ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "n_block_gt_scattered", "count"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "n_total", "count"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "churn_higher_block", "count", r"higher in"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "churn_lower_block", "count", r"lower in"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "churn_equal", "count", r"equal in"),
    ("manuscript/tables/tableS10_volumematch.md", "@phrase", {}, "block churn is higher in 33 papers, lower in 18, and equal in 71", "phrase"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_scattered", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio_ci_lo", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio_ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_scattered", "stat"),
    # ---- 100-suite random drift benchmark (sim_random_drift_benchmark_summary) ----
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="raw_ratio"), "exceedances", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="raw_ratio"), "p_value", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="raw_ratio"), "p_value", "stat"),
    # ---- Pattern tie structure & churn (ex_pattern_structure & ex_pattern_churn) ----
    ("manuscript/latex/main_rev2.tex", "ex_pattern_structure.csv", dict(metric="unique_patterns"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_structure.csv", dict(metric="shared_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_structure.csv", dict(metric="pct_shared_rows"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_structure.csv", dict(metric="rows_in_groups_gt_5"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_structure.csv", dict(metric="pct_rows_in_groups_gt_5"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "median_pat_churn", "stat"),
    # ---- F6 tail analysis (ex_f6_tail_summary) ----
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="raw_drift_ratio"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="n_events"), "value", "count"),
    # ---- R5-1 armour: Fig 1 paper-level co-reporting counts ----
    ("manuscript/latex/supplementary_main.tex", "fig1_coreport_pairs.csv", dict(param_a="DO", param_b="pH"), "papers_co_reporting", "count"),
    ("manuscript/latex/supplementary_main.tex", "fig1_coreport_pairs.csv", dict(param_a="pH", param_b="Temperature"), "papers_co_reporting", "count"),
    ("manuscript/latex/supplementary_main.tex", "fig1_coreport_pairs.csv", dict(param_a="DO", param_b="Temperature"), "papers_co_reporting", "count"),
    ("manuscript/latex/supplementary_main.tex", "fig1_coreport_pairs.csv", dict(param_a="pH", param_b="Salinity"), "papers_co_reporting", "count"),
    ("manuscript/latex/supplementary_main.tex", "fig1_coreport_pairs.csv", dict(param_a="DO", param_b="Salinity"), "papers_co_reporting", "count"),
    # ---- R5-3 armour: reproducible verdict-provenance strata ----
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="n_author_rows"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "verdict_agreement_summary.csv", dict(metric="n_author_rows_with_values"), "value", "count"),
    # 2026-10-26: the numeric pin for n_author_rows_without_values matched an unrelated
    # "81" (a Table 4 interval bound) while the prose never stated the value. The claim
    # is now in 3.3 p2 and pinned as an ordered phrase, which fails if the wording moves.
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "81 lack harmonized values", "phrase"),
    # 2026-10-26: the B128 dependence is a derived count (35 rows minus the 17 that
    # remain), so no CSV column carries the 18. Pinned as an ordered phrase instead.
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "supplies 18 of the 35 rows", "phrase"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="n_compared"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="agreement_pct"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa"), "value", "stat"),
    # ---- R5-2 armour: density-quartile churn medians (presence PCA LOPO) ----
    ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="pca_pres", mode="lopo"), "median_frac_Q1", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="pca_pres", mode="lopo"), "median_frac_Q2", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="pca_pres", mode="lopo"), "median_frac_Q3", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="pca_pres", mode="lopo"), "median_frac_Q4", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="pca_pres", mode="lopo"), "median_frac", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exC1_churn_corrected.csv", dict(variant="mds_pres", mode="lopo"), "median_frac", "stat"),
    # ---- R5-6/R4-2 armour: absolute ratios under the PRIMARY distance-based variant ----
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F1_clarity"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F4_organic"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F1_physical"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "fam_med", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "reportable", "count"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F6_aeration"), "reportable", "count"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F1_physical"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F3_nutrients"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F2_metals"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F1_physicala"), "raw_ratio", "stat"),
    # ---- Pattern churn now covers all 122 papers for MDS and t-SNE (R4-11) ----
#     ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "events_evaluated", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="tsne_pres", mode="lopo"), "events_evaluated", "count"),
#     ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "median_pat_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="tsne_pres", mode="lopo"), "median_pat_churn", "stat"),
    # ---- Phase F1: 1,000-replicate power sweep with Monte Carlo SEs ----
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.0", gamma="0.0"), "rej_row", "stat", r"row 0\.049"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.0", gamma="0.0"), "rej_block", "stat", r"paper 0\.052"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.2", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.5", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.5", gamma="0.0"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.0"), "rej_block", "stat", r"0\.060"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "rej_block", "stat", r"0\.065"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.6"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.6"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "contrast_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "n_rep", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "rej_block_se", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.2", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.5", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.5", gamma="0.0"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.0"), "rej_block", "stat", r"0\.060"),
    # ---- Phase F2: fitted log-linear model (Table S10 + §3.2) ----
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "coef", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "coef", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "mult_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "mult_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "n_obs", "count"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", dict(), "n_draws", "count"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", dict(), "pooled_si_lopo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", dict(), "spearman_rho_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", dict(), "spearman_rho_scattered", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "coef", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_factor", "stat"),
    # ---- Phase G1: tie-aware neighbour churn ----
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "frac_rows_tie_aware_churn_gt0", "stat_pct100"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="tsne_pres", mode="lopo"), "frac_rows_tie_aware_churn_gt0", "stat_pct100"),
#     ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "frac_rows_tie_aware_churn_gt0", "stat_pct100"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "median_tie_aware_churn", "stat"),
#     ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "median_tie_aware_churn", "stat"),
    # ---- bounded total-variation version (C6) ----
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "median_tv_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="tsne_pres", mode="lopo"), "median_tv_churn", "stat"),
#     ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "median_tv_churn", "stat"),
    # ---- 2026-10-24 tie-rule sensitivity: index churn is tie-resolution dependent,
    # the tie-aware fraction is not. artifact: results/tie_order_sensitivity.csv.
    ("manuscript/latex/main_rev2.tex", "tie_order_sensitivity.csv", dict(variant="pca_pres", tie_rule="default"), "median_idx_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "tie_order_sensitivity.csv", dict(variant="pca_pres", tie_rule="index-stable"), "median_idx_churn", "stat"),
#     ("manuscript/latex/main_rev2.tex", "tie_order_sensitivity.csv", dict(variant="mds_pres", tie_rule="index-stable"), "median_idx_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "tie_order_sensitivity.csv", dict(variant="tsne_pres", tie_rule="index-stable"), "median_idx_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "tie_order_sensitivity.csv", dict(variant="pca_pres", tie_rule="default"), "frac_tie_aware_gt0", "stat_pct100"),
#     ("manuscript/latex/main_rev2.tex", "tie_order_sensitivity.csv", dict(variant="mds_pres", tie_rule="default"), "frac_tie_aware_gt0", "stat_pct100"),
    ("manuscript/latex/main_rev2.tex", "tie_order_sensitivity.csv", dict(variant="tsne_pres", tie_rule="default"), "frac_tie_aware_gt0", "stat_pct100"),
    # 2026-10-25: LOPO stress-1 median quoted in 2.1 (fit stability under deletion)
#     ("manuscript/latex/main_rev2.tex", "mds_stress_summary.csv", dict(scope="lopo"), "stress1_med", "stat"),
#     ("manuscript/latex/main_rev2.tex", "@phrase", {}, "0.291 on the full data and 0.291 under leave-one-paper-out deletion", "phrase"),
#     ("manuscript/latex/main_rev2.tex", "@phrase", {}, "0.00, 0.20 and 0.00", "phrase"),
#     ("manuscript/latex/main_rev2.tex", "@phrase", {}, r"9.2\%, 37.4\% and 25.1\%", "phrase"),
    ("manuscript/tables/tableS4_churn.md", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "n_rows_in_multiconfig_neigh", "count"),
    ("manuscript/tables/tableS4_churn.md", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "n_rows_in_multiconfig_neigh", "count"),
    # ---- 2026-10-26: Table S4 panel C, paper-level interval for the tie-aware
    # fraction. artifact: results/ex_pattern_churn_ci.csv (code/churn_frac_ci.py).
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="pca_pres"), "median_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="tsne_pres"), "median_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="mds_pres"), "median_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="pca_pres"), "ci_lo_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="tsne_pres"), "ci_lo_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="mds_pres"), "ci_lo_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="pca_pres"), "ci_hi_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="tsne_pres"), "ci_hi_pct", "stat"),
    ("manuscript/latex/supplementary_main.tex", "ex_pattern_churn_ci.csv", dict(variant="mds_pres"), "ci_hi_pct", "stat"),
    # ---- Phase G2: density baseline (the dense-arm comparator) ----
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.2"), "med_drift", "stat", r"drift is\s+8\.49"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.4"), "med_drift", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.6"), "med_drift", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.8"), "med_drift", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.2"), "med_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.8"), "med_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.2"), "realised_density", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.8"), "realised_density", "stat"),
    # ---- Phase H2: pin the SUPPLEMENTARY TABLE NOTES to their source CSVs ----
    # S1 note: W5C3 vs F1 physical opposite-sign explanation
    ("manuscript/tables/tableS1_sensitivity.md", "exC2_sensitivity.csv", dict(sensitivity="ward_k5", family="W5C3"), "G_median", "stat"),
    ("manuscript/tables/tableS1_sensitivity.md", "exC2_summary.csv", dict(variant="pca_pres", family="F1_physical"), "G_median", "stat"),
    ("manuscript/tables/tableS1_sensitivity.md", "exC2_sensitivity.csv", dict(sensitivity="pca3d", family="F6_aeration"), "G_median", "stat"),
    # S3 note: raw-value PCA variance share + Holm range endpoints
    ("manuscript/tables/tableS3_blockperm.md", "sdr_weighting.csv", dict(variant="pca_vstd"), "event_paper_p", "stat"),
    ("manuscript/tables/tableS3_blockperm.md", "sdr_weighting.csv", dict(variant="mds_pres"), "event_paper_p", "stat"),
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(variant="pca_pres", family="F6_aeration"), "G_median", "stat"),
    # S4 note: tie structure + quartile medians
    ("manuscript/tables/tableS4_churn.md", "ex_pattern_structure.csv", dict(metric="pct_shared_rows"), "value", "stat"),
    ("manuscript/tables/tableS4_churn.md", "ex_pattern_structure.csv", dict(metric="rows_in_groups_gt_5"), "value", "count"),
    ("manuscript/tables/tableS4_churn.md", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "median_idx_churn", "stat"),
    ("manuscript/tables/tableS4_churn.md", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "frac_rows_tie_aware_churn_gt0", "stat_pct100"),
    # S5 note: the F1 physical rubric-only / author-only separation
    ("manuscript/tables/tableS5_verdict.md", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/tables/tableS12a_verdict_stratified.md", "exB_stratified.csv", dict(family="F1_physical", layer="value", stratum="author"), "pseudo_F", "stat"),
    # S7 note: F6 p moves
    ("manuscript/tables/tableS7_d91.md", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    # S8 note: F6 tail statistics quoted in the note
    ("manuscript/tables/tableS9_f6_absdrift.md", "ex_f6_tail_summary.csv", dict(metric="pooled_share_above_tau6"), "value", "stat_pct100"),
    ("manuscript/tables/tableS8_absdrift.md", "ex_f6_tail_summary.csv", dict(metric="raw_drift_ratio"), "value", "stat"),
    ("manuscript/tables/tableS8_absdrift.md", "sim_random_drift_benchmark_summary.csv", dict(metric="G"), "p_value", "stat"),
    # S9 note: correlation between configuration-weight shift and max drift
    ("manuscript/tables/tableS9_f6_absdrift.md", "ex_f6_tail_summary.csv", dict(metric="median_fam_max"), "value", "stat_sci2"),
    # S11 note: SI_fam / SI_all for F6
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(variant="pca_pres", family="F6_aeration"), "SI_family_median", "stat"),
    ("manuscript/tables/tableS11_absratio.md", "exC2_summary.csv", dict(variant="pca_pres", family="F6_aeration"), "SI_all_median", "stat"),
    # S12a note: multi-class re-run figures
    ("manuscript/tables/tableS12a_verdict_stratified.md", "exB_stratified.csv", dict(family="F1_physical", layer="value", stratum="author"), "pseudo_F", "stat"),
    # S12b note: B128 leave-one-out
    ("manuscript/tables/tableS12b_f3_f4_sensitivity.md", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "pseudo_F", "stat"),
    ("manuscript/tables/tableS12b_f3_f4_sensitivity.md", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "p_perm", "stat"),
    # ---- Phase 1 (D133): review-driven consistency fixes ----
    # R1: Table S13 now honors the tau floor rule (generator flag bug fixed);
    # pin the regenerated table to the results layer
    ("manuscript/tables/tableS13_absratio_variants.md", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "raw_ratio", "stat"),
    ("manuscript/tables/tableS13_absratio_variants.md", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F1_physical"), "raw_ratio", "stat"),
    ("manuscript/tables/tableS13_absratio_variants.md", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F6_aeration"), "fam_med", "stat_sci2"),
    # R2: MDS tie-aware median (was mis-stated as 0.00) now pinned in the S4 note
    ("manuscript/tables/tableS4_churn.md", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "median_tie_aware_churn", "stat"),
    # ---- Phase 3 (D135): tau-provenance disclosure numbers, pinned in Section 2.2 ----
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="pooled_share_above_tau6"), "value", "stat_pct100"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="median_fam_med"), "value", "stat_sci2"),
    # ---- Phase M (D158): family-expansion audit — benchmark + continuous audit
    # numbers pinned against the co-author digest (report_discuss.tex) and the
    # analysis registry. Suite-metric claims land with the md rewrite (Phase 5).
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "target_paired_ratio", "stat"),
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "n_total", "count"),
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "exceed_paired_all", "count"),
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "n_non_deg", "count"),
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "exceed_paired_nondeg", "count"),
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "p_paired_all", "stat"),
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "p_paired_nondeg", "stat"),
    ("report_discuss.tex", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "exceed_paired_nondeg", "count"),
    ("report_discuss.tex", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "n_non_deg", "count"),
    ("report_discuss.tex", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "p_paired_nondeg", "stat", r"0\.864"),
    ("report_discuss.tex", "exM_continuous_audit_summary.csv", {}, "n_complete_rows", "count"),
    ("report_discuss.tex", "exM_continuous_audit_summary.csv", {}, "n_papers", "count"),
    ("report_discuss.tex", "exM_continuous_audit_summary.csv", {}, "uniqueness_pct", "stat"),
    ("report_discuss.tex", "exM_continuous_audit_summary.csv", {}, "med_val_drift", "stat_sci2", r"4\.08"),
    ("report_discuss.tex", "exM_continuous_audit_summary.csv", {}, "ratio_to_tau", "stat"),
    ("docs/analysis_registry_familyexpansion.md", "exM_benchmark_4param_summary.csv", {}, "n_total", "count"),
    ("docs/analysis_registry_familyexpansion.md", "exM_benchmark_4param_summary.csv", {}, "exceed_paired_all", "count"),
    ("docs/analysis_registry_familyexpansion.md", "exM_benchmark_4param_summary.csv", {}, "n_non_deg", "count"),
    ("docs/analysis_registry_familyexpansion.md", "exM_benchmark_4param_summary.csv", {}, "exceed_paired_nondeg", "count"),
    ("docs/analysis_registry_familyexpansion.md", "exM_benchmark_4param_summary.csv", {}, "p_paired_all", "stat"),
    ("docs/analysis_registry_familyexpansion.md", "exM_benchmark_4param_summary.csv", {}, "p_paired_nondeg", "stat"),
    # ---- Phase M (D159): generated verification report + canonical T7 benchmark.
    ("docs/family_group_investigation_findings.md", "exM_suite_metrics.csv", dict(name="F6_aeration_pH"), "paired_ratio_2d_pca", "stat"),
    ("docs/family_group_investigation_findings.md", "exM_suite_metrics.csv", dict(name="F1_clarity"), "med_fam_2d_pca", "stat_sci2"),
    ("docs/family_group_investigation_findings.md", "exM_suite_metrics.csv", dict(name="F1_clarity_chla"), "tie_rate", "stat_pct100"),
    ("docs/family_group_investigation_findings.md", "exM_suite_metrics.csv", dict(name="F1_clarity_core"), "tie_rate", "stat_pct100"),
    ("docs/family_group_investigation_findings.md", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "exceed_paired_nondeg", "count"),
    ("docs/family_group_investigation_findings.md", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "n_non_deg", "count"),
    ("docs/family_group_investigation_findings.md", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "p_paired_nondeg", "stat"),
    ("docs/family_group_investigation_findings.md", "exM_benchmark_matched_summary.csv", dict(candidate="F1_phys_nutr"), "exceed_paired_nondeg", "count"),
    ("docs/family_group_investigation_findings.md", "exM_benchmark_4param_summary.csv", {}, "exceed_paired_all", "count"),
    ("docs/family_group_investigation_findings.md", "exM_benchmark_4param_summary.csv", {}, "p_paired_all", "stat"),
    ("docs/family_group_investigation_findings.md", "exM_continuous_audit_summary.csv", {}, "n_complete_rows", "count"),
    ("docs/family_group_investigation_findings.md", "exM_continuous_audit_summary.csv", {}, "n_papers", "count"),
    ("docs/family_group_investigation_findings.md", "exM_continuous_audit_summary.csv", {}, "uniqueness_pct", "stat"),
    ("docs/family_group_investigation_findings.md", "exM_continuous_audit_summary.csv", {}, "ratio_to_tau", "stat"),
    # ---- D203: screening-agreement interval (percentile bootstrap over the 237
    # screened records, 2,000 resamples, seed 42) pinned in all three documents
    # that quote the screening kappa.
    ("manuscript/latex/main_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "value", "stat", r"kappa = 0\.529"),
    ("manuscript/latex/main_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "ci_lo", "stat", r"CI 0\.412"),
    ("manuscript/latex/main_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "ci_hi", "stat", r"to 0\.640"),
    ("manuscript/latex/supplementary_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "ci_lo", "stat", r"CI 0\.412"),
    ("manuscript/latex/supplementary_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "ci_hi", "stat", r"to 0\.640"),
    ("manuscript/tables/tableS14_provenance.md", "screening_agreement.csv", dict(metric="operational_kappa"), "ci_lo", "stat", r"CI 0\.412"),
    ("manuscript/tables/tableS14_provenance.md", "screening_agreement.csv", dict(metric="operational_kappa"), "ci_hi", "stat", r"to 0\.640"),
    # ---- D204: main-text statistics pinned to the released layers and the
    # experiment outputs. Coverage before this block: 62% of the distinct
    # statistics in main_rev2.tex. New render kinds: stat_int (whole-number
    # rounding) and stat_dp2 (two-decimal rounding) for values the
    # manuscript prints at reduced precision.
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="total_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="data_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="padding_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="placeholder_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="context_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="parameters"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="presence_y"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="presence_ne"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="presence_blank"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="value_cells"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_labels"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="low_confidence"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="unassigned_parameters"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="configurations"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rows_sharing_pattern"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rows_in_groups_gt5"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="presence_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="data_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="multi_row_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="identical_panel_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="median_rows_per_paper"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="density_median"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="density_q1"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="density_q3"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="max_parameters_per_row"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="f6_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="f6_do_only_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="papers_in_manifest"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="papers_not_retrieved"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="papers_extracted"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="trace_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="trace_hits"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_favorable"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_stable"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_mild_stress"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_stressful"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_harmful"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_reference_standard"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "n_papers", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "scope_rows", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "value_cells", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "filled_cells", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "value_records", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "censored_values", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "documented_blanks", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "family_complete_cells", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "presence_blank_data", "count"),
    ("manuscript/latex/supplementary_main.tex", "@baseline", {}, "blank_records", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "conversions", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "qa_flags", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "statuses_labeled", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "derived", "count"),
    ("manuscript/latex/main_rev2.tex", "@baseline", {}, "paper_labeled", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "ratio_raw", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "ratio_raw_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "ratio_raw_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "median_paired_ratio", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "ratio_raw", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "ratio_raw_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "ratio_raw_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "median_paired_ratio", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physicala", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "ratio_raw", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "ratio_raw_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "ratio_raw_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "median_paired_ratio", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F2_metals", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "ratio_raw", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "ratio_raw_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "ratio_raw_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "median_paired_ratio", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "ratio_raw", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "ratio_raw_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "ratio_raw_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "median_paired_ratio", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "ratio_raw", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "ratio_raw_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "ratio_raw_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "median_paired_ratio", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_clarity", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "ratio_raw", "stat_int"),
    # P1 audit: F6 intervals are not text-bearing at the coordinate floor.
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "median_paired_ratio", "stat_int"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "n_events", "count"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "ratio_unique_cfg", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F1_clarity"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F1_physical"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F1_physicala"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F2_metals"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F3_nutrients"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F4_organic"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F1_physical"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F1_physicala"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F2_metals"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F3_nutrients"), "raw_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "paired_ratio", "stat_int"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "fam_med", "stat_sci2"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="pca_pres", dims="3", family="F6_aeration"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "d_lopo_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "d_matched_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "ratio_of_medians", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "ratio_of_medians_ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "ratio_of_medians_ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio_ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio_ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "n_block_gt_scattered", "count"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "churn_higher_block", "count", r"higher in"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "churn_equal", "count", r"equal in"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_scattered", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "pooled_si_lopo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "mult_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "mult_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_hi", "stat"),
#     ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B111", test="C1 mds_pres LOPO SI"), "value", "stat"),
#     ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B175", test="C1 mds_pres LOPO SI"), "value", "stat"),
# 2026-11-03: the duplicate-free row-weighted SDR claims (1.0243 / 1.0334) were
# retired from the main text when the density result was demoted to exploratory.
# The per-event duplicate-free values are wired to p0_sensitivity.csv below.
#     ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B175", test="C1 pca_vstd LOPO SI"), "value", "stat"),
#     ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B111", test="C1 pca_vstd LOPO SI"), "value", "stat"),
#     ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B111", test="C1 mds_pres LOPO SI"), "p_two", "stat"),
#     ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="full set", test="C1 mds_pres LOPO SI"), "p_two", "stat"),
    ("manuscript/latex/main_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B175", test="C1 pca_vstd LOPO SI"), "p_two", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vraw"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vraw"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vraw"), "p_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vraw"), "SI", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_vraw_scale.csv", dict(rank="1"), "variance", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_vraw_scale.csv", dict(rank="2"), "variance", "stat"),
    ("manuscript/latex/main_rev2.tex", "fig2_churn_quartiles.csv", dict(quartile="Q1"), "median_churn_reported", "stat"),
    ("manuscript/latex/main_rev2.tex", "fig2_churn_quartiles.csv", dict(quartile="Q2"), "median_churn_reported", "stat"),
    ("manuscript/latex/main_rev2.tex", "fig2_churn_quartiles.csv", dict(quartile="Q3"), "median_churn_reported", "stat"),
    ("manuscript/latex/main_rev2.tex", "fig2_churn_quartiles.csv", dict(quartile="Q4"), "median_churn_reported", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="pca_pres", mode="lopo"), "median_idx_churn", "stat"),
#     ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="mds_pres", mode="lopo"), "median_idx_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_pattern_churn.csv", dict(variant="tsne_pres", mode="lopo"), "median_idx_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "pseudo_F", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_perm", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "anosim_R", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_anosim", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "silhouette", "stat_dp2"),
#     ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "p_silhouette", "stat_dp2"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "n", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "papers", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_multiclass.csv", {}, "groups", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_silhouette.csv", dict(family="F1_physical", layer="value"), "sil", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_stratified.csv", dict(family="F3_nutrients", layer="value", stratum="rubric"), "n", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_stratified.csv", dict(family="F3_nutrients", layer="value", stratum="rubric"), "papers", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.2", gamma="0.0"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.5", gamma="0.0"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.0", gamma="0.0"), "rej_row_se", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.6"), "contrast_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "n_rep", "count"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.2"), "med_drift_mc_se", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.2"), "med_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.4"), "med_drift_mc_se", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.4"), "med_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.6"), "med_drift_mc_se", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.6"), "med_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.8"), "med_drift_mc_se", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.8"), "med_churn", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_radius_check.csv", dict(association="density rho vs per-row median drift"), "spearman", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_radius_check.csv", dict(association="radius vs per-row median drift"), "spearman", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_radius_check.csv", dict(association="density rho vs per-row median drift"), "n", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="total_evaluations"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="events_gt_10pct_above_tau"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="events_gt_50pct_above_tau"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="spearman_l1shift_vs_fammax"), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="pearson_l1shift_vs_fammax"), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="spearman_l1shift_vs_fammax"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_density_restricted"), "exceedances", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_density_restricted"), "total_suites", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_density_restricted"), "p_value", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_five_configuration"), "exceedances", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_five_configuration"), "total_suites", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_five_configuration"), "p_value", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_nondeg.csv", dict(metric="nondeg_exceeders_min_configs"), "exceedances", "count"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_nondeg.csv", dict(metric="nondeg_exceeders_max_configs"), "exceedances", "count"),
    # P1 audit: the 1e-12 to 1e-9 floor-level phrase was removed from the text.
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_nondeg.csv", dict(metric="degenerate_count"), "exceedances", "count"),
    # ---- D205: second ledger batch — verdict-layer splits, per-family
    # Table 2 counts, funnel counts, screening detail and derived ranges
    # quoted in the main text.
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="labelled_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="unlabelled_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="unlabelled_placeholder"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="unlabelled_blank"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="insufficient_value_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="insufficient_value_labelled"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="author_labels_data"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_compared_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="undecidable_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="intake_removed"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="fig3_b086_unlabelled_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="trace_share_pct"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physical_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physical_cover_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physical_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physical_complete_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f2_metals_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f2_metals_cover_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f2_metals_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f2_metals_complete_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f3_nutrients_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f3_nutrients_cover_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f3_nutrients_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f3_nutrients_complete_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f4_organic_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f4_organic_cover_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f4_organic_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f4_organic_complete_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physicala_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physicala_cover_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physicala_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_physicala_complete_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_clarity_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_clarity_cover_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_clarity_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f1_clarity_complete_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f6_aeration_cover_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f6_aeration_cover_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f6_aeration_complete_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="family_f6_aeration_complete_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "@funnel", {}, "raw", "count"),
    ("manuscript/latex/main_rev2.tex", "@funnel", {}, "screened", "count"),
    ("manuscript/latex/main_rev2.tex", "@funnel", {}, "retained", "count"),
    ("manuscript/latex/main_rev2.tex", "@funnel", {}, "qualified", "count"),
    ("manuscript/latex/main_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "n", "count"),
    ("manuscript/latex/main_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "n_disagree", "count"),
    ("manuscript/latex/main_rev2.tex", "screening_agreement.csv", dict(metric="operational_kappa"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="n_author_rows"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="n_compared"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="agreement_pct"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "n", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "papers", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "n", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "papers", "count"),
    ("manuscript/latex/main_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "pseudo_F", "stat_dp2"),
    ("manuscript/latex/main_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "p_perm", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "p_anosim", "stat"),
    ("manuscript/latex/main_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "silhouette", "stat_dp2"),
    ("manuscript/latex/main_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "p_silhouette", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_vstd", mode="lopo"), "spearman", "stat_dp2"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "lie between 4 and 32-fold", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "95\\% CI 247.8 to 826.7", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "95\\% CI 194.6 to 820.1", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "two lie between 0.85 and 1.16 (0.85, 0.49 to 1.65, and 1.16, 0.78 to 2.13", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "collapses to numerical residue below", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "two exceed 100-fold (134.6, 95\\% CI 80.0 to 236.2, and 214.6, 153.3 to 304.8)", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "no row reports more than 24 of the 44 parameters", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "Tiers are unchanged across $\\tau = 10^{-9}$ to $10^{-4}$", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "At $p < 0.10$, only F2 metals meets the five stability criteria", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "presence-complete base is three papers", "phrase"),
    ("manuscript/latex/supplementary_rev2.tex", "@phrase", {}, "Under the joint loosest rubric, F6 aeration also meets the criteria", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "equals 1.000 for all F1 physical (121/121) and F4 organic (53/53) events and for 87 of 88 F3 nutrient events", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "One of 100 random families exceeds F6 ($p = 0.020$), and one of 19 random families with five configurations does so ($p = 0.10$)", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "exceeded by 19 of 100 ($p = 0.198$)", "phrase"),
    # P1 audit: the 11/92 non-degenerate phrase was replaced by the tau-aligned benchmark.
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "a positive value-layer silhouette (0.487, $p = 0.002$, Tables S5 and S12b)", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "14.3-fold MDS sensitivity", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "3 outliers to -29.2 omitted", "phrase"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_min"), "exceedances", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_random_drift_benchmark_summary.csv", dict(metric="G_le_minus3_count"), "exceedances", "count"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="f6_do_only_pct"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "corpus_composition.csv", dict(metric="rubric_share_of_labelled_pct"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "screening_agreement.csv", dict(metric="agreement_pct"), "value", "stat"),
    # ---- D206: rubric-vs-author kappa interval (cluster bootstrap over papers,
    # 2,000 resamples, seed 42) pinned in every document that quotes it.
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_lo"), "value", "stat", r"CI 0\.046"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_hi"), "value", "stat", r"to 0\.209"),
    ("manuscript/latex/supplementary_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_lo"), "value", "stat", r"CI 0\.046"),
    ("manuscript/latex/supplementary_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_hi"), "value", "stat", r"to 0\.209"),
    ("manuscript/tables/tableS12a_verdict_stratified.md", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_lo"), "value", "stat", r"CI 0\.046"),
    ("manuscript/tables/tableS12a_verdict_stratified.md", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_hi"), "value", "stat", r"to 0\.209"),
    ("manuscript/tables/tableS14_provenance.md", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_lo"), "value", "stat", r"CI 0\.046"),
    ("manuscript/tables/tableS14_provenance.md", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_hi"), "value", "stat", r"to 0\.209"),
    ("report_discuss.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_lo"), "value", "stat", r"CI 0\.046"),
    ("report_discuss.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_ci_hi"), "value", "stat", r"to 0\.209"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_papers"), "value", "count"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa_n_boot"), "value", "count"),
    # ---- D207: SI Extended-analyses statistics pinned to the artifacts, plus the
    # single-class-freezing counts and the density-residualized Holm values.
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_multiclass.csv", {}, "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F1_physical", layer="value", stratum="author"), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F1_physical", layer="value", stratum="author"), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F1_physical", layer="value", stratum="author"), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F1_physical", layer="value", stratum="author"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F3_nutrients", layer="value", stratum="rubric"), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F6_aeration", layer="presence"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "p_perm_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F1_physical", layer="presence"), "p_perm_rho_residualized_holm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "p_perm_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "p_perm_rho_residualized_holm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F1_physical", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F3_nutrients", layer="value"), "R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F3_nutrients", layer="value"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F1_physical", layer="presence"), "R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F1_physical", layer="presence"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F6_aeration", layer="presence"), "R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F6_aeration", layer="presence"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_silhouette.csv", dict(family="F1_physical", layer="value"), "sil", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil_p", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "corpus_composition.csv", dict(metric="f1_value_author_rows"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "corpus_composition.csv", dict(metric="f1_value_author_papers"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "corpus_composition.csv", dict(metric="f1_value_single_class_papers"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "corpus_composition.csv", dict(metric="f1_value_single_class_rows"), "value", "count"),
    ("manuscript/tables/tableS12a_verdict_stratified.md", "corpus_composition.csv", dict(metric="f1_value_single_class_papers"), "value", "count"),
    ("manuscript/tables/tableS12a_verdict_stratified.md", "corpus_composition.csv", dict(metric="f1_value_single_class_rows"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "P_G_gt0_holm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "SI_family_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "SI_all_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "P_G_gt0_holm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exG_absratio.csv", dict(family="F3_nutrients", variant="pca_pres"), "fam_sparse_eq_dense_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(family="W5C4", sensitivity="ward_k5"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(family="W5C4", sensitivity="ward_k5"), "P_G_gt0", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(family="W5C4", sensitivity="ward_k5"), "n_pairs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(family="F6_aeration", sensitivity="pca3d"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(family="F6_aeration", sensitivity="pca3d"), "P_G_gt0", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(family="F3_nutrients", sensitivity="pca3d"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_sensitivity.csv", dict(family="F1_physical", sensitivity="pca3d"), "G_median", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment="full set", test="C2 F6_aeration paired G"), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment="drop B175", test="C2 F6_aeration paired G"), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "n_pairs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "n_pairs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "a_within_mean", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "a_within_mean", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "b_shared_cond_mean", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "b_shared_cond_mean", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "c_cross_other_mean", "stat_dp2"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "c_cross_other_mean", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "within_minus_between", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "within_minus_between", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "shared_minus_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "shared_minus_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "p_shared_lt_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "p_shared_lt_other", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "concordance_Q", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "concordance_Q", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="presence"), "paper_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_tests.csv", dict(layer="value"), "paper_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exD_bpairs_site_probe.csv", {}, "n_shared_site_tokens", "count"),
    # ---- D160: E6 arithmetic fix (S14 606/268 pairing) pinned from both documents.
    ("manuscript/tables/tableS14_provenance.md", "@baseline", {}, "paper_labeled", "count"),
    ("manuscript/tables/tableS14_provenance.md", "@baseline", {}, "statuses_labeled", "count"),
    # ---- D165: formal tier table (registry §9) pinned to the Phase-M outputs.
    ("manuscript/tables/tableM_tiers.md", "exM_suite_metrics.csv", dict(name="F6_aeration"), "paired_ratio_2d_pca", "stat"),
    ("manuscript/tables/tableM_tiers.md", "exM_suite_metrics.csv", dict(name="F1_phys_nutr"), "paired_ratio_2d_pca", "stat"),
    ("manuscript/tables/tableM_tiers.md", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "exceed_paired_nondeg", "count"),
    ("manuscript/tables/tableM_tiers.md", "exM_benchmark_matched_summary.csv", dict(candidate="F1_phys_nutr"), "exceed_paired_nondeg", "count"),
    ("manuscript/tables/tableM_tiers.md", "exM_benchmark_matched_summary.csv", dict(candidate="F2_metals"), "exceed_paired_nondeg", "count"),
    # ---- D198: ORDERED phrases. Value-only matching cannot see that a sentence
    # reordered its own numbers: swapping any two of the four drifts, or the two
    # MDS medians, left the suite green. These pin the order as written.
    ("manuscript/latex/main_rev2.tex", "@phrase", {},
     r"drift is 8.49 $\times 10^{-4}$, 7.66 $\times 10^{-4}$, 8.59 $\times 10^{-4}$ and 8.89 $\times 10^{-4}$ respectively", "phrase"),
#     ("manuscript/latex/main_rev2.tex", "@phrase", {},
#      r"$r = 0.175$ for MDS, $-0.021$ for standardized PCA", "phrase"),
    # ---- D199: unstandardized-scale diagnostics (new artifact) + funnel copies
    ("manuscript/latex/main_rev2.tex", "exC1_vraw_scale.csv", dict(rank="1"), "variance", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_vraw_scale.csv", dict(rank="2"), "variance", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_vraw_scale.csv", dict(rank="1"), "share_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_vraw_scale.csv", dict(rank="1"), "pc1_loading", "stat"),
    ("manuscript/tables/tableS3_blockperm.md", "exC1_vraw_scale.csv", dict(rank="1"), "share_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "@funnel", {}, "raw", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "@funnel", {}, "raw", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_drift_taxonomy.csv", dict(metric="events_with_all_configs"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_drift_taxonomy.csv", dict(metric="events"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_drift_taxonomy.csv", dict(metric="configs_realized"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_drift_taxonomy.csv", dict(metric="configs_possible"), "value", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "@phrase", {}, "11 distinct values in 119 of 121 deletion events", "phrase"),
    ("manuscript/latex/supplementary_rev2.tex", "@phrase", {}, "11 of 15 possible non-empty configurations", "phrase"),
    # ---- D201: coverage extension. The audit found ~108 distinct numbers in the main
    # draft unwired; these are the results-and-discussion statistics most exposed to a
    # silent transcription error (family ratios/intervals, the volume-matched and
    # log-linear layer, the SI intervals, the calibration sweep, and their restatements).
    #  Section 3.4 — family-restricted drift ratios and intervals (exG)
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "ratio_raw", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "median_paired_ratio", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "median_paired_ratio_ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F4_organic", variant="pca_pres"), "median_paired_ratio_ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "ratio_raw", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "median_paired_ratio", "stat"),
    # D201: the raw F4/F1 intervals are rendered as integers in the prose, which
    # exceeds the 0.5% stat tolerance; both renderings are pinned as one ordered phrase.
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "F4 organic (156-fold) and F1 physical (73-fold)", "phrase"),
    #  Section 3.4 — the MDS cross-family comparison (exK)
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F4_organic"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F1_physical"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F3_nutrients"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F1_physicala"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F2_metals"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "raw_ratio", "stat"),
#     ("manuscript/latex/main_rev2.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "paired_ratio", "stat"),
    #  Section 3.2 — volume-matched control and the log-linear decomposition (exH)
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "d_lopo_med", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "d_matched_med", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "ratio_of_medians", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "median_paired_ratio", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "n_block_gt_scattered", "count"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "churn_higher_block", "count", r"higher in"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_scattered", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "pooled_si_lopo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "coef", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "coef", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="log_volume"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "n_obs", "count"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_pres", mode="loro"), "med_drift", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_summary.csv", dict(variant="pca_pres", mode="loro"), "SI", "stat"),
    #  Section 3.2 — the sparsity drift ratio intervals (exC1_blockperm)
    ("manuscript/latex/supplementary_main.tex", "exC1_blockperm.csv", dict(variant="mds_pres"), "ci_lo", "stat"),
    ("manuscript/latex/supplementary_main.tex", "exC1_blockperm.csv", dict(variant="mds_pres"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vstd"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vstd"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="tsne_pres"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="tsne_pres"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vraw"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_vraw"), "ci_hi", "stat"),
    #  Section 3.5 — calibration sweep and density baseline
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.2", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.5", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.0"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.0"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.1"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.6"), "rej_row", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.6"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "contrast_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.6"), "contrast_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.4"), "realised_density", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.6"), "realised_density", "stat"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.2"), "med_drift_mc_se", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exL_density_baseline_summary.csv", dict(coverage="0.8"), "med_drift_mc_se", "stat_sci2"),
    #  Discussion / Introduction restatements of the headline numbers.
    #  The old 11/92 non-degenerate benchmark was replaced by the tau-aligned
    #  benchmark wired above; the floor-level claim is no longer text-bearing.
    ("manuscript/latex/supplementary_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_f3_lopo_sensitivity.csv", dict(omitted_paper="B128"), "silhouette", "stat"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="agreement_pct"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="agreement_pct"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "verdict_agreement_summary.csv", dict(metric="cohen_kappa"), "value", "stat"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="pooled_share_above_tau6"), "value", "stat_pct100"),
    ("manuscript/latex/main_rev2.tex", "ex_f6_tail_summary.csv", dict(metric="median_fam_med"), "value", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="pca_pres"), "G_median", "stat"),
    #  D201 batch 2 — remaining load-bearing statistics (SI verdict layer, radius/density
    #  correlations, F6 all-parameter drift, presence-PCA interval, discussion echoes)
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_pres"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_blockperm.csv", dict(variant="pca_pres"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exG_absratio.csv", dict(family="F6_aeration", variant="pca_pres"), "all_med", "stat_sci2"),
    ("manuscript/latex/supplementary_rev2.tex", "exG_absratio.csv", dict(family="F1_physical", variant="pca_pres"), "fam_med", "stat_sci2"),
    ("manuscript/latex/main_rev2.tex", "exC1_radius_check.csv", dict(association="radius vs density rho"), "spearman", "stat"),
    # 2026-10-22: the Discussion quotes the density/drift coefficient with a
    # cluster bootstrap interval over papers (results/exC1_rho_paper_ci.csv).
    ("manuscript/latex/main_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="rho_drift"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="rho_drift"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="radius_drift"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="radius_drift"), "ci_hi", "stat"),
    # 2026-10-22: the family status is categorical, so it is wired as an ordered
    # literal. The derivation is results/family_tiers.csv (code/family_tiers.py),
    # which applies the Table 5 rubric to all seven families.
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "All seven families are provisional", "phrase"),
    ("manuscript/latex/main_rev2.tex", "@phrase", {}, "assigns provisional status to all seven", "phrase"),
    # 2026-10-22: the per-family benchmark exceedances that hold F1 physical and
    # F3 nutrients at provisional are quoted in the supplement, where they are
    # wired to results/family_tiers.csv.
    ("manuscript/latex/supplementary_rev2.tex", "family_tiers.csv", dict(family="F1_physical"), "benchmark_exceed", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "family_tiers.csv", dict(family="F3_nutrients"), "benchmark_exceed", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "@phrase", {}, "returns provisional for every one", "phrase"),
    # 2026-10-22: the density/radius diagnostics now have a printed home in the
    # supplement (Extended analyses), with paper-level intervals for both coefficients.
    ("manuscript/latex/supplementary_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="radius_drift"), "ci_lo", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="radius_drift"), "ci_hi", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="rho_pc1"), "abs_value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_rho_paper_ci.csv", dict(metric="rho_pc2"), "abs_value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_radius_check.csv", dict(association="radius vs density rho"), "spearman", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_radius_check.csv", dict(association="radius vs per-row median drift"), "spearman", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_radius_check.csv", dict(association="density rho vs per-row median drift"), "spearman", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_anosim.csv", dict(family="F6_aeration", layer="value"), "R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_silhouette.csv", dict(family="F6_aeration", layer="value"), "sil", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_silhouette.csv", dict(family="F1_physical", layer="value"), "sil", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_silhouette.csv", dict(family="F3_nutrients", layer="value"), "sil", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F4_organic", layer="presence"), "pseudo_F_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_permanova.csv", dict(family="F6_aeration", layer="presence"), "p_perm_rho_residualized", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family="F1_physical", layer="presence", stratum="rubric"), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC2_summary.csv", dict(family="F6_aeration", variant="mds_pres"), "G_median", "stat"),
    ("report_discuss.tex", "exC2_summary.csv", dict(family="F1_physical", variant="pca_pres"), "G_median", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.6"), "rej_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "sim_calibration_summary.csv", dict(sigma_paper="0.8", gamma="0.2"), "contrast_pct", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_volumematch_summary.csv", {}, "spearman_rho_block", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "coef", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_factor", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_lo", "stat"),
    ("manuscript/latex/main_rev2.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_hi", "stat"),
    #  (D202) Digest claims are evaluated against the PROSE view only: a value that
    #  lives solely in an embedded table is guarded by that table's own builder, and
    #  four such claims were dropped here (two redundant D160 S14 pins, two mis-targeted).
    #  D201 batch 3 — the co-author digest's HAND-TYPED prose numbers (its embedded
    #  tables are generated, so only the note/exec-briefing statistics are at risk).
    # D201: the digest is LaTeX-dense — a bare value can be satisfied by a TikZ
    # dimension (0.35cm satisfied beta_block=0.3486). Pin the symbol context.
    ("report_discuss.tex", "exH_loglinear_model.csv", dict(term="block"), "coef", "stat",
     r"beta_\{\\mathrm\{block\}\}\s*=\s*0\.3486"),
    ("report_discuss.tex", "exH_loglinear_model.csv", dict(term="block"), "mult_factor", "stat"),
    ("report_discuss.tex", "exK_absratio_summary.csv", dict(variant="mds_pres", dims="2", family="F6_aeration"), "raw_ratio", "stat"),
    ("report_discuss.tex", "ex_f6_tail_summary.csv", dict(metric="median_fam_med"), "value", "stat_sci2"),
    ("report_discuss.tex", "ex_f6_tail_summary.csv", dict(metric="pooled_share_above_tau6"), "value", "stat_pct100"),
    ("report_discuss.tex", "ex_f6_tail_summary.csv", dict(metric="median_fam_max"), "value", "stat_sci2"),
    ("report_discuss.tex", "exM_continuous_audit_summary.csv", {}, "uniqueness_pct", "stat"),
    ("report_discuss.tex", "exM_continuous_audit_summary.csv", {}, "med_val_drift", "stat_sci2"),
    ("report_discuss.tex", "exM_benchmark_4param_summary.csv", {}, "n_non_deg", "count"),
    ("report_discuss.tex", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "n_non_deg", "count"),
    ("report_discuss.tex", "exM_benchmark_matched_summary.csv", dict(candidate="F6_aeration_pH"), "p_paired_nondeg", "stat"),
    ("report_discuss.tex", "exM_suite_metrics.csv", dict(name="F6_aeration_pH"), "paired_ratio_2d_pca", "stat"),
    # ---- D223: supplement tables wired positionally (Tables S2, S3, S5, S12a, S12b).
    # ---- Each column is mapped BY POSITION to a named results column and each row
    # ---- label to its filter key, so the mapping follows the table's structure
    # ---- rather than a value coincidence. See docs/claim_wiring.md.
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='author'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='presence', stratum='rubric'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='author'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F1_physical', layer='value', stratum='rubric'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='author'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='presence', stratum='rubric'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='author'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F6_aeration', layer='value', stratum='rubric'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='author'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='presence', stratum='rubric'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='rubric'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='author'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "pseudo_F_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "p_perm_res", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='presence', stratum='rubric'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='all'), "p_silhouette", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='author'), "pseudo_F", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='author'), "p_perm", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='author'), "anosim_R", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='author'), "p_anosim", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='author'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='author'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F3_nutrients', layer='value', stratum='author'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='rubric'), "n", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='rubric'), "papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exB_stratified.csv", dict(family='F4_organic', layer='value', stratum='rubric'), "groups", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 pca_pres LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 pca_pres LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 pca_pres LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 mds_pres LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 mds_pres LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 mds_pres LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 pca_vstd LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 pca_vstd LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C1 pca_vstd LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C2 F6_aeration paired G'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C2 F6_aeration paired G'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='full set', test='C2 F6_aeration paired G'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 pca_pres LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 pca_pres LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 pca_pres LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 mds_pres LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 mds_pres LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 mds_pres LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 pca_vstd LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 pca_vstd LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C1 pca_vstd LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C2 F6_aeration paired G'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C2 F6_aeration paired G'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B111', test='C2 F6_aeration paired G'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 pca_pres LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 pca_pres LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 pca_pres LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 mds_pres LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 mds_pres LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 mds_pres LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 pca_vstd LOPO SI'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 pca_vstd LOPO SI'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C1 pca_vstd LOPO SI'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C2 F6_aeration paired G'), "value", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C2 F6_aeration paired G'), "p_one", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "duplicate_handling.csv", dict(treatment='drop B175', test='C2 F6_aeration paired G'), "n_events", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='pca_pres', mode='lopo'), "n_obs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='pca_pres', mode='lopo'), "SI", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_pres'), "n_papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_pres'), "p_block", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_pres'), "p_block_two", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_pres'), "p_rowlevel_carried", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='pca_vraw', mode='lopo'), "n_obs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='pca_vraw', mode='lopo'), "SI", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vraw'), "n_papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vraw'), "p_block", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vraw'), "p_block_two", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vraw'), "p_rowlevel_carried", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='pca_vstd', mode='lopo'), "n_obs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='pca_vstd', mode='lopo'), "SI", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vstd'), "n_papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vstd'), "p_block", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vstd'), "p_block_two", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='pca_vstd'), "p_rowlevel_carried", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='mds_pres', mode='lopo'), "n_obs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='mds_pres', mode='lopo'), "SI", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='mds_pres'), "n_papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='mds_pres'), "p_block", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='mds_pres'), "p_block_two", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='mds_pres'), "p_rowlevel_carried", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='tsne_pres', mode='lopo'), "n_obs", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_summary.csv", dict(variant='tsne_pres', mode='lopo'), "SI", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='tsne_pres'), "n_papers", "count"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='tsne_pres'), "p_block", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='tsne_pres'), "p_block_two", "stat"),
    ("manuscript/latex/supplementary_rev2.tex", "exC1_blockperm.csv", dict(variant='tsne_pres'), "p_rowlevel_carried", "stat"),
    # P2 duplicate sensitivity: Table S2 Panel B (20-draw sensitivity)
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="none"), "block_factor", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="none"), "block_lo", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="none"), "median_index_churn", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="none"), "median_pattern_churn", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B111"), "block_factor", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B111"), "block_lo", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B111"), "median_index_churn", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B111"), "median_pattern_churn", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B175"), "block_factor", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B175"), "block_lo", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B175"), "block_hi", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B175"), "median_index_churn", "stat"),
    ("manuscript/latex/supplementary_main.tex", "dup_sensitivity.csv", dict(exclude="B175"), "median_pattern_churn", "stat"),
    # ---- P0 extraction-confidence / duplicate per-event sensitivity (2026-11-03) ----
    ("manuscript/latex/main_rev2a.tex", "p0_sensitivity.csv", dict(subset="low_confidence_excluded"), "sdr", "stat"),
    ("manuscript/latex/supplementary_main.tex", "p0_sensitivity.csv", dict(subset="low_confidence_excluded"), "p1", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p0_sensitivity.csv", dict(subset="low_confidence_excluded"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p0_sensitivity.csv", dict(subset="low_confidence_excluded"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p0_sensitivity.csv", dict(subset="low_confidence_excluded"), "median_density_retained", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p0_sensitivity.csv", dict(subset="low_confidence_excluded"), "median_density_excluded", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p0_sensitivity.csv", dict(subset="drop_B175"), "sdr", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p0_sensitivity.csv", dict(subset="drop_B111"), "sdr", "stat"),
    # ---- P1 model-fit and density-baseline slope (2026-11-03) ----
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_fit.csv", {}, "r2", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_fit.csv", {}, "adj_r2", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exH_loglinear_model_fit.csv", {}, "r2_onedraw", "stat"),
    ("manuscript/latex/main_rev2a.tex", "density_baseline_slope.csv", {}, "slope", "stat"),
    ("manuscript/latex/main_rev2a.tex", "density_baseline_slope.csv", {}, "slope_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "density_baseline_slope.csv", {}, "slope_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "density_baseline_slope.csv", {}, "p_two", "stat"),
    ("manuscript/latex/main_rev2a.tex", "density_baseline_slope.csv", {}, "r2", "stat"),
    # ---- P2 sensitivity and provenance (2026-11-03) ----
    ("manuscript/latex/main_rev2a.tex", "p2_k_sensitivity.csv", dict(k=3), "median_index_churn", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_k_sensitivity.csv", dict(k=5), "median_index_churn", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_k_sensitivity.csv", dict(k=10), "median_index_churn", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_3d_sdr.csv", dict(variant="presence_pca_3d"), "sdr", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_3d_sdr.csv", dict(variant="presence_pca_3d"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_3d_sdr.csv", dict(variant="presence_pca_3d"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_3d_sdr.csv", dict(variant="std_value_pca_3d"), "sdr", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_3d_sdr.csv", dict(variant="std_value_pca_3d"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_3d_sdr.csv", dict(variant="std_value_pca_3d"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "p2_3d_sdr.csv", dict(variant="std_value_pca_3d"), "p1", "stat"),
    # 2026-11-04 audit fix: guard the newly disclosed Stage-A and 49-paper trace
    # numbers as ordered phrases, because both are hand-typed in the main text.
    ("manuscript/latex/main_rev2a.tex", "@phrase", {}, "Stage-A presence audit (39 papers, 117 claims) found 104 correct before adjudication (88.9\\%, below the 95\\% protocol target), and 13 over-marked cells were corrected", "phrase"),
    ("manuscript/latex/main_rev2a.tex", "@phrase", {}, "A stratified 100-cell value trace across 49 of 122 papers (40\\% of papers, 3.8\\% of the 2,658 numeric cells)", "phrase"),
    ("manuscript/latex/main_rev2a.tex", "ex_pattern_churn_ci.csv", dict(variant="pca_pres"), "median_pct", "stat"),
    ("manuscript/latex/main_rev2a.tex", "ex_pattern_churn_ci.csv", dict(variant="pca_pres"), "ci_lo_pct", "stat"),
    ("manuscript/latex/main_rev2a.tex", "ex_pattern_churn_ci.csv", dict(variant="pca_pres"), "ci_hi_pct", "stat"),
    ("manuscript/latex/main_rev2a.tex", "ex_pattern_churn_ci.csv", dict(variant="tsne_pres"), "median_pct", "stat"),
    ("manuscript/latex/main_rev2a.tex", "ex_pattern_churn_ci.csv", dict(variant="tsne_pres"), "ci_lo_pct", "stat"),
    ("manuscript/latex/main_rev2a.tex", "ex_pattern_churn_ci.csv", dict(variant="tsne_pres"), "ci_hi_pct", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exC2_sensitivity.csv", dict(sensitivity="ward_k5", family="W5C4"), "G_median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "exC2_sensitivity.csv", dict(sensitivity="ward_k5", family="W5C4"), "n_pairs", "count"),
    ("manuscript/latex/main_rev2a.tex", "nonretrieved_comparison.csv", dict(group="not_retrieved"), "median_year", "stat"),
    ("manuscript/latex/main_rev2a.tex", "nonretrieved_comparison.csv", dict(group="retrieved"), "median_year", "stat"),
    ("manuscript/latex/main_rev2a.tex", "nonretrieved_comparison.csv", dict(group="not_retrieved"), "doi_count", "count"),
    # ---- v1.40: interaction, decision consequences, structural factorial ----
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="log_volume_x_block"), "coef", "stat"),
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="log_volume_x_block"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="log_volume_x_block"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="block_no_interaction_factor"), "coef", "stat"),
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="block_no_interaction_factor"), "ci_lo", "stat"),
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="block_no_interaction_factor"), "ci_hi", "stat"),
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="block_factor_median5"), "coef", "stat"),
    ("manuscript/latex/main_rev2a.tex", "block_model_diagnostics.csv", dict(term="block_factor_q95"), "coef", "stat"),
    ("manuscript/latex/main_rev2a.tex", "decision_consequences_summary.csv", dict(variant="presence PCA", metric="top10"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "decision_consequences_summary.csv", dict(variant="std-value PCA", metric="top10"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "decision_consequences_summary.csv", dict(variant="presence PCA", metric="ari"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "decision_consequences_summary.csv", dict(variant="std-value PCA", metric="ari"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "decision_consequences_summary.csv", dict(variant="presence PCA", metric="ari_below_0.80"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "decision_consequences_summary.csv", dict(variant="std-value PCA", metric="ari_below_0.80"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="overall", level="all"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="overall", level="all"), "q1", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="overall", level="all"), "q3", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="overall", level="all"), "pct_gt1", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="panel", level="repeated"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="panel", level="repeated"), "q1", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="panel", level="repeated"), "pct_gt1", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="panel", level="varying"), "median", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="panel", level="varying"), "q1", "stat"),
    ("manuscript/latex/main_rev2a.tex", "structural_generality_summary.csv", dict(factor="panel", level="varying"), "pct_gt1", "stat"),
]

TOKEN_RE = re.compile(r"(?<![\d.])-?\d+\.\d+|(?<![\d.])-\.\d+|(?<![\d.])\.\d+|(?<![\d.])-?\d+(?![\d]|\.\d)")
# scientific notation as written in this repo's documents (D197):
#   8.49 $\times$ 10$^{-4}$   ·   $6.09\times 10^{4}$   ·   2.85 × 10⁻⁶
#   1.74e-04                  ·   4.9E-09
SCI_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:\$?\\times\$?|\u00d7|\bx\b)\s*10\s*\$?\^?\{?\s*(-?\d+)\s*\}?\$?")
SCI_E_RE = re.compile(r"(?<![\d.])(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![\d.])")


def norm(text: str) -> str:
    t = text.replace("\u2212", "-").replace("\u2013", "-")
    t = t.replace("{,}", ",")  # LaTeX brace-protected thousands (latex pipeline: $78{,}900$)
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)  # 279,378 -> 279378


def load_csv(name: str) -> list[dict]:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lookup(rows: list[dict], filt: dict, col: str) -> float:
    hits = [r for r in rows if all(r.get(k) == str(v) for k, v in filt.items())]
    if len(hits) != 1:
        raise AssertionError(f"filter {filt} matched {len(hits)} rows (want 1)")
    raw = hits[0][col]
    if raw == "" or raw.lower() == "nan":
        raise AssertionError(f"empty/nan in {col} for {filt}")
    return float(raw)


# --- D202: LaTeX typography tokens must never satisfy a numeric claim -----------
# The digest is LaTeX-dense: a bare value could be matched by a TikZ dimension
# (0.35cm satisfied beta_block = 0.3486) or a table-column width. Dimensions are
# typography, not statistics, so they are dropped at tokenisation.
_UNIT_AFTER = re.compile(r"\s*(?:cm|mm|pt|em|ex|in\b|bp|pc|\\linewidth|\\textwidth|\\columnwidth|\\textheight)")
_GEOM_BEFORE = re.compile(r"(?:width=|height=|scale=|text width=|inner sep=|outer sep=|line width=|"
                          r"left=|right=|above=|below=|node distance=|xshift=|yshift=|"
                          r"\\tabcolsep|\\arraystretch|\\setlength\{|\\includegraphics\[)")


def _is_typography(text: str, a: int, b: int) -> bool:
    if _UNIT_AFTER.match(text[b:b + 14]):
        return True
    if _GEOM_BEFORE.search(text[max(0, a - 30):a]):
        return True
    # fraction geometry like "0.92\linewidth" or "[width=0.82\linewidth]"
    return bool(re.match(r"\s*\\?(?:line|text|column)width", text[b:b + 16]))


def scan(text: str) -> list:
    """(value, start, end) for every number in the text, spans retained (D198).

    Spans let a claim carry an optional context requirement (`near`, see has_value):
    value-only matching cannot see a sentence reorder its own numbers, so a claim on
    "sparse and dense medians 0.0146 and 0.0143" was satisfied after the two were
    swapped. Claims whose ordering carries meaning now pin it.
    """
    text = norm(text)
    sci, spans = [], []
    for rx in (SCI_RE, SCI_E_RE):
        for m in rx.finditer(text):
            spans.append((float(m.group(1)) * (10.0 ** int(m.group(2))), m.start(), m.end()))
            sci.append((m.start(), m.end()))
    for m in TOKEN_RE.finditer(text):
        if any(a <= m.start() < b for a, b in sci):
            continue  # mantissa/exponent fragment of a scientific-notation number
        if _is_typography(text, m.start(), m.end()):
            continue  # LaTeX dimension (cm/pt/\linewidth, geometry options) — not a statistic
        spans.append((float(m.group(0)), m.start(), m.end()))
    return spans


def tokens_of(text: str) -> list:
    """Values only (convenience wrapper around scan; D197 semantics unchanged).

    D197: '4.9 $\\times$ 10$^{-9}$' is ONE number (4.9e-9), not the fragments
    4.9 / 10 / -9. Resolving the pairs also stops mantissas and exponents from
    posing as standalone tokens — previously the token "0.0" alone could satisfy
    a claim on 4.9e-9.
    """
    return [v for v, _, _ in scan(text)]


SPAN = 220  # characters of context searched either side of a matched number (D198)


def _strip_generated_blocks(text: str) -> str:
    """Remove TikZ / graphics scaffolding from a LaTeX document (D202, D210).

    Hand-written result tables are manuscript content and now carry claims
    (e.g. the Section 3.2 summary table), so tabular bodies stay in view.
    TikZ and includegraphics geometry still carry no claims and are removed.
    """
    # D210: hand-written result tables now carry claims too; keep tabular bodies in view.
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", text, flags=re.S)
    text = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", " ", text)
    text = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", " ", text)
    # drop the LaTeX preamble / package scaffolding: it carries no claims
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    return text


def _doc_text(doc_cache: dict, doc: str) -> str:
    key = ("#text", doc)
    if key not in doc_cache:
        raw = norm(open(os.path.join(ROOT, doc), encoding="utf-8").read())
        doc_cache[key] = _strip_generated_blocks(raw) if doc.endswith(".tex") else raw
    return doc_cache[key]


def doc_tokens(doc_cache: dict, doc: str) -> list:
    """(value, start, end) triples for the document (spans kept for near=, D198)."""
    key = ("#tok", doc)
    if key not in doc_cache:
        doc_cache[key] = scan(_doc_text(doc_cache, doc))
    return doc_cache[key]


def has_value(toks: list, v: float, kind: str, text: str = None, near: str = None) -> bool:
    if kind == "count":
        cand = [c for c in toks if c[0] == float(int(round(v)))]
    elif kind == "stat_int":
        # the document prints the value rounded to a whole number
        # (e.g. 62.53 -> 63); 0.5-wide band around round(v)
        cand = [c for c in toks if abs(round(v) - c[0]) <= 0.5]
    elif kind == "stat_dp2":
        # the document prints the value rounded to two decimals
        # (e.g. -0.0344 -> -0.03); 0.005-wide band around round(v, 2)
        cand = [c for c in toks if abs(round(v, 2) - c[0]) <= 0.005]
    elif kind == "stat_pct100":
        # the results layer stores a proportion; the document quotes a percentage
        cand = [c for c in toks if abs(c[0] - v * 100.0) <= 0.06]
    else:
        # Relative tolerance; the flat 0.001 floor applies only to |v| >= 0.1.
        # D197: the old rule used tol = 0.001 whenever |v| < 0.1, i.e. a tolerance
        # WIDER than the value it guarded, so every sub-0.001 claim (the whole drift
        # layer: Table 3, Table S8/S9/S13, the density baseline) was satisfied by an
        # unrelated nearby token such as 0.0 and passed vacuously.
        # 3% covers a value quoted at 1 s.f. (this manuscript writes 0.0068 as
        # "0.007", 2.9%) and stays far tighter than the values it guards: for a drift
        # of 4.9e-9 the band is 1.5e-10.
        # 'stat_sci2' is kept as an alias: same rule, since the results layer stores a
        # full float while the document quotes 2 s.f. in scientific notation.
        if abs(v) < 0.1:
            tol = max(abs(v) * 0.03, 1e-13)
        else:
            tol = max(0.001, abs(v) * 0.005)
        cand = [c for c in toks if abs(c[0] - v) <= tol]
    if not cand:
        return False
    if near is None:
        return True
    # D198: ordering matters in some sentences (e.g. "sparse and dense medians
    # 0.0146 and 0.0143"). A claim may require a regex within +/-SPAN characters of
    # the matched number; value-only matching accepted the swapped sentence.
    assert text is not None, "near= requires the document text"
    for t, a, b in cand:
        lo, hi = max(0, a - SPAN), min(len(text), b + SPAN)
        if re.search(near, text[lo:hi]):
            return True
    return False


def text_has(doc_cache: dict, doc: str, v: float, kind: str, near: str = None) -> bool:
    return has_value(doc_tokens(doc_cache, doc), v, kind,
                     text=_doc_text(doc_cache, doc), near=near)


# Known-answer fixtures for the matcher above (run: python3 code/check_numbers.py
# --selftest). Each case pins one branch; the D197 cases are the regression guard
# for the tolerance repair — every one of them PASSED vacuously before it.
SELFTEST = [
    # (document text, value, kind, expected)
    ("the corpus holds 122 papers", 122, "count", True),
    ("the corpus holds 123 papers", 122, "count", False),
    ("raw ratio 60900 (9151, 102939)", 60900.28, "stat", True),
    ("raw ratio 60000", 60900.28, "stat", False),
    ("median drift 4.9 $\\times$ 10$^{-9}$", 4.903020993414152e-09, "stat", True),
    ("median drift 4.90 $\\times$ 10$^{-9}$", 4.903020993414152e-09, "stat", True),
    ("median drift 0.0", 4.903020993414152e-09, "stat", False),
    ("median drift 0.001", 4.903020993414152e-09, "stat", False),
    ("drift 8.49 $\\times$ 10$^{-4}$", 0.0008486896894588344, "stat", True),
    ("drift 0.0", 0.0008486896894588344, "stat", False),
    ("max drift 2.85 $\\times$ 10$^{-6}$", 2.8530455269395016e-06, "stat_sci2", True),
    ("$6.09\\times 10^{4}$-fold", 60900.28, "stat", True),
    ("tie rate 1.74e-04", 1.7407811116025367e-04, "stat", True),
    ("rank correlation 0.007", 0.0068, "stat", True),      # 1 s.f. quote
    ("rank correlation 0.009", 0.0068, "stat", False),     # 32% off
    ("Holm .007", 0.007, "stat", True),                    # lead-in dot form
    ("Holm 0.007", 0.007, "stat", True),
    ("Holm 0.02", 0.007, "stat", False),
    ("17.8% of evaluations", 0.17812087942573504, "stat_pct100", True),
    ("18.0% of evaluations", 0.17812087942573504, "stat_pct100", False),
    ("paired interval 63", 62.53, "stat_int", True),
    ("paired interval 62", 62.53, "stat_int", False),
    ("silhouette -0.03", -0.0344, "stat_dp2", True),
    ("silhouette -0.02", -0.0344, "stat_dp2", False),
    # D228: a paragraph-final integer used to be invisible because the integer
    # branch excluded a following full stop (the same guard that stops "3.2"
    # yielding 3). Both directions are pinned here.
    ("churn is higher in 15 and equal in 107.", 107, "count", True),
    ("the ratio is 3.2 in total", 3, "count", False),
]

# D198 fixtures: context requirement and ordered phrase (see has_value / main())
SELFTEST_NEAR = [
    # (text, value, kind, near, expected)
    ("medians 0.0146 and 0.0143", 0.01463, "stat", r"medians\s+0\.0146\s+and\s+0\.0143", True),
    ("medians 0.0143 and 0.0146", 0.01463, "stat", r"medians\s+0\.0146\s+and\s+0\.0143", False),
    ("p = 0.043 and p = 0.044", 0.043, "stat", r"0\.043", True),
    ("p = 0.044 only", 0.043, "stat", r"0\.043", False),
]
SELFTEST_PHRASES = [
    ("drift is 8.49 $\\times$ 10$^{-4}$, 8.59", "drift is 8.49 $\\times$ 10$^{-4}$, 8.59", True),
    ("drift is 8.59 $\\times$ 10$^{-4}$, 8.49", "drift is 8.49 $\\times$ 10$^{-4}$, 8.59", False),
]


def selftest() -> int:
    bad = n = 0
    for text, v, kind, want in SELFTEST:
        n += 1
        if has_value(scan(text), v, kind) != want:
            bad += 1
            print(f"FAIL  selftest  kind={kind} v={v!r} text={text!r} want={want}")
    for text, v, kind, near, want in SELFTEST_NEAR:
        n += 1
        if has_value(scan(text), v, kind, text=norm(text), near=near) != want:
            bad += 1
            print(f"FAIL  near      v={v!r} near={near!r} text={text!r} want={want}")
    for text, phrase, want in SELFTEST_PHRASES:
        n += 1
        if (norm(phrase) in norm(text)) != want:
            bad += 1
            print(f"FAIL  phrase    {phrase!r} in {text!r} want={want}")
    print(f"[check-numbers --selftest] {n - bad}/{n} matcher fixtures pass")
    return 1 if bad else 0


FUNNEL = {"raw": 11612, "dedup_rm": 2681, "after_dedup": 8931, "filter_rm": 2838,
          "after_filter": 6093, "frame": 375, "screened": 237, "include": 134, "unsure": 14,
          "retained": 148, "doi_confirmed": 117, "no_doi": 25, "offtopic": 8,
          "predatory": 5, "qualified": 135, "pdf_at_close": 119}
FUNNEL_LITERALS = {"raw": "11,612", "dedup_rm": "2,681", "after_dedup": "8,931",
                   "filter_rm": "2,838", "after_filter": "6,093", "screened": "237",
                   "retained": "148", "qualified": "135", "pdf_at_close": "119"}


def funnel_get(base_cache: dict, col: str) -> float:
    """PRISMA funnel counts, taken from manuscript/corpus_funnel.md with the same
    arithmetic integrity checks the S14 / FigS1 builders enforce (D199). The draft
    and the caption quote these numbers; this is their artifact."""
    assert FUNNEL["raw"] - FUNNEL["dedup_rm"] == FUNNEL["after_dedup"], "funnel leg 1"
    assert FUNNEL["after_dedup"] - FUNNEL["filter_rm"] == FUNNEL["after_filter"], "funnel leg 2"
    assert FUNNEL["include"] + FUNNEL["unsure"] == FUNNEL["retained"], "funnel leg 3"
    assert FUNNEL["retained"] - FUNNEL["offtopic"] - FUNNEL["predatory"] == FUNNEL["qualified"], "funnel leg 4"
    assert FUNNEL["frame"] - 138 == FUNNEL["screened"], "funnel leg 5 (pilot consumption)"
    if "#funnel" not in base_cache:
        text = open(os.path.join(ROOT, "manuscript", "corpus_funnel.md"), encoding="utf-8").read()
        for k, lit in FUNNEL_LITERALS.items():
            assert lit in text, f"funnel literal {lit} ({k}) missing from corpus_funnel.md"
        base_cache["#funnel"] = True
    return float(FUNNEL[col])


def baseline_get(base: dict, col: str) -> float:
    """Support dotted paths into the baseline dict, e.g. 'verdict_dist.Favorable'."""
    node = base["baselines"]
    for part in col.split("."):
        node = node[part]
    return float(node)


COVERAGE_DOCS = ["manuscript/latex/main_rev2.tex", "manuscript/latex/supplementary_rev2.tex",
                 "manuscript/latex/main_rev2.tex", "report_discuss.tex"]
# structural constants that are not statistics (thresholds, dimensions, percentages)
_COV_SKIP = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
             16.0, 17.0, 18.0, 19.0, 20.0, 24.0, 44.0, 50.0, 95.0, 99.0, 100.0, 199.0, 999.0, 1000.0}


def report_coverage(claims=None, main=None):
    """Distinct statistic-shaped numbers per draft vs how many are wired as claims (D201).

    Makes the gate's meaning explicit: `check_numbers` verifies WIRED claims, so the
    headline count is a coverage-relative measure, not a statement that every number
    in the manuscript is machine-checked. Run with --coverage.
    """
    claims = claims or CLAIMS
    base = json.load(open(os.path.join(RESULTS, "phase4_baseline.json"), encoding="utf-8"))
    wired: dict = {}
    for claim in claims:
        doc, csv_name, filt, col, kind = claim[:5]
        if main and doc == "manuscript/latex/main_rev2.tex":
            doc = main
        if csv_name in ("@phrase",):
            continue
        try:
            v = baseline_get(base, col) if csv_name == "@baseline" else (
                funnel_get({}, col) if csv_name == "@funnel" else lookup(load_csv(csv_name), filt, col))
        except Exception:
            continue
        wired.setdefault(doc, []).append((kind, v))

    out = []
    docs = [main if (main and d == "manuscript/latex/main_rev2.tex") else d for d in COVERAGE_DOCS]
    for d in docs:
        if not os.path.exists(os.path.join(ROOT, d)):
            continue
        vals = set()
        for line in _doc_text({}, d).splitlines():
            if line.lstrip().startswith(("#", ">")):
                continue
            for v, _, _ in scan(line):
                if v in _COV_SKIP or (v == int(v) and 1900 <= v <= 2030) or abs(v) >= 1000:
                    continue
                if v != int(v) or abs(v) >= 20:
                    vals.add(round(v, 10))
        # respect each claim's KIND when deciding whether a document number is covered
        n_wired = 0
        for v in vals:
            hit = False
            for kind, cv in wired.get(d, []):
                target = cv * 100.0 if kind == "stat_pct100" else cv
                band = 0.06 if kind == "stat_pct100" else (
                    max(abs(target) * 0.05, 1e-13) if abs(target) < 0.1 else max(0.001, abs(target) * 0.005))
                if abs(target - v) <= band:
                    hit = True
                    break
            n_wired += 1 if hit else 0
        out.append((os.path.basename(d), n_wired, len(vals)))
    return out


def _claim_key(csv_name: str, filt: dict, col: str) -> tuple:
    return (csv_name, col, tuple(sorted(filt.items())))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--main", default="manuscript/latex/main_rev2a.tex")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    args, _ = ap.parse_known_args()
    if args.selftest:
        return selftest()
    verbose = args.verbose
    triage_moved: set = set()
    triage_dropped: set = set()
    triage_added: list = []
    if args.main:
        triage_path = os.path.join(ROOT, "code", "rev2a_triage.json")
        if os.path.exists(triage_path):
            triage = json.load(open(triage_path, encoding="utf-8"))
            triage_moved = {_claim_key(c[0], c[2], c[1]) for c in triage["moved_to_si"]}
            triage_dropped = {_claim_key(c[0], c[2], c[1]) for c in triage["dropped"]}
            triage_added = [tuple(c) for c in triage.get("added_rev2a", [])]
    base = json.load(open(os.path.join(RESULTS, "phase4_baseline.json"), encoding="utf-8"))
    doc_cache: dict = {}
    fails, ok, skipped = [], 0, 0
    triage_skipped = 0
    triage_retargeted = 0
    missing_docs = set()
    claims = list(CLAIMS) + triage_added
    for claim in claims:
        doc, csv_name, filt, col, kind = claim[:5]
        if args.main and doc == "manuscript/latex/main_rev2.tex":
            ck = _claim_key(csv_name, filt, col)
            if ck in triage_dropped:
                triage_skipped += 1
                continue
            doc = args.main
            if ck in triage_moved:
                doc = "manuscript/latex/supplementary_main.tex"
                triage_retargeted += 1
        if not doc.startswith("@") and not os.path.exists(os.path.join(ROOT, doc)):
            # the target document is not part of this tree (e.g. the co-author
            # digest in the scoped code mirror): its claims are skipped, not failed
            skipped += 1
            missing_docs.add(doc)
            continue
        near = claim[5] if len(claim) > 5 else None
        if csv_name == "@phrase":
            # D198: an ORDERED literal must appear (value-only matching cannot see
            # that a sentence reordered its own numbers, e.g. the MDS sparse/dense
            # medians or the density-baseline drift list).
            if norm(col) in _doc_text(doc_cache, doc):
                ok += 1
            else:
                fails.append(f"FAIL  {doc}  phrase not found: {col[:70]!r}")
            continue
        if csv_name == "@baseline":
            v = baseline_get(base, col)
        elif csv_name == "@funnel":
            v = funnel_get(doc_cache, col)
        else:
            v = lookup(load_csv(csv_name), filt, col)
        if text_has(doc_cache, doc, v, kind, near=near):
            ok += 1
            if verbose:
                print(f"PASS  {doc}  {col}{filt} = {v}")
        else:
            fails.append(f"FAIL  {doc}  {col}{filt} = {v}  (value not found in doc)")
    for f in fails:
        print(f)
    n_docs = len({c[0] for c in claims})
    n_phrase = sum(1 for c in claims if c[1] == "@phrase")
    n_funnel = sum(1 for c in claims if c[1] == "@funnel")
    n_total = len(claims) - skipped - triage_skipped
    print(f"[check-numbers] {ok}/{n_total} claims verified across {n_docs} docs "
          f"(numeric {n_total - n_phrase - n_funnel} · ordered-phrase {n_phrase} · funnel {n_funnel})"
          + ("" if not fails else f" — {len(fails)} FAILURES")
          + (f" · rev2a triage: {triage_retargeted} retargeted to SI, {triage_skipped} dropped"
             if args.main else "")
          + (f" · {skipped} claims skipped (documents not in this tree: "
             + ", ".join(sorted(missing_docs)) + ")" if skipped else ""))
    if args.coverage:
        coverage = report_coverage(claims, args.main)
        for name, wired_n, total in coverage:
            print(f"  [coverage] {name:38s} {wired_n:4d}/{total:<4d} distinct statistics wired "
                  f"({100.0 * wired_n / total:.0f}%)" if total else "")
        print("  [coverage] note: the gate above checks WIRED claims; a number that is not "
              "wired is not checked for drift.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
