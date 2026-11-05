# P1 Family-PCA — reproducible build (Analysis_Design §10; D117 restructure)
#
#   make all         FAST reproducibility check (~1 min): everything rebuilt from
#                    frozen stat layers — baseline gate -> selftest -> all figures
#                    (decoupled builders) -> tables -> numbers check
#   make recompute   FULL stat re-run (~40 min): 4 experiment scripts + fig4
#                    (re-writes results/*.csv; must stay byte-identical) + all figures
#   make check       G3 numbers-consistency check only (read-only, seconds)
#   make paper       stub until P1-6.2 (Methods/Results lock)
# FIG_DPI=180 gives the legacy quick figures; default 600 = submission quality.

PY ?= python3
FIG_DPI ?= 600

.PHONY: all baseline selftest composition stage-layers recompute reproduce figures fig4 fig2-disk fig3 fig5 figD \
        figs-built tables check paper corrections blockperm churn sim d91 tableS absdrift verdict-replay vraw-scale f1-taxonomy \
        absratio volumematch configmatch diagnostics \
        fig2-print fig3-print fig5-print si-panels fig7-replay figS13 figS14 duplicates radius \
        churn-pattern f6-tail f6-completeness sim-benchmark absratio-variants tableS13 tableS14 \
        block-null power-extension robustness-weighting extraction-sensitivity family-search verdict-power family-tier-sensitivity \
        provenance-agreement density-baseline familyexpansion familyinvestigation tableM wordcount seed-baseline mds-classical-check mds-scope-audit r1-anchor r2-calibration sdr-paper-weighted sdr-hierarchical paper-stats decision-consequences structural-generality block-diagnostics

# NOTE: `all` rebuilds submission artifacts from frozen layers: baseline gate ->
# selftest -> all figures -> supplementary tables -> numbers check.
all: stage-layers baseline selftest family-tiers family-tier-sensitivity fig2-print fig3-print fig5-print fig7-replay figD figS13 figS14 fig4 figs-built si-panels tables results/exC1_vraw_scale.csv f1-taxonomy tableS13 verdict-replay tableS14 tableS provenance-agreement results/exL_density_baseline_summary.csv composition check style dups wordcount
	@echo "[make] ALL GREEN — frozen layers -> submission figures -> tables -> numbers check complete"

# NOTE: `family-tiers` is NOT in this target. It reads exK_absratio_summary.csv
# (absratio-variants) and exM_benchmark_matched_*.csv (familyexpansion-matched),
# which are produced after this target runs; it is placed in `reproduce` after
# `familyexpansion` and in `all` for the fast path from committed outputs.
corrections: blockperm churn d91 sim absdrift absratio volumematch configmatch diagnostics duplicates radius rho-bootstrap vraw-scale churn-pattern tie-order f6-tail f6-completeness tableS sim-benchmark fig2-print fig3-print fig5-print si-panels fig7-replay
	@echo "[make] corrections pass complete (blockperm, churn, d91, simulation, absdrift, volume-matched, absolute-ratio, configuration-matched, benchmarks, SI tables, print figures)"

.PHONY: mdsstress
mdsstress:
	$(PY) code/mds_stress.py

# ---- Phase 1 (D253): estimand-defined seed baseline. Standalone: ~80-90 min
# ---- for seeds 0-9, deliberately not wired into all/reproduce.
seed-baseline:
	$(PY) code/mds_seed_baseline.py --seeds 0-9 --protocol all

# ---- Phase 2: deterministic classical-MDS gate (read-only, ~8 s).
mds-classical-check:
	$(PY) code/mds_classical_check.py

# ---- D260: classified MDS scope audit against the random-start archive.
mds-scope-audit:
	$(PY) code/audit_mds_scope.py

blockperm:
	$(PY) code/stability_blockperm.py

churn:
	$(PY) code/churn_corrected.py

d91:
	$(PY) code/verdict_d91_sensitivity.py

sim:
	$(PY) code/experiment_S_validation.py

absdrift:
	$(PY) code/experiment_F_absdrift.py

absratio:
	$(PY) code/experiment_G_absratio.py

volumematch:
	$(PY) code/experiment_H_volumematch.py

configmatch:
	$(PY) code/experiment_I_configmatch.py

diagnostics:
	$(PY) code/experiment_J_diagnostics.py

# ---- Crossref registry check for references.bib. NOT part of `all`/`reproduce`:
# ---- it needs network access (the Crossref REST API), which the rest of the
# ---- pipeline does not. Run it by hand when a network is available.
verify-refs:
	$(PY) code/verify_references.py

# ---- `tableS` reads five generated inputs: the vraw-scale diagnostics, the
# ---- churn-pattern and F6-tail summaries, the stratified B analysis and the
# ---- author-multi-class check. The last two producers ship in the archive but
# ---- had no target of their own — their CSVs were committed, so a fresh unpack
# ---- could not rebuild them.
# ---- These are FILE targets, not phony ones: a producer runs only when its
# ---- output is missing or older than its script. A fresh unpack has no
# ---- results/, so the producers run there, while in the project tree the
# ---- committed CSVs are current and the fast `all` path stays fast.
# ---- `make reproduce` forces the full rebuild via the phony targets below.
results/exC1_vraw_scale.csv: code/make_vraw_scale_diagnostics.py
	$(PY) code/make_vraw_scale_diagnostics.py

results/ex_pattern_churn.csv: code/pattern_churn.py
	$(PY) code/pattern_churn.py

results/ex_f6_tail_summary.csv: code/f6_tail_analysis.py
	$(PY) code/f6_tail_analysis.py

results/exB_stratified.csv: code/experiment_B_stratified.py
	$(PY) code/experiment_B_stratified.py

results/exB_multiclass.csv: code/verdict_multiclass.py
	$(PY) code/verdict_multiclass.py

tableS: results/exC1_vraw_scale.csv results/ex_pattern_churn.csv results/ex_f6_tail_summary.csv results/exB_stratified.csv results/exB_multiclass.csv
	$(PY) code/make_tableS_corrections.py

# force-rebuild entry points for the same producers
b-stratified:
	$(PY) code/experiment_B_stratified.py

verdict-multiclass:
	$(PY) code/verdict_multiclass.py

churn-pattern:
	$(PY) code/pattern_churn.py

tie-order:
	$(PY) code/tie_order_sensitivity.py

f6-tail:
	$(PY) code/f6_tail_analysis.py

f6-completeness:
	$(PY) code/f6_completeness_sensitivity.py

# D252: validation artifacts cited in Section 3.5, Table S15 and Section 3.2.
# Tags must match the filenames checked by code/check_numbers.py. The block null
# is expensive; these targets sit in `reproduce`, not in the fast `all` path.
block-null:
	$(PY) code/experiment_S_block_null.py --variant pca_pres --reps 100 --deletes 30 --tag pca
	$(PY) code/experiment_S_block_null.py --variant pcoa_mds --reps 100 --deletes 30 --tag pcoa
	$(PY) code/experiment_S_block_null.py --variant mds_pres --reps 20 --deletes 10 --tag mds20

power-extension:
	$(PY) code/experiment_S_power_extension.py

robustness-weighting:
	$(PY) code/robustness_weighting.py

extraction-sensitivity:
	$(PY) code/extraction_confidence_sensitivity.py

family-search:
	$(PY) code/family_search_census.py

sim-benchmark:
	$(PY) code/sim_random_drift_benchmark.py

# ---- Phase C1/C2: absolute drift ratios under BOTH primary variants, 2-D and 3-D.
# The canonical layer is the HEADLINE set: --mds-2d-only (presence PCA 2-D/3-D +
# Jaccard MDS 2-D). Running the full 4-variant set (default below, MDS-bound ~32 min)
# writes the same paths with an extra MDS 3-D arm and will FAIL the Table S13 scope
# guard on purpose — that artifact's note documents MDS 3-D as deliberately omitted,
# so widening it must be a conscious edit (D197). The canonical scope is therefore the
# DEFAULT; the wider set is opt-in and must be requested explicitly:
#   make absratio-variants ABSRATIO_VARIANTS_FAST=            # full 4-variant set
#   make absratio-variants ABSRATIO_VARIANTS_FAST=--pca-only  # fast, *-pca_only file
ABSRATIO_VARIANTS_FAST ?= --mds-2d-only
absratio-variants:
	$(PY) code/experiment_K_absratio_variants.py $(ABSRATIO_VARIANTS_FAST)

# ---- D199: unstandardized value-PCA scale diagnostics -> results/exC1_vraw_scale.csv.
# The Table S3 note and the Section 3.2 sentence read it; must run before `tableS`.
vraw-scale:
	$(PY) code/make_vraw_scale_diagnostics.py

# ---- D199: per-event drift taxonomy for the F1-physical family fit -> backs the
# SI Extended-analyses "11 distinct values in 119 of 121 events" sentence.
f1-taxonomy:
	$(PY) code/make_f1_drift_taxonomy.py

# ---- Table S13 is generated from results/exK_absratio_summary.csv (never hand-written)
tableS13:
	$(PY) code/make_tableS13_absratio.py

# ---- Verdict-layer replay (D197): re-derives all 338 stored verdicts with the
# shipped decision function; Table S14 prints the count it produces.
verdict-replay:
	$(PY) code/verdict_rubric_replay.py

# ---- Table S14 reads two generated inputs: the rubric replay agreement and
# ---- the verdict-agreement summary. The latter is written by
# ---- `verdict_provenance_agreement.py`, which the aggregate targets used to
# ---- schedule AFTER `tableS14` — so a fresh unpack could not build the table.
# ---- File targets, as elsewhere: built when missing or stale, not on every run.
results/verdict_replay_agreement.csv: code/verdict_rubric_replay.py
	$(PY) code/verdict_rubric_replay.py

results/verdict_agreement_summary.csv: code/verdict_provenance_agreement.py
	$(PY) code/verdict_provenance_agreement.py

tableS14: results/verdict_replay_agreement.csv results/verdict_agreement_summary.csv
	$(PY) code/make_tableS14_provenance.py

# ---- Phase C8: reproducible verdict-provenance strata + rubric-vs-author concordance
# ---- Phase G2: density baseline (the dense-arm comparator)
# ---- File target: `check` asserts 20 claims against this summary and no
# ---- aggregate target used to produce it, so a fresh unpack could not run the
# ---- numbers gate. It is a file target so the project tree does not re-run it.
results/exL_density_baseline_summary.csv: code/experiment_L_density_baseline.py
	$(PY) code/experiment_L_density_baseline.py $(LB_REPS)

density-baseline:
	$(PY) code/experiment_L_density_baseline.py $(LB_REPS)

# ---- Phase M: family-suite expansion audit (promoted from scratch 2026-10-13, D-row pending).
# MDS-bound (minutes); deliberately NOT wired into `all` until the extended-battery
# registry lands — this script emits raw metrics only, no tier verdicts.
# Fast smoke: FAMILYEXP_FAST=--pca-only make familyexpansion
#
# D158/registry §8: the canonical T7 is the SPAN-MATCHED random-suite benchmark,
# and `--benchmark-matched` is a separate invocation that short-circuits the
# default run. Without it the script writes only the sensitivity variant
# (exM_benchmark_4param*), and `tableM` and `check` read the canonical
# exM_benchmark_matched_summary.csv — which the default run never produces.
FAMILYEXP_FAST ?=
familyexpansion: familyexpansion-matched
	$(PY) code/experiment_M_familyexpansion.py $(FAMILYEXP_FAST)

results/exM_benchmark_matched_summary.csv: code/experiment_M_familyexpansion.py
	$(PY) code/experiment_M_familyexpansion.py --benchmark-matched

familyexpansion-matched:
	$(PY) code/experiment_M_familyexpansion.py --benchmark-matched

# ---- Phase M (D159): generated verification report for the family-expansion audit.
# Reads results/exM_*.csv (incl. canonical T7 summary); writes docs/family_group_investigation_findings.md.
familyinvestigation:
	$(PY) code/make_familyinvestigation_report.py

# ---- Phase M (D165): formal tier table under registry v1.0 (registry §9).
# Reads results/exM_*.csv; writes manuscript/tables/tableM_tiers.md.
tableM:
	$(PY) code/make_tableM_tiers.py

provenance-agreement:
	$(PY) code/verdict_provenance_agreement.py

fig2-print:
	$(PY) code/make_fig2_print.py

fig3-print:
	$(PY) code/make_fig3_print.py

fig5-print:
	$(PY) code/make_fig5_print.py

si-panels:
	$(PY) code/make_si_panels.py

fig7-replay:
	$(PY) code/make_fig7_replay.py

duplicates:
	$(PY) code/robustness_duplicates.py

# Paper-level interval for the tie-aware churn fraction (Table S4 panel C).
# Standalone because it re-ordinates every variant 122 times (~10 min).
churn-frac-ci:
	$(PY) code/churn_frac_ci.py

radius:
	$(PY) code/radius_check.py

rho-bootstrap:
	$(PY) code/rho_paper_bootstrap.py

family-tiers:
	$(PY) code/family_tiers.py

family-tier-sensitivity: family-tiers
	$(PY) code/family_tier_sensitivity.py

verdict-power:
	$(PY) code/experiment_P_verdict_power.py
	$(PY) code/experiment_P_verdict_power_57.py

baseline:
	$(PY) -c "import importlib.util as u; s=u.spec_from_file_location('f','code/phase4_freeze.py'); m=u.module_from_spec(s); s.loader.exec_module(m); b=m.verify_baseline(); print('baseline OK:', b['data_version'], '| papers', b['baselines']['n_papers'])"

selftest:
	$(PY) code/stats_selftest.py

# Full reproduction from the released layers: the computation targets first
# (the repo's daily `all` starts from committed outputs), then every figure,
# table and gate. This is the command the code archive documents.
# `corrections` carries the correction-suite experiments (block permutation,
# churn correction, the sensitivity families, the benchmarks); the figure and
# table builders read their outputs, so the chain must run both.
reproduce: stage-layers figures sdr-paper-weighted sdr-hierarchical p2-3d r1-anchor corrections absratio-variants mdsstress familyexpansion family-tiers family-tier-sensitivity verdict-power rho-bootstrap block-null power-extension robustness-weighting extraction-sensitivity family-search fig4 figD figS13 figS14 figs-built tables vraw-scale f1-taxonomy tableS13 tableM verdict-replay tableS14 provenance-agreement density-baseline composition check style dups wordcount
	@echo "[make] REPRODUCE complete - layers -> figures -> tables -> gates"

recompute: figures fig4
	@echo "[make] recompute complete — verify: git diff results/ must be empty"

figures:
	FIG_DPI=$(FIG_DPI) $(PY) code/experiment_C1.py
	FIG_DPI=$(FIG_DPI) $(PY) code/experiment_B_familymaps.py
	FIG_DPI=$(FIG_DPI) $(PY) code/experiment_C2_stability.py
	FIG_DPI=$(FIG_DPI) $(PY) code/experiment_D_concordance.py

fig2-disk:
	$(PY) code/make_fig2_print.py

fig3:
	$(PY) code/make_fig3_print.py

fig5:
	$(PY) code/make_fig5_print.py

figD:
	FIG_DPI=$(FIG_DPI) $(PY) code/make_figD.py

figS13:
	$(PY) code/make_figS13_value_f1f3.py

figS14:
	$(PY) code/make_figS14_value_f1f4.py


fig4:
	FIG_DPI=$(FIG_DPI) $(PY) code/fig4_f2_v1_vs_expanded.py

figs-built:
	$(PY) code/make_prisma_flow.py
	$(PY) code/make_fig1_coreport.py

ga:
	$(PY) code/make_ga_workflow.py

tables:
	$(PY) code/make_tables.py

# D205: corpus composition recomputed from the released layers; the numbers
# checker pins Table 1, the family table and the verdict-layer splits to it.
composition: results/verdict_agreement_summary.csv
	$(PY) code/make_corpus_composition.py

# D208: place the released layers where the scripts expect them, so the code
# archive runs from a fresh unpack of the data record (no-op in this tree).
stage-layers:
	$(PY) code/stage_layers.py

# Canonical gate. paragraph_map remains a rev2-only historical check; the full
# focused gate is check-a.
check: composition results/verdict_agreement_summary.csv
	$(PY) code/check_numbers.py --main manuscript/latex/main_rev2a.tex
	$(PY) code/check_highlights.py
	$(PY) code/check_si_refs.py --main manuscript/latex/main_rev2a.tex
	$(PY) code/check_si_parity.py
	$(PY) code/paragraph_map.py --check

# D207: md->tex pipeline retired for the manuscript track — main_rev2/supplementary_rev2.tex
# are the canonical hand-edited sources; check/style/dups/wordcount gates point at them.
# md2latex.py is kept only for make_report_discuss.py's table import and is not run here.
# NOTE always compile main BEFORE supplementary (the SI resolves cross-refs via main_rev2.aux).
# A clean build (no .bbl) prints transient first-pass "Citation ... undefined" warnings;
# the guard below judges the FINAL logs, after latexmk's bibtex + rerun chain.
# Legacy rev2 targets: retained for the archived full manuscript. The canonical
# focused submission uses paper-a / check-a.
paper:
	cd manuscript/latex && latexmk -pdf -interaction=nonstopmode main_rev2.tex
	cd manuscript/latex && latexmk -pdf -interaction=nonstopmode supplementary_rev2.tex
	cd manuscript/latex && latexmk -pdf -interaction=nonstopmode supplementary_main.tex
	@if grep -iE "undefined|^!" manuscript/latex/main_rev2.log manuscript/latex/supplementary_rev2.log manuscript/latex/supplementary_main.log >/dev/null; then \
	  echo "[paper] FAIL — undefined refs/citations or errors remain in the final logs"; exit 1; \
	else echo "[paper] OK — final logs clean: 0 errors, 0 undefined refs/citations"; fi

# Focused rev2a build. Style is deliberately separate: rev2a still has prose
# semicolons (check-a-style is expected red until the copy-edit pass).
paper-a:
	cd manuscript/latex && latexmk -pdf -interaction=nonstopmode main_rev2a.tex
	cd manuscript/latex && latexmk -pdf -interaction=nonstopmode supplementary_main.tex
	@if grep -iE "undefined|^!" manuscript/latex/main_rev2a.log manuscript/latex/supplementary_main.log >/dev/null; then \
	  echo "[paper-a] FAIL — undefined refs/citations or errors remain in the final logs"; exit 1; \
	else echo "[paper-a] OK — final logs clean: 0 errors, 0 undefined refs/citations"; fi

check-a:
	$(PY) code/check_numbers.py --main manuscript/latex/main_rev2a.tex
	$(PY) code/check_dup.py --main manuscript/latex/main_rev2a.tex
	$(PY) code/check_style.py --main manuscript/latex/main_rev2a.tex
	$(PY) code/check_highlights.py
	$(PY) code/check_si_refs.py --main manuscript/latex/main_rev2a.tex
	$(PY) code/check_si_parity.py
	$(PY) code/wordcount_body.py --tex manuscript/latex/main_rev2a.tex
	$(PY) code/wordcount.py

check-a-style:
	$(PY) code/check_style.py --main manuscript/latex/main_rev2a.tex

# ---- D213: seconds-scale checks on the shipping code archive. The two-hour
# `make reproduce` inside a fresh unpack stays the deposit gate; this gate
# catches the defects that have actually reached the archive (a script the
# Makefile invokes but the bundle omits, a target ordered before its producers,
# a stale inventory, checksum or manifest).
archive-preflight:
	$(PY) code/archive_preflight.py

# ---- D214: the two-hour reproduction, run in a scratch directory and recorded
# in release/archive_verification.json, so a later session can tell which build a
# run validated. `make archive-preflight` compares the current bundle with it.
archive-verify:
	$(PY) code/archive_verify.py --run

si-refs:
	$(PY) code/check_si_refs.py

style:
	$(PY) code/check_style.py

wordcount:
	$(PY) code/wordcount.py

r1-anchor:
	$(PY) code/r1_anchor.py

sdr-paper-weighted:
	$(PY) code/sdr_paper_weighted.py

sdr-hierarchical:
	$(PY) code/sdr_hierarchical.py

# Three-component sensitivity for the main SDR. Uses the presence density
# (rho_v) for the value layer; the old script used the mean z-score, which is
# not a density and produced the wrong 3-D value-layer SDR.
p2-3d:
	$(PY) code/p2_3d_sdr.py

paper-stats: r1-anchor sdr-paper-weighted
	@echo "[paper-stats] R1 anchor and per-event SDR reproduced"

decision-consequences:
	$(PY) code/decision_consequences.py

structural-generality:
	$(PY) code/experiment_structural_generality.py --reps 3

block-diagnostics:
	$(PY) code/block_model_diagnostics.py

r2-calibration:
	$(PY) code/r2_per_event_calibration.py --rep 300 --events 30 --perm 199

# D206: body word count (prose and full-body scopes, math-removed and
# math-as-words) with the scope printed alongside the numbers.
wordcount-body:
	$(PY) code/wordcount_body.py

dups:
	$(PY) code/check_dup.py
