# Ordination stability audit — core code
#
#   make all        stage the data layers and run the statistical self-test
#   make core       re-run the core deletion-stability analysis
#   make figures    build the EES main/SI figures from generated results
#   make mds        MDS diagnostics
#
# The data record is required for anything that reads the extraction layers:
# https://doi.org/10.5281/zenodo.23973301

PY ?= python3

.PHONY: all stage-layers selftest core figures mds

all: stage-layers selftest

stage-layers:
	$(PY) code/data/stage_layers.py

selftest:
	$(PY) code/core/stats_selftest.py

core:
	$(PY) code/core/experiment_C1.py
	$(PY) code/core/experiment_H_volumematch.py
	$(PY) code/core/experiment_H_volumematch_value.py
	$(PY) code/core/experiment_L_density_baseline.py
	$(PY) code/core/experiment_S_block_null.py
	$(PY) code/core/experiment_S_validation.py
	$(PY) code/core/experiment_S_power_extension.py
	$(PY) code/core/experiment_structural_generality.py
	$(PY) code/core/p2_3d_sdr.py
	$(PY) code/ees/make_vraw_scale_diagnostics.py
	$(PY) code/core/decision_consequences.py
	$(PY) code/core/block_model_diagnostics.py
	$(PY) code/core/churn_corrected.py
	$(PY) code/core/pattern_churn.py
	$(PY) code/core/churn_frac_ci.py
	$(PY) code/core/sdr_paper_weighted.py
	$(PY) code/core/sdr_hierarchical.py
	$(PY) code/core/r1_anchor.py
	$(PY) code/core/r2_per_event_calibration.py --rep 300 --events 30 --perm 199
	$(PY) code/core/sim_random_drift_benchmark.py
	$(PY) code/ees/ees_rev3_splits.py
	$(PY) code/ees/preprocessing_sensitivity.py
	$(PY) code/ees/preprocessing_block_factor.py
	$(PY) code/ees/composition_matched_control.py --matching optimal --workers 4
	$(PY) code/ees/composition_formal_test.py --vol-csv results/exH_volumematch_draws.csv --comp-csv results/composition_matched_control.csv --median-draws --out results/composition_formal_test.csv
	$(PY) code/ees/wp3_composition_extra.py
	$(PY) code/ees/coverage_paired_ratio_rules.py
	$(PY) code/core/block_interaction_primary.py
	$(PY) code/ees/study_level_adjusted.py
	$(PY) code/core/observed_sigma_calibration.py
	$(PY) code/core/subset_sdr_calibration.py --reps 50 --subsets 100 --perm 199 --sigma 0.87 --out results/subset_sdr_calibration.csv
	$(PY) code/ees/presence_misclassification_sensitivity.py
	$(PY) code/ees/extraction_confidence_sensitivity.py
	$(PY) code/ees/study_level_aggregate.py
	$(PY) code/ees/joint_2d3d_ratio.py
	$(PY) code/core/robustness_duplicates.py
	$(PY) code/core/robustness_exclB175.py
	$(PY) code/core/robustness_weighting.py
	$(PY) code/core/rho_paper_bootstrap.py
	$(PY) code/core/radius_check.py
	$(PY) code/core/tie_order_sensitivity.py
	$(PY) code/core/dedup_scan.py

figures:
	$(PY) code/figures/make_fig1_coreport.py
	$(PY) code/figures/make_fig2_print.py
	$(PY) code/figures/make_fig8_block_effect.py
	$(PY) code/figures/make_si_panels.py
	$(PY) code/figures/make_fig7_replay.py

curated:
	$(PY) code/core/clean_primary_audit.py
	$(PY) code/ees/row_count_sensitivity.py
	$(PY) code/ees/p0_tie_aware_grade.py
	$(PY) code/ees/p0_composition_percentile.py
	$(PY) code/figures/make_fig_main_coverage.py

mds:
	$(PY) code/mds/mds_classical_check.py
	$(PY) code/mds/mds_multistart_check.py
	$(PY) code/mds/mds_stress.py
	$(PY) code/mds/mds_seed_baseline.py --seeds 0-9 --protocol all
	$(PY) code/mds/audit_mds_scope.py

