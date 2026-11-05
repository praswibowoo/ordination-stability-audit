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
	$(PY) code/stage_layers.py

selftest:
	$(PY) code/stats_selftest.py

core:
	$(PY) code/experiment_C1.py
	$(PY) code/experiment_H_volumematch.py
	$(PY) code/experiment_H_volumematch_value.py
	$(PY) code/experiment_L_density_baseline.py
	$(PY) code/experiment_S_block_null.py
	$(PY) code/experiment_S_validation.py
	$(PY) code/experiment_S_power_extension.py
	$(PY) code/experiment_structural_generality.py
	$(PY) code/p2_3d_sdr.py
	$(PY) code/make_vraw_scale_diagnostics.py
	$(PY) code/decision_consequences.py
	$(PY) code/block_model_diagnostics.py
	$(PY) code/churn_corrected.py
	$(PY) code/pattern_churn.py
	$(PY) code/churn_frac_ci.py
	$(PY) code/sdr_paper_weighted.py
	$(PY) code/sdr_hierarchical.py
	$(PY) code/r1_anchor.py
	$(PY) code/r2_per_event_calibration.py --rep 300 --events 30 --perm 199
	$(PY) code/sim_random_drift_benchmark.py
	$(PY) code/ees_rev3_splits.py
	$(PY) code/preprocessing_sensitivity.py
	$(PY) code/preprocessing_block_factor.py
	$(PY) code/composition_matched_control.py
	$(PY) code/presence_misclassification_sensitivity.py
	$(PY) code/extraction_confidence_sensitivity.py
	$(PY) code/robustness_duplicates.py
	$(PY) code/robustness_exclB175.py
	$(PY) code/robustness_weighting.py
	$(PY) code/rho_paper_bootstrap.py
	$(PY) code/radius_check.py
	$(PY) code/tie_order_sensitivity.py
	$(PY) code/dedup_scan.py

figures:
	$(PY) code/make_fig1_coreport.py
	$(PY) code/make_fig2_print.py
	$(PY) code/make_fig8_block_effect.py
	$(PY) code/make_si_panels.py
	$(PY) code/make_fig7_replay.py

mds:
	$(PY) code/mds_classical_check.py
	$(PY) code/mds_multistart_check.py
	$(PY) code/mds_stress.py
	$(PY) code/mds_seed_baseline.py --seeds 0-9 --protocol all
	$(PY) code/audit_mds_scope.py

