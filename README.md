# Ordination stability audit — core code

Companion code for:

> *When does deleting one study change a pooled ordination? A study-level
> deletion audit of a water-quality corpus.*

This repository is the **code release** for the audit. It contains the reusable
diagnostic, the core deletion-stability experiments, the EES sensitivity
analyses, and the figure builders.

It does **not** contain manuscript sources, derived data, result tables, or
generated figures. The data record supplies the input layers; outputs are
written to `results/` and `figures/`.

## What it does

- Leave-one-study-out deletion with Procrustes drift and neighbour churn.
- Volume-matched scattered controls for block structure versus deleted volume.
- Composition/density-matched controls.
- Row-weighted, pooled study-level, and per-event study-weighted sparsity drift
  ratios (SDR).
- Rank, cluster, and decision-level stability.
- R1 drift scaling and R2 size/power calibration.
- `code/core/stability_diagnostic.py`: a standalone Python API and CLI for applying
  the diagnostic to another meta-matrix.

## Code layout

| Area | Main files |
|---|---|
| Diagnostic | `code/core/stability_diagnostic.py` |
| Data staging | `code/data/stage_layers.py`, `phase4_freeze.py`, `harmonize_units.py`, `stageB_merge.py` |
| Core deletion audit | `code/core/experiment_C1.py`, `experiment_H_volumematch.py`, `experiment_H_volumematch_value.py`, `experiment_L_density_baseline.py`, `experiment_S_block_null.py`, `experiment_structural_generality.py` |
| Drift, churn, SDR | `code/core/churn_corrected.py`, `pattern_churn.py`, `churn_frac_ci.py`, `sdr_paper_weighted.py`, `sdr_hierarchical.py` |
| Rank and decisions | `code/core/decision_consequences.py`, `p2_3d_sdr.py`, `block_model_diagnostics.py` |
| Calibration | `code/core/r1_anchor.py`, `r2_per_event_calibration.py`, `experiment_S_validation.py`, `experiment_S_power_extension.py`, `sim_random_drift_benchmark.py` |
| EES sensitivities | `code/ees/ees_rev3_splits.py`, `preprocessing_sensitivity.py`, `preprocessing_block_factor.py`, `composition_matched_control.py`, `composition_formal_test.py`, `composition_vs_density.py`, `composition_exact_subset.py`, `wp3_composition_extra.py`, `coverage_paired_ratio_rules.py`, `study_level_aggregate.py`, `study_level_adjusted.py`, `joint_2d3d_ratio.py`, `presence_misclassification_sensitivity.py`, `extraction_confidence_sensitivity.py` |
| Primary interaction | `code/core/block_interaction_primary.py` |
| Density calibration | `code/core/observed_sigma_calibration.py`, `code/core/subset_sdr_calibration.py` |
| Robustness | `code/core/robustness_duplicates.py`, `robustness_exclB175.py`, `robustness_weighting.py`, `rho_paper_bootstrap.py`, `radius_check.py`, `tie_order_sensitivity.py`, `dedup_scan.py` |
| MDS diagnostics | `code/mds/mds_classical_check.py`, `mds_multistart_check.py`, `mds_stress.py`, `mds_seed_baseline.py`, `audit_mds_scope.py` |
| Figures | `code/figures/make_fig1_coreport.py`, `make_fig2_print.py`, `make_fig8_block_effect.py`, `make_si_panels.py`, `make_fig7_replay.py` |

Manuscript-only material — family/verdict legacy analyses, manuscript checks,
table builders, PRISMA/corpus-funnel builders, `docs/`, `results/`, and
`figures/` — is kept in the main manuscript repository, not here.

## Install

```bash
conda env create -f environment.yml
# or
pip install -r requirements.txt
```

Pins: Python 3.13.7, NumPy 2.4, SciPy 1.17, scikit-learn 1.9, pandas 3.0,
Matplotlib 3.11. Seed 42 throughout.

## Quickstart

```bash
python3 code/core/stability_diagnostic.py --help
```

The analysis scripts read the staged extraction layers. The data layers are in
the Zenodo record; unpack it beside this repository, then run:

```bash
python3 code/data/stage_layers.py   # place the data layers at the expected paths
make all                       # stage layers + statistical self-test
make core                      # full core analysis
make figures                   # figures from the generated results
make mds                       # MDS diagnostics
```

`make all` is the fast check. `make core` and `make figures` require the data
record; `make figures` also requires the results produced by `make core`.

**Primary analysis set.** `experiment_C1.load()` now returns the manuscript's
primary set: 730 coverage rows from 122 studies and 410 value-complete rows from
69 studies. The legacy 764/421 set is available as
`experiment_C1.load_legacy()` or with `--layer legacy` in scripts that expose
that option. Outputs are written to `results/`; the manuscript's
`results/wp3_primary/` contains the same primary estimates under the manuscript
file names.

**Released presence table.** `presence.csv` populates `batch_id` on the first
row of each paper block. To recover paper identity without the analysis loader,
merge `family_map.row_id` to the presence row ordinal and use
`family_map.batch_id_filled`; `stage_layers.py` does this staging automatically.
Filtering on `row_kind = data` alone is not sufficient.

## Data record

- **Data (CC BY 4.0):** concept DOI https://doi.org/10.5281/zenodo.23552334
  (v1.3 documentation; the v1.2 extraction files are byte-identical; version DOI
  https://doi.org/10.5281/zenodo.23973301).
- **Code (MIT):** this repository.

The data record is a general corpus release, not a single-paper supplement.

## Reproducibility note

The public `make core` target runs the primary 730/410 analysis and writes its
outputs to `results/`. The manuscript's `results/wp3_primary/` is the same
primary set under the manuscript file names; the parity check is described in
`docs/revision/EEA_correction_2026-11-07/PHASE5_audit.md` in the manuscript
repository. The archived 764/421 analysis is available through
`experiment_C1.load_legacy()` and is not the default.

t-SNE is a descriptive variant and its embedding depends on the BLAS/threading
environment. t-SNE rows are run-dependent; the metric variants (PCA and MDS)
are deterministic under the pinned environment. The bootstrap intervals are
seed-42 reproducible given the same environment.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, cite the paper and this repository. A `CITATION.cff` is
included. Author identifiers: P. Wibowo 0000-0002-2738-744X · C. Veesommai
Sillberg 0000-0001-8400-7469.
