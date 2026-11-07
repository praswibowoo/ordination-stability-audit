# code-v1.9 release notes

**Date:** 2026-11-07
**Primary analysis set:** 730 coverage rows / 122 studies; 410 value rows / 69 studies.

## Changes from code-v1.8

- `experiment_C1.load()` now returns the primary 730/410 analysis set.
- `experiment_C1.load_legacy()` returns the archived 764/421 set.
- Added `revision_layers.py` (primary-set derivation).
- Added `block_interaction_primary.py` (block x log-volume interaction and
  volume-specific factors).
- Added `composition_formal_test.py`, `composition_vs_density.py`,
  `composition_exact_subset.py`, and `wp3_composition_extra.py`.
- `composition_matched_control.py` now defaults to the optimal one-to-one
  min-cost assignment with a `ceil(n_p/2)` donor-study cap; the greedy capped
  path remains available with `--matching greedy`.
- Added `observed_sigma_calibration.py` and `subset_sdr_calibration.py`.
- Added `study_level_adjusted.py` (study-size/configuration-rarity adjustment).
- Added `coverage_paired_ratio_rules.py` (one paired-ratio rule for all coverage
  arms).
- `stage_layers.py` recognises the v1.3 data bundle.
- `networkx` added to the environment.

## Parity check

Run with the v1.3 data bundle on 2026-11-07:

| Check | Public output | Manuscript |
|---|---|---|
| Primary coverage rows / studies | 730 / 122 | 730 / 122 |
| Primary value rows / studies | 410 / 69 | 410 / 69 |
| Coverage block factor | 1.402 (1.247-1.588) | 1.40 (1.25-1.58) |
| Coverage ratio of medians | 1.08 (0.96-1.46) | 1.08 (0.96-1.46) |
| Coverage paired ratio (first draw) | 1.38 (1.21-1.64) | 1.38 (1.21-1.64) |
| Block x log-volume | 0.2869 (0.1205-0.4640) | 0.287 (0.121-0.464) |
| Block factor at 1 / 5 / 14 rows | 0.887 / 1.407 / 1.891 | 0.89 / 1.41 / 1.89 |
| Value block factor (2-D) | 1.477 (1.203-1.860) | 1.48 (1.20-1.86) |
| Composition exact fraction | 0.114481; 99/122 zero | 0.114; 99/122 zero |

The coverage block model and interaction reproduce the manuscript's primary
numbers at the reported precision. The value-layer block factor differs in the
third decimal because the public run uses its own seeded draw stream; it
rounds to the same 1.48 (1.20-1.86).

## Not bit-identical

- t-SNE is environment-dependent and remains descriptive.
- Bootstrap draws are seed-42 reproducible within the pinned environment but
  are not bit-identical across separate RNG streams in the value-layer script.
