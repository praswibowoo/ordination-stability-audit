# EES result tables

These tables accompany the EES manuscript *Calibrating deletion stability in
pooled multivariate environmental data: a study-level resampling framework*.
They are the outputs of the seven scripts listed in the repository README.

| File | Source | Key values |
|---|---|---|
| `exH_value_volumematch.csv` | `code/experiment_H_volumematch_value.py` | Value-layer block and first scattered events (70 + 70) |
| `exH_value_volumematch_draws.csv` | same | All 7,000 scattered draws |
| `exH_value_summary.csv` | same | Paired ratio 1.57 (0.96–2.12); ratio of medians 1.50 (0.93–2.49) |
| `exH_value_loglinear_model.csv` | same | Block factor 1.22 (0.96–1.60); volume elasticity 0.92 |
| `ees_rev3_splits.csv` | `code/ees_rev3_splits.py` | Repeated 1.45 (1.25–1.68); varying 1.33 (1.05–1.70); single-row 1.10 (0.76–1.75); multi-row 1.43 (1.26–1.61); drop-substitutes 1.42 (1.27–1.61) |
| `preprocessing_sensitivity.csv` | `code/preprocessing_sensitivity.py` | Per-paper fixed-preprocessing block/matched drift |
| `preprocessing_sensitivity_summary.csv` | same | Presence 1.40 (1.17–1.76); value 1.12 (0.76–1.74); ratio of medians 1.00; B154 57% |
| `composition_matched_control.csv` | `code/composition_matched_control.py` | Per-draw composition/density-matched drift |
| `composition_matched_summary.csv` | same | Overall 1.18 (1.05–1.32); exact-config subset 0.98 (0.82–1.15); no-exact median ratio 1.20 |
| `presence_misclassification_sensitivity.csv` | `code/presence_misclassification_sensitivity.py` | Per-replicate perturbed block factors |
| `presence_misclassification_summary.csv` | same | 5–11% perturbation: block factor 1.24–1.38 |
| `preprocessing_block_factor.csv` | `code/preprocessing_block_factor.py` | Presence 1.39 (1.21–1.60); value 1.07 (0.74–1.52) |

The scripts require the frozen extraction layers and the main volume-matched
results, which are in the Zenodo data bundle. The repository is the readable
core, not the full reproduction bundle. The value-layer paired ratio falls from
1.57 under recomputed z-scores to 1.12 under fixed z-scores, so the manuscript
reports the value-layer average effect as not robust to the preprocessing
protocol; B154 remains extreme under both protocols (53% vs 57%). The
presence-layer volume-matched factor falls from 1.42 to 1.18 under
composition/density matching, and the exact-configuration subset shows no
effect; the manuscript reports the block effect as partly composition/rarity
dependent.
