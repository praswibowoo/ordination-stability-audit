# EES rev3 result tables

These tables accompany the EES revision of *Calibrating deletion stability in
pooled multivariate environmental data: a study-level resampling framework*.
They are the outputs of the four scripts listed in the repository README.

| File | Source | Key values |
|---|---|---|
| `exH_value_volumematch.csv` | `code/experiment_H_volumematch_value.py` | Value-layer block and first scattered events (70 + 70) |
| `exH_value_volumematch_draws.csv` | same | All 7,000 scattered draws |
| `exH_value_summary.csv` | same | Paired ratio 1.57 (0.96–2.12); ratio of medians 1.50 (0.93–2.49) |
| `exH_value_loglinear_model.csv` | same | Block factor 1.22 (0.96–1.60); volume elasticity 0.92 |
| `ees_rev3_splits.csv` | `code/ees_rev3_splits.py` | Repeated 1.45 (1.25–1.68); varying 1.33 (1.05–1.70); single-row 1.10 (0.76–1.75); multi-row 1.43 (1.26–1.61); drop-substitutes 1.42 (1.27–1.61) |
| `preprocessing_sensitivity.csv` | `code/preprocessing_sensitivity.py` | Per-paper fixed-preprocessing block/matched drift |
| `preprocessing_sensitivity_summary.csv` | same | Presence 1.40 (1.17–1.76); value 1.12 (0.76–1.74); ratio of medians 1.00; B154 57% |

The scripts require the frozen extraction layers and the main volume-matched
results, which are in the Zenodo data bundle. The repository is the readable
core, not the full reproduction bundle. The value-layer paired ratio falls from
1.57 under recomputed z-scores to 1.12 under fixed z-scores, so the manuscript
reports the value-layer average effect as not robust to the preprocessing
protocol; B154 remains extreme under both protocols (53% vs 57%).
