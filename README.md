# Ordination stability audit — core code

Companion code for: *Calibrating deletion stability in pooled multivariate
environmental data: a study-level resampling framework*.

This repository is the **code release** for the audit: the reusable diagnostic,
the R1 drift anchor, the R2 per-event calibration, the SDR estimators, the
verification gates, the analysis scripts, and figure and table builders. The
data layers are in the Zenodo data record (see *Full reproduction* below).

## What it does

- Leave-one-study-out deletion with Procrustes drift and neighbour churn.
- A volume-matched scattered control that separates block structure from
  deleted volume.
- Row-weighted, pooled study-level, and per-event study-weighted sparsity drift
  ratios (SDR).
- R2 calibration of the per-event estimator (size and power).
- `code/stability_diagnostic.py`: a standalone Python API and CLI for applying
  the diagnostic to another meta-matrix.

## EES additions

The EES revision adds seven scripts:

| File | Purpose |
|---|---|
| `code/experiment_H_volumematch_value.py` | Value-layer volume-matched replication (70 studies, 421 rows) |
| `code/ees_rev3_splits.py` | Real-corpus repeated/varying-panel, single/multi-row, and drop-substitute splits |
| `code/preprocessing_sensitivity.py` | Fixed-centring/z-scoring sensitivity for the presence and value layers |
| `code/make_fig8_block_effect.py` | Figure 3: per-study paired ratios and block-versus-matched drift |
| `code/composition_matched_control.py` | Composition/density-matched control (same presence configurations where donors exist) |
| `code/presence_misclassification_sensitivity.py` | Cell-level present-to-NR perturbation sensitivity |
| `code/preprocessing_block_factor.py` | Fixed-preprocessing block factor from the per-paper layer |

These scripts require the frozen extraction layers from the Zenodo data record;
their tables are generated under `results/`, and figures under `figures/`.

## Install

```
conda env create -f environment.yml
# or
pip install -r requirements.txt
```

Pins: Python 3.13.7, NumPy 2.4, SciPy 1.17, scikit-learn 1.9, pandas 3.0,
Matplotlib 3.11. Seed 42 throughout.

## Quickstart

```
python3 code/stability_diagnostic.py --help
```

The individual analysis scripts read the staged extraction layers
(`docs/extraction/stageA`, `docs/extraction/stageB`). Outputs are written to
`results/` and `figures/`. The data layers are in the Zenodo data record; unpack
it beside this repository, then run:

```
python3 code/stage_layers.py   # place the data layers at the expected paths
make all                       # fast rebuild: figures, tables, gates (~1 min)
make paper-stats               # R1 anchor + per-event SDR (~1 min)
make r2-calibration            # per-event size/power calibration (~5 min)
make reproduce                 # full pipeline (~2.5 h)
```

## Full reproduction

- **Data record (CC BY 4.0):** https://doi.org/10.5281/zenodo.23973301
  (version 2; concept DOI https://doi.org/10.5281/zenodo.23552334). This is a
  general corpus release, not a single-paper supplement.
- **Code (MIT):** this repository.

The repository contains the analysis scripts and the figure and table
builders. The data record supplies the input layers.

## Reproducibility note

t-SNE is a descriptive variant and its embedding depends on the BLAS/threading
environment. t-SNE rows in the SDR and churn tables are run-dependent; the
metric variants (PCA and MDS) and the reported inferential statistics reproduce
exactly.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, cite the paper and this repository. A `CITATION.cff` is
included. Author identifiers: P. Wibowo 0000-0002-2738-744X · C. Veesommai
Sillberg 0000-0001-8400-7469.
