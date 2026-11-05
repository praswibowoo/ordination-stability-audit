# Ordination stability audit — core code

Companion code for: *Auditing deletion stability in pooled ordinations: a
paper-level protocol for sparse ecological meta-matrices*.

This repository is the **core code** of the audit: the reusable diagnostic, the
R1 drift anchor, the R2 per-event calibration, the SDR estimators, and the
verification gates. It also carries the Stage-A presence-verification sheet
(`docs/verification/`) so the 39-paper audit in Table S14 can be checked. It is
**not** the full reproduction bundle. The data layers, working documents,
generated results, and manuscript tables ship in the Zenodo archives (see
*Full reproduction* below).

## What it does

- Leave-one-paper-out deletion with Procrustes drift and neighbour churn.
- A volume-matched scattered control that separates block structure from
  deleted volume.
- Row-weighted, pooled paper-level, and per-event paper-weighted sparsity drift
  ratios (SDR).
- R2 calibration of the per-event estimator (size and power).
- `code/stability_diagnostic.py`: a standalone Python API and CLI for applying
  the diagnostic to another meta-matrix.

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
(`docs/extraction/stageA`, `docs/extraction/stageB`) and the generated result
tables (`results/`). Those inputs are part of the full Zenodo bundle, not this
core repository. With the bundle unpacked:

```
make stage-layers   # place the released data layers at the expected paths
make all            # fast rebuild: figures, tables, gates (~1 min)
make paper-stats    # R1 anchor + per-event SDR (~1 min)
make r2-calibration # per-event size/power calibration (~5 min)
make reproduce      # full pipeline (~2.5 h)
```

## Full reproduction

- **Data record (CC BY 4.0):** https://doi.org/10.5281/zenodo.23552335
  (concept DOI https://doi.org/10.5281/zenodo.23552334)
- **Code (MIT):** this repository, tag `code-v1.5`.

The code bundle contains the 188 input files (plus the archive manifest), the generated results, and the
figures, so `make reproduce` runs unmodified. This repository is the readable
core of that bundle. The Stage-A presence-verification sheet and its result log
are in `docs/verification/`.

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
