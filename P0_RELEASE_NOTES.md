# P0 release (code-v1.11)

Adds the post-hoc EES sensitivities and the current coverage figure builder:

- `code/ees/row_count_sensitivity.py` (row-count threshold sensitivity; 1.30)
- `code/ees/p0_tie_aware_grade.py`
- `code/ees/p0_composition_percentile.py`
- `code/ees/make_fig_percentile_ecdf.py` (legacy two-panel ECDF alternative)
- `code/figures/make_fig_main_coverage.py` (current main-text coverage figures)
- `code/ees_revision_checks.py` (fixes the `clean_primary_audit` import)

These expect the manuscript `results/wp3_clean/{coverage,value}_events.csv` and
`composition_matched_control.csv`, produced by
`code/core/clean_primary_audit.py`. `make curated` runs the curated audit and
these sensitivities. `results/` and `figures/` are gitignored; the scripts are
the release artifact, and outputs are regenerated from the Zenodo data record.

## Release commands

```bash
cd public-code
git add code/ees/row_count_sensitivity.py \
        code/ees/p0_tie_aware_grade.py \
        code/ees/p0_composition_percentile.py \
        code/ees/make_fig_percentile_ecdf.py \
        code/figures/make_fig_main_coverage.py \
        code/ees_revision_checks.py \
        Makefile README.md CITATION.cff P0_RELEASE_NOTES.md
git commit -m "Add row-count and coverage-figure builders; EES v1.54 alignment"
git tag -a code-v1.11 -m "Row-count sensitivity and coverage figure builders"
git push origin main --tags
```

After the push, the manuscript cites `code-v1.11` in `main.tex`,
`cover_letter.md`, `title_page.tex`, and the data-record metadata.

**Do not submit** with the manuscript citing `code-v1.10` for the current
results unless `code-v1.11` is pushed.
