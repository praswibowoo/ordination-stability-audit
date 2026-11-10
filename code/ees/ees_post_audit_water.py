#!/usr/bin/env python3
"""Water-only full-set and panel-collapsed audit for the EES revision."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from core_revision_checks import fit, load_curated  # noqa: E402
from ees_revision_checks import coverage_rows, classify_coverage  # noqa: E402
from ees_post_audit import (  # noqa: E402
    OUT, block_model, collapse_panels, run_audit, summarise,
)

EXCLUDE = ("B175", "B111", "B095", "B182", "B239")


def main():
    X, rho, paper = load_curated()
    recs = coverage_rows()
    types = np.asarray([classify_coverage(r) for r in recs], dtype=object)
    rec_paper = np.asarray([r["paper"] for r in recs], dtype=object)
    keep_cur = np.asarray([p not in EXCLUDE for p in rec_paper])
    assert int(keep_cur.sum()) == 716
    assert len(paper) == int(keep_cur.sum())
    # X is the 716-row curated matrix in the same order as recs[keep_cur].
    water716 = (types == "water")[keep_cur]
    assert int(water716.sum()) == 671
    Xw, rhow, paperw = X[water716], rho[water716], paper[water716]
    assert Xw.shape == (671, 44), Xw.shape
    assert len(set(paperw)) == 115

    out = {"n_rows": int(len(Xw)), "n_papers": int(len(set(paperw)))}
    b, m = run_audit(Xw, rhow, paperw, fit, dims=2, draws=100, align="sign",
        label="water_full")
    out["full"] = summarise(b, m, "water_full_")
    out["full_model"] = block_model(b, m)
    b.to_csv(os.path.join(OUT, "water_full_block.csv"), index=False)
    m.to_csv(os.path.join(OUT, "water_full_matched.csv"), index=False)

    Xc, rhoc, paperc, _ = collapse_panels(Xw, rhow, paperw)
    bc, mc = run_audit(Xc, rhoc, paperc, fit, dims=2, draws=100, align="sign",
        label="water_collapsed")
    out["collapsed"] = summarise(bc, mc, "water_collapsed_")
    out["collapsed_model"] = block_model(bc, mc)
    out["collapsed_rows"] = int(len(Xc))
    bc.to_csv(os.path.join(OUT, "water_collapsed_block.csv"), index=False)
    mc.to_csv(os.path.join(OUT, "water_collapsed_matched.csv"), index=False)

    with open(os.path.join(OUT, "water.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
