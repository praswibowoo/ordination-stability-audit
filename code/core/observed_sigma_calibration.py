#!/usr/bin/env python3
"""observed_sigma_calibration.py — null calibration at the observed SDs.

The original observed-sigma CSVs had no generating script. This script:

  1. computes the observed between-study SD of log LOPO drift for the presence
     and value layers;
  2. runs the null calibration (gamma = 0) of experiment_S_validation.calibration
     at those computed SDs for both layers (override with --sigmas);
  3. writes results/observed_sigma_calibration_presence.csv and
     results/observed_sigma_calibration_value.csv.

The label records the layer; sigma_paper records the nuisance SD used in the
simulation. The legacy layer is the default so the cited values are reproducible.

Run: python3 code/observed_sigma_calibration.py [--reps 1000] [--n-perm 199]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")

from experiment_C1 import load, ordinate, knn_idx, drift_after_procrustes  # noqa: E402
from experiment_S_validation import calibration  # noqa: E402
from revision_layers import (add_layer_args, layer_kwargs, load_layers,
                             value_lopo_drifts)  # noqa: E402


def presence_lopo_drifts(X, paper):
    """Median LOPO Procrustes drift per study for the presence PCA."""
    S_full = ordinate("pca_pres", X)
    rows = []
    for p in sorted(set(paper)):
        keep = paper != p
        keep_idx = np.where(keep)[0]
        S_red = ordinate("pca_pres", X[keep])
        drift, _ = drift_after_procrustes(S_full[keep_idx], S_red)
        rows.append(float(np.median(drift)))
    return np.asarray(rows, float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--sigmas", default=None,
                    help="comma-separated nuisance SDs (default: the computed observed SDs)")
    ap.add_argument("--outdir", default=RESULTS,
                    help="output directory (default: results)")
    add_layer_args(ap)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X, rho, paper, params, Xv, keysv, rho_v = load(**layer_kwargs(args))
    vpaper = np.asarray([k[0] for k in keysv], dtype=object)

    pres_drift = presence_lopo_drifts(X, paper)
    val_rows = value_lopo_drifts(load_layers(**layer_kwargs(args)))
    val_drift = np.asarray([r["median_drift"] for r in val_rows], float)
    observed = {
        "presence": float(np.std(np.log(pres_drift), ddof=1)),
        "value": float(np.std(np.log(val_drift), ddof=1)),
    }
    print("observed SD of log LOPO drift: " +
          ", ".join(f"{k}={v:.4f}" for k, v in observed.items()))
    if args.sigmas:
        sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    else:
        sigmas = sorted({round(observed["presence"], 2),
                         round(observed["value"], 2)})
    print("calibration sigmas: " + ", ".join(f"{s:.2f}" for s in sigmas))

    layers = {
        "presence": (X, rho, paper),
        "value": (Xv, rho_v, vpaper),
    }
    for label, (Xlayer, rholayer, player) in layers.items():
        scenarios = tuple((s, 0.0) for s in sigmas)
        rows, _meta = calibration(
            Xlayer, rholayer, player, scenarios=scenarios,
            n_rep=args.reps, n_perm=args.n_perm)
        out = []
        for s in sigmas:
            sub = [r for r in rows if abs(float(r["sigma_paper"]) - s) < 1e-9]
            out.append(dict(
                sigma_paper=s,
                rej_row=float(np.mean([r["rej_row"] for r in sub])),
                rej_block=float(np.mean([r["rej_block"] for r in sub])),
                n=len(sub),
                label=label,
            ))
        path = os.path.join(args.outdir,
                            f"observed_sigma_calibration_{label}.csv")
        with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["sigma_paper", "rej_row",
                                               "rej_block", "n", "label"])
            w.writeheader()
            w.writerows(out)
        os.replace(path + ".tmp", path)
        print(f"[observed-sigma] wrote {os.path.relpath(path, ROOT)}")
        for r in out:
            print(f"  {label} sigma={r['sigma_paper']:.2f} "
                  f"row={r['rej_row']:.3f} paper={r['rej_block']:.3f} n={r['n']}")


if __name__ == "__main__":
    main()
