#!/usr/bin/env python3
"""mds_classical_check.py — Phase 2 gate for the deterministic MDS protocol.

Checks that the production Jaccard MDS:
  1. uses classical-MDS initialization (sklearn >= 1.8);
  2. returns a bit-identical embedding on two consecutive fits;
  3. reaches a stress-1 close to the best of the 100 committed random starts
     (0.29033158), not the retired seed-42 local minimum (0.30065329).

Read-only: no artifact is written.
Run: python3 code/mds_classical_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.manifold import MDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from experiment_C1 import classical_mds_init, jac_dist, load  # noqa: E402
from mds_multistart_check import stress1  # noqa: E402
from phase4_freeze import verify_baseline  # noqa: E402

BEST_100_STRESS = 0.29033158
RETIRED_SEED42_STRESS = 0.30065329


def fit(D: np.ndarray) -> np.ndarray:
    return MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        **classical_mds_init(),
        n_init=1,
        normalized_stress="auto",
    ).fit_transform(D)


def main() -> int:
    verify_baseline()
    X, _rho, _paper, *_ = load()
    D = jac_dist(X)
    a = fit(D)
    b = fit(D)
    identical = np.array_equal(a, b)
    s = stress1(D, a)
    rel = abs(s - BEST_100_STRESS) / BEST_100_STRESS * 100.0
    print(f"[mds-classical] two fits bit-identical: {identical}")
    print(f"[mds-classical] stress-1: {s:.8f}")
    print(f"[mds-classical] best of 100 random starts: {BEST_100_STRESS:.8f}")
    print(f"[mds-classical] relative gap: {rel:.3f}%")
    print(f"[mds-classical] retired seed-42 stress: {RETIRED_SEED42_STRESS:.8f}")
    ok = identical and rel <= 0.5
    print("[mds-classical] PASS" if ok else "[mds-classical] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
