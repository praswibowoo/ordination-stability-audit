#!/usr/bin/env python3
"""Fig D — F1 pair-distance structure — rebuilt from stored D results (D115).

DECOUPLED figure factory: derives ONLY figD from results/exD_tests.csv via
experiment_D's own figD function. No stat recompute, no stat files written.

Run: python3 code/make_figD.py
"""
from __future__ import annotations
import csv
import importlib.util
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117/D132)
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "code", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    d = load_module("experiment_D_concordance", "experiment_D_concordance.py")
    res = {}
    with open(os.path.join(ROOT, "results", "exD_tests.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if k == "layer":
                    continue
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
            res[r["layer"]] = r
    assert {"presence", "value"} <= set(res), set(res)
    d.figD(res["presence"], res["value"],
           os.path.join(ROOT, "figures", "figD_distances.png"))
    print("[make_figD] figD rebuilt from stored exD_tests.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
