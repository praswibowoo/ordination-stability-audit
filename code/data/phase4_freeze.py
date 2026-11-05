#!/usr/bin/env python3
"""phase4_baseline.py — P1-4.5 seed/env freeze verification (Analysis_Design pre-reg §2–3).

Produces results/phase4_baseline.json:
  - runtime versions (python/numpy/scipy/sklearn/pandas/matplotlib)
  - seed + input sha256 pins (full 64-hex) for the three core matrices
  - row/cell/status counts = Phase-4 baselines (must match D95/D96 numbers)
Re-run any time: `python3 code/data/phase4_freeze.py` (atomic, idempotent).
C1+ experiment scripts MUST call verify_baseline() before running (hard stop on mismatch).
"""
from __future__ import annotations
import hashlib, json, os, platform, sys
from datetime import date

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
CFG = os.path.join(ROOT, "config.yaml")
OUT = os.path.join(ROOT, "results/phase4_baseline.json")

INPUTS = [
    ("matrix_presence", "docs/extraction/stageA/matrix_presence_v2.csv"),
    ("matrix_values", "docs/extraction/stageB/matrix_values_v2.csv"),
    ("family_map", "docs/family_map.csv"),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    cfg = {}
    with open(CFG, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line and ":" in line:
                k, _, v = line.partition(":")
                try:
                    cfg[k.strip()] = int(v.strip())
                except ValueError:
                    cfg[k.strip()] = v.strip().strip('"')

    assert cfg.get("seed") == 42, cfg.get("seed")
    pins = {}
    for name, rel in INPUTS:
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p), p
        pins[name] = {"path": rel, "sha256": sha(p)}
    lock = {
        "generated": __import__("datetime").date.today().isoformat(),
        "task": "P1-4.5 seed/env freeze",
        "seed": 42,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": __import__("numpy").__version__,
        "pandas": __import__("pandas").__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": __import__("sklearn").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "data_version": cfg["data_version"],
        "inputs": pins,
        "baselines": {
            "n_papers": 138, "scope_rows": 593, "value_cells": 3216,
            # value-layer accounting (2026-11-01 audit): 3,216 Stage-B template
            # cells; 2,705 value records = 2,658 numeric + 7 censored + 40 blank;
            # 511 template cells have no record; documented blanks = 40 + 511.
            "value_records": 2705, "filled_cells": 2658,
            "censored_values": 7, "blank_records": 40,
            "documented_blanks": 551, "family_complete_cells": 3305,
            "presence_blank_data": 26329,
            "conversions": 290, "qa_flags": 25,
            "statuses_labeled": 606, "derived": 338, "paper_labeled": 268,
            "low_confidence_rows_in_log": 110,
            "verdict_dist": {"Favorable": 25, "Stable": 18, "Mild stress": 87,
                             "Stressful": 134, "Harmful": 57, "Reference standard": 17},
            "families": {"F1_physical": 519, "F2_metals": 14, "F3_nutrients": 85,
                         "F4_organic": 23, "F1_physicala": 26, "F1_clarity": 29,
                         "F6_aeration": 159},
            "gates": {"F3_papers": 38, "F4_papers": 34, "lethal_rows": 0},
        },
    }
    tmp = OUT + ".tmp"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
    os.replace(tmp, OUT)
    print(f"env lock frozen → {OUT}")
    print("python", lock["python"], "| numpy", lock["numpy"], "| scipy", lock["scipy"],
          "| sklearn", lock["sklearn"], "| pandas", lock["pandas"], "| matplotlib", lock["matplotlib"])


def verify_baseline():
    """Hard pre-run gate for every Phase-4 experiment (Analysis_Design §3)."""
    with open(OUT, encoding="utf-8") as f:
        lock = json.load(f)
    for name, rel in INPUTS:
        p = os.path.join(ROOT, rel)
        h = sha(p)
        assert h == lock["inputs"][name]["sha256"], (
            f"baseline mismatch {name}: disk {h[:16]} != frozen {lock['inputs'][name]['sha256'][:16]} "
            f"— data changed since the P1-4.5 freeze; re-run phase4_freeze.py consciously")
    return lock


if __name__ == "__main__":
    main()
