#!/usr/bin/env python3
"""make_fig7_replay.py — rebuild the validation figure from stored CSVs.

Calls the (print-sized) fig7() in experiment_S_validation.py using the frozen
simulation outputs, so the figure can be re-rendered without re-running S1/S2.

Run: python3 code/make_fig7_replay.py
"""
import csv
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117/D132)
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import experiment_S_validation as S  # noqa: E402


def load(name):
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    cal = load("sim_calibration.csv")
    rand = load("sim_random_families.csv")
    S.fig7(cal, rand, 0.9634, os.path.join(ROOT, "figures", "fig7_simulation.png"))
    print("[fig7-replay] wrote figures/fig7_simulation.png/pdf (print size)")


if __name__ == "__main__":
    main()
