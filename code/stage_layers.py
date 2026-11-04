#!/usr/bin/env python3
"""stage_layers.py — place the released layers where the scripts expect them.

The analysis scripts read the frozen layers from fixed paths
(`docs/extraction/stageA/matrix_presence_v2.csv`, `docs/extraction/stageB/
matrix_values_v2.csv`, `docs/family_map.csv`) and two builders read the
released data record at `release/p1_family_pca_data_v1.0/`. In the authors'
working tree both locations already hold the same bytes, so this script does
nothing. In a fresh unpack of the two Zenodo records it copies the data
record's files into those locations, so every script runs unmodified.

Sources are looked for in this order (v1.1 is documentation-only; its data
files are byte-identical to v1.0):
  1. release/p1_family_pca_data_v1.1/            (working tree)
  2. release/p1_family_pca_data_v1.0/            (working tree, legacy)
  3. ../p1_family_pca_data_v1.1/                  (sibling unpack)
  4. ../p1_family_pca_data_v1.0/                  (sibling unpack, legacy)
  5. p1_family_pca_data_v1.1/                    (unpacked into the code directory)
  6. p1_family_pca_data_v1.0/                    (unpacked into the code directory)

Idempotent: a destination is only written when it is missing or differs.

Run: python3 code/stage_layers.py
"""
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAYERS = [
    ("presence.csv", "docs/extraction/stageA/matrix_presence_v2.csv"),
    ("values.csv", "docs/extraction/stageB/matrix_values_v2.csv"),
    ("family_map.csv", "docs/family_map.csv"),
    # the two committed result layers that mirror released files
    ("verdicts.csv", "results/verdict_derivation_log.csv"),
    ("conversions.csv", "results/unit_conversion_log.csv"),
]


def find_record():
    for rel in ("release/p1_family_pca_data_v1.1",
                "release/p1_family_pca_data_v1.0",
                "../p1_family_pca_data_v1.1",
                "../p1_family_pca_data_v1.0",
                "p1_family_pca_data_v1.1",
                "p1_family_pca_data_v1.0"):
        p = os.path.normpath(os.path.join(ROOT, rel))
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "presence.csv")):
            return p
    return None


def copy_if_needed(src, dst):
    if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
        return "same"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def main() -> int:
    # the output directories the builders write into (absent in a fresh unpack)
    for d in ("results", "figures", "manuscript/tables"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    rec = find_record()
    if rec is None:
        print("[stage-layers] no data record found next to the code; nothing to stage "
              "(unpack p1_family_pca_data_v1.1.zip beside this directory)")
        return 0
    print(f"[stage-layers] data record: {rec}")
    for name, rel in LAYERS:
        src = os.path.join(rec, name)
        if not os.path.exists(src):
            print(f"  skip {name}: not in the record")
            continue
        print(f"  {copy_if_needed(src, os.path.join(ROOT, rel)):7s} {rel}")
    # the two builders that read the record from release/
    rel_dst = os.path.join(ROOT, "release", "p1_family_pca_data_v1.0")
    if os.path.normpath(rec) != os.path.normpath(rel_dst):
        for name in sorted(os.listdir(rec)):
            src = os.path.join(rec, name)
            if os.path.isfile(src):
                copy_if_needed(src, os.path.join(rel_dst, name))
        print(f"  mirrored the record into release/p1_family_pca_data_v1.0/ "
              f"(the analysis builders read this path; v1.1 data files are byte-identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
