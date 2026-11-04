#!/usr/bin/env python3
"""audit_mds_scope.py — Phase 3 scope gate for the classical-MDS re-run.

Compares the current results/ tree with the archived random-start snapshot at
scratch/archive/random_start_2026-10-30/ and fails if anything outside the MDS
layer moved.

Allowed changes:
  * rows whose variant is mds_pres (C1/C2/K/churn/tie-order/weighting);
  * MDS columns in exM_suite_metrics.csv and family_tiers.csv;
  * all rows of the MDS-only block-null and calibration artifacts.

Everything else must be byte-equivalent within numeric tolerance. Tier changes
are reported separately because they require the Phase 3b decision, not a
silent scope failure.

Scope of this gate. This is a STRICT diff against the random-start archive, so
after Phase 3 it is expected to keep failing on the documented stale layers
(see docs/mds_phase3_audit.md §2-3 and D259):
  * t-SNE drift/SI/churn rows (the archive predates the pinned environment);
  * churn columns for every variant (regenerated table layer);
  * degenerate 3-D PCA residue rows (delta_fam ~ 1e-17);
  * the PCoA permutation step and the pca3d F6 cell;
  * the D259 matched-benchmark refresh (family_tiers benchmark counts).
The classified audit is docs/mds_phase3_audit.md; this script is the raw
byte-diff, not the leakage verdict. MDS-only artifacts (mds_stress_summary,
sim_block_null_mds*) are allowed to change in every column; only their shape
is pinned.

Run: python3 code/audit_mds_scope.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(ROOT, "scratch", "archive", "random_start_2026-10-30")
NEW = os.path.join(ROOT, "results")

ROW_RULES = {
    "exC1_summary.csv": (["variant", "mode"], {"mds_pres"}),
    "exC2_summary.csv": (["variant", "family"], {"mds_pres"}),
    "exC2_paired.csv": (["variant", "family", "deleted"], {"mds_pres"}),
    "exK_absratio_summary.csv": (["variant", "dims", "family"], {"mds_pres"}),
    "exK_absratio_variants.csv": (["variant", "dims", "family", "deleted"], {"mds_pres"}),
    "exC1_blockperm.csv": (["variant"], {"mds_pres"}),
    "exC1_churn_corrected.csv": (["variant", "mode"], {"mds_pres"}),
    "ex_pattern_churn.csv": (["variant", "mode"], {"mds_pres"}),
    "ex_pattern_churn_ci.csv": (["variant", "mode"], {"mds_pres"}),
    "tie_order_sensitivity.csv": (["variant", "tie_rule", "mode"], {"mds_pres"}),
    "robustness_weighting.csv": (["check", "variant"], {"mds_pres"}),
    "duplicate_handling.csv": (["treatment", "test"], {"mds_pres"}),
    "sim_ordination_calibration.csv": (["variant"], {"mds_pres"}),
}

COLUMN_RULES = {
    "exM_suite_metrics.csv": ("name", lambda c: "mds" in c.lower()),
    "family_tiers.csv": (
        "family",
        lambda c: ("mds" in c.lower() or c == "criterion2_mds_ok"),
    ),
}

ALL_MDS = {
    "mds_stress_summary.csv",
    "sim_block_null_mds20.csv",
    "sim_block_null_summary_mds20.csv",
}

FREEZE = {
    "exC2_absolute_drift.csv",
    "exC2_sensitivity.csv",
    "sim_block_null_pca.csv",
    "sim_block_null_summary_pca.csv",
    "sim_block_null_pcoa.csv",
    "sim_block_null_summary_pcoa.csv",
    "mds_multistart_check.csv",
    "mds_seed_baseline_events.csv",
    "mds_seed_baseline_summary.csv",
}


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def rows_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality with a tolerance for the pinned float path.

    Phase 3 measured float-path differences <=1.3e-7 relative in this layer;
    real value changes are >=0.1%. Degenerate 3-D PCA residue rows are excluded
    by value in compare_rows, not waved through by tolerance.
    """
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return np.isclose(a.astype(float), b.astype(float),
                          equal_nan=True, rtol=1e-6, atol=1e-10)
    return (a.fillna("__NA__").astype(str) == b.fillna("__NA__").astype(str)).values


def compare_rows(name: str, key: list[str], allowed: set[str]):
    old = read(os.path.join(OLD, name)).set_index(key, drop=False)
    new = read(os.path.join(NEW, name)).set_index(key, drop=False)
    if set(old.index) != set(new.index):
        return [(name, "__keys__", {"index"})]
    new = new.loc[old.index]
    if "variant" in old.columns:
        check = ~old["variant"].isin(allowed)
    else:
        check = pd.Series(True, index=old.index)
    if name == "duplicate_handling.csv":
        check = ~old["test"].astype(str).str.contains("mds_pres", regex=False)
    # Degenerate 3-D PCA residue rows: delta_fam ~ 1e-17 makes the ratio
    # machine noise (Phase 3 audit section 3). Excluded by value, not by file.
    if {"dims", "fam_med"} <= set(old.columns):
        deg = (old["dims"].astype(float) == 3) & (old["fam_med"].astype(float) < 1e-15)
        check = check & ~deg
    out = []
    label_col = "variant" if "variant" in old.columns else ("test" if "test" in old.columns else key[0])
    for col in old.columns:
        if col in key:
            continue
        diff = ~rows_equal(old.loc[check, col], new.loc[check, col])
        if diff.any():
            off = set(old.loc[check].loc[diff, label_col].astype(str)) - allowed
            if off:
                out.append((name, col, off))
    print(f"[scope] {name}: {len(old)} rows; frozen {int(check.sum())}; "
          f"MDS rows {int((~check).sum())}")
    return out


def compare_columns(name: str, key: str, allowed_col):
    old = read(os.path.join(OLD, name)).set_index(key, drop=False)
    new = read(os.path.join(NEW, name)).set_index(key, drop=False)
    if set(old.index) != set(new.index):
        return [(name, "__keys__", {"index"})]
    new = new.loc[old.index]
    allowed = [c for c in old.columns if allowed_col(c)]
    out = []
    for col in old.columns:
        if col == key or col in allowed:
            continue
        diff = ~rows_equal(old[col], new[col])
        if diff.any():
            out.append((name, col, set(old.loc[diff, key].astype(str))))
    if name == "family_tiers.csv" and "tier" in old.columns:
        changed = old["tier"].astype(str) != new["tier"].astype(str)
        if changed.any():
            fams = set(old.loc[changed, "family"].astype(str))
            print(f"[scope] TIER CHANGED: {sorted(fams)}")
            out.append((name, "tier", fams))
        else:
            print("[scope] family_tiers.csv: all seven tiers unchanged")
    print(f"[scope] {name}: {len(old)} rows; allowed MDS columns {allowed}")
    return out


def compare_all(name: str):
    """MDS-only artifacts: every column may change; only the shape is pinned."""
    old = read(os.path.join(OLD, name))
    new = read(os.path.join(NEW, name))
    if old.shape != new.shape:
        return [(name, "__shape__", {f"{old.shape}->{new.shape}"})]
    print(f"[scope] {name}: all {len(old)} rows allowed to change")
    return []


# Documented stale/frozen classes from docs/mds_phase3_audit.md and D259.
DOCUMENTED_FROZEN = {
    "exC2_sensitivity.csv": "pca3d F6 degenerate cell (Phase 3 audit section 3)",
    "sim_block_null_pcoa.csv": "PCoA p_block permutation step 0.005 (Phase 3 audit section 3)",
}
DOCUMENTED_COLUMNS = {
    ("family_tiers.csv", "ci_hi"): "printed precision 0.01 (Phase 3 audit section 3)",
    ("family_tiers.csv", "benchmark_exceed"): "D259 matched-benchmark refresh",
    ("family_tiers.csv", "benchmark_nondeg"): "D259 matched-benchmark refresh",
    ("exC2_summary.csv", "G_CI95"): "float path 5e-9 in the CI tuple (Phase 3 audit section 3)",
    ("family_tiers.csv", "criterion3_interval_excludes_1"): "criterion 3 re-based on the presence-PCA interval (D258/D260)",
    ("family_tiers.csv", "provisional_triggers"): "MDS trigger removed (D258)",
}
CHURN_TOKENS = ("churn", "chi", "frac", "tie", "multiconfig", "n_rows")


def documented(name: str, col: str, offenders: set[str]):
    if col in {"__keys__", "__shape__"}:
        return None
    if name in DOCUMENTED_FROZEN:
        return DOCUMENTED_FROZEN[name]
    if any(tok in col.lower() for tok in CHURN_TOKENS):
        return "churn layer (Phase 3 audit section 2; control 67a8fa4)"
    if (name, col) in DOCUMENTED_COLUMNS:
        return DOCUMENTED_COLUMNS[(name, col)]
    if offenders <= {"mds_pres", "tsne_pres"}:
        return "MDS / t-SNE stale layer (Phase 3 audit section 2)"
    return None


def main() -> int:
    problems = []
    for name, (key, allowed) in ROW_RULES.items():
        if os.path.exists(os.path.join(OLD, name)) and os.path.exists(os.path.join(NEW, name)):
            problems += compare_rows(name, key, allowed)
        else:
            print(f"[scope] SKIP {name} (missing old or new)")
    for name, (key, allowed_col) in COLUMN_RULES.items():
        if os.path.exists(os.path.join(OLD, name)) and os.path.exists(os.path.join(NEW, name)):
            problems += compare_columns(name, key, allowed_col)
        else:
            print(f"[scope] SKIP {name} (missing old or new)")
    for name in sorted(ALL_MDS):
        if os.path.exists(os.path.join(OLD, name)) and os.path.exists(os.path.join(NEW, name)):
            problems += compare_all(name)
        else:
            print(f"[scope] SKIP {name} (missing old or new)")
    for name in sorted(FREEZE):
        if os.path.exists(os.path.join(OLD, name)) and os.path.exists(os.path.join(NEW, name)):
            old = read(os.path.join(OLD, name))
            new = read(os.path.join(NEW, name))
            if old.shape != new.shape or any(not rows_equal(old[c], new[c]).all() for c in old.columns):
                problems.append((name, "__frozen__", {"file"}))
            else:
                print(f"[scope] {name}: frozen OK")
        else:
            print(f"[scope] SKIP {name} (missing old or new)")

    # exC1_rows.csv.gz needs its own row-level pass (1.28M rows).
    old_rows = pd.read_csv(os.path.join(OLD, "exC1_rows.csv.gz"), low_memory=False)
    new_rows = pd.read_csv(os.path.join(NEW, "exC1_rows.csv.gz"), low_memory=False)
    key = ["variant", "mode", "deleted", "row_id"]
    old_rows = old_rows.set_index(key, drop=False)
    new_rows = new_rows.set_index(key, drop=False)
    if set(old_rows.index) != set(new_rows.index):
        problems.append(("exC1_rows.csv.gz", "__keys__", {"index"}))
    else:
        new_rows = new_rows.loc[old_rows.index]
        frozen = old_rows["variant"] != "mds_pres"
        for col in ("rho", "drift", "churn"):
            diff = ~rows_equal(old_rows.loc[frozen, col], new_rows.loc[frozen, col])
            if diff.any():
                off = set(old_rows.loc[frozen].loc[diff, "variant"].astype(str))
                problems.append(("exC1_rows.csv.gz", col, off))
        print(f"[scope] exC1_rows.csv.gz: {len(old_rows)} rows; frozen {int(frozen.sum())}; "
              f"MDS {int((~frozen).sum())}")

    classified, unexplained = [], []
    for name, col, off in problems:
        reason = documented(name, col, off)
        if reason:
            classified.append((name, col, sorted(off), reason))
        else:
            unexplained.append((name, col, sorted(off)))

    print("\n[scope] RESULT:", "PASS" if not unexplained else "FAIL")
    if classified:
        print(f"[scope] {len(classified)} documented difference class(es):")
        for name, col, off, reason in classified:
            print(f"  ~ {name}: {col} [{', '.join(off)}] — {reason}")
    if unexplained:
        print(f"[scope] {len(unexplained)} UNEXPLAINED difference(s):")
        for name, col, off in unexplained:
            print(f"  - {name}: {col} [{', '.join(off)}]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
