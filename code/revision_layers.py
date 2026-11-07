#!/usr/bin/env python3
"""revision_layers.py - WP1 canonical analysis layers.

This module is the single source of truth for the revision's primary analysis
set. It keeps the raw extraction files unchanged and applies four explicit
rules at load time:

  1. Standard/regulatory rows (``scenario_type == "standard"``) are excluded
     from the primary set. They are context anchors, not observations.
  2. Ranged F1 cells use the midpoint of ``value`` and ``value_max`` in the
     primary set. Inverted ranges are flagged and fall back to the lower bound.
  3. B154 salinity is reported as EC µS/cm but described in the source as
     ~2 ppt. The primary set uses the reported approximate value (2.0 ppt).
     The exclusion and raw variants are available as sensitivities.
  4. F1 QA bounds flag implausible values and unit-ambiguous salinity cells.
     The primary set excludes flagged cells; the flag-only and QA-off variants
     are available as sensitivities.

The legacy behaviour (include standard rows, lower-bound ranges, raw B154,
no QA) is available by passing the legacy flags explicitly. This module does
not write files.

Run a summary:
    python3 code/revision_layers.py
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESENCE_PATH = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")
VALUES_PATH = os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv")

F1_PARAMS = [
    "DO (mg/L)",
    "pH",
    "Salinity (ppt/\u2030)",
    "Temperature (\u00b0C)",
]

# Broad plausibility bounds for the F1 physical parameters. These are QA
# screens, not ecological thresholds. They are deliberately wide.
F1_BOUNDS = {
    "DO (mg/L)": (0.0, 20.0),
    "pH": (0.0, 14.0),
    "Salinity (ppt/\u2030)": (0.0, 70.0),
    "Temperature (\u00b0C)": (-5.0, 45.0),
}

B154_BATCH = "B154"
B154_SALINITY_PPT = 2.0


@dataclass
class Layers:
    """Container for the coverage and value layers."""

    X: np.ndarray
    rho: np.ndarray
    paper: np.ndarray
    params: list[str]
    Xv: np.ndarray
    keysv: list[tuple[str, str]]
    rho_v: np.ndarray
    flags: list[dict] = field(default_factory=list)
    include_standard: bool = False
    b154_rule: str = "source_override"
    range_rule: str = "midpoint"
    qa_action: str = "exclude"
    n_presence_rows: int = 0
    n_presence_studies: int = 0
    n_standard_rows: int = 0
    n_value_rows: int = 0
    n_value_studies: int = 0
    n_value_standard_rows: int = 0
    n_range_cells: int = 0
    n_midpoint_cells: int = 0
    n_inverted_ranges: int = 0
    n_b154_override_cells: int = 0
    n_b154_excluded_cells: int = 0
    n_qa_flagged_cells: int = 0
    n_qa_dropped_cells: int = 0

    @property
    def summary(self) -> dict:
        return {
            "include_standard": self.include_standard,
            "b154_rule": self.b154_rule,
            "range_rule": self.range_rule,
            "qa_action": self.qa_action,
            "n_presence_rows": self.n_presence_rows,
            "n_presence_studies": self.n_presence_studies,
            "n_standard_rows": self.n_standard_rows,
            "n_value_rows": self.n_value_rows,
            "n_value_studies": self.n_value_studies,
            "n_value_standard_rows": self.n_value_standard_rows,
            "n_range_cells": self.n_range_cells,
            "n_midpoint_cells": self.n_midpoint_cells,
            "n_inverted_ranges": self.n_inverted_ranges,
            "n_b154_override_cells": self.n_b154_override_cells,
            "n_b154_excluded_cells": self.n_b154_excluded_cells,
            "n_qa_flagged_cells": self.n_qa_flagged_cells,
            "n_qa_dropped_cells": self.n_qa_dropped_cells,
        }


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().upper() == "TRUE"


def load_layers(
    include_standard: bool = False,
    b154_rule: str = "source_override",
    range_rule: str = "midpoint",
    qa_action: str = "exclude",
) -> Layers:
    """Load the coverage and value layers under the requested revision rules.

    Parameters
    ----------
    include_standard:
        If False (primary), exclude ``scenario_type == "standard"`` rows.
    b154_rule:
        ``"source_override"`` (primary; use the reported ~2 ppt),
        ``"exclude"`` (drop B154 salinity cells), or ``"raw"`` (legacy).
    range_rule:
        ``"midpoint"`` (primary) or ``"lower"`` (legacy/extraction value).
    qa_action:
        ``"exclude"`` (primary), ``"keep"`` (flag but keep), or ``"off"``.
    """
    if b154_rule not in {"source_override", "exclude", "raw"}:
        raise ValueError(f"unknown b154_rule: {b154_rule}")
    if range_rule not in {"midpoint", "lower"}:
        raise ValueError(f"unknown range_rule: {range_rule}")
    if qa_action not in {"exclude", "keep", "off"}:
        raise ValueError(f"unknown qa_action: {qa_action}")

    presence = _read_csv(PRESENCE_PATH)
    if not presence:
        raise RuntimeError("empty presence layer")
    header = list(presence[0].keys())
    params = header[7:51]
    if len(params) != 44:
        raise RuntimeError(f"expected 44 parameters, found {len(params)}")

    X: list[list[float]] = []
    paper: list[str] = []
    ordinal_to_data: dict[str, int] = {}
    standard_keys: set[tuple[str, str]] = set()
    own = ""
    ordinal = 0
    n_standard_rows = 0
    for row in presence:
        ordinal += 1
        batch = (row.get("batch_id") or "").strip()
        if batch:
            own = batch
        if row.get("row_kind") != "data":
            continue
        key = (own, str(ordinal))
        if (row.get("scenario_type") or "").strip() == "standard":
            n_standard_rows += 1
            standard_keys.add(key)
        if not include_standard and key in standard_keys:
            continue
        X.append([1.0 if (row.get(p) or "").strip() in ("Y", "NE") else 0.0 for p in params])
        paper.append(own)
        ordinal_to_data[key[1]] = len(X) - 1

    X_arr = np.asarray(X, float)
    rho = X_arr.mean(axis=1)
    paper_arr = np.asarray(paper)

    cells: dict[tuple[str, str], dict[str, float]] = {}
    flags: list[dict] = []
    n_range_cells = 0
    n_midpoint_cells = 0
    n_inverted_ranges = 0
    n_b154_override_cells = 0
    n_b154_excluded_cells = 0
    n_qa_flagged_cells = 0
    n_qa_dropped_cells = 0

    for row in _read_csv(VALUES_PATH):
        if not _is_true(row.get("is_real_value")):
            continue
        param = (row.get("param") or "").strip()
        if param not in F1_PARAMS:
            continue
        batch = (row.get("batch_id") or "").strip()
        row_id = str(row.get("row_id") or "").strip()
        if not batch or not row_id:
            continue
        key = (batch, row_id)
        if not include_standard and key in standard_keys:
            continue
        value = _as_float(row.get("value"))
        if value is None:
            continue
        value_max = _as_float(row.get("value_max"))
        unit = (row.get("unit_published") or "").strip()
        info: list[str] = []
        drop = False

        if value_max is not None:
            n_range_cells += 1
            if range_rule == "midpoint":
                if value_max >= value:
                    value = (value + value_max) / 2.0
                    n_midpoint_cells += 1
                else:
                    info.append("inverted_range")
                    n_inverted_ranges += 1

        b154_dropped = False
        if batch == B154_BATCH and param == F1_PARAMS[2]:
            if b154_rule == "source_override":
                value = B154_SALINITY_PPT
                info.append("b154_source_override")
                n_b154_override_cells += 1
            elif b154_rule == "exclude":
                info.append("b154_excluded")
                n_b154_excluded_cells += 1
                drop = True
                b154_dropped = True
            else:
                info.append("b154_raw")

        if qa_action != "off" and not b154_dropped:
            qa_hit = False
            lo, hi = F1_BOUNDS[param]
            if not (lo <= value <= hi):
                info.append("qa_bounds")
                qa_hit = True
                drop = True
            if param == F1_PARAMS[2] and not unit and value > 40.0:
                info.append("qa_unit_ambiguous")
                qa_hit = True
                drop = True
            if qa_hit:
                n_qa_flagged_cells += 1

        if info:
            flags.append(
                dict(
                    batch=batch,
                    row_id=row_id,
                    param=param,
                    value=value,
                    flags=";".join(info),
                    dropped=bool(drop and qa_action == "exclude"),
                )
            )
        if drop and qa_action == "exclude":
            if any(item.startswith("qa_") for item in info):
                n_qa_dropped_cells += 1
            continue
        cells.setdefault(key, {})[param] = value

    keysv: list[tuple[str, str]] = []
    Xv: list[list[float]] = []
    rho_v: list[float] = []
    n_value_standard_rows = 0
    for key, values in cells.items():
        if not all(p in values for p in F1_PARAMS):
            continue
        if key in standard_keys:
            n_value_standard_rows += 1
        keysv.append(key)
        Xv.append([values[p] for p in F1_PARAMS])
        idx = ordinal_to_data.get(key[1])
        rho_v.append(float(rho[idx]) if idx is not None else np.nan)

    return Layers(
        X=X_arr,
        rho=rho,
        paper=paper_arr,
        params=params,
        Xv=np.asarray(Xv, float),
        keysv=keysv,
        rho_v=np.asarray(rho_v, float),
        flags=flags,
        include_standard=include_standard,
        b154_rule=b154_rule,
        range_rule=range_rule,
        qa_action=qa_action,
        n_presence_rows=len(X_arr),
        n_presence_studies=len(set(paper_arr)),
        n_standard_rows=n_standard_rows,
        n_value_rows=len(keysv),
        n_value_studies=len({k[0] for k in keysv}),
        n_value_standard_rows=n_value_standard_rows,
        n_range_cells=n_range_cells,
        n_midpoint_cells=n_midpoint_cells,
        n_inverted_ranges=n_inverted_ranges,
        n_b154_override_cells=n_b154_override_cells,
        n_b154_excluded_cells=n_b154_excluded_cells,
        n_qa_flagged_cells=n_qa_flagged_cells,
        n_qa_dropped_cells=n_qa_dropped_cells,
    )


def value_pca(layers: Layers, n_components: int = 2):
    """Fit the standardized value PCA used by the manuscript."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if len(layers.Xv) < n_components:
        raise ValueError("not enough value-complete rows for PCA")
    Z = StandardScaler().fit_transform(layers.Xv)
    pca = PCA(n_components=n_components).fit(Z)
    return pca, Z


def b154_diagnostics(layers: Layers, pca) -> list[dict]:
    """Return B154 value-PCA coordinates and z-scores under the loaded rules."""
    out: list[dict] = []
    if not layers.keysv:
        return out
    Zstd = (layers.Xv - layers.Xv.mean(axis=0)) / layers.Xv.std(axis=0, ddof=0)
    scores = pca.transform(Zstd)
    for i, key in enumerate(layers.keysv):
        if key[0] != B154_BATCH:
            continue
        out.append(
            dict(
                row_id=key[1],
                do=float(layers.Xv[i, 0]),
                ph=float(layers.Xv[i, 1]),
                salinity=float(layers.Xv[i, 2]),
                temperature=float(layers.Xv[i, 3]),
                z_salinity=float(Zstd[i, 2]),
                pc1=float(scores[i, 0]),
                pc2=float(scores[i, 1]),
            )
        )
    return out


def value_lopo_drifts(layers: Layers) -> list[dict]:
    """Per-study LOPO median drift and percentile for the value PCA.

    Uses the same estimator as the manuscript: standardized value PCA, Procrustes
    alignment on common rows, median row drift per event. Percentile is
    ``100 * rank / n`` with average ranks for ties.
    """
    from scipy.spatial import procrustes
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xv = layers.Xv
    paper = np.asarray([k[0] for k in layers.keysv])
    S_full = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Xv))
    rows: list[dict] = []
    for study in sorted(set(paper)):
        keep = paper != study
        S_red = PCA(n_components=2).fit_transform(
            StandardScaler().fit_transform(Xv[keep])
        )
        a, b, _ = procrustes(S_full[keep], S_red)
        drift = np.linalg.norm(b - a, axis=1)
        rows.append(dict(
            paper=str(study),
            median_drift=float(np.median(drift)),
            n_rows=int(keep.sum()),
            study_rows=int((paper == study).sum()),
        ))
    values = np.asarray([r["median_drift"] for r in rows], float)
    ranks = np.argsort(np.argsort(values)) + 1
    n = len(values)
    for row, rank in zip(rows, ranks):
        row["percentile"] = float(100.0 * rank / n)
    return sorted(rows, key=lambda r: -r["median_drift"])


def legacy_layers() -> Layers:
    """Reproduce the pre-WP1 layer: standard rows, lower ranges, raw B154, no QA."""
    return load_layers(
        include_standard=True,
        b154_rule="raw",
        range_rule="lower",
        qa_action="off",
    )


def add_layer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the canonical layer flags to an argparse parser.

    ``--layer primary`` (default) selects the WP1 730/410 set;
    ``--layer legacy`` reproduces the pre-WP1 764/421 set. Individual flags override
    the layer defaults.
    """
    parser.add_argument("--layer", choices=["legacy", "primary"], default="primary",
                        help="legacy 764/421 (default) or WP1 primary 730/410")
    parser.add_argument("--include-standard", dest="include_standard",
                        action="store_true", default=None,
                        help="include scenario_type == 'standard' rows")
    parser.add_argument("--exclude-standard", dest="include_standard",
                        action="store_false",
                        help="exclude scenario_type == 'standard' rows")
    parser.add_argument("--b154-rule", choices=["source_override", "exclude", "raw"],
                        default=None)
    parser.add_argument("--range-rule", choices=["midpoint", "lower"], default=None)
    parser.add_argument("--qa-action", choices=["exclude", "keep", "off"],
                        default=None)
    return parser


def layer_kwargs(args: argparse.Namespace) -> dict:
    """Resolve --layer plus explicit overrides into load_layers() keyword args."""
    if args.layer == "primary":
        kwargs = dict(include_standard=False, b154_rule="source_override",
                      range_rule="midpoint", qa_action="exclude")
    else:
        kwargs = dict(include_standard=True, b154_rule="raw",
                      range_rule="lower", qa_action="off")
    for key in ("include_standard", "b154_rule", "range_rule", "qa_action"):
        value = getattr(args, key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


def main() -> None:
    for label, layers in (
        ("legacy", legacy_layers()),
        ("primary", load_layers()),
    ):
        pca, _ = value_pca(layers)
        ev = pca.explained_variance_ratio_
        print(f"{label}: {layers.summary}")
        print(f"  value PCA 2-D variance: {ev.sum():.4f} ({ev[0]:.4f}, {ev[1]:.4f})")
        for row in b154_diagnostics(layers, pca):
            print(f"  B154 {row['row_id']}: salinity={row['salinity']:.2f} z={row['z_salinity']:.2f} pc2={row['pc2']:.2f}")


if __name__ == "__main__":
    main()
