#!/usr/bin/env python3
"""mds_seed_baseline.py — estimand-defined seed baseline for the Jaccard MDS arm.

Phase 1 of the MDS classical-init correction. The reviewer's M-D.3 request is a
seed-only baseline at the LOPO configuration, not a repetition of the deletion
protocol. This script therefore computes four distinct quantities per event:

  deletion    full(s)  vs reduced(s)   deletion drift at seed s
  seed_only   reduced(42) vs reduced(s) same reduced data, different seed
  hybrid      full(42) vs reduced(s)   cross-seed deletion (mixed-seed harness)
  full_seed   full(42) vs full(s)      full-map seed drift on the kept rows

In the event CSV, `stress_full` is the reference fit's stress and
`stress_reduced` is the comparison fit's stress. For seed_only the reference is
the reduced(42) fit, so `stress_full` is blank; for full_seed the comparison is
a second full fit, so `stress_reduced` is that fit's full-data stress.

For each protocol the summary records both the pooled row-level mean/median and
the mean/median of the 122 event means. Those are different estimands; the
manuscript's exC1_summary.csv `mean_drift` is pooled, while the Phase 1 decision
statistics are event-level.

The random-start protocol is retained here on purpose: this script establishes
the baseline against which the deterministic classical-init re-run is judged.
Phase 2 changes the production ordination; this diagnostic is historical after
that change.

Usage:
  python3 code/mds/mds_seed_baseline.py --seeds 0-9 --protocol all
  python3 code/mds/mds_seed_baseline.py --seeds 0-9 --protocol all --check
  python3 code/mds/mds_seed_baseline.py --seeds 0 --max-events 5   # smoke test

Outputs:
  results/mds_seed_baseline_events.csv
  results/mds_seed_baseline_summary.csv

Run time: ~7.4 min per seed for 122 events on the audit machine; seeds 0-9 with
fit caching is ~80-90 min. The script is standalone and is not wired into
`make all` or `make reproduce`.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
from sklearn.manifold import MDS

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))

from experiment_C1 import (  # noqa: E402
    KNN_K,
    drift_after_procrustes,
    jac_dist,
    knn_churn,
    knn_idx,
    load,
)
from mds_multistart_check import stress1  # noqa: E402

REF_SEED = 42
DEFAULT_SUMMARY = os.path.join(ROOT, "results", "mds_seed_baseline_summary.csv")
DEFAULT_EVENTS = os.path.join(ROOT, "results", "mds_seed_baseline_events.csv")
PROTOCOLS = ("deletion", "seed_only", "hybrid", "full_seed")

# Regression checks. The seed-42 deletion values are pinned to the committed
# exC1_summary.csv. The seed-0 and cross-seed values come from the 2026-10-30
# exploratory probe (scikit-learn 1.7.2); they are checked with a wider
# tolerance because Phase 1 runs under the pinned 1.9.0 environment.
CHECKS = (
    ("deletion", 42, 42, "mean_pooled", 0.0182216, 2e-5),
    ("deletion", 42, 42, "mean_of_event_means", 0.0182312, 2e-5),
    ("deletion", 0, 0, "mean_pooled", 0.0095690, 2e-4),
    ("deletion", 0, 0, "mean_of_event_means", 0.0095830, 2e-4),
    ("seed_only", 42, 0, "mean_of_event_means", 0.0160810, 2e-4),
    ("hybrid", 42, 0, "mean_of_event_means", 0.0217170, 2e-4),
    ("full_seed", 42, 0, "mean_of_event_means", 0.0209020, 2e-4),
)


def parse_seeds(text: str) -> list[int]:
    """Parse '0-9', '0,1,2' or '0-9,42' into a sorted unique list."""
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            a, b = int(lo), int(hi)
            if b < a:
                raise ValueError(f"seed range is reversed: {part}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("no seeds parsed")
    return sorted(out)


def fit_mds(D: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Random-start 2-D metric SMACOF, matching the committed C1 protocol."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            n_init=1,
            init="random",
            normalized_stress="auto",
        )
        emb = model.fit_transform(D)
    return np.asarray(emb, float), float(stress1(D, emb))


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def check_summary(rows: list[dict], strict: bool = False, verbose: bool = True) -> bool:
    """Check the pinned regression values; return True if all present and close."""
    by_key = {(r["protocol"], int(r["full_seed"]), int(r["refit_seed"])): r for r in rows}
    ok = True
    for protocol, fs, rs, col, expected, tol in CHECKS:
        key = (protocol, fs, rs)
        if key not in by_key:
            if verbose:
                print(f"  SKIP  {protocol} full={fs} refit={rs}: row absent")
            continue
        got = float(by_key[key][col])
        use_tol = tol if (strict or protocol == "deletion" and fs == REF_SEED) else max(tol, 2e-4)
        good = abs(got - expected) <= use_tol
        if verbose:
            print(
                f"  {'PASS' if good else 'FAIL'}  {protocol} full={fs} refit={rs} "
                f"{col}: {got:.7f} vs {expected:.7f} (tol {use_tol:g})"
            )
        ok &= good
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9", help="seed list/range, e.g. 0-9 or 0,1,2")
    ap.add_argument("--protocol", default="all", choices=("all",) + PROTOCOLS)
    ap.add_argument("--out", default=DEFAULT_SUMMARY, help="summary CSV path")
    ap.add_argument("--events-out", default=DEFAULT_EVENTS, help="event CSV path")
    ap.add_argument("--max-events", type=int, default=0, help="limit events for a smoke test")
    ap.add_argument("--check", action="store_true", help="check the summary after writing (or check an existing summary)")
    ap.add_argument("--strict", action="store_true", help="apply the 2e-5 probe tolerance to all checks")
    args = ap.parse_args()

    if args.check and os.path.exists(args.out) and not args.max_events:
        with open(args.out, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        print(f"[seed-baseline] checking existing {os.path.relpath(args.out, ROOT)}")
        return 0 if check_summary(rows, strict=args.strict) else 1

    seeds = parse_seeds(args.seeds)
    requested = set(PROTOCOLS) if args.protocol == "all" else {args.protocol}
    all_seeds = sorted(set(seeds) | {REF_SEED})
    cross_seeds = [s for s in seeds if s != REF_SEED]
    need_full = bool(requested & {"deletion", "hybrid", "full_seed"})
    t0 = time.time()

    X, _rho, paper, _params, *_ = load()
    papers = sorted(set(paper))
    if args.max_events:
        papers = papers[: args.max_events]
    D_full = jac_dist(X)

    print(
        f"[seed-baseline] events={len(papers)} seeds={seeds} protocols={sorted(requested)} "
        f"full_fits={need_full} ref={REF_SEED}",
        flush=True,
    )

    full_emb: dict[int, np.ndarray] = {}
    full_stress: dict[int, float] = {}
    full_knn: dict[int, np.ndarray] = {}
    if need_full:
        for s in all_seeds:
            full_emb[s], full_stress[s] = fit_mds(D_full, s)
            full_knn[s] = knn_idx(full_emb[s], KNN_K)
        print(f"[seed-baseline] full fits done ({time.time() - t0:.1f}s)", flush=True)

    event_rows: list[dict] = []
    pooled: dict[tuple[str, int, int], list[np.ndarray]] = {}
    event_stats: dict[tuple[str, int, int], list[tuple[float, float]]] = {}

    def emit_with_pooled(
        protocol: str,
        full_seed: int,
        refit_seed: int,
        event: str,
        n_keep: int,
        S_ref: np.ndarray,
        S_comp: np.ndarray,
        knn_ref: np.ndarray,
        knn_comp: np.ndarray,
        mapped_comp: bool,
        keep_idx: np.ndarray,
        stress_ref: float,
        stress_comp: float,
    ) -> None:
        drift, _ = drift_after_procrustes(S_ref, S_comp)
        comp = keep_idx[knn_comp] if mapped_comp else knn_comp
        ch = knn_churn(knn_ref, comp)
        mean_d, med_d = float(np.mean(drift)), float(np.median(drift))
        mean_c = float(np.mean(ch))
        key = (protocol, full_seed, refit_seed)
        pooled.setdefault(key, []).append(drift.astype(float))
        event_stats.setdefault(key, []).append((mean_d, mean_c))
        event_rows.append(
            dict(
                protocol=protocol,
                full_seed=full_seed,
                refit_seed=refit_seed,
                event=event,
                n_keep=n_keep,
                mean_drift=round(mean_d, 8),
                median_drift=round(med_d, 8),
                mean_churn=round(mean_c, 8),
                churn_frac=round(mean_c / 2.0, 8),
                stress_full="" if not np.isfinite(stress_ref) else round(stress_ref, 8),
                stress_reduced="" if not np.isfinite(stress_comp) else round(stress_comp, 8),
            )
        )

    for j, p in enumerate(papers, 1):
        keep = paper != p
        keep_idx = np.where(keep)[0]
        D_keep = jac_dist(X[keep])
        red_emb: dict[int, np.ndarray] = {}
        red_stress: dict[int, float] = {}
        red_knn: dict[int, np.ndarray] = {}
        for s in all_seeds:
            red_emb[s], red_stress[s] = fit_mds(D_keep, s)
            red_knn[s] = knn_idx(red_emb[s], KNN_K)

        if "deletion" in requested:
            for s in all_seeds:
                emit_with_pooled(
                    "deletion", s, s, p, len(keep_idx),
                    full_emb[s][keep_idx], red_emb[s],
                    full_knn[s][keep_idx], red_knn[s],
                    True, keep_idx, full_stress[s], red_stress[s],
                )

        for s in cross_seeds:
            if "seed_only" in requested:
                emit_with_pooled(
                    "seed_only", REF_SEED, s, p, len(keep_idx),
                    red_emb[REF_SEED], red_emb[s],
                    red_knn[REF_SEED], red_knn[s],
                    False, keep_idx,
                    float("nan"), red_stress[s],
                )
            if "hybrid" in requested:
                emit_with_pooled(
                    "hybrid", REF_SEED, s, p, len(keep_idx),
                    full_emb[REF_SEED][keep_idx], red_emb[s],
                    full_knn[REF_SEED][keep_idx], red_knn[s],
                    True, keep_idx, full_stress[REF_SEED], red_stress[s],
                )
            if "full_seed" in requested:
                emit_with_pooled(
                    "full_seed", REF_SEED, s, p, len(keep_idx),
                    full_emb[REF_SEED][keep_idx], full_emb[s][keep_idx],
                    full_knn[REF_SEED][keep_idx], full_knn[s][keep_idx],
                    False, keep_idx, full_stress[REF_SEED], full_stress[s],
                )

        if j % 10 == 0 or j == len(papers):
            elapsed = time.time() - t0
            rate = elapsed / j
            eta = rate * (len(papers) - j)
            print(
                f"[seed-baseline] event {j}/{len(papers)} elapsed {elapsed/60:.1f} min "
                f"eta {eta/60:.1f} min",
                flush=True,
            )

    # Summary. The deletion(42) event means are the reference distribution for
    # percentile_in_deletion42; this is descriptive, not a null p-value.
    ref_means = [
        r["mean_drift"] for r in event_rows
        if r["protocol"] == "deletion" and int(r["full_seed"]) == REF_SEED
        and int(r["refit_seed"]) == REF_SEED
    ]
    ref_means = np.asarray([float(v) for v in ref_means], float)

    summary_rows: list[dict] = []
    for key, pairs in sorted(event_stats.items()):
        protocol, fs, rs = key
        drift_all = np.concatenate(pooled[key]) if pooled.get(key) else np.asarray([])
        means = np.asarray([x[0] for x in pairs], float)
        churns = np.asarray([x[1] for x in pairs], float)
        mean_event = float(np.mean(means))
        pct = (
            float(100.0 * np.mean(ref_means <= mean_event))
            if ref_means.size and np.isfinite(mean_event) else float("nan")
        )
        summary_rows.append(
            dict(
                protocol=protocol,
                full_seed=fs,
                refit_seed=rs,
                n_events=len(pairs),
                mean_pooled=round(float(np.mean(drift_all)), 8) if drift_all.size else "",
                median_pooled=round(float(np.median(drift_all)), 8) if drift_all.size else "",
                mean_of_event_means=round(mean_event, 8),
                median_of_event_means=round(float(np.median(means)), 8),
                mean_churn=round(float(np.mean(churns)), 8),
                churn_frac=round(float(np.mean(churns) / 2.0), 8),
                percentile_in_deletion42=round(pct, 3) if np.isfinite(pct) else "",
            )
        )

    write_csv(
        args.events_out,
        [
            "protocol", "full_seed", "refit_seed", "event", "n_keep",
            "mean_drift", "median_drift", "mean_churn", "churn_frac",
            "stress_full", "stress_reduced",
        ],
        event_rows,
    )
    write_csv(
        args.out,
        [
            "protocol", "full_seed", "refit_seed", "n_events",
            "mean_pooled", "median_pooled", "mean_of_event_means",
            "median_of_event_means", "mean_churn", "churn_frac",
            "percentile_in_deletion42",
        ],
        summary_rows,
    )
    print(
        f"[seed-baseline] wrote {os.path.relpath(args.events_out, ROOT)} "
        f"({len(event_rows)} rows) and {os.path.relpath(args.out, ROOT)} "
        f"({len(summary_rows)} rows); elapsed {(time.time()-t0)/60:.1f} min",
        flush=True,
    )

    if args.max_events:
        print("[seed-baseline] smoke run; regression checks skipped")
        return 0

    print("[seed-baseline] regression checks")
    ok = check_summary(summary_rows, strict=args.strict)
    if not ok:
        print("[seed-baseline] CHECK FAILED", file=sys.stderr)
        return 1
    print("[seed-baseline] CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
