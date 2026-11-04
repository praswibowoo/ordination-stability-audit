#!/usr/bin/env python3
"""make_f1_drift_taxonomy.py — per-event drift taxonomy for the F1-physical family fit (D199).

Backs the Supplementary Extended-analyses sentence that no artifact previously
produced:
  "F1 physical realizes 11 of 15 possible non-empty configurations (2^4 - 1) among
   its 748 cover rows ... and its per-event drift takes 11 distinct values in
   119 of 121 deletion events, causing the sparse- and dense-quartile medians to
   coincide exactly (SI_fam = 1.000)."

The claim was correct but unbacked (a previous session had already flagged it as
"unsourced" inside the digest generator). Mechanism: rows sharing a presence
configuration land on identical coordinates, so each LOPO refit yields one drift
value per realized configuration; deleting a paper that owns every row of a
configuration removes that value for that event, which is exactly why two of the
121 events carry 10 rather than 11.

Uses the SHIPPED primitives — `experiment_C1.load` for the layers and the same
family-fit + Procrustes path as `experiment_C2_stability` — so the numbers come
from the production code path, not from a reimplementation.

Reads frozen layers via the loader. Writes results/exC2_drift_taxonomy.csv (atomic).
Deterministic (no randomness). Run: python3 code/make_f1_drift_taxonomy.py
"""
from __future__ import annotations
import csv
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from phase4_freeze import verify_baseline  # noqa: E402
from experiment_C1 import load, drift_after_procrustes  # noqa: E402
from experiment_C2_stability import FAMILIES  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from scipy.spatial import procrustes  # noqa: E402

FAM = "F1_physical"
OUT = os.path.join(ROOT, "results", "exC2_drift_taxonomy.csv")


def _fit(X, dim=2):
    return PCA(n_components=dim, random_state=42).fit_transform(X)


def main() -> int:
    verify_baseline()
    X, rho, paper, params, _, _, _ = load()
    fam_cols = [params.index(p) for p in FAMILIES[FAM]]
    paper = np.asarray(paper)

    cover = np.array([any(X[i][c] > 0 for c in fam_cols) for i in range(len(X))])
    idx = np.where(cover)[0]
    Xf = X[idx][:, fam_cols]
    pm = paper[idx]

    configs = [tuple(int(v) for v in row) for row in Xf]
    realized = len(set(configs))
    possible = 2 ** len(fam_cols) - 1

    S_full = _fit(Xf)
    dist_by_event, cfgs_kept = {}, {}
    for p in sorted(set(pm)):
        keep = pm != p
        if keep.sum() < 8:
            continue
        S_red = _fit(Xf[keep])
        a, b, _ = procrustes(S_full[keep], S_red)
        d = np.round(np.linalg.norm(b - a, axis=1), 9)
        dist_by_event[p] = int(len(set(d.tolist())))
        cfgs_kept[p] = int(len(set(c for c, k in zip(configs, keep) if k)))

    n_events = len(dist_by_event)
    hist = Counter(dist_by_event.values())
    full_events = hist.get(realized, 0)
    preserved = sum(1 for p in cfgs_kept if cfgs_kept[p] == realized)

    rows = [dict(metric="cover_rows", value=len(idx)),
            dict(metric="papers", value=len(set(pm))),
            dict(metric="configs_realized", value=realized),
            dict(metric="configs_possible", value=possible),
            dict(metric="events", value=n_events),
            dict(metric="events_with_all_configs", value=full_events),
            dict(metric="events_missing_one_config", value=n_events - full_events),
            dict(metric="events_config_preserved", value=preserved)]

    # --- the numbers the Extended analyses quotes ---
    assert (len(idx), len(set(pm))) == (748, 121), (len(idx), len(set(pm)))
    assert (realized, possible) == (11, 15), (realized, possible)
    assert n_events == 121, n_events
    assert full_events == 119, hist            # "11 distinct values in 119 of 121 events"
    # NOTE (D199): unlike F6 aeration — whose 5 configurations survive every one of
    # its 112 deletions (asserted in ex_f6_tail_summary.all_configurations_preserved) —
    # the F1-physical set does NOT survive every deletion: two events (B096, B222)
    # lose one configuration outright, which is precisely why they are the two events
    # with 10 rather than 11 distinct drift values. The F6 "5 -> 5" statement is
    # F6-specific and must not be generalised to other families.
    assert n_events - full_events == 2, hist
    assert preserved == full_events, (preserved, full_events)

    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)

    print(f"{FAM}: {len(idx)} cover rows / {len(set(pm))} papers; "
          f"{realized} of {possible} configurations realized")
    print(f"  distinct per-row drift values per event: {dict(sorted(hist.items()))}")
    print(f"  events carrying all {realized} values: {full_events} of {n_events}; "
          f"configurations preserved in {preserved}/{n_events} events")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
