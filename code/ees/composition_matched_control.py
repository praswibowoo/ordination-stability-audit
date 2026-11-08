#!/usr/bin/env python3
"""composition_matched_control.py — composition- and density-matched deletion controls.

For each study, delete a set of rows from OTHER studies under two controls:

  composition_matched : exact presence configuration first, density second,
                        with at most ceil(n_p/2) rows from any one donor study;
  density_only        : density-matched, configuration blind, same donor cap.

The donor cap replaces the old all-from-one-study behaviour without forcing
one row per study (which destroyed exact matching). Assignments are randomized
and checked to differ across draws.

Reads: frozen presence layer via experiment_C1.load(); block events from
       results/wp3_primary/exH_volumematch.csv by default.
Writes: results/wp3_primary/composition_matched_control.csv
        results/wp3_primary/composition_matched_summary.csv

Run: python3 code/composition_matched_control.py [--draws 100]
     (primary + optimal are the defaults; use --layer legacy for the old set)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zlib
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import (load, ordinate, knn_idx, knn_churn,
                           drift_after_procrustes)  # noqa: E402
from revision_layers import add_layer_args, layer_kwargs  # noqa: E402


def event_drift(S_full, knn_full, X, rho, keep):
    S_red = ordinate("pca_pres", X[keep])
    drift, _ = drift_after_procrustes(S_full[keep], S_red)
    ki = np.where(keep)[0]
    ch = knn_churn(knn_full[keep], ki[knn_idx(S_red)])
    return float(np.median(drift)), float(np.median(ch))


def cluster_boot_block(ev: pd.DataFrame, n_boot: int = 2000, seed: int = SEED):
    y = np.log(ev["med_drift"].to_numpy(float))
    v = np.log(ev["n_rows_deleted"].to_numpy(float))
    b = (ev["arm"] == "block").to_numpy(float)
    Xd = np.column_stack([np.ones(len(y)), v, b])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    uniq = np.asarray(sorted(set(ev["paper"])), dtype=object)
    gid = pd.Categorical(ev["paper"], categories=uniq).codes
    idx = {j: np.where(gid == j)[0] for j in range(len(uniq))}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[j] for j in pick])
        boots.append(np.linalg.lstsq(Xd[sel], y[sel], rcond=None)[0])
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots[:, 2], [0.025, 0.975])
    return float(np.exp(beta[2])), float(np.exp(lo)), float(np.exp(hi))


def assign_capped(rng, idx_p, donor_rows, donor_cfg, row_pos, study_donor_idx,
                  counts, used, paper, rho, cfg, cap, mode):
    """Exact-first greedy assignment with a per-donor-study cap.

    Exact configuration matches are taken first, preferring the least-used
    donor study. Rows without an available exact match are filled by the
    nearest density donor. The cap keeps one donor study from supplying the
    whole matched set while preserving exact matches where they exist.
    """
    n_p = len(idx_p)
    assigned: list[int | None] = [None] * n_p

    def take(a, j):
        pos = row_pos[int(j)]
        assigned[a] = int(j)
        used[pos] = True
        counts[study_donor_idx[pos]] += 1

    exact_rows = [a for a in range(n_p)
                  if mode == "composition" and donor_cfg.get(cfg[idx_p[a]])]
    exact_set = set(exact_rows)
    other_rows = [a for a in range(n_p) if a not in exact_set]
    rng.shuffle(exact_rows)
    rng.shuffle(other_rows)

    for a in exact_rows + other_rows:
        i = idx_p[a]
        if mode == "composition":
            exact = donor_cfg.get(cfg[i], [])
            if exact:
                cand = [j for j in exact
                        if not used[row_pos[j]]
                        and counts[study_donor_idx[row_pos[j]]] < cap]
                if cand:
                    # Shuffle first, then stable-sort by donor-study load so
                    # ties are randomized but the least-used study wins.
                    rng.shuffle(cand)
                    cand.sort(key=lambda j: counts[study_donor_idx[row_pos[j]]])
                    take(a, cand[0])
                    continue

        cost = np.abs(rho[i] - rho[donor_rows])
        cost[used] = np.inf
        cost[counts[study_donor_idx] >= cap] = np.inf
        cost = cost + rng.random(len(cost)) * 1e-6
        pos = int(np.argmin(cost))
        if not np.isfinite(cost[pos]):
            # Cap too tight for this row: relax it but still use an unused row.
            cost = np.abs(rho[i] - rho[donor_rows])
            cost[used] = np.inf
            pos = int(np.argmin(cost))
        take(a, int(donor_rows[pos]))
    return assigned


def assign_optimal_capped(rng, idx_p, donor_rows, paper, rho, cfg, cap, mode):
    """Optimal one-to-one assignment with a per-donor-study cap.

    This is the method described in the manuscript: a min-cost flow assigns
    each deleted row to a distinct donor row from another study, with at most
    ``cap`` rows from any one donor study.  For the composition arm, exact
    configuration matches receive a small cost and all other matches a large
    cost, so exact matches are maximized first and density is minimized second.
    The density-only arm ignores configuration and minimizes density alone.
    """
    n_p = len(idx_p)
    source, sink = "__source__", "__sink__"
    graph = nx.DiGraph()
    graph.add_node(source, demand=-n_p)
    graph.add_node(sink, demand=n_p)

    for a in range(n_p):
        node = f"row{a}"
        graph.add_node(node, demand=0)
        graph.add_edge(source, node, capacity=1, weight=0)

    for study in sorted(set(paper[donor_rows])):
        node = f"study{study}"
        graph.add_node(node, demand=0)
        graph.add_edge(node, sink, capacity=cap, weight=0)

    for j in donor_rows:
        node = f"donor{int(j)}"
        graph.add_node(node, demand=0)
        graph.add_edge(node, f"study{paper[j]}", capacity=1, weight=0)

    for a, i in enumerate(idx_p):
        for j in donor_rows:
            density_cost = int(round(abs(float(rho[i]) - float(rho[j])) * 1_000_000))
            exact = mode == "composition" and cfg[i] == cfg[j]
            if exact:
                cost = 1 + density_cost
            else:
                cost = 100_000_000 + density_cost
            cost += int(rng.integers(0, 1000))
            graph.add_edge(f"row{a}", f"donor{int(j)}", capacity=1, weight=cost)

    flow = nx.min_cost_flow(graph)
    chosen: list[int] = []
    for a in range(n_p):
        for j in donor_rows:
            if flow[f"row{a}"].get(f"donor{int(j)}", 0) == 1:
                chosen.append(int(j))
                break
    if len(chosen) != n_p:
        raise AssertionError(f"optimal assignment returned {len(chosen)} of {n_p} rows")
    return chosen


def _process_paper(task):
    p, X, rho, paper, cfg, block_df, draws, matching = task
    n = len(X)
    n_p = int(np.sum(paper == p))
    idx_p = np.where(paper == p)[0]
    donor_rows = np.where(paper != p)[0]
    donor_cfg: dict[bytes, list[int]] = defaultdict(list)
    for j in donor_rows:
        donor_cfg[cfg[j]].append(int(j))
    row_pos = {int(j): k for k, j in enumerate(donor_rows)}
    study_ids = {s: k for k, s in enumerate(sorted(set(paper[donor_rows])))}
    study_donor_idx = np.array([study_ids[paper[j]] for j in donor_rows], int)
    cap = max(1, int(np.ceil(0.5 * n_p)))
    S_full = ordinate("pca_pres", X)
    knn_full = knn_idx(S_full)

    out = []
    for mode, arm in (("composition", "composition_matched"),
                      ("density", "density_only")):
        for dr in range(draws):
            seed = (SEED + zlib.crc32(f"{p}:{mode}".encode()) % 100000
                    + 7919 * dr)
            rng = np.random.default_rng(seed)
            if matching == "optimal":
                chosen = assign_optimal_capped(
                    rng, idx_p, donor_rows, paper, rho, cfg, cap, mode)
            else:
                counts = np.zeros(len(study_ids), int)
                used = np.zeros(len(donor_rows), bool)
                chosen = assign_capped(
                    rng, idx_p, donor_rows, donor_cfg, row_pos,
                    study_donor_idx, counts, used, paper, rho, cfg, cap, mode)
            counts_chosen = Counter(paper[chosen])
            keep = np.ones(n, bool)
            keep[np.asarray(chosen)] = False
            d, ch = event_drift(S_full, knn_full, X, rho, keep)
            out.append(dict(
                paper=p, draw=dr, arm=arm,
                med_drift=d, med_churn=ch,
                n_rows_deleted=n_p,
                cap=cap,
                exact_fraction=float(np.mean([cfg[idx_p[a]] == cfg[chosen[a]]
                                              for a in range(n_p)])),
                n_distinct_donor_studies=int(len(counts_chosen)),
                n_donor_studies_available=int(len(study_ids)),
                max_donor_rows=int(max(counts_chosen.values())),
                max_donor_fraction=float(max(counts_chosen.values()) / n_p),
                assignment_id=hashlib.sha1(
                    ",".join(map(str, sorted(chosen))).encode()
                ).hexdigest()[:12],
            ))
    return out


def summary_for_arm(arm, matched, block, n_boot):
    sub = matched[matched["arm"] == arm]
    block_by_paper = dict(zip(block["paper"], block["med_drift"]))
    papers = sorted(block_by_paper)
    by_paper = {p: g["med_drift"].to_numpy(float)
                for p, g in sub.groupby("paper")}
    d_l = np.array([block_by_paper[p] for p in papers], float)
    d_m = np.array([float(np.median(by_paper[p])) for p in papers], float)
    paired = d_l / d_m
    ratio_of_meds = float(np.median(d_l) / np.median(d_m))
    med_paired = float(np.median(paired))
    rng = np.random.default_rng(SEED)
    boot = np.empty(10000)
    for i in range(10000):
        idx = rng.integers(0, len(paired), len(paired))
        boot[i] = np.median(paired[idx])
    ci = np.quantile(boot, [0.025, 0.975])
    p_w = stats.wilcoxon(d_l, d_m).pvalue

    ev = pd.concat([
        block.assign(arm="block")[["paper", "med_drift", "n_rows_deleted", "arm"]],
        sub[["paper", "med_drift", "n_rows_deleted", "arm"]],
    ], ignore_index=True)
    bf, lo, hi = cluster_boot_block(ev, n_boot=n_boot)
    unique = sub.groupby("paper")["assignment_id"].nunique()
    multi = sub[sub["n_rows_deleted"] > 1]
    return dict(
        arm=arm,
        n_papers=len(papers),
        block_med=float(np.median(d_l)),
        matched_med=float(np.median(d_m)),
        ratio_of_medians=ratio_of_meds,
        median_paired_ratio=med_paired,
        paired_ci_lo=float(ci[0]),
        paired_ci_hi=float(ci[1]),
        wilcoxon_p=float(p_w),
        block_factor=bf,
        block_ci_lo=lo,
        block_ci_hi=hi,
        mean_exact_fraction=float(sub["exact_fraction"].mean()),
        min_exact_fraction=float(sub["exact_fraction"].min()),
        n_papers_with_exact=int((sub.groupby("paper")["exact_fraction"].first() > 0).sum()),
        n_papers_zero_exact=int((sub.groupby("paper")["exact_fraction"].first() == 0).sum()),
        mean_cap=float(sub["cap"].mean()),
        mean_distinct_donor_studies=float(sub["n_distinct_donor_studies"].mean()),
        mean_donor_studies_available=float(sub["n_donor_studies_available"].mean()),
        max_donor_rows=int(sub["max_donor_rows"].max()),
        max_donor_fraction=float(sub["max_donor_fraction"].max()),
        max_donor_fraction_multi=(
            float(multi["max_donor_fraction"].max()) if len(multi) else float("nan")
        ),
        n_unique_assignments=int(sub["assignment_id"].nunique()),
        n_papers_with_varying_assignments=int((unique > 1).sum()),
        n_events=len(ev),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--matching", choices=("greedy", "optimal"), default="optimal",
                    help="optimal capped min-cost flow (manuscript) or legacy greedy")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (default 1)")
    ap.add_argument("--block-csv", default=None,
                    help="presence block-event CSV (default: primary wp3_primary)")
    ap.add_argument("--exclude-papers", default="",
                    help="comma-separated batch IDs to remove before the audit")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: wp3_primary for primary layer)")
    add_layer_args(ap)
    ap.set_defaults(layer="primary")
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = (os.path.join(RESULTS, "wp3_primary")
                       if args.layer == "primary" else RESULTS)
    if args.block_csv is None:
        args.block_csv = (os.path.join(RESULTS, "wp3_primary", "exH_volumematch.csv")
                          if args.layer == "primary"
                          else os.path.join(RESULTS, "exH_volumematch.csv"))
    os.makedirs(args.outdir, exist_ok=True)

    X, rho, paper, params, *_ = load(**layer_kwargs(args))
    paper = np.asarray(paper)
    excluded = {p.strip() for p in args.exclude_papers.split(",") if p.strip()}
    if excluded:
        keep = ~np.isin(paper, sorted(excluded))
        X, rho, paper = X[keep], rho[keep], paper[keep]
        print(f"excluded {sorted(excluded)}: {X.shape[0]} rows / {len(set(paper))} papers")
    n = len(X)
    uniq = sorted(set(paper))
    S_full = ordinate("pca_pres", X)
    knn_full = knn_idx(S_full)

    # Presence configuration per row, as a hashable bytes key.
    cfg = [bytes(X[i].astype(np.uint8)) for i in range(n)]

    block = pd.read_csv(args.block_csv)
    block = block[block["arm"].isin(["lopo", "block"])][
        ["paper", "med_drift", "n_rows_deleted"]]
    block = block[block["paper"].isin(uniq)].reset_index(drop=True)
    if len(block) != len(uniq):
        raise ValueError(f"block CSV has {len(block)} LOPO rows for {len(uniq)} studies")

    tasks = [(p, X, rho, paper, cfg, block, args.draws, args.matching)
             for p in uniq]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for batch in pool.map(_process_paper, tasks, chunksize=1)
                    for row in batch]
    else:
        rows = [row for task in tasks for row in _process_paper(task)]
    rows.sort(key=lambda r: (r["paper"], r["arm"], r["draw"]))

    matched = pd.DataFrame(rows)
    out = os.path.join(args.outdir, "composition_matched_control.csv")
    matched.to_csv(out, index=False, float_format="%.6g")

    # Self-checks: cap respected, all draws present, assignments randomized.
    for arm in ("composition_matched", "density_only"):
        sub = matched[matched["arm"] == arm]
        per_paper_draws = sub.groupby("paper")["draw"].nunique()
        if not (per_paper_draws == args.draws).all():
            raise AssertionError(f"{arm}: not every paper has {args.draws} draws")
        if not (sub["max_donor_rows"] <= sub["cap"]).all():
            raise AssertionError(f"{arm}: donor cap violated")
        if not (sub["max_donor_fraction"]
                <= sub["cap"] / sub["n_rows_deleted"] + 1e-12).all():
            raise AssertionError(f"{arm}: donor fraction exceeds cap/n_rows")
        min_studies = np.ceil(sub["n_rows_deleted"] / sub["cap"])
        if not (sub["n_distinct_donor_studies"] >= min_studies).all():
            raise AssertionError(f"{arm}: too few distinct donor studies")
        varying = sub.groupby("paper")["assignment_id"].nunique()
        multi = sub.groupby("paper")["n_rows_deleted"].first()
        eligible = varying[multi > 1]
        if args.draws > 1 and len(eligible) and not (eligible > 1).all():
            raise AssertionError(f"{arm}: identical assignments across draws")

    summaries = [
        summary_for_arm(arm, matched, block, args.bootstrap)
        for arm in ("composition_matched", "density_only")
    ]
    summ = pd.DataFrame(summaries)
    spath = os.path.join(args.outdir, "composition_matched_summary.csv")
    summ.to_csv(spath, index=False, float_format="%.6g")
    print(summ.to_string(index=False))
    print(f"\nwrote {os.path.relpath(out, ROOT)} and "
          f"{os.path.relpath(spath, ROOT)}")


if __name__ == "__main__":
    main()
