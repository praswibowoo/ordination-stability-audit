#!/usr/bin/env python3
"""p0_tie_aware_grade.py — tie-aware quartile-grade audit (P0-4).

The manuscript assigns a row on a quartile cut to the upper grade. Binary PCA
scores have many exact ties, so a small shift can move a whole tie group across
a cut. This script recomputes the block-event grade flips with a jitter-based
tie-aware rule and reports the tie fraction.

Run: python3 code/p0_tie_aware_grade.py
Out: results/wp3_clean/p0_tie_aware_grade.csv
"""
from __future__ import annotations
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "core"))
import clean_primary_audit as cpa  # noqa: E402

SEED = 42
N_JITTER = 200
EXCLUDE = cpa.EXCLUDE
OUT = os.path.join(ROOT, "results", "wp3_clean", "p0_tie_aware_grade.csv")


def tie_aware_flip(a, b, n_jitter=N_JITTER, seed=SEED):
    """Randomly break exact quartile ties and average the flip rate.

    The manuscript rule assigns a row on a cut to the upper grade. Here, each
    row exactly on a cut is assigned upper/lower with probability 0.5, which
    gives the tie-averaged flip rate. Cuts stay fixed at the full-data values.
    """
    rng = np.random.default_rng(seed)
    cuts = np.quantile(a, [0.25, 0.50, 0.75])
    qa0 = np.searchsorted(cuts, a, side="right")
    qb0 = np.searchsorted(cuts, b, side="right")
    strict = float(np.mean(qa0 != qb0))
    tie_a_mask = np.zeros(a.size, bool)
    tie_b_mask = np.zeros(b.size, bool)
    for c in cuts:
        tie_a_mask |= (a == c)
        tie_b_mask |= (b == c)
    tie_a = float(tie_a_mask.mean())
    tie_b = float(tie_b_mask.mean())
    flips = np.empty(n_jitter)
    for k in range(n_jitter):
        qa = qa0.copy()
        qb = qb0.copy()
        qa[tie_a_mask & (rng.random(a.size) < 0.5)] -= 1
        qb[tie_b_mask & (rng.random(b.size) < 0.5)] -= 1
        flips[k] = float(np.mean(qa != qb))
    return dict(strict_flip_pct=100.0 * strict,
                tie_frac_a_pct=100.0 * tie_a,
                tie_frac_b_pct=100.0 * tie_b,
                tie_aware_median_pct=100.0 * float(np.median(flips)),
                tie_aware_q05_pct=100.0 * float(np.quantile(flips, 0.05)),
                tie_aware_q95_pct=100.0 * float(np.quantile(flips, 0.95)))


def main():
    L = cpa.load_layers()
    X, paper = L.X, np.asarray(L.paper, dtype=object)
    Xv = L.Xv
    vpaper = np.asarray([k[0] for k in L.keysv], dtype=object)
    keep_c = ~np.isin(paper, EXCLUDE)
    keep_v = ~np.isin(vpaper, EXCLUDE)
    Xc, paperc = X[keep_c], paper[keep_c]
    Xvc, vpaperc = Xv[keep_v], vpaper[keep_v]
    row_types = [cpa.classify_coverage(r) for r in cpa.coverage_rows()]
    water = np.asarray([t == "water" for t in row_types])
    keep_w = keep_c & water
    Xw, paperw = X[keep_w], paper[keep_w]

    rows = []
    for layer, Xl, pl, fit in [
        ("coverage", Xc, paperc, cpa.fit_presence),
        ("coverage_water", Xw, paperw, cpa.fit_presence),
        ("value", Xvc, vpaperc, cpa.fit_value),
    ]:
        S_full = fit(Xl)
        for p in sorted(set(pl)):
            mask = pl == p
            keep = ~mask
            idx = np.where(keep)[0]
            S_red = fit(Xl[keep])
            a = S_full[idx, 0].copy()
            b = S_red[:, 0].copy()
            c = np.corrcoef(a, b)[0, 1]
            if np.isfinite(c) and c < 0:
                b = -b
            b = (b - b.mean()) / b.std() * a.std() + a.mean()
            m = tie_aware_flip(a, b)
            rows.append(dict(layer=layer, paper=p, rows=int(mask.sum()), **m))

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for layer in ("coverage", "coverage_water", "value"):
        sub = [r for r in rows if r["layer"] == layer]
        print(f"\n[{layer}] n={len(sub)}")
        for key in ("strict_flip_pct", "tie_aware_median_pct",
                    "tie_frac_a_pct", "tie_frac_b_pct"):
            vals = np.array([r[key] for r in sub], float)
            print(f"  {key}: median {np.median(vals):.3f}  IQR "
                  f"{np.quantile(vals, .25):.3f}-{np.quantile(vals, .75):.3f}")
        for p in ("B063", "B128", "B156", "B020", "B084"):
            hit = [r for r in sub if r["paper"] == p]
            if hit:
                r = hit[0]
                print(f"  {p}: strict {r['strict_flip_pct']:.3f} -> "
                      f"tie-aware {r['tie_aware_median_pct']:.3f} "
                      f"[{r['tie_aware_q05_pct']:.3f}, {r['tie_aware_q95_pct']:.3f}], "
                      f"tie_a {r['tie_frac_a_pct']:.2f}%, tie_b {r['tie_frac_b_pct']:.2f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
