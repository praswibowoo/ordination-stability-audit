#!/usr/bin/env python3
"""family_tier_sensitivity.py — tier stability under tau and rubric thresholds.

Reads the committed tier table (results/family_tiers.csv) and re-applies the
five tier-setting criteria under:
  (a) a tau grid (1e-9 ... 1e-4),
  (b) a benchmark-p grid (0.01, 0.05, 0.10, 0.20),
  (c) structural switches (drop the 3-D criterion, drop the collapse criterion,
      and the joint loosest rubric).

The base scenario must reproduce the committed tiers exactly; the script
asserts that before writing anything. Writes:
  results/family_tier_sensitivity.csv
  manuscript/tables/tableS19_tier_sensitivity.md

The LaTeX table in supplementary_rev2.tex / supplementary_main.tex is
hand-copied from the Markdown. If the CSV changes, regenerate the Markdown,
update both SI copies, and re-run `python3 code/check_si_parity.py`.

Run: python3 code/family_tier_sensitivity.py
"""
from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "manuscript", "tables")

BASE_TAU = 1e-6
BASE_P = 0.05
TAU_GRID = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
P_GRID = (0.01, 0.05, 0.10, 0.20)


def tier(row: dict, tau: float = BASE_TAU, p_thr: float = BASE_P,
         require_3d: bool = True, require_collapse: bool = True) -> dict:
    """Apply the published rubric to one family row."""
    raw = float(row["raw_pca2d"])
    lo = float(row["ci_lo"])
    fam_med = float(row["fam_med"])
    rep3 = int(row["reportable_3d"])
    ratio3 = float(row["ratio_3d"]) if row["ratio_3d"] else float("nan")
    collapse = int(row["collapse_events"])
    exceed = int(row["benchmark_exceed"])
    nondeg = int(row["benchmark_nondeg"])
    p_bench = (1 + exceed) / (1 + nondeg)

    c1 = raw > 1.0 and lo > 1.0
    c3 = lo > 1.0
    c4 = fam_med > tau
    c5 = (rep3 == 1 and ratio3 > 1.0) if require_3d else True
    c6 = ((collapse == 0) or (fam_med > tau)) if require_collapse else True
    c7 = p_bench < p_thr
    incidental = raw <= 1.0

    if c1 and c3 and c4 and c5 and c6 and c7:
        label = "ready"
    elif not incidental and ((not c4) or (not c5) or (not c7)):
        label = "provisional"
    else:
        label = "incidental"

    return dict(c1=int(c1), c3=int(c3), c4=int(c4), c5=int(c5),
                c6=int(c6), c7=int(c7), p_bench=p_bench,
                at_floor=int(fam_med <= tau), tier=label)


def read_tiers() -> list[dict]:
    path = os.path.join(RESULTS, "family_tiers.csv")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def short(label: str) -> str:
    return {"ready": "R", "provisional": "P", "incidental": "I"}[label]


def main() -> None:
    rows = read_tiers()
    assert rows, "results/family_tiers.csv is empty"

    # Gate: the base rubric must reproduce the committed tiers.
    for r in rows:
        base = tier(r)
        assert base["tier"] == r["tier"], (
            f"base tier mismatch for {r['family']}: "
            f"{base['tier']} vs stored {r['tier']}")

    out = []
    for r in rows:
        fam = r["family"]
        # Panel A: tau grid (other criteria at base)
        for tau in TAU_GRID:
            s = tier(r, tau=tau)
            out.append(dict(panel="tau", scenario="tau", param="tau",
                            value=f"{tau:.0e}", family=fam,
                            raw_pca2d=r["raw_pca2d"], ci_lo=r["ci_lo"],
                            fam_med=r["fam_med"], at_floor=s["at_floor"],
                            p_bench=f"{s['p_bench']:.3f}",
                            c1=s["c1"], c4=s["c4"], c5=s["c5"],
                            c6=s["c6"], c7=s["c7"], criteria_tier=s["tier"]))
        # Panel B: benchmark-p grid (other criteria at base)
        for p in P_GRID:
            s = tier(r, p_thr=p)
            out.append(dict(panel="p", scenario="benchmark", param="p",
                            value=f"{p:.2f}", family=fam,
                            raw_pca2d=r["raw_pca2d"], ci_lo=r["ci_lo"],
                            fam_med=r["fam_med"], at_floor=s["at_floor"],
                            p_bench=f"{s['p_bench']:.3f}",
                            c1=s["c1"], c4=s["c4"], c5=s["c5"],
                            c6=s["c6"], c7=s["c7"], criteria_tier=s["tier"]))
        # Panel C: structural switches
        variants = (
            ("base", {}),
            ("no_3d", dict(require_3d=False)),
            ("no_collapse", dict(require_collapse=False)),
            ("loosest", dict(tau=1e-9, p_thr=0.20,
                             require_3d=False, require_collapse=False)),
        )
        for name, kw in variants:
            s = tier(r, **kw)
            out.append(dict(panel="structure", scenario=name, param="switch",
                            value=name, family=fam,
                            raw_pca2d=r["raw_pca2d"], ci_lo=r["ci_lo"],
                            fam_med=r["fam_med"], at_floor=s["at_floor"],
                            p_bench=f"{s['p_bench']:.3f}",
                            c1=s["c1"], c4=s["c4"], c5=s["c5"],
                            c6=s["c6"], c7=s["c7"], criteria_tier=s["tier"]))

    out_path = os.path.join(RESULTS, "family_tier_sensitivity.csv")
    with open(out_path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    os.replace(out_path + ".tmp", out_path)

    # Markdown table (the SI LaTeX table mirrors this file).
    def cell(panel: str, fam: str, value: str) -> str:
        for r in out:
            if r["panel"] == panel and r["family"] == fam and r["value"] == value:
                return short(r["criteria_tier"]) + ("*" if r["at_floor"] else "")
        raise KeyError((panel, fam, value))

    lines = [
        "# Table S19 — Tier sensitivity to τ and rubric thresholds",
        "",
        "Panel A: τ grid; all other criteria at base (benchmark p < 0.05, 3-D and collapse on).",
        "",
        "| Family | " + " | ".join(f"τ={t:.0e}" for t in TAU_GRID) + " |",
        "|---|" + "|".join([":--:"] * len(TAU_GRID)) + "|",
    ]
    for r in rows:
        fam = r["family"]
        lines.append("| " + fam + " | " + " | ".join(
            cell("tau", fam, f"{t:.0e}") for t in TAU_GRID) + " |")
    lines += [
        "",
        "Panel B: benchmark-p grid; τ = 1e-6, 3-D and collapse on.",
        "",
        "| Family | " + " | ".join(f"p<{p:.2f}" for p in P_GRID) + " | benchmark p |",
        "|---|" + "|".join([":--:"] * len(P_GRID)) + "|:--:|",
    ]
    for r in rows:
        fam = r["family"]
        pb = next(x["p_bench"] for x in out
                  if x["panel"] == "p" and x["family"] == fam and x["value"] == "0.05")
        lines.append("| " + fam + " | " + " | ".join(
            cell("p", fam, f"{p:.2f}") for p in P_GRID) + f" | {pb} |")
    lines += [
        "",
        "Panel C: structural switches. Base = τ 1e-6, p < 0.05, all criteria on.",
        "",
        "| Family | base | no 3-D | no collapse | loosest |",
        "|---|:--:|:--:|:--:|:--:|",
    ]
    for r in rows:
        fam = r["family"]
        lines.append("| " + fam + " | " + " | ".join(
            cell("structure", fam, v) for v in
            ("base", "no_3d", "no_collapse", "loosest")) + " |")
    lines += [
        "",
        "R = all five stability criteria met at the stated threshold; P = provisional; I = incidental; tiers in Table 5 use the base threshold (p < 0.05). `*` = median family drift at or below τ.",
        "Loosest = τ 1e-9, benchmark p < 0.20, 3-D and collapse criteria off.",
        "The base column reproduces `results/family_tiers.csv` exactly. F2 metals meets the five criteria at p < 0.10 on three presence-complete papers; that column is a threshold sensitivity, not a tier assignment. Under the joint loosest rubric, F6 aeration also meets the criteria.",
        "",
    ]
    table_path = os.path.join(TABLES, "tableS19_tier_sensitivity.md")
    os.makedirs(TABLES, exist_ok=True)
    with open(table_path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    os.replace(table_path + ".tmp", table_path)

    n_ready = sum(r["criteria_tier"] == "ready" for r in out if r["panel"] == "p"
                  and r["value"] == "0.10")
    print(f"[tier-sensitivity] {len(rows)} families × "
          f"{len(TAU_GRID)} τ + {len(P_GRID)} p + 4 structural variants")
    print(f"[tier-sensitivity] ready at p<0.10: {n_ready}")
    print(f"[tier-sensitivity] wrote {os.path.relpath(out_path, ROOT)}")
    print(f"[tier-sensitivity] wrote {os.path.relpath(table_path, ROOT)}")


if __name__ == "__main__":
    main()
