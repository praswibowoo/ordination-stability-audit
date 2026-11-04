#!/usr/bin/env python3
"""check_si_parity.py — submission SI must contain the same floats as the canonical SI.

The submitted supplement (`supplementary_main.tex`) is a floats-only extract of
the canonical prose supplement (`supplementary_rev2.tex`). Captions are
deliberately shortened for submission, so this check compares the ordered list
of float labels, not caption text. It catches the failure mode that produced
the S1-S7 / S8 mismatch: a float added to the canonical SI but not to the
submission extract (or vice versa).

It also checks Table S19 against `results/family_tier_sensitivity.csv`, because
that float is hand-copied from a generated Markdown table and has no other
value gate. The check is skipped when the results layer is absent (code archive).

Read-only. Exit 1 on any difference.
"""
from __future__ import annotations

import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "manuscript/latex/supplementary_rev2.tex")
SUBMISSION = os.path.join(ROOT, "manuscript/latex/supplementary_main.tex")

FLOAT_RE = re.compile(
    r"\\begin\{(figure|table|sidewaystable)\}(.*?)\\end\{\1\}", re.S)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def labels(path: str) -> list[tuple[str, str]]:
    text = open(path, encoding="utf-8").read()
    out = []
    for env, body in FLOAT_RE.findall(text):
        m = LABEL_RE.search(body)
        if not m:
            out.append((env, "<unlabelled>"))
        else:
            out.append((env, m.group(1)))
    return out


# ---- S19 value check -------------------------------------------------------
S19_CSV = os.path.join(ROOT, "results/family_tier_sensitivity.csv")
S19_FAMILY = {
    "F1 physical": "F1_physical",
    "F1a": "F1_physicala",
    "F1 clarity": "F1_clarity",
    "F2 metals": "F2_metals",
    "F3 nutrients": "F3_nutrients",
    "F4 organic": "F4_organic",
    "F6 aeration": "F6_aeration",
}
TIER_LETTER = {"ready": "R", "provisional": "P", "incidental": "I"}
TAU_VALUES = ["1e-09", "1e-08", "1e-07", "1e-06", "1e-05", "1e-04"]
P_VALUES = ["0.01", "0.05", "0.10", "0.20"]
SCENARIOS = ["base", "no_3d", "no_collapse", "loosest"]


def _cell(value: str) -> tuple[str, bool]:
    """Return (tier letter, star) from a LaTeX panel cell."""
    raw = value.replace("$", "").replace("{", "").replace("}", "")
    star = "*" in raw
    letters = re.sub(r"[^PRI]", "", raw.replace("*", ""))
    return letters, star


def _row_cells(line: str) -> list[str]:
    return [c.strip() for c in line.rstrip().rstrip("\\").split("&")]


def check_s19_values() -> list[str]:
    if not (os.path.exists(S19_CSV) and os.path.exists(CANON)):
        return []  # code archive: no results layer / no manuscript
    text = open(CANON, encoding="utf-8").read()
    m = re.search(r"\\label\{tab:S19\}(.*?)\\end\{table\}", text, re.S)
    if not m:
        return ["S19: float not found in canonical SI"]
    tabs = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", m.group(1), re.S)
    if len(tabs) != 3:
        return [f"S19: expected 3 panels, found {len(tabs)}"]

    rows = {}
    with open(S19_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["panel"], r["scenario"], r["value"], r["family"])] = r

    def expect(key: tuple[str, str, str, str]) -> tuple[str, bool]:
        r = rows[key]
        return TIER_LETTER[r["criteria_tier"]], r["at_floor"] == "1"

    problems: list[str] = []

    def compare(panel: str, fam: str, value: str, cell: str, where: str) -> None:
        if panel == "tau":
            key = ("tau", "tau", value, fam)
        elif panel == "p":
            key = ("p", "benchmark", value, fam)
        else:
            key = ("structure", value, value, fam)
        if key not in rows:
            problems.append(f"S19 {where}: no CSV row for {fam} {value}")
            return
        want = expect(key)
        got = _cell(cell)
        if got != want:
            problems.append(
                f"S19 {where}: {fam} {value} shows {got[0]}{'*' if got[1] else ''} "
                f"but CSV says {want[0]}{'*' if want[1] else ''}")

    for panel, tab, values in (("tau", tabs[0], TAU_VALUES),
                               ("p", tabs[1], P_VALUES),
                               ("structure", tabs[2], SCENARIOS)):
        for line in tab.splitlines():
            if "&" not in line or line.lstrip().startswith("\\") or "Family" in line:
                continue
            cells = _row_cells(line)
            family_label = cells[0]
            fam = S19_FAMILY.get(family_label)
            if fam is None:
                continue
            if panel == "p":
                for value, cell in zip(values, cells[1:5]):
                    compare(panel, fam, value, cell, "Panel B")
                pb = cells[5] if len(cells) > 5 else ""
                try:
                    if abs(float(pb) - float(rows[("p", "benchmark", "0.05", fam)]["p_bench"])) > 5e-4:
                        problems.append(f"S19 Panel B: {family_label} benchmark p {pb} "
                                        f"!= CSV {rows[('p','benchmark','0.05',fam)]['p_bench']}")
                except ValueError:
                    problems.append(f"S19 Panel B: {family_label} unreadable benchmark p {pb!r}")
            else:
                for value, cell in zip(values, cells[1:]):
                    compare(panel, fam, value, cell, "Panel A" if panel == "tau" else "Panel C")
    return problems


def main() -> int:
    missing = [p for p in (CANON, SUBMISSION) if not os.path.exists(p)]
    if missing:
        print("[si-parity] SKIPPED — manuscript sources not in this tree: "
              + ", ".join(os.path.relpath(p, ROOT) for p in missing))
        return 0
    a = labels(CANON)
    b = labels(SUBMISSION)
    la = [x[1] for x in a]
    lb = [x[1] for x in b]

    problems = []
    if len(la) != len(lb):
        problems.append(f"float count: canonical {len(la)} vs submission {len(lb)}")
    only_a = [x for x in la if x not in lb]
    only_b = [x for x in lb if x not in la]
    if only_a:
        problems.append("only in canonical: " + ", ".join(only_a))
    if only_b:
        problems.append("only in submission: " + ", ".join(only_b))
    for i, (x, y) in enumerate(zip(la, lb), 1):
        if x != y:
            problems.append(f"order differs at float {i}: {x} vs {y}")
            break

    if problems:
        print("[si-parity] FAIL — submission SI is not a faithful float extract")
        for p in problems:
            print("  " + p)
        print("  run: python3 code/check_si_parity.py")
        return 1
    s19 = check_s19_values()
    if s19:
        print("[si-parity] FAIL — Table S19 does not match family_tier_sensitivity.csv")
        for p in s19:
            print("  " + p)
        return 1
    print(f"[si-parity] OK — {len(la)} floats, same labels and order; S19 values match the CSV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
