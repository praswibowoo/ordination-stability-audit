#!/usr/bin/env python3
"""verdict_provenance_agreement.py — C8: reproducible verdict-provenance strata and
rubric-vs-author concordance (reviewer R5-3, carry-over R4-8).

Problem this fixes (docs/archive/peer_review/Result_PeerReview.md R5-3):
  §3.3 quotes "n = 178" author-labeled rows with measurable parameters and
  "274 of 452 author rows lack measured chemical parameters entirely". Neither
  denominator reconciles with the 267 author-labeled data rows stated in §3.1 and
  §4.4 of the same paper, and no artifact reproduces them. The kappa = 0.033
  claim is load-bearing for the "ordinations do not predict verdicts" conclusion,
  so the denominator has to be machine-derivable.

What it does
  * classifies every data row as author-labeled or rubric-derived, using the
    "[derived v1.0" marker that code/verdict_derivation.py writes into
    cause_effect_note (the same marker the derivation log uses);
  * determines which rows carry measurable water-quality values, joining the value
    layer on (batch_id, row_id) through docs/family_map.csv (the row_id bridge);
  * re-applies the FROZEN rubric v1.0 value route to every author-labeled row that
    has enough values, so the rubric's predicted label can be compared with the
    label the authors actually gave;
  * emits agreement %, Cohen's kappa, and the full confusion table.

Outputs (atomic, UTF-8):
  results/provenance_strata.csv        stratum counts (reproducible denominators)
  results/verdict_rubric_vs_author.csv per-row comparison for kappa
  results/verdict_agreement_summary.csv  agreement %, kappa, n

Read-only on frozen inputs. No randomness, no seed dependence.
Run: python3 code/verdict_provenance_agreement.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
PRES = os.path.join(ROOT, "docs", "extraction", "stageA", "matrix_presence_v2.csv")
VALS = os.path.join(ROOT, "docs", "extraction", "stageB", "matrix_values_v2.csv")
FMAP = os.path.join(ROOT, "docs", "family_map.csv")

from verdict_derivation import band, CORE, METAL_STD, LETHAL_RE, STD_RE, CITE  # noqa: E402

DERIVED_MARKER = "[derived v1.0"
AUTHOR_VOCAB = {"Favorable", "Stable", "Mild stress", "Stressful", "Harmful",
                "Reference standard", "Reference baseline", "Degraded"}
RUBRIC_VOCAB = {"Favorable", "Stable", "Mild stress", "Stressful", "Harmful",
                "Reference standard"}


def _atomic(path, header, rows):
    """Atomic CSV write. Accepts either dict rows (header = fieldnames) or
    list rows (header = first row written verbatim)."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        if rows and isinstance(rows[0], dict):
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        else:
            w = csv.writer(fh)
            w.writerow(header)
            for r in rows:
                w.writerow(r)
    os.replace(tmp, path)


def cohen_kappa(a, b):
    """Cohen's kappa for two categorical label lists."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if abs(1.0 - pe) < 1e-15:
        return po, float("nan")
    return po, (po - pe) / (1.0 - pe)


def bootstrap_kappa_ci(compare, n_boot=2000, seed=42):
    """Percentile cluster bootstrap for Cohen's kappa on the decidable rows.

    Resampling unit = paper. Author labels and rubric labels are assigned
    within a study, so rows from one paper are not independent; papers are
    drawn with replacement and all of their decidable rows enter the
    resample. Seed and resample count are fixed so the interval reproduces.
    """
    import numpy as np
    rows = [c for c in compare if c["rubric_decidable"] == "TRUE"]
    by_paper = defaultdict(list)
    for c in rows:
        by_paper[c["batch_id"]].append((c["author_verdict"], c["rubric_verdict"]))
    papers = sorted(by_paper)
    rng = np.random.default_rng(seed)
    ks = np.empty(n_boot, float)
    for b in range(n_boot):
        pick = rng.integers(0, len(papers), len(papers))
        a, r = [], []
        for j in pick:
            a.extend(x[0] for x in by_paper[papers[j]])
            r.extend(x[1] for x in by_paper[papers[j]])
        ks[b] = cohen_kappa(a, r)[1]
    lo, hi = np.nanpercentile(ks, [2.5, 97.5])
    return float(lo), float(hi), int(n_boot), int(seed), len(papers)


def rubric_predict(row_vals, scen, era, notes):
    """Re-implementation of the FROZEN rubric v1.0 value route.

    Returns the rubric's label, or None when the row carries too little
    information for the rubric to decide (the 'insufficient' branch).
    """
    stype = (row_vals.get("__scenario_type__") or "").strip()
    if stype == "standard" or STD_RE.search(scen or ""):
        return "Reference standard"
    if era == "classic-tox":
        return "Lethal/Mortality risk" if LETHAL_RE.search((scen or "") + " " + (notes or "")) else "Stressful"

    bands, harmed, core = [], [], False
    for p, rec in row_vals.items():
        if p.startswith("__"):
            continue
        v = (rec.get("value") or "").strip()
        c = (rec.get("censored") or "").strip()
        if p in METAL_STD:
            if not v:
                continue
            b = 0 if c == "lt" else (3 if c == "gt" else
                                     (0 if float(v) / METAL_STD[p] < 0.1 else
                                      1 if float(v) / METAL_STD[p] < 0.5 else
                                      2 if float(v) / METAL_STD[p] <= 1 else 3))
            bands.append((p, b))
            core = True
            if b == 3:
                harmed.append(CITE[p])
            continue
        b = None
        if v:
            b = band(p, float(v))
            vm = (rec.get("value_max") or "").strip()
            if vm:
                try:
                    b2 = band(p, float(vm))
                    if b2 is not None and b2 > b:
                        b = b2
                except ValueError:
                    pass
        elif c == "lt":
            b = 0
        elif c == "gt":
            b = 3
        if b is None:
            continue
        bands.append((p, b))
        if p in CORE:
            core = True
        if b == 3:
            harmed.append(CITE.get(p, ""))
    if len(bands) < 2 or not core:
        return None
    mb = max(b for _, b in bands)
    do = next((float(r["value"]) for p, r in row_vals.items()
               if p == "DO (mg/L)" and (r.get("value") or "").strip()), None)
    ph = next((float(r["value"]) for p, r in row_vals.items()
               if p == "pH" and (r.get("value") or "").strip()), None)
    col = next((float(r["value"]) for p, r in row_vals.items()
                if p == "Total Coliform (CFU/100mL)" and (r.get("value") or "").strip()), None)
    if all(b == 0 for _, b in bands) and do is not None and do >= 6 \
            and (col is None or col <= 1000) and (ph is not None and 7 <= ph <= 8.5):
        return "Favorable"
    if all(b == 0 for _, b in bands):
        return "Stable"
    if mb == 1:
        return "Mild stress"
    if mb == 2:
        return "Stressful"
    return "Harmful" if harmed else "Stressful"


def main():
    rows = list(csv.DictReader(open(PRES, encoding="utf-8-sig")))
    hdr = list(rows[0].keys())
    params = hdr[7:51]
    assert len(params) == 44

    # forward-fill paper ownership over all rows (R5-1 convention)
    own = ""
    for r in rows:
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        r["_p"] = own

    # row_id bridge: presence file row index <-> value layer row_id
    fmap = list(csv.DictReader(open(FMAP, encoding="utf-8-sig")))

    # value layer: real values keyed by (batch, presence row index)
    vals = {}
    for r in csv.DictReader(open(VALS, encoding="utf-8")):
        if r["is_real_value"].strip() != "TRUE":
            continue
        if not (r["value"] or "").strip():
            continue
        try:
            int(r["row_id"])
        except ValueError:
            continue
        vals.setdefault((r["batch_id"], int(r["row_id"]) - 1), {})[r["param"]] = r

    data = [(i, r) for i, r in enumerate(rows) if r["row_kind"] == "data"]
    assert len(data) == 764, len(data)

    strata = Counter()
    compare = []
    for gi, r in data:
        note = r.get("cause_effect_note") or ""
        verdict = (r.get("verdict") or "").strip()
        is_derived = DERIVED_MARKER in note
        if verdict in ("", "TBD-StageB"):
            prov = "unlabeled_tbd" if verdict == "TBD-StageB" else "unlabeled_blank"
            has_val = (r["_p"], gi) in vals
            strata[(prov, has_val)] += 1
            continue
        prov = "rubric" if is_derived else "author"
        if verdict not in (RUBRIC_VOCAB | AUTHOR_VOCAB):
            prov = "author_other"
        has_val = (r["_p"], gi) in vals
        strata[(prov, has_val)] += 1

        if prov == "author" and has_val:
            rec = dict(vals[(r["_p"], gi)])
            rec["__scenario_type__"] = (r.get("scenario_type") or "").strip()
            era = next(iter(rec.values())).get("era", "") if rec else ""
            notes = " ".join(v.get("notes", "") for k, v in rec.items()
                             if not k.startswith("__"))
            pred = rubric_predict(rec, r.get("scenario", ""), era, notes)
            compare.append(dict(batch_id=r["_p"], row_index=gi,
                                scenario=r.get("scenario", ""),
                                author_verdict=verdict,
                                rubric_verdict=pred or "",
                                rubric_decidable="TRUE" if pred else "FALSE",
                                agree="TRUE" if pred == verdict else "FALSE"))

    n_author = sum(v for (p, _), v in strata.items() if p.startswith("author"))
    n_rubric = strata.get(("rubric", False), 0) + strata.get(("rubric", True), 0)
    n_auth_val = strata.get(("author", True), 0)
    n_auth_noval = strata.get(("author", False), 0)
    n_auth_other_val = strata.get(("author_other", True), 0)

    auth = [c["author_verdict"] for c in compare]
    pred = [c["rubric_verdict"] for c in compare if c["rubric_decidable"] == "TRUE"]
    auth_d = [c["author_verdict"] for c in compare if c["rubric_decidable"] == "TRUE"]
    agree, kappa = cohen_kappa(auth_d, pred)
    k_lo, k_hi, k_boot, k_seed, k_papers = bootstrap_kappa_ci(compare)

    strata_rows = [
        dict(stratum=k[0], has_values=str(k[1]), rows=v)
        for k, v in sorted(strata.items(), key=lambda kv: (-kv[1], kv[0]))
    ] + [
        dict(stratum="TOTAL author-labeled", has_values="", rows=n_author),
        dict(stratum="TOTAL author with values", has_values="", rows=n_auth_val + n_auth_other_val),
        dict(stratum="TOTAL author without values", has_values="", rows=n_auth_noval),
        dict(stratum="TOTAL rubric-derived", has_values="", rows=n_rubric),
    ]
    _atomic(os.path.join(RESULTS, "provenance_strata.csv"),
            list(strata_rows[0].keys()), strata_rows)
    _atomic(os.path.join(RESULTS, "verdict_rubric_vs_author.csv"),
            list(compare[0].keys()) if compare else ["batch_id"],
            compare if compare else [dict(batch_id="none")])
    _atomic(os.path.join(RESULTS, "verdict_agreement_summary.csv"),
            ["metric", "value"], [
                ["n_author_rows", n_author],
                ["n_author_rows_with_values", n_auth_val + n_auth_other_val],
                ["n_author_rows_without_values", n_auth_noval],
                ["n_rubric_derived_rows", n_rubric],
                ["n_compared", len(auth_d)],
                ["n_rubric_decidable", len(pred)],
                ["agreement_pct", round(agree * 100, 2)],
                ["cohen_kappa", round(kappa, 4)],
                ["cohen_kappa_ci_lo", round(k_lo, 4)],
                ["cohen_kappa_ci_hi", round(k_hi, 4)],
                ["cohen_kappa_papers", k_papers],
                ["cohen_kappa_n_boot", k_boot],
                ["cohen_kappa_seed", k_seed],
            ])

    print("strata:")
    for s in strata_rows:
        print(f"  {s['stratum']:22s} has_values={s['has_values']:5s} {s['rows']}")
    print(f"\ncompared on rubric-decidable rows: n = {len(auth_d)}")
    print(f"  agreement = {agree * 100:.1f}%   Cohen's kappa = {kappa:.4f} "
          f"(95% CI {k_lo:.4f} to {k_hi:.4f}; {k_boot}-resample cluster bootstrap over "
          f"{k_papers} papers, seed {k_seed})")
    print(f"\nconfusion (author -> rubric):")
    cf = Counter(zip(auth_d, pred))
    for (a, b), n in sorted(cf.items(), key=lambda kv: -kv[1]):
        print(f"  {a:20s} -> {b:20s} {n}")
    print("\nwrote results/provenance_strata.csv, results/verdict_rubric_vs_author.csv, "
          "results/verdict_agreement_summary.csv")


if __name__ == "__main__":
    main()
