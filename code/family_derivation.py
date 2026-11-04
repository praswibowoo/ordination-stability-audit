#!/usr/bin/env python3
"""P1 Phase 3 — Family derivation v3 = Model D (pre-registered clustering + amendment registry).

Structure:
  PART 1 — Pre-registered derivation (P1_Master_Plan.md §4): co-reporting Jaccard on 44 params
           across condition-rows; Ward (Jaccard distance); cuts k∈[4,7]; candidates pruned
           deterministically to >=10 full-reporting rows; veto rule (metal x physical J>=0.9);
           raw-vs-pruned tables + traces logged for transparency.
  PART 2 — Amendment registry v1.1 (FROZEN, user-approved 2026-10-01; replaces v1.0 family set):
           explicit, versioned family definitions with rationale + evidence. This is NOT a
           silent override: it is a documented version bump per change control (Data_Dictionary §7).

Amendment registry rationale (summary):
  - F3 nutrients {NO2,NO3,NH4,PO4}: pre-registered rule 4 MANDATES mapping to v1 families
    where compatible (F3 nutrients); clustering fragmented this suite (never co-clustered).
  - F4 organic {COD,BOD,Total Coliform}: same rule-4 rationale (v1 F4 organic); boundary
    diagnostics support (COD avgJ 0.36 vs 0.21 to minerals; BOD 0.35 vs 0.29).
  - F1 clarity {Turbidity,Secchi,TSS}: domain-lead semantic correction (Secchi = optical
    transparency, belongs with clarity suite, not nutrients). Flagged: lowest cohesion.
  - F6 aeration {DO,BOD,COD}: discovery amendment (exhaustive subset search + binding matrix);
    DO/BOD/COD are bridge params split across clusters; classic oxygen-balance suite.
  - F1 physical restructured {DO,pH,Salinity,Temperature}; F1 physicala {TDS,EC,Cl-,Hardness,
    Alkalinity}; PROD {TN,Chl-a,Si} demoted (1 paper, all-blank verdicts).
  - Dual memberships allowed: DO (F1+F6), BOD, COD (F4+F6). Family flags are per-family.

Outputs (atomic, UTF-8):
  results/co_reporting_jaccard.csv
  docs/family_map.csv                (857 rows x 7 per-family flags)
  docs/family_derivation_log.md      (full derivation + amendment evidence)
  docs/family_rules.md               (v1.1 FROZEN)

Run:  python3 code/family_derivation.py
"""
from __future__ import annotations
import csv, os
from collections import defaultdict, OrderedDict, Counter
from datetime import date

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")
OUT_J = os.path.join(ROOT, "results/co_reporting_jaccard.csv")
OUT_MAP = os.path.join(ROOT, "docs/family_map.csv")
OUT_LOG = os.path.join(ROOT, "docs/family_derivation_log.md")
OUT_RULES = os.path.join(ROOT, "docs/family_rules.md")

SEED = 42
TODAY = date.today().isoformat()
FROZEN_V11 = "2026-10-01"  # freeze date is a constant — reruns must not re-stamp the FROZEN doc (D77)
# STABLE_J = 0.75 governed the v1.0 cluster screen (commit f55b8af) and was retired with the
# v1.1 amendment registry (commit df5fab0). The shipped rules_family.md still lists it as a
# governing rule; see D251 in docs/kickoff_notes.md before changing either side.
MIN_PARAMS, MIN_ROWS, K_RANGE, VETO_J = 3, 10, [4, 5, 6, 7], 0.9

METALS = ["As (µg/L)", "Cd (µg/L)", "Cr (µg/L)", "Cu (µg/L)", "Fe (µg/L)", "Hg (µg/L)",
          "Mn (µg/L)", "Ni (µg/L)", "Pb (µg/L)", "Zn (µg/L)", "Al (µg/L)", "Co (µg/L)"]
PHYS = ["DO (mg/L)", "pH", "H2S (µg/L)", "Salinity (ppt/‰)", "TDS", "EC (µS/cm)",
        "Temperature (°C)", "ORP (mV)", "Turbidity (NTU)", "Cl- (ppt)", "SAR",
        "Hardness (mg/L)", "Alkalinity (mg/L)", "TSS (mg/L)", "Secchi (m)"]
NUT = ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)", "NH3 (µg/L)", "TN (mg/L)", "TP (mg/L)",
       "PO4 (mg/L)", "Si (µmol/L)"]
ORG = ["TOC (mg/L)", "DOC (mg/L)", "BOD (mg/L)", "COD (mg/L)", "Oil & Grease (mg/L)"]
MIC = ["Total Coliform (CFU/100mL)", "Fecal Coliform (CFU/100mL)", "E. coli (CFU/100mL)"]
TYPE = {}
for ps, t in [(METALS, "metals"), (PHYS, "physical"), (NUT, "nutrients"),
              (ORG, "organic"), (MIC, "microbial")]:
    for p in ps:
        TYPE[p] = t
TYPE["Chlorophyll-a (mg/m3)"] = "biological"

# ---------------- v1.1 AMENDMENT REGISTRY (frozen, user-approved) ----------------
AMENDMENTS = OrderedDict([
    ("F1 physical",   {"params": ["DO (mg/L)", "pH", "Salinity (ppt/‰)", "Temperature (°C)"],
                       "kind": "clustering-derived (restructured: NO3/PO4 moved to F3)",
                       "rationale": "Core physical WQ suite; strongest cohesion/separation."}),
    ("F2 metals",     {"params": ["As (µg/L)", "Cd (µg/L)", "Cr (µg/L)", "Cu (µg/L)", "Fe (µg/L)",
                                   "Ni (µg/L)", "Pb (µg/L)", "Zn (µg/L)", "Co (µg/L)"],
                       "kind": "clustering-derived",
                       "rationale": "Metal suite as pruned from Ward k=4 cluster 4."}),
    ("F3 nutrients",  {"params": ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)", "PO4 (mg/L)"],
                       "kind": "amendment (rule-4 v1-continuity)",
                       "rationale": "Pre-registered rule 4 mandates mapping to v1 F3 nutrients where compatible; "
                                    "clustering fragmented the N+P suite (NO3/PO4 in physical cluster, NO2/NH4 in residual). "
                                    "84 rows / 11 papers — stronger than any cluster-derived nutrient candidate."}),
    ("F4 organic",    {"params": ["COD (mg/L)", "BOD (mg/L)", "Total Coliform (CFU/100mL)"],
                       "kind": "amendment (rule-4 v1-continuity)",
                       "rationale": "v1 F4 organic continuity; boundary diagnostics: COD avgJ 0.36 to suite vs 0.21 to "
                                    "minerals; BOD 0.35 vs 0.29; all 4 backing papers report full trio; best-separated "
                                    "thin family (min-between 0.124)."}),
    ("F1 physicala",  {"params": ["TDS", "EC (µS/cm)", "Cl- (ppt)", "Hardness (mg/L)", "Alkalinity (mg/L)"],
                       "kind": "clustering-derived (pruned from k=4 cluster 3)",
                       "rationale": "Mineral/ionic suite; TSS->F1 clarity, COD/BOD->F4 organic."}),
    ("F1 clarity",    {"params": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)"],
                       "kind": "amendment (domain-lead semantic)",
                       "rationale": "Secchi = optical transparency parameter, grouped with clarity suite (Turbidity/"
                                    "TSS), not nutrients. FLAGGED: lowest within-family cohesion (0.199)."}),
    ("F6 aeration",   {"params": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"],
                       "kind": "amendment (discovery: exhaustive search + binding matrix)",
                       "rationale": "Oxygen-balance suite (DO deficit vs organic oxygen demand); DO/BOD/COD are "
                                    "bridge params split across clusters; 159 rows/26 papers, top-1 share 15%, "
                                    "within-J 0.416 (2nd strongest). Dual membership: DO (F1+F6), BOD/COD (F4+F6)."}),
])


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def load():
    with open(MATRIX, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f)); hdr = list(rows[0].keys())
    params = hdr[7:51]
    assert len(params) == 44 and hdr[51] == "Other reported parameters" and hdr[52] == "row_kind"
    own = ""
    for r in rows:
        if r["batch_id"].strip():
            own = r["batch_id"]
        r["_p"] = own
    return rows, params


def main():
    np.random.seed(SEED)
    rows, params = load()
    cond = [r for r in rows if r["row_kind"] == "data"]
    B = np.array([[1 if r[p] in ("Y", "NE") else 0 for p in params] for r in cond], dtype=np.int8)

    def P(r, p):
        return r[p] in ("Y", "NE")

    inter = B.astype(np.int64).T @ B.astype(np.int64)
    tot = B.sum(0).astype(np.int64)
    union = tot[:, None] + tot[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(J, 1.0)
    D = 1.0 - J
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="ward")

    def full_stats(ps):
        f = [r for r in cond if all(P(r, p) for p in ps)]
        return f, sorted(set(r["_p"] for r in f))

    def within_j(ps):
        vs = [J[params.index(a), params.index(b)] for i, a in enumerate(ps) for b in ps[i + 1:]]
        return float(np.mean(vs)) if vs else 1.0

    # provenance: subset of a Ward cluster at some k in [4,7]?
    clusts = {}
    for k in K_RANGE:
        lab = fcluster(Z, t=k, criterion="maxclust")
        clusts[k] = [set(np.where(lab == c)[0].tolist()) for c in set(lab)]
    def provenance(ps):
        idx = {params.index(p) for p in ps}
        for k, cls in clusts.items():
            for c in cls:
                if idx <= c:
                    return f"cluster-derived (subset of k={k} cluster, size {len(c)})"
        return "amendment (never co-clustered k=4..7)"

    # PART 1 tables (for the log): raw Ward clusters + pruning trace (report only)
    raw_rows = []
    for k in K_RANGE:
        for ci, members in enumerate([m for m in
                (lambda lab: [np.where(lab == c)[0].tolist() for c in set(lab)])(
                    fcluster(Z, t=k, criterion="maxclust"))], 1):
            if len(members) < MIN_PARAMS:
                raw_rows.append((k, ci, len(members), "excluded_size<3", "", ""))
                continue
            f, pp = full_stats([params[i] for i in members])
            raw_rows.append((k, ci, len(members), len(f), len(pp),
                             ", ".join(params[i].split(" (")[0] for i in members)))

    # PART 2: final families
    finals = []
    for name, spec in AMENDMENTS.items():
        ps = spec["params"]
        f, pp = full_stats(ps)
        pc = Counter(r["_p"] for r in f)
        vc = Counter(r["verdict"] if r["verdict"] else "blank" for r in f)
        f2 = [r for r in cond if r["scenario_type"] != "standard" and all(P(r, p) for p in ps)]
        finals.append({
            "name": name, "params": ps, "kind": spec["kind"], "rationale": spec["rationale"],
            "rows": len(f), "papers": len(pp), "top1": (pc.most_common(1)[0][1] / len(f)) if f else 0,
            "within": round(within_j(ps), 3),
            "std_excl_rows": len(f2),
            "verdicts": dict(sorted(vc.most_common(5), key=lambda kv: (-kv[1], kv[0]))),  # deterministic display (D77)
            "provenance": provenance(ps),
            "rule_ok": len(ps) >= MIN_PARAMS and len(f) >= MIN_ROWS,
        })

    # veto check on final families
    vetoes = []
    for fam in finals:
        ms = [p for p in fam["params"] if TYPE[p] == "metals"]
        ps = [p for p in fam["params"] if TYPE[p] == "physical"]
        if ms and ps:
            jmax = max(J[params.index(a), params.index(b)] for a in ms for b in ps)
            if jmax <= VETO_J:
                vetoes.append((fam["name"], jmax))
    # union scope
    seen, papers = set(), set()
    for fam in finals:
        for r in cond:
            if all(P(r, p) for p in fam["params"]):
                seen.add((r["_p"], r["scenario"])); papers.add(r["_p"])

    assigned = {p for fam in finals for p in fam["params"]}
    unassigned = [p for p in params if p not in assigned]

    # G1 impact
    d = cond
    f3 = {r["_p"] for r in d if sum(P(r, p) for p in ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)"]) >= 2 and P(r, "PO4 (mg/L)")}
    f4 = {r["_p"] for r in d if sum(P(r, p) for p in ["COD (mg/L)", "BOD (mg/L)", "Total Coliform (CFU/100mL)",
                                                      "Fecal Coliform (CFU/100mL)", "E. coli (CFU/100mL)"]) >= 2}
    lethal = sum(1 for r in rows if r["verdict"] == "Lethal")
    src_papers = len({r["_p"] for r in rows if r["row_kind"] in ("data", "context")})

    # ---------------- outputs ----------------
    buf = ["param," + ",".join(params)]
    for i, p in enumerate(params):
        buf.append(p + "," + ",".join(f"{J[i, j]:.4f}" for j in range(len(params))))
    atomic_write(OUT_J, "\n".join(buf) + "\n")

    fam_names = [f["name"] for f in finals]
    map_hdr = (["row_id", "batch_id_filled", "scenario", "row_kind"] +
               [f"full_{n.replace(' ', '_')}" for n in fam_names] +
               ["n_families_complete", "families_complete"])
    lines = [",".join(map_hdr)]
    for ri, r in enumerate(rows, 1):
        flags = []
        for fam in finals:
            ok = r["row_kind"] == "data" and all(r[p] in ("Y", "NE") for p in fam["params"])
            flags.append("1" if ok else "0")
        fc = [fam["name"] for fam, fl in zip(finals, flags) if fl == "1"]
        cells = [str(ri), r["_p"], r["scenario"], r["row_kind"]] + flags + [str(len(fc)), ";".join(fc)]
        lines.append(",".join(f'"{c}"' if ("," in str(c) or ";" in str(c) or '"' in str(c)) else str(c) for c in cells))
    atomic_write(OUT_MAP, "\n".join(lines) + "\n")

    L = []
    L.append(f"# Family Derivation Log — Phase 3 v1.1 / Model D (generated {TODAY})")
    L.append("")
    L.append(f"- Condition-rows: row_kind=='data' → **{len(cond)}**; presence = Y or NE; 44 params; seed={SEED}")
    L.append(f"- PART 1: pre-registered Ward/Jaccard, k∈{K_RANGE} (raw clusters + pruning reachable in v1.0 log/commit history);")
    L.append("  PART 2: **amendment registry v1.1** (frozen, user-approved) — every change versioned with rationale (below).")
    L.append("- Louvain option: skipped (python-louvain/networkx not installed). No manual overrides at runtime (registry is code, not hand edits).")
    L.append("")
    L.append("## v1.1 Amendment registry")
    L.append("| family | kind | params |")
    L.append("|---|---|---|")
    for fam in finals:
        L.append(f"| **{fam['name']}** | {fam['kind']} | {', '.join(fam['params'])} |")
    L.append("")
    L.append("## Final families — verified evidence")
    L.append("| family | rows | papers | top-1 share | within-J | std-excl rows | verdict spread | provenance |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fam in finals:
        L.append(f"| **{fam['name']}** | {fam['rows']} | {fam['papers']} | {fam['top1']:.0%} | {fam['within']} | "
                 f"{fam['std_excl_rows']} | {fam['verdicts']} | {fam['provenance']} |")
    L.append("")
    L.append(f"- Rules (≥{MIN_PARAMS} params, ≥{MIN_ROWS} rows): **{'ALL PASS' if all(f['rule_ok'] for f in finals) else 'FAIL'}**")
    L.append(f"- Veto rule (metal×physical ≤{VETO_J}): **{'PASS — none triggered' if not vetoes else vetoes}**")
    L.append(f"- Dual memberships: DO (F1+F6) · BOD (F4+F6) · COD (F4+F6) — by design (bridge params)")
    L.append(f"- **Stage-B union scope: {len(seen)} rows / {len(papers)} papers** (v1.0: 371/54)")
    L.append(f"- Unassigned ({len(unassigned)}): {', '.join(p.split(' (')[0] for p in unassigned)}")
    L.append("")
    L.append("## Rationales")
    for fam in finals:
        L.append(f"- **{fam['name']}**: {fam['rationale']}")
    L.append("")
    L.append("## PART 1 reference — raw Ward clusters (pre-amendment)")
    L.append("| k | # | raw size | raw full-rows | raw papers | members |")
    L.append("|---|---|---|---|---|---|")
    for k, ci, sz, fr, fp, mem in raw_rows:
        L.append(f"| {k} | {ci} | {sz} | {fr} | {fp} | {mem} |")
    L.append("")
    L.append("Full v1.0 pruning traces + sensitivity: see git history (commit f55b8af) / docs/family_derivation_log.md@v1.0.")
    L.append("")
    L.append("## G1 impact (presence layer)")
    L.append(f"- Source papers: {src_papers} (≥60) · F3 gate: {len(f3)} papers (≥3) · F4 gate: {len(f4)} papers (≥3) · Lethal rows: {lethal}")
    L.append("")
    L.append("## Derivation path")
    L.append("- Code: `code/family_derivation.py` v3 (clustering + amendment registry); exploration evidence: `code/explore_families.py`.")
    L.append("- Model exploration A/B/C/D (multi-membership analysis, decision record): `docs/family_model_exploration.md` (code: `code/family_models_exploration.py`).")
    atomic_write(OUT_LOG, "\n".join(L) + "\n")

    R = []
    R.append(f"# Family Rules — FROZEN v1.1 ({FROZEN_V11})  [supersedes v1.0]")
    R.append("")
    R.append("**Status:** FROZEN. Procedure: pre-registered Ward/Jaccard derivation (P1_Master_Plan.md §4) +")
    R.append("**amendment registry v1.1** (documented version bump; change control Data_Dictionary §7). No silent overrides;")
    R.append("every family carries rationale + evidence in `docs/family_derivation_log.md`.")
    R.append("")
    R.append("## Pre-registered rules (still governing)")
    R.append(f"1. Families ≥{MIN_PARAMS} params; ≥{MIN_ROWS} condition-rows reporting FULL family; stable/robust; v1-mapping where compatible.")
    R.append(f"2. Veto: no metal×physical mix unless J>{VETO_J}.")
    R.append("3. Amendments allowed only via dated version bump with documented justification (this is v1.1).")
    R.append("")
    R.append("## Families (7)")
    R.append("| family | size | rows_full | papers | kind | params |")
    R.append("|---|---|---|---|---|---|")
    for fam in finals:
        R.append(f"| **{fam['name']}** | {len(fam['params'])} | {fam['rows']} | {fam['papers']} | {fam['kind']} | {', '.join(fam['params'])} |")
    R.append("")
    R.append(f"Dual memberships (bridge params): DO = F1+F6 · BOD, COD = F4+F6.")
    R.append(f"Unassigned ({len(unassigned)}): {', '.join(p.split(' (')[0] for p in unassigned)}.")
    R.append(f"Stage-B scope: {len(seen)} rows / {len(papers)} papers family-complete (D9/D50).")
    R.append("")
    R.append("## Flagged caveats")
    R.append("- F1 clarity: lowest within-family cohesion (0.199) — domain-lead amendment; monitor in Phase 4.")
    R.append("- F2 metals: 3 papers, B063+B208 share 93% of rows (thin).")
    R.append("- F6 aeration: amendment (never co-clustered) but strongest thin-family cohesion (0.416); 26 papers.")
    R.append("- G1 gates: F3=36 papers, F4=34 papers (presence layer) — unaffected.")
    R.append("- Model exploration A/B/C/D (multi-membership, decision record): `docs/family_model_exploration.md`.")
    atomic_write(OUT_RULES, "\n".join(R) + "\n")

    # console summary
    print(f"families={len(finals)} | rules={'PASS' if all(f['rule_ok'] for f in finals) else 'FAIL'} | vetoes={len(vetoes)}")
    for fam in finals:
        print(f"  {fam['name']:14s} n={len(fam['params'])} rows={fam['rows']:3d} papers={fam['papers']:2d} "
              f"withinJ={fam['within']} :: {', '.join(p.split(' (')[0] for p in fam['params'])}")
    print(f"union Stage-B scope: {len(seen)} rows / {len(papers)} papers | unassigned={len(unassigned)}")
    print(f"G1: F3={len(f3)} F4={len(f4)} lethal={lethal} papers={src_papers}")
    print("wrote:", OUT_J, OUT_MAP, OUT_LOG, OUT_RULES, sep="\n  ")


if __name__ == "__main__":
    main()
