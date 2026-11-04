#!/usr/bin/env python3
"""dedup_scan.py — systematic duplicate scan of the released corpus (D109).

Checks: (1) exact/normalized DOI duplicates; (2) title token-Jaccard pairs >= 0.5;
(3) cross-paper identical-value fingerprints (param, value_as_published) — the signal
that found B111/B175; (4) overlap with the v1 H-registry.
Positive control: B111/B175 must be flagged by (3). Flags only — no corpus mutation.
Output: results/dedup_scan.csv
"""
from __future__ import annotations
import csv, itertools, os, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, "release/p1_family_pca_data_v1.0")


def norm_doi(d):
    return (d or "").strip().lower().replace("https://doi.org/", "")


def toks(t):
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) > 3}


def main():
    man = list(csv.DictReader(open(os.path.join(REL, "corpus_manifest.csv"), encoding="utf-8-sig")))
    V = list(csv.DictReader(open(os.path.join(REL, "values.csv"), encoding="utf-8-sig")))
    flags = []

    # (1) DOI duplicates
    by_doi = defaultdict(list)
    for r in man:
        d = norm_doi(r["doi"])
        if d:
            by_doi[d].append(r["batch_id"])
    for d, ids in by_doi.items():
        if len(ids) > 1:
            flags.append(dict(check="doi_duplicate", a=ids[0], b=",".join(ids[1:]),
                              score=1.0, detail=d))
    print("DOI dups:", sum(1 for f in flags if f["check"] == "doi_duplicate"))

    # (2) title overlap
    for a, b in itertools.combinations(man, 2):
        ta, tb = toks(a["title"]), toks(b["title"])
        if not ta or not tb:
            continue
        j = len(ta & tb) / len(ta | tb)
        if j >= 0.5:
            flags.append(dict(check="title_overlap", a=a["batch_id"], b=b["batch_id"],
                              score=round(j, 3),
                              detail=f"{a['title'][:45]} || {b['title'][:45]}"))
    print("title pairs >=0.5:", sum(1 for f in flags if f["check"] == "title_overlap"))

    # (3) distinctive identical-value fingerprints (standard rows EXCLUDED — shared
    #     regulation limits are anchors, not duplication)
    standard_rows = {(r["batch_id"], r["row_id"]) for r in V
                     if "standard" in (r["scenario"] or "").lower() or
                     r.get("param", "") and r.get("value_as_published", "") and
                     "baku" in (r.get("scenario") or "").lower()}
    fp = defaultdict(set)
    for r in V:
        if (r["batch_id"], str(r.get("row_id", ""))) in standard_rows:
            continue
        v = (r["value_as_published"] or "").strip()
        if not v:
            continue
        key = v.replace(",", ".")
        try:
            key = f"{float(key):.6g}"
        except ValueError:
            pass
        if ("." in key and len(key.split(".")[-1]) >= 2) or "-" in key or "\u00b1" in v or "\u00b1" in v:
            fp[(r["param"], key)].add(r["batch_id"])
    shared = defaultdict(lambda: [0, 0])
    for (param, v), papers in fp.items():
        if len(papers) < 2:
            continue
        for a, b in itertools.combinations(sorted(papers), 2):
            shared[(a, b)][0] += 1
    for (a, b), (n, _) in sorted(shared.items(), key=lambda x: -x[1][0]):
        if n >= 3:
            flags.append(dict(check="value_fingerprint", a=a, b=b, score=n,
                              detail=f"{n} distinctive identical (param,value) anchors (standard rows excluded)"))

    # (3b) range-containment detector (catches pooled-range vs point-value re-publications
    #      like B111/B175 — granularity differs, exact values never match)
    import math
    cells = defaultdict(lambda: defaultdict(list))
    for r in V:
        if (r["batch_id"], str(r.get("row_id", ""))) in standard_rows:
            continue
        try:
            v = float(r["value"]) if (r["value"] or "").strip() else None
            vm = float(r["value_max"]) if (r["value_max"] or "").strip() else None
        except ValueError:
            continue
        if v is None:
            continue
        cells[r["batch_id"]][r["param"]].append((v, vm))
    papers = sorted(cells)
    for a, b in itertools.combinations(papers, 2):
        contain_params = 0
        points_covered = 0
        for p, va in cells[a].items():
            if p not in cells[b]:
                continue
            ranges = [(lo, hi or lo) for lo, hi in va if hi is not None and hi != lo]
            pts = [x for x, hi in cells[b].get(p, []) if hi is None]
            if len(ranges) >= 1 and len(pts) >= 3:
                if all(any(lo - 1e-9 <= pt <= hi + 1e-9 for lo, hi in ranges) for pt in pts):
                    contain_params += 1
                    points_covered += len(pts)
        if contain_params >= 3 and points_covered >= 6:
            flags.append(dict(check="range_containment", a=a, b=b, score=contain_params,
                              detail=f"{contain_params} params: pooled ranges in one contain ALL {points_covered} point-values of the other (pooled-vs-monthly re-publication pattern)"))

    # (4) v1 registry overlap
    v1 = list(csv.DictReader(open(os.path.join(ROOT, "docs/h_registry.csv"), encoding="utf-8-sig")))
    v1_dois = {norm_doi(r.get("doi")) for r in v1 if r.get("doi")}
    man_dois = {norm_doi(r["doi"]): r["batch_id"] for r in man if r["doi"]}
    ov = sorted(set(v1_dois) & set(man_dois))
    for d in ov:
        flags.append(dict(check="v1_registry_overlap", a=man_dois[d], b=d, score=1.0, detail="same DOI as v1 H-paper"))
    print("v1 DOI overlaps:", len(ov))

    out = os.path.join(ROOT, "results/dedup_scan.csv")
    tmp = out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "a", "b", "score", "detail"])
        w.writeheader()
        w.writerows(flags)
    os.replace(tmp, out)
    print(f"flags total: {len(flags)} → results/dedup_scan.csv")
    for f in flags:
        print("  ", f["check"], f["a"], "<->", f["b"], "|", f["score"], "|", f["detail"][:70])


if __name__ == "__main__":
    main()
