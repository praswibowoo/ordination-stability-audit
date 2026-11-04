#!/usr/bin/env python3
"""verify_references.py — registry check for references.bib (E3).

For every bibliography entry that carries a DOI, the DOI is resolved against
the Crossref REST API and the registry record is compared with the local
entry on three fields:

  title    normalised string similarity (difflib ratio, >= 0.90)
  year     registry issued year vs the local year (|diff| <= 1 allowed for
           online-first vs issue year)
  author   first-author family name vs the local first author

An entry that resolves and matches on all three is VERIFIED. An entry that
resolves but differs on any field is MISMATCH (the registry values are
reported so the entry can be corrected). An entry whose DOI does not resolve
is UNRESOLVED. Entries without a DOI are reported as NO-DOI and are not
counted as failures: method-origin papers in journals that do not mint DOIs
(scikit-learn, t-SNE) legitimately have none.

The script is a report, not a gate: it exits 0 even when a mismatch is
found, so it can be run inside a build. It writes
results/references_verification.csv for the numbers checker to pin.

Network: the Crossref polite pool requires a User-Agent with a contact
address. Override with P1_VERIFY_UA="name (mailto:you@example.org)".

Run: python3 code/verify_references.py [--bib references.bib] [--timeout 25]
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
UA = os.environ.get("P1_VERIFY_UA", "p1-family-pca-verify/1.0 (mailto:p1-team@localhost)")
TITLE_MIN = 0.90
YEAR_TOL = 1


def norm_text(s):
    s = re.sub(r"\\[a-zA-Z]+\s*\{?([^{}]*)\}?", r"\1", s)      # drop LaTeX macros
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = s.replace("'", "").replace("`", "")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return " ".join(s.lower().split())


def parse_bib(path):
    entries, key = [], None
    fields = {}
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        m = re.match(r"@(\w+)\{([^,]+),", line)
        if m:
            if key:
                entries.append((key, fields))
            key, fields = m.group(2).strip(), {}
            continue
        if not key:
            continue
        m = re.match(r"\s*(\w+)\s*=\s*\{?(.*?)\}?,?\s*$", line)
        if m:
            fields[m.group(1).lower()] = m.group(2).strip().rstrip(",").strip("{}")
        elif line.strip() == "}" and fields:
            entries.append((key, fields))
            key, fields = None, {}
    if key:
        entries.append((key, fields))
    return entries


def url_ok(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return fh.status < 400
    except urllib.error.HTTPError as e:
        return e.code in (401, 403)
    except Exception:
        return False


def first_author_family(author_field):
    if not author_field:
        return ""
    first = re.split(r"\s+and\s+", author_field)[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    parts = first.split()
    return parts[-1] if parts else ""


def resolves(doi, timeout, ua=None):
    """Does the DOI resolve? (redirect chain followed; a publisher bot-block
    still counts as resolution, a 404 at doi.org does not)."""
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": ua or UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return fh.status in (200, 403)
    except urllib.error.HTTPError as e:
        return e.code in (301, 302, 303, 307, 403)
    except Exception:
        return False


def fetch(doi, timeout):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)["message"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default=os.path.join(ROOT, "references.bib"))
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.25, help="politeness delay")
    args = ap.parse_args()

    entries = parse_bib(args.bib)
    rows, n_verified, n_mismatch, n_unresolved, n_nodoi, n_resolves_only = [], 0, 0, 0, 0, 0
    for key, f in entries:
        doi = (f.get("doi") or "").strip()
        title = (f.get("title") or "").strip()
        year = f.get("year", "")
        author = (f.get("author") or "").strip()
        if not doi:
            url = (f.get("url") or "").strip()
            if url:
                ok_url = url_ok(url, args.timeout)
                n_nodoi += 1
                rows.append(dict(key=key, doi="", status="URL-OK" if ok_url else "URL-FAILED",
                                 title=title, year=year, registry_title="", registry_year="",
                                 registry_author="", title_similarity="", note=url))
            else:
                n_nodoi += 1
                rows.append(dict(key=key, doi="", status="NO-DOI", title=title,
                                 year=year, registry_title="", registry_year="",
                                 registry_author="", title_similarity="", note=""))
            continue
        try:
            msg = fetch(doi, args.timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404 and resolves(doi, args.timeout):
                n_resolves_only += 1
                rows.append(dict(key=key, doi=doi, status="RESOLVES-NOT-IN-CROSSREF",
                                 title=title, year=year, registry_title="",
                                 registry_year="", registry_author="",
                                 title_similarity="", note="resolves at doi.org; no Crossref record"))
            else:
                n_unresolved += 1
                rows.append(dict(key=key, doi=doi, status="UNRESOLVED", title=title,
                                 year=year, registry_title="", registry_year="",
                                 registry_author="", title_similarity="", note=f"HTTP {e.code}"))
            time.sleep(args.sleep)
            continue
        except Exception as e:                                   # noqa: BLE001
            n_unresolved += 1
            rows.append(dict(key=key, doi=doi, status="UNRESOLVED", title=title,
                             year=year, registry_title="", registry_year="",
                             registry_author="", title_similarity="", note=type(e).__name__))
            time.sleep(args.sleep)
            continue
        reg_title = (msg.get("title") or [""])[0]
        parts = (msg.get("issued") or {}).get("date-parts") or [[""]]
        reg_year = parts[0][0]
        reg_author = ""
        for a in msg.get("author") or []:
            if a.get("family"):
                reg_author = a["family"]
                break
        sim = difflib.SequenceMatcher(None, norm_text(title), norm_text(reg_title)).ratio()
        year_ok = (str(reg_year) == str(year)) or (abs(int(reg_year) - int(year)) <= YEAR_TOL
                                                  if str(year).isdigit() and str(reg_year).isdigit() else False)
        author_ok = (norm_text(first_author_family(author)) in norm_text(reg_author)
                     or norm_text(reg_author) in norm_text(first_author_family(author)))
        ok = sim >= TITLE_MIN and year_ok and author_ok
        note = ""
        if not ok:
            n_mismatch += 1
            bits = []
            if sim < TITLE_MIN:
                bits.append(f"title similarity {sim:.2f}")
            if not year_ok:
                bits.append(f"year {year} vs {reg_year}")
            if not author_ok:
                bits.append(f"author {first_author_family(author)!r} vs {reg_author!r}")
            note = "; ".join(bits)
        else:
            n_verified += 1
        rows.append(dict(key=key, doi=doi, status="VERIFIED" if ok else "MISMATCH",
                         title=title, year=year, registry_title=reg_title,
                         registry_year=reg_year, registry_author=reg_author,
                         title_similarity=f"{sim:.3f}", note=note))
        time.sleep(args.sleep)

    out = os.path.join(RESULTS, "references_verification.csv")
    os.makedirs(RESULTS, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)

    print(f"[verify-references] {len(entries)} entries | verified {n_verified} | "
          f"mismatch {n_mismatch} | unresolved {n_unresolved} | "
          f"resolves-not-in-crossref {n_resolves_only} | no-DOI/URL {n_nodoi}")
    for r in rows:
        if r["status"] in ("MISMATCH", "UNRESOLVED", "URL-FAILED"):
            print(f"  {r['status']:10s} {r['key']:28s} {r['doi']:40s} {r['note']}")
    for r in rows:
        if r["status"] == "NO-DOI":
            print(f"  NO-DOI      {r['key']:28s} (no DOI minted by the venue)")
    print(f"  wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    import urllib.parse
    raise SystemExit(main())
