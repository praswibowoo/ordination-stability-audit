#!/usr/bin/env python3
"""goldset_check.py — local pre-flight for T1 gold-set retrieval validation.

What it does (LOCAL proxy, not a Scopus replacement):
  For each v1 paper with a filled title (+ optional abstract) in the H-registry
  CSV, evaluate whether our query bank (Q1-Q9 + Q3b) WOULD match its text.
  Answers: "is the bank logically capable of retrieving this paper?"

Official Scopus/WoS gold-set test still required for final validation
(Scopus indexes full records; this proxy uses title+abstract only).

Usage:
    python code/goldset_check.py docs/h_registry.csv [docs/goldset_validation.csv]

Output:
  - prints per-paper match report
  - writes goldset_validation.csv (h_number,doi,title,retrieved_by_query,miss_flag)
  - exit 0 if >= target, else 1 (target: 90% of papers WITH text filled)

Query cores mirror docs/search_strings.md v1.3 (Q7 redesigned D18; keep in sync manually;
any string change = protocol v1.3 entry).
"""
from __future__ import annotations

import csv
import re
import sys

TARGET_RATIO = 0.90

MANG = ["mangrove*", "mangal", "rhizophora", "avicennia", "sonneratia", "bruguiera", "kandelia"]

QUERY_CORES: dict[str, str] = {
    "Q1": '{MANG} AND ("water quality" OR "physico-chemical" OR physicochemical) AND (pH OR "dissolved oxygen" OR salinity OR turbidity OR conductivity OR "total dissolved solids" OR TDS)',
    "Q2": '{MANG} AND ("heavy metal*" OR metals) AND (arsenic OR cadmium OR chromium OR copper OR iron OR lead OR manganese OR mercury OR nickel OR zinc) AND (water OR "surface water" OR sediment porewater)',
    "Q3": '{MANG} AND (ammonium OR ammonia OR nitrate OR nitrite OR phosphate OR "dissolved inorganic nitrogen" OR "total nitrogen" OR "total phosphorus" OR nutrient*) AND (season* OR site* OR transect* OR station* OR spatial* OR "water column")',
    "Q3b": '{MANG} AND (ammonium OR ammonia OR nitrate OR nitrite OR phosphate OR "dissolved inorganic nitrogen" OR "total nitrogen" OR "total phosphorus" OR nutrient*) AND ("water quality" OR water OR estuarine OR sediment)',
    "Q4": '{MANG} AND (BOD OR "biochemical oxygen demand" OR COD OR "chemical oxygen demand" OR coliform* OR TSS OR "total suspended solids") AND (water OR estuarine OR "water quality")',
    "Q5": '{MANG} AND (chlorophyll-a OR "chlorophyll a" OR eutrophication OR "trophic state" OR "trophic status" OR "algal bloom") AND ("water quality" OR nutrient* OR water)',
    "Q6": '{MANG} AND "water quality" AND (standard* OR guideline* OR regulation OR criteri* OR threshold*)',
    "Q7": '((mangrove OR estuarine OR brackish) AND (shrimp OR prawn OR crab OR fish OR oyster OR mussel OR copepod) AND (LC50 OR LC96 OR lethal OR mortality OR survival*) AND (water OR toxicity OR pollutant* OR salinity OR pH OR "dissolved oxygen"))',
    "Q8": '{MANG} AND (urban* OR "pollution gradient" OR "land use" OR "land-use" OR aquacultur* OR effluent OR sewage) AND "water quality"',
    "Q9": '{MANG} AND ("total organic carbon" OR TOC OR sulfide OR "hydrogen sulfide" OR ORP OR "oxidation reduction potential" OR alkalinity OR "oil and grease" OR SAR OR "sodium adsorption ratio" OR chloride) AND "water quality"',
}


def expand_mang(core: str) -> str:
    arm = "(" + " OR ".join(MANG) + ")"
    return core.replace("{MANG}", arm)


def tokenize(expr: str) -> list[str]:
    # NOTE: [^\s()]+ (not \S+) so parentheses attached to words split correctly
    return re.findall(r'\(|\)|"[^"]*"|[^\s()]+', expr)


class Parser:
    def __init__(self, tokens: list[str], text: str):
        self.tokens = tokens
        self.i = 0
        self.text = text

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self) -> bool:
        return self.expr()

    def expr(self) -> bool:
        val = self.term()
        while self.peek() and self.peek().upper() == "OR":
            self.take()
            val = self.term() or val
        return val

    def term(self) -> bool:
        val = self.factor()
        while True:
            p = self.peek()
            if p is None:
                break
            pu = p.upper()
            if pu == "AND":
                self.take()
                val = self.factor() and val
            elif pu == "OR" or p == ")":
                break
            else:
                # implicit AND: databases auto-AND adjacent terms (e.g. "sediment porewater")
                val = self.factor() and val
        return val

    def factor(self) -> bool:
        t = self.take()
        if t == "(":
            val = self.expr()
            assert self.take() == ")", "unbalanced parentheses"
            return val
        assert t is not None, "unexpected end of expression"
        return self.atom(t)

    def atom(self, t: str) -> bool:
        if t.startswith('"') and t.endswith('"'):
            phrase = t[1:-1].lower()
            # wildcard inside phrase (e.g. "organism*"): treat as prefix on last word
            if phrase.endswith("*"):
                stem = phrase[:-1]
                return stem in self.text
            return phrase in self.text
        if t.endswith("*"):
            stem = re.escape(t[:-1].lower())
            return re.search(rf"\b{stem}\w*", self.text) is not None
        w = re.escape(t.lower())
        # optional plural approximates database lemmatization ("waters" ~ "water")
        return re.search(rf"\b{w}(?:s|es)?\b", self.text) is not None


def matches(core: str, text: str) -> bool:
    expr = expand_mang(core)
    return Parser(tokenize(expr), text).parse()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/goldset_validation.csv"
    rows = list(csv.DictReader(open(src)))

    filled = [r for r in rows if (r.get("title") or "").strip()]
    results, hits = [], 0
    for r in filled:
        text = ((r.get("title") or "") + " " + (r.get("abstract") or "")).lower()
        matched = [q for q, core in QUERY_CORES.items() if matches(core, text)]
        miss = "1" if not matched else "0"
        hits += 0 if miss == "1" else 1
        results.append({
            "h_number": r.get("h_number", "").strip(),
            "doi": r.get("doi", "").strip(),
            "title": r.get("title", "").strip(),
            "retrieved_by_query": "|".join(matched) if matched else "",
            "miss_flag": miss,
        })
        print(f"{r.get('h_number',''):6s} {'MISS' if miss=='1' else 'ok  '} {','.join(matched):18s} {r.get('title','')[:70]}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["h_number", "doi", "title", "retrieved_by_query", "miss_flag"])
        w.writeheader()
        w.writerows(results)

    total = len(filled)
    print(f"\nGold-set (local proxy): {hits}/{total} retrievable "
          f"({(hits/total*100 if total else 0):.0f}%) — target ≥90% of filled rows")
    print(f"Unfilled registry rows: {len(rows)-total} (fill titles to complete the test)")
    print(f"Written: {out}")
    if total == 0:
        print("NOTE: no titles filled yet — fill docs/h_registry.csv first.")
        return 0
    return 0 if hits / total >= TARGET_RATIO else 1


if __name__ == "__main__":
    raise SystemExit(main())
