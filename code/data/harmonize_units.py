#!/usr/bin/env python3
"""harmonize_units.py — Stage-B value parser + unit harmonizer (P1-2.2).

Usage:
  from harmonize_units import parse_cell, harmonize, QA_RULES

  parse_cell("7.2±0.3")      -> {'value': 7.2, 'value_max': None, 'flags': ['sd:0.3']}
  parse_cell("23-28")        -> {'value': 23.0, 'value_max': 28.0, 'flags': ['was_range']}
  parse_cell("<0.001")       -> {'NE': True, 'flags': ['censored:LT']}
  parse_cell("")             -> {}

  harmonize('metals', 0.5, 'mg/L')  -> (500.0, 'U1')   # value, rule stamp

CLI:  python code/data/harmonize_units.py selftest
"""
from __future__ import annotations
import re, sys, csv

# ---------- parsing ----------
NUM = re.compile(r'-?\d+(?:[.,]\d+)?')

def _num(x: str) -> float:
    """Number with Indonesian/European decimal comma support."""
    return float(x.replace(',', '.'))

def parse_cell(s: str) -> dict:
    """Parse a published cell into normalized fields."""
    if s is None: return {}
    s = str(s).strip()
    if s == '': return {}
    out = {}
    low = s.lower()
    if low in ('nd', 'bdl', 'n.d.', 'not detected') or re.fullmatch(r'[<≤]\s*[\d\.,]+', s) or s == '-':
        out['NE'] = True
        m = NUM.search(s)
        if m: out['flags'] = ['censored:LT']
        return out
    if low.startswith('nt') or 'not tested' in low:
        return {'NR_note': s}
    # strip +/- sd
    sd = None
    m = re.match(r'([\d\.,\s]+)\s*(?:±|\+/-)\s*([\d\.,]+)', s)
    if m:
        out['flags'] = [f'sd:{m.group(2).strip()}']
        s = m.group(1)
    # range
    m = re.match(r'\s*(-?[\d\.,]+)\s*(?:-|–|—|to)\s*(-?[\d\.,]+)', s)
    if m:
        out['value'] = _num(m.group(1)); out['value_max'] = _num(m.group(2))
        out.setdefault('flags', []).append('was_range')
        return out
    # value with unit/uncertainty text
    m = NUM.search(s)
    if m:
        out['value'] = _num(m.group(0))
        if s.strip().startswith('>'): out['flags'] = out.get('flags', []) + ['censored:GT']
        if '(' in s: out['flags'] = out.get('flags', []) + ['note:' + re.sub(r'[()]', '', s)[:40]]
        return out
    return {'unparsed': s}

# ---------- unit harmonization ----------
ATOMIC_MASS = {'as': 74.92, 'cd': 112.4, 'cr': 52.0, 'cu': 63.5, 'fe': 55.8, 'hg': 200.6,
               'mn': 54.9, 'ni': 58.7, 'pb': 207.2, 'zn': 65.4, 'al': 26.98, 'co': 58.9}

def harmonize(element: str, value: float, unit: str) -> tuple:
    """Return (normalized_value, rule_id) or (value, None) if identity."""
    u = (unit or '').lower().strip()
    el = element.lower()
    if el in ATOMIC_MASS and u in ('mmol/l', 'µmol/l', 'umol/l', 'mol/l'):
        if u == 'mol/l': value *= 1e6
        elif u == 'mmol/l': value *= 1e3
        return value * ATOMIC_MASS[el] / 1000.0, 'U12'   # µmol -> µg
    if u in ('mg/l',) and el in ATOMIC_MASS:
        return value * 1000.0, 'U1'                      # mg/L -> µg/L
    if u == 'ppm' and el in ATOMIC_MASS:
        return value * 1000.0, 'U1'                      # ppm == mg/L (identity) -> µg/L
    if el == 'salinity' and (u in ('mg/l', 'mg l-1') or (u == 'ppm' and value >= 100)):
        return value / 1000.0, 'U4c'      # mg/L (or freshwater-scale ppm) -> ppt (adjudicated 2026-10-01; small-value ppm left as-printed, e.g. B115 ambiguity)
    if el == 'cl' and u in ('mg/l', 'mg l-1', 'ppm'):
        return value / 1000.0, 'U4d'      # chloride mg/L -> ppt-equivalent (adjudicated 2026-10-01)
    if el == 'hardness' and u in ('°dh', 'odh', 'dh', '°d'):
        return value * 17.85, 'U15'       # German degrees -> mg/L as CaCO3 (adjudicated 2026-10-01)
    if el in ('no2', 'no3', 'nh4') and u in ('µmol/l', 'umol/l', 'µg-at./l', 'ug-at./l'):
        return value * 0.014, 'U16a'      # molar (as N) -> mg N/L (adjudicated 2026-10-01)
    if el == 'po4' and u in ('µmol/l', 'umol/l', 'µg-at./l', 'ug-at./l'):
        return value * 0.095046, 'U16b'   # molar (as P) -> mg PO4/L (×0.031 then P->PO4 ×3.066; adjudicated 2026-10-01)
    if u in ('ms/m',):
        return value * 100.0, 'U2'
    if u in ('ms/cm',):
        return value * 1000.0, 'U2b'      # mS/cm -> µS/cm (RATIFIED 2026-10-02, D83)
    if el == 'salinity' and u in ('g/l',):
        return value, 'U4b'               # g/L ~= ppt (approx, density~1; reserved — not applied to data)
    if u in ('g/l',):
        return value * 1000.0, 'U3'
    if u in ('°f', 'f'):
        return (value - 32) * 5 / 9.0, 'U7'
    if el == 'si' and u in ('mg/l',) and element == 'sio2':
        return value * 33.37, 'U11'
    if el == 'do' and u in ('ml/l', 'ml l-1', 'ml o2/l', 'mll-1'):
        return value * 1.429, 'U13'       # ml O2/L -> mg/L at STP (RATIFIED 2026-10-02, D83)
    if el == 'secchi' and u in ('cm',):
        return value / 100.0, 'U14'       # cm -> m (RATIFIED 2026-10-02, D83)
    if el in ('h2s', 'sulfide', 'sulphide') and u in ('mg/l', 'mg l-1'):
        return value * 1000.0, 'U17'      # H2S mg/L -> µg/L (×1000; registered post-D92 audit — D96)
    return value, None

# ---------- QA ----------
QA_RULES = {
 'DO (mg/L)': (-1, 20), 'pH': (3, 12), 'Temperature (°C)': (-2, 42),
 'Salinity (ppt/‰)': (0, 45), 'Turbidity (NTU)': (0, 4000),
 'Chlorophyll-a (mg/m3)': (0, 500), 'Secchi (m)': (0, 200),
}
def qa(param: str, value) -> str:
    if value is None: return ''
    lo, hi = QA_RULES.get(param, (None, None))
    if lo is not None and (value < lo or value > hi):
        return f'OUT-OF-RANGE[{lo},{hi}]'
    if value > 1000 and re.search(r'\(µg/L\)|\(ug/L\)', param):
        return 'CHECK: metal >1000 µg/L'
    return ''

# ---------- self-test ----------
def selftest() -> int:
    cases = [
        ('7.2', 7.2, None), ('23-28', 23.0, 28.0), ('7.2±0.3', 7.2, None),
        ('<0.001', None, None), ('ND', None, None), ('', None, None),
    ]
    ok = True
    for cell, ev, evm in cases:
        r = parse_cell(cell)
        v = r.get('value'); vm = r.get('value_max')
        good = (v == ev) and (vm == evm)
        ok = ok and good
        print(f"  parse {cell!r:14s} -> {r} {'OK' if good else 'FAIL'}")
    n1, r1 = harmonize('cu', 0.5, 'mg/l'); print(f"  harmonize Cu 0.5 mg/L -> {n1} rule={r1}", 'OK' if (n1 == 500.0 and r1 == 'U1') else 'FAIL')
    n2, r2 = harmonize('cu', 10.0, 'µmol/l'); print(f"  harmonize Cu 10 µmol/L -> {n2:.2f} rule={r2}", 'OK' if r2 == 'U12' else 'FAIL')
    n3, _ = harmonize('air', 77.0, '°f'); print(f"  F->C 77 -> {n3}", 'OK' if abs(n3 - 25.0) < 0.01 else 'FAIL')
    print(f"  QA pH 13 -> {qa('pH', 13)}", 'OK' if qa('pH', 13) else 'FAIL')
    print('selftest', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1

if __name__ == '__main__':
    raise SystemExit(selftest())
