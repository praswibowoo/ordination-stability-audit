#!/usr/bin/env python3
"""composition_exact_subset.py — exact-configuration subset block factor.

Thin wrapper around wp3_composition_extra.main(), which also writes the direct
composition-vs-density contrast.  Reads
results/wp3_primary/composition_matched_control.csv.
"""
from wp3_composition_extra import main

if __name__ == "__main__":
    main()
