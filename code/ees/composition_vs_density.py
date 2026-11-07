#!/usr/bin/env python3
"""composition_vs_density.py — direct composition-vs-density contrast.

Thin wrapper around wp3_composition_extra.main(), which also writes the exact
subset table.  Reads results/wp3_primary/composition_matched_control.csv.
"""
from wp3_composition_extra import main

if __name__ == "__main__":
    main()
