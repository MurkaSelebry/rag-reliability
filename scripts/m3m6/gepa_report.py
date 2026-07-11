#!/usr/bin/env python
"""Markdown-отчёт эволюции GEPA из stats-json (после scripts/m3m6/run_gepa.py).

Тонкая обёртка над `rag_reliability_m3m6.methods.m3.gepa_report` — вся логика и аргументы CLI там.
"""

from rag_reliability_m3m6.methods.m3.gepa_report import main

if __name__ == "__main__":
    main()
