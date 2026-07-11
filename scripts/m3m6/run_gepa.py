#!/usr/bin/env python
"""Метод 3: GEPA-оптимизация промпта судьи (DSPy), stats-json в artifacts/.

Тонкая обёртка над `rag_reliability_m3m6.methods.m3.run_gepa` — вся логика и аргументы CLI там.
"""

from rag_reliability_m3m6.methods.m3.run_gepa import main

if __name__ == "__main__":
    main()
