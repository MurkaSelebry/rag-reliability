#!/usr/bin/env python
"""Метод 6: абляция semantic entropy (сетка thr × N) на готовых сэмплах.

Тонкая обёртка над `rag_reliability_m3m6.methods.m6.entropy_ablation` — вся логика и аргументы CLI там.
"""

from rag_reliability_m3m6.methods.m6.entropy_ablation import main

if __name__ == "__main__":
    main()
