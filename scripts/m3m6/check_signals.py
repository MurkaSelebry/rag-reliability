#!/usr/bin/env python
"""Сигналы работоспособности m3/m6 по типам кейсов (kind).

Тонкая обёртка над `rag_reliability_m3m6.analysis.check_signals` — вся логика и аргументы CLI там.
"""

from rag_reliability_m3m6.analysis.check_signals import main

if __name__ == "__main__":
    main()
