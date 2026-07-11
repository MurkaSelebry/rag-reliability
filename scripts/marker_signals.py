#!/usr/bin/env python
"""Согласованность маркеров судьи с разметкой кураторов.

Тонкая обёртка над `rag_reliability.analysis.marker_signals` — вся логика и аргументы CLI там.
"""

from rag_reliability.analysis.marker_signals import main

if __name__ == "__main__":
    main()
