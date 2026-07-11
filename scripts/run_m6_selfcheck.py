#!/usr/bin/env python
"""Метод 6, этап 3: калибровка фич на val → predictions/{profile}/m6/.

Тонкая обёртка над `rag_reliability.methods.m6.predict` — вся логика и аргументы CLI там.
"""

from rag_reliability.methods.m6.predict import main

if __name__ == "__main__":
    main()
