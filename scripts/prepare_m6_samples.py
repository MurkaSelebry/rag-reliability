#!/usr/bin/env python
"""Метод 6, этап 1: N сэмплов «бота» на (Q, CTX), поэлементный кэш в artifacts/.

Тонкая обёртка над `rag_reliability.methods.m6.sample` — вся логика и аргументы CLI там.
"""

from rag_reliability.methods.m6.sample import main

if __name__ == "__main__":
    main()
