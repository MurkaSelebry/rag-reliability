#!/usr/bin/env python
"""Групповые сплиты train/val/test из корпуса (данные платформы не пересоздаёт).

Тонкая обёртка над `rag_reliability_m3m6.data.make_splits` — вся логика и аргументы CLI там.
"""

from rag_reliability_m3m6.data.make_splits import main

if __name__ == "__main__":
    main()
