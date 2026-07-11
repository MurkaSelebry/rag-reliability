#!/usr/bin/env python
"""Метод 3: инференс судьи (zero_shot/few_shot/gepa) → predictions/{profile}/m3/...

Тонкая обёртка над `rag_reliability.methods.m3.predict` — вся логика и аргументы CLI там.
"""

from rag_reliability.methods.m3.predict import main

if __name__ == "__main__":
    main()
