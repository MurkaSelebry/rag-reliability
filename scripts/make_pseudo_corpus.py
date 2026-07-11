#!/usr/bin/env python
"""Синтетический псевдо-корпус (SberQuAD) для cloud-отладки, поэлементный кэш.

Тонкая обёртка над `rag_reliability.data.pseudo_corpus` — вся логика и аргументы CLI там.
"""

from rag_reliability.data.pseudo_corpus import main

if __name__ == "__main__":
    main()
