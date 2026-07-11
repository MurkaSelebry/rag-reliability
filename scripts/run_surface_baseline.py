#!/usr/bin/env python
"""Бейзлайн: surface-эвристики (+опционально e5-эмбеддинги) → predictions/local/baselines/.

Тонкая обёртка над `rag_reliability.baselines.surface` — вся логика и аргументы CLI там.
"""

from rag_reliability.baselines.surface import main

if __name__ == "__main__":
    main()
