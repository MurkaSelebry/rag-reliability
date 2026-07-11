#!/usr/bin/env python
"""Единый offline HTML-отчёт m3/m6 (plotly) по всем прогонам в predictions/.

Тонкая обёртка над `rag_reliability.analysis.report` — вся логика и аргументы CLI там.
"""

from rag_reliability.analysis.report import main

if __name__ == "__main__":
    main()
