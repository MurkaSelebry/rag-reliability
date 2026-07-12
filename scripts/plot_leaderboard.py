#!/usr/bin/env python
"""Render the organizer-corpus reliability leaderboard as a slide-ready PNG."""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# category -> colour
CAT = {
    "cloud": "#8250df",       # m3_openai*
    "rule": "#1f9d55",        # independent
    "local": "#3776ab",       # prompt_*, lora_*, m3_zero/few (MLX)
    "encoder": "#d1495b",     # supervised
    "baseline": "#9aa0a6",    # dummy_*
}


def categorise(name: str) -> str:
    if name.startswith("m3_openai"):
        return "cloud"
    if name == "independent":
        return "rule"
    if name == "encoder":
        return "encoder"
    if name.startswith("dummy"):
        return "baseline"
    return "local"


def main() -> None:
    rows = []
    for f in glob.glob("results/lb/*/metrics.json"):
        name = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        rows.append((name, d["reliable_f1_macro"]))
    rows.sort(key=lambda r: r[1])

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [CAT[categorise(n)] for n in names]

    fig, ax = plt.subplots(figsize=(10, 0.62 * len(rows) + 1.6))
    bars = ax.barh(names, vals, color=colors, height=0.68)
    for b, v in zip(bars, vals):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", ha="left", fontsize=11, fontweight="bold")

    # Alfa reference (different corpus, not directly comparable)
    ax.axvline(0.584, ls=":", lw=1.3, color="#555", alpha=0.7)
    ax.text(0.584, 0.2, " Alfa m3\n 0.584 (n=223)", color="#555", fontsize=8.5, va="bottom", ha="left")

    ax.set_xlim(0.40, 0.62)
    ax.set_xlabel("reliability macro-F1", fontsize=12)
    ax.set_title("RAG reliability — organizer corpus, unified 225-row held-out split",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (CAT["cloud"], CAT["local"], CAT["rule"], CAT["encoder"], CAT["baseline"])]
    ax.legend(handles, ["cloud LLM judge", "local MLX", "rule-based", "supervised encoder", "baseline"],
              loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig("results/lb/leaderboard.png", dpi=160, bbox_inches="tight")
    print("wrote results/lb/leaderboard.png")


if __name__ == "__main__":
    main()
