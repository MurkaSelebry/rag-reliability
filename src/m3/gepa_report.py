"""Дамп эволюции GEPA-промпта: stats-json -> markdown-отчёт.

Запуск:
  python -m src.m3.gepa_report --variant markers --seed 0
  python -m src.m3.gepa_report --stats artifacts/cloud/m3_gepa_stats_markers_seed0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_EXCERPT = 400  # символов с начала и конца кандидата


def _candidate_text(cand) -> str | None:
    """Текст инструкции кандидата из детальной статистики (формат dspy может меняться)."""
    if isinstance(cand, dict):
        for v in cand.values():          # {predictor_name: instruction}
            if isinstance(v, str) and v.strip():
                return v
        return None
    return cand if isinstance(cand, str) else None


def _excerpt(text: str) -> str:
    if len(text) <= 2 * _EXCERPT:
        return text
    return text[:_EXCERPT] + f"\n…[{len(text) - 2 * _EXCERPT} символов пропущено]…\n" + text[-_EXCERPT:]


def main() -> None:
    ap = argparse.ArgumentParser(description="markdown-отчёт эволюции GEPA-промпта")
    ap.add_argument("--variant", choices=["markers", "plain"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--artifacts-dir", default="artifacts/cloud")
    ap.add_argument("--stats", default=None,
                    help="путь к stats-json (иначе строится из --variant/--seed)")
    ap.add_argument("--out", default=None, help="путь к markdown (иначе рядом со stats)")
    args = ap.parse_args()

    if args.stats:
        stats_path = Path(args.stats)
    elif args.variant is not None and args.seed is not None:
        stats_path = Path(args.artifacts_dir) / f"m3_gepa_stats_{args.variant}_seed{args.seed}.json"
    else:
        ap.error("укажи --stats ЛИБО пару --variant и --seed")
    if not stats_path.exists():
        raise SystemExit(f"нет файла статистики: {stats_path} (сначала запусти src.m3.run_gepa)")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    variant, seed = stats.get("variant", "?"), stats.get("seed", "?")
    dr = stats.get("detailed_results") or {}
    scores = dr.get("val_aggregate_scores") or []
    candidates = dr.get("candidates") or []
    best_idx = dr.get("best_idx")

    lines = [
        f"# Эволюция GEPA-промпта — variant={variant}, seed={seed}",
        "",
        f"- auto: `{stats.get('auto')}`, train_size: {stats.get('train_size')}, "
        f"val_size: {stats.get('val_size')}",
        f"- use_marker_feedback: {stats.get('use_marker_feedback')}",
        f"- модели: task `{stats.get('task_model')}`, reflection `{stats.get('reflection_model')}`",
        f"- LM-вызовы: task {stats.get('task_lm_calls')}, reflection {stats.get('reflection_lm_calls')}",
        f"- git: `{stats.get('git_hash')}`, profile: `{stats.get('profile')}`",
        "",
        "## Кандидаты",
        "",
        "| # | val-score | лучший |",
        "|---|---|---|",
    ]
    for i in range(max(len(scores), len(candidates))):
        sc = f"{scores[i]:.3f}" if i < len(scores) and scores[i] is not None else "—"
        lines.append(f"| {i} | {sc} | {'✅' if best_idx == i else ''} |")

    lines += ["", "## Что менялось в инструкции", ""]
    if candidates:
        for i, cand in enumerate(candidates):
            text = _candidate_text(cand)
            lines.append(f"### Кандидат {i}" + (" (лучший)" if best_idx == i else ""))
            lines.append("")
            lines.append("```" if text else "_текст кандидата недоступен в статистике_")
            if text:
                lines += [_excerpt(text), "```"]
            lines.append("")
    else:
        lines += ["_Полные тексты кандидатов недоступны — только скоры выше "
                  "и финальная инструкция ниже._", ""]

    lines += ["## Финальная инструкция", "", "```",
              stats.get("best_instruction", ""), "```", ""]

    out_path = Path(args.out) if args.out else \
        stats_path.parent / f"m3_gepa_report_{variant}_seed{seed}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"отчёт: {out_path}")


if __name__ == "__main__":
    main()
