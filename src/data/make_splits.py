"""Provisional-сплиты реального корпуса кураторов.

ВНИМАНИЕ: сплиты ВРЕМЕННЫЕ — они замещаются платформенными каноническими
сплитами (`data/processed/dev_*.jsonl` с общей платформы команды), как только
те появятся. Модуль нужен для ранней отладки методов на реальном корпусе.

Режимы:
- ``group`` (дефолт): group-aware 80/10/10 — кейсы с одинаковым
  нормализованным запросом клиента попадают в ОДИН сплит (защита от утечки:
  почти-дубликатные диалоги делят ответы).
- ``curator``: точное воспроизведение бейзлайн-сплита кураторов —
  sklearn ``train_test_split(test_size=0.2, random_state=seed,
  stratify=reliable)``, без val — только для сопоставимости с их ноутбуком.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from src.common.schemas import Case

_SPLIT_ORDER = ("train", "val", "test")
_TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}

_NON_ALNUM_RE = re.compile(r"[^\w ]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def group_key(case: Case) -> str:
    """Нормализованный запрос клиента: lower, без пунктуации, схлопнутые пробелы."""
    s = case.query.lower().replace("ё", "е")
    s = _NON_ALNUM_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s if s else case.id  # пустой запрос → синглтон-группа по id


def _dedup(cases: list[Case]) -> list[Case]:
    """Схлопывает точные дубликаты по (полный текст диалога, ответ)."""
    seen: set[tuple[str, str]] = set()
    kept: list[Case] = []
    for c in cases:
        key = ("\n".join(c.dialog) if c.dialog else c.query, c.answer)
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    dropped = len(cases) - len(kept)
    if dropped:
        print(f"[make_splits] отброшено точных дубликатов: {dropped}")
    return kept


def _split_group_aware(cases: list[Case], seed: int) -> dict[str, list[str]]:
    """Детерминированное жадное распределение групп в train/val/test ~80/10/10."""
    cases = _dedup(cases)
    groups: dict[str, list[Case]] = {}
    for c in cases:
        groups.setdefault(group_key(c), []).append(c)

    n = len(cases)
    targets = {name: _TARGETS[name] * n for name in _SPLIT_ORDER}
    sizes = {name: 0 for name in _SPLIT_ORDER}
    result: dict[str, list[str]] = {name: [] for name in _SPLIT_ORDER}

    # сортировка групп: крупные первыми, тай-брейк по минимальному id — детерминизм
    ordered = sorted(groups.values(), key=lambda g: (-len(g), min(c.id for c in g)))
    for grp in ordered:
        # жадно: в сплит с наибольшим дефицитом; при равенстве — train > val > test
        best = max(
            _SPLIT_ORDER,
            key=lambda name: (targets[name] - sizes[name], -_SPLIT_ORDER.index(name)),
        )
        sizes[best] += len(grp)
        result[best].extend(sorted(c.id for c in grp))
    return result


def _split_curator(cases: list[Case], seed: int) -> dict[str, list[str]]:
    """Бейзлайн кураторов: 80/20 stratify(reliable), без val, без дедупа и групп."""
    from sklearn.model_selection import train_test_split  # ленивый импорт

    ids = [c.id for c in cases]
    rel_y = [int(bool(c.faith) and bool(c.rel)) for c in cases]
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=seed, stratify=rel_y)
    return {"train": list(train_ids), "val": [], "test": list(test_ids)}


def make_splits(cases: list[Case], seed: int = 42, mode: str = "group") -> dict[str, list[str]]:
    """Возвращает {"train": [ids], "val": [ids], "test": [ids]} (curator: val=[])."""
    if mode == "group":
        return _split_group_aware(cases, seed)
    if mode == "curator":
        return _split_curator(cases, seed)
    raise ValueError(f"неизвестный режим сплита: {mode!r}")


def _write_split(cases: list[Case], ids: list[str], path: Path) -> None:
    """Пишет jsonl в формате, совместимом с src.common.schemas.load_cases."""
    by_id = {c.id: c for c in cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps(asdict(by_id[cid]), ensure_ascii=False) + "\n")


def _print_stats(name: str, cases: list[Case], ids: list[str]) -> None:
    by_id = {c.id: c for c in cases}
    subset = [by_id[cid] for cid in ids]
    n = len(subset)
    if n == 0:
        print(f"  {name}: n=0")
        return
    faith = sum(int(bool(c.faith)) for c in subset) / n
    rel = sum(int(bool(c.rel)) for c in subset) / n
    reliable = sum(int(bool(c.faith) and bool(c.rel)) for c in subset) / n
    print(f"  {name}: n={n}, reliable={reliable:.1%}, faith={faith:.1%}, rel={rel:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provisional-сплиты корпуса кураторов")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["group", "curator"], default="group")
    parser.add_argument("--limit", type=int, default=None, help="взять первые N кейсов")
    args = parser.parse_args()

    from src.common.config import load_config
    from src.data.alfa_loader import load_alfa

    cfg = load_config(args.config)
    seed = cfg["alfa"].get("seed", 42)
    cases = load_alfa(cfg["alfa"]["raw_csv"])
    if args.limit:
        cases = cases[: args.limit]

    splits = make_splits(cases, seed=seed, mode=args.mode)

    out_dir = Path("data/processed")
    if args.mode == "group":
        paths = {name: out_dir / f"alfa_dev_{name}.jsonl" for name in _SPLIT_ORDER}
    else:
        paths = {name: out_dir / f"alfa_curator_{name}.jsonl" for name in ("train", "test")}

    print(f"[make_splits] mode={args.mode}, seed={seed}, кейсов: {len(cases)}")
    for name, path in paths.items():
        _write_split(cases, splits[name], path)
        _print_stats(name, cases, splits[name])
        print(f"  → {path}")


if __name__ == "__main__":
    main()
