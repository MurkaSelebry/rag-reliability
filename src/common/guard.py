"""Защита от утечки: при cloud-профиле на внешний API уходят только синтетические кейсы.

Абсолютное правило проекта (CLAUDE.md): ни одна строка корпуса кураторов
не должна попасть на внешний endpoint. Проверка — до отправки запроса.
"""
from __future__ import annotations

from .schemas import Case


class DataLeakError(RuntimeError):
    """Попытка отправить не-синтетические данные на внешний API."""


def is_synthetic(case: Case) -> bool:
    return case.id.startswith("pseudo_") or bool(case.meta.get("synthetic"))


def assert_case_cloud_safe(case: Case, profile: str) -> None:
    """Guard одного кейса — вызывается LLM-клиентом перед каждым запросом."""
    if profile == "cloud" and not is_synthetic(case):
        raise DataLeakError(
            f"cloud-профиль: кейс {case.id!r} не синтетический (нет префикса pseudo_ "
            f"и флага synthetic) — запрос заблокирован")


def assert_cloud_safe(cases: list[Case], profile: str) -> None:
    """Guard всего файла данных — вызывается скриптами сразу после load_cases."""
    if profile != "cloud":
        return
    bad = [c.id for c in cases if not is_synthetic(c)]
    if bad:
        raise DataLeakError(
            f"cloud-профиль: {len(bad)} не-синтетических кейсов (первые: {bad[:5]}) — стоп")
