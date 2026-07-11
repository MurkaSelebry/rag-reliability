"""Защита от утечки: при cloud-профиле на внешний API уходят только синтетические кейсы.

Правило проекта (CLAUDE.md): корпус кураторов не уходит на внешний endpoint.
Единственное исключение — ЯВНЫЙ opt-in `guard.allow_real_data: true` в конфиге:
владелец данных разрешил конкретный внешний endpoint для конкретного корпуса
(датасет объявлен открытым). Дефолт всегда блокирующий; opt-in логируется.
"""

from __future__ import annotations

import logging

from rag_reliability_m3m6.common.schemas import Case

_log = logging.getLogger(__name__)
_warned_once = False


class DataLeakError(RuntimeError):
    """Попытка отправить не-синтетические данные на внешний API."""


def is_synthetic(case: Case) -> bool:
    return case.id.startswith("pseudo_") or bool(case.meta.get("synthetic"))


def _warn_opt_in() -> None:
    """Однократное громкое предупреждение об активном opt-in."""
    global _warned_once
    if not _warned_once:
        _log.warning(
            "guard.allow_real_data=true: реальные кейсы уходят на внешний API "
            "по явному разрешению владельца данных (см. комментарий в конфиге)"
        )
        _warned_once = True


def assert_case_cloud_safe(case: Case, profile: str, allow_real: bool = False) -> None:
    """Guard одного кейса — вызывается LLM-клиентом перед каждым запросом."""
    if profile == "cloud" and not is_synthetic(case):
        if allow_real:
            _warn_opt_in()
            return
        raise DataLeakError(
            f"cloud-профиль: кейс {case.id!r} не синтетический (нет префикса pseudo_ "
            f"и флага synthetic) — запрос заблокирован"
        )


def assert_cloud_safe(cases: list[Case], profile: str, allow_real: bool = False) -> None:
    """Guard всего файла данных — вызывается скриптами сразу после load_cases."""
    if profile != "cloud":
        return
    bad = [c.id for c in cases if not is_synthetic(c)]
    if bad:
        if allow_real:
            _warn_opt_in()
            return
        raise DataLeakError(
            f"cloud-профиль: {len(bad)} не-синтетических кейсов (первые: {bad[:5]}) — стоп"
        )
