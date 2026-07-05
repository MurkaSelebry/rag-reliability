"""tracking: тонкая обёртка над MLflow для локального (file-store) трекинга прогонов.

Только file:-uri, никаких http — правило проекта «никаких внешних вызовов данных».
Конфиг редактируется (api_key → ***) перед логированием параметров.
"""

from __future__ import annotations

import os
from pathlib import Path

# Разрешаем file-store backend (иначе новые версии MLflow бросают исключение).
# Проект принципиально использует только локальный file:-store, без http.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402  # импорт после установки env-флага file-store

from src.common.run_meta import _redact

_MAX_PARAM_LEN = 500  # лимит длины значения параметра MLflow


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Рекурсивно разворачивает вложенный dict в плоские dot-ключи со строковыми значениями."""
    out: dict[str, str] = {}
    for key, val in d.items():
        full = f"{prefix}{key}"
        if isinstance(val, dict):
            out.update(_flatten(val, prefix=f"{full}."))
        else:
            out[full] = str(val)[:_MAX_PARAM_LEN]
    return out


def log_run(
    tracking_uri: str,
    experiment: str,
    run_name: str,
    cfg: dict,
    metrics: dict,
    artifacts: list | None = None,
    tags: dict | None = None,
) -> str:
    """Логирует один прогон в локальный MLflow-store и возвращает его run_id.

    Секрет api_key вычищается через _redact до разворачивания cfg в параметры.
    Отсутствующие пути в artifacts пропускаются молча.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten(_redact(cfg)))
        mlflow.log_metrics(metrics)
        for path in artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        mlflow.set_tags(tags or {})
        return run.info.run_id
