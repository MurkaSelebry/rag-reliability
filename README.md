# RAG Reliability — Методы 3 и 6 (SMILES × Alfa Bank)

Ветки исследовательского проекта **Assessing the Reliability of Responses in RAG Systems**:
per-case вероятности **faithfulness** и **relevance** для ответов русскоязычного
банковского RAG-бота.

- **Метод 3** — LLM-as-judge: вердикты PASS/FAIL из logprobs токенов, с промпт-оптимизацией
  GEPA/DSPy (гипотеза H5).
- **Метод 6** — SelfCheckGPT + semantic entropy: unsupervised consistency-сигнал (гипотеза H4).

> **Статус:** этап −1 (cloud-режим отладки) завершён — оба пайплайна живые на синтетическом
> псевдо-корпусе. Числа cloud-режима отладочные, в сводную таблицу проекта не идут.
> Подробности и результаты: [`docs/08`](docs/08_stage_minus1_results.md),
> [`docs/09`](docs/09_amendments_b_results.md).

## Быстрый старт

```bash
pip install -r requirements.txt

# ключ провайдера — в .env (см. ниже), не в коде
echo 'OPENROUTER_API_KEY=sk-or-...' > .env

# тесты (без GPU и без сети — чистая логика на стабах)
pytest tests/ -x -q            # 48 passed

# smoke-тест провайдера: приходят ли top_logprobs на токенах вердикта
python -m tools.smoke_logprobs --config configs/config.cloud.yaml -n 3
```

## Конфигурация и секреты (.env)

Код **не знает о провайдере ничего, кроме конфига** — `api_base`/`api_key`/`model`.
Ключи не хардкодятся: в YAML стоит плейсхолдер `${OPENROUTER_API_KEY}`, загрузчик
(`src/common/config.py`) подставляет значение из окружения, автоматически подхватывая
`.env` из корня проекта.

- `.env` **в `.gitignore`** — секрет в репозиторий не попадает.
- В `run.yaml` рядом с каждым прогоном `api_key` редактируется до `***`.
- Профили: `configs/config.yaml` (local, дефолт) и `configs/config.cloud.yaml` (OpenRouter).
  Отличаются только `api_base`/`api_key`/`model` — код между профилями не меняется.

```yaml
# configs/config.cloud.yaml (фрагмент)
llm:
  api_base: "https://openrouter.ai/api/v1"
  api_key: "${OPENROUTER_API_KEY}"     # берётся из окружения / .env
  model: "qwen/qwen-2.5-7b-instruct"
```

**Правило данных (абсолют):** в cloud-профиле на внешний API уходят **только**
синтетические кейсы (id с префиксом `pseudo_` или `meta.synthetic: true`). Guard
(`src/common/guard.py`) проверяет это перед каждым запросом; корпус кураторов на внешний
endpoint не уходит никогда.

## Запуск методов

```bash
# псевдо-корпус (SberQuAD, 300 кейсов, поэлементный кэш; --limit N -> отдельные __smokeN файлы)
python -m tools.make_pseudo_corpus --config configs/config.cloud.yaml

# Метод 3 — zero_shot
python -m src.m3.predict --config configs/config.cloud.yaml --mode zero_shot --split val

# Метод 6 — sample -> features -> predict (20 кейсов)
for s in train val test; do
  python -m src.m6.sample   --config configs/config.cloud.yaml --split $s --limit 20
  python -m src.m6.features --config configs/config.cloud.yaml --split $s --limit 20
done
python -m src.m6.predict --config configs/config.cloud.yaml --limit 20

# сигналы работоспособности по типам кейсов
python -m tools.check_signals --config configs/config.cloud.yaml --split val \
    --m3-pred predictions/cloud/m3/zero_shot/val.jsonl \
    --m6-features artifacts/cloud/m6_features/val.jsonl
```

## Структура

```
src/common/    config (.env + ${VAR}), schemas (Case/Pred), guard (утечка),
               llm_client (LLMClient + JudgeClient), eval_local, run_meta
src/m3/        prompts (SEED_INSTRUCTION), predict (судья из logprobs)
src/m6/        nli (mDeBERTa-XNLI), sample, features (selfcheck/entropy/cos), predict
tools/         smoke_logprobs, make_pseudo_corpus, check_signals
tests/         48 юнит-тестов (чистая логика на стабах, без GPU/сети)
configs/       config.yaml (local), config.cloud.yaml (OpenRouter), few_shot, markers
docs/          00–07 спецификации; 08 отчёт этапа −1; 09 отчёт Amendments B
reference_src/ референс ранней итерации (НЕ подключён, только для заимствования)
```

## Ключевые правила проекта (см. `CLAUDE.md`)

- **Local models only** для данных: банковские тройки — только локальный vLLM; облако
  допускается лишь для публичных датасетов и `pseudo_*` (отладка механики).
- **Сплиты неприкосновенны**, dev-test — только финальный замер, не для решений.
- **Вероятности из logprobs** (softmax PASS/FAIL), fallback logprobs → regex → 0.5/0.5;
  кейс не теряется никогда.
- **Кэшируй всё дорогое**, атомарная запись; перезапуск продолжает, а не пересчитывает.
- **Детерминизм**: каждый прогон пишет `run.yaml` (конфиг + git-хэш + seed).

## Текущие результаты (этап −1, псевдо-корпус)

- Метод 3 zero_shot: **100 % извлечение из logprobs**, f1_reliable ≈ 0.83 на val.
- Метод 6: contradiction — сильный сигнал на галлюцинации (Δ+0.94); semantic entropy
  бесполезна как faithfulness-признак (проверено абляцией N, [`docs/09`](docs/09_amendments_b_results.md)).
- Два честных негативных результата, задающих приоритеты GEPA-этапа: off_topic faith-путаница
  промптом не лечится (нужен few-shot); энтропию не наращивать. Детали в `docs/08`, `docs/09`.

## Дальше

Этап 0 (реальный корпус + локальный vLLM) и GEPA-этап (few_shot, run_gepa, H5) — см.
чекбоксы в [`docs/05_tasks.md`](docs/05_tasks.md).
