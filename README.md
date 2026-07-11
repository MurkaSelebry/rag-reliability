# rag-reliability-m3m6

Ветки «Метод 3» и «Метод 6» командного проекта **Assessing the Reliability of
Responses in RAG Systems** @SMILES × Alfa Bank. По тройке `QUESTION`, `CONTEXT`,
`ANSWER` методы выдают per-case вероятности **faithfulness** и **relevance**
для ответов русскоязычного банковского RAG-бота
(`reliable = faithfulness AND relevance`).

Реализованные семейства методов:

- **Метод 3 — LLM-as-judge** (`scripts/run_m3.py`): вердикты PASS/FAIL из
  logprobs токенов (softmax по паре на позиции вердикта), режимы
  `zero_shot` / `few_shot` / `gepa`; промпт-оптимизация GEPA/DSPy
  (`scripts/run_gepa.py`, гипотеза H5).
- **Метод 6 — SelfCheckGPT + semantic entropy** (`scripts/prepare_m6_samples.py`
  → `scripts/prepare_m6_features.py` → `scripts/run_m6_selfcheck.py`):
  unsupervised consistency-сигнал по N сэмплам бота (гипотеза H4).
- **Бейзлайны**: surface-эвристики ±e5-эмбеддинги
  (`scripts/run_surface_baseline.py`), supervised-энкодер кураторов
  (`scripts/train_encoder_baseline.py`).

## Карта документации

| Документ | Что внутри |
|---|---|
| [docs/00](docs/00_project_overview.md) … [docs/07](docs/07_api_policy_pseudo_corpus.md) | Постановка, формат данных, контракт платформы, спецификации методов, задачи, ресурсы, политика API |
| [docs/08](docs/08_stage_minus1_results.md), [docs/09](docs/09_amendments_b_results.md) | Итоги этапа −1 (cloud-отладка) и Amendments B |
| [docs/10](docs/10_gepa_stage.md) | GEPA-этап: стоп-правила, абляция entropy |
| [docs/11](docs/11_viz_scale.md), [docs/viz/](docs/viz/) | Визуализации, дашборд, интерактивный отчёт, [метрики](docs/viz/METRICS.md) |
| [docs/12](docs/12_alfa_data.md), [docs/13](docs/13_openrouter_stage.md) | Корпус Альфы и этап OpenRouter (72B-абляции) |

## Быстрый старт

Требуется Python ≥ 3.11 и `uv`. Тесты и вся чистая логика работают без GPU и сети.

```bash
make install            # uv venv + ядро/dev-зависимости
make install-m6         # опционально: NLI/эмбеддинг-стек Метода 6 (torch)
make install-gepa       # опционально: DSPy для GEPA
make install-viz        # опционально: фигуры, HTML-отчёт, streamlit-эксплорер
make check              # тесты (100) + линт
make help               # все цели: методы, бейзлайны, отчёты
```

(Без make: `uv venv --python 3.11 && uv pip install -e ".[dev]"`, затем `pytest`.)

Секреты — только через `.env` (в `.gitignore`); в YAML стоит плейсхолдер:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

```yaml
# configs/config.cloud.yaml (фрагмент)
llm:
  api_base: "https://openrouter.ai/api/v1"
  api_key: "${OPENROUTER_API_KEY}"     # берётся из окружения / .env
  model: "qwen/qwen-2.5-7b-instruct"
```

Профили: `configs/config.yaml` (local vLLM, дефолт), `configs/config.cloud.yaml`
(OpenRouter) — код между профилями не меняется, различия только в
`api_base`/`api_key`/`model`.

Smoke-тест провайдера (приходят ли top_logprobs на токенах вердикта):

```bash
python scripts/smoke_logprobs.py --config configs/config.cloud.yaml -n 3
```

## Запуск методов

Все скрипты идемпотентны (кэши в `artifacts/`, перезапуск продолжает) и
принимают `--limit N` для smoke-прогона.

```bash
# псевдо-корпус (SberQuAD, 300 кейсов) для cloud-отладки
python scripts/make_pseudo_corpus.py --config configs/config.cloud.yaml

# Метод 3 — судья, zero_shot
python scripts/run_m3.py --config configs/config.cloud.yaml --mode zero_shot --split val

# Метод 6 — sample -> features -> predict
for s in train val test; do
  python scripts/prepare_m6_samples.py  --config configs/config.cloud.yaml --split $s --limit 20
  python scripts/prepare_m6_features.py --config configs/config.cloud.yaml --split $s --limit 20
done
python scripts/run_m6_selfcheck.py --config configs/config.cloud.yaml --limit 20

# оценка predictions (пороги: --fit только на val!)
python scripts/evaluate.py --cases data/processed/pseudo_dev_val.jsonl \
    --preds predictions/cloud/m3/zero_shot/val.jsonl --fit

# сигналы работоспособности и отчёты
python scripts/check_signals.py --config configs/config.cloud.yaml --split val \
    --m3-pred predictions/cloud/m3/zero_shot/val.jsonl \
    --m6-features artifacts/cloud/m6_features/val.jsonl
python scripts/make_report.py --root . --out artifacts/report/index.html
make explorer           # интерактивный разбор кейсов (streamlit)
```

Каждый метод пишет `predictions/{profile}/{method}/{variant}/{split}.jsonl`
(строки `{"id", "p_faith", "p_rel", "meta"}`) + `run.yaml` (конфиг, git-хэш,
seed) — формат зафиксирован контрактом платформы ([docs/02](docs/02_platform_contract.md)).

## Метрики

Считает `rag_reliability.common.eval_local` (зеркало замороженного
`evaluate.py` платформы; см. `scripts/evaluate.py`):

- **`f1_macro_reliable`** — основная метрика (joint faith ∧ rel)
- `f1_macro_faith`, `f1_macro_rel`
- пороги `t_faith`/`t_rel` подбираются сеткой **только на val**

## Ключевые правила проекта (см. `CLAUDE.md`)

- **Local models only** для данных: банковские тройки — только локальный vLLM;
  облако допускается лишь для публичных датасетов и `pseudo_*` (guard в
  `rag_reliability.common.guard` проверяет каждый запрос).
- **Сплиты неприкосновенны**, dev-test — только финальный замер, не для решений.
- **Вероятности из logprobs** (softmax PASS/FAIL), fallback
  logprobs → regex → 0.5/0.5; кейс не теряется никогда.
- **Кэшируй всё дорогое**, атомарная запись; перезапуск продолжает.
- **Детерминизм**: каждый прогон пишет `run.yaml`; cloud-числа помечаются
  `profile: cloud` и в сводную таблицу не идут.

## Статус и результаты

- ✅ Этап −1 (cloud-отладка на псевдо-корпусе) завершён: оба пайплайна живые
  ([docs/08](docs/08_stage_minus1_results.md), [docs/09](docs/09_amendments_b_results.md)).
- ✅ Метод 3: few_shot f1_reliable **0.824** (val, псевдо-корпус) — лучший
  вариант; GEPA-light его не превзошёл (medium не эскалировался по
  стоп-правилу); 100 % извлечение вердиктов из logprobs.
- ✅ Метод 6: contradiction — рабочий faith-сигнал (Δ+0.31); semantic entropy
  ненадёжна как faithfulness-признак (абляция thr×N, [docs/10](docs/10_gepa_stage.md) §4).
- ✅ Бейзлайны на корпусе Альфы: surface 0.615 / surface+e5 0.537;
  curator_encoder — CPU-smoke, полноценно — на GPU-этапе.
- ⚠️ Главный негативный результат: off_topic faith-путаница — ограничение
  7B-судьи (не лечится ни промптом, ни few-shot); проверка на крупном
  backbone — 72B-абляции ([docs/13](docs/13_openrouter_stage.md)).
- ⏳ Дальше: local-этап (корпус кураторов + vLLM), боевые H1/H5-прогоны —
  чекбоксы в [docs/05](docs/05_tasks.md).

## Структура проекта

```
src/rag_reliability/     пакет с логикой:
  common/                config (.env + ${VAR}), schemas (Case/Pred), guard (утечка),
                         llm_client/async_llm, eval_local, run_meta, results_index, tracking
  data/                  alfa_loader, make_splits, pseudo_corpus
  methods/m3/            prompts (SEED_INSTRUCTION), predict (судья из logprobs),
                         run_gepa, gepa_report
  methods/m6/            sample, nli (mDeBERTa-XNLI), features (selfcheck/entropy/cos),
                         predict, entropy_ablation
  baselines/             surface (+e5), curator_encoder
  analysis/              check_signals, marker_signals, figs, report (HTML), eda_alfa
scripts/                 тонкие CLI-обёртки — запускать из корня репо
tests/                   100 юнит-тестов (чистая логика на стабах, без GPU/сети)
configs/                 config.yaml (local), config.cloud.yaml, config.alfa_cloud.yaml,
                         few_shot.yaml, markers.yaml
docs/                    00–13 спецификации и отчёты этапов; viz/ — дашборд и фигуры
predictions/             выходы методов (контракт платформы, коммитятся)
reference_src/           референс ранней итерации (НЕ подключён, только для заимствования)
data/, artifacts/        корпуса и кэши (в .gitignore; версии — dvc.yaml)
```
