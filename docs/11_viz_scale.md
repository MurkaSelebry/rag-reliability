# 11. Этап −0.25 (cloud, «viz & scale») — итоги

**Дата:** 2026-07-05. **Ветка:** `claude/writing-plans-feature-q6te47` (второй PR).
**Профиль:** cloud; LLM-вызовы этого этапа — только smoke async-клиента (~30 дешёвых
вызовов 7B), всё остальное CPU.

> **Статус чисел.** Все визуализируемые числа — с синтетического псевдо-корпуса
> (docs/07–10): отладочные, в отчёт проекта не идут, H5 не проверяется. HTML-отчёт и
> эксплорер на реальном корпусе будут содержать тексты кейсов — они наследуют статус
> данных (не публиковать за периметром команды).

## Зачем этап

Корпуса кураторов и GPU ещё нет. Пока ждём — (а) выжимаем максимум из результатов
этапов −1…−0.5 интерактивной визуализацией и инструментом error-analysis, (б) готовим
репозиторий к масштабу local-этапа: 5k кейсов × 6+ вариантов × десятки прогонов.
Боевая логика не менялась — всё это надстройка над существующими артефактами и
идемпотентными скриптами.

## 1. Что построено

| Компонент | Файлы | Команда |
|---|---|---|
| Единый индекс прогонов | `src/rag_reliability_m3m6/common/results_index.py` | `build_index()` → 5 DataFrame (runs/predictions/m6_features/gepa/entropy_ablation) |
| Интерактивный HTML-отчёт | `scripts/m3m6/make_report.py` | `python scripts/m3m6/make_report.py --root . --out artifacts/report/index.html` |
| Обозреватель кейсов | `scripts/m3m6/explorer.py` | `streamlit run scripts/m3m6/explorer.py` (локально) |
| Async-клиент | `src/rag_reliability_m3m6/common/async_llm.py` | `--concurrency N` в `scripts/m3m6/run_m3.py` и `scripts/m3m6/prepare_m6_samples.py` |
| MLflow-трекинг | `src/rag_reliability_m3m6/common/tracking.py` | `mlflow ui --backend-store-uri file:./mlruns` |
| DVC-DAG | `dvc.yaml` | `dvc repro report` |
| Гигиена | `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | `ruff check .`, `pre-commit run -a` |

Тесты: **73 passed** (62 прежних + 11 новых: индекс 4, async 2, tracking 2, логика
отчёта 3). Всё дерево прогнано через `ruff check --fix` + `ruff format` (единственное
исключение — `src/rag_reliability_m3m6/methods/m3/prompts.py`: текст промптов не переносим, E501 per-file-ignore —
смена текста инвалидировала бы кэш судьи).

### HTML-отчёт (`artifacts/report/index.html`, ~4.9 МБ, plotly.js inline — без сети)

11 секций: статус-плашка; лидерборд вариантов с подсветкой переобучения
(gap val−test > 0.04 — подсвечен `gepa_markers_s1`, +0.241); grouped bar f1 val/test с
переключателем метрик; боксплоты p_faith/p_rel по kind × все 6 вариантов (виден главный
негативный результат: off_topic p_faith низкий у ВСЕХ вариантов); heatmap kind×variant;
калибровочные кривые; f1(t); confusion-плитки (пороги каждого варианта из его report);
GEPA-эволюция (val_score по кандидатам, звезда = лучший) + бюджеты вызовов; m6-scatter +
heatmap абляции энтропии; метаданные прогонов и доля extract_method (санити
извлечения — 100% logprobs).

Правило отчёта: ни одна цифра не считается «на глаз» в plotting-коде — только
`results_index` + чистые функции с тестами (`variant_summary`, `confusion_counts`,
`gepa_evolution_frame`, `kind_pivot`).

### Streamlit-эксплорер (4 страницы)

- **Кейсы** — фильтры (split/kind/variant/extract_method/«только ошибочные при порогах
  из report»/диапазоны p) → карточка кейса: query, чанки, ответ, метки и **вердикты всех
  вариантов рядом** + raw-вывод судьи по каждому. Главный инструмент разбора
  off_topic-путаницы.
- **Разногласия** — пары вариантов A/B, кейсы с разошедшимися reliable-вердиктами по |Δp|;
  для GEPA vs few_shot — прямой ответ «что изменил оптимизированный промпт».
- **m6** — сэмплы кейса из кэша бок о бок с ответом + фичи (виден alignment-collapse).
- **GEPA-промпты** — финальная инструкция, таблица кандидатов, diff против SEED_INSTRUCTION.

## 2. Async-клиент: замер и вывод о детерминизме

Smoke на 10 кейсах few_shot (val), свежий кэш, OpenRouter/Phala:

| Прогон | Время (весь процесс) |
|---|---|
| sync (`--concurrency 1`) | 21.3 с |
| async (`--concurrency 4`) | 9.7 с |

С учётом ~5 с импортов ускорение самих запросов ≈ **3.5×** при N=4 — масштаб для
local-этапа (vLLM, N=32–64) подтверждён.

**Эквивалентность sync/async:** через общий файловый кэш — **побитовая** (async-прогон
30 кейсов по существующему кэшу дал p, идентичные референсным predictions; ключ кэша
sha256(model, system, user) и fallback-цепочка общие с `JudgeClient`). **Важная находка:**
свежие повторные вызовы OpenRouter при T=0 недетерминированы на стороне провайдера
(батчинг): два независимых **синхронных** прогона разошлись на тех же кейсах так же, как
sync vs async (max |Δp_faith| ≈ 0.82 на пограничном кейсе `pseudo_00169`). То есть
недетерминизм — свойство провайдера, не async-пути; гарантия воспроизводимости в проекте
обеспечивается кэшем (и на local-этапе — vLLM с фиксированным seed).

Guard как в синхронном клиенте: `assert_cloud_safe` по всем кейсам **до** первого
запроса (тест это фиксирует).

## 3. MLflow (локальный) и DVC

- `tracking.log_run(...)`: file-store `file:./mlruns` (в .gitignore), flatten-конфиг c
  редакцией `api_key → ***` (переиспользован `_redact` из run_meta), метрики из report,
  артефакты (report json, run.yaml), теги (variant/split/method). Вызов встроен в
  `scripts/m3m6/run_m3.py` и `scripts/m3m6/run_m6_selfcheck.py` под флагом `tracking.enabled` (в local-конфиге
  выключен по умолчанию — ноль побочных эффектов). Backfill сделан перепрогоном 12
  прогонов m3 по кэшу (0 LLM-вызовов): mlflow содержит 6 вариантов × val/test.
  Установленный MLflow требует `MLFLOW_ALLOW_FILE_STORE=true` — выставляется в
  `tracking.py` через `os.environ.setdefault` (file-store и есть наша политика).
- `dvc.yaml`: DAG pseudo → m3_{zero,few}_shot → m6(sample→features) → report.
  GEPA-прогоны в DAG не включены (дорогие, ручной запуск по стоп-правилам) — report
  подхватывает их выходы через deps на `predictions/cloud`. predictions — `cache: false`
  (коммитятся в git по контракту платформы); корпус — кэшируемый out стейджа pseudo.
  Первичная фиксация без перезапуска: `dvc commit -f`; `dvc repro report` → up to date.
  Remote — локальный плейсхолдер `/tmp/dvc-rag-m3m6` (реальный выберет команда).

## 4. Инструменты: принято / отклонено

| Инструмент | Зачем | Почему локально |
|---|---|---|
| plotly | интерактивный отчёт | `include_plotlyjs="inline"`, 0 внешних src |
| streamlit | error-analysis UI | локальный запуск, headless |
| mlflow | сравнение десятков прогонов local-этапа | только `file:./mlruns` |
| dvc | версии корпуса + DAG прогонов | remote — локальная директория |
| uv/ruff/pre-commit/CI | скорость установки, единый стиль, страховка утечки | всё офлайн, CI без секретов |

**Отклонено (анти-overengineering):**
- **W&B (cloud)** — телеметрия наружу, запрещено политикой данных; mlflow file-store
  закрывает потребность.
- **Hydra** — `config.py` (${VAR}+.env, 50 строк) работает; миграция всех CLI не
  окупается на двух конфигах.
- **Prefect/Airflow** — DAG из 5 стейджей закрывает DVC без демонов и БД.
- **Docker** — отложен до переезда на серверный GPU (local-этап), там решится вместе с
  vLLM-окружением.

## 5. Гигиена

- `pyproject.toml`: зависимости (зеркалируются в `requirements.txt` для pip/CI), ruff
  (line-length 100, E/F/I/UP, E741 ignore — идиома `for l in fh`), pytest testpaths.
- Окружение — чистый uv venv (`.venv`), torch ставится отдельным шагом с CPU-индексом
  (смешение индексов в одном резолве uv отвергает: «No solution found»).
- `.pre-commit-config.yaml`: ruff + ruff-format + end-of-file-fixer + check-yaml +
  detect-private-key; `pre-commit run -a` — все Passed.
- CI (`.github/workflows/ci.yml`): ruff + pytest (стабы, без секретов) + job
  `no-data-leak` (git ls-files не содержит `data/processed/` и `.env`).

## 6. Что переносится на local-этап

- `--concurrency 32–64` под vLLM (семафор уже есть; тюнинг по нагрузке сервера).
- mlflow: experiment `local`, тот же `tracking.enabled` в local-конфиге.
- dvc: заменить remote-плейсхолдер на решение команды (`dvc remote add -d ...`,
  `dvc push`); стейджи получат local-конфиг вместо cloud.
- Отчёт и эксплорер работают без изменений — сменится только root данных/предсказаний.
- Правило периметра: HTML-отчёт с реальными кейсами не покидает команду.
