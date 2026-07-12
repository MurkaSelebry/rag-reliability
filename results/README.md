# Результаты веток m3/m6 на полном датасете организаторов

Дата: 2026-07-12 · ветка `m3-m6` · тесты: **100 passed**, ruff clean (`tests/make_check_output.txt`)

## Что прогнано

- **Полный корпус организаторов**: `data/data/data.zip → data.csv` (2245 строк, −12 точных
  дубликатов = **2233 кейса**), детерминированные group-aware сплиты `make splits`
  (seed 42): train 1787 / val 223 / test 223. Сплиты бит-в-бит совпали с прежними
  прогонами (id проверены).
- **Метод 3 (LLM-as-judge)**: zero_shot и few_shot прогнаны на **всех** сплитах —
  train докуплен в этом прогоне, val/test из этапа 1 (OpenRouter/Phala,
  qwen-2.5-7b-instruct, opt-in владельца данных в `configs/config.alfa_cloud.yaml`).
  Извлечение вероятностей: **100% logprobs** на всех 2233 кейсах обоих вариантов
  (ни одного regex/константного фолбэка).
- **Бейзлайны** (CPU, локально): majority, surface-логрег, surface+e5 —
  пересчитаны на этих же сплитах, числа воспроизвелись точь-в-точь.
- Smoke-тест logprobs провайдера перед прогоном: OK (`scripts/smoke_logprobs.py`).

## Главная таблица — f1-macro(reliable)

Пороги подобраны **только на val** (контракт платформы) и применены ко всем сплитам.
`full` = train + val + test = весь корпус организаторов.

| метод / вариант | val | test | train | **full (2233)** |
|---|---|---|---|---|
| baseline majority | 0.4297 | 0.4053 | — | — |
| **baseline surface** | **0.6133** | **0.5982** | — | — |
| baseline surface_e5 | 0.6150 | 0.5373 | — | — |
| m3 zero_shot (7B) | 0.5678 | 0.5841 | 0.5478 | **0.5541** |
| m3 few_shot (7B) | 0.5718 | 0.5309 | 0.5318 | **0.5359** |
| m3 zero_shot_72b (абляция) | 0.5894 | — | — | — |
| m3 few_shot_72b (абляция) | 0.5235 | — | — | — |
| m3 gepa_markers_s0 (light) | 0.5502 | — | — | — |

Бейзлайны обучаются на train, поэтому их train/full-числа были бы in-sample и не приводятся.

## Детализация Метода 3 (полный корпус)

| вариант | пороги (t_faith / t_rel) | f1 reliable | f1 faith | f1 rel |
|---|---|---|---|---|
| zero_shot | 0.30 / 0.63 | 0.5541 | 0.5550 | 0.5002 |
| few_shot | 0.98 / 0.63 | 0.5359 | 0.5340 | 0.4767 |

У few_shot порог t_faith=0.98 экстремальный — судья с примерами сдвигает
p_faith к 1.0, и разделение достигается только у самой границы; это ещё один
симптом переуверенности в PASS.

## Вывод

Картина этапа 1 подтверждается на полном корпусе: **judge < surface-логрег**
(0.554 против 0.613 val / 0.598 test), few-shot не помогает (0.536).
Диагностика и стоп-правила — `docs/13_openrouter_stage.md`; эмпирика идёт
в дискуссию H1 как аргумент за supervised-ветки.

Метод 6 на полном корпусе не прогнан: NLI-фичи требуют GPU
(~60 ч на CPU для 446×10 пар; train ещё дороже) — статус «до GPU»,
сэмплы val/test закэшированы.

## Структура папки

```
results/
├── README.md                 # этот файл
├── summary.json              # все метрики одним json
├── summary_table.md          # главная таблица отдельно
├── tests/make_check_output.txt
├── baselines/{majority,surface,surface_e5}/report.json
└── m3/{zero_shot,few_shot,zero_shot_72b,few_shot_72b,gepa_markers_s0}/report.json
```

## Воспроизведение

```bash
make check                                   # тесты + линт
unzip -j data/data/data.zip data.csv -d data/raw/alfa/
make splits                                  # детерминированно, seed 42
make baseline-surface                        # CPU
export OPENROUTER_API_KEY=...                # opt-in: configs/config.alfa_cloud.yaml
python scripts/run_m3.py --config configs/config.alfa_cloud.yaml \
    --mode zero_shot --split train --concurrency 8   # + few_shot; val/test аналогично
```

Сырые предсказания: `predictions/alfa_openrouter/m3/{zero_shot,few_shot}/{train,val,test}.jsonl`
(+ `run.yaml` с конфигом, git-хэшем и seed рядом с каждым прогоном).
Числа профиля cloud помечены в run.yaml и в сводную таблицу проекта не входят
до подтверждения на локальном vLLM (правило проекта).
