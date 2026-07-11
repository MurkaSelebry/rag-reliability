# 10. Этап −0.5 (cloud, GEPA-этап) — подробный отчёт

**Дата:** 2026-07-04. **Ветка:** `claude/writing-plans-feature-q6te47`.
**Профиль:** cloud (OpenRouter, провайдер **Phala**; задача `qwen/qwen-2.5-7b-instruct`,
рефлексия GEPA `qwen/qwen-2.5-72b-instruct`).
**Корпус:** синтетический псевдо-корпус (SberQuAD → 300 кейсов, микс 2/1/1/1
clean/hallucination/incomplete_answer/off_topic_answer, сплит 240/30/30), регенерирован
этим этапом (seed 42), метки искусственные.

> **Статус чисел — читать обязательно.** Всё ниже получено в cloud-режиме на синтетике с
> искусственными метками. По правилам проекта (CLAUDE.md, docs/07) эти числа **не
> доказывают гипотез H1/H4/H5 и не идут в сводную таблицу проекта/отчёт**. Финальные числа —
> только local-профиль на реальном корпусе кураторов.
>
> **Отдельно про H5.** Маркеры в псевдо-корпусе **детерминированы типом кейса** и идеально
> коррелируют с истинной меткой. Значит feedback варианта `markers` ≈ feedback `plain` +
> дубль-подсказка, и любой выигрыш маркеров здесь **завышен и малоинформативен**. Этот этап
> отлаживает **механику** GEPA (обвязку DSPy, метрику с feedback, бюджеты, стоп-правила,
> сохранение и инференс эволюционированных промптов), а **не проверяет H5**. H5 проверяется
> только на реальном корпусе, где маркеры несут ортогональную метке информацию (за что
> именно FAIL).

## Цель этапа и как достигнута

Закрыть cloud-режим отладки до конца, до получения GPU и корпуса кураторов:

1. Довести ступень `few_shot` Метода 3 до якорей обеих осей (по итогам Amendments B).
2. Дать инструмент визуализации для отладки судьи и m6.
3. Поднять и отладить всю механику GEPA (оптимизация промпта, метрика ±маркеры, H5-обвязка,
   дамп эволюции), с бюджетными стоп-правилами.
4. Закрыть хвосты Метода 6: абляция `entail_threshold` и кривая N из готового кэша сэмплов.
5. Свести всё в этот отчёт и подготовить перенос на local-этап.

Все 6 задач плана выполнены. Тесты: **62 passed** (48 прежних + 14 новых). Работа
велась по TDD: тест писался вместе с функцией, на стабах вместо моделей/сети.

---

## 1. Инвентарь: что построено

### Новый код

| Файл | Что делает | Тесты |
|---|---|---|
| `configs/m3m6/few_shot.yaml` | 7 few-shot-примеров с ручным «Анализом», якоря обеих осей | `tests/test_few_shot_config.py` (5) |
| `scripts/m3m6/make_figs.py` | боксплоты p по kind, reliability diagram, f1(t), scatter m6 | `tests/test_figs_logic.py` (2) |
| `src/rag_reliability_m3m6/methods/m3/run_gepa.py` | DSPy/GEPA-оптимизация инструкции судьи; метрика ±маркеры; глосс из yaml | `tests/test_gepa_metric.py` (5) |
| `src/rag_reliability_m3m6/methods/m3/gepa_report.py` | markdown-отчёт эволюции промпта из track_stats | I/O-глю, без юнит-теста |
| `scripts/m3m6/entropy_ablation.py` | абляция entropy по (порог × N) из кэша, 0 LLM-вызовов | `tests/test_entropy_ablation.py` (2) |

### Изменённый код

| Файл | Изменение |
|---|---|
| `src/rag_reliability_m3m6/methods/m3/predict.py` | `--prompt-file` (override пути промпта для mode=gepa) и `--variant-name` (поддиректория predictions, чтобы прогоны markers/plain×seed не перетирали друг друга) |
| `src/rag_reliability_m3m6/methods/m6/sample.py` | `--n` (override `m6.n_samples`; нужен для добора до 10 сэмплов под абляцию) |
| `configs/config.cloud.yaml` | секция `m3.gepa`: `train_size`, `auto`, `seeds`, `variants` (markers/plain → `use_marker_feedback` + `out_prompt`), `reflection` (72B, api_base, max_tokens) |
| `requirements.txt` | `+matplotlib` |
| `docs/05_tasks.md` | блок «Этап −0.5 — GEPA (cloud)», чекбоксы закрыты |

### Артефакты прогона (в `artifacts/`, `predictions/` — gitignored, кроме predictions/cloud)

- `artifacts/cloud/m3_prompt_{markers,plain}_seed{0,1}.txt` — 4 эволюционированных инструкции;
- `artifacts/cloud/m3_gepa_stats_*.json` — полный track_stats + счётчики LM-вызовов;
- `artifacts/cloud/m3_gepa_report_*.md` — 4 markdown-отчёта эволюции (таблица кандидатов);
- `artifacts/cloud/m3_entropy_ablation_val.json` — таблица абляции;
- `artifacts/figs/*.png` — 5 фигур;
- `predictions/cloud/m3/{zero_shot,few_shot,gepa_markers_s0,gepa_markers_s1,gepa_plain_s0,gepa_plain_s1}/`
  — predictions val+test + `report_*.json` + `run.yaml`.

---

## 2. Ступень few_shot — якоря обеих осей (Task 1)

### Что сделано

7 примеров отобраны из `pseudo_dev_train` (не из val/test — нет пересечения с оценкой),
`ctx` укорочен до абзаца-источника + одного дистрактора, «Анализ» к каждому написан вручную
с явной логикой оси. Состав (закрывает выводы Amendments B):

| # | kind | faith | rel | роль «Анализа» |
|---|---|---|---|---|
| 1 | clean | PASS | PASS | всё подтверждено контекстом и по вопросу |
| 2 | clean | PASS | PASS | второй позитив, другой домен |
| 3 | hallucination | FAIL | PASS | подменён факт: «в ответе 2008, в чанке 2010» |
| 4 | incomplete_answer | FAIL | PASS | «опущена деталь — название "Кембрийский взрыв"» |
| 5 | incomplete_answer | FAIL | PASS | «ответ неполный: опущено "столбовое расписание"» |
| 6 | off_topic_answer | **PASS** | **FAIL** | **якорь B1**: «верно по фрагментам, но не на вопрос — оси независимы» |
| 7 | off_topic_answer | **PASS** | **FAIL** | **якорь rel**: строгая формула «не отвечает на вопрос» |

Тесты фиксируют: 6–8 примеров, все поля, ≥2 incomplete с «опущ/неполн», наличие
off_topic-якоря (PASS/FAIL), строгий rel=FAIL, оба PASS/PASS.

### Приёмка на val (30 кейсов): few_shot vs zero_shot

| Метрика | zero_shot | few_shot | критерий | итог |
|---|---|---|---|---|
| off_topic средний **p_faith** | 0.202 | **0.240** | > 0.5 | ❌ **не взят** |
| сигнал rel (clean − off_topic) | +0.581 | **+0.528** | > +0.5 | ✅ |
| сигнал faith (clean − hallucination) | +0.532 | **+0.635** | > +0.5 | ✅ |
| сигнал faith (clean − incomplete) | +0.371 | **+0.311** | ≥ +0.2 | ✅ |
| f1_reliable (val) | 0.792 | **0.824** | — | ↑ +0.032 |

Средние p по kind (few_shot): clean p_faith **0.969** / p_rel 0.854; hallucination
p_faith **0.334**; incomplete p_faith 0.658; off_topic p_faith **0.240** / p_rel 0.326.

### Ключевой вывод (негативный, ценный)

**Главный критерий — off_topic p_faith > 0.5 — не взят ни промптом (B1), ни few-shot'ом.**
Это **устойчивое ограничение 7B-судьи**, а не дефект примеров. Судья трактует «ответ не на
тот вопрос» как искажение и ставит FAITHFULNESS: FAIL, хотя факты берутся из [CTX]. Сырые
вердикты (val, few_shot):

- `pseudo_00059` (Q: на какой глубине образуются алмазы) — «в чанках ~200 км, а ответ
  утверждает, что глубина не связана с теориями — искажение, FAITHFULNESS: FAIL» →
  p_faith 0.016 (**но** p_rel 0.622 — оси всё же разведены).
- `pseudo_00169` (Q: с кем предки Байрона пришли в Англию) — «ответ про смену фамилии при
  Генрихе VIII, а не про приход — основная информация не представлена» → двойной FAIL,
  p_faith 0.182, p_rel 0.009.

**Итог:** правило осей B1 разводит relevance (rel-сигнал сохранён +0.528, размытия из B1
нет), но faith-путаницу на off_topic 7B-судья не преодолевает. Больше **одной** итерации
примеры не крутились (правило стоп-приёмки). Гипотеза, что более крупный backbone возьмёт
порог, проверяется на local-этапе.

---

## 3. GEPA — оптимизация промпта и механика H5 (Task 3–4)

### Архитектура

- **Роли моделей:** DSPy — только для **оптимизации** (эволюция инструкции); **инференс** —
  собственный `JudgeClient` (вероятности из logprobs PASS/FAIL), схема совпадает с Методами
  1–2 ради честного H1.
- **Программа:** `dspy.ChainOfThought(Judge)` — signature с входами query/context/answer и
  выходами `faithfulness`/`relevance` (`Literal["PASS","FAIL"]`); docstring-инструкция =
  актуальный `SEED_INSTRUCTION` (с правилом независимости осей B1).
- **Метрика** `make_metric(use_markers, gloss)`: `score = (ok_faith + ok_rel)/2`; при ошибке
  feedback даёт истинные метки `FAITHFULNESS={..}, RELEVANCE={..}`, а в варианте `markers` —
  дополнительно построчный глосс маркеров кейса из `configs/markers.yaml`; при верном
  ответе — «Обе оценки верны.». **Единственное различие markers/plain — эта строка глосса.**
- **Данные:** trainset — детерминированная подвыборка `pseudo_dev_train` (seed прогона),
  valset — `pseudo_dev_val` целиком. Guard `assert_cloud_safe` на train и val до старта.
- **LM:** task `dspy.LM(openai/qwen-2.5-7b, T=0, max_tokens=600, extra_body=Phala)`;
  reflection — 72B (`m3.gepa.reflection`).
- **Сохранение:** инструкция → `out_prompt` (подстановка `{seed}`), `program.save(json)`,
  `detailed_results.to_dict()` + счётчики LM-вызовов → stats-json.

### Стоп-правило бюджета (checkpoint: smoke markers/seed0, train=50, auto=light)

| Вариант | f1_reliable (val) | f1_faith | f1_rel |
|---|---|---|---|
| zero_shot | 0.792 | 0.653 | 0.711 |
| few_shot | **0.824** | 0.659 | 0.804 |
| gepa_markers smoke (train=50) | 0.792 | 0.653 | **0.444** |

GEPA-light-smoke **не превзошёл few_shot** и просадил f1_rel (раздувание инструкции
примерами повредило relevance-дискриминацию). Согласно стоп-правилу **medium НЕ
эскалировался**. Полная механика H5 прогнана на **train=100, auto=light** — как отладка
обвязки, не как проверка H5.

### Полные прогоны H5: сводная таблица (f1_reliable; GEPA train=100, auto=light)

| Вариант | val | test | зазор val−test |
|---|---|---|---|
| zero_shot | 0.792 | 0.661 | +0.131 |
| few_shot | **0.824** | 0.691 | +0.133 |
| gepa_markers seed0 | 0.729 | **0.764** | −0.035 |
| gepa_markers seed1 | 0.824 | 0.583 | **+0.241 ⚠** |
| gepa_plain seed0 | 0.722 | 0.661 | +0.061 |
| gepa_plain seed1 | 0.760 | 0.697 | +0.063 |
| **среднее markers** | **0.776** | **0.674** | |
| **среднее plain** | **0.741** | **0.679** | |

**Средний выигрыш markers − plain: val +0.035, test −0.005.**

### Бюджет прогонов (фактический)

| Прогон | train | task LM (7B) | reflection (72B) | кандидатов | best_idx | best val-score |
|---|---|---|---|---|---|---|
| markers seed0 | 100 | 596 | 20 | 8 | 2 | 0.717 |
| markers seed1 | 100 | 501 | 21 | 5 | 1 | 0.733 |
| plain seed0 | 100 | 501 | 20 | 11 | 3 | 0.717 |
| plain seed1 | 100 | 501 | 25 | 7 | 5 | 0.733 |
| **ИТОГО (4)** | | **2099** | **86** | | | |

Плюс smoke markers/seed0 (train=50): task 501 / reflection 22. Всё уложилось в ориентир
$2–5 на этап.

### Как это читать (и почему это НЕ вывод по H5)

Маркеры дают крошечное преимущество на val (+0.035), которое **не удерживается на test**
(−0.005). Именно так и должно выглядеть на этом корпусе: маркеры детерминированы типом
кейса, feedback markers ≈ feedback plain + дубль-подсказка → выигрыш почти нулевой и тонет
в шуме 30-кейсовых сплитов. **H5 на псевдо-корпусе не проверяется — отлажена механика.**

**Переобучение промпта.** `gepa_markers seed1` — зазор val−test **+0.241** (> 4 пунктов,
флаг переобучения по стоп-правилу). На 30-кейсовых сплитах это ожидаемая дисперсия
(`gepa_markers seed0`, наоборот, test > val на 0.035). Так как этап — репетиция механики,
корректирующая итерация с меньшим train_size **не запускалась** (на 30 test-кейсах синтетики
она не даст сигнала); флаг зафиксирован как демонстрация работающего контроля. На local тот
же контроль применяется к реальным сплитам.

**Эволюция промптов.** Рефлексия-72B за ~20 вызовов находит лучшего кандидата; порождённые
инструкции осмысленны, на русском, сохраняют правило осей B1 и добавляют структурированные
указания + few-shot-примеры из train. Полностью сохранены в
`artifacts/cloud/m3_gepa_report_{variant}_seed{seed}.md` (таблица кандидатов с val-score +
тексты) и `..._stats_*.json`.

---

## 4. Абляция semantic entropy по порогу и N (Task 5)

### Что сделано

`scripts/m3m6/entropy_ablation.py` считает NLI-матрицу **один раз на кейс** (полная матрица пар
`[answer]+samples` для max N) и строит кластеры (union-find по двунаправленному entailment)
при всех (thr, N) срезами этой матрицы — **ноль LLM-вызовов**, из кэша сэмплов N=10
(добраны `sample.py --n 10`). Тест на стабе NLI фиксирует: порог 0.45 даёт 1 кластер при
thr=0.4 и 2 при thr=0.5; срез по N использует префикс сэмплов; NLI зовётся один батч на кейс.

### Δ (hallucination − clean) semantic_entropy, val, 30 кейсов

| thr \ N | N=3 | N=5 | N=10 |
|---|---|---|---|
| 0.30 | +0.229 | +0.065 | +0.441 |
| 0.40 | +0.416 | +0.017 | +0.343 |
| 0.50 | +0.335 | **−0.064** | +0.249 |

### Интерпретация

На регенерированном корпусе Δ entropy **преимущественно положительна**, но **мала и
немонотонна по N** (thr=0.5: +0.335 → −0.064 → +0.249 для N=3/5/10 — знак флипается). Это
**не воспроизводит чистую инверсию B4** (−0.164 на прежнем корпусе), но и не даёт
устойчивого сигнала: entropy оказывается **шумным, чувствительным к порогу NLI-эквивалентности,
к N и к самой выборке ответов** признаком.

**Вывод B4 в силе (уточнён):** semantic entropy — ненадёжный признак faithfulness; рабочий
faith-сигнал Метода 6 — **selfcheck-contradiction**. На val Δ (hallucination − clean):
contra **+0.307** против entropy +0.249. Полная сводка m6-фич по kind (val):

| kind | contra_mean | semantic_entropy | n_clusters | cos_q_a |
|---|---|---|---|---|
| clean | 0.039 | 1.083 | 4.625 | 0.879 |
| hallucination | **0.347** | 1.332 | 5.667 | 0.903 |
| incomplete_answer | 0.005 | 0.595 | 2.500 | 0.834 |
| off_topic_answer | 0.034 | 0.981 | 4.000 | 0.800 |

Contradiction чётко выделяет hallucination; entropy — нет; incomplete слепа для consistency
(ожидаемо, H4 — ловится только judge-методом). Решение B4 (не наращивать энтропию, faith на
contradiction) не пересматривается. На реальном корпусе порог NLI-эквивалентности подбирать
на dev-val.

---

## 5. Фигуры для отладки (Task 2)

Одна команда на любых predictions:

```bash
python scripts/m3m6/make_figs.py --config configs/config.cloud.yaml --split val \
    --m3-pred predictions/cloud/m3/few_shot/val.jsonl \
    --m6-features artifacts/cloud/m6_features/val.jsonl
```

→ 5 PNG в `artifacts/figs/` (dpi=150, backend Agg, matplotlib импортируется лениво):

1. `m3_box_p_faith.png`, `m3_box_p_rel.png` — боксплоты p по kind (порядок фикс.:
   clean, hallucination, incomplete_answer, off_topic_answer);
2. `m3_reliability_faith.png` — калибровочная диаграмма (mean_prob vs frac_pos, диагональ
   пунктиром, число кейсов в бине);
3. `m3_f1_curve.png` — f1(t) по обеим осям на одном полотне;
4. `m6_scatter.png` — selfcheck_contra_mean × semantic_entropy, цвет = kind,
   размер = n_clusters.

Чистые функции (`reliability_bins`, `f1_threshold_curve`) отделены от отрисовки и покрыты
тестами. Заголовок каждой фигуры включает split и путь predictions (трассируемость).

---

## 6. Как воспроизвести

```bash
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-...' > .env       # .env в .gitignore
pytest tests/ -x -q                              # 62 passed

# псевдо-корпус (300 кейсов) — если ещё нет
python scripts/m3m6/make_pseudo_corpus.py --config configs/config.cloud.yaml

# Метод 3: базы
python scripts/m3m6/run_m3.py --config configs/config.cloud.yaml --mode zero_shot --split val
python scripts/m3m6/run_m3.py --config configs/config.cloud.yaml --mode few_shot  --split val

# GEPA: smoke → полная механика H5
python scripts/m3m6/run_gepa.py --config configs/config.cloud.yaml --variant markers --seed 0 \
    --train-size 50 --auto light
python scripts/m3m6/run_gepa.py --config configs/config.cloud.yaml --variant markers --seed 0 --train-size 100
#   … markers/plain × seed 0/1, затем инференс:
python scripts/m3m6/run_m3.py --config configs/config.cloud.yaml --mode gepa \
    --prompt-file artifacts/cloud/m3_prompt_markers_seed0.txt --variant-name gepa_markers_s0 --split val
python scripts/m3m6/gepa_report.py --variant markers --seed 0

# Метод 6: сэмплы (N=10 для абляции) → фичи → абляция
python scripts/m3m6/prepare_m6_samples.py   --config configs/config.cloud.yaml --split val --n 10
python scripts/m3m6/prepare_m6_features.py  --config configs/config.cloud.yaml --split val
python scripts/m3m6/entropy_ablation.py --config configs/config.cloud.yaml --split val \
    --thresholds 0.3 0.4 0.5 --ns 3 5 10

# Фигуры
python scripts/m3m6/make_figs.py --config configs/config.cloud.yaml --split val \
    --m3-pred predictions/cloud/m3/few_shot/val.jsonl \
    --m6-features artifacts/cloud/m6_features/val.jsonl
```

Все скрипты идемпотентны (кэш судьи по sha256(model, system, user); кэш сэмплов поэлементно;
атомарная запись tmp+replace) — прерванный прогон продолжает, а не пересчитывает.

---

## 7. Соблюдение правил проекта

- **Local-only для данных.** На внешний API уходили **только** `pseudo_*` кейсы и публичный
  SberQuAD; guard `assert_cloud_safe` вызывается в `run_gepa.py` и `predict.py` до первого
  LLM-вызова (проверено grep'ом). Ни одной строки корпуса кураторов.
- **Сплиты неприкосновенны.** Пороги и отбор — только на dev-val; dev-test — единственный
  финальный замер. Few-shot-примеры взяты из train, не пересекаются с оценкой.
- **Вероятности из logprobs** (softmax PASS/FAIL), fallback logprobs→regex→0.5/0.5.
- **Детерминизм.** Все случайности — через seed; каждый прогон пишет `run.yaml`
  (конфиг + git-хэш + seed), `api_key` редактируется до `***`, `profile: cloud`.
- **Конфигурация — только yaml.** Бюджеты GEPA, модели, пути, N сэмплов, пороги — из конфига;
  глосс маркеров — из `configs/markers.yaml`, не хардкод.

---

## 8. Что переносится на local-этап

- **few_shot-состав** (7 якорей обеих осей) → реальные кейсы с реальными маркерами кураторов;
  проверить, берёт ли более крупный backbone off_topic p_faith > 0.5.
- **GEPA-обвязка** целиком: `run_gepa.py` (метрика ±маркеры, глосс из yaml), `--prompt-file`/
  `--variant-name`, `gepa_report.py`, стоп-правила бюджета (light→medium только по решению),
  контроль зазора val−test.
- **MARKER_GLOSS** ← словарь 13 маркеров кураторов (`configs/markers.yaml` заменяется без
  правок кода).
- **Абляции** N и порога — из готового кэша сэмплов, без новых вызовов.
- **m6:** резка длинных premise с перекрытием; реальный промпт бота от кураторов.
- **Action item команде:** передать итоговый `SEED_INSTRUCTION` (с правилом осей B1) ветке
  Метода 1 — синхронизация промпта для честного H1.

### Открытые ограничения (в limitations отчёта)

1. off_topic faith-путаница 7B-судьи не лечится ни промптом, ни few-shot — нужен более
   крупный backbone (проверка на local).
2. Reflection-LM в облаке — 72B (локально заменится на 32B; расхождение чисел — норма).
3. Выигрыш markers vs plain на псевдо-корпусе завышен (детерминированные маркеры) — H5
   мерить только на реальном корпусе.
4. semantic entropy — ненадёжный faith-признак; m6 строит faith на contradiction.
