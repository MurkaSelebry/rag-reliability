# 08. Этап −1 (cloud-режим отладки) — что сделано и результаты

Дата: 2026-07-04. Профиль: **cloud** (OpenRouter, провайдер Phala, `qwen/qwen-2.5-7b-instruct`).

> **Статус чисел.** Всё в этом документе получено в cloud-режиме на **синтетическом**
> псевдо-корпусе с искусственными метками. По правилам проекта (CLAUDE.md, docs/07)
> эти числа **не доказывают гипотез H1/H4/H5 и не идут в сводную таблицу/отчёт**.
> Назначение этапа −1 — доказать, что пайплайны Метода 3 и Метода 6 живые и
> дискриминируют, до появления GPU и корпуса кураторов. Финальные числа проекта —
> только local-профиль на реальном корпусе.

---

## 1. Цель этапа и как она достигнута

Обкатать обе ветки (Метод 3 — LLM-as-judge из logprobs; Метод 6 — SelfCheckGPT +
semantic entropy) против внешнего API на синтетических данных, соблюдая абсолютное
правило «ни одна строка корпуса кураторов не уходит на внешний endpoint».

Реализовано 16 задач (все 6 чекбоксов этапа −1 в `docs/05_tasks.md` закрыты).
Разработка велась по плану `superpowers:writing-plans` + исполнение
`subagent-driven-development`: каждая задача — TDD, отдельное ревью спеки и качества,
финальное whole-branch review. 25 коммитов в ветке `main`.

---

## 2. Что построено

### Общий слой (`src/common/`)

| Модуль | Назначение |
|---|---|
| `config.py` | Загрузка YAML + `.env`, подстановка `${OPENROUTER_API_KEY}` из окружения. Секрет нигде не хардкодится. |
| `schemas.py` | `Case` (+`meta`), `Pred`, чтение/запись jsonl. Канонический формат docs/01. |
| `guard.py` | Защита от утечки: `is_synthetic` (префикс `pseudo_` или `meta.synthetic`), `assert_case_cloud_safe` (per-request), `assert_cloud_safe` (per-file). |
| `llm_client.py` | `LLMClient` (chat с retry, `openrouter_extra_body`, n-fallback, guard) + `JudgeClient` (вероятности PASS/FAIL из logprobs с BPE-склейкой, кэш). |
| `eval_local.py` | Байт-в-байт копия замороженного `evaluate.py` платформы: `fit_thresholds`, `evaluate`. |
| `run_meta.py` | `run.yaml` рядом с predictions: конфиг (с редакцией `api_key` → `***`) + git-хэш + seed. |

### Метод 3 (`src/m3/`)

- `prompts.py` — `SEED_INSTRUCTION` + шаблон `[Q]/[CTX]/[A]` (копия reference).
- `predict.py` — CLI (`--mode/--split/--limit`); guard всего файла до первого запроса;
  `JudgeClient.judge` на кейс; пороги с val через `eval_local`; `run.yaml` + `report_{split}.json`.

### Метод 6 (`src/m6/`)

- `nli.py` — батчевый `NLIScorer` на mDeBERTa-XNLI (ленивый импорт torch).
- `sample.py` — N сэмплов ответа «бота», поэлементный кэш.
- `features.py` — selfcheck-NLI (contradiction), semantic entropy (union-find по
  двунаправленному entailment), `cos_q_a` на multilingual-e5.
- `predict.py` — изотоника (p_faith, калибровка на val) + логрег (p_rel, обучение на train).

### Инструменты (`tools/`)

- `smoke_logprobs.py` — проверка провайдера (top_logprobs на токенах вердикта).
- `make_pseudo_corpus.py` — генерация псевдо-корпуса из SberQuAD по таксономии docs/07.2.
- `check_signals.py` — средние по типам кейсов и проверка ожидаемых сигналов.

### Тесты

44 юнит-теста (чистая логика на стабах, без GPU/сети): загрузчик конфига, схемы,
guard, LLM-клиент (n-fallback, guard-by-default), извлечение вердикта из logprobs
(4 сценария + BPE-склейка + приоритет точных токенов), eval_local, run.yaml,
логика псевдо-корпуса, фичи m6 (кластеризация, энтропия, selfcheck).

---

## 3. Псевдо-корпус

- **Источник:** SberQuAD (`kuznetsoffandrey/sberquad`, split train), не более одного
  кейса на абзац, детерминированная подвыборка (seed=42).
- **Объём:** 300 кейсов, сплиты **240 / 30 / 30** (train/val/test).
- **Микс 2/1/1/1:** clean 120, hallucination 60, incomplete_answer 60, off_topic_answer 60.
- **Генератор:** `qwen/qwen-2.5-7b-instruct` (тот же backbone, что судья).
- **Контекст:** абзац-источник + 1–2 дистрактора, порядок перемешан (имитация RAG-retrieval).

**Контроль качества.** Первая smoke-итерация (20 кейсов) дала 4/20 брака: эхо-формат
у hallucination, отсутствие искажения, ответ на заданный вопрос у off_topic, китайские
символы. Промпты генерации починены (добавлены «отвечай только на русском», «не
повторяй вопрос», «обязательно подмени факт», «не давай прямой ответ»), 4 кейса
перегенерированы → 20/20 OK. Выборочный QC полного корпуса (5×4 = 20 кейсов): **18/20 OK**,
2 изолированных брака (~10%) без систематики (один пропуск подмены факта, один off_topic
с добавленным утверждением). Для отладочного корпуса это приемлемо; метки синтетические.

---

## 4. Результаты Метода 3 (zero_shot, 30 val-кейсов)

Судья выносит вердикты `FAITHFULNESS: PASS/FAIL`, `RELEVANCE: PASS/FAIL`; вероятности —
softmax по паре logprobs на позиции вердикта.

**Извлечение из logprobs: 100% кейсов** (ни regex-fallback, ни 0.5/0.5). Основная
научная ветка Метода 3 в облаке отлаживается — это ключевой результат этапа.

**Метрики** (`report_val.json`, пороги подобраны на val; числа — после правки B1
«независимость осей» в SEED_INSTRUCTION, см. §6 п.2):

| Метрика | Значение |
|---|---|
| f1_macro_reliable | 0.829 |
| f1_macro_faith | 0.603 |
| f1_macro_rel | 0.755 |

**Средние p по типам кейсов (после B1):**

| kind | p_faith | p_rel |
|---|---|---|
| clean | 0.884 | 0.936 |
| hallucination | 0.110 | 0.581 |
| incomplete_answer | 0.665 | 0.950 |
| off_topic_answer | 0.005 | 0.289 |

**Сигналы (docs/07.3):**
- faith (clean − hallucination): **+0.774** — судья уверенно ловит подмену факта.
- rel (clean − off_topic): **+0.647** — судья ловит ответ не на тот вопрос (сигнал
  ослаб после B1: у off_topic вырос p_rel, см. §6 п.2).
- faith (clean − incomplete): **+0.219** — incomplete отслеживается с B2, сигнал слабый
  (ожидаемо трудный тип).

---

## 5. Результаты Метода 6 (20 кейсов на сплит)

Пайплайн sample → features → predict отработал целиком. mDeBERTa-XNLI (~1.1 ГБ) и
multilingual-e5-large (~2.2 ГБ) скачаны и работают на CPU.

**Метрики** (`report_test.json`, 20 test-кейсов):

| Метрика | Значение |
|---|---|
| f1_macro_reliable | 0.596 |
| f1_macro_faith | 0.394 |
| f1_macro_rel | 0.627 |
| share_single_cluster_test | 0.20 |

**Средние фичи по типам кейсов (val):**

| kind | selfcheck_contra_mean | semantic_entropy | n_clusters | cos_q_a |
|---|---|---|---|---|
| clean | 0.031 | 0.996 | 3.55 | 0.874 |
| hallucination | 0.971 | 1.006 | 3.50 | 0.907 |
| incomplete_answer | 0.001 | 0.593 | 2.25 | 0.829 |
| off_topic_answer | 0.007 | 0.638 | 2.33 | 0.804 |

**Сигналы (hallucination − clean):**
- selfcheck contradiction: **+0.939** — очень сильный сигнал на галлюцинацию.
- semantic entropy: **+0.009** — практически нулевой (см. ограничения).

---

## 6. Честные наблюдения и ограничения

Это важнее «зелёных» чисел — они объясняют, чему на этих данных верить нельзя.

1. **Метки синтетические, числа отладочные.** Псевдо-корпус — прокси, генератор и судья
   имеют один backbone. Числа не переносятся на реальный корпус и не идут в отчёт проекта.

2. **Судья путает оси на `off_topic`, и промптом это НЕ чинится (проверено).** По
   таксономии off_topic-ответ верен по абзацу (faith=1), но судья давал p_faith ≈ 0.000 —
   переносил провал relevance на faithfulness. Правка B1 добавила в SEED_INSTRUCTION явное
   правило независимости осей с примером off_topic. Результат повторного прогона: off_topic
   p_faith **0.000 → 0.005** — правило на 7B-судье практически не сработало для этого типа.
   Побочные сдвиги были: clean p_faith 0.792→0.884, hallucination 0.000→0.110, а off_topic
   p_rel вырос 0.029→0.289 (сигнал rel ослаб с +0.899 до +0.647). Итог f1_reliable подрос
   0.799→0.829 в основном за счёт лучшего подбора порогов, f1_faith чуть просел 0.612→0.603.
   **Вывод:** независимость осей — задача для GEPA/few-shot (нужен few-shot-пример off_topic
   с явным PASS по faith), а не для одной строки промпта. На реальных данных ось faith/rel
   будет чище (off_topic — артефакт синтетической разметки).

3. **Semantic entropy почти не даёт сигнала (Δ+0.009).** На 5 сэмплах и 20 кейсах
   кластеризация шумная, а на «уверенных» типах сэмплы схлопываются в один кластер
   (`share_single_cluster_test = 0.20`). Это ожидаемый alignment-collapse (см. docs/04):
   на кейсах, где все сэмплы согласованы, consistency-сигнал слеп. Рабочую нагрузку по
   faithfulness в Методе 6 сейчас несёт **selfcheck contradiction** (Δ+0.94), не энтропия.

4. **Метод 6 обучен на 20 кейсах.** Изотоника и логрег на таком объёме шумны (пороги
   вроде t_faith=0.01 — симптом малой выборки, не качество). Цель этапа — «пайплайн жив»,
   не метрика.

5. **Reflection-LM и GEPA не проверялись.** Оптимизация промптов (Метод 3, ступень gepa)
   и few_shot вне scope этапа −1 — это отдельный план. `few_shot.yaml` ещё не существует.

6. **cos_q_a слабо разделяет.** Косинус вопрос/ответ у всех типов 0.80–0.91; как
   отдельный relevance-сигнал он слаб, работает только в связке логрега.

7. **Псевдо-корпус систематически смещён в пользу Метода 6 по оси H4.** Галлюцинации в нём —
   инъецированные пертурбации: правильный ответ существует в модели, пересэмплирование его
   воспроизводит, а подменённый факт выделяется как рассогласование — поэтому
   selfcheck-contradiction на них так силён (Δ+0.94). На реальных галлюцинациях бота, где
   модель уверенно повторяет один и тот же неверный факт во всех сэмплах, consistency-сигнал
   будет слабее. Числа Метода 6 на псевдо-корпусе — оптимистичная верхняя граница, не
   перенос на реальные данные.

---

## 7. Соблюдение правил проекта

- **Local-vs-cloud data rule:** каждый внешний вызов гейтится — m3/m6 через `case=`
  (per-request guard) + `assert_cloud_safe` на весь файл; генератор псевдо-корпуса —
  явным `public_data=True` (SberQuAD публичен). Не-`pseudo_` кейс на OpenRouter не уйдёт.
- **Секреты:** `.env`, `artifacts/`, `data/` в `.gitignore`; `api_key` в `run.yaml`
  редактируется до `***`; в коммитах ключа нет.
- **Детерминизм:** каждый прогон пишет `run.yaml` (конфиг + git-хэш + seed).
- **Изоляция cloud-выходов:** `predictions/cloud/`, `artifacts/cloud/` — отдельно от боевых.
- **Кэш переживает перезапуск:** судья, сэмплы m6, фичи m6, генерации псевдо-корпуса —
  атомарная запись (`tmp`+`replace`) и устойчивое чтение (битый файл от обрыва → промах,
  а не падение).

---

## 8. Как воспроизвести

```bash
# ключ провайдера
export OPENROUTER_API_KEY=...   # или в .env

# тесты (без GPU/сети)
pytest tests/ -x -q                       # 44 passed

# 1) провайдер отдаёт top_logprobs на токенах вердикта?
python -m tools.smoke_logprobs --config configs/config.cloud.yaml -n 3

# 2) псевдо-корпус (кэш в artifacts/pseudo_gen; --limit N -> отдельные __smokeN файлы)
python -m tools.make_pseudo_corpus --config configs/config.cloud.yaml

# 3) Метод 3 zero_shot
python -m src.m3.predict --config configs/config.cloud.yaml --mode zero_shot --split val

# 4) Метод 6 (20 кейсов)
for s in train val test; do
  python -m src.m6.sample   --config configs/config.cloud.yaml --split $s --limit 20
  python -m src.m6.features --config configs/config.cloud.yaml --split $s --limit 20
done
python -m src.m6.predict --config configs/config.cloud.yaml --limit 20

# 5) сигналы работоспособности
python -m tools.check_signals --config configs/config.cloud.yaml --split val \
    --m3-pred predictions/cloud/m3/zero_shot/val.jsonl \
    --m6-features artifacts/cloud/m6_features/val.jsonl
```

Переключение на local-профиль (`configs/config.yaml`) кода не меняет — только
`api_base`/`model`/данные.

---

## 9. Что дальше (вне этапа −1)

- Метод 3: few_shot (`configs/few_shot.yaml`), GEPA-оптимизация (auto=light, train=50),
  H5-прогоны gepa_markers vs gepa_plain — отдельный план.
- Метод 6: абляции N∈{3,5,10} и T из того же кэша, резка длинных premise с перекрытием.
- Переход на local vLLM + реальный корпус кураторов → боевые числа в сводную таблицу.

Отложенные мелкие замечания ревью (docstrings, type hints, хардкод пути few_shot)
зафиксированы в `.superpowers/sdd/progress.md` — они вне scope этапа −1.
