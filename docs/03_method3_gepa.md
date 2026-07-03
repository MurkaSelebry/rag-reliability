# 03. Метод 3 — LLM-as-judge с промпт-оптимизацией (GEPA/DSPy)

## Идея

Backbone без изменения весов выносит вердикты PASS/FAIL по двум осям.
Три ступени, каждая — отдельный variant и строка сводной таблицы:

1. `zero_shot` — рукописная инструкция с определениями faith/rel;
2. `few_shot` — та же инструкция + 6–8 вручную отобранных примеров;
3. `gepa` — инструкция, автоматически эволюционированная GEPA по f1-macro на
   dev_val; в двух под-вариантах: `gepa_markers` (feedback с маркерами) и
   `gepa_plain` (feedback без маркеров) — их разница при равном бюджете и
   seed = проверка **H5**.

## Архитектура

```
prompts.py: SEED_INSTRUCTION + USER_TEMPLATE([Q]/[CTX]/[A]) + few-shot builder
                     │
     ┌───────────────┴────────────────────────────┐
     │ run_gepa.py (только оптимизация)           │
     │  DSPy ChainOfThought(Judge signature)      │
     │  metric(gold, pred, trace, pred_name,      │
     │         pred_trace) -> Prediction(         │
     │            score  = (ok_faith+ok_rel)/2,   │
     │            feedback = истинные метки        │
     │                     [+ расшифровки маркеров])│
     │  reflection_lm = локальная 32B             │
     │  результат: artifacts/m3_prompt_{variant}.txt│
     └───────────────┬────────────────────────────┘
                     │ текст инструкции
     ┌───────────────▼────────────────────────────┐
     │ predict.py (инференс всех ступеней)        │
     │  JudgeClient.judge(system, user):          │
     │   одна генерация T=0, logprobs=True;       │
     │   p = softmax(logprob_PASS, logprob_FAIL)  │
     │   на 1-й (faith) и 2-й (rel) позиции       │
     │   вердикта; fallback regex -> 0.9/0.1;     │
     │   не распарсили -> 0.5/0.5                 │
     └────────────────────────────────────────────┘
```

**Ключевое решение:** DSPy — только для оптимизации; инференс — через
собственный JudgeClient, потому что (а) нужны калибруемые вероятности из
logprobs, (б) схема извлечения должна совпадать с Методами 1–2 ради честного H1.

## Формат промпта судьи

system (SEED_INSTRUCTION, ступень 1; у ступени 3 заменяется эволюционированной):

```
Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.
FAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО на предоставленные
фрагменты: не добавляет фактов вне [CTX]; не искажает числа, ставки, сроки;
не смешивает информацию из разных фрагментов; не опускает важные детали.
Иначе FAIL.
RELEVANCE = PASS, если ответ полностью соответствует вопросу клиента (с учётом
истории диалога), покрывает все части вопроса, не подменяет ответ общими
словами. Иначе FAIL.
Сначала кратко проанализируй ответ, затем выдай вердикты строго в формате:
FAITHFULNESS: PASS или FAIL
RELEVANCE: PASS или FAIL
```

user:

```
[Q]\n{q}\n\n[CTX]\n{ctx}\n\n[A]\n{a}\n\nПроанализируй и выдай вердикты в заданном формате.
```

## Этапы реализации

1. **JudgeClient + извлечение вероятностей.** OpenAI-клиент на vLLM;
   `chat(...)` с retry/backoff; `judge(...)` с извлечением PASS/FAIL logprobs
   (искать в top_logprobs=20 позиции, где сгенерирован токен PASS или FAIL;
   softmax по паре; порядок появления: 1-я позиция = faith, 2-я = rel).
   Юнит-тесты на стабах: обе позиции найдены / одна / ни одной / regex-fallback.
2. **Ступень zero_shot.** predict.py: прогон val и test, отчёт через eval_local
   (пороги с val). Это первая строка и санити всей ветки.
3. **Ступень few_shot.** Отобрать 6–8 кейсов из dev_train: покрыть основные
   маркеры + 2 PASS/PASS-кейса; каждому написать короткий «Анализ: ...» вручную.
   Примеры живут в `configs/few_shot.yaml`, не в коде.
4. **GEPA.** DSPy-программа (signature с Literal["PASS","FAIL"] выходами,
   docstring = SEED_INSTRUCTION), метрика с feedback, `dspy.GEPA(metric=...,
   auto="light"→"medium", reflection_lm=..., track_stats=True, seed=...)`,
   `compile(program, trainset=подвыборка dev_train 300, valset=dev_val)`.
   Сохранить: эволюционированную инструкцию (txt), program.json, статистику
   эволюции (для отчёта). Сначала smoke на auto="light" и train=50.
5. **H5-прогоны.** gepa_markers и gepa_plain: одинаковые seed, бюджет,
   подвыборка; различие только в feedback-функции.
6. **Инференс gepa-вариантов + отчёты.** Зазор val/test > 3–4 пунктов f1 =
   промпт переобучился: уменьшить gepa_train_size или бюджет, повторить.
7. **Стоимость и сдача.** run.yaml, report_test.json, чекбоксы в 05_tasks.md.

## Конфиг (configs/config.yaml, секция m3)

mode (zero_shot|few_shot|gepa), gepa_auto (light|medium), gepa_train_size (300),
use_marker_feedback (bool), seed, out_prompt, out_pred_dir.

## Грабли

- vLLM должен возвращать logprobs у chat.completions (`logprobs=True,
  top_logprobs=20`); если конкретная версия отдаёт токены иначе (BPE-подтокены
  «PA», «SS») — матчить по склейке подтокенов, тест на это обязателен.
- Не использовать guided_choice при CoT-выводе — задушит рассуждение;
  формат держится инструкцией + fallback.
- Кандидаты GEPA выбираются только по dev_val; dev_test — единственный
  финальный замер.
- Reflection-LM = backbone (если 32B недоступна) заметно ослабляет GEPA —
  зафиксировать в limitations.
- Кэшировать вызовы судьи по хэшу (system, user) — повторные прогоны бесплатны.

## Definition of done

predictions val+test для zero_shot, few_shot, gepa_markers, gepa_plain;
2 seed у gepa-вариантов; report_test.json на каждый; эволюция промптов
сохранена; тесты извлечения вероятностей зелёные.
