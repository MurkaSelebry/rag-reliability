# 04. Метод 6 — SelfCheckGPT + Semantic Entropy

## Идея

Unsupervised consistency: сгенерировать N альтернативных ответов на тот же
(Q, CTX) и померить согласованность исходного ответа с ними. Разметка
используется только для калибровки порога/вероятности на dev_val (как в
статье проекта: «threshold calibration on val»). Проверяет **H4**: сигнал на
faithfulness есть, на relevance — нет (модель может консистентно отвечать не
на тот вопрос).

## Архитектура (3 независимых этапа, всё кэшируется)

```
этап 1  sample.py
  промпт «бота» (максимально близкий к реальному боту банка; уточнить у
  кураторов, дефолт в спецификации ниже) + [CTX] + [Q]
  -> N=10 сэмплов, T=0.8, top_p=0.95, один запрос с n=10
  -> artifacts/m6_samples/{split}/{id}.json   (поэлементный кэш)

этап 2  features.py                        -> artifacts/m6_features/{split}.jsonl
  ├─ SelfCheck-NLI: ответ -> предложения (razdel.sentenize);
  │    пары (premise=сэмпл_j, hypothesis=предложение_i) -> mDeBERTa-XNLI
  │    P(contradiction); скор предложения = mean_j; фичи кейса:
  │    selfcheck_contra_mean, selfcheck_contra_max (по предложениям)
  ├─ Semantic entropy: тексты = [answer] + samples;
  │    для всех пар двунаправленный P(entail); эквивалентность =
  │    оба направления > entail_threshold (0.5); union-find ->
  │    кластеры; фичи: semantic_entropy = -Σ p log p по размерам кластеров,
  │    n_clusters, answer_in_top_cluster ∈ {0,1}
  └─ cos_q_a: multilingual-e5 ("query: {q}" vs "passage: {a}"),
       normalize, скалярное произведение (relevance-сигнал)

этап 3  predict.py                          -> predictions/m6/{split}.jsonl
  ├─ p_faith: raw_unfaith = z(selfcheck_contra_mean) + z(semantic_entropy)
  │    (z-нормировка по train); IsotonicRegression на dev_val:
  │    iso.fit(-raw_unfaith(val), y_faith(val)); p_faith = iso.predict(-raw)
  ├─ p_rel: LogisticRegression(class_weight="balanced") на dev_train по всем
  │    фичам + cos_q_a  (сознательное отступление от чистого unsupervised —
  │    оговорить в отчёте; faith-ветка supervised-сигнал не использует)
  └─ отчёт: метрики + share_single_cluster_test + метрики на подмножестве
       n_clusters == 1 (иллюстрация alignment-collapse: на кейсах, где все
       сэмплы схлопнулись в один кластер, consistency-сигнал слеп)
```

## Промпт сэмплера (дефолт до уточнения у кураторов)

system: «Ты — ассистент банка для корпоративных клиентов. Отвечай на вопрос
клиента, используя только предоставленные фрагменты документации. Если ответа
в фрагментах нет, скажи об этом.»
user: «Фрагменты документации:\n{ctx}\n\nВопрос клиента: {q}\n\nОтвет:»

## Этапы реализации

1. **NLIScorer.** Батчевый скорер на
   `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`: пары
   (premise, hypothesis) -> {entail, contra}; индексы классов читать из
   model.config.id2label (не хардкодить); fp16 на CUDA; max_length 512,
   длинные premise резать с перекрытием и брать max по кускам.
2. **sample.py.** Поэлементный кэш (пропуск существующих файлов), `--limit N`
   для smoke, retry на сетевые ошибки. Прогнать train/val/test.
3. **features.py.** Инкрементальная дозапись jsonl (skip по id); юнит-тесты
   чистой логики на стабе NLI: кластеризация (известные группы), энтропия,
   selfcheck-агрегация, сегментация предложений.
4. **predict.py.** Изотоника + логрег + отчёт; пороги через общий eval_local.
5. **Абляции.** N ∈ {3,5,10} — подмножества того же кэша (не пересэмплировать);
   T ∈ {0.5, 0.8, 1.0} — отдельные кэш-директории (`m6_samples_t05` и т.п.);
   entail_threshold ∈ {0.4, 0.5, 0.6}. Кривая качество/цена по N — в отчёт.
6. **Стоимость.** Мс/кейс на каждом этапе, число LLM-вызовов (=1 запрос n=10),
   число NLI-пар на кейс.

## Конфиг (configs/config.yaml, секция m6)

n_samples, temperature, top_p, max_new_tokens, nli_model, embed_model,
entail_threshold, samples_cache, features_cache, out_pred_dir.

## Грабли

- Число NLI-пар растёт как (число предложений × N) + (N+1)² — батчить,
  следить за временем; при длинных ответах ограничить предложения топ-К
  по длине? Нет: считать все, но логировать 95-й перцентиль времени.
- Пустые/односложные сэмплы («Информации нет») ломают NLI — не фильтровать,
  это легитимный сигнал согласованности, но убедиться, что NLI их переваривает.
- Прокси-генератор ≠ реальный бот: limitation в отчёте; если дадут реальный
  промпт/модель — пересэмплировать в новую кэш-директорию.
- e5 требует префиксы "query: "/"passage: " — не забывать.
- contradiction ≠ not-entailed: для SelfCheck используем именно contradiction
  (по оригинальной статье), для кластеризации — entailment. Не путать.

## Definition of done

Кэши sample/features на трёх сплитах; predictions val+test; report_test.json
со share_single_cluster и подмножеством; кривая N∈{3,5,10}; тесты зелёные.
