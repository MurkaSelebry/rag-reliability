# 05. Задачи (вести чекбоксы здесь)

## Этап -1 — cloud-режим (пока нет GPU/корпуса; см. docs/07)
- [x] configs/config.cloud.yaml: профиль OpenRouter (создан; вписать OPENROUTER_API_KEY в окружение)
- [x] Поддержка в загрузчике конфига: подстановка ${VAR} и проброс openrouter_extra_body
- [x] Smoke-тест logprobs провайдера (top_logprobs на токенах вердикта)
- [x] Guard в LLM-клиенте: cloud-профиль + не-synthetic данные = ошибка
- [x] tools/make_pseudo_corpus.py по спецификации docs/07.2 (+ ручная проверка 5×4 кейсов)
- [x] Прогон m3 zero_shot и m6 (20 кейсов) на псевдо-корпусе; проверка сигналов из docs/07.3

## Этап 0 — фундамент
- [ ] Получить сплиты платформы -> data/processed/ ; проверить схему loader-ом
- [ ] Получить у кураторов словарь 13 маркеров -> configs/markers.yaml
- [ ] Уточнить у кураторов промпт/модель реального бота (для Метода 6)
- [ ] Поднять vLLM с backbone; проверить logprobs у chat.completions
- [ ] src/common: schemas.py (Case/Pred, jsonl IO), eval_local.py
      (fit_thresholds, evaluate), judge_client.py (chat, judge, кэш вызовов)
- [ ] tests: извлечение PASS/FAIL вероятностей (4 сценария), пороги, метрики
- [ ] Smoke: eval_local на синтетике даёт осмысленные числа

## Метод 3
- [ ] prompts.py: SEED_INSTRUCTION, USER_TEMPLATE, few-shot builder
- [ ] predict.py: CLI (--mode, --split, --limit), run.yaml, report
- [ ] Прогон zero_shot val+test -> первая строка таблицы
- [ ] configs/few_shot.yaml: 6–8 примеров с ручным «Анализом»
- [ ] Прогон few_shot val+test
- [ ] run_gepa.py: DSPy-программа, метрика с feedback, сохранение промпта и статистики
- [ ] GEPA smoke: auto=light, train=50
- [ ] gepa_markers: 2 seed, auto=medium
- [ ] gepa_plain: те же seed/бюджет (H5)
- [ ] Инференс gepa-вариантов val+test; контроль зазора val/test
- [ ] Стоимость; сдача predictions платформе

## Метод 6
- [ ] nli.py: батчевый NLIScorer (id2label из конфига модели, fp16, overlap)
- [ ] sample.py: кэш поэлементно, --limit, retry; train/val/test
- [ ] features.py: selfcheck-NLI, semantic entropy (union-find), cos_q_a;
      инкрементальная запись; тесты на стабе NLI
- [ ] predict.py: изотоника (val) для p_faith, логрег (train) для p_rel,
      отчёт + share_single_cluster + подмножество n_clusters==1
- [ ] Абляции: N ∈ {3,5,10} из того же кэша; (опц.) T, entail_threshold
- [ ] Стоимость (мс/кейс по этапам); сдача predictions платформе

## Отчёт
- [ ] H5: таблица gepa_markers vs gepa_plain (равный бюджет), 2 seed
- [ ] H4: faith vs rel колонки Метода 6 + alignment-collapse анализ
- [ ] Вклад в H1: наша строка gepa с тем же backbone, что Метод 1
- [ ] Эволюция GEPA-промпта (для презентации)
- [ ] limitations: reflection-LM, прокси-генератор, supervised p_rel в m6
