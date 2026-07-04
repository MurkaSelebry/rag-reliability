# 05. Задачи (вести чекбоксы здесь)

## Этап -1 — cloud-режим (пока нет GPU/корпуса; см. docs/07)
- [x] configs/config.cloud.yaml: профиль OpenRouter (создан; вписать OPENROUTER_API_KEY в окружение)
- [x] Поддержка в загрузчике конфига: подстановка ${VAR} и проброс openrouter_extra_body
- [x] Smoke-тест logprobs провайдера (top_logprobs на токенах вердикта)
- [x] Guard в LLM-клиенте: cloud-профиль + не-synthetic данные = ошибка
- [x] tools/make_pseudo_corpus.py по спецификации docs/07.2 (+ ручная проверка 5×4 кейсов)
- [x] Прогон m3 zero_shot и m6 (20 кейсов) на псевдо-корпусе; проверка сигналов из docs/07.3

## Этап −0.5 — GEPA (cloud; docs/10)
- [x] configs/few_shot.yaml: 7 примеров с якорями обеих осей (B1/B2); приёмка на val
      (off_topic p_faith 0.202→0.240 — порог 0.5 не взят, ограничение 7B-судьи; rel-сигнал сохранён)
- [x] tools/make_figs.py: боксплоты по kind, reliability, f1(t), scatter m6 (+тесты логики)
- [x] configs/config.cloud.yaml m3.gepa; run_gepa.py (метрика ±markers, глосс из yaml); gepa_report.py; predict --prompt-file/--variant-name
- [x] GEPA smoke: auto=light, train=50 (checkpoint: ≤ few_shot → medium НЕ запускался, docs/10 §3)
- [x] H5-механика: gepa_markers/gepa_plain × 2 seed, train=100 auto=light (репетиция механики, не проверка H5)
- [x] tools/entropy_ablation.py: абляция thr×N из кэша (0 LLM-вызовов); интерпретация в docs/10 §4
- [x] docs/10_gepa_stage.md — итоговый отчёт этапа

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
- [x] run_gepa.py: DSPy-программа, метрика с feedback, сохранение промпта и статистики (cloud; docs/10)
- [x] GEPA smoke: auto=light, train=50 (cloud; docs/10 §3)
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
