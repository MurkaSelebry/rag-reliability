"""Промпты Метода 3. Ступень 1 (zero-shot) и ступень 2 (few-shot вручную).
Ступень 3 (GEPA) стартует с SEED_INSTRUCTION и эволюционирует её."""

SEED_INSTRUCTION = """Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.

Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.

FAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты: не добавляет фактов, которых нет в [CTX]; не искажает числа, ставки, сроки и условия; не смешивает информацию из разных фрагментов так, что получается неверное утверждение; не опускает важные детали и оговорки из [CTX], меняющие смысл.
FAITHFULNESS = FAIL в противном случае.

RELEVANCE = PASS, если ответ полностью соответствует вопросу клиента: отвечает именно на заданный вопрос (с учётом истории диалога), а не на смежный; покрывает все части вопроса; не уходит в общие слова вместо ответа.
RELEVANCE = FAIL в противном случае.

ВАЖНО: оси независимы. FAITHFULNESS оценивается ТОЛЬКО против [CTX]: если ответ верен по фрагментам, но не относится к вопросу — это FAITHFULNESS: PASS и RELEVANCE: FAIL, а не двойной FAIL. И наоборот: ответ точно по теме вопроса, но с фактами не из [CTX] — это RELEVANCE: PASS и FAITHFULNESS: FAIL. Не переноси ошибку одной оси на другую.

Сначала кратко проанализируй ответ, затем выдай вердикты строго в формате:
FAITHFULNESS: PASS или FAIL
RELEVANCE: PASS или FAIL"""

USER_TEMPLATE = """[Q]
{q}

[CTX]
{ctx}

[A]
{a}

Проанализируй и выдай вердикты в заданном формате."""


def build_user_prompt(case, max_ctx_chars: int | None = None) -> str:
    return USER_TEMPLATE.format(q=case.q_text(), ctx=case.ctx_text(max_ctx_chars), a=case.answer)


# --- Few-shot (ступень 2): примеры подставляются в system после инструкции.
# Отбор вручную из dev-train: покрыть главные маркеры + оба PASS-кейса.
# Заполнить реальными кейсами после получения данных; структура:
FEW_SHOT_TEMPLATE = """

Примеры оценок:

Пример {i}.
[Q] {q}
[CTX] {ctx}
[A] {a}
Анализ: {analysis}
FAITHFULNESS: {faith}
RELEVANCE: {rel}"""


def build_few_shot_system(examples: list[dict]) -> str:
    """examples: [{q, ctx, a, analysis, faith: 'PASS|FAIL', rel: 'PASS|FAIL'}]"""
    blocks = [FEW_SHOT_TEMPLATE.format(i=i + 1, **ex) for i, ex in enumerate(examples)]
    return SEED_INSTRUCTION + "".join(blocks)
