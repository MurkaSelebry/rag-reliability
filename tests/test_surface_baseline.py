"""Поверхностные фичи: длины, overlap, извлечение — без моделей."""

from rag_reliability.baselines.surface import ngram_overlap, surface_features
from rag_reliability.common.schemas import Case


def test_ngram_overlap_bounds():
    assert ngram_overlap("ставка 19 процентов", "ставка 19 процентов", n=2) == 1.0
    assert ngram_overlap("абв где", "жзи клм", n=2) == 0.0


def test_surface_features_keys_and_types():
    c = Case(
        id="alfa_x",
        query="Какая ставка?",
        context=["Ставка 19.84%."],
        answer="Ставка 19.84%.",
        meta={"synthetic": False},
    )
    f = surface_features(c)
    for k in (
        "len_answer",
        "len_ctx",
        "len_query",
        "n_chunks",
        "overlap_ans_ctx_2",
        "overlap_ans_q_1",
        "digit_match_ratio",
    ):
        assert k in f and isinstance(f[k], (int, float))
    assert f["digit_match_ratio"] == 1.0


def test_digit_match_ratio_missing_digit():
    c = Case(
        id="alfa_y",
        query="Какая ставка?",
        context=["Ставка 19.84% годовых."],
        answer="Ставка 25% и комиссия 19.84%.",
        meta={"synthetic": False},
    )
    f = surface_features(c)
    # 25 нет в контексте, 19.84 есть → ratio строго между 0 и 1
    assert 0.0 < f["digit_match_ratio"] < 1.0


def test_empty_answer_and_context_robust():
    c = Case(id="alfa_z", query="", context=[], answer="", meta={"synthetic": False})
    f = surface_features(c)
    for v in f.values():
        assert isinstance(v, (int, float))
        assert v == v  # не NaN
    # ответ без цифр → digit_match_ratio == 1.0 по спецификации
    assert f["digit_match_ratio"] == 1.0
    assert ngram_overlap("", "", n=2) == 0.0
