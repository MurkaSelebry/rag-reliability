"""Оконное разбиение длинной премисы в NLIScorer (docs/04, долг задачи 6 шаг 1).

Чистая логика (split_tokens / aggregate_windows / _split_premise / score)
тестируется на стабах токенизатора и модели — без сети и реальных весов.
"""

from types import SimpleNamespace

import torch

from src.m6.nli import NLIScorer, aggregate_windows, split_tokens

# --------------------------------------------------------------------------
# Чистые функции


def test_split_tokens_windows():
    toks = list(range(10))
    wins = split_tokens(toks, budget=4, overlap=2)
    assert wins[0] == [0, 1, 2, 3] and wins[1][0] == 2  # шаг = budget - overlap
    assert wins[-1][-1] == 9  # хвост покрыт
    assert split_tokens(list(range(3)), budget=4, overlap=2) == [[0, 1, 2]]


def test_split_tokens_exact_fit_single_window():
    assert split_tokens(list(range(4)), budget=4, overlap=2) == [[0, 1, 2, 3]]


def test_split_tokens_every_token_covered():
    toks = list(range(23))
    wins = split_tokens(toks, budget=7, overlap=3)
    seen = {t for w in wins for t in w}
    assert seen == set(toks)
    assert all(len(w) <= 7 for w in wins)


def test_aggregate_windows_max():
    scores = [
        {"entail": 0.1, "contra": 0.0},
        {"entail": 0.9, "contra": 0.2},
        {"entail": 0.3, "contra": 0.05},
    ]
    groups = [0, 0, 1]  # первые два окна -> пара 0, третье -> пара 1
    agg = aggregate_windows(scores, groups, n_pairs=2)
    assert agg[0]["entail"] == 0.9 and agg[0]["contra"] == 0.2
    assert agg[1]["entail"] == 0.3


# --------------------------------------------------------------------------
# Стабы токенизатора/модели (пословная токенизация, «токен» = слово)


class _StubEnc(dict):
    def to(self, device):  # noqa: ARG002 - совместимость с .to(self.device)
        return self


class StubTok:
    """Пословный токенизатор: одиночный вызов -> ids-слова, парный батч -> _StubEnc."""

    def __call__(self, a, b=None, add_special_tokens=True, **kw):
        if isinstance(a, str):  # режим _split_premise: один текст, без спецтокенов
            assert add_special_tokens is False
            return {"input_ids": a.split()}
        # батчевый режим score: списки премис/гипотез
        return _StubEnc(windows=list(a), hyps=list(b))

    def decode(self, ids):
        return " ".join(ids)


class StubModel:
    """entail высокий, только если в окне есть слово 'signal'; иначе neutral."""

    def __call__(self, windows, hyps):  # noqa: ARG002
        logits = [[8.0, 0.0, -8.0] if "signal" in w.split() else [-8.0, 0.0, 8.0] for w in windows]
        return SimpleNamespace(logits=torch.tensor(logits))


def _make_scorer(max_length: int = 48, overlap: int = 8, bs: int = 2) -> NLIScorer:
    s = NLIScorer.__new__(NLIScorer)
    s.tok = StubTok()
    s.model = StubModel()
    s.device = "cpu"
    s.bs = bs
    s.max_length = max_length
    s.overlap = overlap
    s.i_ent, s.i_con = 0, 2
    return s


# --------------------------------------------------------------------------
# _split_premise


def test_split_premise_short_returns_original():
    s = _make_scorer()
    prem = "a b c"
    assert s._split_premise(prem, "h1 h2") == [prem]


def test_split_premise_long_windows_with_overlap():
    s = _make_scorer(max_length=48, overlap=8)
    prem = " ".join(f"w{i}" for i in range(100))
    hyp = "h1 h2 h3 h4"  # budget = 48 - 4 - 4 = 40, stride = 32
    wins = s._split_premise(prem, hyp)
    assert len(wins) > 1
    assert wins[0].split() == [f"w{i}" for i in range(40)]
    assert wins[1].split()[0] == "w32"  # шаг = budget - overlap
    assert wins[-1].split()[-1] == "w99"  # хвост не потерян


def test_split_premise_tiny_budget_fallback():
    # budget = 10 - 5 - 4 = 1 <= 32 -> одно окно, обрезка до 32 токенов
    s = _make_scorer(max_length=10)
    prem = " ".join(f"w{i}" for i in range(40))
    wins = s._split_premise(prem, "h1 h2 h3 h4 h5")
    assert len(wins) == 1
    assert wins[0].split() == [f"w{i}" for i in range(32)]


# --------------------------------------------------------------------------
# score: сигнал в хвосте длинной премисы не теряется, контракт сохранён


def test_score_catches_signal_in_tail():
    s = _make_scorer(max_length=48, overlap=8, bs=2)
    # signal стоит на позиции 90 — при усечении до budget=40 он был бы потерян
    words = [f"w{i}" for i in range(100)]
    words[90] = "signal"
    long_prem = " ".join(words)
    pairs = [
        (long_prem, "h1 h2 h3 h4"),  # сигнал только в хвостовом окне
        ("a b c", "h1 h2 h3 h4"),  # короткая премиса без сигнала
    ]
    out = s.score(pairs)
    assert len(out) == len(pairs)  # контракт: один dict на пару, порядок сохранён
    assert set(out[0]) == {"entail", "contra"}
    assert out[0]["entail"] > 0.9  # хвостовое окно поймано (max по окнам)
    assert out[1]["entail"] < 0.1


def test_score_empty_pairs():
    assert _make_scorer().score([]) == []
