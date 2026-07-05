"""Санити eval_local на синтетических предсказаниях (docs/05 этап 0)."""

from src.common.eval_local import evaluate, fit_thresholds
from src.common.schemas import Case, Pred


def _mk(n_good=20, n_bad=20):
    cases, preds = [], []
    for i in range(n_good):  # надёжные кейсы с высокими p
        cases.append(Case(id=f"g{i}", query="q", context=["c"], answer="a", faith=1, rel=1))
        preds.append(Pred(id=f"g{i}", p_faith=0.9, p_rel=0.9))
    for i in range(n_bad):  # ненадёжные с низкими p
        cases.append(Case(id=f"b{i}", query="q", context=["c"], answer="a", faith=0, rel=1))
        preds.append(Pred(id=f"b{i}", p_faith=0.1, p_rel=0.9))
    return cases, preds


def test_perfect_separation_gives_f1_one():
    cases, preds = _mk()
    tf, tr, val_f1 = fit_thresholds(cases, preds)
    assert val_f1 == 1.0
    rep = evaluate(cases, preds, tf, tr)
    assert rep["f1_macro_reliable"] == 1.0
    assert rep["n"] == 40


def test_cases_without_labels_skipped():
    cases, preds = _mk(5, 5)
    cases.append(Case(id="nolabel", query="q", context=["c"], answer="a"))  # faith=None
    preds.append(Pred(id="nolabel", p_faith=0.5, p_rel=0.5))
    tf, tr, _ = fit_thresholds(cases, preds)
    assert evaluate(cases, preds, tf, tr)["n"] == 10
