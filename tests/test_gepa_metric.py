"""Метрика GEPA: глосс маркеров из configs/markers.yaml, score и feedback ±маркеры."""
import dspy
import pytest

from src.m3.run_gepa import load_marker_gloss, make_metric


def _gold(f="PASS", r="FAIL", markers=()):
    return dspy.Example(faithfulness=f, relevance=r, markers=list(markers))


def _pred(f="PASS", r="FAIL"):
    return dspy.Prediction(faithfulness=f, relevance=r)


def test_gloss_loaded_from_yaml():
    gloss = load_marker_gloss("configs/markers.yaml")
    assert "hallucination" in gloss and "факт" in gloss["hallucination"]


def test_metric_score_halves():
    m = make_metric(use_markers=True, gloss={})
    assert m(_gold(), _pred()).score == 1.0                     # обе оси верны
    assert m(_gold(), _pred(f="FAIL")).score == 0.5             # одна ось
    assert m(_gold("FAIL", "PASS"), _pred()).score == 0.0       # ни одной


def test_feedback_contains_markers_when_enabled():
    gloss = {"hallucination": "в ответе есть факты, отсутствующие в контексте"}
    m = make_metric(use_markers=True, gloss=gloss)
    fb = m(_gold(f="FAIL", markers=["hallucination"]), _pred(f="PASS")).feedback
    assert "hallucination" in fb and "отсутствующие в контексте" in fb


def test_feedback_no_markers_when_disabled():
    """Вариант plain: истинные метки есть, маркеров нет (единственное различие H5)."""
    m = make_metric(use_markers=False, gloss={"hallucination": "x"})
    fb = m(_gold(f="FAIL", markers=["hallucination"]), _pred(f="PASS")).feedback
    assert "hallucination" not in fb and "FAITHFULNESS=FAIL" in fb


def test_correct_prediction_positive_feedback():
    m = make_metric(use_markers=True, gloss={})
    assert "верн" in m(_gold(), _pred()).feedback.lower()
