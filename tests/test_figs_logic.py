"""Чистая логика фигур: бины reliability diagram и кривая f1(t)."""

import pytest

from tools.make_figs import f1_threshold_curve, reliability_bins


def test_reliability_bins_means():
    probs = [0.05, 0.15, 0.95, 0.85, 0.9]
    labels = [0, 0, 1, 1, 0]
    bins = reliability_bins(probs, labels, n_bins=10)
    b0 = next(b for b in bins if b["lo"] == 0.0)  # [0.0, 0.1)
    assert b0["n"] == 1 and b0["mean_prob"] == pytest.approx(0.05)
    b9 = next(b for b in bins if b["hi"] == 1.0)  # [0.9, 1.0]
    assert b9["n"] == 2 and b9["frac_pos"] == pytest.approx(0.5)
    assert all(b["n"] == 0 or 0 <= b["frac_pos"] <= 1 for b in bins)


def test_f1_threshold_curve_peak():
    """Идеальное разделение: f1=1.0 на порогах между 0.2 и 0.8."""
    probs = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    ts, f1s = f1_threshold_curve(probs, labels, step=0.1)
    assert len(ts) == len(f1s)
    assert max(f1s) == pytest.approx(1.0)
    assert f1s[ts.index(pytest.approx(0.5, abs=1e-9))] == pytest.approx(1.0)
