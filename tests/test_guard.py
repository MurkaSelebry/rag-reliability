"""Guard: cloud-профиль допускает только синтетические кейсы (CLAUDE.md, docs/07.1)."""

import pytest

from rag_reliability_m3m6.common.guard import (
    DataLeakError,
    assert_case_cloud_safe,
    assert_cloud_safe,
    is_synthetic,
)
from rag_reliability_m3m6.common.schemas import Case


def _case(id="case_001", **meta):
    return Case(id=id, query="q", context=["c"], answer="a", meta=meta)


def test_is_synthetic_by_prefix():
    assert is_synthetic(_case(id="pseudo_00001"))


def test_is_synthetic_by_flag():
    assert is_synthetic(_case(id="whatever", synthetic=True))


def test_not_synthetic():
    assert not is_synthetic(_case(id="case_00317"))


def test_local_profile_allows_anything():
    assert_cloud_safe([_case()], profile="local")  # не бросает


def test_cloud_profile_rejects_real_data():
    with pytest.raises(DataLeakError):
        assert_cloud_safe([_case(id="pseudo_1"), _case(id="case_00317")], profile="cloud")


def test_cloud_profile_allows_synthetic():
    assert_cloud_safe([_case(id="pseudo_1"), _case(id="x", synthetic=True)], profile="cloud")


def test_allow_real_opt_in_passes_and_default_blocks():
    """Явный opt-in владельца данных пропускает реальный кейс; дефолт — блокирует."""
    real = Case(
        id="alfa_ab12cd34ef56",
        query="q",
        context=["c"],
        answer="a",
        meta={"synthetic": False, "source": "alfa"},
    )
    assert_case_cloud_safe(real, "cloud", allow_real=True)  # не бросает
    assert_cloud_safe([real], "cloud", allow_real=True)  # не бросает
    with pytest.raises(DataLeakError):
        assert_case_cloud_safe(real, "cloud")  # дефолт строгий


def test_real_alfa_case_blocked_in_cloud():
    """Реальный кейс кураторов не проходит в cloud ни при каких условиях."""
    real = Case(
        id="alfa_ab12cd34ef56",
        query="q",
        context=["c"],
        answer="a",
        meta={"synthetic": False, "source": "alfa"},
    )
    assert not is_synthetic(real)
    with pytest.raises(DataLeakError):
        assert_case_cloud_safe(real, "cloud")
