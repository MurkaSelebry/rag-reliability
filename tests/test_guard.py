"""Guard: cloud-профиль допускает только синтетические кейсы (CLAUDE.md, docs/07.1)."""

import pytest

from src.common.guard import DataLeakError, assert_cloud_safe, is_synthetic
from src.common.schemas import Case


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
