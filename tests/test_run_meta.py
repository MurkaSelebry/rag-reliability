"""run.yaml: конфиг с редакцией секретов + git-хэш + произвольные поля."""

import yaml

from src.common.run_meta import cost_stats, save_run_yaml


def test_run_yaml_redacts_api_key(tmp_path):
    cfg = {"profile": "cloud", "llm": {"api_key": "sk-secret", "model": "m"}, "m3": {"seed": 0}}
    save_run_yaml(tmp_path, cfg, seed=0, split="val")
    d = yaml.safe_load((tmp_path / "run.yaml").read_text(encoding="utf-8"))
    assert d["config"]["llm"]["api_key"] == "***"  # секрет не утёк
    assert cfg["llm"]["api_key"] == "sk-secret"  # исходный cfg не тронут
    assert d["profile"] == "cloud" and d["seed"] == 0 and d["split"] == "val"
    assert "git_hash" in d


def test_cost_stats():
    s = cost_stats([10, 20, 30], n_calls=2, n_cases=2)
    assert s["median_ms_per_case"] == 20.0  # медиана длительностей
    assert s["llm_calls_per_case"] == 1.0  # 2 вызова / 2 кейса
