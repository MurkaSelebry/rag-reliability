"""run.yaml: конфиг с редакцией секретов + git-хэш + произвольные поля."""
import yaml

from src.common.run_meta import save_run_yaml


def test_run_yaml_redacts_api_key(tmp_path):
    cfg = {"profile": "cloud", "llm": {"api_key": "sk-secret", "model": "m"}, "m3": {"seed": 0}}
    save_run_yaml(tmp_path, cfg, seed=0, split="val")
    d = yaml.safe_load((tmp_path / "run.yaml").read_text(encoding="utf-8"))
    assert d["config"]["llm"]["api_key"] == "***"          # секрет не утёк
    assert cfg["llm"]["api_key"] == "sk-secret"            # исходный cfg не тронут
    assert d["profile"] == "cloud" and d["seed"] == 0 and d["split"] == "val"
    assert "git_hash" in d
