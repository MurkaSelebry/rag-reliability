"""tracking: локальный file-store, параметры/метрики/артефакты, редакция секретов."""

import mlflow

from rag_reliability_m3m6.common.tracking import log_run


def test_log_run_local_store(tmp_path):
    uri = f"file:{tmp_path / 'mlruns'}"
    cfg = {
        "profile": "cloud",
        "llm": {"api_key": "sk-secret", "model": "m"},
        "m3": {"mode": "few_shot", "seed": 0},
    }
    rep = tmp_path / "report_val.json"
    rep.write_text('{"f1_macro_reliable": 0.82}')
    run_id = log_run(
        tracking_uri=uri,
        experiment="m3",
        run_name="few_shot/val",
        cfg=cfg,
        metrics={"f1_macro_reliable": 0.82},
        artifacts=[rep],
        tags={"split": "val", "variant": "few_shot"},
    )
    client = mlflow.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.metrics["f1_macro_reliable"] == 0.82
    assert run.data.params["llm.model"] == "m"
    assert "sk-secret" not in str(run.data.params)  # секрет не утёк
    assert run.data.tags["variant"] == "few_shot"


def test_log_run_twice_and_missing_artifact(tmp_path):
    """Два прогона в одном experiment сосуществуют; отсутствующий артефакт не роняет прогон."""
    uri = f"file:{tmp_path / 'mlruns'}"
    cfg = {
        "profile": "local",
        "llm": {"api_key": "sk-x", "model": "m"},
        "m3": {"mode": "zero_shot", "seed": 1},
    }
    missing = tmp_path / "does_not_exist.json"  # отсутствующий путь — терпимо
    run_id1 = log_run(
        tracking_uri=uri,
        experiment="m3",
        run_name="a",
        cfg=cfg,
        metrics={"f1_macro_reliable": 0.5},
        artifacts=[missing],
        tags={"split": "val"},
    )
    run_id2 = log_run(
        tracking_uri=uri,
        experiment="m3",
        run_name="b",
        cfg=cfg,
        metrics={"f1_macro_reliable": 0.7},
        artifacts=None,
        tags={"split": "val"},
    )
    assert run_id1 != run_id2

    client = mlflow.MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("m3")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 2
    # отсутствующий артефакт пропущен молча, но метрики залогированы
    run1 = client.get_run(run_id1)
    assert run1.data.metrics["f1_macro_reliable"] == 0.5
