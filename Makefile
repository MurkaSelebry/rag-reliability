# Быстрые команды. Всё выполняется в .venv (создаётся `make install`).

PY := .venv/bin/python
CONFIG ?= configs/config.yaml
SPLIT ?= val
MODE ?= zero_shot
VARIANT ?= markers
SEED ?= 0
M3_PRED ?= predictions/cloud/m3/zero_shot/$(SPLIT).jsonl
LIMIT ?=
LIMIT_FLAG := $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: help install install-gepa install-m6 install-encoder install-viz install-tracking \
        install-data test lint check smoke m3 gepa gepa-report m6-samples m6-features \
        m6-predict m6 baseline-surface baseline-encoder pseudo-corpus splits figs report \
        explorer clean

help: ## Список доступных целей
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install: ## venv + ядро и dev-зависимости (uv)
	uv venv --python 3.11
	uv pip install -e ".[dev]"

install-gepa: ## + DSPy для GEPA-оптимизации промпта (Метод 3)
	uv pip install -e ".[gepa]"

install-m6: ## + NLI/эмбеддинг-стек Метода 6 (torch, transformers, sbert)
	uv pip install -e ".[m6]"

install-encoder: ## + стек supervised-энкодера кураторов
	uv pip install -e ".[encoder]"

install-viz: ## + matplotlib/plotly/streamlit для фигур, отчёта и эксплорера
	uv pip install -e ".[viz]"

install-tracking: ## + mlflow (локальный file-store)
	uv pip install -e ".[tracking]"

install-data: ## + datasets (псевдо-корпус) и dvc
	uv pip install -e ".[data]"

test: ## Юнит-тесты (стабы, без GPU и сети)
	$(PY) -m pytest

lint: ## Ruff lint
	$(PY) -m ruff check .

check: test lint ## Тесты + линт

smoke: ## Smoke-прогон Метода 3 на 10 кейсах (нужен доступ к LLM из CONFIG)
	$(PY) scripts/run_m3.py --config $(CONFIG) --mode $(MODE) --split $(SPLIT) --limit 10

m3: ## Метод 3: инференс судьи (CONFIG/MODE/SPLIT/LIMIT)
	$(PY) scripts/run_m3.py --config $(CONFIG) --mode $(MODE) --split $(SPLIT) $(LIMIT_FLAG)

gepa: ## Метод 3: GEPA-оптимизация (VARIANT/SEED; дорого — см. стоп-правила docs/10)
	$(PY) scripts/run_gepa.py --config $(CONFIG) --variant $(VARIANT) --seed $(SEED)

gepa-report: ## Markdown-отчёт эволюции GEPA из stats-json (VARIANT/SEED)
	$(PY) scripts/gepa_report.py --variant $(VARIANT) --seed $(SEED)

m6-samples: ## Метод 6, этап 1: сэмплы бота (поэлементный кэш)
	$(PY) scripts/prepare_m6_samples.py --config $(CONFIG) --split $(SPLIT) $(LIMIT_FLAG)

m6-features: ## Метод 6, этап 2: фичи selfcheck/entropy/cos
	$(PY) scripts/prepare_m6_features.py --config $(CONFIG) --split $(SPLIT) $(LIMIT_FLAG)

m6-predict: ## Метод 6, этап 3: калибровка на val -> predictions
	$(PY) scripts/run_m6_selfcheck.py --config $(CONFIG) $(LIMIT_FLAG)

m6: m6-samples m6-features m6-predict ## Метод 6: полный конвейер на SPLIT

baseline-surface: ## Бейзлайн surface(+e5)
	$(PY) scripts/run_surface_baseline.py --config $(CONFIG) $(LIMIT_FLAG)

baseline-encoder: ## Бейзлайн: supervised-энкодер кураторов
	$(PY) scripts/train_encoder_baseline.py --config $(CONFIG)

pseudo-corpus: ## Синтетический псевдо-корпус для cloud-отладки
	$(PY) scripts/make_pseudo_corpus.py --config $(CONFIG) $(LIMIT_FLAG)

splits: ## Групповые сплиты из корпуса (данные платформы не трогает)
	$(PY) scripts/make_splits.py --config $(CONFIG)

figs: ## Фигуры отладки по M3_PRED (matplotlib, см. install-viz)
	$(PY) scripts/make_figs.py --config $(CONFIG) --split $(SPLIT) --m3-pred $(M3_PRED)

report: ## Единый offline HTML-отчёт по всем прогонам
	$(PY) scripts/make_report.py --root . --out artifacts/report/index.html

explorer: ## Интерактивный разбор кейсов (streamlit, см. install-viz)
	.venv/bin/streamlit run scripts/explorer.py

clean: ## Кэши инструментов и сборки (artifacts/ и predictions/ не трогает)
	rm -rf .pytest_cache .ruff_cache .mypy_cache \
		src/*.egg-info src/rag_reliability/__pycache__ tests/__pycache__
