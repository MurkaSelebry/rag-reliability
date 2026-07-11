# Quick commands. Everything runs against .venv (create with `make install`).

PY := .venv/bin/python
DATA ?= data/dummy.jsonl
MODEL ?= mlx-community/Qwen2.5-1.5B-Instruct-4bit
ENCODER_MAX_LENGTH ?= 512
ENCODER_BATCH_SIZE ?= 4
ENCODER_EPOCHS ?= 3
ENCODER_LEARNING_RATE ?= 2e-5
ENCODER_POS_WEIGHT_MODE ?= none
# --- переменные пайплайна m3-m6 (scripts/m3m6/) ---
CONFIG ?= configs/config.yaml
SPLIT ?= val
MODE ?= zero_shot
VARIANT ?= markers
SEED ?= 0
M3_PRED ?= predictions/cloud/m3/zero_shot/$(SPLIT).jsonl
LIMIT ?=
LIMIT_FLAG := $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: help install install-mlx install-lettucedetect install-m6 install-cloud install-gepa test lint check dummy \
        install-encoder baseline-direct baseline-marker encoder-baseline train-direct \
        train-marker train-lettucedetect infer-direct infer-marker infer-lettucedetect \
        install-demo serve-demo benchmark-dummy eval-all clean \
        install-viz install-tracking install-data m3m6-smoke m3 gepa gepa-report \
        m6-samples m6-features m6-predict m6 baseline-surface baseline-curator-encoder \
        pseudo-corpus splits figs report explorer

help: ## List available targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-24s %s\n", $$1, $$2}'

install: ## Create venv and install core + dev deps (uv)
	uv venv --python 3.12
	uv pip install -e ".[dev]"

install-mlx: ## Add MLX backend + LoRA training deps (Apple Silicon)
	uv pip install -e ".[mlx]"

install-lettucedetect: ## Add LettuceDetect feature-classifier deps
	uv pip install -e ".[lettucedetect]"

install-m6: ## Add Method 6 SelfCheck/NLI feature deps
	uv pip install -e ".[m6]"

install-cloud: ## Add OpenAI-compatible cloud backend deps
	uv pip install -e ".[cloud]"

install-gepa: ## Add DSPy deps for GEPA prompt evolution (Method 3)
	uv pip install -e ".[gepa]"

install-encoder: ## Add supervised encoder baseline deps
	uv pip install -e ".[encoder]"

install-demo: ## Add local Gradio demo deps
	uv pip install -e ".[demo]"

install-viz: ## Add matplotlib/plotly/streamlit for m3-m6 figures & explorer
	uv pip install -e ".[viz]"

install-tracking: ## Add mlflow (local file-store) for m3-m6 runs
	uv pip install -e ".[tracking]"

install-data: ## Add datasets (pseudo-corpus) and dvc for m3-m6
	uv pip install -e ".[data]"

test: ## Run unit tests
	$(PY) -m pytest -q

lint: ## Ruff lint
	$(PY) -m ruff check .

check: test lint ## Tests + lint

dummy: ## Smoke-test pipeline without a model (keyword strategy, marker mode)
	$(PY) scripts/run_prompt_baseline.py --data $(DATA) \
		--output results/dummy_marker_predictions.jsonl \
		--mode marker --backend dummy --dummy-strategy keyword
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/dummy_marker_predictions.jsonl \
		--output results/dummy_marker_metrics.json

baseline-direct: ## Zero-shot MLX baseline, direct mode
	$(PY) scripts/run_prompt_baseline.py --data $(DATA) \
		--output results/qwen_direct_predictions.jsonl \
		--mode direct --backend mlx --model $(MODEL)
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/qwen_direct_predictions.jsonl \
		--output results/qwen_direct_metrics.json

baseline-marker: ## Zero-shot MLX baseline, marker mode
	$(PY) scripts/run_prompt_baseline.py --data $(DATA) \
		--output results/qwen_marker_predictions.jsonl \
		--mode marker --backend mlx --model $(MODEL)
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/qwen_marker_predictions.jsonl \
		--output results/qwen_marker_metrics.json

encoder-baseline: ## Supervised RuModernBERT reliability classifier
	$(PY) scripts/train_encoder_baseline.py --data $(DATA) \
		--output results/encoder_baseline_512_best_metrics.json \
		--output-dir results/encoder_checkpoints_512_best \
		--max-length $(ENCODER_MAX_LENGTH) --batch-size $(ENCODER_BATCH_SIZE) \
		--epochs $(ENCODER_EPOCHS) --learning-rate $(ENCODER_LEARNING_RATE) \
		--pos-weight-mode $(ENCODER_POS_WEIGHT_MODE)

train-direct: ## Prepare direct SFT splits and print the mlx_lm.lora command
	$(PY) scripts/train_direct_lora.py --data $(DATA)

train-marker: ## Prepare marker SFT splits and print the mlx_lm.lora command
	$(PY) scripts/train_marker_lora.py --data $(DATA)

train-lettucedetect: ## Train LettuceDetect feature classifier
	$(PY) scripts/train_lettucedetect.py --data $(DATA)

infer-direct: ## Inference with the trained direct adapter + evaluation
	$(PY) scripts/infer.py --data $(DATA) \
		--output results/direct_lora_predictions.jsonl \
		--mode direct --adapter-path results/adapters_direct
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/direct_lora_predictions.jsonl \
		--output results/direct_lora_metrics.json

infer-marker: ## Inference with the trained marker adapter + evaluation
	$(PY) scripts/infer.py --data $(DATA) \
		--output results/marker_lora_predictions.jsonl \
		--mode marker --adapter-path results/adapters_marker
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/marker_lora_predictions.jsonl \
		--output results/marker_lora_metrics.json

infer-lettucedetect: ## Inference with LettuceDetect classifier + evaluation
	$(PY) scripts/infer_lettucedetect.py --data $(DATA) \
		--model results/lettucedetect/classifier.joblib \
		--output results/lettucedetect/predictions.jsonl
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/lettucedetect/predictions.jsonl \
		--output results/lettucedetect/metrics.json

serve-demo: ## Local manual web UI
	$(PY) scripts/serve_demo.py

benchmark-dummy: ## Unified benchmark smoke test with dummy methods
	$(PY) scripts/run_benchmark.py --data $(DATA) \
		--methods dummy_direct,dummy_marker \
		--output-dir results/benchmark_dummy

eval-all: ## Print every metrics json in results/
	@for f in results/*_metrics.json; do echo "== $$f"; cat "$$f"; done

# --- Пайплайн ветки m3-m6 (пакет rag_reliability_m3m6, скрипты scripts/m3m6/) ---

m3m6-smoke: ## m3-m6: smoke-прогон Метода 3 на 10 кейсах (нужен LLM из CONFIG)
	$(PY) scripts/m3m6/run_m3.py --config $(CONFIG) --mode $(MODE) --split $(SPLIT) --limit 10

m3: ## m3-m6: инференс судьи Метода 3 (CONFIG/MODE/SPLIT/LIMIT)
	$(PY) scripts/m3m6/run_m3.py --config $(CONFIG) --mode $(MODE) --split $(SPLIT) $(LIMIT_FLAG)

gepa: ## m3-m6: GEPA-оптимизация промпта (VARIANT/SEED; дорого — стоп-правила docs/10)
	$(PY) scripts/m3m6/run_gepa.py --config $(CONFIG) --variant $(VARIANT) --seed $(SEED)

gepa-report: ## m3-m6: markdown-отчёт эволюции GEPA (VARIANT/SEED)
	$(PY) scripts/m3m6/gepa_report.py --variant $(VARIANT) --seed $(SEED)

m6-samples: ## m3-m6: Метод 6, этап 1 — сэмплы бота (поэлементный кэш)
	$(PY) scripts/m3m6/prepare_m6_samples.py --config $(CONFIG) --split $(SPLIT) $(LIMIT_FLAG)

m6-features: ## m3-m6: Метод 6, этап 2 — фичи selfcheck/entropy/cos
	$(PY) scripts/m3m6/prepare_m6_features.py --config $(CONFIG) --split $(SPLIT) $(LIMIT_FLAG)

m6-predict: ## m3-m6: Метод 6, этап 3 — калибровка на val -> predictions
	$(PY) scripts/m3m6/run_m6_selfcheck.py --config $(CONFIG) $(LIMIT_FLAG)

m6: m6-samples m6-features m6-predict ## m3-m6: Метод 6, полный конвейер на SPLIT

baseline-surface: ## m3-m6: бейзлайн surface(+e5)
	$(PY) scripts/m3m6/run_surface_baseline.py --config $(CONFIG) $(LIMIT_FLAG)

baseline-curator-encoder: ## m3-m6: supervised-энкодер корпуса кураторов
	$(PY) scripts/m3m6/train_encoder_baseline.py --config $(CONFIG)

pseudo-corpus: ## m3-m6: синтетический псевдо-корпус для cloud-отладки
	$(PY) scripts/m3m6/make_pseudo_corpus.py --config $(CONFIG) $(LIMIT_FLAG)

splits: ## m3-m6: групповые сплиты из корпуса (данные платформы не трогает)
	$(PY) scripts/m3m6/make_splits.py --config $(CONFIG)

figs: ## m3-m6: фигуры отладки по M3_PRED (matplotlib, см. install-viz)
	$(PY) scripts/m3m6/make_figs.py --config $(CONFIG) --split $(SPLIT) --m3-pred $(M3_PRED)

report: ## m3-m6: единый offline HTML-отчёт по всем прогонам
	$(PY) scripts/m3m6/make_report.py --root . --out artifacts/report/index.html

explorer: ## m3-m6: интерактивный разбор кейсов (streamlit, см. install-viz)
	.venv/bin/streamlit run scripts/m3m6/explorer.py

clean: ## Remove tool caches and build artifacts (keeps results/, artifacts/, predictions/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache \
		src/*.egg-info src/rag_reliability/__pycache__ \
		src/rag_reliability_m3m6/__pycache__ tests/__pycache__
