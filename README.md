# rag-reliability-judge

Part of the team project **"Assessing the Reliability of Responses in RAG
Systems"** @SMILES-2026. Given `QUESTION`, `CONTEXT`, `ANSWER`, the methods
predict whether the answer is reliable (`reliable = faithfulness AND relevance`).

Implemented method families:

- **Method 1 — direct** (`mode=direct`): the model outputs
  `{"faithfulness": 0|1, "relevance": 0|1}`.
- **Method 2 — marker** (`mode=marker`): the model additionally names the
  error type first: `{"marker": "...", "faithfulness": 0|1, "relevance": 0|1}`.
- **LettuceDetect features**: LettuceDetect token-level scores are
  aggregated into three features, then a logistic regression predicts
  faithfulness and relevance.
- **Method 3/6 imports from `m3-m6`**: Method 3 prompt judge and Method 6
  SelfCheck-style feature scoring are integrated through the shared
  predictions/metrics contract.
- **Full m3-m6 branch pipeline**: the original Method 3/6 implementation
  (logprob judge, GEPA evolution, SelfCheck sampling/features/calibration,
  baselines, analysis and reporting) lives as the separate
  `rag_reliability_m3m6` package with CLI wrappers under `scripts/m3m6/`
  (see the [m3-m6 pipeline](#m3-m6-branch-pipeline) section).

## Documentation map

| Doc | What's inside |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline, module map, design decisions (conservative parsing, chat-template symmetry, 4-bit model, dependency pins) |
| [docs/data.md](docs/data.md) | Sample schema, marker vocabulary, dummy dataset, plugging in the real dataset |
| [docs/training.md](docs/training.md) | LoRA workflow, configs, why `--mask-prompt`, scaling up |
| [docs/lettucedetect.md](docs/lettucedetect.md) | LettuceDetect feature extraction + logistic regression |
| [docs/m3_m6.md](docs/m3_m6.md) | Selective Method 3/6 port from the `m3-m6` branch |
| [docs/experiments.md](docs/experiments.md) | All results so far, how to reproduce, environment gotchas |
| [docs/00](docs/00_project_overview.md)…[docs/13](docs/13_openrouter_stage.md) | m3-m6 branch docs (Russian): problem statement, data format, platform contract, method specs, stage reports, [viz](docs/viz/) |

## Quickstart

Requires Python ≥ 3.11. Target hardware: Apple Silicon (MLX); everything
except the `mlx` backend runs anywhere.

```bash
make install                    # uv venv + core/dev deps
make install-mlx                # optional, Apple Silicon: mlx backend + LoRA
make install-lettucedetect      # optional: LettuceDetect feature method
make install-encoder            # optional: RuModernBERT supervised baseline
make install-m6                 # optional: Method 6 NLI/embedding features
make install-cloud              # optional: OpenAI-compatible Method 3 backend
make install-gepa               # optional: DSPy for GEPA prompt evolution
make install-demo               # optional: local Gradio demo UI
make install-viz                # optional: m3-m6 figures / HTML report / explorer
make check                      # tests + lint
make help                       # all shortcuts: dummy, baselines, LoRA, m3-m6, eval
```

(No make: `uv venv --python 3.12 && uv pip install -e ".[dev]"`, then `pytest`.)

Smoke-test the pipeline without any model:

```bash
python scripts/run_prompt_baseline.py \
  --data data/dummy.jsonl \
  --output results/dummy_predictions.jsonl \
  --mode marker --backend dummy --dummy-strategy keyword

python scripts/evaluate.py \
  --data data/dummy.jsonl \
  --predictions results/dummy_predictions.jsonl \
  --output results/dummy_metrics.json
```

Real zero-shot baseline (downloads ~840 MB once):

```bash
python scripts/run_prompt_baseline.py \
  --data data/dummy.jsonl \
  --output results/qwen_direct_predictions.jsonl \
  --mode direct --backend mlx
```

LoRA fine-tuning: see [docs/training.md](docs/training.md). LettuceDetect
feature-classifier training: see [docs/lettucedetect.md](docs/lettucedetect.md).

Unified benchmark interface:

```bash
python scripts/run_benchmark.py \
  --data data/dummy.jsonl \
  --methods dummy_direct,dummy_marker \
  --output-dir results/benchmark_dummy
```

Supported methods: `dummy_direct`, `dummy_marker`, `prompt_direct`,
`prompt_marker`, `lora_direct`, `lora_marker`, `lettucedetect`, `encoder`,
`m3_zero_shot`, `m3_few_shot`, `m3_gepa`, `m3_openai`, `m6_selfcheck`.
Each method writes `predictions.jsonl`; the shared evaluator then writes
`metrics.json`.

`m3_zero_shot`, `m3_few_shot`, and `m3_gepa` are Method 3 judge prompt modes
ported from the `m3-m6` branch.
`m3_openai` runs the Method 3 zero-shot prompt through an OpenAI-compatible
chat completions endpoint with a local file cache.
`m6_selfcheck` consumes a precomputed Method 6 feature JSONL via
`--m6-features`; sample generation, NLI scoring, and calibration remain explicit
preparation steps instead of hidden benchmark side effects.

Manual local demo UI:

```bash
make install-demo
make serve-demo
```

The demo accepts `question`, `context`, `answer`, optional gold labels, and a
method selector. Methods that need missing artifacts or dependencies return a
clear unavailable status instead of crashing; the encoder method can run from a
configured local checkpoint when one is available.
It also supports dataset presets, side-by-side method comparison, compact
correctness display, raw-output inspection, run history, method configuration,
and batch benchmark command generation from either a path or uploaded JSONL.

Supervised encoder baseline from the organizer notebook:

```bash
python scripts/prepare_data.py \
  --input from_organizators/data/data.zip \
  --output data/organizers.jsonl

python scripts/train_encoder_baseline.py \
  --data data/organizers.jsonl \
  --output results/encoder_baseline_512_best_metrics.json \
  --output-dir results/encoder_checkpoints_512_best \
  --max-length 512 --batch-size 4 \
  --epochs 3 --learning-rate 2e-5 --pos-weight-mode none
```

## m3-m6 branch pipeline

The `rag_reliability_m3m6` package is the original branch implementation,
kept intact next to the benchmark harness. It works on config-driven splits
(`configs/config*.yaml`, local vLLM by default, OpenRouter in the cloud
profile) and writes per-case probabilities to
`predictions/{profile}/{method}/{variant}/{split}.jsonl` plus a `run.yaml`
(config, git hash, seed) — the frozen platform contract
([docs/02](docs/02_platform_contract.md)). All scripts are idempotent
(element-wise caches in `artifacts/`) and accept `--limit N` for smoke runs.

```bash
# Method 3 — logprob judge (zero_shot / few_shot / gepa)
python scripts/m3m6/run_m3.py --config configs/config.cloud.yaml --mode zero_shot --split val
python scripts/m3m6/run_gepa.py --config configs/config.cloud.yaml --variant markers --seed 0

# Method 6 — sample -> features -> calibrated predictions
python scripts/m3m6/prepare_m6_samples.py  --config configs/config.cloud.yaml --split val
python scripts/m3m6/prepare_m6_features.py --config configs/config.cloud.yaml --split val
python scripts/m3m6/run_m6_selfcheck.py    --config configs/config.cloud.yaml

# evaluation, signals, figures, offline HTML report, streamlit explorer
python scripts/m3m6/evaluate.py --cases data/processed/pseudo_dev_val.jsonl \
    --preds predictions/cloud/m3/zero_shot/val.jsonl --fit
python scripts/m3m6/make_report.py --root . --out artifacts/report/index.html
make explorer
```

Key rules (details in `CLAUDE.md` and docs/00–13): bank data goes to the
local vLLM only (a guard blocks non-synthetic cases in the cloud profile),
dev-test is never used for decisions, judge probabilities come from PASS/FAIL
token logprobs with the fallback chain logprobs → regex → 0.5/0.5, and every
run records `run.yaml` for determinism. Cloud-profile numbers are debug-only.

## Metrics

Reported by `scripts/evaluate.py`:

- **`reliable_f1_macro`** — primary metric
- `faithfulness_f1_macro`, `relevance_f1_macro`
- `invalid_output_rate` — outputs unparseable even with fallbacks; counted
  conservatively as `faithfulness=0, relevance=0`
- marker mode only: `marker_f1_macro`, `marker_per_class_f1`,
  `marker_confusion` (gold → predicted counts)

The m3-m6 pipeline reports `f1_macro_reliable` / `f1_macro_faith` /
`f1_macro_rel` via `rag_reliability_m3m6.common.eval_local`
(`scripts/m3m6/evaluate.py`), with thresholds fitted on val only.

Current best on dummy data: zero-shot direct `reliable_f1 = 0.86`
(full table + caveats in [docs/experiments.md](docs/experiments.md)).

Current best on organizer data: RuModernBERT encoder at `max_length=512`,
3 epochs, no positive-class weighting, and a validation-selected threshold:
reliability macro-F1 `0.5879`.

m3-m6 branch results (pseudo-corpus, cloud debug): Method 3 few_shot
`f1_reliable = 0.824` (val); Method 6 contradiction is a working
faithfulness signal (Δ+0.31) while semantic entropy is not
([docs/10](docs/10_gepa_stage.md) §4); baselines on the curator corpus:
surface 0.615 / surface+e5 0.537.

## Status

- ✅ Pipeline (data → prompt → inference → parsing → metrics) built and
  verified end-to-end: dummy backends, zero-shot MLX baseline, LoRA training
  + adapter inference — all with 0% invalid outputs.
- ✅ Organizer CSV/ZIP dataset format is supported by `prepare_data.py`.
- ✅ Organizer encoder baseline is reproducible at `max_length=512` with a
  held-out test set, no class weighting, 3 epochs, and validation-selected
  threshold.
- ✅ Method 3/6 code from `m3-m6` is being integrated selectively into the
  shared prediction/evaluation contract without replacing the current package
  layout.
- ✅ The full `m3-m6` branch pipeline is merged in as the standalone
  `rag_reliability_m3m6` package (`scripts/m3m6/`), with its prediction
  artifacts under `predictions/` and stage reports under docs/00–13.
- ⏳ Next: real Method 1 vs Method 2 fine-tuning comparison and targeted
  hyperparameter tuning around the 512-token encoder setup; m3-m6 local
  stage (curator corpus + vLLM, H1/H5 runs — [docs/05](docs/05_tasks.md)).

## Project layout

```
data/dummy.jsonl             36 synthetic Russian banking RAG examples
configs/                     LoRA training configs (direct, marker);
                             m3-m6: config.yaml (local), config.cloud.yaml,
                             config.alfa_cloud.yaml, markers.yaml, m3m6/few_shot.yaml
src/rag_reliability/         benchmark harness: schema, prompts, formatting, parsing,
                             metrics, dataset IO, dummy predictors, mlx backend, methods
src/rag_reliability_m3m6/    m3-m6 branch package: common (config/schemas/guard/
                             llm clients/eval_local), data, baselines, analysis,
                             methods/m3 (logprob judge + GEPA), methods/m6 (SelfCheck)
scripts/                     CLI entry points — run from repo root
scripts/m3m6/                m3-m6 pipeline CLI wrappers
tests/                       unit tests (both suites, no MLX/GPU required)
docs/                        architecture / data / training / experiments;
                             00–13 + viz/ — m3-m6 branch docs (Russian)
predictions/                 m3-m6 prediction artifacts (platform contract, committed)
results/                     predictions, metrics, adapters (gitignored)
artifacts/                   m3-m6 caches and reports (gitignored, dvc.yaml)
reference_src/               early-iteration reference (not wired in)
```
