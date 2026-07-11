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
- **Independent rule-based evaluator**: heuristic thresholds over
  faithfulness/relevance signals, no model required.

All methods are registered in one place
(`src/rag_reliability/methods/registry.py`) and driven through a single CLI,
`rag-judge`.

## At a glance

`rag-judge` is the single entry point for running, benchmarking, and scoring
every method against the shared `predictions.jsonl` → `metrics.json`
contract. Fifteen methods are registered, from a zero-config dummy baseline
to LoRA-tuned and Method 3/6 judges; `rag-judge list-methods` prints exactly
what's available and what each one requires. Training and data-prep
pipelines stay as standalone `scripts/*.py` invocations (see
[Advanced / pipelines](#advanced--pipelines)) since they produce artifacts
(adapters, checkpoints, prompts) that methods later consume.

## Documentation map

| Doc | What's inside |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline, module map, design decisions (conservative parsing, chat-template symmetry, 4-bit model, dependency pins) |
| [docs/data.md](docs/data.md) | Sample schema, marker vocabulary, dummy dataset, plugging in the real dataset |
| [docs/training.md](docs/training.md) | LoRA workflow, configs, why `--mask-prompt`, scaling up |
| [docs/lettucedetect.md](docs/lettucedetect.md) | LettuceDetect feature extraction + logistic regression |
| [docs/m3_m6.md](docs/m3_m6.md) | Selective Method 3/6 port from the `m3-m6` branch |
| [docs/experiments.md](docs/experiments.md) | All results so far, how to reproduce, environment gotchas |

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
make install-demo               # optional: local Gradio demo UI
make check                      # tests + lint
make help                       # all shortcuts: dummy, baselines, LoRA, eval
```

(No make: `uv venv --python 3.12 && uv pip install -e ".[dev]"`, then `pytest`.)

This installs the `rag-judge` console script. List every registered method,
its family, and what it requires:

```bash
rag-judge list-methods
```

Smoke-test the pipeline without any model (dummy backend, no downloads):

```bash
rag-judge run --method dummy_marker --data data/dummy.jsonl --output-dir results/run
```

Run several methods through the shared predictions → metrics contract:

```bash
rag-judge benchmark --methods dummy_direct,dummy_marker --data data/dummy.jsonl --output-dir results/benchmark_dummy
```

Use `--methods all` to run every registered method, or a comma list to pick
specific ones. Each run writes `predictions.jsonl` and `metrics.json` per
method plus a `summary.json` in `--output-dir`.

Real zero-shot baseline (downloads ~840 MB once):

```bash
rag-judge run --method prompt_direct --data data/dummy.jsonl --output-dir results/run
```

Score an existing predictions file against gold labels directly:

```bash
rag-judge eval --data data/dummy.jsonl --predictions results/run/predictions.jsonl --output results/run/metrics.json
```

Launch the local Gradio demo UI:

```bash
make install-demo
rag-judge serve
```

The demo accepts `question`, `context`, `answer`, optional gold labels, and a
method selector sourced from the same registry as the CLI. Methods that need
missing artifacts or dependencies return a clear unavailable status instead
of crashing; the encoder method can run from a configured local checkpoint
when one is available. It also supports dataset presets, side-by-side method
comparison, compact correctness display, raw-output inspection, run history,
method configuration, and batch benchmark command generation from either a
path or uploaded JSONL.

LoRA fine-tuning: see [docs/training.md](docs/training.md). LettuceDetect
feature-classifier training: see [docs/lettucedetect.md](docs/lettucedetect.md).

## Methods

| Method | Family | What it needs | In demo? |
|---|---|---|---|
| `dummy_direct` | dummy | — | yes |
| `dummy_marker` | dummy | — | yes |
| `prompt_direct` | prompt | MLX model | yes |
| `prompt_marker` | prompt | MLX model | yes |
| `lora_direct` | lora | `results/adapters_direct` | yes |
| `lora_marker` | lora | `results/adapters_marker` | yes |
| `lettucedetect` | lettucedetect | `results/lettucedetect/classifier.joblib` | yes |
| `encoder` | encoder | `results/encoder_checkpoints_512_best` | yes |
| `m3_zero_shot` | m3 | MLX model | yes |
| `m3_few_shot` | m3 | `configs/few_shot.yaml` | yes |
| `m3_gepa` | m3 | evolved prompt file | batch-only |
| `m3_openai` | m3 | OpenAI-compatible endpoint | batch-only |
| `m3_openai_judge` | m3 | OpenAI-compatible endpoint | batch-only |
| `m6_selfcheck` | m6 | `results/m6/features.jsonl` | batch-only |
| `independent` | independent | — | yes |

This table mirrors `registry.METHODS` in
[`src/rag_reliability/methods/registry.py`](src/rag_reliability/methods/registry.py);
run `rag-judge list-methods` for the same information straight from the code.

`m3_zero_shot`, `m3_few_shot`, and `m3_gepa` are Method 3 judge prompt modes
ported from the `m3-m6` branch. `m3_openai` and `m3_openai_judge` run the
Method 3 prompt through an OpenAI-compatible chat completions endpoint (the
latter as a logprob-based judge) with a local file cache. `m6_selfcheck`
consumes a precomputed Method 6 feature JSONL via `--m6-features`; sample
generation, NLI scoring, and calibration remain explicit preparation steps
instead of hidden benchmark side effects.

## Advanced / pipelines

Training and data-prep steps produce artifacts (adapters, checkpoints,
converted datasets, evolved prompts) that the methods above consume — they
are not part of the `run`/`benchmark`/`eval` contract, so they stay as raw
script invocations.

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

LoRA training (`train_direct_lora.py` / `mlx_lm.lora`): see
[docs/training.md](docs/training.md).

GEPA prompt evolution (`run_gepa.py`, produces the prompt file consumed by
`m3_gepa`): see [docs/m3_m6.md](docs/m3_m6.md).

## Metrics

Reported by `rag-judge eval` (`scripts/evaluate.py`):

- **`reliable_f1_macro`** — primary metric
- `faithfulness_f1_macro`, `relevance_f1_macro`
- `invalid_output_rate` — outputs unparseable even with fallbacks; counted
  conservatively as `faithfulness=0, relevance=0`
- marker mode only: `marker_f1_macro`, `marker_per_class_f1`,
  `marker_confusion` (gold → predicted counts)

Current best on dummy data: zero-shot direct `reliable_f1 = 0.86`
(full table + caveats in [docs/experiments.md](docs/experiments.md)).

Current best on organizer data: RuModernBERT encoder at `max_length=512`,
3 epochs, no positive-class weighting, and a validation-selected threshold:
reliability macro-F1 `0.5879`.

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
- ✅ All 15 methods are registered in `methods/registry.py` and reachable
  through the `rag-judge` CLI (`run`, `benchmark`, `eval`, `serve`,
  `list-methods`); 11 of them are also wired into the Gradio demo (see
  the Methods table above).
- ⏳ Next: real Method 1 vs Method 2 fine-tuning comparison and targeted
  hyperparameter tuning around the 512-token encoder setup.

## Project layout

```
data/dummy.jsonl        36 synthetic Russian banking RAG examples
configs/                LoRA training configs (direct, marker)
src/rag_reliability/    schema, prompts, formatting, parsing, metrics,
                        dataset IO, dummy predictors, mlx backend, methods,
                        method registry, rag-judge CLI
scripts/                CLI entry points — run from repo root
tests/                  unit tests (65, no MLX required)
docs/                   architecture / data / training / experiments
results/                predictions, metrics, adapters (gitignored)
```
