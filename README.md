# rag-reliability-judge

Part of the team project **"Assessing the Reliability of Responses in RAG
Systems"** @SMILES-2026. A local instruct LLM acts as a judge: given `QUESTION`, `CONTEXT`,
`ANSWER` it predicts whether the answer is reliable
(`reliable = faithfulness AND relevance`).

Two fine-tuned methods are compared:

- **Method 1 — direct** (`mode=direct`): the model outputs
  `{"faithfulness": 0|1, "relevance": 0|1}`.
- **Method 2 — marker** (`mode=marker`): the model additionally names the
  error type first: `{"marker": "...", "faithfulness": 0|1, "relevance": 0|1}`.

## Documentation map

| Doc | What's inside |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline, module map, design decisions (conservative parsing, chat-template symmetry, 4-bit model, dependency pins) |
| [docs/data.md](docs/data.md) | Sample schema, marker vocabulary, dummy dataset, plugging in the real dataset |
| [docs/training.md](docs/training.md) | LoRA workflow, configs, why `--mask-prompt`, scaling up |
| [docs/experiments.md](docs/experiments.md) | All results so far, how to reproduce, environment gotchas |

## Quickstart

Requires Python ≥ 3.11. Target hardware: Apple Silicon (MLX); everything
except the `mlx` backend runs anywhere.

```bash
make install                    # uv venv + core/dev deps
make install-mlx                # optional, Apple Silicon: mlx backend + LoRA
make install-encoder            # optional: RuModernBERT supervised baseline
make check                      # tests (47, no MLX required) + lint
make help                       # all shortcuts: dummy, baselines, LoRA, eval
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

LoRA fine-tuning: see [docs/training.md](docs/training.md).

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

## Metrics

Reported by `scripts/evaluate.py`:

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
- ⏳ Next: real Method 1 vs Method 2 fine-tuning comparison and targeted
  hyperparameter tuning around the 512-token encoder setup.

## Project layout

```
data/dummy.jsonl        36 synthetic Russian banking RAG examples
configs/                LoRA training configs (direct, marker)
src/rag_reliability/    schema, prompts, formatting, parsing, metrics,
                        dataset IO, dummy predictors, mlx backend
scripts/                CLI entry points — run from repo root
tests/                  unit tests (47, no MLX required)
docs/                   architecture / data / training / experiments
results/                predictions, metrics, adapters (gitignored)
```
