# Architecture

The pipeline is a straight line; every experiment (dummy, zero-shot, LoRA)
walks the same five steps, so results are always comparable:

```
data (jsonl) ──► prompt formatting ──► inference ──► parsing ──► evaluation
  dataset.py       prompts.py         scripts/      parsing.py   metrics.py
                   formatting.py
```

## Modules (`src/rag_reliability/`)

| Module | Responsibility |
|---|---|
| `schema.py` | Pydantic models: `RagSample` (input + gold labels), `Prediction` (parsed model output), `EvaluationResult` (metrics). `reliable = faithfulness AND relevance` is a derived property on both sides. |
| `prompts.py` | English judge prompts for both modes; `ALLOWED_MARKERS` is the single source of truth for the marker vocabulary. |
| `formatting.py` | Builds SFT targets and `{"prompt", "completion"}` training records; `resolve_marker()` implements the `none`/`unknown` fallback used by both training and evaluation. |
| `parsing.py` | Raw LLM text → `Prediction`. Three-stage fallback: balanced-JSON extraction → regex → conservative `(0, 0, invalid_output=True)`. Never raises on model output. |
| `metrics.py` | Macro-F1 for reliable/faithfulness/relevance; marker F1 + confusion (marker mode only). Joins predictions to samples by `id`, raises on missing ids. |
| `dataset.py` | JSONL IO, stratified 80/10/10 split by `reliable` (seed=42), training-file writer. |
| `dummy_model.py` | Deterministic pseudo-LLMs so the whole pipeline runs without a model (see [experiments.md](experiments.md)). |

## Scripts (`scripts/`)

| Script | Role |
|---|---|
| `run_prompt_baseline.py` | Zero-shot judge over a dataset; `--backend dummy` or `--backend mlx`. |
| `infer.py` | Same as the mlx baseline but loads a trained LoRA adapter (`--adapter-path`). Output format is identical, so `evaluate.py` works for both. |
| `evaluate.py` | Predictions + gold → metrics json. |
| `train_direct_lora.py` / `train_marker_lora.py` | Prepare SFT splits and print the exact `mlx_lm.lora` command (they do not train themselves). |
| `prepare_data.py` | Raw dataset → `RagSample` jsonl (see [data.md](data.md)). |

## Two judge methods

- **Method 1 — direct** (`mode=direct`): model outputs
  `{"faithfulness": 0|1, "relevance": 0|1}`.
- **Method 2 — marker** (`mode=marker`): model first names the error type,
  then the labels: `{"marker": "...", "faithfulness": 0|1, "relevance": 0|1}`.
  Hypothesis: forcing an error-type decision improves label quality and gives
  diagnosable failure categories for free.

## Design decisions

- **Conservative parsing.** An unparseable output counts as
  `faithfulness=0, relevance=0` and increments `invalid_output_rate`. A judge
  that produces garbage must not look reliable.
- **Chat template symmetry.** Both training (`mlx_lm` `CompletionsDataset`
  applies the tokenizer chat template to prompt/completion pairs) and
  inference (`apply_chat_template` in the scripts) wrap prompts identically —
  verified, not assumed.
- **`--mask-prompt` in training.** The judge prompt is ~50× longer than the
  JSON completion; without prompt masking the loss is dominated by prompt
  tokens.
- **4-bit base model.** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~840 MB)
  instead of bf16 (~3 GB): faster download, ~8 s inference over 36 samples,
  QLoRA-style training works out of the box.
- **`transformers<5` pin.** transformers 5.x breaks `mlx-lm` at import time
  (`TOKENIZER_MAPPING.register` receives a `str`); pinned in the `mlx` extra.
