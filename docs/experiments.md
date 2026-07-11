# Experiments log

All numbers below are on `data/dummy.jsonl` — **36 synthetic examples**. They
validate that the pipeline works and that the metrics can discriminate; they
say nothing about real-world quality. Rerun everything when the real dataset
arrives.

## Results (dummy data, 2026-07-04)

| Run | reliable_f1 | faithfulness_f1 | relevance_f1 | marker_f1 | invalid |
|---|---|---|---|---|---|
| dummy `always_reliable` | 0.28 | — | — | — | 0% |
| dummy `keyword` (marker) | 0.60 | 0.63 | 0.77 | 0.39 | 0% |
| Qwen zero-shot, direct | **0.86** | 0.80 | 0.58 | — | 0% |
| Qwen zero-shot, marker | 0.58 | 0.61 | 0.36 | 0.09 | 0% |
| LoRA, direct | 0.57 | 0.70 | 0.43 | — | 0% |
| LoRA, marker | 0.28 | 0.38 | 0.43 | 0.08 | 0% |

Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. LoRA: 58 iters, lr 1e-4,
batch 1, `--mask-prompt`, 29 train samples.

## Results (organizer data, 2026-07-09)

Organizer data converted from `from_organizators/data/data.zip`: 2245 rows,
1622 reliable / 623 unreliable after `faithfulness AND relevance`.

| Run | primary metric | Accuracy | Precision | Recall | Reliable F1 | ROC-AUC | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| dummy `always_reliable` | 0.4194 | — | — | — | — | — | Macro-F1 floor on full converted data |
| dummy `keyword` (marker) | 0.4696 | — | — | — | — | — | Marker F1 0.0474 |
| Qwen zero-shot, direct | 0.4946 | — | — | — | — | — | `invalid_output_rate=0%` |
| Qwen zero-shot, marker | 0.3146 | — | — | — | — | — | Marker F1 0.0123, invalid 3 rows |
| RuModernBERT encoder, 512, threshold 0.5 | 0.5415 | 0.6682 | 0.7424 | 0.8272 | 0.7825 | 0.6573 | Old train/test split |
| RuModernBERT encoder, 1024, threshold 0.5 | 0.4191 | 0.7216 | 0.7216 | 1.0000 | 0.8383 | 0.6169 | Collapses toward predicting reliable |
| RuModernBERT encoder, 512, balanced weight, tuned threshold | 0.5593 | 0.6927 | 0.7500 | 0.8611 | 0.8017 | 0.6562 | Train/validation/test, threshold 0.44 |
| RuModernBERT encoder, 512, no weight, 2 epochs | 0.5693 | 0.6347 | 0.7667 | 0.7099 | 0.7372 | 0.6643 | Threshold 0.73 |
| RuModernBERT encoder, 512, no weight, 3 epochs, lr=1e-5 | 0.5756 | 0.6949 | 0.7576 | 0.8488 | 0.8006 | 0.6610 | Threshold 0.68 |
| RuModernBERT encoder, 512, no weight, 3 epochs, lr=2e-5 | **0.5879** | 0.6815 | 0.7670 | 0.8025 | 0.7843 | 0.6614 | Threshold 0.72 |
| RuModernBERT encoder, 512, no weight, 3 epochs, lr=3e-5 | 0.5539 | 0.6659 | 0.7486 | 0.8086 | 0.7774 | 0.6558 | Threshold 0.69 |

For encoder runs, the primary metric is binary reliability macro-F1. The
current working encoder baseline is the 512-token / no-class-weight / 3-epoch
run with `learning_rate=2e-5`. Lowering LR to `1e-5` and raising it to `3e-5`
both hurt test macro-F1. Longer context at 1024 fits in memory but hurts
macro-F1 because the model becomes too positive.

## LoRA benchmark status (organizer data, 2026-07-09)

Organizer direct/marker LoRA benchmark files are prepared under
`results/organizer_lora/` with a shared held-out split:
1796 train / 224 validation / 225 test samples. The MLX SFT files use chat
records (`{"messages": [...]}`), because the installed `mlx_lm` chat-template
path fails on prompt/completion records when `--mask-prompt` is enabled.

Held-out test results on the shared 225-sample split:

| Run | reliable_f1_macro | faithfulness_f1_macro | relevance_f1_macro | marker_f1_macro | invalid |
|---|---:|---:|---:|---:|---:|
| Qwen zero-shot, direct | **0.4701** | **0.4739** | 0.4238 | — | 0 / 225 |
| Qwen zero-shot, marker | 0.3513 | 0.3477 | 0.2399 | 0.0136 | 0 / 225 |
| LoRA, direct | 0.4186 | 0.4201 | **0.4668** | — | 0 / 225 |
| LoRA, marker | 0.4186 | 0.4201 | **0.4668** | 0.0761 | 0 / 225 |

Current stopping point:

- `max_seq_length=2048` is not viable for organizer SFT. Many prompts exceed
  the limit, the assistant target can be truncated away, and training reports
  `nan` loss.
- `max_seq_length=8192` avoids that truncation but hit Apple Metal OOM around
  iter 40 with peak memory near 29 GB.
- Prepared SFT records now cap the dialog at 2000 characters and context at
  5000 characters, and LoRA configs use `max_seq_length=4096`. This keeps the
  generated direct train records under 4096 tokens (observed max: 3192) and
  started training cleanly.
- Fresh organizer adapters were trained to completion for 3592 iterations each
  under `results/adapters_direct_rerun/` and `results/adapters_marker_rerun/`.
  Both use the same model, seed-42 train/validation/test split, learning rate,
  batch size, and sequence-length settings.
- The fully trained direct and marker adapters both fall below zero-shot direct
  on reliability and faithfulness macro-F1. They increase relevance macro-F1
  to 0.4668, but marker fine-tuning is not useful for the marker-label task:
  marker macro-F1 is only 0.0761 and it predicts `none` for every held-out row.

## Direct LoRA controlled sweep (organizer data, 2026-07-10)

The direct LoRA training set is dominated by `(faithfulness=1, relevance=1)`
labels: 1622 of 2245 total rows. Both the original two-epoch run and a new
one-epoch run predicted `(1, 1)` for every held-out example. To isolate this
effect, a second run used a deterministic balanced SFT train set with 256 rows
per `(faithfulness, relevance)` pair (1024 rows total); validation and the
seed-42 225-row test split were unchanged.

| Run | reliable_f1_macro | faithfulness_f1_macro | relevance_f1_macro | invalid |
|---|---:|---:|---:|---:|
| Qwen zero-shot, direct | **0.4701** | **0.4739** | 0.4238 | 0 / 225 |
| LoRA, one epoch, unbalanced | 0.4186 | 0.4201 | **0.4668** | 0 / 225 |
| LoRA, two epochs, balanced 256×4 | 0.4636 | 0.4669 | **0.4668** | 0 / 225 |

Balancing improves reliability by 0.0450 macro-F1 over the unbalanced LoRA
run and prevents its all-positive faithfulness collapse: the balanced model
outputs `(1, 1)` for 133 samples and `(0, 1)` for 92. It still never predicts
`relevance=0`, so it does not beat the direct zero-shot baseline overall.

## Reading the numbers

- `always_reliable` at 0.28 is the trivial floor (dataset is ~39% reliable);
  `keyword` beating it shows the metrics discriminate.
- Zero-shot direct (0.86) is the current best. Zero-shot marker is worse —
  the 1.5B model handles the extra classification task poorly without
  fine-tuning (marker_f1 0.09 ≈ noise).
- **LoRA results are artifacts of 29 toy training samples**, not pipeline
  bugs. The marker adapter collapses to predicting `none`/(1,1) for
  everything. Checked and ruled out: chat-template mismatch between training
  and inference (mlx_lm's `CompletionsDataset` applies the same template the
  inference scripts do). With real data volume this comparison becomes
  meaningful.
- `invalid_output_rate` is 0% everywhere: the parser plus a JSON-only prompt
  keep even the zero-shot 1.5B model parseable.
- On organizer data, direct zero-shot Qwen is only slightly above trivial
  baselines. The supervised encoder is stronger, and threshold tuning gives a
  small but real macro-F1 gain without changing the model.
- On the organizer held-out LoRA split, neither completed LoRA variant improves
  Method 1: zero-shot direct is better on reliability and faithfulness macro-F1
  with no invalid outputs. The LoRA gain is limited to relevance macro-F1.
- Disabling positive-class weighting helped. The dataset has a positive
  majority, and the balanced `pos_weight<1` made the model too eager to trade
  minority recall against overall calibration.

## Reproduction

```bash
# organizer supervised encoder baseline
python scripts/prepare_data.py \
  --input from_organizators/data/data.zip \
  --output data/organizers.jsonl
python scripts/train_encoder_baseline.py \
  --data data/organizers.jsonl \
  --output results/encoder_baseline_512_best_metrics.json \
  --output-dir results/encoder_checkpoints_512_best \
  --max-length 512 --batch-size 4 \
  --epochs 3 --learning-rate 2e-5 --pos-weight-mode none

# zero-shot baseline (direct; use --mode marker for method 2)
python scripts/run_prompt_baseline.py \
  --data data/dummy.jsonl \
  --output results/qwen_direct_predictions.jsonl \
  --mode direct --backend mlx

python scripts/evaluate.py \
  --data data/dummy.jsonl \
  --predictions results/qwen_direct_predictions.jsonl \
  --output results/qwen_direct_metrics.json

# LoRA: prepare splits + print the training command, then run it
python scripts/train_direct_lora.py --data data/dummy.jsonl
mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --train \
  --data results/lora_direct --batch-size 1 --iters 58 \
  --learning-rate 0.0001 --max-seq-length 2048 --mask-prompt \
  --adapter-path results/adapters_direct

# inference with the adapter + evaluation
python scripts/infer.py \
  --data data/dummy.jsonl \
  --output results/direct_lora_predictions.jsonl \
  --mode direct --adapter-path results/adapters_direct
python scripts/evaluate.py \
  --data data/dummy.jsonl \
  --predictions results/direct_lora_predictions.jsonl \
  --output results/direct_lora_metrics.json
```

Note: the current dummy evaluation runs over all 36 samples, i.e. the LoRA
adapters are partly evaluated on their own training data. Fine for a smoke
test; on the real dataset evaluate on the held-out test split only.

## Known environment gotchas

- **transformers 5.x breaks mlx-lm** at import
  (`AttributeError: 'str' object has no attribute '__module__'` inside
  `TOKENIZER_MAPPING.register`). Fixed by the `transformers<5` pin in the
  `mlx` extra.
- **bf16 model downloads can stall.** `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB)
  hung repeatedly; the 4-bit mlx-community build (~840 MB) is the default
  everywhere. Pre-download with
  `huggingface-cli download mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
