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

## Reproduction

```bash
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
