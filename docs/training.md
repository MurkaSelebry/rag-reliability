# Training

## LoRA

LoRA training itself is done by `mlx_lm.lora` (Apple Silicon). The project scripts
only prepare data and print the exact command — this keeps the repo free of a
training-loop reimplementation.

### Workflow

```bash
# 1. prepare splits + get the command (direct or marker)
python scripts/train_direct_lora.py --data data/dummy.jsonl
python scripts/train_marker_lora.py --data data/dummy.jsonl

# 2. run the printed mlx_lm.lora command, e.g.:
mlx_lm.lora \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --train \
    --data results/lora_direct \
    --batch-size 1 \
    --grad-accumulation-steps 8 \
    --iters 58 \
    --learning-rate 0.0001 \
    --max-seq-length 2048 \
    --mask-prompt \
    --adapter-path results/adapters_direct

# 3. inference with the adapter, then evaluate
python scripts/infer.py \
  --data data/dummy.jsonl \
  --output results/direct_lora_predictions.jsonl \
  --mode direct --adapter-path results/adapters_direct
python scripts/evaluate.py \
  --data data/dummy.jsonl \
  --predictions results/direct_lora_predictions.jsonl \
  --output results/direct_lora_metrics.json
```

### Configs (`configs/*.yaml`)

`direct_lora.yaml` / `marker_lora.yaml` drive what the training scripts print:
`model`, `batch_size`, `grad_accumulation_steps`, `epochs` (converted to
`--iters` as `len(train) * epochs / batch_size`), `learning_rate`,
`max_seq_length`. Every key in the config is used — if you need LoRA rank or
other `mlx_lm.lora` options, pass them via its own `-c` yaml config.

### Why `--mask-prompt`

The judge prompt (instructions + question + context + answer) is far longer
than the JSON completion. Without masking, loss is computed over the whole
sequence and prompt tokens dominate the gradient; the model optimizes
"repeat the prompt" instead of "produce the right labels". With
`--mask-prompt`, loss covers only completion tokens.

### Training data format

`{"prompt": ..., "completion": ...}` jsonl. `mlx_lm`'s `CompletionsDataset`
wraps these in the tokenizer chat template (user/assistant turns) — the same
template the inference scripts apply, so train and test distributions match.

### Scaling up

- Larger base model: swap `model:` in the config (e.g.
  `mlx-community/Qwen2.5-3B-Instruct-4bit`) — everything else stays the same.
- Real dataset: expect the marker-mode collapse seen on dummy data
  ([experiments.md](experiments.md)) to disappear with real volume; if not,
  raise epochs, check per-marker sample counts, and consider class weighting
  via data repetition.

## LettuceDetect Classifier

The LettuceDetect method does not use `mlx-lm` and does not fine-tune a
generative model. It extracts three aggregate features from LettuceDetect
token-level scores and trains a multi-output logistic regression classifier for
faithfulness and relevance.

Install the optional dependencies:

```bash
uv pip install -e ".[lettucedetect]"
```

Train:

```bash
python scripts/train_lettucedetect.py \
  --data data/dummy.jsonl \
  --output results/lettucedetect/classifier.joblib
```

Run inference:

```bash
python scripts/infer_lettucedetect.py \
  --data data/dummy.jsonl \
  --model results/lettucedetect/classifier.joblib \
  --output results/lettucedetect/predictions.jsonl
```

Evaluate with the same script used for all methods:

```bash
python scripts/evaluate.py \
  --data data/dummy.jsonl \
  --predictions results/lettucedetect/predictions.jsonl \
  --output results/lettucedetect/metrics.json
```

See [lettucedetect.md](lettucedetect.md) for method details and caveats.
