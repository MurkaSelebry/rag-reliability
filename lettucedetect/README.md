# LettuceDetect Feature Classifier

This directory contains an isolated implementation of the LettuceDetect-based
method. It does not modify the existing direct/marker LoRA pipeline.

The method uses LettuceDetect as a feature extractor:

```text
RagSample(question, context, answer)
  -> token-level LettuceDetect scores over answer tokens
  -> [max score, mean score, fraction(score > threshold)]
  -> StandardScaler + multi-output LogisticRegression
  -> Prediction(faithfulness_pred, relevance_pred)
```

LettuceDetect is expected to be a stronger signal for faithfulness than for
relevance. Relevance is learned only indirectly from the same three features.

## Install

From the repository root:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv pip install -r lettucedetect/requirements.txt
```

The `mlx-lm` extra is not needed for this method.

## Train

```bash
python lettucedetect/train_classifier.py \
  --data data/dummy.jsonl \
  --output results/lettucedetect/classifier.joblib
```

The script reuses `rag_reliability.dataset.load_jsonl`,
`rag_reliability.dataset.split_samples`, and writes `RagSample` split files to:

```text
results/lettucedetect/splits/train.jsonl
results/lettucedetect/splits/val.jsonl
results/lettucedetect/splits/test.jsonl
```

## Infer

```bash
python lettucedetect/infer_classifier.py \
  --data data/dummy.jsonl \
  --model results/lettucedetect/classifier.joblib \
  --split test \
  --output results/lettucedetect/predictions_test.jsonl
```

The output is standard `Prediction` JSONL and can be evaluated with the existing
script:

```bash
python scripts/evaluate.py \
  --data results/lettucedetect/splits/test.jsonl \
  --predictions results/lettucedetect/predictions_test.jsonl \
  --output results/lettucedetect/metrics_test.json
```

## Notes

- Run these scripts as files (`python lettucedetect/train_classifier.py`), not
  as modules. The local directory is named `lettucedetect`, so `python -m ...`
  can shadow the third-party package.
- The current implementation recomputes features during training and inference.
  A feature cache can be added later without changing the evaluation contract.
- The code reads the same `RagSample` source used by the LoRA methods. It does
  not depend on `wandb/RAGTruth-processed`; that dataset is only useful as a
  prototype/reference source.
