# LettuceDetect feature classifier

This method uses LettuceDetect as a feature extractor, then trains a small
multi-output logistic regression classifier on top of the extracted features.
It shares the same input schema and evaluation script as the direct/marker
LoRA methods.

```
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
uv pip install -e ".[dev,lettucedetect]"
```

The `mlx-lm` extra is not needed for this method.

## Train

```bash
python scripts/train_lettucedetect.py \
  --data data/dummy.jsonl \
  --output results/lettucedetect/classifier.joblib
```

The script reuses `rag_reliability.dataset.load_jsonl` and
`rag_reliability.dataset.split_samples`. It trains on the train split and
prints validation metrics, but does not save split files yet.

## Infer

```bash
python scripts/infer_lettucedetect.py \
  --data data/dummy.jsonl \
  --model results/lettucedetect/classifier.joblib \
  --output results/lettucedetect/predictions.jsonl
```

The output is standard `Prediction` JSONL and can be evaluated with the existing
script:

```bash
python scripts/evaluate.py \
  --data data/dummy.jsonl \
  --predictions results/lettucedetect/predictions.jsonl \
  --output results/lettucedetect/metrics.json
```

## Implementation

| Module | Responsibility |
|---|---|
| `rag_reliability.methods.lettucedetect.features` | LettuceDetect detector setup, token-score aggregation, feature extraction. |
| `rag_reliability.methods.lettucedetect.classifier` | sklearn pipeline construction, target extraction, `Prediction` conversion. |
| `scripts/train_lettucedetect.py` | CLI for extracting train/validation features and saving the classifier artifact. |
| `scripts/infer_lettucedetect.py` | CLI for extracting inference features and writing `Prediction` JSONL. |

Feature caching is intentionally not implemented yet. It can be added later
under `results/lettucedetect/` without changing the prediction/evaluation
contract.
