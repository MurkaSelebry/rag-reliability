# LettuceDetect Feature Classifier

This directory contains an isolated implementation of the LettuceDetect-based
method. It is shaped as a staging area for a later move into:

```text
src/rag_reliability/methods/lettucedetect/
scripts/train_lettucedetect.py
scripts/infer_lettucedetect.py
```

The directory is named `lettucedetect_method` to avoid shadowing the installed
third-party package named `lettucedetect`.

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

## Layout

```text
features.py              LettuceDetect detector setup and feature extraction
classifier.py            sklearn classifier helpers and Prediction conversion
train_lettucedetect.py   training CLI
infer_lettucedetect.py   inference CLI
train_classifier.py      compatibility wrapper
infer_classifier.py      compatibility wrapper
common.py                compatibility imports
requirements.txt         temporary isolated dependencies
```

## Install

From the repository root:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv pip install -r lettucedetect_method/requirements.txt
```

The `mlx-lm` extra is not needed for this method.

## Train

```bash
python lettucedetect_method/train_lettucedetect.py \
  --data data/dummy.jsonl \
  --output results/lettucedetect/classifier.joblib
```

The script reuses `rag_reliability.dataset.load_jsonl` and
`rag_reliability.dataset.split_samples`. It trains on the train split and
prints validation metrics, but it does not save split files yet.

The old entry point still works during the staging phase:

```bash
python lettucedetect_method/train_classifier.py \
  --data data/dummy.jsonl \
  --output results/lettucedetect/classifier.joblib
```

## Infer

```bash
python lettucedetect_method/infer_lettucedetect.py \
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

The old entry point still works during the staging phase:

```bash
python lettucedetect_method/infer_classifier.py \
  --data data/dummy.jsonl \
  --model results/lettucedetect/classifier.joblib \
  --output results/lettucedetect/predictions.jsonl
```

## Integration Notes

- Move `features.py` and `classifier.py` into
  `src/rag_reliability/methods/lettucedetect/`.
- Move `train_lettucedetect.py` and `infer_lettucedetect.py` into `scripts/`
  and replace local imports with package imports.
- Replace this `requirements.txt` with a `lettucedetect` optional extra in
  `pyproject.toml`.
- Delete compatibility wrappers and this staging directory after the package
  integration is complete.
