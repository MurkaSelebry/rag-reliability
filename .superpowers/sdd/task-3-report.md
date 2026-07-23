# Task 3 Report: Method 6 probabilities

## RED

Added three tests to `tests/test_m6_method.py` for Method 6 probability
emission, clipping to `[0, 1]`, and unchanged binary predictions.

Command:

```text
.venv/bin/python -m pytest tests/test_m6_method.py -q
```

Result before implementation: `2 failed, 8 passed`. The failing assertions
showed `faithfulness_prob` and `relevance_prob` were `None`, as expected.

## GREEN

`prediction_from_features` now derives and clips `p_faith = 1 - contradiction`
and `p_rel = cosine`, and sets `prob_method="m6_features"`. The existing binary
threshold calculations were left unchanged.

Focused command result:

```text
.venv/bin/python -m pytest tests/test_m6_method.py -q
10 passed in 0.13s
```

## Full verification

```text
make check
150 passed in 4.85s
All checks passed!

git diff --check
```

`git diff --check` completed without output or errors.

## Self-review

- Faithfulness probability is exactly `clip(1 - selfcheck_contra_mean)`.
- Relevance probability is exactly `clip(cos_q_a)`, including negative cosine.
- `prob_method` is exactly `m6_features`.
- Faithfulness binary logic still uses contradiction AND entropy thresholds.
- Relevance binary logic still uses the cosine threshold.
- Comments, docstrings, and type annotations remain English.
