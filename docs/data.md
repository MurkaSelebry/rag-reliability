# Data

## Sample schema (jsonl, one object per line)

```json
{
  "id": "sample_001",
  "question": "...",
  "context": "...",
  "answer": "...",
  "faithfulness": 1,
  "relevance": 1,
  "marker": "none"
}
```

- `faithfulness`, `relevance` — int 0/1, gold labels.
- `reliable` is **not** stored; it is derived as
  `faithfulness AND relevance` everywhere.
- `marker` — optional error type. When missing, training targets and marker
  metrics derive it: `none` for reliable samples, `unknown` for unreliable
  ones (`resolve_marker()` in `formatting.py`).

## Marker vocabulary

Defined once in `prompts.py::ALLOWED_MARKERS`:

| Marker | Meaning |
|---|---|
| `none` | Answer is reliable (only valid for reliable samples) |
| `unknown` | Unreliable, but no specific error type labeled |
| `hallucination` | Facts not present in the context |
| `off_topic_answer` | Doesn't address the question |
| `incomplete_answer` | Addresses the question only partially |
| `context_mixing` | Mixes facts from unrelated parts of the context |
| `contradiction` | Contradicts the context |
| `unsupported_claim` | Claim that the context neither supports nor denies |

## Dummy dataset

`data/dummy.jsonl` — 36 synthetic Russian banking RAG examples:
14 reliable + 22 covering the six concrete error markers. Exists so the whole
pipeline (including metrics discrimination) can be exercised without the real
dataset; see [experiments.md](experiments.md).

## Converting the real dataset

```bash
python scripts/prepare_data.py \
  --input raw_dataset.jsonl \
  --output data/processed.jsonl \
  --column-map '{"query": "question", "passage": "context", "response": "answer"}'
```

`--column-map` renames source fields into the schema above; unknown fields are
passed through and rejected by validation if extraneous.

**TODO when the real format is known:**
- CSV branch in `prepare_data.py` (currently `NotImplementedError`):
  delimiter, encoding, label columns, 0/1 type coercion.
- Verify label semantics match (what exactly was annotated as
  faithfulness/relevance) and the marker set — extend `ALLOWED_MARKERS` if the
  real annotation uses different categories.

## Splits

`split_samples()` (dataset.py): stratified by `reliable`, 80/10/10, seed 42.
Training scripts write both a flat `results/train_<mode>.jsonl` and the
`results/lora_<mode>/{train,valid,test}.jsonl` layout that `mlx_lm.lora`
expects.
