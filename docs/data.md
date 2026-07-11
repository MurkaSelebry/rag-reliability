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

The organizer dataset also uses the official `reason_*` marker labels:

| Marker | Meaning |
|---|---|
| `reason_hallucinated_fact` | The model hallucinated an incorrect fact |
| `reason_off_topic_answer` | The model answered a different question |
| `reason_irrelevant_chunk_used` | The model used an irrelevant knowledge chunk |
| `reason_chunk_fact_mixup` | The model mixed facts from multiple chunks |
| `reason_incomplete_answer` | The answer omitted important information |
| `reason_false_verification` | The model falsely claimed verification or calculation |
| `reason_outdated_fact` | The answer uses outdated information |
| `reason_answer_for_operator` | The answer contains operator-only information |
| `reason_other` | Other reason |
| `reason_reveals_ai_identity` | The answer reveals or implies AI identity |
| `reason_wrong_navigation` | The answer points to the wrong article/path |
| `reason_missed_complaint_handoff` | The model missed a required complaint handoff |
| `reason_missed_chunk_conditions` | The model missed conditions present in chunks |

## Dummy dataset

`data/dummy.jsonl` — 36 synthetic Russian banking RAG examples:
14 reliable + 22 covering the six concrete error markers. Exists so the whole
pipeline (including metrics discrimination) can be exercised without the real
dataset; see [experiments.md](experiments.md).

## Converting the real dataset

Organizer CSV/ZIP format:

```bash
python scripts/prepare_data.py \
  --input from_organizators/data/data.zip \
  --output data/organizers.jsonl
```

This maps:

- `full_dialog` -> `question` (the prompts treat it as a dialog and evaluate
  relevance against the latest client request)
- `chunk_1` ... `chunk_8` -> numbered, merged `context`
- `answer` -> `answer`
- `binary_faithfulness` -> `faithfulness`
- `binary_relevancy` -> `relevance`
- first value from `markers` -> `marker`; empty reliable rows become `none`,
  empty unreliable rows become `unknown`

The organizer markers are sparse and may contain multiple reasons in one row.
The current `RagSample` schema stores one marker, so conversion keeps the first
reason. Binary reliability experiments do not depend on marker completeness.

Generic json/jsonl format:

```bash
python scripts/prepare_data.py \
  --input raw_dataset.jsonl \
  --output data/processed.jsonl \
  --column-map '{"query": "question", "passage": "context", "response": "answer"}'
```

`--column-map` renames source fields into the schema above; unknown fields are
passed through and rejected by validation if extraneous.

**TODO after first real-data runs:**
- Decide whether marker-mode should use only rows with explicit marker labels,
  all rows with `unknown` fallback, or a multi-label formulation outside the
  current single-marker schema.
- Add a held-out split export for final evaluation so LoRA smoke tests do not
  evaluate on training rows.

## Splits

`split_samples()` (dataset.py): stratified by `reliable`, 80/10/10, seed 42.
Training scripts write both a flat `results/train_<mode>.jsonl` and the
`results/lora_<mode>/{train,valid,test}.jsonl` layout that `mlx_lm.lora`
expects.
