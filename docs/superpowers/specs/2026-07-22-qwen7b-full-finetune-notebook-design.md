# Qwen2.5-7B full fine-tune notebook (Methods 1 & 2)

**Date:** 2026-07-22
**Status:** approved design
**Author:** brainstorming session

## Goal

A single Jupyter notebook that **fully fine-tunes** (not LoRA)
`Qwen/Qwen2.5-7B-Instruct` on the project's judge task, runnable on Colab,
Kaggle, or Yandex DataSphere. It supports **Method 1 (direct)** and **Method 2
(marker)** via a `MODE` switch, auto-detects the available GPU/RAM and picks a
memory strategy, and evaluates the result with the same prediction/metrics
contract the repo already uses so numbers are comparable to the LoRA runs.

## Hard constraints

- **Full fine-tuning only.** LoRA/QLoRA is not the deliverable. QLoRA exists
  solely as an explicit, clearly-labeled `QLORA_FALLBACK` escape hatch for when
  the only hardware available (a single 16GB T4) cannot do full FT at all.
- **Train == inference format.** The notebook reuses the repo's format builders
  and parser rather than reimplementing them, so the training distribution
  matches what `scripts/infer.py` produces at test time.

## Non-goals

- No changes to the existing repo training scripts or CLI.
- No LoRA-vs-full comparison harness — this notebook produces one full-FT
  checkpoint per invocation for the selected `MODE`.
- No multi-node / cluster orchestration beyond single-machine multi-GPU.

## Format symmetry (the critical part)

The notebook clones this repo and `pip install -e .`, then imports:

- `rag_reliability.prompts.build_direct_prompt`, `build_marker_prompt`
- `rag_reliability.formatting.build_direct_target`, `build_marker_target`,
  `build_chat_training_record`, `resolve_marker`
- `rag_reliability.dataset.load_jsonl`, `split_samples`
- `rag_reliability.parsing.parse_prediction`
- `rag_reliability.metrics.*` (same contract as `scripts/evaluate.py`)
- `rag_reliability.schema.RagSample`, `Prediction`, `ALLOWED_MARKERS`

If the clone/import fails (repo unreachable), a fallback cell inlines verbatim
copies of the small builders + parser so the notebook still runs standalone.
The inline copies are a mirror, and the design notes they must be kept in sync.

**SFT record** (identical to `build_chat_training_record`):

```json
{"messages": [
  {"role": "user", "content": "<build_{mode}_prompt(sample)>"},
  {"role": "assistant", "content": "<build_{mode}_target(sample)>"}
]}
```

- Loss on assistant tokens only via TRL `SFTConfig(assistant_only_loss=True)` —
  the completion-only-loss equivalent of mlx-lm `--mask-prompt`.
- Chat template comes from the Qwen2.5 tokenizer (same template the inference
  path applies through `apply_chat_template`).

## Hardware auto-detection → 3 real profiles

A detection cell measures `n_gpus`, per-GPU + total VRAM, and CPU RAM, then
selects a profile. All three do **real full fine-tuning**. Naive bf16 + AdamW
full FT of 7B needs ~84GB, so smaller cards use ZeRO-3 offload.

| Detected | Profile | Strategy |
|---|---|---|
| Total VRAM ≥ ~48GB (A100/H100 80GB) | `full_single` | Single-process full FT: bf16, gradient checkpointing, 8-bit AdamW (bitsandbytes `adamw_bnb_8bit`). Simplest, fastest. |
| 1 GPU, 24–40GB (A100/L4 40GB) | `full_zero3_offload` | Full FT via DeepSpeed ZeRO-3 with optimizer **and** parameter CPU offload. Requires ≥~60GB CPU RAM. |
| Multi-GPU ≤16GB each (2×T4) | `full_zero3_offload` (multi-process via `accelerate.notebook_launcher`) | ZeRO-3 + CPU offload sharded across processes. Fully supported path (not just a warning). |
| Single 16GB T4 | `insufficient` | Hard stop with a clear message. `QLORA_FALLBACK=True` opts into 4-bit QLoRA, labeled "NOT full FT". |

Decisions locked in brainstorming:
- **DeepSpeed path is a first-class, fully-supported path**, not experimental.
- A DeepSpeed ZeRO-3 offload JSON config is written by the notebook and passed
  to the HF `Trainer`/TRL `SFTTrainer`. Multi-GPU launches use
  `accelerate.notebook_launcher` wrapping the train function.
- The detection cell prints the chosen profile, VRAM/RAM numbers, and the
  reasoning, so the user sees exactly which flow ran.

## MODE switch (not a loop)

`MODE = "direct"` or `MODE = "marker"` at the top of the config cell. One run
trains one method. Decided in brainstorming over a two-method loop: keeps each
run's memory footprint and logs clean, and the two targets have different
formats. Output dir is `ft_{MODE}/`.

## Notebook structure (cells)

1. **Install** — pinned `transformers`, `trl`, `peft`, `bitsandbytes`,
   `deepspeed`, `accelerate`, `datasets`, versions known-compatible with
   Qwen2.5. Detect platform (Colab / Kaggle / DataSphere / other).
2. **Clone repo + import** format builders/parser/metrics (with inline
   fallback).
3. **Config** — `MODE`, `BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"`, `DATA_PATH`
   (dummy↔real switch), hyperparameters (epochs, LR, max_seq_len, batch, grad
   accumulation), `SAVE_TARGET`, `QLORA_FALLBACK`.
4. **Hardware detect** → profile selection + printout.
5. **Data** — `load_jsonl(DATA_PATH)` → `split_samples` (stratified by
   `reliable`, seed 42) → chat records for train/val via the imported builder.
6. **Model + tokenizer** loaded per profile (dtype, device_map, gradient
   checkpointing, quant only if `QLORA_FALLBACK`).
7. **Train** — TRL `SFTTrainer` with `assistant_only_loss=True`; `full_single`
   runs inline, ZeRO paths run via the DeepSpeed config /
   `notebook_launcher`. Saves to `ft_{MODE}/`.
8. **Evaluate** — greedy generation (`max_new_tokens≈64`) over the test split
   using `apply_chat_template`, parse with `parse_prediction(expect_marker=MODE=="marker")`,
   compute metrics with the repo's `metrics` module. Print per-method metrics
   JSON and invalid-output rate.
9. **Save + next steps** — copy `ft_{MODE}/` to the save target, print the
   `scripts/infer.py` command to run the checkpoint through the repo pipeline.

## Data flow

- `DATA_PATH` default: `data/dummy.jsonl` from the clone (pipeline smoke test).
- Real dataset: an upload cell accepts `organizers.jsonl`, or runs
  `scripts/prepare_data.py --input <data.zip> --output data/organizers.jsonl`,
  then point `DATA_PATH` at it.
- Split: repo `split_samples` (80/10/10, stratified by `reliable`, seed 42) —
  same split logic as the LoRA scripts so test sets are comparable.

## Evaluation

- Generate on the held-out **test** split only.
- Parse with the repo `parse_prediction` (first balanced JSON substring;
  invalid outputs flagged, not crashed).
- Metrics via the repo `metrics` module (same as `scripts/evaluate.py`):
  accuracy / precision / recall / f1 / f1_macro on `reliable`, ROC-style where
  available, plus marker metrics for Method 2.
- Report invalid-output rate (the LoRA path reports 0% — this is the
  comparison anchor).

## Saving

Decided: **Google Drive or a plain output folder** (no HF Hub requirement).

- Colab detected → offer `drive.mount('/content/drive')` and copy `ft_{MODE}/`
  there.
- Kaggle → write under `/kaggle/working/` (persisted as notebook output).
- DataSphere / other → local output dir; print the path.
- HF Hub push is an **optional** commented block gated on a token, off by
  default.

## Testing / verification

- Smoke run on `data/dummy.jsonl` end-to-end (train few steps → eval → metrics
  print) must complete without error on whichever profile the runner lands on.
- Assert the built SFT record for a known sample equals
  `build_chat_training_record(sample, MODE)` from the repo — guards format
  drift between the notebook and the repo.
- Eval cell must produce a valid `metrics.json`-shaped dict.

## Risks / notes

- DeepSpeed ZeRO-3 CPU offload needs substantial CPU RAM; the detect cell
  checks RAM and warns if below the profile's requirement before training.
- `notebook_launcher` re-imports the module state; the train function must be
  self-contained (no reliance on globals mutated in other cells) — the design
  puts everything it needs inside the function or passes it as args.
- Version pins matter: TRL `assistant_only_loss` + Qwen2.5 chat template must be
  on compatible `transformers`/`trl` releases; the install cell pins them.
