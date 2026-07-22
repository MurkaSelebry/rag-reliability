# Notebooks

## `qwen7b_full_finetune.ipynb`

Full fine-tuning (**not LoRA**) of `Qwen/Qwen2.5-7B-Instruct` on the repo's
RAG-reliability-judge task, for GPU-cloud notebook environments (Google
Colab, Kaggle, Yandex DataSphere) rather than the Apple Silicon / MLX
workflow used elsewhere in this repo (see [docs/training.md](../docs/training.md)
for the LoRA/MLX path). It trains **one method per run** — Method 1 (direct)
or Method 2 (marker) — selected via the `MODE` switch, and produces a
Hugging Face full-fine-tune checkpoint plus in-notebook evaluation against
the repo's own parsing and metrics code.

### How to open it

- **Google Colab**: upload the `.ipynb` (or open it from GitHub via
  `File > Open notebook > GitHub`), select a GPU runtime (A100/L4/T4
  depending on availability), and run cells top to bottom.
- **Kaggle**: create a new notebook, upload/import this file, turn on a GPU
  accelerator (e.g. 2xT4) in the notebook settings, and run top to bottom.
- **Yandex DataSphere**: import the notebook into a project, attach a GPU
  configuration, and run top to bottom. The platform-detection cell
  recognizes DataSphere via `/home/jupyter` or a `DATASPHERE` marker in
  `HOSTNAME` and adjusts the default save location accordingly.

In all three environments the notebook is self-contained: the first cell
installs the pinned dependency stack (skipped if already satisfied), and the
second cell clones this repo so training reuses the repo's own formatting
and parsing code.

### `REPO_URL` — set this before running

Cell 2 clones the repo and imports its format builders, dataset loader,
parser, and metrics module (`nb_format.build_sft_messages`,
`dataset.load_jsonl`/`split_samples`, `parsing.parse_prediction`,
`schema.RagSample`, `metrics`) so that the chat-template formatting used for
training is byte-for-byte the same code path used at inference time. Before
running, **edit `REPO_URL` in cell 2** to a reachable clone of this repo
(your fork, or this repo's remote) — the placeholder
`https://github.com/<owner>/rag-reliability-judge.git` will not resolve as
committed.

If the clone or import fails, the cell **raises** rather than silently
degrading — the only sanctioned way around that is the documented manual
step: paste the inline fallback (the bodies of the prompt/target builders,
`RagSample`, `ALLOWED_MARKERS`, `parse_prediction`, `load_jsonl`,
`split_samples`) directly into the cell, keeping it in sync with the repo
modules and `tests/test_nb_format.py`. The repo-import path is the tested
default; the inline fallback exists only for environments that genuinely
cannot reach the repo.

### Config switches (cell 3)

- **`MODE`** — `"direct"` (Method 1) or `"marker"` (Method 2). Trains one
  method per run; to get both, run the notebook twice with different
  `MODE` values (each writes to its own `OUTPUT_DIR = f"ft_{MODE}"`).
- **`USE_REAL_DATA`** — `False` (default) trains on the repo's small dummy
  set (`data/dummy.jsonl`, copied in via the cloned repo) for a fast smoke
  run; `True` points at `data/organizers.jsonl` (the real organizer
  dataset, produced by `scripts/prepare_data.py` — see the main
  [README](../README.md#advanced--pipelines)) — you must have that file in
  place before switching this on.
- **`SAVE_TARGET`** — `"auto"` (default) saves the checkpoint to Google
  Drive on Colab, `/kaggle/working` on Kaggle, or the notebook's working
  directory elsewhere; set it to an explicit path to override.
- **`QLORA_FALLBACK`** — `False` (default). **This is not full fine-tuning.**
  It is an escape hatch for hardware that cannot support any of the full-FT
  profiles below: when set to `True`, the notebook loads the model 4-bit
  quantized and trains a LoRA adapter instead of full weights. Leave it
  `False` unless the hardware-detection cell (cell 4) hard-stops with
  "insufficient" and you specifically want a degraded, adapter-only run
  instead of moving to bigger hardware.

### Hardware profiles (auto-detected in cell 4)

The notebook inspects GPU count, per-GPU VRAM, and CPU RAM, then picks one
of four profiles — no manual selection needed, but you should provision
hardware that lands you in one of the first three:

| Profile | Requirement | Notes |
|---|---|---|
| `full_single` | 1 GPU, ≥ ~70GB VRAM (i.e. an 80GB A100/H100) | Whole model + 8-bit-AdamW optimizer states fit on one GPU; no DeepSpeed needed. A bare 48GB card is intentionally routed to offload instead — too tight for full 7B FT. |
| `full_zero3_offload` (single-proc) | 1 GPU, ≥22GB and below the 70GB `full_single` threshold (e.g. 40–48GB A100/L4/L40), and ≥ ~60GB CPU RAM | DeepSpeed ZeRO-3 with CPU offload of optimizer + parameters; the notebook warns (but does not stop) if CPU RAM looks short. |
| `full_zero3_offload` (multi-proc) | ≥ 2 GPUs (e.g. 2×T4) | Same DeepSpeed ZeRO-3 CPU-offload config, sharded across processes launched via `accelerate.notebook_launcher`. |
| `insufficient` | A single GPU below the thresholds above | Hard stop (`RuntimeError`) unless `QLORA_FALLBACK=True` — full fine-tuning of a 7B model is not possible on that hardware. |

DeepSpeed ZeRO-3 offload (cell 7) is configured with `bf16`, CPU-offloaded
optimizer and parameter state, and `stage3_gather_16bit_weights_on_model_save`
so the final saved checkpoint is a normal consolidated Hugging Face model
rather than sharded ZeRO shards.

### Pinned stack

Dependencies are floor-pinned (`>=`), not exact-pinned, with one deliberate
exception: **`trl==1.8.0`** is pinned exactly because it is the version that
supports `assistant_only_loss` in `SFTConfig` — completion-only loss masking
that trains only on assistant-turn tokens. This mirrors the repo's MLX LoRA
training recipe's `--mask-prompt` flag (see
[docs/training.md](../docs/training.md)), keeping the loss-masking behavior
consistent between the MLX/LoRA path and this full-FT path. The rest of the
stack (`transformers`, `accelerate`, `datasets`, `peft`, `bitsandbytes`,
`deepspeed`) is floor-pinned deliberately: Colab and Kaggle ship a
preinstalled, CUDA-matched build of these libraries, and a floor pin lets
that preinstalled build satisfy the requirement instead of being reinstalled
against a mismatched CUDA toolkit.

### Evaluation caveat

The checkpoint this notebook produces is a plain Hugging Face full-fine-tune
model (config + `safetensors`/`bin` weights + tokenizer files), evaluated
**in-notebook** in cell 8: greedy generation over the held-out test split,
parsed with the repo's own `parsing.parse_prediction`, and scored with the
repo's own `metrics.evaluate_predictions`. That in-notebook cell is the
**primary evaluation path** for this checkpoint.

The repo's `scripts/infer.py` is **not** a drop-in evaluator for this
checkpoint: it loads models through `rag_reliability.mlx_backend`, i.e. the
MLX runtime for Apple Silicon. Running this HF/CUDA checkpoint through
`scripts/infer.py` as-is will not work — doing so would require a
transformers-based backend equivalent (not part of this task). Cell 9 prints
the `scripts/infer.py` / `scripts/evaluate.py` invocation for reference (and
as a target for such a backend, if one is added later), but until that
backend exists, treat cell 8's in-notebook evaluation as authoritative.

### Output

Training writes to `OUTPUT_DIR` (`ft_direct` or `ft_marker`), then cell 9
copies the checkpoint to the resolved `SAVE_TARGET` location (Drive /
`/kaggle/working` / local working dir) so it survives runtime disconnects.
Pushing to the Hugging Face Hub is available but commented out and opt-in.

## Yandex DataSphere — extra setup (real full FT on A100 80GB)

DataSphere's `DS Default` image is older than the 2026 training stack and its
project disk is tiny (~10 GB), so a run there needs three DataSphere-specific
steps that Colab/Kaggle don't. This recipe is battle-tested for the `g2.1`
config (1× A100 80 GB, 119 GB RAM).

**1. Attach a large file storage.** The 10 GB project disk cannot hold the
~15 GB model cache *and* the ~15 GB checkpoint. Create a File Storage
(Ресурсы проекта → Файловое хранилище → **Активировать**), then **restart the
compute VM** (stop the running instance — a kernel restart is not enough; the
mount is injected only when a fresh VM starts). It appears at
`/home/jupyter/filestore/<name>/`. Point both the HF cache and the output there.

**2. Install a driver-matched stack.** DataSphere's driver is CUDA 12.2, so the
PyPI `torch` that `trl`/`transformers` would otherwise pull (CUDA 13) fails with
"driver too old", and the base `torch` (2.0.1/cu118) is too old for `trl 1.8`.
Pin `torch==2.5.1` on **cu121**. Also pin `numpy==1.26.4` **last** — new enough
for `trl`'s typing, old enough (`<2`) to keep the base image's numpy-1-compiled
C-extensions (`soxr`, `scipy`, `sklearn`, `numba`) ABI-compatible; numpy 2 breaks
them. Run this as the first cell (replacing cell 1's installer), then
**Restart Kernel**:

```python
import os, subprocess, sys
BASE = "/home/jupyter/filestore/<name>"          # your mounted storage
os.environ["HF_HOME"] = BASE + "/hf"              # model cache on the big disk
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def pip(*a): subprocess.check_call([sys.executable, "-m", "pip", "install", *a])
pip("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
    "--index-url", "https://download.pytorch.org/whl/cu121")
pip("transformers>=4.56.2", "trl==1.8.0", "accelerate>=1.4.0", "datasets==4.7.0",
    "peft>=0.8.0", "bitsandbytes>=0.44.1", "pydantic>=2.5", "sentencepiece", "psutil")
pip("numpy==1.26.4")   # LAST — overrides any numpy 2 pulled above
```

`HF_HOME` and `PYTORCH_CUDA_ALLOC_CONF` reset on every kernel restart, so also
set them at the very top of cell 1 (before any `transformers` import).

**3. Point the output at the big disk and don't re-run cell 7 dirty.** In cell 3
set `OUTPUT_DIR = f"{BASE}/ft_{MODE}"` (the default relative `ft_{MODE}` lands on
the 10 GB disk and the ~15 GB save fails with `No space left on device`). In
cell 7 set `save_strategy="no"` (per-epoch checkpoints include the optimizer
state and blow the disk). Each cell-7 run leaks GPU memory, so **Restart Kernel
before re-running cell 7** — otherwise the second run OOMs on an already-full
80 GB GPU.

Everything else (cells 2, 4–9) runs unchanged. `PLATFORM` auto-detects as
`datasphere`, so cell 9 leaves the checkpoint in place on the mounted storage
(no copy). Remember DataSphere billing: stop the VM and deactivate/delete the
file storage when done.
