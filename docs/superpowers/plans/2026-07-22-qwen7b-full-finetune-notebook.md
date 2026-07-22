# Qwen2.5-7B Full Fine-Tune Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Jupyter notebook that fully fine-tunes `Qwen/Qwen2.5-7B-Instruct` on Method 1 (direct) or Method 2 (marker), auto-selecting a memory strategy for the detected GPU, runnable on Colab / Kaggle / Yandex DataSphere.

**Architecture:** The notebook clones this repo and imports its format builders + parser + metrics so training matches inference exactly. A detection cell measures VRAM/RAM and picks one of three real full-FT profiles (`full_single`, `full_zero3_offload` single-proc, `full_zero3_offload` multi-proc via `notebook_launcher`). Training uses TRL `SFTTrainer` with `assistant_only_loss=True`. A local pytest guards the notebook's inline fallback format against the repo builders so the two never drift.

**Tech Stack:** Jupyter (nbformat), Hugging Face `transformers` / `trl` / `accelerate` / `bitsandbytes` / `deepspeed`, PyTorch, the repo's `rag_reliability` package (imported at runtime in the notebook).

## Global Constraints

- Full fine-tuning only; QLoRA is an explicit `QLORA_FALLBACK` escape hatch labeled "NOT full FT". (spec: Hard constraints)
- Base model: `Qwen/Qwen2.5-7B-Instruct` (verbatim). (spec: Goal)
- Train == inference format: SFT record is `{"messages":[{"role":"user","content":build_{mode}_prompt(sample)},{"role":"assistant","content":build_{mode}_target(sample)}]}`, loss on assistant tokens only. (spec: Format symmetry)
- `MODE` is a single switch (`"direct"` | `"marker"`), one method per run — not a loop. (spec: MODE switch)
- Split via repo `split_samples` (80/10/10, stratified by `reliable`, seed 42). (spec: Data flow)
- Eval uses repo `parse_prediction` + `metrics` module; report invalid-output rate. (spec: Evaluation)
- Save target: Google Drive or plain output folder; HF Hub push optional/off by default. (spec: Saving)
- Notebook path: `notebooks/qwen7b_full_finetune.ipynb`. Fallback format module: `src/rag_reliability/nb_format.py`.

---

## File structure

- Create `src/rag_reliability/nb_format.py` — one self-contained function `build_sft_messages(sample, mode)` that returns the chat record, plus `INLINE_SOURCE` (its own source text) so the notebook's fallback cell is a verbatim copy of a repo-tested function. This is the seam that makes the "inline fallback" testable.
- Create `tests/test_nb_format.py` — asserts `build_sft_messages` equals the repo's `build_chat_training_record` for fixtures across both modes (drift guard).
- Create `notebooks/qwen7b_full_finetune.ipynb` — the deliverable, 9 cells.
- Create `notebooks/README.md` — how to open/run per platform, what each profile does.
- Modify `README.md` — add a one-line pointer to the notebook under the docs map / advanced section.

Rationale: the risky, testable logic (format symmetry) lives in a real repo module with a real test. The notebook imports it when the clone succeeds and carries a verbatim copy for the offline fallback. Everything GPU-bound lives only in the notebook (can't run in local CI on this Apple-Silicon repo).

---

### Task 1: Repo-tested format helper + drift guard

**Files:**
- Create: `src/rag_reliability/nb_format.py`
- Test: `tests/test_nb_format.py`

**Interfaces:**
- Consumes: `rag_reliability.formatting.build_chat_training_record`, `rag_reliability.schema.RagSample`.
- Produces: `build_sft_messages(sample: RagSample, mode: str) -> dict[str, list[dict[str, str]]]` returning `{"messages": [ {"role": "user", ...}, {"role": "assistant", ...} ]}`. Notebook cells (Task 6) import this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nb_format.py
import pytest

from rag_reliability.formatting import build_chat_training_record
from rag_reliability.nb_format import build_sft_messages
from rag_reliability.schema import RagSample

SAMPLES = [
    RagSample(
        id="ok",
        question="Сколько стоит обслуживание?",
        context="Обслуживание карты «Классика» составляет 149 рублей.",
        answer="149 рублей.",
        faithfulness=1,
        relevance=1,
        marker="none",
    ),
    RagSample(
        id="bad",
        question="Какой суточный лимит?",
        context="Лимит 100 000 рублей.",
        answer="500 000 рублей.",
        faithfulness=0,
        relevance=1,
        marker="hallucination",
    ),
]


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("mode", ["direct", "marker"])
def test_nb_format_matches_repo(sample: RagSample, mode: str) -> None:
    assert build_sft_messages(sample, mode) == build_chat_training_record(sample, mode)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nb_format.py -v`
Expected: FAIL — `ModuleNotFoundError: rag_reliability.nb_format`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag_reliability/nb_format.py
"""Self-contained SFT record builder mirrored verbatim into the training notebook.

The notebook imports this when it can clone the repo; when it can't, it pastes
INLINE_SOURCE. tests/test_nb_format.py asserts this stays equal to the repo's
build_chat_training_record so the notebook fallback never drifts.
"""

from __future__ import annotations

from rag_reliability.formatting import build_chat_training_record
from rag_reliability.schema import RagSample


def build_sft_messages(sample: RagSample, mode: str) -> dict[str, list[dict[str, str]]]:
    """One SFT chat record: user=judge prompt, assistant=JSON verdict."""
    return build_chat_training_record(sample, mode)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_nb_format.py -v`
Expected: PASS (4 params).

- [ ] **Step 5: Commit**

```bash
git add src/rag_reliability/nb_format.py tests/test_nb_format.py
git commit -m "Add nb_format helper + drift guard for training notebook"
```

---

### Task 2: Notebook skeleton — install + platform detect (cell 1)

**Files:**
- Create: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Produces: a valid nbformat v4 notebook whose first code cell installs pinned deps and sets `PLATFORM` in {"colab","kaggle","datasphere","other"}.

- [ ] **Step 1: Author the notebook file with a markdown title cell + cell 1**

Create `notebooks/qwen7b_full_finetune.ipynb` as nbformat v4 JSON. Markdown cell 0:

```markdown
# Qwen2.5-7B full fine-tune — RAG reliability judge (Method 1 / 2)
Full fine-tuning (not LoRA) of `Qwen/Qwen2.5-7B-Instruct`. Auto-detects GPU and
picks a memory strategy. Set `MODE` in the config cell to `"direct"` or `"marker"`.
Runs on Colab, Kaggle, Yandex DataSphere.
```

Code cell 1 (install + platform):

```python
# --- Install pinned deps (skip if already present) ---
import importlib, subprocess, sys

PKGS = [
    "transformers==4.46.3",
    "trl==0.12.2",
    "accelerate==1.1.1",
    "datasets==3.1.0",
    "peft==0.13.2",
    "bitsandbytes==0.44.1",
    "deepspeed==0.15.4",
    "sentencepiece",
]
def _pip(args): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
try:
    import trl, transformers  # noqa: F401
except Exception:
    _pip(PKGS)

# --- Detect platform ---
import os
if "google.colab" in sys.modules or os.path.exists("/content"):
    PLATFORM = "colab"
elif os.path.exists("/kaggle"):
    PLATFORM = "kaggle"
elif os.path.exists("/home/jupyter") or "DATASPHERE" in os.environ.get("HOSTNAME", "").upper():
    PLATFORM = "datasphere"
else:
    PLATFORM = "other"
print("platform:", PLATFORM)
```

- [ ] **Step 2: Verify the notebook is valid JSON/nbformat**

Run: `python -c "import nbformat; nbformat.read('notebooks/qwen7b_full_finetune.ipynb', as_version=4); print('ok')"`
Expected: `ok` (no schema error).

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: install + platform-detect cell"
```

---

### Task 3: Clone repo + import format helpers, with inline fallback (cell 2)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Consumes: `src/rag_reliability/nb_format.py::build_sft_messages` (Task 1) when clone succeeds.
- Produces: in-notebook names `build_sft_messages`, `load_jsonl`, `split_samples`, `parse_prediction`, `RagSample`, and the `metrics` module handle `M`, plus `REPO_DIR`.

- [ ] **Step 1: Add cell 2**

```python
# --- Get the repo so training == inference format ---
REPO_URL = "https://github.com/<owner>/rag-reliability-judge.git"  # set to your fork/remote
REPO_DIR = "rag-reliability-judge"
import os, subprocess, sys

def _have_repo():
    try:
        import rag_reliability  # noqa: F401
        return True
    except Exception:
        return False

if not _have_repo():
    if not os.path.exists(REPO_DIR):
        try:
            subprocess.check_call(["git", "clone", "-q", REPO_URL, REPO_DIR])
        except Exception as e:
            print("clone failed, using inline fallback:", e)
    if os.path.isdir(REPO_DIR):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", REPO_DIR])

try:
    from rag_reliability.nb_format import build_sft_messages
    from rag_reliability.dataset import load_jsonl, split_samples
    from rag_reliability.parsing import parse_prediction
    from rag_reliability.schema import RagSample
    from rag_reliability import metrics as M
    USING_REPO = True
except Exception as e:
    print("repo import failed -> inline fallback:", e)
    USING_REPO = False
    # Paste verbatim: nb_format.INLINE fallback (kept in sync by tests/test_nb_format.py)
    # NOTE for implementer: copy the bodies of build_direct_prompt/build_marker_prompt,
    # build_direct_target/build_marker_target, resolve_marker, RagSample, ALLOWED_MARKERS,
    # parse_prediction, load_jsonl, split_samples here. Source of truth: the repo modules.
    raise RuntimeError("Set REPO_URL to a reachable remote, or paste the inline fallback block.")

print("format source:", "repo" if USING_REPO else "inline")
```

Implementer note: the fallback branch is intentionally a raise until `REPO_URL` is filled; the verbatim inline paste is a documented manual step because it duplicates ~120 lines of repo code that `tests/test_nb_format.py` already guards for the primary path.

- [ ] **Step 2: Verify nbformat validity**

Run: `python -c "import nbformat; nbformat.read('notebooks/qwen7b_full_finetune.ipynb', as_version=4); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: clone repo + import format helpers (inline fallback)"
```

---

### Task 4: Config cell (cell 3)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Produces: globals `MODE`, `BASE_MODEL`, `DATA_PATH`, `USE_REAL_DATA`, hyperparams (`EPOCHS`, `LR`, `MAX_SEQ_LEN`, `PER_DEVICE_BATCH`, `GRAD_ACCUM`, `SEED`), `SAVE_TARGET`, `QLORA_FALLBACK`, `OUTPUT_DIR`.

- [ ] **Step 1: Add cell 3**

```python
# --- Config: edit these ---
MODE = "direct"              # "direct" (Method 1) or "marker" (Method 2)
assert MODE in ("direct", "marker")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

USE_REAL_DATA = False        # False -> dummy smoke set; True -> your organizers.jsonl
DATA_PATH = (f"{REPO_DIR}/data/dummy.jsonl" if not USE_REAL_DATA
             else "data/organizers.jsonl")

EPOCHS = 3
LR = 1e-5                    # full FT wants a smaller LR than LoRA
MAX_SEQ_LEN = 2048
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
SEED = 42

SAVE_TARGET = "auto"        # "auto" -> Drive on Colab / working dir elsewhere; or a path
QLORA_FALLBACK = False      # True only if hardware can't do full FT (NOT full FT)
OUTPUT_DIR = f"ft_{MODE}"
print(dict(MODE=MODE, model=BASE_MODEL, data=DATA_PATH, qlora=QLORA_FALLBACK))
```

- [ ] **Step 2: Verify nbformat validity** (same command as Task 3 Step 2). Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: config cell (MODE switch, data toggle, hyperparams)"
```

---

### Task 5: Hardware detect → profile (cell 4)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Produces: `PROFILE` in {"full_single","full_zero3_offload","insufficient"}, `N_GPUS`, `TOTAL_VRAM_GB`, `CPU_RAM_GB`, `MULTI_PROC` (bool).

- [ ] **Step 1: Add cell 4**

```python
# --- Detect GPU/RAM, pick a real full-FT profile ---
import torch, psutil

N_GPUS = torch.cuda.device_count()
per_gpu = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(N_GPUS)]
TOTAL_VRAM_GB = sum(per_gpu)
MIN_GPU_GB = min(per_gpu) if per_gpu else 0.0
CPU_RAM_GB = psutil.virtual_memory().total / 1e9

MULTI_PROC = False
if TOTAL_VRAM_GB >= 48 and N_GPUS == 1:
    PROFILE = "full_single"
elif N_GPUS >= 1 and MIN_GPU_GB >= 22:          # 1x40GB single-proc ZeRO-3 offload
    PROFILE = "full_zero3_offload"
elif N_GPUS >= 2:                                # e.g. 2xT4: shard via notebook_launcher
    PROFILE, MULTI_PROC = "full_zero3_offload", True
else:
    PROFILE = "insufficient"

if PROFILE == "full_zero3_offload" and CPU_RAM_GB < 55:
    print(f"WARNING: ZeRO-3 CPU offload wants >=~60GB RAM, have {CPU_RAM_GB:.0f}GB — may OOM.")

if PROFILE == "insufficient" and not QLORA_FALLBACK:
    raise RuntimeError(
        f"Full FT of 7B needs more than {TOTAL_VRAM_GB:.0f}GB VRAM across {N_GPUS} GPU(s). "
        "Use an 80GB A100/H100, a 40GB card, or 2xT4 — or set QLORA_FALLBACK=True (NOT full FT)."
    )
print(dict(profile=PROFILE, gpus=N_GPUS, vram_gb=round(TOTAL_VRAM_GB), ram_gb=round(CPU_RAM_GB),
           multi_proc=MULTI_PROC, qlora=QLORA_FALLBACK))
```

- [ ] **Step 2: Verify nbformat validity.** Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: hardware detect + profile selection"
```

---

### Task 6: Data → split → chat dataset (cell 5)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Consumes: `load_jsonl`, `split_samples`, `build_sft_messages`, `MODE`, `DATA_PATH`, `SEED`.
- Produces: `train_ds`, `val_ds` (HF `Dataset` with a `messages` column), `test_samples` (list[RagSample]).

- [ ] **Step 1: Add cell 5**

```python
# --- Load, split (stratified by reliable, seed 42), build chat records ---
from datasets import Dataset

samples = load_jsonl(DATA_PATH)
train, val, test_samples = split_samples(samples, seed=SEED)
print(f"loaded {len(samples)} -> train={len(train)} val={len(val)} test={len(test_samples)}")

def _to_ds(rows):
    return Dataset.from_list([build_sft_messages(s, MODE) for s in rows])

train_ds, val_ds = _to_ds(train), _to_ds(val)

# In-notebook symmetry assertion (same guard as tests/test_nb_format.py)
from rag_reliability.formatting import build_chat_training_record
assert train_ds[0] == build_chat_training_record(train[0], MODE), "format drift!"
print("format symmetry OK. sample:", train_ds[0]["messages"][1]["content"])
```

- [ ] **Step 2: Verify nbformat validity.** Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: data load, split, chat-record dataset + symmetry assert"
```

---

### Task 7: Model + tokenizer per profile (cell 6)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Consumes: `BASE_MODEL`, `PROFILE`, `QLORA_FALLBACK`, `MULTI_PROC`.
- Produces: `load_model_and_tokenizer()` returning `(model, tokenizer)`. Defined as a function (not run inline) because ZeRO/`notebook_launcher` must construct the model inside the worker process.

- [ ] **Step 1: Add cell 6**

```python
# --- Model + tokenizer loader (called inside the train fn) ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=torch.bfloat16)
    if QLORA_FALLBACK:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    # ZeRO-3 sets device placement itself; full_single loads to the one GPU.
    if PROFILE == "full_single" and not QLORA_FALLBACK:
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    model.config.use_cache = False
    return model, tok

print("loader ready for profile:", PROFILE)
```

- [ ] **Step 2: Verify nbformat validity.** Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: profile-aware model+tokenizer loader"
```

---

### Task 8: DeepSpeed config + train function + launch (cell 7)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Consumes: `load_model_and_tokenizer`, `train_ds`, `val_ds`, hyperparams, `PROFILE`, `MULTI_PROC`, `OUTPUT_DIR`, `QLORA_FALLBACK`.
- Produces: side effect — a trained full-FT checkpoint saved to `OUTPUT_DIR/`.

- [ ] **Step 1: Add cell 7**

```python
# --- DeepSpeed ZeRO-3 offload config (used by both offload paths) ---
import json
ZERO3 = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True, "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": GRAD_ACCUM,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH,
    "gradient_clipping": 1.0,
}
with open("ds_zero3.json", "w") as f: json.dump(ZERO3, f, indent=2)

def train_fn():
    from trl import SFTConfig, SFTTrainer
    model, tok = load_model_and_tokenizer()
    if QLORA_FALLBACK:
        from peft import LoraConfig, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                              target_modules=["q_proj","k_proj","v_proj","o_proj",
                                              "gate_proj","up_proj","down_proj"])
    else:
        peft_cfg = None

    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_seq_length=MAX_SEQ_LEN,
        bf16=True,
        gradient_checkpointing=True,
        assistant_only_loss=True,      # == mlx --mask-prompt: loss on assistant tokens only
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=SEED,
        deepspeed=("ds_zero3.json" if PROFILE == "full_zero3_offload" and not QLORA_FALLBACK else None),
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=val_ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

# --- Launch ---
if MULTI_PROC:
    from accelerate import notebook_launcher
    notebook_launcher(train_fn, num_processes=N_GPUS)
else:
    train_fn()
print("training done ->", OUTPUT_DIR)
```

- [ ] **Step 2: Verify nbformat validity.** Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: DeepSpeed config + SFTTrainer train fn + launch"
```

---

### Task 9: Evaluate on test split (cell 8)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Consumes: `OUTPUT_DIR`, `test_samples`, `MODE`, `parse_prediction`, `M` (metrics), the repo prompt builders.
- Produces: `metrics_out` dict; prints per-method metrics + invalid-output rate.

- [ ] **Step 1: Add cell 8**

```python
# --- Evaluate: greedy generate on test, parse, score with repo metrics ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_reliability.prompts import build_direct_prompt, build_marker_prompt

build_prompt = build_direct_prompt if MODE == "direct" else build_marker_prompt
tok = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch.bfloat16,
                                             device_map="auto")
model.eval()

@torch.no_grad()
def generate(prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=64, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

preds = []
for s in test_samples:
    raw = generate(build_prompt(s))
    preds.append(parse_prediction(raw, s.id, expect_marker=(MODE == "marker")))

# metrics.evaluate_predictions(samples, predictions) -> EvaluationResult
# (confirmed signature; it already computes invalid_output_rate internally).
metrics_out = M.evaluate_predictions(test_samples, preds)
print(metrics_out.model_dump_json(indent=2) if hasattr(metrics_out, "model_dump_json") else metrics_out)
```

- [ ] **Step 2: Verify nbformat validity.** Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: eval cell (generate, parse, repo metrics, invalid rate)"
```

---

### Task 10: Save + next steps (cell 9)

**Files:**
- Modify: `notebooks/qwen7b_full_finetune.ipynb`

**Interfaces:**
- Consumes: `OUTPUT_DIR`, `SAVE_TARGET`, `PLATFORM`, `MODE`.
- Produces: side effect — checkpoint copied to the save location; prints the repo inference command.

- [ ] **Step 1: Add cell 9**

```python
# --- Save checkpoint + print how to run it through the repo pipeline ---
import shutil, os

def resolve_dest():
    if SAVE_TARGET != "auto":
        return SAVE_TARGET
    if PLATFORM == "colab":
        from google.colab import drive
        drive.mount("/content/drive")
        return f"/content/drive/MyDrive/{OUTPUT_DIR}"
    if PLATFORM == "kaggle":
        return f"/kaggle/working/{OUTPUT_DIR}"
    return os.path.abspath(OUTPUT_DIR)

dest = resolve_dest()
if os.path.abspath(dest) != os.path.abspath(OUTPUT_DIR):
    shutil.copytree(OUTPUT_DIR, dest, dirs_exist_ok=True)
print("saved to:", dest)

# Optional HF Hub push (off by default):
# from huggingface_hub import login; login(token="hf_..."); model.push_to_hub("you/qwen7b-judge-"+MODE)

print(f"""
Next — run this checkpoint through the repo pipeline:
  python scripts/infer.py --data data/dummy.jsonl --mode {MODE} \\
    --model {dest} --output results/{MODE}_ft_predictions.jsonl
  python scripts/evaluate.py --data data/dummy.jsonl \\
    --predictions results/{MODE}_ft_predictions.jsonl --output results/{MODE}_ft_metrics.json
""")
```

Implementer note: `scripts/infer.py` currently loads models through the mlx backend. A full-FT HF checkpoint runs directly in the notebook's eval cell; using it via `scripts/infer.py` on CUDA may need a transformers backend. Flag this in `notebooks/README.md` rather than modifying the CLI (out of scope per spec non-goals).

- [ ] **Step 2: Verify nbformat validity.** Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/qwen7b_full_finetune.ipynb
git commit -m "Notebook: save checkpoint + next-steps cell"
```

---

### Task 11: Docs — notebook README + main README pointer

**Files:**
- Create: `notebooks/README.md`
- Modify: `README.md`

- [ ] **Step 1: Write `notebooks/README.md`**

Cover: purpose (full FT of Qwen2.5-7B, Methods 1/2); how to open on Colab / Kaggle / DataSphere; the `MODE` and `USE_REAL_DATA` switches; the three hardware profiles and their VRAM/RAM needs; that `QLORA_FALLBACK` is NOT full FT; the `REPO_URL` must be set for the import cell; and the caveat that the saved HF checkpoint is evaluated in-notebook (repo `scripts/infer.py` uses the mlx backend).

- [ ] **Step 2: Add a pointer in `README.md`**

Add one row/line under the documentation map or Advanced/pipelines section:

```markdown
| [notebooks/qwen7b_full_finetune.ipynb](notebooks/README.md) | Full fine-tune Qwen2.5-7B on Method 1/2 in Colab/Kaggle/DataSphere (GPU cloud) |
```

- [ ] **Step 3: Verify the link resolves**

Run: `test -f notebooks/README.md && grep -q "qwen7b_full_finetune" README.md && echo ok`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/README.md README.md
git commit -m "Docs: notebook README + main README pointer"
```

---

## Self-review

**Spec coverage:**
- Full-FT-only + QLoRA escape hatch → Tasks 5, 7 (`QLORA_FALLBACK`). ✓
- Format symmetry (import builders, inline fallback) → Tasks 1, 3, 6 (+ drift test). ✓
- 3 hardware profiles, DeepSpeed first-class → Tasks 5, 8. ✓
- MODE switch not loop → Task 4. ✓
- split_samples stratified seed 42 → Task 6. ✓
- Eval via repo parse + metrics, invalid rate → Task 9. ✓
- Save to Drive/output, HF optional → Task 10. ✓
- Testing/verification (nbformat validity, format assert) → each notebook task + Task 1 pytest + Task 6 in-cell assert. ✓
- Notebook README / pointer → Task 11. ✓

**Placeholder scan:** One intentional, documented manual step remains — the inline fallback paste (Task 3), flagged with an implementer note (the repo-import path is the tested default). `REPO_URL` `<owner>` is a required user value, noted in Task 3 and README. Metrics API (`evaluate_predictions`, `EvaluationResult` with `invalid_output_rate`) confirmed against the repo — Task 9 uses the exact signature.

**Type consistency:** `build_sft_messages(sample, mode)` (Task 1) is the name imported in Tasks 3/6. `PROFILE` values, `MULTI_PROC`, `OUTPUT_DIR`, `train_ds/val_ds/test_samples` are named consistently across Tasks 4–10. `load_model_and_tokenizer()` (Task 7) is called only inside `train_fn` (Task 8), satisfying the notebook_launcher self-containment note.
