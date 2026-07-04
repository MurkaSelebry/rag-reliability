"""Shared MLX loading/generation used by run_prompt_baseline.py and infer.py.

Kept in one place so training and evaluation cannot drift apart in how they
wrap prompts (chat template) or load the model.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

INSTALL_HINT = (
    "ERROR: mlx-lm is not installed. Install it with:\n"
    '    uv pip install -e ".[mlx]"'
)


def make_generate_fn(
    model_name: str,
    max_tokens: int,
    adapter_path: str | None = None,
) -> Callable[[str], str]:
    """Load an MLX model (optionally with a LoRA adapter) and return prompt -> text.

    Prompts are wrapped in the tokenizer chat template as a single user turn —
    the same shape mlx_lm's CompletionsDataset uses for training records.
    Exits with a clear message when mlx-lm is missing.
    """
    try:
        from mlx_lm import generate, load  # noqa: PLC0415
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        sys.exit(1)

    print(f"Loading MLX model {model_name} (adapter: {adapter_path or 'none'}) ...")
    model, tokenizer = load(model_name, adapter_path=adapter_path)

    def generate_fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens)

    return generate_fn
