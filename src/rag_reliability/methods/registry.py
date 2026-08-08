"""Single source of truth for reliability methods: metadata + CLI command builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CommandContext:
    data: Path
    run_dir: Path
    predictions_path: Path
    python: str = "python"

    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    max_tokens: int = 64

    direct_adapter_path: str = "results/adapters_direct"
    marker_adapter_path: str = "results/adapters_marker"

    lettucedetect_model: str = "results/lettucedetect/classifier.joblib"

    encoder_model: str = "deepvk/RuModernBERT-base"
    encoder_output_dir: str | None = None
    encoder_max_length: int = 512
    encoder_batch_size: int = 4
    encoder_epochs: float = 3
    encoder_learning_rate: float = 2e-5
    encoder_pos_weight_mode: str = "none"

    m3_backend: str = "mlx"
    m3_max_tokens: int = 400
    m3_max_context_chars: int | None = None
    m3_examples: str = "configs/few_shot.yaml"
    m3_prompt_file: str = "artifacts/m3_optimized_prompt.txt"
    m3_api_base: str = "http://localhost:8000/v1"
    m3_api_key_env: str = "OPENAI_API_KEY"
    m3_cache_dir: str = "results/m3/cache"
    m3_concurrency: int = 1

    m6_features: str = "results/m6/features.jsonl"
    m6_contradiction_threshold: float = 0.5
    m6_entropy_threshold: float = 1.0
    m6_relevance_threshold: float = 0.25

    independent_faithfulness_threshold: float = 0.20
    independent_relevance_threshold: float = 0.10

    independent_v2_model: str = "results/independent_v2/model.joblib"

    limit: int | None = None


BuildCommand = Callable[[CommandContext], list[str]]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    family: str
    mode: str | None
    build_command: BuildCommand
    demo_runner: str | None
    requires: tuple[str, ...] = field(default_factory=tuple)


def _maybe_limit(
    command: list[str],
    ctx: CommandContext,
) -> list[str]:
    if ctx.limit is not None:
        command.extend(["--limit", str(ctx.limit)])

    return command


def _dummy(mode: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        return _maybe_limit(
            [
                ctx.python,
                "scripts/run_prompt_baseline.py",
                "--data",
                str(ctx.data),
                "--output",
                str(ctx.predictions_path),
                "--mode",
                mode,
                "--backend",
                "dummy",
                "--dummy-strategy",
                "keyword" if mode == "marker" else "always_reliable",
            ],
            ctx,
        )

    return build


def _prompt(mode: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        return _maybe_limit(
            [
                ctx.python,
                "scripts/run_prompt_baseline.py",
                "--data",
                str(ctx.data),
                "--output",
                str(ctx.predictions_path),
                "--mode",
                mode,
                "--backend",
                "mlx",
                "--model",
                ctx.model,
                "--max-tokens",
                str(ctx.max_tokens),
            ],
            ctx,
        )

    return build


def _lora(mode: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        adapter = (
            ctx.direct_adapter_path
            if mode == "direct"
            else ctx.marker_adapter_path
        )

        return [
            ctx.python,
            "scripts/infer.py",
            "--data",
            str(ctx.data),
            "--output",
            str(ctx.predictions_path),
            "--mode",
            mode,
            "--model",
            ctx.model,
            "--adapter-path",
            adapter,
            "--max-tokens",
            str(ctx.max_tokens),
        ]

    return build


def _lettucedetect(ctx: CommandContext) -> list[str]:
    return [
        ctx.python,
        "scripts/infer_lettucedetect.py",
        "--data",
        str(ctx.data),
        "--model",
        ctx.lettucedetect_model,
        "--output",
        str(ctx.predictions_path),
    ]


def _encoder(ctx: CommandContext) -> list[str]:
    checkpoint_dir = (
        ctx.encoder_output_dir
        or str(ctx.run_dir / "checkpoints")
    )

    return [
        ctx.python,
        "scripts/train_encoder_baseline.py",
        "--data",
        str(ctx.data),
        "--output",
        str(ctx.run_dir / "encoder_binary_metrics.json"),
        "--predictions-output",
        str(ctx.predictions_path),
        "--model",
        ctx.encoder_model,
        "--output-dir",
        checkpoint_dir,
        "--max-length",
        str(ctx.encoder_max_length),
        "--batch-size",
        str(ctx.encoder_batch_size),
        "--epochs",
        str(ctx.encoder_epochs),
        "--learning-rate",
        str(ctx.encoder_learning_rate),
        "--pos-weight-mode",
        ctx.encoder_pos_weight_mode,
    ]


def _m3(name: str) -> BuildCommand:
    def build(ctx: CommandContext) -> list[str]:
        if name in ("m3_openai", "m3_openai_judge"):
            m3_mode = "zero_shot"
            backend = (
                "openai"
                if name == "m3_openai"
                else "openai_judge"
            )
        else:
            m3_mode = name.removeprefix("m3_")
            backend = ctx.m3_backend

        command = [
            ctx.python,
            "scripts/run_m3.py",
            "--data",
            str(ctx.data),
            "--output",
            str(ctx.predictions_path),
            "--mode",
            m3_mode,
            "--backend",
            backend,
            "--model",
            ctx.model,
            "--max-tokens",
            str(ctx.m3_max_tokens),
        ]

        if name in ("m3_openai", "m3_openai_judge"):
            command.extend(
                [
                    "--api-base",
                    ctx.m3_api_base,
                    "--api-key-env",
                    ctx.m3_api_key_env,
                    "--cache-dir",
                    ctx.m3_cache_dir,
                ]
            )

        if name == "m3_openai_judge":
            command.extend(
                [
                    "--concurrency",
                    str(ctx.m3_concurrency),
                ]
            )

        if name == "m3_few_shot":
            command.extend(
                [
                    "--examples",
                    ctx.m3_examples,
                ]
            )

        elif name == "m3_gepa":
            command.extend(
                [
                    "--prompt-file",
                    ctx.m3_prompt_file,
                ]
            )

        if ctx.m3_max_context_chars is not None:
            command.extend(
                [
                    "--max-context-chars",
                    str(ctx.m3_max_context_chars),
                ]
            )

        return _maybe_limit(command, ctx)

    return build


def _m6(ctx: CommandContext) -> list[str]:
    return _maybe_limit(
        [
            ctx.python,
            "scripts/run_m6_selfcheck.py",
            "--data",
            str(ctx.data),
            "--features",
            ctx.m6_features,
            "--output",
            str(ctx.predictions_path),
            "--contradiction-threshold",
            str(ctx.m6_contradiction_threshold),
            "--entropy-threshold",
            str(ctx.m6_entropy_threshold),
            "--relevance-threshold",
            str(ctx.m6_relevance_threshold),
        ],
        ctx,
    )


def _independent(ctx: CommandContext) -> list[str]:
    return _maybe_limit(
        [
            ctx.python,
            "scripts/run_independent.py",
            "--data",
            str(ctx.data),
            "--output",
            str(ctx.predictions_path),
            "--faithfulness-threshold",
            str(ctx.independent_faithfulness_threshold),
            "--relevance-threshold",
            str(ctx.independent_relevance_threshold),
        ],
        ctx,
    )


def _independent_v2(ctx: CommandContext) -> list[str]:
    return _maybe_limit(
        [
            ctx.python,
            "scripts/run_independent_v2.py",
            "--data",
            str(ctx.data),
            "--model",
            ctx.independent_v2_model,
            "--output",
            str(ctx.predictions_path),
        ],
        ctx,
    )


METHODS: dict[str, MethodSpec] = {
    "dummy_direct": MethodSpec(
        "dummy_direct",
        "Dummy — direct",
        "dummy",
        "direct",
        _dummy("direct"),
        "dummy",
    ),
    "dummy_marker": MethodSpec(
        "dummy_marker",
        "Dummy — marker",
        "dummy",
        "marker",
        _dummy("marker"),
        "dummy",
    ),
    "prompt_direct": MethodSpec(
        "prompt_direct",
        "Zero-shot prompt — direct",
        "prompt",
        "direct",
        _prompt("direct"),
        "prompt",
        ("MLX model",),
    ),
    "prompt_marker": MethodSpec(
        "prompt_marker",
        "Zero-shot prompt — marker",
        "prompt",
        "marker",
        _prompt("marker"),
        "prompt",
        ("MLX model",),
    ),
    "lora_direct": MethodSpec(
        "lora_direct",
        "LoRA — direct",
        "lora",
        "direct",
        _lora("direct"),
        "lora",
        ("results/adapters_direct",),
    ),
    "lora_marker": MethodSpec(
        "lora_marker",
        "LoRA — marker",
        "lora",
        "marker",
        _lora("marker"),
        "lora",
        ("results/adapters_marker",),
    ),
    "lettucedetect": MethodSpec(
        "lettucedetect",
        "LettuceDetect features",
        "lettucedetect",
        None,
        _lettucedetect,
        "lettucedetect",
        ("results/lettucedetect/classifier.joblib",),
    ),
    "encoder": MethodSpec(
        "encoder",
        "RuModernBERT encoder",
        "encoder",
        None,
        _encoder,
        "encoder",
        ("results/encoder_checkpoints_512_best",),
    ),
    "m3_zero_shot": MethodSpec(
        "m3_zero_shot",
        "Method 3 — zero-shot judge",
        "m3",
        None,
        _m3("m3_zero_shot"),
        "m3",
        ("MLX model",),
    ),
    "m3_few_shot": MethodSpec(
        "m3_few_shot",
        "Method 3 — few-shot judge",
        "m3",
        None,
        _m3("m3_few_shot"),
        "m3",
        ("configs/few_shot.yaml",),
    ),
    "m3_gepa": MethodSpec(
        "m3_gepa",
        "Method 3 — GEPA prompt",
        "m3",
        None,
        _m3("m3_gepa"),
        None,
        ("evolved prompt file",),
    ),
    "m3_openai": MethodSpec(
        "m3_openai",
        "Method 3 — OpenAI endpoint",
        "m3",
        None,
        _m3("m3_openai"),
        None,
        ("OpenAI-compatible endpoint",),
    ),
    "m3_openai_judge": MethodSpec(
        "m3_openai_judge",
        "Method 3 — OpenAI logprob judge",
        "m3",
        None,
        _m3("m3_openai_judge"),
        None,
        ("OpenAI-compatible endpoint",),
    ),
    "m6_selfcheck": MethodSpec(
        "m6_selfcheck",
        "Method 6 — SelfCheck features",
        "m6",
        None,
        _m6,
        None,
        ("results/m6/features.jsonl",),
    ),
    "independent": MethodSpec(
        "independent",
        "Independent rule-based evaluator",
        "independent",
        None,
        _independent,
        "independent",
    ),
    "independent_v2": MethodSpec(
        "independent_v2",
        "Independent evaluator V2 — learned features",
        "independent",
        None,
        _independent_v2,
        None,
        ("results/independent_v2/model.joblib",),
    ),
}


def all_method_names() -> tuple[str, ...]:
    return tuple(METHODS)


def get(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown method {name!r}; "
            f"available: {', '.join(METHODS)}"
        ) from exc


def resolve_names(raw: str) -> list[str]:
    if raw.strip() == "all":
        return list(all_method_names())

    names = [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]

    unknown = [
        name
        for name in names
        if name not in METHODS
    ]

    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. "
            f"Available: {list(METHODS)}"
        )

    return names