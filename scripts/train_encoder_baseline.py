#!/usr/bin/env python
"""Train a supervised encoder baseline for binary answer reliability.

This is the organizer notebook converted into a reproducible CLI:
RagSample jsonl -> tokenizer -> encoder + binary classification head -> metrics.
Heavy ML dependencies are imported lazily so unit tests can cover preprocessing
without requiring torch/transformers/datasets.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.schema import Prediction, RagSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/organizers.jsonl", help="Input RagSample jsonl")
    parser.add_argument("--output", default="results/encoder_baseline_metrics.json")
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Optional standard Prediction JSONL for the held-out test split",
    )
    parser.add_argument("--model", default="deepvk/RuModernBERT-base")
    parser.add_argument("--output-dir", default="results/encoder_checkpoints")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--train-data", default=None, help="Explicit train split (bypasses internal split)")
    parser.add_argument("--val-data", default=None, help="Explicit validation split")
    parser.add_argument("--test-data", default=None, help="Explicit test split; scores these exact rows")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pos-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-threshold-tuning", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_encoder_text(sample: RagSample) -> str:
    """Build the encoder input used by the organizer baseline."""
    return f"dialog: {sample.question}\nanswer: {sample.answer}\ncontext: {sample.context}"


def reliability_labels(samples: list[RagSample]) -> list[int]:
    return [sample.reliable for sample in samples]


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_array >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true_array, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, y_prob_array))
        if len(set(y_true_array.tolist())) > 1
        else float("nan"),
    }


def find_best_threshold(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        if best_metrics is None or metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def split_samples(
    samples: list[RagSample],
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[list[RagSample], list[RagSample], list[RagSample]]:
    labels = reliability_labels(samples)
    train_validation_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_validation_labels = reliability_labels(train_validation_samples)
    relative_validation_size = validation_size / (1.0 - test_size)
    train_samples, validation_samples = train_test_split(
        train_validation_samples,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=train_validation_labels,
    )
    return train_samples, validation_samples, test_samples


def compute_pos_weight(labels: list[int], mode: str) -> float:
    if mode == "none":
        return 1.0
    positives = sum(labels)
    return (len(labels) - positives) / max(positives, 1)


def predictions_from_probabilities(
    samples: list[RagSample],
    probabilities: list[float] | np.ndarray,
    threshold: float,
) -> list[Prediction]:
    """Export encoder reliability scores as standard Prediction records.

    The encoder is a binary reliability classifier, not a separate
    faithfulness/relevance classifier. For the shared evaluator we map
    reliable=1 to (faithfulness=1, relevance=1), and reliable=0 to (0, 0).
    """
    predictions: list[Prediction] = []
    for sample, probability in zip(samples, probabilities, strict=True):
        probability_float = float(probability)
        reliable_pred = int(probability_float >= threshold)
        predictions.append(
            Prediction(
                id=sample.id,
                faithfulness_pred=reliable_pred,
                relevance_pred=reliable_pred,
                raw_output=(
                    f"encoder_probability={probability_float:.6f}; "
                    f"threshold={threshold:.6f}"
                ),
            )
        )
    return predictions


def set_seed(seed: int) -> None:
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dataset(samples: list[RagSample], tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset  # noqa: PLC0415

    rows = {
        "text": [build_encoder_text(sample) for sample in samples],
        "labels": reliability_labels(samples),
    }
    dataset = Dataset.from_dict(rows)
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )
    return dataset.remove_columns("text")


def build_model(model_name: str, pos_weight: float) -> Any:
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    def mean_pool(hidden: Any, mask: Any) -> Any:
        expanded_mask = mask.unsqueeze(-1).float()
        return (hidden * expanded_mask).sum(1) / expanded_mask.sum(1).clamp(min=1e-9)

    class Classifier(nn.Module):
        def __init__(self, name: str, weight: float) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name, trust_remote_code=True)
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            labels: Any | None = None,
        ) -> dict[str, Any]:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.head(mean_pool(output.last_hidden_state, attention_mask)).squeeze(-1)
            loss = self.loss_fn(logits, labels.float()) if labels is not None else None
            return {"loss": loss, "logits": logits}

    return Classifier(model_name, pos_weight)


def train_and_evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments  # noqa: PLC0415

    set_seed(args.seed)
    if args.test_data:
        train_samples = load_jsonl(args.train_data)
        validation_samples = load_jsonl(args.val_data)
        test_samples = load_jsonl(args.test_data)
    else:
        samples = load_jsonl(args.data)
        train_samples, validation_samples, test_samples = split_samples(
            samples,
            test_size=args.test_size,
            validation_size=args.validation_size,
            seed=args.seed,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    train_ds = make_dataset(train_samples, tokenizer, args.max_length)
    validation_ds = make_dataset(validation_samples, tokenizer, args.max_length)
    test_ds = make_dataset(test_samples, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer)

    train_labels = reliability_labels(train_samples)
    pos_weight = compute_pos_weight(train_labels, args.pos_weight_mode)
    model = build_model(args.model, pos_weight)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=collator,
    )
    trainer.train()

    threshold = args.threshold
    validation_metrics: dict[str, float] | None = None
    if not args.no_threshold_tuning:
        validation_logits = trainer.predict(validation_ds).predictions
        validation_prob = torch.sigmoid(torch.tensor(validation_logits)).numpy()
        validation_true = np.asarray(validation_ds["labels"], dtype=int)
        threshold, validation_metrics = find_best_threshold(validation_true, validation_prob)

    logits = trainer.predict(test_ds).predictions
    y_prob = torch.sigmoid(torch.tensor(logits)).numpy()
    y_true = np.asarray(test_ds["labels"], dtype=int)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    if args.predictions_output is not None:
        save_jsonl(
            predictions_from_probabilities(test_samples, y_prob, threshold=threshold),
            args.predictions_output,
        )
    metrics.update(
        {
            "model": args.model,
            "train_total": len(train_samples),
            "validation_total": len(validation_samples),
            "test_total": len(test_samples),
            "threshold": threshold,
            "threshold_tuned": not args.no_threshold_tuning,
            "pos_weight": pos_weight,
            "pos_weight_mode": args.pos_weight_mode,
            "max_length": args.max_length,
        }
    )
    if validation_metrics is not None:
        metrics["validation_f1_macro"] = validation_metrics["f1_macro"]
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(args)
    rendered = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(rendered)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
