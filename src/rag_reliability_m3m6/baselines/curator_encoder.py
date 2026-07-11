"""Воспроизведение бейзлайна кураторов (data/baseline.ipynb) как скрипта.

Рецепт кураторов: RuModernBERT-base + mean-pool + одна голова на `reliable`
(= faith AND rel), BCEWithLogitsLoss с pos_weight, MAX_LENGTH 8096, batch 8,
2 эпохи, lr 2e-5, weight_decay 0.01, seed 42, порог 0.5 на sigmoid.
Их числа на их 4782 строках: f1-macro 0.62, f1(reliable) 0.705, ROC-AUC 0.693;
наши числа на 2245 строках будут другими — фиксируем НАШЕ воспроизведение.

Отличия от ноутбука — только вынужденные:
- у кураторов вход строился из колонки `full_query`, которой у нас нет;
  подставляем извлечённую последнюю реплику клиента (case.query) —
  вопрос кураторам о содержимом `full_query` открыт;
- их sklearn-сплит train_test_split(test_size=0.2, random_state=42,
  stratify=reliable) уже материализован платформенным кодом в
  data/processed/alfa_curator_{train,test}.jsonl — читаем готовые файлы;
- Trainer заменён на эквивалентный plain-torch цикл (AdamW с исключением
  bias/LayerNorm из weight decay, линейный lr-спад без warmup, clip 1.0).

Статус: на этой машине нет GPU — здесь выполняется только CPU-smoke
(--smoke: подвыборка + короткий max_length), полный прогон — позже на GPU:
    python scripts/m3m6/train_encoder_baseline.py --config configs/config.yaml

CLI: python scripts/m3m6/train_encoder_baseline.py --config configs/config.yaml
     [--smoke] [--limit N] [--max-length M] [--epochs E] [--batch-size B]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from rag_reliability_m3m6.common.config import load_config
from rag_reliability_m3m6.common.run_meta import save_run_yaml
from rag_reliability_m3m6.common.schemas import Case, Pred, load_cases, save_preds

# Выбор модели — ИХ решение из ноутбука, не наш конфиг; поэтому константа.
MODEL_NAME = "deepvk/RuModernBERT-base"

# Гиперпараметры кураторов (baseline.ipynb, ячейка Config).
MAX_LENGTH = 8096
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

# Сплит кураторов, материализованный платформой; имена фиксированы.
_TRAIN_PATH = Path("data/processed/alfa_curator_train.jsonl")
_TEST_PATH = Path("data/processed/alfa_curator_test.jsonl")
_OUT_DIR = Path("predictions/local/baselines/curator_encoder")


def build_input(case: Case) -> str:
    """Текст входа как у кураторов: query/answer/context, чанки через \\n\\n.

    В ноутбуке query = full_query; у нас — последняя реплика клиента.
    """
    chunks = "\n\n".join(t for t in (c.strip() for c in case.context) if t)
    return f"query: {case.query.strip()}\nanswer: {case.answer.strip()}\ncontext: {chunks}"


def subsample(cases: list[Case], limit: int | None, seed: int) -> list[Case]:
    """Детерминированная случайная подвыборка с сохранением исходного порядка."""
    if limit is None or limit >= len(cases):
        return cases
    idx = sorted(random.Random(seed).sample(range(len(cases)), limit))
    return [cases[i] for i in idx]


def _labels(cases: list[Case]) -> list[int]:
    ys = []
    for c in cases:
        if c.reliable is None:
            raise ValueError(f"кейс {c.id} без меток faith/rel — сплит должен быть размечен")
        ys.append(c.reliable)
    return ys


def _set_seed(seed: int) -> None:
    """set_seed из ноутбука (без cudnn-флагов на CPU — они бесплатны, ставим всегда)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_model(pos_weight: float):  # noqa: ANN202 — тип требует torch, импорт ленивый
    """Classifier из ноутбука: encoder + Dropout(0.1) + Linear(hidden, 1)."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    class Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.head = nn.Sequential(
                nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1)
            )
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        def forward(self, input_ids, attention_mask, labels=None):  # noqa: ANN001, ANN201
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.head(mean_pool(out.last_hidden_state, attention_mask)).squeeze(-1)
            loss = self.loss_fn(logits, labels.float()) if labels is not None else None
            return loss, logits

    return Classifier()


def train_and_predict(
    train: list[Case],
    test: list[Case],
    max_length: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> list[float]:
    """Обучение по рецепту кураторов; возвращает p(reliable) для test в исходном порядке."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def encode(cases: list[Case]) -> list[dict]:
        y = _labels(cases)
        rows = []
        for c, label in zip(cases, y):
            enc = tokenizer(build_input(c), truncation=True, max_length=max_length)
            rows.append({"input_ids": enc["input_ids"], "labels": label})
        return rows

    def collate(batch: list[dict]) -> dict:
        # динамический паддинг по батчу, как DataCollatorWithPadding в ноутбуке
        padded = tokenizer.pad([{"input_ids": r["input_ids"]} for r in batch], return_tensors="pt")
        padded["labels"] = torch.tensor([r["labels"] for r in batch])
        return padded

    y_train = _labels(train)
    pos = sum(y_train)
    pos_weight = (len(y_train) - pos) / max(pos, 1)
    model = _build_model(pos_weight).to(device)
    # bf16 на cuda — через autocast (веса fp32, GradScaler не нужен)
    from contextlib import nullcontext

    def amp_ctx():  # noqa: ANN202
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        encode(train), batch_size=batch_size, shuffle=True, generator=gen, collate_fn=collate
    )
    test_loader = DataLoader(encode(test), batch_size=batch_size, collate_fn=collate)

    # Как Trainer: AdamW, decay не применяется к bias и параметрам нормализации.
    no_decay = [p for n, p in model.named_parameters() if n.endswith("bias") or "norm" in n.lower()]
    no_decay_ids = {id(p) for p in no_decay}
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, (total_steps - step) / total_steps)
    )

    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_ctx():
                loss, _ = model(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(f"  step {step}/{total_steps} loss={loss.item():.4f}", flush=True)
        mean_loss = epoch_loss / max(1, len(train_loader))
        print(f"эпоха {epoch + 1}/{epochs}: train loss={mean_loss:.4f}")

    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_ctx():
                _, logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
            probs.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return probs


def compute_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    """Метрики кураторов + f1_macro; порог 0.5, как в ноутбуке."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = [int(p >= threshold) for p in y_prob]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Бейзлайн кураторов: RuModernBERT → reliable")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="подвыборка train/test до N")
    parser.add_argument("--max-length", type=int, default=None, help=f"override {MAX_LENGTH}")
    parser.add_argument("--epochs", type=int, default=None, help=f"override {EPOCHS}")
    parser.add_argument("--batch-size", type=int, default=None, help=f"override {BATCH_SIZE}")
    parser.add_argument(
        "--smoke", action="store_true", help="CPU-smoke: --limit 200 --max-length 1024 --epochs 1"
    )
    args = parser.parse_args()

    limit = args.limit if args.limit is not None else (200 if args.smoke else None)
    smoke_len = 1024 if args.smoke else MAX_LENGTH
    max_length = args.max_length if args.max_length is not None else smoke_len
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke else EPOCHS)
    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE

    cfg = load_config(args.config)
    train = subsample(load_cases(_TRAIN_PATH), limit, SEED)
    test = subsample(load_cases(_TEST_PATH), limit, SEED)
    print(
        f"curator_encoder: train={len(train)} test={len(test)} "
        f"max_length={max_length} epochs={epochs} batch={batch_size} smoke={args.smoke}"
    )

    t0 = time.time()
    probs = train_and_predict(train, test, max_length, batch_size, epochs, SEED)
    wall = time.time() - t0

    meta = {"variant": "curator_encoder", "heads": "joint", "note": "p_faith=p_rel=p_reliable"}
    preds = [Pred(id=c.id, p_faith=p, p_rel=p, meta=dict(meta)) for c, p in zip(test, probs)]
    save_preds(preds, _OUT_DIR / "test.jsonl")

    metrics = compute_metrics(_labels(test), probs)
    metrics.update({"n_train": len(train), "n_test": len(test), "wall_time_sec": round(wall, 1)})
    (_OUT_DIR / "curator_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_run_yaml(
        _OUT_DIR,
        cfg,
        split="curator_test",
        method="baseline",
        variant="curator_encoder",
        model_name=MODEL_NAME,
        smoke=args.smoke,
        limit=limit,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        seed=SEED,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wall time: {wall:.1f}s; predictions: {_OUT_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    main()
