"""Батчевый NLI-скорер на мультиязычной mDeBERTa (XNLI).

Даёт P(entailment) и P(contradiction) для пар (premise, hypothesis).
Используется и для SelfCheck-NLI, и для кластеризации semantic entropy.

Длинные премисы (чанки корпуса ~6k символов) не влезают в max_length и раньше
молча обрезались. Теперь премиса режется на перекрывающиеся окна, скор пары —
max по окнам («сигнал есть хотя бы в одном окне»).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Минимальный бюджет токенов на премису: ниже него окна вырождаются,
# оставляем одно усечённое окно.
_MIN_BUDGET = 32
# Запас под спецтокены пары ([CLS]/[SEP] и т.п.).
_SPECIAL_TOKENS_MARGIN = 4


def split_tokens(tokens: list, budget: int, overlap: int) -> list[list]:
    """Режет список токенов на окна длины <= budget с шагом budget - overlap.

    Последнее окно всегда покрывает хвост. Если токены влезают целиком —
    одно окно без изменений. Чистая функция, torch не нужен.
    """
    if len(tokens) <= budget:
        return [tokens]
    stride = max(budget - overlap, 1)
    windows: list[list] = []
    for start in range(0, len(tokens), stride):
        windows.append(tokens[start : start + budget])
        if start + budget >= len(tokens):
            break
    return windows


def aggregate_windows(scores: list[dict], groups: list[int], n_pairs: int) -> list[dict]:
    """Схлопывает скоры окон в скор пары: max по окнам для entail и contra.

    groups[i] — индекс исходной пары для i-го окна. Порядок и длина выхода
    соответствуют исходным парам. Чистая функция, torch не нужен.
    """
    agg = [{"entail": 0.0, "contra": 0.0} for _ in range(n_pairs)]
    for s, g in zip(scores, groups):
        agg[g]["entail"] = max(agg[g]["entail"], s["entail"])
        agg[g]["contra"] = max(agg[g]["contra"], s["contra"])
    return agg


class NLIScorer:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
        overlap: int = 128,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            )
            .to(self.device)
            .eval()
        )
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, l in id2label.items() if "entail" in l)
        self.i_con = next(i for i, l in id2label.items() if "contra" in l)
        self.bs, self.max_length = batch_size, max_length
        self.overlap = overlap

    def _split_premise(self, premise: str, hypothesis: str) -> list[str]:
        """Окна премисы так, чтобы окно + гипотеза + спецтокены влезали в max_length."""
        prem_ids = self.tok(premise, add_special_tokens=False)["input_ids"]
        hyp_ids = self.tok(hypothesis, add_special_tokens=False)["input_ids"]
        budget = self.max_length - len(hyp_ids) - _SPECIAL_TOKENS_MARGIN
        if budget <= _MIN_BUDGET:
            # гипотеза съела почти весь контекст — одно усечённое окно
            if len(prem_ids) <= _MIN_BUDGET:
                return [premise]
            return [self.tok.decode(prem_ids[:_MIN_BUDGET])]
        if len(prem_ids) <= budget:
            return [premise]
        return [self.tok.decode(w) for w in split_tokens(prem_ids, budget, self.overlap)]

    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """pairs: (premise, hypothesis) -> [{'entail': p, 'contra': p}]

        Длинная премиса скорится по окнам с перекрытием; агрегат — max по окнам.
        Выход: ровно один dict на входную пару, порядок сохранён.
        """
        expanded: list[tuple[str, str]] = []
        groups: list[int] = []
        for idx, (prem, hyp) in enumerate(pairs):
            for window in self._split_premise(prem, hyp):
                expanded.append((window, hyp))
                groups.append(idx)
        scores: list[dict] = []
        for i in range(0, len(expanded), self.bs):
            batch = expanded[i : i + self.bs]
            enc = self.tok(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
            for row in probs:
                scores.append({"entail": row[self.i_ent].item(), "contra": row[self.i_con].item()})
        return aggregate_windows(scores, groups, len(pairs))
