"""Батчевый NLI-скорер на мультиязычной mDeBERTa (XNLI).

Даёт P(entailment) и P(contradiction) для пар (premise, hypothesis).
Используется и для SelfCheck-NLI, и для кластеризации semantic entropy.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIScorer:
    def __init__(self, model_name: str, device: str | None = None,
                 batch_size: int = 64, max_length: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, l in id2label.items() if "entail" in l)
        self.i_con = next(i for i, l in id2label.items() if "contra" in l)
        self.bs, self.max_length = batch_size, max_length

    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """pairs: (premise, hypothesis) -> [{'entail': p, 'contra': p}]"""
        out = []
        for i in range(0, len(pairs), self.bs):
            batch = pairs[i:i + self.bs]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch],
                           truncation=True, max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
            for row in probs:
                out.append({"entail": row[self.i_ent].item(),
                            "contra": row[self.i_con].item()})
        return out
