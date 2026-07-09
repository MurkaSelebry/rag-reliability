"""Batch NLI scorer used by Method 6 feature preparation."""

from __future__ import annotations


class NLIScorer:
    """Thin wrapper around a multilingual sequence-classification NLI model."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        id2label = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
        self.entail_index = next(index for index, label in id2label.items() if "entail" in label)
        self.contra_index = next(index for index, label in id2label.items() if "contra" in label)
        self.batch_size = batch_size
        self.max_length = max_length

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        output = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probabilities = self.torch.softmax(self.model(**encoded).logits.float(), dim=-1)
                for row in probabilities:
                    output.append(
                        {
                            "entail": float(row[self.entail_index].item()),
                            "contra": float(row[self.contra_index].item()),
                        }
                    )
        return output
