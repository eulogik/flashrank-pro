from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)


@dataclass
class FlashRankProConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 8192
    num_labels: int = 1
    torch_dtype: str = "float16"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


class FlashRankPro(nn.Module):
    def __init__(self, config: FlashRankProConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            torch_dtype=getattr(torch, config.torch_dtype),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

    def forward(self, query_doc_pairs):
        encoded = self.tokenizer(
            query_doc_pairs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**encoded)
        return outputs.logits.squeeze(-1)

    @torch.no_grad()
    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        formatted = [f"{q} {self.tokenizer.sep_token} {d}" if self.tokenizer.sep_token else f"{q} {d}" for q, d in pairs]
        logits = self.forward(formatted)
        scores = torch.sigmoid(logits).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores


class Reranker:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model = FlashRankPro(FlashRankProConfig(model_name=model_path))
        self.model.to(device)
        self.model.eval()

    def rerank(self, query: str, docs: list[str], top_k: Optional[int] = None) -> list[dict]:
        scores = self.model.score(query, docs)
        results = [{"text": doc, "score": score} for doc, score in zip(docs, scores)]
        results.sort(key=lambda x: x["score"], reverse=True)
        if top_k:
            results = results[:top_k]
        return results
