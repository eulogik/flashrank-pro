"""
Stage 2: Knowledge Distillation from teacher to ModernBERT student.

Trains a small ModernBERT-based reranker using soft labels from a
powerful teacher (Qwen3-Reranker-8B or mxbai-rerank-large-v2).

Loss: Hybrid Pointwise-MSE + Margin-MSE for ranking calibration.

Colab T4: ~3h for ModernBERT-base (full), ~3h for ModernBERT-large (LoRA).
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    HfArgumentParser,
    TrainingArguments,
)


class RerankingDataset(Dataset):
    """Each sample: query, positive doc, negative docs, teacher scores."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"query": s["query"], "positive": s["positive"], "negatives": s["negatives"], "teacher_scores": s["teacher_scores"]}


def collate_fn(batch, tokenizer, max_length):
    n_negs = len(batch[0]["negatives"])
    texts = []
    teacher_scores = []
    for b in batch:
        docs = [b["positive"]] + b["negatives"]
        teacher_scores.append(b["teacher_scores"])
        texts.extend(f"{b['query']} {tokenizer.sep_token or '[SEP]'} {d}" for d in docs)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "teacher_scores": torch.tensor(teacher_scores, dtype=torch.float32),
        "n_negs": n_negs,
    }


def hybrid_distillation_loss(student_logits, teacher_scores, n_negs, margin_beta=1.0):
    batch_size = student_logits.size(0)
    n_docs = n_negs + 1
    student_scores = student_logits.view(batch_size // n_docs, n_docs)
    teacher = teacher_scores.to(student_scores.device)

    if teacher.dim() == 3 and teacher.size(-1) == 2:
        teacher = teacher[..., -1]

    teacher = teacher.to(student_scores.dtype)
    pointwise_loss = F.mse_loss(student_scores, teacher)

    margin_loss = 0.0
    count = 0
    for i in range(teacher.size(0)):
        pos_score = teacher[i, 0]
        for j in range(1, n_docs):
            teacher_margin = pos_score - teacher[i, j]
            student_margin = student_scores[i, 0] - student_scores[i, j]
            margin_loss += F.mse_loss(student_margin, teacher_margin)
            count += 1
    margin_loss = margin_loss / max(count, 1)

    return pointwise_loss + margin_beta * margin_loss


def find_latest_checkpoint(output_dir: str) -> Optional[int]:
    """Find the latest completed epoch checkpoint."""
    if not os.path.exists(output_dir):
        return None
    epochs = []
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-epoch-"):
            try:
                epochs.append(int(d.split("-")[-1]))
            except ValueError:
                pass
    return max(epochs) if epochs else None


def save_checkpoint(accelerator, output_dir, epoch):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        print(f"Checkpoint saved: {ckpt_dir}")


def main(
    model_name: str = "answerdotai/ModernBERT-base",
    data_path: str = "data/synthetic_training_data.jsonl",
    output_dir: str = "models/flashrank-pro-base",
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    num_epochs: int = 3,
    max_length: int = 512,
    margin_beta: float = 1.0,
    use_lora: bool = False,
    lora_r: int = 16,
    use_wandb: bool = False,
):
    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.float32,
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["Wqkv", "Wo", "Wi"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = RerankingDataset(data_path, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    resume_epoch = find_latest_checkpoint(output_dir)
    start_epoch = 0
    if resume_epoch is not None:
        ckpt = os.path.join(output_dir, f"checkpoint-epoch-{resume_epoch}")
        accelerator.load_state(ckpt)
        start_epoch = resume_epoch
        if accelerator.is_main_process:
            print(f"Resumed from epoch {resume_epoch} checkpoint ({ckpt})")

    bad = [n for n, p in model.named_parameters() if torch.isnan(p).any() or torch.isinf(p).any()]
    if bad and accelerator.is_main_process:
        print(f"WARNING: {len(bad)} params contain NaN/inf at start (e.g. {bad[:3]})")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Training {model_name} | Batch {batch_size} | LR {learning_rate} | Epochs {num_epochs}")
        print(f"Total steps: {total_steps} | Samples: {len(dataset)}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.squeeze(-1)
            loss = hybrid_distillation_loss(logits, batch["teacher_scores"], batch["n_negs"], margin_beta)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 100 == 0 and accelerator.is_main_process:
                print(f"Step {step}/{total_steps} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        save_checkpoint(accelerator, output_dir, epoch + 1)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in unwrapped.parameters()):
            print("ERROR: Final model has NaN/inf weights — NOT saving. Check loss/inputs.")
        else:
            unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
