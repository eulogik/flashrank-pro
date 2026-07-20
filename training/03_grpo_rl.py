"""
Stage 3: GRPO Reinforcement Learning for fine-grained scoring.

GRPO (ProRank paper): group-relative policy optimization.
- Each query has K docs scored: pairwise ranking reward vs teacher
- Relative reward normalization
- KL divergence against reference model to prevent collapse

Runs as LoRA on top of Stage 2 model.
Colab T4: ~2h.
"""

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False


class RLDataset(Dataset):
    def __init__(self, data_path, k_samples=8):
        self.examples = []
        with open(data_path) as f:
            for sid, line in enumerate(f):
                s = json.loads(line)
                for k_idx in range(k_samples):
                    for doc_idx, score in enumerate(s["teacher_scores"]):
                        doc = s["positive"] if doc_idx == 0 else s["negatives"][doc_idx - 1]
                        self.examples.append({
                            "query": s["query"],
                            "doc": doc,
                            "teacher_score": score,
                            "query_id": sid,
                        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def grpo_reward(student_scores, teacher_scores, query_ids):
    """Pairwise ranking accuracy vs teacher per query group."""
    rewards = torch.zeros(len(query_ids), device=student_scores.device)
    for qid in query_ids.unique():
        mask = query_ids == qid
        idxs = mask.nonzero().squeeze(-1)
        n = idxs.size(0)
        if n < 2:
            continue
        s = student_scores[idxs]
        t = teacher_scores[idxs]
        correct = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (s[i] > s[j]) == (t[i] > t[j]):
                    correct += 1
                total += 1
        acc = correct / max(total, 1)
        rewards[idxs] = acc
    return rewards


def main(
    model_path: str = "models/flashrank-pro-base",
    data_path: str = "data/synthetic_training_data.jsonl",
    output_dir: str = "models/flashrank-pro-base-rl",
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    k_samples: int = 8,
    beta: float = 0.04,
    num_epochs: int = 1,
    max_length: int = 512,
):
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["Wqkv", "Wo", "Wi"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    ref_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, torch_dtype=torch.float16)
    for p in ref_model.parameters():
        p.requires_grad = False

    dataset = RLDataset(data_path, k_samples=k_samples)

    def collate_fn(batch):
        queries = [b["query"] for b in batch]
        docs = [b["doc"] for b in batch]
        teacher_scores = torch.tensor([b["teacher_score"] for b in batch], dtype=torch.float32)
        query_ids = torch.tensor([b["query_id"] for b in batch])
        texts = [f"{q} {tokenizer.sep_token or '[SEP]'} {d}" for q, d in zip(queries, docs)]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "teacher_scores": teacher_scores,
            "query_ids": query_ids,
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    model, ref_model, optimizer, loader, scheduler = accelerator.prepare(model, ref_model, optimizer, loader, scheduler)

    resume_epoch = 0
    if os.path.exists(output_dir):
        for d in os.listdir(output_dir):
            if d.startswith("checkpoint-epoch-"):
                try:
                    resume_epoch = max(resume_epoch, int(d.split("-")[-1]))
                except ValueError:
                    pass
        if resume_epoch > 0:
            ckpt = os.path.join(output_dir, f"checkpoint-epoch-{resume_epoch}")
            accelerator.load_state(ckpt)
            if accelerator.is_main_process:
                print(f"Resumed from epoch {resume_epoch} ({ckpt})")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"GRPO RL | Model: {model_path} | K: {k_samples} | Beta: {beta}")
        print(f"Samples: {len(dataset)} | Steps/epoch: {len(loader)}")

    for epoch in range(resume_epoch, num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(loader, desc=f"RL Epoch {epoch+1}")):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            teacher_scores = batch["teacher_scores"].to(accelerator.device)
            query_ids = batch["query_ids"].to(accelerator.device)

            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

            with torch.no_grad():
                ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

            rewards = grpo_reward(student_logits, teacher_scores, query_ids)
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            kl_div = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(ref_logits.detach(), dim=-1),
                reduction="batchmean",
            )

            pg_loss = -(normalized_rewards * student_logits).mean()
            loss = pg_loss + beta * kl_div

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 50 == 0 and accelerator.is_main_process:
                print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | Reward: {rewards.mean().item():.4f} | KL: {kl_div.item():.4f}")

        ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        accelerator.save_state(ckpt_dir)
        if accelerator.is_main_process:
            print(f"Checkpoint saved: {ckpt_dir}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"RL model saved to {output_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
