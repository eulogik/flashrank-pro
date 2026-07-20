"""
Stage 3: GRPO Reinforcement Learning for fine-grained scoring.

GRPO (ProRank paper): group-relative policy optimization.
- Each query's K docs are scored together as a group
- Pairwise ranking reward vs teacher within each group
- Relative reward normalization
- KL divergence against reference model

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
    """Each sample = one query with all its docs (positive + negatives).

    Returns grouped query-doc pairs so the reward compares all docs
    within a query.
    """

    def __init__(self, data_path, max_docs=8):
        self.groups = []
        self.max_docs = max_docs
        with open(data_path) as f:
            for sid, line in enumerate(f):
                s = json.loads(line)
                docs = [s["positive"]] + s["negatives"]
                scores = s["teacher_scores"]
                if isinstance(scores, list) and len(scores) > 0 and isinstance(scores[0], (list, tuple)):
                    scores = [sc[-1] for sc in scores]
                if isinstance(scores, list) and len(scores) > max_docs:
                    scores = scores[:max_docs]
                    docs = docs[:max_docs]
                self.groups.append({
                    "query": s["query"],
                    "docs": docs,
                    "teacher_scores": scores if isinstance(scores, list) else [scores],
                    "query_id": sid,
                })

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]


def collate_fn(batch, tokenizer, max_length):
    all_queries = []
    all_docs = []
    all_scores = []
    all_qids = []

    for g in batch:
        k = len(g["docs"])
        all_queries.extend([g["query"]] * k)
        all_docs.extend(g["docs"])
        all_scores.extend(g["teacher_scores"][:k])
        all_qids.extend([g["query_id"]] * k)

    texts = [f"{q} {tokenizer.sep_token or '[SEP]'} {d}" for q, d in zip(all_queries, all_docs)]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "teacher_scores": torch.tensor(all_scores, dtype=torch.float32),
        "query_ids": torch.tensor(all_qids),
    }


def grpo_reward(student_scores, teacher_scores, query_ids):
    """Pairwise ranking accuracy vs teacher per query group."""
    rewards = torch.zeros(len(student_scores), device=student_scores.device)
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
    max_docs: int = 5,
    beta: float = 0.04,
    num_epochs: int = 1,
    max_length: int = 512,
):
    accelerator = Accelerator(mixed_precision="fp16")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["Wqkv", "Wo", "Wi"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    ref_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, torch_dtype=torch.float32)
    for p in ref_model.parameters():
        p.requires_grad = False

    dataset = RLDataset(data_path, max_docs=max_docs)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

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
        print(f"GRPO RL | Model: {model_path} | max_docs: {max_docs} | Beta: {beta}")
        print(f"Queries: {len(dataset)} | Groups/batch: {batch_size} | Total steps: {total_steps}")

    for epoch in range(resume_epoch, num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(loader, desc=f"RL Epoch {epoch+1}")):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            teacher_scores = batch["teacher_scores"].to(accelerator.device)
            query_ids = batch["query_ids"].to(accelerator.device)

            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            student_logits = student_logits.clamp(-50, 50)

            with torch.no_grad():
                ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                ref_logits = ref_logits.clamp(-50, 50)

            rewards = grpo_reward(student_logits, teacher_scores, query_ids)

            r_mean = rewards.mean()
            r_std = rewards.std() + 1e-8
            normalized_rewards = (rewards - r_mean) / r_std

            flat_s = student_logits.view(-1).float()
            flat_r = ref_logits.view(-1).float().detach()

            ref_probs = F.softmax(flat_r, dim=-1).clamp(min=1e-7)
            student_log_probs = F.log_softmax(flat_s, dim=-1)
            kl_div = (ref_probs * (ref_probs.log() - student_log_probs)).sum()

            pg_loss = -(normalized_rewards * student_logits).mean()
            loss = pg_loss + beta * kl_div

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 50 == 0 and accelerator.is_main_process:
                print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | Reward: {r_mean.item():.4f} | KL: {kl_div.item():.4f}")

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
