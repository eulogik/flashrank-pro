"""
Stage 3: GRPO Reinforcement Learning for prompt warmup + fine-grained scoring.

Implements GRPO from ProRank paper:
- Sample K=8 outputs per query-doc pair
- Reward based on ranking quality (pairwise accuracy vs teacher)
- Relative reward normalization (max-min)
- Fine-grained score learning via Yes/No logit ratio

This runs as LoRA on top of the Stage 2 model.
Colab T4: ~2h.
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model


def compute_fine_grained_score(model, input_ids, attention_mask, tokenizer):
    """Compute fine-grained score from Yes/No logit ratio (ProRank method)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
    logits = outputs.logits
    last_token_logits = logits[:, -1, :]
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    yes_logit = last_token_logits[:, yes_id]
    no_logit = last_token_logits[:, no_id]
    scores = yes_logit - no_logit
    return scores


def grpo_reward(student_scores, teacher_scores, query_ids):
    """Compute reward for GRPO: pairwise ranking accuracy vs teacher."""
    query_ids = torch.tensor(query_ids, device=student_scores.device)
    rewards = []
    for qid in query_ids.unique():
        mask = query_ids == qid
        s_scores = student_scores[mask]
        t_scores = teacher_scores[mask]
        n = s_scores.size(0)
        if n < 2:
            rewards.append(torch.tensor(0.0, device=student_scores.device))
            continue
        correct = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (s_scores[i] > s_scores[j]) == (t_scores[i] > t_scores[j]):
                    correct += 1
                total += 1
        rewards.append(torch.tensor(correct / max(total, 1), device=student_scores.device))
    return torch.stack(rewards)


def grpo_loss(model, ref_model, batch, tokenizer, k_samples=8, beta=0.04):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    teacher_scores = batch["teacher_scores"]
    query_ids = batch["query_ids"]

    bsz = input_ids.size(0)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

    student_scores = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

    rewards = grpo_reward(student_scores, teacher_scores, query_ids)
    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    kl_div = F.kl_div(
        F.log_softmax(student_scores.view(-1), dim=-1),
        F.softmax(ref_logits.view(-1).detach(), dim=-1),
        reduction="batchmean",
    )

    pg_loss = -(normalized_rewards * student_scores).mean()
    loss = pg_loss + beta * kl_div

    return loss, {"loss": loss.item(), "reward": rewards.mean().item(), "kl": kl_div.item()}


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
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, torch_dtype=torch.float16, ignore_mismatched_sizes=True)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    ref_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, torch_dtype=torch.float16)
    for p in ref_model.parameters():
        p.requires_grad = False

    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    all_examples = []
    for sid, s in enumerate(samples):
        for k_idx in range(k_samples):
            all_examples.append({
                "query": s["query"],
                "positive": s["positive"],
                "negatives": s["negatives"],
                "teacher_scores": s["teacher_scores"],
                "query_id": sid,
            })

    def collate(batch):
        queries = [b["query"] for b in batch]
        docs = [b["positive"] for b in batch]
        teacher_scores = []
        query_ids = []
        for b in batch:
            teacher_scores.extend(b["teacher_scores"])
            query_ids.extend([b["query_id"]] * len(b["teacher_scores"]))
        texts = [f"{q} {tokenizer.sep_token or '[SEP]'} {d}" for q, d in zip(queries, docs)]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "teacher_scores": torch.tensor(teacher_scores[:len(queries)], dtype=torch.float32),
            "query_ids": query_ids[:len(queries)],
        }

    loader = DataLoader(all_examples, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    model, ref_model, optimizer, loader, scheduler = accelerator.prepare(model, ref_model, optimizer, loader, scheduler)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"GRPO RL Training | Model: {model_path} | K: {k_samples} | Beta: {beta}")

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(loader, desc=f"RL Epoch {epoch+1}")):
            loss, metrics = grpo_loss(model, ref_model, batch, tokenizer, k_samples, beta)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 50 == 0 and accelerator.is_main_process:
                print(f"Step {step}: {metrics}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"RL model saved to {output_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
