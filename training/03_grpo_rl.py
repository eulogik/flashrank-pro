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
from transformers import ModernBertConfig
from peft import LoraConfig, get_peft_model
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False


def _ensure_model_type(model_path: str) -> None:
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if "model_type" not in cfg or not cfg["model_type"]:
            cfg["model_type"] = "modernbert"
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)


def find_latest_checkpoint(output_dir: str) -> tuple[Optional[str], int]:
    if not os.path.exists(output_dir):
        return None, 0
    best_type, best_num = None, 0
    for d in os.listdir(output_dir):
        for prefix in ["checkpoint-step-", "checkpoint-epoch-"]:
            if d.startswith(prefix):
                try:
                    num = int(d[len(prefix):])
                    if num > best_num:
                        best_type, best_num = prefix.rstrip("-"), num
                except ValueError:
                    pass
    return best_type, best_num


def restore_from_checkpoint(model_path: str, model_name: str) -> None:
    """If top-level model weights are missing, restore from latest checkpoint."""
    if os.path.exists(os.path.join(model_path, "model.safetensors")) or \
       os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        return
    ckpt_type, ckpt_num = find_latest_checkpoint(model_path)
    if ckpt_type is None:
        print(f"Stage 2 output at {model_path} has no trained model or checkpoints.")
        print("Run Stage 2 first (needs at least one epoch checkpoint before Stage 3).")
        raise SystemExit(1)
    ckpt_dir = os.path.join(model_path, f"{ckpt_type}-{ckpt_num}")
    print(f"Restoring model from checkpoint {ckpt_dir} ...")
    acc = Accelerator()
    config = ModernBertConfig.from_pretrained(model_path)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, torch_dtype=torch.float32)
    model = acc.prepare(model)
    acc.load_state(ckpt_dir)
    unwrapped = acc.unwrap_model(model)
    unwrapped.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)
    _ensure_model_type(model_path)
    print(f"Restored model to {model_path}")


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
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "teacher_scores": scores_tensor,
        "query_ids": torch.tensor(all_qids),
    }


def grpo_reward(student_scores, teacher_scores, query_ids):
    """Group-relative advantage: teacher score centered within each query.

    Uses the teacher's relative preference as the reward signal so gradients
    are always non-zero (within-query variance > 0). Returns both the
    advantage used for the loss and the pairwise ranking accuracy (for logging).
    """
    advantages = torch.zeros(len(student_scores), device=student_scores.device)
    accs = []
    for qid in query_ids.unique():
        mask = query_ids == qid
        idxs = mask.nonzero().squeeze(-1)
        n = idxs.size(0)
        if n < 2:
            continue
        s = student_scores[idxs]
        t = teacher_scores[idxs]
        t = torch.nan_to_num(t, nan=0.0)

        t_mean = t.mean()
        t_std = t.std() + 1e-8
        advantages[idxs] = (t - t_mean) / t_std

        correct = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (s[i] > s[j]) == (t[i] > t[j]):
                    correct += 1
                total += 1
        accs.append(correct / max(total, 1))

    mean_acc = sum(accs) / len(accs) if accs else 0.0
    return advantages, mean_acc


def main(
    model_path: str = "models/flashrank-pro-base",
    model_name: str = "answerdotai/ModernBERT-base",
    data_path: str = "data/synthetic_training_data.jsonl",
    output_dir: str = "models/flashrank-pro-base-rl",
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    max_docs: int = 5,
    beta: float = 0.04,
    num_epochs: int = 1,
    max_length: int = 512,
    max_steps: int = -1,
):
    if os.path.exists(os.path.join(output_dir, "model.safetensors")) or os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
        print(f"   ✅ {output_dir} already exists — skipping Stage 3")
        return

    accelerator = Accelerator()

    restore_from_checkpoint(model_path, model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        accelerator.print(f"Local tokenizer not found at {model_path}, loading from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    config = ModernBertConfig.from_pretrained(model_path)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, torch_dtype=torch.float32)
    bad = [n for n, p in model.named_parameters() if torch.isnan(p).any() or torch.isinf(p).any()]
    if bad and accelerator.is_main_process:
        print(f"FATAL: {len(bad)} params in checkpoint contain NaN/inf (e.g. {bad[:3]}). "
              f"The Stage 2 checkpoint is corrupted — retrain Stage 2 in fp32.")
        raise SystemExit(1)
    if accelerator.is_main_process:
        print(f"Model loaded in fp32 ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

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
        if max_steps > 0:
            print(f"Smoke mode: max_steps={max_steps}")

    global_step = 0
    for epoch in range(resume_epoch, num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(loader, desc=f"RL Epoch {epoch+1}")):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            teacher_scores = batch["teacher_scores"].to(accelerator.device)
            query_ids = batch["query_ids"].to(accelerator.device)

            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            if torch.isnan(student_logits).any():
                if accelerator.is_main_process:
                    print(f"  WARNING: student_logits has NaN before nan_to_num at step {step}")
            student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50)

            with torch.no_grad():
                ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                ref_logits = torch.nan_to_num(ref_logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50)

            advantages, mean_acc = grpo_reward(student_logits, teacher_scores, query_ids)
            advantages = torch.nan_to_num(advantages, nan=0.0)

            kl_div = F.mse_loss(student_logits.float(), ref_logits.float().detach())

            pg_loss = -(advantages * student_logits.float()).mean()
            loss = pg_loss + beta * kl_div

            if torch.isnan(loss):
                if accelerator.is_main_process:
                    print(f"  NaN detected | pg_loss: {pg_loss.item():.6f} | kl: {kl_div.item():.6f} | "
                          f"adv: min={advantages.min().item():.4f} max={advantages.max().item():.4f} | "
                          f"scores: min={student_logits.min().item():.4f} max={student_logits.max().item():.4f}")
                loss = pg_loss  # fallback to just policy gradient

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if step % 50 == 0 and accelerator.is_main_process:
                print(f"Step {global_step}/{total_steps} | Loss: {loss.item():.6f} | Acc: {mean_acc:.4f} | "
                      f"KL: {kl_div.item():.6f} | GradNorm: {grad_norm.item():.4f}")
            if 0 < max_steps <= global_step:
                break

        ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        accelerator.save_state(ckpt_dir)
        if accelerator.is_main_process:
            print(f"Checkpoint saved: {ckpt_dir}")
        if 0 < max_steps <= global_step:
            break

    accelerator.wait_for_everyone()
    if max_steps > 0 and global_step < total_steps:
        if accelerator.is_main_process:
            print(f"Trimmed training to {global_step} steps (max_steps={max_steps})")
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        merged = unwrapped.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"RL model saved to {output_dir} (LoRA merged into base)")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
