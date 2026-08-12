"""
Fine-tune FlashRank-Pro on BEIR train splits with BM25-mined hard negatives.

The model was trained on synthetic data with easy negatives -> weak discrimination
on BM25-selected candidates. This fine-tune trains exactly on that distribution:
for each training query, positives come from qrels, hard negatives are top BM25
candidates that are NOT relevant. Margin ranking loss + BCE calibration.

Usage:
    python training/05_beir_finetune.py \
        --model_path eulogik/flashrank-pro-base \
        --datasets scifact,fiqa,arguana,scidocs,nfcorpus \
        --output_dir models/flashrank-pro-beir \
        --max_queries_per_dataset 4000 \
        --num_hard_negatives 5 \
        --num_epochs 2
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")),
)
from evaluate_beir_bm25 import build_or_load_index, retrieve


def load_beir_split(name: str, split: str):
    from beir.datasets.data_loader import GenericDataLoader

    return GenericDataLoader(data_folder=f"data/beir/{name}").load(split=split)


def build_training_data(datasets, max_queries_per_dataset, num_hard_negatives, seed):
    """Each example: (query, pos_text, [neg_texts]) using BM25 hard negatives."""
    random.seed(seed)
    examples = []
    for name in datasets:
        t0 = time.time()
        try:
            corpus, queries, qrels = load_beir_split(name, "train")
        except Exception as e:
            print(f"  {name}: no train split ({e}); skipping")
            continue
        bm25, doc_ids = build_or_load_index(name, corpus, "data/beir_index")

        n_neg_ok = 0
        for qid, query in queries.items():
            if n_neg_ok >= max_queries_per_dataset:
                break
            pos = [c for c in qrels.get(qid, {}) if qrels[qid][c] > 0]
            if not pos:
                continue
            rel = set(qrels[qid])
            bm25_cands = list(retrieve(bm25, doc_ids, {qid: query}, 100)[qid].keys())
            hard_neg = [c for c in bm25_cands if c not in rel]
            if not hard_neg:
                continue
            random.shuffle(hard_neg)
            negs = hard_neg[:num_hard_negatives]
            examples.append(
                {
                    "dataset": name,
                    "query": query,
                    "pos": [corpus[c]["text"] for c in pos[:2]],
                    "negs": [corpus[c]["text"] for c in negs],
                }
            )
            n_neg_ok += 1
        print(f"  {name}: {n_neg_ok} train examples ({time.time()-t0:.0f}s)")
    return examples


class RerankDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(examples, tokenizer, max_length):
    """Pack all positives+negatives of a batch of queries into one padded batch."""
    bos = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id

    items = []  # (qid_in_batch, is_pos, input_ids, attention_mask)
    for bi, ex in enumerate(examples):
        q = tokenizer.encode(ex["query"], add_special_tokens=False)
        for p in ex["pos"]:
            items.append((bi, 1.0, q, tokenizer.encode(p, add_special_tokens=False)))
        for n in ex["negs"]:
            items.append((bi, 0.0, q, tokenizer.encode(n, add_special_tokens=False)))

    # length-sort for tight packing
    items.sort(key=lambda x: len(x[2]) + len(x[3]))

    input_ids, attention_mask, labels = [], [], []
    for bi, is_pos, q, d in items:
        d = d[: max_length - len(q) - 3]
        n_real = len(q) + len(d) + 3
        ids = [bos] + q + [sep] + d + [sep] + [pad] * (max_length - n_real)
        input_ids.append(ids)
        attention_mask.append([1] * n_real + [0] * (max_length - n_real))
        labels.append((bi, is_pos))
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        labels,
        len(examples),
    )


def evaluate_on_test(model, tokenizer, dataset="scifact", n_queries=40, max_length=512):
    """Quick NDCG@10 estimate on a random test subset (BM25 top-100 + rerank)."""
    corpus, queries, qrels = load_beir_split(dataset, "test")
    bm25, doc_ids = build_or_load_index(dataset, corpus, "data/beir_index")
    res = retrieve(bm25, doc_ids, queries, 100)

    from beir.retrieval.evaluation import EvaluateRetrieval

    random.seed(42)
    qids = list(res.keys())
    random.shuffle(qids)
    qids = qids[:n_queries]
    doc_cache = {}
    for q in qids:
        for c in res[q]:
            if c not in doc_cache:
                doc_cache[c] = tokenizer.encode(corpus[c]["text"], add_special_tokens=False)
    bos = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id

    reranked = {}
    with torch.inference_mode():
        for qid in qids:
            cands = list(res[qid].keys())
            q = tokenizer.encode(queries[qid], add_special_tokens=False)
            pairs = [(c, q, doc_cache[c][: max_length - len(q) - 3]) for c in cands]
            pairs.sort(key=lambda x: len(x[1]) + len(x[2]))
            scores = {}
            for start in range(0, len(pairs), 96):
                batch = pairs[start : start + 96]
                max_len = min(max_length, max(len(qq) + len(d) + 3 for _, qq, d in batch))
                input_ids, amask = [], []
                for c, qq, d in batch:
                    n_real = len(qq) + len(d) + 3
                    input_ids.append([bos] + qq + [sep] + d + [sep] + [pad] * (max_len - n_real))
                    amask.append([1] * n_real + [0] * (max_len - n_real))
                logits = model(
                    input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(amask)
                ).logits
                for (c, _, _), s in zip(batch, torch.sigmoid(logits.float()).squeeze(-1).tolist()):
                    scores[c] = s
            reranked[qid] = dict(sorted(scores.items(), key=lambda x: -x[1]))
    ndcg = EvaluateRetrieval.evaluate(qrels, reranked, [10])[0]["NDCG@10"]
    return ndcg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="eulogik/flashrank-pro-base")
    ap.add_argument("--datasets", default="scifact,fiqa,arguana,scidocs,nfcorpus")
    ap.add_argument("--data_jsonl", default=None, help="JSONL of {query,pos,negs} — skips BEIR build")
    ap.add_argument("--output_dir", default="models/flashrank-pro-beir")
    ap.add_argument("--max_queries_per_dataset", type=int, default=4000)
    ap.add_argument("--num_hard_negatives", type=int, default=5)
    ap.add_argument("--num_epochs", type=int, default=2)
    ap.add_argument("--batch_queries", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--bce_weight", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_queries", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    print("Building training data (BM25 hard negatives)...")
    ds_list = args.datasets.split(",")
    cache_path = os.path.join(args.output_dir, "train_examples.pkl")
    import pickle

    examples = None
    if args.data_jsonl:
        import json as _json

        with open(args.data_jsonl) as f:
            examples = [_json.loads(line) for line in f]
        print(f"  loaded {len(examples)} examples from {args.data_jsonl}")
    elif os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            examples = pickle.load(f)
        print(f"  loaded {len(examples)} cached examples")
    if examples is None:
        examples = build_training_data(ds_list, args.max_queries_per_dataset, args.num_hard_negatives, seed=0)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(examples, f)
    print(f"Total training examples: {len(examples)}")
    for e in examples[:2]:
        print(f"  sample: q={e['query'][:60]} pos={len(e['pos'])} negs={len(e['negs'])}")

    dataset = RerankDataset(examples, tokenizer, args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_queries, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, args.max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.num_epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    start_step = 0
    if args.resume:
        ckpt = os.path.join(args.output_dir, "checkpoint.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = state["step"]
            print(f"Resumed from step {start_step}")

    print(f"Training: {total_steps} total steps, {len(loader)} steps/epoch")
    best_ndcg = 0.0
    step = start_step
    t0 = time.time()
    loss_ema = None
    for epoch in range(args.num_epochs):
        for batch in loader:
            if step % 10 == 0:
                print(f"  heartbeat step {step}", flush=True)
            input_ids, amask, labels, n_queries = batch
            logits = model(input_ids=input_ids, attention_mask=amask).logits.float().squeeze(-1)
            scores = torch.sigmoid(logits)

            # group scores by query; margin loss between pos and each hard neg
            margin_loss = torch.zeros((), dtype=torch.float32)
            bce = torch.zeros((), dtype=torch.float32)
            groups = {}
            for (bi, is_pos), s in zip(labels, scores):
                groups.setdefault(bi, []).append((is_pos, s))
            n_pos = 0
            for bi in range(n_queries):
                items = groups.get(bi, [])
                pos_scores = [s for is_pos, s in items if is_pos == 1.0]
                neg_scores = [s for is_pos, s in items if is_pos == 0.0]
                for ps in pos_scores:
                    for ns in neg_scores:
                        margin_loss += torch.clamp(args.margin - (ps - ns), min=0.0)
                    bce += -(torch.log(ps + 1e-7)) - torch.log(1 - torch.stack(neg_scores) + 1e-7).mean()
                    n_pos += 1
            margin_loss = margin_loss / max(n_pos, 1)
            bce = bce / max(n_pos, 1)
            loss = margin_loss + args.bce_weight * bce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
            step += 1

            ema = loss_ema if loss_ema is not None else loss.item()
            loss_ema = 0.95 * ema + 0.05 * loss.item()
            if step % 50 == 0:
                rate = (time.time() - t0) / (step - start_step)
                eta = rate * (total_steps - step) / 60
                print(
                    f"  step {step}/{total_steps} | loss {loss_ema:.4f} (margin {margin_loss.item():.4f}, bce {bce.item():.4f}) | {rate:.2f}s/step | ETA {eta:.0f}min",
                    flush=True,
                )

            if step % args.eval_every == 0:
                model.eval()
                ndcg = evaluate_on_test(model, tokenizer, n_queries=args.eval_queries)
                model.train()
                print(f"  >>> EVAL scifact({args.eval_queries}q): NDCG@10 = {ndcg:.4f} (best {best_ndcg:.4f})", flush=True)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    print(f"  >>> saved best to {args.output_dir}")
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "step": step},
                    os.path.join(args.output_dir, "checkpoint.pt"),
                )
                print(f"  >>> checkpoint saved at step {step}", flush=True)

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "step": step},
            os.path.join(args.output_dir, "checkpoint.pt"),
        )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training done. Best eval NDCG@10: {best_ndcg:.4f}")


if __name__ == "__main__":
    main()
