"""
Evaluate FlashRank-Pro on MTEB reranking tasks (new corpus/queries/data format).

Tasks: AskUbuntuDupQuestions, SciDocsRR, StackOverflowDupQuestions
Metrics (official MTEB reranking family): MAP/NDCG/MRR @ 10,100,1000, main = MAP@1000.

Usage:
    python scripts/evaluate_mteb_rerank.py --model_path eulogik/flashrank-pro-base
"""

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TASKS = {
    "AskUbuntuDupQuestions": {"dataset": "mteb/AskUbuntuDupQuestions", "split": "test"},
    "SciDocsRR": {"dataset": "mteb/SciDocsRR", "split": "test"},
    "StackOverflowDupQuestions": {"dataset": "mteb/StackOverflowDupQuestions", "split": "test"},
}

BASE_URL = "https://huggingface.co/datasets/{dataset}/resolve/main/{folder}/{split}-00000-of-00001.parquet"


def load_task(spec: dict):
    url_data = BASE_URL.format(dataset=spec["dataset"], folder="data", split=spec["split"])
    url_q = BASE_URL.format(dataset=spec["dataset"], folder="queries", split=spec["split"])
    url_c = BASE_URL.format(dataset=spec["dataset"], folder="corpus", split=spec["split"])

    data = pd.read_parquet(url_data)
    queries = pd.read_parquet(url_q)
    corpus = pd.read_parquet(url_c)

    q_map = dict(zip(queries["_id"], queries["text"]))
    c_map = dict(zip(corpus["_id"], corpus["text"]))
    return data, q_map, c_map


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    ideal = np.sort(ranks)[::-1][:k]
    ideal_dcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal))
    if ideal_dcg == 0:
        return 0.0
    dcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ranks[:k]))
    return dcg / ideal_dcg


def map_at_k(ranks: np.ndarray, k: int) -> float:
    relevant_pos = [i for i, rel in enumerate(ranks[:k]) if rel == 1]
    if not relevant_pos:
        return 0.0
    return sum((i + 1) / (pos + 1) for i, pos in enumerate(relevant_pos)) / len(relevant_pos)


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    for pos, rel in enumerate(ranks[:k]):
        if rel == 1:
            return 1.0 / (pos + 1)
    return 0.0


def evaluate_task(model, tokenizer, name: str, results_dir: str,
                  max_length: int, batch_size: int, max_queries: Optional[int]):
    result_path = os.path.join(results_dir, f"{name}_rerank.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
        print(f"{name}: already done -> MAP@1000 {existing.get('MAP@1000', 0):.4f}")
        return existing

    print(f"\n=== {name} ===")
    data, q_map, c_map = load_task(TASKS[name])
    print(f"  {len(data)} pairs, {data['query-id'].nunique()} queries")

    bos = tokenizer.cls_token_id or tokenizer.bos_token_id
    sep = tokenizer.sep_token_id

    needed = set(data["corpus-id"].unique())
    doc_cache: dict = {}
    t0 = time.time()
    for cid in needed:
        doc_cache[cid] = tokenizer.encode(c_map[cid], add_special_tokens=False)
    print(f"  Tokenized {len(doc_cache)} docs in {time.time()-t0:.1f}s")

    groups = data.groupby("query-id")
    qids = list(groups.groups.keys())
    if max_queries:
        qids = qids[:max_queries]

    t0 = time.time()
    agg = {m: [] for m in ["MAP@10", "MAP@100", "MAP@1000", "NDCG@10", "NDCG@100", "NDCG@1000", "MRR@10", "MRR@100", "MRR@1000"]}

    with torch.inference_mode():
        for qi, qid in enumerate(qids):
            rows = groups.get_group(qid)
            query = q_map[qid]
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            max_doc = max_length - len(query_ids) - 3
            cids = rows["corpus-id"].tolist()
            labels = rows["score"].to_numpy().astype(int)

            pairs = [(query_ids, doc_cache[cid][:max_doc]) for cid in cids]
            order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
            pairs = [pairs[i] for i in order]

            scores = []
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                max_len = min(max_length, max(len(q) + len(d) + 3 for q, d in batch))
                input_ids, attention_mask = [], []
                for q, d in batch:
                    n_real = len(q) + len(d) + 3
                    pad = max_len - n_real
                    ids = [bos] + q + [sep] + d + [sep] + [tokenizer.pad_token_id] * pad
                    input_ids.append(ids)
                    attention_mask.append([1] * n_real + [0] * pad)
                ids_t = torch.tensor(input_ids, dtype=torch.long)
                mask_t = torch.tensor(attention_mask, dtype=torch.long)
                logits = model(input_ids=ids_t, attention_mask=mask_t).logits
                scores.extend(torch.sigmoid(logits.float()).squeeze(-1).tolist())

            ranks = np.array([labels[i] for i in order])[np.argsort(-np.array(scores))]
            for m in agg:
                k = int(m.split("@")[1])
                fn = {"MAP": map_at_k, "NDCG": ndcg_at_k, "MRR": mrr_at_k}[m.split("@")[0]]
                agg[m].append(fn(ranks, k))

            if (qi + 1) % 500 == 0:
                rate = (time.time() - t0) / (qi + 1)
                print(f"  {qi+1}/{len(qids)} queries ({rate:.2f}s/q, ETA {rate*(len(qids)-qi-1)/60:.0f}min)")

    metrics = {m: float(np.mean(v)) for m, v in agg.items()}
    os.makedirs(results_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {name}: MAP@1000 {metrics['MAP@1000']:.4f} | NDCG@10 {metrics['NDCG@10']:.4f} | MRR@10 {metrics['MRR@10']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="eulogik/flashrank-pro-base")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--results_dir", default="results/mteb")
    args = ap.parse_args()

    print(f"Loading model {args.model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    torch.set_num_threads(min(args.threads, os.cpu_count() or 8))

    task_list = args.tasks.split(",") if args.tasks else list(TASKS.keys())
    summary = {}
    for task in task_list:
        try:
            summary[task] = evaluate_task(
                model, tokenizer, task, args.results_dir,
                args.max_length, args.batch_size, args.max_queries,
            )
        except Exception as e:
            print(f"  FAILED {task}: {e}")
            summary[task] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("MTEB Reranking results:")
    for task, m in summary.items():
        if "error" in m:
            print(f"  {task}: ERROR {m['error']}")
        else:
            print(f"  {task}: MAP@1000 {m.get('MAP@1000', 0):.4f} | NDCG@10 {m.get('NDCG@10', 0):.4f}")
    vals = [m["MAP@1000"] for m in summary.values() if "error" not in m]
    if vals:
        print(f"  AVERAGE MAP@1000 ({len(vals)} tasks): {np.mean(vals):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
