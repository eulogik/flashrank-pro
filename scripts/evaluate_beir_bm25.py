"""
Evaluate FlashRank-Pro on BEIR with proper Okapi BM25 retrieval + reranking.

Usage:
    python scripts/evaluate_beir_bm25.py --model_path eulogik/flashrank-pro-base
    python scripts/evaluate_beir_bm25.py --model_path eulogik/flashrank-pro-base --datasets nfcorpus,scifact
    python scripts/evaluate_beir_bm25.py --model_path eulogik/flashrank-pro-base --include_quora
"""

import argparse
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class EvalConfig:
    model_path: str
    datasets: list
    top_k: int = 100
    max_length: int = 512
    batch_size: int = 128
    results_dir: str = "results/beir"
    index_dir: str = "data/beir_index"
    include_quora: bool = False
    max_queries: Optional[int] = None


DEFAULT_DATASETS = [
    "nfcorpus", "scifact", "fiqa", "arguana",
    "webis-touche2020", "scidocs",
]
QUORA = "quora"


def tokenize(text: str) -> list[str]:
    """Lucene-standard-analyzer-like tokenization: lowercase, split non-alphanumerics."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_dataset(name: str, data_dir: str = "data/beir"):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = os.path.join(data_dir, name)
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    except Exception:
        print(f"  Downloading {name}...")
        util.download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
            data_dir,
        )
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def build_or_load_index(name: str, corpus: dict, index_dir: str):
    """BM25Okapi index with k1=1.5, b=0.75, cached to disk per dataset."""
    os.makedirs(index_dir, exist_ok=True)
    cache_path = os.path.join(index_dir, f"{name}_bm25.pkl")
    doc_ids = list(corpus.keys())

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached["doc_ids"] == doc_ids:
            return cached["bm25"], doc_ids

    t0 = time.time()
    tokenized = [tokenize(corpus[did]["text"]) for did in doc_ids]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    with open(cache_path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)
    print(f"  Indexed {len(doc_ids)} docs in {time.time()-t0:.1f}s")
    return bm25, doc_ids


def retrieve(bm25: BM25Okapi, doc_ids: list, queries: dict, top_k: int):
    results = {}
    for qid, query in queries.items():
        scores = bm25.get_scores(tokenize(query))
        top_idx = np.argsort(-scores)[:top_k]
        results[qid] = {doc_ids[i]: float(scores[i]) for i in top_idx if scores[i] > 0}
    return results


def rerank(model, tokenizer, corpus: dict, bm25_results: dict, queries: dict,
           top_k: int, max_length: int, batch_size: int):
    """Score all (query, doc) pairs.

    Optimizations:
    - corpus docs tokenized ONCE and cached as token id lists
    - pairs pooled across queries, binned by length, batched within bins (tight padding)
    """
    qids = list(bm25_results.keys())
    reranked: dict[str, dict] = {}
    t0 = time.time()

    bos = tokenizer.cls_token_id or tokenizer.bos_token_id
    eos = tokenizer.sep_token_id or tokenizer.eos_token_id
    sep = tokenizer.sep_token_id

    doc_ids_needed = set()
    for qid in qids:
        for doc_id in list(bm25_results[qid].keys())[:top_k]:
            doc_ids_needed.add(doc_id)
    doc_cache: dict = {}
    for doc_id in doc_ids_needed:
        doc_cache[doc_id] = tokenizer.encode(corpus[doc_id]["text"], add_special_tokens=False)

    print(f"  Tokenized {len(doc_cache)} unique docs ({time.time()-t0:.1f}s)")

    t0 = time.time()
    qids_done = 0
    for chunk_start in range(0, len(qids), 100):
        chunk_qids = qids[chunk_start:chunk_start + 100]

        pairs = []  # (qid, doc_id, query_ids, doc_ids)
        for qid in chunk_qids:
            query_ids = tokenizer.encode(queries[qid], add_special_tokens=False)
            max_doc = max_length - len(query_ids) - 3
            for doc_id in list(bm25_results[qid].keys())[:top_k]:
                pairs.append((qid, doc_id, query_ids, doc_cache[doc_id][:max_doc]))

        if not pairs:
            for qid in chunk_qids:
                reranked[qid] = {}
            continue

        bins = [(0, 64), (65, 128), (129, 256), (257, 512)]
        for lo, hi in bins:
            bin_pairs = [
                p for p in pairs
                if lo <= len(p[2]) + len(p[3]) <= hi
            ]
            if not bin_pairs:
                continue
            order = sorted(range(len(bin_pairs)), key=lambda i: len(bin_pairs[i][2]) + len(bin_pairs[i][3]))
            bin_pairs = [bin_pairs[i] for i in order]

            for start in range(0, len(bin_pairs), batch_size):
                batch = bin_pairs[start:start + batch_size]
                max_len = min(max_length, max(len(q) + len(d) + 3 for _, _, q, d in batch))
                input_ids, attention_mask = [], []
                for _, _, q, d in batch:
                    n_real = len(q) + len(d) + 3
                    pad = max_len - n_real
                    ids = [bos] + q + [sep] + d + [eos] + [tokenizer.pad_token_id] * pad
                    input_ids.append(ids)
                    attention_mask.append([1] * n_real + [0] * pad)
                ids_t = torch.tensor(input_ids, dtype=torch.long)
                mask_t = torch.tensor(attention_mask, dtype=torch.long)
                with torch.inference_mode():
                    logits = model(input_ids=ids_t, attention_mask=mask_t).logits
                scores = torch.sigmoid(logits.float()).squeeze(-1).tolist()
                for (qid, doc_id, _, _), score in zip(batch, scores):
                    reranked.setdefault(qid, {})[doc_id] = float(score)

        qids_done += len(chunk_qids)
        elapsed = time.time() - t0
        rate = elapsed / qids_done if qids_done else 0
        print(f"  Reranked {qids_done}/{len(qids)} queries ({rate:.2f}s/q, ETA {rate*(len(qids)-qids_done)/60:.0f}min)")

    return reranked


def evaluate_dataset(model, tokenizer, name: str, cfg: EvalConfig):
    from beir.retrieval.evaluation import EvaluateRetrieval

    result_path = os.path.join(cfg.results_dir, f"{name}_ndcg10.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
        print(f"  {name}: already done -> NDCG@10 {existing.get('NDCG@10', 0):.4f}")
        return existing

    print(f"\n=== {name} ===")
    corpus, queries, qrels = load_dataset(name)
    if cfg.max_queries:
        qids = list(queries.keys())[: cfg.max_queries]
        queries = {qid: queries[qid] for qid in qids}

    print(f"  Corpus {len(corpus)} docs, {len(queries)} queries")
    bm25, doc_ids = build_or_load_index(name, corpus, cfg.index_dir)

    t0 = time.time()
    bm25_results = retrieve(bm25, doc_ids, queries, cfg.top_k)
    print(f"  BM25 retrieval done in {time.time()-t0:.1f}s")

    reranked = rerank(model, tokenizer, corpus, bm25_results, queries,
                      cfg.top_k, cfg.max_length, cfg.batch_size)

    ndcg = EvaluateRetrieval.evaluate(qrels, reranked, [10])
    metrics = ndcg[0]
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {name} NDCG@10: {metrics.get('NDCG@10', 0):.4f} | MAP@10: {metrics.get('MAP@10', 0):.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="eulogik/flashrank-pro-base")
    ap.add_argument("--datasets", default=None, help="comma-separated; default all except quora")
    ap.add_argument("--include_quora", action="store_true")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    cfg = EvalConfig(
        model_path=args.model_path,
        datasets=args.datasets.split(",") if args.datasets else DEFAULT_DATASETS,
        top_k=args.top_k,
        max_length=args.max_length,
        batch_size=args.batch_size,
        include_quora=args.include_quora,
        max_queries=args.max_queries,
    )
    if args.include_quora and QUORA not in cfg.datasets:
        cfg.datasets.append(QUORA)

    print(f"Loading model {cfg.model_path}...")
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    model.eval()
    torch.set_num_threads(min(args.threads, os.cpu_count() or 8))
    print(f"Model loaded in {time.time()-t0:.1f}s, threads={torch.get_num_threads()}")

    summary = {}
    for name in cfg.datasets:
        try:
            summary[name] = evaluate_dataset(model, tokenizer, name, cfg)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            summary[name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("BEIR NDCG@10 (BM25 top-100 + FlashRank-Pro rerank):")
    for name, m in summary.items():
        if "error" in m:
            print(f"  {name}: ERROR {m['error']}")
        else:
            print(f"  {name}: {m.get('NDCG@10', 0):.4f}")
    vals = [m["NDCG@10"] for m in summary.values() if "error" not in m]
    if vals:
        print(f"  AVERAGE ({len(vals)} datasets): {np.mean(vals):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
