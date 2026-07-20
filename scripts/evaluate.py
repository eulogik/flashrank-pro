"""
Evaluate FlashRank-Pro on BEIR/MTEB benchmarks.

Usage:
    python scripts/evaluate.py --model_path models/flashrank-pro-merged
    python scripts/evaluate.py --model_path models/flashrank-pro-merged --benchmark beir
"""

import json
import os
import sys
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm


BENCHMARK_DATASETS = {
    "beir": [
        "nfcorpus", "scifact", "fiqa", "arguana",
        "webis-touche2020", "quora", "scidocs",
    ],
    "mteb_reranking": [
        "askubuntu", "mind_small", "yahoo_answers_topics",
    ],
}


def evaluate_beir(model_path: str, datasets: list[str] = None, top_k: int = 100):
    """Evaluate on BEIR benchmark datasets using BM25 + reranker."""
    try:
        from beir import util, LoggingHandler
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.lexical import BM25Search as BM25
    except ImportError:
        print("BEIR library not installed. Install with: pip install beir")
        return {}

    if datasets is None:
        datasets = BENCHMARK_DATASETS["beir"]

    reranker = CrossEncoder(model_path, max_length=512)

    results = {}
    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = os.path.join("data/beir", dataset_name)
        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        except Exception:
            print(f"Downloading {dataset_name}...")
            util.download_and_unzip(url, "data/beir")
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        bm25 = BM25(index_name=dataset_name, hostname="localhost", initialize=True)
        bm25_retriever = EvaluateRetrieval(bm25, k_values=[top_k])
        bm25_results = bm25_retriever.retrieve(corpus, queries)

        reranked = {}
        for qid in tqdm(queries, desc=f"Reranking {dataset_name}"):
            query = queries[qid]
            doc_ids = list(bm25_results[qid].keys())[:top_k]
            docs = [corpus[doc_id]["text"] for doc_id in doc_ids]
            if not docs:
                reranked[qid] = {}
                continue
            pairs = [[query, doc] for doc in docs]
            scores = reranker.predict(pairs)
            sorted_idx = np.argsort(-scores)
            reranked[qid] = {doc_ids[i]: float(scores[i]) for i in sorted_idx}

        ndcg = EvaluateRetrieval.evaluate(qrels, reranked, [10])
        results[dataset_name] = ndcg
        print(f"  NDCG@10: {ndcg.get('NDCG@10', 'N/A'):.4f}")

    return results


def evaluate_mteb_reranking(model_path: str, datasets: list[str] = None):
    """Evaluate on MTEB reranking subset using mteb library."""
    try:
        import mteb
    except ImportError:
        print("MTEB library not installed. Install with: pip install mteb")
        return {}

    tasks = mteb.get_tasks(tasks=datasets or BENCHMARK_DATASETS["mteb_reranking"])
    model = CrossEncoder(model_path, max_length=512)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="results/mteb")
    return results


def main(
    model_path: str = "models/flashrank-pro-merged",
    benchmark: str = "beir",
    datasets: Optional[str] = None,
):
    print(f"Evaluating {model_path} on {benchmark}...")

    ds_list = datasets.split(",") if datasets else None

    if benchmark == "beir":
        results = evaluate_beir(model_path, datasets=ds_list)
    elif benchmark == "mteb":
        results = evaluate_mteb_reranking(model_path, datasets=ds_list)
    else:
        print(f"Unknown benchmark: {benchmark}")
        return

    os.makedirs("results", exist_ok=True)
    output_path = f"results/eval_{benchmark}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if results:
        avg = np.mean([v.get("NDCG@10", 0) for v in results.values() if isinstance(v, dict)])
        print(f"Average NDCG@10: {avg:.4f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
