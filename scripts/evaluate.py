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
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """Load tokenizer directly from tokenizer.json to avoid tokenizer_class issues."""
    # Try local path first, then HF hub cache
    tok_path = None
    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        tok_path = os.path.join(model_path, "tokenizer.json")
    else:
        # Try HF hub cache
        import glob
        cache_pattern = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}/snapshots/*/tokenizer.json"
        )
        matches = glob.glob(cache_pattern)
        if matches:
            tok_path = matches[0]
    
    if not tok_path or not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer.json not found for {model_path}")
    
    tok = Tokenizer.from_file(tok_path)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        model_max_length=8192,
        padding_side="right",
        truncation_side="right",
    )


BENCHMARK_DATASETS = {
    "beir": [
        "nfcorpus", "scifact", "fiqa", "arguana",
        "webis-touche2020", "quora", "scidocs",
    ],
    "mteb_reranking": [
        "askubuntu", "mind_small", "yahoo_answers_topics",
    ],
}


def tfidf_retrieve(corpus: dict, queries: dict, top_k: int = 100):
    """Simple TF-IDF retrieval as BM25 substitute."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did]["text"] for did in doc_ids]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    doc_vectors = vectorizer.fit_transform(doc_texts)

    results = {}
    for qid, query in queries.items():
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, doc_vectors).flatten()
        top_idx = np.argsort(-sims)[:top_k]
        results[qid] = {doc_ids[i]: float(sims[i]) for i in top_idx}

    return results


def evaluate_beir(model_path: str, datasets: list[str] = None, top_k: int = 100):
    """Evaluate on BEIR benchmark datasets using TF-IDF + reranker."""
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError:
        print("BEIR library not installed. Install with: pip install beir")
        return {}

    if datasets is None:
        datasets = BENCHMARK_DATASETS["beir"]

    reranker = CrossEncoder(model_path, max_length=512, trust_remote_code=True)

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

        tfidf_results = tfidf_retrieve(corpus, queries, top_k=top_k)

        reranked = {}
        for qid in tqdm(queries, desc=f"Reranking {dataset_name}"):
            query = queries[qid]
            doc_ids = list(tfidf_results[qid].keys())[:top_k]
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
    model = CrossEncoder(model_path, max_length=512, trust_remote_code=True)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="results/mteb")
    return results


def main(
    model_path: str = "models/flashrank-pro-merged",
    benchmark: str = "beir",
    datasets: Optional[str] = None,
):
    print(f"Evaluating {model_path} on {benchmark}...")

    if isinstance(datasets, tuple):
        ds_list = list(datasets)
    else:
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
