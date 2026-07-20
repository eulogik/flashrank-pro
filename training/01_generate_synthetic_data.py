"""
Stage 1: Generate training data for knowledge distillation.

Default: Uses existing query-doc pairs from any HF dataset (free).
Optional: Generates synthetic queries via OpenRouter or OpenAI.

Pipeline:
  1. Load corpus (e.g., GooAQ with question/answer fields)
  2. Mine hard negatives via embedding similarity
  3. Score all pairs with a teacher reranker for soft labels

Cost: Free by default. Only costs if you use --llm_endpoint for generation.
"""

import json
import math
import os
import random
from typing import Optional

import torch

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


FIELD_MAP = {
    "query": ["question", "query", "title", "text"],
    "positive": ["answer", "positive", "relevant", "document", "text", "passage"],
}


def resolve_fields(dataset) -> tuple[str, str]:
    """Auto-detect query and positive doc field names in the dataset."""
    cols = dataset.column_names
    q_field = next((c for c in FIELD_MAP["query"] if c in cols), cols[0])
    p_field = next((c for c in FIELD_MAP["positive"] if c in cols), cols[-1])
    return q_field, p_field


def load_queries(
    corpus_name: str,
    split: str = "train",
    n_queries: int = 50000,
) -> tuple[dict[str, str], list[str]]:
    """Load query-doc pairs from a HuggingFace dataset.

    Returns (queries dict mapping query→positive doc, full corpus list).
    Public datasets with query-doc pairs:
      - sentence-transformers/gooaq  (3M, English Q&A)
      - sentence-transformers/natural-questions  (QA pairs)
      - ms_marco  (500K+ search queries)
      - BeIR/*  (per-dataset queries)
    """
    dataset = load_dataset(corpus_name, split=split)
    q_field, p_field = resolve_fields(dataset)
    print(f"    Fields: query='{q_field}', positive='{p_field}'")

    queries = {}
    corpus = []
    for i, row in enumerate(dataset):
        q = str(row[q_field]).strip()
        d = str(row[p_field]).strip()
        if q and d and len(q) > 3 and len(d) > 20:
            queries[q] = d
            corpus.append(d)
        if len(queries) >= n_queries:
            break

    print(f"    Loaded {len(queries)} query-doc pairs from {n_queries} sampled rows")
    return queries, corpus


def generate_queries_via_llm(
    documents: list[str],
    n_queries: int = 50000,
    endpoint: str = "https://openrouter.ai/api/v1",
    model: str = "meta-llama/llama-3.2-3b-instruct:free",
    api_key: Optional[str] = None,
) -> dict[str, str]:
    """Generate synthetic queries from documents using an LLM API.

    Supports any OpenAI-compatible API (OpenRouter, OpenAI, Together, etc.).
    Example free OpenRouter models:
      - meta-llama/llama-3.2-3b-instruct:free
      - google/gemini-2.0-flash-exp:free
      - mistralai/mistral-small-3.1-24b-instruct:free
    """
    import openai

    client = openai.OpenAI(api_key=api_key or os.getenv("LLM_API_KEY"), base_url=endpoint)

    queries = {}
    for i in tqdm(range(0, n_queries, 10), desc="Generating queries"):
        batch = [documents[j % len(documents)] for j in range(i, min(i + 10, n_queries))]
        for doc in batch:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"Write a search query this document answers perfectly.\nDocument: {doc[:1536]}\nQuery:"}],
                    max_tokens=48,
                    temperature=0.7,
                )
                q = resp.choices[0].message.content.strip().strip('"')
                if q:
                    queries[q] = doc
            except Exception as e:
                words = doc.split()
                if len(words) > 10:
                    q = " ".join(random.sample(words, min(6, len(words) // 2)))
                    queries[q] = doc
    return queries


def mine_hard_negatives(
    queries: dict[str, str],
    corpus: list[str],
    n_negatives: int = 4,
    model_name: str = "sentence-transformers/static-retrieval-mrl-en-v1",
) -> list[dict]:
    """Mine hard negatives using an efficient embedding model (CPU)."""
    embedder = SentenceTransformer(model_name, device="cpu")

    q_texts = list(queries.keys())
    q_embs = embedder.encode(q_texts, show_progress_bar=True, normalize_embeddings=True)
    c_embs = embedder.encode(corpus, show_progress_bar=True, normalize_embeddings=True)

    pairs = []
    for i, (q, pos) in enumerate(tqdm(list(queries.items()), desc="Mining negatives")):
        sims = q_embs[i] @ c_embs.T
        ranked = np.argsort(-sims)
        negs = []
        for idx in ranked:
            if corpus[idx] != pos and len(negs) < n_negatives:
                negs.append(corpus[idx])
        pairs.append({"query": q, "positive": pos, "negatives": negs})

    return pairs


def score_with_teacher(
    pairs: list[dict],
    teacher_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
) -> list[dict]:
    """Score pairs using a teacher reranker for distillation soft labels."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(teacher_name, device="cuda" if torch.cuda.is_available() else "cpu")

    for p in tqdm(pairs, desc="Scoring with teacher"):
        docs = [p["positive"]] + p["negatives"]
        pairs_list = [(p["query"], d) for d in docs]
        scores = model.predict(pairs_list, show_progress_bar=False)
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, -1]
        scores = scores.tolist()
        if isinstance(scores, float):
            scores = [scores]
        scores = [float(s) for s in scores]
        if any(math.isnan(s) or math.isinf(s) for s in scores):
            raise RuntimeError(f"Teacher produced NaN/inf scores for query: {p['query'][:60]}")
        p["teacher_scores"] = scores

    return pairs


def main(
    output_path: str = "data/synthetic_training_data.jsonl",
    corpus_name: str = "sentence-transformers/gooaq",
    n_queries: int = 50000,
    n_negatives: int = 4,
    llm_endpoint: Optional[str] = None,
    llm_model: str = "meta-llama/llama-3.2-3b-instruct:free",
):
    """
    Generate training data for knowledge distillation.

    Args:
        output_path: Output JSONL file
        corpus_name: HuggingFace dataset name with query-doc pairs
        n_queries: Number of query-doc pairs to sample
        n_negatives: Hard negatives per query
        llm_endpoint: API base URL for LLM query generation (optional).
                       If set, generates synthetic queries instead of using
                       dataset's existing queries. Supports OpenRouter,
                       OpenAI, Together, etc.
                       Examples:
                         https://openrouter.ai/api/v1
                         https://api.openai.com/v1
        llm_model: Model name for LLM generation.
                   OpenRouter free: meta-llama/llama-3.2-3b-instruct:free
                   OpenAI: gpt-4o-mini
    """
    print(f"Loading corpus from {corpus_name}...")
    if llm_endpoint:
        dataset = load_dataset(corpus_name, split="train")
        docs = dataset["answer"] if "answer" in dataset.column_names else dataset["text"]
        print(f"    Generating {n_queries} synthetic queries via {llm_endpoint} ({llm_model})...")
        queries = generate_queries_via_llm(
            docs, n_queries=n_queries,
            endpoint=llm_endpoint, model=llm_model,
        )
        corpus = docs
    else:
        queries, corpus = load_queries(corpus_name, n_queries=n_queries)

    print(f"Mining {n_negatives} hard negatives per query...")
    pairs = mine_hard_negatives(queries, corpus, n_negatives=n_negatives)

    print("Scoring with teacher model...")
    pairs = score_with_teacher(pairs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(pairs)} examples to {output_path}")
    print(f"  Each example: query + positive + {n_negatives} negatives + teacher scores")

    sample = pairs[0]
    ts = sample["teacher_scores"]
    assert isinstance(ts, list) and len(ts) == n_negatives + 1, f"Bad score count: {ts}"
    assert all(math.isfinite(s) for s in ts), f"Non-finite teacher score: {ts}"
    spread = max(ts) - min(ts)
    if spread < 1e-3:
        print("  WARNING: teacher scores have near-zero spread (set has no ranking signal)")
    print(f"  Sample teacher scores (pos first): {[round(s, 4) for s in ts]}")
    print(f"  Spread: {spread:.4f} (higher = better ranking signal)")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
