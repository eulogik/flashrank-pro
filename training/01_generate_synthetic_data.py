"""
Stage 1: Generate synthetic training data using an LLM.
- Generate 50K queries from diverse document corpora
- Mine hard negatives via embedding retrieval
- Score pairs with teacher reranker for distillation labels

Cost: ~$7.50 via GPT-4o-mini, or free via open-source LLM.
"""

import json
import os
import random
from typing import Optional

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_queries(
    documents: list[str],
    n_queries: int = 50000,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    batch_size: int = 10,
) -> dict[str, str]:
    """Generate synthetic queries from documents using an LLM."""
    import openai

    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    queries = {}
    for i in tqdm(range(0, n_queries, batch_size), desc="Generating queries"):
        batch_docs = [documents[j % len(documents)] for j in range(i, min(i + batch_size, n_queries))]
        prompts = [
            {
                "role": "user",
                "content": (
                    f"Generate a realistic search query that this document would be the perfect answer for.\n"
                    f"Document: {doc[:2048]}\n"
                    f"Output only the query, nothing else."
                ),
            }
            for doc in batch_docs
        ]
        try:
            for doc, prompt in zip(batch_docs, prompts):
                resp = client.chat.completions.create(model=model, messages=[prompt], max_tokens=64, temperature=0.8)
                q = resp.choices[0].message.content.strip()
                queries[q] = doc
        except Exception as e:
            print(f"API error: {e}, using fallback random queries")
            for j, doc in enumerate(batch_docs):
                words = doc.split()
                if len(words) > 10:
                    q = " ".join(random.sample(words, min(8, len(words) // 2)))
                    queries[q] = doc
    return queries


def mine_hard_negatives(
    queries: dict[str, str],
    corpus: list[str],
    n_negatives: int = 4,
    model_name: str = "sentence-transformers/static-retrieval-mrl-en-v1",
) -> list[dict]:
    """Mine hard negatives using an efficient embedding model."""
    embedder = SentenceTransformer(model_name, device="cpu")

    query_texts = list(queries.keys())
    q_embs = embedder.encode(query_texts, show_progress_bar=True, normalize_embeddings=True)
    c_embs = embedder.encode(corpus, show_progress_bar=True, normalize_embeddings=True)

    pairs = []
    for i, (q, pos_doc) in enumerate(tqdm(list(queries.items()), desc="Mining negatives")):
        sims = q_embs[i] @ c_embs.T
        ranked = np.argsort(-sims)
        negatives = []
        for idx in ranked:
            if corpus[idx] != pos_doc and len(negatives) < n_negatives:
                negatives.append(corpus[idx])
            if len(negatives) >= n_negatives:
                break
        pairs.append({"query": q, "positive": pos_doc, "negatives": negatives})

    return pairs


def score_with_teacher(
    pairs: list[dict],
    teacher_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
) -> list[dict]:
    """Score pairs using a teacher reranker for distillation labels."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher_name,
        torch_dtype="float16",
        device_map="auto",
    )
    model.eval()

    import torch

    for p in tqdm(pairs, desc="Scoring with teacher"):
        all_docs = [p["positive"]] + p["negatives"]
        texts = [f"{p['query']} {tokenizer.sep_token or '[SEP]'} {doc}" for doc in all_docs]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        p["teacher_scores"] = scores

    return pairs


def main(
    output_path: str = "data/synthetic_training_data.jsonl",
    corpus_name: str = "sentence-transformers/gooaq",
    n_queries: int = 50000,
    n_negatives: int = 4,
):
    print(f"Loading corpus from {corpus_name}...")
    dataset = load_dataset(corpus_name, split="train")
    documents = dataset["answer"] if "answer" in dataset.column_names else dataset["text"]

    print(f"Generating {n_queries} synthetic queries...")
    queries = generate_queries(documents, n_queries=n_queries)

    print("Mining hard negatives...")
    pairs = mine_hard_negatives(queries, documents, n_negatives=n_negatives)

    print("Scoring with teacher model...")
    pairs = score_with_teacher(pairs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(pairs)} examples to {output_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
