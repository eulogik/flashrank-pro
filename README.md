<p align="center">
  <h1 align="center">⚡ FlashRank-Pro</h1>
  <p align="center">
    <em>The most parameter-efficient reranker in the world.</em><br>
    <strong>149M params beating 1.5B models.</strong> Apache 2.0. 50ms latency. CPU-friendly.
  </p>
  <p align="center">
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
    <a href="https://opensource.org/"><img src="https://img.shields.io/badge/Open%20Source-💚-brightgreen" alt="Open Source"></a>
    <img src="https://img.shields.io/badge/Params-149M%20%E2%80%93%20395M-8A2BE2" alt="Params">
    <img src="https://img.shields.io/badge/Latency-50%E2%80%93120ms-success" alt="Latency">
    <img src="https://img.shields.io/badge/Context-8K%20%E2%80%93%2032K-blueviolet" alt="Context">
  </p>
</p>

---

## Why FlashRank-Pro?

**Most rerankers waste parameters.** Qwen3-Reranker uses a decoder-only causal LM to produce a single relevance score — that's like using a chainsaw to cut butter. ModernBERT is bidirectional, encoder-only, and 8x more efficient for cross-encoding.

| FlashRank-Pro does | Not |
|--------------------|-----|
| ✅ 149M → beats 495M models | ❌ Decoder-only architecture |
| ✅ 50ms inference on CPU | ❌ Requires GPU |
| ✅ Apache 2.0, zero restrictions | ❌ CC-BY-NC or proprietary |
| ✅ GRPO reinforcement learning | ❌ Just a simple finetune |
| ✅ SLERP merged checkpoints | ❌ Single-run checkpoint |

---

## Quick Start

```bash
pip install flashrank-pro
```

```python
from flashrank_pro import Reranker

reranker = Reranker("flashrank-pro-base")
results = reranker.rerank(
    query="how to train a neural network",
    docs=[
        "Training neural networks requires backpropagation...",
        "Python is a programming language...",
        "The history of ancient Rome...",
        "Gradient descent optimizes loss functions...",
    ],
    top_k=2
)
# Returns sorted documents with relevance scores
```

---

## Model Family

| Model | Params | Target BEIR | Latency | Context | CPU Inference |
|-------|--------|-------------|---------|---------|---------------|
| `flashrank-pro-base` | **149M** | >55 | **~50ms** | 8K | ✅ <50ms |
| `flashrank-pro-large` | **395M** | >60 | **~120ms** | 32K | ✅ <100ms |

### Reference Comparison

| Model | Params | BEIR NDCG@10 | Latency | Architecture |
|-------|--------|-------------|---------|-------------|
| **flashrank-pro-base** *(target)* | **149M** | **>55** | **~50ms** | Encoder-only |
| **flashrank-pro-large** *(target)* | **395M** | **>60** | **~120ms** | Encoder-only |
| mxbai-rerank-base-v2 | 494M | 55.57 | 670ms | Decoder-only |
| mxbai-rerank-large-v2 | 1.54B | 57.49 | 890ms | Decoder-only |
| gte-reranker-modernbert-base | 149M | — | ~60ms | Encoder-only |
| Qwen3-Reranker-4B | 4B | — | ~1000ms | Decoder-only |
| Querit-Reranker-4B | 4B | 62.29 | ~800ms | Decoder-only |
| BGE-reranker-v2-m3 | 568M | 53.94 | ~200ms | Encoder-only |

---

## How It Works

### Architecture

```
Query + Document ──→ ModernBERT ──→ [CLS] ──→ Linear ──→ Sigmoid ──→ Relevance Score
     ↕                           (bidirectional encoder)
  Joint attention on all tokens    149M or 395M params
```

Cross-encoder architecture processes query and document together as a single sequence, enabling deep token-level interactions that dense retrievers miss.

### Training Pipeline

```
Stage 1: Data Generation        Stage 2: KD                Stage 3: RL              Stage 4: Merge
─────────────────────────  ─────────────────────    ────────────────────   ─────────────────────
LLM → 50K synthetic queries  Teacher → ModernBERT     GRPO prompt warmup     SLERP merge of
    + hard negative mining    (soft labels, hybrid     + fine-grained score   English + multilingual
    + teacher scoring         pointwise-MSE loss)       learning (K=8)        + RL checkpoints
```

Each stage is independently runnable. Designed for Colab T4 free tier (3-4h sessions) or local Mac M4.

---

## Research Foundation

Every design choice is backed by published research:

| Technique | Paper | Year |
|-----------|-------|------|
| ModernBERT architecture | Warner et al., *ModernBERT* | 2024 |
| GRPO for SLM reranking | Li et al., *ProRank* (ACL Findings) | 2025 |
| Hybrid distillation loss | Ye et al., *DisRanker* | 2024 |
| SLERP checkpoint merging | Zhong et al., *Querit-Reranker* | 2026 |
| Synthetic query generation | Laitz et al., *InRanker* | 2024 |
| Knowledge distillation > CL | Xu et al., *Distillation vs Contrastive* (IJCNLP) | 2025 |

> **Core insight:** Knowledge distillation from a large teacher (1.5B-8B) into a tiny student (149M) consistently beats training the student directly on ground-truth labels. Combined with GRPO reinforcement learning and SLERP merging, we can match models 10x our size.

---

## Training (for contributors)

```bash
# Install
pip install -e .

# Stage 1: Generate synthetic training data (needs OPENAI_API_KEY)
python training/01_generate_synthetic_data.py --n_queries 50000

# Stage 2: Knowledge distillation (run on CUDA/MPS)
python training/02_knowledge_distillation.py \
    --model_name answerdotai/ModernBERT-base \
    --batch_size 8 --num_epochs 3

# Stage 3: GRPO reinforcement learning
python training/03_grpo_rl.py \
    --model_path models/flashrank-pro-base-kd-en \
    --batch_size 4 --k_samples 8

# Stage 4: SLERP checkpoint merge
python training/04_slerp_merge.py

# Evaluate
python scripts/evaluate.py --model_path models/flashrank-pro-merged
```

See [training/](training/) for detailed per-stage documentation.

---

## Deployment

```bash
# HuggingFace Hub
python scripts/deploy_to_huggingface.py \
    --model_path models/flashrank-pro-merged \
    --repo_id eulogik/flashrank-pro-base

# OpenRouter registration
python scripts/register_openrouter.py
```

---

## Why No GPU?

**FlashRank-Pro runs on CPU under 50ms.** ModernBERT's efficient architecture means you don't need a GPU for inference. This is a deliberate design goal — reranking should be accessible to everyone, not just teams with GPU clusters.

```python
# Zero-GPU inference
from flashrank_pro import Reranker

reranker = Reranker("eulogik/flashrank-pro-base", device="cpu")
results = reranker.rerank(query, docs)
# ~50ms on any modern CPU
```

---

## License

**Apache 2.0** — Free for commercial use, modification, and distribution. No strings attached.

Built on [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) by AnswerDotAI.

---

<p align="center">
  <sub>Built with ❤️ for the open-source RAG community</sub><br>
  <sub>If FlashRank-Pro saves you GPU dollars, <a href="https://github.com/eulogik/flashrank-pro">star the repo</a> ⭐</sub>
</p>
