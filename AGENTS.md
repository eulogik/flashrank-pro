# FlashRank-Pro — Agent Guide

## 1. Project Identity

**Vision:** Build the most parameter-efficient reranker in the world. 149M parameters beating 1.5B models. Apache 2.0. 50ms latency. CPU-friendly.

**Why this wins:** The market gap is clear — nobody has combined ModernBERT (most efficient encoder arch) + GRPO reinforcement learning + multi-stage knowledge distillation + SLERP checkpoint merging into a single Apache 2.0 package. The current best (mxbai-rerank-v2, Qwen3-Reranker, Querit-Reranker) all use decoder-only architectures that waste parameters on causal LM when all you need is relevance scoring. ModernBERT is bidirectional and 8x smaller for the same quality.

**Key differentiators:**
- 149M params (ModernBERT-base) targeting 1.5B-model territory
- 395M params (ModernBERT-large) targeting 4B-model territory
- 50-120ms inference — fastest on any platform
- CPU inference under 100ms — no GPU required
- 32K context window
- 100+ language multilingual
- Function call / code / MCP support
- Apache 2.0 — zero restrictions

---

## 2. Architecture

### Model
- **Base:** `answerdotai/ModernBERT-base` (149M params, bidirectional encoder)
- **Large:** `answerdotai/ModernBERT-large` (395M params)
- **Head:** Single linear classification layer on [CLS] token → scalar relevance score
- **Output:** `sigmoid(logits)` → relevance probability [0, 1]
- **Max length:** 8,192 tokens (32K compatible with architectural changes)

### Training Pipeline (4 Stages)

```
Stage 1: Data Generation ───→ Stage 2: KD ───→ Stage 3: RL ───→ Stage 4: Merge
     (LLM queries)         (Teacher→Student)   (GRPO)          (SLERP)
```

| Stage | File | What | Hardware | Time | Cost |
|-------|------|------|----------|------|------|
| 1 | `01_generate_synthetic_data.py` | Load query-doc pairs from HF dataset (free), mine hard negatives, score with teacher | Any (CPU) | ~30min | $0 |
| 2 | `02_knowledge_distillation.py` | Distill from mxbai-rerank-large-v2 or Qwen3-Reranker-8B into ModernBERT | Colab T4 (16GB) | ~3h (base), ~3h (large LoRA) | $0 |
| 3 | `03_grpo_rl.py` | GRPO prompt warmup + fine-grained scoring with LoRA | Colab T4 (16GB) | ~2h | $0 |
| 4 | `04_slerp_merge.py` | SLERP merge of KD + RL + multilingual checkpoints | Any (CPU) | 5min | $0 |

### Inference
```python
from flashrank_pro import Reranker
r = Reranker("flashrank-pro-base")
results = r.rerank(query="how to train a model", docs=[...], top_k=5)
```

---

## 3. Research Backing (Papers That Justify Every Decision)

Every design choice is grounded in published research. Cite these when questioned.

### Architecture Choice — ModernBERT
- **ModernBERT paper** (Warner et al., 2024) — `answerdotai/ModernBERT` — Encoder-only, 2x faster than BERT, supports 8K context natively, rotary embeddings, GeGLU activations, unbiased layer norm.
- **Benchmark evidence:** gte-reranker-modernbert-base (149M) achieves identical Hit@1 to nemotron-rerank-1b (1.2B) on Amazon reviews benchmark (source: AIMultiple benchmark, 2026). First result accuracy is what matters for RAG.

### Knowledge Distillation Strategy
- **"Distillation vs Contrastive Learning"** (Xu et al., 2025, IJCNLP) — KD consistently beats contrastive learning for student models <7B when teacher is stronger. Our teacher is 1.5B/8B → student is 149M.
- **"Best Practices for Distilling LLMs into BERT"** (Ye et al., 2024) — Hybrid pointwise-MSE + Margin-MSE loss is optimal for reranker distillation. We implement this exactly.
- **InRanker** (Laitz et al., 2024) — Two-phase distillation (soft human labels → soft synthetic labels) improves zero-shot. We mirror this.

### RL Fine-Tuning
- **ProRank** (Li et al., 2025, ACL Findings) — SLMs (0.5B) with GRPO + fine-grained score learning beat 32B LLM rerankers on BEIR. We use K=8 samples, relative reward normalization, Yes/No logit ratio scoring.
- **mxbai-rerank-v2** (Lee et al., 2025) — Three-phase RL (GRPO + contrastive + preference) lifts BEIR by 8+ points over v1.

### Checkpoint Merging
- **Querit-Reranker** (Zhong et al., 2026) — SLERP merging of task-specific checkpoints yields SOTA on MTEB Multilingual v2 (71.08) without ensemble overhead.

### Synthetic Data
- **TWOLAR** (Baldelli et al., 2024) — 20K synthetic queries from 4 diverse retrievers + LLM reranking = model matching 1000x larger models.
- **PRD** (Oosterhuis, 2025) — Pairwise distillation with 2% pair sampling achieves full-pair performance. We can scale this approach.

---

## 4. Project Structure

```
reranker/
├── AGENTS.md                    ← You are here
├── MEMORY.md                    ← Living handoff document
├── README.md                    ← Public-facing README
├── setup.py                     ← Package definition
├── requirements.txt             ← Dependencies
├── .gitignore
├── configs/
│   ├── training_config.yaml     ← All hyperparameters in one place
│   └── slerp_config.json        ← SLERP merge weights
├── flashrank_pro/
│   ├── __init__.py              ← Package init, exports Reranker
│   └── model.py                 ← FlashRankPro nn.Module, Reranker class
├── training/
│   ├── __init__.py
│   ├── 01_generate_synthetic_data.py
│   ├── 02_knowledge_distillation.py
│   ├── 03_grpo_rl.py
│   └── 04_slerp_merge.py
├── scripts/
│   ├── train_full_pipeline.sh   ← End-to-end training runner
│   ├── evaluate.py              ← BEIR/MTEB evaluation
│   ├── deploy_to_huggingface.py ← HF Hub upload
│   ├── register_openrouter.py   ← OpenRouter config generation
│   └── launch_colab.py          ← Colab helper
└── data/                        ← gitignored, synthetic data lives here
```

---

## 5. Code Conventions

### Style
- No comments in code unless absolutely necessary for clarity
- Type hints everywhere
- Dataclasses for config
- `fire.Fire(main)` for CLI entrypoints
- Accelerate for distributed training (not raw DDP)
- `fp16` (not `bf16`) for T4 compatibility

### Training Scripts
Each training script follows this pattern:
```python
def main(
    # required args with defaults
    model_path: str = "answerdotai/ModernBERT-base",
    data_path: str = "data/synthetic_training_data.jsonl",
    output_dir: str = "models/flashrank-pro-base",
    # hyperparameters
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    ...
):
```
- Every hyperparameter is a CLI arg via `fire`
- Accelerate wraps model/optimizer/loader/scheduler
- Main process prints step losses
- `accelerator.wait_for_everyone()` before save
- Only main process saves

### Model Class
- `Reranker` is the user-facing class (simple API)
- `FlashRankPro` is the nn.Module (internal)
- `FlashRankProConfig` is the dataclass holding model hyperparams
- `score()` returns raw sigmoid scores
- `rerank()` returns sorted list of dicts with `text` and `score` keys

---

## 6. Hardware Constraints

| Machine | GPU | VRAM | Usable For |
|---------|-----|------|------------|
| MacBook Air M4 16GB | No GPU (MPS backend, ~40% of CUDA perf) | System RAM shared | Data gen, SLERP merge, light eval, code dev |
| Mac Mini M4 16GB | Same as above | Same | Same as above |
| Colab Free T4 | Tesla T4 (16GB VRAM) | 16GB | KD full FT for base (batch 8, ~12GB), LoRA for large, GRPO RL |

**Critical constraint:** T4 does NOT support bf16. All training scripts use fp16.

**Colab session limits:** 3-4 hours. Each stage is designed to fit in one session:
- Stage 2 (base full FT): ~3h ✓
- Stage 2 (large LoRA FT): ~3h ✓
- Stage 3 (RL LoRA): ~2h ✓
- Stage 1 and 4 are offline/API, not affected.

---

## 7. Benchmark Targets

| Model | Params | BEIR (NDCG@10) | MTEB-R | Latency |
|-------|--------|----------------|--------|---------|
| flashrank-pro-base (target) | 149M | >55 | >65 | ~50ms |
| flashrank-pro-large (target) | 395M | >60 | >70 | ~120ms |
| mxbai-rerank-base-v2 (reference) | 494M | 55.57 | — | 670ms |
| mxbai-rerank-large-v2 (reference) | 1.54B | 57.49 | — | 890ms |
| gte-reranker-modernbert-base (reference) | 149M | — | — | ~60ms |
| Qwen3-Reranker-4B (reference) | 4B | — | 69.76 | ~1000ms |
| Querit-Reranker-4B (reference) | 4B | 62.29 | 71.08 | ~800ms |

**Key comparison:** If flashrank-pro-base hits BEIR >55, it matches mxbai-base-v2 (494M) at 1/3 the params and 13x less latency. If it hits BEIR >57, it matches mxbai-large-v2 (1.5B) at 10x the params and 18x less latency.

---

## 8. Deployment Strategy

### HuggingFace
```
your-org/flashrank-pro-base
your-org/flashrank-pro-large
```
- Model card with benchmark comparisons
- "First 149M reranker beating 1.5B models" narrative
- Apache 2.0 badge
- Direct `CrossEncoder` loading support

### GitHub
- `pip install flashrank-pro`
- 5-line API
- LangChain integration PR
- LlamaIndex integration PR
- "No GPU required" narrative (CPU inference <100ms)

### OpenRouter
- Register as `flashrank-pro-base` and `flashrank-pro-large`
- Pricing: $0.001/1K tokens (10x cheaper than Cohere)
- Fastest model on the platform (50ms)
- Instant adoption from RAG devs hitting OpenRouter for embedding + reranking

---

## 9. List of Files and What Each Does

| File | Responsibility |
|------|---------------|
| `AGENTS.md` | This file — complete project guide for AI agents |
| `MEMORY.md` | Living handoff document — state, decisions, session log |
| `README.md` | Public README — quick start, model family table |
| `setup.py` | Package build config — `pip install -e .` |
| `requirements.txt` | All dependencies |
| `.gitignore` | Standard Python + model/data ignores |
| `configs/training_config.yaml` | Single source of truth for all training hyperparams |
| `configs/slerp_config.json` | Checkpoint paths and merge weights |
| `flashrank_pro/__init__.py` | Exports `Reranker`, `FlashRankProConfig`, `__version__` |
| `flashrank_pro/model.py` | `FlashRankPro` (nn.Module) and `Reranker` (user-facing API) |
| `training/01_generate_synthetic_data.py` | LLM-based synthetic query generation + hard negative mining + teacher scoring |
| `training/02_knowledge_distillation.py` | Hybrid Pointwise-MSE + Margin-MSE distillation from teacher to ModernBERT |
| `training/03_grpo_rl.py` | GRPO reinforcement learning with K=8 sampling, pairwise reward, Yes/No scoring |
| `training/04_slerp_merge.py` | Spherical linear interpolation checkpoint merging |
| `scripts/train_full_pipeline.sh` | Orchestrates all 4 stages end-to-end |
| `scripts/evaluate.py` | BEIR and MTEB reranking evaluation |
| `scripts/deploy_to_huggingface.py` | Uploads model to HuggingFace Hub |
| `scripts/register_openrouter.py` | Generates OpenRouter model registration config |
| `scripts/launch_colab.py` | Colab notebook instructions for cloud training |

---

## 10. Key Implementation Details

### Why Pointwise + Margin-MSE Hybrid Loss
- Pure pointwise MSE loses ranking signal between docs
- Pure margin MSE ignores score calibration
- Hybrid (from DisRanker paper) gives both: absolute score accuracy + relative ordering
- Implementation: `L = MSE(student, teacher) + β * MSE((pos-neg)_student, (pos-neg)_teacher)`

### Why GRPO Instead of PPO
- PPO requires a critic model (doubles memory)
- GRPO uses group-relative rewards (no critic needed)
- ProRank showed GRPO works better for SLM rerankers
- K=8 samples per query-doc gives stable reward estimates

### Why SLERP Instead of Weight Averaging
- Linear interpolation (Model Soups) loses directional information
- SLERP stays on the geodesic path in parameter space
- Querit-Reranker proved SLERP merges complementary checkpoints without performance loss
- Sequential SLERP handles multiple checkpoints by merging one at a time

### Why ModernBERT Instead of Qwen3-0.6B
- ModernBERT is encoder-only (bidirectional); Qwen3-0.6B is decoder-only (causal)
- Cross-encoder needs joint query-doc attention — bidirectional is strictly better
- ModernBERT has 8K native context via RoPE; Qwen3-0.6B requires architecture modification for same
- ModernBERT-base (149M) uses 4x less compute than Qwen3-0.6B (600M) for same quality cross-encoding

---

## 11. Quick Reference Commands

```bash
# Install project
cd /Users/gautamkishore/Code/reranker
pip install -e .

# Smoke test (validates all 4 stages with tiny data, ~5 min)
python scripts/smoke_test.py

# Stage 1: Generate data (run locally, no API key needed)
python training/01_generate_synthetic_data.py --n_queries 50000

# Stage 2: Knowledge Distillation (run on Colab T4)
python training/02_knowledge_distillation.py \
    --model_name answerdotai/ModernBERT-base \
    --batch_size 8 --num_epochs 3

# Stage 3: GRPO RL (run on Colab T4)
python training/03_grpo_rl.py \
    --model_path models/flashrank-pro-base-kd-en \
    --batch_size 4

# Stage 4: SLERP merge (run anywhere, CPU fine)
python training/04_slerp_merge.py --config_path configs/slerp_config.json

# Evaluate
python scripts/evaluate.py --model_path models/flashrank-pro-merged

# Deploy to HF
python scripts/deploy_to_huggingface.py --model_path models/flashrank-pro-merged

# Register on OpenRouter
python scripts/register_openrouter.py

# Full pipeline
bash scripts/train_full_pipeline.sh base
```

---

## 12. Common Pitfalls

1. **T4 doesn't support bf16** — All scripts now use `fp32` (was fp16). fp16 MSE loss silently produces NaN in gradients → corrupts saved weights. fp32 is slower but safe.
2. **Stage 3 save_pretrained saves LoRA adapters, not full model** — `merge_and_unload()` must be called before `save_pretrained()` so Stage 4 SLERP merge gets full weight keys/shapes.
3. **Stage 4 needs safetensors support** — Stages 2/3 save `model.safetensors`, not `pytorch_model.bin`. Stage 4 must handle both.
4. **Notebook clone in Colab** — `https://token@github.com/repo.git` format conflicts with credential managers. Use Python `requests` to download zipball from GitHub API instead.
5. **Colab session timeout** — Stages 2-3 each fit in one session. Stage 1 (data gen) takes ~2.5h — use incremental checkpoints + Drive writes.
6. **ModernBERT large OOM on T4** — Use LoRA (`--use_lora true`) for large. Base fits full FT with batch 8.
7. **Teacher model CPU OOM** — `mxbai-rerank-large-v2` (1.5B) needs ~6GB. `Qwen3-Reranker-8B` needs ~16GB. Use the smaller teacher for local scoring.
8. **OpenAI API rate limits** — Stage 1 generates queries using existing HF datasets (free). Only uses API if `--llm_endpoint` is explicitly passed.
9. **SLERP merge path mismatch** — The pytorch_model.bin or model.safetensors keys must match between checkpoints. Use the same base model config for all.
10. **sentence-transformers CrossEncoder vs our model** — Our `Reranker` class is a thin wrapper. For MTEB/BEIR eval, we use `sentence-transformers.CrossEncoder(model_path)` because those frameworks expect it.
11. **Teacher scoring is slow per-example** — `CrossEncoder.predict()` called per-query (50K calls) → 14h on T4. Fix: collect ALL pairs into one list, single `predict()` call. Batch 64 on T4 with fp16 (`model.model.half()`).
12. **Notebook writes to local, not Drive** — If notebook writes to `/content/...` and only copies to Drive after completion, checkpoints are lost on session break. Fix: Stage 1 writes directly to Drive path.
13. **Notebook restore must always overwrite** — If local `data/` exists from a failed run, the restore logic skips Drive→local copy. Stale local data = wrong state. Fix: always delete local and copy from Drive.
14. **Config model_type missing in merged model** — `AutoModelForSequenceClassification.from_pretrained` fails if `config.json` lacks `model_type`. Fix: `_ensure_model_type()` patches config on disk before loading.
15. **MPS crashes on Qwen2-based models** — `mxbai-rerank-*` uses Qwen2 architecture which has MPS matmul bugs. Cannot run teacher on Mac GPU. Use CPU or Colab T4.
