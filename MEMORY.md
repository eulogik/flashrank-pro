# FlashRank-Pro — Memory & Handoff Log

> **Living document.** Append new sessions at the top. Never delete history.
> Last updated: 2026-07-20

---

## Session 003 — Notebook hardened for private repo

**Date:** 2026-07-20

### Changes
- **GH_TOKEN support:** Notebook now accepts `GH_TOKEN` secret for cloning private repos
- **Fallback upload:** If no GH_TOKEN, prompts user to upload repo zip directly via Colab's file upload widget
- **Public clone fallback:** Tries anonymous clone first (will fail for private, but supports SSH auth)
- **Self-contained merge cell:** Stage 4 now defines all its own paths instead of relying on kernel variables from previous cells
- **Clearer instructions:** Header doc now explains exactly what each secret is for and why

### How to use with private repo
1. Download notebook from local clone, upload to Colab
2. Add `GH_TOKEN` in Secrets (optional but recommended)
3. Runtime → Change runtime type → T4 GPU
4. Runtime → Run all

### State
- Notebook validated (all cells pass basic Python syntax + brace balance checks)
- Last commit: `ef89395`

---

## Session 002 — Colab Notebook & Git Push

**Date:** 2026-07-20

### What Was Built
- `notebooks/FlashRank_Pro_Training.ipynb` — bulletproof resumable notebook
- README upgraded with beautiful badges, comparison tables, architecture diagrams
- Eulogik branding added throughout (readme, deploy scripts, OpenRouter config)
- Pushed to `eulogik/flashrank-pro` under org

### Notebook Design Decisions
- **Resumable:** Each cell checks Drive for previous output before running. Runtime → Run all picks up where it left off.
- **Drive-backed:** All data/models saved to `MyDrive/flashrank-pro/` after each stage
- **Run-all safe:** No manual steps between cells — dependencies, mount, clone, resume all automatic
- **LORA_FLAG fix:** `--use_lora` only passed as flag for `large` model (not as `true`/`false` string)
- **Secrets-based auth:** Uses Colab's 🔑 Secrets panel for GH_TOKEN, LLM_API_KEY (optional), and HF_TOKEN

### Notebook Cell Map

| Cell | Stage | Resume Check | Drive Save |
|------|-------|-------------|------------|
| 0 | Setup (deps, mount, clone, restore) | DEPS_INSTALLED marker ✅ | Restores data/ and models/ |
| 1 | Stage 1: Data generation | `data/synthetic_training_data.jsonl` exists | Copied to Drive |
| 2 | Stage 2: KD | `models/*-kd-en/` exists | Copied to Drive |
| 3 | Stage 2b: Multilingual KD | `models/*-kd-multilingual/` exists | Copied to Drive |
| 4 | Stage 3: GRPO RL | `models/*-rl/` exists | Copied to Drive |
| 5 | Stage 4: SLERP merge | `models/*-merged/` exists | Copied to Drive |
| 6 | Quick sanity eval | Merged model exists | N/A |
| 7 | HF deploy | Merged model exists + HF_TOKEN | N/A |

### State at Handoff
- Repo is live at `github.com/eulogik/flashrank-pro` (private)
- All 20 files committed
- Notebook ready for direct upload to Colab
- Next: run Stage 1 (free, uses HuggingFace datasets), then Stages 2-3 on T4

---

## Session 001 — Project Scaffold & Strategic Definition

**Date:** 2026-07-20
**Who:** Agent (opencode/deepseek-v4-flash-free) + User

### Summary
Initial project creation. Deep research on reranker landscape → strategy definition → full scaffold built.

### What Was Built

| File | Status | Notes |
|------|--------|-------|
| `AGENTS.md` | ✅ Created | Complete project guide — vision, architecture, research, conventions, pitfalls |
| `MEMORY.md` | ✅ Created | This file — handoff document |
| `README.md` | ✅ Created | Public-facing quick start |
| `setup.py` | ✅ Created | Package definition |
| `requirements.txt` | ✅ Created | All dependencies pinned |
| `.gitignore` | ✅ Created | Ignores models/, data/, results/, .env |
| `configs/training_config.yaml` | ✅ Created | Single source of truth for hyperparams |
| `configs/slerp_config.json` | ✅ Created | Merge weights for 4 checkpoints |
| `flashrank_pro/__init__.py` | ✅ Created | Exports Reranker, FlashRankProConfig, __version__ |
| `flashrank_pro/model.py` | ✅ Created | FlashRankPro (nn.Module) + Reranker (user-facing) |
| `training/__init__.py` | ✅ Created | Empty |
| `training/01_generate_synthetic_data.py` | ✅ Created | LLM query gen + hard negative mining + teacher scoring |
| `training/02_knowledge_distillation.py` | ✅ Created | Hybrid Pointwise-MSE + Margin-MSE, Accelerate, fp16 |
| `training/03_grpo_rl.py` | ✅ Created | GRPO with K=8, pairwise reward, Yes/No logit scoring |
| `training/04_slerp_merge.py` | ✅ Created | Sequential SLERP merging |
| `scripts/train_full_pipeline.sh` | ✅ Created | End-to-end runner for all 4 stages |
| `scripts/evaluate.py` | ✅ Created | BEIR + MTEB evaluation harness |
| `scripts/deploy_to_huggingface.py` | ✅ Created | HF Hub upload |
| `scripts/register_openrouter.py` | ✅ Created | OpenRouter config generator |
| `scripts/launch_colab.py` | ✅ Created | Colab session instructions per stage |

### Key Decisions Made

1. **ModernBERT over Qwen3-0.6B** — Encoder-only is strictly better for cross-encoder. 4x less params for same quality.
2. **GRPO over PPO** — No critic model needed. Fits T4 memory. ProRank paper validates it for SLM rerankers.
3. **Hybrid loss (Pointwise-MSE + Margin-MSE)** — DisRanker paper shows both absolute accuracy + ranking signal needed.
4. **SLERP over linear interpolation** — Querit-Reranker proved SLERP preserves directional info in parameter space.
5. **fp16 over bf16** — T4 doesn't support bf16. All scripts use fp16 for Colab compatibility.
6. **LoRA for large, full FT for base** — ModernBERT-base (149M) fits T4 at batch 8. ModernBERT-large (395M) needs LoRA.
7. **Apache 2.0** — Must be Apache 2.0 to maximize adoption and allow OpenRouter listing.

### Training Data Strategy
- **Corpus:** `sentence-transformers/gooaq` (Q&A, 3M pairs) — used by Sentence-Transformers blog for ModernBERT reranker
- **Queries:** Loaded from HuggingFace datasets (GooAQ has 3M question→answer pairs), free
- **Negatives:** 4 hard negatives per query via embedding retriever
- **Teacher:** `mixedbread-ai/mxbai-rerank-large-v2` (1.5B) for soft labels
- **Alternative teacher:** `Qwen/Qwen3-Reranker-8B` — larger but free and Apache 2.0

### Open Questions / Future Work
- [ ] Should we use a multilingual corpus for Stage 1 or just English?
- [ ] Should we add pairwise ranking loss (PRD) as an additional loss term?
- [ ] Need to test actual T4 memory usage for ModernBERT-base FT (batch 8)
- [ ] Determine optimal margin_beta in hybrid loss (currently 1.0)
- [ ] Evaluate whether `Lion` optimizer beats `AdamW` for ModernBERT (paper suggests yes)
- [ ] Build Colab notebooks (currently just script-based)
- [ ] Write model card for HuggingFace
- [ ] Add LangChain integration PR
- [ ] Add LlamaIndex integration PR
- [ ] Create benchmark comparison table for README

### Research Sources Used
The following were analyzed to formulate this strategy:
- AIMultiple Reranker Benchmark (2026) — 8 models, 145K Amazon reviews
- Mixedbread mxbai-rerank-v2 blog — RL training for rerankers
- Querit-Reranker paper (Zhong et al., 2026) — SLERP merging + synthetic data
- ProRank paper (Li et al., 2025, ACL Findings) — GRPO for SLM rerankers
- Distillation vs Contrastive Learning (Xu et al., 2025, IJCNLP)
- DisRanker (Ye et al., 2024) — Hybrid pointwise + margin MSE loss
- InRanker (Laitz et al., 2024) — Two-phase distillation
- PRD (Oosterhuis, 2025) — 2% pair sampling for full performance
- ModernBERT paper (Warner et al., 2024) — Architecture
- gte-reranker-modernbert-base — Existing ModernBERT reranker (no RL)
- Sentence-Transformers blog on ModernBERT reranker training (March 2025)
- Lion vs AdamW for cross-encoders (Kumar et al., 2025)

### State at Handoff
- No training has been run yet
- No data has been generated
- All code is in scaffold state
- Project is ready for Stage 1 execution
