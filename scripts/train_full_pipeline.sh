#!/bin/bash
# Full training pipeline for FlashRank-Pro
# Run each stage sequentially. Stages 1-3 run on Colab T4 (~3h each).
# Stage 4 runs on CPU in ~5min.
#
# Usage:
#   bash scripts/train_full_pipeline.sh [base|large]

set -e

SIZE=${1:-base}
MODEL_NAME="answerdotai/ModernBERT-${SIZE}"
OUTPUT_PREFIX="models/flashrank-pro-${SIZE}"

echo "=== FlashRank-Pro Training Pipeline ($SIZE) ==="

# Stage 1: Generate synthetic data
echo ""
echo "=== Stage 1: Generating synthetic training data ==="
python training/01_generate_synthetic_data.py \
    --output_path data/synthetic_training_data.jsonl \
    --corpus_name sentence-transformers/gooaq \
    --n_queries 50000 \
    --n_negatives 4

# Stage 2: Knowledge Distillation
echo ""
echo "=== Stage 2: Knowledge Distillation ==="
python training/02_knowledge_distillation.py \
    --model_name $MODEL_NAME \
    --data_path data/synthetic_training_data.jsonl \
    --output_dir ${OUTPUT_PREFIX}-kd-en \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 2e-5

# Stage 2b: Multilingual KD (requires multilingual corpus)
echo ""
echo "=== Stage 2b: Multilingual Knowledge Distillation ==="
python training/02_knowledge_distillation.py \
    --model_name $MODEL_NAME \
    --data_path data/synthetic_training_data.jsonl \
    --output_dir ${OUTPUT_PREFIX}-kd-multilingual \
    --batch_size 8 \
    --num_epochs 2 \
    --learning_rate 2e-5

# Stage 3: GRPO RL
echo ""
echo "=== Stage 3: GRPO RL Fine-Tuning ==="
python training/03_grpo_rl.py \
    --model_path ${OUTPUT_PREFIX}-kd-en \
    --data_path data/synthetic_training_data.jsonl \
    --output_dir ${OUTPUT_PREFIX}-rl \
    --batch_size 4 \
    --k_samples 8 \
    --num_epochs 1

# Stage 4: SLERP Merge
echo ""
echo "=== Stage 4: SLERP Checkpoint Merging ==="
python training/04_slerp_merge.py \
    --config_path configs/slerp_config.json \
    --output_path ${OUTPUT_PREFIX}-merged

echo ""
echo "=== Training Complete! ==="
echo "Final model: ${OUTPUT_PREFIX}-merged"
echo ""
echo "To evaluate: python scripts/evaluate.py --model_path ${OUTPUT_PREFIX}-merged"
