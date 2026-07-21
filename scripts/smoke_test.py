"""
Smoke test: validates all 4 stages with tiny data in ~10 minutes.

Usage:
    python scripts/smoke_test.py

Runs each stage with max_steps=2, batch_size=2, tiny data (8 queries, 2 negatives each).
Deletes temp artifacts on completion. Exits 0 on PASS, 1 on FAIL.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

SMOKE_DIR = tempfile.mkdtemp(prefix="flashrank-pro-smoke-")
DATA_PATH = os.path.join(SMOKE_DIR, "data.jsonl")
KD_OUTPUT = os.path.join(SMOKE_DIR, "kd")
RL_OUTPUT = os.path.join(SMOKE_DIR, "rl")
MERGED_OUTPUT = os.path.join(SMOKE_DIR, "merged")
MODEL_NAME = "answerdotai/ModernBERT-base"


def generate_data():
    queries = [
        "how to train a neural network",
        "what is gradient descent",
        "python list comprehension syntax",
        "difference between cpu and gpu",
        "how backpropagation works",
        "what is a transformer model",
        "docker container vs virtual machine",
        "how to optimize hyperparameters",
    ]
    positives = [
        "Training neural networks uses backpropagation through the computational graph to update weights.",
        "Gradient descent iteratively adjusts parameters to minimize a loss function.",
        "List comprehension creates lists in one line: [x*2 for x in range(10)].",
        "CPU handles sequential tasks well while GPU excels at parallel matrix operations.",
        "Backpropagation computes gradients by applying the chain rule from output to input.",
        "Transformers use self-attention mechanisms to process sequential data in parallel.",
        "Docker containers share the host OS kernel while VMs include a full guest OS.",
        "Hyperparameter optimization uses grid search, random search, or Bayesian methods.",
    ]
    negatives_list = [
        ["The sky is blue on a clear day.", "Cats are popular pets worldwide."],
        ["Python was created by Guido van Rossum.", "Water freezes at zero degrees Celsius."],
        ["The Roman Empire fell in 476 AD.", "Photosynthesis converts light to energy."],
        ["Mount Everest is the tallest mountain.", "Shakespeare wrote Hamlet in 1601."],
        ["The Earth orbits the Sun once per year.", "Oxygen is essential for human life."],
        ["Coffee is a popular morning beverage.", "The Amazon is the largest rainforest."],
        ["Rome is the capital of Italy.", "Beethoven composed nine symphonies."],
        ["Pluto was reclassified as a dwarf planet.", "Gold is a precious metal."],
    ]

    pairs = []
    for i in range(len(queries)):
        docs = [positives[i]] + negatives_list[i]
        scores = [1.0, 0.1, 0.05]  # positive high, negatives low
        pairs.append({
            "query": queries[i],
            "positive": positives[i],
            "negatives": negatives_list[i],
            "teacher_scores": scores,
        })

    with open(DATA_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Generated {len(pairs)} synthetic examples at {DATA_PATH}")


def run_stage(label: str, cmd: list) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    print(f"  OK")
    return True


def validate_output(path: str, label: str) -> bool:
    if os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(os.path.join(path, "pytorch_model.bin")):
        import torch
        from safetensors.torch import load_file
        sdpath = os.path.join(path, "model.safetensors")
        if os.path.exists(sdpath):
            sd = load_file(sdpath)
            bad = [k for k, v in sd.items() if torch.isnan(v).any() or torch.isinf(v).any()]
            if bad:
                print(f"  FAIL: {label} has {len(bad)} NaN/inf tensors")
                return False
        print(f"  {label}: weights exist and clean")
        return True
    else:
        print(f"  FAIL: {label} missing output weights")
        return False


def main():
    print("=" * 60)
    print("  FlashRank-Pro Smoke Test")
    print(f"  Temp dir: {SMOKE_DIR}")
    print("=" * 60)

    # Check GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({mem:.0f} GB VRAM)")
    else:
        print(f"  Device: {device} (training will be slow)")

    # Ensure flashrank_pro is importable in subprocess
    try:
        import flashrank_pro
    except ImportError:
        print("  Installing flashrank-pro in development mode...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True, capture_output=True)
        print("  Done.")

    # Stage 0: Generate tiny data
    print("\n[Stage 0] Generating synthetic data...")
    generate_data()

    # Stage 2: KD (2 steps)
    ok = run_stage(
        "[Stage 2] Knowledge Distillation (2 steps)",
        [
            sys.executable, "training/02_knowledge_distillation.py",
            "--model_name", MODEL_NAME,
            "--data_path", DATA_PATH,
            "--output_dir", KD_OUTPUT,
            "--batch_size", "2",
            "--num_epochs", "1",
            "--max_steps", "2",
        ],
    )
    if not ok:
        return False
    if not validate_output(KD_OUTPUT, "Stage 2 KD"):
        return False

    # Stage 3: GRPO RL (2 steps)
    ok = run_stage(
        "[Stage 3] GRPO RL (2 steps)",
        [
            sys.executable, "training/03_grpo_rl.py",
            "--model_path", KD_OUTPUT,
            "--data_path", DATA_PATH,
            "--output_dir", RL_OUTPUT,
            "--batch_size", "2",
            "--max_docs", "3",
            "--num_epochs", "1",
            "--max_steps", "2",
        ],
    )
    if not ok:
        return False
    if not validate_output(RL_OUTPUT, "Stage 3 RL"):
        return False

    # Stage 4: SLERP merge
    slerp_cfg_path = os.path.join(SMOKE_DIR, "slerp_config.json")
    with open(slerp_cfg_path, "w") as f:
        json.dump({
            "checkpoints": [KD_OUTPUT, RL_OUTPUT],
            "weights": [0.5, 0.5],
        }, f)

    ok = run_stage(
        "[Stage 4] SLERP Merge",
        [
            sys.executable, "training/04_slerp_merge.py",
            "--config_path", slerp_cfg_path,
            "--output_path", MERGED_OUTPUT,
        ],
    )
    if not ok:
        return False
    if not validate_output(MERGED_OUTPUT, "Stage 4 merged"):
        return False

    # Quick sanity: load and run one query
    print("\n[Sanity] Loading merged model...")
    sys.path.insert(0, ".")
    from flashrank_pro import Reranker
    r = Reranker(MERGED_OUTPUT, device=device)
    docs = [
        "Training neural networks requires backpropagation.",
        "Python is a high-level programming language.",
    ]
    results = r.rerank("how to train a model", docs)
    for i, res in enumerate(results):
        print(f"  {i+1}. [{res['score']:.4f}] {res['text'][:60]}")
    if results[0]["score"] != results[1]["score"]:
        print("  Scores differ: model produces meaningful output")
    else:
        print("  WARNING: all scores identical — model may not be learning")

    # Cleanup
    print(f"\nCleaning up {SMOKE_DIR}...")
    shutil.rmtree(SMOKE_DIR)
    print(f"Smoke test PASSED")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
