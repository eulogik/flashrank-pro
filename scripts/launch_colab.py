"""
Generate Colab notebook link for cloud training.
Opens the notebook in Colab with your GitHub repo.

Usage:
    python scripts/launch_colab.py --stage 1
    python scripts/launch_colab.py --stage 2 --model base
"""

import webbrowser
from typing import Optional

NOTEBOOK_TEMPLATES = {
    1: {
        "title": "Stage 1: Generate Synthetic Data",
        "steps": [
            "pip install openai sentence-transformers datasets tqdm",
            "import openai; import os; openai.api_key = 'YOUR_KEY'",
            "!python training/01_generate_synthetic_data.py --n_queries 50000",
            "# Upload the generated data/synthetic_training_data.jsonl to Google Drive",
            "from google.colab import drive; drive.mount('/content/drive')",
            "!cp data/synthetic_training_data.jsonl /content/drive/MyDrive/",
        ],
    },
    2: {
        "title": "Stage 2: Knowledge Distillation",
        "steps": [
            "pip install torch transformers accelerate datasets sentence-transformers tqdm",
            "# Mount Google Drive and copy data",
            "from google.colab import drive; drive.mount('/content/drive')",
            "!cp /content/drive/MyDrive/synthetic_training_data.jsonl data/",
            "# Check GPU: should be Tesla T4",
            "!nvidia-smi",
            "# Run distillation (base: ~3h, large with LoRA: ~3h)",
            "!python training/02_knowledge_distillation.py --model_name answerdotai/ModernBERT-base --batch_size 8",
            "# Save model to Drive",
            "!cp -r models/flashrank-pro-base-kd-en /content/drive/MyDrive/",
        ],
    },
    3: {
        "title": "Stage 3: GRPO RL Fine-Tuning",
        "steps": [
            "pip install torch transformers accelerate datasets peft tqdm",
            "from google.colab import drive; drive.mount('/content/drive')",
            "!cp -r /content/drive/MyDrive/flashrank-pro-base-kd-en models/",
            "!cp /content/drive/MyDrive/synthetic_training_data.jsonl data/",
            "!python training/03_grpo_rl.py --batch_size 4 --k_samples 8",
            "!cp -r models/flashrank-pro-base-rl /content/drive/MyDrive/",
        ],
    },
}


def main(stage: int = 1, repo_url: str = "https://github.com/YOUR_USER/flashrank-pro"):
    if stage in NOTEBOOK_TEMPLATES:
        template = NOTEBOOK_TEMPLATES[stage]
        print(f"=== {template['title']} ===")
        print(f"\nInstructions for Stage {stage}:")
        for i, step in enumerate(template["steps"], 1):
            print(f"  {i}. {step}")

        colab_url = f"https://colab.research.google.com/github/{repo_url.replace('https://github.com/', '')}/blob/main/notebooks/stage{stage}.ipynb"
        print(f"\nOr open in Colab: {colab_url}")
        try:
            webbrowser.open(colab_url)
            print("Browser opened.")
        except Exception:
            pass
    else:
        print(f"Stage {stage} not found. Available: {list(NOTEBOOK_TEMPLATES.keys())}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
