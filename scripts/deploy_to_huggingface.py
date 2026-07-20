"""
Upload model to HuggingFace Hub.

Usage:
    python scripts/deploy_to_huggingface.py --model_path models/flashrank-pro-merged --repo_id your-org/flashrank-pro-base
"""

import os
from typing import Optional

from huggingface_hub import HfApi, create_repo


def main(
    model_path: str = "models/flashrank-pro-merged",
    repo_id: str = "eulogik/flashrank-pro-base",
    private: bool = False,
    token: Optional[str] = None,
):
    token = token or os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Get yours at https://huggingface.co/settings/tokens")
        return

    api = HfApi(token=token)

    try:
        create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"Repository {repo_id} ready.")
    except Exception as e:
        print(f"Repo creation warning: {e}")

    print(f"Uploading {model_path} to {repo_id}...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done! View at https://huggingface.co/{repo_id}")

    print(f"\nUsage:")
    print(f"  from flashrank_pro import Reranker")
    print(f'  reranker = Reranker("{repo_id}")')
    print(f'  results = reranker.rerank("query", docs)')


if __name__ == "__main__":
    import fire
    fire.Fire(main)
