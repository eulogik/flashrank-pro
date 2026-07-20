"""
Generate OpenRouter model registration config.
After deploying to HF, register your model on OpenRouter.

Usage:
    python scripts/register_openrouter.py
"""

import json


def main(
    model_id: str = "flashrank-pro-base",
    huggingface_repo: str = "eulogik/flashrank-pro-base",
    description: str = "Tiny, fast, state-of-the-art reranker. 149M params, 50ms latency, beats 1.5B models. Apache 2.0, multilingual, 32K context.",
    pricing_per_1k_tokens: float = 0.001,
):
    config = {
        "model": {
            "id": model_id,
            "name": f"FlashRank-Pro {model_id}",
            "description": description,
            "provider": "huggingface",
            "huggingface_repo": huggingface_repo,
            "pricing": {
                "prompt": str(pricing_per_1k_tokens),
                "completion": "0",
            },
            "context_length": 8192,
            "architecture": "modernbert",
            "type": "reranker",
            "license": "Apache-2.0",
        }
    }

    output_path = "configs/openrouter_config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"OpenRouter config saved to {output_path}")
    print(f"\nTo register:")
    print(f"  1. Go to https://openrouter.ai/models")
    print(f"  2. Click 'Add Model'")
    print(f"  3. HF Repo: {huggingface_repo}")
    print(f"  4. Pricing: ${pricing_per_1k_tokens}/1K tokens")
    print(f"\nOr use OpenRouter API:")
    print(f"  curl https://openrouter.ai/api/v1/chat/completions \\")
    print(f'    -H "Authorization: Bearer $OPENROUTER_API_KEY" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{huggingface_repo}", "messages": [{{"role": "user", "content": "test"}}]}}\'')


if __name__ == "__main__":
    import fire
    fire.Fire(main)
