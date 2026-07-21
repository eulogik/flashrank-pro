"""
Stage 4: SLERP (Spherical Linear Interpolation) checkpoint merging.

Merges multiple fine-tuned checkpoints into a single stronger model
without ensembling overhead. Based on Querit-Reranker approach.

Run on CPU. Takes ~5 minutes.
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def slerp(tensor_a: torch.Tensor, tensor_b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Shape mismatch: {tensor_a.shape} vs {tensor_b.shape}")
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()
    dot = (a_flat * b_flat).sum()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    dot = dot / (norm_a * norm_b + 1e-8)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if sin_theta < 1e-8:
        result = (1 - t) * a_flat + t * b_flat
    else:
        result = (torch.sin((1 - t) * theta) / sin_theta) * a_flat + (torch.sin(t * theta) / sin_theta) * b_flat
    return result.reshape(tensor_a.shape).to(tensor_a.dtype)


def merge_checkpoints_slerp(
    checkpoint_paths: list[str],
    weights: Optional[list[float]] = None,
    output_path: str = "models/flashrank-pro-merged",
):
    """
    Merge multiple checkpoints using sequential SLERP.

    Args:
        checkpoint_paths: Paths to model checkpoints
        weights: Weight for each checkpoint (must sum to 1)
        output_path: Output path for merged model
    """
    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    assert len(checkpoint_paths) == len(weights), "Mismatched checkpoints and weights"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    print(f"Merging {len(checkpoint_paths)} checkpoints via SLERP...")
    for ckpt, w in zip(checkpoint_paths, weights):
        print(f"  {ckpt} (weight: {w})")

    merged_state = None
    cumulative_weight = 0.0

    for i, (ckpt_path, w) in enumerate(zip(checkpoint_paths, weights)):
        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        safetensors_path = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        elif os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state = load_file(safetensors_path)
        else:
            raise FileNotFoundError(f"No model weights found in {ckpt_path}")
        if i == 0:
            merged_state = {k: v.clone().float() for k, v in state.items()}
            cumulative_weight = w
        else:
            cumulative_weight += w
            t = w / cumulative_weight
            for key in merged_state:
                merged_state[key] = slerp(merged_state[key], state[key].float(), t)

    os.makedirs(output_path, exist_ok=True)
    merged_state = {k: v.to(torch.float16) for k, v in merged_state.items()}
    torch.save(merged_state, os.path.join(output_path, "pytorch_model.bin"))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_paths[0])
    tokenizer.save_pretrained(output_path)

    config = AutoModelForSequenceClassification.from_pretrained(checkpoint_paths[0]).config
    if not getattr(config, "model_type", None):
        config.model_type = "modernbert"
    config.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")


def main(
    config_path: str = "configs/slerp_config.json",
    output_path: str = "models/flashrank-pro-merged",
):
    with open(config_path) as f:
        config = json.load(f)
    merge_checkpoints_slerp(config["checkpoints"], config.get("weights"), output_path)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
