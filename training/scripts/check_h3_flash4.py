"""Check native H3+LoRA+checkpointing outputs and gradients against FA4."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from diffusers import MiniMaxH3Transformer3DModel

from video_blade.h3.attention import use_flash4
from video_blade.h3.flow import H3Layout
from video_blade.h3.models import configure_role, native_dtypes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(321)
    teacher = native_dtypes(
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=128,
            hidden_size=64,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=128,
            text_dim=16,
            freq_dim=16,
            time_embed_hidden_dim=64,
            time_embed_dim=32,
            rope_freq_dim=4,
        )
    ).cuda()
    model = configure_role(teacher, "student", rank=4, alpha=4, checkpointing=True)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.01)
    candidate = deepcopy(model)
    model.set_attention_backend("_native_cudnn")
    use_flash4(candidate)
    layout = H3Layout(64, 64, 22, 5, "cuda")
    samples = layout.noise()
    embeds = torch.randn(1, 5, 16, device="cuda", dtype=torch.bfloat16)
    sigmas = (torch.tensor(0.7, device="cuda"), torch.tensor(0.3, device="cuda"))
    results = []
    for network in (model, candidate):
        outputs = layout.predict(network, samples, sigmas, embeds)
        sum(x.square().mean() for x in outputs).backward()
        grads = {n: p.grad for n, p in network.named_parameters() if p.requires_grad}
        if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
            raise AssertionError("Missing or nonfinite LoRA gradient")
        results.append((outputs, grads))
    errors = {}
    for key in results[0][1]:
        reference, actual = results[0][1][key], results[1][1][key]
        relative = (
            (reference - actual).norm() / reference.norm().clamp_min(1e-12)
        ).item()
        if relative > 0.05:
            raise AssertionError(
                f"Excessive LoRA gradient difference: {key}: {relative}"
            )
        errors[key] = relative
    output_errors = [
        ((a - b).norm() / a.norm()).item() for a, b in zip(results[0][0], results[1][0])
    ]
    if max(output_errors) > 0.02:
        raise AssertionError(f"Excessive H3 output difference: {output_errors}")
    result = {
        "passed": True,
        "gradient_checkpointing": True,
        "lora_gradients_checked": len(errors),
        "max_gradient_relative_l2_error": max(errors.values()),
        "output_relative_l2_errors": output_errors,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
