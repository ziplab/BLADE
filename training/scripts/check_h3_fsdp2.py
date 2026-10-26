"""Create a tiny real H3 fixture for distributed update/resume regression."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from diffusers import MiniMaxH3Transformer3DModel
from safetensors.torch import save_file

from video_blade.h3.config import H3Config
from video_blade.h3.models import native_dtypes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/h3-fsdp2-check-v1")
    )
    parser.add_argument(
        "--attention-backend", choices=("native", "flash_4"), default="native"
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--algorithm", choices=("dmd", "tdm"), default="dmd")
    parser.add_argument("--checkpoint-skip-every", type=int, default=0)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_root = root / "base"
    torch.manual_seed(123)
    model = native_dtypes(
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=128 if args.attention_backend == "flash_4" else 32,
            hidden_size=64,
            num_layers=args.layers,
            num_refiner_layers=1,
            ffn_dim=128,
            text_dim=16,
            freq_dim=16,
            time_embed_hidden_dim=64,
            time_embed_dim=32,
            rope_freq_dim=4,
        )
    )
    model.save_pretrained(model_root / "transformer")
    embeds = root / "embeddings"
    embeds.mkdir(exist_ok=True)
    entries = []
    for i in range(32):
        shard = f"prompt-{i:03d}.safetensors"
        save_file(
            {"embeddings": torch.randn(3 + i % 4, 16, dtype=torch.bfloat16)},
            str(embeds / shard),
        )
        entries.append({"shard": shard, "key": "embeddings", "prompt": f"Fixture {i}"})
    manifest = {
        "format_version": 1,
        "family": "minimax_h3",
        "model": str(model_root),
        "text_encoder_layer": 50,
        "transformer_config_sha256": hashlib.sha256(
            (model_root / "transformer/config.json").read_bytes()
        ).hexdigest(),
        "entries": entries,
    }
    (embeds / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cfg = H3Config(
        algorithm=args.algorithm,
        model=str(model_root),
        embeddings=str(embeds),
        height=64,
        width=64,
        frames=22,
        max_steps=3,
        stop_after_steps=2,
        rank=4,
        lora_alpha=4,
        validation_embeddings=None,
        attention_backend=args.attention_backend,
        accumulation_steps=2,
        deterministic=args.attention_backend == "native",
        checkpoint_skip_every=args.checkpoint_skip_every,
        adapter_every=1,
    )
    for name, config in (
        (
            "continuous",
            replace(cfg, output=str(root / "continuous"), stop_after_steps=3),
        ),
        ("interrupted", replace(cfg, output=str(root / "interrupted"))),
        (
            "resumed",
            replace(
                cfg,
                output=str(root / "resumed"),
                resume=str(root / "interrupted/checkpoint-000002"),
                stop_after_steps=3,
            ),
        ),
    ):
        (root / f"{name}.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
