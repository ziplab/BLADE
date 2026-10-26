"""Cache variable-length H3 conditioning using the native Qwen3-VL layer 50."""

import argparse
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
from safetensors.torch import save_file
from transformers import (
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

from ..checkpoint import runtime_versions
from ..data import PromptDataset


def cache_entries(prompts, world, shard_size):
    return [
        {
            "shard": f"shard-rank-{index % world:03d}-{(index // world) // shard_size:05d}.safetensors",
            "key": str(index),
            "prompt": prompt,
        }
        for index, prompt in enumerate(prompts)
    ]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--keep-full-encoder", action="store_true")
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and args.device:
        parser.error("torchrun assigns devices through LOCAL_RANK; omit --device")
    device = torch.device(args.device or f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    if args.shard_size <= 0:
        parser.error("shard-size must be positive")
    lines = [
        line for line in args.prompts.read_text().splitlines() if line.strip()
    ]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        lines = lines[: args.limit]
    if not lines:
        parser.error("No prompts found")
    occupied = [False]
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        occupied[0] = any(args.output.iterdir())
    if world > 1:
        dist.broadcast_object_list(occupied, src=0)
    if occupied[0]:
        raise FileExistsError(f"Embedding output must be empty: {args.output}")
    reference = PromptDataset(args.verify_against) if args.verify_against else None
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.model / "tokenizer", local_files_only=True
    )
    processor = Qwen3VLProcessor.from_pretrained(
        args.model / "processor", local_files_only=True
    )
    encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model / "text_encoder", dtype=torch.bfloat16, local_files_only=True
        )
        .requires_grad_(False)
        .eval()
    )
    if not args.keep_full_encoder:
        # One extra layer is essential: hidden_states[50] must precede the final
        # RMSNorm. Stopping after exactly 50 layers would change the condition.
        encoder.model.language_model.layers = torch.nn.ModuleList(
            list(encoder.model.language_model.layers[:51])
        )
        encoder.config.text_config.num_hidden_layers = 51
    # Text-only encoding never calls the vision tower or language-model head.
    encoder.model.language_model.to(device)
    entries = cache_entries(lines, world, args.shard_size)
    tensors = {}
    verified = 0
    indices = list(range(rank, len(lines), world))
    for local_index, index in enumerate(indices):
        prompt = lines[index]
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        embeddings = get_qwen3vl_prompt_embeds(
            encoder,
            processor,
            tokens,
            {},
            text_encoder_layer=50,
            device=device,
            dtype=torch.bfloat16,
        )
        value = embeddings[0].cpu().contiguous()
        if reference is not None and index < len(reference):
            if reference.entries[index]["prompt"] != prompt:
                raise ValueError(f"Reference prompt differs at index {index}")
            torch.testing.assert_close(value, reference[index], rtol=0, atol=0)
            verified += 1
        tensors[str(index)] = value
        if (local_index + 1) % args.shard_size == 0 or local_index + 1 == len(indices):
            save_file(tensors, str(args.output / entries[index]["shard"]))
            tensors = {}
            print(
                f"Rank {rank}: encoded {local_index + 1}/{len(indices)} H3 prompts",
                flush=True,
            )
    verified_tensor = torch.tensor(verified, device=device)
    if world > 1:
        dist.all_reduce(verified_tensor)
        dist.barrier()
    manifest = {
        "format_version": 1,
        "family": "minimax_h3",
        "task": "t2va",
        "model": str(args.model.resolve()),
        "transformer_config_sha256": hashlib.sha256(
            (args.model / "transformer/config.json").read_bytes()
        ).hexdigest(),
        "text_encoder_config_sha256": hashlib.sha256(
            (args.model / "text_encoder/config.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "text_encoder_layer": 50,
        "decoder_layers_executed": encoder.config.text_config.num_hidden_layers,
        "encoder_world_size": world,
        "native_embeddings_verified_exactly": int(verified_tensor.item()),
        "chat_template": False,
        "negative_prompt": None,
        "versions": runtime_versions(),
        "entries": entries,
    }
    if rank == 0:
        temporary = args.output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, args.output / "manifest.json")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
