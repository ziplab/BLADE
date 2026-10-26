"""Encode prompt shards using only the tokenizer and UMT5 encoder."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import ftfy
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, UMT5EncoderModel

NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)


@torch.inference_mode()
def encode(tokenizer, encoder, prompts, length, device):
    import html

    cleaned = [
        " ".join(html.unescape(html.unescape(ftfy.fix_text(p))).strip().split())
        for p in prompts
    ]
    tokens = tokenizer(
        cleaned,
        padding="max_length",
        max_length=length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)
    embeds = encoder(
        input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
    ).last_hidden_state
    # Match WanPipeline: discard encoder outputs at padded positions, then zero-pad.
    embeds = embeds * tokens.attention_mask.unsqueeze(-1)
    return embeds.cpu().contiguous()


def main():
    torch.set_num_threads(8)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=128)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--negative-prompt", default=NEGATIVE)
    args = p.parse_args()
    if min(args.batch_size, args.shard_size, args.max_length) <= 0 or (
        args.limit is not None and args.limit <= 0
    ):
        p.error("Batch size, shard size, max length and limit must be positive")
    prompts = [
        line.strip() for line in args.prompts.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        prompts = prompts[: args.limit]
    if not prompts:
        p.error("Prompt file contains no nonempty lines")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(
            f"Output must be empty to prevent mixing embedding versions: {args.output}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", local_files_only=True
    )
    encoder = (
        UMT5EncoderModel.from_pretrained(
            args.model,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .to(args.device)
    )
    entries = []
    for start in range(0, len(prompts), args.shard_size):
        tensors = {}
        end = min(start + args.shard_size, len(prompts))
        shard = f"shard-{start // args.shard_size:05d}.safetensors"
        for offset in range(start, end, args.batch_size):
            batch_end = min(offset + args.batch_size, end)
            embeddings = encode(
                tokenizer,
                encoder,
                prompts[offset:batch_end],
                args.max_length,
                args.device,
            )
            for j, tensor in enumerate(embeddings):
                key = str(offset + j)
                tensors[key] = tensor.clone()
                entries.append(
                    {"shard": shard, "key": key, "prompt": prompts[offset + j]}
                )
        save_file(tensors, str(args.output / shard))
        print(f"Encoded {end}/{len(prompts)} prompts", flush=True)
    negative = encode(
        tokenizer, encoder, [args.negative_prompt], args.max_length, args.device
    )
    save_file({"embeddings": negative}, str(args.output / "negative.safetensors"))
    manifest = {
        "format_version": 1,
        "model": str(Path(args.model).resolve()),
        "transformer_config_sha256": hashlib.sha256(
            (Path(args.model) / "transformer/config.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "encoder_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers")
        },
        "max_length": args.max_length,
        "negative_prompt": args.negative_prompt,
        "entries": entries,
    }
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output / "manifest.json")


if __name__ == "__main__":
    main()
