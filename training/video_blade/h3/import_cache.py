"""Import the pinned PDMD H3 prompt cache without running a text encoder."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

REPOSITORY = "pdmd2026/rcm-vidprom-h3-qwenvl-cache"
REVISION = "5abcdbbe08ae594da1fe07128fac80e1a3c1ae0a"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def cache_entries(root):
    """Preserve published text and source order while checking row identities."""
    prompts = read_jsonl(root / "prompts/prompts.jsonl")
    rows = []
    for path in sorted((root / "cache").glob("rank-*.jsonl")):
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: row["source_row"])
    if len(rows) != len(prompts) or any(
        row["source_row"] != index for index, row in enumerate(rows)
    ):
        raise ValueError("Cache records must cover the source exactly once")
    entries = []
    for row, prompt in zip(rows, prompts):
        text = prompt["prompt"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or row["id"] != prompt["id"]
            or hashlib.sha256(text.encode()).hexdigest() != row["prompt_sha256"]
        ):
            raise ValueError(f"Prompt identity mismatch at row {row['source_row']}")
        shard = Path(row["shard"])
        if shard.name != str(shard) or shard.suffix != ".safetensors":
            raise ValueError(f"Invalid cache shard: {shard}")
        entries.append(
            {
                "shard": f"cache/{shard}",
                "key": row["key"],
                "prompt": text,
                "tokens": row["tokens"],
            }
        )
    return entries, rows


def verify_encoder(model, audit):
    provenance = audit["provenance"]
    if (
        provenance["base_revision"] != BASE_REVISION
        or provenance["encoder_layer"] != "50"
        or provenance["final_norm"] != "identity"
        or provenance["prompt_rewrite"] != "false"
        or provenance["tokenizer_sha256"] != sha256(model / "tokenizer/tokenizer.json")
        or provenance["encoder_index_sha256"]
        != sha256(model / "text_encoder/model.safetensors.index.json")
    ):
        raise ValueError("Public cache does not match the native H3 text encoder")


def import_cache(root, model, workers=8):
    audit = json.loads((root / "cache-audit.json").read_text())
    complete = json.loads((root / "BACKUP_COMPLETE.json").read_text())
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    verify_encoder(model, audit)
    entries, rows = cache_entries(root)
    # The publisher's backup marker includes the README and checksum inventory.
    # The inventory cannot checksum itself and lists only the data files.
    backup_extra = {"README.md", "SHA256SUMS.json"} - sums.keys()
    backup_bytes = sum(value["size"] for value in sums.values()) + sum(
        (root / name).stat().st_size for name in backup_extra
    )
    if (
        complete["repo_id"] != REPOSITORY
        or complete["cache_signature"] != audit["cache_signature"]
        or len(entries) != audit["samples"]
        or len(entries) != complete["samples"]
        or len(sums) + len(backup_extra) != complete["files"]
        or backup_bytes != complete["bytes"]
        or sha256(root / "prompts/prompts.jsonl") != audit["source_sha256"]
        or any(row["source_sha256"] != audit["source_sha256"] for row in rows)
        or hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        != audit["cache_signature"]
    ):
        raise ValueError("Public cache inventory or source signature differs")
    by_shard = defaultdict(list)
    for entry in entries:
        by_shard[entry["shard"]].append(entry)
    if len(by_shard) != audit["shards"] or not by_shard.keys() <= sums.keys():
        raise ValueError("Cache shard count differs")

    def verify_file(item):
        name, expected = item
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid cache path: {name}")
        path = root / relative
        if (
            path.stat().st_size != expected["size"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"Cache checksum mismatch: {name}")
        if name not in by_shard:
            return
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if handle.metadata() != audit["provenance"]:
                raise ValueError(f"Cache shard provenance differs: {name}")
            for entry in by_shard[name]:
                value = handle.get_tensor(entry["key"])
                tags = handle.get_tensor(entry["key"] + "_tags")
                if (
                    value.shape != (entry["tokens"], 5120)
                    or value.dtype != torch.bfloat16
                    or not torch.isfinite(value).all()
                    or tags.shape != (entry["tokens"],)
                    or tags.dtype != torch.int64
                    or not (tags == 1).all()
                ):
                    raise ValueError(
                        f"Invalid H3 text condition: {name}:{entry['key']}"
                    )

    # Each worker reads one shard at a time; avoid nested CPU thread pools.
    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(verify_file, sums.items()), 1):
            if index % 100 == 0 or index == len(sums):
                print(f"Verified {index}/{len(sums)} cache files", flush=True)
    manifest = {
        "format_version": 1,
        "family": "minimax_h3",
        "task": "t2va",
        "model": str(model.resolve()),
        "transformer_config_sha256": sha256(model / "transformer/config.json"),
        "text_encoder_config_sha256": sha256(model / "text_encoder/config.json"),
        "source_sha256": audit["source_sha256"],
        "text_encoder_layer": 50,
        "chat_template": False,
        "negative_prompt": None,
        "external_cache": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "cache_signature": audit["cache_signature"],
            "verified_files": len(sums),
            "verified_bytes": sum(value["size"] for value in sums.values()),
            "backup_files": complete["files"],
            "backup_bytes": complete["bytes"],
            "encoder_provenance": audit["provenance"],
            "prompt_provenance": json.loads(
                (root / "prompts/provenance.json").read_text()
            ),
        },
        "entries": entries,
    }
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, root / "manifest.json")
    print(
        f"Imported {len(entries)} native H3 prompt conditions into {root}", flush=True
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    if not args.local_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            local_dir=args.output,
            max_workers=args.workers,
        )
    import_cache(args.output, args.model, args.workers)


if __name__ == "__main__":
    main()
