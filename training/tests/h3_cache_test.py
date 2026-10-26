"""Check provenance, text preservation, and invalid external H3 conditions."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from video_blade.data import PromptDataset
from video_blade.h3.import_cache import (
    BASE_REVISION,
    REPOSITORY,
    import_cache,
    sha256,
)


def fixture_cache(tmp_path, tags=1):
    root, model = tmp_path / "cache", tmp_path / "model"
    for name in ("cache", "prompts"):
        (root / name).mkdir(parents=True)
    for name in (
        "tokenizer/tokenizer.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/config.json",
        "transformer/config.json",
    ):
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    provenance = {
        "base_revision": BASE_REVISION,
        "encoder_layer": "50",
        "final_norm": "identity",
        "prompt_rewrite": "false",
        "tokenizer_sha256": sha256(model / "tokenizer/tokenizer.json"),
        "encoder_index_sha256": sha256(
            model / "text_encoder/model.safetensors.index.json"
        ),
    }
    prompts = [
        {"id": str(i), "prompt": text}
        for i, text in enumerate((" Leading space", "Trailing space "))
    ]
    (root / "prompts/prompts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prompts)
    )
    source_hash = sha256(root / "prompts/prompts.jsonl")
    provenance["source_sha256"] = source_hash
    rows = [
        {
            "id": row["id"],
            "source_row": i,
            "key": str(i),
            "tokens": 2 + i,
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "source_sha256": source_hash,
            "shard": "features.safetensors",
        }
        for i, row in enumerate(prompts)
    ]
    (root / "cache/rank-0000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows))
    )
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    values = {}
    for i in range(2):
        values[str(i)] = torch.full((2 + i, 5120), i, dtype=torch.bfloat16)
        values[str(i) + "_tags"] = torch.full((2 + i,), tags, dtype=torch.int64)
    save_file(values, str(root / "cache/features.safetensors"), metadata=provenance)
    (root / "cache-audit.json").write_text(
        json.dumps(
            {
                "samples": 2,
                "shards": 1,
                "cache_signature": signature,
                "source_sha256": source_hash,
                "provenance": provenance,
            }
        )
    )
    (root / "prompts/provenance.json").write_text("{}")
    (root / "README.md").write_text("H3 cache\n")
    sums = {
        str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": sha256(p)}
        for p in root.rglob("*")
        if p.is_file() and p.name != "README.md"
    }
    (root / "SHA256SUMS.json").write_text(json.dumps(sums))
    (root / "BACKUP_COMPLETE.json").write_text(
        json.dumps(
            {
                "repo_id": REPOSITORY,
                "cache_signature": signature,
                "samples": 2,
                "files": len(sums) + 2,
                "bytes": sum(v["size"] for v in sums.values())
                + (root / "README.md").stat().st_size
                + (root / "SHA256SUMS.json").stat().st_size,
            }
        )
    )
    return root, model


def test_external_cache_preserves_text_order_and_reads_without_conversion(tmp_path):
    root, model = fixture_cache(tmp_path)
    manifest = import_cache(root, model, workers=1)
    dataset = PromptDataset(root)
    assert [row["prompt"] for row in manifest["entries"]] == [
        " Leading space",
        "Trailing space ",
    ]
    assert dataset[0].shape == (2, 5120)
    assert dataset[1].shape == (3, 5120)
    assert dataset[1].dtype == torch.bfloat16
    assert (dataset[1] == 1).all()


def test_external_cache_rejects_corruption_before_publishing_manifest(tmp_path):
    root, model = fixture_cache(tmp_path)
    with (root / "cache/features.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        import_cache(root, model, workers=1)
    assert not (root / "manifest.json").exists()


def test_external_cache_rejects_nontext_modality_tags(tmp_path):
    root, model = fixture_cache(tmp_path, tags=0)
    with pytest.raises(ValueError, match="Invalid H3 text condition"):
        import_cache(root, model, workers=1)
    assert not (root / "manifest.json").exists()


def test_external_cache_rejects_different_tokenizer(tmp_path):
    root, model = fixture_cache(tmp_path)
    (model / "tokenizer/tokenizer.json").write_text('{"different": true}')
    with pytest.raises(ValueError, match="does not match"):
        import_cache(root, model, workers=1)
