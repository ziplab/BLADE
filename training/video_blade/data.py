"""Sharded prompt embeddings with bounded per-worker file caching."""

import json
from collections import OrderedDict
from pathlib import Path

from safetensors import safe_open
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format_version"] != 1:
            raise ValueError("Unsupported prompt manifest version")
        self.entries = self.manifest["entries"]
        if not self.entries:
            raise ValueError("Prompt dataset is empty")
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        shard = entry["shard"]
        if shard not in self.cache:
            self.cache[shard] = safe_open(
                str(self.root / shard), framework="pt", device="cpu"
            )
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard].get_tensor(entry["key"])

    def negative(self):
        with safe_open(
            str(self.root / "negative.safetensors"), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor("embeddings")
