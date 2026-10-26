"""Sample an exported H3 PEFT adapter with the recorded distilled schedule."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh

from .config import H3Config
from .evaluate import evaluate
from .models import load_sharded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    cfg = H3Config(**json.loads(args.config.read_text())).validate()
    cfg.output = args.output
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if (
        adapter_config["r"] != cfg.rank
        or adapter_config["lora_alpha"] != cfg.lora_alpha
    ):
        raise ValueError("LoRA rank/alpha differ from the training configuration")
    if not cfg.validation_embeddings:
        raise ValueError("Set validation_embeddings to the H3 prompt cache to sample")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Match FP32 projection precision used by in-training validation.
    torch.backends.cuda.matmul.allow_tf32 = not cfg.deterministic
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    try:
        torch.manual_seed(cfg.seed)
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        student = load_sharded(cfg, "student", mesh, device).eval()
        state = {}
        if dist.get_rank() == 0:
            adapter = load_file(str(args.adapter / "adapter_model.safetensors"))
            for name, param in student.named_parameters():
                if param.requires_grad:
                    key = name.replace(".default.", ".")
                    state[name] = adapter.pop(key)
            if adapter:
                raise ValueError(f"Unexpected adapter weights: {sorted(adapter)}")
        set_model_state_dict(
            student,
            state,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )
        evaluate(cfg, None, student, device, args.step)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
