"""Atomic distributed checkpoints plus a portable student-only PEFT adapter."""

import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)

from .config import H3Config

STATE_OPTIONS = StateDictOptions(ignore_frozen_params=True, strict=False)


def write_adapter(student, root):
    adapter = get_model_state_dict(
        student,
        options=StateDictOptions(
            full_state_dict=True, cpu_offload=True, ignore_frozen_params=True
        ),
    )
    if dist.get_rank() == 0:
        state = get_peft_model_state_dict(student, state_dict=adapter)
        save_file(
            {k: v.contiguous() for k, v in state.items()},
            str(root / "adapter_model.safetensors"),
        )
        student.peft_config["default"].save_pretrained(root)


def export_adapter(output, step, cfg, student):
    rank = dist.get_rank()
    parent = Path(output) / "adapters"
    target = parent / f"step-{step:06d}"
    staging = parent / f".step-{step:06d}.incomplete"
    if rank == 0:
        staging.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    write_adapter(student, staging)
    if rank == 0:
        (staging / "metadata.json").write_text(
            json.dumps(
                {"resumable": False, "step": step, "config": cfg.to_dict()}, indent=2
            )
            + "\n"
        )
        if target.exists():
            raise FileExistsError(target)
        os.replace(staging, target)
    dist.barrier()


def link_checkpoint_adapter(output, step, cfg, checkpoint):
    parent = Path(output) / "adapters"
    target = parent / f"step-{step:06d}"
    staging = parent / f".step-{step:06d}.incomplete"
    staging.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        os.link(checkpoint / name, staging / name)
    (staging / "metadata.json").write_text(
        json.dumps(
            {
                "resumable": False,
                "step": step,
                "checkpoint": str(checkpoint),
                "config": cfg.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    if target.exists():
        raise FileExistsError(target)
    os.replace(staging, target)


def save_checkpoint(
    output, step, cfg, student, fake, optimizers, cursor, manifest_hash
):
    rank, world = dist.get_rank(), dist.get_world_size()
    target = Path(output) / f"checkpoint-{step:06d}"
    staging = Path(output) / f".checkpoint-{step:06d}.incomplete"
    occupied = [target.exists() or staging.exists() if rank == 0 else None]
    dist.broadcast_object_list(occupied, src=0)
    if occupied[0]:
        raise FileExistsError(
            f"Checkpoint destination already exists: {target} / {staging}"
        )
    if rank == 0:
        staging.mkdir(parents=True)
    dist.barrier()
    # Save shards instead of gathering the 33B fake and its Adam states on GPU 0.
    for role, model, optimizer in zip(("student", "fake"), (student, fake), optimizers):
        weights, opt_state = get_state_dict(model, optimizer, options=STATE_OPTIONS)
        dcp.save(
            {"model": weights, "optimizer": opt_state}, checkpoint_id=staging / role
        )
        del weights, opt_state
    write_adapter(student, staging)
    if rank == 0:
        (staging / "metadata.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "resumable": True,
                    "step": step,
                    "fake_updates": step * cfg.fake_update_ratio,
                    "cursor": cursor,
                    "world_size": world,
                    "config": cfg.to_dict(),
                    "manifest_sha256": manifest_hash,
                },
                indent=2,
            )
            + "\n"
        )
    torch.save(
        {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(),
        },
        staging / f"rng-rank-{rank:05d}.pt",
    )
    dist.barrier()
    if rank == 0:
        os.replace(staging, target)
        link_checkpoint_adapter(output, step, cfg, target)
    dist.barrier()


def restore_checkpoint(root, cfg, student, fake, optimizers, manifest_hash):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    if not metadata.get("resumable", True):
        raise ValueError(
            "An adapter-only export has no fake/optimizer states for exact resume"
        )
    if metadata["world_size"] != dist.get_world_size():
        raise ValueError(
            "Exact resume requires the original world size and per-rank RNG states"
        )
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("Prompt cache changed since the checkpoint")
    flexible = {
        "output",
        "resume",
        "stop_after_steps",
        "checkpoint_every",
        "adapter_every",
        "max_steps",
        "validation_every",
    }
    saved_config = H3Config(**metadata["config"]).to_dict()
    for key, value in cfg.to_dict().items():
        if key not in flexible and saved_config.get(key) != value:
            raise ValueError(f"Resume config mismatch: {key}")
    for role, model, optimizer in zip(("student", "fake"), (student, fake), optimizers):
        weights, opt_state = get_state_dict(model, optimizer, options=STATE_OPTIONS)
        state = {"model": weights, "optimizer": opt_state}
        dcp.load(state, checkpoint_id=root / role)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=STATE_OPTIONS,
        )
    rng = torch.load(
        root / f"rng-rank-{dist.get_rank():05d}.pt",
        weights_only=False,
        map_location="cpu",
    )
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    random.setstate(rng["python"])
    return metadata["step"], metadata["cursor"]
