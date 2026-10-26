"""Atomic model checkpoints, optimizer state, RNG state, and runtime provenance."""

import importlib.metadata
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import WanPipeline
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from .runtime import distributed


def checkpoint_files(training_mode="lora", fake_training_mode=None):
    if training_mode == "lora" and fake_training_mode == "full":
        return [
            "student.safetensors",
            "pytorch_lora_weights.safetensors",
            "fake/config.json",
            "fake/diffusion_pytorch_model.safetensors",
            "training.pt",
        ]
    if training_mode == "full":
        return [
            "student/config.json",
            "student/diffusion_pytorch_model.safetensors",
            "fake/config.json",
            "fake/diffusion_pytorch_model.safetensors",
            "training.pt",
        ]
    return [
        "student.safetensors",
        "fake.safetensors",
        "pytorch_lora_weights.safetensors",
        "training.pt",
    ]


def load_model_weights(model, root, name, training_mode="lora"):
    root = Path(root)
    if training_mode == "full":
        model.load_state_dict(
            load_file(str(root / name / "diffusion_pytorch_model.safetensors")),
            strict=True,
        )
    else:
        result = set_peft_model_state_dict(
            model, load_file(str(root / f"{name}.safetensors"))
        )
        if result.unexpected_keys:
            raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def save_checkpoint(
    output,
    step,
    config,
    student,
    fake,
    optimizers,
    cursor,
    rank,
    world,
    manifest_hash,
    scheduler=None,
    source=None,
):
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "python": random.getstate(),
    }
    rngs = [None] * world if rank == 0 else None
    if distributed():
        dist.gather_object(rng, rngs, dst=0)
    else:
        rngs = [rng]
    if rank == 0:
        target = output / f"checkpoint-{step:06d}"
        if target.exists():
            raise FileExistsError(target)
        staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=output))
        for name, model in (("student", student), ("fake", fake)):
            mode = config.training_mode
            if name == "fake":
                mode = getattr(config, "fake_training_mode", None) or mode
            if mode == "full":
                model_root = staging / name
                model.save_config(model_root)
                state = {
                    k: v.detach().cpu().contiguous()
                    for k, v in model.state_dict().items()
                }
                save_file(
                    state,
                    str(model_root / "diffusion_pytorch_model.safetensors"),
                    metadata={"format": "pt"},
                )
                del state
                continue
            state = {
                k: v.detach().cpu().contiguous()
                for k, v in get_peft_model_state_dict(model).items()
            }
            save_file(state, str(staging / f"{name}.safetensors"))
            if name == "student":
                WanPipeline.save_lora_weights(
                    staging,
                    transformer_lora_layers=state,
                    transformer_lora_adapter_metadata=model.peft_config[
                        "default"
                    ].to_dict(),
                )
        torch.save(
            {
                "step": step,
                "cursor": cursor,
                "config": config.to_dict(),
                "world_size": world,
                "rngs": rngs,
                "optimizers": [o.state_dict() for o in optimizers],
                "manifest_sha256": manifest_hash,
                "versions": runtime_versions(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "source": source,
            },
            staging / "training.pt",
        )
        (staging / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n"
        )
        os.replace(staging, target)
        print(f"Saved {target}", flush=True)
    if distributed():
        dist.barrier()


def restore_checkpoint(
    path, cfg, student, fake, optimizers, rank, world, manifest_hash, scheduler=None
):
    root = Path(path)
    state = torch.load(root / "training.pt", map_location="cpu", weights_only=False)
    config_type = type(cfg)
    saved_config = (
        config_type.from_dict(state["config"])
        if hasattr(config_type, "from_dict")
        else config_type(**state["config"])
    ).to_dict()
    if (
        getattr(cfg, "lr_schedule", "constant") != "constant"
        and saved_config["max_steps"] != cfg.max_steps
    ):
        raise ValueError("Scheduled training must resume with the original max_steps")
    flexible = {
        "output",
        "max_steps",
        "stop_after_steps",
        "resume",
        "checkpoint_every",
        "workers",
        "adapter_every",
        "save_final_checkpoint",
    }
    for key, value in cfg.to_dict().items():
        if key not in flexible and saved_config[key] != value:
            raise ValueError(
                f"Resume config mismatch for {key}: {value!r} != {saved_config[key]!r}"
            )
    if state["world_size"] != world or state["manifest_sha256"] != manifest_hash:
        raise ValueError(
            "Stateful resume requires the original world size and prompt manifest"
        )
    for name, model in (("student", student), ("fake", fake)):
        mode = cfg.training_mode
        if name == "fake":
            mode = getattr(cfg, "fake_training_mode", None) or mode
        load_model_weights(model, root, name, mode)
    for optimizer, values in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(values)
    if scheduler is not None:
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        elif getattr(cfg, "lr_schedule", "constant") != "constant":
            raise ValueError(
                "Checkpoint does not contain the required LR scheduler state"
            )
    rng = state["rngs"][rank]
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    random.setstate(rng["python"])
    return state["step"], state["cursor"]



def source_metadata():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def runtime_versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "diffusers",
            "transformers",
            "peft",
            "accelerate",
            "huggingface-hub",
            "tokenizers",
        )
    }
