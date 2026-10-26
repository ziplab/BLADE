"""Validated configuration for the dense Wan distillation runner."""

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    model: str = "models/Wan2.1-T2V-1.3B-Diffusers"
    embeddings: str = "outputs/prompts-wan21"
    output: str = "outputs/wan21-dense"
    height: int = 480
    width: int = 832
    frames: int = 81
    batch_size: int = 1
    accumulation_steps: int = 1
    max_steps: int = 300
    stop_after_steps: int | None = None
    distill_steps: int = 8
    learning_rate: float = 4e-5
    fake_learning_rate: float = 8e-5
    weight_decay: float = 0.01
    fake_target_from_fp32: bool = False
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    training_mode: str = "lora"
    update_order: str = "synchronous"
    rank: int = 64
    lora_alpha: float = 64.0
    lora_target_modules: list[str] | None = None
    guidance_scale: float = 5.0
    eta: float = 0.9
    lambda_reg: float = 0.5
    max_grad_norm: float = 1.0
    checkpoint_every: int = 30
    gradient_checkpointing: bool = True
    workers: int = 2
    seed: int = 42
    deterministic: bool = False
    attention_backend: str = "native"
    resume: str | None = None
    validation_embeddings: str | None = None
    validation_every: int = 30
    validation_seed: int = 8888
    validation_fps: int = 16
    validation_sampler: str = "trajectory"

    def validate(self):
        if self.update_order not in {"synchronous", "alternating"}:
            raise ValueError("update_order must be synchronous or alternating")
        if self.training_mode not in {"lora", "full"}:
            raise ValueError("training_mode must be lora or full")
        if self.training_mode == "full" and self.lora_target_modules is not None:
            raise ValueError("Full training does not accept lora_target_modules")
        for name in (
            "gradient_checkpointing",
            "deterministic",
            "fake_target_from_fp32",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "height",
            "width",
            "frames",
            "batch_size",
            "accumulation_steps",
            "max_steps",
            "distill_steps",
            "rank",
            "checkpoint_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.height % 16 or self.width % 16 or (self.frames - 1) % 4:
            raise ValueError(
                "Wan requires height/width divisible by 16 and frames = 4n + 1; model-specific geometry is checked at load time"
            )
        if not 2 <= self.distill_steps <= 50:
            raise ValueError("distill_steps must be between 2 and 50")
        if not 0 <= self.eta <= 1 or self.lambda_reg < 0:
            raise ValueError("eta must be in [0, 1] and lambda_reg must be nonnegative")
        if (
            min(
                self.learning_rate,
                self.fake_learning_rate,
                self.max_grad_norm,
                self.lora_alpha,
            )
            <= 0
        ):
            raise ValueError(
                "Learning rates, max_grad_norm and lora_alpha must be positive"
            )
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.lora_target_modules is not None and (
            not isinstance(self.lora_target_modules, list)
            or not self.lora_target_modules
            or any(
                not isinstance(name, str) or not name.strip()
                for name in self.lora_target_modules
            )
        ):
            raise ValueError(
                "lora_target_modules must be a nonempty list of module names"
            )
        if self.validation_sampler not in {"unipc", "trajectory"}:
            raise ValueError("validation_sampler must be unipc or trajectory")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError(
                "warmup_steps must be nonnegative and smaller than max_steps"
            )
        if self.stop_after_steps is not None and (
            type(self.stop_after_steps) is not int
            or not 0 < self.stop_after_steps <= self.max_steps
        ):
            raise ValueError("stop_after_steps must be a positive integer <= max_steps")
        if self.lr_schedule == "constant" and self.warmup_steps:
            raise ValueError("warmup_steps is only supported by the cosine schedule")
        if self.validation_every <= 0 or self.validation_fps <= 0:
            raise ValueError("Validation interval and fps must be positive")
        if self.attention_backend not in {"native", "_native_cudnn"}:
            raise ValueError("attention_backend must be native or _native_cudnn")
        if self.deterministic and self.attention_backend == "_native_cudnn":
            raise ValueError("Use native attention for deterministic regression runs")
        return self

    def to_dict(self):
        return asdict(self)


def parse_config():
    parser = argparse.ArgumentParser(description="Dense Wan trajectory distillation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    unknown = values.keys() - {f.name for f in fields(TrainConfig)}
    if unknown:
        parser.error(f"Unknown config keys: {sorted(unknown)}")
    for key in ("output", "max_steps", "stop_after_steps", "resume"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    return TrainConfig(**values).validate()
