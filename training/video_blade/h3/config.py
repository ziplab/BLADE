"""Configuration isolated from the established Wan experiments."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class H3Config:
    algorithm: str = "dmd"
    model: str = "models/MiniMax-H3"
    embeddings: str = "outputs/prompts-h3-train"
    output: str = "outputs/h3-run"
    height: int = 768
    width: int = 1344
    frames: int = 124
    batch_size: int = 1
    accumulation_steps: int = 2
    max_steps: int | None = None
    stop_after_steps: int | None = 3
    distill_steps: int = 4
    learning_rate: float = 5e-5
    fake_learning_rate: float = 4e-7
    rank: int = 128
    lora_alpha: float = 8.0
    gradient_checkpointing: bool = True
    checkpoint_skip_every: int = 0
    video_shift: float = 6.0
    audio_shift: float = 3.0
    eta: float = 1.0
    audio_loss_weight: float = 1.0
    fake_update_ratio: int = 5
    adam_beta2: float = 0.999
    score_sigma_min: float = 0.02
    score_sigma_max: float = 0.98
    retain_rollout_weights: bool = True
    benchmark_rollout_residency: bool = False
    benchmark_partial_checkpointing: bool = False
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_every: int = 50
    adapter_every: int = 50
    save_final_checkpoint: bool = True
    validation_every: int = 50
    validation_embeddings: str | None = "outputs/prompts-h3-validation-v1"
    validation_count: int = 2
    teacher_steps: int = 49
    validation_seed: int = 8888
    seed: int = 42
    deterministic: bool = False
    attention_backend: str = "_native_cudnn"
    resume: str | None = None

    def __post_init__(self):
        if self.max_steps is None:
            self.max_steps = 300 if self.algorithm == "tdm" else 1000

    def validate(self):
        if self.algorithm not in {"dmd", "tdm"}:
            raise ValueError("H3 algorithm must be dmd or tdm")
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
            "adapter_every",
            "validation_every",
            "validation_count",
            "teacher_steps",
            "fake_update_ratio",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size != 1:
            raise ValueError(
                "H3 currently requires local batch_size=1 for unpadded prompt layouts"
            )
        if (
            self.height % 32
            or self.width % 32
            or self.frames < 22
            or self.frames % 17 != 5
        ):
            raise ValueError(
                "H3 requires height/width divisible by 32 and frames=17n+5"
            )
        if self.stop_after_steps is not None and (
            type(self.stop_after_steps) is not int
            or not 0 < self.stop_after_steps <= self.max_steps
        ):
            raise ValueError("stop_after_steps must be a positive integer <= max_steps")
        for name in (
            "learning_rate",
            "fake_learning_rate",
            "lora_alpha",
            "video_shift",
            "audio_shift",
            "max_grad_norm",
            "audio_loss_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.eta != 1.0 or self.weight_decay < 0:
            raise ValueError("The H3 DMD baseline requires deterministic Euler (eta=1)")
        if not 0 < self.score_sigma_min < self.score_sigma_max < 1:
            raise ValueError("Score sigma bounds must satisfy 0 < min < max < 1")
        if self.algorithm == "tdm" and any(
            max(1 - (i + 1) / self.distill_steps, self.score_sigma_min)
            >= min(1 - i / self.distill_steps, self.score_sigma_max)
            for i in range(self.distill_steps)
        ):
            raise ValueError(
                "Every TDM base-time segment must intersect the score bounds"
            )
        if not 0 < self.adam_beta2 < 1:
            raise ValueError("adam_beta2 must be in (0, 1)")
        for name in (
            "gradient_checkpointing",
            "retain_rollout_weights",
            "benchmark_rollout_residency",
            "deterministic",
            "save_final_checkpoint",
            "benchmark_partial_checkpointing",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.attention_backend not in {"native", "_native_cudnn", "flash_4"}:
            raise ValueError("Unsupported attention backend")
        if self.deterministic and self.attention_backend != "native":
            raise ValueError("Deterministic regression requires native math attention")
        if (
            type(self.checkpoint_skip_every) is not int
            or self.checkpoint_skip_every < 0
            or self.checkpoint_skip_every == 1
        ):
            raise ValueError("checkpoint_skip_every must be 0 or an integer >= 2")
        if self.checkpoint_skip_every and not self.gradient_checkpointing:
            raise ValueError("Partial checkpointing requires gradient_checkpointing")
        return self

    def to_dict(self):
        return asdict(self)


def parse_config():
    parser = argparse.ArgumentParser(description="FSDP2 MiniMax H3 distillation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    for name in ("output", "stop_after_steps", "resume"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    return H3Config(**values).validate()
