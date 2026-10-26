"""Protect full-backbone gradients, portable exports, and exact optimizer resume."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from diffusers import WanTransformer3DModel
from diffusers.optimization import get_scheduler

from video_blade.checkpoint import checkpoint_files, restore_checkpoint, save_checkpoint
from video_blade.config import TrainConfig
from video_blade.models import clone_student, trainable


def make_teacher():
    return (
        WanTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=48,
            out_channels=48,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
        )
        .to(torch.bfloat16)
        .requires_grad_(False)
        .eval()
    )


def test_full_training_updates_backbone_and_preserves_frozen_teacher():
    teacher = make_teacher()
    original = deepcopy(teacher.state_dict())
    student = clone_student(teacher, 64, 64, checkpointing=True, training_mode="full")
    assert all(
        p.requires_grad and p.dtype == torch.float32 for p in student.parameters()
    )
    assert not any("lora_" in name for name, _ in student.named_parameters())
    value = student(
        hidden_states=torch.randn(1, 48, 1, 4, 4),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 3, 16),
        return_dict=False,
    )[0]
    value.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in student.parameters()
    )
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    for key, tensor in original.items():
        torch.testing.assert_close(tensor, teacher.state_dict()[key], rtol=0, atol=0)


def test_full_checkpoint_exports_and_resumes_next_update(tmp_path, monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda: torch.zeros(1, dtype=torch.uint8)
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda _: None)
    cfg = TrainConfig(output=str(tmp_path), training_mode="full", max_steps=4)
    teacher = make_teacher()

    def setup():
        models = [
            clone_student(teacher, 64, 64, training_mode="full") for _ in range(2)
        ]
        optimizers = [
            torch.optim.AdamW(trainable(m), lr=lr)
            for m, lr in zip(models, (2e-6, 1e-5))
        ]
        scheduler = get_scheduler("constant", optimizer=optimizers[0])
        return models, optimizers, scheduler

    def update(models, optimizers, scheduler):
        for model, optimizer in zip(models, optimizers):
            sum((p - 1).square().mean() for p in trainable(model)).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    models, optimizers, scheduler = setup()
    update(models, optimizers, scheduler)
    save_checkpoint(
        tmp_path,
        1,
        cfg,
        *models,
        optimizers,
        {"epoch": 0, "batch": 2},
        0,
        1,
        "manifest",
        scheduler=scheduler,
    )
    root = tmp_path / "checkpoint-000001"
    assert all((root / name).is_file() for name in checkpoint_files("full"))
    assert not (root / "pytorch_lora_weights.safetensors").exists()
    exported = WanTransformer3DModel.from_pretrained(
        root / "student", torch_dtype=torch.float32, local_files_only=True
    )
    for key, tensor in models[0].state_dict().items():
        torch.testing.assert_close(tensor, exported.state_dict()[key], rtol=0, atol=0)
    resumed, resumed_opts, resumed_scheduler = setup()
    step, cursor = restore_checkpoint(
        root, cfg, *resumed, resumed_opts, 0, 1, "manifest", scheduler=resumed_scheduler
    )
    assert step == 1 and cursor == {"epoch": 0, "batch": 2}
    with pytest.raises(ValueError, match="training_mode"):
        restore_checkpoint(
            root,
            replace(cfg, training_mode="lora"),
            *resumed,
            resumed_opts,
            0,
            1,
            "manifest",
        )
    with pytest.raises(ValueError, match="update_order"):
        restore_checkpoint(
            root,
            replace(cfg, update_order="alternating"),
            *resumed,
            resumed_opts,
            0,
            1,
            "manifest",
        )
    update(models, optimizers, scheduler)
    update(resumed, resumed_opts, resumed_scheduler)
    for first, second in zip(models, resumed):
        for key, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[key], rtol=0, atol=0)
