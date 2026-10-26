"""Verify that resumed cosine training preserves adapters and optimizer progress."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from diffusers import WanTransformer3DModel
from diffusers.optimization import get_scheduler
from peft import get_peft_model_state_dict

from video_blade.checkpoint import restore_checkpoint, save_checkpoint
from video_blade.config import TrainConfig
from video_blade.models import add_lora, trainable


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Checkpoint includes CUDA RNG"
)
@pytest.mark.parametrize("stop_after_steps", [None, 2])
def test_cosine_resume_preserves_next_update(tmp_path, stop_after_steps):
    cfg = TrainConfig(
        output=str(tmp_path),
        lr_schedule="cosine",
        warmup_steps=1,
        max_steps=4,
        stop_after_steps=stop_after_steps,
    )
    base = WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=16,
        out_channels=16,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
    ).requires_grad_(False)

    def setup():
        models = [add_lora(deepcopy(base), 2, 4) for _ in range(2)]
        optimizers = [
            torch.optim.AdamW(trainable(m), lr=lr)
            for m, lr in zip(models, (1e-3, 2e-3))
        ]
        scheduler = get_scheduler(
            "cosine", optimizer=optimizers[0], num_warmup_steps=1, num_training_steps=4
        )
        return models, optimizers, scheduler

    def update(models, optimizers, scheduler):
        for model, optimizer in zip(models, optimizers):
            sum((p - 1).square().mean() for p in trainable(model)).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    models, optimizers, scheduler = setup()
    for _ in range(2):
        update(models, optimizers, scheduler)
    save_checkpoint(
        tmp_path,
        2,
        cfg,
        *models,
        optimizers,
        {"epoch": 0, "batch": 2},
        0,
        1,
        "manifest",
        scheduler=scheduler,
    )
    resumed, resumed_opts, resumed_scheduler = setup()
    step, cursor = restore_checkpoint(
        tmp_path / "checkpoint-000002",
        replace(cfg, stop_after_steps=None),
        *resumed,
        resumed_opts,
        0,
        1,
        "manifest",
        scheduler=resumed_scheduler,
    )
    assert step == 2 and cursor == {"epoch": 0, "batch": 2}
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    update(models, optimizers, scheduler)
    update(resumed, resumed_opts, resumed_scheduler)
    for first, second in zip(models, resumed):
        a, b = get_peft_model_state_dict(first), get_peft_model_state_dict(second)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
