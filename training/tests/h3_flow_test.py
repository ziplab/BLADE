"""Protect H3's velocity sign, paired clock, packing, and DMD gradients."""

import json
from copy import deepcopy

import torch
from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from safetensors.torch import save_file

from video_blade.data import PromptDataset
from video_blade.h3.config import H3Config
from video_blade.h3.flow import (
    H3Layout,
    dmd_loss,
    paired_decompose,
    shifted_sigma,
    sigma_grid,
)
from video_blade.h3.models import checkpoint_policy, configure_role
from video_blade.h3.prepare import cache_entries


def test_h3_schedules_count_forwards_and_shift_once():
    for grid, shift in zip(sigma_grid(4, 6.0, 3.0), (6.0, 3.0)):
        native = MiniMaxH3Scheduler(shift=shift)
        native.set_timesteps(5)
        torch.testing.assert_close(grid, native.sigmas, rtol=0, atol=0)
        assert len(grid) == 5
        torch.testing.assert_close(grid, shifted_sigma(torch.linspace(1, 0, 5), shift))


def test_h3_cleanward_velocity_matches_native_scheduler():
    scheduler = MiniMaxH3Scheduler(shift=6)
    scheduler.set_timesteps(5)
    noisy, cleanward = torch.randn(1, 8, 96), torch.randn(1, 8, 96)
    sigma = scheduler.sigmas[0]
    clean, _ = paired_decompose(
        (noisy, noisy), (-cleanward, -cleanward), (sigma, sigma)
    )
    expected = scheduler.step(cleanward, scheduler.timesteps[0], noisy).prev_sample
    actual = noisy + (scheduler.sigmas[1] - sigma) * (-cleanward)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(clean[0], noisy + sigma * cleanward)


def test_h3_packing_and_native_geometry():
    layout = H3Layout(768, 1344, 124, 23)
    assert layout.latent_frames == 37
    assert layout.audio_frames == 207
    samples = layout.noise()
    assert samples[0].shape == (1, 37296, 96)
    assert samples[1].shape == (1, 414, 32)
    video, audio = layout.unpack(samples)
    torch.testing.assert_close(
        patchify_video_latents(video, (1, 2, 2))[None], samples[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        audio.permute(0, 2, 1).reshape(1, -1, 32), samples[1], rtol=0, atol=0
    )


def test_dmd_gradient_and_teacher_distance_normalization():
    x = torch.tensor([[[2.0, 4.0]]], requires_grad=True)
    teacher = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    fake = torch.tensor([[[3.0, 3.0]]], requires_grad=True)
    loss = dmd_loss(x, teacher, fake)
    loss.backward()
    torch.testing.assert_close(
        x.grad, (fake.detach() - teacher.detach()) / 1.5 / x.numel()
    )
    assert teacher.grad is None and fake.grad is None
    assert torch.isfinite(dmd_loss(x, x, x))


def test_h3_reference_configuration():
    cfg = H3Config().validate()
    assert (cfg.rank, cfg.lora_alpha) == (128, 8)
    assert (cfg.learning_rate, cfg.fake_learning_rate) == (5e-5, 4e-7)
    assert cfg.fake_update_ratio == 5


def test_partial_checkpointing_preserves_full_model_gradients():
    torch.manual_seed(17)
    reference = configure_role(
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=32,
            hidden_size=32,
            num_layers=4,
            num_refiner_layers=1,
            ffn_dim=64,
            text_dim=16,
            freq_dim=16,
            time_embed_hidden_dim=32,
            time_embed_dim=16,
            rope_freq_dim=4,
        ),
        "fake",
        checkpointing=True,
    )
    alternative = deepcopy(reference)
    checkpoint_policy(alternative, skip_every=4)
    layout = H3Layout(64, 64, 22, 3)
    samples = layout.noise()
    embeddings = torch.randn(1, 3, 16)
    sigmas = (torch.tensor(0.7), torch.tensor(0.3))
    for model in (reference, alternative):
        output = layout.predict(model, samples, sigmas, embeddings)
        sum(x.square().mean() for x in output).backward()
    for original, partial in zip(reference.parameters(), alternative.parameters()):
        assert original.grad is not None and partial.grad is not None
        torch.testing.assert_close(original.grad, partial.grad, rtol=1e-6, atol=1e-7)


def test_parallel_prompt_shards_preserve_global_order_and_variable_lengths(tmp_path):
    entries = cache_entries([f"Prompt {i}" for i in range(17)], world=3, shard_size=2)
    shards = {}
    for index, entry in enumerate(entries):
        shards.setdefault(entry["shard"], {})[entry["key"]] = torch.full(
            (2 + index % 3, 4), index, dtype=torch.bfloat16
        )
    for name, values in shards.items():
        save_file(values, str(tmp_path / name))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format_version": 1, "entries": entries})
    )
    dataset = PromptDataset(tmp_path)
    assert len(dataset) == 17
    for index in range(len(dataset)):
        value = dataset[index]
        assert value.shape == (2 + index % 3, 4)
        assert (value == index).all()
