"""Protect Wan2.1/2.2 geometry and uniform T2V timestep equivalence."""

import json
from copy import deepcopy

import pytest
import torch
from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

from video_blade.architecture import WanSpec
from video_blade.models import add_lora, trainable
from video_blade.sampling import make_pipeline


def test_wan22_native_geometry_and_patch_alignment():
    spec = WanSpec(48, 4, 16, (1, 2, 2), True, 5.0)
    shape = spec.latent_shape(2, 121, 704, 1280)
    assert shape == (2, 48, 31, 44, 80)
    assert spec.tokens(shape) == 27280
    with pytest.raises(ValueError, match="Invalid Wan video geometry"):
        spec.latent_shape(1, 121, 720, 1280)
    assert WanSpec(16, 4, 8, (1, 2, 2), False, 3.0).latent_shape(2, 81, 480, 832) == (
        2,
        16,
        21,
        60,
        104,
    )


@pytest.mark.parametrize("channels", [16, 48])
def test_compact_t2v_timesteps_match_expanded_forward_and_gradients(channels):
    torch.manual_seed(14)
    compact = add_lora(
        WanTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=channels,
            out_channels=channels,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
        ).requires_grad_(False),
        rank=2,
        alpha=2,
    )
    with torch.no_grad():
        for name, param in compact.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.01)
    expanded = deepcopy(compact)
    x = torch.randn(2, channels, 3, 4, 4)
    prompt = torch.randn(2, 4, 16)
    t = torch.tensor([875.0, 250.0])
    outputs, gradients = [], []
    for model, timestep in ((compact, t), (expanded, t[:, None].expand(-1, 12))):
        value = model(x, timestep, prompt, return_dict=False)[0]
        outputs.append(value)
        gradients.append(torch.autograd.grad(value.square().mean(), trainable(model)))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_pipeline_preserves_wan22_geometry_and_token_timesteps_without_vae(tmp_path):
    model = WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=48,
        out_channels=48,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
    ).eval()
    model.save_config(tmp_path / "transformer")
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae/config.json").write_text(
        json.dumps(
            {
                "z_dim": 48,
                "latents_mean": [0.0] * 48,
                "latents_std": [1.0] * 48,
                "scale_factor_temporal": 4,
                "scale_factor_spatial": 16,
            }
        )
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "expand_timesteps": True,
            }
        )
    )
    UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=5.0,
    ).save_pretrained(tmp_path / "scheduler")
    observed = []
    hook = model.register_forward_pre_hook(
        lambda _, args, kwargs: observed.append(tuple(kwargs["timestep"].shape)),
        with_kwargs=True,
    )
    pipe = make_pipeline(tmp_path, model)
    result = pipe(
        prompt_embeds=torch.randn(1, 4, 16),
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=1.0,
        output_type="latent",
        generator=torch.Generator().manual_seed(42),
    ).frames
    hook.remove()
    assert result.shape == (1, 48, 3, 4, 4)
    assert observed == [(1, 12), (1, 12)]
    assert pipe.scheduler.config.flow_shift == 5.0
