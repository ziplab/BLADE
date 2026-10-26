"""Ensure exported LoRA preserves non-default alpha and trained weights."""

from copy import deepcopy

import pytest
import torch
from diffusers import WanPipeline, WanTransformer3DModel
from peft import get_peft_model_state_dict

from video_blade.models import add_lora


@pytest.mark.parametrize("with_ffn", [False, True])
def test_diffusers_export_preserves_alpha_and_weights(tmp_path, with_ffn):
    base = WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=16,
        out_channels=16,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
    )
    base.requires_grad_(False)
    targets = ["to_q", "to_k", "to_v", "to_out.0"]
    if with_ffn:
        targets += ["ffn.net.0.proj", "ffn.net.2"]
    student = add_lora(deepcopy(base), rank=2, alpha=4, target_modules=targets)
    with torch.no_grad():
        for name, param in student.named_parameters():
            if "lora_B" in name:
                param.normal_()
    expected = get_peft_model_state_dict(student)
    WanPipeline.save_lora_weights(
        tmp_path,
        transformer_lora_layers=expected,
        transformer_lora_adapter_metadata=student.peft_config["default"].to_dict(),
    )
    base.load_lora_adapter(
        str(tmp_path),
        weight_name="pytorch_lora_weights.safetensors",
        use_safetensors=True,
        local_files_only=True,
        adapter_name="default",
    )
    actual = get_peft_model_state_dict(base)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])
    assert base.peft_config["default"].lora_alpha == 4
    assert set(base.peft_config["default"].target_modules) == set(targets)


def test_ffn_adapters_receive_gradients_while_backbone_stays_frozen():
    model = WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=48,
        out_channels=48,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
    ).requires_grad_(False)
    inputs = {
        "hidden_states": torch.randn(1, 48, 1, 4, 4),
        "timestep": torch.tensor([500.0]),
        "encoder_hidden_states": torch.randn(1, 3, 16),
        "return_dict": False,
    }
    with torch.no_grad():
        expected = model(**inputs)[0]
    add_lora(
        model,
        rank=2,
        alpha=2,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ffn.net.0.proj",
            "ffn.net.2",
        ],
    )
    output = model(**inputs)[0]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    ffn_updates = [
        p
        for name, p in model.named_parameters()
        if ".ffn." in name and "lora_B" in name
    ]
    assert len(ffn_updates) == 2
    assert all(
        torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in ffn_updates
    )
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
