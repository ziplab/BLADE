"""Check flow identities, interval ordering, and detached training targets."""

import pytest
import torch

from video_blade.config import TrainConfig
from video_blade.flow import (
    decompose,
    forward_noise,
    generator_loss,
    mix,
    sample_training_sigmas,
    schedule,
    select_states,
    velocity_training_pair,
)


def test_flow_decomposition_round_trip():
    x0, eps = torch.randn(3, 2, 2, 2, 2), torch.randn(3, 2, 2, 2, 2)
    sigma = torch.tensor([0.0, 0.4, 1.0])
    noisy = mix(x0, eps, sigma)
    actual_x0, actual_eps = decompose(noisy, eps - x0, sigma)
    torch.testing.assert_close(actual_x0, x0)
    torch.testing.assert_close(actual_eps, eps)


def test_forward_transition_preserves_flow_variance():
    torch.manual_seed(7)
    x0, eps = torch.randn(100000, 1), torch.randn(100000, 1)
    a, b = torch.full((len(x0),), 0.25), torch.full((len(x0),), 0.8)
    actual = forward_noise(mix(x0, eps, a), a, b)
    assert abs(actual.var().item() - ((1 - 0.8) ** 2 + 0.8**2)) < 0.015


def test_all_intervals_have_finite_training_range():
    for k in (2, 4, 8, 50):
        sigmas = schedule(k, "cpu")
        mid = sigmas[1:].repeat(20)
        noisy_sigma = sample_training_sigmas(mid)
        assert torch.all(noisy_sigma >= mid)
        assert torch.all(noisy_sigma >= 0.001)
        assert torch.all(noisy_sigma < 1)
        assert sigmas[0] == 1 and sigmas[-1] == 0


def test_generator_target_has_no_score_gradients():
    x = torch.tensor([[[[[2.0]]]]], requires_grad=True)
    real = torch.tensor([[[[[1.0]]]]], requires_grad=True)
    fake = torch.tensor([[[[[3.0]]]]], requires_grad=True)
    loss = generator_loss(x, real, fake)
    loss.backward()
    assert x.grad.item() > 0
    assert real.grad is None and fake.grad is None


def test_low_noise_velocity_labels_precede_bf16_input_quantization():
    torch.manual_seed(3)
    x0 = torch.randn(100000, 1)
    epsilon = torch.randn_like(x0)
    sigma = torch.full((len(x0),), 0.001)
    noisy = mix(x0, epsilon, sigma)
    model_input, target = velocity_training_pair(noisy, x0, sigma, True)
    legacy_input, legacy_target = velocity_training_pair(noisy, x0, sigma, False)
    assert model_input.dtype == torch.bfloat16
    assert torch.equal(model_input, legacy_input)
    assert (target - (epsilon - x0)).square().mean() < 1e-7
    assert (legacy_target - (epsilon - x0)).square().mean() > 1.0


def test_selected_states_preserve_batch_correspondence():
    states = [torch.tensor([[10], [11]]), torch.tensor([[20], [21]])]
    assert select_states(states, torch.tensor([1, 0])).tolist() == [[20], [11]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"frames": 8},
        {"width": 130},
        {"eta": 1.1},
        {"accumulation_steps": 0},
        {"stop_after_steps": 0},
        {"stop_after_steps": True},
        {"stop_after_steps": 301},
        {"lora_target_modules": []},
        {"lora_target_modules": "all-linear"},
        {"lora_target_modules": [""]},
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        TrainConfig(**overrides).validate()
