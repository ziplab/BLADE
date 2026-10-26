"""Check paired TDM clocks, endpoint gradients, and algorithm isolation."""

import json

import pytest
import torch

from video_blade.h3 import checkpoint
from video_blade.h3.config import H3Config
from video_blade.h3.flow import dmd_loss, score_training_pair, sigma_grid
from video_blade.h3.tdm import (
    endpoint_loss,
    paired_endpoint_loss,
    query_sigmas,
    trajectory_query,
)
from video_blade.h3.train import backward_simulation


def test_tdm_queries_share_base_time_and_stay_in_both_segments():
    grids = sigma_grid(4, 6, 3)
    torch.manual_seed(24)
    for index in range(4):
        for _ in range(64):
            base, sigmas = query_sigmas(index, 4, 6, 3, 0.02, 0.98, "cpu")
            assert max(1 - (index + 1) / 4, 0.02) < base
            assert base < min(1 - index / 4, 0.98)
            for grid, sigma, shift in zip(grids, sigmas, (6, 3)):
                assert grid[index + 1] < sigma < grid[index]
                recovered_base = sigma / (shift - (shift - 1) * sigma)
                torch.testing.assert_close(recovered_base, base)


def test_tdm_endpoint_transition_has_correct_mean_and_variance():
    torch.manual_seed(33)
    endpoints = (torch.full((1, 100000), 0.7), torch.full((1, 100000), -0.4))
    lows = (torch.tensor(0.6), torch.tensor(0.3))
    sigmas = (torch.tensor(0.8), torch.tensor(0.5))
    samples = trajectory_query(endpoints, lows, sigmas)
    residuals = []
    for endpoint, sample, low, sigma in zip(endpoints, samples, lows, sigmas):
        ratio = (1 - sigma) / (1 - low)
        residual = sample - ratio * endpoint
        residuals.append(residual)
        expected_variance = sigma**2 - (ratio * low) ** 2
        torch.testing.assert_close(
            residual.mean(), torch.tensor(0.0), atol=0.006, rtol=0
        )
        torch.testing.assert_close(
            residual.var(), expected_variance, atol=0.006, rtol=0
        )
    assert abs((residuals[0] * residuals[1]).mean()) < 0.006


@pytest.mark.parametrize("draw", [0.0, 1.0])
def test_shifted_queries_exclude_endpoints_even_at_rng_boundaries(monkeypatch, draw):
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(draw))
    grids = sigma_grid(4, 6, 3)
    for index in range(4):
        base, sigmas = query_sigmas(index, 4, 6, 3, 0.02, 0.98, "cpu")
        assert max(1 - (index + 1) / 4, 0.02) < base
        assert base < min(1 - index / 4, 0.98)
        for grid, sigma in zip(grids, sigmas):
            assert grid[index + 1] < sigma < grid[index]


def test_last_segment_matches_existing_dmd_loss_and_gradient():
    torch.manual_seed(4)
    clean = torch.randn(1, 3, 8, requires_grad=True)
    teacher, fake = torch.randn_like(clean), torch.randn_like(clean)
    expected = dmd_loss(clean, teacher, fake)
    actual = endpoint_loss(clean, clean, teacher, fake, torch.tensor(0.0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, clean)[0], torch.autograd.grad(expected, clean)[0]
    )


def test_endpoint_gradient_matches_explicit_weighted_score_chain():
    torch.manual_seed(5)
    for low, sigma in ((0.9473684, 0.98), (0.75, 0.85), (0.0, 0.3)):
        clean = torch.randn(1, 2, 4, requires_grad=True)
        endpoint = torch.randn_like(clean).requires_grad_()
        teacher = torch.randn_like(clean).requires_grad_()
        fake = torch.randn_like(clean).requires_grad_()
        noisy = torch.randn_like(clean)
        normalizer = (clean.detach() - teacher.detach()).abs().mean()
        teacher_score = ((1 - sigma) * teacher.detach() - noisy) / sigma**2
        fake_score = ((1 - sigma) * fake.detach() - noisy) / sigma**2
        ratio = (1 - sigma) / (1 - low)
        query_weight = sigma**2 / (1 - sigma) ** 2
        expected = ratio * (fake_score - teacher_score) * query_weight / normalizer
        loss = endpoint_loss(clean, endpoint, teacher, fake, torch.tensor(low))
        loss.backward()
        torch.testing.assert_close(endpoint.grad, expected / endpoint.numel())
        assert clean.grad is None and teacher.grad is None and fake.grad is None


def test_audio_video_reductions_have_independent_scales():
    clean = (torch.full((1, 100, 8), 2.0), torch.full((1, 2, 4), 4.0))
    endpoints = tuple(x.clone().requires_grad_() for x in clean)
    teacher = (torch.zeros_like(clean[0]), torch.zeros_like(clean[1]))
    fake = (torch.ones_like(clean[0]), torch.ones_like(clean[1]))
    lows = (torch.tensor(0.5), torch.tensor(0.25))
    paired_endpoint_loss(clean, endpoints, teacher, fake, lows, 2.0).backward()
    torch.testing.assert_close(endpoints[0].grad.sum(), torch.tensor(1.0))
    torch.testing.assert_close(endpoints[1].grad.sum(), torch.tensor(2 / 3))


def test_only_selected_euler_step_receives_endpoint_gradient(monkeypatch):
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.velocities = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.tensor(float(i + 1))) for i in range(4)]
            )

    class Layout:
        calls = 0

        def noise(self):
            return (torch.ones(1, 2, 3), torch.ones(1, 1, 4))

        def predict(self, model, samples, sigmas, embeddings):
            value = model.velocities[self.calls]
            self.calls += 1
            return tuple(torch.ones_like(x) * value for x in samples)

    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(2))
    model, layout = Student(), Layout()
    grids = sigma_grid(4, 6, 3)
    clean, endpoints, index = backward_simulation(
        model, layout, torch.zeros(1, 3, 4), grids, True, False, return_segment=True
    )
    assert index == 2 and layout.calls == 3
    sum(x.sum() for x in endpoints).backward()
    assert model.velocities[0].grad is None
    assert model.velocities[1].grad is None
    assert model.velocities[2].grad is not None
    assert model.velocities[3].grad is None
    assert all(x.requires_grad for x in clean)


def test_fake_targets_keep_fp32_clean_and_velocity_identity():
    clean = (torch.randn(1, 3, 4), torch.randn(1, 2, 5))
    sigmas = (torch.tensor(0.04), torch.tensor(0.02))
    noisy, velocity = score_training_pair(clean, sigmas)
    for x0, x, v, sigma in zip(clean, noisy, velocity, sigmas):
        assert v.dtype == torch.float32
        torch.testing.assert_close(x - sigma * v, x0)


def test_config_rejects_empty_segments_and_preserves_legacy_default():
    assert H3Config().validate().algorithm == "dmd"
    assert H3Config().validate().max_steps == 1000
    assert H3Config(algorithm="tdm").validate().max_steps == 300
    assert H3Config(algorithm="tdm", max_steps=1000).validate().max_steps == 1000
    with pytest.raises(ValueError, match="Every TDM"):
        H3Config(algorithm="tdm", score_sigma_max=0.7).validate()
    with pytest.raises(ValueError, match="algorithm"):
        H3Config(algorithm="other").validate()


@pytest.mark.parametrize(
    "saved_algorithm,current_algorithm", [("dmd", "tdm"), ("tdm", "dmd")]
)
def test_resume_rejects_cross_algorithm_states_before_loading(
    tmp_path, monkeypatch, saved_algorithm, current_algorithm
):
    saved = H3Config(algorithm=saved_algorithm).to_dict()
    if saved_algorithm == "dmd":
        saved.pop("algorithm")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "world_size": 8,
                "manifest_sha256": "fixture",
                "config": saved,
            }
        )
    )
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="algorithm"):
        checkpoint.restore_checkpoint(
            tmp_path, H3Config(algorithm=current_algorithm), None, None, [], "fixture"
        )
