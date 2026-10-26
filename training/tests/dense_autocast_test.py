"""Guard gradients when no-grad rollout precedes a mixed-precision train forward."""

import pytest
import torch

from video_blade.flow import decompose, generator_loss, predict, rollout, sampling_rng
from video_blade.models import model_autocast


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_rollout_preserves_subsequent_student_gradients():
    class TinyVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = torch.nn.Linear(
                2, 2, bias=False, device="cuda", dtype=torch.float32
            )

        def forward(self, hidden_states, **kwargs):
            return (self.lora(hidden_states),)

    model = TinyVelocity()
    noise = torch.randn(1, 2, 1, 2, 2, device="cuda", dtype=torch.bfloat16)
    sigma = torch.ones(1, device="cuda")
    with model_autocast():
        states, _ = rollout(model, noise, None, steps=2, eta=0.9)
        v = predict(model, states[0], sigma, None)
        x0, _ = decompose(states[0], v, sigma)
        loss = generator_loss(x0, torch.zeros_like(x0), torch.ones_like(x0))
        assert loss.requires_grad
        loss.backward()
    assert torch.isfinite(model.lora.weight.grad).all()
    assert model.lora.weight.grad.abs().sum() > 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_sampling_seed_targets_selected_device_and_restores_rng():
    torch.cuda.init()
    before = [torch.cuda.get_rng_state(i) for i in (0, 1)]
    with sampling_rng("cuda:1", 123):
        first = torch.randn(16, device="cuda:1")
    with sampling_rng("cuda:1", 123):
        second = torch.randn(16, device="cuda:1")
    assert torch.equal(first, second)
    assert all(torch.equal(before[i], torch.cuda.get_rng_state(i)) for i in (0, 1))
