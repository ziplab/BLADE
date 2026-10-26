"""Check dense cuDNN attention outputs and gradients against native Flash SDPA."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA attention backends required"
)
def test_cudnn_dense_attention_matches_flash():
    torch.manual_seed(123)
    inputs = [
        torch.randn(1, 12, 256, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    results = []
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        tensors = [t.clone().requires_grad_() for t in inputs]
        with sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(*tensors)
            output.float().square().mean().backward()
        results.append([output.detach().float(), *[t.grad.float() for t in tensors]])
    for first, second in zip(*results):
        relative_l2 = (first - second).norm() / first.norm().clamp_min(1e-12)
        assert relative_l2 < 0.01
