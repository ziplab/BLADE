"""Flow-matching algebra shared by training and distilled sampling.

Wan predicts velocity v = epsilon - x0, with x_sigma = (1-sigma)x0 + sigma*epsilon.
All schedule arithmetic is float32, independently of the model's compute dtype.
"""

import math
from contextlib import contextmanager

import torch


@contextmanager
def sampling_rng(device, seed):
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device), torch.random.fork_rng(devices=[device.index]):
        torch.cuda.manual_seed(seed)
        yield


def expand(sigma, sample):
    return sigma.float().reshape(-1, *([1] * (sample.ndim - 1)))


def predict(model, noisy, sigma, embeds):
    # Pure T2V has a uniform timestep across tokens. Wan accepts the compact [B]
    # form, avoiding redundant per-token timestep embeddings in the 5B model.
    return model(
        hidden_states=noisy,
        timestep=sigma.float() * 1000,
        encoder_hidden_states=embeds,
        return_dict=False,
    )[0].float()


def decompose(noisy, velocity, sigma):
    s = expand(sigma, noisy)
    x0 = noisy.float() - s * velocity.float()
    epsilon = noisy.float() + (1 - s) * velocity.float()
    return x0, epsilon


def mix(x0, epsilon, sigma):
    s = expand(sigma, x0)
    return (1 - s) * x0.float() + s * epsilon.float()


def renoise(x0, epsilon, sigma, eta):
    noise = eta * epsilon + math.sqrt(1 - eta**2) * torch.randn_like(epsilon)
    return mix(x0, noise, sigma)


def forward_noise(sample, sigma_from, sigma_to):
    """Sample the forward Markov transition without dividing by zero at sigma=1."""
    a = 1 - expand(sigma_from, sample)
    b = 1 - expand(sigma_to, sample)
    ratio = b / a
    variance = (
        expand(sigma_to, sample).square()
        - (ratio * expand(sigma_from, sample)).square()
    )
    return ratio * sample.float() + variance.clamp_min(0).sqrt() * torch.randn_like(
        sample
    )


def velocity_training_pair(noisy, x0, sigma, target_from_fp32=False):
    """Quantize model inputs independently of the continuous flow velocity label."""
    model_input = noisy.to(torch.bfloat16)
    label_input = noisy.float() if target_from_fp32 else model_input.float()
    return model_input, (label_input - x0.float()) / expand(sigma, x0)


def schedule(steps, device):
    return torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)


@torch.no_grad()
def rollout(model, noise, embeds, steps, eta):
    sigmas = schedule(steps, noise.device)
    states = []
    current = noise
    for i in range(steps):
        states.append(current)
        sigma = sigmas[i].expand(noise.shape[0])
        velocity = predict(model, current, sigma, embeds)
        x0, epsilon = decompose(current, velocity, sigma)
        if i + 1 < steps:
            current = renoise(
                x0, epsilon, sigmas[i + 1].expand(noise.shape[0]), eta
            ).to(noise.dtype)
    return states, x0


def select_states(states, indices):
    # Gather only the selected samples; avoid stacking all full-resolution trajectory tensors.
    return torch.cat([states[i][j : j + 1] for j, i in enumerate(indices.tolist())])


def sample_training_sigmas(sigma_mid):
    # The upper bound stays above every interval's lower bound, including K=50.
    upper = torch.maximum(torch.full_like(sigma_mid, 0.98), (1 + sigma_mid) / 2)
    return (sigma_mid + (upper - sigma_mid) * torch.rand_like(sigma_mid)).clamp_min(
        0.001
    )


def generator_loss(x0, real_x0, fake_x0):
    target = (x0.float() + real_x0.float() - fake_x0.float()).detach()
    c = 1e-3 * math.sqrt(x0[0].numel() / (64 * 64 * 4))
    # Rationalized pseudo-Huber avoids cancellation for small differences.
    delta2 = (x0.float() - target).square()
    return (delta2 / (torch.sqrt(delta2 + c * c) + c)).mean()
