"""Paired-clock TDM with endpoint matching and direct fake denoising."""

import torch

from ..flow import forward_noise
from .flow import modality_sum, shifted_sigma


def query_sigmas(index, steps, video_shift, audio_shift, minimum, maximum, device):
    """Draw one base time strictly inside the shared audio/video segment."""
    lower = torch.tensor(max(1 - (index + 1) / steps, minimum), device=device)
    upper = torch.tensor(min(1 - index / steps, maximum), device=device)
    base = lower + (upper - lower) * torch.rand((), device=device)
    base = base.maximum(torch.nextafter(lower, upper)).minimum(
        torch.nextafter(upper, lower)
    )
    sigmas = []
    for shift in (video_shift, audio_shift):
        lo, hi = shifted_sigma(lower, shift), shifted_sigma(upper, shift)
        # A strict base-time bound can round back onto a shifted endpoint.
        sigmas.append(
            shifted_sigma(base, shift)
            .maximum(torch.nextafter(lo, hi))
            .minimum(torch.nextafter(hi, lo))
        )
    return base, tuple(sigmas)


def trajectory_query(endpoints, lows, sigmas):
    """Diffuse the joint endpoint with independent conditional modality noises."""
    return tuple(
        forward_noise(endpoint.detach(), low.reshape(1), sigma.reshape(1))
        for endpoint, low, sigma in zip(endpoints, lows, sigmas)
    )


def endpoint_loss(clean, endpoint, teacher_clean, fake_clean, low):
    """Normalized L2 endpoint surrogate, with a DMD-compatible time weight.

    For a=1-sigma and r=a_tau/a_low, the forward-chain KL direction is
    r*a_tau/sigma_tau**2 * (fake_clean-teacher_clean). Choosing the explicit
    query weight lambda_tau=sigma_tau**2/a_tau**2 leaves delta/a_low.
    The endpoint remains differentiable; all score and scale terms are detached.
    At low=0 this is exactly the established normalized clean-space DMD loss.
    """
    with torch.no_grad():
        normalizer = (
            (clean.float() - teacher_clean.float()).abs().mean().clamp_min(1e-8)
        )
        direction = (fake_clean.float() - teacher_clean.float()) / (
            (1 - low) * normalizer
        )
        target = endpoint.float() - direction
    return 0.5 * (endpoint.float() - target).square().mean()


def paired_endpoint_loss(
    clean, endpoints, teacher_clean, fake_clean, lows, audio_weight
):
    return modality_sum(
        tuple(
            endpoint_loss(x0, endpoint, teacher, fake, low)
            for x0, endpoint, teacher, fake, low in zip(
                clean, endpoints, teacher_clean, fake_clean, lows
            )
        ),
        audio_weight,
    )
