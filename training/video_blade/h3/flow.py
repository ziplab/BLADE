"""H3 packed layouts and paired flow paths using native modality schedules."""

from dataclasses import dataclass

import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    audio_latent_num_frames,
    video_latent_num_frames,
)

from ..flow import decompose, mix, renoise


def shifted_sigma(base, shift):
    return shift * base / (1 + (shift - 1) * base)


def sigma_grid(steps, video_shift=12.0, audio_shift=3.0, device="cpu"):
    # The public scheduler counts grid points; this API counts model evaluations.
    grids = []
    for shift in (video_shift, audio_shift):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps + 1, device=device)
        if scheduler.num_inference_steps != steps:
            raise ValueError("The shifted schedule collapsed distinct grid points")
        grids.append(scheduler.sigmas)
    return tuple(grids)


@dataclass
class H3Layout:
    height: int
    width: int
    frames: int
    text_length: int
    device: torch.device | str = "cpu"
    video_channels: int = 24
    audio_channels: int = 32

    def __post_init__(self):
        self.latent_frames = video_latent_num_frames(self.frames, 17, 5)
        self.audio_frames = audio_latent_num_frames(self.frames)
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.ones(self.text_length, dtype=torch.long),
            self.latent_frames,
            self.height // 16,
            self.width // 16,
            self.audio_frames,
            (1, 2, 2),
            2,
            2,
            0,
        )
        names = (
            "position_ids",
            "token_tags",
            "video_indices",
            "audio_indices",
            "text_indices",
        )
        self.arguments = {
            name: tensor.to(self.device) for name, tensor in zip(names, layout[:5])
        }
        # Keep two entries even when their values coincide at sigma=1. This has
        # identical AdaLN semantics and avoids a GPU synchronization per forward.
        self.timestep_indices = torch.zeros_like(self.arguments["token_tags"])
        self.timestep_indices[self.arguments["audio_indices"]] = 1

    def noise(self, generator=None):
        video = torch.randn(
            1,
            self.video_channels,
            self.latent_frames,
            self.height // 16,
            self.width // 16,
            device=self.device,
            generator=generator,
        )
        video = patchify_video_latents(video, (1, 2, 2)).unsqueeze(0)
        audio = torch.randn(
            1,
            self.audio_frames * 2,
            self.audio_channels,
            device=self.device,
            generator=generator,
        )
        return video, audio

    def predict(self, model, samples, sigmas, embeddings):
        result = model(
            hidden_states=samples[0],
            audio_hidden_states=samples[1],
            encoder_hidden_states=embeddings,
            timestep=1
            - torch.stack(
                [torch.as_tensor(s, device=self.device).reshape(()) for s in sigmas]
            ),
            timestep_indices=self.timestep_indices,
            **self.arguments,
            return_dict=False,
        )
        # The shared TDM algebra uses epsilon-x0. H3 predicts x0-epsilon.
        return tuple(-value.float() for value in result)

    def unpack(self, samples):
        video, audio = samples
        video = video.reshape(
            1,
            self.latent_frames,
            self.height // 32,
            self.width // 32,
            self.video_channels,
            1,
            2,
            2,
        )
        video = video.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            1,
            self.video_channels,
            self.latent_frames,
            self.height // 16,
            self.width // 16,
        )
        audio = audio.reshape(2, self.audio_frames, self.audio_channels).permute(
            0, 2, 1
        )
        return video.contiguous(), audio.contiguous()


def paired_decompose(samples, velocities, sigmas):
    parts = [
        decompose(x, v, torch.as_tensor(s, device=x.device).reshape(1))
        for x, v, s in zip(samples, velocities, sigmas)
    ]
    return tuple(p[0] for p in parts), tuple(p[1] for p in parts)


def paired_renoise(x0, eps, sigmas, eta):
    return tuple(
        renoise(x, e, torch.as_tensor(s, device=x.device).reshape(1), eta)
        for x, e, s in zip(x0, eps, sigmas)
    )


def score_training_pair(x0, sigmas):
    noise = tuple(torch.randn_like(x) for x in x0)
    noisy = tuple(mix(x, e, s.reshape(1)) for x, e, s in zip(x0, noise, sigmas))
    return noisy, tuple(e - x.float() for e, x in zip(noise, x0))


def modality_sum(values, audio_weight=1.0):
    # Match the public H3 recipe: sum independent per-modality means.
    return values[0] + audio_weight * values[1]


def paired_mse(predictions, targets, audio_weight=1.0):
    return modality_sum(
        tuple((p - t).square().mean() for p, t in zip(predictions, targets)),
        audio_weight,
    )


def dmd_loss(x0, teacher_x0, fake_x0):
    with torch.no_grad():
        normalizer = (x0.float() - teacher_x0.float()).abs().mean().clamp_min(1e-8)
        direction = (fake_x0.float() - teacher_x0.float()) / normalizer
    target = (x0.float() - direction).detach()
    return 0.5 * (x0.float() - target).square().mean()


def paired_generator_loss(x0, teacher_x0, fake_x0, audio_weight=1.0):
    return modality_sum(
        tuple(dmd_loss(x, t, f) for x, t, f in zip(x0, teacher_x0, fake_x0)),
        audio_weight,
    )


@torch.no_grad()
def rollout(model, layout, embeddings, grids, eta=1.0, generator=None):
    current = layout.noise(generator)
    states = []
    for index in range(len(grids[0]) - 1):
        states.append(current)
        sigma = tuple(grid[index] for grid in grids)
        velocity = layout.predict(model, current, sigma, embeddings)
        x0, epsilon = paired_decompose(current, velocity, sigma)
        if index + 2 < len(grids[0]):
            current = paired_renoise(
                x0, epsilon, tuple(grid[index + 1] for grid in grids), eta
            )
    return states, x0
