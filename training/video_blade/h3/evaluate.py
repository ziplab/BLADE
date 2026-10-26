"""Collective FSDP2 sampling and native video/stereo-audio visualization."""

import json
import subprocess
import wave
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from safetensors.torch import save_file

from ..data import PromptDataset
from ..runtime import preserve_rng, require_finite
from .flow import H3Layout, rollout, sigma_grid
from .train import rollout_residency, validate_manifest


@torch.no_grad()
def decode(cfg, layout, samples, root, device):
    video_latents, audio_latents = layout.unpack(samples)
    vae = (
        AutoencoderKLMiniMaxH3.from_pretrained(
            cfg.model,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        .requires_grad_(False)
        .eval()
        .to(device)
    )
    vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    decoded = vae.decode(video_latents * std + mean, return_dict=False)[0].float()
    pixels_mean = decoded.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    pixels_std = decoded.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    video = (
        ((decoded * pixels_std + pixels_mean).clamp(0, 1)[0].permute(1, 2, 3, 0) * 255)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    del vae, decoded
    audio_vae = (
        AutoencoderKLMiniMaxH3Audio.from_pretrained(
            cfg.model,
            subfolder="audio_vae",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        .requires_grad_(False)
        .eval()
        .to(device)
    )
    mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    audio = (
        audio_vae.decode(audio_latents * std + mean, return_dict=False)[0]
        .float()[:, 0]
        .transpose(0, 1)
        .cpu()
        .numpy()
    )
    del audio_vae
    silent = root.with_suffix(".silent.mp4")
    wav = root.with_suffix(".wav")
    imageio.mimwrite(
        silent, video, fps=24, codec="libx264", quality=8, macro_block_size=1
    )
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(32000)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-i",
            str(wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(root.with_suffix(".mp4")),
        ],
        check=True,
    )
    silent.unlink()
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(cfg, teacher, student, device, step):
    rank, world = dist.get_rank(), dist.get_world_size()
    dataset = PromptDataset(cfg.validation_embeddings)
    validate_manifest(dataset, cfg)
    count = min(cfg.validation_count, len(dataset))
    pending = []
    with preserve_rng(device):
        for role, model in (("teacher", teacher), ("student", student)):
            if model is None:
                continue
            parent = Path(cfg.output) / "validation" / role
            if role == "student":
                parent /= f"step-{step:06d}"
            parent.mkdir(parents=True, exist_ok=True)
            existing = [
                (parent / f"prompt-{i:03d}.json").is_file() for i in range(count)
            ]
            if role == "teacher" and all(existing):
                continue
            steps = cfg.teacher_steps if role == "teacher" else cfg.distill_steps
            shifts = (
                (12.0, 3.0) if role == "teacher" else (cfg.video_shift, cfg.audio_shift)
            )
            grids = sigma_grid(steps, *shifts, device)
            for start in range(0, count, world):
                index = start + rank
                # Every rank must execute the same number of FSDP forwards.
                actual = min(index, count - 1)
                embeds = dataset[actual].unsqueeze(0).to(device)
                layout = H3Layout(
                    cfg.height, cfg.width, cfg.frames, embeds.shape[1], device
                )
                generator = torch.Generator(device=device).manual_seed(
                    cfg.validation_seed + actual
                )
                with rollout_residency(model, cfg.retain_rollout_weights):
                    _, samples = rollout(
                        model, layout, embeds, grids, generator=generator
                    )
                for sample in samples:
                    require_finite(sample, "H3 validation latents")
                if index < count:
                    root = parent / f"prompt-{index:03d}"
                    save_file(
                        {
                            "video": samples[0].cpu().contiguous(),
                            "audio": samples[1].cpu().contiguous(),
                        },
                        str(root.with_suffix(".safetensors")),
                    )
                    pending.append(
                        (
                            layout,
                            tuple(x.cpu() for x in samples),
                            root,
                            {
                                "prompt": dataset.entries[index]["prompt"],
                                "seed": cfg.validation_seed + index,
                                "role": role,
                                "algorithm": cfg.algorithm,
                                "step": step if role == "student" else None,
                                "transformer_forwards": steps,
                                "video_shift": shifts[0],
                                "audio_shift": shifts[1],
                                "cfg": 1.0,
                                "fps": 24,
                                "audio_sample_rate": 32000,
                            },
                        )
                    )
        dist.barrier()
        for layout, samples, root, metadata in pending:
            decode(cfg, layout, tuple(x.to(device) for x in samples), root, device)
            root.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        dist.barrier()
