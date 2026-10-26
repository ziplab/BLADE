"""Distributed held-out sampling without changing the training RNG streams."""

import hashlib
import json
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

from .architecture import WanSpec
from .data import PromptDataset
from .flow import rollout, sampling_rng
from .models import model_autocast
from .sampling import make_pipeline


class Evaluator:
    def __init__(self, cfg, teacher, device, rank, world):
        self.cfg, self.teacher, self.device = cfg, teacher, device
        self.spec = WanSpec.load(cfg.model)
        self.rank, self.world = rank, world
        self.dataset = PromptDataset(cfg.validation_embeddings)
        actual = hashlib.sha256(
            (Path(cfg.model) / "transformer/config.json").read_bytes()
        ).hexdigest()
        if self.dataset.manifest[
            "transformer_config_sha256"
        ] != actual or self.dataset.manifest["model"] != str(Path(cfg.model).resolve()):
            raise ValueError(
                "Validation embeddings were prepared for a different model"
            )
        train_manifest = json.loads(
            (Path(cfg.embeddings) / "manifest.json").read_text()
        )
        train_prompts = {e["prompt"].strip() for e in train_manifest["entries"]}
        if any(e["prompt"].strip() in train_prompts for e in self.dataset.entries):
            raise ValueError("Validation prompts overlap with training prompts")
        self.vae = (
            AutoencoderKLWan.from_pretrained(
                cfg.model,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        self.vae.enable_tiling()
        self.processor = VideoProcessor(self.spec.spatial_scale)

    def _write(self, path, frames, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.mp4")
        export_to_video(frames, str(temporary), fps=self.cfg.validation_fps)
        os.replace(temporary, path)
        sidecar = path.with_suffix(".json")
        temporary = sidecar.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, sidecar)

    @torch.inference_mode()
    def run(self, student, step):
        cfg, device = self.cfg, self.device
        root = Path(cfg.output) / "validation"
        self.vae.to(device)
        pipe = make_pipeline(cfg.model, self.teacher, self.vae)
        student_pipe = (
            make_pipeline(cfg.model, student)
            if cfg.validation_sampler == "unipc"
            else None
        )
        was_training = student.training
        student.eval()
        try:
            for index in range(self.rank, len(self.dataset), self.world):
                seed = cfg.validation_seed + index
                prompt = self.dataset.entries[index]["prompt"]
                embeds = (
                    self.dataset[index].unsqueeze(0).to(device, dtype=torch.bfloat16)
                )
                shape = self.spec.latent_shape(1, cfg.frames, cfg.height, cfg.width)
                generator = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                common = {
                    "prompt": prompt,
                    "index": index,
                    "seed": seed,
                    "height": cfg.height,
                    "width": cfg.width,
                    "frames": cfg.frames,
                    "fps": cfg.validation_fps,
                }
                teacher_path = root / "teacher" / f"{index:02d}.mp4"
                if not teacher_path.exists():
                    with sampling_rng(device, seed):
                        frames = pipe(
                            prompt_embeds=embeds,
                            negative_prompt_embeds=self.dataset.negative().to(
                                device, dtype=torch.bfloat16
                            ),
                            height=cfg.height,
                            width=cfg.width,
                            num_frames=cfg.frames,
                            num_inference_steps=50,
                            guidance_scale=cfg.guidance_scale,
                            latents=noise.clone(),
                        ).frames[0]
                    self._write(
                        teacher_path,
                        frames,
                        {
                            **common,
                            "role": "teacher",
                            "steps": 50,
                            "sampler": "unipc",
                            "guidance_scale": cfg.guidance_scale,
                            "flow_shift": self.spec.flow_shift,
                            "expand_timesteps": self.spec.expand_timesteps,
                        },
                    )
                path = root / "student" / f"step-{step:06d}" / f"{index:02d}.mp4"
                if path.exists():
                    continue
                with sampling_rng(device, seed), model_autocast():
                    if student_pipe is None:
                        _, latents = rollout(
                            student, noise, embeds, cfg.distill_steps, cfg.eta
                        )
                    else:
                        latents = student_pipe(
                            prompt_embeds=embeds,
                            height=cfg.height,
                            width=cfg.width,
                            num_frames=cfg.frames,
                            num_inference_steps=cfg.distill_steps,
                            guidance_scale=1.0,
                            latents=noise.clone(),
                            output_type="latent",
                        ).frames.float()
                if not torch.isfinite(latents).all().item():
                    raise FloatingPointError(
                        f"Nonfinite validation latents at step {step}, prompt {index}"
                    )
                mean = latents.new_tensor(self.vae.config.latents_mean).view(
                    1, -1, 1, 1, 1
                )
                std = latents.new_tensor(self.vae.config.latents_std).view(
                    1, -1, 1, 1, 1
                )
                decoded = self.vae.decode(
                    (latents * std + mean).float(), return_dict=False
                )[0]
                if not torch.isfinite(decoded).all().item():
                    raise FloatingPointError(
                        f"Nonfinite decoded validation at step {step}, prompt {index}"
                    )
                frames = self.processor.postprocess_video(decoded, output_type="np")[0]
                metadata = {
                    **common,
                    "role": "student",
                    "step": step,
                    "steps": cfg.distill_steps,
                    "sampler": cfg.validation_sampler,
                    "guidance_scale": 1.0,
                    "flow_shift": self.spec.flow_shift
                    if student_pipe
                    else getattr(cfg, "trajectory_shift", None),
                    "expand_timesteps": self.spec.expand_timesteps
                    if student_pipe
                    else False,
                    "eta": getattr(cfg, "eta", 1.0) if student_pipe is None else None,
                    "pixel_mean": float(frames.mean()),
                    "pixel_std": float(frames.std()),
                    "motion_l1": float(abs(frames[1:] - frames[:-1]).mean())
                    if len(frames) > 1
                    else 0.0,
                }
                self._write(path, frames, metadata)
                print(f"Rank {self.rank}: saved validation {path}", flush=True)
        finally:
            student.train(was_training)
            del pipe, student_pipe
            self.vae.cpu()
            torch.cuda.empty_cache()
