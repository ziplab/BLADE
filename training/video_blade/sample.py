"""Render checkpoints with published UniPC inference or the training trajectory."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

from .architecture import WanSpec
from .checkpoint import load_model_weights
from .data import PromptDataset
from .flow import rollout, sampling_rng
from .models import load_transformer, model_autocast
from .sampling import make_pipeline


def main(default_sampler="unipc"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--embeddings", type=Path)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=int, help="Default: checkpoint validation_fps")
    p.add_argument(
        "--sampler",
        choices=("unipc", "trajectory", "checkpoint"),
        default=default_sampler,
        help="Use UniPC, a training trajectory, or the checkpoint validation sampler",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Also render a 50-step CFG teacher using the same noise",
    )
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        p.error("This sampling runner requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    if args.sampler == "checkpoint":
        args.sampler = cfg.get("validation_sampler", "unipc")
    fps = args.fps if args.fps is not None else cfg.get("validation_fps", 16)
    if fps <= 0:
        p.error("fps must be positive")
    spec = WanSpec.load(cfg["model"])
    shape = spec.latent_shape(1, cfg["frames"], cfg["height"], cfg["width"])
    dataset = PromptDataset(args.embeddings or cfg["embeddings"])
    actual_hash = hashlib.sha256(
        (Path(cfg["model"]) / "transformer/config.json").read_bytes()
    ).hexdigest()
    if dataset.manifest["transformer_config_sha256"] != actual_hash or dataset.manifest[
        "model"
    ] != str(Path(cfg["model"]).resolve()):
        raise ValueError("Prompt embeddings were prepared for a different base model")
    embeds = dataset[args.index].unsqueeze(0).to(args.device, dtype=torch.bfloat16)
    training_mode = cfg.get("training_mode", "lora")
    model = load_transformer(
        cfg["model"],
        args.device,
        cfg["rank"] if training_mode == "lora" else None,
        cfg["lora_alpha"] if training_mode == "lora" else None,
        attention_backend=cfg.get("attention_backend", "native"),
        target_modules=cfg.get("lora_target_modules"),
    ).eval()
    if training_mode == "full":
        model.float()
    adapter_targets = (
        sorted(model.peft_config["default"].target_modules)
        if training_mode == "lora"
        else None
    )
    load_model_weights(model, args.checkpoint, "student", training_mode)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    noise = torch.randn(
        shape, device=args.device, dtype=torch.bfloat16, generator=generator
    )
    # The stochastic trajectory uses the default device generator.
    with sampling_rng(device, args.seed):
        with torch.inference_mode(), model_autocast():
            if args.sampler == "trajectory":
                _, latents = rollout(
                    model, noise, embeds, cfg["distill_steps"], cfg["eta"]
                )
            else:
                pipe = make_pipeline(cfg["model"], model)
                latents = pipe(
                    prompt_embeds=embeds,
                    height=cfg["height"],
                    width=cfg["width"],
                    num_frames=cfg["frames"],
                    num_inference_steps=cfg["distill_steps"],
                    guidance_scale=1.0,
                    latents=noise,
                    output_type="latent",
                ).frames.float()
                del pipe
    del model
    vae = AutoencoderKLWan.from_pretrained(
        cfg["model"], subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    ).to(args.device)
    vae.enable_tiling()
    mean = latents.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = latents.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    with torch.inference_mode():
        decoded = vae.decode((latents * std + mean).float(), return_dict=False)[0]
    from diffusers.video_processor import VideoProcessor

    frames = VideoProcessor(spec.spatial_scale).postprocess_video(
        decoded, output_type="np"
    )[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=fps)
    metadata = {
        "prompt": dataset.entries[args.index]["prompt"],
        "index": args.index,
        "seed": args.seed,
        "distill_steps": cfg["distill_steps"],
        "sampler": args.sampler,
        "guidance_scale": 1.0,
        "eta": cfg["eta"]
        if args.sampler == "trajectory"
        else None,
        "flow_shift": spec.flow_shift
        if args.sampler == "unipc"
        else cfg.get("trajectory_shift"),
        "expand_timesteps": spec.expand_timesteps if args.sampler == "unipc" else False,
        "height": cfg["height"],
        "width": cfg["width"],
        "frames": cfg["frames"],
        "fps": fps,
        "checkpoint": str(args.checkpoint),
        "training_mode": training_mode,
        "lora_target_modules": adapter_targets,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved {args.output}", flush=True)
    if args.baseline:
        teacher = load_transformer(
            cfg["model"],
            args.device,
            attention_backend=cfg.get("attention_backend", "native"),
        )
        pipe = make_pipeline(cfg["model"], teacher, vae)
        with torch.inference_mode():
            frames = pipe(
                prompt_embeds=embeds,
                negative_prompt_embeds=dataset.negative().to(
                    args.device, dtype=torch.bfloat16
                ),
                height=cfg["height"],
                width=cfg["width"],
                num_frames=cfg["frames"],
                num_inference_steps=50,
                guidance_scale=cfg["guidance_scale"],
                latents=noise.clone(),
            ).frames[0]
        baseline_path = args.output.with_name(args.output.stem + "-teacher.mp4")
        export_to_video(frames, str(baseline_path), fps=fps)
        print(f"Saved {baseline_path}", flush=True)


if __name__ == "__main__":
    main()
