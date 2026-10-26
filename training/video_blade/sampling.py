"""Construct published Wan sampling pipelines without losing model metadata."""

from diffusers import UniPCMultistepScheduler, WanPipeline
from diffusers.video_processor import VideoProcessor

from .architecture import WanSpec


def make_pipeline(model_path, transformer, vae=None):
    spec = WanSpec.load(model_path)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )
    pipe = WanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        expand_timesteps=spec.expand_timesteps,
    )
    # A pipeline without a VAE otherwise falls back to Wan2.1's 8x spatial stride.
    pipe.vae_scale_factor_temporal = spec.temporal_scale
    pipe.vae_scale_factor_spatial = spec.spatial_scale
    pipe.video_processor = VideoProcessor(spec.spatial_scale)
    pipe.set_progress_bar_config(disable=True)
    return pipe
