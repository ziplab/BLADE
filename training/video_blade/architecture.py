"""Read single-Transformer Wan T2V geometry from the model's own configuration."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WanSpec:
    channels: int
    temporal_scale: int
    spatial_scale: int
    patch_size: tuple[int, int, int]
    expand_timesteps: bool
    flow_shift: float

    @classmethod
    def load(cls, model):
        root = Path(model)
        transformer = json.loads((root / "transformer/config.json").read_text())
        vae = json.loads((root / "vae/config.json").read_text())
        pipeline = json.loads((root / "model_index.json").read_text())
        scheduler = json.loads((root / "scheduler/scheduler_config.json").read_text())
        channels = transformer["in_channels"]
        if (
            pipeline["_class_name"] != "WanPipeline"
            or pipeline.get("boundary_ratio") is not None
            or pipeline.get("transformer_2", [None, None]) != [None, None]
            or channels not in (16, 48)
            or transformer["out_channels"] != channels
            or vae["z_dim"] != channels
        ):
            raise ValueError("Expected single-Transformer Wan2.1 T2V or Wan2.2 TI2V-5B")
        if (
            scheduler.get("prediction_type") != "flow_prediction"
            or not scheduler.get("use_flow_sigmas")
            or scheduler["num_train_timesteps"] != 1000
        ):
            raise ValueError("Expected a 1000-timestep flow-prediction scheduler")
        if len(vae["latents_mean"]) != channels or len(vae["latents_std"]) != channels:
            raise ValueError("VAE normalization does not match the latent channels")
        return cls(
            channels=channels,
            temporal_scale=vae.get("scale_factor_temporal", 4),
            spatial_scale=vae.get("scale_factor_spatial", 8),
            patch_size=tuple(transformer["patch_size"]),
            expand_timesteps=pipeline.get("expand_timesteps", False),
            flow_shift=scheduler["flow_shift"],
        )

    def latent_shape(self, batch, frames, height, width):
        pt, ph, pw = self.patch_size
        latent_frames = (frames - 1) // self.temporal_scale + 1
        if (
            (frames - 1) % self.temporal_scale
            or latent_frames % pt
            or height % (self.spatial_scale * ph)
            or width % (self.spatial_scale * pw)
        ):
            raise ValueError(
                f"Invalid Wan video geometry: frames={frames}, height={height}, width={width}; "
                f"VAE strides={self.temporal_scale}/{self.spatial_scale}, patches={self.patch_size}"
            )
        return (
            batch,
            self.channels,
            latent_frames,
            height // self.spatial_scale,
            width // self.spatial_scale,
        )

    def tokens(self, latent_shape):
        return math.prod(s // p for s, p in zip(latent_shape[2:], self.patch_size))

    def to_dict(self):
        return asdict(self)
