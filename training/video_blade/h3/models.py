"""FSDP2 H3 models with FP32 optimizer masters and native compute precision."""

import gc

import torch
import torch.distributed as dist
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from ..runtime import require_finite

LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
FP32_MODULES = (
    "proj_in",
    "audio_proj_in",
    "time_embedder",
    "proj_out",
    "audio_proj_out",
)


def checkpoint_policy(model, skip_every=0):
    if not hasattr(model, "_h3_full_checkpoint"):
        model._h3_full_checkpoint = model._gradient_checkpointing_func
    original = model._h3_full_checkpoint
    indices = {id(block): index for index, block in enumerate(model.transformer_blocks)}

    def checkpoint(module, *args, **kwargs):
        index = indices.get(id(module))
        if skip_every and index is not None and (index + 1) % skip_every == 0:
            return module(*args, **kwargs)
        return original(module, *args, **kwargs)

    model.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint)


def configure_role(model, role, rank=128, alpha=128.0, checkpointing=True):
    if role not in {"student", "fake", "teacher"}:
        raise ValueError(f"Unknown model role: {role}")
    model.requires_grad_(False)
    if role == "fake":
        model.float().requires_grad_(True)
    if role == "student":
        model.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=LORA_TARGETS,
                init_lora_weights=True,
            )
        )
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    if checkpointing and role != "teacher":
        model.enable_gradient_checkpointing()
    return model.train(role != "teacher")


def native_dtypes(model):
    model.to(torch.bfloat16)
    model.rope.inv_freq = model.rope.inv_freq.float()
    for name in FP32_MODULES:
        getattr(model, name).float()
    return model


def shard_model(model, mesh):
    bf16 = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    fp32 = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
    )
    for block in (*model.token_refiner.refiner_blocks, *model.transformer_blocks):
        fully_shard(block, mesh=mesh, mp_policy=bf16, reshard_after_forward=True)
    for name in FP32_MODULES:
        fully_shard(
            getattr(model, name), mesh=mesh, mp_policy=fp32, reshard_after_forward=True
        )
    fully_shard(model, mesh=mesh, mp_policy=bf16, reshard_after_forward=True)
    return model


def load_sharded(cfg, role, mesh, device):
    # Rank zero reads EFS once. No GPU ever materializes the complete 33B model.
    state = {}
    if dist.get_rank() == 0:
        print(f"Reading H3 weights for {role}", flush=True)
        source = MiniMaxH3Transformer3DModel.from_pretrained(
            cfg.model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        configure_role(
            source, role, cfg.rank, cfg.lora_alpha, cfg.gradient_checkpointing
        )
        state = source.state_dict()
        del source
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(
            MiniMaxH3Transformer3DModel.load_config(cfg.model, subfolder="transformer")
        )
        native_dtypes(model)
        configure_role(
            model, role, cfg.rank, cfg.lora_alpha, cfg.gradient_checkpointing
        )
        if cfg.gradient_checkpointing and role != "teacher":
            checkpoint_policy(model, cfg.checkpoint_skip_every)
    if cfg.attention_backend == "flash_4":
        from .attention import use_flash4

        use_flash4(model)
    else:
        model.set_attention_backend(cfg.attention_backend)
    shard_model(model, mesh)
    model.to_empty(device=device)
    # Nonpersistent buffers are absent from checkpoints and to_empty discards them.
    rope = model.rope
    rope.inv_freq.copy_(
        1.0
        / (
            model.config.rope_theta
            ** (
                torch.arange(0, 2 * rope.rope_freq_dim, 2, device=device).float()
                / (2 * rope.rope_freq_dim)
            )
        )
    )
    set_model_state_dict(
        model,
        state,
        options=StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=True, strict=True
        ),
    )
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return model


def clip_gradients(parameters, maximum):
    # DTensor's norm redistributes shards, giving the global model norm. The
    # scalar must be replicated before using the ordinary runtime finite check.
    norm = torch.nn.utils.clip_grad_norm_(parameters, maximum, foreach=False)
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    require_finite(norm, "global FSDP2 gradient norm")
    return norm
