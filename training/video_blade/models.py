"""Wan models and LoRA adapters, without sparse-attention dependencies."""

from copy import deepcopy

import torch
from diffusers import WanTransformer3DModel
from peft import LoraConfig

from .architecture import WanSpec


def model_autocast():
    # Sampling and training share FP32 trainable parameters. A cached cast created under
    # no_grad would otherwise be reused without autograd in the next train forward.
    return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)


def load_transformer(
    path,
    device,
    rank=None,
    alpha=None,
    checkpointing=False,
    attention_backend="native",
    target_modules=None,
):
    WanSpec.load(path)
    model = WanTransformer3DModel.from_pretrained(
        path, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.requires_grad_(False)
    model = model.to(device).eval()
    model.set_attention_backend(attention_backend)
    return (
        add_lora(model, rank, alpha, checkpointing, target_modules=target_modules)
        if rank is not None
        else model
    )


def add_lora(model, rank, alpha, checkpointing=False, target_modules=None):
    model.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=True,
            target_modules=target_modules
            if target_modules is not None
            else ["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if checkpointing:
        model.enable_gradient_checkpointing()
    return model.train()


def clone_student(
    teacher, rank, alpha, checkpointing=False, target_modules=None, training_mode="lora"
):
    # Copy the resident frozen backbone on GPU instead of reading it from EFS three times.
    if training_mode == "full":
        model = deepcopy(teacher).float().requires_grad_(True).train()
        if checkpointing:
            model.enable_gradient_checkpointing()
        return model
    if training_mode != "lora":
        raise ValueError(f"Unsupported training mode: {training_mode}")
    return add_lora(
        deepcopy(teacher), rank, alpha, checkpointing, target_modules=target_modules
    )


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]
