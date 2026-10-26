"""Single-node DDP runner for dense Wan trajectory distillation."""

import hashlib
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .architecture import WanSpec
from .checkpoint import (
    restore_checkpoint,
    runtime_versions,
    save_checkpoint,
    source_metadata,
)
from .config import parse_config
from .data import PromptDataset
from .evaluate import Evaluator
from .flow import (
    decompose,
    forward_noise,
    generator_loss,
    predict,
    renoise,
    rollout,
    sample_training_sigmas,
    schedule,
    select_states,
    velocity_training_pair,
)
from .models import clone_student, load_transformer, model_autocast, trainable
from .optimization import distillation_update
from .runtime import distributed, preserve_rng, require_finite


def run_training():
    cfg = parse_config()
    output = Path(cfg.output)
    if (output / "metrics.jsonl").exists() and not cfg.resume:
        raise FileExistsError(f"Use a new output directory or --resume: {output}")
    spec = WanSpec.load(cfg.model)
    shape = spec.latent_shape(cfg.batch_size, cfg.frames, cfg.height, cfg.width)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_cudnn_sdp(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training runner")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    if rank == 0:
        print(f"Initialized {world} rank(s); loading base Transformer", flush=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset = PromptDataset(cfg.embeddings)
    manifest_hash = hashlib.sha256(
        (Path(cfg.embeddings) / "manifest.json").read_bytes()
    ).hexdigest()
    actual_hash = hashlib.sha256(
        (Path(cfg.model) / "transformer/config.json").read_bytes()
    ).hexdigest()
    if dataset.manifest["transformer_config_sha256"] != actual_hash or dataset.manifest[
        "model"
    ] != str(Path(cfg.model).resolve()):
        raise ValueError("Prompt embeddings were prepared for a different base model")
    if len(dataset) < cfg.batch_size * world:
        raise ValueError("Dataset must contain at least one full global batch")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=cfg.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=cfg.workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed + rank),
    )
    teacher = load_transformer(
        cfg.model, device, attention_backend=cfg.attention_backend
    )
    student = clone_student(
        teacher,
        cfg.rank,
        cfg.lora_alpha,
        cfg.gradient_checkpointing,
        target_modules=cfg.lora_target_modules,
        training_mode=cfg.training_mode,
    )
    fake = clone_student(
        teacher,
        cfg.rank,
        cfg.lora_alpha,
        cfg.gradient_checkpointing,
        target_modules=cfg.lora_target_modules,
        training_mode=cfg.training_mode,
    )
    student_params, fake_params = trainable(student), trainable(fake)
    optimizer = torch.optim.AdamW(
        student_params,
        lr=cfg.learning_rate,
        betas=(0.0, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )
    fake_optimizer = torch.optim.AdamW(
        fake_params,
        lr=cfg.fake_learning_rate,
        betas=(0.0, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )
    lr_scheduler = get_scheduler(
        cfg.lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )
    student_ddp = (
        DDP(
            student,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=cfg.training_mode == "full",
        )
        if world > 1
        else student
    )
    fake_ddp = (
        DDP(
            fake,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=cfg.training_mode == "full",
        )
        if world > 1
        else fake
    )
    # Identical model initialization across ranks, independent subsequent trajectories.
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    output.mkdir(parents=True, exist_ok=True)
    step, cursor = 0, {"epoch": 0, "batch": 0}
    if cfg.resume:
        step, cursor = restore_checkpoint(
            cfg.resume,
            cfg,
            student,
            fake,
            (optimizer, fake_optimizer),
            rank,
            world,
            manifest_hash,
            scheduler=lr_scheduler,
        )
    stop_step = cfg.stop_after_steps or cfg.max_steps
    if step > stop_step:
        raise ValueError("The stopping step cannot be lower than the resumed step")
    source = source_metadata()
    if rank == 0:
        (output / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
        (output / "runtime.json").write_text(
            json.dumps(runtime_versions(), indent=2) + "\n"
        )
        (output / "model-spec.json").write_text(
            json.dumps(
                {
                    **spec.to_dict(),
                    "latent_shape": shape,
                    "tokens": spec.tokens(shape),
                    "training_timestep_layout": "batch",
                    "validation_sampler": cfg.validation_sampler,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "world_size": world,
                    "gpu": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "trainable_student": sum(p.numel() for p in student_params),
                    "trainable_fake": sum(p.numel() for p in fake_params),
                    "training_mode": cfg.training_mode,
                    "update_order": cfg.update_order,
                    "effective_batch": world * cfg.batch_size * cfg.accumulation_steps,
                }
            ),
            flush=True,
        )
    neg = dataset.negative().to(device, dtype=torch.bfloat16)
    sigmas = schedule(cfg.distill_steps, device)
    epoch, offset = cursor["epoch"], cursor["batch"]
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    optimizer.zero_grad(set_to_none=True)
    fake_optimizer.zero_grad(set_to_none=True)
    with preserve_rng(device):
        evaluator = (
            Evaluator(cfg, teacher, device, rank, world)
            if cfg.validation_embeddings
            else None
        )

    def evaluate(current_step):
        if evaluator is None:
            return
        started = time.perf_counter()
        with preserve_rng(device):
            evaluator.run(student, current_step)
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        if world > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if rank == 0:
            with (output / "evaluation.jsonl").open("a") as f:
                f.write(
                    json.dumps({"step": current_step, "seconds": elapsed.item()}) + "\n"
                )

    def microbatches():
        nonlocal epoch, offset, iterator
        for micro in range(cfg.accumulation_steps):
            try:
                embeddings = next(iterator)
            except StopIteration:
                epoch, offset = epoch + 1, 0
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                embeddings = next(iterator)
            offset += 1
            embeddings = embeddings.to(device, dtype=torch.bfloat16, non_blocking=True)
            with torch.no_grad(), model_autocast():
                noise = torch.randn(shape, device=device, dtype=torch.bfloat16)
                states, _ = rollout(
                    student, noise, embeddings, cfg.distill_steps, cfg.eta
                )
            yield micro, embeddings, states

    def backward_fake(batch):
        micro, embeddings, states = batch
        sync = micro + 1 == cfg.accumulation_steps
        context = fake_ddp.no_sync() if not sync and world > 1 else nullcontext()
        with context, model_autocast():
            # Fit the fake score on detached student trajectories.
            with torch.no_grad():
                indices = torch.randint(
                    cfg.distill_steps, (cfg.batch_size,), device=device
                )
                selected = select_states(states, indices)
                s, mid = sigmas[indices], sigmas[indices + 1]
                v = predict(student, selected, s, embeddings)
                x0, eps = decompose(selected, v, s)
                t = sample_training_sigmas(mid)
                noisy, target_v = velocity_training_pair(
                    forward_noise(renoise(x0, eps, mid, cfg.eta), mid, t),
                    x0,
                    t,
                    target_from_fp32=cfg.fake_target_from_fp32,
                )
                real_v = (
                    predict(teacher, noisy, t, embeddings) if cfg.lambda_reg else None
                )
            fake_v = predict(fake_ddp, noisy, t, embeddings)
            fake_loss = (fake_v - target_v).square().mean()
            reg_loss = (
                (fake_v - real_v).square().mean()
                if real_v is not None
                else fake_loss.new_zeros(())
            )
            critic_loss = fake_loss + cfg.lambda_reg * reg_loss
            require_finite(critic_loss, "fake loss")
            (critic_loss / cfg.accumulation_steps).backward()
            metrics[:2].add_(
                torch.stack([fake_loss.detach(), reg_loss.detach()])
                / cfg.accumulation_steps
            )

    def backward_student(batch):
        micro, embeddings, states = batch
        sync = micro + 1 == cfg.accumulation_steps
        context = student_ddp.no_sync() if not sync and world > 1 else nullcontext()
        with context, model_autocast():
            # Backpropagate only through a single selected student denoising step.
            with torch.no_grad():
                indices = torch.randint(
                    cfg.distill_steps, (cfg.batch_size,), device=device
                )
                selected = select_states(states, indices)
                s, mid = sigmas[indices], sigmas[indices + 1]
                t = sample_training_sigmas(mid)
            v = predict(student_ddp, selected, s, embeddings)
            x0, eps = decompose(selected, v, s)
            with torch.no_grad():
                noisy = forward_noise(renoise(x0, eps, mid, cfg.eta), mid, t).to(
                    torch.bfloat16
                )
                # Batch conditional and negative predictions into a single teacher call.
                teacher_v = predict(
                    teacher,
                    torch.cat([noisy, noisy]),
                    torch.cat([t, t]),
                    torch.cat([embeddings, neg.expand(cfg.batch_size, -1, -1)]),
                )
                cond, uncond = teacher_v.chunk(2)
                guided_v = uncond + cfg.guidance_scale * (cond - uncond)
                real_x0, _ = decompose(noisy, guided_v, t)
                fake_x0, _ = decompose(noisy, predict(fake, noisy, t, embeddings), t)
            loss = generator_loss(x0, real_x0, fake_x0)
            require_finite(loss, "generator loss")
            (loss / cfg.accumulation_steps).backward()
            metrics[2].add_(loss.detach() / cfg.accumulation_steps)

    evaluate(step)
    while step < stop_step:
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        metrics = torch.zeros(3, device=device)
        used_lr = optimizer.param_groups[0]["lr"]
        student_norm, fake_norm = distillation_update(
            microbatches(),
            backward_fake,
            backward_student,
            student_params,
            fake_params,
            optimizer,
            fake_optimizer,
            lr_scheduler,
            cfg.max_grad_norm,
            cfg.update_order,
        )
        step += 1
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
        if world > 1:
            dist.all_reduce(metrics)
            metrics /= world
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            record = {
                "step": step,
                "student_lr": used_lr,
                "fake_lr": fake_optimizer.param_groups[0]["lr"],
                "fake_loss": metrics[0].item(),
                "regularization_loss": metrics[1].item(),
                "generator_loss": metrics[2].item(),
                "student_grad_norm": student_norm.item(),
                "fake_grad_norm": fake_norm.item(),
                "seconds": elapsed.item(),
                "samples_per_second": world
                * cfg.batch_size
                * cfg.accumulation_steps
                / elapsed.item(),
                "peak_allocated_gib": peak.item(),
            }
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if step % cfg.checkpoint_every == 0 or step == stop_step:
            save_checkpoint(
                output,
                step,
                cfg,
                student,
                fake,
                (optimizer, fake_optimizer),
                {"epoch": epoch, "batch": offset},
                rank,
                world,
                manifest_hash,
                scheduler=lr_scheduler,
                source=source,
            )
        if step % cfg.validation_every == 0 or step == stop_step:
            evaluate(step)


def main():
    try:
        run_training()
    finally:
        if distributed():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
