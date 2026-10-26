"""Dense H3 DMD and paired-clock TDM training on FSDP2."""

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.utils.data import DataLoader, DistributedSampler

from ..checkpoint import runtime_versions, source_metadata
from ..data import PromptDataset
from ..models import trainable
from ..runtime import require_finite
from .checkpoint import export_adapter, restore_checkpoint, save_checkpoint
from .config import parse_config
from .flow import (
    H3Layout,
    paired_decompose,
    paired_generator_loss,
    paired_mse,
    score_training_pair,
    shifted_sigma,
    sigma_grid,
)
from .models import checkpoint_policy, clip_gradients, load_sharded
from .tdm import paired_endpoint_loss, query_sigmas, trajectory_query


@contextmanager
def rollout_residency(model, enabled):
    """Retain frozen rollout weights across successive forwards, then release."""
    if enabled:
        model.set_reshard_after_forward(False)
    try:
        yield
    finally:
        if enabled:
            model.set_reshard_after_forward(True)
            for module in reversed(list(model.modules())):
                if isinstance(module, FSDPModule):
                    module.reshard()


def backward_simulation(
    student, layout, embeddings, grids, gradient, retain_weights, return_segment=False
):
    # The number of FSDP collectives must be identical on every rank. Prompts,
    # noise and score sigma remain independent; only the exit index is shared.
    end = torch.randint(len(grids[0]) - 1, (), device=embeddings.device)
    if dist.is_initialized():
        dist.broadcast(end, src=0)
    end = end.item()
    current = layout.noise()
    with rollout_residency(student, retain_weights):
        for index in range(end + 1):
            with torch.set_grad_enabled(gradient and index == end):
                sigma = tuple(grid[index] for grid in grids)
                velocity = layout.predict(student, current, sigma, embeddings)
                clean, _ = paired_decompose(current, velocity, sigma)
                if index < end or return_segment:
                    current = tuple(
                        x + (grid[index + 1] - grid[index]) * v
                        for x, v, grid in zip(current, velocity, grids)
                    )
    return (clean, current, end) if return_segment else clean


def validate_manifest(dataset, cfg):
    manifest = dataset.manifest
    if (
        manifest.get("family") != "minimax_h3"
        or manifest.get("text_encoder_layer") != 50
    ):
        raise ValueError("H3 requires native layer-50 Qwen3-VL prompt embeddings")
    digest = hashlib.sha256(
        (Path(cfg.model) / "transformer/config.json").read_bytes()
    ).hexdigest()
    if (
        manifest["model"] != str(Path(cfg.model).resolve())
        or manifest["transformer_config_sha256"] != digest
    ):
        raise ValueError("Prompt embeddings belong to another H3 model")


def run_training():
    cfg = parse_config()
    output = Path(cfg.output)
    if (output / "metrics.jsonl").exists() and not cfg.resume:
        raise FileExistsError(f"Use a new output or --resume: {output}")
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(8)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = not cfg.deterministic
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    dataset = PromptDataset(cfg.embeddings)
    validate_manifest(dataset, cfg)
    if len(dataset) < world:
        raise ValueError("Prompt cache must hold at least one complete global batch")
    manifest_hash = hashlib.sha256(
        (Path(cfg.embeddings) / "manifest.json").read_bytes()
    ).hexdigest()
    mesh = init_device_mesh("cuda", (world,))
    teacher = load_sharded(cfg, "teacher", mesh, device)
    student = load_sharded(cfg, "student", mesh, device)
    fake = load_sharded(cfg, "fake", mesh, device)
    # All roles use eval-mode forwards, as in the reference. Autograd and the
    # explicit checkpointing flags independently control trainable execution.
    teacher.eval()
    student.eval()
    fake.eval()
    parameters = [trainable(model) for model in (student, fake)]
    optimizers = [
        torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.0, cfg.adam_beta2),
            weight_decay=cfg.weight_decay,
            fused=True,
        )
        for params, lr in zip(parameters, (cfg.learning_rate, cfg.fake_learning_rate))
    ]
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
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
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(cfg.seed + rank),
    )
    step, cursor = 0, {"epoch": 0, "batch": 0}
    if cfg.resume:
        step, cursor = restore_checkpoint(
            cfg.resume, cfg, student, fake, optimizers, manifest_hash
        )
    epoch, offset = cursor["epoch"], cursor["batch"]
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    grids = sigma_grid(cfg.distill_steps, cfg.video_shift, cfg.audio_shift, device)
    layouts = {}
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for filename, value in (
            ("config.json", cfg.to_dict()),
            ("source.json", source_metadata()),
            ("runtime.json", runtime_versions()),
            (
                "model-spec.json",
                {
                    "family": "minimax_h3",
                    "algorithm": cfg.algorithm,
                    "task": "t2va",
                    "student_parameters": sum(p.numel() for p in parameters[0]),
                    "fake_parameters": sum(p.numel() for p in parameters[1]),
                    "world_size": world,
                    "effective_student_batch": world * cfg.accumulation_steps,
                    "fake_updates_per_student": cfg.fake_update_ratio,
                    "student_sigmas": [g.tolist() for g in grids],
                    "training_order": "student_then_fake",
                    "student_query": (
                        "q(x_sigma | x_segment_endpoint)"
                        if cfg.algorithm == "tdm"
                        else "q(x_sigma | predicted_clean)"
                    ),
                    "student_loss": (
                        "normalized endpoint L2; lambda=sigma^2/(1-sigma)^2"
                        if cfg.algorithm == "tdm"
                        else "normalized clean L2"
                    ),
                    "fake_query": "direct q(x_sigma | predicted_clean)",
                    "fake_loss": "velocity MSE; clean weight=1/sigma^2; no importance sampling",
                    "precision": "FP32 trainable masters and reductions; native mixed-precision forward",
                },
            ),
        ):
            (output / filename).write_text(json.dumps(value, indent=2) + "\n")
        print((output / "model-spec.json").read_text(), flush=True)

    def next_batch():
        nonlocal epoch, offset, iterator
        try:
            embeddings = next(iterator)
        except StopIteration:
            epoch, offset = epoch + 1, 0
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            embeddings = next(iterator)
        offset += 1
        embeddings = embeddings.to(device, non_blocking=True)
        length = embeddings.shape[1]
        if length not in layouts:
            # Bound layout caching for the full, variable-length prompt corpus.
            if len(layouts) >= 8:
                layouts.pop(next(iter(layouts)))
            layouts[length] = H3Layout(
                cfg.height, cfg.width, cfg.frames, length, device
            )
        return embeddings, layouts[length]

    def stage(is_student, apply_update=True):
        stage_started = time.perf_counter()
        model_index = 0 if is_student else 1
        optimizer = optimizers[model_index]
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        segment_records = []
        for _ in range(cfg.accumulation_steps):
            embeddings, layout = next_batch()
            simulation = backward_simulation(
                student,
                layout,
                embeddings,
                grids,
                is_student,
                cfg.retain_rollout_weights,
                return_segment=cfg.algorithm == "tdm",
            )
            if cfg.algorithm == "tdm":
                clean, endpoints, index = simulation
                lows = tuple(grid[index + 1] for grid in grids)
                base_sigma, sigmas = query_sigmas(
                    index,
                    cfg.distill_steps,
                    cfg.video_shift,
                    cfg.audio_shift,
                    cfg.score_sigma_min,
                    cfg.score_sigma_max,
                    device,
                )
                if rank == 0:
                    segment_records.append(
                        {
                            "segment": index,
                            "base_query": base_sigma.item(),
                            "high": [grid[index].item() for grid in grids],
                            "low": [value.item() for value in lows],
                            "query": [value.item() for value in sigmas],
                        }
                    )
            else:
                clean = simulation
                base_sigma = torch.rand((), device=device).clamp(
                    cfg.score_sigma_min, cfg.score_sigma_max
                )
                sigmas = (
                    shifted_sigma(base_sigma, cfg.video_shift),
                    shifted_sigma(base_sigma, cfg.audio_shift),
                )
            with torch.no_grad():
                if cfg.algorithm == "tdm" and is_student:
                    noisy = trajectory_query(endpoints, lows, sigmas)
                else:
                    noisy, target = score_training_pair(clean, sigmas)
            if is_student:
                with torch.no_grad():
                    fake_prediction = layout.predict(fake, noisy, sigmas, embeddings)
                    teacher_prediction = layout.predict(
                        teacher, noisy, sigmas, embeddings
                    )
                    fake_x0, _ = paired_decompose(noisy, fake_prediction, sigmas)
                    teacher_x0, _ = paired_decompose(noisy, teacher_prediction, sigmas)
                if cfg.algorithm == "tdm":
                    loss = paired_endpoint_loss(
                        clean,
                        endpoints,
                        teacher_x0,
                        fake_x0,
                        lows,
                        cfg.audio_loss_weight,
                    )
                else:
                    loss = paired_generator_loss(
                        clean, teacher_x0, fake_x0, cfg.audio_loss_weight
                    )
            else:
                prediction = layout.predict(fake, noisy, sigmas, embeddings)
                loss = paired_mse(prediction, target, cfg.audio_loss_weight)
            require_finite(loss, "H3 student loss" if is_student else "H3 fake loss")
            (loss / cfg.accumulation_steps).backward()
            loss_total += loss.detach() / cfg.accumulation_steps
        norm = clip_gradients(parameters[model_index], cfg.max_grad_norm)
        if apply_update:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if apply_update and rank == 0:
            with (output / "stages.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "student_update": step + 1,
                            "role": "student" if is_student else "fake",
                            "loss_rank0": loss_total.item(),
                            "global_grad_norm": norm.item(),
                            "seconds": time.perf_counter() - stage_started,
                            "cursor": {"epoch": epoch, "batch": offset},
                            "algorithm": cfg.algorithm,
                            "segments": segment_records,
                        }
                    )
                    + "\n"
                )
        return loss_total, norm

    def benchmark_residency():
        nonlocal epoch, offset, iterator
        original = cfg.retain_rollout_weights
        original_skip = cfg.checkpoint_skip_every
        saved_cursor = (epoch, offset)
        results = []
        # Compare matched layouts, noise, exit steps and unchanged model weights.
        profiles = (
            [(False, 0), (False, 8)]
            if cfg.benchmark_partial_checkpointing
            else [(value, original_skip) for value in (False, True, True, False)]
        )
        for retained, skip_every in profiles:
            epoch, offset = saved_cursor
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            for _ in range(offset):
                next(iterator)
            cfg.retain_rollout_weights = retained
            for network in (student, fake):
                checkpoint_policy(network, skip_every)
            with torch.random.fork_rng(devices=[device.index]):
                record = {
                    "retain_rollout_weights": retained,
                    "checkpoint_skip_every": skip_every,
                }
                torch.cuda.reset_peak_memory_stats()
                for role, is_student in (("student", True), ("fake", False)):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    loss, norm = stage(is_student, apply_update=False)
                    torch.cuda.synchronize()
                    values = torch.tensor(
                        [
                            time.perf_counter() - started,
                            torch.cuda.max_memory_allocated() / 2**30,
                        ],
                        device=device,
                    )
                    dist.all_reduce(values, op=dist.ReduceOp.MAX)
                    record[f"{role}_seconds"] = values[0].item()
                    record[f"{role}_peak_gib"] = values[1].item()
                    record[f"{role}_loss_rank0"] = loss.item()
                    record[f"{role}_grad_norm"] = norm.item()
                results.append(record)
                if cfg.benchmark_partial_checkpointing and skip_every == 8:
                    baseline_peak = max(
                        results[0]["student_peak_gib"], results[0]["fake_peak_gib"]
                    )
                    measured_peak = max(
                        record["student_peak_gib"], record["fake_peak_gib"]
                    )
                    projected = baseline_peak + 2 * max(
                        0, measured_peak - baseline_peak
                    )
                    budget = (
                        0.8
                        * torch.cuda.get_device_properties(device).total_memory
                        / 2**30
                    )
                    shared_budget = torch.tensor(budget, device=device)
                    dist.all_reduce(shared_budget, op=dist.ReduceOp.MIN)
                    budget = shared_budget.item()
                    record["projected_skip4_peak_gib"] = projected
                    record["skip4_budget_gib"] = budget
                    record["skip4_test_authorized_by_memory_budget"] = (
                        projected <= budget
                    )
                    if projected <= budget:
                        profiles.append((False, 4))
                if rank == 0:
                    print(
                        f"Rollout residency benchmark: {json.dumps(record)}", flush=True
                    )
        cfg.retain_rollout_weights = original
        for network in (student, fake):
            checkpoint_policy(network, original_skip)
        epoch, offset = saved_cursor
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        if rank == 0:
            (output / "rollout-residency-benchmark.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )

    def evaluate(current_step):
        if not cfg.validation_embeddings:
            return
        from .evaluate import evaluate

        evaluate(cfg, teacher, student, device, current_step)

    evaluate(step)
    stop = cfg.stop_after_steps or cfg.max_steps
    if step > stop:
        raise ValueError("The stopping step is lower than the checkpoint step")
    while step < stop:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        student_loss, student_norm = stage(True)
        student_seconds = time.perf_counter() - started
        fake_loss, fake_norm = (
            torch.zeros((), device=device),
            torch.zeros((), device=device),
        )
        for _ in range(cfg.fake_update_ratio):
            loss, norm = stage(False)
            fake_loss += loss / cfg.fake_update_ratio
            fake_norm = torch.maximum(fake_norm, norm)
        step += 1
        torch.cuda.synchronize()
        seconds = torch.tensor(time.perf_counter() - started, device=device)
        metrics = torch.stack([student_loss, fake_loss])
        dist.all_reduce(metrics)
        metrics /= world
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        if rank == 0:
            record = {
                "step": step,
                "fake_updates": step * cfg.fake_update_ratio,
                "generator_loss": metrics[0].item(),
                "fake_loss": metrics[1].item(),
                "student_grad_norm": student_norm.item(),
                "fake_grad_norm_max": fake_norm.item(),
                "student_lr": cfg.learning_rate,
                "fake_lr": cfg.fake_learning_rate,
                "seconds": seconds.item(),
                "student_stage_seconds": student_seconds,
                "peak_allocated_gib": peak.item(),
                "student_samples_per_second": world
                * cfg.accumulation_steps
                / seconds.item(),
            }
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if step == stop and cfg.benchmark_rollout_residency:
            benchmark_residency()
        if step % cfg.checkpoint_every == 0 or (
            step == stop and cfg.save_final_checkpoint
        ):
            with torch.random.fork_rng(devices=[device.index]):
                save_checkpoint(
                    output,
                    step,
                    cfg,
                    student,
                    fake,
                    optimizers,
                    {"epoch": epoch, "batch": offset},
                    manifest_hash,
                )
        elif step % cfg.adapter_every == 0 or step == stop:
            with torch.random.fork_rng(devices=[device.index]):
                export_adapter(output, step, cfg, student)
        if step % cfg.validation_every == 0 or step == stop:
            evaluate(step)


def main():
    try:
        run_training()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
