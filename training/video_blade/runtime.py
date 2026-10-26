"""Distributed runtime checks shared by training and checkpointing."""

import random
from contextlib import contextmanager

import torch
import torch.distributed as dist


def distributed():
    return dist.is_available() and dist.is_initialized()


@contextmanager
def preserve_rng(device):
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[torch.device(device).index]):
            yield
    finally:
        random.setstate(python_state)


def require_finite(value, label):
    valid = torch.isfinite(value.detach()).all().to(torch.int32)
    if distributed():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise FloatingPointError(
            f"Nonfinite {label}; all ranks stopped without applying this update"
        )
