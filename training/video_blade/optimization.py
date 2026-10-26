"""Coordinate optimizer updates without changing microbatch loss scaling."""

import torch

from .runtime import require_finite


def distillation_update(
    batches,
    backward_fake,
    backward_student,
    student_params,
    fake_params,
    optimizer,
    fake_optimizer,
    scheduler,
    max_grad_norm,
    update_order="synchronous",
):
    """Run callbacks that each divide losses by the accumulation count.

    Alternating updates retain detached trajectories while the complete fake
    batch is accumulated, then train the student against the updated fake.
    """
    if update_order not in {"synchronous", "alternating"}:
        raise ValueError(f"Unsupported update order: {update_order}")
    if update_order == "alternating":
        cached = []
        for batch in batches:
            backward_fake(batch)
            cached.append(batch)
        fake_norm = torch.nn.utils.clip_grad_norm_(fake_params, max_grad_norm)
        require_finite(fake_norm, "fake gradient norm")
        fake_optimizer.step()
        fake_optimizer.zero_grad(set_to_none=True)
        for batch in cached:
            backward_student(batch)
    else:
        for batch in batches:
            backward_fake(batch)
            backward_student(batch)

    student_norm = torch.nn.utils.clip_grad_norm_(student_params, max_grad_norm)
    if update_order == "synchronous":
        fake_norm = torch.nn.utils.clip_grad_norm_(fake_params, max_grad_norm)
    require_finite(torch.stack([student_norm, fake_norm]), "gradient norm")
    optimizer.step()
    scheduler.step()
    if update_order == "synchronous":
        fake_optimizer.step()
        fake_optimizer.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    return student_norm, fake_norm
