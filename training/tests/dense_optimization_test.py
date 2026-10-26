"""Check alternating updates against a full-batch analytic SGD reference."""

import pytest
import torch

from video_blade.config import TrainConfig
from video_blade.optimization import distillation_update


@pytest.mark.parametrize("accumulation_steps", [1, 2, 4])
@pytest.mark.parametrize("order", ["synchronous", "alternating"])
def test_accumulation_matches_full_batch_and_student_uses_expected_fake(
    accumulation_steps, order
):
    student = torch.nn.Parameter(torch.tensor(1.0))
    fake = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([student], lr=0.02)
    fake_optimizer = torch.optim.SGD([fake], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    samples = torch.tensor([1.0, 2.0, 3.0, 4.0])
    observed_fake = []
    observed_student = []

    def backward_fake(batch):
        observed_student.append(student.item())
        loss = 0.5 * (fake * batch - student.detach()).square().mean()
        (loss / accumulation_steps).backward()

    def backward_student(batch):
        observed_fake.append(fake.item())
        loss = 0.5 * (student * batch - fake.detach()).square().mean()
        (loss / accumulation_steps).backward()

    student_norm, fake_norm = distillation_update(
        iter(samples.chunk(accumulation_steps)),
        backward_fake,
        backward_student,
        [student],
        [fake],
        optimizer,
        fake_optimizer,
        scheduler,
        100.0,
        order,
    )
    # The complete fake batch gives grad=-mean(x)=-2.5, hence fake=0.25.
    expected_fake = 0.25 if order == "alternating" else 0.0
    expected_student_grad = 7.5 - 2.5 * expected_fake
    assert observed_student == [1.0] * accumulation_steps
    assert observed_fake == pytest.approx([expected_fake] * accumulation_steps)
    assert fake.item() == pytest.approx(0.25)
    assert student.item() == pytest.approx(1.0 - 0.02 * expected_student_grad)
    assert student_norm.item() == pytest.approx(expected_student_grad)
    assert fake_norm.item() == pytest.approx(2.5)
    assert scheduler.last_epoch == 1
    assert student.grad is None and fake.grad is None


def test_invalid_update_order_rejected():
    with pytest.raises(ValueError, match="update_order"):
        TrainConfig(update_order="unknown").validate()
