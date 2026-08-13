from __future__ import annotations

import torch

from ddh.losses import longitudinal_loss


def test_longitudinal_loss_is_zero_for_baseline_only_batch() -> None:
    x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.double)
    prediction = torch.zeros_like(x, requires_grad=True)

    loss = longitudinal_loss(prediction, x)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None
