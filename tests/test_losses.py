from __future__ import annotations

import torch

from ddh.losses import longitudinal_loss, negative_log_likelihood


def test_longitudinal_loss_is_zero_for_baseline_only_batch() -> None:
    x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.double)
    prediction = torch.zeros_like(x, requires_grad=True)

    loss = longitudinal_loss(prediction, x)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None


def test_negative_log_likelihood_matches_explicit_formula() -> None:
    """Guards the per-sample refactor: same sum-over-batch / K reduction as before."""
    outcomes = [
        torch.tensor([[0.1, 0.2, 0.1], [0.05, 0.05, 0.1]], dtype=torch.double),
        torch.tensor([[0.2, 0.1, 0.05], [0.3, 0.1, 0.05]], dtype=torch.double),
    ]
    cif = [torch.cumsum(o, 1) for o in outcomes]
    t = torch.tensor([2, 2])
    e = torch.tensor([1, 0])          # patient 0 has the event, patient 1 is censored
    s = torch.tensor([1, 1])

    expected = 0.0
    for i, contribution in enumerate(
        [outcomes[0][0, 2], 1 - cif[0][1, 2] - cif[1][1, 2]]
    ):
        survival = (1 - cif[0][i, 1] - cif[1][i, 1]).clamp(min=1e-3)
        expected += float(torch.log(contribution + 1e-10) - torch.log(survival))

    loss = negative_log_likelihood(outcomes, cif, t, e, s=s)
    assert abs(float(loss) - (-expected / 2)) < 1e-12
