import pytest
import torch
from gqasr.gradients import clip_gradients


class Runtime:
    rank = 0

    def all_finite(self, value):
        return bool(torch.isfinite(value).all())


def test_standard_norm_overflow_still_fails_closed():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 1e30)
    with pytest.raises(FloatingPointError):
        clip_gradients(model, 1.0, Runtime(), 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_bad_gradient_reports_parameter(value, capsys):
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, value)
    with pytest.raises(FloatingPointError, match="update 4"):
        clip_gradients(model, 1.0, Runtime(), 3)
    assert '"parameters": ["weight"]' in capsys.readouterr().out


def test_matches_standard_clipping():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    assert clip_gradients(model, 1.0, Runtime(), 0).item() == 5.0
    torch.testing.assert_close(model.weight.grad, torch.tensor([[0.6, 0.8]]))
