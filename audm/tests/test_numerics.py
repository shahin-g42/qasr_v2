import json

import torch

from audm.numerics import save_failure


def test_failure_batch_tensor_only_roundtrip(tmp_path):
    value = torch.randn(2, 3, requires_grad=True)
    report = save_failure(
        tmp_path, {"input_features": value, "labels": torch.tensor([[1, -100]]), "unused": object()},
        rank=3, step=1, error=RuntimeError("backward failed"),
    )
    saved = torch.load(report["batch"], weights_only=True)
    torch.testing.assert_close(saved["input_features"], value)
    assert not saved["input_features"].requires_grad
    assert set(saved) == {"input_features", "labels"}
    metadata = json.loads((tmp_path / "numerics/rank-3-step-1.json").read_text())
    assert metadata["tensors"]["input_features"]["finite"]
    assert metadata["error"] == "backward failed"


# Reproduce ZeRO-2's norm sentinel/combination/update order without requiring CUDA.
class Zero2:
    partition_gradients = True
    cpu_offload = False

    def __init__(self, norms):
        self.norms = norms
        self.bit16_groups = [None] * len(norms)
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.optimizer = torch.optim.AdamW([self.parameter], lr=0.1)

    def get_grad_norm_direct(self, index):
        return torch.tensor(self.norms[index])

    def scaled_global_norm(self):
        return torch.linalg.vector_norm(torch.stack([
            self.get_grad_norm_direct(i) for i in range(len(self.bit16_groups))
        ]))

    def step(self):
        self.scaled_global_norm()
        self.parameter.grad = torch.ones_like(self.parameter)
        self.optimizer.step()


def test_zero2_sentinel_rejected_before_any_update():
    import pytest

    from audm.numerics import install_zero2_guard

    optimizer = Zero2([-1.] * 5)
    assert torch.isclose(optimizer.scaled_global_norm(), torch.tensor(5.).sqrt())
    install_zero2_guard(optimizer)
    install_zero2_guard(optimizer)  # idempotent
    with pytest.raises(FloatingPointError, match="optimizer update blocked"):
        optimizer.step()
    torch.testing.assert_close(optimizer.parameter, torch.ones(1))
    assert not optimizer.optimizer.state


def test_zero2_finite_results_preserved():
    from audm.numerics import install_zero2_guard

    baseline, guarded = Zero2([3., 4.]), Zero2([3., 4.])
    install_zero2_guard(guarded)
    baseline.step()
    guarded.step()
    torch.testing.assert_close(baseline.parameter, guarded.parameter)


def test_zero2_offload_and_nonfinite_norms():
    import pytest

    from audm.numerics import install_zero2_guard

    for bad in (float("nan"), float("inf"), -1.):
        optimizer = Zero2([bad])
        optimizer.cpu_offload = True
        optimizer.complete_grad_norm_calculation_for_cpu_offload = optimizer.get_grad_norm_direct
        optimizer.scaled_global_norm = lambda: torch.stack([
            optimizer.complete_grad_norm_calculation_for_cpu_offload(0)
        ]).norm()
        install_zero2_guard(optimizer)
        with pytest.raises(FloatingPointError):
            optimizer.step()
        assert not optimizer.optimizer.state


def test_loss_and_plain_gradient_guards():
    import pytest

    from audm.numerics import check_gradients, check_loss

    check_loss(torch.tensor(1.))
    for bad in (float("nan"), float("inf")):
        with pytest.raises(FloatingPointError):
            check_loss(torch.tensor(bad))
    model = torch.nn.Linear(1, 1)
    model(torch.ones(1, 1)).sum().backward()
    check_gradients(model)
    model.weight.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        check_gradients(model)


def test_unsupported_zero_optimizer_fails_closed():
    import pytest

    from audm.numerics import install_zero2_guard

    with pytest.raises(RuntimeError, match="Unsupported DeepSpeed"):
        install_zero2_guard(object())


def test_zero2_overflow_detection_aborts_instead_of_skipping():
    import pytest

    from audm.numerics import install_zero2_guard

    optimizer = Zero2([1.])
    optimizer.check_grad_overflow = True
    optimizer.device = "cpu"
    optimizer.check_overflow = lambda: setattr(optimizer, "overflow", True)
    install_zero2_guard(optimizer)
    with pytest.raises(FloatingPointError, match="overflow"):
        optimizer.check_overflow()
    assert not optimizer.optimizer.state


def test_trainer_installs_guard_before_deepspeed_internal_step():
    from types import SimpleNamespace

    import pytest

    from audm.trainer import AUDTrainer

    trainer = object.__new__(AUDTrainer)
    trainer.is_deepspeed_enabled = True
    engine = SimpleNamespace(optimizer=Zero2([-1.]))
    # Accelerate's DeepSpeed backward performs the optimizer step internally.
    trainer._aud_training_step = lambda model, inputs, count: model.optimizer.step()
    with pytest.raises(FloatingPointError):
        trainer.training_step(engine, {})
    assert not engine.optimizer.optimizer.state


def test_trainer_checks_plain_accumulated_gradients_before_return():
    from types import SimpleNamespace

    import pytest

    from audm.trainer import AUDTrainer

    trainer = object.__new__(AUDTrainer)
    trainer.is_deepspeed_enabled = False
    trainer.accelerator = SimpleNamespace(sync_gradients=True)
    model = torch.nn.Linear(1, 1)
    model(torch.ones(1, 1)).sum().backward()
    model.weight.grad.fill_(float("nan"))
    trainer._aud_training_step = lambda *args: torch.tensor(1.)
    with pytest.raises(FloatingPointError):
        trainer.training_step(model, {})


def test_trainer_checks_loss_before_backward(monkeypatch):
    from types import SimpleNamespace

    import pytest
    from qasr.train import PredictionLoggingTrainer

    from audm.trainer import AUDTrainer

    trainer = object.__new__(AUDTrainer)
    trainer.args = SimpleNamespace(prediction_loss_only=True)
    model = torch.nn.Linear(1, 1).train()
    monkeypatch.setattr(
        PredictionLoggingTrainer, "compute_loss",
        lambda *args, **kwargs: torch.tensor(float("nan"), requires_grad=True),
    )
    with pytest.raises(FloatingPointError, match="training loss"):
        trainer.compute_loss(model, {"labels": torch.tensor([1])})
    assert all(p.grad is None for p in model.parameters())
