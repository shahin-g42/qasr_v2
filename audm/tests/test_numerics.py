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
