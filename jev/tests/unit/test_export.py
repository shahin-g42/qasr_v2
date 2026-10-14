from __future__ import annotations

import json

import pytest
import torch

from jev import export
from jev.calibrate import write_calibration
from jev.checkpoint import save_checkpoint
from jev.export import (
    checkpoint_identity,
    export_from_state,
    export_model,
    load_model_from_checkpoint,
)
from jev.modeling import load_model
from jev_contracts import atomic_write_json, write_manifest

from .test_modeling import runtime as _runtime
from .test_modeling import snapshot_fixture
from .test_processing import make_row

# Cross-module fixture re-exported under its fixture name so pytest resolves it
# here; the parameters below intentionally reuse that name.
runtime = _runtime


def stepped_decision_model(snapshot_path):
    model = load_model(snapshot_path)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3,
    )
    batch = model.prepare([make_row()])
    model(batch).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def step_once(model, optimizer):
    batch = model.prepare([make_row()])
    model(batch).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def sealed_checkpoint(tmp_path, model, optimizer, *, update=5, identity=None):
    step_once(model, optimizer)
    directory = tmp_path / f"step_{update:012d}"
    save_checkpoint(
        directory, model=model, optimizer=optimizer, run_state={"update": update},
        identity=identity or {"config": "export"}, mesh_shape=(1, 1), update=update,
    )
    return directory


def test_checkpoint_identity_binds_descriptor_and_manifest(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    model, optimizer = stepped_decision_model(path)
    directory = sealed_checkpoint(tmp_path, model, optimizer)
    identity = checkpoint_identity(directory)
    assert len(identity["manifest"]) == 64
    assert identity["descriptor"]["identity"] == {"config": "export"}
    forged = tmp_path / "forged"
    forged.mkdir()
    descriptor = {"schema_version": 1, "kind": "jev-checkpoint", "update": 5,
                  "identity": {"config": "other"}, "mesh": [1, 1],
                  "run": {"update": 5}, "world_size": 1, "torch": "test"}
    sealed = {**descriptor, "identity": {"config": "export"}}
    atomic_write_json(forged / "descriptor.json", descriptor)
    write_manifest(forged, ["descriptor.json"],
                   {"kind": "jev-checkpoint", "descriptor": sealed})
    with pytest.raises(ValueError, match="disagree"):
        checkpoint_identity(forged)
    unsealed = tmp_path / "unsealed"
    unsealed.mkdir()
    # An unsealed directory surfaces as the missing manifest file, the same
    # contract behavior the artifacts suite pins for verify_manifest.
    with pytest.raises(FileNotFoundError):
        checkpoint_identity(unsealed)


def test_reshard_export_and_reload_roundtrip(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    model, optimizer = stepped_decision_model(path)
    with torch.no_grad():
        original = model.readout.weight.clone()
        model.readout.weight.add_(1.0)
    directory = sealed_checkpoint(tmp_path, model, optimizer, update=6)
    loaded = load_model_from_checkpoint(directory, path, initialization="autojev")
    torch.testing.assert_close(loaded.readout.weight, model.readout.weight)
    assert not torch.equal(loaded.readout.weight, original)
    destination = tmp_path / "bundle"
    decision = export_model(loaded, destination, temperature=1.5, metadata={"phase": "test"})
    assert decision["temperature"] == 1.5
    assert decision["metadata"]["phase"] == "test"
    assert (destination / "manifest.json").is_file()
    reloaded = load_model(destination)
    assert reloaded.temperature == 1.5
    torch.testing.assert_close(reloaded.readout.weight, loaded.readout.weight)
    with pytest.raises(FileExistsError):
        export_model(loaded, destination, temperature=1.5)


def test_export_refuses_nonfinite_parameters(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    model, optimizer = stepped_decision_model(path)
    with torch.no_grad():
        model.readout.weight[0, 0] = float("nan")
    directory = tmp_path / "step_000000000005"
    save_checkpoint(
        directory, model=model, optimizer=optimizer, run_state={"update": 5},
        identity={"config": "export"}, mesh_shape=(1, 1), update=5,
    )
    with pytest.raises(ValueError, match="nonfinite"):
        load_model_from_checkpoint(directory, path, initialization="autojev")


def test_export_model_validates_temperature_and_metadata(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    model = load_model(path)
    with pytest.raises(ValueError, match="temperature"):
        export_model(model, tmp_path / "never", temperature=0.0)
    for bad in (-1.0, float("inf"), True):
        with pytest.raises(ValueError):
            export_model(model, tmp_path / "never", temperature=bad)
    with pytest.raises(ValueError, match="metadata"):
        export_model(model, tmp_path / "never", temperature=1.0, metadata=["bad"])


def test_export_from_state_rejects_incomplete_state(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    model = load_model(path)
    full_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    destination = tmp_path / "from-state"
    decision = export_from_state(destination, path, initialization="autojev",
                                 full_state=full_state, temperature=2.0)
    assert decision["temperature"] == 2.0
    broken = {key: value for key, value in full_state.items() if "visual" not in key}
    with pytest.raises(RuntimeError):
        export_from_state(tmp_path / "broken", path, initialization="autojev",
                          full_state=broken, temperature=2.0)


def cli_arguments(checkpoint, snapshot, destination, *, calibration=None, temperature=None):
    arguments = [
        "--checkpoint", str(checkpoint), "--snapshot", str(snapshot),
        "--initialization", "autojev", "--destination", str(destination),
    ]
    if calibration is not None:
        arguments += ["--calibration", str(calibration)]
    if temperature is not None:
        arguments += ["--temperature", str(temperature)]
    return arguments


def test_cli_exports_with_an_explicit_temperature(tmp_path, runtime, capsys):
    path, _, _ = snapshot_fixture(tmp_path)
    model, optimizer = stepped_decision_model(path)
    directory = sealed_checkpoint(tmp_path, model, optimizer)
    destination = tmp_path / "bundle"
    assert export.main(cli_arguments(directory, path, destination, temperature=1.25)) == 0
    decision = json.loads((destination / "decision_config.json").read_text(encoding="utf-8"))
    assert decision["temperature"] == 1.25
    assert decision["metadata"]["source_checkpoint"] == checkpoint_identity(directory)["manifest"]
    printed = json.loads(capsys.readouterr().out)
    assert printed["temperature"] == 1.25 and printed["codes"] == 255
    with pytest.raises(SystemExit, match="Provide"):
        export.main(cli_arguments(directory, path, tmp_path / "other"))


def test_cli_binds_a_matching_calibration_artifact(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    model, optimizer = stepped_decision_model(path)
    directory = sealed_checkpoint(tmp_path, model, optimizer)
    manifest_hash = checkpoint_identity(directory)["manifest"]
    artifact = {
        "schema_version": 1,
        "temperature": {"schema_version": 1, "temperature": 1.75},
        "policy": {"schema_version": 1, "threshold": 0.5},
        "verification": {"schema_version": 1, "verified": True},
    }
    matching = write_calibration(tmp_path / "matching", artifact,
                                 identity={"checkpoint": manifest_hash,
                                           "release": "r" * 64})
    assert matching["identity"]["checkpoint"] == manifest_hash
    destination = tmp_path / "bundle"
    assert export.main(
        cli_arguments(directory, path, destination, calibration=tmp_path / "matching"),
    ) == 0
    decision = json.loads((destination / "decision_config.json").read_text(encoding="utf-8"))
    assert decision["temperature"] == 1.75
    assert decision["metadata"]["calibration"]["checkpoint"] == manifest_hash
    write_calibration(tmp_path / "wrong", artifact,
                      identity={"checkpoint": "0" * 64, "release": "r" * 64})
    with pytest.raises(SystemExit, match="different checkpoint"):
        export.main(
            cli_arguments(directory, path, tmp_path / "other",
                          calibration=tmp_path / "wrong"),
        )
