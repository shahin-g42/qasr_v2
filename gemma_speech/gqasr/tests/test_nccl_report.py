"""Read existing diagnostics and bind distributed workers to their local GPU."""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from gqasr.distributed import Runtime
from gqasr.nccl_report import collect, inspect_trace

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("cuda", [True, False])
def test_runtime_binds_local_device_not_global_rank(monkeypatch, cuda):
    monkeypatch.setenv("RANK", "73")
    monkeypatch.setenv("WORLD_SIZE", "96")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    select, init = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "set_device", select)
    monkeypatch.setattr(torch.distributed, "init_process_group", init)
    rt = Runtime("bf16")
    if cuda:
        select.assert_called_once_with(1)
        init.assert_called_once_with("nccl", device_id=torch.device("cuda", 1))
        assert rt.device == torch.device("cuda", 1)
    else:
        select.assert_not_called()
        init.assert_called_once_with("gloo")


def entry(sequence, name="nccl:all_reduce", state="scheduled"):
    return {
        "record_id": sequence, "collective_seq_id": sequence,
        "process_group": ("0", "default_pg"),
        "profiling_name": name, "input_sizes": [[1]], "output_sizes": [[1]],
        "input_dtypes": ["Int"], "state": state,
        "frames": [{"name": "all_finite", "filename": "distributed.py", "line": 53}],
    }


def write_trace(path, entries):
    path.write_bytes(pickle.dumps({
        "version": "2.10", "pg_config": {"0": {"ranks": "[0, 1]"}},
        "pg_status": {"0": {"last_enqueued_collective": 16,
                            "last_completed_collective": 15}},
        "entries": entries,
    }, protocol=2))


def test_preserves_requested_collective_and_tail_for_missing_ranks(tmp_path):
    write_trace(tmp_path / "nccl_trace_rank_72", [entry(i) for i in range(10, 20)])
    write_trace(tmp_path / "nccl_trace_rank_73", [entry(i) for i in range(10, 16)])
    write_trace(tmp_path / "nccl_trace_rank_8", [entry(16)])
    log = tmp_path / "gqasr-encoder-attempt-9.log"
    log.write_text("GQASR stage=encoder host=node056\n"
                   "Watchdog caught collective operation timeout: SeqNum=16\n"
                   "#2 all_finite from distributed.py:53\n")
    report = collect(tmp_path, node_rank=9)
    assert [trace["rank"] for trace in report["traces"]] == [72, 73]
    a, b = report["traces"]
    assert [e["collective_seq_id"] for e in a["entries"]] == [16, 18, 19]
    assert a["entries"][0]["frames"][0]["name"] == "all_finite"
    assert a["entries"][0]["input_dtypes"] == ["Int"]
    assert a["requested_sequence_present"]
    assert not b["requested_sequence_present"]
    assert [e["collective_seq_id"] for e in b["entries"]] == [14, 15]
    assert report["latest_log"]["file"] == str(log)
    assert not report["root_cause_confirmed"]
    json.dumps(report)


def test_partial_and_missing_dumps_remain_missing_evidence(tmp_path):
    (tmp_path / "nccl_trace_rank_72").write_bytes(b"\x80\x04")
    result = collect(tmp_path, node_rank=9)
    assert "read_error" in result["traces"][0]
    assert result["latest_log"] is None
    assert not result["root_cause_confirmed"]
    assert collect(tmp_path, node_rank=0)["traces"] == []
    assert "read_error" in inspect_trace(tmp_path / "missing", 16)


def test_trace_reader_rejects_executable_pickle(tmp_path):
    path = tmp_path / "trace"
    # GLOBAL/REDUCE would call eval when loaded by unrestricted pickle.load.
    path.write_bytes(b"cbuiltins\neval\n(S'40 + 2'\ntR.")
    assert "Refusing executable pickle global" in inspect_trace(path, 16)["read_error"]


def test_cli_writes_report_without_importing_torch(tmp_path):
    write_trace(tmp_path / "nccl_trace_rank_73", [entry(16)])
    output = tmp_path / "reports/node-9.json"
    # Running the stdlib-only script directly also works if the ML environment
    # is unusable; the normal entry point is python -m gqasr.nccl_report.
    source = ROOT / "gemma_speech/gqasr/src/gqasr/nccl_report.py"
    result = subprocess.run(
        [sys.executable, "-S", str(source), "--log-dir", str(tmp_path),
         "--node-rank", "9", "--output", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["trace_ranks"] == [73]
    report = json.loads(output.read_text())
    assert not report["root_cause_confirmed"] and len(report["traces"]) == 1


@pytest.mark.parametrize("overrides,expected", [
    ({}, ("2000", "/tmp/local-diagnostics/nccl_trace_rank_")),
    ({"TORCH_NCCL_TRACE_BUFFER_SIZE": "4000",
      "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/legacy_"}, ("4000", "/tmp/legacy_")),
    ({"TORCH_NCCL_TRACE_BUFFER_SIZE": "4000", "TORCH_FR_BUFFER_SIZE": "6000",
      "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/legacy_",
      "TORCH_FR_DUMP_TEMP_FILE": "/tmp/current_"}, ("6000", "/tmp/current_")),
])
def test_launcher_uses_current_flight_recorder_variables(tmp_path, overrides, expected):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_gqasr_12node.sh").write_text(
        '#!/bin/bash\nprintf "%s\\n%s\\n%s\\n" "$TORCH_FR_BUFFER_SIZE" '
        '"$TORCH_FR_DUMP_TEMP_FILE" "$TORCH_NCCL_DUMP_ON_TIMEOUT"\n'
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("TORCH_FR_", "TORCH_NCCL_", "GQASR_"))}
    env.update(PROJECT_DIR=str(tmp_path), DRY_RUN="1",
               GQASR_LOG_DIR="/tmp/local-diagnostics", **overrides)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/resume_gqasr_encoder_12node.sh"), "9"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines()[-3:] == [*expected, "1"]
