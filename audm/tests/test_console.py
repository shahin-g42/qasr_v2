"""Real subprocess checks for diagnostic routing, failures and signal forwarding."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/audm/console.py"
# Load by file so subprocesses test the dependency-free relay, without torch.
LOAD = f"""
import importlib.util, logging, os, sys
spec = importlib.util.spec_from_file_location('relay', {str(SCRIPT)!r})
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)
"""


def run_logged(tmp_path, code, *, env=None):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-root",
            str(tmp_path / "logs"),
            "--",
            sys.executable,
            "-c",
            LOAD + code,
        ],
        env=dict(os.environ, **(env or {})),
        capture_output=True,
        text=True,
        timeout=20,
    )


def logs(tmp_path):
    directory = Path((tmp_path / "logs/latest-node0.txt").read_text().strip())
    return directory, (directory / "rank-000.log").read_text()


def test_complete_rank_log_and_small_console(tmp_path):
    result = run_logged(
        tmp_path,
        """
with relay.worker_logging():
    print('raw stdout detail')
    print('raw stderr detail', file=sys.stderr)
    os.write(2, b'native warning\\n')
    logging.info('full model architecture')
    for i in range(30): logging.warning('dataset replacement %d', i)
    relay.emit('Train 20/2000 | loss 1.2345')
    import subprocess
    subprocess.run([sys.executable, '-c', 'import os; os.write(2, b"worker detail\\\\n")'], check=True)
""",
    )
    assert result.returncode == 0, result.stderr
    directory, raw = logs(tmp_path)
    for detail in (
        "raw stdout detail",
        "raw stderr detail",
        "native warning",
        "full model architecture",
        "dataset replacement 29",
        "worker detail",
    ):
        assert detail in raw
        assert detail not in result.stdout
    assert result.stdout.count("WARNING |") == 3
    assert "Train 20/2000 | loss 1.2345" in result.stdout
    assert "30 warning/error records" in result.stdout
    assert (directory / "console.log").read_text() == result.stdout
    assert not result.stderr


def test_rank_one_keeps_details_without_duplicate_progress(tmp_path):
    result = run_logged(
        tmp_path,
        """
with relay.worker_logging():
    print('rank one detail')
    relay.emit('must not duplicate')
""",
        env={"RANK": "1"},
    )
    assert result.returncode == 0
    assert "must not duplicate" not in result.stdout
    directory = Path((tmp_path / "logs/latest-node0.txt").read_text().strip())
    assert "rank one detail" in (directory / "rank-001.log").read_text()


def test_failure_is_visible_and_traceback_retained(tmp_path):
    result = run_logged(
        tmp_path,
        """
with relay.worker_logging():
    raise ValueError('bad manifest at line 14')
""",
    )
    assert result.returncode == 1
    _, raw = logs(tmp_path)
    assert "Traceback (most recent call last)" in raw
    assert "ValueError: bad manifest at line 14" in raw
    assert "FAILED | rank 0 | ValueError: bad manifest at line 14" in result.stdout
    assert "Traceback" not in result.stdout
    assert "FAILED | exit 1" in result.stdout


def test_launcher_failure_and_large_native_output_are_recorded(tmp_path):
    result = run_logged(tmp_path, "print('X' * 100000); sys.exit(7)")
    assert result.returncode == 7
    assert "FAILED | exit 7" in result.stdout
    directory = Path((tmp_path / "logs/latest-node0.txt").read_text().strip())
    assert "X" * 100000 in (directory / "launcher.log").read_text()
    assert "XXX" not in result.stdout


def test_each_launch_retains_previous_logs_and_does_not_create_run_output(tmp_path):
    output = tmp_path / "run"
    for _ in range(2):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--",
                sys.executable,
                "-c",
                "print('ok')",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
    assert not output.exists()
    assert len(list((tmp_path / "logs").glob("*/launcher.log"))) == 2


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_reaches_owned_process_group_and_preserves_exit_status(tmp_path, sig):
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    code = (
        LOAD
        + f"""
import signal, time
from pathlib import Path
def stop(sig, frame):
    Path({str(stopped)!r}).write_text(str(sig))
    raise SystemExit(128 + sig)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
with relay.worker_logging():
    Path({str(ready)!r}).touch()
    while True: time.sleep(.01)
"""
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--log-root",
            str(tmp_path / "logs"),
            "--",
            sys.executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        process.send_signal(sig)
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == 128 + sig, stdout
        assert stopped.read_text() == str(sig)
        assert "Stopping | signal" in stdout
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_callback_formats_metrics_without_mutating_them(monkeypatch):
    from audm.console_callback import ConsoleCallback

    shown = []
    monkeypatch.setattr("audm.console_callback.emit", shown.append)
    callback = ConsoleCallback("audflash", "full")
    args = SimpleNamespace(process_index=0)
    state = SimpleNamespace(global_step=3700, max_steps=10000)
    metrics = dict(loss=1.23, learning_rate=2e-5, grad_norm=3.45)
    before = dict(metrics)
    callback.on_log(args, state, None, metrics)
    callback.on_log(
        args, state, None, dict(eval_wer_en=0.11, eval_cer_en=0.05, eval_wer_macro=0.11)
    )
    assert metrics == before
    assert "Train 3,700/10,000 (37.0%) | loss 1.2300 | grad 3.45 | lr 2.00e-05" in shown
    assert any("WER 11.00% | CER 5.00%" in line for line in shown)
    args.process_index = 1
    count = len(shown)
    callback.on_log(args, state, None, metrics)
    assert len(shown) == count


def test_reject_log_directory_inside_model_output(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-root",
            str(output / "logs"),
            "--",
            sys.executable,
            "-c",
            "pass",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "AUD_LOG_DIR must be outside" in result.stderr
    assert not output.exists()


def test_unicode_events_remain_atomic(tmp_path):
    result = run_logged(
        tmp_path,
        """
with relay.worker_logging():
    relay._EVENT_LIMIT = 512
    relay.emit('unicode ' + '\U0001f600' * 700)
""",
    )
    assert result.returncode == 0, result.stderr
    _, raw = logs(tmp_path)
    assert "unicode" in raw and "unicode" in result.stdout
    assert "@@AUD_CONSOLE@@" not in result.stdout


def test_closed_console_reader_does_not_stop_training(tmp_path):
    marker = tmp_path / "done"
    command = [
        sys.executable,
        str(SCRIPT),
        "--log-root",
        str(tmp_path / "logs"),
        "--",
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.stdout.close()
    assert process.wait(timeout=20) == 0, process.stderr.read().decode()
    process.stderr.close()
    assert marker.exists()
