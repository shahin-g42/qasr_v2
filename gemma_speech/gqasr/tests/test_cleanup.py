import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest

from gqasr import cleanup


ROOT = Path(__file__).resolve().parents[3]


def process(pid=100, ppid=1, argv=("python3", "-m", "gqasr", "train"), **kwargs):
    return cleanup.Process(pid, ppid, 1234 + pid, "S", argv, str(ROOT), **kwargs)


def test_training_tree_and_orphans_are_selected_but_other_jobs_are_not():
    parent = process(argv=("python3", "/env/bin/torchrun", "--nproc_per_node=8",
                           "-m", "gqasr", "train", "--config", "config.yaml"))
    rank = process(101, 100)
    worker = process(102, 101, ("python3", "-c", "from multiprocessing.spawn import spawn_main"))
    grandchild = process(103, 102, ("ffmpeg", "-i", "audio.wav"))
    orphan = process(104, 1, worker.argv, config=str(ROOT / "cfg.yaml"), distributed=True)
    no_proof = process(105, 1, worker.argv)
    other_project = replace(process(106), cwd="/another/repo")
    benchmark = process(107, 1, ("python3", "-m", "gqasr.arabic_benchmark.cli", "run"))
    shell = process(108, 1, ("bash", "-c", "echo python -m gqasr train"))
    other_app = process(109, 1, ("echo", "-m", "gqasr", "train"))
    zombie = replace(process(110), state="Z")
    processes = {p.pid: p for p in (parent, rank, worker, grandchild, orphan, no_proof,
                                   other_project, benchmark, shell, other_app, zombie)}
    assert set(cleanup.select_processes(processes, ROOT, 9999)) == {100, 101, 102, 103, 104}


def test_own_process_ancestors_and_unrelated_shell_children_are_protected():
    parent = process(100, 1)
    own = process(101, 100, ("python3", "-m", "gqasr.cleanup", "--apply"))
    other = process(102, 100, ("bash",))
    assert cleanup.select_processes({p.pid: p for p in (parent, own, other)}, ROOT, 101) == {}


@pytest.mark.parametrize("script,args,expected", [
    ("scripts/run_gqasr_12node.sh", ("9",), True),
    ("scripts/run_gqasr_12node.sh", ("9", "--preflight"), False),
    ("scripts/resume_gqasr_encoder_12node.sh", ("9",), True),
    ("scripts/resume_gqasr_encoder_12node.sh", ("check",), False),
    ("gemma_speech/cluster/gqasr_stage.sh", ("encoder",), True),
    ("gemma_speech/cluster/gqasr_stage.sh", ("scan",), False),
    ("gemma_speech/cluster/gqtts_stage.sh", ("export",), False),
])
def test_launcher_matching(script, args, expected):
    p = process(argv=("bash", str(ROOT / script), *args))
    assert bool(cleanup.training_role(p, ROOT)) == expected


def test_proc_reader_checks_uid_and_handles_parenthesized_comm(tmp_path):
    directory = tmp_path / "123"
    directory.mkdir()
    (directory / "stat").write_text("123 (python (worker)) S 12 " + "0 " * 17 + "4567 0")
    (directory / "cmdline").write_bytes(b"python3\0-m\0gqasr\0train\0")
    (directory / "cwd").symlink_to(ROOT, target_is_directory=True)
    (directory / "environ").write_bytes(
        b"GQASR_CONFIG=/repo/config.yaml\0RANK=1\0LOCAL_RANK=1\0WORLD_SIZE=96\0"
        b"SECRET_TOKEN=must-not-be-reported\0"
    )
    p = cleanup.read_process(123, os.getuid(), tmp_path)
    assert p.start == 4567 and p.ppid == 12 and p.distributed
    assert p.config == "/repo/config.yaml"
    assert "SECRET" not in repr(p)
    assert cleanup.read_process(123, os.getuid() + 1, tmp_path) is None


def test_reused_pid_is_never_signalled(monkeypatch):
    original = process()
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.setattr(cleanup, "read_process", lambda *args: replace(original, start=99999))
    kill = Mock()
    monkeypatch.setattr(os, "kill", kill)
    assert cleanup.send_signal(original, signal.SIGTERM, os.getuid()) == "exited_or_replaced"
    kill.assert_not_called()


def test_pidfd_is_used_and_closed(monkeypatch):
    original = process()
    monkeypatch.setattr(os, "pidfd_open", lambda pid: 321, raising=False)
    monkeypatch.setattr(cleanup, "read_process", lambda *args: original)
    send, close, kill = Mock(), Mock(), Mock()
    monkeypatch.setattr(signal, "pidfd_send_signal", send, raising=False)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(os, "kill", kill)
    assert cleanup.send_signal(original, signal.SIGTERM, os.getuid()) == "sent"
    send.assert_called_once_with(321, signal.SIGTERM)
    close.assert_called_once_with(321)
    kill.assert_not_called()


def test_signal_stops_only_a_test_owned_subprocess(monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        target = process(child.pid)
        # macOS has no /proc; only this child is eligible in this live signal test.
        monkeypatch.setattr(cleanup, "read_process",
                            lambda pid, uid: target if pid == child.pid else None)
        monkeypatch.delattr(os, "pidfd_open", raising=False)
        assert cleanup.send_signal(target, signal.SIGTERM, os.getuid()) == "sent"
        assert child.wait(timeout=5) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("apply,force", [(False, False), (True, False), (True, True)])
def test_preview_and_force_policy_preserve_files(tmp_path, monkeypatch, apply, force):
    (tmp_path / "gemma_speech/gqasr/src/gqasr").mkdir(parents=True)
    (tmp_path / "gemma_speech/gqasr/src/gqasr/train.py").write_text("")
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"unchanged checkpoint")
    target = replace(process(), cwd=str(tmp_path))
    monkeypatch.setattr(cleanup.sys, "platform", "linux")
    exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda p: True if str(p) == "/proc/self/stat" else exists(p))
    monkeypatch.setattr(cleanup, "snapshot", lambda uid: {target.pid: target})
    signal_calls = []
    monkeypatch.setattr(cleanup, "send_signal",
                        lambda p, sig, uid: signal_calls.append(sig) or "sent")
    monkeypatch.setattr(cleanup, "wait_for_exit", lambda *args: [target])
    monkeypatch.setattr(subprocess, "run",
                        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""))
    args = argparse.Namespace(project_root=tmp_path, apply=apply, force=force,
                              grace_seconds=0, report_dir=tmp_path / "reports",
                              log_dir=tmp_path / "no-logs")
    result = cleanup.local_cleanup(args)
    assert signal_calls == (
        [] if not apply else [signal.SIGTERM, signal.SIGKILL] if force else [signal.SIGTERM]
    )
    assert result == (1 if apply else 0)  # Deliberately simulate an unkillable process.
    assert checkpoint.read_bytes() == b"unchanged checkpoint"
    report = json.loads(next(args.report_dir.glob("*.json")).read_text())
    assert report["targets"][0]["pid"] == target.pid


def test_hosts_and_ssh_quoting(tmp_path):
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("# actual machines\nnode051\nuser@node056\n")
    assert cleanup.read_hosts(hosts) == ["node051", "user@node056"]
    args = argparse.Namespace(project_root=tmp_path / "project with spaces",
                              grace_seconds=30, log_dir=Path("/tmp/node logs"),
                              report_dir=None, apply=True, force=True)
    cmd = shlex.split(cleanup.remote_command(args))
    assert cmd[:2] == ["bash", str(args.project_root / "scripts/cleanup_gqasr_nodes.sh")]
    assert cmd[-2:] == ["--apply", "--force"]
    for invalid in ("node051\nnode051\n", "node051; touch /tmp/bad\n", "-oProxyCommand=bad\n", ""):
        hosts.write_text(invalid)
        with pytest.raises(ValueError):
            cleanup.read_hosts(hosts)


def test_remote_cleanup_is_parallel_and_reports_failed_hosts(tmp_path, monkeypatch, capsys):
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("node051\nnode056\nnode034\n")
    args = argparse.Namespace(project_root=ROOT, grace_seconds=30, log_dir=tmp_path,
                              report_dir=None, apply=False, force=False,
                              hosts_file=hosts, jobs=3)
    # All three calls must start before any can finish. No SSH is executed.
    barrier = threading.Barrier(3, timeout=5)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        barrier.wait()
        host = command[-2]
        if host == "node034":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 255 if host == "node056" else 0,
                                           "preview result\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert cleanup.remote_cleanup(args) == 1
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["failed_hosts"] == ["node034", "node056"]
    assert len(calls) == 3
    for command, kwargs in calls:
        assert command[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        assert "--apply" not in shlex.split(command[-1])
        assert "--hosts-file" not in shlex.split(command[-1])
        assert kwargs["timeout"] == 120


def test_force_requires_explicit_apply(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cleanup", "--force"])
    local, remote = Mock(), Mock()
    monkeypatch.setattr(cleanup, "local_cleanup", local)
    monkeypatch.setattr(cleanup, "remote_cleanup", remote)
    with pytest.raises(SystemExit) as error:
        cleanup.main()
    assert error.value.code == 2
    local.assert_not_called()
    remote.assert_not_called()
