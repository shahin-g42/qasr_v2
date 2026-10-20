"""Stop this user's GQASR training processes on Linux; preview by default."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import datetime
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    start: int
    state: str
    argv: tuple
    cwd: str
    config: str = ""
    distributed: bool = False


def read_process(pid, uid, proc_root=Path("/proc")):
    """Read only our own processes; do not retain or print environment values."""
    directory = proc_root / str(pid)
    try:
        if directory.stat().st_uid != uid:
            return None
        # comm can contain spaces and parentheses; fields after its final ')'
        # start at field 3 (state). starttime is field 22.
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        argv = tuple(os.fsdecode(x) for x in (directory / "cmdline").read_bytes().split(b"\0") if x)
        try:
            cwd = os.readlink(directory / "cwd")
        except OSError:
            cwd = ""
        try:
            env = dict(item.split(b"=", 1) for item in
                       (directory / "environ").read_bytes().split(b"\0") if b"=" in item)
        except OSError:
            env = {}
        distributed = all(key in env for key in (b"LOCAL_RANK", b"RANK", b"WORLD_SIZE"))
        return Process(pid, int(fields[1]), int(fields[19]), fields[0], argv, cwd,
                       os.fsdecode(env.get(b"GQASR_CONFIG", b"")), distributed)
    except (OSError, ValueError, IndexError):
        return None  # Exited or became inaccessible during enumeration.


def snapshot(uid):
    result = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            process = read_process(int(directory.name), uid)
            if process is not None:
                result[process.pid] = process
    return result


def within(value, root, cwd=""):
    if not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        if not cwd:
            return False
        path = Path(cwd) / path
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def option(argv, key):
    for i, arg in enumerate(argv):
        if arg == key and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(key + "="):
            return arg.split("=", 1)[1]
    return ""


def training_role(process, root):
    argv = process.argv
    if not argv or process.state == "Z":
        return None
    scoped = (within(process.cwd, root) or
              within(option(argv, "--config"), root, process.cwd))
    module = any(argv[i:i + 3] == ("-m", "gqasr", "train")
                 for i in range(len(argv) - 2))
    console = any(Path(arg).name == "gqasr" and argv[i + 1:i + 2] == ("train",)
                  for i, arg in enumerate(argv))
    executable = Path(argv[0]).name
    runner = bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable)) or executable in {
        "gqasr", "torchrun",
    }
    if scoped and runner and (module or console):
        return "GQASR train/torchrun"
    # Match actual script arguments, never a substring in `bash -c` or an editor.
    if Path(argv[0]).name in {"bash", "sh", "zsh"} and len(argv) > 1:
        script = Path(argv[1])
        if not script.is_absolute():
            if not process.cwd:
                return None
            script = Path(process.cwd) / script
        script = script.resolve()
        scripts = {
            root / "scripts/run_gqasr_12node.sh",
            root / "scripts/resume_gqasr_encoder_12node.sh",
        }
        if script in scripts and "--preflight" not in argv and "check" not in argv[2:]:
            return "GQASR launcher"
        if script == root / "gemma_speech/cluster/gqasr_stage.sh" and argv[2:3] in {
            ("bridge",), ("full",), ("encoder",),
        }:
            return "GQASR stage"
    # Orphaned DataLoader/loky/resource-tracker processes lose their parent.
    # Require BOTH a project-specific inherited config and torchrun rank env.
    python = Path(argv[0]).name.startswith("python")
    worker = (
        ("-c" in argv and any(marker in option(argv, "-c") for marker in (
            "from multiprocessing.spawn import spawn_main",
            "from multiprocessing.forkserver import main",
            "from multiprocessing.resource_tracker import main",
            "from joblib.externals.loky.backend.resource_tracker import main",
        )))
        or any(argv[i:i + 2] == ("-m", "joblib.externals.loky.backend.popen_loky_posix")
               for i in range(len(argv) - 1))
    )
    sidecar = Path(argv[0]).name in {"wandb-core", "gpu_stats"}
    if (process.distributed and within(process.config, root, process.cwd)
            and ((python and worker) or sidecar)):
        return "inherited training worker"
    return None


def select_processes(processes, root, own_pid):
    protected = {0, 1}
    current = own_pid
    while current not in protected:
        protected.add(current)
        parent = processes.get(current)
        if parent is None:
            break
        current = parent.ppid
    selected = {pid: training_role(p, root) for pid, p in processes.items()
                if pid not in protected}
    selected = {pid: role for pid, role in selected.items() if role}
    while True:
        children = {pid: f"child of {p.ppid}" for pid, p in processes.items()
                    if pid not in selected and pid not in protected
                    and p.state != "Z" and p.ppid in selected}
        if not children:
            return selected
        selected.update(children)


def alive(original, uid):
    current = read_process(original.pid, uid)
    return current is not None and current.start == original.start and current.state != "Z"


def send_signal(original, sig, uid):
    """Bind a pidfd where supported, then recheck identity before signalling."""
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            descriptor = os.pidfd_open(original.pid)
        if not alive(original, uid):
            return "exited_or_replaced"
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(original.pid, sig)
        return "sent"
    except ProcessLookupError:
        return "exited"
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        if descriptor is not None:
            os.close(descriptor)


def wait_for_exit(targets, uid, seconds):
    deadline = time.monotonic() + seconds
    while True:
        remaining = [p for p in targets if alive(p, uid)]
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))


def local_cleanup(args):
    if sys.platform != "linux" or not Path("/proc/self/stat").exists():
        raise ValueError("Run cleanup on the Linux training node; it uses /proc.")
    if os.getuid() == 0:
        raise ValueError("Run as the training user, not root.")
    root = args.project_root.resolve()
    if not (root / "gemma_speech/gqasr/src/gqasr/train.py").is_file():
        raise ValueError(f"Not the expected GQASR project: {root}")
    processes = snapshot(os.getuid())
    roles = select_processes(processes, root, os.getpid())
    targets = [processes[pid] for pid in sorted(roles)]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname()
    report_dir = args.report_dir or args.log_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"cleanup-{host}-{stamp}-{os.getpid()}.json"
    report = {
        "hostname": host, "uid": os.getuid(), "project_root": str(root),
        "apply": args.apply, "force": args.force, "grace_seconds": args.grace_seconds,
        "targets": [{"pid": p.pid, "ppid": p.ppid, "start_ticks": p.start,
                     "state": p.state, "role": roles[p.pid]} for p in targets],
    }
    try:
        if __package__:
            from .nccl_report import collect
        else:
            from nccl_report import collect
        report["existing_diagnostics"] = collect(args.log_dir)
    except (OSError, ValueError) as exc:
        report["diagnostic_read_error"] = str(exc)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{host}: {'STOP' if args.apply else 'PREVIEW'} {len(targets)} GQASR processes; "
          f"report={report_path}", flush=True)
    for p in targets:
        print(f"  PID {p.pid:<8} parent {p.ppid:<8} state {p.state}  {roles[p.pid]}", flush=True)
    if not args.apply:
        print("Preview only. Use --apply to send SIGTERM; add --force for surviving processes.")
        return 0
    # All PIDs are captured before stopping parents, so reparented workers remain
    # known. Parent/launcher processes get TERM too, preventing elastic restarts.
    report["signals"] = [
        {"pid": p.pid, "signal": "SIGTERM", "result": send_signal(p, signal.SIGTERM, os.getuid())}
        for p in targets
    ]
    print(f"SIGTERM sent; waiting up to {args.grace_seconds:g} seconds.", flush=True)
    remaining = wait_for_exit(targets, os.getuid(), args.grace_seconds)
    if remaining and args.force:
        for p in remaining:
            report["signals"].append({
                "pid": p.pid, "signal": "SIGKILL",
                "result": send_signal(p, signal.SIGKILL, os.getuid()),
            })
        remaining = wait_for_exit(remaining, os.getuid(), 5)
    # Catch workers or supervisors which appeared after the initial snapshot.
    # Report these rather than silently killing new PIDs outside that snapshot.
    after = snapshot(os.getuid())
    extra = select_processes(after, root, os.getpid())
    remaining_ids = {p.pid for p in remaining} | set(extra)
    report["remaining"] = [
        {"pid": pid, "state": after[pid].state if pid in after else "unknown"}
        for pid in sorted(remaining_ids)
    ]
    if Path("/dev/shm").exists():
        usage = shutil.disk_usage("/dev/shm")
        report["shared_memory"] = {"total_bytes": usage.total, "free_bytes": usage.free}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10,
        )
        report["gpu_processes"] = gpu.stdout.strip()
        if gpu.returncode:
            report["gpu_query_error"] = gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["gpu_query_error"] = str(exc)
    report["matched_processes_stopped"] = not remaining_ids
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "hostname": host, "matched_processes_stopped": not remaining_ids,
        "remaining": report["remaining"], "report": str(report_path),
        "gpu_processes": report.get("gpu_processes"), "gpu_query_error": report.get("gpu_query_error"),
    }), flush=True)
    return 1 if remaining_ids else 0


def read_hosts(path):
    hosts = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    hosts = [host for host in hosts if host]
    if not hosts or len(hosts) > 12:
        raise ValueError("Host file must contain 1–12 actual training hostnames, one per line.")
    if len(set(hosts)) != len(hosts):
        raise ValueError("Duplicate hostname in host file.")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host) for host in hosts):
        raise ValueError("Use plain SSH hostnames (optionally user@hostname), not shell commands.")
    return hosts


def remote_command(args):
    command = ["bash", str(args.project_root / "scripts/cleanup_gqasr_nodes.sh"),
               "--grace-seconds", str(args.grace_seconds), "--log-dir", str(args.log_dir)]
    if args.apply:
        command.append("--apply")
    if args.force:
        command.append("--force")
    if args.report_dir:
        command.extend(["--report-dir", str(args.report_dir)])
    return shlex.join(command)


def remote_cleanup(args):
    hosts = read_hosts(args.hosts_file)
    command = remote_command(args)
    failed = []
    def run(host):
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
            capture_output=True, text=True, timeout=args.grace_seconds + 90,
        )
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run, host): host for host in hosts}
        for future in as_completed(pending):
            host = pending[future]
            try:
                result = future.result()
                print(f"\n--- {host} ---\n{result.stdout}", flush=True)
                if result.stderr:
                    print(result.stderr, file=sys.stderr, flush=True)
                if result.returncode:
                    failed.append(host)
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"{host}: {exc}", file=sys.stderr, flush=True)
                failed.append(host)
    print(json.dumps({"hosts": hosts, "failed_hosts": sorted(failed),
                      "apply": args.apply}), flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--apply", action="store_true", help="Send SIGTERM to selected processes.")
    parser.add_argument("--force", action="store_true", help="SIGKILL survivors after the grace period.")
    parser.add_argument("--grace-seconds", type=float, default=30)
    parser.add_argument("--hosts-file", type=Path, help="SSH to these hosts; otherwise run locally.")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--log-dir", type=Path, default=Path(os.environ.get(
        "GQASR_LOG_DIR", f"/tmp/gqasr-{getpass.getuser()}-logs")))
    parser.add_argument("--report-dir", type=Path, help="Default: node-local log directory.")
    args = parser.parse_args()
    if args.force and not args.apply:
        parser.error("--force requires --apply")
    if not 0 <= args.grace_seconds <= 300 or not 1 <= args.jobs <= 12:
        parser.error("Use grace-seconds 0–300 and jobs 1–12.")
    args.project_root = args.project_root.resolve()
    try:
        return remote_cleanup(args) if args.hosts_file else local_cleanup(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cleanup failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
