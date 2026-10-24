"""Small, dependency-free console relay and complete worker diagnostics.

Executed by path by the shell launcher so the relay never imports torch. Worker
FD redirection also captures native stderr and DataLoader subprocess output.
"""

import argparse
import contextlib
import datetime
import json
import logging
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

_MARKER = b"@@AUD_CONSOLE@@"
_EVENT_FD = None
_OWNER_PID = None
_EVENT_LIMIT = 4096


def active():
    return _EVENT_FD is not None and _OWNER_PID == os.getpid()


def emit(message, *, all_ranks=False):
    """No tensors or collectives here. Each event fits in one atomic pipe write."""
    if not active() or (not all_ranks and int(os.environ.get("RANK", "0")) != 0):
        return
    message = " ".join(str(message).split())[:700]
    print(f"[AUD console] {message}", file=sys.stderr, flush=True)
    event = json.dumps({"message": message}, ensure_ascii=False).encode()
    # PIPE_BUF differs by platform. Keep Unicode/error summaries atomic even
    # when several ranks fail at once; the full message is already in the file.
    while len(event) + len(_MARKER) + 2 > _EVENT_LIMIT:
        message = message[: len(message) // 2] + "..."
        event = json.dumps({"message": message}, ensure_ascii=False).encode()
    os.write(_EVENT_FD, b"\n" + _MARKER + event + b"\n")


class _Notices(logging.Handler):
    def __init__(self, rank, path):
        super().__init__(logging.WARNING)
        self.rank, self.path = rank, path
        self.count = 0
        self.shown = set()

    def emit(self, record):
        if not active():
            return
        self.count += 1
        message = record.getMessage()
        # Repeated bad-sample notices must not overwhelm progress. All records
        # are still captured verbatim in the rank file, including other ranks.
        key = (record.name, message)
        if self.rank == 0 and key not in self.shown and len(self.shown) < 3:
            self.shown.add(key)
            emit(f"{record.levelname} | {record.name}: {message[:220]} | Details: {self.path}")


@contextlib.contextmanager
def worker_logging():
    """Only activate inside our launcher; library calls retain caller logging."""
    global _EVENT_FD, _OWNER_PID, _EVENT_LIMIT
    directory = os.environ.get("AUD_CONSOLE_RUN_DIR")
    if not directory or active():
        yield
        return
    rank = int(os.environ.get("RANK", "0"))
    path = Path(directory) / f"rank-{rank:03d}.log"
    # A new launch has its own directory, never inside the Trainer output. This
    # preserves existing-run/atomic-export validation and never truncates logs.
    with path.open("ab", buffering=0) as log:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        _EVENT_FD, _OWNER_PID = saved[0], os.getpid()
        _EVENT_LIMIT = min(4096, os.fpathconf(saved[0], "PC_PIPE_BUF"))
        notice = _Notices(rank, path)
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
            logging.getLogger().addHandler(notice)
            emit("Loading model and preparing data...")
            try:
                yield
            except BaseException as exc:
                traceback.print_exc()
                if not isinstance(exc, SystemExit) or exc.code:
                    emit(
                        f"FAILED | rank {rank} | {type(exc).__name__}: {exc} | Details: {path}",
                        all_ranks=True,
                    )
                raise
            finally:
                if notice.count:
                    emit(f"Diagnostics | {notice.count} warning/error records on rank 0: {path}")
        finally:
            logging.getLogger().removeHandler(notice)
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            _EVENT_FD = _OWNER_PID = None
            for descriptor in saved:
                os.close(descriptor)


def _argument(command, name):
    for i, value in enumerate(command):
        if value == name and i + 1 < len(command):
            return command[i + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def launch(command, log_root=None, name=None):
    output = _argument(command, "--output")
    root = Path(
        log_root
        or os.environ.get("AUD_LOG_DIR")
        or (str(Path(output).absolute().parent / "logs") if output else "output/aud-logs")
    ).absolute()
    if output and root.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError("AUD_LOG_DIR must be outside --output to preserve checkpoint safety")
    root.mkdir(parents=True, exist_ok=True)
    label = name or (Path(output).name if output else "distributed-check")
    if "--smoke" in command:
        label += "-smoke"
    label = re.sub(r"[^a-zA-Z0-9_.-]", "-", label)
    node = os.environ.get("AUD_NODE_RANK", "0")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = Path(tempfile.mkdtemp(prefix=f"{label}-{timestamp}-node{node}-", dir=root))
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    # Separate node pointers avoid races on shared storage. Existing logs remain.
    pointer = root / f"latest-node{node}.txt"
    temporary = directory / "latest.tmp"
    temporary.write_text(str(directory) + "\n")
    temporary.replace(pointer)
    console = (directory / "console.log").open("w", buffering=1)
    console_open = True

    def show(message):
        nonlocal console_open
        line = f"{time.strftime('%H:%M:%S')}  {message}"
        console.write(line + "\n")
        if console_open:
            try:
                print(line, flush=True)
            except BrokenPipeError:
                # Losing a tail/pipe reader must not corrupt training or logs.
                console_open = False
                with open(os.devnull, "w") as sink:
                    os.dup2(sink.fileno(), sys.stdout.fileno())

    show(f"AUD | {label} | node {node}")
    show(f"Debug logs: {directory}")
    env = dict(os.environ, AUD_CONSOLE_RUN_DIR=str(directory), PYTHONUNBUFFERED="1")
    process = None
    handlers = {}
    received = None
    signalled_at = None

    def forward(sig, frame):
        nonlocal received, signalled_at
        received, signalled_at = sig, time.monotonic()
        show(f"Stopping | signal {signal.Signals(sig).name}")
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, sig)

    try:
        with (directory / "launcher.log").open("ab", buffering=0) as raw:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                bufsize=0,
            )
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, forward)
            pending = b""
            last_event = time.monotonic()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    for key, _ in selector.select(timeout=1):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        raw.write(data)
                        pending += data.replace(b"\r", b"\n")
                        lines = pending.split(b"\n")
                        pending = lines.pop()
                        # Native tools may emit enormous lines without newlines.
                        pending = pending[-8192:]
                        for line in lines:
                            if not line.startswith(_MARKER):
                                continue
                            try:
                                event = json.loads(line[len(_MARKER) :])
                                show(event["message"])
                                last_event = time.monotonic()
                            except (ValueError, KeyError, TypeError):
                                pass  # Raw bytes remain in launcher.log.
                    if time.monotonic() - last_event >= 60:
                        show("Working | detailed progress is in the debug logs")
                        last_event = time.monotonic()
                    if signalled_at and time.monotonic() - signalled_at > 30:
                        # Only the process group this launcher created, never
                        # unrelated jobs. Torchrun owns its worker teardown.
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        signalled_at = None
            code = process.wait()
        if received:
            code = 128 + received
        elif code < 0:
            code = 128 - code
        show(
            f"Complete | {label} | logs: {directory}"
            if code == 0
            else f"FAILED | exit {code} | inspect {directory}"
        )
        return code
    except BaseException as exc:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        show(f"FAILED | launcher: {exc} | logs: {directory}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        console.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root")
    parser.add_argument("--name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    raise SystemExit(launch(command, args.log_root, args.name))


if __name__ == "__main__":
    main()
