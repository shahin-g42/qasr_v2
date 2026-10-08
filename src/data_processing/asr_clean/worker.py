"""Workers: claim chunks, ASR -> LLM -> ordered, buffered, resumable writes.

Layout under ``<run_root>`` (one run = one RUN_ID):

    <lang>/<stem>/part-NNNNN.jsonl          cleaned records (5 columns, exactly)
    _rejects/<lang>/<stem>/part-NNNNN.jsonl  every record not written, with why
    _state/plan.json, dups/                  frozen by `plan`
    _state/claims/<chunk>                    who owns a chunk (O_EXCL create)
    _state/progress/<chunk>.json             committed position + counters

Exactly-once: records are committed in input order, ``flush_every`` at a time
(write, fsync, then atomically replace the progress file). After a crash the
part files are truncated back to the committed byte sizes and the input is
re-read from the committed offset, so nothing is lost or written twice.

Load balance: chunks are claimed dynamically, so fast nodes (Flash-Next) take
more chunks than slow ones (27B). A claim whose heartbeat is older than
``stale_minutes`` (its worker died) is taken over by an idle worker.

Server outages never turn into rejects: a transient ASR/LLM failure blocks the
worker until /health answers again. Only per-record verdicts (bad record,
duration out of range, audio the server cannot decode, LLM output that fails
the guards, an explicit LLM drop) are rejects.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .endpoints import ASRClient, LLMClient, PermanentError, TransientError
from .plan import iter_chunk_lines, load_dup_mask, load_plan
from .prompts import PROMPT_VERSION, guard, messages, parse_response
from .text import cer, header_duration, strip_envelope

LOG = logging.getLogger("asr_clean")
OUTPUT_KEYS = ("audio_filepath", "duration", "text", "org_text", "asr_text")


@dataclass
class WorkerConfig:
    run_root: str
    asr_urls: list[str]
    llm_url: str = "http://localhost:8010"
    asr_model: str = "qasr"
    llm_model: str = "corrector"
    flush_every: int = 1024          # records per commit (and per window)
    windows_in_flight: int = 3       # pipelined windows per process
    asr_concurrency: int = 16        # in-flight ASR requests per process
    llm_concurrency: int = 24        # in-flight LLM requests per process
    format_batch: int = 16           # items per request, transcripts agree
    adjudicate_batch: int = 8        # items per request, transcripts disagree
    agree_cer: float = 0.02          # normalized CER(org, asr) at or below = "agree"
    thinking: str = "auto"           # auto: think only when adjudicating | always | never
    llm_context: int = 8192          # the corrector's --max-model-len
    format_max_tokens: int = 4096
    adjudicate_max_tokens: int = 6144
    max_divergence: float = 0.5      # guard: output must be this close to org or asr
    min_duration: float = 0.1        # the training filter (configs/v7.6)
    max_duration: float = 35.0
    max_item_attempts: int = 4
    stale_minutes: float = 30.0
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}")


# ------------------------------------------------------------------- files --

def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


class RunPaths:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.state = self.root / "_state"
        self.claims = self.state / "claims"
        self.progress = self.state / "progress"
        for d in (self.claims, self.progress):
            d.mkdir(parents=True, exist_ok=True)

    def part(self, c: dict) -> Path:
        return self.root / c["lang"] / c["stem"] / f"part-{c['part']:05d}.jsonl"

    def rejects(self, c: dict) -> Path:
        return self.root / "_rejects" / c["lang"] / c["stem"] / f"part-{c['part']:05d}.jsonl"

    def progress_of(self, c: dict) -> dict | None:
        p = self.progress / f"{c['id']}.json"
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None


# ------------------------------------------------------------------ claims --

class Claims:
    def __init__(self, paths: RunPaths, worker_id: str, stale_minutes: float) -> None:
        self.paths, self.me, self.stale_s = paths, worker_id, stale_minutes * 60

    def _file(self, c: dict) -> Path:
        return self.paths.claims / c["id"]

    def owner(self, c: dict) -> str | None:
        try:
            return json.loads(self._file(c).read_text())["owner"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def heartbeat(self, c: dict) -> None:
        with contextlib.suppress(OSError):
            os.utime(self._file(c))

    def _try_create(self, c: dict) -> bool:
        try:
            fd = os.open(self._file(c), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"owner": self.me, "since": time.time()}))
        return True

    def _try_steal(self, c: dict) -> bool:
        f = self._file(c)
        try:
            age = time.time() - f.stat().st_mtime
        except OSError:
            return False
        if age < self.stale_s:
            return False
        _atomic_json(f, {"owner": self.me, "since": time.time(), "stolen_from": self.owner(c)})
        time.sleep(2.0)  # two thieves: the last replace wins, the other backs off
        return self.owner(c) == self.me

    def next_chunk(self, chunks: list[dict]) -> dict | None:
        def done(c: dict) -> bool:
            p = self.paths.progress_of(c)
            return bool(p and p.get("status") == "done")

        mine = [c for c in chunks if self.owner(c) == self.me and not done(c)]
        if mine:
            return mine[0]
        for c in chunks:
            if not self._file(c).exists() and not done(c) and self._try_create(c):
                return c
        for c in chunks:
            if not done(c) and self.owner(c) != self.me and self._try_steal(c):
                LOG.warning("took over stale chunk %s", c["id"])
                return c
        return None


# ---------------------------------------------------------------- pipeline --

@dataclass
class Item:
    line: int
    end_offset: int
    audio_filepath: str = ""
    duration: float | None = None
    org_text: str = ""
    asr_text: str = ""
    asr_label: str | None = None
    text: str = ""
    choice: str = ""
    reject: str | None = None
    detail: str = ""
    lane: str = ""
    llm_raw: str = ""
    agree: bool = False


class Window:
    def __init__(self, items: list[Item], skipped: int) -> None:
        self.items = items
        self.skipped_dups = skipped
        self.asr_futs: list[Future] = []
        self.llm_futs: list[Future] = []
        self.llm_submitted = False


class ChunkProcessor:
    def __init__(self, cfg: WorkerConfig, asr: ASRClient, llm: LLMClient,
                 asr_pool: ThreadPoolExecutor, llm_pool: ThreadPoolExecutor) -> None:
        self.cfg, self.asr, self.llm = cfg, asr, llm
        self.asr_pool, self.llm_pool = asr_pool, llm_pool

    # -- stage A: record -> ASR ------------------------------------------------
    def _parse(self, raw: bytes, item: Item) -> None:
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            item.reject, item.detail = "bad_record", "invalid JSON"
            return
        path = rec.get("audio_filepath") or rec.get("wav_path")
        text = rec.get("text") if rec.get("text") is not None else rec.get("transcript")
        if not isinstance(path, str) or not path.strip():
            item.reject, item.detail = "bad_record", "no audio_filepath"
            return
        item.audio_filepath = path
        item.org_text = strip_envelope(text) if isinstance(text, str) else ""
        d = rec.get("duration")
        item.duration = float(d) if isinstance(d, (int, float)) and d > 0 else None
        if not item.org_text:
            item.reject, item.detail = "bad_record", "empty original transcript"

    def _asr_one(self, item: Item, lang: str) -> None:
        if item.reject:
            return
        if item.duration is None:
            from .text import local_path

            item.duration = header_duration(local_path(item.audio_filepath))
            if item.duration is None:
                item.reject, item.detail = "no_duration", "header unreadable"
                return
        if not self.cfg.min_duration <= item.duration <= self.cfg.max_duration:
            item.reject, item.detail = "duration_out_of_range", f"{item.duration:.2f}s"
            return
        while True:
            try:
                res = self.asr.transcribe(item.audio_filepath, lang)
                break
            except PermanentError as exc:
                item.reject, item.detail = "asr_error", str(exc)[:500]
                return
            except TransientError as exc:
                _wait_healthy(self.asr.clients, f"ASR ({exc})")
        item.asr_text, item.asr_label = res["text"], res["language"]
        # Routing signal, computed here so the driver thread never blocks on it.
        item.agree = cer(item.org_text, item.asr_text, lang) <= self.cfg.agree_cer
        if res["finish_reason"] == "length":
            item.detail = "asr_truncated"  # informative only; the LLM adjudicates

    # -- stage B: ORIGINAL + ASR -> LLM ---------------------------------------
    def _think(self, lane: str) -> bool:
        return self.cfg.thinking == "always" or (self.cfg.thinking == "auto" and lane == "adjudicate")

    def _correct(self, items: list[Item], lang: str, lane: str, attempt: int = 1) -> None:
        payload = [{"org_text": it.org_text, "asr_text": it.asr_text} for it in items]
        msgs = messages(payload, lang)
        budget = self.cfg.adjudicate_max_tokens if lane == "adjudicate" else self.cfg.format_max_tokens
        est_prompt = sum(len(m["content"]) for m in msgs) // 2 + 64  # conservative chars/token
        max_tokens = max(256, min(budget, self.cfg.llm_context - est_prompt))
        while True:
            try:
                content, meta = self.llm.chat(msgs, thinking=self._think(lane), max_tokens=max_tokens)
                break
            except PermanentError as exc:  # e.g. context overflow: shrink the batch
                content, meta = "", {"error": str(exc)[:500]}
                break
            except TransientError as exc:
                _wait_healthy([self.llm.client], f"LLM ({exc})")
        parsed = parse_response(content, len(items))
        missing = []
        for i, it in enumerate(items):
            res = parsed.get(i)
            if res is None:
                missing.append(it)
                continue
            it.lane, it.choice = lane, res["choice"]
            if res["drop"]:
                it.reject, it.detail = "llm_drop", "both transcripts unusable"
                continue
            reason = guard(res["text"], it.org_text, it.asr_text, lang, self.cfg.max_divergence)
            if reason:
                it.reject, it.detail, it.llm_raw = f"guard_{reason}", "", res["text"]
            else:
                it.text = res["text"]
        if not missing:
            return
        if attempt >= self.cfg.max_item_attempts:
            for it in missing:
                it.reject = "llm_unresolved"
                it.detail = (meta.get("error") or f"no valid answer after {attempt} attempts")[:500]
                it.llm_raw = content[:2000]
            return
        if len(missing) == 1 or len(missing) < len(items):
            self._correct(missing, lang, lane, attempt + 1)  # retry just the unanswered ones
        else:  # nothing usable at all: halve the batch
            half = len(missing) // 2
            self._correct(missing[:half], lang, lane, attempt + 1)
            self._correct(missing[half:], lang, lane, attempt + 1)

    def _submit_llm(self, window: Window, lang: str) -> None:
        ready = [it for it in window.items if not it.reject]
        agree = [it for it in ready if it.agree]
        disagree = [it for it in ready if not it.agree]
        for lane, group, size in (("format", agree, self.cfg.format_batch),
                                  ("adjudicate", disagree, self.cfg.adjudicate_batch)):
            for k in range(0, len(group), size):
                window.llm_futs.append(self.llm_pool.submit(self._correct, group[k:k + size], lang, lane))
        window.llm_submitted = True

    # -- driver ----------------------------------------------------------------
    def run(self, chunk: dict, paths: RunPaths, claims: Claims, stop_after_windows: int | None = None) -> bool:
        prog = paths.progress_of(chunk) or {
            "chunk": chunk["id"], "status": "running", "offset": chunk["start"], "line": 0,
            "out_bytes": 0, "rej_bytes": 0, "counts": {}, "started": time.time(), "prompt": PROMPT_VERSION}
        if prog.get("status") == "done":
            return True
        part, rej = paths.part(chunk), paths.rejects(chunk)
        for f in (part, rej):
            f.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as files:
            out_fh = files.enter_context(open(part, "ab"))
            rej_fh = files.enter_context(open(rej, "ab"))
            return self._drive(chunk, prog, out_fh, rej_fh, paths, claims, stop_after_windows)

    def _drive(self, chunk: dict, prog: dict, out_fh, rej_fh, paths: RunPaths, claims: Claims,
               stop_after_windows: int | None) -> bool:
        cfg, lang = self.cfg, chunk["lang"]
        # Roll back anything written after the last commit.
        out_fh.truncate(prog["out_bytes"])
        rej_fh.truncate(prog["rej_bytes"])
        out_fh.seek(0, os.SEEK_END)
        rej_fh.seek(0, os.SEEK_END)
        counts = Counter(prog["counts"])
        mask = load_dup_mask(cfg.run_root, chunk)

        lines = _iter_from(chunk, prog["offset"], prog["line"])

        def next_window() -> Window | None:
            items, skipped = [], 0
            for i, end, raw in lines:
                if mask is not None and mask[i]:
                    skipped += 1
                    if skipped + len(items) >= cfg.flush_every:
                        break
                    continue
                it = Item(line=i, end_offset=end)
                self._parse(raw, it)
                items.append(it)
                if len(items) + skipped >= cfg.flush_every:
                    break
            if not items and not skipped:
                return None
            w = Window(items, skipped)
            w.last_line, w.last_offset = (i, end)  # type: ignore[possibly-undefined]
            w.asr_futs = [self.asr_pool.submit(self._asr_one, it, lang) for it in items]
            return w

        inflight: deque[Window] = deque()
        last_beat = time.time()
        exhausted = False
        windows_done = 0
        while True:
            while not exhausted and len(inflight) < cfg.windows_in_flight:
                w = next_window()
                if w is None:
                    exhausted = True
                else:
                    inflight.append(w)
            if not inflight:
                break
            for w in inflight:
                if not w.llm_submitted and all(f.done() for f in w.asr_futs):
                    for f in w.asr_futs:
                        f.result()  # surface bugs, never swallow them
                    self._submit_llm(w, lang)
            head = inflight[0]
            if head.llm_submitted and all(f.done() for f in head.llm_futs):
                for f in head.llm_futs:
                    f.result()
                self._commit(head, chunk, prog, counts, out_fh, rej_fh, paths, claims)
                inflight.popleft()
                windows_done += 1
                if stop_after_windows and windows_done >= stop_after_windows:
                    return False  # test hook: simulate a crash after N commits
                continue
            pending = [f for w in inflight for f in (w.llm_futs if w.llm_submitted else w.asr_futs) if not f.done()]
            if pending:
                wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            if time.time() - last_beat > 120:  # alive even while a slow window is in flight
                claims.heartbeat(chunk)
                last_beat = time.time()

        prog["status"], prog["finished"] = "done", time.time()
        _atomic_json(paths.progress / f"{chunk['id']}.json", prog)
        return True

    def _commit(self, w: Window, chunk: dict, prog: dict, counts: Counter,
                out_fh, rej_fh, paths: RunPaths, claims: Claims) -> None:
        out_lines, rej_lines = [], []
        for it in w.items:
            counts["lines"] += 1
            if it.reject:
                counts[f"reject:{it.reject}"] += 1
                rej_lines.append(json.dumps({
                    "audio_filepath": it.audio_filepath, "duration": it.duration, "org_text": it.org_text,
                    "asr_text": it.asr_text, "reason": it.reject, "detail": it.detail,
                    "lane": it.lane, "choice": it.choice, "llm_output": it.llm_raw}, ensure_ascii=False))
                continue
            counts["written"] += 1
            counts[f"lane:{it.lane}"] += 1
            counts[f"choice:{it.choice}"] += 1
            if it.detail == "asr_truncated":
                counts["asr_truncated"] += 1
            rec = {"audio_filepath": it.audio_filepath, "duration": round(it.duration, 3),
                   "text": it.text, "org_text": it.org_text, "asr_text": it.asr_text}
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        counts["duplicates"] += w.skipped_dups
        for fh, rows in ((out_fh, out_lines), (rej_fh, rej_lines)):
            if rows:
                fh.write(("\n".join(rows) + "\n").encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        prog.update(offset=w.last_offset, line=w.last_line + 1, out_bytes=out_fh.tell(),
                    rej_bytes=rej_fh.tell(), counts=dict(counts), updated=time.time())
        _atomic_json(paths.progress / f"{chunk['id']}.json", prog)
        claims.heartbeat(chunk)


def _iter_from(chunk: dict, offset: int, first_line: int):
    for i, end, raw in iter_chunk_lines(chunk["source"], offset, chunk["end"]):
        yield first_line + i, end, raw


def _wait_healthy(clients, what: str) -> None:
    delay = 10.0
    while True:
        LOG.warning("%s unavailable; waiting %.0fs for /health", what, delay)
        time.sleep(delay)
        if all(c.healthy() for c in clients):
            return
        delay = min(delay * 2, 300.0)


# ------------------------------------------------------------------ entry --

def run_worker(cfg: WorkerConfig, stop_after_windows: int | None = None) -> None:
    plan = load_plan(cfg.run_root)
    paths = RunPaths(cfg.run_root)
    claims = Claims(paths, cfg.worker_id, cfg.stale_minutes)
    asr = ASRClient(cfg.asr_urls, model=cfg.asr_model)
    llm = LLMClient(cfg.llm_url, model=cfg.llm_model)
    for clients, what in ((asr.clients, "ASR"), ([llm.client], "LLM")):
        if not all(c.healthy() for c in clients):
            _wait_healthy(clients, what)
    with ThreadPoolExecutor(cfg.asr_concurrency, thread_name_prefix="asr") as asr_pool, \
            ThreadPoolExecutor(cfg.llm_concurrency, thread_name_prefix="llm") as llm_pool:
        proc = ChunkProcessor(cfg, asr, llm, asr_pool, llm_pool)
        while (chunk := claims.next_chunk(plan["chunks"])) is not None:
            t0 = time.time()
            LOG.info("[%s] chunk %s (%s lines)", cfg.worker_id, chunk["id"], chunk["lines"])
            if not proc.run(chunk, paths, claims, stop_after_windows):
                return
            p = paths.progress_of(chunk) or {}
            c = p.get("counts", {})
            LOG.info("[%s] done %s: %d written, %d rejected, %d dups in %.0fs", cfg.worker_id, chunk["id"],
                     c.get("written", 0), sum(v for k, v in c.items() if k.startswith("reject:")),
                     c.get("duplicates", 0), time.time() - t0)
    LOG.info("[%s] no chunks left", cfg.worker_id)


# ----------------------------------------------------------- status/assemble --

def status(run_root: str) -> dict:
    plan = load_plan(run_root)
    paths = RunPaths(run_root)
    total = Counter()
    done = started = 0
    first, last = None, None
    for c in plan["chunks"]:
        p = paths.progress_of(c)
        if not p:
            continue
        started += 1
        done += p.get("status") == "done"
        total.update(p.get("counts", {}))
        first = min(first or p["started"], p["started"])
        last = max(last or 0, p.get("updated", p["started"]))
    to_clean = plan["lines"] - plan["duplicates"]
    processed = total["lines"]
    rate = processed / max((last or 0) - (first or 0), 1) if processed else 0.0
    return {
        "chunks": len(plan["chunks"]), "chunks_started": started, "chunks_done": done,
        "lines": plan["lines"], "duplicates_planned": plan["duplicates"], "to_clean": to_clean,
        "processed": processed, "written": total["written"],
        "rejects": {k[7:]: v for k, v in sorted(total.items()) if k.startswith("reject:")},
        "lanes": {k[5:]: v for k, v in total.items() if k.startswith("lane:")},
        "choices": {k[7:]: v for k, v in total.items() if k.startswith("choice:")},
        "asr_truncated": total["asr_truncated"],
        "records_per_s": round(rate, 1),
        "eta_hours": round((to_clean - processed) / rate / 3600, 1) if rate else None,
    }


def assemble(run_root: str, allow_partial: bool = False) -> list[str]:
    """Concatenate each source's finished parts into ``<lang>/<stem>.jsonl``."""
    plan = load_plan(run_root)
    paths = RunPaths(run_root)
    by_src: dict[tuple[str, str], list[dict]] = {}
    for c in plan["chunks"]:
        by_src.setdefault((c["lang"], c["stem"]), []).append(c)
    written = []
    for (lang, stem), chunks in sorted(by_src.items()):
        states = [(paths.progress_of(c) or {}).get("status") for c in chunks]
        if not allow_partial and any(s != "done" for s in states):
            continue
        target = paths.root / lang / f"{stem}.jsonl"
        tmp = target.with_suffix(".jsonl.tmp")
        n = 0
        with open(tmp, "wb") as out:
            for c in sorted(chunks, key=lambda c: c["part"]):
                p = paths.progress_of(c)
                if not p:
                    continue
                with open(paths.part(c), "rb") as fh:  # committed bytes only
                    data = fh.read(p["out_bytes"])
                out.write(data)
                n += data.count(b"\n")
        os.replace(tmp, target)
        written.append(f"{target} ({n:,} records{'' if all(s == 'done' for s in states) else ', PARTIAL'})")
    return written


def run_child(cfg_dict: dict, log_file: str) -> None:
    """Entry point of one spawned worker process (must be importable for `spawn`)."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_worker(WorkerConfig(**cfg_dict))
    except Exception:
        LOG.exception("worker %s crashed", cfg_dict.get("worker_id"))
        raise


def to_dict(cfg: WorkerConfig) -> dict:
    return asdict(cfg)
