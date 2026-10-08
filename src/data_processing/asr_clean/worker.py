"""Workers: claim chunks, ASR -> LLM -> ordered, buffered, resumable writes.

Layout under ``<run_root>`` (one run = one RUN_ID):

    <lang>/<tree>/<stem>/part-NNNNN.jsonl          cleaned records (5 columns, exactly)
    _rejects/<lang>/<tree>/<stem>/part-NNNNN.jsonl  every record not written, with why
    (<tree> = v7.6 | q3asr_sft_manifests | ..., the source manifest tree)
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
import threading
import time
import unicodedata
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import diacritics
from .endpoints import ASRClient, LLMClient, PermanentError, TransientError
from .plan import iter_chunk_lines, load_dup_mask, load_plan
from .prompts import PROMPT_VERSION, guard, messages, parse_response
from .text import added_letters, cer, header_duration, strip_envelope

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
    windows_in_flight: int = 4       # pipelined windows per process
    asr_concurrency: int = 32        # in-flight ASR requests per process
    llm_concurrency: int = 32        # in-flight LLM requests per process
    format_batch: int = 24           # items per request, transcripts agree (shares the long system prompt)
    adjudicate_batch: int = 8        # items per request, transcripts disagree
    agree_cer: float = 0.02          # normalized CER(org, asr) at or below = "agree"
    # Off by default: on the 8k-context corrector nodes 91% of thinking requests
    # hit max_tokens before answering (test1, 2026-10-09). auto = adjudication only.
    thinking: str = "never"          # never | auto | always
    llm_context: int = 8192          # the corrector's --max-model-len
    format_max_tokens: int = 4096
    adjudicate_max_tokens: int = 6144
    max_divergence: float = 0.5      # guard: output must be this close to org or asr
    # Format lane (ORIGINAL == ASR): letters the LLM may insert/substitute before
    # its rewrite is overruled in favour of the agreed words. -1 turns the guard
    # off. Every overrule is logged to _review/ for inspection either way.
    agree_max_added: int = 1
    # A real transcript of the longest (35 s) clips is ~100-300 tokens; hitting
    # this cap means the ASR looped. 1024 keeps a looping-but-long transcript's
    # real tail; test3 had 57 caps in 51.5k requests at 512.
    asr_max_tokens: int = 1024
    ar_diacritics: str = "critical"  # Arabic diacritization pass: critical | full | none
    diac_batch: int = 16             # items per diacritization request
    diac_max_tokens: int = 6144
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

    def review(self, c: dict) -> Path:
        return self.root / "_review" / c["lang"] / c["stem"] / f"part-{c['part']:05d}.jsonl"

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


# --------------------------------------------------------------- telemetry --

class Telemetry:
    """Process-wide request statistics, snapshotted to ``_state/workers/<id>.json``.

    Answers "where does the time go": per LLM lane the request count, items,
    latency, prompt/completion tokens, how often generation hit max_tokens
    (a thinking run that never reached its answer) and how many items the
    model left out; per ASR call the latency and audio seconds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.c: Counter = Counter()

    def add(self, **kv: float) -> None:
        with self._lock:
            self.c.update(kv)

    def snapshot(self) -> dict:
        with self._lock:
            return {"started": self.started, "updated": time.time(), "counters": dict(self.c)}


def _snapshot_loop(tel: Telemetry, path: Path, stop: threading.Event, every: float = 30.0) -> None:
    while not stop.wait(every):
        _atomic_json(path, tel.snapshot())
    _atomic_json(path, tel.snapshot())


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
    diac: str = ""                   # "", "ok" or "failed:<why>"


class Window:
    def __init__(self, items: list[Item], skipped: int) -> None:
        self.items = items
        self.skipped_dups = skipped
        self.asr_futs: list[Future] = []
        self.llm_futs: list[Future] = []
        self.llm_submitted = False
        self.diac_futs: list[Future] = []
        self.diac_submitted = False


class ChunkProcessor:
    def __init__(self, cfg: WorkerConfig, asr: ASRClient, llm: LLMClient,
                 asr_pool: ThreadPoolExecutor, llm_pool: ThreadPoolExecutor,
                 tel: Telemetry | None = None) -> None:
        self.cfg, self.asr, self.llm = cfg, asr, llm
        self.asr_pool, self.llm_pool = asr_pool, llm_pool
        self.tel = tel or Telemetry()

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
        t0 = time.perf_counter()
        while True:
            try:
                res = self.asr.transcribe(item.audio_filepath, lang)
                self.tel.add(asr_requests=1, asr_seconds=time.perf_counter() - t0, asr_audio_s=item.duration)
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

    def _correct(self, items: list[Item], lang: str, lane: str, attempt: int = 1,
                 think: bool | None = None) -> None:
        think = self._think(lane) if think is None else think
        payload = [{"org_text": it.org_text, "asr_text": it.asr_text} for it in items]
        msgs = messages(payload, lang, lane)
        budget = self.cfg.adjudicate_max_tokens if lane == "adjudicate" else self.cfg.format_max_tokens
        est_prompt = sum(len(m["content"]) for m in msgs) // 2 + 64  # conservative chars/token
        max_tokens = max(256, min(budget, self.cfg.llm_context - est_prompt))
        t0 = time.perf_counter()
        while True:
            try:
                content, meta = self.llm.chat(msgs, thinking=think, max_tokens=max_tokens)
                break
            except PermanentError as exc:  # e.g. context overflow: shrink the batch
                content, meta = "", {"error": str(exc)[:500]}
                break
            except TransientError as exc:
                _wait_healthy([self.llm.client], f"LLM ({exc})")
        parsed = parse_response(content, len(items))
        key = f"llm_{lane}{'_think' if think else ''}"
        truncated = meta.get("finish_reason") == "length"
        self.tel.add(**{f"{key}_requests": 1, f"{key}_items": len(items), f"{key}_answered": len(parsed),
                        f"{key}_seconds": time.perf_counter() - t0,
                        f"{key}_prompt_tokens": meta.get("prompt_tokens", 0),
                        f"{key}_completion_tokens": meta.get("completion_tokens", 0),
                        f"{key}_truncated": int(truncated), f"{key}_errors": int("error" in meta)})
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
            if not reason and it.detail == "asr_truncated" and res["choice"] == "asr":
                # The ASR stopped at max_tokens (it looped) and the final text follows
                # it: the transcript likely ends before the audio does.
                reason = "follows_truncated_asr"
            if reason:
                it.reject, it.detail, it.llm_raw = f"guard_{reason}", "", res["text"]
            elif (self.cfg.agree_max_added >= 0 and lane == "format"
                  and added_letters(it.org_text, res["text"], lang) > self.cfg.agree_max_added):
                # Both transcripts agree on these words; an LLM that still rewrites
                # them (test3: ml 55%, hi 13%) is overruled. Its version is kept in
                # _review/ so the decision can be audited.
                it.text, it.choice, it.detail, it.llm_raw = it.org_text, "original", "agree_guard", res["text"]
                self.tel.add(agree_guard_overrules=1)
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
        if think and truncated:
            # The reasoning ate the whole budget before the answer began:
            # retrying with thinking would burn the same tokens again.
            self.tel.add(llm_think_fallbacks=len(missing))
            self._correct(missing, lang, lane, attempt + 1, think=False)
        elif len(missing) == 1 or len(missing) < len(items):
            self._correct(missing, lang, lane, attempt + 1, think)  # retry just the unanswered ones
        else:  # nothing usable at all: halve the batch
            half = len(missing) // 2
            self._correct(missing[:half], lang, lane, attempt + 1, think)
            self._correct(missing[half:], lang, lane, attempt + 1, think)

    def _submit_llm(self, window: Window, lang: str) -> None:
        ready = [it for it in window.items if not it.reject]
        agree = [it for it in ready if it.agree]
        disagree = [it for it in ready if not it.agree]
        for lane, group, size in (("format", agree, self.cfg.format_batch),
                                  ("adjudicate", disagree, self.cfg.adjudicate_batch)):
            for k in range(0, len(group), size):
                window.llm_futs.append(self.llm_pool.submit(self._correct, group[k:k + size], lang, lane))
        window.llm_submitted = True

    # -- stage C: Arabic diacritics (marks only) --------------------------------
    def _diacritize(self, items: list[Item], attempt: int = 1) -> None:
        policy = self.cfg.ar_diacritics
        msgs = diacritics.messages([it.text for it in items], policy)
        est_prompt = sum(len(m["content"]) for m in msgs) // 2 + 64
        max_tokens = max(256, min(self.cfg.diac_max_tokens, self.cfg.llm_context - est_prompt))
        t0 = time.perf_counter()
        while True:
            try:
                content, meta = self.llm.chat(msgs, thinking=False, max_tokens=max_tokens)
                break
            except PermanentError as exc:
                content, meta = "", {"error": str(exc)[:500]}
                break
            except TransientError as exc:
                _wait_healthy([self.llm.client], f"LLM ({exc})")
        parsed = parse_response(content, len(items))
        self.tel.add(llm_diacritize_requests=1, llm_diacritize_items=len(items),
                     llm_diacritize_answered=len(parsed), llm_diacritize_seconds=time.perf_counter() - t0,
                     llm_diacritize_prompt_tokens=meta.get("prompt_tokens", 0),
                     llm_diacritize_completion_tokens=meta.get("completion_tokens", 0),
                     llm_diacritize_truncated=int(meta.get("finish_reason") == "length"),
                     llm_diacritize_errors=int("error" in meta))
        retry = []
        for i, it in enumerate(items):
            res = parsed.get(i)
            why = "no_answer" if res is None or res["drop"] else diacritics.check(it.text, res["text"])
            if why is None:
                it.text, it.diac = unicodedata.normalize("NFC", res["text"]), "ok"
            elif attempt == 1:
                retry.append(it)
            else:
                it.diac = f"failed:{why}"  # keeps its undiacritized text
        for it in retry:  # alone: one bad item must not cost the others
            self._diacritize([it], attempt + 1)

    def _submit_diac(self, window: Window, lang: str) -> None:
        if lang == "ar" and self.cfg.ar_diacritics in ("critical", "full"):
            ready = [it for it in window.items if not it.reject and it.text]
            for it in ready:  # the original's shadda/tanween onto unchanged words
                it.text = diacritics.transfer_marks(it.org_text, it.text)
            for k in range(0, len(ready), self.cfg.diac_batch):
                window.diac_futs.append(self.llm_pool.submit(self._diacritize, ready[k:k + self.cfg.diac_batch]))
        window.diac_submitted = True

    # -- driver ----------------------------------------------------------------
    def run(self, chunk: dict, paths: RunPaths, claims: Claims, stop_after_windows: int | None = None) -> bool:
        prog = paths.progress_of(chunk) or {
            "chunk": chunk["id"], "status": "running", "offset": chunk["start"], "line": 0,
            "out_bytes": 0, "rej_bytes": 0, "counts": {}, "started": time.time(), "prompt": PROMPT_VERSION}
        if prog.get("status") == "done":
            return True
        part, rej, rev = paths.part(chunk), paths.rejects(chunk), paths.review(chunk)
        for f in (part, rej, rev):
            f.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as files:
            out_fh = files.enter_context(open(part, "ab"))
            rej_fh = files.enter_context(open(rej, "ab"))
            rev_fh = files.enter_context(open(rev, "ab"))
            return self._drive(chunk, prog, out_fh, rej_fh, rev_fh, paths, claims, stop_after_windows)

    def _drive(self, chunk: dict, prog: dict, out_fh, rej_fh, rev_fh, paths: RunPaths, claims: Claims,
               stop_after_windows: int | None) -> bool:
        cfg, lang = self.cfg, chunk["lang"]
        # Roll back anything written after the last commit.
        for fh, key in ((out_fh, "out_bytes"), (rej_fh, "rej_bytes"), (rev_fh, "rev_bytes")):
            fh.truncate(prog.get(key, 0))
            fh.seek(0, os.SEEK_END)
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
                if w.llm_submitted and not w.diac_submitted and all(f.done() for f in w.llm_futs):
                    for f in w.llm_futs:
                        f.result()
                    self._submit_diac(w, lang)
            head = inflight[0]
            if head.diac_submitted and all(f.done() for f in head.diac_futs):
                for f in head.diac_futs:
                    f.result()
                self._commit(head, chunk, prog, counts, out_fh, rej_fh, rev_fh, paths, claims)
                inflight.popleft()
                windows_done += 1
                if stop_after_windows and windows_done >= stop_after_windows:
                    return False  # test hook: simulate a crash after N commits
                continue
            pending = [f for w in inflight
                       for f in (w.diac_futs if w.diac_submitted else w.llm_futs if w.llm_submitted else w.asr_futs)
                       if not f.done()]
            if pending:
                wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            if time.time() - last_beat > 120:  # alive even while a slow window is in flight
                claims.heartbeat(chunk)
                last_beat = time.time()

        prog["status"], prog["finished"] = "done", time.time()
        _atomic_json(paths.progress / f"{chunk['id']}.json", prog)
        return True

    def _commit(self, w: Window, chunk: dict, prog: dict, counts: Counter,
                out_fh, rej_fh, rev_fh, paths: RunPaths, claims: Claims) -> None:
        out_lines, rej_lines, rev_lines = [], [], []
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
            if it.detail == "agree_guard":
                counts["agree_guard"] += 1
                rev_lines.append(json.dumps({
                    "audio_filepath": it.audio_filepath, "duration": it.duration, "org_text": it.org_text,
                    "asr_text": it.asr_text, "llm_text": it.llm_raw, "kept": "org_text",
                    "reason": "agree_guard"}, ensure_ascii=False))
            if it.diac:
                counts["diac:" + it.diac] += 1
            if chunk["lang"] == "ar":
                letters = len(diacritics._ARABIC_LETTER_RE.findall(it.text))
                counts["ar_letters"] += letters
                counts["ar_marks"] += round(diacritics.density(it.text) * letters)
                counts["ar_records_marked"] += diacritics.density(it.text) > 0
                counts["ar_records"] += 1
            rec = {"audio_filepath": it.audio_filepath, "duration": round(it.duration, 3),
                   "text": it.text, "org_text": it.org_text, "asr_text": it.asr_text}
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        counts["duplicates"] += w.skipped_dups
        for fh, rows in ((out_fh, out_lines), (rej_fh, rej_lines), (rev_fh, rev_lines)):
            if rows:
                fh.write(("\n".join(rows) + "\n").encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        prog.update(offset=w.last_offset, line=w.last_line + 1, out_bytes=out_fh.tell(),
                    rej_bytes=rej_fh.tell(), rev_bytes=rev_fh.tell(), counts=dict(counts), updated=time.time())
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
    asr = ASRClient(cfg.asr_urls, model=cfg.asr_model, max_tokens=cfg.asr_max_tokens)
    llm = LLMClient(cfg.llm_url, model=cfg.llm_model)
    for clients, what in ((asr.clients, "ASR"), ([llm.client], "LLM")):
        if not all(c.healthy() for c in clients):
            _wait_healthy(clients, what)
    tel = Telemetry()
    (paths.state / "workers").mkdir(exist_ok=True)
    stop = threading.Event()
    snap = threading.Thread(target=_snapshot_loop, daemon=True,
                            # One file per incarnation: a restarted worker keeps its id, and
                            # must add to the totals, not overwrite its earlier counters.
                            args=(tel, paths.state / "workers" /
                                  f"{cfg.worker_id.replace(':', '_')}_{int(tel.started * 1000)}.json", stop))
    snap.start()
    try:
        _work(cfg, plan, paths, claims, asr, llm, tel, stop_after_windows)
    finally:
        stop.set()
        snap.join(timeout=10)


def _work(cfg: WorkerConfig, plan: dict, paths: RunPaths, claims: Claims, asr: ASRClient, llm: LLMClient,
          tel: Telemetry, stop_after_windows: int | None) -> None:
    with ThreadPoolExecutor(cfg.asr_concurrency, thread_name_prefix="asr") as asr_pool, \
            ThreadPoolExecutor(cfg.llm_concurrency, thread_name_prefix="llm") as llm_pool:
        proc = ChunkProcessor(cfg, asr, llm, asr_pool, llm_pool, tel)
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
    tel = Counter()
    t_first = t_last = None
    for f in (paths.state / "workers").glob("*.json") if (paths.state / "workers").exists() else []:
        try:
            snap = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tel.update(snap["counters"])
        t_first = min(t_first or snap["started"], snap["started"])
        t_last = max(t_last or 0, snap["updated"])
    wall = max((t_last or 0) - (t_first or 0), 1.0)
    lanes_tel = {}
    for key in sorted({k.rsplit("_requests", 1)[0] for k in tel if k.startswith("llm_") and k.endswith("_requests")}):
        n = tel[f"{key}_requests"]
        lanes_tel[key[4:]] = {
            "requests": int(n),
            "items_per_request": round(tel[f"{key}_items"] / n, 1),
            "items_answered_pct": round(100 * tel[f"{key}_answered"] / max(tel[f"{key}_items"], 1), 1),
            "mean_latency_s": round(tel[f"{key}_seconds"] / n, 1),
            "mean_prompt_tokens": int(tel[f"{key}_prompt_tokens"] / n),
            "mean_completion_tokens": int(tel[f"{key}_completion_tokens"] / n),
            "hit_max_tokens_pct": round(100 * tel[f"{key}_truncated"] / n, 1),
            "errors": int(tel[f"{key}_errors"]),
        }
    llm_tokens = sum(v for k, v in tel.items() if k.endswith("_completion_tokens"))
    telemetry = {
        "llm": lanes_tel,
        "llm_output_tok_per_s": round(llm_tokens / wall, 1),
        "think_fallback_items": int(tel["llm_think_fallbacks"]),
        "asr_requests": int(tel["asr_requests"]),
        "asr_mean_latency_s": round(tel["asr_seconds"] / max(tel["asr_requests"], 1), 2),
        "asr_rtfx": round(tel["asr_audio_s"] / wall, 1),
    }
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
        "agree_guard": total["agree_guard"],
        "arabic_diacritics": {
            "marks_per_letter": round(total["ar_marks"] / max(total["ar_letters"], 1), 3),
            "records_with_marks_pct": round(100 * total["ar_records_marked"] / max(total["ar_records"], 1), 1),
            "diacritized": total["diac:ok"],
            "failed": {k[12:]: v for k, v in total.items() if k.startswith("diac:failed:")},
        },
        "telemetry": telemetry,
        "records_per_s": round(rate, 1),
        "eta_hours": round((to_clean - processed) / rate / 3600, 1) if rate else None,
    }


def assemble(run_root: str, allow_partial: bool = False) -> list[str]:
    """Concatenate each source's finished parts into ``<lang>/<tree>/<stem>.jsonl``."""
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
        target.parent.mkdir(parents=True, exist_ok=True)
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
