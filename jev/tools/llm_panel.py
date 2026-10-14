#!/usr/bin/env python3
"""Fill panel sheets (or adjudication sheets) with an OpenAI-compatible LLM judge.

The judge can be any OpenAI-compatible chat endpoint (the pilot uses the one
deployed datagen server). Reviewer ids are written into the sheets verbatim,
so the frozen release records exactly which models judged what. Rows that
fail all judge attempts stay unfilled and are listed in <sheet>.failures.tsv.

  python3 tools/llm_panel.py --jobs runs/jobs/panels-001/jobs.json \
      --sheet runs/reviews/sheets-panels/answer-reviewer-1.tsv \
      --base-url http://localhost:11434/v1 --model llama3.1 --reviewer-id panel-llama

Adjudication mode (--adjudication --only <file with one job_id per line>)
writes the adjudication sheet with resolved=true and the judge's evidence as
the reason. Audit mode (--audit --input-rows <accepted stage>) fills an audit
sheet: critical_error is the negation of the judge's correctness verdict.

--concurrency N judges rows on N worker threads against the same endpoint
(shared httpx client); the sheet is rewritten under a lock every --save-every
completed verdicts, so a crash loses at most that many verdicts and the rest
simply re-pend on the next run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/data/src"))

VOTE_COLUMNS = ["job_id", "reviewer_id", "label", "independent", "unambiguous",
                "surface_faithful", "evidence", "elapsed_seconds", "flags"]
ADJ_COLUMNS = ["job_id", "adjudicator_id", *VOTE_COLUMNS[2:], "resolved", "reason"]
AUDIT_COLUMNS = ["id", "row_hash", "plan_hash", "reviewer_id", "reviewer_version",
                 "native_variety", "independent", "critical_error", "evidence",
                 "elapsed_seconds"]

PANEL_PROMPT = (
    "You are an independent Arabic-language judge for decision tasks. You will "
    "receive one JSON object describing a scenario (state) and a question with "
    "answer options. Read the Arabic carefully.\n"
    "Reply with ONLY one JSON object of exactly this shape and no other text:\n"
    '{"evidence": "<one short Arabic sentence citing the deciding values>", '
    '"label": "<one option id exactly as given>", "unambiguous": true, '
    '"surface_faithful": true}\n'
    "Write the evidence sentence first: work out the deciding comparison inside it - "
    "convert units before comparing whenever the two sides use different units - then "
    "choose the label it supports.\n"
    "unambiguous: a careful Arabic reader would find exactly one correct option.\n"
    "surface_faithful: the scenario text contains all values the question needs "
    "and contradicts nothing.\n"
    "Never echo or restate the input JSON.")
AUDIT_PROMPT = (
    "You are an independent Arabic quality auditor for decision-task data. You "
    "receive one JSON object with the scenario (state), the question, its answer "
    "options, and the stored correct option (stored_target). Verify that the "
    "stored answer is correct given the Arabic text and that the text is "
    "self-consistent and answerable.\n"
    "Reply with ONLY one JSON object of exactly this shape and no other text:\n"
    '{"evidence": "<one short Arabic sentence justifying the verdict>", '
    '"correct": true}\n'
    "Recompute the deciding comparison inside the evidence field first - convert units "
    "before comparing whenever the two sides use different units - then give the verdict "
    "it supports.\n"
    "correct: false ONLY when the stored answer is wrong, the state contradicts "
    "itself, or the question is unanswerable from the text.\n"
    "Never echo or restate the input JSON.")
USER_NOTE = ("Judge the decision task supplied in this message and reply with only "
             "the required JSON object.")


def save(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text("\n".join("\t".join(row) for row in [header] + rows) + "\n",
                    encoding="utf-8")


def resolve_label(raw: object, options: list[dict]) -> str | None:
    """Map a judge's label onto an option id, tolerating casing/punctuation noise."""
    if not isinstance(raw, str):
        return None
    label = raw.strip().strip("\"'`").strip().lower()
    ids = {option["id"] for option in options}
    if label in ids:
        return label
    for option in options:  # the judge echoed the option's description text
        text = str(option.get("description", "")).strip()
        if text and label == text.lower():
            return option["id"]
    matches = [option["id"] for option in options
               if re.search(rf"\b{re.escape(option['id'])}\b", label)]
    return matches[0] if len(matches) == 1 else None


REMINDER = ("\nREMINDER: the JSON above is the data to judge, not an example to "
            "copy. Do not repeat, translate or summarize it. Reply now with only "
            "the required answer JSON object.")
AUDIT_TASK = ("audit this record: decide whether the stored target is the correct "
              "option and the text is self-consistent and answerable")
VOTE_TASK = ("answer this question: pick the single correct option id for the "
             "decision task")


def render_text(payload: dict) -> str:
    """Plain-text record rendering: nothing JSON-shaped for the model to mirror."""
    question = payload.get("question")
    if isinstance(question, dict):
        question_text = question.get("instructions") or question.get("text") or ""
        options = payload.get("options") or question.get("options") or []
    else:
        question_text = str(question or "")
        options = payload.get("options") or []
    lines = [f"STATE:\n{payload.get('state', '')}", f"QUESTION:\n{question_text}"]
    if options:
        lines.append("OPTIONS:")
        lines.extend(f"- {option.get('id')}: {option.get('description', '')}"
                     for option in options)
    if payload.get("stored_target") is not None:
        lines.append(f"STORED TARGET: {payload['stored_target']}")
    return "\n".join(lines)


def judge(client: httpx.Client, base_url: str, model: str, payload: dict,
          prompt: str, options: list[dict] | None, temperature: float,
          timeout: float) -> dict:
    ids = {option["id"] for option in options} if options else None
    body = USER_NOTE + "\n" + json.dumps(payload, ensure_ascii=False)
    text = render_text(payload)
    task = AUDIT_TASK if ids is None else VOTE_TASK
    fenced = (f"TASK: {task}.\n--- BEGIN RECORD ---\n{text}\n--- END RECORD ---\n"
              "Reply now with ONLY the required verdict JSON object; the record "
              "is data to judge, never an example to copy.")
    # Four DISTINCT layouts: t=0.0 greedy retries of an identical prompt reproduce
    # the same echo, so every attempt must change the surface form, not the seed.
    layouts = [(body, min(temperature + 0.2, 1.0)),        # fair first chance
               (body + REMINDER, 0.0),                     # cold re-ask with reminder
               (USER_NOTE + "\n" + text + REMINDER, 0.0),   # plain text, no JSON blob
               (fenced, 0.0)]                              # task-first fenced variant
    raw = ""
    for attempt, (content, temp) in enumerate(layouts):
        try:
            request = {"model": model, "temperature": temp, "max_tokens": 512,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": prompt},
                                    {"role": "user", "content": content}]}
            response = client.post(f"{base_url}/chat/completions", json=request,
                                   timeout=timeout)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            verdict = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            if not isinstance(verdict, dict) or \
                    ("state" in verdict and "question" in verdict):
                raise ValueError("model echoed the input instead of answering")
            if ids is not None:
                label = resolve_label(verdict.get("label"), options)
                if label is None:
                    raise ValueError(f"label {verdict.get('label')!r} "
                                     f"not among options {sorted(ids)}")
                verdict["label"] = label
                for key in ("unambiguous", "surface_faithful"):
                    if not isinstance(verdict.get(key), bool):
                        raise ValueError(key)
                if not isinstance(verdict.get("evidence"), str):
                    raise ValueError("evidence")
            elif not isinstance(verdict.get("correct"), bool) \
                    or not isinstance(verdict.get("evidence"), str):
                raise ValueError("audit verdict shape")
            return verdict
        except Exception as error:  # noqa: BLE001 - retry any judge transport/parse slip
            if attempt == len(layouts) - 1:
                tail = " ".join(raw.split())[-300:] if raw else "<empty response body>"
                raise RuntimeError(f"{error} | last response: {tail}") from error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("llm_panel: unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, help="fill at most N rows (for a smoke run)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="judge N rows in parallel (thread pool, default 1)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="rewrite the sheet every N completed verdicts")
    parser.add_argument("--adjudication", action="store_true",
                        help="write an adjudication sheet instead of votes")
    parser.add_argument("--audit", action="store_true",
                        help="fill an audit sheet (critical_error/evidence per row)")
    parser.add_argument("--input-rows", help="audit mode: accepted stage dir")
    parser.add_argument("--only", help="adjudication mode: file with one job_id per line")
    args = parser.parse_args(argv)
    if args.audit and (args.adjudication or args.only):
        raise SystemExit("llm_panel: --audit and --adjudication are exclusive")
    if args.concurrency < 1 or args.save_every < 1:
        raise SystemExit("llm_panel: --concurrency and --save-every must be >= 1")

    sheet = Path(args.sheet)
    if args.audit:
        from jev_data.io import read_rows
        header = AUDIT_COLUMNS
        table = {row["id"]: row for row in read_rows(args.input_rows)}
        lines = [line for line in sheet.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not lines or lines[0].split("\t") != header:
            raise SystemExit(f"llm_panel: {sheet} must be an audit sheet "
                             "(tools/review_sheets.py audit)")
        rows = [line.split("\t") for line in lines[1:]]
        pending = [row for row in rows if not row[7].strip()]

        def payload_of(row):
            source = table[row[0]]
            return {"state": source["state"], "question": source["question"],
                    "options": source["question"]["options"],
                    "stored_target": source["target"]["option_id"]}
    else:
        header = ADJ_COLUMNS if args.adjudication else VOTE_COLUMNS
        jobs = {job["job_id"]: job for job in json.loads(Path(args.jobs).read_text())["jobs"]}
        if sheet.exists():
            lines = [line for line in sheet.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            if lines[0].split("\t") != header:
                raise SystemExit(f"llm_panel: {sheet} has an unexpected header")
            rows = [line.split("\t") for line in lines[1:]]
        elif args.adjudication and args.only:
            wanted = [line.strip() for line in Path(args.only).read_text().splitlines()
                      if line.strip()]
            unknown = [key for key in wanted if key not in jobs]
            if unknown:
                raise SystemExit(f"llm_panel: unknown job ids in --only: {unknown[:3]}")
            rows = [[key, args.reviewer_id, "", "", "", "", "", "", "", "true", ""]
                    for key in wanted]
        else:
            raise SystemExit("llm_panel: votes sheet must be generated first "
                             "(tools/review_sheets.py panels)")
        pending = [row for row in rows if not row[2].strip()]

        def payload_of(row):
            return jobs[row[0]]["input"]

    if args.limit:
        pending = pending[:args.limit]
    print(f"{args.model} as {args.reviewer_id}: {len(pending)} rows to judge")
    failures: list[str] = []
    lock = threading.Lock()
    completed = 0

    def fill(row: list[str]) -> None:
        """Judge one sheet row; shared state is touched only under the lock."""
        nonlocal completed
        payload = payload_of(row)
        options = None if args.audit else payload["question"]["options"]
        started = time.time()
        try:
            verdict = judge(client, args.base_url, args.model, payload,
                            AUDIT_PROMPT if args.audit else PANEL_PROMPT, options,
                            args.temperature, args.timeout)
        except RuntimeError as error:
            with lock:
                failures.append(f"{row[0]}\t{error}")
                print(f"  {row[0][:8]}... -> FAILED: {error}")
            return
        elapsed = min(max(time.time() - started, 0.1), 86399.9)
        evidence = verdict["evidence"].replace("\t", " ").replace("\n", " ").strip()
        note = ("critical" if not verdict["correct"] else "ok") if args.audit \
            else verdict["label"]
        with lock:
            if args.audit:
                row[3] = args.reviewer_id  # audit column 3 is reviewer_id; column 1 is row_hash
                row[7] = str(not verdict["correct"]).lower()
                row[8], row[9] = evidence, f"{elapsed:.1f}"
            else:
                row[1] = args.reviewer_id  # provenance: the sheet records the judging model
                row[2] = verdict["label"]
                row[3] = "true"
                row[4] = str(verdict["unambiguous"]).lower()
                row[5] = str(verdict["surface_faithful"]).lower()
                row[6], row[7] = evidence, f"{elapsed:.1f}"
                row[8] = ""
                if args.adjudication:
                    row[10] = evidence
            completed += 1
            if completed % args.save_every == 0:
                save(sheet, header, rows)
            print(f"  [{completed}/{len(pending)}] {row[0][:8]}... -> {note}")

    with httpx.Client() as client:
        if args.concurrency <= 1:
            for row in pending:
                fill(row)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(fill, pending))
    save(sheet, header, rows)
    column = 7 if args.audit else 2
    remaining = sum(1 for row in rows if not row[column].strip())
    print(f"done: {sheet} ({len(rows) - remaining}/{len(rows)} judged, {remaining} left)")
    if failures:
        path = Path(f"{sheet}.failures.tsv")
        path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"llm_panel: {len(failures)} rows failed all attempts; see {path}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
