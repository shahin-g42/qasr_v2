#!/usr/bin/env python3
"""Stdlib-only client for the QASR vLLM server (scripts/serve_asr_vllm.sh).

Uses the chat-completions route with ``file://`` audio: the server reads the
clip from the shared filesystem, so nothing is uploaded, and the raw answer
keeps the model's own ``language X`` header (a free language-ID signal).

    python3 scripts/asr_vllm_client.py transcribe --url http://inception-H100-hpc-029:8020 \
        --lang ar /lustrefs/.../clip.wav
    python3 scripts/asr_vllm_client.py check-prompt --url ... --lang ar /lustrefs/.../clip.wav
    python3 scripts/asr_vllm_client.py sample --url ... --manifest train_ar.jsonl --lang ar -n 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request

LANGUAGES = {"ar": "Arabic", "en": "English", "zh": "Chinese", "hi": "Hindi", "ml": "Malayalam"}
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
MODEL = "qasr"


def local_path(path: str) -> str:
    # Same remap the training loader applies (qasr.data.parse_record).
    return "/lustrefs/taiga/vast40" + path[len("/vast"):] if path.startswith("/vast/") else path


def _post(url: str, route: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def messages(path: str, lang: str | None) -> list[dict]:
    return [
        {"role": "system", "content": LANGUAGES.get(lang, lang) if lang else ""},
        {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "file://" + local_path(path)}}]},
    ]


def transcribe(
    url: str, path: str, lang: str | None, logprobs: bool = False, max_tokens: int = 512,
    timeout: float = 600,
) -> dict:
    """Greedy transcription. Returns raw text, parsed language/text, token usage, optional logprobs."""
    body = {
        "model": MODEL, "messages": messages(path, lang), "temperature": 0.0,
        "max_tokens": max_tokens, "skip_special_tokens": True,
    }
    if logprobs:
        body["logprobs"] = True
    res = _post(url, "/v1/chat/completions", body, timeout=timeout)
    choice = res["choices"][0]
    raw = choice["message"]["content"] or ""
    head, sep, text = raw.rpartition("<asr_text>")
    out = {
        "raw": raw,
        "language": head.strip()[len("language "):].strip() if sep and head.strip().startswith("language ") else None,
        "text": text if sep else raw,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": res.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": res.get("usage", {}).get("completion_tokens"),
    }
    if logprobs and choice.get("logprobs"):
        out["token_logprobs"] = [t["logprob"] for t in choice["logprobs"]["content"]]
    return out


def check_prompt(url: str, path: str, lang: str | None) -> bool:
    """The chat route must render exactly the training prompt (audio run collapsed)."""
    chat = _post(url, "/tokenize", {"model": MODEL, "messages": messages(path, lang),
                                    "add_generation_prompt": True})["tokens"]
    name = LANGUAGES.get(lang, lang) if lang else ""
    expected_text = (f"<|im_start|>system\n{name}<|im_end|>\n<|im_start|>user\n{AUDIO_PLACEHOLDER}"
                     "<|im_end|>\n<|im_start|>assistant\n")
    expected = _post(url, "/tokenize", {"model": MODEL, "prompt": expected_text,
                                        "add_special_tokens": False})["tokens"]
    # The chat route expands the single <|audio_pad|> to one token per encoder
    # frame; collapsing consecutive repeats undoes exactly that (the template
    # itself never emits a token twice in a row).
    collapsed = [t for i, t in enumerate(chat) if i == 0 or t != chat[i - 1]]
    ok = collapsed == expected
    print(f"prompt parity: {'OK' if ok else 'MISMATCH'} "
          f"({len(chat)} tokens, {len(chat) - len(collapsed) + 1} audio)")
    if not ok:
        print("  chat    :", collapsed[:60])
        print("  expected:", expected[:60])
    return ok


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("transcribe", "check-prompt"):
        p = sub.add_parser(name)
        p.add_argument("--url", required=True)
        p.add_argument("--lang")
        p.add_argument("audio")
    p = sub.add_parser("sample")
    p.add_argument("--url", required=True)
    p.add_argument("--lang", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "transcribe":
        print(json.dumps(transcribe(args.url, args.audio, args.lang, logprobs=True), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "check-prompt":
        return 0 if check_prompt(args.url, args.audio, args.lang) else 1

    with open(args.manifest, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    total = 0.0
    for row in rows:
        path = row.get("audio_filepath") or row.get("wav_path")
        ref = row.get("text") or row.get("transcript") or ""
        out = transcribe(args.url, path, args.lang)
        c = cer(ref, out["text"])
        total += c
        print(f"[{c:5.1%}] lang={out['language']}\n  REF: {ref}\n  HYP: {out['text']}")
    print(f"mean CER over {len(rows)}: {total / max(len(rows), 1):.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
