"""Explicit bounded OpenAI-compatible HTTP; no discovery, fallback, or secret logging."""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from jev_contracts import ValidationError, atomic_write_json, canonical_json

from .io import digest, fields, integer, number, parse_json, read_json, require, strings, text


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    family: str
    model: str
    revision: str
    endpoint: str
    api_key_env: str | None
    timeout_seconds: float
    transport_retries: int
    max_content_attempts: int
    max_requests: int
    concurrency: int
    max_tokens: int
    temperature: float
    allowed_data_classes: list[str]
    qualification_ref: str

    @classmethod
    def from_dict(cls, value: dict) -> TeacherConfig:
        fields(value, set(cls.__dataclass_fields__))
        for key in ("name", "family", "model", "revision", "endpoint", "qualification_ref"):
            text(value[key])
        require(value["revision"].lower() not in {"main", "latest", "master", "head"},
                "teacher revision must be pinned")
        url = urlsplit(value["endpoint"])
        require(url.scheme in {"http", "https"} and bool(url.hostname),
                "teacher endpoint unset/invalid")
        require(not url.username and not url.password and not url.query and not url.fragment,
                "credentials/query strings are forbidden in endpoint")
        require(url.scheme == "https" or url.hostname in {"localhost", "127.0.0.1", "::1"},
                "nonlocal teacher endpoints require TLS")
        if value["api_key_env"] is not None:
            text(value["api_key_env"])
        number(value["timeout_seconds"], 0.01, 600)
        integer(value["transport_retries"], 0, 5)
        integer(value["max_content_attempts"], 1, 3)
        integer(value["max_requests"], 1, 10**7)
        integer(value["concurrency"], 1, 128)
        integer(value["max_tokens"], 1, 65536)
        number(value["temperature"], 0, 2)
        strings(value["allowed_data_classes"])
        require(bool(value["allowed_data_classes"]) and
                set(value["allowed_data_classes"]) <= {"public", "private", "commissioned"},
                "invalid allowed data classes")
        return cls(**value)

    def identity(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "api_key_env"}


class TeacherError(ValidationError):
    """Safe error: remote payloads, credentials, and private text are never included."""


class TeacherClient:
    def __init__(self, config: TeacherConfig, cache: str | Path, *,
                 allow_network: bool = False, transport=None):
        self.config = TeacherConfig.from_dict(asdict(config))
        self.cache = Path(cache)
        self.allow_network = allow_network is True
        self.transport = transport
        self.requests = 0
        self.cache_hits = 0
        self.transport_failures = 0
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._locks: dict[str, asyncio.Lock] = {}

    async def complete(self, payload: dict, *, prompt: str, data_class: str,
                       attempt: int = 0, user_note: str = "") -> dict:
        config = self.config
        require(data_class in config.allowed_data_classes, "teacher data class not approved")
        integer(attempt, 0, config.max_content_attempts - 1)
        identity = {"teacher": config.identity(), "payload": payload, "prompt": text(prompt),
                    "user_note": str(user_note), "data_class": data_class,
                    "content_attempt": attempt}
        key = digest(identity)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        async with self._locks.setdefault(key, asyncio.Lock()), self._semaphore:
            path = self.cache / (key + ".json")
            if path.exists():
                record = read_json(path)
                fields(record, {"schema_version", "request_hash", "response_hash", "response"})
                require(type(record["schema_version"]) is int
                        and record["schema_version"] == 1 and record["request_hash"] == key,
                        "teacher cache identity mismatch")
                require(record["response_hash"] == digest(record["response"]),
                        "teacher cache corrupt")
                self.cache_hits += 1
                return record["response"]
            require(self.allow_network, "teacher network access is not explicitly enabled")
            import httpx

            headers = {"Content-Type": "application/json"}
            if config.api_key_env:
                secret = os.environ.get(config.api_key_env)
                require(bool(secret), "configured teacher credential is missing")
                headers["Authorization"] = "Bearer " + secret
            request = {"model": config.model, "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": (str(user_note) + "\n" if user_note else "")
                    + canonical_json(payload)},
            ], "max_tokens": config.max_tokens, "temperature": config.temperature,
                "response_format": {"type": "json_object"}}
            raw = None
            async with httpx.AsyncClient(timeout=config.timeout_seconds, transport=self.transport,
                                         follow_redirects=False, trust_env=False) as client:
                for retry in range(config.transport_retries + 1):
                    require(self.requests < config.max_requests, "teacher request budget exhausted")
                    self.requests += 1
                    try:
                        response = await client.post(
                            config.endpoint.rstrip("/") + "/chat/completions",
                            headers=headers, json=request)
                        if response.status_code == 429 or response.status_code >= 500:
                            raise TeacherError("retryable teacher transport failure")
                        if response.status_code != 200:
                            raise TeacherError("teacher HTTP request rejected")
                        try:
                            raw = parse_json(response.content)
                        except ValidationError:
                            raw = {"invalid_json_body_base64": base64.b64encode(
                                response.content).decode("ascii")}
                        if type(raw) is not dict:
                            raw = {"invalid_response_object": raw}
                        break
                    except (httpx.TimeoutException, httpx.TransportError, TeacherError) as exc:
                        self.transport_failures += 1
                        rejected = (isinstance(exc, TeacherError)
                                    and str(exc) == "teacher HTTP request rejected")
                        if rejected or retry == config.transport_retries:
                            raise TeacherError(
                                "teacher transport failed within configured bounds") from None
                        await asyncio.sleep(min(0.1 * 2**retry, 2))
            require(type(raw) is dict, "teacher response must be an object")
            # Freeze even malformed content: retries get a new explicit attempt identity.
            record = {"schema_version": 1, "request_hash": key, "response_hash": digest(raw),
                      "response": raw}
            try:
                atomic_write_json(path, record)
            except FileExistsError:
                existing = read_json(path)
                require(existing == record,
                        "concurrent teacher response differs; retry with shard ownership")
            return raw


def content(response: dict) -> dict:
    try:
        choices = response["choices"]
        require(type(choices) is list and len(choices) == 1, "expected one teacher completion")
        choice = choices[0]
        require(type(choice) is dict, "invalid teacher choice")
        require(choice["finish_reason"] == "stop", "truncated teacher completion")
        message = choice["message"]
        require(type(message) is dict and message.get("role") == "assistant",
                "invalid teacher message role")
        result = parse_json(text(message["content"]))
        require(type(result) is dict, "teacher content must be an object")
        return result
    except (KeyError, TypeError, ValidationError):
        raise TeacherError("malformed structured teacher completion") from None
