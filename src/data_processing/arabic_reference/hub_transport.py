"""Exercise the Hub's actual HTTP decoder without network, CUDA or credentials."""

from __future__ import annotations

import importlib
import json
import shlex
import subprocess
import sys


def probe_transport():
    import huggingface_hub
    from huggingface_hub.utils import _http

    client = getattr(_http, "httpx2", None) or getattr(_http, "httpx", None)
    result = {"status": "passed", "huggingface_hub": huggingface_hub.__version__}
    if client is None:
        # Legacy Hub releases use requests and do not take the failing httpx2 path.
        return {**result, "backend": "requests", "brotli_probe": "not_applicable"}
    decoder = importlib.import_module(client.__name__ + "._decoders")
    codec = decoder.brotli
    result.update(backend=client.__name__, http_version=client.__version__)
    if codec is None:
        # Optional Brotli is absent, so this client does not advertise it.
        return {**result, "brotli_probe": "not_installed"}
    result.update(
        brotli_backend=codec.__name__,
        brotli_version=getattr(codec, "__version__", "unknown"),
        brotli_path=getattr(codec, "__file__", "unknown"),
    )
    payload = bytes(range(256)) * 8192
    compressed = codec.compress(payload)

    class Stream(client.SyncByteStream):
        def __iter__(self):
            for offset in range(0, len(compressed), 7):
                yield compressed[offset : offset + 7]

    try:
        response = client.Response(200, headers={"Content-Encoding": "br"}, stream=Stream())
        try:
            decoded = b"".join(response.iter_bytes())
        finally:
            response.close()
        if decoded != payload:
            raise ValueError("Brotli streaming round-trip changed bytes")
    except Exception as exc:
        raise ValueError(
            f"{client.__name__} / {result['brotli_backend']} {result['brotli_version']} "
            f"({result['brotli_path']}): {type(exc).__name__}: {exc}"
        ) from None
    return {**result, "brotli_probe": "passed", "decoded_bytes": len(decoded)}


def check_hub_transport():
    """Contain broken native decoder crashes and report a repair for this interpreter."""
    command = [sys.executable, "-m", "data_processing.arabic_reference.hub_transport", "--probe"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        detail = "HTTP decoder probe timed out after 30 seconds"
    else:
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        detail = (proc.stderr or proc.stdout).strip()[-2000:]
        detail = f"HTTP decoder probe exited {proc.returncode}: {detail}"
    repair = shlex.join(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "Brotli==1.2.0",
            "brotlicffi==1.2.0.2",
        ]
    )
    raise ValueError(
        f"Hugging Face HTTP compression is incompatible. {detail}\n"
        f"Repair the inference venv (leave the shared environment unchanged):\n  {repair}\n"
        "Then rerun the transport stage and the same benchmark command. Keep the model cache."
    )


if __name__ == "__main__":
    try:
        print(json.dumps(probe_transport() if "--probe" in sys.argv else check_hub_transport()))
    except (ValueError, ImportError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
