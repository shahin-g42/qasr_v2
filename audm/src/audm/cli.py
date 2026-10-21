import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="audm")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Export one complete offline AUD model")
    export.add_argument("--variant", choices=("audpro", "audturbo", "audflash"), required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--qasr-checkpoint")
    export.add_argument("--gqasr-checkpoint")
    export.add_argument("--encoder-source")
    export.add_argument("--encoder-revision", default="main")
    export.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    export.add_argument("--adapter-width", type=int, default=384)
    export.add_argument(
        "--bridge-init",
        choices=("gqasr", "new"),
        default="gqasr",
        help="Gemma AudFlash: reuse compatible trained GQASR bridge or initialize a new bridge",
    )
    export.add_argument("--flash-audio-stride", type=int, choices=(1, 2), default=1)
    export.add_argument("--seed", type=int, default=42)
    verify = sub.add_parser(
        "verify", help="Strict offline reload, checksums and forward/generation smoke"
    )
    verify.add_argument("directory")
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--no-smoke", dest="smoke", action="store_false")
    for name in ("train", "preflight"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument("--smoke", action="store_true")
            command.add_argument("--resume", help="Trainer checkpoint directory or auto")
        else:
            command.add_argument(
                "--scan",
                action="store_true",
                help="Scan all manifest records and calculate coverage",
            )
    check = sub.add_parser("distributed-check", help="NCCL check; launch with torchrun")
    check.add_argument("--expected-world-size", type=int, required=True)
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
    if command == "export":
        from .export import export_model

        result = export_model(**values)
    elif command == "verify":
        from .verify import verify

        result = verify(**values)
    elif command == "distributed-check":
        from .verify import distributed_check

        result = distributed_check(**values)
    else:
        try:
            from .train import preflight, run
        except ImportError as exc:
            raise RuntimeError(
                "Training uses the QASR checkout: install the root project or set PYTHONPATH=<repo>/src"
            ) from exc
        values["recipe"] = values.pop("config")
        result = run(**values) if command == "train" else preflight(**values)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
