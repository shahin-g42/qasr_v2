import argparse
import json
from pathlib import Path


def preflight(cfg, do_scan=False, verify_audio=False):
    from .data import ManifestIndex, scan
    from .processing import GQASRProcessor

    manifests = [
        p
        for key in ("train_manifest", "eval_manifest")
        for paths in cfg[key].values()
        for p in paths
    ]
    missing = [p for p in manifests if not Path(p).is_file()]
    result = {
        "model_name": "gqasr",
        "base_model_exists": Path(cfg["base_model"]).is_dir(),
        "train_manifests": sum(map(len, cfg["train_manifest"].values())),
        "eval_manifests": sum(map(len, cfg["eval_manifest"].values())),
        "languages": list(cfg["language_batch_counts"]),
        "missing_manifests": missing,
        "global_batch": cfg["batch_size"] * cfg["accumulation_steps"] * cfg["expected_world_size"],
        "language_quotas_per_rank_update": cfg["language_batch_counts"],
        "scan_complete": False,
        "ready_to_train": False,
    }
    if do_scan:
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cluster manifests are unavailable; first: {missing[0]}"
            )
        result["index"] = scan(cfg, GQASRProcessor(cfg["base_model"]), verify_audio)
    if Path(cfg["index_path"]).is_file():
        index = ManifestIndex(cfg["index_path"], "train")
        from .data import data_contract

        if index.metadata["data_contract"] != data_contract(cfg):
            raise ValueError("existing index/config mismatch")
        result["scan_complete"] = True
        result["counts"] = index.metadata["counts"]
    result["ready_to_train"] = (
        not missing and result["base_model_exists"] and result["scan_complete"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="GQASR ASR preparation, training and inference")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export-base")
    for name in ("acoustic-path", "decoder-path", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bridge-width", type=int, default=384)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--config", required=True)
    p.add_argument("--scan", action="store_true")
    p.add_argument("--verify-audio", action="store_true")
    p.add_argument("--report")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p = sub.add_parser("verify-base")
    p.add_argument("--model", required=True)
    p.add_argument("--report")
    p = sub.add_parser("transcribe")
    p.add_argument("--model", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=768)
    args = vars(parser.parse_args())
    command = args.pop("command")
    report = args.pop("report", None)
    if command == "export-base":
        import torch

        from .bootstrap import export_base

        torch.set_num_threads(4)
        result = export_base(**args)
    elif command == "prepare":
        from .config import prepare

        result = prepare(**args)
    elif command == "preflight":
        from .config import load_config

        result = preflight(load_config(args["config"]), args["scan"], args["verify_audio"])
    elif command == "train":
        from .train import train

        train(**args)
        return
    elif command == "verify-base":
        from .verify import verify_base

        result = verify_base(args["model"])
    else:
        import torch

        from .data import load_wave
        from .modeling import GQASRForSpeechRecognition
        from .processing import GQASRProcessor

        model = GQASRForSpeechRecognition.from_pretrained(args["model"], args["device"]).eval()
        processor = GQASRProcessor(args["model"])
        batch = processor.batch([load_wave(args["audio"], 0.1, 35.0)])
        with torch.no_grad():
            output = model.generate_transcripts(
                **{k: v.to(args["device"]) for k, v in batch.items()},
                max_new_tokens=args["max_new_tokens"],
            )
        result = {"text": processor.decode(output[0])}
    if report:
        with Path(report).open("x") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2, ensure_ascii=False))
