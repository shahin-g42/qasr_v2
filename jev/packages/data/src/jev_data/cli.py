"""CPU pipeline entry point. Every write is exclusive; every remote call is opt-in."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from jev_contracts import (
    ValidationError,
    atomic_write_json,
    canonical_json,
    verify_manifest,
    write_manifest,
)

from .io import (
    dataset_identity,
    guard_test,
    publish_rows,
    read_config,
    read_json,
    read_rows,
    require,
)
from .registry import Registry


def _inputs(paths: list[str]) -> list[dict]:
    rows = [row for path in paths for row in read_rows(path)]
    dataset_identity(rows)
    return rows


def _optional(path: str | None, default):
    return read_json(path) if path else default


def _teachers(args) -> list:
    from .teachers import TeacherClient, TeacherConfig

    require(args.teacher_config is not None and args.cache is not None,
            "explicit teacher configuration and response cache are required")
    config = read_config(args.teacher_config)
    from .io import version

    version(config.get("schema_version"))
    require(type(config.get("teachers")) is dict, "invalid teacher configuration")
    names = args.teacher or []
    require(bool(names), "explicit teacher names required")
    result = []
    for name in names:
        require(name in config["teachers"], "unknown configured teacher")
        teacher = TeacherConfig.from_dict(config["teachers"][name])
        from .io import digest

        result.append(TeacherClient(teacher, Path(args.cache) / digest(name),
                                    allow_network=args.allow_network))
    return result


async def _network_job(args, job_id: str, request: dict, operation) -> dict:
    from .io import digest, read_verified_json
    from .state import ShardState

    require(args.shard is not None and args.owner is not None,
            "network stages require shard-local ownership and an explicit owner")
    job_id = args.command + ":" + job_id
    with ShardState(args.shard, args.owner, local_storage=args.local_storage) as state:
        job = state.plan(job_id, request)
        if job["state"] == "accepted":
            return read_verified_json(Path(args.shard) / job["artifact"], "artifact.json")
        require(job["state"] != "quarantined", "use explicit retry for quarantined jobs")
        try:
            result = await operation()
            if args.command == "verify" and result["quality"]["tier"] != "silver":
                filename = "quarantine-" + digest([job_id, job["attempt"]]) + ".json"
                path = Path(args.shard) / filename
                atomic_write_json(path, result)
                state.transition(job_id, "quarantined", reason="independent_verification_failed")
                return result
            if job["state"] == "planned":
                state.transition(job_id, "authored")
            if state.get(job_id)["state"] == "authored":
                state.transition(job_id, "verified")
            state.publish(job_id, result)
            return result
        except (ValidationError, OSError):
            state.reconcile()
            current = state.get(job_id)
            if current["state"] not in {"accepted", "quarantined", "excluded"}:
                state.transition(job_id, "quarantined", reason="stage_failed")
            raise


def _emit(value: dict, output: str | None) -> None:
    if output:
        atomic_write_json(output, value)
    else:
        print(canonical_json(value))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="jev-data", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    sources = commands.add_parser("sources", help="Inspect strict pending/approved rights records")
    sources.add_argument("--registry", required=True)
    sources.add_argument("--output")

    ingest = commands.add_parser("ingest", help="Validate pinned local JSONL; no downloads")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--registry", required=True)
    ingest.add_argument("--output", required=True)

    plan = commands.add_parser("plan", help="Assign union-linked families and report actual quotas")
    plan.add_argument("--input", nargs="+", required=True)
    plan.add_argument("--config", required=True,
                      help="YAML/JSON containing task_cards and optional split_weights")
    plan.add_argument("--seed", required=True)
    plan.add_argument("--union-groups")
    plan.add_argument("--reservations")
    plan.add_argument("--output", required=True)

    annotation = commands.add_parser("annotate", help="Export/import blind human judgments")
    annotation.add_argument("action", choices=["export", "import"])
    annotation.add_argument("--input", nargs="+", required=True)
    annotation.add_argument("--registry", required=True)
    annotation.add_argument("--rubric-version")
    annotation.add_argument("--jobs", help="Immutable export directory for import")
    annotation.add_argument("--reviews", help="JSON array of submissions keyed by opaque job ID")
    annotation.add_argument("--output", required=True)

    generate = commands.add_parser("generate", help="Native symbolic or grounded authoring")
    generate.add_argument("--mode", choices=["oracle", "grounded"], required=True)
    generate.add_argument("--input", required=True,
                          help="Symbolic world or {slot, card, document} JSON")
    generate.add_argument("--registry", required=True)
    generate.add_argument("--source")
    generate.add_argument("--identifier")
    generate.add_argument("--split", default="train",
                          choices=["train", "development", "calibration", "test"])
    generate.add_argument("--calibration-subset",
                          choices=["temperature_fit", "policy_selection", "risk_verification"])
    generate.add_argument("--output", required=True)

    verify = commands.add_parser(
        "verify", help="Blind verification (dual-family default); never Gold")
    verify.add_argument("--input", nargs="+", required=True)
    verify.add_argument("--registry", required=True)
    verify.add_argument("--output", required=True)
    for command in (generate, verify):
        command.add_argument("--teacher-config")
        command.add_argument("--teacher", action="append")
        command.add_argument("--cache")
        command.add_argument("--allow-network", action="store_true")
        command.add_argument("--shard")
        command.add_argument("--owner")
        command.add_argument("--local-storage", action="store_true")
    verify.add_argument("--verification-mode", choices=["dual_family", "single_family"],
                        default="dual_family",
                        help="single_family allows one configured teacher; recorded in evidence")

    curate = commands.add_parser("curate", help="Rights, evidence, dedup and leakage report")
    audit = commands.add_parser("audit", help="Preregister/evaluate human audits or check releases")
    for command in (curate, audit):
        command.add_argument("--input", nargs="+")
        command.add_argument("--registry")
        command.add_argument("--protected", nargs="*")
        command.add_argument("--policy")
        command.add_argument("--resolutions")
        command.add_argument("--union-groups")
        command.add_argument("--output")
    audit.add_argument("--phase", choices=["plan", "run", "release"], default="run")
    audit.add_argument("--seed")
    audit.add_argument("--audit-plan")
    audit.add_argument("--reviews")
    audit.add_argument("--release")
    audit.add_argument("--split", default="train")

    freeze = commands.add_parser("freeze", help="Publish an audited, explicitly approved subset")
    freeze.add_argument("--input", nargs="+", required=True)
    freeze.add_argument("--registry", required=True)
    freeze.add_argument("--audit", required=True)
    freeze.add_argument("--approval", required=True)
    freeze.add_argument("--output", required=True)

    retry = commands.add_parser(
        "retry", help="Retry quarantined jobs; accepted rows stay immutable")
    retry.add_argument("--shard", required=True)
    retry.add_argument("--owner", required=True)
    retry.add_argument(
        "--job-id", required=True, help="Stage-prefixed ID: generate:ID or verify:ID")
    retry.add_argument("--local-storage", action="store_true")
    retry.add_argument("--output")

    acquire = commands.add_parser("acquire", help="Train-only uncertainty/diversity acquisition")
    acquire.add_argument("--input", nargs="+", required=True)
    acquire.add_argument("--registry", required=True)
    acquire.add_argument("--predictions", required=True)
    acquire.add_argument("--protected", nargs="*")
    acquire.add_argument("--count", required=True, type=int)
    acquire.add_argument("--seed", required=True)
    acquire.add_argument("--output")

    for command in (ingest, plan, annotation, generate, curate, audit, freeze):
        command.add_argument("--allow-test", action="store_true")
        command.add_argument("--purpose", choices=["training", "evaluation"], default="training")
    return root


def run(args) -> int:
    name = args.command
    if name == "sources":
        _emit(Registry.load(args.registry).to_dict(), args.output)
    elif name == "ingest":
        from .ingest import ingest

        rows = ingest(args.input, Registry.load(args.registry), args.source, output=args.output,
                      allow_test=args.allow_test, purpose=args.purpose)
        print(canonical_json({"ingested": len(rows)}))
    elif name == "plan":
        from .planning import plan

        rows = _inputs(args.input)
        guard_test(rows, allow_test=args.allow_test, purpose=args.purpose)
        config = read_config(args.config)
        result = plan(rows, seed=args.seed, cards=config["task_cards"],
                      weights=config.get("split_weights"),
                      union_groups=_optional(args.union_groups, []),
                      reservations=_optional(args.reservations, {}))
        guard_test(result["rows"], allow_test=args.allow_test, purpose=args.purpose)
        publish_rows(args.output, result["rows"], "planned", artifacts={"plan.json": result})
    elif name == "annotate":
        from .annotation import export_jobs, import_reviews
        from .io import read_verified_json

        rows = _inputs(args.input)
        registry = Registry.load(args.registry)
        for row in rows:
            registry.validate_row(row)
        if args.action == "export":
            require(args.rubric_version is not None, "rubric version is required")
            bundle = export_jobs(rows, rubric_version=args.rubric_version,
                                 allow_test=args.allow_test, purpose=args.purpose)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=False, mode=0o700)
            atomic_write_json(output / "jobs.json", {"schema_version": 1, "jobs": bundle["jobs"]})
            atomic_write_json(output / "private.json", bundle)
            write_manifest(output, ["jobs.json", "private.json"],
                           {"kind": "human_jobs", "dataset": bundle["dataset"]})
        else:
            require(args.jobs is not None and args.reviews is not None,
                    "jobs and reviews are required")
            bundle = read_verified_json(args.jobs, "private.json")
            public = read_verified_json(args.jobs, "jobs.json")
            require(public == {"schema_version": 1, "jobs": bundle["jobs"]},
                    "public annotation artifact differs from private binding")
            result = import_reviews(rows, bundle, read_json(args.reviews),
                                    allow_test=args.allow_test, purpose=args.purpose)
            publish_rows(args.output, result, "human_reviewed")
    elif name == "generate":
        from .generate import AUTHOR_PROMPT, author, oracle_candidate
        from .io import digest

        request = read_json(args.input)
        registry = Registry.load(args.registry)
        if args.mode == "oracle":
            require(args.source is not None and args.identifier is not None,
                    "source and identifier required")
            row = oracle_candidate(request, identifier=args.identifier, source_id=args.source,
                                   registry=registry, split=args.split,
                                   calibration_subset=args.calibration_subset,
                                   allow_test=args.allow_test, purpose=args.purpose)
        else:
            clients = _teachers(args)
            require(len(clients) == 1, "grounded author needs exactly one configured teacher")
            async def produce():
                return await author(request["slot"], request["card"], request["document"],
                                    registry, clients[0])

            identity = {"request": request, "teacher": clients[0].config.identity(),
                        "registry_hash": digest(registry.to_dict()),
                        "prompt_hash": digest(AUTHOR_PROMPT)}
            row = asyncio.run(_network_job(args, request["slot"]["id"], identity, produce))
        publish_rows(args.output, [row], "authored")
    elif name == "verify":
        from .io import digest
        from .verify import VERIFY_PROMPT, verify

        clients, rows, registry = _teachers(args), _inputs(args.input), Registry.load(args.registry)

        async def batch():
            output = []
            for row in rows:
                async def produce(current=row):
                    return await verify(current, registry, clients,
                                        mode=args.verification_mode)

                request = {"row": row, "teachers": [client.config.identity() for client in clients],
                           "registry_hash": digest(registry.to_dict()),
                           "prompt_hash": digest(VERIFY_PROMPT)}
                output.append(await _network_job(args, row["id"], request, produce))
            return output

        result = asyncio.run(batch())
        publish_rows(args.output, result, "verified")
        return 0 if all(row["quality"]["tier"] == "silver" for row in result) else 2
    elif name in {"curate", "audit"}:
        from .curate import audit, audit_plan, curate

        if name == "audit" and args.phase == "release":
            from .freeze import load_release

            require(args.release is not None, "release directory required")
            rows = load_release(args.release, args.split,
                                allow_test=args.allow_test, purpose=args.purpose)
            _emit({"verified": True, "count": len(rows),
                   "manifest": verify_manifest(args.release)}, args.output)
            return 0
        require(bool(args.input) and args.registry is not None, "input rows and registry required")
        rows, registry = _inputs(args.input), Registry.load(args.registry)
        protected = _inputs(args.protected) if args.protected else []
        guard_test(rows + protected, allow_test=args.allow_test, purpose=args.purpose)
        policy = read_config(args.policy) if args.policy else None
        if name == "curate":
            result = curate(rows, registry, protected=protected, policy=policy,
                            union_groups=_optional(args.union_groups, []),
                            resolutions=_optional(args.resolutions, []))
        elif args.phase == "plan":
            require(args.seed is not None, "preregistered sample seed required")
            result = audit_plan(rows, registry, seed=args.seed, policy=policy,
                                union_groups=_optional(args.union_groups, []))
        else:
            require(args.audit_plan is not None and args.reviews is not None,
                    "human audit requires a preregistered plan and actual review evidence")
            result = audit(rows, registry, plan=read_json(args.audit_plan),
                           reviews=read_json(args.reviews),
                           protected=protected, resolutions=_optional(args.resolutions, []))
        _emit(result, args.output)
        return 0 if result.get("passed", True) else 2
    elif name == "freeze":
        from .freeze import freeze

        result = freeze(args.output, _inputs(args.input), Registry.load(args.registry),
                        read_json(args.audit), approval=read_json(args.approval),
                        allow_test=args.allow_test, purpose=args.purpose)
        print(canonical_json(result))
    elif name == "retry":
        from .state import ShardState

        with ShardState(args.shard, args.owner, local_storage=args.local_storage) as state:
            job = state.retry(args.job_id)
            _emit({"job": job, "reconciliation": state.counts()}, args.output)
    elif name == "acquire":
        from .active_learning import acquire

        protected = _inputs(args.protected) if args.protected else []
        guard_test(protected)
        result = acquire(_inputs(args.input), read_json(args.predictions),
                         Registry.load(args.registry), count=args.count, seed=args.seed,
                         protected=protected)
        _emit(result, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        # No remote error bodies, raw rows, URLs or credentials are printed.
        print(f"jev-data: {type(exc).__name__}: operation rejected; "
              "check inputs and approval gates", file=sys.stderr)
        return 2
