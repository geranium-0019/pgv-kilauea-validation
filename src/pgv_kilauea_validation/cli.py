"""Command-line entry point; numerical imports follow optional CPU setup."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from . import __version__
from .io import Config, digest, write_json


def cpu_setup(enabled):
    if enabled:
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            os.environ[key] = "1"
        if platform.machine().lower() in ["x86_64", "amd64"]:
            os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX512_SKX,AVX2,FMA3"
        if hasattr(os, "sched_getaffinity"):
            allowed = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {min(allowed)})
    return {
        k: os.environ.get(k)
        for k in [
            "NPY_DISABLE_CPU_FEATURES",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ]
    }


def runtime():
    versions = {}
    for name in ["numpy", "scipy", "rasterio", "matplotlib", "PyYAML"]:
        versions[name] = importlib.metadata.version(name)
    return dict(
        tool_version=__version__,
        python=sys.version,
        platform=platform.platform(),
        dependencies=versions,
        cpu_affinity=sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    )


def run_pgv(config, pairs=None, overwrite=False, geotiff=False):
    from .pgv import run_pair, export_geotiffs

    rows = config.pairs(eligible_only=True)
    if pairs:
        unknown = set(pairs) - {r["pair_id"] for r in rows}
        if unknown:
            raise ValueError("Unknown/ineligible pairs: " + ", ".join(sorted(unknown)))
        rows = [r for r in rows if r["pair_id"] in pairs]
    state = dict(stage="pgv", state="running", completed=0, total=len(rows))
    try:
        for i, row in enumerate(rows, 1):
            state["current_pair"] = row["pair_id"]
            write_json(config.output / "pgv_status.json", state)
            print(f"[{i}/{len(rows)}] {row['pair_id']}", flush=True)
            run_pair(config, row, overwrite)
            if geotiff:
                export_geotiffs(config, row)
            state["completed"] = i
            write_json(config.output / "pgv_status.json", state)
        state.update(state="complete")
        state.pop("current_pair", None)
    except BaseException as e:
        state.update(state="failed", error=str(e))
        raise
    finally:
        write_json(config.output / "pgv_status.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce Kilauea PGV and GNSS validation from wrapped interferograms."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, default=Path("configs/kilauea.yaml"))
    parser.add_argument("--output", type=Path, help="Override output directory")
    parser.add_argument(
        "--safe-cpu",
        action="store_true",
        help="Use one CPU and reduced NumPy SIMD features on x86",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on errors"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare-paper", help="Attach existing paper inputs without modifying them"
    )
    prep.add_argument("--source", required=True, type=Path)
    prep.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    v = sub.add_parser(
        "validate", help="Check input manifest, files, and optional SHA256 hashes"
    )
    v.add_argument("--hashes", action="store_true")
    sub.add_parser(
        "classify", help="Recompute timing classes and check the audited manifest"
    )
    p = sub.add_parser(
        "pgv", help="Compute eligible pairs, resuming only verified matching outputs"
    )
    p.add_argument("--pair", action="append")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--geotiff", action="store_true")
    e = sub.add_parser("evaluate", help="Aggregate frequency/DI and compare GNSS signs")
    e.add_argument("--recompute-endpoints", action="store_true")
    f = sub.add_parser("figures", help="Render frequency, GNSS, and processing figures")
    r = sub.add_parser(
        "run", help="Classify, compute PGV, estimate GNSS, evaluate, and render"
    )
    for a in [f, r]:
        a.add_argument("--font", help="Font family, e.g. Aptos")
        a.add_argument("--font-dir", type=Path)
        a.add_argument("--method-pair", help="Processing example pair ID")
    r.add_argument(
        "--cached-endpoints",
        action="store_true",
        help="Use bundled GNSS endpoint estimates instead of raw five-minute data",
    )
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--geotiff", action="store_true")
    check = sub.add_parser(
        "verify", help="Compare results against independent paper reference artifacts"
    )
    check.add_argument("--no-map-comparison", action="store_true")
    sub.add_parser("doctor", help="Report numerical dependencies and paths")
    args = parser.parse_args(argv)
    environment = cpu_setup(args.safe_cpu)
    try:
        config = Config.load(args.config, args.output)
        if args.command == "doctor":
            print(
                json.dumps(
                    dict(
                        **runtime(),
                        config=str(config.file),
                        output=str(config.output),
                        environment=environment,
                    ),
                    indent=2,
                )
            )
            return 0
        config.output.mkdir(parents=True, exist_ok=True)
        import fcntl

        with (config.output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = dict(
                command=args.command,
                state="running",
                started_utc=datetime.now(timezone.utc).isoformat(),
                config_sha256=digest(config.file),
                settings=config.values,
                runtime=runtime(),
                environment=environment,
            )
            write_json(config.output / "run_status.json", state)
            try:
                if args.command == "prepare-paper":
                    from .dataset import attach

                    result = attach(config, args.source, args.mode)
                elif args.command == "validate":
                    from .dataset import validate

                    result = validate(config, args.hashes)
                elif args.command == "classify":
                    from .analysis import classification

                    result = dict(classified_pairs=len(classification(config)))
                elif args.command == "pgv":
                    result = run_pgv(config, args.pair, args.overwrite, args.geotiff)
                elif args.command == "evaluate":
                    from .analysis import aggregate
                    from .gnss import compare

                    result = compare(
                        config, aggregate(config), args.recompute_endpoints
                    )
                elif args.command == "figures":
                    from .figures import render

                    render(config, args.font, args.font_dir, args.method_pair)
                    result = dict(figures=str(config.output / "figures"))
                elif args.command == "verify":
                    from .analysis import verify

                    result = verify(config, not args.no_map_comparison)
                elif args.command == "run":
                    from .dataset import validate
                    from .analysis import classification, aggregate
                    from .gnss import compare
                    from .figures import render

                    validate(config)
                    classification(config)
                    run_pgv(config, overwrite=args.overwrite, geotiff=args.geotiff)
                    result = compare(
                        config, aggregate(config), not args.cached_endpoints
                    )
                    render(config, args.font, args.font_dir, args.method_pair)
                else:
                    raise ValueError("Unknown command")
                state.update(
                    state="complete",
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    result=result,
                )
                print(json.dumps(result, indent=2), flush=True)
            except BaseException as exc:
                state.update(
                    state="failed", error_type=type(exc).__name__, error=str(exc)
                )
                raise
            finally:
                write_json(config.output / "run_status.json", state)
        return 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
