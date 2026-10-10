"""Separate CUDA Warm V2 registration and independent CPU arithmetic reference.

Export declares the device-KV/I/O-binding contract. It neither creates a CUDA
session nor registers a native physical lease. CPU reference success never
certifies CUDA capability, device residency, or an accepted private seed.
"""
import argparse
import json

from .onnx_warm import (PRIVATE_CUDA_WARM_SCHEMA, CUDA_WARM_LAYOUT,
                        CUDA_WARM_GRAPH_SEMANTICS, CUDA_PRIVATE_SEED_POLICY,
                        CUDA_QUERY_SEMANTICS, CUDA_EXECUTION_DOMAIN)

CUDA_WARM_GRAPH_FILE = CUDA_WARM_LAYOUT + ".onnx"
CUDA_WARM_MANIFEST_FILE = "cuda_warm_export.json"


def export_cuda_warm_checkpoint(checkpoint, directory, rules_profile_json):
    from .warm_artifacts import export_warm_checkpoint
    return export_warm_checkpoint(checkpoint, directory, rules_profile_json, cuda=True)


def validate_cuda_warm_export_manifest(manifest, metadata):
    from .warm_artifacts import validate_warm_export_manifest
    if manifest.get("schema") != PRIVATE_CUDA_WARM_SCHEMA:
        raise ValueError("CUDA Warm V2 rejects host Warm artifact domains")
    return validate_warm_export_manifest(manifest, metadata)


def numeric_check_cuda_warm_cpu_reference(checkpoint, directory):
    from .warm_artifacts import numeric_check_warm
    return numeric_check_warm(checkpoint, directory, cuda=True)


def numeric_check_cuda_warm_actual_reference(checkpoint, export_manifest, export_manifest_sha256,
                                             capture_json, capture_sha256, output_json, *, max_seconds=300):
    from .cuda_warm_actual_reference import numeric_check_cuda_warm_actual_reference as replay
    return replay(checkpoint, export_manifest, export_manifest_sha256, capture_json, capture_sha256,
                  output_json, max_seconds=max_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--rules-profile-json", required=True)
    numeric = commands.add_parser("cpu-reference")
    numeric.add_argument("--checkpoint", required=True)
    numeric.add_argument("--output", required=True)
    from .cuda_warm_actual_reference import add_actual_reference_parser
    add_actual_reference_parser(commands)
    arguments = parser.parse_args(argv)
    if arguments.command == "export":
        result = export_cuda_warm_checkpoint(arguments.checkpoint, arguments.output, arguments.rules_profile_json)
    elif arguments.command == "cpu-reference":
        result = numeric_check_cuda_warm_cpu_reference(arguments.checkpoint, arguments.output)
    else:
        from .cuda_warm_actual_reference import run_actual_reference_cli
        result = run_actual_reference_cli(arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
