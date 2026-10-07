"""Finite forward/export commands; this CLI has no training command."""
import argparse
import json

from .config import ModelConfig, ROLES, matmul_flops


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output", required=True)
    init.add_argument("--seed", type=int, default=1)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--include-validator", action="store_true")
    export.add_argument("--layout", choices=("separate_pc", "shared_pc_if"), default="separate_pc")
    export.add_argument("--rules-profile-json")
    check = commands.add_parser("numeric-check")
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--export", required=True)
    fixtures = commands.add_parser("rust-fixtures")
    fixtures.add_argument("--checkpoint", required=True)
    fixtures.add_argument("--export", required=True)
    fixtures.add_argument("--output", required=True)
    flops = commands.add_parser("flops")
    flops.add_argument("--role", choices=ROLES, default="proposer")
    flops.add_argument("--records", type=int, default=128)
    flops.add_argument("--candidates", type=int, default=32)
    flops.add_argument("--divergences", type=int, default=0)
    flops.add_argument("--batch", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "flops":
        result = matmul_flops(ModelConfig(), args.role, args.records, args.candidates, args.divergences, args.batch)
    else:
        import torch
        from .artifacts import export_checkpoint, initialize_checkpoint, numeric_check, rust_fixtures
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.command == "init": result = initialize_checkpoint(args.output, args.seed)
        elif args.command == "export": result = export_checkpoint(args.checkpoint, args.output, args.include_validator, args.layout, args.rules_profile_json)
        elif args.command == "rust-fixtures": result = rust_fixtures(args.checkpoint, args.export, args.output)
        else: result = numeric_check(args.checkpoint, args.export)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
