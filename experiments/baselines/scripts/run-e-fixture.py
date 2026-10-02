#!/usr/bin/env python3
"""Reproduce E's synthetic external-runner smoke; never a strength evaluation.

Requires a separately built, licensed Fastchess f618e345 binary. No binaries or
external data are downloaded by this script. All outputs go below ARTIFACT_ROOT.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
FASTCHESS_SHA = "f618e34540f94f4719ad3817950618dabe441318"
RULES_SHA = "118dc0311a88e143be285703940321dc16261f6a"
CONTRACT_SHA = "67284c4f66f7a7ae9f46fa63dfd50e7410eb6845"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--fastchess", type=Path, required=True)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--label", default="e-runner-checkpoint")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=["normal", "cutoff", "cancel"], default=["normal", "cutoff", "cancel"])
    args = parser.parse_args()
    if not args.label or not all(c.isascii() and (c.isalnum() or c in "-_") for c in args.label):
        parser.error("label requires an ASCII basename")
    artifact_root = args.artifact_root.resolve()
    if artifact_root == REPO or REPO in artifact_root.parents:
        parser.error("artifact root must be outside the checkout")
    root = artifact_root / "runs" / args.label
    root.mkdir(parents=True, exist_ok=False)
    reports = root / "checks"
    reports.mkdir()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source_dirty = subprocess.run(["git", "diff", "--quiet", "--", "crates/rz-arena", "crates/rz-experiments"], cwd=REPO).returncode != 0
    if source_dirty:
        raise RuntimeError("commit E source before recording a clean binary identity")
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(artifact_root / "build")
    commands = []

    def run(name, command, expected=0, interrupt=False):
        start = time.monotonic_ns()
        with (reports / f"{name}.stdout").open("xb") as out, (reports / f"{name}.stderr").open("xb") as err:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=out, stderr=err)
            if interrupt:
                # Wait for the actual runner child, beyond input hashing.
                deadline = time.monotonic() + 10
                while child.poll() is None and time.monotonic() < deadline:
                    if (root / "attempt-cancel/opening.pgn").exists():
                        time.sleep(0.05)
                        if child.poll() is None:
                            child.send_signal(signal.SIGINT)
                            break
                    time.sleep(0.001)
                else:
                    raise RuntimeError("no live runner child available for cancellation")
            try:
                status = child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                raise
        commands.append({"name": name, "argv": [str(v) for v in command], "exit_code": status,
                         "elapsed_ns": time.monotonic_ns() - start, "expected_exit_code": expected})
        dump(reports / f"{name}.command.json", commands[-1])
        if status != expected:
            raise RuntimeError(f"{name}: exit {status}, expected {expected}; raw logs preserved")

    if not args.skip_build:
        run("build", [args.cargo, "build", "--manifest-path", "crates/rz-arena/Cargo.toml", "--all-features", "--bins", "--locked"])
        run("build-input", [args.cargo, "build", "--manifest-path", "crates/rz-experiments/Cargo.toml", "--bins", "--locked"])
    binaries = artifact_root / "build" / "debug"
    arena = binaries / "rz-arena"
    experiments = binaries / "rz-experiments"
    for origin, name in [(args.fastchess, "fastchess"), (binaries / "rz-arena-fixture-engine", "fixture-engine")]:
        shutil.copyfile(origin, root / name)
        (root / name).chmod(0o500)
    shutil.copyfile(REPO / "experiments/baselines/fixtures/e01-openings.json", root / "e01-openings.json")
    compiler = subprocess.check_output([args.cargo, "--version"], text=True).strip()
    base = json.loads((REPO / "experiments/baselines/fixtures/e01-input.json").read_text())
    base["contract_revision"] = "0.1"
    base["question"] = "Synthetic process and checked Rules smoke only; no NN or strength result"
    base["protocol"]["rules_reference_id"] = "rz-position-owned-rules"
    base["protocol"]["rules_version"] = RULES_SHA
    base["protocol"]["uci_adapter_version"] = "rz-e02-fastchess-fixture-v1"
    base["budget"].update(max_child_processes=3, max_output_bytes=2 * 1024**2, max_artifact_bytes=32 * 1024**2)
    base["lifecycle"].update(handshake_timeout_ms=500, drain_timeout_ms=500, shutdown_timeout_ms=200)
    for engine in base["engines"]:
        binary = root / "fixture-engine"
        engine["tool"].update(version="rz-arena-scripted-fixture-v1", source_commit=source,
                              build_mode="debug-synthetic-fixture", compiler=compiler,
                              target="x86_64-unknown-linux-gnu", isa="compiler-default")
        engine["tool"]["binary"].update(path=binary.name, sha256=sha(binary), bytes=binary.stat().st_size,
                                            license="MIT - synthetic scripted fixture")
    runner = base["protocol"]["runner"]
    binary = root / "fastchess"
    runner.update(version="fastchess alpha 1.8.2 20260726-f618e34", source_url="https://github.com/Disservin/fastchess",
                  source_commit=FASTCHESS_SHA, build_mode="release-O3-march-native-ZLIB-disabled",
                  compiler="g++ 14.2.0 (separate externally built tool)", target="x86_64-linux", isa="host-native")
    runner["binary"].update(path=binary.name, sha256=sha(binary), bytes=binary.stat().st_size,
                           source="https://github.com/Disservin/fastchess", license="MIT (ZLIB disabled)")
    cases = {}
    try:
        for scenario in args.scenarios:
            max_plies = 6 if scenario == "cutoff" else 256
            manifest = copy.deepcopy(base)
            manifest["run_id"] = f"e02-{scenario}"
            manifest["protocol"]["max_plies"] = max_plies
            input_path = root / f"{scenario}-input.json"
            lock = root / f"{scenario}-input-lock.json"
            plan = root / f"{scenario}-plan.json"
            dump(input_path, manifest)
            run(f"{scenario}-lock", [str(experiments), "lock", str(input_path), str(lock)])
            run(f"{scenario}-plan", [str(arena), "plan", str(lock), str(plan), "--max-pairs", "1", "--max-plan-bytes", "1048576"])
            run(scenario, [str(arena), "fixture-pair", str(plan), str(root), f"attempt-{scenario}", "--max-pairs", "1", "--max-plan-bytes", "1048576"], expected=2 if scenario == "cancel" else 0, interrupt=scenario == "cancel")
            receipt = json.loads((root / f"attempt-{scenario}/process-receipt.json").read_text())
            assert not receipt["execution_ready"]
            if scenario != "cancel":
                assert receipt["process"]["group_cleanup"] == "gone"
            else:
                # Some init environments retain orphan descendant zombies.
                # Keep Unverified and exclude the pair; never upgrade it to Gone.
                assert receipt["process"]["group_cleanup"] in {"gone", "unverified"}
            if scenario == "normal":
                assert [game["uci_moves"] for game in receipt["pgn_audit"]["games"]] == [["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]] * 2
                assert all(game["terminal_reason"] == "checkmate" for game in receipt["pgn_audit"]["games"])
            elif scenario == "cutoff":
                assert all(game["classification"] == "incomplete" for game in receipt["pgn_audit"]["games"])
            else:
                assert receipt["process"]["stop"] == "cancelled" and receipt["pgn_audit"] is None
            audit_name = f"{scenario}-ledger-audit"
            run(audit_name, [str(arena), "audit", str(plan), str(root / f"attempt-{scenario}/ledger.jsonl"), "--max-pairs", "1", "--max-plan-bytes", "1048576", "--max-events", "4", "--max-ledger-bytes", "65536"])
            summary = json.loads((reports / f"{audit_name}.stdout").read_text())["summary"]
            assert summary["completed_pairs"] == (1 if scenario == "normal" else 0)
            assert (summary["wins"], summary["draws"], summary["losses"]) == ((1, 0, 1) if scenario == "normal" else (0, 0, 0))
            cases[scenario] = {"process_stop": receipt["process"]["stop"], "group_cleanup": receipt["process"]["group_cleanup"], "summary": summary}
    finally:
        dump(root / "reproduction-receipt.json", {"source_commit": source, "rules_source_commit": RULES_SHA,
             "contract_source_commit": CONTRACT_SHA, "runner_source_commit": FASTCHESS_SHA,
             "actual_process_execution": True, "engines_and_results": "synthetic scripted fixture; no NN, GPU or strength claim",
             "commands": commands, "cases": cases, "binaries_excluded_from_public_evidence": ["fastchess", "fixture-engine"]})
    print(json.dumps({"scenarios": list(cases), "actual_execution": True, "synthetic_engines": True, "execution_ready": False}))


if __name__ == "__main__":
    main()
