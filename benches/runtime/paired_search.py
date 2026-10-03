#!/usr/bin/env python3
"""OPT-00 diagnostic runner: bounded, serial fresh UCI processes, no shell.

The manifest locks binaries/config/fixtures before pilot/confirmation. No result
is a formal performance acceptance. Unknown resource facts remain unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import threading
import time

SCHEMA = "rz-opt00-paired-search/1"
REPO = Path(__file__).resolve().parents[2]
MAX_LOG = 2 * 1024 * 1024
MAX_LINE = 64 * 1024
NATIVE_FLAGS = {"--source-weights", "--onnx-model", "--export-manifest",
                "--manifest-sha256", "--ort-library", "--ort-sha256",
                "--cuda-bundle", "--cuda-bundle-sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, limit=65536):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "JSON exceeds byte limit")
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj
    return json.loads(data, object_pairs_hook=pairs)


def validate(manifest):
    require(set(manifest) <= {"schema", "provider", "seed", "witness_sha256", "timeout_ms", "max_rss_mib",
                             "max_throttled_usec", "run_wall_limit_ms", "max_run_bytes", "cpu_affinity",
                             "runner_cpu_affinity", "baseline", "candidate", "fixtures", "resource_notes", "profile_mode", "primary_metric"}, "unknown manifest field")
    require(manifest.get("schema") == SCHEMA, "unsupported schema")
    provider = manifest.get("provider")
    require(provider in {"cpu_mock", "onnx_cpu", "onnx_cuda"}, "explicit provider required")
    require(manifest.get("profile_mode") == "off", "this runner requires source profile off")
    require(manifest.get("primary_metric") == "position_bestmove_ns", "charge position preparation to the primary metric")
    require(type(manifest.get("seed")) is int and 0 <= manifest["seed"] < 2**32, "invalid seed")
    require(re.fullmatch(r"[0-9a-f]{64}", manifest.get("witness_sha256", "")), "correctness witness hash required")
    require(type(manifest.get("timeout_ms")) is int and 100 <= manifest["timeout_ms"] <= 60000, "invalid timeout")
    require(type(manifest.get("max_rss_mib")) is int and 16 <= manifest["max_rss_mib"] <= 65536, "invalid RSS cap")
    require(type(manifest.get("max_throttled_usec")) is int and manifest["max_throttled_usec"] >= 0, "invalid throttle threshold")
    require(type(manifest.get("run_wall_limit_ms")) is int and 100 <= manifest["run_wall_limit_ms"] <= 3600000, "invalid run wall cap")
    require(type(manifest.get("max_run_bytes")) is int and 1024 * 1024 <= manifest["max_run_bytes"] <= 1024**3, "invalid run artifact cap")
    affinity = manifest.get("cpu_affinity")
    require(isinstance(affinity, list) and len(affinity) <= 256 and all(type(x) is int and 0 <= x < 4096 for x in affinity), "invalid affinity")
    require(len(set(affinity)) == len(affinity), "duplicate affinity")
    runner_affinity = manifest.get("runner_cpu_affinity", [])
    require(isinstance(runner_affinity, list) and len(runner_affinity) <= 256 and all(type(x) is int and 0 <= x < 4096 for x in runner_affinity), "invalid runner affinity")
    require(not set(affinity) & set(runner_affinity), "runner and engine affinity overlap")
    for role in ("baseline", "candidate"):
        variant = manifest[role]
        require(set(variant) <= {"binary", "binary_sha256", "source_commit", "build_features", "compiler", "args", "expected_profile"}, "unknown variant field")
        require(re.fullmatch(r"[0-9a-f]{40}", variant.get("source_commit", "")), "full source SHA required")
        binary = Path(variant["binary"])
        require(binary.is_absolute() and binary.is_file() and binary.stat().st_size <= 64 * 1024 * 1024, "invalid engine binary")
        require(sha256(binary) == variant.get("binary_sha256"), "engine binary hash mismatch")
        require(isinstance(variant.get("build_features"), list) and all(isinstance(x, str) for x in variant["build_features"]), "feature declaration required")
        require(isinstance(variant.get("compiler"), str) and 1 <= len(variant["compiler"]) <= 256, "compiler declaration required")
        args = variant.get("args")
        require(isinstance(args, list) and len(args) <= 32 and all(isinstance(x, str) and len(x) <= 4096 and '\0' not in x for x in args), "invalid argv")
        if provider == "cpu_mock":
            require(args and args[0] == "--cpu-mock" and len(args) <= 2, "mock argv differs")
            if len(args) == 2:
                require(re.fullmatch(r"--mock-delay-ms=\d{1,4}", args[1]) and int(args[1].split('=')[1]) <= 1000, "mock delay invalid")
        else:
            require(args and args[0] == "--" + provider.replace('_', '-'), "native provider differs")
            require(len(args) % 2 == 1 and all(args[i] in NATIVE_FLAGS for i in range(1, len(args), 2)), "unsupported native argument")
            flags = args[1::2]
            required = NATIVE_FLAGS if provider == "onnx_cuda" else NATIVE_FLAGS - {"--cuda-bundle", "--cuda-bundle-sha256"}
            require(len(flags) == len(set(flags)) and set(flags) == required,
                    "native arguments must contain each required asset/hash flag exactly once")
            expected = variant.get("expected_profile", {})
            require(set(expected) == {"backend_sha256", "model_manifest_sha256", "encoding_manifest_sha256"}, "unknown native profile field")
            require(isinstance(expected, dict) and all(re.fullmatch(r"[0-9a-f]{64}", expected.get(k, "")) for k in
                ("backend_sha256", "model_manifest_sha256", "encoding_manifest_sha256")), "native identity lock required")
    fixtures = manifest.get("fixtures")
    require(isinstance(fixtures, list) and 1 <= len(fixtures) <= 32, "invalid fixture count")
    names = set()
    for fixture in fixtures:
        require(set(fixture) == {"name", "position", "go", "legal_moves", "terminal"}, "unknown fixture field")
        name = fixture.get("name", "")
        require(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) and name not in names, "invalid fixture name")
        names.add(name)
        position = fixture.get("position", "")
        require(isinstance(position, str) and len(position) <= 16384 and re.fullmatch(r"position (?:startpos|fen) [a-zA-Z0-9 /-]*|position startpos", position), "invalid position command")
        go = fixture.get("go", "")
        require(re.fullmatch(r"go nodes (?:[1-9]|[1-9]\d|1[01]\d|12[0-8])|go movetime \d{1,5}", go), "bounded go required")
        if go.startswith("go movetime"):
            require(1 <= int(go.split()[-1]) <= 10000, "invalid movetime")
        moves = fixture.get("legal_moves")
        require(isinstance(moves, list) and len(moves) <= 512 and all(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", m) for m in moves), "legal move witness required")
        require(type(fixture.get("terminal")) is bool and (bool(moves) != fixture["terminal"]), "terminal/move witness differs")
    require(isinstance(manifest.get("resource_notes"), dict), "known/unknown resource declarations required")
    return manifest


def order(seed, session, blocks):
    rng = random.Random(seed + session * 104729)
    patterns = (["ABBA", "BAAB"] * ((blocks + 1) // 2))[:blocks]
    rng.shuffle(patterns)
    return patterns


def resource_snapshot():
    result = {}
    for name, path in {
        "cpu_max": "/sys/fs/cgroup/cpu.max", "memory_max": "/sys/fs/cgroup/memory.max",
        "cpu_stat": "/sys/fs/cgroup/cpu.stat", "memory_events": "/sys/fs/cgroup/memory.events",
        "cpu_pressure": "/proc/pressure/cpu", "memory_pressure": "/proc/pressure/memory",
    }.items():
        try:
            result[name] = Path(path).read_text()[:4096]
        except OSError:
            result[name] = None
    return result


def throttle(snapshot):
    match = re.search(r"^throttled_usec (\d+)$", snapshot.get("cpu_stat") or "", re.M)
    return int(match[1]) if match else None


def memory_events(snapshot):
    return {key: int(value) for key, value in re.findall(r"^(oom|oom_kill|high) (\d+)$", snapshot.get("memory_events") or "", re.M)}


class Reader:
    def __init__(self, stream, path, lines=False):
        self.queue = queue.Queue(64) if lines else None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.read, args=(stream, path), daemon=True)
        self.thread.start()

    def put(self, item):
        if self.queue is not None:
            while not self.stop.is_set():
                try:
                    self.queue.put(item, timeout=.1)
                    return
                except queue.Full:
                    pass

    def read(self, stream, path):
        pending = b""
        total = 0
        try:
            with path.open("xb") as output:
                while not self.stop.is_set():
                    chunk = stream.read1(4096)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_LOG:
                        raise ValueError("engine log exceeds 2 MiB cap")
                    output.write(chunk)
                    if self.queue is not None:
                        pending += chunk
                        while b'\n' in pending:
                            line, pending = pending.split(b'\n', 1)
                            require(len(line) <= MAX_LINE, "engine line exceeds cap")
                            self.put(line.decode("utf-8", errors="strict").rstrip('\r'))
                        require(len(pending) <= MAX_LINE, "engine line exceeds cap")
                require(not pending, "unterminated protocol line")
        except (OSError, ValueError) as error:
            self.error = str(error)
        finally:
            self.put(None)


def read_until(reader, predicate, deadline, process, memory_cap, artifact_check=lambda: None):
    while time.monotonic() < deadline:
        if reader.error:
            raise ValueError(reader.error)
        usage = process_memory(process.pid)
        require(usage is None or usage <= memory_cap, "engine RSS cap exceeded")
        artifact_check()
        try:
            line = reader.queue.get(timeout=min(.05, max(.001, deadline - time.monotonic())))
        except queue.Empty:
            continue
        require(line is not None, "engine closed protocol early")
        if predicate(line):
            return line
    raise TimeoutError("engine protocol/deadline timeout")


def process_memory(pid):
    try:
        data = Path(f"/proc/{pid}/status").read_text()
        match = re.search(r"^VmHWM:\s+(\d+) kB$", data, re.M)
        return int(match[1]) * 1024 if match else None
    except OSError:
        return None


def artifact_bytes(root):
    return sum(path.lstat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())


def native_receipts(run_dir, variant, provider, pid):
    prefix = "native-cuda" if provider == "onnx_cuda" else "native-cpu"
    roots = list((run_dir / "native").glob(f"*/{prefix}-startup.v1.json"))
    require(len(roots) == 1, "one native process startup receipt required")
    startup = read_json(roots[0], 256 * 1024)
    termination = read_json(roots[0].with_name(f"{prefix}-termination.v1.json"), 256 * 1024)
    require(startup.get("schema_version") == 1 and startup.get("kind") == "startup" and startup.get("loaded") is True, "native startup incomplete")
    require(startup.get("process_id") == pid and startup["executable"]["sha256"] == variant["binary_sha256"], "native process/binary binding differs")
    profile = startup["profile"]
    for key, expected in {"provider": "cuda" if provider == "onnx_cuda" else "cpu", "precision": "fp32", "max_batch_items": 1,
                          "intra_threads": 1, "max_workers": 1, "full_steps": 1, "min_steps": 1, "max_steps": 1,
                          "require_full": True, "contract_major": 0, "contract_minor": 1}.items():
        require(profile.get(key) == expected, "native compute/contract profile differs")
    if provider == "onnx_cuda":
        require(profile.get("runtime_mapping_verified") is True and type(profile.get("executed_cuda_nodes")) is int and profile["executed_cuda_nodes"] > 0, "CUDA execution/mapping proof missing")
    expected = variant["expected_profile"]
    for key, observed in {"backend_sha256": profile["backend_sha256"], "model_manifest_sha256": profile["model"]["manifest_sha256"],
                          "encoding_manifest_sha256": profile["encoding"]["manifest_sha256"]}.items():
        require(observed == expected[key], "native model/encoding/backend identity differs")
    require(termination.get("schema_version") == 1 and termination.get("kind") == "termination" and termination.get("process_run_id") == startup["process_run_id"], "native termination binding differs")
    require(termination.get("startup") == startup and termination.get("run_succeeded") is True and termination.get("physical_drain") == "confirmed", "native run/drain failed")
    report = termination.get("report")
    require(isinstance(report, dict) and report.get("origin") == ("cuda_onnx" if provider == "onnx_cuda" else "cpu_onnx"), "native report origin differs")
    require(report.get("backend_sha256") == profile["backend_sha256"] and report.get("model_manifest_sha256") == profile["model"]["manifest_sha256"], "native termination profile differs")
    return {"startup_sha256": sha256(roots[0]), "termination_sha256": sha256(roots[0].with_name(f"{prefix}-termination.v1.json")),
            "report": termination.get("report"), "actual_inference_observed": termination.get("actual_cuda_inference_observed" if provider == "onnx_cuda" else "actual_cpu_inference_observed")}


def run_process(manifest, role, fixture, run_dir, run_deadline=None):
    run_dir.mkdir()
    variant = manifest[role]
    result = {"role": role, "fixture": fixture["name"], "source_commit": variant["source_commit"],
              "binary_sha256": variant["binary_sha256"], "provider": manifest["provider"], "started_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              "classification": "product_failure", "go_bestmove_ns": None, "bestmove": None,
              "position_ready_ns": None, "position_bestmove_ns": None,
              "reported_nodes": None, "peak_rss_bytes": None, "formal_acceptance": False}
    before = resource_snapshot()
    process = None
    readers = []
    began = time.monotonic()
    deadline = min(began + manifest["timeout_ms"] / 1000, run_deadline or float('inf'))
    stage = "lock"
    previous_artifacts = artifact_bytes(run_dir.parent)
    try:
        require(time.monotonic() < deadline, "run wall cap exceeded")
        require(sha256(variant["binary"]) == variant["binary_sha256"], "binary changed since lock")
        args = list(variant["args"])
        if manifest["provider"] != "cpu_mock":
            (run_dir / "native").mkdir()
            # The manifest retains pairs, but NativeConfig accepts named values.
            # Build one argv item per value; no shell parsing or whitespace split.
            args = [args[0]] + [f"{flag}={value}" for flag, value in zip(args[1::2], args[2::2])]
            args += ["--attestation", f"--output-root={run_dir / 'native'}"]
        stage = "spawn"
        process = subprocess.Popen([variant["binary"], *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stage = "affinity"
        if manifest["cpu_affinity"]:
            require(hasattr(os, "sched_setaffinity"), "requested affinity unsupported")
            os.sched_setaffinity(process.pid, manifest["cpu_affinity"])
            result["observed_affinity"] = sorted(os.sched_getaffinity(process.pid))
            require(result["observed_affinity"] == sorted(manifest["cpu_affinity"]), "affinity differs")
        stdout = Reader(process.stdout, run_dir / "stdout.log", lines=True)
        stderr = Reader(process.stderr, run_dir / "stderr.log")
        readers = [stdout, stderr]
        def send(command):
            process.stdin.write((command + '\n').encode())
            process.stdin.flush()
        def artifact_check():
            require(previous_artifacts + artifact_bytes(run_dir) <= manifest["max_run_bytes"], "run artifact cap exceeded")
        wait = lambda predicate: read_until(stdout, predicate, deadline, process, manifest["max_rss_mib"] * 1024 * 1024, artifact_check)
        stage = "uci"
        send("uci")
        wait(lambda line: line == "uciok")
        send("ucinewgame")
        send("isready")
        wait(lambda line: line == "readyok")
        stage = "search"
        position_started_ns = time.perf_counter_ns()
        send(fixture["position"])
        send("isready")
        wait(lambda line: line == "readyok")
        result["position_ready_ns"] = time.perf_counter_ns() - position_started_ns
        # Own the timestamp before writing/flushing go: include transport overhead.
        started_ns = time.perf_counter_ns()
        send(fixture["go"])
        def completed(line):
            if line.startswith("info "):
                match = re.search(r"(?:^| )nodes (\d+)(?: |$)", line)
                if match:
                    result["reported_nodes"] = int(match[1])
            return line.startswith("bestmove ")
        best = wait(completed)
        result["go_bestmove_ns"] = time.perf_counter_ns() - started_ns
        result["position_bestmove_ns"] = time.perf_counter_ns() - position_started_ns
        move = best.split()[1]
        require(move == "0000" if fixture["terminal"] else move in fixture["legal_moves"], "bestmove differs from fixture legal witness")
        result["bestmove"] = move
        result["peak_rss_bytes"] = process_memory(process.pid)
        stage = "shutdown"
        send("quit")
        process.stdin.close()
        # Drain protocol output while the engine exits; bounded queue cannot hide a second bestmove.
        while process.poll() is None or not stdout.queue.empty():
            require(time.monotonic() < deadline, "engine quit/drain timeout")
            try:
                line = stdout.queue.get(timeout=.01)
                require(line is None or not line.startswith("bestmove "), "duplicate bestmove")
            except queue.Empty:
                pass
        require(process.wait(timeout=max(.01, deadline - time.monotonic())) == 0, "engine failed exit")
        for reader in readers:
            reader.thread.join(timeout=max(.01, deadline - time.monotonic()))
            require(not reader.thread.is_alive() and reader.error is None, reader.error or "log reader incomplete")
        while not stdout.queue.empty():
            line = stdout.queue.get_nowait()
            require(line is None or not line.startswith("bestmove "), "duplicate bestmove")
        stderr_lines = (run_dir / "stderr.log").read_text().splitlines()
        if manifest["provider"] == "cpu_mock":
            require(not stderr_lines, "engine emitted diagnostic stderr")
        else:
            prefix = "ONNX CUDA evidence:" if manifest["provider"] == "onnx_cuda" else "ONNX CPU evidence:"
            require(len(stderr_lines) == 1 and stderr_lines[0].startswith(prefix), "native diagnostic stderr differs")
            stage = "native-attestation"
        if manifest["provider"] != "cpu_mock":
            result["native"] = native_receipts(run_dir, variant, manifest["provider"], process.pid)
        artifact_check()
        result["classification"] = "diagnostic"
    except (KeyError, TypeError, OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        result["failure_stage"] = stage
        if stage in {"lock", "spawn", "affinity"}:
            result["classification"] = "infrastructure_failure"
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()  # Only this runner's child, never another user's process.
            process.wait(timeout=5)
            for reader in readers:
                reader.stop.set()
                reader.thread.join(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream and not stream.closed:
                    stream.close()
            result["exit_code"] = process.returncode
        after = resource_snapshot()
        result["resources_before"] = before
        result["resources_after"] = after
        a, b = throttle(before), throttle(after)
        result["throttled_usec_delta"] = b - a if a is not None and b is not None and b >= a else None
        a_mem, b_mem = memory_events(before), memory_events(after)
        result["memory_event_deltas"] = {key: b_mem[key] - a_mem[key] if key in a_mem and key in b_mem and b_mem[key] >= a_mem[key] else None for key in ("oom", "oom_kill", "high")}
        tainted = (result["throttled_usec_delta"] or 0) > manifest["max_throttled_usec"] or any((value or 0) > 0 for value in result["memory_event_deltas"].values())
        if result["classification"] == "diagnostic" and tainted:
            result["classification"] = "contaminated"
        result["process_wall_ns"] = int((time.monotonic() - began) * 1e9)
        for stream in ("stdout", "stderr"):
            path = run_dir / f"{stream}.log"
            result[f"{stream}_sha256"] = sha256(path) if path.exists() else None
        result["log_errors"] = [reader.error for reader in readers if reader.error]
    return result


def run(manifest, output, phase, session):
    output = Path(output).resolve()
    require(not output.is_relative_to(REPO), "outputs must stay outside source repository")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.lock.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    runner_affinity = manifest.get("runner_cpu_affinity", [])
    if runner_affinity:
        require(hasattr(os, "sched_setaffinity"), "requested runner affinity unsupported")
        os.sched_setaffinity(0, runner_affinity)
    run_deadline = time.monotonic() + manifest["run_wall_limit_ms"] / 1000
    comparisons = ["variant"] if phase != "pilot" else ["control", "variant"]
    blocks = {"smoke": 1, "pilot": 3, "confirm": 4}[phase]
    had_failure = False
    summaries = []
    with (output / "processes.jsonl").open("x") as records:
        index = 0
        for comparison in comparisons:
            for fixture in manifest["fixtures"]:
                for block, pattern in enumerate(order(manifest["seed"], session, blocks)):
                    block_results = []
                    for ordinal, item in enumerate(pattern):
                        role = "baseline" if item == 'A' or comparison == "control" else "candidate"
                        result = run_process(manifest, role, fixture, output / f"process-{index:05}", run_deadline)
                        result.update(phase=phase, session=session, comparison=comparison, block=block,
                                      pattern=pattern, ordinal=ordinal, slot=item, process_index=index)
                        records.write(json.dumps(result, sort_keys=True) + '\n')
                        records.flush()
                        block_results.append(result)
                        index += 1
                        if result["classification"].endswith("_failure"):
                            had_failure = True
                            # Preserve the failure and abort the locked run; no replacement/slow-sample exclusion.
                            break
                    metric = manifest["primary_metric"]
                    complete = len(block_results) == 4 and all(r[metric] is not None and not r["classification"].endswith("_failure") for r in block_results)
                    ratio = None
                    if complete:
                        a = sum(r[metric] for r in block_results if r["slot"] == 'A') / 2
                        b = sum(r[metric] for r in block_results if r["slot"] == 'B') / 2
                        ratio = b / a if a else None
                    summaries.append({"fixture": fixture["name"], "comparison": comparison, "block": block,
                                      "pattern": pattern, "complete": complete, "b_over_a_ratio": ratio,
                                      "classes": [r["classification"] for r in block_results],
                                      "eligible_for_diagnostic": complete and all(r["classification"] == "diagnostic" for r in block_results)})
                    if had_failure:
                        break
                if had_failure:
                    break
            if had_failure:
                break
    (output / "execution.json").write_text(json.dumps({"phase": phase, "session": session, "processes": index,
        "complete": not had_failure, "performance_acceptance": False,
        "resource_notes": manifest["resource_notes"], "manifest_sha256": sha256(output / "manifest.lock.json")}, indent=2) + '\n')
    (output / "blocks.json").write_text(json.dumps({"session": session, "blocks": summaries,
        "confidence_interval": None, "formal_acceptance": False}, indent=2) + '\n')
    return 2 if had_failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("smoke", "pilot", "confirm"), required=True)
    parser.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    try:
        return run(validate(read_json(args.manifest)), args.output, args.phase, args.session)
    except (KeyError, TypeError, OSError, ValueError) as error:
        parser.exit(2, f"paired search: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
