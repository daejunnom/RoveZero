#!/usr/bin/env python3
"""Preregistered B1 adapter equivalence: fresh cgroups, fixed work, AA then AB."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

from bounded_local import OwnedRun, outside_git, put, resolve_affinity
from paired_search import native_receipts, read_json, require, sha256

SCHEMA = "rz-adapter-regression/2"
POSITIONS = ["position startpos", "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8"]
SIMULATIONS = 4096
TREE_MAX_EDGES = 262_144
# An untimed `go nodes` inherits the engine's 30-second resource wall. Give
# fixed work explicit headroom; the independent whole-run 180-second cap remains.
SEARCH_COMMAND = "go nodes 4096 movetime 75000"
PROFILE_KEYS = {"backend_sha256", "model_manifest_sha256", "encoding_manifest_sha256"}
ARGUMENTS = {"--source-weights", "--onnx-model", "--export-manifest", "--manifest-sha256",
             "--ort-library", "--ort-sha256", "--cuda-bundle", "--cuda-bundle-sha256",
             "--runtime-cache-root", "--search-simulations", "--search-max-edges", "--final-selection"}

def resource_affinity(notes, allowed=None):
    require(isinstance(notes, dict), "registered resource policy required")
    affinity=resolve_affinity(notes.get("cpu_affinity"),allowed)
    require(notes.get("cpu_affinity")==affinity,"explicit sorted CPU affinity required")
    fixed={"memory_high_GiB":6,"memory_max_GiB":12,"swap":0,"precision":"fp32",
           "tf32":False,"history_fill":"no","batch":1}
    require(all(type(notes.get(key)) is type(value) and notes.get(key)==value
                for key,value in fixed.items()),"registered fixed resource/input policy differs")
    return affinity


def validate(manifest):
    require(set(manifest) == {"schema", "baseline", "candidate", "assets", "numerical_witnesses", "resource_notes", "runtime_cache_preparation"}, "unknown or missing registration field")
    require(manifest["schema"] == SCHEMA, "unsupported adapter regression schema")
    for role in ("baseline", "candidate"):
        variant = manifest[role]
        require(set(variant) == {"binary", "binary_sha256", "source_commit", "source_patch", "build_features", "compiler", "args", "expected_profile"}, "variant fields differ")
        require(re.fullmatch(r"[0-9a-f]{40}", variant["source_commit"]) is not None, "full source commit required")
        patch = variant["source_patch"]
        if patch is not None:
            require(set(patch) == {"path", "bytes", "sha256"}, "source patch identity invalid")
            path = outside_git(patch["path"])
            require(path.is_file() and 0 < path.stat().st_size == patch["bytes"] <= 1024**2 and sha256(path) == patch["sha256"], "registered source patch differs")
        binary = outside_git(variant["binary"])
        require(binary.is_file() and binary.stat().st_size <= 64*1024**2 and sha256(binary) == variant["binary_sha256"], "registered engine binary differs")
        require(variant["build_features"] == ["onnx-cuda"], "this recipe fixes the B1 onnx-cuda build")
        require(isinstance(variant["compiler"], str) and 1 <= len(variant["compiler"]) <= 256, "compiler identification missing")
        args = variant["args"]
        require(isinstance(args, list) and len(args) <= 20 and args[0] == "--onnx-cuda", "explicit bounded CUDA argv required")
        flags = {}
        for arg in args[1:]:
            require(isinstance(arg, str) and len(arg) <= 4096 and not any(c in arg for c in "\n\r\0"), "invalid argv item")
            key, separator, value = arg.partition("=")
            require(separator and key in ARGUMENTS and key not in flags and value, "unknown/duplicate native flag")
            flags[key] = value
        require(set(flags) == ARGUMENTS, "all native asset, cache and search arguments must be fixed")
        require(flags["--search-simulations"] == "4096" and flags["--final-selection"] == "visits", "search configuration differs")
        require(flags["--search-max-edges"] == str(TREE_MAX_EDGES), "shared explicit tree envelope differs")
        require(set(variant["expected_profile"]) == PROFILE_KEYS and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in variant["expected_profile"].values()), "semantic profile identity missing")
    for field in ("args", "compiler", "build_features", "expected_profile"):
        require(manifest["baseline"][field] == manifest["candidate"][field], "adapter control changed " + field)
    require(isinstance(manifest["assets"], list) and 1 <= len(manifest["assets"]) <= 32, "finite asset registration required")
    for artifact in manifest["assets"]:
        require(set(artifact) == {"path", "bytes", "sha256"}, "artifact fields differ")
        path = Path(artifact["path"])
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.stat().st_size == artifact["bytes"] and 0 < artifact["bytes"] <= 2*1024**3, "registered asset size/path differs")
        require(sha256(path) == artifact["sha256"], "registered asset hash differs")
    require(isinstance(manifest["numerical_witnesses"], list) and 1 <= len(manifest["numerical_witnesses"]) <= 8, "independent numerical evidence required")
    for witness in manifest["numerical_witnesses"]:
        path = outside_git(witness["path"])
        require(sha256(path) == witness["sha256"] and read_json(path, 1024**2).get("status") == "passed", "numerical witness was not passed")
    resource_affinity(manifest["resource_notes"])
    from shared_runtime_cache import validate as validate_cache
    cache=manifest["runtime_cache_preparation"]
    root,_=validate_cache(cache)
    flags=dict(arg.split("=",1) for arg in manifest["baseline"]["args"][1:])
    require(root.parent==Path(flags["--runtime-cache-root"]),"cache preparation targets a different runtime root")
    bundle=read_json(flags["--cuda-bundle"],64*1024)
    files=[{k:f[k] for k in ("filename","bytes","sha256")} for f in bundle["files"]]
    require(sorted(cache["files"],key=lambda f:f["filename"])==sorted(files,key=lambda f:f["filename"]),"cache preparation differs from registered bundle")
    played=next(a for a in manifest["assets"] if a["path"]==flags["--onnx-model"])
    raw=[read_json(w["path"],1024**2) for w in manifest["numerical_witnesses"]]
    raw=[w for w in raw if w.get("onnx_sha256")==played["sha256"]]
    require(len(raw)==1 and raw[0]["cuda_bundle_sha256"]==cache["canonical_sha256"],"cache canonical identity lacks the played model's numerical proof")
    return manifest


def prepare_cache(spec, output, deadline, affinity=None):
    directory=output/"cache-preparation"
    owner=OwnedRun(directory,wall=min(60,deadline-time.monotonic()),affinity=affinity)
    report=None
    try:
        put(directory/"spec.json",spec)
        owner.spawn([sys.executable,str(Path(__file__).with_name("shared_runtime_cache.py")),
                     str(directory/"spec.json"),str(directory/"report.json")])
        require(owner.wait_exit()==0,"shared runtime cache preparation failed")
        report=read_json(directory/"report.json",64*1024)
        require(report["status"]=="passed","shared runtime preparation did not pass")
    finally:
        capture=owner.finish()
        put(directory/"capture.json",capture)
    events=dict(line.split() for line in capture["resources"]["memory.events"].splitlines())
    require(capture["exit_code"]==0 and not capture["forced_cleanup"] and not capture["cleanup_error"] and
            not capture["remaining_owned_processes"] and not int(events["oom"]) and not int(events["oom_kill"]),
            "cache preparation resource or process drain failed")
    return {"status":"passed","report_sha256":sha256(directory/"report.json"),
            "capture_sha256":sha256(directory/"capture.json"),"report":report,"capture":capture}


def capture_valid(result):
    require(result.get("diagnostic_only", False) is False and
            result.get("performance_measurement", True) is True,
            "diagnostic evidence is not an AA/AB performance sample")
    require(result.get("accepted") is True, "failed fixed-work run")
    capture = result["capture"]
    require(capture["exit_code"] == 0 and not capture["forced_cleanup"] and not capture["cleanup_error"] and not capture["remaining_owned_processes"], "natural process/physical drain missing")
    t = capture["whole_wall_seconds"]
    p = int(capture["resources"]["memory.peak"])
    require(math.isfinite(t) and t > 0 and p > 0, "unknown primary time/peak")
    return t, p


def pair_result(left, right, comparison):
    ta, pa = capture_valid(left)
    tb, pb = capture_valid(right)
    require(left["work"] == right["work"], "workload, consumption or bestmove differs")
    tr, pr = tb/ta, pb/pa
    if comparison == "AA":
        passed = max(tr, 1/tr) <= 1.05 and max(pr, 1/pr) <= 1.05
    else:
        passed = tr <= 1.05 and pr <= 1.05
    return {"comparison": comparison, "time_ratio": tr, "peak_ratio": pr, "passed": passed,
            "A_time": ta, "B_time": tb, "A_peak": pa, "B_peak": pb}


def summarize(pairs):
    # Accumulate only this registered comparison. Per-pair gates remain decisive.
    ab = [p for p in pairs if p["comparison"] == "AB"]
    totals = {k: sum(p[k] for p in ab) for k in ("A_time", "B_time", "A_peak", "B_peak")}
    complete = len(pairs) == 8 and len(ab) == 5 and all(p["passed"] for p in pairs)
    return {"status": "passed" if complete else "hold", "pair_count": len(pairs), "AB_totals": totals,
            "AB_time_ratio_of_sums": totals["B_time"]/totals["A_time"] if ab else None,
            "AB_peak_ratio_of_sums": totals["B_peak"]/totals["A_peak"] if ab else None,
            "all_individual_pair_gates_required": True, "past_other_experiments_in_denominator": False}


def run_once(variant, directory, overall_deadline, affinity=None, *, diagnostic_only=False):
    require(type(diagnostic_only) is bool, "explicit diagnostic classification required")
    owner = OwnedRun(directory, wall=min(180, overall_deadline-time.monotonic()),affinity=affinity)
    result = {"source_commit": variant["source_commit"], "source_patch": variant["source_patch"], "binary_sha256": variant["binary_sha256"], "accepted": False,
              "diagnostic_only": diagnostic_only, "performance_measurement": not diagnostic_only}
    stage = "binary_verification"
    try:
        require(sha256(variant["binary"]) == variant["binary_sha256"], "binary changed after registration")
        (directory/"native").mkdir()
        child = owner.spawn([variant["binary"], *variant["args"], "--attestation", "--output-root=" + str(directory/"native")])
        stage = "uci_and_model_ready"
        owner.send("uci")
        owner.until(lambda line: line == "uciok")
        owner.send("isready")
        owner.until(lambda line: line == "readyok")
        ready = time.monotonic()
        result["ready_seconds"] = ready-owner.started
        result["resource_checkpoints"] = [owner.resource_checkpoint("model_ready")]
        work = []
        for position in POSITIONS:
            stage = "fixed_work_search"
            owner.send("ucinewgame")
            owner.send("isready")
            owner.until(lambda line: line == "readyok")
            began = time.monotonic()
            owner.send(position)
            owner.send("isready")
            owner.until(lambda line: line == "readyok")
            owner.send(SEARCH_COMMAND)
            reported = []
            def observe(line):
                if line.startswith("info "):
                    match = re.search(r"(?:^| )nodes (\d+)(?: |$)", line)
                    if match:
                        reported.append(int(match[1]))
            line = owner.until(lambda line: line.startswith("bestmove "), observe)
            move = line.split()[1]
            require(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is not None, "nonterminal fixed input returned invalid move syntax")
            work.append({"position": position, "bestmove": move, "requested_simulations": SIMULATIONS})
            result.setdefault("searches", []).append({"position_bestmove_seconds": time.monotonic()-began, "reported_nodes": reported[-1] if reported else None})
            result["resource_checkpoints"].append(owner.resource_checkpoint(f"input_{len(work)}_complete"))
        stage = "natural_quit_and_physical_drain"
        drain_started = time.monotonic()
        owner.send("quit")
        def no_duplicate(line):
            require(not line.startswith("bestmove "), "duplicate final move")
        require(owner.wait_exit(no_duplicate) == 0, "native process failed")
        result["search_sequence_seconds"] = drain_started-ready
        result["drain_seconds"] = time.monotonic()-drain_started
        result["resource_checkpoints"].append(owner.resource_checkpoint("physical_process_exited"))
        for file in owner.logs.values():
            file.flush()
        errors=(directory/"stderr.log").read_text(encoding="utf-8").splitlines()
        failure=next((line for line in errors if line.startswith("SearchFailed:")),None)
        require(failure is None,"fixed-work search failed: "+str(failure))
        stage = "native_evidence"
        native = native_receipts(directory, variant, "onnx_cuda", child.pid)
        term_path = next((directory/"native").glob("*/native-cuda-termination.v1.json"))
        term = read_json(term_path, 256*1024)
        report = native["report"]
        require(native["actual_inference_observed"] and term["actual_rules_search_backup_observed"] and not term["retained_owner_and_evidence"], "physical inference/consumption evidence missing")
        require(all(term[k] is None for k in ("original_service_failure", "collection_failure", "runtime_mapping_failure")), "native termination failure")
        require(all(report[k] is None for k in ("overflow", "boundary_error", "poison_error")) and not report["failures"] and not report["canceled"]["count"] and not report["expired"]["count"], "request failure/cancel/loss")
        require(all(report["observations"][k] == 0 for k in ("scheduler_dropped", "scheduler_counter_overflow", "delivery_dropped", "delivery_counter_overflow", "drain_discarded_results")), "observation loss")
        roots, backups = report["search_root_initializations"]["count"], report["search_non_root_backups"]["count"]
        require(roots == 2 and backups == 2*SIMULATIONS and report["completed_by_runtime"] == roots+backups, "fixed 8192 NN-backed simulations plus two roots not reconciled; terminal work is not silently substituted")
        profile = read_json(term_path.with_name("native-cuda-search-config.v1.json"), 64*1024)
        require(profile["process_run_id"] == term["process_run_id"], "search configuration belongs to another process")
        startup_bytes=(json.dumps(term["startup"],ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
        require(profile["startup_sha256"] == hashlib.sha256(startup_bytes).hexdigest(), "search configuration startup digest differs")
        served=term["startup"]["profile"]
        flags=dict(arg.split("=",1) for arg in variant["args"][1:])
        export=read_json(flags["--export-manifest"],64*1024)
        require(served["history_fill"] == "no" and served["tf32"] is False and served["onnx_sha256"] == export["onnx_sha256"] and served["source_weights_gzip_sha256"] == export["source_gzip_sha256"] and served["runtime_bundle_manifest_sha256"] == flags["--cuda-bundle-sha256"], "served model/input semantics differ")
        require(profile["simulations"] == SIMULATIONS and profile["final_selection"] == "visits" and profile["policy_temperature_milli"] == 1000 and profile["batch_size"] == 1 and profile["search_workers"] == 1 and not profile["raw_cache"], "served search configuration differs")
        require(profile["max_edges"] == TREE_MAX_EDGES and profile["max_nodes"] == 20_000 and profile["max_depth"] == 128, "served tree envelope differs")
        result["work"] = {"inputs": work, "nn_root_consumed": roots, "nn_backup_consumed": backups, "completed_by_runtime": report["completed_by_runtime"], "physical_invocations": "unknown_not_a_full_physical_journal"}
        result["native"] = native
        result["accepted"] = True
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as exc:
        result.update(error=str(exc), failure_stage=stage)
    finally:
        result["capture"] = owner.finish()
        capture = result["capture"]
        events = dict(line.split() for line in capture["resources"]["memory.events"].splitlines())
        if capture["forced_cleanup"] or capture["cleanup_error"] or capture["exit_code"] != 0 or capture["remaining_owned_processes"] or int(events["oom"]) or int(events["oom_kill"]):
            result["accepted"] = False
            result.setdefault("error", "process/resource/physical cleanup failed")
        for name in ("stdout", "stderr"):
            path=directory/(name+".log")
            result[name+"_sha256"] = sha256(path) if path.exists() else None
        put(directory/"capture.json", result)
    return result


def run(manifest, output, overall_seconds=3600):
    require(type(overall_seconds) is int and 210<=overall_seconds<=3600,
            "overall budget must admit one bounded run and never exceed 3600 seconds")
    output = outside_git(output)
    output.mkdir()
    put(output/"registration.json", {"manifest": manifest, "helper_sha256": sha256(__file__),
        "cache_reader_sha256": sha256(Path(__file__).with_name("shared_runtime_cache.py")),
        "owner_sha256": sha256(Path(__file__).with_name("bounded_local.py")), "positions": POSITIONS,
        "simulations_per_input": SIMULATIONS, "AA_pairs": 3, "AB_pairs": 5,
        "tree_max_edges": TREE_MAX_EDGES, "tree_max_nodes": 20_000, "tree_max_depth": 128,
        "search_command": SEARCH_COMMAND,
        "run_wall_seconds": 180, "cleanup_seconds": 30, "overall_seconds": overall_seconds,
        "cache_preparation_wall_seconds":60,"cache_preparation_cleanup_seconds":30,
        "cache_condition":"shared-runtime-warm-readhash_no_residency_guarantee",
        "primary_time": "whole_wall_start_through_native_exit_receipts_and_cgroup_collection",
        "primary_peak": "fresh_cgroup_memory.peak", "each_pair_time_and_peak_ratio_max": 1.05,
        "AA_variability_max": .05, "first_failure_stops": True, "automatic_retry": False})
    started = time.monotonic()
    deadline = started+overall_seconds
    pairs=[]
    reference_work=None
    error=None
    cache_preparation=None
    try:
        affinity=resource_affinity(manifest["resource_notes"])
        cache_preparation=prepare_cache(manifest["runtime_cache_preparation"],output,deadline,affinity)
        for index in range(8):
            comparison = "AA" if index<3 else "AB"
            order = ("baseline", "baseline") if comparison=="AA" else (("baseline", "candidate") if index%2 else ("candidate", "baseline"))
            results=[]
            for ordinal, role in enumerate(order):
                require(time.monotonic()+210 < deadline, "overall budget cannot admit another bounded run")
                result = run_once(manifest[role], output/f"{comparison}-{index:02}-{ordinal}-{role}", deadline,affinity)
                require(result["accepted"], result.get("error", "fixed work failed"))
                if reference_work is None:
                    reference_work=result["work"]
                require(result["work"] == reference_work, "fixed work/outcome differs across registered runs")
                results.append(result)
                print(json.dumps({"comparison": comparison, "pair": index, "role":role, "T":result["capture"]["whole_wall_seconds"], "P":result["capture"]["resources"]["memory.peak"]}), flush=True)
            if order[0]=="candidate":
                results.reverse()
            pair = pair_result(*results, comparison)
            pairs.append(pair)
            put(output/f"pair-{index:02}.json", pair)
            require(pair["passed"], "AA environment variability or AB per-pair regression exceeds five percent")
            if index==2:
                aa = [r for p in pairs for r in ((p["A_time"],p["A_peak"]),(p["B_time"],p["B_peak"]))]
                require(max(t for t,p in aa)/min(t for t,p in aa) <= 1.05 and max(p for t,p in aa)/min(p for t,p in aa) <= 1.05, "AA overall spread exceeds five percent; AB was not started")
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as exc:
        error=str(exc)
    result=summarize(pairs)
    if error:
        result.update(status="hold", error=error)
    result["overall_wall_seconds"] = time.monotonic()-started
    result["registration_sha256"] = sha256(output/"registration.json")
    result["candidate"] = {k: manifest["candidate"][k] for k in
        ("source_commit", "binary_sha256", "expected_profile")}
    result["resource_notes"] = manifest["resource_notes"]
    result["tree_max_edges"] = TREE_MAX_EDGES
    result["cache_preparation"] = cache_preparation
    result["memory_accounting_scope"] = "fresh_engine_cgroup_peak_after_shared_file_read_hash_preparation_not_whole_host_peak"
    put(output/"summary.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args=parser.parse_args()
    raise SystemExit(run(validate(read_json(args.manifest, 1024**2)), args.output))
