"""Read-only cumulative paired E evidence. Never launches or promotes an engine.

Raw measurements stay outside Git. A ledger references bounded, SHA-pinned JSON
results and declares compatible workload/peak definitions. Source epochs remain
visible even in descriptive cumulative totals. See benches/runtime/README.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

CAP = 16 * 1024 * 1024
COMPAT_KEYS = {
    "option", "workload", "model", "runtime", "precision", "batch", "resources",
    "fixed_work", "time_scope", "peak_kind", "other_options",
}


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric observation required")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("finite positive observation required")
    return value


def read_json(path, digest=None):
    path = Path(path)
    # A ledger cannot authorize reading secrets or model/library binaries.
    name = path.name.lower()
    if path.suffix.lower() != ".json" or name.startswith(".env") or any(
        word in name for word in ("credential", "service-account", "service_account", "api-key", "api_key", "ssh-key", "ssh_key")
    ):
        raise ValueError("only explicit non-secret JSON evidence is supported")
    if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
        raise ValueError("linked evidence is not supported")
    if not path.is_file():
        raise ValueError("regular JSON evidence required")
    with path.open("rb") as stream:
        payload = stream.read(CAP + 1)
    if len(payload) > CAP:
        raise ValueError("evidence exceeds bounded JSON limit")
    observed = hashlib.sha256(payload).hexdigest()
    if digest is not None and observed != digest:
        raise ValueError("evidence SHA mismatch")
    return json.loads(payload, object_pairs_hook=unique_object), observed


def pinned(root, ref):
    if set(ref) != {"path", "sha256"} or not re.fullmatch("[0-9a-f]{64}", ref["sha256"]):
        raise ValueError("explicit path and SHA required")
    return read_json(root / ref["path"], ref["sha256"])[0]


def events(text):
    return {key: int(value) for key, value in (line.split() for line in text.splitlines())}


def binding_fixed_work(report, sample, item, fixed_work):
    """Validate a single controlled B1 option before normalizing its work.

    Binding runs include a separate provider probe. They are not CUDA Graph
    capture/replay counts, and the two option families must keep distinct ledgers.
    Existing inference families retain their original compatibility fingerprints.
    """
    kind = report.get("kind")
    kinds = {"b1_io_binding_fixed_work", "b1_cuda_graph_option_fixed_work"}
    if kind not in kinds or report.get("schema_version") != 1:
        raise ValueError("registered B1 binding family required")
    for flag in ("io_binding", "cuda_graph"):
        if type(item.get(flag)) is not bool or report.get(flag) is not item[flag]:
            raise ValueError("registered B1 binding option differs")
    if ((kind == "b1_io_binding_fixed_work" and item["cuda_graph"] is not False)
        or (kind == "b1_cuda_graph_option_fixed_work" and item["io_binding"] is not True)):
        raise ValueError("only the registered binding or graph option may change")
    if (report.get("precision") != "fp32" or report.get("history_fill") != "no"
        or any(report.get(flag) is not False for flag in (
            "tf32", "cache", "dedup", "reuse_buffers", "cpu_ep_fallback", "disable_cuda_cpu_arena"))
        or any(report.get(flag, False) is not False for flag in ("zero_copy_ort", "copy_ort_model"))
        or sample.get("batch") != 1 or sample.get("warmup_nn_items") != 3):
        raise ValueError("independent fixed CUDA B1 work differs")
    positive(sample.get("cuda_placement", {}).get("executed_cuda_nodes"))
    for key in ("input_sha256", "output_sha256", "model_sha256", "export_manifest_sha256", "runtime_bundle_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", report.get(key, "")):
            raise ValueError("binding fixed-work identity missing")
    expected = {"provider_probe_binding_runs": 1, "warm_and_measured_binding_runs": 23,
                "successful_binding_runs": 24} if item["io_binding"] else {
                    "provider_probe_binding_runs": 0, "warm_and_measured_binding_runs": 0,
                    "successful_binding_runs": 0}
    if any(type(report.get(key)) is not int or report[key] != count for key, count in expected.items()):
        raise ValueError("provider probe and measured binding counts differ")
    if kind == "b1_cuda_graph_option_fixed_work" and report.get("capture_replay_observed") is not False:
        raise ValueError("graph option receipt cannot attest capture/replay")
    # Only the closed, validated option differs. Output/model/input identities
    # still have to match, and family identity prevents cross-family reuse.
    fixed_work.pop("io_binding")
    fixed_work.pop("cuda_graph")
    fixed_work.update(binding_family=kind, warmup_nn_items=3, provider_probe_separate=True)


def observation(root, item, peak_kind):
    result = pinned(root, item["result"])
    if result.get("accepted") is not True or result.get("exit_code") != 0:
        raise ValueError("failed or incomplete run cannot enter a comparison")
    if result.get("diagnostic_only") or result.get("performance_measurement") is False:
        raise ValueError("diagnostic or non-performance run cannot enter a comparison")
    if result.get("conditioning", False) or result.get("cancelled", False):
        raise ValueError("conditioning or cancelled run cannot enter a comparison")
    if result.get("forced_cgroup_cleanup") is not False:
        raise ValueError("normal owned cleanup must be confirmed")
    if result.get("marker_errors") or result.get("phase_memory_errors"):
        raise ValueError("incomplete measurement evidence")
    legacy = item["layout"] == "inference"
    if item["layout"] not in ("native", "prepare", "oracle", "inference"):
        raise ValueError("unknown evidence layout")
    remaining = "remaining_after_cleanup" if legacy else "remaining_owned_processes"
    if result.get(remaining) != [] or result.get("errors") or result.get("timed_out") or result.get("host_cancelled"):
        raise ValueError("owned exit, error and deadline checks must pass")
    resources = result["cgroup" if legacy else "resources"]
    counters = events(resources["memory.events"])
    if any(counters.get(key, 0) for key in ("max", "oom", "oom_kill", "oom_group_kill")):
        raise ValueError("allocation or OOM failure cannot be averaged away")
    time = positive(result["elapsed_s" if legacy else "whole_wall_seconds"])
    fixed_work = result.get("fixed_work")
    if legacy:
        report = pinned(root, item["report"])
        if report.get("accepted") is not True or report.get("failure") is not None:
            raise ValueError("inference report not accepted")
        if report["batches"] != [1] or report["warmup_runs"] != 3 or report["measured_runs"] != 20:
            raise ValueError("fixed B1 workload required")
        sample, = report["samples"]
        if sample["completed_nn_items"] != 20 or sample["physical_completion_confirmed"] is not True:
            raise ValueError("physical fixed-work completion required")
        fixed_work = {key: report[key] for key in (
            "input_sha256", "output_sha256", "model_sha256", "export_manifest_sha256",
            "runtime_bundle_sha256", "precision", "history_fill", "tf32", "cache", "dedup",
            "io_binding", "cuda_graph", "warmup_runs", "measured_runs",
        )}
        fixed_work["completed_nn_items"] = sample["completed_nn_items"]
        if peak_kind != "process_vm_hwm_bytes":
            raise ValueError("inference series must keep its registered RSS peak")
        match = re.fullmatch(r"([0-9]+) kB", report["memory"]["self_vm_hwm"])
        if match is None:
            raise ValueError("peak RSS was not observed")
        peak = positive(int(match[1]) * 1024)
        if report["reuse_buffers"] is not item["reuse_buffers"]:
            raise ValueError("registered buffer option differs")
        binding = (report.get("kind") in ("b1_io_binding_fixed_work", "b1_cuda_graph_option_fixed_work")
            or any(flag in item for flag in ("io_binding", "cuda_graph")))
        if binding:
            binding_fixed_work(report, sample, item, fixed_work)
        if ("disable_cuda_cpu_arena" in item
            and report.get("kind") in ("b1_owned_ort_fixed_work", "b1_copied_ort_fixed_work")):
            if item["disable_cuda_cpu_arena"] is not False:
                raise ValueError("owned ORT comparison requires its independent CPU arena flag off")
        elif "disable_cuda_cpu_arena" in item or report.get("kind") == "b1_cuda_cpu_arena_fixed_work":
            if (type(item.get("disable_cuda_cpu_arena")) is not bool
                or report.get("kind") != "b1_cuda_cpu_arena_fixed_work"
                or report.get("disable_cuda_cpu_arena") is not item["disable_cuda_cpu_arena"]
                or report.get("cpu_ep_fallback") is not False
                or report["reuse_buffers"] is not False):
                raise ValueError("registered CUDA CPU arena option or fallback differs")
        copied_ort = "copy_ort_model" in item or report.get("kind") == "b1_copied_ort_fixed_work"
        if copied_ort or "zero_copy_ort" in item or report.get("kind") == "b1_owned_ort_fixed_work":
            flag = "copy_ort_model" if copied_ort else "zero_copy_ort"
            kind = "b1_copied_ort_fixed_work" if copied_ort else "b1_owned_ort_fixed_work"
            if (type(item.get(flag)) is not bool
                or report.get("kind") != kind
                or report.get(flag) is not item[flag]
                or (copied_ort and (item.get("zero_copy_ort") is not False
                    or report.get("zero_copy_ort") is not False))
                or (not copied_ort and report.get("copy_ort_model", False) is not False)
                or report.get("cpu_ep_fallback") is not False
                or report.get("disable_cuda_cpu_arena") is not False
                or report["reuse_buffers"] is not False):
                raise ValueError("registered owned ORT option or other experiment differs")
            derived = pinned(root, item["derived_manifest"])
            if (derived["source_onnx_sha256"] != report["model_sha256"]
                or derived["source_export_manifest_sha256"] != report["export_manifest_sha256"]
                or derived["runtime_bundle_sha256"] != report["runtime_bundle_sha256"]
                or derived["runtime_version"] != "1.22.0"
                or derived["optimization_level"] != 1
                or derived["provider"] != "cuda" or derived["precision"] != "fp32"
                or derived["tf32"] is not False or derived["redistribution_ready"] is not False):
                raise ValueError("derived ORT provenance differs")
            expected_retained = derived["ort_bytes"] if not copied_ort and item[flag] else 0
            if report.get("retained_model_bytes") != expected_retained:
                raise ValueError("owned ORT lifetime/storage receipt differs")
            if item[flag] and (report.get("serialized_model_sha256") != derived["ort_sha256"]
                or report.get("derived_manifest_sha256") != item["derived_manifest"]["sha256"]):
                raise ValueError("actual ORT serialization identity differs")
    else:
        if peak_kind != "cgroup_memory_peak_bytes":
            raise ValueError("cgroup series must keep its registered whole-group peak")
        peak = positive(int(resources["memory.peak"]))
        if item["layout"] == "native":
            receipt = pinned(root, item["receipt"])
            if not receipt["integration_checks_passed"] or not receipt["cleanup_verified"] or receipt["unresolved_owner_retained"]:
                raise ValueError("production integration and physical cleanup evidence missing")
            if receipt["incomplete_games"] != 0 or len(receipt["provider_sessions"]) != 4 or not fixed_work:
                raise ValueError("four fresh native sessions and fixed work required")
    # Optional additional small evidence is verified without replacing the raw metrics.
    for ref in item.get("additional_evidence", []):
        pinned(root, ref)
    source = result.get("source_commit")
    if "registration" in item:
        registered_source = pinned(root, item["registration"])["source_commit"]
        if source is not None and source != registered_source:
            raise ValueError("raw source differs from pinned registration")
        source = registered_source
    if not source:
        raise ValueError("raw source or pinned source registration required")
    return dict(id=item["result"]["sha256"], time_seconds=time, peak_bytes=peak,
                source_commit=source, fixed_work=fixed_work,
                high_events=counters.get("high", 0))


def summarize(pairs):
    def totals(side):
        values = [pair[side] for pair in pairs]
        return dict(time_seconds=math.fsum(value["time_seconds"] for value in values),
                    peak_observation_sum_bytes=sum(value["peak_bytes"] for value in values),
                    mean_run_peak_bytes=statistics.mean(value["peak_bytes"] for value in values))
    baseline, variant = totals("baseline"), totals("variant")
    ratios = {key: [pair["variant"][field] / pair["baseline"][field] for pair in pairs]
              for key, field in (("time", "time_seconds"), ("peak", "peak_bytes"))}
    return dict(pairs=len(pairs), baseline=baseline, variant=variant,
                cumulative_time_ratio=variant["time_seconds"] / baseline["time_seconds"],
                cumulative_peak_ratio=variant["peak_observation_sum_bytes"] / baseline["peak_observation_sum_bytes"],
                paired_ratios={key: dict(min=min(values), median=statistics.median(values), max=max(values),
                    geometric_mean=math.exp(statistics.mean(math.log(v) for v in values)))
                    for key, values in ratios.items()})


def aggregate(root, ledger):
    if ledger["schema_version"] != 1 or not ledger["comparisons"]:
        raise ValueError("nonempty ledger schema 1 required")
    groups, ids, executions, used_in_series = {}, {}, {}, set()
    duplicates = 0
    for comparison in ledger["comparisons"]:
        identifier = comparison["id"]
        fingerprint = canonical(comparison)
        if identifier in ids:
            if ids[identifier] != fingerprint:
                raise ValueError("conflicting comparison ID")
            duplicates += 1
            continue
        ids[identifier] = fingerprint
        compat = comparison["compatibility"]
        if set(compat) != COMPAT_KEYS or any(v is None for v in compat.values()):
            raise ValueError("complete compatibility dimensions required")
        binding_comparison = (compat["option"] in ("io-binding", "cuda-graph")
            or any(flag in comparison[side] for flag in ("io_binding", "cuda_graph") for side in ("baseline", "variant")))
        if binding_comparison:
            option = compat["option"]
            flag = "io_binding" if option == "io-binding" else "cuda_graph"
            fixed_flag = "cuda_graph" if option == "io-binding" else "io_binding"
            expected_fixed = option == "cuda-graph"
            if (option not in ("io-binding", "cuda-graph")
                or comparison["baseline"].get(flag) is not False
                or comparison["variant"].get(flag) is not True
                or any(comparison[side].get(fixed_flag) is not expected_fixed
                    or comparison[side]["layout"] != "inference" for side in ("baseline", "variant"))):
                raise ValueError("binding comparison requires its single controlled baseline and variant")
        copied_comparison = (compat["option"] == "copied-ort-flatbuffer"
            or any("copy_ort_model" in comparison[side] for side in ("baseline", "variant")))
        ort_comparison = (copied_comparison or compat["option"] == "owned-ort-flatbuffer"
            or any("zero_copy_ort" in comparison[side] for side in ("baseline", "variant")))
        if copied_comparison:
            if (compat["option"] != "copied-ort-flatbuffer"
                or comparison["baseline"].get("copy_ort_model") is not False
                or comparison["variant"].get("copy_ort_model") is not True
                or any(comparison[side].get("zero_copy_ort") is not False
                    or comparison[side]["layout"] != "inference" for side in ("baseline", "variant"))):
                raise ValueError("copied ORT comparison requires original baseline and copied variant")
        elif ort_comparison:
            if (comparison["baseline"].get("zero_copy_ort") is not False
                or comparison["variant"].get("zero_copy_ort") is not True
                or any(comparison[side]["layout"] != "inference" for side in ("baseline", "variant"))):
                raise ValueError("owned ORT comparison requires original baseline and direct variant")
        if (compat["option"] == "cuda-cpu-arena-off"
            or (not ort_comparison
                and any("disable_cuda_cpu_arena" in comparison[side] for side in ("baseline", "variant")))):
            if (comparison["baseline"].get("disable_cuda_cpu_arena") is not False
                or comparison["variant"].get("disable_cuda_cpu_arena") is not True
                or any(comparison[side]["layout"] != "inference" for side in ("baseline", "variant"))):
                raise ValueError("CUDA CPU arena comparison requires baseline on and variant off")
        group_id = hashlib.sha256(canonical(compat).encode()).hexdigest()
        pair = {side: observation(root, comparison[side], compat["peak_kind"])
                for side in ("baseline", "variant")}
        if pair["baseline"]["id"] == pair["variant"]["id"]:
            raise ValueError("same physical run cannot be both comparison arms")
        if pair["baseline"]["fixed_work"] != pair["variant"]["fixed_work"]:
            raise ValueError("paired completed work or output differs")
        epoch = canonical(dict(environment=comparison["environment_epoch"],
                              sources=[pair[side]["source_commit"] for side in ("baseline", "variant")]))
        for value in pair.values():
            key = (group_id, value["id"])
            if key in used_in_series:
                raise ValueError("physical run reused inside the same option series")
            used_in_series.add(key)
            if value["id"] in executions and executions[value["id"]] != value:
                raise ValueError("conflicting physical run identity")
            executions[value["id"]] = value
        group = groups.setdefault(group_id, dict(compatibility=compat, pairs=[], epochs={}))
        group["pairs"].append(pair)
        group["epochs"].setdefault(epoch, []).append(pair)
    output = []
    for key, group in groups.items():
        output.append(dict(series_id=key, compatibility=group["compatibility"],
            cumulative=summarize(group["pairs"]),
            epochs=[dict(identity=json.loads(epoch), **summarize(pairs)) for epoch, pairs in group["epochs"].items()]))
    excluded, excluded_ids = [], {}
    for item in ledger.get("excluded", []):
        if not item["reason"]:
            raise ValueError("exclusion requires a preserved reason")
        fingerprint = canonical(item)
        if item["id"] in excluded_ids:
            if excluded_ids[item["id"]] != fingerprint:
                raise ValueError("conflicting exclusion ID")
            continue
        for ref in item["evidence"]:
            pinned(root, ref)
        excluded_ids[item["id"]] = fingerprint
        excluded.append(item)
    return dict(schema_version=1, series=output, duplicate_comparisons_skipped=duplicates,
        unique_compared_physical_runs=len(executions), excluded=excluded,
        shared_control_occurrences=2 * len(ids) - len(executions),
        default_changed=False, promotion_decision="not_made_by_aggregation",
        limitations=["peak sums describe independent run observations, not simultaneous/system peak",
            "cumulative ratios are descriptive; source/environment epochs remain separately visible",
            "shared controls across options are correlated, not independent repetitions",
            "conditioning and failures stay visible; no RSS/cgroup/VRAM substitution",
            "does not launch workloads, alter prior judgments or infer Elo/confidence"])


def combine_ledgers(paths):
    combined = dict(schema_version=1, comparisons=[], excluded=[])
    inputs = []
    for path in paths:
        path = Path(path).absolute()
        ledger, digest = read_json(path)
        if ledger["schema_version"] != 1:
            raise ValueError("unsupported included ledger")
        value = copy.deepcopy(ledger)
        def rebase(ref):
            ref["path"] = str((path.parent / ref["path"]).absolute())
        for comparison in value["comparisons"]:
            for side in ("baseline", "variant"):
                arm = comparison[side]
                for key in ("result", "report", "receipt", "registration", "derived_manifest"):
                    if key in arm:
                        rebase(arm[key])
                for ref in arm.get("additional_evidence", []):
                    rebase(ref)
        for item in value.get("excluded", []):
            for ref in item["evidence"]:
                rebase(ref)
        combined["comparisons"].extend(value["comparisons"])
        combined["excluded"].extend(value.get("excluded", []))
        inputs.append(dict(path=str(path), sha256=digest))
    return combined, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("report", type=Path, help="fresh JSON report outside Git")
    parser.add_argument("--include-ledger", type=Path, action="append", default=[],
                        help="preserved earlier ledger; repeat to add more epochs")
    args = parser.parse_args()
    ledger, inputs = combine_ledgers([*args.include_ledger, args.ledger])
    result = aggregate(args.ledger.parent, ledger)
    result["input_ledgers"] = inputs
    if any((parent / ".git").exists() for parent in args.report.absolute().parents):
        raise ValueError("report must be outside Git")
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("unique_compared_physical_runs", "shared_control_occurrences", "duplicate_comparisons_skipped")}))


if __name__ == "__main__":
    main()
