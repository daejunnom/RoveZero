"""Bounded offline GQA node/profile accounting; no model or device execution.

Inputs are explicit serialized graph inventory and ORT ChromeTrace profile JSON.
Node event durations are microseconds in that profile's scope. They never prove
physical allocation/copy bytes, isolated CUDA kernel time or active-engine cost.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys

MAX_INPUT_BYTES = 32 * 1024 * 1024  # Combined inventory + profile, not per file.
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_NODES = 5000
MAX_EVENTS = 100_000
MAX_DATAFLOW_STEPS = 2_000_000
PROVIDERS = {"CPUExecutionProvider", "CUDAExecutionProvider"}
FUSED_ATTENTION = {"Attention", "MultiHeadAttention", "GroupQueryAttention", "GroupedQueryAttention"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def checked_path(path, output=False):
    path = Path(path)
    require(path.is_absolute() and len(path.parts) <= 64, "bounded absolute artifact path required")
    for part in (path, *path.parents):
        lower = part.name.lower()
        require(not (lower == ".env" or lower.startswith(".env.") or "credential" in lower
                     or "service-account" in lower or "service_account" in lower
                     or lower.startswith(("id_ed25519", "id_rsa"))), "secret path is outside scope")
        require(not part.is_symlink(), "linked artifact paths unsupported")
        if hasattr(part, "is_junction"):
            require(not part.is_junction(), "linked artifact paths unsupported")
        if output:
            require(not (part / ".git").exists(), "report must stay outside Git")
    return path


def read_inputs(inventory_path, profile_path):
    result, total = [], 0
    for path in (inventory_path, profile_path):
        path = checked_path(path)
        named = path.stat()
        require(stat.S_ISREG(named.st_mode), "input must be a regular file")
        require(named.st_size <= MAX_INPUT_BYTES - total, "combined inputs exceed 32 MiB")
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES - total + 1)
        total += len(raw)
        require(total <= MAX_INPUT_BYTES, "combined inputs grew beyond 32 MiB")
        result.append((decode(raw), {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}))
    return result


def text(value, label):
    require(isinstance(value, str) and 0 < len(value) <= 512, f"invalid {label}")
    return value


def inventory_nodes(inventory, graph_role, graph_sha256):
    require(isinstance(inventory, dict) and isinstance(inventory.get("graphs"), list), "graph inventory required")
    require(1 <= len(inventory["graphs"]) <= 16, "graph inventory bound exceeded")
    require(all(isinstance(graph, dict) for graph in inventory["graphs"]), "invalid graph inventory entry")
    selected = [graph for graph in inventory["graphs"] if graph.get("role") == graph_role]
    require(len(selected) == 1, "selected graph role absent or duplicated")
    graph = selected[0]
    require(graph.get("sha256") == graph_sha256, "selected serialized graph pin differs")
    nodes, pending, scopes = {}, [(graph.get("inventory"), 0)], set()
    while pending:
        scope, depth = pending.pop()
        require(isinstance(scope, dict) and depth <= 8, "invalid or unbounded graph scope")
        path = text(scope.get("path"), "scope name")
        require(path not in scopes and len(scopes) < MAX_NODES, "duplicate or unbounded graph scopes")
        scopes.add(path)
        require(isinstance(scope.get("nodes"), list), "scope nodes required")
        for node in scope["nodes"]:
            require(isinstance(node, dict), "invalid inventory node")
            name, op = text(node.get("name"), "node name"), text(node.get("op"), "opcode")
            require(name not in nodes, "ambiguous duplicate node name across scopes")
            for field in ("inputs", "outputs"):
                require(isinstance(node.get(field), list) and len(node[field]) <= 32, "invalid node ports")
                for value in node[field]:
                    text(value, "tensor name")
            require(node["outputs"], "inventory node has no outputs")
            nodes[name] = {**node, "_scope": path, "op": op}
            require(len(nodes) <= MAX_NODES, "selected graph exceeds 5000 nodes")
        children = scope.get("subgraphs", [])
        require(isinstance(children, list) and len(children) <= 32, "invalid subgraph bound")
        pending.extend((child, depth + 1) for child in children)
    return nodes


def indexes(nodes):
    producers, consumers = {}, defaultdict(list)
    for node in nodes.values():
        for output in node["outputs"]:
            key = (node["_scope"], output)
            require(key not in producers, "ambiguous duplicate tensor producer")
            producers[key] = node
        for axis, value in enumerate(node["inputs"]):
            consumers[(node["_scope"], value)].append((node, axis))
    return producers, consumers


def profile_events(profile):
    require(isinstance(profile, list) and len(profile) <= MAX_EVENTS, "bounded ORT profile event list required")
    events, other = defaultdict(list), []
    for index, event in enumerate(profile):
        require(isinstance(event, dict), "invalid profile event")
        if event.get("cat") != "Node" or not str(event.get("name", "")).endswith("_kernel_time"):
            continue
        name = text(event["name"][:-len("_kernel_time")], "profile node name")
        args = event.get("args", {})
        require(isinstance(args, dict), "invalid profile node arguments")
        duration = event.get("dur") if event.get("ph", "X") == "X" else None
        if duration is not None:
            require(type(duration) in (int, float) and math.isfinite(duration) and 0 <= duration <= 10**12,
                    "invalid node event duration")
        record = {"node": name, "op": args.get("op_name"), "provider": args.get("provider"),
                  "duration_us": duration, "output_type_shape": args.get("output_type_shape"), "index": index}
        events[name].append(record)
        other.append(record)
    return events, other


def event_stats(events):
    durations = [event["duration_us"] for event in events if event["duration_us"] is not None]
    providers = sorted({str(event.get("provider", "unknown")) for event in events})
    return {"events": len(events), "providers": providers,
            "duration_complete": bool(events) and len(durations) == len(events),
            "duration_sum_us": sum(durations) if events and len(durations) == len(events) else None,
            "known_duration_sum_us": sum(durations), "unknown_duration_events": len(events) - len(durations),
            "known_event_duration_min_us": min(durations) if durations else None,
            "known_event_duration_max_us": max(durations) if durations else None}


def float_shape(events):
    shapes = set()
    for event in events:
        description = event.get("output_type_shape")
        if not isinstance(description, list) or len(description) != 1 or not isinstance(description[0], dict):
            return None
        shape = description[0].get("float")
        if not isinstance(shape, list) or not 1 <= len(shape) <= 5:
            return None
        if any(type(dim) is not int or not 1 <= dim <= 16_384 for dim in shape):
            return None
        shapes.add(tuple(shape))
    return list(next(iter(shapes))) if len(shapes) == 1 else None


class DataflowBudget:
    def __init__(self):
        self.steps = 0

    def visit(self):
        self.steps += 1
        require(self.steps <= MAX_DATAFLOW_STEPS, "dataflow traversal budget exceeded")


def softmax_link(value, scope, producers, consumers, budget, upstream):
    pending, seen, found = deque([(value, 0)]), set(), {}
    while pending:
        tensor, depth = pending.popleft()
        if tensor in seen or depth > 5:
            continue
        seen.add(tensor)
        edges = [(producers.get((scope, tensor)), 0)] if upstream else consumers.get((scope, tensor), [])
        for node, axis in edges:
            budget.visit()
            if node is None:
                continue
            if node["op"] == "Softmax" and axis == 0:
                found[node["name"]] = node
            allowed = {"Identity", "Cast", "Transpose", "Mul"} if upstream else {"Identity", "Cast", "Mul", "Add", "Where"}
            # Where data inputs are 1/2; input 0 is the condition.
            if node["op"] in allowed and (upstream or node["op"] != "Where" or axis in (1, 2)):
                next_values = node["inputs"] if upstream else node["outputs"]
                pending.extend((port, depth + 1) for port in next_values)
    # Ambiguous dataflow cannot identify an attention endpoint.
    return next(iter(found.values())) if len(found) == 1 else None


def attention_endpoints(value, scope, producers, consumers, budget):
    pending, seen, result = deque([(value, 0)]), set(), {}
    while pending:
        tensor, depth = pending.popleft()
        if tensor in seen or depth > 3:
            continue
        seen.add(tensor)
        for node, axis in consumers.get((scope, tensor), []):
            budget.visit()
            if node["op"] == "MatMul" and axis == 1 and len(node["inputs"]) == 2:
                left = softmax_link(node["inputs"][0], scope, producers, consumers, budget, True)
                right = softmax_link(node["outputs"][0], scope, producers, consumers, budget, False)
                if bool(left) != bool(right):
                    kind, softmax = ("value", left) if left else ("key", right)
                    result[(kind, node["name"], softmax["name"])] = (kind, node, softmax)
            elif node["op"] in {"Identity", "Transpose"} and axis == 0:
                pending.extend((output, depth + 1) for output in node["outputs"])
    return list(result.values())


def repeat_groups(nodes, producers, consumers, budget):
    groups = []
    for repeat in nodes.values():
        if repeat["op"] not in {"Expand", "Tile"} or not repeat["inputs"]:
            continue
        scope = repeat["_scope"]
        before = producers.get((scope, repeat["inputs"][0]))
        if not before or before["op"] != "Unsqueeze" or not before["inputs"]:
            continue
        after = [node for node, axis in consumers.get((scope, repeat["outputs"][0]), [])
                 if axis == 0 and node["op"] == "Reshape"]
        if len(after) != 1:
            continue
        ends = attention_endpoints(after[0]["outputs"][0], scope, producers, consumers, budget)
        if len(ends) != 1:
            continue
        kind, matmul, softmax = ends[0]
        groups.append({"kind": kind, "scope": scope, "nodes": [before["name"], repeat["name"], after[0]["name"]],
                       "input_value": before["inputs"][0], "repeat_node": repeat["name"],
                       "reshape_node": after[0]["name"], "attention_nodes": [matmul["name"], softmax["name"]]})
    return groups


def dimensions(group, events, context):
    shape = float_shape(events.get(group["reshape_node"], []))
    before = float_shape(events.get(group["nodes"][0], []))
    expanded = float_shape(events.get(group["repeat_node"], []))
    if not shape or len(shape) != 4 or shape[1] != 6 or shape[3] != 64:
        return {"status": "unknown", "reason": "missing_or_nonregistered_observed_output_shape", "output_shape": shape}
    b, _, tokens, d = shape
    if before != [b, 2, 1, tokens, d] or expanded != [b, 2, 3, tokens, d]:
        return {"status": "unknown", "reason": "repeat_path_shapes_do_not_prove_registered_2_to_6_head_layout",
                "output_shape": shape, "before_shape": before, "expanded_shape": expanded}
    batch, records, memory = context["batch"], context["records"], context["memory_tokens"]
    if context["graph"] == "public" and (b, tokens) == (batch, 66):
        scope = "public_board_self_attention"
    elif context["graph"] == "public" and (b, tokens) == (batch * max(records, 1), 4):
        scope = "public_record_field_self_attention"
    elif context["graph"] != "public" and (b, tokens) == (batch, memory):
        scope = "private_cross_public_memory"
    elif context["graph"] != "public" and (b, tokens) == (batch, 16):
        scope = "private_latent_self_attention"
    else:
        return {"status": "unknown", "reason": "observed_shape_differs_from_declared_B_R_S", "output_shape": shape}
    original = b * 2 * tokens * d * 4
    logical = b * 6 * tokens * d * 4
    return {"status": "matched", "attention_scope": scope, "observed_output_shape": shape,
            "logical_original_fp32_bytes": original, "logical_expanded_fp32_bytes": logical,
            "logical_additional_fp32_bytes": logical - original,
            "physical_copy_or_allocation_bytes": "unknown"}


def validate_context(context):
    require(isinstance(context, dict), "invocation context must be an object")
    require(context.get("graph") in {"public", "shared_pc", "shared_pc_warm", "proposer", "critic"}, "unsupported graph role")
    require(context.get("role") in {"public", "proposer", "critic"}, "unsupported invocation role")
    require((context["graph"] == "public") == (context["role"] == "public"), "graph/invocation role differs")
    require(context["graph"] not in {"proposer", "critic"} or context["graph"] == context["role"], "separate graph/invocation role differs")
    for key, low, high in (("batch", 1, 16), ("records", 0, 128), ("memory_tokens", 67, 194)):
        require(type(context.get(key)) is int and low <= context[key] <= high, "unbounded B/R/S context")
    require(context["memory_tokens"] == 66 + max(context["records"], 1), "registered memory token context differs")
    require(isinstance(context.get("graph_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", context["graph_sha256"]), "graph SHA-256 required")


def analyze(inventory, profile, context):
    validate_context(context)
    nodes = inventory_nodes(inventory, context["graph"], context["graph_sha256"])
    producers, consumers = indexes(nodes)
    events, all_events = profile_events(profile)
    budget = DataflowBudget()
    groups = repeat_groups(nodes, producers, consumers, budget)
    observed_roles = sorted({role for role in ("proposer", "critic")
                             if any(name.startswith(role + "_") and name in nodes
                                    and any(event["op"] == nodes[name]["op"] for event in events[name])
                                    for name in events)})
    role = "public" if context["graph"] == "public" else (observed_roles[0] if len(observed_roles) == 1 else "unknown")
    rows, attention_nodes = [], set()
    for group in groups:
        matched = [event for name in group["nodes"] for event in events.get(name, [])
                   if event["op"] == nodes[name]["op"]]
        missing = [name for name in group["nodes"] if not events.get(name)]
        mismatched = [name for name in group["nodes"] if any(event["op"] != nodes[name]["op"] for event in events.get(name, []))]
        stats = event_stats(matched)
        shape = dimensions(group, events, context)
        per_node_counts = {name: len(events.get(name, [])) for name in group["nodes"]}
        consistent_counts = len(set(per_node_counts.values())) == 1 and bool(matched)
        status = "matched" if not missing and not mismatched and consistent_counts and shape["status"] == "matched" and stats["duration_complete"] and set(stats["providers"]).issubset(PROVIDERS) else "unknown"
        attention_nodes.update(group["attention_nodes"])
        rows.append({**group, "status": status, "role": role, "declared_role": context["role"],
                     "missing_nodes": missing, "opcode_mismatch_nodes": mismatched,
                     "observed_event_duration": stats, "dimensions": shape,
                     "observed_node_event_counts": per_node_counts, "node_event_counts_equal": consistent_counts,
                     "duration_aggregation": "all_matching_events_in_profile_window;not_per_invocation_wall_time",
                     "repeat_path_duration_us": stats["duration_sum_us"] if status == "matched" else None,
                     "provider_scope": "cuda_profile_events" if stats["providers"] == ["CUDAExecutionProvider"]
                     else "cpu_diagnostic_events" if stats["providers"] == ["CPUExecutionProvider"] else "mixed_or_unknown_events"})
    fused = []
    for name, node in nodes.items():
        # A fused node is attributable only when its concrete inventory name/op
        # and both explicit public KV input edges match. Its time is combined
        # attention time; the implicit KV repetition component stays unknown.
        if node["op"] in FUSED_ATTENTION and {"memory_key", "memory_value"}.issubset(node["inputs"]):
            selected = [event for event in events.get(name, []) if event["op"] == node["op"]]
            if selected:
                fused.append({"status": "fused", "node": name, "op": node["op"], "scope": node["_scope"],
                              "role": role, "combined_attention_duration": event_stats(selected),
                              "repeat_kv_duration_us": None, "logical_expanded_bytes": None,
                              "physical_copy_or_allocation_bytes": "unknown"})
    attention = [event for name in sorted(attention_nodes) for event in events.get(name, []) if event["op"] == nodes[name]["op"]]
    transfers = [event for event in all_events if str(event["op"]).startswith("Memcpy")]
    unknown = [{"node": event["node"], "op": event["op"], "provider": event["provider"],
                "possible_fused": "fused" in str(event["op"]).lower() or event["op"] in FUSED_ATTENTION}
               for event in all_events if event["node"] not in nodes or event["op"] != nodes[event["node"]]["op"]]
    return {"schema": "rz-pals-gqa-profile-cost-v1", "status": "analyzed" if rows or fused else "unknown",
            "context": context, "observed_private_roles": observed_roles, "attributed_role": role,
            "role_context_matches_observation": role == context["role"] if role != "unknown" else None,
            "role_attribution": "aggregate_profile_has_multiple_or_no_private_roles" if role == "unknown" else "profile_node_prefix_observation",
            "repeat_kv_groups": rows, "explicit_fused_attention": fused,
            "matching_summary": {"matched": sum(row["status"] == "matched" for row in rows),
                                 "unknown": sum(row["status"] == "unknown" for row in rows), "fused": len(fused)},
            "attention_score_value_matmul_and_softmax": event_stats(attention),
            "attention_aggregation": "unique_related_node_names;each_matching_event_counted_once;projection_ffn_and_mask_ops_excluded",
            "memcpy_profile_events": event_stats(transfers), "physical_transfer_bytes": "unknown",
            "unattributed_runtime_nodes": unknown[:256], "unattributed_runtime_event_count": len(unknown),
            "unknown_record_limit": 256, "parser_nn_execution": False, "parser_gpu_execution": False,
            "bounds": {"combined_input_bytes": MAX_INPUT_BYTES, "nodes": MAX_NODES, "events": MAX_EVENTS,
                       "dataflow_steps": MAX_DATAFLOW_STEPS, "observed_dataflow_steps": budget.steps},
            "required_supervision": "one_process;at_most_two_CPU_IDs;60_second_wall_timeout",
            "profile_graph_binding": "explicit_input_pairing_and_name_op_dataflow_match;runtime_graph_pin_not_independently_attested",
            "duration_scope": "ORT_Node_ChromeTrace_kernel_time_event_microseconds;not_isolated_CUDA_kernel_stopwatch;node_sums_may_overlap",
            "graph_run_wall_time": "unknown;requires_separate_Run_and_physical_fence_receipt",
            "isolated_gpu_repeat_time": "unknown", "active_engine_bottleneck_claim": False,
            "missing_or_fused_nodes": "no_imputation;absent_nodes_do_not_mean_zero_cost"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--records", required=True, type=int)
    parser.add_argument("--memory-tokens", required=True, type=int)
    args = parser.parse_args(argv)
    output = checked_path(args.output, output=True)
    require(not output.exists(), "output must be a new report")
    loaded = read_inputs(args.inventory, args.profile)
    context = {key: getattr(args, key) for key in ("graph", "graph_sha256", "role", "batch", "records", "memory_tokens")}
    report = analyze(loaded[0][0], loaded[1][0], context)
    report["inputs"] = {"inventory": loaded[0][1], "profile": loaded[1][1]}
    report["parser_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(encoded) <= MAX_OUTPUT_BYTES, "output exceeds 4 MiB")
    with output.open("xb") as stream:
        stream.write(encoded)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RecursionError) as error:
        print(f"pals_gqa_cost: {error}", file=sys.stderr)
        raise SystemExit(1)
