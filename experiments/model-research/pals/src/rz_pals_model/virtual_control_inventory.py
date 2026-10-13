"""Opt-in V3 serialized control inventory; never starts an NN session.

Only exact manifest/graph bytes are read. Type/shape metadata and lexical
origins are recorded without reading Tensor values. An annotation is a static
candidate, never CPU permission or provider placement evidence. The separate
Rust V3 policy revalidates provenance and owns pre-Run placement admission.
The legacy producer, schema and profile registration are unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import cuda_control_inventory as legacy
from .config import ModelConfig, MODEL_SCHEMA_V3, VIRTUAL_REVIEW_PROFILE
from .onnx_virtual_review import PUBLIC_LAYOUT, SHARED_LAYOUT, VALIDATOR_LAYOUT, WARM_LAYOUT, WARM_MODE_NODE

SCHEMA = "rovezero.pals-virtual-static-control-inventory.v1"
SCOPE = "read_only_serialized_graph_metadata_no_runtime"
MAX_MANIFEST_BYTES = 256 * 1024
MAX_GRAPH_BYTES = legacy.MAX_GRAPH_BYTES
MAX_TOTAL_GRAPH_BYTES = 512 * 1024 * 1024
MAX_INVENTORY_BYTES = 4 * 1024 * 1024
MAX_NODES = 12000
MAX_GRAPH_NODES = 5000
MAX_INITIALIZERS_PER_SCOPE = legacy.MAX_INITIALIZERS_PER_SCOPE
MAX_ATTRIBUTES_PER_NODE = legacy.MAX_ATTRIBUTES_PER_NODE
MAX_RECURSIVE_DEPTH = legacy.MAX_RECURSIVE_DEPTH
MAX_PATH_BYTES = legacy.MAX_PATH_BYTES
DATA, ROLE, META, INTEGRAL = legacy.DATA, legacy.ROLE, legacy.META, legacy.INTEGRAL
WARM = "virtual_warm_mode_control"
WARM_ANNOTATION = "virtual_warm_if_dispatch_candidate"
FILES = {"public": "public_memory.onnx", "shared_pc": "shared_pc_if.onnx",
         "validator": "role_validator.onnx", "shared_warm": "shared_pc_if_approx_warm.onnx"}
GRAPH_NAMES = {"public": PUBLIC_LAYOUT, "shared_pc": SHARED_LAYOUT,
               "validator": VALIDATOR_LAYOUT, "shared_warm": WARM_LAYOUT}


def _manifest(raw):
    def nonfinite(_):
        raise ValueError("nonfinite_json_refused")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=legacy._unique_object, parse_constant=nonfinite)
    if type(value) is not dict or value.get("schema") != MODEL_SCHEMA_V3:
        raise ValueError("virtual_v3_export_manifest_required")
    if type(value.get("config")) is not dict:
        raise ValueError("registered_virtual_config_required")
    config = ModelConfig(**value["config"])
    config.validate()
    if config.profile != VIRTUAL_REVIEW_PROFILE or not legacy._sha(value.get("checkpoint_sha256")):
        raise ValueError("registered_virtual_profile_and_checkpoint_pin_required")
    if type(value.get("graphs")) is not list or not value["graphs"] or any(type(entry) is not dict for entry in value["graphs"]):
        raise ValueError("registered_virtual_graph_descriptors_required")
    warm = value.get("layout") == WARM_LAYOUT
    metadata = {key: value.get(key) for key in ("config", "checkpoint_sha256", "trained", "training_steps")}
    # These existing validators are dependency-free. They check actual V3
    # Rules/config/I-O declarations, not checkpoint contents or NN behavior.
    from .artifacts import validate_virtual_export_manifest, validate_virtual_warm_export_manifest
    if warm:
        validate_virtual_warm_export_manifest(value, metadata)
    else:
        validate_virtual_export_manifest(value, metadata)
    expected_roles = ["public", "shared_warm"] if warm else ["public", "shared_pc"] + (["validator"] if value["validator_present"] else [])
    if [entry.get("role") for entry in value["graphs"]] != expected_roles:
        raise ValueError("virtual_graph_role_set_or_order_differs")
    if any(entry.get("file") != FILES[entry["role"]] for entry in value["graphs"]):
        raise ValueError("virtual_graph_file_is_not_registered")
    return value, warm


class _State:
    def __init__(self, role, warm, deadline):
        if role not in GRAPH_NAMES or type(warm) is not bool:
            raise ValueError("unregistered_virtual_inventory_role")
        self.role, self.warm, self.deadline = role, warm, deadline
        self.count, self.graph_count, self.names = 0, 0, set()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise ValueError("inventory_wall_deadline_exceeded")


def _scope(graph, path, inherited, state, depth=0):
    """Same legacy JSON field meanings, with an explicit V3 Warm route."""
    state.check()
    if depth > MAX_RECURSIVE_DEPTH or len(path.encode("utf-8")) > MAX_PATH_BYTES:
        raise ValueError("recursive_metadata_bound_exceeded")
    inputs, outputs = legacy._descriptors(graph.input), legacy._descriptors(graph.output)
    origins = dict(inherited)
    for descriptor in inputs:
        origin = DATA
        if descriptor["name"] == "role_is_critic":
            if state.role not in ("shared_pc", "shared_warm") or descriptor["dtype"] != "BOOL" or descriptor["shape"] != []:
                raise ValueError("registered_role_control_is_not_scalar_bool")
            origin = ROLE
        elif descriptor["name"] == "warm_start":
            if not state.warm or state.role != "shared_warm" or path != "shared_warm" or descriptor["dtype"] != "BOOL" or descriptor["shape"] != []:
                raise ValueError("registered_virtual_warm_control_is_not_scalar_bool")
            origin = WARM
        if descriptor["name"] in origins:
            raise ValueError("nested_input_shadows_outer_value")
        origins[descriptor["name"]] = origin
    if graph.sparse_initializer or len(graph.initializer) > MAX_INITIALIZERS_PER_SCOPE:
        raise ValueError("sparse_or_unbounded_initializer_metadata")
    seeds = []
    for tensor in graph.initializer:
        name = legacy._name(tensor.name)
        if name in origins:
            raise ValueError("initializer_shadows_value")
        seed = legacy._tensor_seed(tensor, name, "graph_initializer_metadata")
        if seed is not None:
            seeds.append(seed)
        origins[name] = INTEGRAL if seed else DATA
    nodes, children = [], []
    for index, node in enumerate(graph.node):
        state.check()
        state.count += 1
        state.graph_count += 1
        name = legacy._name(node.name)
        if (state.count > MAX_NODES or state.graph_count > MAX_GRAPH_NODES or name in state.names or node.domain not in ("", "ai.onnx")
                or getattr(node, "overload", "") or len(node.input) > 64 or not 0 < len(node.output) <= 16):
            raise ValueError("ambiguous_or_unsupported_serialized_node")
        state.names.add(name)
        attrs = [attribute.name for attribute in node.attribute]
        if len(attrs) > MAX_ATTRIBUTES_PER_NODE or len(attrs) != len(set(attrs)) or any(not legacy._name(attr) for attr in attrs):
            raise ValueError("duplicate_or_invalid_node_attribute")
        legacy._name(node.op_type)
        for input_name in node.input:
            if input_name:
                legacy._name(input_name)
        actual_origins = [origins.get(value, DATA) for value in node.input]
        provenance = [origin if value else "absent_optional_input" for value, origin in zip(node.input, actual_origins)]
        annotation, output_origin, constant_seeds = "none", DATA, []
        for attribute in node.attribute:
            if attribute.type == 4:
                seed_name = node.output[0] if node.op_type == "Constant" and len(node.output) == 1 else name
                seed = legacy._tensor_seed(attribute.t, seed_name, "constant_node_tensor_metadata")
                if node.op_type == "Constant" and attribute.name == "value" and seed is not None:
                    constant_seeds.append(seed)
            elif attribute.type in (9, 11, 12):
                raise ValueError("repeated_or_sparse_tensor_attribute_is_not_registered")
            elif attribute.type in (5, 10) and not (node.op_type == "If" and attribute.type == 5):
                raise ValueError("unregistered_nested_graph_attribute")
        if constant_seeds:
            if len(constant_seeds) != len(node.output):
                raise ValueError("ambiguous_integral_constant")
            output_origin = INTEGRAL
        if node.op_type in ("Shape", "Size") and len(node.input) == len(node.output) == 1:
            annotation, output_origin = "shape_metadata_candidate", META
        elif (node.op_type in legacy.METADATA_CHAIN and actual_origins and META in actual_origins
              and all(origin in (META, INTEGRAL) for origin in actual_origins)):
            annotation, output_origin = "shape_metadata_chain_candidate", META
        elif node.op_type == "Identity" and list(node.input) == ["role_is_critic"] and actual_origins == [ROLE]:
            annotation, output_origin = "role_scalar_identity_candidate", ROLE
        elif node.op_type in legacy.MAJOR:
            annotation = "major_nn_requires_cuda"
        # Range is deliberately not a metadata-control chain. Dtype/shape alone
        # does not prove start=0, step=1, or its runtime output bound. The current
        # hand-written dynamic-L graph uses Slice; float ply-embedding Slice is
        # DATA and gains no CPU authorization. No Tensor values are inspected.
        if node.op_type == "If":
            if set(attrs) != {"then_branch", "else_branch"} or len(node.input) != 1:
                raise ValueError("exact_two_branch_if_required")
            if list(node.input) == ["role_is_critic"] and actual_origins == [ROLE]:
                annotation = "role_if_dispatch_candidate"
            elif (state.warm and path == "shared_warm" and index == 0 and name == WARM_MODE_NODE
                  and list(node.input) == ["warm_start"] and actual_origins == [WARM]):
                annotation = WARM_ANNOTATION
            else:
                raise ValueError("arbitrary_bool_or_data_if_refused")
            for attribute in node.attribute:
                if attribute.type != 5:
                    raise ValueError("if_attribute_is_not_graph")
                child_path = f"{path}/node{index}:{name}/{attribute.name}"
                children.append(_scope(attribute.g, child_path, origins, state, depth + 1))
        nodes.append({"index": index, "name": name, "domain": node.domain, "op": node.op_type,
                      "inputs": list(node.input), "outputs": list(node.output), "input_origins": provenance,
                      "constant_integral_seeds": constant_seeds, "static_annotation": annotation,
                      "actual_provider": "unknown", "attribute_names": sorted(attrs)})
        for output in node.output:
            if legacy._name(output) in origins:
                raise ValueError("node_output_aliases_outer_value")
            origins[output] = output_origin
    return {"path": path, "inputs": inputs, "outputs": outputs, "integral_initializers": seeds,
            "initializer_count": len(graph.initializer), "nodes": nodes, "subgraphs": children}


def _children(scope, node):
    prefix = f"{scope['path']}/node{node['index']}:{node['name']}/"
    branches = {child["path"][len(prefix):]: child for child in scope["subgraphs"] if child["path"].startswith(prefix)}
    if set(branches) != {"then_branch", "else_branch"}:
        raise ValueError("exact_virtual_if_branch_paths_required")
    return branches


def _role_route(scope, node, kind):
    if (node["op"] != "If" or node["inputs"] != ["role_is_critic"]
            or node["input_origins"] != [ROLE] or node["static_annotation"] != "role_if_dispatch_candidate"
            or not re.fullmatch("route_private_" + kind + r"_[1-9][0-9]*", node["name"])):
        raise ValueError("unregistered_virtual_private_role_route")
    branches = _children(scope, node)
    for branch_name, expert in (("then_branch", "critic"), ("else_branch", "proposer")):
        branch = branches[branch_name]
        if branch["inputs"] or branch["initializer_count"] or branch["subgraphs"]:
            raise ValueError("virtual_role_branch_must_capture_single_outer_bank")
        referenced = [name for item in branch["nodes"] for name in item["inputs"] if name.startswith("experts.")]
        if not referenced or any(not name.startswith("experts." + expert + ".") for name in referenced):
            raise ValueError("virtual_role_branch_private_bank_differs")


def _validate_routes(role, scope, warm):
    if role in ("public", "validator"):
        if scope["subgraphs"] or any(node["op"] == "If" for node in scope["nodes"]):
            raise ValueError("public_and_validator_have_no_control_if")
        return
    role_identity = [node for node in scope["nodes"] if node["name"] == "output_is_critic"]
    if len(role_identity) != 1 or any(role_identity[0].get(key) != expected for key, expected in (
            ("op", "Identity"), ("inputs", ["role_is_critic"]), ("outputs", ["is_critic"]),
            ("input_origins", [ROLE]), ("static_annotation", "role_scalar_identity_candidate"),
            ("attribute_names", []))):
        raise ValueError("exact_virtual_role_output_identity_required")
    routes = [node for node in scope["nodes"] if node["op"] == "If"]
    if role == "shared_pc" and not warm:
        if len(routes) != 6:
            raise ValueError("fresh_virtual_pc_requires_six_role_routes")
        for node, kind in zip(routes, ("initial", "ffn", "ffn", "ffn", "ffn", "heads")):
            _role_route(scope, node, kind)
        return
    if role != "shared_warm" or not warm or len(routes) != 6 or not scope["nodes"]:
        raise ValueError("registered_virtual_warm_graph_required")
    mode = scope["nodes"][0]
    if (routes[0] is not mode or mode["name"] != WARM_MODE_NODE or mode["inputs"] != ["warm_start"]
            or mode["input_origins"] != [WARM] or mode["static_annotation"] != WARM_ANNOTATION
            or len(mode["outputs"]) != 1 or not re.fullmatch(r"selected_private_initial_[1-9][0-9]*", mode["outputs"][0])):
        raise ValueError("exact_virtual_warm_mode_route_required")
    selected = mode["outputs"][0]
    branches = _children(scope, mode)
    seed, fresh = branches["then_branch"], branches["else_branch"]
    for branch, name in ((seed, selected + "__warm_branch"), (fresh, selected + "__fresh_branch")):
        if branch["inputs"] or branch["initializer_count"] or branch["outputs"] != [{"name": name, "dtype": "FLOAT", "shape": ["batch", 16, 384]}]:
            raise ValueError("virtual_warm_branch_bank_or_output_differs")
    if seed["subgraphs"] or len(seed["nodes"]) != 1:
        raise ValueError("virtual_seed_branch_must_be_identity_only")
    identity = seed["nodes"][0]
    if (identity["name"] != "consume_complete_virtual_private_seed" or identity["op"] != "Identity"
            or identity["inputs"] != ["initial_latent"] or identity["outputs"] != [selected + "__warm_branch"]
            or identity["attribute_names"] or identity["input_origins"] != [DATA]
            or identity["static_annotation"] != "none"):
        raise ValueError("virtual_seed_branch_query_math_or_alias_refused")
    if not fresh["nodes"]:
        raise ValueError("complete_virtual_fresh_prefix_required")
    initial = fresh["nodes"][-1]
    _role_route(fresh, initial, "initial")
    if initial["outputs"] != [selected + "__fresh_branch"] or any(node["op"] == "If" for node in fresh["nodes"][:-1]):
        raise ValueError("virtual_fresh_prefix_last_initial_route_required")
    required = {"query_line_features", "relation_features", "relation_active_float", "query_anchor_features"}
    actual = {output for node in fresh["nodes"][:-1] if node["op"] == "Identity" for output in node["outputs"]}
    if not required.issubset(actual):
        raise ValueError("virtual_fresh_query_preparation_is_incomplete")
    expected_children = {f"{fresh['path']}/node{initial['index']}:{initial['name']}/then_branch",
                         f"{fresh['path']}/node{initial['index']}:{initial['name']}/else_branch"}
    if {child["path"] for child in fresh["subgraphs"]} != expected_children:
        raise ValueError("virtual_fresh_prefix_extra_nested_control_refused")
    for node, kind in zip(routes[1:], ("ffn", "ffn", "ffn", "ffn", "heads")):
        _role_route(scope, node, kind)


def _check_document(document, entry, manifest):
    legacy._check_document(document, entry, manifest)
    role = entry["role"]
    if document.graph.name != GRAPH_NAMES[role]:
        raise ValueError("exact_virtual_serialized_graph_name_required")
    if any(tensor.name.startswith("experts.") for tensor in document.graph.initializer) and role == "public":
        raise ValueError("public_graph_private_initializer_refused")
    for tensor in document.graph.initializer:
        if tensor.name.startswith("experts."):
            allowed = ("experts.validator.",) if role == "validator" else ("experts.proposer.", "experts.critic.")
            if not tensor.name.startswith(allowed):
                raise ValueError("virtual_graph_private_bank_role_differs")


def _weight_bindings(document, role, manifest):
    """Connect pinned ledger declarations to actual type/shape, never values.

    Tensor content SHA fields are export-ledger declarations covered by the
    manifest byte pin. This producer does not independently hash Tensor data.
    Runtime admission separately maps these source nodes to actual kernels.
    """
    banks = manifest.get("initializer_banks")
    bank = banks.get(role) if type(banks) is dict else None
    rows = bank.get("parameters") if type(bank) is dict else None
    if type(rows) is not list or not rows or len(rows) > MAX_INITIALIZERS_PER_SCOPE:
        raise ValueError("bounded_export_parameter_ledger_required")
    actual = {tensor.name: tensor for tensor in document.graph.initializer}
    if len(actual) != len(document.graph.initializer):
        raise ValueError("duplicate_initializer_ledger_source")
    fields = {"initializer", "parameter", "shape", "bytes", "sha256", "source_parameter_sha256"}
    bindings, names = [], set()
    for row in rows:
        if type(row) is not dict or set(row) != fields:
            raise ValueError("exact_export_parameter_ledger_row_required")
        name, parameter = legacy._name(row["initializer"]), legacy._name(row["parameter"])
        if name in names or name not in actual or name not in (parameter, parameter + ".linear_transpose"):
            raise ValueError("parameter_initializer_identity_differs")
        names.add(name)
        tensor = actual[name]
        shape = row["shape"]
        if (tensor.data_type != 1 or type(shape) is not list or not 0 < len(shape) <= 8
                or any(type(dim) is not int or not 0 < dim <= 1048576 for dim in shape)
                or list(tensor.dims) != shape or type(row["bytes"]) is not int
                or not legacy._sha(row["sha256"]) or not legacy._sha(row["source_parameter_sha256"])):
            raise ValueError("parameter_ledger_dtype_shape_or_pin_differs")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if row["bytes"] != elements * 4 or row["bytes"] > MAX_GRAPH_BYTES:
            raise ValueError("parameter_ledger_byte_shape_differs")
        if not parameter.startswith(("public_encoder.", "reader_blocks.", "move_embedding.", "experts.")):
            raise ValueError("unregistered_virtual_parameter_namespace")
        if role == "public" and not parameter.startswith("public_encoder."):
            raise ValueError("public_parameter_ledger_is_not_role_neutral")
        if role != "public" and parameter.startswith("public_encoder.") and not parameter.startswith("public_encoder.line_encoder."):
            raise ValueError("private_parameter_ledger_has_public_board_bank")
        if parameter.startswith("experts."):
            allowed = ("experts.validator.",) if role == "validator" else ("experts.proposer.", "experts.critic.")
            if role == "public" or not parameter.startswith(allowed):
                raise ValueError("parameter_ledger_private_role_differs")
        bindings.append({"initializer": name, "parameter": parameter, "dtype": "FLOAT", "shape": shape,
                         "bytes": row["bytes"], "sha256": row["sha256"],
                         "source_parameter_sha256": row["source_parameter_sha256"]})
    # The registered builder's non-parameter float constants are scalar
    # constant_N initializers only. No alternate matrix/embedding namespace
    # may hide a source bank outside the export ledger. Scalars remain DATA,
    # never integral control or a CPU permission.
    unbound = [name for name, tensor in actual.items() if tensor.data_type == 1 and name not in names
               and not (re.fullmatch(r"constant_[1-9][0-9]*", name) and list(tensor.dims) == [])]
    if unbound:
        raise ValueError("source_parameter_missing_from_export_ledger")
    return bindings


def _private_dependencies(scope, bindings):
    by_name = {row["initializer"]: row for row in bindings}
    result = []
    def visit(body):
        for node in body["nodes"]:
            if node["op"] not in legacy.MAJOR:
                continue
            for name in node["inputs"]:
                row = by_name.get(name)
                if row is not None and row["parameter"].startswith("experts."):
                    result.append({"scope_path": body["path"], "node_index": node["index"],
                                   "node_name": node["name"], "op": node["op"],
                                   "initializer": name, "parameter": row["parameter"]})
        for child in body["subgraphs"]:
            visit(child)
    visit(scope)
    return result


def _write_exclusive(path, inventory):
    raw = (json.dumps(inventory, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_INVENTORY_BYTES:
        raise ValueError("inventory_output_bound_exceeded")
    path = legacy._external(path, output=True)
    with path.open("xb") as stream:
        written = stream.write(raw)
        if written != len(raw):
            raise OSError("incomplete_inventory_write")
        stream.flush()
        os.fsync(stream.fileno())
    return {"schema": SCHEMA, "inventory_sha256": hashlib.sha256(raw).hexdigest(),
            "inventory_bytes": len(raw), "total_nodes": inventory["total_nodes"],
            "actual_provider_placement": "not_observed", "cpu_allowlist": "not_created"}


def _admission_limits():
    return {"max_manifest_bytes": MAX_MANIFEST_BYTES, "max_graph_bytes": MAX_GRAPH_BYTES,
            "max_inventory_bytes": MAX_INVENTORY_BYTES, "max_nodes": MAX_NODES,
            "max_graph_nodes": MAX_GRAPH_NODES, "max_initializers_per_scope": MAX_INITIALIZERS_PER_SCOPE,
            "max_attributes_per_node": MAX_ATTRIBUTES_PER_NODE,
            "max_recursive_depth": MAX_RECURSIVE_DEPTH, "max_path_bytes": MAX_PATH_BYTES}


def produce_virtual_control_inventory(export_manifest, export_manifest_sha256, output_json, *, opt_in=False, max_seconds=30):
    """Pin all bytes before ONNX import; never instantiate a runtime/session."""
    if opt_in is not True:
        raise ValueError("explicit_virtual_control_inventory_opt_in_required")
    if type(max_seconds) is not int or not 1 <= max_seconds <= 60:
        raise ValueError("explicit_finite_metadata_wall_required")
    deadline = time.monotonic() + max_seconds
    output = legacy._external(output_json, output=True)
    manifest_path = legacy._external(export_manifest)
    raw_manifest = legacy._read_pinned(manifest_path, export_manifest_sha256, MAX_MANIFEST_BYTES)
    manifest, warm = _manifest(raw_manifest)
    raw_graphs, graph_bytes = [], 0
    for entry in manifest["graphs"]:
        if time.monotonic() >= deadline:
            raise ValueError("inventory_wall_deadline_exceeded")
        raw = legacy._read_pinned(manifest_path.parent / entry["file"], entry["sha256"], MAX_GRAPH_BYTES)
        graph_bytes += len(raw)
        if graph_bytes > MAX_TOTAL_GRAPH_BYTES:
            raise ValueError("total_pinned_graph_byte_bound_exceeded")
        raw_graphs.append(raw)
    import onnx
    graphs, total_nodes = [], 0
    for entry, raw in zip(manifest["graphs"], raw_graphs):
        state = _State(entry["role"], warm, deadline)
        state.count = total_nodes
        document = onnx.load_model_from_string(raw)
        _check_document(document, entry, manifest)
        onnx.checker.check_model(document, full_check=False)
        body = _scope(document.graph, entry["role"], {}, state)
        _validate_routes(entry["role"], body, warm)
        bindings = _weight_bindings(document, entry["role"], manifest)
        total_nodes = state.count
        graphs.append({"role": entry["role"], "file": entry["file"], "sha256": entry["sha256"],
                       "serialized_graph_name": document.graph.name, "inventory": body,
                       "weight_bindings": bindings, "private_neural_dependencies": _private_dependencies(body, bindings)})
    if total_nodes == 0:
        raise ValueError("empty_virtual_graph_inventory_refused")
    inventory = {"schema": SCHEMA, "scope": SCOPE, "artifact_kind": "approx_warm" if warm else "fresh",
                 "layout": manifest["layout"], "manifest_sha256": export_manifest_sha256,
                 "actual_provider_placement": "not_observed", "cpu_allowlist": "not_created",
                 "total_nodes": total_nodes, "graphs": graphs,
                 "producer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "admission_limits": _admission_limits()}
    if time.monotonic() >= deadline:
        raise ValueError("inventory_wall_deadline_exceeded")
    return _write_exclusive(output, inventory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-virtual-control-inventory", action="store_true", required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        receipt = produce_virtual_control_inventory(args.export_manifest, args.export_manifest_sha256,
                  args.output_json, opt_in=args.enable_virtual_control_inventory, max_seconds=args.max_seconds)
    except (OSError, ValueError, TypeError, ImportError):
        print(json.dumps({"schema": SCHEMA, "inventory_created": False,
                          "failure": "bounded_pinned_virtual_metadata_inventory_rejected"}))
        return 1
    print(json.dumps(receipt, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
