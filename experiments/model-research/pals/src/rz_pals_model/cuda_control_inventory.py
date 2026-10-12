"""Pinned serialized ONNX metadata inventory; never creates an NN session.

The Rust control policy independently recomputes this provenance and owns
admission. This module does not create a CPU allowlist, observe providers,
read Tensor values, or certify CUDA kernels, residency, or GQA performance.
The caller must impose the registered process/resource deadline as well.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

SCHEMA = "rovezero.pals-static-control-inventory.v2"
CUDA_WARM_SCHEMA = "rovezero.pals-private-cuda-warm.v2"
CUDA_WARM_LAYOUT = "shared_pc_if_approx_cuda_warm_v2"
MAX_MANIFEST_BYTES = 128 * 1024
MAX_GRAPH_BYTES = 256 * 1024 * 1024
MAX_INVENTORY_BYTES = 1024 * 1024
MAX_NODES = 5000
MAX_INITIALIZERS_PER_SCOPE = 5000
MAX_ATTRIBUTES_PER_NODE = 32
MAX_RECURSIVE_DEPTH = 32
MAX_PATH_BYTES = 4096
DTYPES = {1: "FLOAT", 6: "INT32", 7: "INT64", 9: "BOOL"}
MAJOR = {"MatMul", "Gemm", "Conv", "Attention", "MultiHeadAttention", "Softmax",
         "LayerNormalization", "SimplifiedLayerNormalization", "SkipLayerNormalization",
         "Gelu", "FastGelu", "BiasGelu", "Relu", "Sigmoid", "Tanh"}
METADATA_CHAIN = {"Gather", "Concat", "Unsqueeze", "Slice", "Mul"}
DATA, ROLE, WARM = "model_data_or_unknown", "role_control", "cuda_warm_mode_control"
META, INTEGRAL = "shape_metadata", "small_integral_constant"


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _external(path, *, output=False):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute_external_path_required")
    for part in path.parts:
        word = part.lower()
        if (word == ".env" or word.startswith(".env.") or "credential" in word
                or "service-account" in word or "api_key" in word
                or word.startswith(("id_rsa", "id_ed25519"))):
            raise ValueError("secret_path_refused")
    if output:
        if path.exists() or path.is_symlink():
            raise ValueError("exclusive_output_already_exists")
        resolved = path.parent.resolve(strict=True)
    else:
        if path.is_symlink():
            raise ValueError("symlink_input_refused")
        resolved = path.resolve(strict=True)
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / ".git").exists():
            raise ValueError("checkout_input_or_output_refused")
    return path


def _read_pinned(path, digest, maximum):
    if not _sha(digest):
        raise ValueError("invalid_sha256_pin")
    path = _external(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("bounded_regular_input_required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if len(raw) != before.st_size or len(raw) > maximum or stamp(before) != stamp(after):
        raise ValueError("pinned_input_changed_or_exceeded_bound")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("pinned_input_sha256_differs")
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_field")
        result[key] = value
    return result


def _manifest(raw):
    def nonfinite(_):
        raise ValueError("nonfinite_json_refused")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("manifest_object_required")
    warm = value.get("schema") == CUDA_WARM_SCHEMA
    if warm:
        if (value.get("layout") != CUDA_WARM_LAYOUT or value.get("layout_revision") != 2
                or value.get("execution_domain") != "cuda_device_io_binding_v2"
                or value.get("precision") != "fp32" or value.get("tf32") is not False
                or value.get("model_profile") not in ("full_line_v2", "full_line_interaction_v2")):
            raise ValueError("unregistered_cuda_warm_manifest")
        private_file = CUDA_WARM_LAYOUT + ".onnx"
    else:
        if (value.get("schema") != "rovezero.pals-model.v2"
                or value.get("layout") != "shared_pc_if" or value.get("layout_revision") != 1):
            raise ValueError("unregistered_shared_control_manifest")
        private_file = "shared_pc_if.onnx"
    if value.get("roles") != ["proposer", "critic"] or value.get("validator_present") is not False:
        raise ValueError("control_inventory_is_v_free_pc_only")
    graphs = value.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != 2:
        raise ValueError("two_pinned_graphs_required")
    for graph, role, filename in zip(graphs, ("public", "shared_pc"), ("public_memory.onnx", private_file)):
        if (not isinstance(graph, dict) or graph.get("role") != role or graph.get("file") != filename
                or not _sha(graph.get("sha256")) or graph.get("opset") != 17
                or not isinstance(graph.get("inputs"), list) or not isinstance(graph.get("outputs"), list)):
            raise ValueError("manifest_graph_pin_or_abi_is_invalid")
    return value, warm


def _name(name):
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 256:
        raise ValueError("bounded_named_graph_value_required")
    return name


def _descriptors(values):
    if len(values) > 64:
        raise ValueError("descriptor_count_bound_exceeded")
    result = []
    names = set()
    for value in values:
        name = _name(value.name)
        if name in names or not value.type.HasField("tensor_type"):
            raise ValueError("duplicate_or_nontensor_descriptor")
        names.add(name)
        tensor = value.type.tensor_type
        dtype = DTYPES.get(tensor.elem_type)
        if dtype is None or not tensor.HasField("shape") or len(tensor.shape.dim) > 8:
            raise ValueError("unsupported_tensor_descriptor")
        shape = []
        for dimension in tensor.shape.dim:
            if dimension.HasField("dim_param"):
                shape.append(_name(dimension.dim_param))
            elif dimension.HasField("dim_value") and 0 <= dimension.dim_value <= 1_048_576:
                shape.append(dimension.dim_value)
            else:
                raise ValueError("unbounded_or_unknown_shape_dimension")
        result.append({"name": name, "dtype": dtype, "shape": shape})
    return result


def _tensor_seed(tensor, name, source):
    # dtype/shape only: do not access raw_data, *_data, string_data, or values.
    if getattr(tensor, "external_data", ()) or getattr(tensor, "data_location", 0) != 0:
        raise ValueError("external_tensor_storage_refused")
    dtype = DTYPES.get(tensor.data_type)
    shape = list(tensor.dims)
    if dtype in ("INT64", "INT32", "BOOL") and len(shape) <= 1 and all(0 <= dim <= 64 for dim in shape):
        return {"name": _name(name), "dtype": dtype, "shape": shape, "source": source}
    return None


class _State:
    def __init__(self, warm, deadline):
        self.warm = warm
        self.deadline = deadline
        self.count = 0
        self.names = set()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise ValueError("inventory_wall_deadline_exceeded")


def _scope(graph, path, inherited, state, depth=0):
    state.check()
    if depth > MAX_RECURSIVE_DEPTH or len(path.encode("utf-8")) > MAX_PATH_BYTES:
        raise ValueError("recursive_metadata_bound_exceeded")
    inputs, outputs = _descriptors(graph.input), _descriptors(graph.output)
    origins = dict(inherited)
    for value in inputs:
        origin = DATA
        if value["name"] == "role_is_critic":
            if value["dtype"] != "BOOL" or value["shape"] != []:
                raise ValueError("role_control_is_not_scalar_bool")
            origin = ROLE
        elif value["name"] == "warm_start" and state.warm and path == "shared_pc":
            if value["dtype"] != "BOOL" or value["shape"] != []:
                raise ValueError("cuda_warm_mode_is_not_scalar_bool")
            origin = WARM
        if value["name"] in origins:
            raise ValueError("nested_input_shadows_outer_value")
        origins[value["name"]] = origin
    if graph.sparse_initializer or len(graph.initializer) > MAX_INITIALIZERS_PER_SCOPE:
        raise ValueError("sparse_or_unbounded_initializer_metadata")
    seeds = []
    for tensor in graph.initializer:
        name = _name(tensor.name)
        if name in origins:
            raise ValueError("initializer_shadows_value")
        seed = _tensor_seed(tensor, name, "graph_initializer_metadata")
        if seed is not None:
            seeds.append(seed)
        origins[name] = INTEGRAL if seed else DATA
    nodes, children = [], []
    for index, node in enumerate(graph.node):
        state.check()
        state.count += 1
        name = _name(node.name)
        if (state.count > MAX_NODES or name in state.names or node.domain not in ("", "ai.onnx")
                or getattr(node, "overload", "") or len(node.input) > 64 or not 0 < len(node.output) <= 16):
            raise ValueError("ambiguous_or_unsupported_serialized_node")
        state.names.add(name)
        attrs = [attribute.name for attribute in node.attribute]
        if (len(attrs) > MAX_ATTRIBUTES_PER_NODE or len(attrs) != len(set(attrs))
                or any(not _name(attr) for attr in attrs)):
            raise ValueError("duplicate_or_invalid_node_attribute")
        _name(node.op_type)
        for input_name in node.input:
            if input_name:
                _name(input_name)
        actual_origins = [origins.get(value, DATA) for value in node.input]
        provenance = [origin if value else "absent_optional_input" for value, origin in zip(node.input, actual_origins)]
        annotation, output_origin, constant_seeds = "none", DATA, []
        for attribute in node.attribute:
            if attribute.type == 4:
                seed_name = node.output[0] if node.op_type == "Constant" and len(node.output) == 1 else name
                seed = _tensor_seed(attribute.t, seed_name, "constant_node_tensor_metadata")
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
        elif (node.op_type in METADATA_CHAIN and actual_origins and META in actual_origins
              and all(origin in (META, INTEGRAL) for origin in actual_origins)):
            annotation, output_origin = "shape_metadata_chain_candidate", META
        elif node.op_type == "Identity" and list(node.input) == ["role_is_critic"] and actual_origins == [ROLE]:
            annotation, output_origin = "role_scalar_identity_candidate", ROLE
        elif node.op_type in MAJOR:
            annotation = "major_nn_requires_cuda"
        if node.op_type == "If":
            if set(attrs) != {"then_branch", "else_branch"} or len(node.input) != 1:
                raise ValueError("exact_two_branch_if_required")
            if list(node.input) == ["role_is_critic"] and actual_origins == [ROLE]:
                annotation = "role_if_dispatch_candidate"
            elif (state.warm and path == "shared_pc" and name == "private_warm_mode"
                  and list(node.input) == ["warm_start"] and actual_origins == [WARM]):
                annotation = "cuda_warm_if_dispatch_candidate"
            else:
                raise ValueError("arbitrary_bool_or_data_if_refused")
            # Capture the lexical outer scope before this If's outputs exist.
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
            if _name(output) in origins:
                raise ValueError("node_output_aliases_outer_value")
            origins[output] = output_origin
    return {"path": path, "inputs": inputs, "outputs": outputs, "integral_initializers": seeds,
            "initializer_count": len(graph.initializer), "nodes": nodes, "subgraphs": children}


def _check_document(document, entry, manifest):
    if document.functions or document.training_info or not 1 <= document.ir_version <= 10:
        raise ValueError("unregistered_function_training_or_ir")
    imports = [(item.domain, item.version) for item in document.opset_import]
    if len(imports) != 1 or imports[0] not in (("", 17), ("ai.onnx", 17)):
        raise ValueError("unregistered_opset")
    _name(document.graph.name)
    if entry["role"] == "shared_pc" and document.graph.name != manifest["layout"]:
        raise ValueError("serialized_private_graph_name_differs")
    if _descriptors(document.graph.input) != entry["inputs"] or _descriptors(document.graph.output) != entry["outputs"]:
        raise ValueError("serialized_graph_abi_differs_from_manifest")
    props = _unique_object([(item.key, item.value) for item in document.metadata_props])
    expected = {"schema": manifest["schema"], "layout": manifest["layout"], "role": entry["role"]}
    for key in ("model_semantics", "encoding_schema", "model_profile", "checkpoint_sha256",
                "rules_input_profile", "rules_input_semantic_sha256", "rules_encoder_source_sha256", "execution_domain"):
        if key in manifest:
            expected[key] = manifest[key]
    if any(props.get(key) != value for key, value in expected.items()):
        raise ValueError("serialized_graph_metadata_pin_differs")


def _write_exclusive(path, inventory):
    raw = (json.dumps(inventory, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_INVENTORY_BYTES:
        raise ValueError("inventory_output_bound_exceeded")
    path = _external(path, output=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"schema": SCHEMA, "inventory_sha256": hashlib.sha256(raw).hexdigest(),
            "inventory_bytes": len(raw), "total_nodes": inventory["total_nodes"],
            "actual_provider_placement": "not_observed", "cpu_allowlist": "not_created"}


def produce_cuda_control_inventory(export_manifest, export_manifest_sha256, output_json, *, max_seconds=30):
    """Read only exact pinned graph bytes, then write one new bounded inventory.

    No checkpoint, Tensor values, model math, NN session, or device access.
    ONNX is imported only after manifest and both graph byte pins pass.
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= 60:
        raise ValueError("explicit_finite_metadata_wall_required")
    deadline = time.monotonic() + max_seconds
    output = _external(output_json, output=True)
    manifest_path = _external(export_manifest)
    raw_manifest = _read_pinned(manifest_path, export_manifest_sha256, MAX_MANIFEST_BYTES)
    manifest, warm = _manifest(raw_manifest)
    raw_graphs = []
    for entry in manifest["graphs"]:
        if time.monotonic() >= deadline:
            raise ValueError("inventory_wall_deadline_exceeded")
        raw_graphs.append(_read_pinned(manifest_path.parent / entry["file"], entry["sha256"], MAX_GRAPH_BYTES))
    import onnx
    graph_inventories, total_nodes = [], 0
    for entry, raw in zip(manifest["graphs"], raw_graphs):
        state = _State(warm and entry["role"] == "shared_pc", deadline)
        state.count = total_nodes
        document = onnx.load_model_from_string(raw)
        _check_document(document, entry, manifest)
        onnx.checker.check_model(document, full_check=False)
        inventory = _scope(document.graph, entry["role"], {}, state)
        total_nodes = state.count
        graph_inventories.append({"role": entry["role"], "sha256": entry["sha256"],
                                  "serialized_graph_name": document.graph.name, "inventory": inventory})
    if total_nodes == 0:
        raise ValueError("empty_graph_inventory_refused")
    inventory = {"schema": SCHEMA, "scope": "read_only_serialized_graph_metadata_no_runtime",
                 "manifest_sha256": export_manifest_sha256, "actual_provider_placement": "not_observed",
                 "cpu_allowlist": "not_created", "total_nodes": total_nodes, "graphs": graph_inventories,
                 "producer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "admission_limits": {"manifest_bytes": MAX_MANIFEST_BYTES, "graph_bytes_each": MAX_GRAPH_BYTES,
                     "inventory_bytes": MAX_INVENTORY_BYTES, "nodes": MAX_NODES,
                     "recursive_depth": MAX_RECURSIVE_DEPTH, "max_seconds": max_seconds}}
    if time.monotonic() >= deadline:
        raise ValueError("inventory_wall_deadline_exceeded")
    return _write_exclusive(output, inventory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        result = produce_cuda_control_inventory(args.export_manifest, args.export_manifest_sha256,
                                               args.output_json, max_seconds=args.max_seconds)
    except (OSError, ValueError, ImportError):
        print(json.dumps({"schema": SCHEMA, "inventory_created": False,
                          "failure": "bounded_pinned_metadata_inventory_rejected"}))
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
