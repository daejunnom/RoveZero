"""Weight-free public K/V routing artifact, separate from the learned exports.

This first unit supplies a fixed ONNX interface and a CPU input witness. It does
not supply a CUDA resident record owner, device aliases, leases, fences, a native
consumer, or private Warm. CPU validation is never evidence of those lifetimes.
Generated artifacts belong outside the source checkout.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType


SCHEMA = "rovezero.pals-device-packing.v1"
DOMAIN = "rovezero.pals.weight-free-public-routing"
GRAPH_FILE = "device_public_pack.onnx"
MANIFEST_FILE = "device-packing.json"
BOARD_TOKENS = 66
MAX_RECORDS = 128
HEADS = 2
HEAD_DIMENSION = 64
MAX_TOKENS = BOARD_TOKENS + MAX_RECORDS
MAX_GRAPH_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024
MAX_CPU_INPUT_BYTES = 32 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}: integer outside [{minimum}, {maximum}]")
    return value


def _digest(value, name):
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError(f"{name}: expected lowercase SHA-256")
    return value


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _shape(name, dtype, dimensions):
    return {"name": name, "dtype": dtype, "shape": dimensions}


def input_descriptors():
    result = [_shape("board_key", "FLOAT", [1, HEADS, "base_source_tokens", HEAD_DIMENSION]),
              _shape("board_value", "FLOAT", [1, HEADS, "base_source_tokens", HEAD_DIMENSION])]
    for index in range(MAX_RECORDS):
        stem = f"{index:03}"
        shape = [1, HEADS, f"source_tokens_{stem}", HEAD_DIMENSION]
        result += [_shape("record_key_" + stem, "FLOAT", shape),
                   _shape("record_value_" + stem, "FLOAT", shape),
                   _shape("record_offset_" + stem, "INT64", [1])]
    return result + [_shape("record_stop", "INT64", [1])]


def output_descriptors():
    shape = [1, HEADS, "memory_tokens", HEAD_DIMENSION]
    return [_shape("memory_key", "FLOAT", shape),
            _shape("memory_value", "FLOAT", shape),
            _shape("finite_output", "BOOL", [])]


def build_device_packing_graph():
    """Build only deterministic routing/control operations; no model is loaded."""
    from onnx import TensorProto, checker, helper
    types = {"FLOAT": TensorProto.FLOAT, "INT64": TensorProto.INT64,
             "BOOL": TensorProto.BOOL}
    nodes, values = [], []

    def node(operation, inputs, output, dtype, shape, **attributes):
        nodes.append(helper.make_node(operation, inputs, [output], name=output,
                                      **attributes))
        values.append(helper.make_tensor_value_info(output, types[dtype], shape))
        return output

    initializers = [helper.make_tensor(name, TensorProto.INT64, dims, data)
                    for name, dims, data in (("start_zero", [1], [0]),
                                             ("board_stop", [1], [BOARD_TOKENS]),
                                             ("token_axis", [1], [2]),
                                             ("unit_step", [1], [1]),
                                             ("finite_zero", [], [0]))]
    final_shape = [1, HEADS, "memory_tokens", HEAD_DIMENSION]
    for kind in ("key", "value"):
        board = node("Slice", ["board_" + kind, "start_zero", "board_stop",
                               "token_axis", "unit_step"], "board_slice_" + kind,
                     "FLOAT", [1, HEADS, BOARD_TOKENS, HEAD_DIMENSION])
        records = [node("Gather", [f"record_{kind}_{i:03}", f"record_offset_{i:03}"],
                        f"gather_{kind}_{i:03}", "FLOAT", [1, HEADS, 1, HEAD_DIMENSION],
                        axis=2) for i in range(MAX_RECORDS)]
        joined = node("Concat", [board] + records, "all_ports_" + kind, "FLOAT",
                      [1, HEADS, MAX_TOKENS, HEAD_DIMENSION], axis=2)
        memory = node("Slice", [joined, "start_zero", "record_stop", "token_axis",
                                "unit_step"], "memory_" + kind, "FLOAT", final_shape)
        nan = node("IsNaN", [memory], "nan_" + kind, "BOOL", final_shape)
        inf = node("IsInf", [memory], "inf_" + kind, "BOOL", final_shape)
        bad = node("Or", [nan, inf], "bad_" + kind, "BOOL", final_shape)
        cast = node("Cast", [bad], "bad_int_" + kind, "INT64", final_shape,
                    to=TensorProto.INT64)
        maximum = node("ReduceMax", [cast], "bad_max_" + kind, "INT64", [],
                       axes=[0, 1, 2, 3], keepdims=0)
        node("Equal", [maximum, "finite_zero"], "finite_" + kind, "BOOL", [])
    node("And", ["finite_key", "finite_value"], "finite_output", "BOOL", [])
    inputs = [helper.make_tensor_value_info(d["name"], types[d["dtype"]], d["shape"])
              for d in input_descriptors()]
    outputs = [helper.make_tensor_value_info(d["name"], types[d["dtype"]], d["shape"])
               for d in output_descriptors()]
    output_names = {value.name for value in outputs}
    graph = helper.make_graph(nodes, DOMAIN, inputs, outputs, initializers,
                              value_info=[v for v in values if v.name not in output_names])
    document = helper.make_model(graph, producer_name="rz-pals-device-packing",
                                  producer_version="1", ir_version=10,
                                  opset_imports=[helper.make_opsetid("", 17)])
    helper.set_model_props(document, {"artifact_domain": DOMAIN,
                                     "learned_weights": "0",
                                     "native_connected": "false"})
    checker.check_model(document, full_check=True)
    return document


def graph_bytes():
    return build_device_packing_graph().SerializeToString(deterministic=True)


def _bounded_graph(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_GRAPH_BYTES:
        raise ValueError("graph: immutable bytes exceed bounded artifact size")
    # The deterministic specification rejects even hash-updated foreign graphs,
    # float initializers, external data, functions, subgraphs and extra operators.
    expected = graph_bytes()
    if raw != expected:
        raise ValueError("graph: not the exact weight-free fixed routing specification")
    return build_device_packing_graph()


def intermediate_arithmetic(document):
    """Sum each declared node output's independent maximum tensor payload.

    This is NOT peak memory. Lifetimes, ORT arenas, allocator overhead, sessions,
    CUDA workspaces and aliased inputs are deliberately not inferred from it.
    """
    from onnx import TensorProto
    widths = {TensorProto.FLOAT: ("FLOAT", 4), TensorProto.INT64: ("INT64", 8),
              TensorProto.BOOL: ("BOOL", 1)}
    bounds = {"base_source_tokens": MAX_TOKENS, "memory_tokens": MAX_TOKENS}
    bounds.update({f"source_tokens_{i:03}": MAX_TOKENS for i in range(MAX_RECORDS)})
    infos = {v.name: v for v in list(document.graph.value_info) + list(document.graph.output)}
    ledger, seen = [], set()
    for node in document.graph.node:
        for name in node.output:
            if name in seen or name not in infos:
                raise ValueError("graph: missing or duplicate node output shape")
            seen.add(name)
            tensor = infos[name].type.tensor_type
            if tensor.elem_type not in widths:
                raise ValueError("graph: unsupported arithmetic dtype")
            dtype, width = widths[tensor.elem_type]
            maximum = []
            for dim in tensor.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    maximum.append(dim.dim_value)
                elif dim.HasField("dim_param") and dim.dim_param in bounds:
                    maximum.append(bounds[dim.dim_param])
                else:
                    raise ValueError("graph: no finite bound for output dimension")
            size = math.prod(maximum) * width
            ledger.append({"node": node.name, "output": name, "dtype": dtype,
                           "maximum_shape": maximum, "maximum_payload_bytes": size})
    return {"meaning": "sum_of_independent_maximum_node_output_payloads_not_peak",
            "node_output_count": len(ledger), "node_outputs": ledger,
            "maximum_sum_payload_bytes": sum(v["maximum_payload_bytes"] for v in ledger),
            "allocator_workspace_and_session_bytes": "unknown",
            "physical_stage_completion": "not_implemented"}


def manifest_for_graph(raw):
    document = _bounded_graph(raw)
    return {"schema": SCHEMA, "artifact_domain": DOMAIN, "layout_revision": 1,
            "graph": {"file": GRAPH_FILE, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(), "ir_version": 10,
                      "opset": 17, "inputs": input_descriptors(),
                      "outputs": output_descriptors()},
            "learned_weights": {"count": 0, "bytes": 0},
            "integer_control_initializers": {"count": 5, "payload_bytes": 40},
            "bounds": {"batch": 1, "precision": "FP32", "board_tokens": 66,
                       "kv_heads": 2, "head_dimension": 64, "record_capacity": 128,
                       "source_tokens_minimum": 67, "source_tokens_maximum": 194,
                       "record_count_minimum": 0, "record_count_maximum": 128,
                       "record_offset_minimum": 66, "record_offset_maximum": 193,
                       "record_stop": "66 + max(record_count, 1)"},
            "routing": {"base": "exactly_one_actual_matching_board_base",
                        "records": "ordered_occurrences_duplicates_preserved",
                        "inactive_ports": "alias_matching_base_token_66",
                        "zero_records": "actual_zero_feature_projection_at_token_66_with_false_mask",
                        "mask": "CPU_owned_current_mask_first_66_true_zero_pad_false",
                        "finite_output": "joined_visible_key_and_value_only"},
            "intermediate_arithmetic": intermediate_arithmetic(document),
            "scope": {"artifact_and_cpu_numeric_only": True, "native_connected": False,
                      "cuda_resident_owner": "not_implemented",
                      "cuda_physical_completion_and_quarantine": "not_implemented",
                      "cuda_validation": "not_run_user_deferred",
                      "private_warm": "outside_this_artifact",
                      "cpu_projection_origin": "caller_identity_and_actual_feature_bits_not_native_producer_attestation",
                      "cpu_owner_accounting": "unique_owned_numpy_payloads_only_not_peak",
                      "device_runtime_bytes": "unknown"}}


def _exact_json_types(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (value.keys() == expected.keys() and
                all(_exact_json_types(value[key], item) for key, item in expected.items()))
    if type(expected) is list:
        return (len(value) == len(expected) and
                all(_exact_json_types(a, b) for a, b in zip(value, expected)))
    return value == expected


def validate_manifest(value, raw):
    if not _exact_json_types(value, manifest_for_graph(raw)):
        raise ValueError("manifest: exact schema, types, bounds and graph identity required")
    return value


def read_manifest_bytes(raw, graph_raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_MANIFEST_BYTES:
        raise ValueError("manifest: bounded immutable bytes required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("manifest: duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("manifest: nonfinite JSON constant " + value)

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=invalid_constant)
    return validate_manifest(value, graph_raw)


def export_device_packing_artifact(output_directory):
    """Write a new, outside-checkout artifact directory; never overwrite."""
    source = Path(__file__).resolve()
    checkout = next((parent for parent in source.parents if (parent / ".git").exists()), None)
    if checkout is None:
        raise ValueError("export: checkout boundary unavailable")
    requested = Path(output_directory)
    if not requested.is_absolute():
        raise ValueError("export: explicit absolute outside-checkout directory required")
    target = requested.resolve()
    if target == checkout or checkout in target.parents:
        raise ValueError("export: generated artifacts cannot enter source checkout")
    target.mkdir(parents=True, exist_ok=False)
    raw = graph_bytes()
    manifest = manifest_for_graph(raw)
    for name, payload in ((GRAPH_FILE, raw), (MANIFEST_FILE, _json_bytes(manifest))):
        with (target / name).open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    # Failure retains any partial output as evidence; no automatic cleanup.
    return manifest


@dataclass(frozen=True, eq=False)
class CpuProjectionBlock:
    """Entire owned CPU backing, never a device slice or physical lease claim."""
    board_identity: str
    model_identity: str
    public_graph_identity: str
    game_generation: int
    key: object
    value: object
    record_features: object
    record_mask: tuple


@dataclass(frozen=True)
class RecordReference:
    block: CpuProjectionBlock
    record_index: int


@dataclass(frozen=True, eq=False)
class PreparedPackingInputs:
    feeds: object
    current_mask: object
    owners: tuple
    references: tuple
    base: CpuProjectionBlock
    record_count: int
    expected_projection_keys: tuple
    model_identity: str
    public_graph_identity: str
    game_generation: int
    board_identity: str
    unique_owned_payload_bytes: int
    maximum_owned_payload_bytes: int


def _owned_array(array, dtype, shape, name):
    import numpy as np
    if type(array) is not np.ndarray or array.dtype != np.dtype(dtype):
        raise ValueError(name + ": exact ndarray dtype required")
    if tuple(array.shape) != tuple(shape) or not array.flags.c_contiguous:
        raise ValueError(name + ": exact contiguous shape required")
    if not array.flags.owndata or array.base is not None or array.flags.writeable:
        raise ValueError(name + ": entire readonly owned backing required")
    if not np.isfinite(array).all():
        raise ValueError(name + ": nonfinite input")
    return array


def _validate_block(block, model_identity, public_graph_identity, game_generation):
    if type(block) is not CpuProjectionBlock:
        raise ValueError("block: typed CPU owner required")
    _digest(block.board_identity, "block board")
    _digest(block.model_identity, "block model")
    _digest(block.public_graph_identity, "block public graph")
    if (block.model_identity != model_identity or
            block.public_graph_identity != public_graph_identity or
            type(block.game_generation) is not int or block.game_generation != game_generation):
        raise ValueError("block: model, projection graph or game generation mismatch")
    if type(block.record_mask) is not tuple or not 1 <= len(block.record_mask) <= MAX_RECORDS:
        raise ValueError("block: bounded nonempty projection slots required")
    if any(type(value) is not bool for value in block.record_mask):
        raise ValueError("block: exact Boolean record mask required")
    rows = len(block.record_mask)
    shape = (1, HEADS, BOARD_TOKENS + rows, HEAD_DIMENSION)
    _owned_array(block.key, "float32", shape, "block key")
    _owned_array(block.value, "float32", shape, "block value")
    _owned_array(block.record_features, "float32", (rows, 16), "block features")
    # Separate owning arrays cannot partially overlap; forbid a repeated K/V owner
    # because input accounting otherwise would hide the expected distinct backing.
    if block.key is block.value:
        raise ValueError("block: key/value owners must be distinct")
    return rows


def projection_key(block, record_index):
    """Actual immutable FP32 feature bits plus projection domain identity."""
    import numpy as np
    _validate_block(block, _digest(block.model_identity, "model"),
                    _digest(block.public_graph_identity, "public graph"),
                    _integer(block.game_generation, 0, (1 << 63) - 1, "generation"))
    index = _integer(record_index, 0, len(block.record_mask) - 1, "record index")
    domain = _json_bytes([DOMAIN, block.model_identity, block.public_graph_identity,
                          block.game_generation])
    feature_bits = block.record_features[index].astype(np.dtype("<f4"), copy=False).tobytes()
    return hashlib.sha256(len(domain).to_bytes(8, "little") + domain + feature_bits).hexdigest()


def prepare_cpu_inputs(base, references, *, board_identity, model_identity,
                       public_graph_identity, game_generation, record_count,
                       expected_projection_keys, maximum_owned_payload_bytes=MAX_CPU_INPUT_BYTES):
    """Prepare exact fixed ports with strong whole-block CPU owners.

    This validates CPU arrays at admission. NumPy flags are not a native immutable
    ownership mechanism; consumers must revalidate immediately before use. Native
    CUDA registration, aliases, budgets and physical completion remain absent.
    """
    import numpy as np
    board_identity = _digest(board_identity, "board")
    model_identity = _digest(model_identity, "model")
    public_graph_identity = _digest(public_graph_identity, "public graph")
    game_generation = _integer(game_generation, 0, (1 << 63) - 1, "generation")
    count = _integer(record_count, 0, MAX_RECORDS, "record count")
    limit = _integer(maximum_owned_payload_bytes, 1, MAX_CPU_INPUT_BYTES, "CPU payload budget")
    if type(references) is not tuple or type(expected_projection_keys) is not tuple:
        raise ValueError("records: bounded tuple references and expected keys required")
    refs, wanted = references, expected_projection_keys
    if len(refs) != count or len(wanted) != count:
        raise ValueError("records: count, ordered references and expected keys disagree")
    for expected in wanted:
        _digest(expected, "projection key")
    _validate_block(base, model_identity, public_graph_identity, game_generation)
    if base.board_identity != board_identity:
        raise ValueError("base: actual matching board identity required")
    if count == 0:
        # FP32 +0 bits, not an arbitrary cached record or synthesized zero K/V.
        if (len(base.record_mask) != 1 or base.record_mask[0] is not False or
                any(base.record_features[0].view(np.uint32))):
            raise ValueError("zero records: actual zero-feature false-mask base projection required")
    owners = [base]
    for reference, expected in zip(refs, wanted):
        if type(reference) is not RecordReference:
            raise ValueError("record: typed reference required")
        block = reference.block
        rows = _validate_block(block, model_identity, public_graph_identity, game_generation)
        index = _integer(reference.record_index, 0, rows - 1, "record index")
        if block.record_mask[index] is not True or projection_key(block, index) != _digest(expected, "projection key"):
            raise ValueError("record: active immutable projection identity mismatch")
        if not any(owner is block for owner in owners):
            owners.append(block)
    feeds = {"board_key": base.key, "board_value": base.value}
    for index in range(MAX_RECORDS):
        reference = refs[index] if index < count else RecordReference(base, 0)
        stem = f"{index:03}"
        feeds["record_key_" + stem] = reference.block.key
        feeds["record_value_" + stem] = reference.block.value
        offset = np.array([BOARD_TOKENS + reference.record_index], dtype=np.int64)
        offset.setflags(write=False)
        feeds["record_offset_" + stem] = offset
    stop = np.array([BOARD_TOKENS + max(count, 1)], dtype=np.int64)
    stop.setflags(write=False)
    feeds["record_stop"] = stop
    mask = np.ones((1, BOARD_TOKENS + max(count, 1)), dtype=np.bool_)
    if count == 0:
        mask[0, -1] = False
    mask.setflags(write=False)
    arrays = [array for owner in owners for array in (owner.key, owner.value, owner.record_features)]
    arrays += [feeds[f"record_offset_{i:03}"] for i in range(MAX_RECORDS)] + [stop, mask]
    unique = {id(array): array for array in arrays}
    payload = sum(array.nbytes for array in unique.values())
    if payload > limit:
        raise ValueError("CPU owner payload exceeds declared budget; no silent clamp")
    prepared = PreparedPackingInputs(MappingProxyType(feeds), mask, tuple(owners), refs,
                                     base, count, wanted, model_identity, public_graph_identity,
                                     game_generation, board_identity, payload, limit)
    validate_prepared_inputs(prepared)
    return prepared


def validate_prepared_inputs(prepared):
    import numpy as np
    if type(prepared) is not PreparedPackingInputs:
        raise ValueError("inputs: typed prepared witness required")
    count = _integer(prepared.record_count, 0, MAX_RECORDS, "record count")
    _digest(prepared.board_identity, "board")
    _digest(prepared.model_identity, "model")
    _digest(prepared.public_graph_identity, "public graph")
    _integer(prepared.game_generation, 0, (1 << 63) - 1, "generation")
    if (type(prepared.references) is not tuple or type(prepared.expected_projection_keys) is not tuple or
            len(prepared.references) != count or len(prepared.expected_projection_keys) != count):
        raise ValueError("inputs: ordered count drift")
    for expected in prepared.expected_projection_keys:
        _digest(expected, "projection key")
    expected_owners = [prepared.base]
    for reference in prepared.references:
        if type(reference) is not RecordReference:
            raise ValueError("inputs: untyped reference")
        if not any(owner is reference.block for owner in expected_owners):
            expected_owners.append(reference.block)
    if (type(prepared.owners) is not tuple or len(prepared.owners) != len(expected_owners) or
            any(a is not b for a, b in zip(prepared.owners, expected_owners))):
        raise ValueError("inputs: whole backing owner/pin roster drift")
    for owner in prepared.owners:
        _validate_block(owner, prepared.model_identity, prepared.public_graph_identity,
                        prepared.game_generation)
    if prepared.base.board_identity != prepared.board_identity:
        raise ValueError("inputs: matching base drift")
    if count == 0 and (len(prepared.base.record_mask) != 1 or
                       prepared.base.record_mask[0] is not False or
                       any(prepared.base.record_features[0].view(np.uint32))):
        raise ValueError("inputs: zero projection drift")
    if (type(prepared.feeds) is not MappingProxyType or
            set(prepared.feeds) != {d["name"] for d in input_descriptors()}):
        raise ValueError("inputs: exact fixed port names required")
    if prepared.feeds["board_key"] is not prepared.base.key or prepared.feeds["board_value"] is not prepared.base.value:
        raise ValueError("inputs: base alias owner mismatch")
    for i in range(MAX_RECORDS):
        reference = prepared.references[i] if i < count else RecordReference(prepared.base, 0)
        if i < count:
            index = _integer(reference.record_index, 0, len(reference.block.record_mask) - 1, "record index")
            if (reference.block.record_mask[index] is not True or
                    projection_key(reference.block, index) != prepared.expected_projection_keys[i]):
                raise ValueError("inputs: active projection drift")
        stem = f"{i:03}"
        if (prepared.feeds["record_key_" + stem] is not reference.block.key or
                prepared.feeds["record_value_" + stem] is not reference.block.value):
            raise ValueError("inputs: record alias owner mismatch")
        offset = _owned_array(prepared.feeds["record_offset_" + stem], "int64", (1,), "offset")
        if int(offset[0]) != BOARD_TOKENS + reference.record_index:
            raise ValueError("inputs: exact source offset drift")
    stop = _owned_array(prepared.feeds["record_stop"], "int64", (1,), "stop")
    tokens = BOARD_TOKENS + max(count, 1)
    if int(stop[0]) != tokens:
        raise ValueError("inputs: exact bounded count control required")
    mask = _owned_array(prepared.current_mask, "bool", (1, tokens), "current mask")
    if not mask[0, :BOARD_TOKENS].all() or (count and not mask.all()) or (count == 0 and mask[0, -1]):
        raise ValueError("inputs: current mask semantics drift")
    arrays = [a for owner in prepared.owners for a in (owner.key, owner.value, owner.record_features)]
    arrays += [prepared.feeds[f"record_offset_{i:03}"] for i in range(MAX_RECORDS)] + [stop, mask]
    payload = sum(array.nbytes for array in {id(a): a for a in arrays}.values())
    limit = _integer(prepared.maximum_owned_payload_bytes, 1, MAX_CPU_INPUT_BYTES, "CPU payload budget")
    if type(prepared.unique_owned_payload_bytes) is not int or payload != prepared.unique_owned_payload_bytes or payload > limit:
        raise ValueError("inputs: full unique backing payload accounting drift")
    return prepared
