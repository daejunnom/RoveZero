"""Opt-in approximate private warm artifacts and bounded CPU numeric checks.

Invoke this module explicitly; the legacy artifacts/CLI/export paths are not
modified. No optimizer, training, CUDA provider, or native seed-bank acceptance
is performed here. Synthetic numerical seeds test tensor math, not eligibility
to reuse a latent from a different Rules context or game.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from .artifacts import (annotate_public_memory_shapes, atomic_json, digest_file,
                        load_checkpoint, output_directory, read_rules_profile)
from .config import ModelConfig, SCHEMA, TASKS
from .onnx_warm import (PRIVATE_SEED_POLICY, WARM_ARTIFACT_SCHEMA,
                        WARM_GRAPH_SEMANTICS, WARM_LAYOUT, WARM_LAYOUT_REVISION,
                        audit_warm_pc_graph, build_warm_pc_graph)

WARM_GRAPH_FILE = WARM_LAYOUT + ".onnx"
WARM_MANIFEST_FILE = "warm_export.json"
MAX_WARM_MANIFEST_BYTES = 64 * 1024
# Combined serialized graph admission budget, not an ORT/process peak claim.
MAX_WARM_GRAPH_BYTES = 256 * 1024 * 1024
PRIVATE_INPUT_NAMES = ["role_is_critic", "memory_key", "memory_value", "memory_mask",
                       "candidates", "candidate_mask", "divergence_features",
                       "divergence_mask", "query", "initial_latent", "warm_start"]
PRIVATE_OUTPUT_NAMES = ["candidate_logits", "wdl_logits", "private_latent",
                        "divergence_logits", "is_critic"]


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file_stamp(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _read_immutable_bytes(path, max_bytes, expected_sha256=None):
    """One bounded regular-file read, retaining immutable content and provenance.

    lstat/fstat checks bind this read to the opened inode and reject symlinks,
    FIFOs and replacement during admission. O_NOFOLLOW/O_NONBLOCK are applied
    where the OS exposes them. Subsequent pathname changes cannot alter the
    returned bytes. This is source-file identity, not process memory evidence.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("invalid immutable artifact byte limit")
    if expected_sha256 is not None and not _sha(expected_sha256):
        raise ValueError("invalid immutable artifact digest")
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
        raise ValueError("artifact must be a bounded nonempty regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _file_stamp(opened) != _file_stamp(before) or not stat.S_ISREG(opened.st_mode):
            raise ValueError("artifact inode changed during admission")
        remaining = opened.st_size
        chunks, digest = [], hashlib.sha256()
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError("artifact was truncated during admission")
            chunks.append(block)
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1) or _file_stamp(os.fstat(descriptor)) != _file_stamp(opened) or _file_stamp(path.lstat()) != _file_stamp(opened):
            raise ValueError("artifact content/path changed during admission")
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError("immutable artifact identity mismatch")
        contents = b"".join(chunks)
        return contents, {"sha256": actual_sha256, "bytes": len(contents),
                          "device": opened.st_dev, "inode": opened.st_ino,
                          "mode": opened.st_mode, "mtime_ns": opened.st_mtime_ns,
                          "ctime_ns": opened.st_ctime_ns,
                          "no_follow_open_available": hasattr(os, "O_NOFOLLOW"),
                          "execution_source": "same_verified_immutable_bytes"}
    finally:
        os.close(descriptor)


def _read_warm_manifest(path):
    contents, provenance = _read_immutable_bytes(path, MAX_WARM_MANIFEST_BYTES)
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate warm manifest key")
            result[key] = value
        return result
    def nonfinite_constant(value):
        raise ValueError("nonfinite JSON constant in warm manifest: " + value)
    manifest = json.loads(contents.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=nonfinite_constant)
    if not isinstance(manifest, dict):
        raise ValueError("warm manifest must be one JSON object")
    return manifest, provenance


def _document_from_immutable_bytes(contents):
    import onnx
    if not isinstance(contents, bytes) or not 0 < len(contents) <= MAX_WARM_GRAPH_BYTES:
        raise ValueError("ONNX admission requires immutable verified bytes")
    document = onnx.load_model_from_string(contents)
    def tensor(value):
        if value.external_data or value.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("warm CPU graph external tensor files are unsupported")
    def nodes(values):
        for node in values:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    tensor(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for value in attribute.tensors:
                        tensor(value)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for value in attribute.graphs:
                        graph(value)
    def graph(value):
        for initializer in value.initializer:
            tensor(initializer)
        for initializer in value.sparse_initializer:
            tensor(initializer.values)
            tensor(initializer.indices)
        nodes(value.node)
    graph(document.graph)
    for function in document.functions:
        nodes(function.node)
    onnx.checker.check_model(document, full_check=True)
    return document


def _descriptors(values):
    import onnx
    return [{"name": value.name,
             "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
             "shape": [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]}
            for value in values]


def _expected_private_inputs():
    return [
        {"name": "role_is_critic", "dtype": "BOOL", "shape": []},
        {"name": "memory_key", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
        {"name": "memory_value", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
        {"name": "memory_mask", "dtype": "BOOL", "shape": ["batch", "memory_tokens"]},
        {"name": "candidates", "dtype": "INT64", "shape": ["batch", "candidates", 3]},
        {"name": "candidate_mask", "dtype": "BOOL", "shape": ["batch", "candidates"]},
        {"name": "divergence_features", "dtype": "FLOAT", "shape": ["batch", "divergences", 8]},
        {"name": "divergence_mask", "dtype": "BOOL", "shape": ["batch", "divergences"]},
        {"name": "query", "dtype": "FLOAT", "shape": ["batch", 16]},
        {"name": "initial_latent", "dtype": "FLOAT", "shape": ["batch", 16, 384]},
        {"name": "warm_start", "dtype": "BOOL", "shape": []},
    ]


def _expected_private_outputs():
    return [
        {"name": "candidate_logits", "dtype": "FLOAT", "shape": ["batch", "candidates"]},
        {"name": "wdl_logits", "dtype": "FLOAT", "shape": ["batch", 3]},
        {"name": "private_latent", "dtype": "FLOAT", "shape": ["batch", 16, 384]},
        {"name": "divergence_logits", "dtype": "FLOAT", "shape": ["batch", "divergences"]},
        {"name": "is_critic", "dtype": "BOOL", "shape": []},
    ]


def validate_warm_export_manifest(manifest, metadata):
    """Reject old domains, unregistered seed semantics, V, or missing graphs."""
    if (manifest.get("schema") != WARM_ARTIFACT_SCHEMA or manifest.get("model_semantics") != SCHEMA
            or manifest.get("layout") != WARM_LAYOUT or manifest.get("layout_revision") != WARM_LAYOUT_REVISION
            or manifest.get("config") != ModelConfig().to_dict()
            or manifest.get("config") != metadata.get("config")
            or manifest.get("checkpoint_sha256") != metadata.get("checkpoint_sha256")
            or not _sha(manifest.get("checkpoint_sha256"))):
        raise ValueError("warm export schema/layout/config/checkpoint mismatch")
    if (manifest.get("warm_graph_semantics") != WARM_GRAPH_SEMANTICS
            or manifest.get("private_seed_policy") != PRIVATE_SEED_POLICY
            or manifest.get("approximate") is not True
            or manifest.get("precision") != "fp32" or manifest.get("tf32") is not False
            or manifest.get("native_seed_owner_support") != "not_registered_by_python_export"
            or manifest.get("batch_mode") != "one_scalar_role_and_one_scalar_mode_per_physical_batch"):
        raise ValueError("warm export requires explicit approximate/seed/mode semantics")
    steps = manifest.get("training_steps")
    if (not isinstance(steps, int) or isinstance(steps, bool) or steps < 0
            or manifest.get("trained") is not (steps > 0)
            or steps != metadata.get("training_steps")
            or manifest.get("trained") is not metadata.get("trained")):
        raise ValueError("warm export training provenance mismatch")
    if (manifest.get("validator_present") is not False or manifest.get("roles") != ["proposer", "critic"]
            or manifest.get("task_names") != list(TASKS)):
        raise ValueError("warm export is strictly V-free P/C")
    if (manifest.get("rules_input_profile") != "rz-pals-rules-fields-v1"
            or manifest.get("learned_input_compatibility") != "unverified_declaration_only"
            or any(not _sha(manifest.get(key)) for key in (
                "rules_input_semantic_sha256", "rules_encoder_source_sha256",
                "rules_profile_descriptor_sha256", "rules_profile_canonical_sha256"))
            or not isinstance(manifest.get("rules_input_declaration"), dict)):
        raise ValueError("warm export requires actual verified Rules source declaration")
    declaration = manifest["rules_input_declaration"]
    fields = declaration.get("semantic_fields")
    features = []
    for key, count in (("metadata_features", 16), ("record_features", 16),
                       ("query_features", 16), ("divergence_features", 8)):
        values = declaration.get(key)
        if not isinstance(values, list) or len(values) != count or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("warm Rules feature vocabulary mismatch")
        features += values
    if (declaration.get("schema") != "rovezero.pals-rules-descriptor.v1"
            or declaration.get("config") != manifest["config"]
            or declaration.get("rules_input_profile") != manifest["rules_input_profile"]
            or declaration.get("learned_input_compatibility") != manifest["learned_input_compatibility"]
            or declaration.get("semantic_digest_algorithm") != "sha256_u64le_length_prefixed_utf8_fields"
            or not isinstance(fields, list) or len(fields) != 58
            or fields != [manifest["rules_input_profile"], declaration.get("rules_base_semantics"), *features]
            or not isinstance(fields[1], str) or not fields[1]):
        raise ValueError("warm Rules semantic fields/config mismatch")
    semantic = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8")
        semantic.update(len(encoded).to_bytes(8, "little"))
        semantic.update(encoded)
    canonical = json.dumps(declaration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if (len(canonical) > 65536 or semantic.hexdigest() != manifest["rules_input_semantic_sha256"]
            or declaration.get("rules_input_semantic_sha256") != semantic.hexdigest()
            or declaration.get("rules_encoder_source_sha256") != manifest["rules_encoder_source_sha256"]
            or hashlib.sha256(canonical).hexdigest() != manifest["rules_profile_canonical_sha256"]):
        raise ValueError("warm Rules canonical/semantic/source digest mismatch")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != 2 or [g.get("role") for g in graphs] != ["public", "shared_pc_warm"]:
        raise ValueError("warm export requires public and warm P/C graphs")
    for graph, filename in zip(graphs, ("public_memory.onnx", WARM_GRAPH_FILE)):
        if graph.get("file") != filename or not _sha(graph.get("sha256")) or graph.get("opset") != 17:
            raise ValueError("warm graph file/digest/opset mismatch")
    private = graphs[1]
    if private.get("inputs") != _expected_private_inputs() or private.get("outputs") != _expected_private_outputs():
        raise ValueError("warm graph seed/mode/role tensor shape/dtype mismatch")
    public = graphs[0]
    expected_public = [
        {"name": "board", "dtype": "INT64", "shape": ["batch", 64]},
        {"name": "metadata", "dtype": "FLOAT", "shape": ["batch", 16]},
        {"name": "records", "dtype": "FLOAT", "shape": ["batch", "records", 16]},
        {"name": "record_mask", "dtype": "BOOL", "shape": ["batch", "records"]},
    ]
    expected_memory = [
        {"name": "memory_key", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
        {"name": "memory_value", "dtype": "FLOAT", "shape": ["batch", 2, "memory_tokens", 64]},
        {"name": "memory_mask", "dtype": "BOOL", "shape": ["batch", "memory_tokens"]},
    ]
    if public.get("inputs") != expected_public or public.get("outputs") != expected_memory:
        raise ValueError("warm public graph changed the existing input/KV profile")
    return ["proposer", "critic"]


def export_warm_checkpoint(checkpoint, directory, rules_profile_json):
    """Explicit new artifact domain; existing checkpoint state_dict stays intact."""
    import onnx
    import torch
    from .model import fixture_input
    model, metadata = load_checkpoint(checkpoint)
    if rules_profile_json is None:
        raise ValueError("warm export requires the native Rules profile declaration")
    rules_profile = read_rules_profile(rules_profile_json, model.config)
    output = output_directory(directory)
    for filename in ("public_memory.onnx", WARM_GRAPH_FILE, WARM_MANIFEST_FILE):
        destination = output / filename
        if destination.exists() or destination.with_suffix(destination.suffix + ".partial").exists():
            raise FileExistsError(f"warm artifact already exists: {filename}")
    free = model.without_validator()
    data = fixture_input()
    data.validate(free.config)
    public_file = output / "public_memory.onnx"
    public_temporary = public_file.with_suffix(".onnx.partial")
    public_names = ["board", "metadata", "records", "record_mask"]
    axes = {name: {0: "batch"} for name in public_names}
    axes["records"][1] = axes["record_mask"][1] = "records"
    axes.update({"memory_key": {0: "batch", 2: "memory_tokens"},
                 "memory_value": {0: "batch", 2: "memory_tokens"},
                 "memory_mask": {0: "batch", 1: "memory_tokens"}})
    with torch.no_grad():
        torch.onnx.export(free.public_encoder, data.public_args(), str(public_temporary),
                          input_names=public_names, output_names=["memory_key", "memory_value", "memory_mask"],
                          dynamic_axes=axes, opset_version=17, dynamo=False,
                          export_params=True, do_constant_folding=True)
    public = onnx.load(str(public_temporary), load_external_data=False)
    annotate_public_memory_shapes(public, free.config)
    private, ownership = build_warm_pc_graph(free)
    graphs = []
    for document, filename, role in ((public, "public_memory.onnx", "public"),
                                     (private, WARM_GRAPH_FILE, "shared_pc_warm")):
        destination = output / filename
        temporary = destination.with_suffix(".onnx.partial")
        onnx.checker.check_model(document, full_check=True)
        properties = {
            "schema": WARM_ARTIFACT_SCHEMA, "model_semantics": SCHEMA,
            "layout": WARM_LAYOUT, "layout_revision": str(WARM_LAYOUT_REVISION), "role": role,
            "checkpoint_sha256": metadata["checkpoint_sha256"], "precision": "fp32",
            "trained": str(metadata["trained"]).lower(), "training_steps": str(metadata["training_steps"]),
            "expected_ort": "1.22.0", "approximate": "true",
            "warm_graph_semantics": WARM_GRAPH_SEMANTICS, "private_seed_policy": PRIVATE_SEED_POLICY,
            "public_kv": "shared_role_neutral",
        }
        properties.update({key: rules_profile[key] for key in (
            "rules_input_profile", "rules_input_semantic_sha256", "rules_encoder_source_sha256")})
        onnx.helper.set_model_props(document, properties)
        onnx.save(document, str(temporary))
        temporary.replace(destination)
        graphs.append({"file": filename, "role": role, "sha256": digest_file(destination),
                       "opset": 17, "inputs": _descriptors(document.graph.input),
                       "outputs": _descriptors(document.graph.output)})
    manifest = {
        "schema": WARM_ARTIFACT_SCHEMA, "model_semantics": SCHEMA,
        "layout": WARM_LAYOUT, "layout_revision": WARM_LAYOUT_REVISION,
        "checkpoint_sha256": metadata["checkpoint_sha256"], "config": free.config.to_dict(),
        "trained": metadata["trained"], "training_steps": metadata["training_steps"],
        "precision": "fp32", "tf32": False, "approximate": True,
        "roles": ["proposer", "critic"], "validator_present": False, "task_names": list(TASKS),
        "warm_graph_semantics": WARM_GRAPH_SEMANTICS, "private_seed_policy": PRIVATE_SEED_POLICY,
        "batch_mode": "one_scalar_role_and_one_scalar_mode_per_physical_batch",
        "native_seed_owner_support": "not_registered_by_python_export",
        "graphs": graphs, "ownership": ownership, "numeric_status": "not_run", "cuda": "not_run",
        "torch_version": torch.__version__, "expected_ort": "1.22.0",
        "rust_ort_crate": "2.0.0-rc.10", **rules_profile,
    }
    validate_warm_export_manifest(manifest, metadata)
    atomic_json(output / WARM_MANIFEST_FILE, manifest)
    return manifest


def validate_warm_feed(feed, max_batch=16):
    """Boundary validation used before ORT; ONNX alone is not a finite checker."""
    import numpy as np
    if set(feed) != set(PRIVATE_INPUT_NAMES) or not isinstance(max_batch, int) or isinstance(max_batch, bool) or not 1 <= max_batch <= 16:
        raise ValueError("warm feed names/batch admission limit mismatch")
    if any(not isinstance(value, np.ndarray) for value in feed.values()):
        raise ValueError("warm feed requires explicit typed numpy tensors")
    for name in ("role_is_critic", "warm_start"):
        if feed[name].shape != () or feed[name].dtype != np.bool_:
            raise ValueError("warm role/mode must be scalar BOOL")
    for name in ("memory_mask", "candidate_mask", "divergence_mask"):
        if feed[name].dtype != np.bool_:
            raise ValueError("warm mask requires BOOL")
    key, value, mask = (feed[name] for name in ("memory_key", "memory_value", "memory_mask"))
    b = key.shape[0] if key.ndim == 4 else 0
    s = key.shape[2] if key.ndim == 4 else 0
    if not 1 <= b <= max_batch or not 67 <= s <= 194 or key.shape != (b, 2, s, 64) or value.shape != key.shape or mask.shape != (b, s):
        raise ValueError("warm feed public KV/mask shape or capacity mismatch")
    candidates, divergences = feed["candidates"], feed["divergence_features"]
    c = candidates.shape[1] if candidates.ndim == 3 else 0
    d = divergences.shape[1] if divergences.ndim == 3 else 0
    if not 1 <= c <= 256 or candidates.shape != (b, c, 3) or feed["candidate_mask"].shape != (b, c):
        raise ValueError("warm candidate shape/capacity mismatch")
    if candidates.dtype != np.int64 or np.any((candidates[:, :, :2] < 0) | (candidates[:, :, :2] > 63)) or np.any((candidates[:, :, 2] < 0) | (candidates[:, :, 2] > 4)):
        raise ValueError("warm candidate move code/dtype mismatch")
    if np.any(feed["candidate_mask"] & (candidates[:, :, 0] == candidates[:, :, 1])):
        raise ValueError("warm active candidate has identical from/to")
    if not 1 <= d <= 128 or divergences.shape != (b, d, 8) or feed["divergence_mask"].shape != (b, d):
        raise ValueError("warm divergence shape/capacity mismatch")
    if feed["query"].shape != (b, 16) or feed["initial_latent"].shape != (b, 16, 384):
        raise ValueError("warm query/initial latent shape mismatch")
    for name in ("memory_key", "memory_value", "divergence_features", "query", "initial_latent"):
        if feed[name].dtype != np.float32 or not np.isfinite(feed[name]).all():
            raise ValueError("warm feed requires finite FP32")
    if not np.all(mask[:, :66]):
        raise ValueError("warm public board/metadata prefix must remain active")


def _cpu_session(source, profile_prefix=None):
    import onnxruntime as ort
    if not isinstance(source, bytes) or not 0 < len(source) <= MAX_WARM_GRAPH_BYTES:
        raise ValueError("warm CPU execution requires bounded immutable graph bytes")
    if ort.__version__ != "1.22.0":
        raise ValueError("warm numeric check requires pinned ORT1.22.0")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    if profile_prefix is not None:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    session = ort.InferenceSession(source, sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("warm numeric check must use only CPUExecutionProvider")
    return session


def _numpy(values):
    return [value.detach().cpu().numpy() for value in values]


def _feed(data, memory, seed, role, mode):
    import numpy as np
    if role not in ("proposer", "critic") or not isinstance(mode, (bool, np.bool_)):
        raise ValueError("warm feed requires explicit P/C role and bool mode")
    names = PRIVATE_INPUT_NAMES[1:9]
    values = [*memory, *_numpy((data.candidates, data.candidate_mask,
                               data.divergence_features, data.divergence_mask, data.query))]
    feed = dict(zip(names, values))
    feed.update({"role_is_critic": np.asarray(role == "critic", dtype=np.bool_),
                 "initial_latent": np.asarray(seed),
                 "warm_start": np.asarray(mode, dtype=np.bool_)})
    validate_warm_feed(feed)
    return feed


def _compare(expected, actual, candidate_mask, maxima):
    import numpy as np
    if len(expected) != 5 or len(actual) != 5:
        raise ValueError("warm private result count mismatch")
    for name, a, b in zip(PRIVATE_OUTPUT_NAMES, expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("warm output shape/dtype mismatch: " + name)
        if name == "is_critic":
            if not np.array_equal(a, b):
                raise ValueError("warm output role tag mismatch")
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite warm output")
        np.testing.assert_allclose(b, a, atol=1e-4, rtol=1e-3, err_msg=name)
        maxima[name] = max(maxima.get(name, 0.0), float(np.max(np.abs(a - b))))
    for name, index, mask in (("policy", 0, candidate_mask), ("wdl", 1, None)):
        probabilities = []
        for output in (expected, actual):
            logits = output[index].astype(np.float64)
            if mask is not None:
                logits = np.where(mask, logits, -np.inf)
                valid = np.any(mask, axis=1)
                probabilities.append(np.zeros_like(logits))
                if np.any(valid):
                    shifted = logits[valid] - np.max(logits[valid], axis=1, keepdims=True)
                    values = np.exp(shifted)
                    probabilities[-1][valid] = values / values.sum(axis=1, keepdims=True)
            else:
                values = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities.append(values / values.sum(axis=1, keepdims=True))
        difference = float(np.max(np.abs(probabilities[0] - probabilities[1])))
        if difference > 1e-4:
            raise ValueError("warm normalized " + name + " exceeds tolerance")
        maxima[name] = max(maxima.get(name, 0.0), difference)


def _reference(model, data, memory, role, seed, mode):
    import numpy as np
    import torch
    with torch.no_grad():
        result = model.warm_role_graph(role)(
            *data.role_args(memory), torch.from_numpy(seed), torch.tensor(mode, dtype=torch.bool))
    outputs = _numpy(result)
    if role == "proposer":
        outputs.append(np.zeros(data.divergence_mask.shape, dtype=np.float32))
    outputs.append(np.asarray(role == "critic", dtype=np.bool_))
    return outputs


def verify_warm_cpu_routes(graph_bytes, feed):
    """CPU role/mode witness consuming the same verified immutable graph bytes.

    This observes four finite CPU runs with optimization disabled. It does not
    establish optimized production routing, CUDA execution, or seed admission.
    The legacy helper stringifies paths, so it cannot consume immutable bytes.
    Its node-selection checks are reused here without changing that legacy API.
    No raw profiling files are retained; their digests and node evidence remain.
    """
    import numpy as np
    import tempfile
    document = _document_from_immutable_bytes(graph_bytes)
    modes = [node for node in document.graph.node if node.name == "private_warm_mode"]
    if len(modes) != 1:
        raise ValueError("warm CPU witness lacks unique mode route")
    fresh = next(attribute.g for attribute in modes[0].attribute if attribute.name == "else_branch")
    initial_route = fresh.node[0]
    initialization_nodes = {}
    for attribute in initial_route.attribute:
        role = "critic" if attribute.name == "then_branch" else "proposer"
        initialization_nodes[role] = {node.name + "_kernel_time" for node in attribute.g.node}
    reports = []
    with tempfile.TemporaryDirectory(prefix="rovezero-pals-warm-route-") as directory:
        for mode in (False, True):
            for role in ("proposer", "critic"):
                current = {**feed, "warm_start": np.asarray(mode, dtype=np.bool_),
                           "role_is_critic": np.asarray(role == "critic", dtype=np.bool_)}
                validate_warm_feed(current)
                session = _cpu_session(graph_bytes, Path(directory) / f"{role}-{mode}")
                result = session.run(None, current)
                profile = Path(session.end_profiling())
                # Profiling JSON is diagnostic output of this exact owned CPU
                # session, still bounded and non-following on admission.
                encoded, provenance = _read_immutable_bytes(profile, 16 * 1024 * 1024)
                events = json.loads(encoded)
                nodes = [event.get("name", "") for event in events
                         if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
                selected = [name for name in nodes if name.startswith(role + "_")]
                other = "proposer" if role == "critic" else "critic"
                inactive = [name for name in nodes if name.startswith(other + "_")]
                shared = [name for name in nodes if name.startswith("shared_pc_if_")]
                if not selected or inactive or not shared:
                    raise ValueError("warm CPU profile does not prove selected-only private execution")
                if result[4].shape != () or result[4].dtype != np.bool_ or bool(result[4]) != (role == "critic"):
                    raise ValueError("warm CPU profile role tag mismatch")
                executed = set(selected) & initialization_nodes[role]
                if (mode and executed) or (not mode and not executed):
                    raise ValueError("CPU mode route did not isolate legacy Fresh initialization")
                reports.append({"role": role, "warm_start": mode,
                                "selected_private_nodes": selected, "inactive_private_nodes": inactive,
                                "shared_nodes_executed": len(shared),
                                "fresh_initial_nodes_executed": sorted(executed),
                                "profile_sha256": provenance["sha256"], "profile_raw_retained": False,
                                "provider": "CPUExecutionProvider",
                                "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
                                "execution_source": "same_verified_immutable_bytes",
                                "graph_optimization": "disabled_for_routing_witness"})
                del session
    return {"status": "passed", "cases": reports,
            "scope": "unoptimized_cpu_if_execution_only", "cuda": "not_run",
            "production_optimized_routing": "requires_native_check"}


def numeric_check_warm(checkpoint, directory):
    """Independent Torch/legacy-ONNX/new-ONNX correctness; never a benchmark."""
    import numpy as np
    import torch
    from .model import fixture_input
    from .onnx_shared import build_shared_pc_graph
    model, metadata = load_checkpoint(checkpoint)
    model = model.without_validator()
    output = output_directory(directory)
    manifest, manifest_provenance = _read_warm_manifest(output / WARM_MANIFEST_FILE)
    validate_warm_export_manifest(manifest, metadata)
    # Revalidate the Rules descriptor itself, not only digest-shaped strings.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rovezero-pals-warm-declaration-") as temporary:
        descriptor = Path(temporary) / "rules.json"
        descriptor.write_text(json.dumps(manifest["rules_input_declaration"], ensure_ascii=False), encoding="utf-8")
        verified = read_rules_profile(descriptor, model.config)
        for key in ("rules_input_profile", "rules_input_semantic_sha256", "rules_encoder_source_sha256",
                    "rules_profile_canonical_sha256", "learned_input_compatibility"):
            if verified[key] != manifest[key]:
                raise ValueError("warm registered Rules declaration mismatch")
    graph_sources, graph_provenance = {}, []
    remaining = MAX_WARM_GRAPH_BYTES
    for graph in manifest["graphs"]:
        contents, provenance = _read_immutable_bytes(output / graph["file"], remaining, graph["sha256"])
        remaining -= len(contents)
        graph_sources[graph["role"]] = contents
        graph_provenance.append({"file": graph["file"], **provenance})
    # All audit/session/profile consumers share these exact admitted bytes;
    # none reopen the mutable registered graph paths for execution.
    _document_from_immutable_bytes(graph_sources["public"])
    legacy_document, legacy_ownership = build_shared_pc_graph(model)
    registered_private = _document_from_immutable_bytes(graph_sources["shared_pc_warm"])
    audit_warm_pc_graph(registered_private, legacy_document, legacy_ownership)
    legacy = _cpu_session(legacy_document.SerializeToString())
    public = _cpu_session(graph_sources["public"])
    warm = _cpu_session(graph_sources["shared_pc_warm"])
    reports, maxima = [], {}
    # Fixed finite matrix, including explicit empty padding and dynamic batch.
    for records, candidates, divergences, batch in ((1, 1, 1, 1), (3, 4, 2, 1), (2, 7, 3, 2)):
        data = fixture_input(records, candidates, divergences, batch)
        if records == candidates == divergences == 1:
            data.record_mask.zero_()
            data.candidate_mask.zero_()
            data.divergence_mask.zero_()
        data.validate(model.config)
        with torch.no_grad():
            reference_memory = model.public_encoder(*data.public_args())
        actual_memory = public.run(None, dict(zip(("board", "metadata", "records", "record_mask"), _numpy(data.public_args()))))
        for a, b in zip(_numpy(reference_memory), actual_memory):
            if a.dtype == np.bool_:
                if not np.array_equal(a, b):
                    raise ValueError("warm public mask changed")
            else:
                np.testing.assert_allclose(b, a, atol=1e-4, rtol=1e-3)
        for role in ("proposer", "critic"):
            zero = np.zeros((batch, 16, 384), dtype=np.float32)
            feed = _feed(data, actual_memory, zero, role, False)
            legacy_result = legacy.run(None, {name: value for name, value in feed.items()
                                              if name not in ("initial_latent", "warm_start")})
            fresh_result = warm.run(None, feed)
            fresh_torch = _reference(model, data, reference_memory, role, zero, False)
            _compare(fresh_torch, legacy_result, feed["candidate_mask"], maxima)
            _compare(legacy_result, fresh_result, feed["candidate_mask"], maxima)
            # Numerical fixture only: this latent has not been committed by a
            # real Rust accepted-output/fence owner and is not training data.
            seed = np.ascontiguousarray(fresh_torch[2])
            ignored_seed_result = warm.run(None, _feed(data, actual_memory, seed, role, False))
            for a, b in zip(fresh_result, ignored_seed_result):
                if not np.array_equal(a, b):
                    raise ValueError("Fresh graph unexpectedly consumes private seed")
            warm_result = warm.run(None, _feed(data, actual_memory, seed, role, True))
            warm_torch = _reference(model, data, reference_memory, role, seed, True)
            _compare(warm_torch, warm_result, feed["candidate_mask"], maxima)
            perturbations = []
            for index in (0, 3071, 6143):
                changed = seed.copy()
                changed.reshape(batch, 6144)[:, index] += np.float32(0.25)
                result = warm.run(None, _feed(data, actual_memory, changed, role, True))
                reference = _reference(model, data, reference_memory, role, changed, True)
                _compare(reference, result, feed["candidate_mask"], maxima)
                if np.array_equal(warm_result[2], result[2]):
                    raise ValueError("Warm graph does not consume perturbed seed coordinate")
                perturbations.append(index)
            reports.append({"records": records, "candidates": candidates, "divergences": divergences,
                            "batch": batch, "role": role, "fresh_legacy_new_and_torch": "passed",
                            "warm_new_and_torch": "passed", "fresh_seed_independent": True,
                            "warm_seed_coordinates_tested": perturbations,
                            "seed_provenance": "numerical_fixture_only_not_native_accepted"})
    route_evidence = verify_warm_cpu_routes(graph_sources["shared_pc_warm"], feed)
    report = {"schema": "rovezero.pals-private-warm-cpu-check.v1", "status": "passed",
              "scope": "fp32_tensor_math_only", "checkpoint_sha256": metadata["checkpoint_sha256"],
              "graphs": graph_provenance, "manifest": manifest_provenance,
              "admission_limits": {"manifest_bytes": MAX_WARM_MANIFEST_BYTES,
                                   "combined_serialized_graph_bytes": MAX_WARM_GRAPH_BYTES,
                                   "scope": "serialized_input_admission_not_process_peak"},
              "cases": reports, "max_abs_difference": maxima,
              "raw_atol": 1e-4, "raw_rtol": 1e-3, "policy_wdl_max_atol": 1e-4,
              "provider": "CPUExecutionProvider", "ort_version": "1.22.0",
              "optimizer_steps": 0, "cuda": "not_run", "native_warm_acceptance": "not_run",
              "rules_seed_eligibility": "not_proven_by_numerical_fixture",
              "cpu_routing_witness": route_evidence}
    atomic_json(output / "numeric_warm_cpu.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Separate opt-in P/C approximate warm artifacts; CPU only")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--rules-profile-json", required=True)
    numeric = commands.add_parser("numeric-check")
    numeric.add_argument("--checkpoint", required=True)
    numeric.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "export":
        result = export_warm_checkpoint(arguments.checkpoint, arguments.output, arguments.rules_profile_json)
    else:
        result = numeric_check_warm(arguments.checkpoint, arguments.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
