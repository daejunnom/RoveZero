"""Owned external artifacts, initialization provenance and independent export."""
import hashlib
import json
import os
from pathlib import Path

from .config import ModelConfig, SCHEMA, TASKS


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sha256_text(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def read_rules_profile(path, config):
    """Verify the source declaration's semantic fields, not learned suitability."""
    encoded = Path(path).read_bytes()
    if len(encoded) > 65536:
        raise ValueError("Rules profile exceeds bounded declaration size")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError("duplicate Rules profile key")
            result[key] = value
        return result
    declaration = json.loads(encoded, object_pairs_hook=unique_pairs)
    if declaration.get("schema") != "rovezero.pals-rules-descriptor.v1" or declaration.get("rules_input_profile") != "rz-pals-rules-fields-v1" or declaration.get("config") != config.to_dict():
        raise ValueError("Rules profile schema/config mismatch")
    for key in ("rules_input_semantic_sha256", "rules_encoder_source_sha256"):
        if not _sha256_text(declaration.get(key)): raise ValueError("Rules declaration digest missing")
    if declaration.get("learned_input_compatibility") != "unverified_declaration_only":
        raise ValueError("source declaration cannot assert learned compatibility")
    features = []
    for key, count in (("metadata_features", 16), ("record_features", 16), ("query_features", 16), ("divergence_features", 8)):
        values = declaration.get(key)
        if not isinstance(values, list) or len(values) != count or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("Rules feature vocabulary mismatch")
        features += values
    fields = declaration.get("semantic_fields")
    if declaration.get("semantic_digest_algorithm") != "sha256_u64le_length_prefixed_utf8_fields" or not isinstance(fields, list) or len(fields) != 58 or fields[0] != declaration["rules_input_profile"] or not isinstance(fields[1], str) or not fields[1] or fields[1] != declaration.get("rules_base_semantics") or fields[2:] != features:
        raise ValueError("Rules semantic fields/canonicalization mismatch")
    semantic = hashlib.sha256()
    for field in fields:
        value = field.encode("utf-8")
        semantic.update(len(value).to_bytes(8, "little"))
        semantic.update(value)
    if semantic.hexdigest() != declaration["rules_input_semantic_sha256"]:
        raise ValueError("Rules semantic digest disagrees with declared fields")
    canonical = json.dumps(declaration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"rules_input_profile": declaration["rules_input_profile"],
            "rules_input_semantic_sha256": semantic.hexdigest(),
            "rules_encoder_source_sha256": declaration["rules_encoder_source_sha256"],
            "rules_profile_descriptor_sha256": hashlib.sha256(encoded).hexdigest(),
            "rules_profile_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
            "rules_input_declaration": declaration, "learned_input_compatibility": "unverified_declaration_only"}


def output_directory(path):
    """Refuse source-tree weights/reports and resolve symlinks before creation."""
    result = Path(path).expanduser().resolve()
    source = Path(__file__).resolve()
    repository = next((p for p in source.parents if (p / ".git").exists()), None)
    if repository is not None and (result == repository or repository in result.parents):
        raise ValueError("generated model artifacts must be outside repository")
    result.mkdir(parents=True, exist_ok=True)
    marker = result / "pals-artifact-owner.json"
    if marker.exists():
        owner = json.loads(marker.read_text(encoding="utf-8"))
        if owner.get("schema") != SCHEMA or owner.get("owner") != "rz-pals-model":
            raise ValueError("output directory belongs to another workload")
    else:
        atomic_json(marker, {"schema": SCHEMA, "owner": "rz-pals-model"})
    return result


def atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite registered artifact: {path.name}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError("incomplete artifact already exists; inspect before retry")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def initialize_checkpoint(directory, seed):
    import torch
    from .model import initialize
    output = output_directory(directory)
    checkpoint = output / "untrained.pt"
    if checkpoint.exists() or checkpoint.with_suffix(".pt.partial").exists():
        raise FileExistsError("checkpoint already exists")
    model = initialize(seed)
    temporary = checkpoint.with_suffix(".pt.partial")
    # Only tensors and JSON-like primitives; loading uses weights_only=True.
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict(), "seed": seed,
                "schema": SCHEMA, "training_steps": 0, "trained": False}, temporary)
    temporary.replace(checkpoint)
    metadata = {"schema": SCHEMA, "checkpoint": checkpoint.name, "checkpoint_sha256": digest_file(checkpoint),
                "config": model.config.to_dict(), "seed": seed, "trained": False, "training_steps": 0,
                "external_weights": False, "external_teacher": False, "precision": "fp32", "tf32": False,
                "torch_version": torch.__version__, "parameter_count": sum(p.numel() for p in model.parameters()),
                "roles": list(model.experts), "status": "random_initialization_only"}
    atomic_json(output / "checkpoint.json", metadata)
    return metadata


def load_checkpoint(path):
    import torch
    from .model import PalsModel
    path = Path(path).resolve()
    metadata_path = path.with_name("checkpoint.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA or metadata.get("checkpoint_sha256") != digest_file(path):
        raise ValueError("checkpoint identity mismatch")
    if metadata.get("external_weights") is not False or metadata.get("external_teacher") is not False:
        raise ValueError("first PALS profile requires explicit own-weight/no-external-teacher provenance")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    steps = checkpoint.get("training_steps")
    if checkpoint.get("schema") != SCHEMA or not isinstance(steps, int) or isinstance(steps, bool) or steps < 0 or checkpoint.get("trained") is not (steps > 0):
        raise ValueError("checkpoint must declare consistent training provenance")
    if metadata.get("training_steps") != steps or metadata.get("trained") is not checkpoint["trained"]:
        raise ValueError("checkpoint/receipt training provenance mismatch")
    if checkpoint.get("config") != metadata.get("config"):
        raise ValueError("checkpoint configuration mismatch")
    config = ModelConfig(**checkpoint["config"])
    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(checkpoint["seed"])
        model = PalsModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata


def validate_export_manifest(manifest, metadata):
    """Never accept public-only execution as P/C neural acceptance."""
    from .onnx_shared import ARTIFACT_SCHEMA, SHARED_LAYOUT, SHARED_LAYOUT_REVISION
    shared = manifest.get("schema") == ARTIFACT_SCHEMA
    if manifest.get("schema") not in (SCHEMA, ARTIFACT_SCHEMA) or manifest.get("config") != metadata.get("config") or manifest.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
        raise ValueError("export schema/config/checkpoint mismatch")
    if shared and (manifest.get("model_semantics") != SCHEMA or manifest.get("layout") != SHARED_LAYOUT or manifest.get("layout_revision") != SHARED_LAYOUT_REVISION):
        raise ValueError("export shared layout/semantic revision mismatch")
    if shared and (manifest.get("rules_input_profile") != "rz-pals-rules-fields-v1" or not _sha256_text(manifest.get("rules_input_semantic_sha256")) or not _sha256_text(manifest.get("rules_encoder_source_sha256"))):
        raise ValueError("shared production layout requires Rules semantic profile")
    if manifest.get("trained") is not metadata.get("trained") or manifest.get("training_steps") != metadata.get("training_steps"):
        raise ValueError("export training provenance mismatch")
    if manifest.get("validator_present") is False:
        roles = ["proposer", "critic"]
    elif manifest.get("validator_present") is True:
        roles = ["proposer", "critic", "validator"]
    else:
        raise ValueError("export must declare validator presence")
    if manifest.get("roles") != roles or manifest.get("task_names") != list(TASKS):
        raise ValueError("export role/task vocabulary mismatch")
    if shared and roles != ["proposer", "critic"]:
        raise ValueError("shared_pc_if requires V-free P/C roles")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != (2 if shared else len(roles) + 1):
        raise ValueError("export required graph count mismatch")
    graph_roles = [graph.get("role") for graph in graphs]
    if graph_roles != (["public", "shared_pc"] if shared else ["public", *roles]):
        raise ValueError("export requires one public graph and exactly the declared private roles")
    for graph in graphs:
        role = graph["role"]
        expected_file = "public_memory.onnx" if role == "public" else ("shared_pc_if.onnx" if role == "shared_pc" else f"role_{role}.onnx")
        digest = graph.get("sha256")
        if graph.get("file") != expected_file or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or graph.get("opset") != 17:
            raise ValueError("export graph file/digest/opset mismatch")
        if role == "public":
            expected_inputs = ["board", "metadata", "records", "record_mask"]
            expected_outputs = ["memory_key", "memory_value", "memory_mask"]
        else:
            expected_inputs = (["role_is_critic"] if role == "shared_pc" else []) + ["memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask"]
            if role in ("critic", "shared_pc"): expected_inputs += ["divergence_features", "divergence_mask"]
            expected_inputs += ["query"]
            expected_outputs = ["candidate_logits", "wdl_logits", "private_latent"]
            if role in ("critic", "shared_pc"): expected_outputs += ["divergence_logits"]
            if role == "shared_pc": expected_outputs += ["is_critic"]
            if role == "validator": expected_outputs += ["task_logits"]
        for key, expected in (("inputs", expected_inputs), ("outputs", expected_outputs)):
            descriptors = graph.get(key)
            if not isinstance(descriptors, list) or [value.get("name") for value in descriptors] != expected:
                raise ValueError("export graph required tensor names mismatch")
    return roles


def annotate_public_memory_shapes(document, config):
    """Restore the architecture's literal KV axes lost by legacy transpose export.

    Batch and token axes stay symbolic. An already contradictory literal shape
    is rejected, never rewritten into an apparently supported descriptor.
    """
    config.validate()
    outputs = {value.name: value for value in document.graph.output}
    for name in ("memory_key", "memory_value"):
        if name not in outputs:
            raise ValueError("public export is missing required memory output")
        dimensions = outputs[name].type.tensor_type.shape.dim
        if len(dimensions) != 4:
            raise ValueError("public KV output must have rank four")
        for axis, expected in ((1, config.kv_heads), (3, config.head_dimension)):
            dimension = dimensions[axis]
            if dimension.dim_value and dimension.dim_value != expected:
                raise ValueError("public KV output contradicts fixed model architecture")
            dimension.ClearField("dim_param")
            dimension.dim_value = expected


def export_checkpoint(checkpoint, directory, include_validator=False, layout="separate_pc", rules_profile_json=None):
    import onnx
    import torch
    from .model import fixture_input
    from .onnx_shared import ARTIFACT_SCHEMA, SHARED_LAYOUT, SHARED_LAYOUT_REVISION, build_shared_pc_graph
    if layout not in ("separate_pc", SHARED_LAYOUT) or (layout == SHARED_LAYOUT and include_validator):
        raise ValueError("unsupported layout or validator in V-free shared_pc_if")
    model, metadata = load_checkpoint(checkpoint)
    if layout == SHARED_LAYOUT and rules_profile_json is None:
        raise ValueError("shared_pc_if requires --rules-profile-json from the native Rules descriptor")
    rules_profile = read_rules_profile(rules_profile_json, model.config) if rules_profile_json is not None else None
    output = output_directory(directory)
    if not include_validator:
        model = model.without_validator()
    data = fixture_input()
    data.validate(model.config)
    graphs = []
    public_args = data.public_args()
    # CPU export does not execute TF32. Never change caller autograd/backend
    # globals; a future preparation consumer may export then build its losses.
    with torch.no_grad():
        memory = model.public_encoder(*public_args)
    for value in memory[:2]:
        if value.ndim != 4 or value.shape[1] != model.config.kv_heads or value.shape[3] != model.config.head_dimension:
            raise ValueError("actual public KV shape contradicts fixed model architecture")

    artifact_schema = ARTIFACT_SCHEMA if layout == SHARED_LAYOUT else SCHEMA
    def save_document(document, temporary, destination, filename, role):
        onnx.checker.check_model(document, full_check=True)
        properties = {"schema": artifact_schema, "model_semantics": SCHEMA,
            "layout": layout, "layout_revision": str(SHARED_LAYOUT_REVISION if layout == SHARED_LAYOUT else 1), "role": role,
            "checkpoint_sha256": metadata["checkpoint_sha256"], "training_steps": str(metadata["training_steps"]), "trained": str(metadata["trained"]).lower(),
            "precision": "fp32", "expected_ort": "1.22.0", "public_kv": "shared_role_neutral"}
        if rules_profile is not None:
            properties.update({key: rules_profile[key] for key in ("rules_input_profile", "rules_input_semantic_sha256", "rules_encoder_source_sha256")})
        onnx.helper.set_model_props(document, properties)
        onnx.save(document, str(temporary))
        temporary.replace(destination)
        inputs = [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                   "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in document.graph.input]
        outputs = [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                    "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in document.graph.output]
        graphs.append({"file": filename, "sha256": digest_file(destination), "role": role,
                       "inputs": inputs, "outputs": outputs, "opset": 17})

    def export(graph, args, filename, names, output_names, dynamic_axes, role):
        destination = output / filename
        temporary = destination.with_suffix(".onnx.partial")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"graph already exists: {filename}")
        with torch.no_grad():
            torch.onnx.export(graph, args, str(temporary), input_names=names, output_names=output_names,
                              dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
                              export_params=True, do_constant_folding=True)
        document = onnx.load(str(temporary), load_external_data=False)
        if role == "public":
            annotate_public_memory_shapes(document, model.config)
        save_document(document, temporary, destination, filename, role)

    public_names = ["board", "metadata", "records", "record_mask"]
    axes = {name: {0: "batch"} for name in public_names}
    axes["records"][1] = axes["record_mask"][1] = "records"
    axes.update({"memory_key": {0: "batch", 2: "memory_tokens"}, "memory_value": {0: "batch", 2: "memory_tokens"}, "memory_mask": {0: "batch", 1: "memory_tokens"}})
    export(model.public_encoder, public_args, "public_memory.onnx", public_names,
           ["memory_key", "memory_value", "memory_mask"], axes, "public")
    names = ["memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask", "divergence_features", "divergence_mask", "query"]
    for role in (model.experts if layout == "separate_pc" else ()):
        outputs = ["candidate_logits", "wdl_logits", "private_latent"]
        if role == "critic": outputs.append("divergence_logits")
        if role == "validator": outputs.append("task_logits")
        axes = {name: {0: "batch"} for name in names + outputs}
        axes["memory_key"][2] = axes["memory_value"][2] = "memory_tokens"
        axes["memory_mask"][1] = "memory_tokens"
        axes["candidates"][1] = axes["candidate_mask"][1] = axes["candidate_logits"][1] = "candidates"
        axes["divergence_features"][1] = axes["divergence_mask"][1] = "divergences"
        if role == "critic": axes["divergence_logits"][1] = "divergences"
        export(model.role_graph(role), data.role_args(memory), f"role_{role}.onnx", names, outputs, axes, role)
    reader_bank = None
    if layout == SHARED_LAYOUT:
        filename = "shared_pc_if.onnx"
        destination = output / filename
        temporary = destination.with_suffix(".onnx.partial")
        if destination.exists() or temporary.exists():
            raise FileExistsError("shared P/C graph already exists")
        document, reader_bank = build_shared_pc_graph(model)
        save_document(document, temporary, destination, filename, "shared_pc")
    manifest = {"schema": artifact_schema, "config": model.config.to_dict(), "checkpoint_sha256": metadata["checkpoint_sha256"],
                "trained": metadata["trained"], "training_steps": metadata["training_steps"], "roles": list(model.experts), "validator_present": include_validator,
                "graphs": graphs, "runtime": {"onnxruntime": "1.22.0", "rust_ort": "2.0.0-rc.10", "compatibility": "requires_actual_numeric_check"},
                "candidate_promotion": {"0": "none", "1": "queen", "2": "rook", "3": "bishop", "4": "knight"},
                "wdl_perspective": "input_side_to_move", "task_names": list(TASKS), "numeric_status": "not_run", "cuda_status": "not_run"}
    if layout == SHARED_LAYOUT:
        manifest.update({"model_semantics": SCHEMA, "layout": layout, "layout_revision": SHARED_LAYOUT_REVISION,
                         "reader_initializer_bank": reader_bank, "role_batching": "one_scalar_role_per_physical_batch"})
    if rules_profile is not None:
        manifest.update(rules_profile)
    atomic_json(output / "export.json", manifest)
    return manifest


def numeric_check(checkpoint, directory):
    """Independent Python ORT1.22 reference, not Rust/CUDA acceptance."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from .model import fixture_input
    model, metadata = load_checkpoint(checkpoint)
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "export.json").read_text(encoding="utf-8"))
    validate_export_manifest(manifest, metadata)
    if ort.__version__ != "1.22.0" or manifest["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("reference runtime/checkpoint identity mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sessions = {}
    for graph in manifest["graphs"]:
        path = directory / graph["file"]
        if digest_file(path) != graph["sha256"]:
            raise ValueError("export graph hash mismatch")
        sessions[graph["role"]] = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    reports = []
    with torch.no_grad():
        for records, candidates, divergences, batch in ((1, 1, 1, 1), (3, 7, 4, 1), (5, 9, 3, 2)):
            data = fixture_input(records, candidates, divergences, batch)
            data.validate()
            public_feed = dict(zip(("board", "metadata", "records", "record_mask"), [x.numpy() for x in data.public_args()]))
            public = sessions["public"].run(None, public_feed)
            reference_memory = model.public_encoder(*data.public_args())
            for name, expected, actual in zip(("memory_key", "memory_value", "memory_mask"), reference_memory, public):
                expected = expected.numpy()
                if expected.shape != actual.shape:
                    raise ValueError(f"public shape mismatch: {name}")
                if expected.dtype == np.bool_:
                    if not np.array_equal(expected, actual): raise ValueError("public mask mismatch")
                elif not np.allclose(expected, actual, atol=1e-4, rtol=1e-3):
                    raise ValueError(f"public numeric mismatch: {name}")
            for role in manifest["roles"]:
                expected = model.role_graph(role)(*data.role_args(reference_memory))
                tensors = data.role_args(tuple(torch.from_numpy(v) for v in public))
                all_feed = dict(zip(("memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask", "divergence_features", "divergence_mask", "query"), [x.numpy() for x in tensors]))
                session = sessions.get(role)
                if manifest.get("layout") == "shared_pc_if":
                    session = sessions["shared_pc"]
                    all_feed["role_is_critic"] = np.asarray(role == "critic", dtype=np.bool_)
                actual = session.run(None, {value.name: all_feed[value.name] for value in session.get_inputs()})
                if manifest.get("layout") == "shared_pc_if":
                    if actual[4].shape != () or actual[4].dtype != np.bool_ or bool(actual[4]) != (role == "critic"):
                        raise ValueError("shared role tag mismatch")
                    if role == "proposer":
                        if actual[3].shape != (batch, divergences) or not np.array_equal(actual[3], np.zeros_like(actual[3])):
                            raise ValueError("P must emit empty-semantic zero divergence tensor")
                        actual = actual[:3]
                    else:
                        actual = actual[:4]
                if len(expected) != len(actual):
                    raise ValueError("role output count mismatch")
                maxima = []
                for index, (e, a) in enumerate(zip(expected, actual)):
                    e = e.numpy()
                    if e.shape != a.shape or not np.isfinite(a).all() or not np.allclose(e, a, atol=1e-4, rtol=1e-3):
                        raise ValueError(f"role numeric mismatch: {role} output {index}")
                    maxima.append(float(np.max(np.abs(e-a))))
                # The probabilities are independently normalized, never compared
                # by summing logits or treating masked padding as a legal move.
                for index in (0, 1):
                    ep = torch.softmax(expected[index], -1).numpy()
                    ap = torch.softmax(torch.from_numpy(actual[index]), -1).numpy()
                    if np.max(np.abs(ep-ap)) > 1e-4:
                        raise ValueError(f"policy/WDL numeric mismatch: {role}")
                reports.append({"role": role, "records": records, "candidates": candidates, "divergences": divergences,
                                "batch": batch, "max_absolute_differences": maxima})
    if len(reports) != 3 * len(manifest["roles"]):
        raise ValueError("numeric role acceptance case count mismatch")
    result = {"schema": SCHEMA, "artifact_schema": manifest["schema"], "layout": manifest.get("layout", "separate_pc"),
              "checkpoint_sha256": metadata["checkpoint_sha256"], "trained": metadata["trained"],
              "runtime": ort.__version__, "provider": "CPUExecutionProvider", "rust_ort": "not_run", "cuda": "not_run", "cases": reports, "status": "passed"}
    if manifest.get("layout") == "shared_pc_if":
        from .numeric import verify_cpu_hard_routes
        data = fixture_input(3, 7, 4, 1)
        with torch.no_grad(): memory = model.public_encoder(*data.public_args())
        feed = dict(zip(("memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask", "divergence_features", "divergence_mask", "query"), [value.numpy() for value in data.role_args(memory)]))
        result["hard_route_cpu_profile"] = verify_cpu_hard_routes(directory / "shared_pc_if.onnx", feed)
    return result


def rust_fixtures(checkpoint, export_directory, destination):
    """Raw independent Torch outputs for the Rust ORT acceptance command.

    These are bounded tensor fixtures and carry no claim of legal game states.
    The Rust Rules tests certify legal candidate construction independently.
    """
    import dataclasses
    import torch
    from .model import fixture_input
    model, metadata = load_checkpoint(checkpoint)
    export_directory = Path(export_directory).resolve()
    manifest = json.loads((export_directory / "export.json").read_text(encoding="utf-8"))
    validate_export_manifest(manifest, metadata)
    if manifest["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("fixture/export checkpoint mismatch")
    output = Path(destination).expanduser().resolve()
    output_directory(output.parent)
    if output.exists(): raise FileExistsError("fixture already registered")
    cases = []
    shapes = (("empty_context", "proposer", 0, 0, 0, False, 1),
              ("critic_divergence", "critic", 3, 7, 4, False, 1),
              ("proposer_order", "proposer", 3, 7, 0, False, 1),
              ("proposer_order_reversed", "proposer", 3, 7, 0, True, 1),
              ("same_board_other_history", "proposer", 3, 7, 0, False, 2),
              ("all_promotions", "critic", 1, 4, 1, False, 3))
    with torch.no_grad():
        for name, role, records, candidates, divergences, reverse, history in shapes:
            if role not in manifest["roles"]:
                raise ValueError("fixture role absent from exported model")
            data = fixture_input(max(records, 1), max(candidates, 1), max(divergences, 1))
            data = dataclasses.replace(data, record_mask=torch.ones((1, max(records, 1)), dtype=torch.bool) if records else torch.zeros((1, 1), dtype=torch.bool),
                                       candidate_mask=torch.ones((1, max(candidates, 1)), dtype=torch.bool) if candidates else torch.zeros((1, 1), dtype=torch.bool),
                                       divergence_mask=torch.ones((1, max(divergences, 1)), dtype=torch.bool) if divergences else torch.zeros((1, 1), dtype=torch.bool))
            if not records: data = dataclasses.replace(data, records=torch.zeros_like(data.records))
            if not candidates: data = dataclasses.replace(data, candidates=torch.zeros_like(data.candidates))
            if not divergences: data = dataclasses.replace(data, divergence_features=torch.zeros_like(data.divergence_features))
            if reverse:
                data = dataclasses.replace(data, candidates=torch.flip(data.candidates, (1,)))
            if name == "all_promotions":
                data = dataclasses.replace(data, candidates=torch.tensor([[[48, 56, promotion] for promotion in (1, 2, 3, 4)]], dtype=torch.int64))
            data.validate()
            memory = model.public_encoder(*data.public_args())
            raw = model.role_graph(role)(*data.role_args(memory))
            input_value = {"role": role, "board": data.board[0].tolist(), "metadata": data.metadata[0].tolist(),
                "records": [{"record_id": index+1, "revision": 1, "critical": index == 0,
                             "features": data.records[0,index].tolist()} for index in range(records)],
                "required_critical_records": [1] if records else [],
                "candidates": [{"from": values[0], "to": values[1], "promotion": values[2]} for values in data.candidates[0,:candidates].tolist()],
                "divergence_features": data.divergence_features[0,:divergences].tolist(), "query": data.query[0].tolist(),
                "situation_revision": 1, "history_digest": [history]*32,
                "model_epoch": list(bytes.fromhex(metadata["checkpoint_sha256"]))}
            expected = {"candidate_logits": raw[0][0,:candidates].tolist(), "wdl_logits": raw[1][0].tolist(),
                        "private_latent": raw[2][0].flatten().tolist(),
                        "divergence_logits": raw[3][0,:divergences].tolist() if role == "critic" else None,
                        "task_logits": raw[3][0].tolist() if role == "validator" else None}
            public_memory = {"tokens": int(memory[0].shape[2]), "memory_key": memory[0].flatten().tolist(),
                             "memory_value": memory[1].flatten().tolist(), "memory_mask": memory[2].flatten().tolist()}
            cases.append({"name": name, "input": input_value, "expected": expected, "public_memory": public_memory})
    result = {"schema": SCHEMA, "checkpoint_sha256": metadata["checkpoint_sha256"], "trained": metadata["trained"],
              "rules_certified": False, "reference": "pytorch_fp32_tf32_off", "cases": cases}
    atomic_json(output, result)
    return {"schema": SCHEMA, "file": output.name, "sha256": digest_file(output), "cases": len(cases), "trained": metadata["trained"]}
