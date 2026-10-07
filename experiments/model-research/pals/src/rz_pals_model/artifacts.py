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
        torch.manual_seed(checkpoint["seed"])
        model = PalsModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata


def export_checkpoint(checkpoint, directory, include_validator=False):
    import onnx
    import torch
    from .model import fixture_input
    model, metadata = load_checkpoint(checkpoint)
    output = output_directory(directory)
    if not include_validator:
        model = model.without_validator()
    data = fixture_input()
    data.validate(model.config)
    graphs = []
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    public_args = data.public_args()
    memory = model.public_encoder(*public_args)

    def export(graph, args, filename, names, output_names, dynamic_axes, role):
        destination = output / filename
        temporary = destination.with_suffix(".onnx.partial")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"graph already exists: {filename}")
        torch.onnx.export(graph, args, str(temporary), input_names=names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
                          export_params=True, do_constant_folding=True)
        document = onnx.load(str(temporary), load_external_data=False)
        onnx.checker.check_model(document, full_check=True)
        onnx.helper.set_model_props(document, {"schema": SCHEMA, "role": role,
            "checkpoint_sha256": metadata["checkpoint_sha256"], "training_steps": str(metadata["training_steps"]), "trained": str(metadata["trained"]).lower(),
            "precision": "fp32", "expected_ort": "1.22.0", "public_kv": "shared_role_neutral"})
        onnx.save(document, str(temporary))
        temporary.replace(destination)
        inputs = [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                   "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in document.graph.input]
        outputs = [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                    "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]} for v in document.graph.output]
        graphs.append({"file": filename, "sha256": digest_file(destination), "role": role,
                       "inputs": inputs, "outputs": outputs, "opset": 17})

    public_names = ["board", "metadata", "records", "record_mask"]
    axes = {name: {0: "batch"} for name in public_names}
    axes["records"][1] = axes["record_mask"][1] = "records"
    axes.update({"memory_key": {0: "batch", 2: "memory_tokens"}, "memory_value": {0: "batch", 2: "memory_tokens"}, "memory_mask": {0: "batch", 1: "memory_tokens"}})
    export(model.public_encoder, public_args, "public_memory.onnx", public_names,
           ["memory_key", "memory_value", "memory_mask"], axes, "public")
    names = ["memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask", "divergence_features", "divergence_mask", "query"]
    for role in model.experts:
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
    manifest = {"schema": SCHEMA, "config": model.config.to_dict(), "checkpoint_sha256": metadata["checkpoint_sha256"],
                "trained": metadata["trained"], "training_steps": metadata["training_steps"], "roles": list(model.experts), "validator_present": include_validator,
                "graphs": graphs, "runtime": {"onnxruntime": "1.22.0", "rust_ort": "2.0.0-rc.10", "compatibility": "requires_actual_numeric_check"},
                "candidate_promotion": {"0": "none", "1": "queen", "2": "rook", "3": "bishop", "4": "knight"},
                "wdl_perspective": "input_side_to_move", "task_names": list(TASKS), "numeric_status": "not_run", "cuda_status": "not_run"}
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
                if expected.dtype == np.bool_:
                    if not np.array_equal(expected, actual): raise ValueError("public mask mismatch")
                elif not np.allclose(expected, actual, atol=1e-4, rtol=1e-3):
                    raise ValueError(f"public numeric mismatch: {name}")
            for role in manifest["roles"]:
                expected = model.role_graph(role)(*data.role_args(reference_memory))
                tensors = data.role_args(tuple(torch.from_numpy(v) for v in public))
                all_feed = dict(zip(("memory_key", "memory_value", "memory_mask", "candidates", "candidate_mask", "divergence_features", "divergence_mask", "query"), [x.numpy() for x in tensors]))
                session = sessions[role]
                actual = session.run(None, {value.name: all_feed[value.name] for value in session.get_inputs()})
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
    return {"schema": SCHEMA, "checkpoint_sha256": metadata["checkpoint_sha256"], "trained": metadata["trained"],
            "runtime": ort.__version__, "provider": "CPUExecutionProvider", "rust_ort": "not_run", "cuda": "not_run", "cases": reports, "status": "passed"}


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
            cases.append({"name": name, "input": input_value, "expected": expected})
    result = {"schema": SCHEMA, "checkpoint_sha256": metadata["checkpoint_sha256"], "trained": metadata["trained"],
              "rules_certified": False, "reference": "pytorch_fp32_tf32_off", "cases": cases}
    atomic_json(output, result)
    return {"schema": SCHEMA, "file": output.name, "sha256": digest_file(output), "cases": len(cases), "trained": metadata["trained"]}
