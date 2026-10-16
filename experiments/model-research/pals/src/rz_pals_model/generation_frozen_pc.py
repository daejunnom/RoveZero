"""Bridge named frozen P/C tensors to pinned, exact manual ONNX artifacts.

The pure registration/parser APIs prove schema agreement only. A content proof
is issued solely by the explicit execution path after read-only checkpoint
decode and graph readback. File SHA, named parameter SHA, and graph SHA are
different namespaces. This never trains, constructs a PalsModel/optimizer,
changes caller/RNG state, executes a neural forward, or proves loaded GPU state.
"""
import copy
import ast
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
import zipfile
from types import SimpleNamespace, FunctionType, CodeType, ModuleType

from .config import ModelConfig, MODEL_SCHEMA_V3, MODEL_SEMANTICS_V3, ENCODING_V3
from .generation_data import digest, fields, identifier, sha, uint
from .generation_export import _hash_file, _owned_file, _atomic_json, _pinned_json
from .generation_training import CHECKPOINT_DOMAIN, ELIGIBILITY, GenerationLedger, parameter_owner, validate_checkpoint_phase, spec_digest

REQUEST_DOMAIN = "rz-pals-frozen-pc-content-request/1"
PROOF_DOMAIN = "rz-pals-frozen-pc-content-proof/1"
PARAMETER_DOMAIN = "rz-pals-frozen-pc-parameter-inventory/1"
MAPPING_DOMAIN = "rz-pals-frozen-pc-initializer-mapping/1"
SCOPE = "original_checkpoint_named_shared_pc_to_exact_manual_onnx"
FILENAME = "frozen-pc-content-proof.json"
SOURCE_NAMES = ("config.py", "model.py", "onnx_shared.py", "onnx_virtual_review.py",
                "nonzero_training.py", "generation_training.py", "generation_export.py",
                "generation_frozen_pc.py", "generation_data.py", "virtual_review.py", "artifacts.py",
                "training.py", "generation_chunks.py")
LIMITS = {"checkpoint_bytes": 1024**3, "graph_bytes": 256 * 1024**2,
          "tensor_bytes": 128 * 1024**2, "proof_bytes": 4 * 1024**2, "wall_seconds": 300}
VERIFICATION = {"checkpoint_tensor_content_verified": True, "export_initializer_content_verified": True,
                "manual_graph_structure_verified": True, "optimizer_content_verified": False,
                "numerical_acceptance": False, "native_loaded_assets_verified": False, "arena_eligible": False}


def registered_pc_parameter_spec(config=None):
    """Exact source-owned V3 name/shape/meaning and serialization allowlist.

    The registered configuration has 104 unique tensors, 112 graph occurrences.
    This is source shape arithmetic, not an observation of model/GPU memory.
    Linear matrices transpose once; GQA repeats activations, never this bank.
    """
    config = config or ModelConfig.for_profile("virtual_review_v1")
    config.validate()
    if config.profile != "virtual_review_v1":
        raise ValueError("frozen P/C mapping requires virtual_review_v1")
    result = {}
    def add(name, shape, graphs=("public",), transform="identity"):
        if name in result:
            raise ValueError("duplicate registered parameter")
        result[name] = {"name": name, "shape": list(shape), "dtype": "fp32_le",
                        "owner": parameter_owner(name), "transform": transform, "graphs": list(graphs),
                        "semantics": MODEL_SEMANTICS_V3 + ":" + name}
    def linear(prefix, output, input_, graphs):
        add(prefix + ".weight", (output, input_), graphs, "transpose_2d")
    def block(prefix):
        add(prefix + ".attention_norm.weight", (384,))
        for name, output in (("query", 384), ("key", 128), ("value", 128), ("output", 384)):
            linear(prefix + ".attention." + name, output, 384, ("public",))
        add(prefix + ".ffn_norm.weight", (384,))
        for name, output, input_ in (("gate", 1024, 384), ("up", 1024, 384), ("down", 384, 1024)):
            linear(prefix + ".ffn." + name, output, input_, ("public",))
    public = "public_encoder."
    add(public + "piece_embedding.weight", (13, 384))
    add(public + "square_embedding", (64, 384))
    add(public + "record_field_embedding", (4, 384))
    for name, output, input_ in (("metadata_projection", 768, 16), ("record_projection", 384, 2),
            ("anchor_record_projection", 384, 384), ("record_line_projection", 384, 64),
            ("memory_key", 128, 384), ("memory_value", 128, 384)):
        linear(public + name, output, input_, ("public",))
    for index in range(2):
        block(public + f"board_blocks.{index}")
    block(public + "record_blocks.0")
    line = public + "line_encoder."
    for name, count in (("from_square", 64), ("to_square", 64), ("promotion", 5)):
        add(line + "moves." + name + ".weight", (count, 64), ("public", "shared_pc"))
    linear(line + "moves.projection", 64, 192, ("public", "shared_pc"))
    add(line + "ply_embedding.weight", (256, 64), ("public", "shared_pc"))
    for index in range(2):
        add(line + f"temporal_blocks.{index}.weight", (64, 64, 3), ("public", "shared_pc"))
    linear(line + "pool_attention", 1, 64, ("public", "shared_pc"))
    for index in range(2):
        prefix = f"reader_blocks.{index}."
        add(prefix + "cross_norm.weight", (384,), ("shared_pc",))
        add(prefix + "self_norm.weight", (384,), ("shared_pc",))
        for attention, names in (("cross", (("query", 384), ("output", 384))),
                ("self_attn", (("query", 384), ("key", 128), ("value", 128), ("output", 384)))):
            for name, output in names:
                linear(prefix + attention + "." + name, output, 384, ("shared_pc",))
    for name, count in (("from_square", 64), ("to_square", 64), ("promotion", 5)):
        add("move_embedding." + name + ".weight", (count, 384), ("shared_pc",))
    linear("move_embedding.projection", 384, 1152, ("shared_pc",))
    for role in ("proposer", "critic"):
        prefix = "experts." + role + "."
        add(prefix + "initial_latent", (16, 384), ("shared_pc",))
        for index in range(2):
            add(prefix + f"ffn_norms.{index}.weight", (384,), ("shared_pc",))
            for name, output, input_ in (("gate", 1024, 384), ("up", 1024, 384), ("down", 384, 1024)):
                linear(prefix + f"ffns.{index}." + name, output, input_, ("shared_pc",))
        for name, output, input_ in (("query_projection", 384, 8), ("wdl_head", 3, 384),
                ("interaction_policy_head.hidden", 128, 768), ("interaction_policy_head.output", 1, 128),
                ("query_line_projection", 384, 192), ("relation_hidden", 128, 768),
                ("relation_output", 384, 128), ("query_anchor_projection", 384, 1152)):
            linear(prefix + name, output, input_, ("shared_pc",))
        add(prefix + "interaction_policy_head.hidden.bias", (128,), ("shared_pc",))
        add(prefix + "relation_hidden.bias", (128,), ("shared_pc",))
        if role == "critic":
            linear(prefix + "divergence_projection", 384, 8, ("shared_pc",))
            linear(prefix + "divergence_head", 1, 384, ("shared_pc",))
    return [result[name] for name in sorted(result)]


def _pin(value, *, domain=False, role=False, maximum=1024**3):
    keys = ("file", "sha256", "bytes") + (("domain",) if domain else ()) + (("role",) if role else ())
    fields(value, keys, "frozen P/C pin")
    file = value["file"]
    if type(file) is not str or len(file) > 256 or "\\" in file or ":" in file or any(ord(c) < 32 or ord(c) == 127 for c in file):
        raise ValueError("relative nonsecret artifact pin required")
    path = PurePosixPath(file)
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts) or str(path) != file:
        raise ValueError("noncanonical artifact pin")
    if any(p.startswith(".") for p in path.parts):
        raise ValueError("hidden/secret artifact pin forbidden")
    sha(value["sha256"])
    if not uint(value["bytes"], maximum):
        raise ValueError("empty artifact cannot establish tensor content")
    return value


def validate_request(value):
    fields(value, ("schema", "source_sha256", "model_epoch", "config", "checkpoint", "graphs",
                   "export_manifest", "verifier_sources", "expected_frozen_pc_sha256", "limits"), "frozen P/C request")
    if value["schema"] != REQUEST_DOMAIN:
        raise ValueError("unregistered frozen P/C request domain")
    sha(value["source_sha256"])
    uint(value["model_epoch"])
    if value["expected_frozen_pc_sha256"] is not None:
        sha(value["expected_frozen_pc_sha256"])
    config = ModelConfig(**value["config"])
    config.validate()
    if config.profile != "virtual_review_v1" or config.to_dict() != value["config"]:
        raise ValueError("unregistered frozen P/C profile/config")
    fields(value["limits"], tuple(LIMITS), "frozen P/C limits")
    for name, limit in value["limits"].items():
        if not uint(limit, LIMITS[name]):
            raise ValueError("finite positive frozen P/C limit required")
    checkpoint = _pin(value["checkpoint"], domain=True, maximum=value["limits"]["checkpoint_bytes"])
    if checkpoint["domain"] not in (CHECKPOINT_DOMAIN, MODEL_SEMANTICS_V3) or not checkpoint["file"].endswith(".pt"):
        raise ValueError("explicit actual-generation or own-initialization checkpoint required")
    manifest = _pin(value["export_manifest"], maximum=4 * 1024**2)
    if PurePosixPath(manifest["file"]).name != "export.json":
        raise ValueError("native export manifest original byte pin required")
    if type(value["graphs"]) is not list or len(value["graphs"]) != 2:
        raise ValueError("exact public/shared P/C graph pair required")
    for pin, (role, file) in zip(value["graphs"], (("public", "public_memory.onnx"), ("shared_pc", "shared_pc_if.onnx"))):
        _pin(pin, role=True, maximum=value["limits"]["graph_bytes"])
        if pin["role"] != role or pin["file"] != str(PurePosixPath(manifest["file"]).parent / file):
            raise ValueError("Fresh manual graph pair/order required")
    paths = [checkpoint["file"], manifest["file"]] + [p["file"] for p in value["graphs"]]
    if len(paths) != len(set(paths)):
        raise ValueError("artifact pins cannot alias")
    sources = value["verifier_sources"]
    if type(sources) is not list or [p.get("file") if type(p) is dict else None for p in sources] != list(SOURCE_NAMES):
        raise ValueError("complete preregistered verifier/builder source list required")
    for pin in sources:
        _pin(pin, maximum=2 * 1024**2)
    return copy.deepcopy(value)


def request_digest(value):
    return digest(REQUEST_DOMAIN, validate_request(value))


def validate_frozen_pc_proof(value, *, expected_request_sha256, expected_parameter_sha256=None):
    """Strict metadata join only; authoritative file pin/issuer remains external.

    A caller cannot infer actual decode/readback from a metadata boolean. Native
    code must consume the separately pinned artifact from the verified issuer
    and join its own loaded graph/manifest hashes before arming a capability.
    """
    fields(value, ("schema", "request", "request_sha256", "scope", "checkpoint_scope",
        "frozen_pc_parameter_sha256", "parameter_inventory", "parameter_inventory_sha256", "graphs",
        "mapping", "mapping_sha256", "encoding_registry_sha256", "verification"), "frozen P/C proof")
    request = validate_request(value["request"])
    sha(expected_request_sha256)
    if value["schema"] != PROOF_DOMAIN or value["scope"] != SCOPE or value["request_sha256"] != expected_request_sha256 or request_digest(request) != expected_request_sha256:
        raise ValueError("frozen P/C proof source/request pin mismatch")
    scope = "own_initialization_tensor_content_only" if request["checkpoint"]["domain"] == MODEL_SEMANTICS_V3 else "actual_pc_frozen_tensor_content_only"
    if value["checkpoint_scope"] != scope or type(value["verification"]) is not dict or value["verification"] != VERIFICATION or any(type(v) is not bool for v in value["verification"].values()):
        raise ValueError("frozen tensor proof cannot claim optimizer/numerical/runtime/arena acceptance")
    sha(value["frozen_pc_parameter_sha256"])
    for expected in (request["expected_frozen_pc_sha256"], expected_parameter_sha256):
        if expected is not None and sha(expected) != value["frozen_pc_parameter_sha256"]:
            raise ValueError("named frozen P/C tensor identity mismatch")
    from .virtual_review import registered_encoding_sha256
    if value["encoding_registry_sha256"] != registered_encoding_sha256():
        raise ValueError("frozen proof encoding registry differs from tensor ABI")
    specs = {p["name"]: p for p in registered_pc_parameter_spec(ModelConfig(**request["config"]))}
    inventory = value["parameter_inventory"]
    if type(inventory) is not list or [p.get("name") if type(p) is dict else None for p in inventory] != sorted(specs):
        raise ValueError("complete sorted named P/C inventory required")
    indexed = {}
    for p in inventory:
        fields(p, ("name", "shape", "dtype", "owner", "semantics", "bytes", "sha256"), "named parameter")
        spec = specs[p["name"]]
        if any(p[k] != spec[k] or type(p[k]) is not type(spec[k]) for k in ("shape", "dtype", "owner", "semantics")) or any(type(s) is not int for s in p["shape"]):
            raise ValueError("parameter name/shape/dtype/meaning allowlist mismatch")
        if uint(p["bytes"], request["limits"]["tensor_bytes"]) != 4 * math.prod(p["shape"]):
            raise ValueError("named FP32 parameter byte size mismatch")
        sha(p["sha256"])
        indexed[p["name"]] = p
    if sum(p["bytes"] for p in inventory) > request["limits"]["tensor_bytes"] or value["parameter_inventory_sha256"] != digest(PARAMETER_DOMAIN, inventory):
        raise ValueError("named parameter inventory pin/budget mismatch")
    if type(value["graphs"]) is not list or len(value["graphs"]) != 2:
        raise ValueError("exact readback graph evidence required")
    graph_inventory = {}
    for graph, pin in zip(value["graphs"], request["graphs"]):
        fields(graph, ("role", "file", "sha256", "bytes", "structure_sha256", "initializers", "initializer_inventory_sha256"), "readback graph")
        if any(graph[k] != pin[k] or type(graph[k]) is not type(pin[k]) for k in pin):
            raise ValueError("original graph byte pin mismatch")
        sha(graph["structure_sha256"])
        values = graph["initializers"]
        if type(values) is not list or not 1 <= len(values) <= 4096:
            raise ValueError("bounded actual initializer inventory required")
        names = []
        by_name = {}
        for entry in values:
            fields(entry, ("name", "dtype", "shape", "bytes", "sha256"), "readback initializer")
            identifier(entry["name"])
            if entry["dtype"] not in ("FLOAT", "INT64", "BOOL") or type(entry["shape"]) is not list or len(entry["shape"]) > 8 or any(type(s) is not int or not 1 <= s <= 1048576 for s in entry["shape"]):
                raise ValueError("unsupported actual initializer representation")
            width = {"FLOAT": 4, "INT64": 8, "BOOL": 1}[entry["dtype"]]
            if uint(entry["bytes"], request["limits"]["graph_bytes"]) != width * math.prod(entry["shape"]):
                raise ValueError("actual initializer byte size mismatch")
            sha(entry["sha256"])
            names.append(entry["name"])
            by_name[entry["name"]] = entry
        if names != sorted(set(names)) or graph["initializer_inventory_sha256"] != digest("rz-pals-onnx-readback-initializers/1", values):
            raise ValueError("duplicate/reordered initializer inventory or bad pin")
        if sum(p["bytes"] for p in values) > min(graph["bytes"], request["limits"]["graph_bytes"]):
            raise ValueError("readback initializer total exceeds pinned graph/registered cap")
        graph_inventory[graph["role"]] = by_name
    mapping = value["mapping"]
    expected_keys = sorted((role, name) for name, p in specs.items() for role in p["graphs"])
    if type(mapping) is not list or [(m.get("graph_role"), m.get("parameter")) if type(m) is dict else None for m in mapping] != expected_keys:
        raise ValueError("complete sorted actual copied initializer mapping required")
    for entry in mapping:
        fields(entry, ("graph_role", "parameter", "initializer", "transform", "source_sha256", "initializer_sha256"), "copied initializer mapping")
        spec = specs[entry["parameter"]]
        expected_name = entry["parameter"] + (".linear_transpose" if spec["transform"] == "transpose_2d" else "")
        if entry["transform"] != spec["transform"] or entry["initializer"] != expected_name or entry["source_sha256"] != indexed[entry["parameter"]]["sha256"]:
            raise ValueError("unsupported transformation/source or copied-name drift")
        target = graph_inventory[entry["graph_role"]].get(expected_name)
        shape = list(reversed(spec["shape"])) if spec["transform"] == "transpose_2d" else spec["shape"]
        if target is None or target["dtype"] != "FLOAT" or target["shape"] != shape or target["bytes"] != indexed[entry["parameter"]]["bytes"] or target["sha256"] != entry["initializer_sha256"]:
            raise ValueError("copied tensor/readback initializer join failed")
        if spec["transform"] == "identity" and entry["source_sha256"] != entry["initializer_sha256"]:
            raise ValueError("identity copy changed FP32 bits")
    if value["mapping_sha256"] != digest(MAPPING_DOMAIN, mapping):
        raise ValueError("copied tensor mapping digest mismatch")
    return copy.deepcopy(value)


def read_frozen_pc_proof(path, *, expected_file_sha256, expected_request_sha256, expected_parameter_sha256=None):
    path = Path(path)
    if path.name != FILENAME:
        raise ValueError("registered frozen P/C proof filename required")
    sha(expected_file_sha256)
    value = _pinned_json(path, expected_file_sha256, path.stat().st_size, LIMITS["proof_bytes"])
    checked = validate_frozen_pc_proof(value, expected_request_sha256=expected_request_sha256,
                                      expected_parameter_sha256=expected_parameter_sha256)
    if path.stat().st_size > checked["request"]["limits"]["proof_bytes"]:
        raise ValueError("readback proof exceeds its own registered limit")
    return checked


def _zip_preflight(snapshot, limit):
    """Bound serialized storage before weights-only decoding, not pickle eval."""
    with zipfile.ZipFile(snapshot) as archive:
        entries = archive.infolist()
        if not 1 <= len(entries) <= 8192 or len({e.filename for e in entries}) != len(entries):
            raise ValueError("bounded unique checkpoint ZIP entries required")
        total = 0
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or str(path) != entry.filename or any(p in (".", "..") for p in path.parts)
                    or "\\" in entry.filename or ":" in entry.filename or any(ord(c) < 32 or ord(c) == 127 for c in entry.filename)
                    or entry.is_dir() or entry.flag_bits & 1 or entry.compress_type != zipfile.ZIP_STORED
                    or entry.file_size != entry.compress_size):
                raise ValueError("encrypted/compressed/escaping checkpoint storage forbidden")
            total += entry.file_size
            if total > limit:
                raise ValueError("checkpoint decoded storage exceeded explicit cap")
    snapshot.seek(0)


def _source_value(node, namespace=None):
    """Read declarations/arithmetic and pinned aliases; never execute source text."""
    if isinstance(node, ast.Name) and namespace is not None and node.id in namespace:
        return namespace[node.id]
    if isinstance(node, ast.Attribute) and namespace is not None:
        owner = _source_value(node.value, namespace)
        if isinstance(owner, ModuleType) and node.attr in vars(owner):
            return vars(owner)[node.attr]
        raise ValueError("source default attribute must belong to an imported module")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
        left, right = _source_value(node.left, namespace), _source_value(node.right, namespace)
        if type(left) not in (int, float) or type(right) not in (int, float) or isinstance(node.op, ast.Pow) and not 0 <= right <= 64:
            raise ValueError("nonliteral source declaration")
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        return left ** right
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_source_value(v, namespace) for v in node.elts]
        return values if isinstance(node, ast.List) else tuple(values) if isinstance(node, ast.Tuple) else set(values)
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("source dictionary expansion is not a literal")
        return {_source_value(k, namespace): _source_value(v, namespace) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def _verify_active_source(module_name, source, path, *, executing_code=None, executing_name=None):
    """Bind loaded Python function code to the pinned disk source snapshot.

    Checking disk SHA alone would allow old functions in an already imported
    process to label their work with newer disk source. Dataclass generated
    code is not compared as authored source; its literal field defaults are.
    This check does not import an otherwise unused neural module.
    """
    module = sys.modules.get(module_name)
    if module is None:
        return
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError("active verifier module came from another source root")
    tree = ast.parse(source, filename=str(path))
    compiled = compile(tree, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    codes, all_codes = {}, []
    def collect(code, prefix=""):
        for child in code.co_consts:
            if isinstance(child, CodeType):
                name = prefix + child.co_name
                all_codes.append((name, child))
                codes[name] = child
                collect(child, name + ".")
    collect(compiled)
    if executing_code is not None:
        if (executing_name not in codes or executing_code != codes[executing_name]
                or Path(executing_code.co_filename).resolve() != path.resolve()):
            raise ValueError("executing verifier alias differs from preregistered source")
    elif executing_name is not None:
        raise ValueError("executing verifier name lacks actual frame code")
    def same_code(actual, qualname):
        if (not isinstance(actual, FunctionType) or qualname not in codes or actual.__code__ != codes[qualname]
                or Path(actual.__code__.co_filename).resolve() != path.resolve()
                or actual.__globals__ is not module.__dict__):
            raise ValueError("active verifier code differs from preregistered source: " + qualname)
    def literal_defaults(node, actual):
        def checked_default(declared_node, active_value):
            # ModelConfig() is the only source-owned constructor used as a
            # default. Compare its dataclass values without executing source
            # text or constructing a neural/config object during verification.
            if (isinstance(declared_node, ast.Call) and isinstance(declared_node.func, ast.Name)
                    and declared_node.func.id == "ModelConfig" and not declared_node.args
                    and not declared_node.keywords):
                cls = module.__dict__.get("ModelConfig")
                declarations = getattr(cls, "__dataclass_fields__", None)
                if declarations is None or type(active_value) is not cls:
                    raise ValueError("active function config default differs from source")
                from dataclasses import MISSING
                expected = {key: value.default for key, value in declarations.items()}
                if any(value is MISSING for value in expected.values()) or not _typed_equal(vars(active_value), expected):
                    raise ValueError("active function config default differs from source")
                return
            try: declared = _source_value(declared_node, module.__dict__)
            except (ValueError, TypeError) as error:
                raise ValueError("unsupported source-owned function default") from error
            if not _typed_equal(active_value, declared):
                raise ValueError("active function default differs from source")
        active = actual.__defaults__ or ()
        if len(active) != len(node.args.defaults):
            raise ValueError("active function default count differs from source")
        for declared_node, active_value in zip(node.args.defaults, active):
            checked_default(declared_node, active_value)
        for key, declared_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if declared_node is None:
                continue
            try:
                checked_default(declared_node, (actual.__kwdefaults__ or {}).get(key.arg, object()))
            except ValueError as error:
                raise ValueError("active keyword default differs from source") from error
    from .generation_training import _typed_equal
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                binding = imported.asname or imported.name.split(".")[0]
                dependency = sys.modules.get(imported.name if imported.asname else imported.name.split(".")[0])
                if dependency is None or module.__dict__.get(binding) is not dependency:
                    raise ValueError("active imported module alias differs from pinned owner")
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            dependency = sys.modules.get(module.__package__ + "." + node.module)
            if dependency is None or any(getattr(module, n.asname or n.name, object()) is not getattr(dependency, n.name, object()) for n in node.names):
                raise ValueError("active source dependency alias differs from pinned owner")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            active = getattr(module, node.name, None)
            if any(isinstance(d, ast.Name) and d.id == "contextmanager" for d in node.decorator_list):
                from contextlib import contextmanager
                wrapped = getattr(active, "__wrapped__", None)
                if not isinstance(active, FunctionType) or not isinstance(wrapped, FunctionType):
                    raise ValueError("active source contextmanager wrapper differs")
                template = contextmanager(wrapped)
                if active.__code__ != template.__code__ or active.__closure__ != template.__closure__ or active.__globals__ is not template.__globals__:
                    raise ValueError("active contextmanager does not wrap the pinned function")
                active = wrapped
            same_code(active, node.name)
            literal_defaults(node, active)
        elif isinstance(node, ast.ClassDef):
            cls = getattr(module, node.name, None)
            if not isinstance(cls, type) or cls.__module__ != module_name:
                raise ValueError("active verifier class differs from registered source")
            dataclass_fields = getattr(cls, "__dataclass_fields__", None)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    active = cls.__dict__.get(member.name)
                    if isinstance(active, (staticmethod, classmethod)):
                        active = active.__func__
                    elif isinstance(active, property):
                        active = active.fget
                    same_code(active, node.name + "." + member.name)
                    literal_defaults(member, active)
                elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name) and member.value is not None:
                    if (isinstance(member.value, ast.Call) and isinstance(member.value.func, ast.Name)
                            and member.value.func.id == "field" and dataclass_fields is not None):
                        descriptor = dataclass_fields.get(member.target.id)
                        if descriptor is None:
                            raise ValueError("active dataclass field missing from registered source")
                        for keyword in member.value.keywords:
                            if keyword.arg == "default_factory" and isinstance(keyword.value, ast.Lambda):
                                factory = descriptor.default_factory
                                matching = [code for qualname, code in all_codes
                                            if code.co_name == "<lambda>" and code.co_firstlineno == keyword.value.lineno
                                            and qualname == node.name + ".<lambda>"]
                                if (not isinstance(factory, FunctionType) or len(matching) != 1
                                        or factory.__code__ != matching[0]
                                        or factory.__globals__ is not module.__dict__
                                        or Path(factory.__code__.co_filename).resolve() != path.resolve()):
                                    raise ValueError("active dataclass default factory differs from registered source")
                            elif keyword.arg == "default":
                                try: declared = _source_value(keyword.value)
                                except (ValueError, TypeError): continue
                                if not _typed_equal(descriptor.default, declared):
                                    raise ValueError("active dataclass field default differs from registered source")
                        continue
                    try: declared = _source_value(member.value)
                    except (ValueError, TypeError): continue
                    if (not _typed_equal(getattr(cls, member.target.id, object()), declared)
                            or dataclass_fields is not None
                            and not _typed_equal(dataclass_fields[member.target.id].default, declared)):
                        raise ValueError("active dataclass default differs from registered source")
            # A descriptor factory check is incomplete if generated __init__
            # invokes another closure or its own defaults were mutated. Bind
            # that actual execution connection without calling the constructor.
            authored_init = any(isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and member.name == "__init__" for member in node.body)
            if dataclass_fields is not None and not authored_init and getattr(cls.__dataclass_params__, "init", False):
                from dataclasses import MISSING, _HAS_DEFAULT_FACTORY
                init = cls.__dict__.get("__init__")
                if not isinstance(init, FunctionType):
                    raise ValueError("active dataclass generated init differs from registered source")
                expected_positional, expected_keyword = [], {}
                cells = dict(zip(init.__code__.co_freevars, init.__closure__ or ()))
                for descriptor in dataclass_fields.values():
                    if not descriptor.init:
                        continue
                    factory = descriptor.default_factory
                    default = _HAS_DEFAULT_FACTORY if factory is not MISSING else descriptor.default
                    if default is not MISSING:
                        if descriptor.kw_only:
                            expected_keyword[descriptor.name] = default
                        else:
                            expected_positional.append(default)
                    if factory is not MISSING:
                        # Python 3.10 and 3.12 use different generated
                        # closure names. Both retain one exact field factory;
                        # accepting neither/both would lose that connection.
                        factory_names = ("_dflt_" + descriptor.name,
                                         "__dataclass_dflt_" + descriptor.name + "__")
                        matched_cells = [cells[name] for name in factory_names if name in cells]
                        if len(matched_cells) != 1 or matched_cells[0].cell_contents is not factory:
                            raise ValueError("active dataclass init factory connection differs from registered source")
                if (not _typed_equal(init.__defaults__ or (), tuple(expected_positional))
                        or not _typed_equal(init.__kwdefaults__ or {}, expected_keyword)):
                    raise ValueError("active dataclass init defaults differ from registered source")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is None or any(not isinstance(t, ast.Name) for t in targets):
                continue
            try: declared = _source_value(node.value)
            except (ValueError, TypeError): continue
            if any(not _typed_equal(getattr(module, t.id, object()), declared) for t in targets):
                raise ValueError("active source declaration differs from registered source")


def _checkpoint_pc(payload, request, torch):
    if type(payload) is not dict or payload.get("schema") != request["checkpoint"]["domain"] or payload.get("config") != request["config"]:
        raise ValueError("original checkpoint domain/config differs from registration")
    if payload["schema"] == MODEL_SEMANTICS_V3:
        fields(payload, ("state_dict", "config", "seed", "schema", "training_steps", "trained"), "own initialization checkpoint")
        if type(payload["seed"]) is not int or payload["seed"] != 23 or type(payload["training_steps"]) is not int or payload["training_steps"] != 0 or payload["trained"] is not False:
            raise ValueError("diagnostic/trained metadata cannot become own initialization")
        state, scope = payload["state_dict"], "own_initialization_tensor_content_only"
        training = {"trained": False, "training_steps": 0}
    else:
        # Reuse the existing actual format and freeze-phase validator. This
        # narrow proof deliberately does not restore/validate optimizer/RNG.
        fields(payload, ("schema", "eligibility", "spec", "spec_sha256", "config", "model", "optimizers", "schedulers", "membership", "parameter_updates",
                         "frozen_pc_sha256", "ledger", "sampler", "bucket", "rng", "cuda_devices"), "actual generation checkpoint")
        ledger = GenerationLedger(**copy.deepcopy(payload["ledger"]))
        validate_checkpoint_phase(ledger, payload["frozen_pc_sha256"])
        if payload["eligibility"] != ELIGIBILITY or payload["spec"] != ledger.spec or payload["spec_sha256"] != spec_digest(ledger.spec) or ledger.phase not in ("v", "complete") or ledger.pending_microbatches or not ledger.checkpoint_boundary() or request["model_epoch"] <= ledger.spec["model_epoch"]:
            raise ValueError("actual P/C must be frozen at an optimizer boundary")
        state, scope = payload["model"], "actual_pc_frozen_tensor_content_only"
        training = {"trained": True, "training_steps": sum(ledger.completed.values())}
    if not isinstance(state, dict) or any(type(name) is not str for name in state):
        raise ValueError("named original checkpoint tensor state required")
    specs = {p["name"]: p for p in registered_pc_parameter_spec(ModelConfig(**request["config"]))}
    pc = {name: tensor for name, tensor in state.items() if parameter_owner(name) != "validator"}
    if set(pc) != set(specs):
        raise ValueError("original checkpoint has missing/extra P/C parameters")
    inventory = []
    for name in sorted(pc):
        tensor, spec = pc[name], specs[name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or list(tensor.shape) != spec["shape"] or not bool(torch.isfinite(tensor).all()):
            raise ValueError("original parameter name/shape/dtype/finite mismatch")
        raw = tensor.detach().contiguous().numpy().tobytes()
        inventory.append({k: spec[k] for k in ("name", "shape", "dtype", "owner", "semantics")} | {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    if sum(p["bytes"] for p in inventory) > request["limits"]["tensor_bytes"]:
        raise ValueError("actual P/C tensor bytes exceed registered cap")
    from .nonzero_training import _state_digest
    parameter_sha = _state_digest(pc)
    if payload["schema"] == CHECKPOINT_DOMAIN and parameter_sha != payload["frozen_pc_sha256"] or request["expected_frozen_pc_sha256"] is not None and parameter_sha != request["expected_frozen_pc_sha256"]:
        raise ValueError("actual named P/C bytes disagree with freeze/registration")
    return pc, inventory, parameter_sha, scope, training


def _join_export_manifest(manifest, request, training):
    """Reuse the source-owned export contract, then bind its exact byte pins."""
    from .artifacts import validate_virtual_export_manifest
    metadata = {"config": request["config"], "checkpoint_sha256": request["checkpoint"]["sha256"], **training}
    validate_virtual_export_manifest(manifest, metadata)
    for declaration, pin in zip(manifest["graphs"][:2], request["graphs"]):
        if declaration["role"] != pin["role"] or declaration["sha256"] != pin["sha256"] or pin["file"] != str(PurePosixPath(request["export_manifest"]["file"]).parent / declaration["file"]):
            raise ValueError("native manifest graph byte pins differ from frozen readback registration")


def _readback_graph(document, expected, weights, pc, pin, request, onnx):
    if (document.ir_version != expected.ir_version or document.domain != expected.domain
            or list(document.opset_import) != list(expected.opset_import)
            or document.functions or document.training_info
            or document.graph.SerializeToString(deterministic=True) != expected.graph.SerializeToString(deterministic=True)):
        raise ValueError("actual graph structure/constants/initializers differ from original checkpoint manual graph")
    metadata = {}
    for prop in document.metadata_props:
        if prop.key in metadata:
            raise ValueError("duplicate graph metadata key")
        metadata[prop.key] = prop.value
    required = {"schema": MODEL_SCHEMA_V3, "model_profile": "virtual_review_v1", "model_semantics": MODEL_SEMANTICS_V3,
                "encoding_schema": ENCODING_V3, "precision": "fp32", "role": pin["role"],
                "checkpoint_sha256": request["checkpoint"]["sha256"]}
    if any(metadata.get(k) != v for k, v in required.items()):
        raise ValueError("readback graph metadata/source byte pin mismatch")
    inventory = []
    for tensor in document.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data or tensor.segment.HasField("begin") or tensor.segment.HasField("end"):
            raise ValueError("external/segmented initializer bytes forbidden")
        if any(len(getattr(tensor, key)) for key in ("float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data")):
            raise ValueError("manual initializer must preserve unambiguous raw bytes")
        raw = bytes(tensor.raw_data)
        inventory.append({"name": tensor.name, "dtype": onnx.TensorProto.DataType.Name(tensor.data_type),
                          "shape": list(tensor.dims), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    inventory.sort(key=lambda v: v["name"])
    actual = {p["name"]: p for p in inventory}
    specs = {p["name"]: p for p in registered_pc_parameter_spec(ModelConfig(**request["config"]))}
    mapping = []
    for weight in weights:
        name = weight["parameter"]
        if name not in specs or pin["role"] not in specs[name]["graphs"]:
            raise ValueError("builder requested unregistered parameter/graph meaning")
        source = pc[name].detach().contiguous().numpy()
        raw_source = source.tobytes()
        transpose = specs[name]["transform"] == "transpose_2d"
        target = source.T.copy(order="C") if transpose else source
        target_raw = target.tobytes()
        source_sha, target_sha = hashlib.sha256(raw_source).hexdigest(), hashlib.sha256(target_raw).hexdigest()
        expected_name = name + (".linear_transpose" if transpose else "")
        entry = actual.get(expected_name)
        if weight["initializer"] != expected_name or weight["source_parameter_sha256"] != source_sha or weight["sha256"] != target_sha or weight["shape"] != list(target.shape) or weight["bytes"] != len(target_raw) or entry is None or entry["sha256"] != target_sha:
            raise ValueError("actual copied tensor mapping/FP32 readback mismatch")
        mapping.append({"graph_role": pin["role"], "parameter": name, "initializer": expected_name,
                        "transform": specs[name]["transform"], "source_sha256": source_sha, "initializer_sha256": target_sha})
    graph = dict(pin, structure_sha256=hashlib.sha256(document.graph.SerializeToString(deterministic=True)).hexdigest(),
                 initializers=inventory, initializer_inventory_sha256=digest("rz-pals-onnx-readback-initializers/1", inventory))
    return graph, mapping


def verify_frozen_pc_content(request, *, artifact_root, output_directory, execute=False):
    """Actual read-only decode/readback, under the caller's explicit CPU slot.

    No caller model, optimizer, session or RNG is supplied or constructed. The
    graph builder reads a lightweight immutable state bank from the verified
    private byte snapshot. The resulting raw JSON file needs a separate pin;
    native startup must independently bind loaded asset hashes to this proof.
    """
    if execute is not True:
        raise ValueError("frozen P/C checkpoint decode requires explicit execution authorization")
    executing_code = sys._getframe().f_code
    request = validate_request(request)
    if sys.byteorder != "little":
        raise ValueError("registered raw FP32/state digest requires little-endian CPU")
    start = time.monotonic()
    def deadline():
        if time.monotonic() - start >= request["limits"]["wall_seconds"]:
            raise TimeoutError("frozen P/C verification window ended")
    source_root = Path(__file__).resolve().parent
    def verify_sources():
        for pin in request["verifier_sources"]:
            path, source_snapshot = _owned_file(source_root, pin["file"]), io.BytesIO()
            if _hash_file(path, 2 * 1024**2, snapshot=source_snapshot) != (pin["sha256"], pin["bytes"]):
                raise ValueError("verifier/manual builder source changed from registration")
            frame = {"executing_code": executing_code, "executing_name": "verify_frozen_pc_content"} if path.stem == "generation_frozen_pc" else {}
            _verify_active_source(__package__ + "." + path.stem, source_snapshot.getvalue(), path, **frame)
    verify_sources()
    raw_root = Path(artifact_root)
    if raw_root.is_symlink() or getattr(raw_root.stat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("artifact root cannot be linked/reparse")
    root = raw_root.resolve(strict=True)
    checkpoint = _owned_file(root, request["checkpoint"]["file"])
    output = Path(output_directory).resolve()
    repository = next((p for p in source_root.parents if (p / ".git").exists()), None)
    if output.exists() or output == root or output.is_relative_to(root) or repository is not None and (output == repository or output.is_relative_to(repository)):
        raise ValueError("new independent owned proof output outside artifacts/repository required")
    if not output.parent.is_dir():
        raise ValueError("proof owner must prepare its external scratch/publication parent")
    # Temporary snapshots are verifier-owned external scratch, never writes to
    # the original read-only artifact root. Their lifetime ends after readback.
    with tempfile.TemporaryFile(mode="w+b", prefix="rz-frozen-pc-pinned-", dir=output.parent) as snapshot:
        pin = request["checkpoint"]
        if _hash_file(checkpoint, request["limits"]["checkpoint_bytes"], snapshot=snapshot) != (pin["sha256"], pin["bytes"]):
            raise ValueError("original checkpoint byte pin mismatch")
        snapshot.flush(); snapshot.seek(0)
        _zip_preflight(snapshot, request["limits"]["checkpoint_bytes"])
        deadline()
        import torch
        # Lazy local imports must be source-checked before any of their decode,
        # metadata or builder helpers execute. The first pin pass intentionally
        # leaves unused neural modules unimported, so it cannot prove their
        # already-loaded code after this import boundary.
        from .nonzero_training import _state_digest as _verified_state_digest
        from .artifacts import validate_virtual_export_manifest as _verified_manifest_validator
        from .onnx_virtual_review import build_virtual_public_graph, build_virtual_shared_pc_graph
        from .virtual_review import registered_encoding_sha256
        verify_sources()
        deadline()
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        pc, inventory, parameter_sha, checkpoint_scope, training = _checkpoint_pc(payload, request, torch)
        manifest_pin = request["export_manifest"]
        manifest = _pinned_json(_owned_file(root, manifest_pin["file"]), manifest_pin["sha256"], manifest_pin["bytes"])
        _join_export_manifest(manifest, request, training)
        deadline()
        # Only existing graph serialization code is used; never instantiate a
        # neural model merely to discover its parameter names or dimensions.
        bank = SimpleNamespace(config=ModelConfig(**request["config"]), experts={"proposer": None, "critic": None}, state_dict=lambda: pc)
        import onnx
        graphs, mapping = [], []
        for pin, builder in zip(request["graphs"], (build_virtual_public_graph, build_virtual_shared_pc_graph)):
            deadline()
            expected, evidence = builder(bank)
            with tempfile.TemporaryFile(mode="w+b", prefix="rz-frozen-graph-pinned-", dir=output.parent) as graph_snapshot:
                if _hash_file(_owned_file(root, pin["file"]), request["limits"]["graph_bytes"], snapshot=graph_snapshot) != (pin["sha256"], pin["bytes"]):
                    raise ValueError("original ONNX graph byte pin mismatch")
                graph_snapshot.seek(0)
                document = onnx.load_model_from_string(graph_snapshot.read())
            graph, copied = _readback_graph(document, expected, evidence["parameters"], pc, pin, request, onnx)
            graphs.append(graph); mapping.extend(copied)
            del expected, document, evidence
        mapping.sort(key=lambda p: (p["graph_role"], p["parameter"]))
        proof = {"schema": PROOF_DOMAIN, "request": request, "request_sha256": request_digest(request), "scope": SCOPE,
                 "checkpoint_scope": checkpoint_scope, "frozen_pc_parameter_sha256": parameter_sha,
                 "parameter_inventory": inventory, "parameter_inventory_sha256": digest(PARAMETER_DOMAIN, inventory),
                 "graphs": graphs, "mapping": mapping, "mapping_sha256": digest(MAPPING_DOMAIN, mapping),
                 "encoding_registry_sha256": registered_encoding_sha256(), "verification": dict(VERIFICATION)}
        validate_frozen_pc_proof(proof, expected_request_sha256=request_digest(request))
        deadline(); verify_sources(); deadline()
        import json
        if len(json.dumps(proof, ensure_ascii=False, allow_nan=False).encode()) > request["limits"]["proof_bytes"]:
            raise ValueError("actual frozen P/C proof exceeded registered output cap")
        deadline()
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(output / FILENAME, proof)
        proof_sha, proof_bytes = _hash_file(output / FILENAME, request["limits"]["proof_bytes"])
        deadline()
        return {"schema": PROOF_DOMAIN, "file": FILENAME, "sha256": proof_sha, "bytes": proof_bytes,
                "request_sha256": proof["request_sha256"], "frozen_pc_parameter_sha256": parameter_sha,
                "optimizer_content_verified": False, "numerical_acceptance": False, "native_loaded_assets_verified": False}
