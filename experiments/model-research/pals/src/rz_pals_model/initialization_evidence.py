"""G0 capture, separate V2 observation and dependency-free evidence readers.

Importing this module does not import Torch or construct a model. Only the
explicit ``initialize_g0_with_evidence`` producer runs initialization and a
weights-only inventory check. A stored receipt is an observation of that
producer, never a numerical/export/arena acceptance or a caller capability.
Historical checkpoint/metadata formats are not rewritten.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import time
import types

from .config import (ENCODING_V3, MODEL_SCHEMA_V3, MODEL_SEMANTICS_V3,
                     ModelConfig, ROLES, SCHEMA as OWNER_SCHEMA,
                     VIRTUAL_REVIEW_PROFILE)

EVIDENCE_SCHEMA = "rz-pals-g0-initialization-evidence/1"
SOURCE_SCHEMA = "rz-pals-g0-initialization-sources/1"
LEGACY_EVIDENCE_SCHEMA = "rz-pals-legacy-initialization-observation/1"
LEGACY_SOURCE_SCHEMA = "rz-pals-legacy-initialization-sources/1"
LEGACY_PROFILE = "full_line_interaction_v2"
# The historical shared-export schema is declared in onnx_shared.py. Importing
# that module would import ONNX, so pin its source and retain the exact domain.
LEGACY_MODEL_SCHEMA = "rovezero.pals-model.v2"
PRODUCER = "rz_pals_model.artifacts.initialize_checkpoint"
EVIDENCE_NAME = "g0-initialization-evidence.json"
LEGACY_EVIDENCE_NAME = "legacy-initialization-observation.json"
SOURCE_FILES = ("__init__.py", "config.py", "model.py", "artifacts.py",
                "controlled_initialization.py", "virtual_review.py",
                "onnx_shared.py", "initialization_evidence.py")
MAX_METADATA_BYTES = 65536
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
MAX_PARAMETER_TENSORS = 4096
MAX_PARAMETER_VALUES = 1 << 28
_CAPTURE_KEY = object()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(label + " fields differ from the registered schema")


def _sha(value, label="digest"):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(label + " requires a lowercase SHA-256")


def _text(value, label, maximum=256):
    if type(value) is not str or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(label + " requires bounded nonempty text")


def _uint(value, label, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(label + " requires a bounded integer, not a boolean")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def _parse_json(raw, maximum):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ValueError("JSON evidence exceeds its bounded size")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _is_link(path):
    details = path.lstat()
    return stat.S_ISLNK(details.st_mode) or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _plain_path(path, *, directory=False):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("artifact path must not traverse a parent")
    for parent in reversed((path, *path.parents)):
        if parent.exists() and _is_link(parent):
            raise ValueError("artifact path crosses a symbolic link or reparse point")
    if not path.exists() or (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError("artifact path must be an existing regular " + ("directory" if directory else "file"))
    return path


def _external_directory(directory, *, require_owner=True):
    directory = _plain_path(directory, directory=True)
    # Check the actual source repository plus other Git roots containing the
    # destination. A missing Git marker does not authorize an arbitrary clone.
    repository = next((p for p in Path(__file__).absolute().parents if (p / ".git").exists()), None)
    if repository is not None and (directory == repository or repository in directory.parents):
        raise ValueError("initialization evidence must remain outside the source repository")
    if any((p / ".git").exists() for p in (directory, *directory.parents)):
        raise ValueError("initialization evidence must remain outside Git")
    if require_owner:
        marker = _read_small(directory / "pals-artifact-owner.json", MAX_METADATA_BYTES)
        _keys(marker, ("schema", "owner"), "artifact owner")
        if marker != {"schema": OWNER_SCHEMA, "owner": "rz-pals-model"}:
            raise ValueError("artifact directory belongs to another workload")
    return directory


def _read_small(path, maximum):
    path = _plain_path(path)
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    return _parse_json(raw, maximum)


def _file_identity(path, expected_name, maximum):
    path = _plain_path(path)
    if path.name != expected_name:
        raise ValueError("only the registered named artifact may be joined")
    before = path.stat()
    if not 0 < before.st_size <= maximum:
        raise ValueError("artifact size exceeds the registered limit")
    hasher = hashlib.sha256()
    consumed = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise ValueError("artifact changed before hashing")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            consumed += len(block)
            if consumed > maximum:
                raise ValueError("artifact grew beyond its registered limit")
            hasher.update(block)
        after_fd = os.fstat(stream.fileno())
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if consumed != before.st_size or identity(before) != identity(after_fd) or identity(before) != identity(after):
        raise ValueError("artifact changed during hashing")
    return {"name": expected_name, "bytes": consumed, "sha256": hasher.hexdigest()}


def validate_source_registration(registration):
    """Validate a predeclared source pin list; do not infer a Git commit."""
    return _validate_sources(registration, legacy=False)


def validate_legacy_source_registration(registration):
    """The V2 observation has its own source domain; never relabel it as G0."""
    return _validate_sources(registration, legacy=True)


def _validate_sources(registration, *, legacy):
    _keys(registration, ("schema", "source_commit", "pins"), "initialization sources")
    if registration["schema"] != (LEGACY_SOURCE_SCHEMA if legacy else SOURCE_SCHEMA):
        raise ValueError("initialization source domain mismatch")
    commit = registration["source_commit"]
    if type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("source commit requires the externally registered full Git SHA")
    _keys(registration["pins"], SOURCE_FILES, "fixed initialization source pins")
    for digest in registration["pins"].values():
        _sha(digest, "source file digest")
    return registration


def observe_initialization_sources():
    """Read only the fixed source modules, without Git or dependency import."""
    root = Path(__file__).absolute().parent
    return {name: _file_identity(root / name, name, MAX_EVIDENCE_BYTES)["sha256"] for name in SOURCE_FILES}


def _join_sources(registration, *, legacy=False):
    _validate_sources(registration, legacy=legacy)
    if observe_initialization_sources() != registration["pins"]:
        raise ValueError("registered source pins differ from the actual initializer sources")


def validate_checkpoint_metadata(metadata):
    """Pure validation of the historical producer's new V3 G0 metadata."""
    return _validate_metadata(metadata, legacy=False)


def validate_legacy_checkpoint_metadata(metadata):
    """Observe the unchanged V2 checkpoint semantics, not V3 initialization."""
    return _validate_metadata(metadata, legacy=True)


def _validate_metadata(metadata, *, legacy):
    _keys(metadata, ("schema", "checkpoint", "checkpoint_sha256", "config", "seed", "trained",
                    "training_steps", "external_weights", "external_teacher", "precision", "tf32",
                    "torch_version", "parameter_count", "roles", "status"), "G0 checkpoint metadata")
    expected = ModelConfig.for_profile(LEGACY_PROFILE if legacy else VIRTUAL_REVIEW_PROFILE)
    if (metadata["schema"] != expected.model_semantics or metadata["checkpoint"] != "untrained.pt"
            or metadata["config"] != expected.to_dict() or type(metadata["seed"]) is not int or metadata["seed"] != 23
            or type(metadata["training_steps"]) is not int or metadata["training_steps"] != 0
            or metadata["roles"] != list(ROLES) or metadata["status"] != "random_initialization_only"
            or metadata["precision"] != "fp32" or any(metadata[key] is not False for key in
                ("trained", "external_weights", "external_teacher", "tf32"))):
        raise ValueError("G0 metadata is not the registered fresh, untrained virtual-review initialization")
    # Equality alone would accept bool dimensions as integers.
    _keys(metadata["config"], expected.to_dict(), "G0 model configuration")
    ModelConfig(**metadata["config"]).validate()
    _sha(metadata["checkpoint_sha256"], "checkpoint digest")
    _text(metadata["torch_version"], "producer Torch version")
    _uint(metadata["parameter_count"], "parameter count", MAX_PARAMETER_VALUES)
    if metadata["parameter_count"] == 0:
        raise ValueError("G0 metadata needs a nonempty actual parameter inventory")
    return metadata


def validate_parameter_inventory(inventory, parameter_count):
    """Pure check of actual weights-only inventory records, not tensor loads."""
    if type(inventory) is not list or not 1 <= len(inventory) <= MAX_PARAMETER_TENSORS:
        raise ValueError("complete parameter inventory must be bounded and nonempty")
    previous = ""
    total = 0
    for entry in inventory:
        _keys(entry, ("name", "shape", "dtype", "bytes", "sha256"), "parameter inventory")
        _text(entry["name"], "parameter name")
        if entry["name"] <= previous:
            raise ValueError("parameter inventory names must be unique and sorted")
        previous = entry["name"]
        shape = entry["shape"]
        if type(shape) is not list or len(shape) > 8:
            raise ValueError("parameter shape rank exceeds the registered limit")
        for size in shape:
            _uint(size, "parameter dimension", 1048576)
            if size == 0:
                raise ValueError("parameter dimensions must be nonzero")
        count = math.prod(shape)
        if (entry["dtype"] != "fp32_le" or type(entry["bytes"]) is not int or entry["bytes"] != count * 4
                or count > MAX_PARAMETER_VALUES):
            raise ValueError("parameter inventory shape/dtype/byte count mismatch")
        _sha(entry["sha256"], "parameter bit digest")
        total += count
    _uint(parameter_count, "parameter count", MAX_PARAMETER_VALUES)
    if total != parameter_count:
        raise ValueError("weights-only inventory does not cover the metadata parameter count")
    return inventory


def validate_partial_copy_evidence(copy, source_inventory, target_inventory):
    """Check complete actual copy records; a prevalidation plan is insufficient.

    This pure validator grants no initializer capability. Partial publication
    needs a separately registered actual copy/checkpoint producer; the current
    G0 producer intentionally supports fresh random initialization only.
    """
    fields = ("schema", "allowlist", "copied", "ignored_source", "ignored_target",
              "source_profile", "target_profile", "learned_transfer_claim",
              "automatic_asset_conversion", "nn_forwards", "optimizer_steps")
    _keys(copy, fields, "partial copy evidence")
    if (copy["schema"] != "rz-pals-actual-partial-initialization/1" or copy["target_profile"] != VIRTUAL_REVIEW_PROFILE
            or copy["learned_transfer_claim"] is not False or copy["automatic_asset_conversion"] is not False
            or type(copy["nn_forwards"]) is not int or copy["nn_forwards"] != 0
            or type(copy["optimizer_steps"]) is not int or copy["optimizer_steps"] != 0):
        raise ValueError("partial initialization cannot assert learned transfer or execution acceptance")
    _text(copy["source_profile"], "source profile")
    indices = []
    for inventory in (source_inventory, target_inventory):
        if type(inventory) is not list or not 1 <= len(inventory) <= MAX_PARAMETER_TENSORS:
            raise ValueError("partial initialization requires both complete parameter inventories")
        indexed = {}
        previous = ""
        values = 0
        for entry in inventory:
            _keys(entry, ("name", "shape", "dtype", "bytes", "sha256", "semantics"), "partial parameter inventory")
            plain = {key: entry[key] for key in ("name", "shape", "dtype", "bytes", "sha256")}
            validate_parameter_inventory([plain], entry["bytes"] // 4 if type(entry["bytes"]) is int else -1)
            _text(entry["semantics"], "parameter semantics")
            if entry["name"] <= previous:
                raise ValueError("partial inventory names must be sorted and unique")
            indexed[entry["name"]] = entry
            previous = entry["name"]
            values += entry["bytes"] // 4
            if values > MAX_PARAMETER_VALUES:
                raise ValueError("complete partial inventory exceeds the registered parameter budget")
        indices.append(indexed)
    source, target = indices
    allowlist = copy["allowlist"]
    if type(allowlist) is not dict or not allowlist or len(allowlist) > MAX_PARAMETER_TENSORS:
        raise ValueError("partial copy needs an explicit nonempty semantic allowlist")
    for name, semantics in allowlist.items():
        _text(name, "allowlisted parameter name")
        _text(semantics, "allowlisted semantics")
        if name not in source or name not in target:
            raise ValueError("allowlisted parameter is absent from complete inventories")
        left, right = source[name], target[name]
        if (left["shape"] != right["shape"] or left["dtype"] != right["dtype"]
                or left["semantics"] != semantics or right["semantics"] != semantics):
            raise ValueError("partial copy changes a name/shape/dtype/semantic meaning")
    if type(copy["copied"]) is not list or len(copy["copied"]) != len(allowlist):
        raise ValueError("actual copied parameters must exactly cover the allowlist")
    expected = []
    for name in sorted(allowlist):
        left, right = source[name], target[name]
        if left["sha256"] != right["sha256"]:
            raise ValueError("actual copied target parameter bits differ from source")
        expected.append({"name": name, "shape": left["shape"], "dtype": "fp32_le",
                         "semantics": allowlist[name], "source_sha256": left["sha256"],
                         "target_sha256": right["sha256"], "operation": "copied_exact_fp32_bits"})
    if copy["copied"] != expected:
        raise ValueError("actual copy records differ; a prepared copy plan is not execution evidence")
    if (copy["ignored_source"] != sorted(set(source) - set(allowlist))
            or copy["ignored_target"] != sorted(set(target) - set(allowlist))):
        raise ValueError("ignored parameter lists must exhaustively account for unselected parameters")
    return copy


def _claims():
    return {"training_steps": 0, "nn_forwards": 0, "optimizer_steps": 0, "trained": False,
            "diagnostic": False, "external_weights": False, "external_teacher": False,
            "learned_transfer_claim": False, "automatic_asset_conversion": False}


def _acceptance(*, legacy=False):
    return {"checkpoint_tensor_validation": "finite_cpu_fp32_weights_only_inventory",
            "independent_numeric_validation": "pending", "export_validation": "pending",
            "arena_eligibility": "not_granted", "capability_scope": "legacy_initializer_observation_only" if legacy else "initializer_return_and_file_identity_only"}


def _model_registration(*, legacy):
    config = ModelConfig.for_profile(LEGACY_PROFILE if legacy else VIRTUAL_REVIEW_PROFILE)
    return {"schema": LEGACY_MODEL_SCHEMA if legacy else MODEL_SCHEMA_V3, "semantics": config.model_semantics,
            "encoding": config.encoding, "profile": config.profile, "config": config.to_dict(),
            "roles": list(ROLES), "seed": 23}


def validate_initialization_evidence(evidence):
    """Pure strict schema validation; never reconstruct an execution capability."""
    return _validate_evidence(evidence, legacy=False)


def validate_legacy_initialization_observation(evidence):
    """Strict V2 observation parser. It cannot admit a V3/G0 generation."""
    return _validate_evidence(evidence, legacy=True)


def _validate_evidence(evidence, *, legacy):
    _keys(evidence, ("schema", "generation_id", "initialization_id", "model", "source", "checkpoint", "metadata",
                     "metadata_fields", "checkpoint_inventory", "initialization", "software",
                     "claims", "acceptance"), "G0 initialization evidence")
    if (evidence["schema"] != (LEGACY_EVIDENCE_SCHEMA if legacy else EVIDENCE_SCHEMA)
            or evidence["generation_id"] != ("legacy_baseline" if legacy else "g0")):
        raise ValueError("G0 initialization evidence domain/generation mismatch")
    _text(evidence["initialization_id"], "initialization execution identity", 128)
    model = evidence["model"]
    _keys(model, ("schema", "semantics", "encoding", "profile", "config", "roles", "seed"), "G0 model")
    expected_model = _model_registration(legacy=legacy)
    if model != expected_model or type(model["seed"]) is not int:
        raise ValueError("G0 model/encoding/role/seed registration mismatch")
    ModelConfig(**model["config"]).validate()
    metadata = _validate_metadata(evidence["metadata_fields"], legacy=legacy)
    _validate_sources(evidence["source"], legacy=legacy)
    for key, name, maximum in (("checkpoint", "untrained.pt", MAX_CHECKPOINT_BYTES),
                               ("metadata", "checkpoint.json", MAX_METADATA_BYTES)):
        item = evidence[key]
        _keys(item, ("name", "bytes", "sha256"), "named " + key)
        _uint(item["bytes"], key + " byte count", maximum)
        if item["name"] != name or item["bytes"] == 0:
            raise ValueError("G0 registered artifact name/size mismatch")
        _sha(item["sha256"], key + " digest")
    if evidence["checkpoint"]["sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes and producer metadata are not joined")
    inventory = validate_parameter_inventory(evidence["checkpoint_inventory"], metadata["parameter_count"])
    initialization = evidence["initialization"]
    expected_initialization = {"mode": "fresh_random", "producer": PRODUCER,
        "invocation_status": "captured_registered_initializer_return", "inventory_producer": "rz_pals_model.controlled_initialization._parameter_inventory",
        "allowlist": {}, "copied": [], "ignored_source": [], "ignored_target": [entry["name"] for entry in inventory],
        "partial_initialization": "requires_separate_actual_producer_registration"}
    if initialization != expected_initialization:
        raise ValueError("fresh initializer evidence cannot assert a partial copy or another producer")
    software = evidence["software"]
    _keys(software, ("pytorch_version", "python_version", "python_implementation", "torch_git_version",
                     "torch_cuda_build", "torch_thread_settings", "deterministic_algorithms",
                     "tf32_matmul", "tf32_cudnn", "observation", "device_capability"), "actual software readback")
    for key in ("pytorch_version", "python_version", "python_implementation"):
        _text(software[key], key)
    for key in ("torch_git_version", "torch_cuda_build"):
        if software[key] is not None:
            _text(software[key], key)
    _keys(software["torch_thread_settings"], ("intra", "inter"), "actual Torch thread settings")
    for count in software["torch_thread_settings"].values():
        _uint(count, "actual Torch thread count", 65536)
        if count == 0:
            raise ValueError("actual Torch thread count must be positive")
    for key in ("deterministic_algorithms", "tf32_matmul", "tf32_cudnn"):
        if type(software[key]) is not bool:
            raise ValueError("actual Torch execution settings require boolean observations")
    if (software["pytorch_version"] != metadata["torch_version"]
            or software["observation"] != "already_loaded_torch_after_registered_initializer"
            or software["device_capability"] != "not_probed"):
        raise ValueError("Torch software declaration is not joined to actual producer readback")
    if (type(evidence["claims"]) is not dict or evidence["claims"] != _claims()
            or any(type(evidence["claims"].get(k)) is not int for k in ("training_steps", "nn_forwards", "optimizer_steps"))
            or any(evidence["claims"].get(k) is not False for k in ("trained", "diagnostic", "external_weights",
                    "external_teacher", "learned_transfer_claim", "automatic_asset_conversion"))
            or evidence["acceptance"] != _acceptance(legacy=legacy)):
        raise ValueError("initializer observation cannot grant training/numeric/export/arena acceptance")
    if len(_canonical(evidence)) > MAX_EVIDENCE_BYTES:
        raise ValueError("G0 initialization evidence exceeds its registered limit")
    return evidence


def read_initialization_evidence(path, *, join_files=True):
    """Read observation evidence. This API cannot issue a producer capability.

    With ``join_files=False`` only the schema is checked; that result cannot
    replace pre-child asset identity validation. The default joins named files
    and exact historical metadata bytes from the same owned external directory.
    """
    return _read_evidence(path, join_files=join_files, legacy=False)


def read_legacy_initialization_observation(path, *, join_files=True):
    """Read only an observed V2 asset; never issue V3/G0 authority."""
    return _read_evidence(path, join_files=join_files, legacy=True)


def _read_evidence(path, *, join_files, legacy):
    if type(join_files) is not bool:
        raise ValueError("file-join scope must be an explicit boolean")
    path = _plain_path(path)
    name = LEGACY_EVIDENCE_NAME if legacy else EVIDENCE_NAME
    if path.name != name:
        raise ValueError("initialization evidence requires its registered filename")
    directory = _external_directory(path.parent)
    identity = _file_identity(path, name, MAX_EVIDENCE_BYTES)
    evidence = _read_small(path, MAX_EVIDENCE_BYTES)
    _validate_evidence(evidence, legacy=legacy)
    if join_files:
        _join_asset_files(directory, evidence)
    if _file_identity(path, name, MAX_EVIDENCE_BYTES) != identity:
        raise ValueError("initialization observation changed while its named assets were joined")
    return {"evidence": evidence, "evidence_file": identity, "validation": "schema_and_named_file_identity" if join_files else "schema_only",
            "execution_capability": "not_issued", "numeric_acceptance": "pending"}


def _join_asset_files(directory, evidence):
    checkpoint = _file_identity(directory / "untrained.pt", "untrained.pt", MAX_CHECKPOINT_BYTES)
    metadata_identity = _file_identity(directory / "checkpoint.json", "checkpoint.json", MAX_METADATA_BYTES)
    metadata = _read_small(directory / "checkpoint.json", MAX_METADATA_BYTES)
    if checkpoint != evidence["checkpoint"] or metadata_identity != evidence["metadata"] or metadata != evidence["metadata_fields"]:
        raise ValueError("registered initialization evidence no longer matches named asset bytes")
    if (_file_identity(directory / "untrained.pt", "untrained.pt", MAX_CHECKPOINT_BYTES) != checkpoint
            or _file_identity(directory / "checkpoint.json", "checkpoint.json", MAX_METADATA_BYTES) != metadata_identity):
        raise ValueError("named assets changed while the initialization observation was joined")


class _InitializationCapture:
    """A one-use in-process capture, deliberately not constructible from JSON."""
    __slots__ = ("_directory", "_encoded", "_used", "_legacy")

    def __init__(self, key, directory, evidence, *, legacy=False):
        if key is not _CAPTURE_KEY:
            raise TypeError("only the registered initializer can issue a capture")
        object.__setattr__(self, "_directory", directory)
        object.__setattr__(self, "_encoded", _canonical(_validate_evidence(evidence, legacy=legacy)))
        object.__setattr__(self, "_used", False)
        object.__setattr__(self, "_legacy", legacy)

    def __setattr__(self, name, value):
        raise TypeError("initializer captures are immutable; JSON or caller edits cannot change their authority")


def _publish_capture(capture):
    if type(capture) is not _InitializationCapture or capture._used:
        raise TypeError("publication requires a live, unused producer capture, not a bool or receipt")
    directory = _external_directory(capture._directory)
    evidence = _parse_json(capture._encoded, MAX_EVIDENCE_BYTES)
    _join_sources(evidence["source"], legacy=capture._legacy)
    _join_asset_files(directory, evidence)
    name = LEGACY_EVIDENCE_NAME if capture._legacy else EVIDENCE_NAME
    _atomic_publish_bytes(directory, capture._encoded + b"\n", name=name)
    object.__setattr__(capture, "_used", True)
    return _read_evidence(directory / name, join_files=True, legacy=capture._legacy)


def _atomic_publish_bytes(directory, encoded, *, name=EVIDENCE_NAME):
    """Internal no-clobber byte publication, not an initializer capability."""
    directory = _external_directory(directory)
    if type(encoded) is not bytes or not 0 < len(encoded) <= MAX_EVIDENCE_BYTES:
        raise ValueError("published evidence must be bounded bytes")
    if name not in (EVIDENCE_NAME, LEGACY_EVIDENCE_NAME):
        raise ValueError("publication permits only registered observation filenames")
    target = directory / name
    temporary = directory / (name + ".partial")
    if target.exists() or temporary.exists():
        raise FileExistsError("initialization evidence already exists; preserve it rather than overwrite")
    # Publish with a same-filesystem hard link, whose creation is atomic and
    # refuses an existing destination. replace() could overwrite a race winner.
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, target)
    except BaseException:
        # Keep the owned partial evidence for diagnosis; no successful receipt.
        raise
    temporary.unlink()
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return target


def _registered_function(module, name, source_name):
    function = getattr(module, name, None)
    expected = Path(__file__).absolute().parent / source_name
    if (not inspect.isfunction(function) or function.__module__ != module.__name__ or function.__name__ != name
            or Path(inspect.getsourcefile(function) or "").absolute() != expected
            or type(getattr(module, "__file__", None)) is not str or Path(module.__file__).absolute() != expected):
        raise ValueError("registered initializer/inventory producer identity was replaced")
    return function


def initialize_g0_with_evidence(directory, source_registration, *, initialization_id, max_seconds=300):
    """Explicit actual G0 initializer; caller owns the process and hard deadline.

    This V3 writer calls the registered producer itself;
    no injected function, ``actual=True`` flag, old checkpoint or JSON receipt
    can issue its capture. Stage bounds do not replace a caller-owned timeout.
    The function performs no NN forward, optimizer update, export or GPU probe.
    """
    return _initialize_evidence(directory, source_registration, initialization_id=initialization_id,
                                max_seconds=max_seconds, legacy=False)


def initialize_legacy_with_observation(directory, source_registration, *, initialization_id, max_seconds=300):
    """Actual new V2 baseline capture, kept outside the V3/G0 domain.

    Uses the unchanged FullLineInteractionV2 producer and checkpoint format.
    It grants no new-model numeric acceptance or actual-generation eligibility.
    """
    return _initialize_evidence(directory, source_registration, initialization_id=initialization_id,
                                max_seconds=max_seconds, legacy=True)


def _initialize_evidence(directory, source_registration, *, initialization_id, max_seconds, legacy):
    if type(max_seconds) is not int or not 1 <= max_seconds <= 3600:
        raise ValueError("initializer requires a finite, registered wall window")
    _text(initialization_id, "initialization execution identity", 128)
    _join_sources(source_registration, legacy=legacy)
    directory = _external_directory(directory, require_owner=False)
    # A dedicated new directory prevents provenance upgrades of old diagnostic
    # or baseline assets. Only an already registered owner marker may preexist.
    if any(item.name != "pals-artifact-owner.json" for item in directory.iterdir()):
        raise FileExistsError("G0 initializer requires a new output directory")
    if (directory / "pals-artifact-owner.json").exists():
        _external_directory(directory)
    from . import artifacts, controlled_initialization
    producer = _registered_function(artifacts, "initialize_checkpoint", "artifacts.py")
    inventory_producer = _registered_function(controlled_initialization, "_parameter_inventory", "controlled_initialization.py")
    start = time.monotonic()
    profile = LEGACY_PROFILE if legacy else VIRTUAL_REVIEW_PROFILE
    metadata = _parse_json(_canonical(producer(directory, 23, profile=profile)), MAX_METADATA_BYTES)
    _validate_metadata(metadata, legacy=legacy)
    directory = _external_directory(directory)
    if _read_small(directory / "checkpoint.json", MAX_METADATA_BYTES) != metadata:
        raise ValueError("producer return differs from its committed metadata")
    checkpoint = _file_identity(directory / "untrained.pt", "untrained.pt", MAX_CHECKPOINT_BYTES)
    if checkpoint["sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("producer metadata does not name the actual checkpoint bytes")
    if time.monotonic() - start > max_seconds:
        raise TimeoutError("initialization window expired; preserve incomplete assets")
    inventory = inventory_producer(directory / "untrained.pt", profile, 23, metadata["parameter_count"])
    validate_parameter_inventory(inventory, metadata["parameter_count"])
    torch = sys.modules.get("torch")
    if not isinstance(torch, types.ModuleType) or not getattr(torch, "__file__", None):
        raise ValueError("actual initializer did not leave a loaded Torch module for software readback")
    torch_version = getattr(torch, "__version__", None)
    if not isinstance(torch_version, str) or torch_version != metadata["torch_version"]:
        raise ValueError("actual Torch readback differs from producer metadata")
    torch_version = str(torch_version)
    version = getattr(torch, "version", None)
    evidence = {"schema": LEGACY_EVIDENCE_SCHEMA if legacy else EVIDENCE_SCHEMA,
        "generation_id": "legacy_baseline" if legacy else "g0", "initialization_id": initialization_id,
        "model": _model_registration(legacy=legacy),
        "source": json.loads(_canonical(source_registration)), "checkpoint": checkpoint,
        "metadata": _file_identity(directory / "checkpoint.json", "checkpoint.json", MAX_METADATA_BYTES),
        "metadata_fields": metadata, "checkpoint_inventory": inventory,
        "initialization": {"mode": "fresh_random", "producer": PRODUCER,
            "invocation_status": "captured_registered_initializer_return",
            "inventory_producer": "rz_pals_model.controlled_initialization._parameter_inventory",
            "allowlist": {}, "copied": [], "ignored_source": [], "ignored_target": [entry["name"] for entry in inventory],
            "partial_initialization": "requires_separate_actual_producer_registration"},
        "software": {"pytorch_version": torch_version, "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(), "torch_git_version": getattr(version, "git_version", None),
            "torch_cuda_build": getattr(version, "cuda", None),
            "torch_thread_settings": {"intra": torch.get_num_threads(), "inter": torch.get_num_interop_threads()},
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "observation": "already_loaded_torch_after_registered_initializer", "device_capability": "not_probed"},
        "claims": _claims(), "acceptance": _acceptance(legacy=legacy)}
    _join_sources(source_registration, legacy=legacy)
    if _file_identity(directory / "untrained.pt", "untrained.pt", MAX_CHECKPOINT_BYTES) != checkpoint:
        raise ValueError("checkpoint changed while actual inventory was collected")
    if time.monotonic() - start > max_seconds:
        raise TimeoutError("initialization evidence window expired; preserve unregistered assets")
    return _publish_capture(_InitializationCapture(_CAPTURE_KEY, directory, evidence, legacy=legacy))
