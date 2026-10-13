"""Explicit first-generation window dispatch for an original native producer.

The normal command remains metadata-only. ``--execute`` is a dispatch request,
not a proof made from a JSON receipt: an embedding host must provide its live
``NativeGenerationTrainingProducer``. A source-owned original trainer child
accepts exact externally pinned startup bytes and original monotonic deadlines.
Its parent retains the original opaque collection/action archive; this child
does not reconstruct that native game authority from metadata. The separate Rust
``VerifiedFrozenPcIntakeV3`` scope applies to newly frozen P/C V-action data;
it is not a universal prerequisite for the P/C optimizer itself.

Registration, file pinning and source validation use the standard library.
Torch/model/optimizer operations occur only in ``execute_registered_window``
after the live producer's original binding check. No import-path plugin or
serialized callback is supported.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager, ExitStack
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time

from .generation_data import digest, fields, identifier, sha, strict_json, uint
from .generation_chunks import ChunkGenerationDataset, MANIFEST_DOMAIN, MAX_CORPUS_BYTES, MAX_INDEX_BYTES
from .generation_training import (CHECKPOINT_DOMAIN, GenerationLedger, GenerationOptimizers,
    GenerationSampler, configure_generation_runtime, generation_admission, restore_checkpoint,
    spec_digest, train_generation_window, validate_checkpoint, validate_spec)

REGISTRATION_DOMAIN = "rz-pals-generation-training-window-registration/1"
RESULT_DOMAIN = "rz-pals-generation-training-cli-window/1"
FROZEN_COLLECTION_DOMAIN = "rz-pals-frozen-pc-collection-export/1"
STARTUP_DOMAIN = "rz-pals-original-generation-trainer-startup/1"
MAX_JSON_BYTES = 16 * 1024**2
MAX_WALL_NS = 60 * 60 * 10**9
SOURCE_COMPONENTS = ("trainer", "checkpoint", "loss_recipe", "model_reference", "model_encoding", "data_v3", "collector")
_QUARANTINED_TRAINING_STATES = []


class NativeTrainingProducerUnavailable(ValueError):
    """No original process/native intake owner was supplied by the host."""


class GenerationTrainingWindowFailed(RuntimeError):
    """Preserve charged state for the original owner; never retry implicitly."""
    def __init__(self, error, state):
        super().__init__(str(error))
        self.error, self.state = error, state


@dataclass
class LiveGenerationTrainingState:
    """Live host-owned values; this type has no JSON parser.

    The host retains native intake, loaded backend, checkpoint byte snapshot,
    corpus descriptors and sole GPU ownership until its context exits. Its
    ``check_original_binding`` callback rechecks the original live authority,
    not a boolean from the registration. ``checkpoint_payload`` is the already
    decoded, independently pinned actual checkpoint on resume; initialization
    uses the live model that the original host checked against the G0 bytes.
    """
    model: object
    pc_dataset: ChunkGenerationDataset
    v_dataset: ChunkGenerationDataset | None
    check_original_binding: object
    checkpoint_payload: dict | None = None
    optimizers: object | None = None
    ledger: GenerationLedger | None = None
    sampler_states: dict | None = None
    retained_pc_sampler_states: dict | None = None
    physical_completion: str = "not_started"
    window_initial_completed: int = 0
    window_initial_dispatched: int = 0
    temporary_snapshot_bytes: int = 0
    owned_samplers: dict | None = None
    saved_checkpoints: list | None = None
    runtime_readback: dict | None = None


class NativeGenerationTrainingProducer(ABC):
    """Embedding-only native owner; never instantiate it from a JSON claim.

    ``open_window`` returns a context manager yielding live state. It must join
    the exact registration/model/runtime/source/data/checkpoint pins against
    the original collection archive owner and actual loaded PyTorch model,
    additionally join frozen P/C intake for the V-action corpus, enforce
    the sole GPU lease, and preserve/drain unknown physical completion on exit.
    An ordinary metadata receipt or a foreign checkpoint is insufficient.
    """
    @abstractmethod
    def open_window(self, registration, prepared, until_ns):
        raise NotImplementedError


def _safe_absolute(path):
    if type(path) is not str or not 1 <= len(path) <= 4096 or any(ord(c) < 32 for c in path):
        raise ValueError("explicit bounded artifact path required")
    result = Path(path)
    if not result.is_absolute():
        raise ValueError("registered training paths must be absolute")
    for part in result.parts:
        lower = part.lower()
        if (lower in (".", "..") or lower.startswith(".env") or lower.endswith((".key", ".pem"))
                or any(x in lower for x in ("credential", "service-account", "service_account", "serviceaccount", "ssh-key", "sshkey",
                    "api-key", "api_key", "apikey", "client_secret", "access_token", "refresh_token"))):
            raise ValueError("secret/noncanonical training path forbidden")
    return result


def _artifact(value, maximum=MAX_JSON_BYTES):
    fields(value, ("path", "sha256", "bytes"), "training artifact pin")
    _safe_absolute(value["path"])
    sha(value["sha256"])
    if not uint(value["bytes"], maximum):
        raise ValueError("nonempty bounded training artifact required")
    return value


def _file_identifier(value):
    identifier(value)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None:
        raise ValueError("training file identifier must be a bounded plain basename")
    return value


def validate_registration(value):
    fields(value, ("schema", "window_id", "spec", "phase", "datasets", "checkpoint",
        "source_components", "runtime_sha256", "model_sha256", "frozen_pc_sha256", "model_epoch",
        "training_runtime", "staging_root", "index_root", "checkpoint_directory", "receipt_path",
        "max_wall_ns", "max_updates", "max_output_bytes"), "training window registration")
    if value["schema"] != REGISTRATION_DOMAIN:
        raise ValueError("unregistered training window domain")
    _file_identifier(value["window_id"])
    _artifact(value["spec"])
    if not value["spec"]["path"].endswith(".json"):
        raise ValueError("training spec must be an explicit JSON metadata file")
    if value["phase"] not in ("pc", "v"):
        raise ValueError("unregistered training phase")
    fields(value["datasets"], ("pc", "v"), "registered phase datasets")
    for phase, pin in value["datasets"].items():
        if pin is not None:
            fields(pin, ("manifest", "manifest_sha256", "corpus_root"), "registered chunk corpus")
            _artifact(pin["manifest"])
            if not pin["manifest"]["path"].endswith(".json"):
                raise ValueError("chunk manifest must be an explicit JSON metadata file")
            sha(pin["manifest_sha256"])
            _safe_absolute(pin["corpus_root"])
        elif phase == "pc" or value["phase"] == "v":
            raise ValueError("phase requires independently pinned chunk corpus")
    checkpoint = value["checkpoint"]
    fields(checkpoint, ("artifact", "domain", "mode"), "original training checkpoint")
    _artifact(checkpoint["artifact"], 1024**3)
    from .config import MODEL_SEMANTICS_V3
    if checkpoint["mode"] == "initialize":
        if checkpoint["domain"] != MODEL_SEMANTICS_V3 or value["phase"] != "pc":
            raise ValueError("initial generation requires the registered own G0 checkpoint")
    elif checkpoint["mode"] != "resume" or checkpoint["domain"] != CHECKPOINT_DOMAIN:
        raise ValueError("resume requires the actual-generation checkpoint domain")
    if not checkpoint["artifact"]["path"].endswith(".pt"):
        raise ValueError("training checkpoint must be an explicit nonsecret .pt artifact")
    fields(value["source_components"], SOURCE_COMPONENTS, "training source components")
    for pin in value["source_components"].values():
        sha(pin)
    for name in ("runtime_sha256", "model_sha256"):
        sha(value[name])
    if value["phase"] == "v":
        sha(value["frozen_pc_sha256"])
        if not uint(value["model_epoch"]):
            raise ValueError("V needs a newly frozen P/C epoch")
    elif value["frozen_pc_sha256"] is not None or value["model_epoch"] is not None:
        raise ValueError("P/C cannot override the G0 freeze/epoch")
    fields(value["training_runtime"], ("torch", "cuda", "cudnn", "device_name"), "actual training runtime registration")
    runtime = value["training_runtime"]
    if (runtime["torch"] != "2.8.0" or type(runtime["cuda"]) is not str or not 1 <= len(runtime["cuda"]) <= 64
            or not uint(runtime["cudnn"], 10**9) or runtime["device_name"] != "NVIDIA L4"):
        raise ValueError("registered PyTorch 2.8 CUDA L4 runtime required")
    for name in ("staging_root", "index_root", "checkpoint_directory", "receipt_path"):
        _safe_absolute(value[name])
    root = Path(value["staging_root"])
    repository = next((p for p in Path(__file__).absolute().parents if (p / ".git").exists()), None)
    if repository is not None and (root == repository or root.is_relative_to(repository)):
        raise ValueError("generation staging/cache/artifacts must remain outside source repository")
    for name in ("index_root", "checkpoint_directory", "receipt_path"):
        if Path(value[name]) == root or not Path(value[name]).is_relative_to(root):
            raise ValueError("index/checkpoint/receipt must remain under the explicit managed staging root")
    if not value["receipt_path"].endswith(".json"):
        raise ValueError("terminal receipt must be an explicit JSON metadata file")
    if not Path(checkpoint["artifact"]["path"]).is_relative_to(root):
        raise ValueError("checkpoint snapshot must be charged to the explicit managed staging root")
    for corpus in value["datasets"].values():
        if corpus is not None and not Path(corpus["corpus_root"]).is_relative_to(root):
            raise ValueError("actual corpus must remain under the explicit managed staging root")
    if not uint(value["max_wall_ns"], MAX_WALL_NS) or type(value["max_updates"]) is not int or value["max_updates"] != 250:
        raise ValueError("one finite registered 250-update window required")
    if not uint(value["max_output_bytes"], MAX_CORPUS_BYTES):
        raise ValueError("finite output ceiling within 18GiB required")
    if value["max_output_bytes"] <= MAX_JSON_BYTES:
        raise ValueError("staging needs separate bounded terminal receipt credit")
    return copy.deepcopy(value)


def registration_digest(value):
    return digest(REGISTRATION_DOMAIN, validate_registration(value))


def _pinned_bytes(pin, *, maximum, until_ns, clock):
    _artifact(pin, maximum)
    path = _safe_absolute(pin["path"])
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("training artifact path contains a link/reparse point")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != pin["bytes"]:
            raise ValueError("original training artifact identity/size differs")
        hasher, chunks, total = hashlib.sha256(), [], 0
        # Checkpoint bytes belong to the original producer snapshot. This helper
        # pins JSON only and never allocates a checkpoint-sized Python buffer.
        while total < before.st_size:
            if clock() >= until_ns:
                raise ValueError("original training window expired during registration")
            block = os.read(descriptor, min(65536, before.st_size - total))
            if not block:
                break
            total += len(block)
            hasher.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if (total != before.st_size or os.read(descriptor, 1) or hasher.hexdigest() != pin["sha256"]
                or identity(before) != identity(after) or identity(after) != identity(current)):
            raise ValueError("training artifact bytes/identity changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def prepare_registration(value, *, expected_registration_sha256, until_ns, clock=time.monotonic_ns):
    """Metadata/source pins only; no execution authority or model allocation."""
    registration = validate_registration(value)
    if registration_digest(registration) != sha(expected_registration_sha256):
        raise ValueError("training registration differs from separately supplied pin")
    spec = strict_json(_pinned_bytes(registration["spec"], maximum=MAX_JSON_BYTES, until_ns=until_ns, clock=clock).decode("utf-8", "strict"))
    validate_spec(spec)
    _file_identifier(spec["generation_id"])
    if (spec["runtime_sha256"] != registration["runtime_sha256"]
            or spec["model_sha256"] != registration["model_sha256"]):
        raise ValueError("registered runtime/model differs from generation recipe")
    if registration["checkpoint"]["mode"] == "initialize" and registration["checkpoint"]["artifact"]["sha256"] != spec["model_sha256"]:
        raise ValueError("own G0 binary pin differs from generation recipe")
    if registration["phase"] == "v" and registration["model_epoch"] <= spec["model_epoch"]:
        raise ValueError("V collection epoch must follow G0")
    manifests = {}
    from .generation_chunks import validate_manifest
    for phase, corpus in registration["datasets"].items():
        if corpus is None:
            continue
        raw = _pinned_bytes(corpus["manifest"], maximum=MAX_JSON_BYTES, until_ns=until_ns, clock=clock)
        manifest = strict_json(raw.decode("utf-8", "strict"))
        if manifest.get("schema") != MANIFEST_DOMAIN or validate_manifest(manifest) != corpus["manifest_sha256"]:
            raise ValueError("registered chunk manifest/domain differs")
        if phase == "pc" and corpus["manifest_sha256"] != spec["dataset_sha256"]:
            raise ValueError("P/C corpus differs from original generation recipe")
        manifests[phase] = manifest
    from .v5_source_registry import source_pins
    if source_pins(SOURCE_COMPONENTS) != registration["source_components"]:
        raise ValueError("actual trainer/data/model sources differ from registration")
    if clock() >= until_ns:
        raise ValueError("original training window expired before producer admission")
    return {"spec": spec, "manifests": manifests, "spec_sha256": spec_digest(spec),
            "registration_sha256": expected_registration_sha256, "actual_training_executed": False}


def _sampler_validator(state, live, payload):
    phase = GenerationLedger(**copy.deepcopy(payload["ledger"])).phase
    # The P/C terminal checkpoint still owns its P/C samplers. Fresh V data
    # initializes a distinct V sampler after validating those original cursors.
    pc_terminal = phase == "v" and payload["ledger"]["completed"]["validator"] == 0
    roles = ("proposer", "critic") if phase == "pc" or pc_terminal else ("validator",)
    fields(state, roles, "phase-owned resume samplers")
    dataset = live.pc_dataset if roles != ("validator",) else live.v_dataset
    if type(dataset) is not ChunkGenerationDataset:
        raise ValueError("resume sampler lacks original admitted corpus owner")
    for role in roles:
        GenerationSampler(dataset.rows("train", optimization=True), role, state=state[role])


def _check_runtime(spec, expected):
    """Actual device/software readback, distinct from collection runtime hash."""
    import torch
    if (torch.__version__.split("+")[0] != expected["torch"] or torch.version.cuda != expected["cuda"]
            or torch.backends.cudnn.version() != expected["cudnn"]
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != expected["device_name"]):
        raise ValueError("actual training runtime/device differs; no fallback or automatic precision change")
    if spec["device"] != "cuda:0" or spec["precision"] != "fp32" or spec["tf32"] is not False:
        raise ValueError("actual registered training must use sole CUDA FP32 TF32-off device")
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "torch_module_path": str(torch.__file__),
        "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(), "device": spec["device"],
        "precision": spec["precision"], "tf32_requested": spec["tf32"],
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "runtime_registration_checked": True}


def _staging_bytes(registration):
    """Bound the expressly registered generated root, without reading contents."""
    root = _safe_absolute(registration["staging_root"])
    for parent in (root, *root.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("managed staging root has a link/reparse ancestor")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("explicit existing unlinked managed staging root required")
    total, entries, seen = 0, 0, set()
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            entries += 1
            if entries > 131072:
                raise ValueError("managed training staging entry cap exceeded")
            path = Path(directory) / name
            _safe_absolute(str(path))
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("managed training staging contains a link/reparse point")
            if stat.S_ISREG(info.st_mode):
                identity = (info.st_dev, info.st_ino)
                if info.st_nlink != 1 or identity in seen:
                    raise ValueError("managed training staging contains aliased mutable files")
                seen.add(identity)
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("managed training staging contains a nonregular artifact")
            if total > registration["max_output_bytes"]:
                raise ValueError("managed training staging byte cap exceeded")
    return total


def _output_credit(registration, *, reserved_bytes=0):
    available = registration["max_output_bytes"] - _staging_bytes(registration) - MAX_JSON_BYTES - uint(reserved_bytes, MAX_CORPUS_BYTES)
    if available <= 0:
        raise ValueError("managed training staging has no artifact credit after terminal receipt reserve")
    return available


@contextmanager
def _bounded_checkpoint_snapshot(pin, *, until_ns, clock=time.monotonic_ns):
    """Reuse hash/snapshot verification with the exact declared file ceiling."""
    _artifact(pin, 1024**3)
    source = _safe_absolute(pin["path"])
    for parent in (source, *source.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("original checkpoint path has a link/reparse ancestor")
    from .generation_export import _hash_file
    with tempfile.TemporaryFile(mode="w+b", prefix="rz-generation-pinned-", dir=source.parent) as snapshot:
        class OriginalCreditWriter:
            def write(self, block):
                if clock() >= until_ns or snapshot.tell() + len(block) > pin["bytes"]:
                    raise ValueError("original checkpoint snapshot exceeded deadline/disk credit")
                return snapshot.write(block)
        actual, size = _hash_file(source, pin["bytes"], snapshot=OriginalCreditWriter())
        if actual != pin["sha256"] or size != pin["bytes"]:
            raise ValueError("checkpoint snapshot differs from independently registered exact bytes")
        snapshot.flush()
        snapshot.seek(0)
        yield snapshot, size


class PinnedOriginalTrainerChild(NativeGenerationTrainingProducer):
    """Source-owned child implementation, not a native-game authority issuer.

    The original Rust parent keeps its opaque series/action archive and passes
    the resulting exact external manifest pins. This child verifies those bytes
    and their native-chain joins, executes optimizer work itself, and reports
    only that work. Flags/JSON cannot manufacture the parent's archive owner.
    """
    def __init__(self):
        self.last_state = None

    @contextmanager
    def open_window(self, registration, prepared, until_ns):
        if os.name != "posix":
            raise ValueError("original generation trainer child requires Linux process/deadline transport")
        live = None
        self.last_state = None
        with ExitStack() as stack:
            _output_credit(registration)
            root_info = Path(registration["staging_root"]).stat()
            if root_info.st_uid != os.geteuid() or stat.S_IMODE(root_info.st_mode) & 0o077:
                raise ValueError("original training staging must be current-owner private")
            receipt = Path(registration["receipt_path"])
            if receipt.exists() or receipt.with_name(receipt.name + ".partial").exists():
                raise FileExistsError("original training terminal receipt already exists")
            snapshot_credit = registration["checkpoint"]["artifact"]["bytes"]
            def index_credit():
                credit = min(MAX_INDEX_BYTES, _output_credit(registration, reserved_bytes=snapshot_credit))
                if credit < 65536:
                    raise ValueError("original trainer has insufficient remaining bounded index credit")
                return credit
            pc = registration["datasets"]["pc"]
            pc_dataset = stack.enter_context(ChunkGenerationDataset(pc["corpus_root"], prepared["manifests"]["pc"],
                expected_manifest_sha256=pc["manifest_sha256"], index_root=registration["index_root"], max_index_bytes=index_credit()))
            v_dataset = None
            if registration["datasets"]["v"] is not None:
                v = registration["datasets"]["v"]
                v_dataset = stack.enter_context(ChunkGenerationDataset(v["corpus_root"], prepared["manifests"]["v"],
                    expected_manifest_sha256=v["manifest_sha256"], index_root=registration["index_root"], max_index_bytes=index_credit()))
            source = _safe_absolute(registration["checkpoint"]["artifact"]["path"])
            pin = registration["checkpoint"]["artifact"]
            if _output_credit(registration) < pin["bytes"]:
                raise ValueError("original pinned checkpoint snapshot exceeds actual remaining staging credit")
            if source.stat().st_size != pin["bytes"]:
                raise ValueError("original checkpoint registered size differs")
            with _bounded_checkpoint_snapshot(pin, until_ns=until_ns) as (snapshot, size):
                if size != pin["bytes"] or time.monotonic_ns() >= until_ns:
                    raise ValueError("original checkpoint size/window changed")
                # Verify the parent-owned deterministic environment before a
                # Torch CUDA query/context or first device allocation occurs.
                if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                    raise ValueError("original deterministic CUBLAS environment missing before CUDA initialization")
                import torch
                payload = torch.load(snapshot, map_location="cpu", weights_only=True)
                spec = prepared["spec"]
                _check_runtime(spec, registration["training_runtime"])
                from .config import ModelConfig, MODEL_SEMANTICS_V3
                from .model import PalsModel
                if type(payload) is not dict or payload.get("schema") != registration["checkpoint"]["domain"]:
                    raise ValueError("original initialization/resume checkpoint domain differs")
                config = ModelConfig(**payload["config"])
                config.validate()
                if config.profile != "virtual_review_v1":
                    raise ValueError("original checkpoint must use registered compact model")
                # Construct a private scratch model. Invalid resume validation
                # cannot mutate an embedding caller's model/RNG/optimizer.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(23)
                    model = PalsModel(config)
                initialize = registration["checkpoint"]["mode"] == "initialize"
                if initialize:
                    fields(payload, ("state_dict", "config", "seed", "schema", "training_steps", "trained"), "original G0 initialization checkpoint")
                    if (payload["schema"] != MODEL_SEMANTICS_V3 or type(payload["seed"]) is not int or payload["seed"] != 23
                            or type(payload["training_steps"]) is not int or payload["training_steps"] != 0 or payload["trained"] is not False):
                        raise ValueError("diagnostic/foreign/trained checkpoint cannot replace registered G0")
                    state = payload["state_dict"]
                    expected = model.state_dict()
                    if not isinstance(state, dict) or set(state) != set(expected):
                        raise ValueError("original G0 named parameter membership differs")
                    total = 0
                    for name, tensor in state.items():
                        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                                or tensor.dtype != torch.float32 or tensor.shape != expected[name].shape
                                or not bool(torch.isfinite(tensor).all())):
                            raise ValueError("original G0 tensor shape/dtype/value differs")
                        total += tensor.numel() * tensor.element_size()
                    if total > 128 * 1024**2:
                        raise ValueError("registered G0 parameter allocation cap exceeded")
                    model.load_state_dict(state, strict=True)
                def original_binding(current_registration, current_prepared, original_until):
                    if current_registration != registration or current_prepared["registration_sha256"] != prepared["registration_sha256"] or original_until != until_ns:
                        raise ValueError("original child registration/owner changed")
                    from .v5_source_registry import source_pins
                    if source_pins(SOURCE_COMPONENTS) != registration["source_components"]:
                        raise ValueError("active training sources changed")
                    live.runtime_readback = _check_runtime(spec, registration["training_runtime"])
                    _staging_bytes(registration)
                live = LiveGenerationTrainingState(model, pc_dataset, v_dataset, original_binding,
                    checkpoint_payload=None if initialize else payload, physical_completion="pending", temporary_snapshot_bytes=size,
                    saved_checkpoints=[])
                self.last_state = live
                try:
                    # Establish the lifetime owner before the first device
                    # allocation; even a partially failed .to() is fenced.
                    live.model = model.to(device=spec["device"], dtype=torch.float32).eval()
                    yield live
                finally:
                    try:
                        # Actual trainer owns Torch's device work. This fence
                        # cannot turn native data origin into a new proof.
                        torch.cuda.synchronize(0)
                        live.physical_completion = "confirmed"
                    except BaseException:
                        live.physical_completion = "unknown_quarantined"
                        import sys
                        _QUARANTINED_TRAINING_STATES.append((live, sys.exc_info()[1]))
                        raise


def execute_registered_window(registration, prepared, producer, *, until_ns, clock=time.monotonic_ns, cancelled=lambda: False):
    """Callable producer boundary. No JSON or default path can invoke it."""
    if type(producer) is not PinnedOriginalTrainerChild:
        raise NativeTrainingProducerUnavailable("OriginalNativeTrainingProducerUnavailable: source-owned private original trainer child required")
    if registration_digest(registration) != prepared["registration_sha256"] or spec_digest(prepared["spec"]) != prepared["spec_sha256"]:
        raise ValueError("prepared training registration changed before admission")
    if clock() >= until_ns:
        raise ValueError("original training window expired before opening native owner")
    with producer.open_window(copy.deepcopy(registration), copy.deepcopy(prepared), until_ns) as live:
        if type(live) is not LiveGenerationTrainingState or not callable(live.check_original_binding):
            raise NativeTrainingProducerUnavailable("live native producer state/binding callback required")
        try:
            live.check_original_binding(registration, prepared, until_ns)
            spec, phase = prepared["spec"], registration["phase"]
            if type(live.pc_dataset) is not ChunkGenerationDataset or live.pc_dataset.sha256 != spec["dataset_sha256"]:
                raise ValueError("original P/C dataset owner differs from registered pin")
            dataset = live.pc_dataset if phase == "pc" else live.v_dataset
            if type(dataset) is not ChunkGenerationDataset or dataset.sha256 != registration["datasets"][phase]["manifest_sha256"]:
                raise ValueError("actual phase corpus owner differs from registration")
            admission = generation_admission(spec, dataset, phase, frozen_pc_sha256=registration["frozen_pc_sha256"], model_epoch=registration["model_epoch"])
            initialize = registration["checkpoint"]["mode"] == "initialize"
            if initialize and any(value is not None for value in (live.checkpoint_payload, live.optimizers, live.ledger, live.sampler_states)):
                raise ValueError("initial window cannot inherit optimizer/checkpoint state")
            configure_generation_runtime(spec, execute=True, initialize_rng=initialize)
            # A source-only command never reaches this construction. On resume
            # the full checkpoint is validated before caller model/RNG changes.
            live.optimizers = GenerationOptimizers(live.model, spec, execute=True)
            if initialize:
                live.ledger, live.sampler_states = GenerationLedger(spec), None
            else:
                if type(live.checkpoint_payload) is not dict:
                    raise ValueError("resume requires original independently pinned checkpoint payload")
                checkpoint_ledger = validate_checkpoint(live.checkpoint_payload, live.model, live.optimizers, spec,
                    lambda value: _sampler_validator(value, live, live.checkpoint_payload))
                if checkpoint_ledger.phase != phase:
                    raise ValueError("registered phase differs from actual checkpoint phase")
                live.ledger, saved, _ = restore_checkpoint(live.checkpoint_payload, live.model, live.optimizers, spec,
                    lambda value: _sampler_validator(value, live, live.checkpoint_payload))
                if phase == "v" and live.ledger.completed["validator"] == 0:
                    live.retained_pc_sampler_states, live.sampler_states = copy.deepcopy(saved), None
                else:
                    live.sampler_states = saved
            live.window_initial_completed = sum(live.ledger.completed.values())
            live.window_initial_dispatched = sum(live.ledger.dispatched.values())
            if live.ledger.cost["output_bytes"] > registration["max_output_bytes"]:
                raise ValueError("retained output cost already exceeds registered window storage ceiling")
            remaining = until_ns - clock()
            if remaining <= 0:
                raise ValueError("original window expired before optimizer dispatch")
            def original_clock():
                if cancelled():
                    raise InterruptedError("original trainer cancellation; retain charged work and original owner")
                now = clock()
                if now >= until_ns:
                    raise TimeoutError("original trainer hard deadline expired; no new optimizer dispatch")
                return now
            def retain_progress(samplers, ledger):
                live.owned_samplers, live.ledger = samplers, ledger
            def retain_checkpoint(path, ledger):
                live.saved_checkpoints.append(path)
                live.ledger = ledger
            result, live.optimizers, live.ledger = train_generation_window(spec, dataset, live.model,
                optimizers=live.optimizers, ledger=live.ledger, sampler_states=live.sampler_states,
                frozen_pc_sha256=registration["frozen_pc_sha256"], model_epoch=registration["model_epoch"],
                checkpoint_directory=registration["checkpoint_directory"], max_wall_ns=remaining,
                execute=True, clock=original_clock, output_guard=lambda: _output_credit(registration, reserved_bytes=live.temporary_snapshot_bytes),
                progress_observer=retain_progress, checkpoint_observer=retain_checkpoint)
            live.sampler_states = result["sampler"]
            live.check_original_binding(registration, prepared, until_ns)
            return {"schema": RESULT_DOMAIN, "window_id": registration["window_id"],
                "registration_sha256": prepared["registration_sha256"], "window": result,
                "data_readback": {"dataset_sha256": dataset.sha256, "phase": phase, "coverage": admission["coverage"],
                    "head_provenance_and_native_chain_bytes_checked": True, "original_game_process_verified_here": False,
                    "original_archive_owner": "retained_by_launching_parent"},
                "v_sampler_handoff": "new_frozen_v_pool_pc_sampler_retained" if live.retained_pc_sampler_states is not None else "same_phase_exact_resume",
                "retained_pc_sampler": copy.deepcopy(live.retained_pc_sampler_states),
                "durable_checkpoint_boundary": bool(result["checkpoints"]),
                "uncheckpointed_updates": 0 if result["checkpoints"] else sum(live.ledger.completed.values()) % 250,
                "authority_scope": "source_owned_original_trainer_child_parent_keeps_native_archive",
                "native_intake_issued_by_this_cli": False, "arena_eligible": False}
        except BaseException as error:
            raise GenerationTrainingWindowFailed(error, live) from error


def validate_frozen_pc_boundary(ledger, *, model_epoch):
    """Pure boundary check; a phase name alone cannot authorize V collection."""
    ledger.validate()
    if (ledger.phase != "v" or ledger.completed != {"proposer": 2000, "critic": 2000, "validator": 0}
            or ledger.pending_microbatches or not ledger.checkpoint_boundary()
            or type(model_epoch) is not int or model_epoch <= ledger.spec["model_epoch"]):
        raise ValueError("frozen P/C collection export requires exact 4000-update boundary and new epoch")
    return ledger


def export_frozen_pc_collection(spec, checkpoint_path, expected_checkpoint_sha256, model, optimizers,
        pc_dataset, *, sampler_validator, model_epoch, directory, rules_profile_json, execute=False,
        until_ns=None, clock=time.monotonic_ns, output_guard=None):
    """Bridge the actual 4000-update checkpoint to Collection ONNX artifacts.

    This reuses the manual exporter. It does not execute a forward/update, issue
    a native loaded-backend capability, claim V training or publish a deployment
    candidate. The separately owned Rust frozen intake must still compare the
    named P/C checkpoint tensors with these actual graph initializers.
    """
    if not execute:
        raise ValueError("frozen P/C collection export requires explicit execution")
    if type(until_ns) is not int or not clock() < until_ns <= clock() + MAX_WALL_NS:
        raise ValueError("frozen P/C export requires an original finite registered deadline")
    validate_spec(spec)
    sha(expected_checkpoint_sha256)
    from .generation_export import (_hash_file, _atomic_json, _export_descriptors)
    source = _safe_absolute(str(checkpoint_path))
    target = _safe_absolute(str(directory))
    if source.suffix != ".pt" or target.exists():
        raise ValueError("original .pt checkpoint and new external export directory required")
    repository = next((p for p in Path(__file__).absolute().parents if (p / ".git").exists()), None)
    if repository is not None and (target == repository or target.is_relative_to(repository)):
        raise ValueError("frozen collection export must stay outside source repository")
    original_size = source.stat().st_size
    pin = {"path": str(source), "sha256": expected_checkpoint_sha256, "bytes": original_size}
    with _bounded_checkpoint_snapshot(pin, until_ns=until_ns, clock=clock) as (snapshot, size):
        import torch
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        ledger = validate_checkpoint(payload, model, optimizers, spec, sampler_validator)
        validate_frozen_pc_boundary(ledger, model_epoch=model_epoch)
        generation_admission(spec, pc_dataset, "pc")
        from .nonzero_training import _state_digest
        if (model.training is not False or optimizers.model is not model or optimizers.spec != spec
                or _state_digest(model.state_dict()) != _state_digest(payload["model"])):
            raise ValueError("frozen collection export caller differs from exact checkpoint state")
        from .artifacts import export_virtual_checkpoint, read_rules_profile
        read_rules_profile(rules_profile_json, model.config)
        if ledger.cost["output_bytes"] + 2 * size > MAX_CORPUS_BYTES:
            raise ValueError("frozen collection snapshot exceeds remaining managed output cap")
        metadata = {"config": model.config.to_dict(), "checkpoint_sha256": expected_checkpoint_sha256,
                    "trained": True, "training_steps": 4000}
        manifest = export_virtual_checkpoint(model, metadata, target / "collection", include_validator=True,
            rules_profile_json=rules_profile_json, output_guard=output_guard)
        descriptors = _export_descriptors(target, "collection", manifest)
        output_bytes = descriptors["manifest"]["bytes"] + sum(g["bytes"] for g in descriptors["graphs"])
        if ledger.cost["output_bytes"] + size + output_bytes > MAX_CORPUS_BYTES:
            raise ValueError("frozen collection graph output exceeds managed staging cap")
        if clock() >= until_ns:
            raise ValueError("frozen collection export exceeded original window; preserve partial artifacts")
        receipt = {"schema": FROZEN_COLLECTION_DOMAIN, "eligibility": "frozen_pc_collection_only",
            "generation_id": spec["generation_id"], "spec_sha256": spec_digest(spec),
            "checkpoint_sha256": expected_checkpoint_sha256, "checkpoint_bytes": size,
            "frozen_pc_sha256": payload["frozen_pc_sha256"], "model_epoch": model_epoch,
            "actual_updates": dict(ledger.completed), "pc_sampler": copy.deepcopy(payload["sampler"]),
            "collection": descriptors, "validator_present": True, "validator_training_updates": 0,
            "deployment_exported": False, "native_loaded_assets_verified": False,
            "numeric_status": "not_run", "cuda_status": "not_run", "arena_eligible": False}
        _atomic_json(target / "frozen-pc-collection.json", receipt, output_guard=output_guard)
        receipt_sha256, receipt_bytes = _hash_file(target / "frozen-pc-collection.json", 4 * 1024**2)
        return {"receipt": receipt, "receipt_file_sha256": receipt_sha256, "receipt_file_bytes": receipt_bytes,
                "actual_export_executed": True, "optimizer_updates_executed": 0, "arena_eligible": False}


def main(argv=None, *, producer=None, clock=time.monotonic_ns):
    """Keep legacy validation unchanged; opt-in dispatch needs a live host."""
    import argparse
    import sys
    actual = list(sys.argv[1:] if argv is None else argv)
    if "--execute" not in actual:
        from .generation_training import main as metadata_main
        return metadata_main(actual)
    if "--original-trainer-child" in actual:
        return original_trainer_child_main(actual, clock=clock)
    parser = argparse.ArgumentParser(prog="train-generation", description="explicit registered window; requires original native producer")
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--execution-registration", required=True)
    parser.add_argument("--registration-sha256", required=True)
    args = parser.parse_args(actual)
    # Legacy embedding arguments never recreate an original process window.
    # Production dispatch must use the exact startup + original deadline child
    # transport; callable owners use execute_registered_window explicitly.
    raise NativeTrainingProducerUnavailable("OriginalNativeTrainingProducerUnavailable: original-trainer-child startup/deadline transport required")


def validate_startup(value, *, expected_raw_sha256, raw_bytes, execution_until_ns, whole_until_ns, now_ns):
    """Pure original child envelope validation; no opaque archive is issued."""
    fields(value, ("schema", "execution_id", "registration_sha256", "registration", "original_started_ns",
        "execution_until_ns", "whole_until_ns", "dataset_pin_scope", "dependency_directories", "python_runtime"), "original trainer startup")
    if value["schema"] != STARTUP_DOMAIN or value["dataset_pin_scope"] != "external_original_archive_pin_native_chain_readback_only":
        raise ValueError("unregistered original trainer startup/scope")
    identifier(value["execution_id"])
    if type(raw_bytes) is not bytes or not 0 < len(raw_bytes) <= MAX_JSON_BYTES or hashlib.sha256(raw_bytes).hexdigest() != sha(expected_raw_sha256):
        raise ValueError("original trainer startup raw bytes differ from parent pin")
    registration = validate_registration(value["registration"])
    if registration_digest(registration) != sha(value["registration_sha256"]):
        raise ValueError("original trainer registration differs from parent startup")
    dependencies = value["dependency_directories"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 8:
        raise ValueError("finite explicitly registered dependency directories required")
    paths, identities = set(), set()
    for dependency in dependencies:
        fields(dependency, ("path", "device", "inode"), "registered dependency directory")
        path = str(_safe_absolute(dependency["path"]))
        uint(dependency["device"], 2**64 - 1)
        if not uint(dependency["inode"], 2**64 - 1):
            raise ValueError("registered dependency directory requires actual inode")
        identity = (dependency["device"], dependency["inode"])
        if path in paths or identity in identities:
            raise ValueError("aliased/duplicate registered dependency directories")
        paths.add(path)
        identities.add(identity)
    runtime = value["python_runtime"]
    fields(runtime, ("version", "executable"), "registered original Python runtime")
    if type(runtime["version"]) is not str or re.fullmatch(r"3\.[0-9]{1,2}\.[0-9]{1,3}", runtime["version"]) is None:
        raise ValueError("registered Python release required")
    _artifact(runtime["executable"], 256 * 1024**2)
    for number in (value["original_started_ns"], value["execution_until_ns"], value["whole_until_ns"], execution_until_ns, whole_until_ns, now_ns):
        uint(number, 2**64 - 1)
    if (value["execution_until_ns"] != execution_until_ns or value["whole_until_ns"] != whole_until_ns
            or not value["original_started_ns"] <= now_ns < execution_until_ns <= whole_until_ns
            or execution_until_ns - value["original_started_ns"] > registration["max_wall_ns"]
            or whole_until_ns - execution_until_ns > 30 * 10**9):
        raise ValueError("original trainer deadline mismatch/renewal/expiration")
    return registration


def _regular_readback(path, *, maximum, until_ns, clock=time.monotonic_ns):
    """Bounded actual byte/identity readback, never serialized execution authority."""
    path = _safe_absolute(str(path))
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("registered runtime/source artifact has a link ancestor")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("registered runtime/source file exceeds byte boundary")
        hasher, total = hashlib.sha256(), 0
        while total < before.st_size:
            if clock() >= until_ns:
                raise ValueError("original runtime/source readback deadline expired")
            block = os.read(fd, min(65536, before.st_size - total))
            if not block:
                break
            total += len(block)
            hasher.update(block)
        after, current = os.fstat(fd), path.stat(follow_symlinks=False)
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if total != before.st_size or os.read(fd, 1) or identity(before) != identity(after) or identity(after) != identity(current):
            raise ValueError("actual runtime/source readback changed during pinning")
        return {"path": str(path), "sha256": hasher.hexdigest(), "bytes": total,
            "device": after.st_dev, "inode": after.st_ino}
    finally:
        os.close(fd)


def _install_original_dependencies(startup, *, until_ns, clock=time.monotonic_ns):
    """No-site isolated bootstrap: add only registered paths; do not execute .pth."""
    import sys
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("original trainer requires isolated -I -S -B bootstrap")
    executable = _regular_readback(startup["python_runtime"]["executable"]["path"],
        maximum=256 * 1024**2, until_ns=until_ns, clock=clock)
    expected = startup["python_runtime"]["executable"]
    loaded = os.stat("/proc/self/exe")
    if (sys.version.split()[0] != startup["python_runtime"]["version"]
            or any(executable[name] != expected[name] for name in ("path", "sha256", "bytes"))
            or (loaded.st_dev, loaded.st_ino) != (executable["device"], executable["inode"])):
        raise ValueError("actual mapped Python executable differs from original runtime pin")
    readbacks = []
    for dependency in startup["dependency_directories"]:
        path = _safe_absolute(dependency["path"])
        for parent in (path, *path.parents):
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("registered dependency directory has a link ancestor")
        info = path.stat(follow_symlinks=False)
        if (not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (dependency["device"], dependency["inode"])
                or clock() >= until_ns):
            raise ValueError("registered dependency directory identity/window differs")
        # sys.path changes only. site.addsitedir would execute .pth files.
        if str(path) in sys.path:
            raise ValueError("dependency already inherited before explicit registration")
        sys.path.append(str(path))
        readbacks.append(dict(dependency))
    return {"version": sys.version, "executable": executable, "sys_executable": sys.executable,
        "mapped_executable_identity": {"device": loaded.st_dev, "inode": loaded.st_ino},
        "os": os.name, "platform": sys.platform, "cwd": os.getcwd(),
        "isolated": True, "no_site": True, "bytecode_disabled": True,
        "dependency_directories": readbacks, "dependency_pth_executed": False}


def _active_source_readback(*, until_ns, clock=time.monotonic_ns):
    import sys
    from .v5_source_registry import source_pins
    root = Path(__file__).absolute().parents[5]
    modules = []
    selected = sorted((name, module) for name, module in sys.modules.items()
        if name == "rz_pals_model" or name.startswith("rz_pals_model."))
    main = sys.modules.get("__main__")
    if main is not None:
        path = Path(getattr(main, "__file__", "")).absolute()
        package = root / "experiments/model-research/pals/src/rz_pals_model"
        if path in (package / "generation_training_cli.py", package / "generation_export_cli.py"):
            selected.append(("__main__", main))
            selected.sort(key=lambda item: item[0])
    if not 1 <= len(selected) <= 128:
        raise ValueError("active original trainer source module cap exceeded")
    for name, module in selected:
        path = _safe_absolute(getattr(module, "__file__", ""))
        if not path.is_relative_to(root / "experiments/model-research/pals/src/rz_pals_model") or path.suffix != ".py":
            raise ValueError("active trainer source module escaped original private source boundary")
        modules.append({"module": name, **_regular_readback(path, maximum=4 * 1024**2, until_ns=until_ns, clock=clock)})
    return {"source_components": source_pins(SOURCE_COMPONENTS), "source_modules": modules}


def _terminal_receipt(registration, result):
    from .generation_export import _atomic_json
    path = _safe_absolute(registration["receipt_path"])
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_JSON_BYTES or _staging_bytes(registration) + len(raw) > registration["max_output_bytes"]:
        raise ValueError("original terminal receipt exceeds reserved staging credit")
    _atomic_json(path, result)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def original_trainer_child_main(argv=None, *, clock=time.monotonic_ns, stdin=None):
    """Concrete Linux child protocol for the root-owned Rust process launch.

    stdin is one bounded startup JSON followed by EOF. The original parent
    pins these exact bytes/compiled sources, holds the original dataset owner,
    supplies CLOCK_MONOTONIC deadlines and supervises process-group closure.
    The emitted receipt distinguishes native-chain byte joins from actual
    optimizer work and cannot claim a native archive capability.
    """
    import argparse
    import signal
    import sys
    parser = argparse.ArgumentParser(prog="train-generation-original-child")
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--original-trainer-child", action="store_true", required=True)
    parser.add_argument("--startup-sha256", required=True)
    parser.add_argument("--startup-bytes", type=int, required=True)
    parser.add_argument("--execution-until-ns", type=int, required=True)
    parser.add_argument("--whole-until-ns", type=int, required=True)
    args = parser.parse_args(argv)
    if os.name != "posix":
        raise ValueError("original trainer child requires Linux monotonic/process transport")
    if not uint(args.startup_bytes, MAX_JSON_BYTES):
        raise ValueError("finite original startup byte credit required")
    source = sys.stdin.buffer if stdin is None else stdin
    raw = source.read(args.startup_bytes + 1)
    if type(raw) is not bytes or len(raw) != args.startup_bytes:
        raise ValueError("original startup bytes/EOF exceed exact parent credit")
    startup = strict_json(raw.decode("utf-8", "strict"))
    registration = validate_startup(startup, expected_raw_sha256=args.startup_sha256, raw_bytes=raw,
        execution_until_ns=args.execution_until_ns, whole_until_ns=args.whole_until_ns, now_ns=clock())
    interpreter = _install_original_dependencies(startup, until_ns=args.execution_until_ns, clock=clock)
    prepared = prepare_registration(registration, expected_registration_sha256=startup["registration_sha256"],
        until_ns=args.execution_until_ns, clock=clock)
    cancelled = [False]
    previous = {}
    def receive_cancel(_signal, _frame):
        cancelled[0] = True
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, receive_cancel)
    producer = PinnedOriginalTrainerChild()
    result = None
    exit_code = 0
    try:
        try:
            result = execute_registered_window(registration, prepared, producer, until_ns=args.execution_until_ns,
                clock=clock, cancelled=lambda: cancelled[0])
            result["status"] = result["window"]["status"]
        except BaseException as error:
            state = error.state if isinstance(error, GenerationTrainingWindowFailed) else producer.last_state
            ledger = None if state is None or state.ledger is None else state.ledger.state()
            sampler = None if state is None else ({role: owner.state() for role, owner in state.owned_samplers.items()}
                if state.owned_samplers is not None else state.sampler_states)
            result = {"schema": RESULT_DOMAIN, "status": "failed", "error_type": type(error).__name__,
                "cause_error_type": type(error.error).__name__ if isinstance(error, GenerationTrainingWindowFailed) else type(error).__name__,
                "error": str(error)[:4096], "window_id": registration["window_id"],
                "registration_sha256": prepared["registration_sha256"], "ledger": ledger,
                "sampler": sampler,
                "retained_pc_sampler": None if state is None else state.retained_pc_sampler_states,
                "actual_training_executed": ledger is not None and sum(ledger["dispatched"].values()) > state.window_initial_dispatched,
                "optimizer_completion": "unknown_after_dispatch" if ledger is not None and ledger["pending_microbatches"] == 8 else "charged_partial_or_completed_boundaries",
                "checkpoint_resumable": False, "automatic_retry": False, "arena_eligible": False}
            exit_code = 1
        state = producer.last_state
        checkpoint_pins = []
        # Capture even a durable save followed by a failing output/fence check.
        # Such a file is retained as evidence, never automatically resumed.
        saved = [] if state is None else state.saved_checkpoints or []
        try:
            for checkpoint in saved:
                checkpoint_pins.append(_regular_readback(checkpoint, maximum=1024**3,
                    until_ns=args.whole_until_ns, clock=clock))
        except BaseException as error:
            result["checkpoint_readback_error"] = {"error_type": type(error).__name__, "error": str(error)[:4096]}
            result["unverified_checkpoint_paths"] = list(saved)
            result["status"], exit_code = "checkpoint_readback_failed", 1
        active_sources = None
        try:
            # Ensure the terminal writer's source is included in the readback.
            from .generation_export import _atomic_json
            active_sources = _active_source_readback(until_ns=args.whole_until_ns, clock=clock)
            if active_sources["source_components"] != registration["source_components"]:
                raise ValueError("active terminal source component pins differ from original registration")
        except BaseException as error:
            result["source_readback_error"] = {"error_type": type(error).__name__, "error": str(error)[:4096]}
            result["status"], exit_code = "source_readback_failed", 1
        result.update({"execution_id": startup["execution_id"], "dataset_origin_scope": startup["dataset_pin_scope"],
            "checkpoint_pins": checkpoint_pins, "original_resume_checkpoint": registration["checkpoint"],
            "active_source": active_sources, "interpreter_readback": interpreter,
            "runtime_readback": None if state is None else state.runtime_readback,
            "native_game_authority_issued": False, "original_process_authority_issued": False,
            "physical_completion": "not_started" if state is None else state.physical_completion,
            "original_execution_until_ns": args.execution_until_ns, "original_whole_until_ns": args.whole_until_ns,
            "parent_must_verify_process_and_artifacts": True})
        if result["physical_completion"] == "unknown_quarantined":
            result["status"], exit_code = "physical_completion_unknown", 1
        receipt_pin = _terminal_receipt(registration, result)
        print(json.dumps({"schema": RESULT_DOMAIN, "status": result["status"], "receipt": receipt_pin,
            "actual_training_executed": result.get("actual_training_executed", result.get("window", {}).get("actual_training_executed", False)),
            "physical_completion": result["physical_completion"], "arena_eligible": False}, sort_keys=True, allow_nan=False))
        return exit_code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
