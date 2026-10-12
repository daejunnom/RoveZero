"""Separate, fail-closed DG02 admission for captured native C divergences.

An auxiliary learning_input=false journal is never a current ordinary row.
The current strict parent anchors the same root Rules identity only: auxiliary
role, revision, sequence and selected public records can differ. All features
come from the auxiliary's exact captured tensor bytes, never parent features.
Rules replay belongs to Rust. Independent source/receipt/launch pins provide
conditional caller assurance, not self-reported execution certification.
Legacy prefix descriptors require checked Rust semantic receipts and are
explicitly derived_from_legacy_prepared, never retroactive before-dispatch.
All target masks stay false. This unit grants no ranking, repair validity,
divergence supervision, policy, WDL, training or product inference admission.
"""
import copy
from dataclasses import dataclass
import hashlib
import math
import struct

import torch

from . import training
from .config import TASKS
from .model import TensorInput
from .semantic_verifier import CheckedSemanticInput, byte_pin, canonical, digest, _fields, _int, _parse, _sha

CONTEXT_VERSION = "rz-pals-native-divergence-context/1"
# rz-eval::pals_model::PALS_ENCODING_SCHEMA, used by both the native model
# identity and rz-arena::pals_collect::native_encoding_sha. This namespace is
# distinct from the raw Rules field-semantic digest declared by the export.
PALS_ENCODING_SCHEMA = b"rovezero.pals-board-records.v1"
CHALLENGED_DOMAIN = "rz-pals-challenged-line/1"
ADMISSION_DOMAIN = "rz-pals-native-divergence-admission/1"
LAUNCH_SCHEMA = "rz-pals-native-divergence-collection-launch/1"
SCOPE = "auxiliary_false_learning_input;same_root_current_parent_anchor_only;all_targets_masked;no_ranking_or_repair_validity"
MAX_BYTES = 128 * 1024 * 1024
_FACTORY = object()
_BASE_NAMES = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
               "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl", "native-divergence-inputs.jsonl",
               "native-divergence-sidecars.jsonl", "public-record-sources.jsonl", "native-events.jsonl", "native-raw-outputs.jsonl")
_COMMON_NAMES = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
                 "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")
_COMMON_PINS = ("artifacts", "registration", "checked_source", "export_manifest", "launch", "parent_input_sha256",
                "current_view_sha256", "frozen_admission_sha256")
_SAME_ROOT = ("game_id", "opening_id", "line_genealogy_id", "position_command", "board_fen", "actual_history",
              "rules_state_sha256", "rules_history_sha256", "transposition_sha256", "encoding_sha256", "source", "frozen_epoch", "white_to_move", "legal_moves")


def _rows(raw):
    values = training._collection_jsonl(raw, max_rows=65536 * 8)
    if any(type(value) is not dict for _, value in values):
        raise ValueError("actual auxiliary object row required")
    return values


def _one(rows, predicate, name):
    if any(type(value) is not dict for _, value in rows):
        raise ValueError("actual auxiliary object row required")
    matches = [(raw, value) for raw, value in rows if predicate(value)]
    if len(matches) != 1:
        raise ValueError("missing/ambiguous actual auxiliary " + name)
    return matches[0]


def _pinned_public_row(rows, expected_sha256):
    """One exact observation, including repeated lexical rows per capture.

    Retained collection artifacts can repeat identical selected source bytes.
    Collapse only byte-identical selected rows; never choose by record ID/revision or weaken request/journal
    uniqueness. The whole artifact and the observation hash stay independently
    pinned, and differently serialized/value-bearing rows cannot substitute.
    """
    _sha(expected_sha256)
    unique = {raw: value for raw, value in rows if byte_pin(raw)["sha256"] == expected_sha256}
    return _one(list(unique.items()), lambda _: True, "public record source")


def _pin_matches(raw, expected):
    _fields(expected, ("bytes", "sha256"))
    _int(expected["bytes"], 0, MAX_BYTES)
    _sha(expected["sha256"])
    return byte_pin(raw) == expected


def _finite(values, width):
    if type(values) is not list or len(values) != width:
        raise ValueError("captured feature shape")
    rounded = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError("captured finite FP32 feature required")
        try:
            actual = struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError("captured finite FP32 feature required") from error
        if not math.isfinite(actual):
            raise ValueError("captured finite FP32 feature required")
        rounded.append(actual)
    # Rust's shortest decimal f32 text can round from a slightly larger Python
    # binary64 (including max-f32 3.4028235e38). Preserve its actual f32 bits;
    # overflow/nonfinite fails, and the exact lexical bytes stay independently
    # pinned in the immutable capability.
    return tuple(rounded)


def _sidecar(value, frozen, source):
    names = ("version", "input_sha256", "encoding_sha256", "encoder_source_sha256", "model_epoch_kind",
             "canonical_tensor_sha256", "tensor_json", "tensor_sha256", "record_sources", "sha256")
    _fields(value, names)
    snapshot = frozen["snapshot"]
    if (value["version"] != "rz-pals-native-input-sidecar/1" or value["input_sha256"] != frozen["sha256"]
            or value["encoding_sha256"] != snapshot["encoding_sha256"]
            or value["encoder_source_sha256"] != source["encoder_source_sha256"]
            or value["model_epoch_kind"] != "frozen_model_epoch" or canonical(value["record_sources"]) != canonical(snapshot["public_records"])):
        raise ValueError("actual auxiliary sidecar/producer identity")
    for name in ("canonical_tensor_sha256", "tensor_sha256", "sha256"):
        _sha(value[name])
    if type(value["tensor_json"]) is not str:
        raise ValueError("actual lexical tensor JSON required")
    tensor_raw = value["tensor_json"].encode("utf-8")
    if byte_pin(tensor_raw)["sha256"] != value["tensor_sha256"] or not 1 <= len(tensor_raw) <= 2 * 1024 * 1024:
        raise ValueError("actual captured tensor byte identity")
    if hashlib.sha256(canonical([value[name] for name in names[:-1]])).hexdigest() != value["sha256"]:
        raise ValueError("auxiliary sidecar seal")
    tensor = _parse(tensor_raw)
    _fields(tensor, ("role", "board", "metadata", "records", "required_critical_records", "candidates",
                     "divergence_features", "query", "situation_revision", "history_digest", "model_epoch"))
    if type(tensor["board"]) is not list or len(tensor["board"]) != 64:
        raise ValueError("exact captured board64 required")
    if (tensor["role"] != "critic" or _int(tensor["situation_revision"], 0, 2**64 - 1) != snapshot["input_revision"]
            or tensor["candidates"] != [] or tuple(tensor["board"]) != training._fen_board(snapshot["board_fen"])):
        raise ValueError("DG02 requires captured critic empty candidates/actual board/revision")
    for key in ("history_digest", "model_epoch"):
        if type(tensor[key]) is not list or len(tensor[key]) != 32:
            raise ValueError("actual history/model epoch shape")
        for value_byte in tensor[key]:
            _int(value_byte, 0, 255)
    if (bytes(tensor["history_digest"]).hex() != snapshot["rules_history_sha256"]
            or bytes(tensor["model_epoch"]).hex() != snapshot["source"]["model_weights_sha256"]):
        raise ValueError("actual auxiliary known history/frozen model epoch")
    if type(tensor["records"]) is not list or len(tensor["records"]) != len(snapshot["public_records"]):
        raise ValueError("actual selected public record extent")
    public, critical, ids = [], set(), set()
    for token, record in zip(tensor["records"], snapshot["public_records"]):
        _fields(token, ("record_id", "revision", "critical", "features"))
        identity = _int(token["record_id"], 1, 2**64 - 1)
        if identity in ids or type(token["critical"]) is not bool or _int(token["revision"], 0, 2**64 - 1) != record["situation_revision"]:
            raise ValueError("actual public record duplicate/revision")
        ids.add(identity)
        if token["critical"]:
            critical.add(identity)
        public.append((record["observation_sha256"], record["situation_revision"], _finite(token["features"], 16)))
    required = tensor["required_critical_records"]
    if type(required) is not list or len(required) > 128:
        raise ValueError("required critical record extent")
    for identity in required:
        _int(identity, 1, 2**64 - 1)
    if len(set(required)) != len(required) or not set(required) <= critical:
        raise ValueError("missing actual critical record")
    features = tensor["divergence_features"]
    if type(features) is not list or not 1 <= len(features) <= 128:
        raise ValueError("explicit captured divergence features required")
    features = tuple(_finite(row, 8) for row in features)
    board = tuple(_int(piece, 0, 12) for piece in tensor["board"])
    return tensor, board, _finite(tensor["metadata"], 16), tuple(public), _finite(tensor["query"], 16), features


def _context(raw, frozen, sidecar, lineage):
    value = _parse(raw)
    _fields(value, ("version", "input_sha256", "native_request", "tensor_sidecar_sha256", "captured_input_revision",
                    "proposal_move16", "challenged_line_sha256", "divergence_sites", "sha256"))
    snapshot = frozen["snapshot"]
    _request(value["native_request"])
    for name in ("input_sha256", "tensor_sidecar_sha256", "challenged_line_sha256", "sha256"):
        _sha(value[name])
    if (value["version"] != CONTEXT_VERSION or value["input_sha256"] != frozen["sha256"]
            or value["native_request"] != [lineage["process_epoch"], lineage["request_sequence"]]
            or value["tensor_sidecar_sha256"] != sidecar["sha256"]
            or type(value["captured_input_revision"]) is not int or value["captured_input_revision"] != snapshot["input_revision"]
            or canonical(value["proposal_move16"]) != canonical(lineage["proposal"])):
        raise ValueError("actual descriptor/input/request/sidecar/revision binding")
    # Exact Rust three-element tuple; no alternate/nested seal is admitted.
    wanted = hashlib.sha256(canonical([CHALLENGED_DOMAIN, snapshot["rules_state_sha256"], lineage["proposal"]])).hexdigest()
    if value["challenged_line_sha256"] != wanted:
        raise ValueError("actual challenged line identity")
    wanted = digest(CONTEXT_VERSION, {key: item for key, item in value.items() if key != "sha256"})
    if value["sha256"] != wanted:
        raise ValueError("actual descriptor seal")
    sites = value["divergence_sites"]
    if type(sites) is not list or len(sites) != len(lineage["divergence_plies"]) or not 1 <= len(sites) <= 128:
        raise ValueError("actual descriptor site extent")
    for slot, site in enumerate(sites):
        _fields(site, ("slot", "divergence_ply", "prefix_rules_state_sha256", "prefix_rules_history_sha256"))
        if _int(site["slot"], 0, 127) != slot or _int(site["divergence_ply"], 0, len(lineage["proposal"]) - 1) != lineage["divergence_plies"][slot]:
            raise ValueError("explicit descriptor slot/ply order")
        _sha(site["prefix_rules_state_sha256"])
        _sha(site["prefix_rules_history_sha256"])
    if len({site["divergence_ply"] for site in sites}) != len(sites):
        raise ValueError("duplicate divergence site")
    return value


def _request(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError("exact native epoch/sequence pair required")
    for component in value:
        _int(component, 0, 2**64 - 1)


def _proposal_sites(lineage):
    if type(lineage) is not dict:
        raise ValueError("explicit auxiliary lineage required")
    for name in ("process_epoch", "request_sequence"):
        _int(lineage.get(name), 0, 2**64 - 1)
    proposal, plies = lineage.get("proposal"), lineage.get("divergence_plies")
    if type(proposal) is not list or not 1 <= len(proposal) <= 64:
        raise ValueError("bounded actual challenged proposal required")
    for movement in proposal:
        training.move_components(movement)
    if type(plies) is not list or not 1 <= len(plies) <= 128:
        raise ValueError("explicit divergence site order required")
    for ply in plies:
        _int(ply, 0, len(proposal) - 1)
    if len(set(plies)) != len(plies):
        raise ValueError("duplicate divergence site")


def _rules_prefixes(parents, parent_index, frozen, lineage, checks):
    _proposal_sites(lineage)
    if type(checks) not in (tuple, list) or len(checks) != len(lineage["divergence_plies"]):
        raise ValueError("each legacy divergence site requires a checked Rust Rules receipt")
    sites = []
    for slot, (ply, checked) in enumerate(zip(lineage["divergence_plies"], checks)):
        if type(checked) is not CheckedSemanticInput or checked.verify_parent(parents) != parent_index:
            raise ValueError("legacy prefix Rules proof needs actual same-root current parent anchor")
        receipt, question = checked.rules_receipt(), checked.common_query()
        if (any(receipt["root"][key] != frozen["snapshot"][key] for key in ("rules_state_sha256", "rules_history_sha256", "board_fen"))
                or question["prefix"] != lineage["proposal"][:ply] or question["root_moves"] or question["claimed_line"]
                or receipt["target"]["side_to_move"] == receipt["root"]["side_to_move"]):
            raise ValueError("legacy prefix Rust receipt does not cover captured proposal/site")
        sites.append({"slot": slot, "divergence_ply": ply, "prefix_rules_state_sha256": receipt["target"]["rules_state_sha256"],
                      "prefix_rules_history_sha256": receipt["target"]["rules_history_sha256"]})
    return sites


def _critic_route(manifest_raw, source):
    """Explicit registered graph semantics, not a guessed Critic basename."""
    native = source["native"]
    manifest = _parse(manifest_raw)
    manifest_sha = byte_pin(manifest_raw)["sha256"]
    if (type(manifest) is not dict or manifest_sha != native["independent_registry"]["export_manifest_sha256"]
            or manifest_sha != bytes(native["loaded_source"]["export_manifest_sha256"]).hex()
            or canonical(manifest.get("config")) != canonical(source["configuration"])
            or manifest.get("checkpoint_sha256") != source["source"]["model_weights_sha256"]
            or manifest.get("trained") is not False or type(manifest.get("training_steps")) is not int or manifest["training_steps"] != 0
            or manifest.get("validator_present") is not False or manifest.get("roles") != ["proposer", "critic"]):
        raise ValueError("registered export manifest/critic graph capability mismatch")
    rules_semantic = _sha(manifest.get("rules_input_semantic_sha256"))
    # These are distinct, independently pinned provenance axes. The declared
    # export-era source records where its feature declaration came from; an
    # actual collector source may differ after worker/lifecycle/comment edits.
    # Their equality would incorrectly make code provenance a feature meaning.
    _sha(manifest.get("rules_encoder_source_sha256"))
    actual_source = _sha(source["encoder_source_sha256"])
    loaded = native["loaded_source"]
    if (training._epoch_hex(loaded.get("adapter_source_sha256")) != actual_source
            or training._epoch_hex(loaded.get("encoding_semantic_sha256")) != source["encoding_sha256"]):
        raise ValueError("actual loaded adapter/encoding differs from registered collector source")
    composite = hashlib.sha256(PALS_ENCODING_SCHEMA + bytes.fromhex(rules_semantic)).hexdigest()
    if manifest.get("rules_input_profile") != "rz-pals-rules-fields-v1" or composite != source["encoding_sha256"]:
        raise ValueError("registered export Rules semantics do not match actual encoding namespace")
    graphs = manifest.get("graphs")
    if type(graphs) is not list or not 1 <= len(graphs) <= 4 or any(type(graph) is not dict for graph in graphs):
        raise ValueError("explicit registered export graph inventory required")
    graph_pins = [(graph.get("role"), _sha(graph.get("sha256"))) for graph in graphs]
    if (any(type(role) is not str for role, _ in graph_pins) or len({role for role, _ in graph_pins}) != len(graph_pins)
            or sorted(graph_pins) != sorted((graph["role"], graph["sha256"]) for graph in native["graphs"])):
        raise ValueError("registered export manifest differs from actual loaded graph inventory")
    roles = {role for role, _ in graph_pins}
    if (manifest.get("schema") == "rovezero.pals-model.v1" and manifest.get("layout") in (None, "separate_pc")
            and manifest.get("model_semantics") in (None, "rovezero.pals-model.v1")
            and manifest.get("reader_initializer_bank") is None and roles == {"public", "proposer", "critic"}):
        return "separate_pc"
    if (manifest.get("schema") != "rovezero.pals-model.v2" or manifest.get("layout") != "shared_pc_if"
            or manifest.get("model_semantics") != "rovezero.pals-model.v1" or type(manifest.get("layout_revision")) is not int
            or manifest["layout_revision"] != 1 or manifest.get("role_batching") != "one_scalar_role_per_physical_batch"
            or roles != {"public", "shared_pc"}):
        raise ValueError("registered export has no supported explicit Critic graph route")
    shared = next(graph for graph in graphs if graph["role"] == "shared_pc")
    for name in ("inputs", "outputs"):
        if type(shared.get(name)) is not list or any(type(tensor) is not dict for tensor in shared[name]):
            raise ValueError("registered shared Critic route tensor signature required")
    # The loader already hash-verified and validated the real graph. This
    # separate manifest admission checks the supported explicit hard route;
    # it does not execute ONNX or promote a filename into an execution proof.
    if (shared["inputs"].count({"name": "role_is_critic", "dtype": "BOOL", "shape": []}) != 1
            or shared["inputs"].count({"name": "divergence_features", "dtype": "FLOAT", "shape": ["batch", "divergences", 8]}) != 1
            or shared["outputs"].count({"name": "divergence_logits", "dtype": "FLOAT", "shape": ["batch", "divergences"]}) != 1):
        raise ValueError("registered shared Critic hard-route signature mismatch")
    return "shared_pc_if"


def derive_legacy_context(*, parents, parent_index, auxiliary_input_bytes, sidecar_bytes, lineage_bytes, prefix_rules_checks):
    """Return an UNADMITTED descriptor proposal, not auxiliary input authority.

    admit_native_divergence must independently verify the original false journal,
    source, collection assets and launch. Rules checks prove prefixes only.
    No past-before-dispatch claim is made by this helper.
    """
    frozen, sidecar, lineage = (_parse(raw) for raw in (auxiliary_input_bytes, sidecar_bytes, lineage_bytes))
    _fields(frozen, ("snapshot", "sha256"))
    _sha(frozen["sha256"])
    _sha(frozen["snapshot"]["rules_state_sha256"])
    _sha(sidecar["sha256"])
    _int(frozen["snapshot"]["input_revision"], 0, 2**64 - 1)
    sites = _rules_prefixes(parents, parent_index, frozen, lineage, prefix_rules_checks)
    body = {"version": CONTEXT_VERSION, "input_sha256": frozen["sha256"],
            "native_request": [lineage["process_epoch"], lineage["request_sequence"]], "tensor_sidecar_sha256": sidecar["sha256"],
            "captured_input_revision": frozen["snapshot"]["input_revision"], "proposal_move16": lineage["proposal"],
            "challenged_line_sha256": hashlib.sha256(canonical([CHALLENGED_DOMAIN, frozen["snapshot"]["rules_state_sha256"], lineage["proposal"]])).hexdigest(),
            "divergence_sites": sites}
    return canonical({**body, "sha256": digest(CONTEXT_VERSION, body)})


@dataclass(frozen=True)
class NativeCollectionFacts:
    """Read-only facts, not individual input admission or a reusable bypass.

    Constructors confer no authority: consumers must call the verifier for
    their actual parent/bytes/pins. No admission factory accepts these facts.
    """
    parent_input_sha256: str
    producer_json: bytes
    source_json: bytes
    registered_critic_graph_route: str
    collection_receipt_sha256: str
    rules_input_semantic_sha256: str
    encoding_sha256: str
    export_declared_encoder_source_sha256: str
    actual_encoder_source_sha256: str
    scope: str = "conditional_registered_collection_facts_only;no_individual_input_or_label_admission"


def verify_native_collection_authority(*, parent, parent_index, artifacts, registration_bytes,
                                       checked_source_bytes, export_manifest_bytes, launch_bytes, expected_pins):
    """Check common raw collection/registration/export/caller launch facts.

    Uses the strict ordinary current parent without duplicating its selector.
    Individually admitted ordinary/aux rows, journals, Rules prefixes, raw
    events, feature semantics and target masks remain the consumer's duty.
    Additional assets are checked against the immutable original receipt, not
    retroactively included in the ordinary dataset's admission.
    """
    parents = parent
    if type(parents) is not training.ValidatedDataset or parents.frozen_admission is None:
        raise ValueError("strict frozen current parent required")
    parents._verify_raw_integrity()
    _int(parent_index, 0, len(parents.records) - 1)
    if parent_index not in parents.current_view.current_indices:
        raise ValueError("historical parent cannot anchor auxiliary admission")
    parent = parents.records[parent_index]
    snapshot = parent["input"]["snapshot"]
    if snapshot["role"] not in ("proposer", "critic") or snapshot["source"]["kind"] != "own_pals":
        raise ValueError("actual native current P/C parent required")
    if (type(artifacts) is not dict or not set(_COMMON_NAMES) <= set(artifacts)
            or not set(artifacts) <= {*_BASE_NAMES, "native-divergence-contexts.jsonl"}):
        raise ValueError("explicit actual native collection artifact set required")
    names = tuple(artifacts)
    pins = _fields(expected_pins, _COMMON_PINS)
    _fields(pins["artifacts"], names)
    registration_raw, source_raw, export_raw, launch_raw = registration_bytes, checked_source_bytes, export_manifest_bytes, launch_bytes
    raws = (*artifacts.values(), registration_raw, source_raw, export_raw, launch_raw)
    if any(type(raw) is not bytes for raw in raws) or sum(map(len, raws)) > MAX_BYTES:
        raise ValueError("aggregate immutable auxiliary byte budget")
    if (pins["parent_input_sha256"] != parent["input"]["sha256"] or pins["current_view_sha256"] != parents.current_view.sha256
            or pins["frozen_admission_sha256"] != parents._frozen_admission_identity):
        raise ValueError("actual strict parent/current admission pins")
    for name in names:
        # The ordinary loader deliberately does not admit/read auxiliary D
        # assets. Its immutable raw receipt still declares their exact pins;
        # this separate adapter reads and checks those bytes independently.
        expected = parents.frozen_admission["receipt"] if name == "receipt.json" else parents.collection_receipt["artifacts"].get(name)
        if expected is None or not _pin_matches(artifacts[name], expected) or not _pin_matches(artifacts[name], pins["artifacts"][name]):
            raise ValueError("actual collection artifact differs from independent receipt pin: " + name)
        ordinary_checked = parents.frozen_admission["artifacts"].get(name)
        if ordinary_checked is not None and not _pin_matches(artifacts[name], ordinary_checked):
            raise ValueError("actual collection bytes differ from checked ordinary asset")
    for name, raw in (("registration", registration_raw), ("checked_source", source_raw), ("export_manifest", export_raw), ("launch", launch_raw)):
        if not _pin_matches(raw, pins[name]):
            raise ValueError("actual auxiliary authority byte pin mismatch: " + name)
    admitted = next((value for value in parents.frozen_admission["inputs"] if value["binding"]["input_sha256"] == parent["input"]["sha256"]), None)
    if admitted is None:
        raise ValueError("parent lacks actual registered capture binding")
    producer = admitted["producer"]
    registration = _parse(registration_raw)
    if byte_pin(registration_raw)["sha256"] != producer["registration_sha256"]:
        raise ValueError("auxiliary producer not independently registered for parent game")
    source = training._checked_producer_source(source_raw, registration, producer)
    checked_source_pin = next((entry["checked_source"] for entry in parents.frozen_admission["producer_registrations"] if entry["pin"] == producer), None)
    if checked_source_pin != byte_pin(source_raw):
        raise ValueError("actual auxiliary checked source bytes lack independent parent registration")
    graph_route = _critic_route(export_raw, source)
    receipt = _parse(artifacts["receipt.json"])
    finish = receipt.get("native_finish", {})
    native = finish.get("receipt", {})
    if (receipt.get("complete") is not True or receipt.get("failure") is not None
            or canonical(receipt.get("source")) != canonical(source) or finish.get("collection_accepted") is not True
            or finish.get("finish_error") is not None or "_collection_failure" not in finish or finish["_collection_failure"] is not None
            or native.get("physical_shutdown_confirmed") is not True
            or native.get("native_buffers_released") is not True or native.get("quarantined") is not False
            or type(native.get("physical_runs_in_flight")) is not int or native["physical_runs_in_flight"] != 0
            or type(native.get("observer_failures")) is not int or native["observer_failures"] != 0):
        raise ValueError("actual native receipt/physical cleanup confirmation required")
    launch = _parse(launch_raw)
    _fields(launch, ("schema", "receipt_artifact", "checked_source_artifact", "collector_binary_sha256", "assurance_scope",
                    "spawned", "reaped", "exit_code", "timed_out"))
    if (launch["schema"] != LAUNCH_SCHEMA or launch["receipt_artifact"] != byte_pin(artifacts["receipt.json"])
            or launch["checked_source_artifact"] != byte_pin(source_raw) or launch["collector_binary_sha256"] != source["implementation_sha256"]
            or launch["assurance_scope"] != "independently_pinned_caller_collection_observation"
            or launch["spawned"] is not True or launch["reaped"] is not True or type(launch["exit_code"]) is not int
            or launch["exit_code"] != 0 or launch["timed_out"] is not False):
        raise ValueError("independent registered native collection launch observation required")
    manifest = _parse(export_raw)
    return NativeCollectionFacts(parent["input"]["sha256"], canonical(producer), canonical(source), graph_route,
                                 byte_pin(artifacts["receipt.json"])["sha256"], manifest["rules_input_semantic_sha256"],
                                 source["encoding_sha256"], manifest["rules_encoder_source_sha256"], source["encoder_source_sha256"])


def _validate(parents, parent_index, artifacts, registration_raw, source_raw, export_raw, launch_raw, context_raw, mode, checks, pins):
    if mode not in ("captured_before_dispatch", "derived_from_legacy_prepared"):
        raise ValueError("explicit native/legacy provenance mode required")
    names = (*_BASE_NAMES, *(("native-divergence-contexts.jsonl",) if mode == "captured_before_dispatch" else ()))
    _fields(artifacts, names)
    _fields(pins, (*_COMMON_PINS, "context", "rules_checks", "auxiliary_input_sha256"))
    raws = (*artifacts.values(), registration_raw, source_raw, export_raw, launch_raw, context_raw)
    if any(type(raw) is not bytes for raw in raws) or sum(map(len, raws)) > MAX_BYTES:
        raise ValueError("aggregate immutable auxiliary byte budget")
    authority = verify_native_collection_authority(parent=parents, parent_index=parent_index, artifacts=artifacts,
        registration_bytes=registration_raw, checked_source_bytes=source_raw, export_manifest_bytes=export_raw,
        launch_bytes=launch_raw, expected_pins={name: pins[name] for name in _COMMON_PINS})
    parent = parents.records[parent_index]
    snapshot = parent["input"]["snapshot"]
    producer, source = _parse(authority.producer_json), _parse(authority.source_json)
    graph_route = authority.registered_critic_graph_route
    if not _pin_matches(context_raw, pins["context"]):
        raise ValueError("actual auxiliary authority byte pin mismatch: context")
    if type(checks) not in (tuple, list) or any(type(checked) is not CheckedSemanticInput for checked in checks):
        raise ValueError("exact immutable checked Rules capabilities required")
    if pins["rules_checks"] != [checked.sha256 for checked in checks]:
        raise ValueError("independent legacy Rules capability pins")
    identity = _sha(pins["auxiliary_input_sha256"])
    if any(row["input"]["sha256"] == identity for row in parents.records):
        raise ValueError("auxiliary input cannot masquerade as ordinary current/historical row")
    input_raw, frozen = _one(_rows(artifacts["native-divergence-inputs.jsonl"]), lambda value: value.get("sha256") == identity, "input")
    training._validate_record({"input": frozen, "future_label": None, "verifier_private": None}, parents.owned_sources["cpu_binary_sha256"], parents.owned_sources["input_sources"])
    aux = frozen["snapshot"]
    if aux["role"] != "critic" or any(canonical(aux[key]) != canonical(snapshot[key]) for key in _SAME_ROOT):
        raise ValueError("auxiliary input is not a same-root registered critic; revision/records are independent")
    sidecar_raw, sidecar = _one(_rows(artifacts["native-divergence-sidecars.jsonl"]), lambda value: value.get("input_sha256") == identity, "sidecar")
    lineage_raw, lineage = _one(_rows(artifacts["input-lineage.jsonl"]), lambda value: value.get("input_sha256") == identity, "lineage")
    lineage_names = ("input_sha256", "game_id", "process_epoch", "request_sequence", "native_query_kind", "actual_played_history",
                     "virtual_prefix", "proposal", "counterexample", "divergence_plies", "actual_outcome_eligible", "counterfactual_wdl", "training_admission")
    _fields(lineage, (*lineage_names, *(("native_divergence_context_sha256",) if mode == "captured_before_dispatch" else ())))
    if (lineage["game_id"] != aux["game_id"] or lineage["native_query_kind"] != "Divergence"
            or canonical(lineage["actual_played_history"]) != canonical(aux["actual_history"]) or lineage["virtual_prefix"] != []
            or lineage["counterexample"] is not None or lineage["actual_outcome_eligible"] is not False
            or lineage["counterfactual_wdl"] != "masked" or lineage["training_admission"] != "deferred_divergence_head"):
        raise ValueError("explicit auxiliary lineage scope required")
    _proposal_sites(lineage)
    journals = _rows(artifacts["producer-prepared.jsonl"])
    _, journal = _one(journals, lambda value: type(value.get("prepared")) is dict and value["prepared"].get("input_sha256") == identity,
                      "false-learning journal")
    _fields(journal, ("prepared", "sha256"))
    body = journal["prepared"]
    _fields(body, ("version", "producer_id", "registration_sha256", "roster_sha256", "checked_source_sha256", "game_id",
                   "input_sha256", "capture_sequence", "input_json", "tensor_sidecar_json", "lineage_json", "native_request",
                   "learning_input", "publication"))
    _int(body["capture_sequence"], 0, 2**64 - 1)
    _request(body["native_request"])
    if (journal["sha256"] != training._sorted_canonical(training.PREPARED_PRODUCER_DOMAIN, body)
            or body.get("version") != training.PREPARED_PRODUCER_DOMAIN or body.get("learning_input") is not False
            or body.get("producer_id") != producer["producer_id"] or body.get("registration_sha256") != producer["registration_sha256"]
            or body.get("checked_source_sha256") != byte_pin(source_raw)["sha256"] or body.get("game_id") != aux["game_id"]
            or body.get("capture_sequence") != aux["capture_sequence"] or body.get("native_request") != [lineage["process_epoch"], lineage["request_sequence"]]
            or body.get("publication") != "seal-before-submit; prepaid-drain-after-search"
            or body.get("roster_sha256") != _parse(artifacts["producer-roster.json"])["sha256"]
            or any(body.get(key) != byte_pin(raw) for key, raw in (("input_json", input_raw), ("tensor_sidecar_json", sidecar_raw), ("lineage_json", lineage_raw)))):
        raise ValueError("actual false journal/input/sidecar/lineage producer binding")
    decoded = _sidecar(sidecar, frozen, source)
    if len(decoded[-1]) != len(lineage["divergence_plies"]):
        raise ValueError("actual 8feature/site count mismatch")
    context = _context(context_raw, frozen, sidecar, lineage)
    if mode == "captured_before_dispatch":
        _, actual_context = _one(_rows(artifacts["native-divergence-contexts.jsonl"]), lambda value: value.get("input_sha256") == identity, "before-dispatch context")
        if canonical(actual_context) != canonical(context) or lineage["native_divergence_context_sha256"] != context["sha256"] or checks:
            raise ValueError("actual before-dispatch descriptor/lineage binding")
    else:
        if context["divergence_sites"] != _rules_prefixes(parents, parent_index, frozen, lineage, checks):
            raise ValueError("legacy descriptor lacks actual Rust prefix Rules proof")
    public_sources = _rows(artifacts["public-record-sources.jsonl"]) if artifacts["public-record-sources.jsonl"] else []
    for token, observation in zip(decoded[0]["records"], aux["public_records"]):
        _, raw_source = _pinned_public_row(public_sources, observation["observation_sha256"])
        _fields(raw_source, ("domain", "game_id", "record_index", "revision", "origin_state_id", "origin_state_id_is_advisory",
                            "origin_rules_state_sha256", "origin_rules_identity_observation", "kind", "line", "value",
                            "completed_depth", "scope", "white_score_perspective", "critical", "source_cpu_profile_sha256"))
        _int(raw_source["record_index"], 1, 2**64 - 1)
        _int(raw_source["revision"], 0, 2**64 - 1)
        if (raw_source.get("domain") != "rz-pals-native-public-source/1" or raw_source.get("game_id") != aux["game_id"]
                or raw_source.get("record_index") != token["record_id"] or raw_source.get("revision") != token["revision"]
                or raw_source.get("critical") is not token["critical"] or raw_source.get("source_cpu_profile_sha256") != source["cpu_profile_sha256"]
                or raw_source["origin_state_id_is_advisory"] is not True or raw_source["origin_rules_state_sha256"] is not None
                or raw_source["origin_rules_identity_observation"] != "unknown"):
            raise ValueError("actual auxiliary selected public raw source attribution")
    request_key = (identity, lineage["process_epoch"], lineage["request_sequence"])
    matches = lambda value: (value.get("input_sha256"), value.get("process_epoch"), value.get("request_sequence")) == request_key
    events = [value for _, value in _rows(artifacts["native-events.jsonl"]) if matches(value)]
    for event in events:
        _fields(event, ("domain", "game_id", "process_epoch", "request_sequence", "input_sha256", "stage", "observer_elapsed_us", "detail"))
        _request([event["process_epoch"], event["request_sequence"]])
    if not events or any(event.get("domain") != "rz-pals-native-call-event/1" or event.get("game_id") != aux["game_id"] for event in events):
        raise ValueError("actual native request event attribution")
    stages = [event.get("stage") for event in events]
    if (stages[:3] != ["prepared", "physically_completed", "delivered"] or len(stages) not in (3, 4)
            or (len(stages) == 4 and stages[-1] not in ("search_consumed", "logically_rejected"))
            or events[0].get("detail", {}).get("prepared_before_submit") is not True
            or events[0].get("detail", {}).get("native_query_kind") != "Divergence"
            or events[0].get("detail", {}).get("producer_metadata_admitted") is not True
            or events[1].get("detail", {}).get("success") is not True
            or events[1].get("detail", {}).get("logical_acceptance_inferred") is not False
            or events[2].get("detail", {}).get("search_consumed") is not False
            or (len(stages) == 4 and events[3].get("detail", {}).get("search_consumed") is not (stages[3] == "search_consumed"))):
        raise ValueError("successful actual prepared/physical/delivered chronology required")
    elapsed = [_int(event.get("observer_elapsed_us"), 0, 2**64 - 1) for event in events]
    if elapsed != sorted(elapsed):
        raise ValueError("native event chronology")
    _, output = _one(_rows(artifacts["native-raw-outputs.jsonl"]), matches, "raw output")
    _fields(output, ("domain", "process_epoch", "request_sequence", "input_sha256", "physical_completion_confirmed", "success", "raw"))
    _request([output["process_epoch"], output["request_sequence"]])
    if (output.get("domain") != "rz-pals-native-physical-raw/1" or output.get("physical_completion_confirmed") is not True
            or output.get("success") is not True):
        raise ValueError("known successful native physical raw output required")
    raw = output.get("raw", {})
    _fields(raw, ("representation", "candidate_logits_bits", "wdl_logits_bits", "divergence_logits_bits", "task_logits_bits",
                  "private_latent_bits", "prediction_is_future_label"))
    if raw.get("representation") != "f32_ieee754_bits" or raw.get("prediction_is_future_label") is not False or raw.get("task_logits_bits") is not None:
        raise ValueError("native predictions cannot become future labels")
    for key, width in (("candidate_logits_bits", 0), ("wdl_logits_bits", 3), ("divergence_logits_bits", len(decoded[-1])), ("private_latent_bits", 6144)):
        bits = raw.get(key)
        if type(bits) is not list or len(bits) != width:
            raise ValueError("actual native raw head shape")
        for value in bits:
            _int(value, 0, 2**32 - 1)
            if not math.isfinite(struct.unpack("<f", struct.pack("<I", value))[0]):
                raise ValueError("actual native raw head contains nonfinite FP32")
    encoding = training.EncodedSnapshot(identity, aux["encoding_sha256"], decoded[1], decoded[2], decoded[3], decoded[4],
        tuple((training.DivergenceContext(context["challenged_line_sha256"], site["divergence_ply"], aux["input_revision"]), features)
              for site, features in zip(context["divergence_sites"], decoded[-1])))
    training._validate_encoding(encoding)
    admission = {"schema": ADMISSION_DOMAIN, "scope": SCOPE, "provenance_mode": mode, "parent_input_sha256": parent["input"]["sha256"],
                 "auxiliary_input_sha256": identity, "current_view_sha256": parents.current_view.sha256, "context_sha256": context["sha256"],
                 "auxiliary_has_current_label": False, "producer_journal_learning_input": False,
                 "captured_revision": aux["input_revision"], "same_parent_public_records_required": False,
                 "caller_assurance": "independently_pinned_caller_collection_observation;registered_native_source;Rules_not_reimplemented",
                 "registered_critic_graph_route": graph_route,
                 "rules_input_semantic_sha256": authority.rules_input_semantic_sha256,
                 "encoding_sha256": authority.encoding_sha256,
                 "export_declared_encoder_source_sha256": authority.export_declared_encoder_source_sha256,
                 "actual_encoder_source_sha256": authority.actual_encoder_source_sha256,
                 "native_search_consumed": stages[-1] == "search_consumed", "all_target_masks_false": True}
    return frozen, context, encoding, admission


class CheckedNativeDivergence:
    __slots__ = ("_parents", "_index", "_artifacts", "_raws", "_mode", "_checks", "_pins", "_identity")
    def __init__(self, token=None, *, parents=None, index=None, artifacts=None, raws=None, mode=None, checks=(), pins=None):
        if token is not _FACTORY:
            raise ValueError("unchecked auxiliary constructor refused")
        for name, value in (("_parents", parents), ("_index", index), ("_artifacts", tuple(sorted(artifacts.items()))),
                            ("_raws", tuple(raws)), ("_mode", mode), ("_checks", tuple(checks)), ("_pins", canonical(pins)),
                            ("_identity", digest(ADMISSION_DOMAIN, {"mode": mode, "pins": pins}))):
            object.__setattr__(self, name, value)
    def __setattr__(self, name, value):
        raise AttributeError("auxiliary native admission is immutable")
    def _views(self):
        pins = _parse(self._pins, MAX_BYTES)
        if self._identity != digest(ADMISSION_DOMAIN, {"mode": self._mode, "pins": pins}):
            raise ValueError("auxiliary capability identity changed")
        return _validate(self._parents, self._index, dict(self._artifacts), *self._raws, self._mode, self._checks, pins)
    def verify(self):
        self._views()
        return self
    @property
    def sha256(self):
        self.verify()
        return self._identity
    @property
    def admission(self):
        return copy.deepcopy(self._views()[3])
    def encoded_snapshot(self):
        return self._views()[2]
    def context(self):
        return copy.deepcopy(self._views()[1])


def admit_native_divergence(*, parents, parent_index, artifacts, registration_bytes, checked_source_bytes, export_manifest_bytes, launch_bytes,
                            context_bytes, expected_pins, provenance_mode="captured_before_dispatch", prefix_rules_checks=()):
    pins = _parse(canonical(expected_pins), MAX_BYTES)
    raws = (registration_bytes, checked_source_bytes, export_manifest_bytes, launch_bytes, context_bytes)
    _validate(parents, parent_index, artifacts, *raws, provenance_mode, prefix_rules_checks, pins)
    return CheckedNativeDivergence(_FACTORY, parents=parents, index=parent_index, artifacts=artifacts, raws=raws,
                                   mode=provenance_mode, checks=prefix_rules_checks, pins=pins)


def collate_native_divergences(checked_inputs, *, split="train", max_tensor_bytes=1024 * 1024):
    if type(checked_inputs) not in (tuple, list) or not 1 <= len(checked_inputs) <= 4 or any(type(value) is not CheckedNativeDivergence for value in checked_inputs):
        raise ValueError("explicit checked auxiliary CPU batch 1..4 required")
    if split not in ("train", "validation", "holdout"):
        raise ValueError("explicit immutable parent game split")
    views = [value._views() for value in checked_inputs]
    if any(value._parents.split[view[0]["snapshot"]["game_id"]] != split for value, view in zip(checked_inputs, views)):
        raise ValueError("auxiliary game cannot cross parent split")
    encodings = [view[2] for view in views]
    batch = len(views)
    records = max(1, max(len(value.public_records) for value in encodings))
    sites = max(len(value.divergences) for value in encodings)
    # Include every output tensor and bounded per-slot temporary copies before
    # any allocation. Python immutable source storage has a separate byte cap.
    needed = batch * (64 * 8 + 16 * 4 + records * (16 * 4 + 1) + 3 * 8 + 1 + sites * (8 * 4 + 1) + 16 * 4
                      + 4 + 1 + 3 * 4 + 1 + sites * (4 + 1) + len(TASKS) * (4 + 1)) + 16 * 4
    if type(max_tensor_bytes) is not int or not 1 <= max_tensor_bytes <= 1024 * 1024 or needed > max_tensor_bytes:
        raise ValueError("auxiliary tensor budget denied before allocation")
    board = torch.tensor([value.board for value in encodings], dtype=torch.int64, device="cpu")
    metadata = torch.tensor([value.metadata for value in encodings], dtype=torch.float32, device="cpu")
    public = torch.zeros((batch, records, 16), dtype=torch.float32, device="cpu")
    public_mask = torch.zeros((batch, records), dtype=torch.bool, device="cpu")
    features = torch.zeros((batch, sites, 8), dtype=torch.float32, device="cpu")
    feature_mask = torch.zeros((batch, sites), dtype=torch.bool, device="cpu")
    for row, encoding in enumerate(encodings):
        for slot, (_, _, values) in enumerate(encoding.public_records):
            public[row, slot] = torch.tensor(values, dtype=torch.float32, device="cpu")
            public_mask[row, slot] = True
        for slot, (_, values) in enumerate(encoding.divergences):
            features[row, slot] = torch.tensor(values, dtype=torch.float32, device="cpu")
            feature_mask[row, slot] = True
    inputs = TensorInput(board, metadata, public, public_mask, torch.zeros((batch, 1, 3), dtype=torch.int64, device="cpu"),
                         torch.zeros((batch, 1), dtype=torch.bool, device="cpu"), features, feature_mask,
                         torch.tensor([value.query for value in encodings], dtype=torch.float32, device="cpu"))
    inputs.validate()
    # input masks indicate captured features; target masks indicate supervision.
    # No raw native logits are copied to targets, and no missing target is zero
    # supervision. Zeros below are inert storage under exclusively false masks.
    return training.TrainingBatch("critic", inputs, tuple(value.input_sha256 for value in encodings),
        torch.zeros((batch, 1), dtype=torch.float32, device="cpu"), torch.zeros(batch, dtype=torch.bool, device="cpu"),
        torch.zeros((batch, 3), dtype=torch.float32, device="cpu"), torch.zeros(batch, dtype=torch.bool, device="cpu"),
        torch.zeros((batch, sites), dtype=torch.float32, device="cpu"), torch.zeros((batch, sites), dtype=torch.bool, device="cpu"),
        torch.zeros((batch, len(TASKS)), dtype=torch.float32, device="cpu"), torch.zeros((batch, len(TASKS)), dtype=torch.bool, device="cpu"))
