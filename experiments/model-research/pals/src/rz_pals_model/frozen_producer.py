"""Pure frozen-producer metadata checks; no model/torch/runtime is imported.

Pins must come from an independent owner. Matching them is neither live-source
proof nor training admission. The caller supplies the current-view SHA from a
fully validated dataset: input/source/split/label and DG05 private-context/chain
checks must already have succeeded over these same raw bytes. This module does
not duplicate that validation or recompute Rust's raw f64 canonical dataset/split
identities. The independently pinned raw receipt must carry the completed Rust
audit and actual records/registry byte hashes. A bare view SHA or locally created
receipt is not independent evidence of that separate validation boundary.
"""
import copy
import hashlib
import json
import unicodedata

ROSTER_DOMAIN = "rz-pals-producer-roster/1"
CAPTURE_DOMAIN = "rz-pals-producer-captures/1"
ENVELOPE_DOMAIN = "rz-pals-producer-envelope/1"
OWNED_SOURCES_DOMAIN = "rz-pals-owned-source-registry/1"
MAX_RECORDS = 65536
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
U64_MAX = 2**64 - 1


def _fields(value, names, description):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(description + " fields")
    return value


def _uint(value, maximum=U64_MAX):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("expected strict bounded u64")
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256")
    return value


def _identity(value):
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= 256 or any(unicodedata.category(c) == "Cc" for c in value):
        raise ValueError("invalid bounded producer identity")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _parse(raw, maximum=MAX_MANIFEST_BYTES):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= maximum:
        raise ValueError("bounded actual artifact bytes required")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _metadata_tree(value):
    if value is None or isinstance(value, str):
        return
    if type(value) is int:
        _uint(value)
    elif isinstance(value, list):
        for child in value:
            _metadata_tree(child)
    elif isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata object keys must be strings")
        for child in value.values():
            _metadata_tree(child)
    else:
        raise ValueError("shared producer metadata contains a float or unsupported value")


def canonical_metadata(domain, value):
    """New domains only: sorted nested keys, minimal UTF-8, no JSON floats."""
    _metadata_tree(value)
    return json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def metadata_digest(domain, value):
    return hashlib.sha256(canonical_metadata(domain, value)).hexdigest()


def _source(value):
    if not isinstance(value, dict):
        raise ValueError("producer source")
    if value.get("kind") == "own_cpu":
        _fields(value, ("kind", "cpu_binary_sha256", "evaluator_configuration_sha256", "model_weights_sha256"), "CPU source")
        _sha(value["cpu_binary_sha256"])
        _sha(value["evaluator_configuration_sha256"])
        if value["model_weights_sha256"] is not None:
            _sha(value["model_weights_sha256"])
    elif value.get("kind") == "own_pals":
        _fields(value, ("kind", "model_configuration_sha256", "model_weights_sha256"), "PALS source")
        _sha(value["model_configuration_sha256"])
        _sha(value["model_weights_sha256"])
    else:
        raise ValueError("unknown producer source")
    return copy.deepcopy(value)


def _encoding(value, source):
    if not isinstance(value, dict):
        raise ValueError("producer encoding policy")
    if value.get("kind") == "native_exact":
        _fields(value, ("kind", "encoding_sha256", "encoder_source_sha256", "native_model_epoch"), "native encoding")
        _sha(value["encoding_sha256"])
        _sha(value["encoder_source_sha256"])
        epoch = value["native_model_epoch"]
        if source["kind"] == "own_cpu":
            _fields(epoch, ("kind",), "encoding-only native epoch")
            if epoch["kind"] != "encoding_only_zero":
                raise ValueError("native epoch/source kind mismatch")
        else:
            _fields(epoch, ("kind", "sha256"), "frozen native epoch")
            if epoch["kind"] != "frozen_model_epoch":
                raise ValueError("native epoch/source kind mismatch")
            _sha(epoch["sha256"])
    elif value.get("kind") == "private_checked_derived_query":
        _fields(value, ("kind", "encoding_schema_sha256", "private_encoder_source_sha256",
                        "parent_encoding_sha256", "parent_encoder_source_sha256"), "private encoding")
        if source["kind"] != "own_pals":
            raise ValueError("private derived encoding requires own PALS")
        for name in ("encoding_schema_sha256", "private_encoder_source_sha256", "parent_encoding_sha256", "parent_encoder_source_sha256"):
            _sha(value[name])
    else:
        raise ValueError("unknown producer encoding policy")
    return copy.deepcopy(value)


def normalize_roster(body):
    _fields(body, ("version", "game_producers"), "producer roster")
    entries = body["game_producers"]
    if body["version"] != ROSTER_DOMAIN or not isinstance(entries, list) or not 1 <= len(entries) <= MAX_RECORDS:
        raise ValueError("producer roster extent/version")
    result, seen, counts = [], set(), {}
    for pin in entries:
        _fields(pin, ("game_id", "producer_id", "registration_sha256", "source", "frozen_epoch", "encoding_policy"), "producer pin")
        key = _identity(pin["game_id"]), _identity(pin["producer_id"])
        if key in seen:
            raise ValueError("duplicate game/producer registration")
        seen.add(key)
        counts[key[0]] = counts.get(key[0], 0) + 1
        if counts[key[0]] > 64:
            raise ValueError("per-game producer limit")
        _sha(pin["registration_sha256"])
        _uint(pin["frozen_epoch"])
        source = _source(pin["source"])
        _encoding(pin["encoding_policy"], source)
        result.append(copy.deepcopy(pin))
    result.sort(key=lambda pin: (pin["game_id"], pin["producer_id"]))
    return {"version": ROSTER_DOMAIN, "game_producers": result}


def normalize_capture(body):
    _fields(body, ("version", "bindings"), "producer capture")
    bindings = body["bindings"]
    if body["version"] != CAPTURE_DOMAIN or not isinstance(bindings, list) or not 1 <= len(bindings) <= MAX_RECORDS:
        raise ValueError("producer capture extent/version")
    seen = set()
    for value in bindings:
        _fields(value, ("input_sha256", "game_id", "producer_id", "capture_sequence", "capture_evidence_sha256"), "input producer binding")
        identity = _sha(value["input_sha256"])
        _identity(value["game_id"])
        _identity(value["producer_id"])
        _uint(value["capture_sequence"])
        _sha(value["capture_evidence_sha256"])
        if identity in seen:
            raise ValueError("duplicate input producer binding")
        seen.add(identity)
    return {"version": CAPTURE_DOMAIN, "bindings": sorted(copy.deepcopy(bindings), key=lambda value: value["input_sha256"])}


def normalize_envelope(body):
    _fields(body, ("version", "roster_sha256", "owned_sources_sha256", "raw_dataset_sha256", "split_sha256",
                   "current_view_sha256", "raw_records", "unique_inputs", "capture_sha256", "capture_artifact"), "producer envelope")
    if body["version"] != ENVELOPE_DOMAIN or not 1 <= _uint(body["raw_records"], MAX_RECORDS) or not 1 <= _uint(body["unique_inputs"], body["raw_records"]):
        raise ValueError("producer envelope extent/version")
    for name in ("roster_sha256", "owned_sources_sha256", "raw_dataset_sha256", "split_sha256", "current_view_sha256", "capture_sha256"):
        _sha(body[name])
    artifact = _fields(body["capture_artifact"], ("bytes", "sha256"), "capture byte pin")
    if not 1 <= _uint(artifact["bytes"], MAX_MANIFEST_BYTES):
        raise ValueError("capture artifact byte limit")
    _sha(artifact["sha256"])
    return copy.deepcopy(body)


def _seal(name, domain, normalize, body):
    body = normalize(body)
    return {name: body, "sha256": metadata_digest(domain, body)}


def _load(raw, name, domain, normalize):
    value = _fields(_parse(raw), (name, "sha256"), "sealed producer metadata")
    expected = _seal(name, domain, normalize, value[name])
    if _sha(value["sha256"]) != expected["sha256"]:
        raise ValueError("producer metadata seal mismatch")
    return expected


def seal_roster(body):
    return _seal("roster", ROSTER_DOMAIN, normalize_roster, body)


def load_roster(raw):
    return _load(raw, "roster", ROSTER_DOMAIN, normalize_roster)


def seal_capture(body):
    return _seal("capture", CAPTURE_DOMAIN, normalize_capture, body)


def load_capture(raw):
    return _load(raw, "capture", CAPTURE_DOMAIN, normalize_capture)


def seal_envelope(body):
    return _seal("envelope", ENVELOPE_DOMAIN, normalize_envelope, body)


def load_envelope(raw):
    return _load(raw, "envelope", ENVELOPE_DOMAIN, normalize_envelope)


def owned_sources_digest(registry):
    _fields(registry, ("cpu_binary_sha256", "input_sources"), "owned source registry")
    cpu, sources = registry["cpu_binary_sha256"], registry["input_sources"]
    if not isinstance(cpu, list) or len(cpu) > 64 or not isinstance(sources, list) or not 1 <= len(sources) <= 256:
        raise ValueError("owned source registry extent")
    cpu = sorted({_sha(value) for value in cpu})
    unique = {}
    for source in sources:
        value = _source(source)
        if value["kind"] == "own_cpu" and value["cpu_binary_sha256"] not in cpu:
            raise ValueError("CPU source absent from owned binary registry")
        key = canonical_metadata("source-order", value)
        unique[key] = value
    return metadata_digest(OWNED_SOURCES_DOMAIN, {"cpu_binary_sha256": cpu,
                                                "input_sources": [unique[key] for key in sorted(unique)]})


def _actual_asset(receipt, name, raw):
    assets = receipt.get("artifacts")
    if not isinstance(assets, dict):
        raise ValueError("raw receipt artifacts must be an object")
    artifact = _fields(assets.get(name), ("bytes", "sha256"), "raw receipt asset")
    if _uint(artifact["bytes"]) != len(raw) or _sha(artifact["sha256"]) != hashlib.sha256(raw).hexdigest():
        raise ValueError("actual raw artifact differs from independently pinned receipt")


def audit_metadata(*, roster_bytes, envelope_bytes, capture_bytes, independently_registered,
                   source_registry_bytes, raw_receipt_bytes, records_bytes,
                   expected_raw_receipt_sha256, checked_current_view_sha256,
                   max_input_bytes=64 * 1024 * 1024):
    """Whole-history freeze checks against independent pins, metadata only.

    checked_current_view_sha256 must come from an existing fully validated
    dataset with independent source/split authority, including DG05 private
    context and label-chain checks over these same raw bytes. It must never be
    copied from the envelope or obtained from a bare chain-only helper. The raw
    receipt's audit must be independently pinned, completed Rust validation;
    its shape or SHA alone does not prove who performed that validation. Native
    tensor epoch/callback/runtime and every RequiresDerivedAdapter remain outside
    this metadata result. The receipt producer and training admission are separate
    integration work; this function neither creates nor authorizes either one.
    """
    _uint(max_input_bytes, 1024 * 1024 * 1024)
    artifacts = (roster_bytes, envelope_bytes, capture_bytes, source_registry_bytes, raw_receipt_bytes, records_bytes)
    if max_input_bytes == 0 or any(not isinstance(raw, bytes) for raw in artifacts) or sum(map(len, artifacts)) > max_input_bytes:
        raise ValueError("producer metadata aggregate byte budget")
    receipt = _parse(raw_receipt_bytes)
    if hashlib.sha256(raw_receipt_bytes).hexdigest() != _sha(expected_raw_receipt_sha256):
        raise ValueError("raw receipt differs from independent pin")
    if not isinstance(receipt, dict) or receipt.get("complete") is not True or receipt.get("failure") is not None:
        raise ValueError("incomplete raw receipt")
    _actual_asset(receipt, "records.jsonl", records_bytes)
    _actual_asset(receipt, "source-registry.jsonl", source_registry_bytes)
    registry_lines = source_registry_bytes.splitlines()
    if len(registry_lines) != 1:
        raise ValueError("one actual source registry required")
    registry = _parse(registry_lines[0])
    registry_sha = owned_sources_digest(registry)
    roster, capture, envelope = load_roster(roster_bytes), load_capture(capture_bytes), load_envelope(envelope_bytes)
    e = envelope["envelope"]
    pins = normalize_roster({"version": ROSTER_DOMAIN, "game_producers": independently_registered})["game_producers"]
    pins = {(pin["game_id"], pin["producer_id"]): pin for pin in pins}
    declarations = {(pin["game_id"], pin["producer_id"]): pin for pin in roster["roster"]["game_producers"]}
    for key, declaration in declarations.items():
        if pins.get(key) != declaration or declaration["source"] not in registry["input_sources"]:
            raise ValueError("producer differs from independent registration/owned source")
    raw_audit = receipt.get("audit")
    if not isinstance(raw_audit, dict):
        raise ValueError("independently pinned raw audit required")
    if (e["roster_sha256"] != roster["sha256"] or e["owned_sources_sha256"] != registry_sha
            or e["raw_dataset_sha256"] != _sha(raw_audit.get("canonical_dataset_sha256"))
            or e["split_sha256"] != _sha(raw_audit.get("canonical_split_sha256"))
            or e["current_view_sha256"] != _sha(checked_current_view_sha256)):
        raise ValueError("producer envelope raw/current/registry/roster mismatch")
    if (e["capture_sha256"] != capture["sha256"] or e["capture_artifact"]["bytes"] != len(capture_bytes)
            or e["capture_artifact"]["sha256"] != hashlib.sha256(capture_bytes).hexdigest()):
        raise ValueError("producer capture byte/seal mismatch")
    lines = records_bytes.splitlines()
    if not 1 <= len(lines) <= MAX_RECORDS or e["raw_records"] != len(lines) or _uint(raw_audit.get("records")) != len(lines):
        raise ValueError("raw historical row count mismatch")
    bindings = {value["input_sha256"]: value for value in capture["capture"]["bindings"]}
    seen, originals, pending, native_count = set(), {}, [], 0
    for line in lines:
        row = _fields(_parse(line, 2 * 1024 * 1024), ("input", "future_label", "verifier_private"), "raw learning row")
        frozen = _fields(row["input"], ("snapshot", "sha256"), "raw frozen input")
        identity = _sha(frozen["sha256"])
        snapshot = frozen["snapshot"]
        if not isinstance(snapshot, dict):
            raise ValueError("raw snapshot")
        binding = bindings.get(identity)
        if binding is None:
            raise ValueError("missing input producer binding")
        if binding["game_id"] != _identity(snapshot.get("game_id")) or binding["capture_sequence"] != _uint(snapshot.get("capture_sequence")):
            raise ValueError("producer binding game/capture mismatch")
        pin = declarations.get((binding["game_id"], binding["producer_id"]))
        if pin is None:
            raise ValueError("unregistered input producer")
        if pin["source"] != _source(snapshot.get("source")) or pin["frozen_epoch"] != _uint(snapshot.get("frozen_epoch")):
            raise ValueError("producer source/frozen epoch drift")
        encoding = pin["encoding_policy"]
        derived = _sha(snapshot.get("encoding_sha256"))
        if encoding["kind"] == "native_exact":
            if row["verifier_private"] is not None:
                raise ValueError("private verifier row requires derived encoding policy")
            if derived != encoding["encoding_sha256"]:
                raise ValueError("producer native encoding drift")
        elif snapshot.get("role") != "verifier":
            raise ValueError("private derived producer requires verifier role")
        if identity in originals and originals[identity] != frozen:
            raise ValueError("historical immutable input collision")
        originals[identity] = frozen
        if identity not in seen:
            seen.add(identity)
            if encoding["kind"] == "native_exact":
                native_count += 1
            else:
                pending.append({"input_sha256": identity, "game_id": binding["game_id"], "producer_id": binding["producer_id"],
                                "derived_encoding_sha256": derived,
                                **{name: encoding[name] for name in ("encoding_schema_sha256", "private_encoder_source_sha256",
                                                                    "parent_encoding_sha256", "parent_encoder_source_sha256")}})
    if len(seen) != len(bindings) or e["unique_inputs"] != len(seen):
        raise ValueError("unobserved or missing producer bindings")
    return {"scope": "metadata_only", "raw_records": len(lines), "unique_inputs": len(seen),
            "roster_sha256": roster["sha256"], "envelope_sha256": envelope["sha256"], "capture_sha256": capture["sha256"],
            "native_exact_metadata_inputs": native_count,
            "requires_derived_adapter": sorted(pending, key=lambda value: value["input_sha256"])}
