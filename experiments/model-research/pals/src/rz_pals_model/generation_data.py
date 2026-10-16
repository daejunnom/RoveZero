"""Actual generation data V3; no neural framework or Rules implementation.

Rust owns legality, Rules trajectory membership, and the submitted ReviewView.
This reader checks their sealed receipts rather than reconstructing legal chess
or adding future targets to encoder inputs. Diagnostic records remain ineligible.
"""
import copy
import hashlib
import json
import math
import struct
from collections import Counter

DATA_DOMAIN = "rz-pals-generation-data/3"
INPUT_DOMAIN = "rz-pals-generation-input/3"
VIEW_DOMAIN = "rz-pals-selected-review-view/1"
ACTUAL_SOURCE = "actual_selfplay"
ROLES = ("proposer", "critic", "validator")
TASKS = ("defend_response", "attack_repair", "widen_responses", "lower_selectivity",
         "resume_task", "cross_profile_recheck", "defer")
RULES_ENDINGS = {"checkmate", "stalemate", "repetition", "fifty_move", "seventy_five_move",
                "insufficient_material", "dead_position"}
DRAW_ENDINGS = RULES_ENDINGS - {"checkmate"}
BUCKETS = (8, 16, 32, 64, 128, 256)
TEACHER_SUBMISSION_DOMAIN = "rz-pals-generation-teacher-submission-evidence/1"
TEACHER_COMPLETION_DOMAIN = "rz-pals-generation-teacher-completion-evidence/1"
TEACHER_INPUT_BINDING_DOMAIN = "rz-pals-generation-teacher-input-binding-evidence/1"
TEACHER_INTERPRETATION = "conditional_exact_endpoint_opinion"
EVIDENCE_DOMAINS = frozenset(("rz-pals-generation-head-evidence/1", "rz-pals-generation-causal-evidence/1",
                            "rz-pals-generation-witness-evidence/1", "rz-pals-rules-outcome-evidence/1",
                            "rz-pals-generation-submission-evidence/1", "rz-pals-generation-native-completion-evidence/1",
                            "rz-pals-generation-native-retirement-evidence/1", "rz-pals-generation-runtime-audit-evidence/1",
                            "rz-pals-generation-action-authority-evidence/1", "rz-pals-generation-game-pgn-evidence/1",
                            "rz-pals-generation-game-failure-evidence/1", "rz-pals-generation-frozen-action-group-evidence/1",
                            TEACHER_SUBMISSION_DOMAIN, TEACHER_COMPLETION_DOMAIN, TEACHER_INPUT_BINDING_DOMAIN))


def fields(value, names, description):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{description} fields mismatch")
    return value


def uint(value, maximum=2**64 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid bounded unsigned integer")
    return value


def identifier(value):
    if type(value) is not str or not value or len(value.encode()) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError("invalid identifier")
    return value


def sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA-256")
    return value


def digest(domain, value):
    return hashlib.sha256(json.dumps([domain, value], sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def validate_evidence_receipt(evidence_sha256, receipt):
    sha(evidence_sha256)
    if (type(receipt) is not dict or receipt.get("schema") not in EVIDENCE_DOMAINS
            or digest(receipt["schema"], receipt) != evidence_sha256):
        raise ValueError("recovered evidence receipt domain/digest mismatch")
    if receipt["schema"] in (TEACHER_SUBMISSION_DOMAIN, TEACHER_COMPLETION_DOMAIN, TEACHER_INPUT_BINDING_DOMAIN):
        _teacher_receipt(receipt)
    elif receipt["schema"] == "rz-pals-generation-runtime-audit-evidence/1":
        fields(receipt, ("schema", "witness_json"), "raw runtime audit receipt")
        witness = receipt["witness_json"]
        if type(witness) is not str or len(witness.encode()) > 1024 * 1024:
            raise ValueError("bounded raw runtime audit JSON required")
        parsed = strict_json(witness)
        if type(parsed) is not dict or not parsed:
            raise ValueError("raw runtime audit witness must be a strict JSON object")
    elif receipt["schema"] == "rz-pals-generation-game-pgn-evidence/1":
        fields(receipt, ("schema", "game_id", "opening_id", "engine_a_white", "pgn", "clock_audit"), "raw collection PGN")
        identifier(receipt["game_id"])
        identifier(receipt["opening_id"])
        if type(receipt["engine_a_white"]) is not bool or type(receipt["pgn"]) is not str or not receipt["pgn"] or len(receipt["pgn"].encode()) > 1024 * 1024:
            raise ValueError("bounded raw collection PGN and exact engine color required")
        audit = fields(receipt["clock_audit"], ("schema", "base_ms", "increment_ms", "opening_plies", "accepted_plies",
                      "white_remaining_ns", "black_remaining_ns", "white_wall_charged_ns", "black_wall_charged_ns",
                      "white_increments", "black_increments"), "raw collection clock audit")
        if audit["schema"] != "rz-pals-collection-clock/3":
            raise ValueError("raw collection clock domain mismatch")
        for name in set(audit) - {"schema"}:
            uint(audit[name])
        uint(audit["opening_plies"], 4096)
        if uint(audit["accepted_plies"], 512) != audit["white_increments"] + audit["black_increments"]:
            raise ValueError("raw collection accepted move/increment accounting mismatch")
    elif receipt["schema"] == "rz-pals-generation-game-failure-evidence/1":
        fields(receipt, ("schema", "game_id", "opening_id", "stage", "error"), "raw failed collection game")
        for name in ("game_id", "opening_id", "stage"):
            identifier(receipt[name])
        if type(receipt["error"]) is not str or not receipt["error"] or len(receipt["error"].encode()) > 1024 * 1024:
            raise ValueError("bounded actual game failure text required")
    return receipt


def _moves(value, maximum=256):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("bounded Move16 list required")
    for move in value:
        uint(move, 65535)
        if move >> 12 > 4 or (move & 63) == ((move >> 6) & 63):
            raise ValueError("invalid Move16 encoding")
    return value


def _anchor(anchor):
    fields(anchor, ("rules_state_sha256", "rules_history_sha256", "board_fen", "white_to_move"), "Rules anchor")
    sha(anchor["rules_state_sha256"])
    sha(anchor["rules_history_sha256"])
    if (type(anchor["white_to_move"]) is not bool or type(anchor["board_fen"]) is not str
            or len(anchor["board_fen"].encode()) > 512 or any(ord(c) < 32 for c in anchor["board_fen"])):
        raise ValueError("invalid anchor board/viewpoint")
    fen = anchor["board_fen"].split()
    if len(fen) != 6 or fen[1] != ("w" if anchor["white_to_move"] else "b"):
        raise ValueError("anchor FEN/viewpoint mismatch")


def validate_view(view):
    """Validate physical selected input; actual game history is not a view field."""
    fields(view, ("context_policy_sha256", "anchors", "records", "query", "token_budget", "length_bucket"), "selected view")
    sha(view["context_policy_sha256"])
    if uint(view["token_budget"], 512) not in (128, 256, 512) or view["length_bucket"] not in BUCKETS:
        raise ValueError("unregistered context budget/bucket")
    anchors, records = view["anchors"], view["records"]
    if type(anchors) is not list or not 1 <= len(anchors) <= 4 or type(records) is not list or len(records) > 64:
        raise ValueError("selected anchors/records capacity")
    for anchor in anchors:
        _anchor(anchor)
    tokens, longest = 0, 0
    for index, record in enumerate(records):
        fields(record, ("record_id", "kind", "anchor", "moves", "parent", "supersedes", "observed_sequence",
                        "perspective", "completed_iteration", "rules_terminal"), "selected record")
        identifier(record["record_id"])
        if record["kind"] not in ("proposal", "counter", "repair", "continuation"):
            raise ValueError("unregistered record kind")
        uint(record["anchor"], len(anchors) - 1)
        _moves(record["moves"])
        tokens += len(record["moves"])
        longest = max(longest, len(record["moves"]))
        uint(record["observed_sequence"])
        if type(record["perspective"]) is not int or record["perspective"] not in (-1, 1) or type(record["completed_iteration"]) is not bool or type(record["rules_terminal"]) is not bool:
            raise ValueError("invalid compact record semantic features")
        for relation in ("parent", "supersedes"):
            if record[relation] is not None and uint(record[relation], len(records) - 1) == index:
                raise ValueError("record self relation")
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("duplicate selected record")
    query = fields(view["query"], ("purpose", "anchors", "prefix", "proposal", "counter", "counter_present", "white_to_move"), "selected query")
    identifier(query["purpose"])
    if type(query["anchors"]) is not list or len(query["anchors"]) != 3:
        raise ValueError("query requires prefix/proposal/counter anchor references")
    for ref in query["anchors"]:
        uint(ref, len(anchors) - 1)
    if (type(query["white_to_move"]) is not bool
            or query["white_to_move"] != (anchors[query["anchors"][0]]["white_to_move"] != bool(len(query["prefix"]) % 2))):
        raise ValueError("query anchor viewpoint mismatch")
    for name in ("prefix", "proposal", "counter"):
        _moves(query[name])
        tokens += len(query[name])  # Duplicated material actually read by NN is counted again.
        longest = max(longest, len(query[name]))
    if tokens > view["token_budget"] or view["length_bucket"] != next(b for b in BUCKETS if b >= longest):
        raise ValueError("view token or minimal length bucket mismatch")
    if type(query["counter_present"]) is not bool or not query["counter_present"] and query["counter"]:
        raise ValueError("query counter existence must preserve the submitted optional line")
    return view


def view_digest(view):
    validate_view(view)
    return digest(VIEW_DOMAIN, view)


INPUT_FIELDS = ("game_id", "opening_id", "line_genealogy_id", "transposition_sha256", "role",
                "model_epoch", "context_epoch", "capture_sequence", "question_id", "worker_id",
                "job_id", "slice_id", "rules_anchor", "actual_history", "selected_view",
                "view_sha256", "legal_moves", "action_support_mask", "encoded_model_input", "divergence_record_ids")


def _components(move):
    return [move & 63, (move >> 6) & 63, move >> 12]


def _validate_encoded_input(snapshot):
    # Encoding originates from exact Rust Rules, never Python FEN reconstruction.
    from .virtual_review import validate_payload
    payload = snapshot["encoded_model_input"]
    validate_payload(payload)
    view = snapshot["selected_view"]
    if (payload["role"] != snapshot["role"] or payload["token_budget"] != view["token_budget"]
            or len(payload["anchors"]) != len(view["anchors"]) or len(payload["records"]) != len(view["records"])
            or payload["candidates"] != [_components(m) for m in snapshot["legal_moves"]]):
        raise ValueError("encoded input role/anchor/candidate allocation mismatch")
    kinds = {"proposal": "proposal", "counter": "counterexample", "repair": "repair", "continuation": "cpu_verification"}
    for source, encoded in zip(view["records"], payload["records"]):
        expected = {"kind": kinds[source["kind"]], "perspective": source["perspective"],
                    "completed_iteration": source["completed_iteration"], "rules_terminal": source["rules_terminal"],
                    "line": [_components(m) for m in source["moves"]], "anchor": source["anchor"],
                    "parent": source["parent"], "supersedes": source["supersedes"]}
        if encoded != expected:
            raise ValueError("encoded record changed selected-view semantics")
    source, encoded = view["query"], payload["query"]
    if (encoded["purpose"] != source["purpose"] or encoded["perspective"] != (1 if source["white_to_move"] else -1)
            or encoded["counter_present"] != source["counter_present"]
            or encoded["anchors"] != source["anchors"]
            or any(encoded[name] != [_components(m) for m in source[name]] for name in ("prefix", "proposal", "counter"))):
        raise ValueError("encoded query changed selected-view semantics")


def validate_input(value):
    fields(value, INPUT_FIELDS, "generation input")
    for name in ("game_id", "opening_id", "line_genealogy_id", "question_id", "worker_id", "job_id", "slice_id"):
        identifier(value[name])
    sha(value["transposition_sha256"])
    if value["role"] not in ROLES:
        raise ValueError("unregistered generation role")
    for name in ("model_epoch", "context_epoch", "capture_sequence"):
        uint(value[name])
    _anchor(value["rules_anchor"])
    _moves(value["actual_history"], 16384)  # Audit/Rules history, never appended to review tensor.
    validate_view(value["selected_view"])
    # The public root anchor and the exact NN head endpoint are different when
    # query.prefix is nonempty. Rust Rules owns replay/state/history; WDL may
    # be known only when this exact endpoint occurs in the actual trajectory.
    if (value["selected_view"]["query"]["white_to_move"] != value["rules_anchor"]["white_to_move"]
            or sha(value["view_sha256"]) != view_digest(value["selected_view"])):
        raise ValueError("exact query endpoint/view digest mismatch")
    if any(r["observed_sequence"] > value["capture_sequence"] for r in value["selected_view"]["records"]):
        raise ValueError("future observation leaked into submitted view")
    _moves(value["legal_moves"])
    if len(set(value["legal_moves"])) != len(value["legal_moves"]):
        raise ValueError("duplicate legal candidate")
    mask = value["action_support_mask"]
    if type(mask) is not list or len(mask) != len(TASKS) or any(type(v) is not bool for v in mask):
        raise ValueError("invalid action support mask")
    _validate_encoded_input(value)
    ids = value["divergence_record_ids"]
    if type(ids) is not list or len(ids) != len(value["encoded_model_input"]["divergences"]) or len(set(ids)) != len(ids):
        raise ValueError("divergence input/provenance ordering mismatch")
    visible = {r["record_id"] for r in value["selected_view"]["records"] if r["kind"] == "counter"}
    if any(identifier(v) not in visible for v in ids):
        raise ValueError("divergence refers to absent selected counter")
    return value


def seal_input(value):
    validate_input(value)
    projected = copy.deepcopy(value)
    # Rust/Python decimal exponent spelling differs. Only the seal projection
    # uses exact f32 bits; the actual submitted model payload stays numeric.
    payload = projected["encoded_model_input"]
    def f32_bits(number):
        try:
            return struct.pack(">f", float(number)).hex()
        except (OverflowError, ValueError, struct.error) as error:
            raise ValueError("model payload value cannot be represented as f32") from error
    for anchor in payload["anchors"]:
        anchor["metadata"] = [f32_bits(v) for v in anchor["metadata"]]
    payload["divergences"] = [[f32_bits(v) for v in row] for row in payload["divergences"]]
    return digest(INPUT_DOMAIN, projected)


def _label_base(label, snapshot, description):
    if label["observed_sequence"] < snapshot["capture_sequence"]:
        raise ValueError(f"{description} predates input")
    uint(label["observed_sequence"])
    if label["question_id"] != snapshot["question_id"] or label["view_sha256"] != snapshot["view_sha256"]:
        raise ValueError(f"{description} question/view mismatch")
    sha(label["raw_evidence_sha256"])


def policy_target(snapshot, label):
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source",
                   "completed", "stale", "anchor", "ranked_moves", "teacher_binary_sha256", "teacher_profile_sha256",
                   "go_sequence", "completed_depth"), "conditional teacher policy")
    _label_base(label, snapshot, "policy")
    if label["source"] != "stockfish19_completed_multipv" or label["completed"] is not True or label["stale"] is not False or label["anchor"] != snapshot["rules_anchor"]:
        raise ValueError("policy requires completed nonstale same-anchor teacher opinion")
    sha(label["teacher_binary_sha256"])
    sha(label["teacher_profile_sha256"])
    if uint(label["go_sequence"]) == 0 or uint(label["completed_depth"], 2**32 - 1) == 0:
        raise ValueError("teacher go/depth completion authority required")
    moves = _moves(label["ranked_moves"], 4)
    if not moves or len(set(moves)) != len(moves) or any(m not in snapshot["legal_moves"] for m in moves):
        raise ValueError("teacher opinion contains invalid legal candidates")
    weights = [math.exp(-rank) for rank in range(len(moves))]
    total = sum(weights)
    return [weights[moves.index(m)] / total if m in moves else 0.0 for m in snapshot["legal_moves"]]


def _teacher_process(value):
    fields(value, ("pid", "process_group", "proc_start_ticks"), "teacher actual process identity")
    uint(value["pid"], 2**32 - 1)
    uint(value["process_group"], 2**32 - 1)
    uint(value["proc_start_ticks"])
    return value


def _teacher_text(value, *, optional=False, maximum=1024):
    if optional and value is None:
        return
    if (type(value) is not str or not value or len(value.encode()) > maximum
            or any(ord(c) < 32 for c in value)):
        raise ValueError("bounded teacher raw text required")


def _teacher_uci(value):
    # Serialization validation only. Exact legality remains the Rust Rules
    # producer's responsibility; no chess Rules are reconstructed here.
    if (type(value) is not str or len(value) not in (4, 5)
            or value[0] not in "abcdefgh" or value[2] not in "abcdefgh"
            or value[1] not in "12345678" or value[3] not in "12345678"
            or value[:2] == value[2:4] or len(value) == 5 and value[4] not in "qrbn"):
        raise ValueError("teacher raw UCI move serialization mismatch")
    return value


def _move16_uci(value):
    _moves([value])
    source, target, promotion = value & 63, (value >> 6) & 63, value >> 12
    square = lambda number: "abcdefgh"[number % 8] + "12345678"[number // 8]
    return square(source) + square(target) + ("qrbn"[promotion - 1] if promotion else "")


def _teacher_identity(value):
    fields(value, ("adapter_semantics", "binary_sha256", "launch_arguments_sha256", "declared_name", "declared_version",
                   "declared_source", "declared_license", "options", "assets", "model_metadata"), "teacher checker identity")
    for name in ("binary_sha256", "launch_arguments_sha256"):
        sha(value[name])
    for name in ("adapter_semantics", "declared_name", "declared_version", "declared_source", "declared_license"):
        _teacher_text(value[name])
    options = value["options"]
    if type(options) is not dict or len(options) > 128:
        raise ValueError("bounded teacher options required")
    for name, option in options.items():
        _teacher_text(name, maximum=256)
        if type(option) is not str or len(option.encode()) > 1024 or any(ord(c) < 32 for c in option):
            raise ValueError("bounded teacher option declaration required")
    assets = value["assets"]
    if type(assets) is not list or len(assets) > 32:
        raise ValueError("bounded teacher asset declarations required")
    for asset in assets:
        fields(asset, ("purpose", "sha256"), "teacher declared asset")
        _teacher_text(asset["purpose"], maximum=256); sha(asset["sha256"])
    metadata = fields(value["model_metadata"], ("weights_sha256", "training", "declared_rights", "precision"), "teacher model metadata")
    if metadata["weights_sha256"] is not None:
        sha(metadata["weights_sha256"])
    if metadata["training"] != "unknown":
        raise ValueError("teacher declaration cannot certify its training knowledge")
    for name in ("declared_rights", "precision"):
        _teacher_text(metadata[name], optional=True)
    return value


def _teacher_namespace(identity):
    """Independent framing of the existing typed Rust identity, not its value."""
    _teacher_identity(identity)
    ordered = {name: identity[name] for name in ("adapter_semantics", "binary_sha256", "launch_arguments_sha256",
        "declared_name", "declared_version", "declared_source", "declared_license", "options", "assets", "model_metadata")}
    ordered["options"] = dict(sorted(identity["options"].items()))
    ordered["assets"] = [{name: asset[name] for name in ("purpose", "sha256")} for asset in identity["assets"]]
    ordered["model_metadata"] = {name: identity["model_metadata"][name]
                               for name in ("weights_sha256", "training", "declared_rights", "precision")}
    raw = json.dumps(ordered, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(b"rz-external-evaluation-namespace/1\0" + len(raw).to_bytes(8, "little") + raw).hexdigest()


def _teacher_receipt(value):
    """Keep failed/unknown raw observations; shape validation does not admit labels."""
    if value["schema"] == TEACHER_INPUT_BINDING_DOMAIN:
        fields(value, ("schema", "input_sha256", "policy_head_evidence_sha256", "teacher_submission_sha256",
                       "teacher_completion_sha256", "rules_anchor", "observed_sequence", "interpretation"), "teacher input binding")
        for name in ("input_sha256", "policy_head_evidence_sha256", "teacher_submission_sha256", "teacher_completion_sha256"):
            sha(value[name])
        _anchor(value["rules_anchor"]); uint(value["observed_sequence"])
        if value["interpretation"] != TEACHER_INTERPRETATION:
            raise ValueError("teacher input binding interpretation mismatch")
    elif value["schema"] == TEACHER_SUBMISSION_DOMAIN:
        fields(value, ("schema", "observed_sequence", "execution_epoch", "execution_sequence", "job_sequence", "root_generation",
            "source_cpu_question_sha256", "source_cpu_view_sha256", "teacher_question_sha256", "evaluation_namespace",
            "teacher_binary_sha256", "teacher_profile_sha256", "root_restriction_sha256", "search_conditions_sha256",
            "root_moves", "process", "uci_name", "uci_author", "rules_anchor", "interpretation"), "teacher owner submission")
        for name in ("observed_sequence", "execution_epoch", "execution_sequence", "job_sequence", "root_generation"):
            uint(value[name])
        for name in ("source_cpu_question_sha256", "source_cpu_view_sha256", "teacher_question_sha256", "evaluation_namespace",
                     "teacher_binary_sha256", "teacher_profile_sha256", "root_restriction_sha256", "search_conditions_sha256"):
            sha(value[name])
        _teacher_process(value["process"]); _anchor(value["rules_anchor"])
        _teacher_text(value["uci_name"]); _teacher_text(value["uci_author"], optional=True)
        moves = value["root_moves"]
        if moves is not None:
            if type(moves) is not list or not 1 <= len(moves) <= 256 or len(set(moves)) != len(moves):
                raise ValueError("bounded unique teacher root restriction required")
            for movement in moves:
                _teacher_uci(movement)
        if value["interpretation"] != TEACHER_INTERPRETATION + "; neural_review_not_read_by_teacher":
            raise ValueError("teacher submission does not state its conditional endpoint scope")
    elif value["schema"] == TEACHER_COMPLETION_DOMAIN:
        fields(value, ("schema", "observed_sequence", "submission_observed_sequence", "execution_epoch", "execution_sequence",
            "job_sequence", "root_generation", "source_scope_joined", "successful_owner_join", "evaluation_namespace",
            "physical_process_identity", "teacher_binary_sha256", "teacher_profile_sha256", "exact_resume_supported",
            "go_issue", "result", "original_question_target_eligible", "cancel_sequence"), "teacher owner completion")
        for name in ("observed_sequence", "submission_observed_sequence", "execution_epoch", "execution_sequence", "job_sequence", "root_generation", "cancel_sequence"):
            uint(value[name])
        for name in ("evaluation_namespace", "teacher_binary_sha256", "teacher_profile_sha256"):
            sha(value[name])
        for name in ("source_scope_joined", "successful_owner_join", "exact_resume_supported", "original_question_target_eligible"):
            if type(value[name]) is not bool:
                raise ValueError("teacher owner completion flags require exact booleans")
        if value["physical_process_identity"] is not None:
            _teacher_process(value["physical_process_identity"])
        if value["go_issue"] not in ("Issued", "NotIssued", "Unknown"):
            raise ValueError("teacher raw go issuance state mismatch")
        result = value["result"]
        if type(result) is dict and set(result) == {"error"}:
            _teacher_text(result["error"])
            return value
        fields(result, ("identity", "uci_name", "completion", "go_sequence", "go_issue", "best_move_move16",
            "readiness_barrier_completed", "policy_target_known", "requested_multipv", "option_application_observed",
            "reported_nodes", "opinions"), "teacher raw successful result")
        _teacher_identity(result["identity"]); _teacher_text(result["uci_name"])
        if result["completion"] not in ("Pending", "BestMove", "StoppedDeadline", "StoppedCanceled", "StoppedYield") or result["go_issue"] not in ("Issued", "NotIssued", "Unknown"):
            raise ValueError("teacher raw terminal state mismatch")
        for name in ("readiness_barrier_completed", "policy_target_known", "option_application_observed"):
            if type(result[name]) is not bool:
                raise ValueError("teacher raw result flags require exact booleans")
        uint(result["go_sequence"]); uint(result["requested_multipv"], 4)
        if result["reported_nodes"] is not None:
            uint(result["reported_nodes"])
        if result["best_move_move16"] is not None:
            _moves([result["best_move_move16"]])
        opinions = result["opinions"]
        if type(opinions) is not list or len(opinions) > 4:
            raise ValueError("bounded teacher raw opinion set required")
        for opinion in opinions:
            fields(opinion, ("rank", "depth", "seldepth", "score", "bound", "reported_wdl", "pv_move16", "pv"), "teacher raw opinion")
            uint(opinion["rank"], 4); uint(opinion["depth"], 2**16 - 1)
            if opinion["seldepth"] is not None:
                uint(opinion["seldepth"], 2**16 - 1)
            _teacher_text(opinion["score"])
            if opinion["bound"] not in ("ExactReported", "Lower", "Upper", "Unknown"):
                raise ValueError("teacher raw bound state mismatch")
            if opinion["reported_wdl"] is not None:
                values = opinion["reported_wdl"]
                if type(values) is not list or len(values) != 3:
                    raise ValueError("teacher raw WDL tuple mismatch")
                for component in values:
                    uint(component, 2**16 - 1)
            if opinion["pv_move16"] is not None:
                _moves(opinion["pv_move16"])
            if type(opinion["pv"]) is not list or len(opinion["pv"]) > 256:
                raise ValueError("bounded teacher raw UCI PV required")
            for movement in opinion["pv"]:
                _teacher_uci(movement)
    else:
        raise ValueError("unregistered teacher receipt domain")
    return value


class _RecoveredEvidence:
    """Legacy in-memory authority index built once, never scanned per target."""
    def __init__(self, values):
        self.values = values
        self.teacher_bindings, self.native_retirements = {}, {}
        for identity, value in values.items():
            if value["schema"] == TEACHER_INPUT_BINDING_DOMAIN:
                key = (value["input_sha256"], value["policy_head_evidence_sha256"])
                if key in self.teacher_bindings:
                    raise ValueError("duplicate teacher input/source bindings")
                self.teacher_bindings[key] = (identity, value)
            elif value["schema"] == "rz-pals-generation-native-retirement-evidence/1" and value.get("accepted") is True:
                key = sha(value["input_sha256"])
                if key in self.native_retirements:
                    raise ValueError("duplicate accepted original native retirements")
                self.native_retirements[key] = identity

    def get(self, key, default=None):
        return self.values.get(key, default)

    def teacher_input_binding(self, input_sha256, policy_head_evidence_sha256):
        found = self.teacher_bindings.get((sha(input_sha256), sha(policy_head_evidence_sha256)))
        if found is None:
            raise ValueError("teacher policy missing actual input/source binding")
        return found

    def accepted_retirement(self, input_sha256):
        found = self.native_retirements.get(sha(input_sha256))
        if found is None:
            raise ValueError("original teacher input lacks accepted native retirement")
        return found


def _teacher_policy_evidence(record, target, receipts):
    """Admit a conditional opinion only after independent native/teacher joins.

    The original selected NN input and teacher endpoint are joined through the
    exact Rules anchor. The source CPU question/view may differ and is never
    relabelled as the NN review the teacher supposedly consumed.
    """
    if record is None or not hasattr(receipts, "teacher_input_binding") or not hasattr(receipts, "accepted_retirement"):
        raise ValueError("teacher policy requires original input and bounded recovered receipt index")
    snapshot, input_sha = record["input"], record["input_sha256"]
    binding_sha, binding = receipts.teacher_input_binding(input_sha, target["raw_evidence_sha256"])
    validate_evidence_receipt(binding_sha, binding)
    if (binding["input_sha256"] != input_sha or binding["policy_head_evidence_sha256"] != target["raw_evidence_sha256"]
            or binding["rules_anchor"] != snapshot["rules_anchor"] or binding["observed_sequence"] != target["observed_sequence"]):
        raise ValueError("teacher binding changed original input/head/anchor/chronology")
    def recover(field, domain):
        identity = sha(binding[field])
        raw = receipts.get(identity)
        validate_evidence_receipt(identity, raw)
        if raw["schema"] != domain:
            raise ValueError("teacher source receipt domain mismatch")
        return raw
    submitted = recover("teacher_submission_sha256", TEACHER_SUBMISSION_DOMAIN)
    completed = recover("teacher_completion_sha256", TEACHER_COMPLETION_DOMAIN)
    result = completed["result"]
    if (not snapshot["capture_sequence"] < submitted["observed_sequence"] < target["observed_sequence"]
            or completed["observed_sequence"] != target["observed_sequence"]
            or completed["submission_observed_sequence"] != submitted["observed_sequence"]
            or submitted["rules_anchor"] != snapshot["rules_anchor"]
            or snapshot["worker_id"] != f"native-owner-{submitted['execution_epoch']}"):
        raise ValueError("teacher source crossed original input chronology/Rules/process context")
    for name in ("execution_epoch", "execution_sequence", "job_sequence", "root_generation", "evaluation_namespace"):
        if submitted[name] != completed[name]:
            raise ValueError("teacher original execution/job/root/namespace changed")
    for name in ("teacher_binary_sha256", "teacher_profile_sha256"):
        if submitted[name] != target[name] or completed[name] != target[name]:
            raise ValueError("teacher registered binary/profile changed")
    process = submitted["process"]
    if (process != completed["physical_process_identity"] or not process["pid"] or not process["proc_start_ticks"]
            or completed["successful_owner_join"] is not True or completed["source_scope_joined"] is not True
            or completed["exact_resume_supported"] is not False or completed["go_issue"] != "Issued"
            or completed["original_question_target_eligible"] is not True or completed["cancel_sequence"] != 0
            or "error" in result or result["completion"] != "BestMove" or result["go_issue"] != "Issued"
            or result["go_sequence"] != target["go_sequence"] or result["readiness_barrier_completed"] is not True
            or result["policy_target_known"] is not True or result["identity"]["binary_sha256"] != target["teacher_binary_sha256"]
            or result["uci_name"] != submitted["uci_name"] or _teacher_namespace(result["identity"]) != submitted["evaluation_namespace"]):
        raise ValueError("teacher actual owner/go/readiness/identity source was not joined")
    opinions = sorted(result["opinions"], key=lambda opinion: opinion["rank"])
    if not opinions or result["requested_multipv"] != len(opinions):
        raise ValueError("teacher completed opinion count differs from requested rank set")
    ranked = []
    for rank, opinion in enumerate(opinions, 1):
        pv = opinion["pv_move16"]
        if (opinion["rank"] != rank or opinion["depth"] != target["completed_depth"] or opinion["bound"] != "ExactReported"
                or not pv or opinion["pv"] != [_move16_uci(movement) for movement in pv]):
            raise ValueError("teacher opinion rank/depth/bound/PV serialization changed")
        ranked.append(pv[0])
    if ranked != target["ranked_moves"] or result["best_move_move16"] != ranked[0]:
        raise ValueError("teacher policy differs from recovered complete ranked first moves")
    if submitted["root_moves"] is not None and any(_move16_uci(movement) not in submitted["root_moves"] for movement in ranked):
        raise ValueError("teacher ranked moves violate original registered root restriction")
    from .generation_chunks import _native_chain
    native = _native_chain(record, receipts, receipts.accepted_retirement(input_sha), with_details=True)
    ticket = native["submission"]["ticket"]
    if (snapshot["role"] == "validator"
            or snapshot["selected_view"]["query"]["purpose"] not in ("propose_policy", "reply_policy", "repair_policy")
            or ticket["epoch"] != submitted["execution_epoch"]
            or ticket["root"] != submitted["root_generation"]):
        raise ValueError("teacher conditional opinion crossed original native process/root authority")


def wdl_target(snapshot, label):
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source",
                   "game_id", "anchor", "trajectory_membership_sha256", "rules_result_sha256", "ending", "outcome"), "actual Rules WDL")
    _label_base(label, snapshot, "WDL")
    sha(label["trajectory_membership_sha256"])
    sha(label["rules_result_sha256"])
    if (label["source"] != "rules_selfplay_trajectory" or label["game_id"] != snapshot["game_id"]
            or label["anchor"] != snapshot["rules_anchor"] or label["ending"] not in RULES_ENDINGS
            or label["outcome"] not in ("white_win", "black_win", "draw")):
        raise ValueError("WDL requires exact actual trajectory anchor and Rules termination")
    if (label["ending"] in DRAW_ENDINGS) != (label["outcome"] == "draw"):
        raise ValueError("Rules termination/outcome mismatch")
    if label["outcome"] == "draw":
        return [0.0, 1.0, 0.0]
    win = (label["outcome"] == "white_win") == snapshot["rules_anchor"]["white_to_move"]
    return [1.0, 0.0, 0.0] if win else [0.0, 0.0, 1.0]


def trajectory_digest(authority):
    fields(authority, ("game_id", "ending", "outcome", "anchors", "result_evidence_sha256"), "actual trajectory membership")
    identifier(authority["game_id"])
    sha(authority["result_evidence_sha256"])
    if authority["ending"] not in RULES_ENDINGS or authority["outcome"] not in ("white_win", "black_win", "draw"):
        raise ValueError("actual trajectory needs Rules termination")
    if (authority["ending"] in DRAW_ENDINGS) != (authority["outcome"] == "draw"):
        raise ValueError("actual trajectory outcome/termination mismatch")
    if type(authority["anchors"]) is not list or not 1 <= len(authority["anchors"]) <= 4096:
        raise ValueError("bounded actual Rules trajectory anchors required")
    for anchor in authority["anchors"]:
        _anchor(anchor)
    return digest("rz-pals-actual-trajectory-membership/1", authority)


def divergence_target(snapshot, label):
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source",
                   "counter_record_id", "repair_question_id", "status"), "conditional divergence")
    _label_base(label, snapshot, "divergence")
    if label["source"] != "actual_repair_continuation" or label["status"] not in ("supported_after_repair", "refuted_by_repair"):
        raise ValueError("unresolved/disputed divergence cannot become a target")
    identifier(label["repair_question_id"])
    if not any(r["record_id"] == label["counter_record_id"] and r["kind"] == "counter" for r in snapshot["selected_view"]["records"]):
        raise ValueError("counter was not visible in submitted input")
    if label["counter_record_id"] not in snapshot["divergence_record_ids"]:
        raise ValueError("counter has no submitted divergence-head candidate")
    return 1.0 if label["status"] == "supported_after_repair" else 0.0


def task_pairs(snapshot, label):
    """Only causal newly witnessed Pareto dominance yields a known action pair."""
    fields(label, ("observed_sequence", "question_id", "view_sha256", "raw_evidence_sha256", "source", "outcomes", "target_known_mask"), "action outcomes")
    _label_base(label, snapshot, "V task")
    if label["source"] != "actual_action_witness" or type(label["outcomes"]) is not list or len(label["outcomes"]) > len(TASKS):
        raise ValueError("actual bounded causal action outcomes required")
    result = derive_task_targets(snapshot, label["outcomes"])
    if any(witness["observed_sequence"] > label["observed_sequence"]
           for outcome in label["outcomes"] for witness in outcome["new_witnesses"]):
        raise ValueError("action target publication predates its collected witness")
    if (type(label["target_known_mask"]) is not list or len(label["target_known_mask"]) != len(TASKS)
            or any(type(v) is not bool for v in label["target_known_mask"]) or label["target_known_mask"] != result["target_known_mask"]):
        raise ValueError("V target-known mask differs from actual causal Pareto evidence")
    return result


def derive_task_targets(snapshot, observed_outcomes):
    """Derive a label mask after collection, without changing the sealed input."""
    if type(observed_outcomes) is not list or len(observed_outcomes) > len(TASKS):
        raise ValueError("bounded actual action outcomes required")
    outcomes, seen_actions = {}, set()
    visible = {r["record_id"] for r in snapshot["selected_view"]["records"]}
    for outcome in observed_outcomes:
        fields(outcome, ("action", "execution_id", "completed", "wall_cost_ns", "new_witnesses", "causal_receipt_sha256"), "action outcome")
        action = outcome["action"]
        if action not in TASKS or action in seen_actions or not snapshot["action_support_mask"][TASKS.index(action)]:
            raise ValueError("unsupported/duplicate action target")
        # Completion and known exclusive cost determine supervision, not the
        # identity of the action event. Unknown/incomplete events still cannot
        # be repeated under the same immutable action context.
        seen_actions.add(action)
        identifier(outcome["execution_id"])
        sha(outcome["causal_receipt_sha256"])
        if outcome["wall_cost_ns"] is not None:
            uint(outcome["wall_cost_ns"])
        if type(outcome["completed"]) is not bool or type(outcome["new_witnesses"]) is not list or len(outcome["new_witnesses"]) > 256:
            raise ValueError("invalid action completion/witness list")
        witnesses = set()
        for witness in outcome["new_witnesses"]:
            fields(witness, ("witness_id", "kind", "observed_sequence", "raw_evidence_sha256"), "new valid witness")
            identifier(witness["witness_id"])
            sha(witness["raw_evidence_sha256"])
            if (witness["kind"] not in ("counter", "repair") or witness["witness_id"] in visible
                    or witness["witness_id"] in witnesses or uint(witness["observed_sequence"]) <= snapshot["capture_sequence"]):
                raise ValueError("duplicate/old/noncausal witness cannot reward V")
            witnesses.add(witness["witness_id"])
        if outcome["completed"] and outcome["wall_cost_ns"] is not None:
            outcomes[action] = (len(witnesses), outcome["wall_cost_ns"])
    known, pairs = [False] * len(TASKS), []
    for preferred, a in outcomes.items():
        for other, b in outcomes.items():
            if preferred == other or max(a[0], b[0]) == 0:
                continue  # No new evidence, including Defer, supplies no utility supervision.
            if a[0] >= b[0] and a[1] <= b[1] and a != b:
                i, j = TASKS.index(preferred), TASKS.index(other)
                known[i] = known[j] = True
                pairs.append((i, j))
    return {"target_known_mask": known, "pairs": pairs, "new_witness_context": any(v[0] for v in outcomes.values())}


def validate_record(record):
    fields(record, ("schema", "source_kind", "eligibility", "input", "input_sha256", "future_targets"), "generation record V3")
    if record["schema"] != DATA_DOMAIN or record["source_kind"] != ACTUAL_SOURCE or record["eligibility"] != "actual_generation_candidate":
        raise ValueError("diagnostic/fixture assets cannot be promoted into actual generation")
    snapshot = validate_input(record["input"])
    if sha(record["input_sha256"]) != seal_input(snapshot):
        raise ValueError("selected input seal mismatch")
    targets = fields(record["future_targets"], ("policy", "wdl", "divergence", "task"), "head-separated future targets")
    result = dict.fromkeys(targets, None)
    for head, function in (("policy", policy_target), ("wdl", wdl_target), ("divergence", divergence_target), ("task", task_pairs)):
        if targets[head] is not None:
            if (snapshot["role"] == "validator" and head != "task") or (snapshot["role"] != "validator" and head == "task") or (head == "divergence" and snapshot["role"] != "critic"):
                raise ValueError("target head violates private role routing")
            result[head] = function(snapshot, targets[head])
    return result


def coverage(records):
    """Count unique actual inputs; duplicate rows never fill admission quotas."""
    counts, seen = Counter(), set()
    for record in records:
        targets = validate_record(record)
        identity = record["input_sha256"]
        if identity in seen:
            counts["duplicate_inputs_excluded"] += 1
            continue
        seen.add(identity)
        _count_target_coverage(counts, record, targets)
    return dict(counts)


def _count_target_coverage(counts, record, targets):
    """Count one previously verified unique input; disk index owns uniqueness."""
    role = record["input"]["role"]
    counts[f"{role}_unique_inputs"] += 1
    if role != "validator" and any(targets[h] is not None for h in ("policy", "wdl", "divergence")):
        counts[f"{role}_valid_samples"] += 1
    for head, target in targets.items():
        counts[f"{role}_{head}_{'masked' if target is None else 'valid'}"] += 1
    if role == "validator" and targets["task"]:
        counts["validator_known_pairs"] += len(targets["task"]["pairs"])
        counts["validator_new_witness_contexts"] += int(bool(targets["task"]["pairs"]) and targets["task"]["new_witness_context"])


def admit_phase(records, phase):
    observed = coverage(records)
    return _admit_coverage(observed, phase)


def _admit_coverage(observed, phase):
    required = {"pc": {"proposer_valid_samples": 1024, "critic_valid_samples": 1024},
                "v": {"validator_known_pairs": 128, "validator_new_witness_contexts": 32}}
    if phase not in required:
        raise ValueError("unknown generation phase")
    missing = {name: minimum - observed.get(name, 0) for name, minimum in required[phase].items() if observed.get(name, 0) < minimum}
    if missing:
        raise ValueError("InsufficientValidTargets: " + json.dumps(missing, sort_keys=True))
    return observed


def split_records(records, opening_splits):
    """Drop all known cross-split genealogy/transposition inputs symmetrically."""
    if type(opening_splits) is not dict or any(v not in ("train", "validation", "holdout") for v in opening_splits.values()):
        raise ValueError("opening split must be registered before outcomes")
    domains = {}
    for record in records:
        validate_record(record)
        s = record["input"]
        if s["opening_id"] not in opening_splits:
            raise ValueError("unregistered opening")
        split = opening_splits[s["opening_id"]]
        for key in (("genealogy", s["line_genealogy_id"]), ("transposition", s["transposition_sha256"]),
                    ("rules_state", s["rules_anchor"]["rules_state_sha256"]), ("game", s["game_id"])):
            domains.setdefault(key, set()).add(split)
    leaked = {key for key, splits in domains.items() if len(splits) > 1}
    outputs, excluded = {name: [] for name in ("train", "validation", "holdout")}, 0
    for record in records:
        s = record["input"]
        if any(k in leaked for k in (("genealogy", s["line_genealogy_id"]), ("transposition", s["transposition_sha256"]),
                                     ("rules_state", s["rules_anchor"]["rules_state_sha256"]), ("game", s["game_id"]))):
            excluded += 1
        else:
            outputs[opening_splits[s["opening_id"]]].append(copy.deepcopy(record))
    return outputs, {"cross_split_records_excluded": excluded, "cross_split_identities": len(leaked)}


class GenerationDataset:
    """Immutable admitted rows bound to real producer and trajectory receipts.

    The caller supplies independently recovered producer authorities. A row's
    own source_kind string is never sufficient evidence of an actual collection.
    This class neither fabricates a receipt nor promotes a diagnostic asset.
    """
    def __init__(self, records, *, producer, trajectories, opening_splits, allowed_evidence):
        fields(producer, ("schema", "source_kind", "source_sha256", "model_sha256", "runtime_sha256",
                          "teacher_binary_sha256", "teacher_profile_sha256", "opening_split_sha256",
                          "registered_before_outcomes", "input_seals", "actual_game_ids"), "generation producer authority")
        if (producer["schema"] != "rz-pals-actual-collection-authority/1" or producer["source_kind"] != ACTUAL_SOURCE
                or producer["registered_before_outcomes"] is not True):
            raise ValueError("actual producer authority with prior split registration required")
        for name in ("source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256", "teacher_profile_sha256", "opening_split_sha256"):
            sha(producer[name])
        if producer["opening_split_sha256"] != digest("rz-pals-opening-split/1", opening_splits):
            raise ValueError("outcome-selected or changed opening split")
        if type(records) is not list or len(records) > 262144 or type(trajectories) is not dict:
            raise ValueError("bounded actual records/trajectory authority required")
        if type(producer["input_seals"]) is not list or type(producer["actual_game_ids"]) is not list:
            raise ValueError("actual producer input/game identities required")
        if len(producer["input_seals"]) > 262144 or len(producer["actual_game_ids"]) > 8192 or len(trajectories) > 8192:
            raise ValueError("actual producer/trajectory authority capacity")
        seals = set(sha(v) for v in producer["input_seals"])
        games = set(identifier(v) for v in producer["actual_game_ids"])
        if len(seals) != len(producer["input_seals"]) or len(games) != len(producer["actual_game_ids"]):
            raise ValueError("duplicate producer publication input/game identity")
        if type(allowed_evidence) is not dict or len(allowed_evidence) > 1048576:
            raise ValueError("independently recovered full evidence receipts required, not digest strings")
        evidence = copy.deepcopy(allowed_evidence)
        for evidence_sha256, receipt in evidence.items():
            validate_evidence_receipt(evidence_sha256, receipt)
        for membership_sha256, authority in trajectories.items():
            if sha(membership_sha256) != trajectory_digest(authority):
                raise ValueError("actual trajectory membership seal mismatch")
        lookup = _RecoveredEvidence(evidence)
        for record in records:
            self._validate_published_record(record, producer, seals, games, trajectories, lookup)
        split_rows, exclusions = split_records(records, opening_splits)
        self._records = copy.deepcopy(split_rows)
        self._producer = copy.deepcopy(producer)
        self._authority = copy.deepcopy(trajectories)
        self._evidence = evidence
        self._splits = copy.deepcopy(opening_splits)
        self.exclusions = exclusions
        self._seal = self._integrity()

    @staticmethod
    def _validate_published_record(record, producer, seals, games, trajectories, evidence):
        """Shared exact joins for in-memory and bounded disk-backed consumers."""
        validate_record(record)
        snapshot = record["input"]
        if record["input_sha256"] not in seals or snapshot["game_id"] not in games:
            raise ValueError("input absent from actual producer publication")
        for head, target in record["future_targets"].items():
            if target is not None:
                GenerationDataset._join_evidence(record["input_sha256"], head, target, evidence, record=record)
                if head == "policy" and (target["teacher_binary_sha256"] != producer["teacher_binary_sha256"]
                                         or target["teacher_profile_sha256"] != producer["teacher_profile_sha256"]):
                    raise ValueError("conditional policy teacher differs from registered CPU_T")
        label = record["future_targets"]["wdl"]
        if label is not None:
            authority = trajectories.get(label["trajectory_membership_sha256"])
            fields(authority, ("game_id", "ending", "outcome", "anchors", "result_evidence_sha256"), "actual Rules trajectory authority")
            if (authority["game_id"] != snapshot["game_id"] or authority["ending"] != label["ending"]
                    or authority["outcome"] != label["outcome"] or authority["result_evidence_sha256"] != label["rules_result_sha256"]
                    or snapshot["rules_anchor"] not in authority["anchors"]):
                raise ValueError("WDL anchor absent from independently recovered actual trajectory")
            outcome = {"schema": "rz-pals-rules-outcome-evidence/1", "game_id": authority["game_id"],
                       "ending": authority["ending"], "outcome": authority["outcome"], "rules_terminal": True}
            if evidence.get(authority["result_evidence_sha256"]) != outcome:
                raise ValueError("actual Rules termination evidence is missing or changed")

    @staticmethod
    def _join_evidence(input_sha256, head, target, receipts, *, record=None):
        content = {key: value for key, value in target.items() if key != "raw_evidence_sha256"}
        expected = {"schema": "rz-pals-generation-head-evidence/1", "input_sha256": input_sha256,
                    "head": head, "target": content}
        if (receipts.get(target["raw_evidence_sha256"]) != expected
                or digest(expected["schema"], expected) != target["raw_evidence_sha256"]):
            raise ValueError("target differs from independently recovered full head evidence")
        if head == "policy":
            _teacher_policy_evidence(record, target, receipts)
        if head == "task":
            for outcome in target["outcomes"]:
                # Only the bounded actual chunk reader can resolve a sealed
                # frozen-group alias. Legacy whole-JSON dictionaries retain
                # the same-input rule and never rewrite source receipts.
                origin = receipts.action_source_input(input_sha256, outcome) if hasattr(receipts, "action_source_input") else input_sha256
                causal = {"schema": "rz-pals-generation-causal-evidence/1", "input_sha256": origin,
                          "outcome": {key: value for key, value in outcome.items() if key != "causal_receipt_sha256"}}
                if receipts.get(outcome["causal_receipt_sha256"]) != causal:
                    raise ValueError("V action differs from recovered causal execution evidence")
                for witness in outcome["new_witnesses"]:
                    raw = {"schema": "rz-pals-generation-witness-evidence/1", "input_sha256": origin,
                           "witness": {key: value for key, value in witness.items() if key != "raw_evidence_sha256"},
                           "rules_validated": True}
                    if receipts.get(witness["raw_evidence_sha256"]) != raw:
                        raise ValueError("V witness differs from recovered Rules validation evidence")

    def _integrity(self):
        return digest("rz-pals-generation-dataset/1", [self._records, self._producer, self._authority, self._evidence, self._splits, self.exclusions])

    def rows(self, split, *, optimization=False):
        if split not in self._records or (optimization and split != "train") or self._seal != self._integrity():
            raise ValueError("immutable dataset/split authority mismatch")
        return copy.deepcopy(self._records[split])

    def admit(self, phase):
        return admit_phase(self.rows("train", optimization=True), phase)

    @property
    def sha256(self):
        if self._seal != self._integrity():
            raise ValueError("immutable actual dataset changed")
        return self._seal

    @property
    def authority(self):
        if self._seal != self._integrity():
            raise ValueError("immutable actual dataset changed")
        return copy.deepcopy(self._producer)

    @property
    def opening_splits(self):
        if self._seal != self._integrity():
            raise ValueError("immutable actual dataset changed")
        return copy.deepcopy(self._splits)


def strict_json(text):
    """Reject duplicate keys/nonfinite values before either canonical seal."""
    if type(text) is not str or len(text.encode()) > 16 * 1024 * 1024:
        raise ValueError("bounded generation JSON required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate generation JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite generation JSON: " + value)
    def finite_float(value):
        number = float(value)
        # Exponent overflow uses valid JSON number syntax and therefore never
        # invokes parse_constant. Reject it during decoding, including values
        # inside a raw nested runtime witness that is not a model tensor.
        if not math.isfinite(number):
            raise ValueError("nonfinite generation JSON number")
        return number
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=finite_float)
