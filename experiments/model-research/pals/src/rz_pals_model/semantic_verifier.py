"""CPU-only, frozen, training-private semantic V frontend.

Rules owns replay and all chess facts. This adapter conditionally admits actual
Rust request/receipt bytes against *independent caller pins*, the strict current
parent, and a prior result-free declaration. Caller registration, launch and
parameter observations are a trust boundary, not self-reported runtime proof.
No search, utility, target, optimizer, Warm seed or product V is created here.

The legacy query16 is derived unchanged by verifier_producer. Semantic hashes
bind bytes only. Board facts, ordered moves, promotion and question are the new
private features. A read-only alias to the existing model adds zero parameters
and zero state_dict entries. Private combined K/V never enters public P/C input.
The 11 MiB reservation covers this frontend's extra tensors/scratch, excluding
the existing checkpoint, PublicEncoder/RoleExpert workspace and caller RSS.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import time

import torch
from torch import nn

from .config import TASKS
from .model import PalsModel, TensorInput
from .preparation_check import _parameter_digest
from .training import TaskContext, ValidatedDataset, move_components
from .verifier_producer import eligible_tasks, private_query, verifier_input

SEMANTIC_SCHEMA = "rz-pals-private-semantic-branch/1"
IMPLEMENTATION = "rz-pals-rules-branch-preparation/1"
DESCRIPTOR_SCHEMA = "rz-pals-private-rules-descriptor/1"
MOVE_DOMAIN = "rz-pals-private-semantic-move-order/1"
MEANING_DOMAIN = "rz-pals-private-semantic-branch-meaning/1"
INPUT_DOMAIN = "rz-pals-private-v-semantic-input/1"
FRONTEND_DOMAIN = "rz-pals-private-v-semantic-frontend/1"
COMMON_SCHEMA = "rz-pals-private-v-common-query/1"
QUESTION_DOMAIN = "rz-pals-private-v-common-question/1"
BEFORE_SCHEMA = "rz-pals-private-v-semantic-before/1"
REGISTRATION_SCHEMA = "rz-pals-private-v-semantic-registration/1"
LAUNCH_SCHEMA = "rz-pals-private-v-semantic-launch/1"
PARAMETER_SCHEMA = "rz-pals-private-v-frozen-parameter-observation/1"
DECLARATION_SCOPE = "caller_declared_not_independently_registered"
BINARY_SCOPE = "dispatcher_compared_verified_argument"
ASSURANCE_SCOPE = "independently_pinned_caller_observation;rules_preparation_only;no_search_no_utility_proof"
MEANING_SCOPE = "rules_checked_branch_only;hashes_are_identity_not_neural_features;no_search_no_model_no_target_no_rank_no_utility_no_tactical_proof"
PIECES = "a1=0,h8=63;empty=0;white_PNBRQK=1..6;black_PNBRQK=7..12;exact_rules_piece_at"
CASTLING = "bit0:white_king;bit1:white_queen;bit2:black_king;bit3:black_queen;exact_rules_rights"
REPETITION = "rules_known_prefix_until_irreversible_boundary;not_inferred_from_fen_or_digest"
QUESTIONS = ("restricted_response", "continuation_challenge", "unrestricted_recheck")
PROMOTIONS = (None, "queen", "rook", "bishop", "knight")
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_SEMANTIC_BYTES = 11 * 1024 * 1024
TOKEN_COUNT = 1394
TILE = 64
_FACTORY = object()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(domain, value):
    return hashlib.sha256(canonical([domain, value])).hexdigest()


def byte_pin(raw):
    if type(raw) is not bytes:
        raise ValueError("immutable actual bytes required")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("lowercase SHA256 identity required")
    return value


def _int(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("finite integer bound")
    return value


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("exact semantic fields required")
    return value


def _parse(raw, maximum=MAX_INPUT_BYTES):
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum:
        raise ValueError("bounded immutable JSON bytes required")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _moves(value, maximum, unique=False):
    if type(value) is not list or len(value) > maximum:
        raise ValueError("ordered move extent")
    for move in value:
        _int(move, 0, 65535)
        move_components(move)
    if unique and len(set(value)) != len(value):
        raise ValueError("duplicate move")
    return value


def _tokens(tokens, moves, kind, legal=None):
    if type(tokens) is not list or len(tokens) != len(moves):
        raise ValueError("move token count")
    for slot, (token, move) in enumerate(zip(tokens, moves)):
        _fields(token, ("token_kind", "slot", "legal_order_slot", "move16", "from", "to", "promotion", "uci"))
        source, target, promotion = move_components(move)
        expected_slot = None if legal is None else legal.index(move)
        for key in ("slot", "move16", "from", "to"):
            _int(token[key], 0, 65535)
        if token["legal_order_slot"] is not None:
            _int(token["legal_order_slot"], 0, 255)
        uci = chr(97 + source % 8) + str(1 + source // 8) + chr(97 + target % 8) + str(1 + target // 8)
        uci += "" if promotion == 0 else ("q", "r", "b", "n")[promotion - 1]
        if token != {"token_kind": kind, "slot": slot, "legal_order_slot": expected_slot,
                     "move16": move, "from": source, "to": target,
                     "promotion": PROMOTIONS[promotion], "uci": uci}:
            raise ValueError("ordered Move16 meaning differs from actual tokens")


_DESCRIPTOR_FIELDS = ("schema", "rules_version", "rules_variant", "rules_state_sha256", "rules_history_sha256",
                     "board_fen", "side_to_move", "board64_piece_codes", "piece_code_semantics", "pals_rules_encoding",
                     "castling_rights", "castling_bit_semantics", "en_passant_square", "halfmove_clock", "fullmove_number",
                     "in_check", "history_completeness", "history_origin", "known_history_positions", "known_repetition_count",
                     "repetition_history_complete", "repetition_scope", "legal_moves", "legal_order_sha256", "legal_tokens",
                     "play_status", "terminal_reason", "terminal_winner", "terminal_source")


def _descriptor(value, kind, registration):
    _fields(value, _DESCRIPTOR_FIELDS)
    if (value["schema"] != DESCRIPTOR_SCHEMA or value["rules_version"] != registration["rules_version"]
            or value["rules_variant"] != "standard_chess" or value["pals_rules_encoding"] != "rz-pals-rules-fields-v1"
            or value["piece_code_semantics"] != PIECES or value["castling_bit_semantics"] != CASTLING
            or value["repetition_scope"] != REPETITION):
        raise ValueError("unsupported actual Rules descriptor profile")
    for key in ("rules_state_sha256", "rules_history_sha256", "legal_order_sha256"):
        _sha(value[key])
    if type(value["board64_piece_codes"]) is not list or len(value["board64_piece_codes"]) != 64:
        raise ValueError("exact board64 required")
    for piece in value["board64_piece_codes"]:
        _int(piece, 0, 12)
    if value["side_to_move"] not in ("white", "black"):
        raise ValueError("unknown turn")
    _int(value["castling_rights"], 0, 15)
    if value["en_passant_square"] is not None:
        _int(value["en_passant_square"], 0, 63)
    _int(value["halfmove_clock"], 0, 2**32 - 1)
    _int(value["fullmove_number"], 1, 2**32 - 1)
    _int(value["known_history_positions"], 1, 2**32 - 1)
    _int(value["known_repetition_count"], 1, 2**32 - 1)
    if value["known_repetition_count"] > value["known_history_positions"]:
        raise ValueError("known repetition count exceeds actual known history")
    if (type(value["in_check"]) is not bool or type(value["repetition_history_complete"]) is not bool
            or value["history_completeness"] not in ("complete", "unknown_prefix")
            or value["history_origin"] not in ("start_position", "fen")):
        raise ValueError("explicit history facts required")
    if type(value["board_fen"]) is not str or not 1 <= len(value["board_fen"].encode()) <= 512:
        raise ValueError("bounded FEN identity required")
    _moves(value["legal_moves"], 256, unique=True)
    if value["legal_order_sha256"] != digest(MOVE_DOMAIN, value["legal_moves"]):
        raise ValueError("full legal order seal")
    _tokens(value["legal_tokens"], value["legal_moves"], kind, value["legal_moves"])
    if value["play_status"] == "ongoing":
        if any(value[name] is not None for name in ("terminal_reason", "terminal_winner", "terminal_source")):
            raise ValueError("ongoing descriptor has terminal facts")
    elif value["play_status"] == "rules_terminal":
        if (type(value["terminal_reason"]) is not str or not 1 <= len(value["terminal_reason"]) <= 128
                or value["terminal_source"] != "rz-position-rules" or value["terminal_winner"] not in (None, "white", "black")):
            raise ValueError("Rules terminal provenance")
    else:
        raise ValueError("unknown play status")


def _same_state(left, right):
    # Identity comparison is only for reuse; never reverse-engineer chess facts
    # from FEN/digests. Root/target legal_tokens have different token-kind roles.
    return all(left[key] == right[key] for key in _DESCRIPTOR_FIELDS if key != "legal_tokens")


def _validated(parent, parent_index, raws, pins):
    if type(parent) is not ValidatedDataset:
        raise ValueError("strict checked parent type required")
    parent._verify_raw_integrity()
    if parent.frozen_admission is None or parent._frozen_admission_identity is None:
        raise ValueError("metadata-only parent cannot admit semantic V")
    _int(parent_index, 0, len(parent.records) - 1)
    if parent_index not in parent.current_view.current_indices:
        raise ValueError("parent is not actual current input")
    row = parent.records[parent_index]
    snapshot = row["input"]["snapshot"]
    if snapshot["role"] not in ("proposer", "critic"):
        raise ValueError("current P/C parent required")
    encoding = parent.encodings.get(row["input"]["sha256"])
    if encoding is None:
        raise ValueError("actual parent encoding required")
    _fields(pins, (*raws, "parent_input_sha256", "current_view_sha256", "frozen_admission_sha256", "encoding_sha256"))
    if sum(len(raw) for raw in raws.values()) > MAX_INPUT_BYTES:
        raise ValueError("aggregate semantic raw input bound")
    for name, raw in raws.items():
        _fields(pins[name], ("bytes", "sha256"))
        if byte_pin(raw) != pins[name]:
            raise ValueError("actual semantic bytes differ from independent expected pin: " + name)
    if (row["input"]["sha256"] != _sha(pins["parent_input_sha256"])
            or parent.current_view.sha256 != _sha(pins["current_view_sha256"])
            or parent._frozen_admission_identity != _sha(pins["frozen_admission_sha256"])
            or encoding.encoding_sha256 != _sha(pins["encoding_sha256"])):
        raise ValueError("strict actual parent/current/encoding admission mismatch")
    registration = _parse(raws["registration"])
    _fields(registration, ("schema", "semantic_schema", "implementation", "source_artifact", "binary_sha256",
                           "rules_version", "platform", "accepted_binary_pin_scope", "assurance_scope"))
    if (registration["schema"] != REGISTRATION_SCHEMA or registration["semantic_schema"] != SEMANTIC_SCHEMA
            or registration["implementation"] != IMPLEMENTATION or registration["assurance_scope"] != "independently_registered_caller_source"
            or registration["source_artifact"] != byte_pin(raws["source"])):
        raise ValueError("independently registered actual Rust source binding")
    _sha(registration["binary_sha256"])
    if type(registration["rules_version"]) is not str or not 1 <= len(registration["rules_version"]) <= 128:
        raise ValueError("registered Rules version required")
    scopes = {"linux": "linux_loaded_executable_inode", "windows": "current_exe_path_hash", "macos": "current_exe_path_hash"}
    if scopes.get(registration["platform"]) != registration["accepted_binary_pin_scope"]:
        raise ValueError("platform-specific caller binary assurance")
    common = _parse(raws["common_query"])
    _fields(common, ("schema", "parent_input_sha256", "current_view_sha256", "question", "prefix", "root_moves", "claimed_line",
                     "allowed_tasks", "baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms",
                     "budget_bucket", "cpu_profile_sha256", "known_completed_depth"))
    if (common["schema"] != COMMON_SCHEMA or common["parent_input_sha256"] != row["input"]["sha256"]
            or common["current_view_sha256"] != parent.current_view.sha256 or common["question"] not in QUESTIONS):
        raise ValueError("result-free common question/current parent binding")
    for name, cap in (("prefix", 64), ("root_moves", 256), ("claimed_line", 64)):
        _moves(common[name], cap, name == "root_moves")
    _int(common["baseline_depth"], 0, 64)
    _int(common["requested_depth"], 1, 64)
    _int(common["known_completed_depth"], 0, 64)
    if common["baseline_depth"] > common["requested_depth"]:
        raise ValueError("fixed horizon below baseline")
    _int(common["max_nodes_per_check"], 1, 2**32 - 1)
    _int(common["max_wall_time_ms"], 1, 300000)
    _int(common["budget_bucket"], 0, 16)
    _sha(common["cpu_profile_sha256"])
    question_body = {key: common[key] for key in ("parent_input_sha256", "question", "prefix", "root_moves", "claimed_line")}
    context = TaskContext(digest(QUESTION_DOMAIN, question_body), common["cpu_profile_sha256"], common["budget_bucket"])
    context.validate()
    mask, _ = eligible_tasks(common["allowed_tasks"], {"prefix": common["prefix"], "root_moves": common["root_moves"]}, snapshot["legal_moves"])
    query = private_query(mask, snapshot, **{key: common[key] for key in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "budget_bucket")})
    before = _parse(raws["before_result"])
    if before != {"schema": BEFORE_SCHEMA, "parent_input_sha256": row["input"]["sha256"],
                  "current_view_sha256": parent.current_view.sha256, "common_query_sha256": byte_pin(raws["common_query"])["sha256"],
                  "registration_sha256": byte_pin(raws["registration"])["sha256"]}:
        raise ValueError("prior result-free declaration differs from common question")
    request = _parse(raws["request"], 512 * 1024)
    _fields(request, ("schema", "question", "parent_input_sha256", "before_result_anchor_sha256", "position_command",
                      "expected_board_fen", "rules_state_sha256", "rules_history_sha256", "prefix", "root_moves", "claimed_line",
                      "current_binary_sha256", "max_wall_time_ms", "max_output_bytes", "context_sha256"))
    _int(request["max_wall_time_ms"], 1, 300000)
    _int(request["max_output_bytes"], 1024, 1024 * 1024)
    if (request["schema"] != SEMANTIC_SCHEMA or request["context_sha256"] != digest(SEMANTIC_SCHEMA, {key: value for key, value in request.items() if key != "context_sha256"})
            or request["current_binary_sha256"] != registration["binary_sha256"]
            or request["before_result_anchor_sha256"] != byte_pin(raws["before_result"])["sha256"]
            or request["expected_board_fen"] != snapshot["board_fen"] or request["position_command"] != snapshot["position_command"]
            or any(request[key] != snapshot[key] for key in ("rules_state_sha256", "rules_history_sha256"))
            or any(request[key] != common[key] for key in ("question", "parent_input_sha256", "prefix", "root_moves", "claimed_line"))
            or request["max_wall_time_ms"] > common["max_wall_time_ms"]):
        raise ValueError("actual request differs from sealed current parent/common declaration")
    if bool(request["root_moves"]) != (request["question"] == "restricted_response"):
        raise ValueError("question root-restriction mismatch")
    receipt = _parse(raws["receipt"], request["max_output_bytes"])
    _fields(receipt, ("schema", "implementation", "context_sha256", "current_binary_sha256", "binary_pin_scope", "parent_input_sha256",
                     "before_result_anchor_sha256", "caller_declaration_scope", "meaning_scope", "question", "root", "prefix", "target",
                     "root_restriction", "claimed_line", "branch_meaning_sha256", "resource_policy", "cpu_checks", "search_executed",
                     "model_executed", "training_target_created", "product_verifier_enabled", "elapsed_ms", "deadline_exceeded"))
    if (receipt["schema"] != SEMANTIC_SCHEMA or receipt["implementation"] != IMPLEMENTATION
            or receipt["binary_pin_scope"] != registration["accepted_binary_pin_scope"] or receipt["caller_declaration_scope"] != DECLARATION_SCOPE
            or receipt["meaning_scope"] != MEANING_SCOPE
            or any(receipt[key] != request[key] for key in ("context_sha256", "current_binary_sha256", "parent_input_sha256", "before_result_anchor_sha256", "question"))
            or type(receipt["cpu_checks"]) is not int or receipt["cpu_checks"] != 0
            or any(receipt[key] is not False for key in ("search_executed", "model_executed", "training_target_created", "product_verifier_enabled", "deadline_exceeded"))):
        raise ValueError("only successful Rules-only preparation receipt admitted")
    if canonical(receipt["resource_policy"]) != canonical({"max_wall_time_ms": request["max_wall_time_ms"], "max_output_bytes": request["max_output_bytes"],
                                     "max_prefix_plies": 64, "max_claim_plies": 64, "max_root_moves": 256, "cpu_checks": 0}):
        raise ValueError("actual preparation resource policy mismatch")
    _int(receipt["elapsed_ms"], 0, request["max_wall_time_ms"] - 1)
    _descriptor(receipt["root"], "root_legal", registration)
    _descriptor(receipt["target"], "target_legal", registration)
    root = receipt["root"]
    if (any(root[key] != snapshot[key] for key in ("rules_state_sha256", "rules_history_sha256", "board_fen", "legal_moves"))
            or root["side_to_move"] != ("white" if snapshot["white_to_move"] else "black")
            or tuple(root["board64_piece_codes"]) != tuple(encoding.board)):
        raise ValueError("actual root differs from checked parent Rules/public board")
    _tokens(receipt["prefix"], request["prefix"], "prefix")
    if not request["prefix"] and not _same_state(root, receipt["target"]):
        raise ValueError("empty prefix changed actual state/history/order")
    restriction = receipt["root_restriction"]
    if request["root_moves"]:
        _fields(restriction, ("declared_order", "effective_legal_order", "declared_order_sha256", "effective_order_sha256", "scope"))
        legal = receipt["target"]["legal_moves"]
        if receipt["target"]["play_status"] != "ongoing" or any(move not in legal for move in request["root_moves"]):
            raise ValueError("restriction escapes actual target legal order")
        effective = [move for move in legal if move in request["root_moves"]]
        _tokens(restriction["declared_order"], request["root_moves"], "root_restriction", legal)
        _tokens(restriction["effective_legal_order"], effective, "effective_root_legal", legal)
        if (restriction["declared_order_sha256"] != digest(MOVE_DOMAIN, request["root_moves"])
                or restriction["effective_order_sha256"] != digest(MOVE_DOMAIN, effective)
                or restriction["scope"] != "exact_target_root_only;request_order_preserved;effective_order_is_rules_order"):
            raise ValueError("restriction order binding")
    elif restriction is not None:
        raise ValueError("unused restriction")
    claim = _fields(receipt["claimed_line"], ("status", "legality_verified", "claim_truth", "movements", "restriction_checked", "final_state"))
    _tokens(claim["movements"], request["claimed_line"], "claimed_continuation")
    if claim["claim_truth"] != "unknown":
        raise ValueError("Rules replay cannot claim tactical truth")
    if request["claimed_line"]:
        if (claim["status"] != "legal_continuation" or claim["legality_verified"] is not True
                or claim["restriction_checked"] is not bool(request["root_moves"])):
            raise ValueError("claimed continuation legality/coverage")
        if request["root_moves"] and request["claimed_line"][0] not in request["root_moves"]:
            raise ValueError("claim escapes restriction")
        _descriptor(claim["final_state"], "claim_end_legal", registration)
    elif (claim["status"] != "no_claim" or claim["legality_verified"] is not None
          or claim["restriction_checked"] is not False or claim["final_state"] is not None):
        raise ValueError("no_claim must remain unknown and masked")
    meaning = {key: receipt[key] for key in ("question", "root", "prefix", "target", "root_restriction", "claimed_line")}
    if receipt["branch_meaning_sha256"] != digest(MEANING_DOMAIN, meaning):
        raise ValueError("complete actual branch meaning seal")
    launch = _parse(raws["launch_observation"])
    _fields(launch, ("schema", "registration_sha256", "request", "receipt", "binary_sha256", "platform", "binary_pin_scope",
                    "before_result_anchor_sha256", "assurance_scope", "anchor_durable_before_spawn", "spawned", "reaped",
                    "exit_code", "elapsed_ms", "timed_out"))
    if (launch["schema"] != LAUNCH_SCHEMA or launch["assurance_scope"] != "independently_pinned_caller_observation"
            or launch["registration_sha256"] != byte_pin(raws["registration"])["sha256"]
            or launch["request"] != byte_pin(raws["request"]) or launch["receipt"] != byte_pin(raws["receipt"])
            or any(launch[key] != registration[key] for key in ("binary_sha256", "platform"))
            or launch["binary_pin_scope"] != registration["accepted_binary_pin_scope"]
            or launch["before_result_anchor_sha256"] != request["before_result_anchor_sha256"]
            or any(launch[key] is not True for key in ("anchor_durable_before_spawn", "spawned", "reaped"))
            or type(launch["exit_code"]) is not int or launch["exit_code"] != 0 or launch["timed_out"] is not False):
        raise ValueError("independent caller launch observation mismatch")
    _int(launch["elapsed_ms"], receipt["elapsed_ms"], common["max_wall_time_ms"] - 1)
    # This is the exact legacy derived snapshot/encoding seam. Semantic input
    # receives a separate encoding identity; public features remain identical.
    encoding_sha = digest(FRONTEND_DOMAIN, {"parent_encoding_sha256": encoding.encoding_sha256,
                                          "common_query_sha256": byte_pin(raws["common_query"])["sha256"],
                                          "branch_meaning_sha256": receipt["branch_meaning_sha256"]})
    derived, prepared = verifier_input(row, encoding, snapshot["source"], context, query, encoding_sha, snapshot["frozen_epoch"])
    view = {**common, "common_query_sha256": byte_pin(raws["common_query"])["sha256"], "common_question_sha256": context.branch_sha256,
            "context": {"branch_sha256": context.branch_sha256, "cpu_profile_sha256": context.cpu_profile_sha256, "budget_bucket": context.budget_bucket},
            "query": list(query), "eligible_tasks": mask, "semantic_context_sha256": request["context_sha256"],
            "branch_meaning_sha256": receipt["branch_meaning_sha256"], "before_result_anchor_sha256": request["before_result_anchor_sha256"],
            "derived_input_sha256": derived["input"]["sha256"], "assurance_scope": ASSURANCE_SCOPE}
    return receipt, view, derived, prepared


class CheckedSemanticInput:
    """Factory-only immutable bytes capability; Python is not a hostile sandbox.

    verify() rechecks the full raw/caller/parent binding. Returned dictionaries
    are copies. Neither arbitrary callbacks nor receipt-only self-seals enroll
    this capability. Independent caller pins remain an external trust boundary.
    """
    __slots__ = ("_parent", "_index", "_raws", "_pins", "_identity", "_locked")

    def __init__(self, token=None, *, parent=None, index=None, raws=None, pins=None):
        if token is not _FACTORY:
            raise ValueError("use admit_semantic_input; unchecked constructor refused")
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_raws", tuple(sorted(raws.items())))
        object.__setattr__(self, "_pins", canonical(pins))
        object.__setattr__(self, "_identity", digest(INPUT_DOMAIN, pins))
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name, value):
        raise AttributeError("checked semantic capability is immutable")

    @property
    def sha256(self):
        self.verify()
        return self._identity

    def _views(self):
        pins = _parse(self._pins)
        if digest(INPUT_DOMAIN, pins) != self._identity:
            raise ValueError("semantic capability identity changed")
        return _validated(self._parent, self._index, dict(self._raws), pins)

    def verify(self):
        self._views()
        return self

    def parent_pins(self):
        self.verify()
        return {key: value for key, value in _parse(self._pins).items()
                if key in ("parent_input_sha256", "current_view_sha256", "frozen_admission_sha256", "encoding_sha256")}

    def verify_parent(self, parent):
        self.verify()
        if type(parent) is not ValidatedDataset:
            raise ValueError("strict actual parent type required")
        parent._verify_raw_integrity()
        pins = self.parent_pins()
        if (parent._frozen_admission_identity != pins["frozen_admission_sha256"]
                or parent.current_view.sha256 != pins["current_view_sha256"]):
            raise ValueError("semantic capability belongs to another strict parent admission")
        matches = [index for index in parent.current_view.current_indices
                   if parent.records[index]["input"]["sha256"] == pins["parent_input_sha256"]]
        if len(matches) != 1 or parent.encodings[pins["parent_input_sha256"]].encoding_sha256 != pins["encoding_sha256"]:
            raise ValueError("semantic current input/encoding is not unique")
        return matches[0]

    def common_query(self):
        return copy.deepcopy(self._views()[1])

    def derived_input(self):
        return copy.deepcopy(self._views()[2])

    def encoded_snapshot(self):
        return self._views()[3]  # frozen EncodedSnapshot of immutable tuples

    def rules_receipt(self):
        return copy.deepcopy(self._views()[0])


def admit_semantic_input(*, parent, parent_index, request_bytes, receipt_bytes, source_bytes,
                         registration_bytes, before_result_bytes, launch_observation_bytes,
                         common_query_bytes, expected_pins):
    raws = {"request": request_bytes, "receipt": receipt_bytes, "source": source_bytes,
            "registration": registration_bytes, "before_result": before_result_bytes,
            "launch_observation": launch_observation_bytes, "common_query": common_query_bytes}
    pins = _parse(canonical(expected_pins))
    _validated(parent, parent_index, raws, pins)
    return CheckedSemanticInput(_FACTORY, parent=parent, index=parent_index, raws=raws, pins=pins)


@dataclass(frozen=True)
class StateFeatures:
    board: tuple
    fen: tuple
    history: tuple
    legal: tuple


def prepare_state_features(descriptor):
    """Read actual structured fields only; reusable after descriptor admission."""
    ep = descriptor["en_passant_square"]
    half, full, count = (descriptor[key] for key in ("halfmove_clock", "fullmove_number", "known_repetition_count"))
    split = lambda value: (float(value & 65535) / 65535, float(value >> 16) / 65535)
    rights = descriptor["castling_rights"]
    fen = (float(descriptor["side_to_move"] == "white"), *(float(bool(rights & (1 << bit))) for bit in range(4)),
           float(ep is not None), 0.0 if ep is None else ep % 8 / 7, 0.0 if ep is None else ep // 8 / 7,
           *split(half), *split(full))
    history = (float(descriptor["history_completeness"] == "complete"), float(descriptor["history_completeness"] == "unknown_prefix"), *split(count))
    return StateFeatures(tuple(descriptor["board64_piece_codes"]), fen, history,
                         tuple(tuple(move_components(move)) for move in descriptor["legal_moves"]))


@dataclass(frozen=True)
class SemanticFeatures:
    states: tuple
    prefix: tuple
    restriction: tuple
    claim: tuple
    question: int
    claim_mode: int
    known_completed_depth: int


def semantic_features(checked):
    if type(checked) is not CheckedSemanticInput:
        raise ValueError("checked semantic input type required")
    receipt = checked.rules_receipt()
    common = checked.common_query()
    descriptors = (receipt["root"], receipt["target"], receipt["claimed_line"]["final_state"])
    states, seen = [], []
    for descriptor in descriptors:
        if descriptor is None:
            states.append(None)
            continue
        previous = next((feature for old, feature in seen if _same_state(old, descriptor)), None)
        feature = previous if previous is not None else prepare_state_features(descriptor)
        states.append(feature)
        seen.append((descriptor, feature))
    restriction = receipt["root_restriction"]
    moves = lambda tokens: tuple(tuple(move_components(token["move16"])) for token in tokens)
    return SemanticFeatures(tuple(states), moves(receipt["prefix"]), () if restriction is None else moves(restriction["declared_order"]),
                            moves(receipt["claimed_line"]["movements"]), QUESTIONS.index(receipt["question"]),
                            int(bool(receipt["claimed_line"]["movements"])), common["known_completed_depth"])


def _segments(features):
    # Prior coverage is a separately pinned before-result declaration. It is
    # never taken from this query's future checker result/utility/target.
    yield 0, "question", ((features.question, features.known_completed_depth),)
    yield 1, "claim_mode", (features.claim_mode,)
    for state_index, state in enumerate(features.states):
        for offset, kind, capacity in ((0, "board", 64), (1, "fen", 12), (2, "history", 4), (3, "move", 256)):
            values = () if state is None else getattr(state, ("board", "fen", "history", "legal")[offset])
            yield 2 + state_index * 4 + offset, kind, values
    yield 14, "move", features.prefix
    yield 15, "move", features.restriction
    yield 16, "move", features.claim


_CAPACITIES = (1, 1, 64, 12, 4, 256, 64, 12, 4, 256, 64, 12, 4, 256, 64, 256, 64)


def token_descriptors(features):
    """Immutable typed token views; inactive padding has no payload/basis."""
    for segment, kind, values in _segments(features):
        for position in range(_CAPACITIES[segment]):
            yield segment, position, kind, (values[position] if position < len(values) else None)


def reservation_bytes(batch, public_tokens, limit=MAX_SEMANTIC_BYTES):
    _int(batch, 1, 4)
    _int(public_tokens, 66, 194)
    _int(limit, 1, MAX_SEMANTIC_BYTES)
    # Combined K/V, tile and projections, bounded per-row MoveEmbedding scratch,
    # masks, typed inputs/views and allocator-independent extra scratch reserve.
    needed = 2 * batch * 2 * (public_tokens + TOKEN_COUNT) * 64 * 4
    needed += batch * TILE * (384 + 2 * 128) * 4
    needed += 2 * 1024 * 1024 + batch * (public_tokens + TOKEN_COUNT)
    if needed > limit:
        raise ValueError("semantic owner byte reservation denied before allocation")
    return needed


def _deadline(deadline):
    if type(deadline) not in (float, int) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("original caller absolute deadline expired")


def _tile(model, descriptors):
    """Project a finite tile without allocating a dense 1394x384 token bank."""
    batch, length = len(descriptors), len(descriptors[0])
    payload = torch.zeros((batch, length, 384), dtype=torch.float32)
    mask = torch.zeros((batch, length), dtype=torch.bool)
    for row, tokens in enumerate(descriptors):
        board_slots, board_values, move_slots, move_values = [], [], [], []
        for slot, (segment, position, kind, value) in enumerate(tokens):
            if value is None:
                continue
            mask[row, slot] = True
            payload[row, slot, position] += 1.0
            payload[row, slot, 256 + segment] += 1.0
            if kind == "board":
                board_slots.append(slot)
                board_values.append((value, position))
            elif kind == "move":
                move_slots.append(slot)
                move_values.append(value)
            elif kind in ("fen", "history"):
                payload[row, slot, 280 + position + (12 if kind == "history" else 0)] += value
            elif kind == "question":
                question, prior_depth = value
                payload[row, slot, 304 + question] += 1.0
                payload[row, slot, 310] += prior_depth / 64
            else:
                payload[row, slot, 307 + value] += 1.0
        if board_slots:
            values = torch.tensor(board_values, dtype=torch.int64)
            vectors = model.public_encoder.piece_embedding(values[:, 0]) + model.public_encoder.square_embedding[values[:, 1]]
            payload[row, board_slots] += vectors
            del values, vectors
        if move_slots:
            values = torch.tensor(move_values, dtype=torch.int64).unsqueeze(0)
            vectors = model.move_embedding(values)[0]
            payload[row, move_slots] += vectors
            del values, vectors
    return payload, mask


@dataclass(frozen=True)
class PrivateSemanticMemory:
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor
    reserved_bytes: int


def _private_memory(model, public, features, *, deadline, byte_limit):
    key, value, public_mask = public
    batch, _, public_tokens, _ = key.shape
    reserved = reservation_bytes(batch, public_tokens, byte_limit)
    _deadline(deadline)
    combined_key = torch.empty((batch, 2, public_tokens + TOKEN_COUNT, 64), dtype=torch.float32)
    combined_value = torch.empty_like(combined_key)
    combined_mask = torch.zeros((batch, public_tokens + TOKEN_COUNT), dtype=torch.bool)
    combined_key[:, :, :public_tokens].copy_(key)
    combined_value[:, :, :public_tokens].copy_(value)
    combined_mask[:, :public_tokens].copy_(public_mask)
    streams = [iter(token_descriptors(feature)) for feature in features]
    for start in range(0, TOKEN_COUNT, TILE):
        _deadline(deadline)
        length = min(TILE, TOKEN_COUNT - start)
        descriptors = [tuple(next(stream) for _ in range(length)) for stream in streams]
        payload, mask = _tile(model, descriptors)
        for target, projection in ((combined_key, model.public_encoder.memory_key), (combined_value, model.public_encoder.memory_value)):
            projected = projection(payload).view(batch, length, 2, 64).transpose(1, 2)
            target[:, :, public_tokens + start:public_tokens + start + length].copy_(projected)
            del projected
        combined_mask[:, public_tokens + start:public_tokens + start + length].copy_(mask)
        del payload, mask, descriptors
    _deadline(deadline)
    return PrivateSemanticMemory(combined_key, combined_value, combined_mask, reserved)


def _tensor_inputs(checked_inputs):
    encodings = [checked.encoded_snapshot() for checked in checked_inputs]
    receipts = [checked.rules_receipt() for checked in checked_inputs]
    batch = len(encodings)
    records = max(1, max(len(value.public_records) for value in encodings))
    candidates = max(1, max(len(receipt["root"]["legal_moves"]) for receipt in receipts))
    board = torch.tensor([value.board for value in encodings], dtype=torch.int64)
    metadata = torch.tensor([value.metadata for value in encodings], dtype=torch.float32)
    public_records = torch.zeros((batch, records, 16), dtype=torch.float32)
    record_mask = torch.zeros((batch, records), dtype=torch.bool)
    moves = torch.zeros((batch, candidates, 3), dtype=torch.int64)
    move_mask = torch.zeros((batch, candidates), dtype=torch.bool)
    for row, encoding in enumerate(encodings):
        for slot, (_, _, features) in enumerate(encoding.public_records):
            public_records[row, slot] = torch.tensor(features, dtype=torch.float32)
            record_mask[row, slot] = True
        for slot, move in enumerate(receipts[row]["root"]["legal_moves"]):
            moves[row, slot] = torch.tensor(move_components(move), dtype=torch.int64)
            move_mask[row, slot] = True
    inputs = TensorInput(board, metadata, public_records, record_mask, moves, move_mask,
                         torch.zeros((batch, 1, 8)), torch.zeros((batch, 1), dtype=torch.bool),
                         torch.tensor([value.query for value in encodings], dtype=torch.float32))
    inputs.validate()
    return inputs


class FrozenSemanticVerifier(nn.Module):
    """Parameter-free read-only model alias. state_dict() is deliberately empty.

    Existing weights remain exclusively owned and pinned by the caller's
    PalsModel. This wrapper has no export/serialization/checkpoint API. A caller
    observation pins the actual parameter digest separately from checkpoint
    bytes; its independence and actual load chronology are caller assurances.
    """
    def __init__(self, model, *, parameter_observation_bytes, expected_observation_sha256):
        super().__init__()
        if type(model) is not PalsModel or "validator" not in model.experts:
            raise ValueError("existing frozen validator required; no fallback or new expert")
        model.config.validate()
        observation = _parse(parameter_observation_bytes, 8192)
        _fields(observation, ("schema", "checkpoint_sha256", "parameter_sha256", "assurance_scope"))
        if (byte_pin(parameter_observation_bytes)["sha256"] != _sha(expected_observation_sha256)
                or observation["schema"] != PARAMETER_SCHEMA
                or observation["assurance_scope"] != "independently_pinned_caller_parameter_observation"):
            raise ValueError("independent caller parameter observation required")
        _sha(observation["checkpoint_sha256"])
        _sha(observation["parameter_sha256"])
        object.__setattr__(self, "_model", model)  # not registered as an nn.Module alias
        self._observation_bytes = parameter_observation_bytes
        self._observation_sha = expected_observation_sha256
        self._parameter_sha = observation["parameter_sha256"]
        self._check_model()

    def _check_model(self):
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("semantic first profile refuses active autocast")
        model = self._model
        if model.training or any(module.training for module in model.modules()):
            raise ValueError("caller must explicitly set frozen model eval mode")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("caller must freeze existing weights; frontend never changes parameter flags")
        if (_parameter_digest(model) != self._parameter_sha
                or byte_pin(self._observation_bytes)["sha256"] != self._observation_sha):
            raise ValueError("actual frozen CPU FP32 parameter bytes changed")

    def forward(self, checked_inputs, *, deadline, byte_limit=MAX_SEMANTIC_BYTES):
        _deadline(deadline)
        if type(checked_inputs) not in (tuple, list) or not 1 <= len(checked_inputs) <= 4:
            raise ValueError("CPU semantic batch must be 1..4")
        if any(type(value) is not CheckedSemanticInput for value in checked_inputs):
            raise ValueError("unchecked semantic capability refused")
        for checked in checked_inputs:
            checked.verify()
        # Reserve before even collating typed inputs/public encoder execution.
        public_tokens = 66 + max(1, max(len(checked.encoded_snapshot().public_records) for checked in checked_inputs))
        reservation_bytes(len(checked_inputs), public_tokens, byte_limit)
        self._check_model()
        _deadline(deadline)
        with torch.no_grad():
            inputs = _tensor_inputs(checked_inputs)
            features = tuple(semantic_features(checked) for checked in checked_inputs)
            public = self._model.public_encoder(*inputs.public_args())
            memory = _private_memory(self._model, public, features, deadline=deadline, byte_limit=byte_limit)
            outputs = self._model.experts["validator"](
                self._model.reader_blocks, self._model.move_embedding, memory.key, memory.value, memory.mask,
                inputs.candidates, inputs.candidate_mask, inputs.divergence_features, inputs.divergence_mask, inputs.query)
            logits, latent = outputs[3], outputs[2]
            if (logits.shape != (len(checked_inputs), len(TASKS)) or logits.dtype != torch.float32 or logits.device.type != "cpu"
                    or latent.shape != (len(checked_inputs), 16, 384) or not torch.all(torch.isfinite(logits))
                    or not torch.all(torch.isfinite(latent))):
                raise ValueError("finite existing seven-task V outputs required")
        self._check_model()
        for checked in checked_inputs:
            checked.verify()
        _deadline(deadline)
        # No policy/value targets, preferences, ranks or cache publication.
        return logits, latent, {"schema": FRONTEND_DOMAIN, "input_sha256": tuple(value.sha256 for value in checked_inputs),
                               "semantic_owner_reserved_bytes": memory.reserved_bytes, "private_tokens": TOKEN_COUNT,
                               "parameter_sha256": self._parameter_sha, "assurance_scope": ASSURANCE_SCOPE,
                               "feature_coverage": "structured_board_turn_castling_ep_split_counters_known_repetition_count_history_completeness_full_legal_order_prefix_declared_restriction_claim_line_question_claim_presence_before_declared_known_completed_depth;claim_truth_unknown;no_claim_end_masked;history_origin_in_check_terminal_facts_repetition_scope_are_admission_only",
                               "actual_training_executed": False, "product_verifier_enabled": False}
