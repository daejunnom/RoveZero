"""Dependency-free V3 selected-review admission and physical tensor packing.

This module consumes an already Rules-checked view. It never reconstructs a
game, scans raw RoleRecords, imports Torch, or treats a digest as an NN feature.
Every query sequence is charged, including a duplicate of a record sequence.
"""
import hashlib
import json
import math
import struct

from .config import ENCODING_V3, MODEL_SCHEMA_V3, MODEL_SEMANTICS_V3, ROLES

INPUT_SCHEMA = "rovezero.pals-virtual-review-input.v1"
BUCKETS = (8, 16, 32, 64, 128, 256)
KINDS = ("proposal", "counterexample", "repair", "cpu_verification")
PURPOSES = ("propose_policy", "reply_policy", "repair_policy", "divergence_policy", "value_fresh", "task_policy")
RECORD_FEATURES = ("kind.proposal", "kind.counterexample", "kind.repair", "kind.cpu_verification",
                   "perspective", "completed_iteration", "rules_terminal", "line_nonempty")
QUERY_FEATURES = (*("purpose." + purpose for purpose in PURPOSES), "perspective", "counter_present")
MAX_RECORDS, MAX_ANCHORS, MAX_CANDIDATES = 64, 4, 256

# Actual input ABI/projection registry, independent from model parameter
# semantics, Rules descriptor feature meaning and source implementation bytes.
# Rust virtual_review::ENCODING_REGISTRY_FIELDS frames these exact same fields.
ENCODING_REGISTRY_FIELDS = (
    "rz-pals-virtual-review-encoding-registry/1",
    "rovezero.pals-virtual-review.v1",
    "batch=1;records=0..64;anchors=1..4;root_anchor=0;candidates=0..256;divergences=0..128",
    "L=min_bucket(max_record_and_query_line_len);buckets=8,16,32,64,128,256;one_dynamic_graph",
    "token_budget=128|256|512;charge_records_and_prefix_proposal_counter_including_duplicates",
    "anchor_board:i64[B,A,64];piece_codes=0..12;a1_to_h8",
    "anchor_metadata:f32[B,A,16];Rules_metadata_meaning_is_separately_pinned",
    "anchor_mask:bool[B,A];all_typed_anchors_active;root0_required",
    "records:f32[B,max(R,1),8]",
    "record_features=kind.proposal,kind.counterexample,kind.repair,kind.cpu_verification,perspective,completed_iteration,rules_terminal,line_nonempty",
    "record_anchor:i64[B,max(R,1)];selected_anchor_local_index",
    "record_line_tokens:i64[B,max(R,1),L,3];from,to,promotion",
    "record_line_mask:bool[B,max(R,1),L];true_for_exact_line_prefix_only",
    "record_mask:bool[B,max(R,1)];true_for_actual_records_only",
    "record_relations:i64[B,max(R,1),2];parent,supersedes;absent=-1;local_DAG;no_self_edges",
    "query:f32[B,8]",
    "query_features=purpose.propose_policy,purpose.reply_policy,purpose.repair_policy,purpose.divergence_policy,purpose.value_fresh,purpose.task_policy,perspective,counter_present",
    "query_anchor:i64[B,3];prefix,proposal,counter_local_anchor_including_empty_sequences",
    "query_line_tokens:i64[B,3,L,3];prefix,proposal,counter;from,to,promotion",
    "query_line_mask:bool[B,3,L];true_for_exact_sequence_prefix_only",
    "candidates:i64[B,max(C,1),3];caller_legal_order;no_duplicates",
    "candidate_mask:bool[B,max(C,1)];true_for_exact_candidates_only",
    "divergence_features:f32[B,max(D,1),8];critic_only;raw_observation_vocabulary_separately_pinned",
    "divergence_mask:bool[B,max(D,1)];true_for_actual_critic_divergences_only",
    "moves=from0..63,to0..63,from!=to;promotion=0_none,1_queen,2_rook,3_bishop,4_knight",
    "perspective=+1_white|-1_black;counter_present=explicit_bool;true_empty_counter_is_present",
    "padding=zero_features_zero_moves_false_masks;record_relations=-1;masked_inputs_cannot_contribute",
    "FP32=finite_little_endian_preserve_signed_zero;I64=little_endian;BOOL=0|1",
    "role=proposer|critic|validator;private_PC_scalar_BOOL_routes_are_separate_from_request_tensors",
    "policy_projection=trim_to_exact_candidate_count_then_stable_softmax_in_caller_order;C0_returns_empty_policy",
    "wdl_projection=stable_softmax_of_win_draw_loss_logits;exact_query_position_side_to_move",
    "critic_projection=trim_to_exact_divergence_count;validator_task_logits=7;private_latent=16x384",
    "no_CP_depth_hash_stateID_revision_numeric_features;Rules_full_history_is_execution_seal_only",
)


def registered_encoding_sha256():
    digest = hashlib.sha256()
    for field in ENCODING_REGISTRY_FIELDS:
        raw = field.encode("utf-8")
        digest.update(struct.pack("<Q", len(raw))); digest.update(raw)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(label + " fields differ from V3 schema")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 3.4028234663852886e38


def _moves(value, label):
    if type(value) is not list or len(value) > 256:
        raise ValueError(label + " line capacity")
    for move in value:
        if type(move) is not list or len(move) != 3 or any(type(x) is not int for x in move) or not (0 <= move[0] < 64 and 0 <= move[1] < 64 and move[0] != move[1] and 0 <= move[2] <= 4):
            raise ValueError(label + " Move16 components/promotion")


def length_bucket(maximum):
    if type(maximum) is not int or not 0 <= maximum <= 256:
        raise ValueError("line length exceeds registered buckets")
    return next(bucket for bucket in BUCKETS if maximum <= bucket)


def validate_payload(value):
    _keys(value, ("schema", "role", "anchors", "records", "query", "candidates", "divergences", "token_budget"), "review")
    if value["schema"] != INPUT_SCHEMA or value["role"] not in ROLES:
        raise ValueError("review schema/role")
    anchors, records, query = value["anchors"], value["records"], value["query"]
    if type(anchors) is not list or not 1 <= len(anchors) <= MAX_ANCHORS:
        raise ValueError("anchor capacity; current root must occupy anchor zero")
    for anchor in anchors:
        _keys(anchor, ("board", "metadata"), "anchor")
        if type(anchor["board"]) is not list or len(anchor["board"]) != 64 or any(type(x) is not int or not 0 <= x <= 12 for x in anchor["board"]):
            raise ValueError("anchor board piece representation")
        if type(anchor["metadata"]) is not list or len(anchor["metadata"]) != 16 or not all(_number(x) for x in anchor["metadata"]):
            raise ValueError("anchor metadata requires 16 finite Rules fields")
    if type(records) is not list or len(records) > MAX_RECORDS:
        raise ValueError("selected record capacity")
    for slot, record in enumerate(records):
        _keys(record, ("kind", "perspective", "completed_iteration", "rules_terminal", "line", "anchor", "parent", "supersedes"), "record")
        if record["kind"] not in KINDS or type(record["perspective"]) is not int or record["perspective"] not in (-1, 1) or type(record["completed_iteration"]) is not bool or type(record["rules_terminal"]) is not bool:
            raise ValueError("record eight-axis vocabulary")
        if type(record["anchor"]) is not int or not 0 <= record["anchor"] < len(anchors):
            raise ValueError("record anchor is not a selected local anchor")
        _moves(record["line"], "record")
        for relation in (record["parent"], record["supersedes"]):
            if relation is not None and (type(relation) is not int or not 0 <= relation < len(records) or relation == slot):
                raise ValueError("record relationship references an absent/self record")
    visiting, complete = set(), set()
    def visit(slot):
        if slot in visiting: raise ValueError("record relationship cycle")
        if slot in complete: return
        visiting.add(slot)
        for target in (records[slot]["parent"], records[slot]["supersedes"]):
            if target is not None: visit(target)
        visiting.remove(slot)
        complete.add(slot)
    for slot in range(len(records)): visit(slot)
    _keys(query, ("purpose", "perspective", "prefix", "proposal", "counter", "counter_present", "anchors"), "query")
    if query["purpose"] not in PURPOSES or type(query["perspective"]) is not int or query["perspective"] not in (-1, 1):
        raise ValueError("query eight-axis vocabulary")
    if type(query["counter_present"]) is not bool or (not query["counter_present"] and query["counter"]):
        raise ValueError("counter presence differs from its supplied sequence")
    if type(query["anchors"]) is not list or len(query["anchors"]) != 3 or any(type(x) is not int or not 0 <= x < len(anchors) for x in query["anchors"]):
        raise ValueError("query requires three exact local anchors")
    lines = [record["line"] for record in records] + [query[name] for name in ("prefix", "proposal", "counter")]
    for line in lines: _moves(line, "selected")
    token_count = sum(map(len, lines))
    if type(value["token_budget"]) is not int or value["token_budget"] not in (128, 256, 512) or token_count > value["token_budget"]:
        raise ValueError("ViewCapacity: full selected/query token count exceeds budget")
    _moves(value["candidates"], "candidate")
    if len(value["candidates"]) > MAX_CANDIDATES or len({tuple(x) for x in value["candidates"]}) != len(value["candidates"]):
        raise ValueError("candidate capacity/duplicate")
    if type(value["divergences"]) is not list or len(value["divergences"]) > 128 or any(type(row) is not list or len(row) != 8 or not all(_number(x) for x in row) for row in value["divergences"]):
        raise ValueError("divergence eight-field capacity/finite values")
    if value["role"] != "critic" and value["divergences"]:
        raise ValueError("private role cannot receive critic divergence input")
    return {"token_count": token_count, "line_bucket": length_bucket(max(map(len, lines), default=0)),
            "records": len(records), "anchors": len(anchors), "schema": MODEL_SCHEMA_V3,
            "model_semantics": MODEL_SEMANTICS_V3, "encoding": ENCODING_V3}


def record_features(record):
    return [float(record["kind"] == kind) for kind in KINDS] + [float(record["perspective"]), float(record["completed_iteration"]), float(record["rules_terminal"]), float(bool(record["line"]))]


def query_features(query):
    return [float(query["purpose"] == purpose) for purpose in PURPOSES] + [float(query["perspective"]), float(query["counter_present"])]


def prepare_payload(value):
    """One B1 owned plain-data feed; callers create tensors only after admission."""
    admission = validate_payload(value)
    bucket, records, query = admission["line_bucket"], value["records"], value["query"]
    def packed(line):
        return [list(move) for move in line] + [[0, 0, 0] for _ in range(bucket - len(line))]
    def mask(line): return [i < len(line) for i in range(bucket)]
    rows = records or [None]
    candidates = value["candidates"] or [[0, 0, 0]]
    divergences = value["divergences"] or [[0.] * 8]
    feed = {"anchor_board": [[list(x["board"]) for x in value["anchors"]]],
            "anchor_metadata": [[list(x["metadata"]) for x in value["anchors"]]], "anchor_mask": [[True] * len(value["anchors"])],
            "records": [[record_features(x) if x else [0.] * 8 for x in rows]],
            "record_anchor": [[x["anchor"] if x else 0 for x in rows]], "record_mask": [[x is not None for x in rows]],
            "record_line_tokens": [[packed(x["line"] if x else []) for x in rows]],
            "record_line_mask": [[mask(x["line"] if x else []) for x in rows]],
            "query": [query_features(query)], "query_anchor": [list(query["anchors"])],
            "query_line_tokens": [[packed(query[name]) for name in ("prefix", "proposal", "counter")]],
            "query_line_mask": [[mask(query[name]) for name in ("prefix", "proposal", "counter")]],
            "record_relations": [[[x["parent"] if x["parent"] is not None else -1, x["supersedes"] if x["supersedes"] is not None else -1] if x else [-1, -1] for x in rows]],
            "candidates": [[list(x) for x in candidates]], "candidate_mask": [[i < len(value["candidates"]) for i in range(len(candidates))]],
            "divergence_features": [[list(x) for x in divergences]], "divergence_mask": [[i < len(value["divergences"]) for i in range(len(divergences))]]}
    return {"admission": admission, "feed": feed}


def input_digest(value):
    feed = prepare_payload(value)["feed"]
    digest = hashlib.sha256(ENCODING_V3.encode() + b"\0")
    def frame(raw):
        digest.update(len(raw).to_bytes(8, "little")); digest.update(raw)
    def flattened(data):
        if type(data) is list:
            for child in data: yield from flattened(child)
        else: yield data
    def shape(data):
        result = []
        while type(data) is list:
            result.append(len(data)); data = data[0] if data else None
        return result
    integers = {"anchor_board", "record_anchor", "record_line_tokens", "candidates", "query_anchor", "query_line_tokens", "record_relations"}
    booleans = {"anchor_mask", "record_line_mask", "record_mask", "candidate_mask", "divergence_mask", "query_line_mask"}
    frame(value["role"].encode())
    for name, data in sorted(feed.items()):
        dtype = "i64" if name in integers else "bool" if name in booleans else "f32"
        dimensions = shape(data)
        frame(name.encode()); frame(dtype.encode()); digest.update(len(dimensions).to_bytes(8, "little"))
        for dimension in dimensions: digest.update(dimension.to_bytes(8, "little"))
        for element in flattened(data):
            digest.update(struct.pack("<q" if dtype == "i64" else "<?" if dtype == "bool" else "<f", element))
    return digest.hexdigest()


def independent_record_key(value, slot):
    validate_payload(value)
    if type(slot) is not int or not 0 <= slot < len(value["records"]): raise ValueError("record page slot")
    record = value["records"][slot]
    page = {"anchor": value["anchors"][record["anchor"]], "features": record_features(record), "line": record["line"]}
    return hashlib.sha256(b"rovezero.pals-independent-review-record.v1\0" + _canonical(page)).hexdigest()


def execution_keys(value, *, model_epoch, rules_history, anchor_history_seals, backend, precision, seed_digest=None):
    """Fresh and path-dependent Warm namespaces never alias each other.

    Rules history/model epoch remain acceptance/cache identity, never features.
    A Warm-compatible context excludes public records but includes exact query
    anchor boards, query sequences, role, history, model, backend and precision.
    Actual accepted-seed ownership is certified by the Rust physical owner.
    """
    validate_payload(value)
    if seed_digest is not None and (value["role"] == "validator" or value["query"]["purpose"] == "value_fresh"):
        raise ValueError("value and V require Fresh; approximate seed is unsupported")
    for digest in (model_epoch, rules_history, seed_digest if seed_digest is not None else "00" * 32):
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest): raise ValueError("execution identity digest")
    if type(anchor_history_seals) is not list or len(anchor_history_seals)!=len(value["anchors"]) or anchor_history_seals[0]!=rules_history or any(type(x)is not str or len(x)!=64 or any(c not in "0123456789abcdef" for c in x) for x in anchor_history_seals):
        raise ValueError("every anchor requires an ordered exact Rules/history seal")
    if type(backend) is not str or not backend or precision != "fp32": raise ValueError("execution backend/registered precision")
    query = value["query"]
    context = {"model": model_epoch, "history": rules_history, "anchor_histories":anchor_history_seals,"backend": backend, "precision": precision, "role": value["role"],
               "query": query, "query_anchors": [value["anchors"][i] for i in query["anchors"]],
               "candidates": value["candidates"], "divergences": value["divergences"]}
    context_key = hashlib.sha256(b"rovezero.pals-review-context.v1\0" + _canonical(context)).hexdigest()
    fresh = hashlib.sha256(b"rovezero.pals-review-fresh.v1\0" + _canonical([context_key, input_digest(value)])).hexdigest()
    warm = None if seed_digest is None else hashlib.sha256(b"rovezero.pals-review-approx-warm.v1\0" + _canonical([context_key, input_digest(value), seed_digest])).hexdigest()
    return {"context_key": context_key, "fresh_key": fresh, "warm_invocation_key": warm, "seed_acceptance": "Rust_owner_required"}


def controlled_copy_plan(source, target, allowlist):
    """Inventory-only exact allowlist; never loads a checkpoint or copies tensors."""
    if type(allowlist) is not dict or not allowlist: raise ValueError("explicit semantic allowlist required")
    result = []
    for name, semantics in sorted(allowlist.items()):
        left, right = source.get(name), target.get(name)
        if type(name) is not str or not name or type(semantics) is not str or not semantics or type(left) is not dict or type(right) is not dict:
            raise ValueError("parameter allowlist/inventory membership")
        if left.get("shape") != right.get("shape") or left.get("dtype") != right.get("dtype") or left.get("dtype") != "fp32_le" or left.get("semantics") != semantics or right.get("semantics") != semantics:
            raise ValueError("controlled copy requires equal name/shape/dtype/meaning")
        if type(left.get("shape")) is not list or not left["shape"] or any(type(x) is not int or x < 1 for x in left["shape"]):
            raise ValueError("controlled copy inventory shape")
        digest = left.get("sha256")
        if type(digest) is not str or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
            raise ValueError("controlled copy source inventory digest")
        result.append({"name": name, "shape": list(left["shape"]), "dtype": "fp32_le", "semantics": semantics,
                       "source_sha256": left.get("sha256"), "operation": "copy_after_complete_prevalidation"})
    return {"schema": "rovezero.pals-controlled-copy.v1", "parameters": result,
            "learned_transfer_claim": False, "copies_executed": 0}
