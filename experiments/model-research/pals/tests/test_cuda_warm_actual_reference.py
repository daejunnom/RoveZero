"""Pure capture admission and bounded worker lifecycle; no NN forward/import."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model.config import ModelConfig
from rz_pals_model.cuda_warm_actual_reference import (
    ATOL, LATENT_ELEMENTS, MAX_CAPTURE_CALLS, CAPTURE_SCHEMA, REFERENCE_SCHEMA, DEFAULT_ROLE_FORWARDS,
    _bind_native_source_identity, _compare_raw, _external_path, _json_bytes, _reference_call, _write_reference,
    canonical_captured_input_key, latent_bytes, seed_provenance_digest, run_actual_reference_cli,
    validate_actual_capture, validate_captured_input, _reference_streams)


MANIFEST_SHA = "11" * 32


def _input(role="proposer"):
    move = {"from": 8, "to": 16, "promotion": 0}
    line = [move, {"from": 48, "to": 40, "promotion": 0}, {"from": 16, "to": 24, "promotion": 0}]
    return {"role": role, "board": [index % 13 for index in range(64)],
            "metadata": [0.0] * 16, "query": [0.0] * 16,
            "records": [{"record_id": 10, "revision": 2, "critical": True, "features": [0.0] * 16},
                        {"record_id": 20, "revision": 3, "critical": False, "features": [0.0] * 16}],
            "required_critical_records": [10], "candidates": [move],
            "divergence_features": [[0.0] * 8] if role == "critic" else [],
            "situation_revision": 5, "history_digest": [7] * 32, "model_epoch": [23] * 32,
            "full_line": {"records": [{"moves": line, "parent": None, "supersedes": None},
                                     {"moves": [], "parent": 0, "supersedes": None}],
                          "query_prefix": [], "query_proposal": copy.deepcopy(line), "query_counter": []}}


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _framed_seed_digest(bits):
    # Independent producer-format fixture; capture initial/final SHA stays raw.
    payload = b"".join(struct.pack("<I", bit) for bit in bits)
    return hashlib.sha256(b"rz-pals-private-finite-fp32-latent/1" + struct.pack("<Q", len(bits)) + payload).hexdigest()


def _call(sequence=1, role="proposer", warm=False, source=None):
    config = ModelConfig.for_profile("full_line_interaction_v2")
    value = _input(role)
    key = canonical_captured_input_key(value, config)
    raw = {"candidate_logits": [0.125], "wdl_logits": [0.25, -0.0, -0.25],
           "divergence_logits": [0.375] if role == "critic" else None,
           "task_logits": None, "private_latent": [-0.0] + [0.5] * (LATENT_ELEMENTS - 1)}
    bits = {name: [_bits(number) for number in numbers] if numbers is not None else None for name, numbers in raw.items()}
    seed = source["raw_output_bits"]["private_latent"] if warm else [_bits(-0.0)] * LATENT_ELEMENTS
    digest = hashlib.sha256(latent_bytes(seed)).hexdigest()
    invocation = {"mode": "approx_cuda_warm_v2" if warm else "fresh", "input_key_hex": key,
                  "invocation_key_hex": "33" * 32 if warm else key}
    provenance = None
    if warm:
        invocation["seed_seal_hex"] = "44" * 32
        provenance = {"seal_hex": "44" * 32, "role": role, "latent_bits_digest_hex": _framed_seed_digest(seed),
                      "model_manifest_sha256_hex": MANIFEST_SHA, "model_epoch_hex": bytes(value["model_epoch"]).hex(),
                      "source_input_hex": source["input_key_hex"]}
    encoded_input = json.dumps(value, separators=(",", ":"), allow_nan=False)
    input_bits = {"metadata": [_bits(number) for number in value["metadata"]],
                  "query": [_bits(number) for number in value["query"]],
                  "records": [[_bits(number) for number in record["features"]] for record in value["records"]],
                  "divergence_features": [[_bits(number) for number in features] for features in value["divergence_features"]]}
    return {"request_id": {"epoch": 1, "sequence": sequence}, "execution_id": {"epoch": 2, "sequence": sequence},
            "input": value, "input_key_hex": key, "input_json_utf8": encoded_input, "input_f32_bits": input_bits,
            "input_json_sha256_hex": hashlib.sha256(encoded_input.encode("utf-8")).hexdigest(),
            "initial_latent_bits": seed, "initial_latent_sha256_hex": digest,
            "warm_start": warm, "invocation": invocation, "seed_provenance": provenance,
            "physical": {"dispatched": True, "ready": True, "completion_unknown": False, "completed_ok": True},
            "accepted": True, "delivered": True, "raw_output": raw, "raw_output_bits": bits,
            "final_latent_sha256_hex": hashlib.sha256(latent_bytes(bits["private_latent"])).hexdigest()}


def _capture():
    fresh_p, fresh_c = _call(), _call(2, "critic")
    return {"schema": CAPTURE_SCHEMA, "phase": "capture", "execution_checks_passed": True,
            "export_manifest_sha256_hex": MANIFEST_SHA,
            "max_actual_role_forwards_including_startup": DEFAULT_ROLE_FORWARDS,
            "actual_role_forwards_including_startup": 5, "actual_completed_role_forwards_including_startup": 5,
            "actual_known_completed_startup_role_forwards": 2, "role_forward_accounting_checked": True,
            "model_configuration": ModelConfig.for_profile("full_line_interaction_v2").to_dict(),
            "calls": [fresh_p, fresh_c, _call(3, warm=True, source=fresh_p)]}


def _cost_capture():
    capture = _capture()
    source = capture["calls"][-1]
    source.update(accepted_with_original_controls=True,
                  logical_context={"purpose": "RepairPolicy", "game_generation": 9, "search_generation": 2,
                                   "situation_revision": 5, "public_revision": 7, "prefix": []})
    source_identity = {"checkpoint_sha256": [23] * 32, "export_manifest_sha256": list(bytes.fromhex(MANIFEST_SHA)),
                       "encoding_semantic_sha256": [0x81] * 32, "adapter_source_sha256": [0x82] * 32,
                       "model_configuration": capture["model_configuration"], "trained": False, "frozen_epoch": 5}
    capture["source_identity"] = source_identity
    source["seed_provenance"].update(encoding_semantic_sha256_hex="81" * 32, frozen_epoch=5)
    seed = source["raw_output_bits"]["private_latent"]
    def receipt(calls, consumers, warm, final=False):
        value = {"process_epoch": 1, "game_generation": 9, "physical_runs_in_flight": 0, "quarantined": False,
                 "physically_completed_role_calls": calls, "completed_role_inputs": calls,
                 "delivered_role_inputs": calls, "search_consumed_role_inputs": consumers,
                 "private_warm_observation": {"pending_acceptance": False, "pinned_entries": 0,
                                              "active_lease": None, "accepted_seeds": 3 if warm else 0},
                 "backend_stats": {"role_nn_runs_attempted": calls + 2, "role_nn_runs_completed": calls + 2,
                                   "role_nn_runs_failed_known": 0}}
        if final:
            value.update(physical_shutdown_confirmed=True, native_buffers_released=True)
        return value
    cost = {"opt_in": True, "execution_checks_passed": True, "owners_run_sequentially_after_actual_worker_join": True,
            "max_concurrent_physical_workers": 1, "selected_max_leases": 1, "selected_device_payload_bytes_max": 397312,
            "preparation_calls_per_lane": 3, "measured_calls_per_lane": 5,
            "extra_role_forward_reservation_including_fresh_startup": 18,
            "original_accepted_repair_source": copy.deepcopy(source),
            "original_accepted_full_6144_seed_bits": copy.deepcopy(seed),
            "same_target_input_json_sha256_hex": source["input_json_sha256_hex"]}
    for name, warm in (("warm", True), ("fresh", False)):
        calls = []
        base = len(capture["calls"]) if warm else 0
        for index in range(8):
            call = _call(base + index + 1, warm=warm, source=source if warm else None)
            call.update(accepted=False, rejected="SearchFailedUnconsumed",
                        logical_context=copy.deepcopy(source["logical_context"]))
            if warm:
                call["initial_latent_bits"] = copy.deepcopy(call["initial_latent_bits"])
                call["seed_provenance"].update(source_lease_id=99, seed_sequence=7,
                                               encoding_semantic_sha256_hex="81" * 32, frozen_epoch=5)
            else:
                call["initial_latent_bits"] = [0] * LATENT_ELEMENTS
                call["initial_latent_sha256_hex"] = hashlib.sha256(latent_bytes(call["initial_latent_bits"])).hexdigest()
            call["cost_sample"] = {"lane": "warm_same_accepted_seed" if warm else "fresh_missing_seed",
                                   "phase": "preparation" if index < 3 else "measurement",
                                   "ordinal_in_phase": index + 1 if index < 3 else index - 2,
                                   "api_returned_ok": True, "close_unconsumed_returned_ok": True,
                                   "receipt_before": receipt(base + index, base, warm),
                                   "receipt_returned": receipt(base + index + 1, base, warm),
                                   "receipt_after_close": receipt(base + index + 1, base, warm)}
            call["cost_observation"] = {"actual_completed_nn_inputs_delta": None, "nn_input_measurement_complete": False}
            call["cost_sample"]["public_cache_preparation_counters"] = None
            calls.append(call)
        lane = {"complete": True, "failure": None, "physical_owner_joined": True,
                "preparation_calls_requested": 3, "measured_calls_requested": 5, "measured_calls_completed": 5,
                "calls": calls, "stable_accepted_seed_provenance": copy.deepcopy(calls[0]["seed_provenance"]),
                "final_receipt": receipt(base + 8, base, warm, True)}
        if not warm:
            lane.update(source_identity=copy.deepcopy(source_identity), work_completed_within_original_deadline=True,
                        cleanup_failure=None, startup_receipt=receipt(0, 0, False))
            lane["startup_receipt"]["startup_probe"] = {"completed_proposer_calls": 1, "completed_critic_calls": 1,
                                                        "reset_completed": True, "runtime_mapping_confirmed": True,
                                                        "cuda_placement_witness": {"fixture": "metadata_only"}}
        cost[name] = lane
    capture.update(cost_same_seed_context=cost, final_receipt=copy.deepcopy(cost["warm"]["final_receipt"]),
                   actual_role_forwards_including_startup=13, actual_completed_role_forwards_including_startup=13)
    return capture


class CapturedReferenceAdmissionTests(unittest.TestCase):
    def test_seed_provenance_framing_has_independent_known_digest_and_preserves_raw_sha(self):
        bits = [0x80000000] + [0x3e800000] * (LATENT_ELEMENTS - 1)
        known = "31d746ae061e3380aa88d34b3a310197361ecec7205c4f35ff3b5a3367b642a5"
        self.assertEqual(_framed_seed_digest(bits), known)
        self.assertEqual(seed_provenance_digest(bits), known)
        self.assertEqual(hashlib.sha256(latent_bytes(bits)).hexdigest(),
                         "99ea5aff8cdd0b1a31d72ba623c983870d2316751d2af3b7f8541a0ebd838ea4")
        bits[0] = 0
        self.assertEqual(seed_provenance_digest(bits),
                         "2a90e9a61891c51937fa4465758740a3f4b41c0d5169a6045ada8963c29303d1")

    def test_original_and_cost_reject_raw_only_wrong_domain_length_and_signed_zero_provenance(self):
        for cost in (False, True):
            for label in ("raw_only", "wrong_domain", "wrong_length", "signed_zero"):
                capture = _cost_capture() if cost else _capture()
                warm = capture["cost_same_seed_context"]["warm"]["calls"][0] if cost else capture["calls"][2]
                bits = warm["initial_latent_bits"]
                self.assertEqual(bits[0], 0x80000000)
                payload = latent_bytes(bits)
                domain = b"rz-pals-private-finite-fp32-latent/1"
                length = len(bits)
                if label == "raw_only":
                    invalid = hashlib.sha256(payload).hexdigest()
                else:
                    if label == "wrong_domain": domain = b"rz-pals-private-finite-fp32-latent/2"
                    elif label == "wrong_length": length -= 1
                    else: payload = struct.pack("<I", 0) + payload[4:]
                    invalid = hashlib.sha256(domain + struct.pack("<Q", length) + payload).hexdigest()
                warm["seed_provenance"]["latent_bits_digest_hex"] = invalid
                before = copy.deepcopy(capture)
                with self.subTest(cost=cost, label=label), self.assertRaisesRegex(ValueError, "seed provenance|original accepted full seed"):
                    validate_actual_capture(capture, MANIFEST_SHA)
                self.assertEqual(capture, before)

    def test_cost_keeps_original_and_two_owner_namespaces_without_id_rewrite(self):
        capture = _cost_capture()
        before = copy.deepcopy(capture)
        validate_actual_capture(capture, MANIFEST_SHA)
        self.assertEqual(capture, before)
        streams = _reference_streams(capture)
        self.assertEqual([(name, len(calls)) for name, calls in streams], [("original", 3), ("warm", 8), ("fresh", 8)])
        for field in ("request_id", "execution_id"):
            self.assertNotEqual(streams[0][1][0][field], streams[1][1][0][field])
            self.assertEqual(streams[0][1][0][field], streams[2][1][0][field])
            self.assertEqual(streams[1][1][0][field], streams[2][1][3][field])
        self.assertEqual(streams[1][1][0]["initial_latent_bits"], capture["cost_same_seed_context"]["original_accepted_full_6144_seed_bits"])
        self.assertEqual(streams[2][1][0]["initial_latent_bits"], [0] * LATENT_ELEMENTS)
        self.assertIsNone(streams[1][1][0]["cost_observation"]["actual_completed_nn_inputs_delta"])
        self.assertIsNone(streams[1][1][0]["cost_sample"]["public_cache_preparation_counters"])
        self.assertEqual(_reference_streams(_capture())[0][0], "original")

    def test_partial_unknown_consumed_or_changed_cost_samples_are_refused(self):
        for label in ("missing_cost", "partial", "unknown", "consumed", "consumer_none", "unclosed", "seed_last",
                      "source_removed", "context", "fresh_seed", "fresh_accepted", "mode", "output_bits", "provenance",
                      "phase", "duplicate", "count", "extra_startup", "warm_count", "budget", "fence", "final_nn_none"):
            capture = _cost_capture()
            cost = capture["cost_same_seed_context"]
            warm = cost["warm"]["calls"][0]
            after = warm["cost_sample"]["receipt_after_close"]
            if label == "missing_cost": capture["cost_same_seed_context"] = None
            elif label == "partial": cost["fresh"]["complete"] = False
            elif label == "unknown": warm["physical"]["completion_unknown"] = True
            elif label == "consumed": after["search_consumed_role_inputs"] += 1
            elif label == "consumer_none": after["search_consumed_role_inputs"] = None
            elif label == "unclosed": warm["cost_sample"]["close_unconsumed_returned_ok"] = False
            elif label == "seed_last": warm["initial_latent_bits"][6143] = 0
            elif label == "source_removed": cost["original_accepted_repair_source"]["accepted"] = False
            elif label == "context": warm["logical_context"]["public_revision"] += 1
            elif label == "fresh_seed": cost["fresh"]["calls"][0]["initial_latent_bits"][0] = _bits(-0.0)
            elif label == "fresh_accepted": cost["fresh"]["calls"][0]["accepted"] = True
            elif label == "mode": warm["invocation"]["mode"] = "fresh"
            elif label == "output_bits": warm["raw_output_bits"]["private_latent"][6143] = 0
            elif label == "provenance": cost["warm"]["calls"][1]["seed_provenance"]["source_lease_id"] += 1
            elif label == "phase": warm["cost_sample"]["phase"] = "measurement"
            elif label == "duplicate": cost["warm"]["calls"][1]["request_id"] = warm["request_id"]
            elif label == "count": cost["fresh"]["calls"].pop()
            elif label == "extra_startup": cost["fresh"]["startup_receipt"]["startup_probe"]["completed_proposer_calls"] += 1
            elif label == "warm_count": capture["actual_role_forwards_including_startup"] -= 8
            elif label == "budget": capture["max_actual_role_forwards_including_startup"] = 128
            elif label == "fence": cost["fresh"]["final_receipt"]["physical_shutdown_confirmed"] = False
            else: cost["fresh"]["final_receipt"]["backend_stats"] = None
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_actual_capture(capture, MANIFEST_SHA)

    def test_warm_cost_refuses_original_request_and_execution_collisions(self):
        for field in ("request_id", "execution_id"):
            capture = _cost_capture()
            capture["cost_same_seed_context"]["warm"]["calls"][0][field] = copy.deepcopy(capture["calls"][0][field])
            before = copy.deepcopy(capture)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "duplicate cost"):
                validate_actual_capture(capture, MANIFEST_SHA)
            self.assertEqual(capture, before)

    def test_cost_combined_forward_budget_accepts_64_and_refuses_65_before_replay(self):
        for original_count in (44, 45):
            capture = _cost_capture()
            delta = original_count - len(capture["calls"])
            capture["calls"][-1:-1] = [_call(sequence) for sequence in range(4, 4 + delta)]
            warm = capture["cost_same_seed_context"]["warm"]
            # Synthetic fixture IDs follow the enlarged original owner prefix.
            for sample in warm["calls"]:
                for field in ("request_id", "execution_id"):
                    sample[field]["sequence"] += delta
            receipts = [warm["final_receipt"]]
            receipts.extend(sample["cost_sample"][name] for sample in warm["calls"]
                            for name in ("receipt_before", "receipt_returned", "receipt_after_close"))
            for receipt in receipts:
                for name in ("physically_completed_role_calls", "completed_role_inputs", "delivered_role_inputs", "search_consumed_role_inputs"):
                    receipt[name] += delta
                for name in ("role_nn_runs_attempted", "role_nn_runs_completed"):
                    receipt["backend_stats"][name] += delta
            capture["final_receipt"] = copy.deepcopy(warm["final_receipt"])
            capture["actual_role_forwards_including_startup"] += delta
            capture["actual_completed_role_forwards_including_startup"] += delta
            with self.subTest(original_count=original_count):
                if original_count == 44:
                    validate_actual_capture(capture, MANIFEST_SHA)
                    self.assertEqual(44 + 2 + 8 + 2 + 8, 64)
                    self.assertEqual(capture["actual_role_forwards_including_startup"], 54)
                    self.assertEqual(capture["cost_same_seed_context"]["fresh"]["final_receipt"]["backend_stats"]["role_nn_runs_completed"], 10)
                else:
                    with self.assertRaisesRegex(ValueError, "reserved forward budget"):
                        validate_actual_capture(capture, MANIFEST_SHA)

    def test_cost_binds_each_actual_owner_without_rewriting_request_ids(self):
        capture = _cost_capture()
        before = copy.deepcopy(capture)
        config = ModelConfig.for_profile("full_line_interaction_v2")
        manifest = {"checkpoint_sha256": "17" * 32}
        self.assertIs(_bind_native_source_identity(capture, manifest, {"trained": False}, config), capture["source_identity"])
        self.assertEqual(capture, before)
        for lane, field in (("warm", "encoding"), ("fresh", "checkpoint")):
            changed = copy.deepcopy(capture)
            if field == "encoding":
                changed["cost_same_seed_context"][lane]["calls"][0]["seed_provenance"]["encoding_semantic_sha256_hex"] = "aa" * 32
            else:
                changed["cost_same_seed_context"][lane]["source_identity"]["checkpoint_sha256"] = [24] * 32
            with self.subTest(lane=lane), self.assertRaises(ValueError):
                _bind_native_source_identity(changed, manifest, {"trained": False}, config)

    def test_full_seed_signed_zero_and_prior_same_role_are_preserved(self):
        capture = _capture()
        self.assertEqual(validate_actual_capture(capture, MANIFEST_SHA).profile, "full_line_interaction_v2")
        self.assertEqual(latent_bytes(capture["calls"][0]["initial_latent_bits"])[:4], b"\x00\x00\x00\x80")
        self.assertEqual(capture["calls"][2]["initial_latent_bits"], capture["calls"][0]["raw_output_bits"]["private_latent"])

    def test_canonical_key_binds_full_middle_query_relations_history_and_signed_zero(self):
        value = _input()
        config = ModelConfig.for_profile("full_line_interaction_v2")
        original = canonical_captured_input_key(value, config)
        for label in ("middle_record", "middle_query", "relation", "history", "record_id", "revision", "signed_zero", "required_flag"):
            changed = copy.deepcopy(value)
            if label == "middle_record": changed["full_line"]["records"][0]["moves"][1]["to"] = 39
            elif label == "middle_query": changed["full_line"]["query_proposal"][1]["to"] = 39
            elif label == "relation": changed["full_line"]["records"][1].update(parent=None, supersedes=0)
            elif label == "history": changed["history_digest"][31] = 8
            elif label == "record_id": changed["records"][1]["record_id"] = 21
            elif label == "revision": changed["situation_revision"] = 6
            elif label == "signed_zero": changed["query"][0] = -0.0
            else: changed["full_line"]["records"][1]["parent_required"] = True
            with self.subTest(label=label):
                self.assertNotEqual(canonical_captured_input_key(changed, config), original)

    def test_malformed_typed_inputs_fail_before_tensor_allocation(self):
        config = ModelConfig.for_profile("full_line_interaction_v2")
        for label in ("bool_id", "missing_critical", "unknown_feature", "cycle", "missing_required", "record_id_feature", "query_revision", "query_deadline", "line_capacity", "bad_move", "duplicate_move", "head_profile", "nonfinite"):
            changed, selected = _input(), config
            if label == "bool_id": changed["records"][0]["record_id"] = True
            elif label == "missing_critical": changed["records"][0]["critical"] = False
            elif label == "unknown_feature": changed["full_line"]["record_id_hash"] = 1
            elif label == "cycle": changed["full_line"]["records"][0]["supersedes"] = 1
            elif label == "missing_required": changed["full_line"]["records"][0]["parent_required"] = True
            elif label == "record_id_feature": changed["records"][0]["features"][6] = 1.0
            elif label == "query_revision": changed["query"][6] = 1.0
            elif label == "query_deadline": changed["query"][7] = 1.0
            elif label == "line_capacity": changed["full_line"]["query_counter"] = changed["candidates"] * 257
            elif label == "bad_move": changed["candidates"][0]["promotion"] = 5
            elif label == "duplicate_move": changed["candidates"] *= 2
            elif label == "head_profile": selected = ModelConfig.for_profile("interaction_head_v2")
            else: changed["metadata"][0] = float("nan")
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_captured_input(changed, selected)

    def test_call_provenance_modes_completion_and_bit_tampering_are_refused(self):
        for label in ("execution_gate", "full_input", "input_utf8", "input_digest", "input_f32_bits", "serde_signed_zero", "seed_bit", "seed_digest", "seed_nan", "warm_mode", "fresh_seal", "wrong_role_seed", "prior_seed_missing", "provenance_seal", "output_bit", "output_length", "final_latent_digest", "request_duplicate", "execution_duplicate", "unknown_completion", "not_delivered", "call_count", "profile", "manifest"):
            changed = copy.deepcopy(_capture())
            first, warm = changed["calls"][0], changed["calls"][2]
            digest = MANIFEST_SHA
            if label == "execution_gate": changed["execution_checks_passed"] = False
            elif label == "full_input": first["input"]["full_line"]["query_proposal"][1]["to"] = 39
            elif label == "input_utf8": first["input_json_utf8"] += " "
            elif label == "input_digest": first["input_json_sha256_hex"] = "66" * 32
            elif label == "input_f32_bits": first["input_f32_bits"]["query"][0] = _bits(-0.0)
            elif label == "serde_signed_zero":
                parsed = copy.deepcopy(first["input"])
                parsed["query"][0] = -0.0
                first["input_json_utf8"] = json.dumps(parsed, separators=(",", ":"))
                first["input_json_sha256_hex"] = hashlib.sha256(first["input_json_utf8"].encode("utf-8")).hexdigest()
            elif label == "seed_bit": first["initial_latent_bits"][6143] = 0
            elif label == "seed_digest": first["initial_latent_sha256_hex"] = "66" * 32
            elif label == "seed_nan": first["initial_latent_bits"][0] = 0x7fc00000
            elif label == "warm_mode": warm["invocation"]["mode"] = "approx_warm_v1_refused"
            elif label == "fresh_seal": first["invocation"]["seed_seal_hex"] = "66" * 32
            elif label == "wrong_role_seed": warm["seed_provenance"]["role"] = "critic"
            elif label == "prior_seed_missing":
                changed["calls"] = [warm]
                changed["actual_role_forwards_including_startup"] = changed["actual_completed_role_forwards_including_startup"] = 3
            elif label == "provenance_seal": warm["seed_provenance"]["seal_hex"] = "66" * 32
            elif label == "output_bit": first["raw_output_bits"]["wdl_logits"][1] = 0
            elif label == "output_length": first["raw_output"]["private_latent"].pop()
            elif label == "final_latent_digest": first["final_latent_sha256_hex"] = "66" * 32
            elif label == "request_duplicate": warm["request_id"] = first["request_id"]
            elif label == "execution_duplicate": warm["execution_id"] = first["execution_id"]
            elif label == "unknown_completion": first["physical"]["completion_unknown"] = True
            elif label == "not_delivered": first["delivered"] = False
            elif label == "call_count": changed["calls"] = changed["calls"] * (MAX_CAPTURE_CALLS + 1)
            elif label == "profile": changed["model_configuration"]["iterations"] = True
            else: digest = "66" * 32
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_actual_capture(changed, digest)

    def test_empty_semantic_slots_stay_explicit(self):
        value = _input("critic")
        value.update(records=[], required_critical_records=[], candidates=[], divergence_features=[])
        value["full_line"] = {"records": [], "query_prefix": [], "query_proposal": [], "query_counter": []}
        validate_captured_input(value, ModelConfig.for_profile("full_line_interaction_v2"))

    def test_explicit_scenario_role_budget_and_actual_accounting_are_required(self):
        for selected in (16, DEFAULT_ROLE_FORWARDS, 128):
            capture = _capture()
            capture["max_actual_role_forwards_including_startup"] = selected
            validate_actual_capture(capture, MANIFEST_SHA)
        for name, number in (("max_actual_role_forwards_including_startup", 15),
                             ("max_actual_role_forwards_including_startup", 129),
                             ("max_actual_role_forwards_including_startup", True),
                             ("actual_role_forwards_including_startup", 6),
                             ("actual_completed_role_forwards_including_startup", 4),
                             ("actual_known_completed_startup_role_forwards", 1),
                             ("role_forward_accounting_checked", False)):
            capture = _capture()
            capture[name] = number
            with self.subTest(name=name, value=number), self.assertRaises(ValueError):
                validate_actual_capture(capture, MANIFEST_SHA)
        capture = _capture()
        del capture["max_actual_role_forwards_including_startup"]
        with self.assertRaises(ValueError):
            validate_actual_capture(capture, MANIFEST_SHA)

    def test_selected_role_budget_fences_calls_before_replay(self):
        capture = _capture()
        capture["max_actual_role_forwards_including_startup"] = 16
        capture["calls"] = [_call(sequence) for sequence in range(1, 16)]
        capture["actual_role_forwards_including_startup"] = capture["actual_completed_role_forwards_including_startup"] = 17
        with self.assertRaisesRegex(ValueError, "explicit registered role-forward budget"):
            validate_actual_capture(capture, MANIFEST_SHA)

    def test_reference_tolerance_checks_last_latent_coordinate_and_role_heads(self):
        original = _call()["raw_output"]
        for label in ("latent_last", "wdl", "head", "nonfinite"):
            changed = copy.deepcopy(original)
            if label == "latent_last": changed["private_latent"][6143] += 0.1
            elif label == "wdl": changed["wdl_logits"][1] += ATOL * 2
            elif label == "head": changed["task_logits"] = [0.0] * 7
            else: changed["candidate_logits"][0] = float("inf")
            with self.subTest(label=label), self.assertRaises(ValueError):
                _compare_raw(changed, original)
        self.assertEqual(_compare_raw(original, original)["private_latent"], 0.0)

    def test_duplicate_or_nonfinite_json_is_refused(self):
        for contents in (b'{"calls": [], "calls": []}', b'{"x": NaN}', b'{"x": Infinity}', b'[]'):
            with self.assertRaises(ValueError):
                _json_bytes(contents)

    def test_normalized_policy_and_wdl_gate_is_separate_from_raw_gate(self):
        for head in ("candidate_logits", "wdl_logits"):
            original = _call()["raw_output"]
            original[head] = [10.0] * (2 if head == "candidate_logits" else 3)
            changed = copy.deepcopy(original)
            changed[head][0] += 0.001
            with self.subTest(head=head), self.assertRaisesRegex(ValueError, "normalized probability"):
                _compare_raw(changed, original)

    def test_reference_packet_keeps_exact_id_input_bits_lines_and_invocation(self):
        call = _capture()["calls"][2]
        packet = _reference_call(call, copy.deepcopy(call["raw_output"]), copy.deepcopy(call["raw_output"]))
        for name in ("request_id", "execution_id", "input_key_hex", "input_json_sha256_hex", "input_f32_bits", "invocation",
                     "initial_latent_sha256_hex", "warm_start"):
            self.assertEqual(packet[name], call[name])
        self.assertEqual(packet["full_line"], call["input"]["full_line"])
        self.assertEqual(len(packet["torch_raw_output"]["private_latent"]), LATENT_ELEMENTS)

    def test_capability_namespace_is_distinct_from_rules_only_digest(self):
        capture = _capture()
        config = ModelConfig.for_profile("full_line_interaction_v2")
        source = {"checkpoint_sha256": [23] * 32, "export_manifest_sha256": list(bytes.fromhex(MANIFEST_SHA)),
                  "encoding_semantic_sha256": [0x81] * 32, "adapter_source_sha256": [0x82] * 32,
                  "model_configuration": config.to_dict(), "trained": False, "frozen_epoch": 1}
        capture["source_identity"] = source
        capture["calls"][2]["seed_provenance"].update(encoding_semantic_sha256_hex="81" * 32, frozen_epoch=1)
        manifest = {"checkpoint_sha256": "17" * 32, "rules_input_semantic_sha256": "aa" * 32,
                    "rules_encoder_source_sha256": "bb" * 32}
        self.assertIs(_bind_native_source_identity(capture, manifest, {"trained": False}, config), source)
        capture["calls"][2]["seed_provenance"]["encoding_semantic_sha256_hex"] = "aa" * 32
        with self.assertRaises(ValueError):
            _bind_native_source_identity(capture, manifest, {"trained": False}, config)

    def test_atomic_exclusive_report_keeps_previous_artifact(self):
        with tempfile.TemporaryDirectory(prefix="rz-pals-actual-reference-") as directory:
            path = Path(directory) / "reference.json"
            report = {"schema": REFERENCE_SCHEMA, "calls": []}
            summary = _write_reference(path, report)
            self.assertEqual(summary["reference_sha256_hex"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(json.loads(path.read_bytes()), report)
            self.assertFalse(path.with_name(path.name + ".partial").exists())
            with self.assertRaises(FileExistsError):
                _write_reference(path, report)
            self.assertEqual(json.loads(path.read_bytes()), report)

    def test_secret_and_relative_paths_are_refused_before_read(self):
        with tempfile.TemporaryDirectory(prefix="rz-pals-actual-reference-") as directory:
            for path in (Path(directory) / ".env", Path(directory) / "credential.json", Path("relative.json")):
                with self.assertRaises(ValueError):
                    _external_path(path)

    def test_cli_timeout_terminates_then_kills_one_worker_without_retry(self):
        class Receiver:
            def poll(self, seconds): return False
            def close(self): pass
        class Sender:
            def close(self): pass
        class Worker:
            def __init__(self): self.alive, self.started, self.terminated, self.killed = True, 0, 0, 0
            def start(self): self.started += 1
            def is_alive(self): return self.alive
            def terminate(self): self.terminated += 1
            def kill(self): self.killed += 1; self.alive = False
            def join(self, seconds): pass
            def close(self): pass
        worker = Worker()
        class Context:
            def Pipe(self, duplex): return Receiver(), Sender()
            def Process(self, **kwargs): return worker
        arguments = argparse.Namespace(checkpoint="checkpoint", export_manifest="manifest", export_manifest_sha256=MANIFEST_SHA,
                                       capture_json="capture", capture_sha256="00" * 32, output_json="output", max_seconds=1)
        with patch("rz_pals_model.cuda_warm_actual_reference.multiprocessing.get_context", return_value=Context()):
            with self.assertRaises(TimeoutError):
                run_actual_reference_cli(arguments)
        self.assertEqual((worker.started, worker.terminated, worker.killed), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
