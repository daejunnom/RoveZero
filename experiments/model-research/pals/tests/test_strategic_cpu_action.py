"""Synthetic CPU binding wiring, not live execution or strategic utility proof.

The existing Query/2 and semantic fixtures/factories are used unchanged. Rust
receipt/error JSON below is synthetic raw data. No child, model operation,
native action, whole-cost comparison, optimizer or target is produced here.
"""

import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model import strategic_cpu_action as action
from rz_pals_model import strategic_verifier_query as query
from rz_pals_model import verifier_producer as legacy
from test_strategic_verifier_query import QueryFixture


class StrategicCpuFixture:
    def __init__(self, directory):
        self.base = QueryFixture(directory)
        self.checked = self.base.admit()

    def request(self, index=0):
        spec = self.checked.catalogue()["actions"][index]
        value = self.base.feedback.request(spec["task"])
        for name in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "max_output_bytes"):
            value[name] = spec[name]
        value.pop("context_sha256")
        value["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, value)
        return value

    def prepare(self, index=0, raw=None):
        raw = query.canonical(self.request(index)) + b"\n" if raw is None else raw
        return action.prepare_strategic_cpu_action(checked_query=self.checked, action_index=index,
            cpu_request_bytes=raw, expected_cpu_request_pin=query.byte_pin(raw))

    def receipt(self, prepared, *, partial=False, deadline=False):
        request = query._parse(prepared.raw_assets()["cpu_request"])
        response = self.base.feedback.response(request, partial=partial)
        if deadline:
            response.update(deadline_exceeded=True, elapsed_ms=request["max_wall_time_ms"] + 1)
        raw_response = query.canonical(response) + b"\n"
        envelope = prepared.envelope()
        value = {name: copy.deepcopy(envelope[name]) for name in ("schema", "context_sha256", "query_sha256",
            "catalogue_artifact", "before_result_artifact", "prior_ledger_sha256", "action", "cpu_request_artifact")}
        value.update(status="cpu_response_observed", scope=action.SCOPE,
            cpu_response_raw=raw_response.decode("utf-8"), cpu_response_artifact=query.byte_pin(raw_response),
            cpu_dispatch_attempted=True, actual_cpu_nodes=response["nodes"], baseline_present=response["baseline"] is not None,
            after_present=response["after"] is not None, binary_pin_scope="library_caller_verified_digest",
            elapsed_ms=response["elapsed_ms"] + 1, deadline_exceeded=deadline, **action._DENIED)
        return value, response

    def failed(self, prepared, *, unknown_nodes=False):
        value, response = self.receipt(prepared)
        before = None if unknown_nodes else copy.deepcopy(response["baseline"])
        work = None if unknown_nodes else {"nodes": 3, "quiescence_nodes": 1, "tt_hits": 0}
        known = None if unknown_nodes else before["nodes"] + work["nodes"]
        inner = {"code": "cpu_task_failed", "stage": "after_search", "message": "synthetic canceled execution",
            "known_nodes": known, "failed_check_work": work, "baseline": before, "after": None,
            "elapsed_ms": value["elapsed_ms"], "deadline_exceeded": False}
        value.update(status="cpu_dispatch_failed", cpu_response_raw=None, cpu_response_artifact=None,
            actual_cpu_nodes=known, baseline_present=before is not None, after_present=False)
        return {"code": "strategic_action_failed", "stage": "cpu_dispatch", "message": "synthetic CPU failure",
            "elapsed_ms": value["elapsed_ms"], "deadline_exceeded": False, "cpu_error": inner, "receipt": value}

    @staticmethod
    def consume(prepared, value):
        raw = action._canonical(value) + b"\n"
        return action.admit_strategic_cpu_action_receipt(prepared_action=prepared, receipt_bytes=raw,
            expected_receipt_pin=query.byte_pin(raw))


class StrategicCpuActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = StrategicCpuFixture(self.temp.name)

    def assert_denied(self, value):
        for name in action._DENIED:
            self.assertIs(value[name], False, name)
        self.assertIs(value["independent_loaded_process_observed"], False)
        self.assertEqual(value["actual_utility_groups"], 0)

    def test_prepared_exact_query_original_request_and_legacy_context(self):
        prepared = self.f.prepare()
        self.assertIs(prepared.verify(), prepared)
        envelope = prepared.envelope()
        self.assertEqual(set(envelope), set(action._ENVELOPE_FIELDS))
        context = envelope.pop("context_sha256")
        self.assertEqual(context, legacy._hash(action.SCHEMA, envelope))
        self.assertEqual(envelope["cpu_request_raw"].encode("utf-8"), prepared.raw_assets()["cpu_request"])
        self.assertEqual(envelope["query_sha256"], self.f.checked.sha256)
        self.assert_denied(prepared.audit())
        self.assertFalse(hasattr(prepared, "dispatch"))

    def test_detached_views_and_factory_only_immutable_types(self):
        prepared = self.f.prepare()
        identity = prepared.sha256
        prepared.envelope()["action"]["slot"] = 7
        prepared.raw_assets()["expected_cpu_request_pin"]["sha256"] = "0" * 64
        self.assertEqual(prepared.sha256, identity)
        with self.assertRaises(AttributeError):
            prepared._index = 1
        for constructor in (action.PreparedStrategicCpuAction, action.CheckedStrategicCpuObservation):
            with self.assertRaises(ValueError):
                constructor()

    def test_raw_request_alias_is_preserved_and_not_an_independent_execution(self):
        request = self.f.request()
        first = self.f.prepare()
        alias_raw = json.dumps(dict(reversed(list(request.items()))), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"
        second = self.f.prepare(raw=alias_raw)
        self.assertNotEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.envelope()["context_sha256"], second.envelope()["context_sha256"])
        self.assertEqual(second.raw_assets()["cpu_request"], alias_raw)
        self.assertEqual(first.envelope()["query_sha256"], second.envelope()["query_sha256"])
        self.assert_denied(second.audit())

    def test_literal_canonical_retains_u64_and_korean(self):
        expected = '["rz-pals-private-strategic-action/1",{"message":"사전 요청","sequence":9007199254740993}]'.encode("utf-8")
        self.assertEqual(action._canonical([action.SCHEMA, {"sequence": 9007199254740993, "message": "사전 요청"}]), expected)
        with self.assertRaises(ValueError):
            action._canonical({"sequence": 9007199254740993.0})

    def test_embedded_original_json_can_exceed_metadata_string_bound(self):
        original = query.canonical(self.f.request()) + b" " * 9000
        prepared = self.f.prepare(raw=original)
        self.assertEqual(prepared.envelope()["cpu_request_raw"].encode("utf-8"), original)
        receipt, _ = self.f.receipt(prepared)
        cpu_raw = receipt["cpu_response_raw"].encode("utf-8") + b" " * 9000
        receipt.update(cpu_response_raw=cpu_raw.decode("utf-8"), cpu_response_artifact=query.byte_pin(cpu_raw))
        checked = self.f.consume(prepared, receipt)
        self.assertEqual(action._parse(checked.raw_assets()["result"])["cpu_response_raw"].encode("utf-8"), cpu_raw)
        self.assert_denied(checked.observation())

    def test_outer_escaped_envelope_has_separate_preflight_request_cap(self):
        request = query.canonical(self.f.request())
        original = request + b" " * (action.MAX_CPU_REQUEST_BYTES - 1 - len(request))
        self.assertLess(len(original), action.MAX_CPU_REQUEST_BYTES)
        with patch.object(action, "_canonical", wraps=action._canonical) as canonical:
            with self.assertRaises(ValueError):
                self.f.prepare(raw=original)
        final_body, maximum = canonical.call_args.args
        self.assertEqual(maximum, action.MAX_CPU_REQUEST_BYTES)
        self.assertEqual(final_body["cpu_request_raw"].encode("utf-8"), original)
        # Exercise the exact prepaid boundary without serializing the oversized
        # final envelope. The original CPU request is neither cut nor rewritten.
        with patch.object(action.json, "dumps", side_effect=AssertionError("no envelope serialization")) as serialize:
            with self.assertRaises(ValueError):
                action._canonical(final_body, maximum)
        serialize.assert_not_called()

    def test_invalid_before_sequence_cannot_create_a_new_checked_action_query(self):
        self.f.base.before["before_sequence"] = 0
        self.f.base.before_raw = query.canonical(self.f.base.before)
        self.f.base.pins["before_result"] = query.byte_pin(self.f.base.before_raw)
        with self.assertRaises(ValueError):
            self.f.base.admit()

    def test_unchecked_query_bool_or_out_of_range_index_refused(self):
        raw = query.canonical(self.f.request())
        for checked, index in ((object(), 0), (self.f.checked, True), (self.f.checked, -1), (self.f.checked, 2)):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    action.prepare_strategic_cpu_action(checked_query=checked, action_index=index,
                        cpu_request_bytes=raw, expected_cpu_request_pin=query.byte_pin(raw))

    def test_missing_mutable_oversized_request_refused_before_query_verify(self):
        for raw in (None, b"", bytearray(b"{}"), b"x" * (action.MAX_CPU_REQUEST_BYTES + 1)):
            with self.subTest(kind=type(raw).__name__):
                with patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no query operation")) as check:
                    with self.assertRaises(ValueError):
                        action.prepare_strategic_cpu_action(checked_query=self.f.checked, action_index=0,
                            cpu_request_bytes=raw, expected_cpu_request_pin={"bytes": 1, "sha256": "0" * 64})
                check.assert_not_called()

    def test_closed_expected_pin_refused_before_shared_helper_or_serialization(self):
        raw = query.canonical(self.f.request())
        oversized_keys = {str(index): None for index in range(2048)}
        for pin in (oversized_keys, {"bytes": True, "sha256": "0" * 64}, {"bytes": 1, "sha256": "x" * 8192}):
            with self.subTest(kind=len(pin)):
                with patch.object(query, "_pin", side_effect=AssertionError("no shared key collection")) as shared:
                    with patch.object(action.json, "dumps", side_effect=AssertionError("no serialization")) as serialize:
                        with self.assertRaises(ValueError):
                            action.prepare_strategic_cpu_action(checked_query=self.f.checked, action_index=0,
                                cpu_request_bytes=raw, expected_cpu_request_pin=pin)
                shared.assert_not_called()
                serialize.assert_not_called()

    def test_serialization_credit_refused_before_dumps_with_repeated_references(self):
        small = "x" * 4096
        with patch.object(action.json, "dumps", side_effect=AssertionError("no oversized serialization")) as serialize:
            with self.assertRaises(ValueError):
                action._canonical({"repeated": [small] * 2048})
        serialize.assert_not_called()

    def test_request_pin_duplicate_json_and_nonfinite_fields_refused(self):
        raw = query.canonical(self.f.request())
        with self.assertRaisesRegex(ValueError, "independent original byte pin"):
            action.prepare_strategic_cpu_action(checked_query=self.f.checked, action_index=0,
                cpu_request_bytes=raw + b"\n", expected_cpu_request_pin=query.byte_pin(raw))
        for raw in (b'{"schema":"a","schema":"b"}', b'{"schema":NaN}'):
            with self.assertRaises(ValueError):
                self.f.prepare(raw=raw)

    def test_actual_dispatcher_equal_horizon_unsupported_without_changing_query(self):
        self.f.base.profiles[0] = self.f.base.profile(horizon=2)
        self.f.base.actions = [self.f.base.action("resume_task"), self.f.base.action("defer")]
        self.f.base.refresh()
        self.f.checked = self.f.base.admit()
        self.assertEqual(self.f.checked.catalogue()["actions"][0]["baseline_depth"], 2)
        self.assertEqual(self.f.checked.catalogue()["actions"][0]["requested_depth"], 2)
        with self.assertRaises(ValueError):
            self.f.prepare()

    def test_selected_task_output_and_no_wall_clipping(self):
        original = self.f.request()
        for name, value in (("task", "defer"), ("max_output_bytes", 32768), ("max_wall_time_ms", 9999),
                            ("max_nodes_per_check", 99999), ("requested_depth", 3), ("tt_entries", 32), ("quiescence_ply", 3)):
            with self.subTest(name=name):
                request = copy.deepcopy(original)
                request[name] = value
                request.pop("context_sha256")
                request["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, request)
                with self.assertRaises(ValueError):
                    self.f.prepare(raw=query.canonical(request))

    def test_parent_branch_rules_and_registered_binary_identity_refused(self):
        original = self.f.request()
        for name, value in (("parent_input_sha256", "0" * 64), ("branch_sha256", "0" * 64),
                            ("rules_state_sha256", "0" * 64), ("rules_history_sha256", "0" * 64),
                            ("cpu_binary_sha256", "0" * 64), ("prefix", [self.f.base.feedback.snapshot["legal_moves"][0]])):
            with self.subTest(name=name):
                request = copy.deepcopy(original)
                request[name] = value
                request.pop("context_sha256")
                request["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, request)
                with self.assertRaises(ValueError):
                    self.f.prepare(raw=query.canonical(request))

    def test_prepared_parent_mutation_revalidated_after_creation(self):
        prepared = self.f.prepare()
        self.f.base.parents.records[self.f.base.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            prepared.verify()

    def test_completed_cpu_binding_is_not_query_execution_or_utility_authority(self):
        prepared = self.f.prepare()
        receipt, response = self.f.receipt(prepared)
        receipt["binary_pin_scope"] = "linux_loaded_executable_inode"
        checked = self.f.consume(prepared, receipt)
        self.assertIs(checked.verify(), checked)
        value = checked.observation()
        self.assertEqual(value["status"], "cpu_binding_observation")
        self.assertTrue(value["cpu_reports_complete"])
        self.assertEqual(value["reported_cpu_nodes"], response["nodes"])
        self.assert_denied(value)
        self.assertIs(value["reported_cpu_nodes_are_whole_action_cost"], False)
        self.assertEqual(checked.raw_assets()["result"], action._canonical(receipt) + b"\n")
        detached = checked.observation()
        detached["raw_cpu_response"]["nodes"] = 0
        self.assertEqual(checked.observation()["raw_cpu_response"]["nodes"], response["nodes"])
        self.assertFalse(hasattr(checked, "collate"))

    def test_partial_deadline_and_defer_receipts_preserve_unknown(self):
        for partial, deadline in ((True, False), (True, True), (False, True)):
            with self.subTest(partial=partial, deadline=deadline):
                prepared = self.f.prepare()
                receipt, response = self.f.receipt(prepared, partial=partial, deadline=deadline)
                checked = self.f.consume(prepared, receipt)
                value = checked.observation()
                self.assertEqual(value["status"], "unresolved")
                self.assertFalse(value["cpu_reports_complete"])
                self.assertEqual(value["raw_cpu_response"], response)
                self.assert_denied(value)
        prepared = self.f.prepare(1)
        receipt, _ = self.f.receipt(prepared)
        value = self.f.consume(prepared, receipt).observation()
        self.assertEqual(value["raw_cpu_response"]["status"], "deferred")
        self.assertFalse(value["cpu_reports_complete"])
        self.assertEqual(value["reported_cpu_nodes"], 0)
        self.assert_denied(value)

    def test_typed_failed_raw_known_work_and_none_preserved_without_double_sum(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                prepared = self.f.prepare()
                error = self.f.failed(prepared, unknown_nodes=unknown)
                checked = self.f.consume(prepared, error)
                value = checked.observation()
                self.assertEqual(value["status"], "unresolved")
                self.assertEqual(value["reported_cpu_nodes"], error["cpu_error"]["known_nodes"])
                self.assertEqual(value["raw_cpu_error"], error)
                self.assertIsNone(value["raw_cpu_response"])
                self.assert_denied(value)

    def test_failed_cpu_work_lower_bound_and_per_check_caps_refused(self):
        prepared = self.f.prepare()
        original = self.f.failed(prepared)
        self.assertGreater(original["cpu_error"]["baseline"]["nodes"], 0)
        maximum = self.f.request()["max_nodes_per_check"]
        for change in ("below_retained_reports", "known_over_two_checks", "failed_check_over_cap", "duplicated_after_work"):
            with self.subTest(change=change):
                error = copy.deepcopy(original)
                inner = error["cpu_error"]
                if change == "below_retained_reports":
                    inner["known_nodes"] = 0
                    inner["failed_check_work"]["nodes"] = 0
                    inner["failed_check_work"]["quiescence_nodes"] = 0
                elif change == "known_over_two_checks":
                    inner["known_nodes"] = 2 * maximum + 1
                elif change == "failed_check_over_cap":
                    inner["failed_check_work"]["nodes"] = maximum + 1
                    inner["known_nodes"] = inner["baseline"]["nodes"] + maximum + 1
                else:
                    _, response = self.f.receipt(prepared)
                    inner["after"] = response["after"]
                    error["receipt"]["after_present"] = True
                error["receipt"]["actual_cpu_nodes"] = inner["known_nodes"]
                with self.assertRaises(ValueError):
                    self.f.consume(prepared, error)

    def test_failed_nodes_without_raw_report_and_unknown_with_work_preserved(self):
        prepared = self.f.prepare()
        error = self.f.failed(prepared)
        error["cpu_error"]["known_nodes"] = None
        error["receipt"]["actual_cpu_nodes"] = None
        observed = self.f.consume(prepared, error).observation()
        self.assertIsNone(observed["reported_cpu_nodes"])
        self.assertIsNotNone(observed["raw_cpu_error"]["cpu_error"]["failed_check_work"])
        self.assertEqual(observed["status"], "unresolved")
        self.assert_denied(observed)
        # Baseline RawReport construction may fail while Rust retains the actual
        # report's node count. Absent report/work bytes are not fabricated here.
        error = self.f.failed(prepared, unknown_nodes=True)
        error["cpu_error"]["known_nodes"] = 11
        error["receipt"]["actual_cpu_nodes"] = 11
        observed = self.f.consume(prepared, error).observation()
        self.assertEqual(observed["reported_cpu_nodes"], 11)
        self.assertIsNone(observed["raw_cpu_error"]["cpu_error"]["baseline"])
        self.assertEqual(observed["status"], "unresolved")
        self.assert_denied(observed)

    def test_outer_output_or_deadline_error_preserves_actual_successful_cpu_raw(self):
        prepared = self.f.prepare()
        receipt, response = self.f.receipt(prepared)
        for stage, deadline in (("stdout", False), ("receipt_serialization", False), ("deadline", True)):
            with self.subTest(stage=stage):
                error = {"code": "strategic_action_failed", "stage": stage, "message": "synthetic outer publication failure",
                    "elapsed_ms": receipt["elapsed_ms"], "deadline_exceeded": deadline, "cpu_error": None, "receipt": receipt}
                value = self.f.consume(prepared, error).observation()
                self.assertEqual(value["status"], "unresolved")
                self.assertEqual(value["raw_cpu_response"], response)
                self.assertEqual(value["reported_cpu_nodes"], response["nodes"])
                self.assertFalse(value["cpu_reports_complete"])
                self.assert_denied(value)

    def test_failure_without_bound_receipt_stays_unknown(self):
        prepared = self.f.prepare()
        error = {"code": "strategic_action_failed", "stage": "decode", "message": "synthetic pre-dispatch rejection",
            "elapsed_ms": None, "deadline_exceeded": False, "cpu_error": None, "receipt": None}
        value = self.f.consume(prepared, error).observation()
        self.assertEqual(value["reason"], "strategic_failure_without_bound_receipt")
        self.assertIsNone(value["cpu_dispatch_function_entered"])
        self.assert_denied(value)

    def test_standalone_failed_receipt_and_mutated_error_summary_refused(self):
        prepared = self.f.prepare()
        error = self.f.failed(prepared)
        with self.assertRaisesRegex(ValueError, "typed inner CPU error"):
            self.f.consume(prepared, error["receipt"])
        for name, value in (("actual_cpu_nodes", 0), ("baseline_present", False), ("after_present", True)):
            with self.subTest(name=name):
                changed = copy.deepcopy(error)
                changed["receipt"][name] = value
                with self.assertRaisesRegex(ValueError, "raw summary"):
                    self.f.consume(prepared, changed)
        changed = copy.deepcopy(error)
        changed["receipt"]["cpu_dispatch_attempted"] = False
        with self.assertRaisesRegex(ValueError, "dispatcher entry"):
            self.f.consume(prepared, changed)

    def test_result_identity_authority_and_bool_aliases_refused(self):
        prepared = self.f.prepare()
        receipt, _ = self.f.receipt(prepared)
        for name, value in (("query_sha256", "0" * 64), ("context_sha256", "0" * 64),
                            ("prior_ledger_sha256", "0" * 64), ("scope", "whole_action_complete"),
                            ("catalogue_artifact", {"bytes": 1, "sha256": "0" * 64}),
                            ("before_result_artifact", {"bytes": 1, "sha256": "0" * 64}),
                            ("cpu_dispatch_attempted", 1), ("actual_cpu_nodes", True)):
            with self.subTest(name=name):
                changed = copy.deepcopy(receipt)
                changed[name] = value
                with self.assertRaises(ValueError):
                    self.f.consume(prepared, changed)
        changed = copy.deepcopy(receipt)
        changed["action"]["slot"] = False
        with self.assertRaises(ValueError):
            self.f.consume(prepared, changed)
        for name in action._DENIED:
            changed = copy.deepcopy(receipt)
            changed[name] = True
            with self.assertRaisesRegex(ValueError, "expand authority"):
                self.f.consume(prepared, changed)

    def test_result_raw_pin_original_cpu_pin_summary_and_elapsed_refused(self):
        prepared = self.f.prepare()
        receipt, _ = self.f.receipt(prepared)
        for change in ("raw_pin", "summary", "elapsed", "inner_identity"):
            with self.subTest(change=change):
                changed = copy.deepcopy(receipt)
                if change == "raw_pin":
                    changed["cpu_response_raw"] += "\n"
                elif change == "summary":
                    changed["actual_cpu_nodes"] += 1
                elif change == "elapsed":
                    changed["elapsed_ms"] = 0
                else:
                    response = query._parse(changed["cpu_response_raw"].encode("utf-8"))
                    response["context_sha256"] = "0" * 64
                    raw = query.canonical(response)
                    changed["cpu_response_raw"], changed["cpu_response_artifact"] = raw.decode("utf-8"), query.byte_pin(raw)
                with self.assertRaises(ValueError):
                    self.f.consume(prepared, changed)

    def test_result_duplicate_float_unknown_fields_and_independent_pin_refused(self):
        prepared = self.f.prepare()
        for raw in (b'{"schema":"x","schema":"y"}', b'{"elapsed_ms":1.5}', b'{"code":"unknown"}'):
            with self.assertRaises(ValueError):
                action.admit_strategic_cpu_action_receipt(prepared_action=prepared, receipt_bytes=raw,
                    expected_receipt_pin=query.byte_pin(raw))
        receipt, _ = self.f.receipt(prepared)
        raw = action._canonical(receipt)
        with self.assertRaisesRegex(ValueError, "independent original byte pin"):
            action.admit_strategic_cpu_action_receipt(prepared_action=prepared, receipt_bytes=raw + b"\n",
                expected_receipt_pin=query.byte_pin(raw))

    def test_result_pin_guard_and_original_output_extent_before_parse(self):
        prepared = self.f.prepare()
        with patch.object(query, "_pin", side_effect=AssertionError("no shared collection")) as shared:
            with self.assertRaises(ValueError):
                action.admit_strategic_cpu_action_receipt(prepared_action=prepared, receipt_bytes=b"{}",
                    expected_receipt_pin={str(index): None for index in range(2048)})
        shared.assert_not_called()
        raw = b"x" * (2 * self.f.request()["max_output_bytes"] + 1)
        with patch.object(action, "_parse", side_effect=AssertionError("no result parse")) as parse:
            with self.assertRaisesRegex(ValueError, "outer result output extent"):
                action.admit_strategic_cpu_action_receipt(prepared_action=prepared, receipt_bytes=raw,
                    expected_receipt_pin=query.byte_pin(raw))
        parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
