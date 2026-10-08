"""Synthetic strict handoff wiring; no actual child, replay or model proof.

The existing strict parent, Query/2, PreparedAction and semantic factories are
used unchanged. Synthetic Rules descriptors/registration/launch originals are
kept byte-exact. Nothing below runs Rust, creates a launcher, forwards a model
or produces utility/targets. Imports of fixture modules expose no TestCase
class aliases to unittest discovery.
"""

import copy
import hashlib
import json
import tempfile
import time
import unittest
from unittest.mock import patch

from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import strategic_cpu_action as action
from rz_pals_model import strategic_replay_caller as caller
from rz_pals_model import strategic_verifier_query as query
from rz_pals_model import verifier_producer as legacy
import test_strategic_verifier_query as query_fixtures


class ReplayCallerFixture:
    def __init__(self, directory):
        self.base = query_fixtures.QueryFixture(directory)
        self.parents, self.index = self.base.parents, self.base.index
        self.semantic_fixtures = []
        for nodes in (100000, 90000):
            original = self.base.feedback.base.semantic
            fixture = copy.copy(original)
            fixture.raws = copy.deepcopy(original.raws)
            fixture.common = copy.deepcopy(original.common)
            fixture.registration = copy.deepcopy(original.registration)
            fixture.receipt = copy.deepcopy(original.receipt)
            profile = query._parse(self.base.profiles[0]["registration"])
            fixture.common.update(cpu_profile_sha256=profile["profile_sha256"], known_completed_depth=0,
                baseline_depth=2, requested_depth=4, max_nodes_per_check=nodes, max_wall_time_ms=10000)
            # Existing source fixture deliberately supplies synthetic target
            # Rules observations. This is wiring, not Python chess replay.
            fixture.restrict(list(fixture.snapshot["legal_moves"]))
            self.semantic_fixtures.append(fixture)
        self.rebuild()

    def rebuild(self):
        self.base.actions = []
        for fixture in self.semantic_fixtures:
            checked = fixture.admit()
            common = checked.common_query()
            spec = {"slot": 0, "task": "defend_response", "semantic_input_sha256": checked.sha256,
                "profile_registration": 0, "max_output_bytes": 65536}
            spec.update({name: common[name] for name in ("baseline_depth", "requested_depth", "max_nodes_per_check",
                                                       "max_wall_time_ms", "budget_bucket")})
            self.base.actions.append((spec, checked))
        self.base.refresh()
        self.checked = self.base.admit()
        self.prepared = self.prepare_action(0)

    def request(self, index):
        spec = self.checked.catalogue()["actions"][index]
        common = self.checked.action_semantic_inputs()[index].common_query()
        value = self.base.feedback.request("defend_response")
        value.update(prefix=list(common["prefix"]), root_moves=list(common["root_moves"]))
        for name in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "max_output_bytes"):
            value[name] = spec[name]
        value["branch_sha256"] = legacy._hash("rz-pals-private-cpu-branch/1", {
            "parent_input_sha256": value["parent_input_sha256"], "prefix": value["prefix"], "root_moves": value["root_moves"]})
        value.pop("context_sha256", None)
        value["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, value)
        return value

    def prepare_action(self, index, raw=None):
        raw = query.canonical(self.request(index)) + b"\n" if raw is None else raw
        return action.prepare_strategic_cpu_action(checked_query=self.checked, action_index=index,
            cpu_request_bytes=raw, expected_cpu_request_pin=query.byte_pin(raw))

    def arguments(self, index=0):
        fixture = self.semantic_fixtures[index]
        return {"parents": self.parents, "parent_index": self.index, "checked_query": self.checked,
            "action_index": index, "prepared_action": self.prepared if index == 0 else self.prepare_action(index),
            "semantic_raws": tuple(fixture.raws[name] for name in caller.SEMANTIC_RAW_NAMES),
            "expected_semantic_pins": copy.deepcopy(fixture.pins),
            "expected_accepted_binary_pin_scope": "linux_loaded_executable_inode", "deadline": time.monotonic() + 60}

    def admit(self, **updates):
        arguments = self.arguments()
        arguments.update(updates)
        return caller.prepare_replay_caller_bindings(**arguments)


class StrategicReplayCallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = ReplayCallerFixture(self.temp.name)

    def test_strict_factories_make_partial_historical_handoff_only(self):
        checked = self.f.admit()
        self.assertIs(checked.verify(), checked)
        values = checked.constructor_inputs()
        self.assertEqual(tuple(values["originals_available"]), ("prepared_action", "semantic_receipt"))
        self.assertNotIn("registration", values["originals_available"])
        self.assertEqual(values["original_semantic_registration"], self.f.semantic_fixtures[0].raws["registration"])
        self.assertEqual(values["originals_available"]["semantic_receipt"], self.f.semantic_fixtures[0].raws["receipt"])
        self.assertEqual(values["originals_available"]["prepared_action"], self.f.prepared.envelope_bytes())
        self.assertEqual(values["semantic_receipt_producer_scope_variant"], "LinuxLoadedExecutableInode")
        self.assertIs(values["complete_replay_expected_pins"], False)
        self.assertIn("replay_consumer_registration_original_bytes_and_independent_pin", values["required_external"])
        for key in caller._DENIED:
            self.assertIs(checked.audit()[key], False)
        self.assertEqual(checked.audit()["actual_utility_groups"], 0)
        self.assertFalse(hasattr(checked, "dispatch"))

    def test_original_deadline_and_exact_resource_declarations_are_not_converted(self):
        arguments = self.f.arguments()
        original = self.f.prepared.envelope()
        checked = caller.prepare_replay_caller_bindings(**arguments)
        self.assertEqual(checked.audit()["original_absolute_deadline"], arguments["deadline"])
        self.assertEqual(action._parse(checked.raw_assets()["prepared_action"])["remaining"], original["remaining"])
        self.assertEqual(action._parse(checked.raw_assets()["prepared_action"])["action"], original["action"])
        for name in ("mode", "config", "resources", "outer_wire", "cli_expected"):
            self.assertNotIn(name, checked.constructor_inputs())

    def test_factory_only_immutable_checked_object(self):
        with self.assertRaises(caller.ReplayCallerRefusal):
            caller.CheckedReplayCallerBindings()
        checked = self.f.admit()
        with self.assertRaises(AttributeError):
            checked._deadline = time.monotonic() + 600

    def test_returned_metadata_is_detached_from_capability_and_caller(self):
        arguments = self.f.arguments()
        checked = caller.prepare_replay_caller_bindings(**arguments)
        identity = checked.sha256
        arguments["expected_semantic_pins"]["receipt"]["sha256"] = "0" * 64
        values = checked.constructor_inputs()
        values["checked_historical_bindings"]["parent"]["current_view_sha256"] = "0" * 64
        assets = checked.raw_assets()
        assets["semantic_originals"]["registration"] = b"replacement"
        assets["expected_semantic_pins"]["receipt"]["sha256"] = "0" * 64
        self.assertEqual(checked.sha256, identity)

    def test_missing_originals_cannot_be_recovered_from_public_rules_dict(self):
        self.assertIsInstance(self.f.checked.action_semantic_inputs()[0].rules_receipt(), dict)
        for raws in (None, {}, (), self.f.arguments()["semantic_raws"][:-1]):
            with self.subTest(kind=type(raws).__name__):
                with self.assertRaises(caller.ReplayCallerRefusal):
                    self.f.admit(semantic_raws=raws)

    def test_original_tuple_order_and_mutable_raws_are_refused(self):
        original = self.f.arguments()["semantic_raws"]
        for raws in (list(original), (bytearray(original[0]), *original[1:]),
                     (original[1], original[0], *original[2:])):
            with self.subTest(kind=type(raws).__name__):
                with self.assertRaises(caller.ReplayCallerRefusal):
                    self.f.admit(semantic_raws=raws)

    def test_independent_original_pin_hash_and_extent_are_required(self):
        for name in caller.SEMANTIC_RAW_NAMES:
            pins = self.f.arguments()["expected_semantic_pins"]
            pins[name]["sha256"] = "0" * 64
            with self.subTest(name=name), self.assertRaises(caller.ReplayCallerRefusal):
                self.f.admit(expected_semantic_pins=pins)
        pins = self.f.arguments()["expected_semantic_pins"]
        pins["receipt"]["bytes"] = True
        with self.assertRaises(caller.ReplayCallerRefusal):
            self.f.admit(expected_semantic_pins=pins)

    def test_exact_initialized_query_and_prepared_types_are_required(self):
        class QuerySubclass(query.CheckedStrategicQuery):
            pass
        class PreparedSubclass(action.PreparedStrategicCpuAction):
            pass
        for updates in ({"checked_query": object()}, {"prepared_action": object()},
                        {"checked_query": object.__new__(QuerySubclass)},
                        {"prepared_action": object.__new__(PreparedSubclass)},
                        {"checked_query": object.__new__(query.CheckedStrategicQuery)},
                        {"prepared_action": object.__new__(action.PreparedStrategicCpuAction)}):
            with self.subTest(updates=tuple(updates)), self.assertRaises(caller.ReplayCallerRefusal):
                self.f.admit(**updates)

    def test_selected_action_and_prepared_owner_cannot_be_swapped(self):
        with self.assertRaises(caller.ReplayCallerRefusal):
            self.f.admit(action_index=1)
        for value in (True, -1, 8):
            with self.subTest(index=value), self.assertRaises(caller.ReplayCallerRefusal):
                self.f.admit(action_index=value)
        arguments = self.f.arguments(1)
        self.assertEqual(caller.prepare_replay_caller_bindings(**arguments).constructor_inputs()["checked_historical_bindings"]["action_index"], 1)

    def test_different_valid_semantic_slot_originals_cannot_bind_selected_action(self):
        other = self.f.arguments(1)
        with self.assertRaisesRegex(caller.ReplayCallerRefusal, "selected_original_semantic_mismatch"):
            self.f.admit(semantic_raws=other["semantic_raws"], expected_semantic_pins=other["expected_semantic_pins"])

    def test_non_replay_task_is_explicitly_unsupported(self):
        # Reuse existing admitted Resume/Defer fixture; neither is relabeled as
        # the separate first supported defend_response replay task.
        other_root = self.temp.name + "/other"
        import pathlib
        pathlib.Path(other_root).mkdir()
        base = query_fixtures.QueryFixture(other_root)
        checked = base.admit()
        request = base.feedback.request("resume_task")
        raw = query.canonical(request)
        prepared = action.prepare_strategic_cpu_action(checked_query=checked, action_index=0,
            cpu_request_bytes=raw, expected_cpu_request_pin=query.byte_pin(raw))
        arguments = self.f.arguments()
        arguments.update(parents=base.parents, parent_index=base.index, checked_query=checked, prepared_action=prepared)
        with self.assertRaisesRegex(caller.ReplayCallerRefusal, "unsupported_replay_task"):
            caller.prepare_replay_caller_bindings(**arguments)

    def test_parent_object_index_current_frozen_and_encoding_are_checked(self):
        with self.assertRaises(caller.ReplayCallerRefusal):
            self.f.admit(parents=copy.copy(self.f.parents))
        with self.assertRaises(caller.ReplayCallerRefusal):
            self.f.admit(parent_index=self.f.index + 1)
        for key in caller.PARENT_PIN_NAMES:
            pins = self.f.arguments()["expected_semantic_pins"]
            pins[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(caller.ReplayCallerRefusal):
                self.f.admit(expected_semantic_pins=pins)

    def test_explicit_scope_mismatch_never_uses_result_or_host_os_fallback(self):
        for scope in (None, "library_dispatcher_argument", "LibraryDispatcherArgument", "unknown", True,
                      "current_exe_path_hash"):
            with self.subTest(scope=scope), self.assertRaises(caller.ReplayCallerRefusal):
                self.f.admit(expected_accepted_binary_pin_scope=scope)

    def test_windows_registered_scope_is_supported_only_when_explicitly_expected(self):
        for fixture in self.f.semantic_fixtures:
            fixture.registration.update(platform="windows", accepted_binary_pin_scope="current_exe_path_hash")
            fixture.receipt["binary_pin_scope"] = "current_exe_path_hash"
            fixture.reseal()
        self.f.rebuild()
        checked = self.f.admit(expected_accepted_binary_pin_scope="current_exe_path_hash")
        self.assertEqual(checked.constructor_inputs()["semantic_receipt_producer_scope_variant"], "CurrentExePathHash")

    def test_original_receipt_lexical_alias_is_kept_without_rerendering(self):
        fixture = self.f.semantic_fixtures[0]
        value = json.loads(fixture.raws["receipt"].decode("utf-8"))
        fixture.raws["receipt"] = json.dumps(dict(reversed(list(value.items()))), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"
        launch = semantic._parse(fixture.raws["launch_observation"])
        launch["receipt"] = semantic.byte_pin(fixture.raws["receipt"])
        fixture.raws["launch_observation"] = semantic.canonical(launch)
        for name in caller.SEMANTIC_RAW_NAMES:
            fixture.pins[name] = semantic.byte_pin(fixture.raws[name])
        self.f.rebuild()
        checked = self.f.admit()
        self.assertEqual(checked.constructor_inputs()["originals_available"]["semantic_receipt"], fixture.raws["receipt"])
        self.assertNotEqual(semantic.canonical(value), fixture.raws["receipt"])

    def test_cpu_request_lexical_alias_is_original_prepared_byte_identity(self):
        raw = json.dumps(dict(reversed(list(self.f.request(0).items()))), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"
        prepared = self.f.prepare_action(0, raw=raw)
        checked = self.f.admit(prepared_action=prepared)
        self.assertEqual(checked.raw_assets()["cpu_request"], raw)
        self.assertEqual(checked.raw_assets()["expected_cpu_request_pin"], query.byte_pin(raw))
        self.assertEqual(checked.constructor_inputs()["originals_available"]["prepared_action"], prepared.envelope_bytes())

    def test_rerendered_rules_dict_cannot_replace_independently_pinned_original(self):
        raw = semantic.canonical(self.f.checked.action_semantic_inputs()[0].rules_receipt())
        originals = self.f.arguments()["semantic_raws"]
        with self.assertRaises(caller.ReplayCallerRefusal):
            self.f.admit(semantic_raws=(originals[0], raw, *originals[2:]))

    def test_invalid_and_expired_deadlines_refuse_before_getters_serialization_or_factory(self):
        for deadline in (float("nan"), float("inf"), -float("inf"), True, "later", None, time.monotonic() - 1):
            with self.subTest(deadline=repr(deadline)), \
                    patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no getter")) as getter, \
                    patch.object(query, "canonical", side_effect=AssertionError("no serialization")) as serialize, \
                    patch.object(semantic, "admit_semantic_input", side_effect=AssertionError("no factory")) as factory:
                with self.assertRaises(caller.ReplayCallerRefusal):
                    self.f.admit(deadline=deadline)
                getter.assert_not_called()
                serialize.assert_not_called()
                factory.assert_not_called()

    def test_same_length_tamper_of_each_original_refuses_before_getters_canonical_or_factory(self):
        originals = self.f.arguments()["semantic_raws"]
        for index, name in enumerate(caller.SEMANTIC_RAW_NAMES):
            raw = originals[index]
            tampered = bytes((raw[0] ^ 1,)) + raw[1:]
            self.assertEqual(len(tampered), len(raw))
            changed = (*originals[:index], tampered, *originals[index + 1:])
            with self.subTest(original=name), \
                    patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no getter")) as getter, \
                    patch.object(query, "canonical", side_effect=AssertionError("no serialization")) as serialize, \
                    patch.object(semantic, "admit_semantic_input", side_effect=AssertionError("no factory")) as factory:
                with self.assertRaises(caller.ReplayCallerRefusal) as refused:
                    self.f.admit(semantic_raws=changed)
                self.assertEqual(refused.exception.code, "independent_original_pin_mismatch")
                self.assertEqual(refused.exception.stage, name)
                getter.assert_not_called()
                serialize.assert_not_called()
                factory.assert_not_called()

    def test_large_alias_pin_metadata_refuses_before_json_or_capability_operation(self):
        pins = self.f.arguments()["expected_semantic_pins"]
        pins["parent_input_sha256"] = ["x" * 8192] * 1000
        with patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no getter")) as getter, \
                patch.object(query.json, "dumps", side_effect=AssertionError("no serialization")) as serialize:
            with self.assertRaises(caller.ReplayCallerRefusal):
                self.f.admit(expected_semantic_pins=pins)
        getter.assert_not_called()
        serialize.assert_not_called()

    def test_unknown_large_metadata_keys_refuse_before_key_set_copy(self):
        pins = self.f.arguments()["expected_semantic_pins"]
        pins.update({"extra" + str(index): None for index in range(32)})
        with patch.object(query, "canonical", side_effect=AssertionError("no serialization")) as serialize:
            with self.assertRaises(caller.ReplayCallerRefusal):
                self.f.admit(expected_semantic_pins=pins)
        serialize.assert_not_called()

    def test_raw_and_scratch_ledgers_reserve_before_capability_getters(self):
        for name in ("MAX_RAW_BYTES", "MAX_SCRATCH_BYTES", "MAX_PIN_METADATA_BYTES"):
            with self.subTest(limit=name), patch.object(caller, name, 1), \
                    patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no getter")) as getter:
                with self.assertRaises(caller.ReplayCallerRefusal):
                    self.f.admit()
                getter.assert_not_called()

    def test_raw_credit_charges_repeated_immutable_references(self):
        shared = b"x" * (1024 * 1024)
        with patch.object(query.CheckedStrategicQuery, "verify", side_effect=AssertionError("no getter")) as getter:
            with self.assertRaisesRegex(caller.ReplayCallerRefusal, "semantic_original_aggregate_extent"):
                self.f.admit(semantic_raws=(shared,) * 7)
        getter.assert_not_called()

    def test_each_getter_reuses_original_deadline_without_renewal(self):
        arguments = self.f.arguments()
        checked = caller.prepare_replay_caller_bindings(**arguments)
        for operation in (checked.verify, lambda: checked.sha256, checked.raw_assets, checked.constructor_inputs, checked.audit):
            with self.subTest(operation=repr(operation)), patch.object(caller.time, "monotonic", return_value=arguments["deadline"]):
                with self.assertRaisesRegex(caller.ReplayCallerRefusal, "expired_original_deadline"):
                    operation()

    def test_current_parent_mutation_invalidates_owned_handoff(self):
        checked = self.f.admit()
        self.f.parents.records[self.f.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(caller.ReplayCallerRefusal):
            checked.verify()

    def test_existing_factory_end_crossing_deadline_refuses_no_fresh_clock(self):
        arguments = self.f.arguments()
        now = time.monotonic()
        arguments["deadline"] = now + 60
        original = semantic.admit_semantic_input
        state = [now]
        def delayed(**kwargs):
            result = original(**kwargs)
            state[0] = arguments["deadline"]
            return result
        with patch.object(caller.time, "monotonic", side_effect=lambda: state[0]), \
                patch.object(semantic, "admit_semantic_input", side_effect=delayed):
            with self.assertRaisesRegex(caller.ReplayCallerRefusal, "expired_original_deadline"):
                caller.prepare_replay_caller_bindings(**arguments)

    def test_factory_reuses_original_semantic_admission_tuple_and_independent_pins(self):
        arguments = self.f.arguments()
        original = semantic.admit_semantic_input
        with patch.object(semantic, "admit_semantic_input", wraps=original) as factory:
            checked = caller.prepare_replay_caller_bindings(**arguments)
        supplied = factory.call_args.kwargs
        self.assertIs(supplied["parent"], arguments["parents"])
        for name, raw in zip(caller.SEMANTIC_RAW_NAMES, arguments["semantic_raws"]):
            self.assertIs(supplied[name + "_bytes"], raw)
        self.assertEqual(supplied["expected_pins"], arguments["expected_semantic_pins"])
        self.assertEqual(checked.constructor_inputs()["checked_historical_bindings"]["semantic_input_sha256"], self.f.checked.action_semantic_inputs()[0].sha256)

    def test_handoff_identity_is_byte_binding_without_complete_replay_authority(self):
        checked = self.f.admit()
        self.assertEqual(len(checked.sha256), 64)
        self.assertEqual(hashlib.sha256(checked._views()[2]).hexdigest(), checked.sha256)
        self.assertEqual(checked.audit()["ledger"]["scope"], "own_handoff_ledger_only_not_resident_or_rss_cap")
        self.assertFalse(hasattr(checked, "outer_request"))


if __name__ == "__main__":
    unittest.main()
