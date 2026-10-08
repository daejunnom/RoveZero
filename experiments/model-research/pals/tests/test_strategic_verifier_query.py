"""Synthetic query/2 wiring and refusal fixtures, never live execution proof.

Existing strict parent and CheckedSemanticInput factories are exercised. All
CPU/profile/launch/source/order observations are synthetic caller fixtures.
No model operation, checker child, Rules replay, optimizer or reward is run.
"""

import copy
import tempfile
import unittest
from unittest.mock import patch

from rz_pals_model import strategic_verifier_query as query
from rz_pals_model import semantic_verifier as semantic
from rz_pals_model import verifier_producer as legacy
from test_verifier_feedback import FeedbackFixture, synthetic_process


class QueryFixture:
    def __init__(self, directory):
        self.feedback = FeedbackFixture(directory)
        self.parents = self.feedback.parents
        self.index = self.feedback.index
        self.parent_pins = self.feedback.base.semantic.checked.parent_pins()
        self.profiles = [self.profile()]
        self.prior = []
        self.actions = [self.action("resume_task"), self.action("defer")]
        self.refresh()

    def profile(self, name=legacy.PROFILE, *, horizon=4, tt=16, q=4):
        profile = legacy.cpu_profile(tt, horizon, q, name)
        source = semantic._parse(self.feedback.base.raws["source"])
        source.update(schema=query.PROFILE_SOURCE_SCHEMA, profile=profile,
                      profile_sha256=legacy.profile_sha256(profile), search_version=legacy.CPU_SEARCH,
                      search_conditions=legacy.cpu_conditions_with_ordering(tt, horizon, q, name))
        source_raw = query.canonical(source)
        capabilities_raw = self.feedback.base.raws["capabilities"]
        registration = {"schema": query.PROFILE_REGISTRATION_SCHEMA, "binary_artifact": self.feedback.binary_pin,
            "source_artifact": query.byte_pin(source_raw), "capabilities_artifact": query.byte_pin(capabilities_raw),
            "profile": profile, "profile_sha256": legacy.profile_sha256(profile), "platform": "linux",
            "binary_pin_scope": "linux_loaded_executable_inode", "assurance_scope": query.CALLER_SCOPE}
        return {"registration": query.canonical(registration), "source": source_raw, "capabilities": capabilities_raw}

    def semantic(self, profile_index=0, *, task="resume_task", known_depth=0, nodes=100000, wall=10000, bucket=1):
        original = self.feedback.base.semantic
        fixture = copy.copy(original)
        fixture.raws = copy.deepcopy(original.raws)
        fixture.common = copy.deepcopy(original.common)
        fixture.registration = copy.deepcopy(original.registration)
        fixture.receipt = copy.deepcopy(original.receipt)
        profile = query._parse(self.profiles[profile_index]["registration"])
        fixture.common.update(allowed_tasks=[task], cpu_profile_sha256=profile["profile_sha256"],
            known_completed_depth=known_depth, baseline_depth=2, requested_depth=profile["profile"]["max_depth"],
            max_nodes_per_check=nodes, max_wall_time_ms=wall, budget_bucket=bucket)
        fixture.reseal()
        return fixture.admit()

    def action(self, task, profile_index=0, **kwargs):
        checked = self.semantic(profile_index, task=task, **kwargs)
        common = checked.common_query()
        spec = {"slot": 0, "task": task, "semantic_input_sha256": checked.sha256,
                "profile_registration": profile_index, "max_output_bytes": 65536}
        spec.update({name: common[name] for name in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "budget_bucket")})
        return spec, checked

    def add_prior(self, *, state="completed", partial=False, task="resume_task"):
        f = self.feedback
        request, response, _, request_raw, receipt_raw, _, _ = f.observation(task, partial=partial)
        checked = self.semantic(task=task)
        source_pin = query.byte_pin(self.profiles[0]["source"])
        launch_raw = query.canonical(synthetic_process(f.binary_pin, source_pin))
        ordinal = len(self.prior)
        previous = [self.prior_pin(bundle) for bundle in self.prior]
        event = {"schema": query.PRIOR_SCHEMA, "ordinal": ordinal,
            "previous_ledger_sha256": query.digest(query.LEDGER_DOMAIN, previous),
            "parent_input_sha256": self.parent_pins["parent_input_sha256"], "semantic_input_sha256": checked.sha256,
            "profile_registration": 0, "started_sequence": 2 * ordinal + 1, "finished_sequence": 2 * ordinal + 2,
            "state": state, "reason": None if state in ("completed", "deferred") else "synthetic incomplete observation",
            "request": query.byte_pin(request_raw), "receipt": query.byte_pin(receipt_raw),
            "stdout": query.byte_pin(receipt_raw), "stderr": query.byte_pin(b""),
            "launch": query.byte_pin(launch_raw), "assurance_scope": query.CALLER_SCOPE}
        bundle = {"observation": query.canonical(event), "request": request_raw, "receipt": receipt_raw,
                  "stdout": receipt_raw, "stderr": b"", "launch": launch_raw, "semantic_input": checked}
        self.prior.append(bundle)
        return bundle

    @staticmethod
    def prior_pin(bundle):
        return {**{name: None if bundle[name] is None else query.byte_pin(bundle[name]) for name in
                ("observation", "request", "receipt", "stdout", "stderr", "launch")},
                "semantic_input_sha256": bundle["semantic_input"].sha256}

    def refresh(self):
        for slot, (spec, _) in enumerate(self.actions):
            spec["slot"] = slot
        self.catalogue = {"schema": query.CATALOGUE_SCHEMA, "parent": self.parent_pins, "recipe_id": "결과 전 V 행동 목록",
            "actions": [copy.deepcopy(spec) for spec, _ in self.actions], "scope": query.SCOPE,
            "limits": {"actions": 8, "prior_observations": 16, "aggregate_immutable_raw_bytes": 4 * 1024 * 1024}}
        self.catalogue_raw = query.canonical(self.catalogue)
        history = [self.prior_pin(bundle) for bundle in self.prior]
        self.before = {"schema": query.BEFORE_SCHEMA, "parent": self.parent_pins,
            "catalogue": query.byte_pin(self.catalogue_raw), "prior_ledger_sha256": query.digest(query.LEDGER_DOMAIN, history),
            "decision_ordinal": len(self.prior), "before_sequence": 2 * len(self.prior) + 1,
            "remaining_steps": 8, "remaining_nodes": 4000000, "remaining_wall_ms": 60000,
            "remaining_output_bytes": query.MAX_RAW_BYTES, "assurance_scope": query.CALLER_SCOPE,
            "utility_authority": False, "target_authority": False, "training_authority": False}
        self.before_raw = query.canonical(self.before)
        self.pins = {"parent": self.parent_pins, "catalogue": query.byte_pin(self.catalogue_raw),
            "before_result": query.byte_pin(self.before_raw),
            "profiles": [{name: query.byte_pin(raw) for name, raw in bundle.items()} for bundle in self.profiles],
            "prior": history, "action_semantic_sha256": [checked.sha256 for _, checked in self.actions]}

    def repin_event(self, bundle, **updates):
        event = query._parse(bundle["observation"])
        event.update(updates)
        bundle["observation"] = query.canonical(event)
        self.refresh()

    def admit(self):
        return query.admit_strategic_query(parents=self.parents, parent_index=self.index,
            catalogue_bytes=self.catalogue_raw, before_result_bytes=self.before_raw,
            registered_profiles=self.profiles, prior_observations=self.prior,
            action_semantic_inputs=[checked for _, checked in self.actions], expected_pins=self.pins)


class StrategicQueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = QueryFixture(self.temp.name)

    def test_query_only_actual_factories_no_execution_or_target_authority(self):
        f = self.f
        historical = copy.deepcopy(f.parents.records)
        checked = f.admit()
        self.assertIs(checked.verify(), checked)
        self.assertEqual(len(checked.features()), 2)
        self.assertEqual(checked.features()[0].task, 4)
        self.assertEqual(checked.features()[0].profile.tt_entries, 16)
        self.assertEqual(checked.features()[0].branch.known_completed_depth, 0)
        self.assertEqual(f.parents.records, historical)
        for key in ("utility_authority", "target_authority", "training_authority", "actual_task_execution_authority",
                    "model_executed", "action_scoring_executed", "binary_loaded_image_verified_here", "source_authority", "build_authority"):
            self.assertIs(checked.audit()[key], False)
        self.assertFalse(hasattr(checked, "collate"))

    def test_literal_canonical_vector_keeps_u64_and_korean(self):
        expected = '["rz-pals-private-v-query/2",{"recipe":"사전 기준","sequence":9007199254740993}]'.encode("utf-8")
        self.assertEqual(query.canonical([query.QUERY_DOMAIN, {"sequence": 9007199254740993, "recipe": "사전 기준"}]), expected)
        with self.assertRaises(ValueError):
            query.canonical({"sequence": 9007199254740993.0})

    def test_views_are_detached_and_capability_factory_only(self):
        checked = self.f.admit()
        original = checked.sha256
        view = checked.catalogue()
        view["actions"][0]["requested_depth"] = 64
        raw = checked.raw_assets()
        raw["profiles"][0]["source"] = b"changed"
        self.assertEqual(checked.sha256, original)
        with self.assertRaises(ValueError):
            query.CheckedStrategicQuery()
        with self.assertRaises(AttributeError):
            checked._identity = "0" * 64

    def test_query_factory_never_invokes_nn_or_cpu_capture(self):
        from rz_pals_model import verifier_feedback as feedback
        with patch.object(semantic.FrozenSemanticVerifier, "forward", side_effect=AssertionError("query must not forward")), \
                patch.object(feedback, "_capture", side_effect=AssertionError("query must not dispatch")):
            self.f.admit().verify()

    def test_parent_mutation_checked_again_after_admission(self):
        checked = self.f.admit()
        self.f.parents.records[self.f.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            checked.features()

    def test_semantic_capability_callback_and_subclass_are_not_admitted(self):
        self.f.actions[0] = (self.f.actions[0][0], object())
        with self.assertRaisesRegex(ValueError, "exact semantic capability"):
            self.f.admit()

    def test_independent_pin_cannot_be_self_resealed(self):
        self.f.catalogue["recipe_id"] = "후속 변경"
        self.f.catalogue_raw = query.canonical(self.f.catalogue)
        with self.assertRaisesRegex(ValueError, "independent actual bytes"):
            self.f.admit()

    def test_current_and_frozen_independent_parent_pins_are_required(self):
        f = self.f
        f.pins = copy.deepcopy(f.pins)
        f.pins["parent"]["current_view_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "independent parent/current/frozen"):
            f.admit()

    def test_effective_duplicate_profile_names_are_not_alternatives(self):
        f = self.f
        f.profiles.append(f.profile(legacy.RECHECK_PROFILE))
        f.actions = [f.action("resume_task"), f.action("resume_task", 1)]
        f.refresh()
        with self.assertRaisesRegex(ValueError, "effective duplicate"):
            f.admit()

    def test_actual_option_difference_is_a_distinct_feature(self):
        f = self.f
        f.profiles.append(f.profile(tt=32))
        f.actions = [f.action("resume_task"), f.action("resume_task", 1)]
        f.refresh()
        features = f.admit().features()
        self.assertNotEqual(features[0].profile, features[1].profile)
        self.assertNotEqual(features[0].numeric_controls(), features[1].numeric_controls())
        self.assertFalse(hasattr(features[0].profile, "profile_sha256"))

    def test_changing_only_legacy_bucket_does_not_create_new_alternative(self):
        f = self.f
        f.actions = [f.action("resume_task", bucket=1), f.action("resume_task", bucket=2)]
        f.refresh()
        with self.assertRaisesRegex(ValueError, "effective duplicate"):
            f.admit()

    def test_unknown_profile_option_and_see_marker_are_rejected(self):
        for key, value in (("ordering_policy", legacy.SEE_ORDERING_IDENTITY), ("nullmove", True)):
            f = self.f
            original = f.profiles[0]["registration"]
            registration = query._parse(original)
            registration["profile"][key] = value
            f.profiles[0]["registration"] = query.canonical(registration)
            f.refresh()
            with self.assertRaises(ValueError):
                f.admit()
            f.profiles[0]["registration"] = original

    def test_registration_actual_source_pin_and_capability_bound_checked(self):
        f = self.f
        original = f.profiles[0]["source"]
        source = query._parse(original)
        source["search_conditions"] += ";unknown"
        f.profiles[0]["source"] = query.canonical(source)
        f.refresh()
        with self.assertRaisesRegex(ValueError, "source/capability pins"):
            f.admit()

    def test_registered_node_cap_applies_before_any_dispatch(self):
        f = self.f
        cap = query._parse(f.profiles[0]["capabilities"])
        cap["max_nodes_per_check"] = 99999
        f.profiles[0]["capabilities"] = query.canonical(cap)
        registration = query._parse(f.profiles[0]["registration"])
        registration["capabilities_artifact"] = query.byte_pin(f.profiles[0]["capabilities"])
        f.profiles[0]["registration"] = query.canonical(registration)
        f.refresh()
        with self.assertRaisesRegex(ValueError, "bounded integer"):
            f.admit()

    def test_historical_source_bytes_are_preserved_without_schema_rewrite(self):
        f = self.f
        original = f.feedback.base.raws["source"]
        self.assertEqual(query._parse(original)["schema"], query.LEGACY_PROFILE_SOURCE_SCHEMA)
        f.profiles[0]["source"] = original
        registration = query._parse(f.profiles[0]["registration"])
        registration["source_artifact"] = query.byte_pin(original)
        f.profiles[0]["registration"] = query.canonical(registration)
        f.add_prior()
        f.actions = [f.action("resume_task", known_depth=4)]
        f.refresh()
        checked = f.admit()
        self.assertEqual(checked.raw_assets()["profiles"][0]["source"], original)

    def test_unknown_source_schema_cannot_enroll_current_hash_automatically(self):
        f = self.f
        source = query._parse(f.profiles[0]["source"])
        source["schema"] = "rz-pals-arbitrary-future-source/1"
        f.profiles[0]["source"] = query.canonical(source)
        registration = query._parse(f.profiles[0]["registration"])
        registration["source_artifact"] = query.byte_pin(f.profiles[0]["source"])
        f.profiles[0]["registration"] = query.canonical(registration)
        f.refresh()
        with self.assertRaisesRegex(ValueError, "registered profile scope"):
            f.admit()

    def test_action_order_controls_and_bool_integer_alias_rejected(self):
        f = self.f
        for key, value in (("slot", True), ("requested_depth", True), ("max_nodes_per_check", 99999)):
            original = f.actions[0][0][key]
            f.actions[0][0][key] = value
            f.refresh()
            if key == "slot":
                f.catalogue["actions"][0]["slot"] = value
                f.catalogue_raw = query.canonical(f.catalogue)
                f.before["catalogue"] = query.byte_pin(f.catalogue_raw)
                f.before_raw = query.canonical(f.before)
                f.pins.update(catalogue=query.byte_pin(f.catalogue_raw), before_result=query.byte_pin(f.before_raw))
            with self.assertRaises(ValueError):
                f.admit()
            f.actions[0][0][key] = original

    def test_duplicate_json_key_and_future_result_field_refused(self):
        f = self.f
        f.catalogue_raw = b'{"schema":"a","schema":"b"}'
        f.pins["catalogue"] = query.byte_pin(f.catalogue_raw)
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            f.admit()
        f.refresh()
        f.before["score"] = 8
        f.before_raw = query.canonical(f.before)
        f.pins["before_result"] = query.byte_pin(f.before_raw)
        with self.assertRaisesRegex(ValueError, "exact fields"):
            f.admit()

    def test_action_nine_and_prior_seventeen_refused_before_raw_decode(self):
        f = self.f
        with patch.object(query, "_parse", side_effect=AssertionError("must reject before decode")):
            with self.assertRaisesRegex(ValueError, "action count"):
                query.admit_strategic_query(parents=f.parents, parent_index=f.index, catalogue_bytes=b"{}",
                    before_result_bytes=b"{}", registered_profiles=f.profiles, prior_observations=[],
                    action_semantic_inputs=[f.actions[0][1]] * 9, expected_pins={})
            with self.assertRaisesRegex(ValueError, "prior observation count"):
                query.admit_strategic_query(parents=f.parents, parent_index=f.index, catalogue_bytes=b"{}",
                    before_result_bytes=b"{}", registered_profiles=f.profiles, prior_observations=[{}] * 17,
                    action_semantic_inputs=[f.actions[0][1]], expected_pins={})

    def test_four_mib_aggregate_refused_before_decode_or_parent_verification(self):
        f = self.f
        with patch.object(query, "_parse", side_effect=AssertionError("must reject before decode")), \
                patch.object(f.parents, "_verify_raw_integrity", side_effect=AssertionError("must reserve first")):
            with self.assertRaisesRegex(ValueError, "aggregate immutable raw"):
                query.admit_strategic_query(parents=f.parents, parent_index=f.index,
                    catalogue_bytes=b"x" * query.MAX_RAW_BYTES, before_result_bytes=b"{}",
                    registered_profiles=f.profiles, prior_observations=[],
                    action_semantic_inputs=[f.actions[0][1]], expected_pins={})

    def test_repeated_pin_metadata_refused_before_json_serialization(self):
        f = self.f
        # One small immutable string/object is referenced many times; do not
        # allocate a giant raw fixture just to test the pre-allocation boundary.
        repeated = {"value": "x" * 4096}
        f.pins = dict(f.pins, parent=[repeated] * 1024)
        with patch.object(query.json, "dumps", side_effect=AssertionError("must reserve before dumps")) as dumps:
            with self.assertRaisesRegex(ValueError, "byte extent before serialization"):
                f.admit()
            dumps.assert_not_called()

    def test_json_escape_expansion_is_reserved_before_serialization(self):
        repeated = "\x00" * 1024
        with patch.object(query.json, "dumps", side_effect=AssertionError("must count JSON escapes first")) as dumps:
            with self.assertRaisesRegex(ValueError, "byte extent before serialization"):
                query.canonical([repeated] * 1024)
            dumps.assert_not_called()
        with patch.object(query.json, "dumps", side_effect=AssertionError("must honor remaining credit")) as dumps:
            with self.assertRaisesRegex(ValueError, "byte extent before serialization"):
                query.canonical(["x" * 64], max_bytes=64)
            dumps.assert_not_called()

    def test_known_prior_uses_raw_receipt_and_updates_exact_scope_header(self):
        f = self.f
        f.add_prior()
        f.actions = [f.action("resume_task", known_depth=4), f.action("defer", known_depth=4)]
        f.refresh()
        checked = f.admit()
        prior = checked.features()[0].prior[0]
        self.assertTrue(prior.known)
        self.assertEqual(prior.completed_depths, (2, 4))
        self.assertEqual(checked.features()[0].branch.known_completed_depth, 4)
        self.assertEqual(checked.raw_assets()["prior"][0]["stdout"], f.prior[0]["stdout"])
        self.assertIs(checked.audit()["utility_authority"], False)

    def test_partial_prior_is_unknown_without_completeness_or_rank(self):
        f = self.f
        f.add_prior(state="partial", partial=True)
        f.refresh()
        prior = f.admit().features()[0].prior[0]
        self.assertFalse(prior.known)
        self.assertEqual(prior.completed_depths, (2, 3))
        self.assertEqual(prior.completion[-1], "node_limit")
        self.assertEqual(f.admit().features()[0].branch.known_completed_depth, 0)

    def test_missing_receipt_preserves_stdout_and_unknown_reason(self):
        f = self.f
        bundle = f.add_prior(state="missing")
        bundle.update(receipt=None, launch=None, stdout=b"partial raw stdout", stderr=b"typed failure")
        f.repin_event(bundle, receipt=None, launch=None, stdout=query.byte_pin(bundle["stdout"]), stderr=query.byte_pin(bundle["stderr"]))
        checked = f.admit()
        self.assertFalse(checked.features()[0].prior[0].known)
        self.assertEqual(checked.raw_assets()["prior"][0]["stderr"], b"typed failure")
        self.assertIsNone(checked.features()[0].prior[0].reported_nodes)

    def test_canceled_capture_retains_complete_cpu_report_as_unknown(self):
        f = self.f
        bundle = f.add_prior(state="canceled")
        launch = query._parse(bundle["launch"])
        launch.update(failure="synthetic post-report cancellation", original_deadline_met=False)
        bundle["launch"] = query.canonical(launch)
        f.repin_event(bundle, launch=query.byte_pin(bundle["launch"]))
        checked = f.admit()
        self.assertFalse(checked.features()[0].prior[0].known)
        self.assertEqual(checked.features()[0].prior[0].completed_depths, (2, 4))
        self.assertEqual(checked.features()[0].branch.known_completed_depth, 0)

    def test_typed_deadline_and_cancel_receipts_preserved_as_unknown(self):
        f = self.f
        for state, completion in (("partial", "deadline"), ("canceled", "canceled"), ("failed", "deadline")):
            with self.subTest(state=state):
                f.prior.clear()
                bundle = f.add_prior(state=state, partial=True)
                response = query._parse(bundle["receipt"])
                response.update(deadline_exceeded=True, elapsed_ms=10001)
                response["after"]["completion"] = completion
                raw = query.canonical(response)
                bundle.update(receipt=raw, stdout=raw)
                launch = query._parse(bundle["launch"])
                launch.update(elapsed_ms=10002, original_deadline_met=False, failure="synthetic original deadline")
                bundle["launch"] = query.canonical(launch)
                f.repin_event(bundle, receipt=query.byte_pin(raw), stdout=query.byte_pin(raw),
                              launch=query.byte_pin(bundle["launch"]))
                with patch.object(legacy, "observed_gain", side_effect=AssertionError("unknown must not call completed gate")):
                    checked = f.admit()
                    prior = checked.features()[0].prior[0]
                    self.assertFalse(prior.known)
                    self.assertEqual(prior.completion[-1], completion)
                    self.assertEqual(prior.reported_elapsed_ms, 10001)
                    self.assertEqual(checked.features()[0].branch.known_completed_depth, 0)
                    self.assertEqual(checked.raw_assets()["prior"][0]["receipt"], raw)
                    self.assertIs(checked.audit()["utility_authority"], False)

    def test_unknown_raw_receipt_still_rejects_identity_shape_and_accounting_drift(self):
        f = self.f
        bundle = f.add_prior(state="canceled", partial=True)
        original = query._parse(bundle["receipt"])
        for axis in ("deadline_type", "context", "nodes", "report_profile", "shape"):
            with self.subTest(axis=axis):
                response = copy.deepcopy(original)
                if axis == "deadline_type":
                    response["deadline_exceeded"] = "true"
                elif axis == "context":
                    response["context_sha256"] = "0" * 64
                elif axis == "nodes":
                    response["nodes"] += 1
                elif axis == "report_profile":
                    response["after"]["profile_sha256"] = "0" * 64
                else:
                    del response["resume_kind"]
                raw = query.canonical(response)
                bundle.update(receipt=raw, stdout=raw)
                f.repin_event(bundle, receipt=query.byte_pin(raw), stdout=query.byte_pin(raw))
                with self.assertRaises(ValueError):
                    f.admit()

    def test_known_completion_still_rejects_deadline_exceeded_receipt(self):
        f = self.f
        bundle = f.add_prior()
        response = query._parse(bundle["receipt"])
        response["deadline_exceeded"] = True
        raw = query.canonical(response)
        bundle.update(receipt=raw, stdout=raw)
        f.repin_event(bundle, receipt=query.byte_pin(raw), stdout=query.byte_pin(raw))
        with self.assertRaises(TimeoutError):
            f.admit()

    def test_duplicate_launch_bytes_are_not_new_physical_observations(self):
        f = self.f
        f.add_prior(state="partial", partial=True)
        f.add_prior(state="partial", partial=True)
        f.refresh()
        with self.assertRaisesRegex(ValueError, "duplicate prior execution"):
            f.admit()

    def test_same_launch_with_request_whitespace_alias_is_not_new_execution(self):
        f = self.f
        first = f.add_prior(state="partial", partial=True)
        second = f.add_prior(state="partial", partial=True)
        self.assertEqual(first["launch"], second["launch"])
        second["request"] += b"\n"
        self.assertNotEqual(first["request"], second["request"])
        self.assertEqual(query._parse(first["request"]), query._parse(second["request"]))
        f.repin_event(second, request=query.byte_pin(second["request"]))
        with self.assertRaisesRegex(ValueError, "duplicate prior execution"):
            f.admit()

    def test_launch_whitespace_and_key_order_aliases_are_same_execution(self):
        f = self.f
        for alias in ("whitespace", "key_order"):
            with self.subTest(alias=alias):
                f.prior.clear()
                first = f.add_prior(state="partial", partial=True)
                second = f.add_prior(state="partial", partial=True)
                if alias == "whitespace":
                    second["launch"] += b"\n"
                else:
                    value = query._parse(second["launch"])
                    second["launch"] = query.json.dumps(dict(reversed(tuple(value.items()))), sort_keys=False,
                        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
                self.assertNotEqual(first["launch"], second["launch"])
                self.assertEqual(query._parse(first["launch"]), query._parse(second["launch"]))
                f.repin_event(second, launch=query.byte_pin(second["launch"]))
                with self.assertRaisesRegex(ValueError, "launch JSON alias"):
                    f.admit()

    def test_launch_identity_canonicalization_uses_remaining_buffer_credit(self):
        f = self.f
        bundle = f.add_prior(state="partial", partial=True)
        f.refresh()
        total = query._aggregate(f.catalogue_raw, f.before_raw, f.profiles, f.prior,
                                 [checked for _, checked in f.actions]) + len(query.canonical(f.pins))
        launch = query._parse(bundle["launch"])
        with patch.object(query, "canonical", wraps=query.canonical) as canonical:
            f.admit()
            calls = [call for call in canonical.call_args_list
                     if call.args and call.args[0] == launch and "max_bytes" in call.kwargs]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].kwargs["max_bytes"], query.MAX_RAW_BYTES - total - 32)

    def test_missing_launch_is_not_counted_as_a_physical_execution(self):
        f = self.f
        for _ in range(2):
            bundle = f.add_prior(state="missing")
            bundle.update(receipt=None, launch=None, stdout=b"", stderr=b"")
            f.repin_event(bundle, receipt=None, launch=None, stdout=query.byte_pin(b""), stderr=query.byte_pin(b""))
        priors = f.admit().features()[0].prior
        self.assertEqual(len(priors), 2)
        self.assertTrue(all(not prior.known for prior in priors))
        self.assertTrue(all(prior.reported_nodes is None for prior in priors))

    def test_prior_chain_fork_order_and_before_sequence_rejected(self):
        f = self.f
        bundle = f.add_prior(state="partial", partial=True)
        f.refresh()
        original = bundle["observation"]
        for updates in ({"previous_ledger_sha256": "0" * 64}, {"ordinal": True}, {"started_sequence": 0}):
            f.repin_event(bundle, **updates)
            with self.assertRaises(ValueError):
                f.admit()
            bundle["observation"] = original
        f.refresh()
        f.before["before_sequence"] = 2
        f.before_raw = query.canonical(f.before)
        f.pins["before_result"] = query.byte_pin(f.before_raw)
        with self.assertRaisesRegex(ValueError, "causal chronology"):
            f.admit()

    def test_known_claim_rejects_partial_report_and_dirty_cleanup(self):
        f = self.f
        bundle = f.add_prior(partial=True)
        f.refresh()
        with self.assertRaisesRegex(ValueError, "declared completion"):
            f.admit()
        f.prior.clear()
        bundle = f.add_prior()
        launch = query._parse(bundle["launch"])
        launch["cleanup_error"] = "synthetic cleanup failure"
        bundle["launch"] = query.canonical(launch)
        f.repin_event(bundle, launch=query.byte_pin(bundle["launch"]))
        with self.assertRaisesRegex(ValueError, "cleanup failed"):
            f.admit()

    def test_total_receipt_elapsed_and_registered_source_path_checked(self):
        f = self.f
        bundle = f.add_prior()
        launch = query._parse(bundle["launch"])
        launch["elapsed_ms"] = 0
        bundle["launch"] = query.canonical(launch)
        f.repin_event(bundle, launch=query.byte_pin(bundle["launch"]))
        with self.assertRaisesRegex(ValueError, "total receipt"):
            f.admit()

    def test_known_source_current_path_after_pin_cannot_drift(self):
        f = self.f
        bundle = f.add_prior()
        launch = query._parse(bundle["launch"])
        launch["source_after"]["artifact"] = query.byte_pin(b"changed source facts")
        bundle["launch"] = query.canonical(launch)
        f.repin_event(bundle, launch=query.byte_pin(bundle["launch"]))
        with self.assertRaises(ValueError):
            f.admit()

    def test_known_depth_does_not_cross_node_or_wall_scope(self):
        f = self.f
        f.add_prior()
        f.actions = [f.action("resume_task", nodes=99999), f.action("defer", nodes=99999)]
        f.refresh()
        self.assertEqual(f.admit().features()[0].branch.known_completed_depth, 0)
        f.actions = [f.action("resume_task", nodes=99999, known_depth=4)]
        f.refresh()
        with self.assertRaisesRegex(ValueError, "known coverage"):
            f.admit()

    def test_defer_ends_episode_instead_of_becoming_negative_reward(self):
        f = self.f
        f.add_prior(state="deferred", task="defer")
        f.refresh()
        with self.assertRaisesRegex(ValueError, "terminated Defer episode"):
            f.admit()

    def test_defer_cpu_options_cannot_create_fake_alternatives(self):
        f = self.f
        f.profiles.append(f.profile(horizon=5, tt=32, q=5))
        f.actions = [f.action("defer"), f.action("defer", 1, nodes=1, wall=1)]
        f.refresh()
        with self.assertRaisesRegex(ValueError, "effective duplicate"):
            f.admit()

    def test_defer_requires_no_cpu_nodes_and_keeps_raw_registered_options(self):
        f = self.f
        f.actions = [f.action("defer")]
        f.refresh()
        f.before["remaining_nodes"] = 0
        f.before_raw = query.canonical(f.before)
        f.pins["before_result"] = query.byte_pin(f.before_raw)
        checked = f.admit()
        feature = checked.features()[0]
        self.assertFalse(feature.cpu_check)
        self.assertEqual(feature.node_budget, 0)
        self.assertEqual(feature.numeric_controls(), (0.0,) * 7)
        self.assertEqual(checked.catalogue()["actions"][0]["max_nodes_per_check"], 100000)
        self.assertEqual(checked.raw_assets()["profiles"][0], f.profiles[0])
        self.assertIs(checked.audit()["actual_task_execution_authority"], False)

    def test_action_allowance_cannot_use_small_actual_usage_to_erase_reserve(self):
        f = self.f
        for key, value in (("remaining_nodes", 199999), ("remaining_wall_ms", 9999), ("remaining_output_bytes", 131071)):
            f.refresh()
            f.before[key] = value
            f.before_raw = query.canonical(f.before)
            f.pins["before_result"] = query.byte_pin(f.before_raw)
            with self.assertRaisesRegex(ValueError, "pre-dispatch allowance"):
                f.admit()


if __name__ == "__main__":
    unittest.main()
