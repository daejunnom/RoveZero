"""Stdlib-only journal fixtures, never native execution or collected evidence."""
import copy
import hashlib
import json
import unittest

from rz_pals_model import native_continuation as consumer


def raw(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def move(start, end):
    return start | end << 6


def context(prefix, purpose, revision):
    return {"game_generation": 1, "search_generation": 2, "situation": {"slot": len(prefix), "generation": 1},
            "state": len(prefix) + 1, "focus": 4, "purpose": purpose, "prefix": list(prefix),
            **{name: sha(name + str(prefix)) for name in ("focus_sha256", "prefix_sha256", "proposal_sha256", "divergence_sha256")},
            "refutation_sha256": sha("refutation"), "public_revision": revision, "situation_revision": revision}


class CollectionFixture:
    """Matches the original wire without pretending fixture hashes are Rules proof."""
    def __init__(self, *, reply_revision=8, later_reply_revision=None):
        self.repaired = [move(12, 28), move(52, 36), move(6, 21), move(57, 42), move(5, 33), move(48, 40)]
        self.refutation = self.repaired[:2] + [move(1, 18), move(62, 45), move(2, 29), move(51, 35)]
        self.counter = self.repaired[:3] + [move(62, 45), move(5, 26), move(51, 35)]
        self.identity = {"game_generation": 1, "search_generation": 2, "root": {"slot": 0, "generation": 1},
                         "root_revision": 1, "repair_record_revision": 8, "repaired_line": 4}
        self.policy = {"version": "rz-pals-native-continuation-registration/1", "base_registry_canonical_sha256": sha("registry"),
                       "collector_binary_sha256": sha("never-executed-fixture-binary"), "search_policy": copy.deepcopy(consumer.POLICY)}
        self.policy_raw = raw(self.policy)
        self.source = {"implementation_sha256": self.policy["collector_binary_sha256"],
                       "source": {"kind": "own_pals", "model_configuration_sha256": sha("config"), "model_weights_sha256": sha("weights")},
                       "frozen_epoch": 1, "encoding_sha256": sha("encoding"), "encoder_source_sha256": sha("encoder"),
                       "native": {"refinement_registration": self.policy,
                                  "refinement_registration_sha256": consumer._pin(self.policy_raw)["sha256"],
                                  "pals_search_policy": copy.deepcopy(consumer.POLICY)}}
        self.files = {name: [] for name in consumer.REQUIRED_ARTIFACTS}
        parent_calls = []
        for sequence, ply in enumerate(range(2, 6), 1):
            call = self.add_call(sequence, self.repaired[:ply], "Repair", self.repaired[ply], self.refutation,
                                 self.refutation, 7)
            call.update(prefix_len=ply, prefix=self.repaired[:ply], chosen_move=self.repaired[ply], proposal=self.refutation,
                        counterexample=self.refutation, logical_context_sha256=consumer._pin(consumer._canonical(call["logical_context"]))["sha256"],
                        selection_observation="actual-raw-policy-ranked-first-and-accepted-context-matches-repaired-ply")
            parent_calls.append(call)
        prepared = {"engine_observer_version": "pals-post-repair-continuation-observer/2",
                    "checked_source_sha256": consumer._pin(consumer._canonical(["rz-pals-collector-checked-source/1", self.source]))["sha256"],
                    "refinement_registration_sha256": consumer._pin(self.policy_raw)["sha256"], "policy": copy.deepcopy(consumer.POLICY),
                    "root_rules_state_sha256": sha("root"), "anchor_rules_state_sha256": sha("state-5"), "anchor_ply": 3,
                    "initial_reply_context": context(self.repaired[:3], "ReplyPolicy", reply_revision),
                    "parent_repair_chain": {"initial_prefix": self.repaired[:2], "initial_prefix_len": 2,
                                            "full_repaired_line": self.repaired, "calls": parent_calls,
                                            "journal_observation": "independent-exact-producer-journal-entry-required; raw-entry-SHA-not-returned-by-producer-API"},
                    "examined_responses": [], "initial_selection_rule": "ranked_unexamined_different_else_ranked_different/1",
                    "repaired": self.repaired, "refutation": self.refutation, "repaired_endpoint": {}, "checker_identity": {},
                    "cpu_condition": None, "limits": {"max_rounds": 1, "max_cpu_nodes": 64, "cpu_depth": 1},
                    "engine_deadline_tick": 100000, "engine_deadline_tick_origin": "engine_monotonic_clock_origin",
                    "prepared_before_reply_submit": True, "maximum_reply_calls": 3,
                    "assurance": "multi-Reply observation; not a legacy witness, all-defenses proof or training target"}
        self.trace = [self.trace_row("prepared", prepared)]
        summaries = []
        for ordinal, ply in enumerate(range(3, 6)):
            sequence = 5 + ordinal
            revision = reply_revision if ordinal == 0 or later_reply_revision is None else later_reply_revision
            call = self.add_call(sequence, self.counter[:ply], "Reply", self.counter[ply], self.repaired, self.refutation, revision)
            call.update(prepared_payload_sha256=self.trace[0]["payload_sha256"], ordinal=ordinal,
                        previous_bound_payload_sha256=self.trace[-1]["payload_sha256"] if ordinal else None,
                        bound_before_submit=True, physical_completion_observed=False, delivery_observed=False,
                        search_consumption_observed=False)
            row = self.trace_row("reply_bound", call)
            self.trace.append(row)
            summaries.append({"ordinal": ordinal, "process_epoch": 17, "request_sequence": sequence,
                              "bound_payload_sha256": row["payload_sha256"], "input_sha256": call["input_sha256"],
                              "physical": True, "delivered": True, "search_consumed": True,
                              "accepted_context_checked": True, "producer_metadata_admitted": True,
                              "rejected": False, "physical_unknown": False})
        finish = {"prepared_payload_sha256": self.trace[0]["payload_sha256"], "calls": summaries,
                  "prepared_accepted": True, "initial_reply_attempted": True, "initial_reply_accepted": True,
                  "selected_response": self.counter[3], "expected_initial_response": self.counter[3], "initial_selection_checked": True,
                  "counterline": self.counter, "counterline_completed": True, "full_suffix_replayed": False,
                  "repaired_endpoint": {}, "counter_endpoint": {}, "comparable": False, "publication": None,
                  "disposition": "IncomparableEvidence", "original_error": None, "engine_deadline_tick": 100000,
                  "cancelled_at_observer": False, "deadline_expired_at_observer": False,
                  "assurance": "actual call lifecycle observation; conditional line result; no all-defenses or training-target authority"}
        self.trace.append(self.trace_row("finished", finish))
        self.files["split.jsonl"] = [{"games": {"fixture-game": "train"}}]

    def add_call(self, sequence, prefix, kind, selected, proposal, refutation, revision):
        legal = [selected, self.repaired[3] if kind == "Reply" and sequence == 5 else move(8, 16)]
        input_sha = sha("input-" + str(sequence))
        snapshot = {"game_id": "fixture-game", "opening_id": "fixture-opening", "line_genealogy_id": "fixture-line",
                    "actual_history": [], "rules_state_sha256": sha("state-" + str(sequence)), "rules_history_sha256": sha("history-" + str(sequence)),
                    "source": self.source["source"], "frozen_epoch": 1, "encoding_sha256": self.source["encoding_sha256"],
                    "input_revision": revision, "role": "critic" if kind == "Reply" else "proposer", "legal_moves": legal}
        frozen = {"sha256": input_sha, "snapshot": snapshot}
        tensor = {"role": snapshot["role"], "board": [0] * 64, "metadata": [0] * 16, "records": [],
                  "required_critical_records": [], "candidates": [{"from": m & 63, "to": m >> 6 & 63, "promotion": m >> 12} for m in legal],
                  "divergence_features": [], "query": [0] * 16, "situation_revision": revision,
                  "history_digest": list(bytes.fromhex(snapshot["rules_history_sha256"])), "model_epoch": [0] * 32}
        tensor_raw = raw(tensor)
        sidecar = {"version": "rz-pals-native-input-sidecar/1", "input_sha256": input_sha,
                   "sha256": sha("sidecar-" + str(sequence)), "canonical_tensor_sha256": sha("tensor-key-" + str(sequence)),
                   "encoder_source_sha256": self.source["encoder_source_sha256"], "tensor_json": tensor_raw.decode(),
                   "tensor_sha256": consumer._pin(tensor_raw)["sha256"]}
        lineage = {"input_sha256": input_sha, "game_id": "fixture-game", "process_epoch": 17, "request_sequence": sequence,
                   "native_query_kind": kind, "virtual_prefix": list(prefix), "proposal": proposal, "counterexample": refutation,
                   "actual_played_history": [], "actual_outcome_eligible": False, "counterfactual_wdl": "masked", "training_admission": "ordinary_role"}
        journal = {"version": "rz-pals-collector-prepared-producer/1", "input_sha256": input_sha, "game_id": "fixture-game",
                   "native_request": [17, sequence], "learning_input": True, "publication": "seal-before-submit; prepaid-drain-after-search",
                   "input_json": consumer._pin(raw(frozen)), "tensor_sidecar_json": consumer._pin(raw(sidecar)), "lineage_json": consumer._pin(raw(lineage))}
        self.files["producer-prepared.jsonl"].append({"prepared": journal,
                                                     "sha256": consumer._pin(consumer._canonical([journal["version"], journal]))["sha256"]})
        for name, value in (("inputs.jsonl", frozen), ("native-inputs.jsonl", sidecar), ("input-lineage.jsonl", lineage)):
            self.files[name].append(value)
        self.files["records.jsonl"].append({"input": frozen, "future_label": None})
        for index, (stage, detail) in enumerate((
                ("prepared", {"prepared_before_submit": True, "native_query_kind": kind, "producer_metadata_admitted": True}),
                ("physically_completed", {"success": True, "logical_acceptance_inferred": False}),
                ("delivered", {"search_consumed": False}), ("search_consumed", {"search_consumed": True}))):
            self.files["native-events.jsonl"].append({"domain": "rz-pals-native-call-event/1", "game_id": "fixture-game",
                "process_epoch": 17, "request_sequence": sequence, "input_sha256": input_sha,
                "stage": stage, "observer_elapsed_us": sequence * 10 + index, "detail": detail})
        self.files["native-raw-outputs.jsonl"].append({"domain": "rz-pals-native-physical-raw/1", "process_epoch": 17,
            "request_sequence": sequence, "input_sha256": input_sha, "physical_completion_confirmed": True, "success": True,
            "raw": {"representation": "f32_ieee754_bits", "candidate_logits_bits": [0x3f800000, 0], "wdl_logits_bits": [0] * 3,
                    "private_latent_bits": [0] * 6144, "divergence_logits_bits": [] if kind == "Reply" else None,
                    "task_logits_bits": None, "prediction_is_future_label": False}})
        return {"process_epoch": 17, "request_sequence": sequence, "input_sha256": input_sha,
                "input_row_sha256": consumer._pin(raw(frozen))["sha256"], "sidecar_row_sha256": consumer._pin(raw(sidecar))["sha256"],
                "sidecar_sha256": sidecar["sha256"], "canonical_tensor_sha256": sidecar["canonical_tensor_sha256"],
                "lineage_row_sha256": consumer._pin(raw(lineage))["sha256"], "logical_context": context(prefix, kind + "Policy", revision),
                "position_rules_state_sha256": snapshot["rules_state_sha256"], "producer_metadata_admitted": True}

    def trace_row(self, stage, data):
        descriptor = (consumer._pin(consumer._canonical([consumer.TRACE_DOMAIN, "prepared-descriptor", self.identity, data]))["sha256"]
                      if stage == "prepared" else self.trace[0]["descriptor_sha256"])
        return {"domain": consumer.TRACE_DOMAIN, "stage": stage, "game_id": "fixture-game", "identity": self.identity,
                "descriptor_sha256": descriptor,
                "payload_sha256": consumer._pin(consumer._canonical([consumer.TRACE_DOMAIN, stage, self.identity, descriptor, data]))["sha256"],
                "observer_elapsed_us": 45 if stage == "prepared" else (75 if stage == "finished" else 10 * (4 + len(self.trace)) - 1),
                "data": data}

    def arguments(self):
        files = dict(self.files)
        files[consumer.TRACE_ARTIFACT] = self.trace
        artifacts = {name: b"".join(raw(row) + b"\n" for row in rows) for name, rows in files.items()}
        receipt = {"version": "rz-pals-own-collector/1", "source": self.source, "complete": True,
                   "failure": None, "actual_training_executed": False, "external_teacher_used": False,
                   "artifacts": {name: consumer._pin(value) for name, value in artifacts.items()}}
        receipt_bytes = raw(receipt)
        return dict(receipt_bytes=receipt_bytes, artifact_bytes=artifacts,
                    expected_receipt_sha256=consumer._pin(receipt_bytes)["sha256"], policy_registration_bytes=self.policy_raw,
                    expected_policy_registration_sha256=consumer._pin(self.policy_raw)["sha256"])

    def reseal(self):
        for row in self.trace:
            row["payload_sha256"] = consumer._pin(consumer._canonical([
                row["domain"], row["stage"], row["identity"], row["descriptor_sha256"], row["data"]]))["sha256"]


class NativeContinuationTests(unittest.TestCase):
    def test_multi_reply_consumes_original_journals_without_authority_or_mutation(self):
        args = CollectionFixture().arguments()
        before = copy.deepcopy(args)
        result = consumer.audit_native_continuation_collection(**args)
        self.assertEqual(args, before)
        self.assertEqual(result["reply_requests_observed"], 3)
        self.assertEqual(len(result["attempts"][0]["parent_repair_calls"]), 4)
        self.assertTrue(all(call["search_consumed"] for call in result["attempts"][0]["reply_calls"]))
        self.assertIsNone(result["physical_nn_rows"])
        self.assertTrue(all(result[name] is False for name in consumer.DENIED_AUTHORITIES))

    def test_current_reply_revision_matches_repair_or_its_single_verification(self):
        for current_revision in (8, 9):
            with self.subTest(current_revision=current_revision):
                fixture = CollectionFixture(reply_revision=current_revision)
                result = consumer.audit_native_continuation_collection(**fixture.arguments())
                prepared = fixture.trace[0]["data"]
                self.assertEqual(fixture.identity["repair_record_revision"], 8)
                self.assertEqual(prepared["initial_reply_context"]["public_revision"], current_revision)
                self.assertTrue(all(call["logical_context"]["public_revision"] == 7
                                    for call in prepared["parent_repair_chain"]["calls"]))
                self.assertEqual([row["snapshot"]["input_revision"] for row in fixture.files["inputs.jsonl"]],
                                 [7] * 4 + [current_revision] * 3)
                attempt = result["attempts"][0]
                self.assertEqual(len(attempt["parent_repair_calls"]), 4)
                self.assertEqual(len(attempt["reply_calls"]), 3)
                self.assertTrue(all(call["search_consumed"] for call in attempt["reply_calls"]))
                self.assertTrue(attempt["finished_observed"])
                self.assertTrue(attempt["counterline_completed_observed"])
                self.assertIsNone(result["physical_nn_rows"])
                self.assertTrue(all(result[name] is False for name in consumer.DENIED_AUTHORITIES))

    def test_reply_revision_cannot_skip_verification_or_drift_from_current_public_view(self):
        for current_revision, later_revision, error in (
                (7, None, "post-Repair anchor/revision"),
                (10, None, "post-Repair anchor/revision"),
                (9, 8, "stale Reply context"),
                (9, 10, "stale Reply context")):
            with self.subTest(current_revision=current_revision, later_revision=later_revision):
                fixture = CollectionFixture(reply_revision=current_revision, later_reply_revision=later_revision)
                with self.assertRaisesRegex(ValueError, error):
                    consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_duplicate_and_reordered_reply_rows_are_rejected(self):
        for duplicate in (True, False):
            fixture = CollectionFixture()
            if duplicate:
                fixture.trace.insert(2, copy.deepcopy(fixture.trace[1]))
            else:
                fixture.trace[1], fixture.trace[2] = fixture.trace[2], fixture.trace[1]
            with self.assertRaisesRegex(ValueError, "chronology|chain|duplicate|reordered|bound exceeded"):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_delivery_never_counts_as_consumption_or_allows_next_call(self):
        fixture = CollectionFixture()
        fixture.files["native-events.jsonl"] = [row for row in fixture.files["native-events.jsonl"]
            if not (row["request_sequence"] == 5 and row["stage"] == "search_consumed")]
        with self.assertRaisesRegex(ValueError, "preceding consumed"):
            consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_duplicate_and_late_consumption_rejected(self):
        for late in (False, True):
            fixture = CollectionFixture()
            rows = fixture.files["native-events.jsonl"]
            consumed = next(row for row in rows if row["request_sequence"] == 5 and row["stage"] == "search_consumed")
            extra = copy.deepcopy(consumed)
            if late:
                extra["stage"] = "logically_rejected"
                extra["detail"] = {"reason": "cancelled", "search_consumed": False}
            rows.append(extra)
            with self.assertRaisesRegex(ValueError, "duplicate/reordered"):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_shared_native_clock_crosses_the_before_submit_and_finish_boundaries(self):
        for boundary in ("parent", "parent_next", "bound", "next", "finish"):
            fixture = CollectionFixture()
            if boundary == "parent":
                fixture.trace[0]["observer_elapsed_us"] = 40  # Parent consumed at 43.
            elif boundary == "parent_next":
                for event in fixture.files["native-events.jsonl"]:
                    if event["request_sequence"] == 2:
                        event["observer_elapsed_us"] -= 10  # Parent 2 prepared before parent 1 consumed.
            elif boundary == "bound":
                fixture.trace[1]["observer_elapsed_us"] = 52  # Reply prepared at 50.
            elif boundary == "next":
                fixture.trace[2]["observer_elapsed_us"] = 52  # Preceding Reply consumed at 53.
            else:
                fixture.trace[-1]["observer_elapsed_us"] = 72  # Last Reply consumed at 73.
            with self.assertRaises(ValueError):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_unknown_physical_completion_cannot_be_delivered_or_consumed(self):
        fixture = CollectionFixture()
        fixture.files["native-raw-outputs.jsonl"][-1]["physical_completion_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "failed/unknown"):
            consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_unknown_rejection_without_physical_output_keeps_buffer_uncertainty(self):
        fixture = CollectionFixture()
        prepared = next(row for row in fixture.files["native-events.jsonl"]
                        if row["request_sequence"] == 5 and row["stage"] == "prepared")
        rejection = copy.deepcopy(prepared)
        rejection.update(stage="logically_rejected", observer_elapsed_us=51,
                         detail={"reason": "PhysicalCompletionUnknown", "search_consumed": False})
        fixture.files["native-events.jsonl"] = [row for row in fixture.files["native-events.jsonl"]
            if row["request_sequence"] != 5] + [prepared, rejection]
        fixture.files["native-raw-outputs.jsonl"] = [row for row in fixture.files["native-raw-outputs.jsonl"]
                                                    if row["request_sequence"] != 5]
        finish = fixture.trace[-1]
        summary = finish["data"]["calls"][0]
        summary.update(physical=False, delivered=False, search_consumed=False, accepted_context_checked=False,
                       rejected=True, physical_unknown=True)
        finish["data"].update(calls=[summary], initial_reply_accepted=False, selected_response=None,
                             expected_initial_response=None, initial_selection_checked=False, counterline=[],
                             counterline_completed=False, original_error="physical completion unknown")
        fixture.trace = fixture.trace[:2] + [finish]
        fixture.reseal()
        result = consumer.audit_native_continuation_collection(**fixture.arguments())
        call = result["attempts"][0]["reply_calls"][0]
        self.assertTrue(call["physical_unknown"])
        self.assertTrue(call["rejected"])
        self.assertFalse(call["physical_completion_observed"])
        self.assertFalse(call["search_consumed"])

        fixture = CollectionFixture()
        fixture.files["native-events.jsonl"][-1].update(stage="logically_rejected",
            detail={"reason": "PhysicalCompletionUnknown", "search_consumed": False})
        with self.assertRaisesRegex(ValueError, "known physical completion"):
            consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_raw_policy_mismatch_and_boolean_alias_rejected(self):
        for bool_alias in (False, True):
            fixture = CollectionFixture()
            if bool_alias:
                fixture.files["native-events.jsonl"][-1]["detail"]["search_consumed"] = 1
            else:
                fixture.files["native-raw-outputs.jsonl"][-1]["raw"]["candidate_logits_bits"] = [0, 0x3f800000]
            with self.assertRaisesRegex(ValueError, "consumption|raw policy"):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_post_repair_revision_and_selected_response_checked(self):
        for changed in ("revision", "response", "anchor"):
            fixture = CollectionFixture()
            if changed == "revision":
                fixture.trace[1]["data"]["logical_context"]["public_revision"] = 7
            elif changed == "response":
                fixture.trace[-1]["data"]["selected_response"] = fixture.repaired[3]
            else:
                fixture.trace[1]["data"]["position_rules_state_sha256"] = sha("another-anchor")
            fixture.reseal()
            with self.assertRaises(ValueError):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_original_inventory_and_tensor_bytes_cannot_be_replaced(self):
        fixture = CollectionFixture()
        args = fixture.arguments()
        args["artifact_bytes"]["native-inputs.jsonl"] += b"{}\n"
        with self.assertRaisesRegex(ValueError, "original receipt"):
            consumer.audit_native_continuation_collection(**args)
        fixture.files["native-inputs.jsonl"][-1]["tensor_json"] = "{}"
        with self.assertRaisesRegex(ValueError, "row bytes"):
            consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_finished_positive_flags_cannot_contradict_selected_line_or_attempts(self):
        for changed in ("attempted", "empty_line", "missing_selection", "selection_unchecked", "no_alternative"):
            fixture = CollectionFixture()
            finish = fixture.trace[-1]["data"]
            if changed == "attempted":
                finish["initial_reply_attempted"] = False
            elif changed == "empty_line":
                finish.update(counterline=[], counterline_completed=False)
            elif changed == "missing_selection":
                finish.update(selected_response=None, counterline_completed=False)
            elif changed == "selection_unchecked":
                finish["initial_selection_checked"] = False
            else:
                finish["disposition"] = "NoAlternativeResponse"
            fixture.reseal()
            with self.assertRaises(ValueError):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_partial_cancelled_tail_is_preserved_without_completed_line(self):
        fixture = CollectionFixture()
        fixture.trace = fixture.trace[:2]  # No finish has been observed.
        result = consumer.audit_native_continuation_collection(**fixture.arguments())
        attempt = result["attempts"][0]
        self.assertEqual(attempt["status"], "unresolved")
        self.assertFalse(attempt["finished_observed"])
        self.assertFalse(attempt["counterline_completed_observed"])
        self.assertIsNone(attempt["cancelled_at_observer"])

        fixture = CollectionFixture()
        finish = fixture.trace[-1]
        finish["data"].update(calls=finish["data"]["calls"][:1], counterline=fixture.counter[:4],
                             counterline_completed=False, cancelled_at_observer=True,
                             original_error="cancelled", disposition="Cancelled")
        fixture.trace = fixture.trace[:2] + [finish]
        fixture.reseal()
        result = consumer.audit_native_continuation_collection(**fixture.arguments())
        attempt = result["attempts"][0]
        self.assertTrue(attempt["finished_observed"])
        self.assertTrue(attempt["cancelled_at_observer"])
        self.assertEqual(attempt["original_error"], "cancelled")
        self.assertFalse(attempt["counterline_completed_observed"])
        self.assertEqual(len(attempt["reply_calls"]), 1)

    def test_nonfinite_raw_and_another_request_attribution_are_rejected(self):
        for alias in (False, True):
            fixture = CollectionFixture()
            if alias:
                other = copy.deepcopy(fixture.files["native-events.jsonl"][-1])
                other["request_sequence"] = 99
                fixture.files["native-events.jsonl"].append(other)
            else:
                fixture.files["native-raw-outputs.jsonl"][-1]["raw"]["wdl_logits_bits"][0] = 0x7f800000
            with self.assertRaisesRegex(ValueError, "nonfinite|another native request"):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_actual_game_wdl_leak_and_missing_split_rejected(self):
        for split in (False, True):
            fixture = CollectionFixture()
            if split:
                fixture.files["split.jsonl"][0]["games"] = {}
            else:
                fixture.files["records.jsonl"][-1]["future_label"] = {"provenance": {"source": "actual_game"}, "value_wdl": [1, 0, 0]}
            with self.assertRaisesRegex(ValueError, "split|leaked"):
                consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_orphan_learning_call_cannot_bypass_strict_dataset_admission(self):
        fixture = CollectionFixture()
        fixture.files["records.jsonl"].pop()
        with self.assertRaisesRegex(ValueError, "matching strict dataset records"):
            consumer.audit_native_continuation_collection(**fixture.arguments())

    def test_aggregate_and_duplicate_json_budgets(self):
        args = CollectionFixture().arguments()
        with self.assertRaisesRegex(ValueError, "aggregate"):
            consumer.audit_native_continuation_collection(**args, max_input_bytes=1024)
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            consumer._parse(b'{"a":1,"a":2}')


class FrozenLoaderContinuationConnectionTests(unittest.TestCase):
    """Existing registered producer fixture traverses the production loader.

    The continuation consumer is spied at its boundary. These prove mandatory
    routing and custody, not execution of the synthetic multi-Reply model.
    """
    def fixture(self, directory):
        from pathlib import Path
        from test_training import write_frozen_fixture_collection, repin_frozen_fixture_artifact
        configuration = write_frozen_fixture_collection(directory, kinds=("native",))
        root = Path(directory)
        for name in (consumer.TRACE_ARTIFACT, "native-raw-outputs.jsonl"):
            (root / name).write_bytes(b"")
            repin_frozen_fixture_artifact(directory, configuration, name)
        policy = root / "independent-continuation-registration.json"
        policy.write_bytes(b'{}')
        configuration["continuation_registration"] = {"registration_path": policy,
                                                      "sha256": consumer._pin(policy.read_bytes())["sha256"]}
        return root, configuration

    def test_production_loader_requires_registration_and_never_silently_skips_trace(self):
        import tempfile
        from rz_pals_model.training import load_frozen_collected_dataset
        with tempfile.TemporaryDirectory() as directory:
            _, configuration = self.fixture(directory)
            configuration.pop("continuation_registration")
            with self.assertRaisesRegex(ValueError, "independent continuation registration"):
                load_frozen_collected_dataset(directory, **configuration)

    def test_production_loader_passes_original_pins_and_preserves_audit_immutably(self):
        import tempfile
        from unittest.mock import patch
        from rz_pals_model.training import load_frozen_collected_dataset
        with tempfile.TemporaryDirectory() as directory:
            root, configuration = self.fixture(directory)
            observation = {"schema": consumer.AUDIT_SCHEMA, **consumer.DENIED_AUTHORITIES}
            with patch.object(consumer, "audit_native_continuation_collection", return_value=observation) as audit:
                loaded = load_frozen_collected_dataset(directory, **configuration)
            arguments = audit.call_args.kwargs
            self.assertEqual(arguments["receipt_bytes"], (root / "receipt.json").read_bytes())
            self.assertEqual(arguments["expected_receipt_sha256"], configuration["expected_receipt_sha256"])
            self.assertEqual(set(arguments["artifact_bytes"]), set(consumer.REQUIRED_ARTIFACTS))
            self.assertEqual(loaded.native_continuation_observation, observation)
            loaded.native_continuation_observation["product_authority"] = True
            with self.assertRaisesRegex(ValueError, "continuation observation changed"):
                loaded.indices("proposer")

    def test_production_loader_propagates_consumer_rejection_without_legacy_retry(self):
        import tempfile
        from unittest.mock import patch
        from rz_pals_model.training import load_frozen_collected_dataset
        with tempfile.TemporaryDirectory() as directory:
            _, configuration = self.fixture(directory)
            with patch.object(consumer, "audit_native_continuation_collection", side_effect=ValueError("actual trace rejected")) as audit:
                with self.assertRaisesRegex(ValueError, "actual trace rejected"):
                    load_frozen_collected_dataset(directory, **configuration)
            self.assertEqual(audit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
