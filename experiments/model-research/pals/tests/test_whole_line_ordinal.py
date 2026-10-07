"""Synthetic whole-line byte contracts, never actual chess/process evidence.

The strict frozen parent fixture is reused without a second current selector.
All endpoint Rules/build/launch facts below are deliberately synthetic. They
exercise the consumer's conditional checks only, not Rust replay, execution,
strategic repair, C slot supervision, model loss or training admission.
"""
import copy
import hashlib
import json
import tempfile
import unittest

from rz_pals_model import comparative_training as cpu
from rz_pals_model import semantic_verifier as rules
from rz_pals_model import whole_line_ordinal as ordinal
from test_semantic_verifier import SemanticFixture, tokens
from test_native_divergence import NativeDivergenceFixture
from test_repair_context import RepairFixture
from test_training import sha


class WholeLineFixture:
    """Synthetic caller fixture; explicit pins are not a process proof."""
    def __init__(self, root, *, line_plies=2, direction="maximize_root_surrogate"):
        base = SemanticFixture(root)
        self.parents, self.index = base.parents, base.index
        self.anchor = ordinal.input_anchor(self.parents, self.index)
        self.binary = b"synthetic registered whole-line image; no real executable"
        self.profile = ordinal.profile_description(2, 16, 4)
        self.build = {"schema": ordinal.BUILD_SCHEMA, "source_commit": "synthetic-source-commit",
                      "binary_artifact": ordinal.byte_pin(self.binary),
                      "files_before": [{"path": "crates/rz-uci/src/pals_cpu_task/continuation.rs",
                                        "bytes": 23, "sha256": sha("synthetic historical source bytes")}],
                      "actual_build_exit_code": 0, "source_verified_before_and_after_build": True,
                      "assurance_scope": "independently_pinned_caller_build_observation"}
        self.build["files_after"] = copy.deepcopy(self.build["files_before"])
        self.raws = {"binary": self.binary, "build": ordinal.canonical(self.build)}
        self.source = {"schema": ordinal.SOURCE_SCHEMA, "source_commit": self.build["source_commit"],
                       "build_artifact": ordinal.byte_pin(self.raws["build"]), "profile": self.profile,
                       "profile_sha256": cpu.wire_digest(self.profile), "search_version": cpu.SEARCH_VERSION,
                       "search_conditions": cpu.search_conditions(2, 16, 4),
                       "value_identity": {"semantics": cpu.SEMANTICS, "weights_sha256": None,
                                          "training": {"kind": "bootstrap"}}}
        self.capabilities = {"schema": ordinal.WIRE_SCHEMA, "max_checks": 1, "terminal_checks": 0,
            "max_line_plies": 64, "max_horizon": 64, "max_node_budget": 2**32 - 1, "max_tt_entries": 1048576,
            "max_quiescence_ply": 32, "max_request_bytes": 524288, "max_response_bytes": 1048576,
            "max_wall_time_ms": 300000, "fresh_engine": True, "restriction": "unrestricted_endpoint", "resume": False,
            "profile": cpu.PROFILE, "search_version": cpu.SEARCH_VERSION, "score_semantics": cpu.SEMANTICS,
            "score_units": "uncalibrated_endpoint_side_to_move_raw", "rules_descriptor_schema": rules.DESCRIPTOR_SCHEMA,
            "line_domain": ordinal.LINE_DOMAIN, "conditions_schema": ordinal.CONDITIONS_SCHEMA,
            "caller_registration_scope": "declared_not_independently_verified",
            "binary_pin_scope": "linux_loaded_executable_inode", "cpu_binary_sha256": ordinal.byte_pin(self.binary)["sha256"],
            "training_target_created": False, "product_verifier_enabled": False, "teacher_gpu": False, "ranking_created": False}
        for name, value in (("source", self.source), ("capabilities", self.capabilities)):
            self.raws[name] = ordinal.canonical(value)
        self.registration = {"schema": ordinal.REGISTRATION_SCHEMA,
            **{name + "_artifact": ordinal.byte_pin(self.raws[name]) for name in ("source", "binary", "capabilities", "build")},
            "source_commit": self.build["source_commit"], "profile_sha256": self.source["profile_sha256"],
            "rules_version": base.registration["rules_version"], "platform": "linux",
            "binary_pin_scope": "linux_loaded_executable_inode"}
        self.registration["checker_namespace_sha256"] = ordinal.checker_namespace(self.registration)
        self.raws["registration"] = ordinal.canonical(self.registration)
        self.criterion = {"schema": ordinal.CRITERION_SCHEMA, "recipe_id": "사전 고정 전체 수순 ordinal",
            "scope": ordinal.SCOPE, "direction": direction, "line_plies": line_plies, "horizon": 2,
            "node_budget": 100000, "quiescence_ply": 4, "tt_entries": 16, "max_wall_time_ms": 10000,
            "max_output_bytes": 65536, "profile_sha256": self.source["profile_sha256"],
            "registration_sha256": ordinal.byte_pin(self.raws["registration"])["sha256"],
            "minimum_margin": 1, "score_limit": 20000, "mate_threshold": 29000,
            "required_history_completeness": "complete"}
        self.raws["criterion"] = ordinal.canonical(self.criterion)
        self.root = copy.deepcopy(base.receipt["root"])
        self.root.update(history_completeness="complete", history_origin="start_position",
                         repetition_history_complete=True)
        # These are synthetic descriptors, not a claim that a fixture FEN
        # actually has this history or that repeated move16 values are legal.
        self.endpoints, plans = {}, []
        legal = self.anchor["legal_moves"]
        for slot, task in enumerate(("left", "right")):
            line = [legal[slot]] * line_plies  # Ordered repetition is retained.
            endpoint = copy.deepcopy(self.root)
            endpoint.update(board_fen=f"synthetic endpoint {task}", rules_state_sha256=sha("endpoint state " + task),
                            rules_history_sha256=sha("endpoint complete history " + task),
                            side_to_move=self.anchor["side_to_move"] if line_plies % 2 == 0 else
                                ("black" if self.anchor["side_to_move"] == "white" else "white"),
                            known_history_positions=self.root["known_history_positions"] + line_plies,
                            legal_tokens=tokens(legal, "claim_end_legal", legal))
            self.endpoints[task] = endpoint
            plans.append({"task_id": task, "line": line,
                          "line_sha256": ordinal.ordered_line_sha256(self.anchor["rules_state_sha256"], self.anchor["rules_history_sha256"], line),
                          "expected_endpoint_board_fen": endpoint["board_fen"],
                          "endpoint_rules_state_sha256": endpoint["rules_state_sha256"],
                          "endpoint_rules_history_sha256": endpoint["rules_history_sha256"],
                          "expected_endpoint_legal_moves": legal.copy(), "endpoint_side_to_move": endpoint["side_to_move"]})
        self.plan = {"schema": ordinal.PLAN_SCHEMA, "pair_id": "whole-line-fixture",
                     "anchor": copy.deepcopy(self.anchor), "criterion_artifact": ordinal.byte_pin(self.raws["criterion"]),
                     "registration_artifact": ordinal.byte_pin(self.raws["registration"]), "direction": direction, "lines": plans}
        self.raws["plan"] = ordinal.canonical(self.plan)
        self.requests, self.receipts, self.executions = {}, {}, {}
        for slot, planned in enumerate(plans):
            task = planned["task_id"]
            request = {"schema": ordinal.WIRE_SCHEMA, "task_id": task,
                "captured_input_sha256": self.anchor["query_input_sha256"],
                "checker_namespace_sha256": self.registration["checker_namespace_sha256"],
                "before_result_anchor_sha256": ordinal.byte_pin(self.raws["plan"])["sha256"],
                **{name: self.anchor[name] for name in ("frozen_epoch", "input_revision", "position_command")},
                "expected_root_board_fen": self.anchor["board_fen"], "root_rules_state_sha256": self.anchor["rules_state_sha256"],
                "root_rules_history_sha256": self.anchor["rules_history_sha256"],
                "expected_root_legal_moves": legal.copy(), "root_side_to_move": self.anchor["side_to_move"],
                **{name: planned[name] for name in ("line", "line_sha256", "expected_endpoint_board_fen",
                    "endpoint_rules_state_sha256", "endpoint_rules_history_sha256", "expected_endpoint_legal_moves", "endpoint_side_to_move")},
                "cpu_binary_sha256": ordinal.byte_pin(self.binary)["sha256"], "cpu_profile_sha256": self.source["profile_sha256"],
                **{name: self.criterion[name] for name in ("horizon", "node_budget", "tt_entries", "quiescence_ply", "max_wall_time_ms", "max_output_bytes")}}
            request["context_sha256"] = ordinal.digest(ordinal.WIRE_SCHEMA, request)
            self.requests[task] = request
            report = {"raw_score": 90 if slot == 0 else 20, "best_move": legal[0], "pv": [legal[0]],
                "score_scope": "completed_iteration", "completion": "depth_limit", "completed_depth": 2, "requested_depth": 2,
                "nodes": 42, "quiescence_nodes": 8, "tt_hits": 2, "elapsed_ms": 5, "reused_completed_depth": 0,
                "root_restricted": False, "score_provenance": cpu.SEMANTICS, "pv_rules_validated": True}
            receipt = {name: request[name] for name in ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256",
                "before_result_anchor_sha256", "frozen_epoch", "input_revision", "context_sha256", "cpu_binary_sha256", "line", "line_sha256")}
            receipt.update(binary_pin_scope="linux_loaded_executable_inode", caller_registration_scope="declared_not_independently_verified",
                status="completed", root=copy.deepcopy(self.root), endpoint=copy.deepcopy(self.endpoints[task]),
                line_rules_validated=True, cpu_calls=1, fresh_engine=True, report=report, elapsed_ms=8,
                deadline_exceeded=False, training_target_created=False, product_verifier_enabled=False)
            receipt["conditions"] = self.conditions(task, receipt)
            receipt["conditions_sha256"] = cpu.wire_digest(receipt["conditions"])
            self.receipts[task] = receipt
            self.executions[task] = {"request": ordinal.canonical(request), "receipt": ordinal.canonical(receipt), "stderr": b""}
            self.observe(task)
        self.refresh_pins()

    def conditions(self, task, receipt):
        request, root, endpoint = self.requests[task], receipt["root"], receipt["endpoint"]
        return {"schema": ordinal.CONDITIONS_SCHEMA, "root_rules_state_sha256": root["rules_state_sha256"],
            "root_rules_history_sha256": root["rules_history_sha256"], "root_legal_order_sha256": root["legal_order_sha256"],
            "endpoint_rules_state_sha256": endpoint["rules_state_sha256"], "endpoint_rules_history_sha256": endpoint["rules_history_sha256"],
            "endpoint_legal_order_sha256": endpoint["legal_order_sha256"], "line_sha256": request["line_sha256"],
            "line_plies": len(request["line"]), "endpoint_side_to_move": endpoint["side_to_move"],
            "profile_sha256": self.source["profile_sha256"], "profile": cpu.PROFILE,
            "value_identity": copy.deepcopy(self.source["value_identity"]), "search_version": cpu.SEARCH_VERSION,
            "search_conditions": self.source["search_conditions"],
            **{name: request[name] for name in ("horizon", "node_budget", "tt_entries", "quiescence_ply")},
            "restriction": "unrestricted_endpoint", "perspective": "endpoint_side_to_move",
            "resource_policy": {"max_wall_time_ms": request["max_wall_time_ms"], "max_output_bytes": request["max_output_bytes"],
                                "max_checks": 1, "search_deadline_reserve_ms": min(request["max_wall_time_ms"] // 10, 1000)}}

    def observe(self, task, *, launch_changes=None, process_changes=None):
        execution = self.executions[task]
        process = {"schema": ordinal.PROCESS_SCHEMA, "task_id": task, "spawned": True, "reaped": True, "exit_code": 0,
            "pipes_finished": True, "owned_group_absent": True, "process_supervision_scope": "posix_owned_process_group",
            "loaded_executable": {"status": "checked", "artifact": ordinal.byte_pin(self.binary),
                                  "scope": "caller_child_loaded_inode", "observed_before_stdin": True},
            "binary_before": ordinal.byte_pin(self.binary), "binary_after": ordinal.byte_pin(self.binary),
            "source_before": ordinal.byte_pin(self.raws["source"]), "source_after": ordinal.byte_pin(self.raws["source"]),
            "binary_path_stable": True, "source_path_stable": True, "transport_failure": None, "overflow": False,
            "original_deadline_met": True, "elapsed_ms": 12}
        process.update(process_changes or {})
        execution["process_observation"] = ordinal.canonical(process)
        pins = {name: None if execution[name] is None else ordinal.byte_pin(execution[name])
                for name in ("request", "receipt", "stderr", "process_observation")}
        launch = {"schema": ordinal.LAUNCH_SCHEMA, "task_id": task, "registration_sha256": ordinal.byte_pin(self.raws["registration"])["sha256"],
            **pins, **{name + "_artifact": ordinal.byte_pin(self.raws[name]) for name in ("criterion", "plan", "binary", "source")},
            "anchor_durable_before_spawn": True, "criterion_fixed_before_spawn": True,
            "assurance_scope": "independently_pinned_caller_execution_observation", "spawned": True, "reaped": True,
            "exit_code": 0, "elapsed_ms": 12, "timed_out": False, "platform": "linux",
            "binary_pin_scope": "linux_loaded_executable_inode"}
        launch.update(launch_changes or {})
        execution["launch"] = ordinal.canonical(launch)

    def refresh_pins(self):
        self.pins = {name: ordinal.byte_pin(raw) for name, raw in self.raws.items()}
        self.pins.update(anchor=copy.deepcopy(self.anchor), executions={
            task: {name: None if raw is None else ordinal.byte_pin(raw) for name, raw in buffers.items()}
            for task, buffers in self.executions.items()})

    def rebind_registered_assets(self):
        """Coherent adversarial pin updates must still fail content checks.

        This is synthetic fixture wiring only, not a real re-registration.
        It changes no receipt raw value or search condition.
        """
        for name in ("source", "binary", "capabilities", "build"):
            self.registration[name + "_artifact"] = ordinal.byte_pin(self.raws[name])
        self.registration["checker_namespace_sha256"] = ordinal.checker_namespace(self.registration)
        self.raws["registration"] = ordinal.canonical(self.registration)
        self.criterion["registration_sha256"] = ordinal.byte_pin(self.raws["registration"])["sha256"]
        self.raws["criterion"] = ordinal.canonical(self.criterion)
        self.plan.update(criterion_artifact=ordinal.byte_pin(self.raws["criterion"]),
                         registration_artifact=ordinal.byte_pin(self.raws["registration"]))
        self.raws["plan"] = ordinal.canonical(self.plan)
        for task in self.executions:
            request = json.loads(self.executions[task]["request"])
            request.update(checker_namespace_sha256=self.registration["checker_namespace_sha256"],
                           before_result_anchor_sha256=ordinal.byte_pin(self.raws["plan"])["sha256"])
            request["context_sha256"] = ordinal.digest(ordinal.WIRE_SCHEMA, {key: value for key, value in request.items() if key != "context_sha256"})
            self.requests[task] = request
            self.executions[task]["request"] = ordinal.canonical(request)
            receipt = json.loads(self.executions[task]["receipt"])
            for key in ("checker_namespace_sha256", "before_result_anchor_sha256", "context_sha256"):
                receipt[key] = request[key]
            self.executions[task]["receipt"] = ordinal.canonical(receipt)
            self.observe(task)
        self.refresh_pins()

    def change_receipt(self, task, mutate, *, recompute_conditions=False):
        value = json.loads(self.executions[task]["receipt"])
        mutate(value)
        if recompute_conditions:
            value["conditions"] = self.conditions(task, value)
        value["conditions_sha256"] = cpu.wire_digest(value["conditions"])
        self.executions[task]["receipt"] = ordinal.canonical(value)
        self.observe(task)
        self.refresh_pins()

    def change_request(self, task, mutate):
        value = json.loads(self.executions[task]["request"])
        mutate(value)
        value["context_sha256"] = ordinal.digest(ordinal.WIRE_SCHEMA, {key: item for key, item in value.items() if key != "context_sha256"})
        self.executions[task]["request"] = ordinal.canonical(value)
        self.observe(task)
        self.refresh_pins()

    def admit(self, **changes):
        arguments = {name + "_bytes": raw for name, raw in self.raws.items()}
        arguments.update(parents=self.parents, parent_index=self.index, executions=self.executions, expected_pins=self.pins)
        arguments.update(changes)
        return ordinal.admit_whole_line_pair(**arguments)


class WholeLineOrdinalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = WholeLineFixture(self.temp.name)

    def test_completed_full_line_keeps_raw_units_and_has_no_role_authority(self):
        checked = self.fixture.admit()
        self.assertIs(checked.verify(), checked)
        self.assertTrue(checked.outcome.mask)
        self.assertEqual(checked.outcome.sign, 1)
        self.assertEqual(checked.outcome.raw_endpoint_scores, (90, 20))
        self.assertEqual(checked.outcome.criterion_root_scores, (90, 20))
        self.assertEqual(checked.outcome.scope, "line_conditioned_surrogate")
        admission = checked.admission
        self.assertTrue(admission["requires_role_projection"])
        for name in ("ordinary_policy_admitted", "auxiliary_has_current_label", "strategic_repair_validity_admitted",
                     "counterexample_validity_admitted", "wdl_admitted", "minimax_admitted", "tactical_proof_admitted",
                     "training_target_created", "actual_training_executed", "gpu_executed", "metadata_alone_is_execution_authority"):
            self.assertIs(admission[name], False)
        self.assertFalse(hasattr(checked, "collate"))

    def test_odd_line_parity_is_criterion_only_and_direction_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = WholeLineFixture(root, line_plies=1)
            outcome = fixture.admit().outcome
            self.assertEqual(outcome.raw_endpoint_scores, (90, 20))
            self.assertEqual(outcome.criterion_root_scores, (-90, -20))
            self.assertEqual(outcome.sign, -1)
        with tempfile.TemporaryDirectory() as root:
            fixture = WholeLineFixture(root, line_plies=1, direction="minimize_root_surrogate")
            self.assertEqual(fixture.admit().outcome.sign, 1)

    def test_canonical_literal_retains_u64_korean_and_sorted_nested_keys(self):
        value = {"recipe": "사전 기준", "counter": 9007199254740993, "nested": {"z": [2, 1], "a": None}}
        literal = '{"counter":9007199254740993,"nested":{"a":null,"z":[2,1]},"recipe":"사전 기준"}'.encode("utf-8")
        self.assertEqual(ordinal.canonical(value), literal)
        self.assertEqual(ordinal.digest(ordinal.CRITERION_SCHEMA, value),
                         hashlib.sha256(b'["rz-pals-whole-line-ordinal-criterion/1",' + literal + b']').hexdigest())
        self.assertEqual(ordinal.digest(ordinal.CRITERION_SCHEMA, value),
                         "71231c54452f5b52ac61e4d49aa8a037b01b29b7b609c41511f2059e6077177e")
        for bad in (1.0, float("nan"), 2**64, -(2**63) - 1):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                ordinal.canonical({"bad": bad})

    def test_factory_and_fake_query_callback_cannot_bypass_current_authority(self):
        with self.assertRaises(ValueError):
            ordinal.CheckedWholeLinePair()
        with self.assertRaises(ValueError):
            self.fixture.admit(query_anchor=lambda: self.fixture.anchor)
        checked = self.fixture.admit()
        with self.assertRaises(AttributeError):
            checked._identity = "0" * 64
        admission = checked.admission
        admission["anchor"]["legal_moves"].reverse()
        self.assertEqual(checked.admission["anchor"]["legal_moves"], self.fixture.anchor["legal_moves"])

    def test_parent_raw_mutation_invalidates_previously_admitted_pair(self):
        checked = self.fixture.admit()
        self.fixture.parents.records[self.fixture.index]["input"]["snapshot"]["input_revision"] += 1
        with self.assertRaises(ValueError):
            checked.verify()

    def test_auxiliary_anchor_keeps_its_own_public_context_and_false_learning_input(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = NativeDivergenceFixture(root)
            auxiliary = fixture.admit()
            anchor = ordinal.input_anchor(fixture.parents, fixture.index, auxiliary)
            parent = ordinal.input_anchor(fixture.parents, fixture.index)
            self.assertEqual(anchor["kind"], "native_divergence")
            self.assertEqual(anchor["parent_input_sha256"], parent["parent_input_sha256"])
            self.assertEqual(anchor["query_input_sha256"], fixture.input["sha256"])
            self.assertNotEqual(anchor["query_input_sha256"], anchor["parent_input_sha256"])
            self.assertEqual(anchor["query_context_sha256"], fixture.context["sha256"])
            self.assertNotEqual(anchor["public_evidence"], parent["public_evidence"])
            self.assertEqual(anchor["input_revision"], 9)
            self.assertIs(auxiliary.admission["producer_journal_learning_input"], False)
            self.assertIs(auxiliary.admission["auxiliary_has_current_label"], False)
            with self.assertRaises(ValueError):
                ordinal.input_anchor(self.fixture.parents, self.fixture.index, auxiliary)

    def test_repair_anchor_requires_its_checked_current_ordinary_row_without_validity(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = RepairFixture(root)
            repair = fixture.admit()
            anchor = ordinal.input_anchor(fixture.parents, fixture.repair_index, repair)
            self.assertEqual(anchor["kind"], "native_initial_repair")
            self.assertEqual(anchor["query_input_sha256"], fixture.rows[1]["input"]["sha256"])
            self.assertEqual(anchor["query_context_sha256"], repair.admission()["context_sha256"])
            self.assertEqual(anchor["query_capability_sha256"], repair.sha256)
            with self.assertRaises(ValueError):
                ordinal.input_anchor(fixture.parents, fixture.root_index, repair)

    def test_independent_asset_and_execution_pins_are_not_self_seals(self):
        for name in ordinal.ASSETS:
            with self.subTest(asset=name):
                pins = copy.deepcopy(self.fixture.pins)
                pins[name]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    self.fixture.admit(expected_pins=pins)
        pins = copy.deepcopy(self.fixture.pins)
        pins["executions"]["left"]["receipt"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.fixture.admit(expected_pins=pins)

    def test_exact_parent_public_source_epoch_revision_history_and_label_anchors(self):
        for name in ("parent_label_sha256", "query_input_sha256", "query_context_sha256", "producer_roster_sha256",
                     "producer_envelope_sha256", "parent_prepared_sha256", "rules_history_sha256", "frozen_epoch", "input_revision"):
            with self.subTest(anchor=name):
                pins = copy.deepcopy(self.fixture.pins)
                pins["anchor"][name] = "changed" if type(pins["anchor"][name]) is not int else pins["anchor"][name] + 1
                with self.assertRaises(ValueError):
                    self.fixture.admit(expected_pins=pins)
        pins = copy.deepcopy(self.fixture.pins)
        pins["anchor"]["public_evidence"]["ordered_observation_sha256"].append("0" * 64)
        with self.assertRaises(ValueError):
            self.fixture.admit(expected_pins=pins)

    def test_recontextualized_request_cannot_change_shared_conditions(self):
        original = self.fixture.executions["left"]["request"]
        changes = {"horizon": 3, "node_budget": 99999, "quiescence_ply": 5, "tt_entries": 32,
                   "max_wall_time_ms": 9999, "max_output_bytes": 32768, "frozen_epoch": True, "input_revision": True,
                   "root_rules_history_sha256": sha("different history"), "cpu_profile_sha256": sha("candidate profile")}
        for name, value in changes.items():
            with self.subTest(field=name):
                self.fixture.executions["left"]["request"] = original
                self.fixture.change_request("left", lambda request: request.update({name: value}))
                with self.assertRaises(ValueError):
                    self.fixture.admit()

    def test_old_factual_request_cannot_acquire_new_before_result_plan(self):
        self.fixture.change_request("left", lambda request: request.update(before_result_anchor_sha256=sha("old factual request before no ordinal criterion")))
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_pre_result_durability_and_criterion_freeze_are_required(self):
        for field in ("anchor_durable_before_spawn", "criterion_fixed_before_spawn"):
            with self.subTest(field=field):
                self.fixture.observe("left", launch_changes={field: False})
                self.fixture.refresh_pins()
                with self.assertRaises(ValueError):
                    self.fixture.admit()

    def test_unknown_partial_canceled_and_missing_are_masked_with_raw_preserved(self):
        original = self.fixture.executions["left"]["receipt"]
        for status, completion in (("partial", "node_limit"), ("partial", "deadline"), ("partial", "quiescence_limit"), ("canceled", "canceled")):
            with self.subTest(status=status, completion=completion):
                self.fixture.executions["left"]["receipt"] = original
                self.fixture.change_receipt("left", lambda receipt: (receipt.update(status=status),
                    receipt["report"].update(completion=completion, completed_depth=1)))
                outcome = self.fixture.admit().outcome
                self.assertFalse(outcome.mask)
                self.assertEqual(outcome.sign, 0)
                self.assertEqual(outcome.raw_endpoint_scores, (90, 20))
        self.fixture.executions["left"]["receipt"] = None
        self.fixture.observe("left")
        self.fixture.refresh_pins()
        outcome = self.fixture.admit().outcome
        self.assertFalse(outcome.mask)
        self.assertEqual(outcome.reason, "missing")
        self.assertEqual(outcome.raw_endpoint_scores, (None, 20))

    def test_unknown_history_is_not_global_coverage_or_chess_truth(self):
        def unknown(receipt):
            for descriptor in (receipt["root"], receipt["endpoint"]):
                descriptor.update(history_completeness="unknown_prefix", history_origin="fen", repetition_history_complete=False)
        self.fixture.change_receipt("left", unknown)
        outcome = self.fixture.admit().outcome
        self.assertFalse(outcome.mask)
        self.assertEqual(outcome.reason, "unknown_history_prefix")
        # Rust repetition completeness can become true after an irreversible
        # move without making the unknown FEN prefix a fully known history.
        self.fixture.change_receipt("left", lambda receipt: receipt["endpoint"].update(repetition_history_complete=True))
        self.assertEqual(self.fixture.admit().outcome.reason, "unknown_history_prefix")
        self.fixture.change_receipt("left", lambda receipt: receipt.update(status="invented_status"))
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_terminal_endpoint_has_no_report_or_fabricated_cpu_score(self):
        def terminal(receipt):
            endpoint = receipt["endpoint"]
            endpoint.update(play_status="rules_terminal", terminal_reason="checkmate", terminal_winner="black",
                            terminal_source="rz-position-rules", legal_moves=[], legal_tokens=[],
                            legal_order_sha256=rules.digest(rules.MOVE_DOMAIN, []), in_check=True)
            receipt.update(status="rules_terminal", cpu_calls=0, fresh_engine=False, report=None)
        # A terminal endpoint expectation also must have been fixed in the
        # before plan, not altered after a nonterminal execution receipt.
        self.fixture.plan["lines"][0]["expected_endpoint_legal_moves"] = []
        self.fixture.raws["plan"] = ordinal.canonical(self.fixture.plan)
        for task in ("left", "right"):
            self.fixture.change_request(task, lambda request: request.update(before_result_anchor_sha256=ordinal.byte_pin(self.fixture.raws["plan"])["sha256"],
                **({"expected_endpoint_legal_moves": []} if task == "left" else {})))
            request = json.loads(self.fixture.executions[task]["request"])
            self.fixture.requests[task] = request
            self.fixture.change_receipt(task, lambda receipt: receipt.update(
                before_result_anchor_sha256=request["before_result_anchor_sha256"], context_sha256=request["context_sha256"]))
        self.fixture.change_receipt("left", terminal, recompute_conditions=True)
        outcome = self.fixture.admit().outcome
        self.assertFalse(outcome.mask)
        self.assertEqual(outcome.reason, "terminal")
        self.assertEqual(outcome.raw_endpoint_scores[0], None)
        self.fixture.change_receipt("left", lambda receipt: receipt.update(cpu_calls=1, fresh_engine=True, report={"raw_score": 0}))
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_mate_band_out_of_range_tie_and_below_margin_are_not_known(self):
        original = self.fixture.executions["left"]["receipt"]
        for raw, reason in ((29000, "mate_band"), (-29000, "mate_band"), (20001, "out_of_range"), (20, "tie_or_below_margin")):
            with self.subTest(score=raw):
                self.fixture.executions["left"]["receipt"] = original
                self.fixture.change_receipt("left", lambda receipt: receipt["report"].update(raw_score=raw))
                outcome = self.fixture.admit().outcome
                self.assertFalse(outcome.mask)
                self.assertEqual(outcome.sign, 0)
                self.assertEqual(outcome.reason, reason)
                self.assertEqual(outcome.raw_endpoint_scores[0], raw)
        self.fixture.criterion["minimum_margin"] = 2
        self.fixture.rebind_registered_assets()
        self.fixture.change_receipt("left", lambda receipt: receipt["report"].update(raw_score=21))
        self.assertEqual(self.fixture.admit().outcome.reason, "tie_or_below_margin")

    def test_masked_pair_cannot_hide_cleanup_identity_or_transport_failure(self):
        self.fixture.change_receipt("left", lambda receipt: receipt["report"].update(raw_score=20))
        self.assertFalse(self.fixture.admit().outcome.mask)
        changes = ({"owned_group_absent": False}, {"pipes_finished": False}, {"reaped": False}, {"spawned": False},
                   {"overflow": True}, {"transport_failure": "dispatch failed"}, {"source_path_stable": False},
                   {"loaded_executable": {"status": "unknown", "artifact": ordinal.byte_pin(self.fixture.binary),
                                          "scope": "caller_child_loaded_inode", "observed_before_stdin": True}})
        for change in changes:
            with self.subTest(process=change):
                self.fixture.observe("left", process_changes=change)
                self.fixture.refresh_pins()
                with self.assertRaises(ValueError):
                    self.fixture.admit()
        self.fixture.observe("left", launch_changes={"exit_code": 1})
        self.fixture.refresh_pins()
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_total_receipt_elapsed_must_fit_independent_caller_launch(self):
        self.fixture.change_receipt("left", lambda receipt: receipt.update(elapsed_ms=9999))
        with self.assertRaises(ValueError):
            self.fixture.admit()
        self.fixture.change_receipt("left", lambda receipt: receipt.update(elapsed_ms=8))
        self.fixture.observe("left", launch_changes={"elapsed_ms": 10000}, process_changes={"elapsed_ms": 10000})
        self.fixture.refresh_pins()
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_full_completion_pv_node_and_fresh_namespace_are_strict(self):
        original = self.fixture.executions["left"]["receipt"]
        changes = ({"pv_rules_validated": False}, {"pv": []}, {"best_move": True}, {"reused_completed_depth": 1},
                   {"root_restricted": True}, {"score_provenance": "unregistered value"}, {"nodes": 100001},
                   {"quiescence_nodes": 43}, {"tt_hits": 43}, {"completed_depth": 1}, {"requested_depth": 3},
                   {"score_scope": "frontier_only"}, {"nodes": 0, "quiescence_nodes": 0, "tt_hits": 0})
        for change in changes:
            with self.subTest(report=change):
                self.fixture.executions["left"]["receipt"] = original
                self.fixture.change_receipt("left", lambda receipt: receipt["report"].update(change))
                with self.assertRaises(ValueError):
                    self.fixture.admit()

    def test_root_board_legal_order_history_length_and_turn_are_exact(self):
        original = self.fixture.executions["left"]["receipt"]
        mutations = (lambda receipt: receipt["root"]["board64_piece_codes"].__setitem__(0, 12),
                     lambda receipt: receipt["endpoint"].update(known_history_positions=99),
                     lambda receipt: receipt["endpoint"].update(side_to_move="black"),
                     lambda receipt: receipt["root"]["legal_moves"].reverse(),
                     lambda receipt: receipt["endpoint"].update(repetition_history_complete=False))
        for mutation in mutations:
            self.fixture.executions["left"]["receipt"] = original
            self.fixture.change_receipt("left", mutation, recompute_conditions=True)
            with self.assertRaises(ValueError):
                self.fixture.admit()

    def test_line_order_repetitions_and_full_history_are_not_set_identity(self):
        state, history = self.fixture.anchor["rules_state_sha256"], self.fixture.anchor["rules_history_sha256"]
        left, right = self.fixture.anchor["legal_moves"][:2]
        self.assertNotEqual(ordinal.ordered_line_sha256(state, history, [left, right]),
                            ordinal.ordered_line_sha256(state, history, [right, left]))
        self.assertNotEqual(ordinal.ordered_line_sha256(state, history, [left]),
                            ordinal.ordered_line_sha256(state, history, [left, left]))
        self.assertNotEqual(ordinal.ordered_line_sha256(state, history, [left, left]),
                            ordinal.ordered_line_sha256(state, sha("different known history"), [left, left]))
        self.assertEqual(self.fixture.admit().outcome.sign, 1)
        self.fixture.plan["lines"][1] = copy.deepcopy(self.fixture.plan["lines"][0])
        self.fixture.raws["plan"] = ordinal.canonical(self.fixture.plan)
        self.fixture.refresh_pins()
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_registered_build_prepost_binary_and_actual_cli_capability_required(self):
        for name, mutation in (("build", lambda body: body["files_after"][0].update(sha256=sha("changed source after build"))),
                               ("capabilities", lambda body: body.pop("cpu_binary_sha256")),
                               ("source", lambda body: body["profile"].update(domain=cpu.REQUEST_SCHEMA))):
            with self.subTest(asset=name):
                original = self.fixture.raws[name]
                value = json.loads(original)
                mutation(value)
                self.fixture.raws[name] = ordinal.canonical(value)
                self.fixture.rebind_registered_assets()
                with self.assertRaises(ValueError):
                    self.fixture.admit()
                self.fixture.raws[name] = original
        self.fixture.registration["binary_artifact"]["sha256"] = sha("different loaded image")
        self.fixture.registration["checker_namespace_sha256"] = ordinal.checker_namespace(self.fixture.registration)
        self.fixture.raws["registration"] = ordinal.canonical(self.fixture.registration)
        self.fixture.refresh_pins()
        with self.assertRaises(ValueError):
            self.fixture.admit()

    def test_duplicate_fields_float_scalars_and_missing_execution_are_rejected(self):
        raw = self.fixture.raws["criterion"]
        self.fixture.raws["criterion"] = raw[:-1] + b',"minimum_margin":1}'
        self.fixture.refresh_pins()
        with self.assertRaises(ValueError):
            self.fixture.admit()
        self.fixture.raws["criterion"] = raw
        self.fixture.refresh_pins()
        executions = {"left": self.fixture.executions["left"]}
        with self.assertRaises(ValueError):
            self.fixture.admit(executions=executions)
        self.fixture.criterion["minimum_margin"] = 1.0
        with self.assertRaises(ValueError):
            ordinal.canonical(self.fixture.criterion)


if __name__ == "__main__":
    unittest.main()
