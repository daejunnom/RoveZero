"""Finite wrapper boundary checks. These tests never spawn an engine or load NN."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pals_followup_run as wrapper


class FollowupPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rovezero-followup-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assets = self.root / "assets"
        self.output = self.root / "output"
        self.assets.mkdir()
        self.output.mkdir()

    def write(self, path, value):
        data = json.dumps(value).encode("utf-8")
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def fixture(self):
        resource = {"cpu_affinity": [2, 3], "cpu_threads": 2,
                    "memory_high_bytes": 6 * wrapper.GIB,
                    "memory_max_bytes": 12 * wrapper.GIB, "swap_max_bytes": 0}
        manifest = {"schema_version": 4, "contract_revision": "pals/0.2",
                    "purpose": "arena_pilot", "training_executed": False,
                    "pilot": {"games": 2, "base_ms": 120000, "increment_ms": 1000,
                              "max_plies": 256, "wall_time_max_ms": 900000,
                              "cleanup_max_ms": 30000},
                    "resources": [resource, copy.deepcopy(resource)]}
        return {"domain": "rz-pals-arena-launch-v4/1", "sha256": "a" * 64,
                "input": {"domain": "rz-pals-arena-launch-v4/1",
                          "semantic_lock": {"manifest": manifest}}}

    def args(self, value):
        path = self.root / "lock.json"
        digest = self.write(path, value)
        return argparse.Namespace(lock=path, lock_sha256=digest,
                                  cpu_affinity=[3, 2], stage="execute", asset_root=self.assets,
                                  output_root=self.output, label="finite-pilot")

    def test_registered_pilot_produces_finite_argv_without_execution(self):
        binary = self.root / "registered-runner"
        command, evidence = wrapper.arena_plan(self.args(self.fixture()), binary)
        self.assertEqual(command, [str(binary), "execute", str(self.root / "lock.json"),
                                   str(self.assets), str(self.output), "finite-pilot"])
        self.assertEqual(evidence["pilot"]["wall_time_max_ms"], 900000)
        self.assertFalse((self.output / "finite-pilot").exists())

    def test_larger_resources_missing_peer_and_diagnostic_are_rejected(self):
        for alteration in ("memory", "resources", "diagnostic", "wall", "ply"):
            value = self.fixture()
            manifest = value["input"]["semantic_lock"]["manifest"]
            if alteration == "memory":
                manifest["resources"][1]["memory_max_bytes"] += 1
            elif alteration == "resources":
                manifest["resources"] = []
            elif alteration == "diagnostic":
                manifest["purpose"] = "diagnostic"
            elif alteration == "wall":
                manifest["pilot"]["wall_time_max_ms"] += 1
            else:
                manifest["pilot"]["max_plies"] += 1
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                wrapper.arena_plan(self.args(value), self.root / "runner")

    def test_duplicate_json_keys_and_mutated_pins_fail_before_plan(self):
        path = self.root / "duplicate.json"
        path.write_text('{"schema":4,"schema":3}', encoding="utf-8")
        with self.assertRaises(ValueError):
            wrapper.bounded_json(path)
        path.write_bytes(b"actual executable")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_bytes(b"replaced executable")
        with self.assertRaises(ValueError):
            wrapper.pinned(path, digest, 1024)

    def test_new_model_profile_does_not_reinterpret_legacy_export(self):
        export = self.root / "export.json"
        digest = self.write(export, {"schema": "rovezero.pals-model.v1", "config": {}})
        args = argparse.Namespace(export=export, export_sha256=digest,
                                  model_profile="full_line_interaction_v2")
        with self.assertRaises(ValueError):
            wrapper.model_plan(args, self.root / "validator")

    def gc_args(self, value):
        path = self.root / "external-gc.json"
        digest = self.write(path, value)
        return argparse.Namespace(external_gc_ledger=path, external_gc_ledger_sha256=digest)

    def gc_fixture(self):
        return {"schema": wrapper.EXTERNAL_GC_LEDGER_SCHEMA,
                "requests": [{"id": "request-1", "product_hook": False},
                             {"id": "request-2", "product_hook": False}],
                "actual_executions": [{"id": "actual-1", "request_id": "request-1",
                    "scope": "owned_external_controller", "full_gc_calls": 1, "product_hook": False}],
                "memory_contention_observed": True, "performance_evidence_eligible": False}

    def test_external_gc_requests_are_not_counted_as_actual_calls(self):
        evidence = wrapper.external_gc_observation(self.gc_args(self.gc_fixture()))
        self.assertEqual(evidence["request_count"], 2)
        self.assertEqual(evidence["actual_execution_count"], 1)
        self.assertEqual(evidence["actual_full_gc_calls"], 1)
        self.assertFalse(evidence["performance_evidence_eligible"])
        self.assertFalse(evidence["GC_invoked_by_wrapper"])
        self.assertEqual(evidence["product_GC_hooks"], 0)

    def test_unregistered_gc_is_unknown_and_partial_pin_is_rejected(self):
        evidence = wrapper.external_gc_observation(argparse.Namespace(
            external_gc_ledger=None, external_gc_ledger_sha256=None))
        self.assertEqual(evidence["actual_full_gc_calls"], "unknown")
        self.assertFalse(evidence["performance_evidence_eligible"])
        with self.assertRaises(ValueError):
            wrapper.external_gc_observation(argparse.Namespace(
                external_gc_ledger=self.root / "missing.json", external_gc_ledger_sha256=None))

    def test_external_gc_duplicate_actual_product_hook_and_numeric_boolean_rejected(self):
        for alteration in ("duplicate", "product", "count", "contention", "unlinked", "performance"):
            value = self.gc_fixture()
            if alteration == "duplicate":
                value["actual_executions"].append(copy.deepcopy(value["actual_executions"][0]))
            elif alteration == "product":
                value["actual_executions"][0]["scope"] = "product"
            elif alteration == "count":
                value["actual_executions"][0]["full_gc_calls"] = True
            elif alteration == "contention":
                value["memory_contention_observed"] = 1
            elif alteration == "unlinked":
                value["actual_executions"][0]["request_id"] = "absent-request"
            else:
                value["performance_evidence_eligible"] = True
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                wrapper.external_gc_observation(self.gc_args(value))

    def test_fresh_attempt_output_budget_does_not_scan_runtime_or_models(self):
        report = self.root / "report.json"
        report.write_bytes(b"12345678")
        args = argparse.Namespace(mode="model-check", model_report=report, output_bytes_max=7)
        with self.assertRaises(RuntimeError):
            wrapper.check_colab_output_budget(args)
        args.output_bytes_max = 8
        wrapper.check_colab_output_budget(args)

    def test_colab_planning_uses_separate_schema_and_never_spawns(self):
        lock = self.fixture()
        args = self.args(lock)
        binary = self.root / "runner"
        binary.write_bytes(b"registered fixture")
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        receipt = self.root / "colab-receipt.json"
        argv = ["wrapper", "--binary", str(binary), "--binary-sha256", digest,
                "--receipt", str(receipt), "--cpu-affinity", "2", "3",
                "--resource-profile", "colab-observed", "arena", "--stage", "prepare",
                "--lock", str(args.lock), "--lock-sha256", args.lock_sha256,
                "--asset-root", str(self.assets), "--output-root", str(self.output), "--label", "dry-run"]
        with (mock.patch.object(sys, "argv", argv), mock.patch.object(wrapper, "run_group") as spawn,
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(wrapper.main(), 0)
        spawn.assert_not_called()
        evidence = json.loads(receipt.read_text())
        self.assertEqual(evidence["schema"], wrapper.COLAB_SCHEMA)
        self.assertFalse(evidence["strict_local_resources_admitted"])
        self.assertFalse(evidence["performance_evidence_eligible"])
        self.assertFalse(evidence["native_arena_policy_relaxed"])
        self.assertEqual(evidence["actual_resources"], "unknown")
        self.assertEqual(evidence["external_GC"]["actual_full_gc_calls"], "unknown")


class ResourceObservationTests(unittest.TestCase):
    def reads(self, values):
        def read(path, **unused):
            if path.as_posix() == "/proc/self/cgroup":
                return "0::/colab-observed\n"
            if path.name not in values:
                raise FileNotFoundError(path.name)
            return values[path.name]
        return read

    def observe(self, values, cpus=(0, 2), writable=False):
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value=set(cpus), create=True), \
                mock.patch.object(Path, "read_text", self.reads(values)), \
                mock.patch.object(wrapper.os, "access", return_value=writable):
            return wrapper.observe_resources([0, 2])

    def values(self):
        return {"cpu.max": "200000 100000", "memory.high": str(6 * wrapper.GIB),
                "memory.max": str(12 * wrapper.GIB), "memory.swap.max": "0", "pids.max": "128"}

    def test_matching_observations_do_not_promote_to_strict_local(self):
        evidence = self.observe(self.values(), writable=True)
        self.assertTrue(all(row["status"] == "satisfied" for row in evidence["observations"].values()))
        self.assertFalse(evidence["strict_local_resources_admitted"])
        self.assertFalse(evidence["settings_installed"])
        self.assertFalse(evidence["cgroup_write_performed"])
        self.assertEqual(evidence["cgroup_write_capability"]["status"], "observed_capability_only")
        self.assertFalse(evidence["cgroup_write_capability"]["applied_settings_verified"])

    def test_colab_unlimited_or_different_resources_are_unfulfilled(self):
        values = self.values()
        values.update({"cpu.max": "max 100000", "memory.high": "max", "memory.max": "max"})
        evidence = self.observe(values, cpus=(0, 1, 2, 3))
        for name in ("cpu_affinity", "cpu_quota", "memory.high", "memory.max"):
            self.assertEqual(evidence["observations"][name]["status"], "unfulfilled")
        self.assertEqual(evidence["cgroup_write_capability"]["status"], "unfulfilled")
        self.assertFalse(evidence["strict_local_resources_admitted"])

    def test_missing_or_malformed_quota_and_memory_are_unknown(self):
        evidence = self.observe({"cpu.max": "nan 0", "memory.high": "not-a-limit"})
        for name in ("cpu_quota", "memory.high", "memory.max", "memory.swap.max", "pids.max"):
            self.assertEqual(evidence["observations"][name]["status"], "unknown")

    def test_unsafe_membership_is_not_used_for_limit_reads(self):
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value={0, 2}, create=True), \
                mock.patch.object(Path, "read_text", return_value="0::/../outside\n") as reads:
            evidence = wrapper.observe_resources([0, 2])
        self.assertEqual(reads.call_count, 1)
        self.assertIsNone(evidence["cgroup_membership"])
        self.assertEqual(evidence["observations"]["cpu_quota"]["status"], "unknown")

    def test_non_linux_colab_environment_is_explicitly_unfulfilled(self):
        with mock.patch.object(wrapper.sys, "platform", "win32"), \
                mock.patch.object(Path, "read_text") as reads:
            evidence = wrapper.observe_resources([0, 2])
        reads.assert_not_called()
        self.assertEqual(evidence["execution_platform_status"], "unfulfilled")
        self.assertFalse(evidence["strict_local_resources_admitted"])

    def test_legacy_strict_local_checks_are_still_fail_closed(self):
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value={0, 2}, create=True), \
                mock.patch.object(Path, "read_text", self.reads(self.values())):
            evidence = wrapper.check_resources([0, 2])
        self.assertEqual(evidence["cpu_affinity"], [0, 2])
        self.assertNotIn("strict_local_resources_admitted", evidence)
        with mock.patch.object(wrapper.sys, "platform", "linux"), \
                mock.patch.object(wrapper.os, "sched_getaffinity", return_value={0, 1, 2}, create=True):
            with self.assertRaises(ValueError):
                wrapper.check_resources([0, 2])


if __name__ == "__main__":
    unittest.main()
