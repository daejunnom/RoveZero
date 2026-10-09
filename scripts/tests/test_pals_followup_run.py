"""Finite wrapper boundary checks. These tests never spawn an engine or load NN."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
