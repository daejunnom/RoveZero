"""Small stdlib bootstrap fixtures; no process migration or training proof."""
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural runtime import forbidden in bootstrap fixtures")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_training_bootstrap as bootstrap
finally:
    sys.meta_path.remove(_guard)


def pin(path, content=b"fixture-only"):
    return {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def fixture():
    """A projection only: no original Rust custody, files or socket exists."""
    source_root = "/tmp/rovent-fixture-output/source"
    package = bootstrap.MODULE_RELATIVE
    sources = [{"relative_path": package + name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in (("__init__.py", b"# package fixture\n"),
                              ("rovent_training_child.py", b"# child fixture\n"),
                              ("rovent_training_bootstrap.py", b"# bootstrap fixture\n"))]
    inventory = {entry["relative_path"]: {name: entry[name] for name in ("bytes", "sha256")} for entry in sources}
    policy = {"cpu_quota_millicpus": 2000, "memory_high_bytes": 6 << 30,
        "memory_max_bytes": 12 << 30, "pids_max": 64, "affinity": [0, 2], "device": "cpu"}
    readback = {"cpu_quota_us": 2000000, "cpu_period_us": 1000000, "memory_high_bytes": 6 << 30,
        "memory_max_bytes": 12 << 30, "memory_swap_max_bytes": 0, "pids_max": 64}
    corpus = {"manifest": pin("/tmp/rovent-fixture-corpus/manifest.json", b"manifest"),
        "index": pin("/tmp/rovent-fixture-corpus/index.json", b"index"), "root": "/tmp/rovent-fixture-corpus"}
    registration = {"schema": bootstrap.REGISTRATION_DOMAIN, "execution_id": "fixture-execution",
        "action": "train", "export_request": None,
        "recipe": pin("/tmp/rovent-fixture-input/recipe.json"), "membership": pin("/tmp/rovent-fixture-input/membership.json"),
        "window_spec": pin("/tmp/rovent-fixture-input/window.json"),
        "trainer_source_sha256": bootstrap._digest(bootstrap.SOURCE_DOMAIN, inventory),
        "python_runtime_sha256": "1" * 64, "model_semantics_sha256": "2" * 64,
        "current_corpus": corpus, "ancestor_corpora": [], "checkpoint": None,
        "pc_initial": pin("/tmp/rovent-fixture-input/untrained.pt"),
        "nnuec_initial": pin("/tmp/rovent-fixture-input/nnuec.json"),
        "rules_profile": pin("/tmp/rovent-fixture-input/rules.json"), "checkpoint_mode": "initialize",
        "expected_initial_completed": dict.fromkeys(("proposer", "critic", "validator"), 0),
        "expected_ancestor_completed": dict.fromkeys(("proposer", "critic", "validator"), 0),
        "resources_policy": policy, "require_total_quota": False,
        "output_root": "/tmp/rovent-fixture-output", "control_root": "/tmp/rovent-fixture-control",
        "checkpoint_directory": "/tmp/rovent-fixture-output/checkpoints", "receipt_path": "/tmp/rovent-fixture-output/result.json",
        "max_rpc_messages": 128, "max_rpc_bytes": 1 << 20, "max_read_bytes": 1 << 20, "max_output_bytes": 32 << 20}
    return {"schema": bootstrap.STARTUP_DOMAIN, "execution_id": "fixture-execution", "window_id": "fixture-window",
        "registration_sha256": "3" * 64, "registration": registration,
        "control": {"socket_path": "/tmp/rovent-fixture-control/control.sock", "parent_pid": 17, "parent_start_ticks": 23,
            "control_device": 1, "control_inode": 2},
        "original_started_ns": 100, "execution_until_ns": 10000, "whole_until_ns": 11000,
        "source_root": source_root, "package_root": source_root + "/" + bootstrap.PACKAGE_RELATIVE,
        "source_files": sources, "dependency_directories": [{"path": "/tmp/rovent-fixture-dependencies", "device": 1, "inode": 3}],
        "runtime_module_pins": [pin("/tmp/rovent-fixture-dependencies/runtime.py")],
        "python_runtime": {"version": "3.12.3", "executable": pin("/tmp/rovent-fixture-python")},
        "resources": {"domain": bootstrap.RESOURCE_DOMAIN, "migration": "child_self_only_before_torch",
            "path": "/sys/fs/cgroup/rovent-fixture", "device": 4, "inode": 5, "membership": "/rovent-fixture",
            "policy": copy.deepcopy(policy), "prepared_readback": copy.deepcopy(readback), "readback": copy.deepcopy(readback),
            "available_controllers_snapshot": ["cpu", "memory", "pids"],
            "subtree_controllers_snapshot": ["cpu", "memory", "pids"],
            "scope": "configured_leaf_readback_not_child_membership_or_lifetime_proof"}}


def validate(value, *, raw=None, **changes):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() if raw is None else raw
    arguments = {"expected_sha256": hashlib.sha256(raw).hexdigest(), "expected_bytes": len(raw),
        "execution_until_ns": 10000, "whole_until_ns": 11000, "now_ns": 200}
    arguments.update(changes)
    return bootstrap.validate_startup(raw, **arguments)


class RoventTrainingBootstrapProjectionTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_exact_startup_is_only_a_projection_and_has_no_os_side_effect(self):
        value = fixture()
        with mock.patch.object(bootstrap, "_open_directory", side_effect=AssertionError("no file access")), \
                mock.patch.object(os, "open", side_effect=AssertionError("no file access")):
            result = validate(value)
        self.assertEqual(result, value)
        self.assertIsNot(result, value)
        self.assertEqual(result["registration_sha256"], "3" * 64)
        self.assertNotIn("execution_authorized", result)

    def test_rejects_byte_hash_deadline_and_unknown_field_changes(self):
        value = fixture()
        for changed in ({"expected_sha256": "0" * 64}, {"expected_bytes": 1},
                {"execution_until_ns": 9999}, {"now_ns": 10000}, {"whole_until_ns": 11001}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate(value, **changed)
        extra = copy.deepcopy(value)
        extra["serialized_authority"] = True
        with self.assertRaises(ValueError):
            validate(extra)

    def test_rejects_duplicate_nonfinite_and_invalid_utf8_before_projection(self):
        for raw in (b'{"schema":"one","schema":"two"}', b'{"value":NaN}', b'{"value":1e9999}', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate({}, raw=raw)

    def test_source_domain_inventory_and_current_entry_pins_are_separate_from_native_source(self):
        value = fixture()
        value["registration"]["trainer_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source inventory"):
            validate(value)
        value = fixture()
        value["source_files"].pop()
        inventory = {item["relative_path"]: {key: item[key] for key in ("bytes", "sha256")} for item in value["source_files"]}
        value["registration"]["trainer_source_sha256"] = bootstrap._digest(bootstrap.SOURCE_DOMAIN, inventory)
        with self.assertRaisesRegex(ValueError, "entry/bootstrap/package"):
            validate(value)

    def test_resources_dependencies_and_checkpoint_modes_fail_closed(self):
        mutations = (
            lambda value: value["resources"]["readback"].update(memory_swap_max_bytes=1),
            lambda value: value["registration"]["resources_policy"].update(affinity=[0, 0]),
            lambda value: value.update(dependency_directories=["/tmp/unpinned-site-packages"]),
            lambda value: value["registration"].update(checkpoint_mode="resume"),
            lambda value: value["registration"].update(max_read_bytes=1),
            lambda value: value["registration"].update(require_total_quota=1),
            lambda value: value["registration"].update(action="export"),
            lambda value: value["registration"].update(action="unknown"),
            lambda value: value["registration"].update(export_request=pin("/tmp/rovent-fixture-input/export.json")),
            lambda value: value["source_files"][0].update(relative_path=bootstrap.MODULE_RELATIVE + "../outside.py"),
        )
        for mutate in mutations:
            value = fixture()
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate(value)

    def test_export_projection_requires_resume_and_a_pin_without_reading_or_issuing_authority(self):
        value = fixture()
        value["registration"].update(action="export", checkpoint_mode="resume",
            checkpoint=pin("/tmp/rovent-fixture-input/completed.pt"), pc_initial=None, nnuec_initial=None,
            export_request=pin("/tmp/rovent-fixture-input/export.json"))
        with mock.patch.object(os, "open", side_effect=AssertionError("export projection must not read a file")):
            result = validate(value)
        self.assertEqual(result["registration"]["action"], "export")
        self.assertNotIn("export_authorized", result)
        self.assertEqual(result["registration"]["export_request"], value["registration"]["export_request"])

    def test_nonisolated_child_rejects_before_file_resource_or_neural_operations(self):
        flags = types.SimpleNamespace(isolated=0, no_site=1, dont_write_bytecode=1)
        with mock.patch.object(bootstrap, "_require_linux_nofollow"), mock.patch.object(sys, "flags", flags), \
                mock.patch.object(bootstrap, "_open_file", side_effect=AssertionError("no file access")), \
                mock.patch.object(bootstrap, "_join_resources", side_effect=AssertionError("no migration")):
            with self.assertRaisesRegex(bootstrap.OriginalBootstrapUnavailable, "-I -S -B"):
                bootstrap.bootstrap_original_child({}, until_ns=10000, clock=lambda: 200)

    def test_shared_external_read_credit_is_not_permission_or_a_refund(self):
        budget = bootstrap.PinnedReadBudget(10, until_ns=1000, clock=lambda: 200)
        self.assertEqual(budget.account_external(6), 6)
        with self.assertRaisesRegex(ValueError, "aggregate read credit"):
            budget.account_external(5)
        self.assertEqual(budget.spent, 6)
        with self.assertRaises(ValueError):
            budget.account_external(True)
        self.assertEqual(budget.account_external(4), 10)
        expired = bootstrap.PinnedReadBudget(10, until_ns=200, clock=lambda: 200)
        with self.assertRaises(TimeoutError):
            expired.account_external(1)


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "O_NOFOLLOW"), "actual nofollow fixture uses Linux")
class RoventTrainingBootstrapPinTests(unittest.TestCase):
    def test_small_json_pin_and_streamed_read_share_exact_aggregate_credit(self):
        with tempfile.TemporaryDirectory(prefix="rovent-bootstrap-fixture-") as temporary:
            path = Path(temporary) / "metadata.json"
            raw = b'{"fixture":"no-model-no-authority"}'
            path.write_bytes(raw)
            declared = pin(str(path), raw)
            budget = bootstrap.PinnedReadBudget(len(raw) * 2, until_ns=1000, clock=lambda: 200)
            self.assertEqual(budget.json(declared), {"fixture": "no-model-no-authority"})
            observed = budget.verify(declared)
            self.assertEqual(observed["sha256"], declared["sha256"])
            self.assertEqual(budget.spent, len(raw) * 2)
            with self.assertRaisesRegex(ValueError, "aggregate read credit"):
                budget.json(declared)

    def test_links_wrong_bytes_and_changed_hash_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="rovent-bootstrap-fixture-") as temporary:
            root = Path(temporary)
            path = root / "metadata.json"
            raw = b'{"fixture":1}'
            path.write_bytes(raw)
            declared = pin(str(path), raw)
            link = root / "linked.json"
            link.symlink_to(path)
            for changed in ({**declared, "path": str(link)}, {**declared, "sha256": "0" * 64},
                    {**declared, "bytes": len(raw) + 1}):
                with self.subTest(changed=changed), self.assertRaises((ValueError, OSError)):
                    bootstrap.read_original_pin(changed, until_ns=1000, clock=lambda: 200)
            hardlink = root / "hardlink.json"
            os.link(path, hardlink)
            with self.assertRaisesRegex(ValueError, "single-link"):
                bootstrap.read_original_pin(declared, until_ns=1000, clock=lambda: 200)

    def test_failed_file_pin_retains_charged_reads_and_quarantines_the_account(self):
        with tempfile.TemporaryDirectory(prefix="rovent-bootstrap-fixture-") as temporary:
            path = Path(temporary) / "metadata.json"
            raw = b'{"fixture":1}'
            path.write_bytes(raw)
            declared = {**pin(str(path), raw), "sha256": "0" * 64}
            budget = bootstrap.PinnedReadBudget(1000, until_ns=1000, clock=lambda: 200)
            with self.assertRaises(ValueError):
                budget.read(declared)
            self.assertEqual(budget.spent, len(raw))
            with self.assertRaisesRegex(ValueError, "quarantined"):
                budget.read(pin(str(path), raw))


@unittest.skipUnless(sys.platform == "linux" and hasattr(os, "O_NOFOLLOW"), "owned directory fixtures use Linux")
class RoventOwnedRuntimeEnvironmentTests(unittest.TestCase):
    def test_only_public_runtime_paths_are_set_under_the_owned_output(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            startup = {"registration": {"output_root": str(root)}}
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {"FIXTURE_MARKER": "unchanged"}, clear=True):
                report = bootstrap._prepare_owned_runtime_environment(startup, until_ns=1000, clock=lambda: 200)
                self.assertEqual(set(report["variables"]), set(bootstrap._RUNTIME_ENVIRONMENT))
                self.assertEqual(set(os.environ), {*bootstrap._RUNTIME_ENVIRONMENT, "FIXTURE_MARKER"})
                self.assertEqual(os.environ["FIXTURE_MARKER"], "unchanged")
                for name, value in report["variables"].items():
                    self.assertEqual(os.environ[name], value)
                    self.assertTrue(Path(value).is_relative_to(root))
                    self.assertEqual(stat.S_IMODE(Path(value).stat().st_mode), 0o700)
                self.assertEqual(len(report["directories"]), 7)
                self.assertFalse(report["global_cache_inspected"])
                self.assertFalse(report["automatic_cleanup"])
            self.assertEqual(set(path.name for path in root.iterdir()), {"tmp", "home", "runtime-cache"})

    def test_existing_nonempty_runtime_directory_is_preserved_and_not_published(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            (root / "tmp").mkdir(mode=0o700)
            retained = root / "tmp" / "retained-fixture.txt"
            retained.write_text("preserve fixture evidence", encoding="utf-8")
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {"FIXTURE_MARKER": "unchanged"}, clear=True):
                with self.assertRaisesRegex(ValueError, "new or empty"):
                    bootstrap._prepare_owned_runtime_environment({"registration": {"output_root": str(root)}},
                        until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {"FIXTURE_MARKER": "unchanged"})
            self.assertEqual(retained.read_text(encoding="utf-8"), "preserve fixture evidence")
            self.assertEqual(set(path.name for path in root.iterdir()), {"tmp"})

    def test_linked_runtime_directory_is_not_followed_or_cleaned(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            other = Path(temporary) / "other-fixture-owner"
            other.mkdir(mode=0o700)
            (root / "tmp").symlink_to(other, target_is_directory=True)
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises((ValueError, OSError)):
                    bootstrap._prepare_owned_runtime_environment({"registration": {"output_root": str(root)}},
                        until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {})
            self.assertTrue((root / "tmp").is_symlink())
            self.assertEqual(list(other.iterdir()), [])

    def test_bad_owner_mode_or_expired_window_does_not_create_a_runtime_tree(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            startup = {"registration": {"output_root": str(root)}}
            with mock.patch.object(tempfile, "tempdir", None), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(TimeoutError):
                    bootstrap._prepare_owned_runtime_environment(startup, until_ns=200, clock=lambda: 200)
                root.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "0700"):
                    bootstrap._prepare_owned_runtime_environment(startup, until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {})
            self.assertEqual(list(root.iterdir()), [])

    def test_previously_cached_temporary_path_is_rejected_without_inspecting_it(self):
        with tempfile.TemporaryDirectory(prefix="rovent-owned-environment-fixture-") as temporary:
            root = Path(temporary) / "output"
            root.mkdir(mode=0o700)
            with mock.patch.object(tempfile, "tempdir", "not-inspected-fixture-path"), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(bootstrap.OriginalBootstrapUnavailable, "cached"):
                    bootstrap._prepare_owned_runtime_environment({"registration": {"output_root": str(root)}},
                        until_ns=1000, clock=lambda: 200)
                self.assertEqual(dict(os.environ), {})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
