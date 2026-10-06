import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import ManagedBuild, StorageError, checked, read_record, tree_size
from managed_process import run_group


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rovezero-storage-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.root = self.base / "owned-build"
        self.model = self.source / "original-model.bin"
        self.model.write_bytes(b"retained-original")

    def acquire(self, source=None, **limits):
        return ManagedBuild(self.root, source or self.source, **limits).acquire()

    def test_fixed_slot_reuse_and_cleanup_even_after_command_failure(self):
        first = self.acquire()
        (first.target / "reusable").write_bytes(b"cache")
        (first.scratch / "test-remnant").write_bytes(b"temporary")
        with patch.dict(os.environ, {"CARGO_TARGET_DIR": str(first.target)}):
            env = first.environment()
        self.assertEqual(env["CARGO_INCREMENTAL"], "0")
        self.assertEqual(env["TMPDIR"], str(first.scratch))
        first.finish(exit_code=7, tree_gone=True)
        self.assertFalse(first.scratch.exists())
        second = self.acquire()
        self.assertEqual(first.slot, second.slot)
        self.assertEqual((second.target / "reusable").read_bytes(), b"cache")
        second.finish(exit_code=0, tree_gone=True)
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    def test_post_run_budget_reset_preserves_source(self):
        owner = self.acquire(slot_limit=16_384, total_limit=49_152)
        (owner.target / "too-large").write_bytes(b"x" * 20_000)
        receipt = owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue(receipt["post_run_budget_reset"])
        self.assertFalse(owner.target.exists())
        self.assertLessEqual(tree_size(owner.slot), owner.slot_limit)
        self.assertTrue(self.model.exists())

    def test_pre_run_oversize_cache_and_aggregate_retention(self):
        first = self.acquire(slot_limit=4096, total_limit=6144)
        first.finish(exit_code=0, tree_gone=True)
        (first.target / "oversize-completed-cache").write_bytes(b"x" * 10_000)
        reuse = self.acquire(slot_limit=4096, total_limit=6144)
        self.assertFalse((reuse.target / "oversize-completed-cache").exists())
        (reuse.target / "within-budget").write_bytes(b"x" * 3000)
        reuse.finish(exit_code=0, tree_gone=True)
        other = self.base / "other-source"
        other.mkdir()
        second = self.acquire(other, slot_limit=4096, total_limit=6144)
        (second.target / "within-budget").write_bytes(b"x" * 3000)
        second.finish(exit_code=0, tree_gone=True)
        self.assertFalse(first.slot.exists())
        self.assertTrue(second.slot.exists())
        self.assertLessEqual(tree_size(self.root / "slots"), 6144)

    def test_live_and_abandoned_owner_are_not_stolen(self):
        owner = self.acquire()
        for _ in range(2):
            with self.assertRaises(StorageError):
                self.acquire()
            self.assertTrue(owner.lock.exists())
        owner.finish(exit_code=0, tree_gone=True)

    def test_unverified_descendants_retain_lease_and_scratch(self):
        owner = self.acquire()
        (owner.scratch / "input").write_bytes(b"still-in-use")
        with self.assertRaises(StorageError):
            owner.finish(exit_code=124, tree_gone=False)
        self.assertTrue(owner.lock.exists())
        self.assertTrue((owner.scratch / "input").exists())

    def test_changed_record_or_unknown_content_preserves_all(self):
        owner = self.acquire()
        record = read_record(owner.slot / "record.json")
        record["token"] = "changed"
        (owner.slot / "record.json").write_text(json.dumps(record))
        with self.assertRaises(StorageError):
            owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue(owner.lock.exists())
        self.assertTrue(owner.scratch.exists())

    def test_unknown_catalog_never_adopted(self):
        self.root.mkdir()
        (self.root / "unowned").mkdir()
        with self.assertRaises(StorageError):
            self.acquire()
        self.assertTrue((self.root / "unowned").is_dir())

    def test_checkout_limit_evicts_only_completed_owned_slot(self):
        slots = []
        for index in range(5):
            source = self.base / f"source-{index}"
            source.mkdir()
            owner = self.acquire(source, slots_limit=4)
            slots.append(owner.slot)
            owner.finish(exit_code=0, tree_gone=True)
        self.assertFalse(slots[0].exists())
        self.assertTrue(all(path.exists() for path in slots[1:]))
        self.assertTrue(self.model.exists())

    def test_link_escape_rejected_before_deleting_any_file(self):
        owner = self.acquire()
        (owner.scratch / "ordinary").write_bytes(b"owned")
        link = owner.scratch / "escape"
        try:
            link.symlink_to(self.source, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this Windows account")
        with self.assertRaises(StorageError):
            owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue((owner.scratch / "ordinary").exists())
        self.assertEqual(self.model.read_bytes(), b"retained-original")
        link.unlink()  # Test fixture cleanup only.

    def test_credential_and_checkout_destinations_rejected_without_read(self):
        with self.assertRaises(StorageError):
            checked(self.base / ".env" / "unopened")
        with self.assertRaises(StorageError):
            ManagedBuild(self.source / "target", self.source)

    def test_internal_venv_link_is_removed_without_traversal(self):
        owner = self.acquire()
        (owner.scratch / "lib").mkdir()
        (owner.scratch / "lib" / "owned-copy").write_bytes(b"owned")
        link = owner.scratch / "lib64"
        try:
            link.symlink_to("lib", target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable on this Windows account")
        amount = tree_size(owner.slot)
        self.assertGreater(amount, 0)
        owner.finish(exit_code=0, tree_gone=True)
        self.assertFalse(owner.scratch.exists())
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    def test_live_sampling_allows_disappearance_but_cleanup_preflight_does_not(self):
        original_scandir = os.scandir
        class VanishingScan:
            def __enter__(inner):
                with original_scandir(self.source) as scan:
                    entries = list(scan)
                self.model.unlink()
                return iter(entries)
            def __exit__(inner, *args):
                pass
        for live in (True, False):
            self.model.write_bytes(b"owned test input")
            with patch("managed_storage.os.scandir", return_value=VanishingScan()):
                if live:
                    self.assertEqual(tree_size(self.source, allow_disappearing=True), 0)
                else:
                    with self.assertRaises(FileNotFoundError):
                        tree_size(self.source)


class ProcessTests(unittest.TestCase):
    def test_normal_exit_and_orphan_descendant_are_drained(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-owned-process-") as directory:
            marker = Path(directory) / "late-write"
            child = f"import time,pathlib; time.sleep(2); pathlib.Path({str(marker)!r}).write_text('escaped')"
            program = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
            code, gone, failure = run_group(
                [sys.executable, "-c", program], directory, os.environ.copy(), 5, lambda: None)
            self.assertEqual(code, 0)
            self.assertTrue(gone)
            self.assertIsNone(failure)
            self.assertFalse(marker.exists())

    def test_timeout_and_budget_error_return_failure_after_owned_cleanup(self):
        for error in (False, True):
            def budget():
                if error:
                    raise StorageError("synthetic storage limit")
            code, gone, failure = run_group(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                Path.cwd(), os.environ.copy(), 0.3, budget)
            self.assertEqual(code, 125 if error else 124)
            self.assertTrue(gone)
            self.assertEqual(failure is not None, error)


if __name__ == "__main__":
    unittest.main()
