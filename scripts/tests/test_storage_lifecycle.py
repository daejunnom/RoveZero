"""Real temporary-file/owner boundaries; no NN, GPU, Cargo or real data cleanup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import ManagedBuild, StorageError, recover_build, tree_size, write_record
from managed_outputs import (OutputLease, OutputBudgetError, publish_runtime,
                             recover_output, sample_output_budget)
from storage_guard import CatalogGuard, lifetime
from storage_maintenance import inventory


def pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rovezero-lifecycle-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()
        self.catalog = self.base / "outputs"
        self.build = self.base / "build"

    def lease(self, name="run", **kwargs):
        owner = OutputLease(self.base / name, self.source, catalog=self.catalog, **kwargs).acquire()
        self.addCleanup(owner.guard.release)
        return owner

    def test_retained_evidence_counts_after_expiry_no_automatic_deletion(self):
        first = self.lease(reserve=8192, total_limit=12000)
        evidence = first.output / "unique.pgn"
        evidence.write_bytes(b"x" * 7000)
        first.finish(status="failed_evidence_preserved", tree_gone=True)
        with patch("managed_outputs.time.time_ns", return_value=10**30):
            with self.assertRaisesRegex(StorageError, "budget"):
                self.lease("next", reserve=8192, total_limit=12000)
        self.assertEqual(evidence.stat().st_size, 7000)
        self.assertFalse((self.base / "next").exists())
        self.assertGreater(sample_output_budget(self.catalog), 7000)

    def test_live_and_interrupted_output_owner_never_stolen(self):
        owner = self.lease()
        with self.assertRaises(StorageError):
            self.lease("second")
        owner.guard.release()  # Simulate lost gate; not a child closure proof.
        with self.assertRaisesRegex(StorageError, "interrupted"):
            self.lease("second")
        with self.assertRaises(StorageError):
            recover_output(self.catalog, owner.key, pin(self.catalog / (owner.key + ".json")))
        self.assertTrue(owner.output.exists())

    def test_unknown_tree_keeps_admission_closed_until_real_outer_closure(self):
        build = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(build.guard.release)
        with patch.dict(os.environ, {"RZ_MANAGED_BUILD_ROOT": str(build.root),
                                     "RZ_MANAGED_BUILD_TOKEN": build.token}):
            owner = self.lease(outer_supervised=True)
        owner.finish(status="child_returned_outer_closure_pending", tree_gone="unknown")
        self.assertEqual(owner.record['state'], 'active')
        with self.assertRaisesRegex(StorageError, "interrupted"):
            self.lease('blocked')
        with patch('managed_outputs.canonical_catalog', return_value=self.catalog):
            build.finish(exit_code=0, tree_gone=True)
        row = json.loads((self.catalog / (owner.key + '.json')).read_text())
        self.assertEqual(row['state'], 'retained')
        self.assertEqual(row['closure'], 'original_outer_supervisor')
        self.assertEqual(row['physical_completion'], 'not_inferred_from_output_lifecycle')
        later = self.lease('admitted')
        later.finish(status='completed', tree_gone=True)

    def test_last_receipt_growth_is_checked_again_by_outer_supervisor(self):
        build = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(build.guard.release)
        with patch.dict(os.environ, {"RZ_MANAGED_BUILD_ROOT": str(build.root),
                                     "RZ_MANAGED_BUILD_TOKEN": build.token}):
            owner = self.lease(reserve=4096, outer_supervised=True)
        owner.finish(status='checks_completed', tree_gone='unknown')
        (owner.output / 'last-receipt.json').write_bytes(b'x' * 5000)
        with patch('managed_outputs.canonical_catalog', return_value=self.catalog):
            with self.assertRaisesRegex(StorageError, 'exceeds reservation'):
                build.finish(exit_code=0, tree_gone=True)
        self.assertTrue(build.lock.exists())
        self.assertEqual(json.loads((self.catalog / (owner.key + '.json')).read_text())['state'], 'active')

    def test_output_overflow_preserves_result_and_records_failure(self):
        owner = self.lease(reserve=4096, total_limit=8192)
        evidence = owner.output / "raw.log"
        evidence.write_bytes(b"x" * 5000)
        with self.assertRaises(OutputBudgetError):
            owner.finish(status="failed", tree_gone=True)
        record = json.loads((self.catalog / (owner.key + ".json")).read_text())
        self.assertEqual(record["state"], "retained")
        self.assertIn("budget", record["budget_error"])
        self.assertEqual(evidence.stat().st_size, 5000)

    def test_changed_owner_is_never_overwritten_by_finish(self):
        owner = self.lease()
        changed = {**owner.record, "token": "a" * 32}
        for path in (owner.output / ".rovezero-output.json", self.catalog / (owner.key + ".json")):
            write_record(path, changed)
        with self.assertRaisesRegex(StorageError, "owner changed"):
            owner.finish(status="completed", tree_gone=True)
        self.assertEqual(json.loads((owner.output / ".rovezero-output.json").read_text()), changed)

    def test_deferred_file_preserves_consumer_fresh_path_and_future_identity(self):
        owner = self.lease("report.json", deferred=True, reserve=4096)
        self.assertFalse(owner.output.exists())
        owner.output.write_bytes(b"actual-result")
        owner.check_budget()
        owner.finish(status="completed", tree_gone=True)
        self.assertEqual(sample_output_budget(self.catalog), len(b"actual-result"))
        other = self.lease("second")
        other.finish(status="completed", tree_gone=True)
        self.assertEqual(owner.output.read_bytes(), b"actual-result")

    def test_deferred_no_child_output_is_retained_as_absent_not_adopted_later(self):
        owner = self.lease("not-created.json", deferred=True)
        owner.finish(status="not_started", tree_gone=True)
        self.assertFalse(owner.output.exists())
        later = self.lease("later")
        later.finish(status="completed", tree_gone=True)
        owner.output.write_bytes(b"unrelated-later-file")
        with self.assertRaisesRegex(StorageError, "custody"):
            self.lease("next")

    def test_historical_and_overlapping_outputs_are_rejected(self):
        historical = self.base / "historical"
        historical.mkdir()
        (historical / "only-result").write_bytes(b"keep")
        with self.assertRaisesRegex(StorageError, "fresh"):
            self.lease("historical")
        owner = self.lease()
        owner.finish(status="completed", tree_gone=True)
        with self.assertRaisesRegex(StorageError, "overlapping"):
            self.lease("run/nested")
        self.assertEqual((historical / "only-result").read_bytes(), b"keep")

    def test_replaced_catalog_gate_is_rejected(self):
        owner = self.lease()
        original = self.catalog / "catalog.lock"
        try:
            original.rename(self.catalog / "retired-gate")
        except PermissionError:
            self.skipTest("Windows denies replacement of the held file")
        original.write_bytes(b"0")
        with self.assertRaisesRegex(StorageError, "gate replaced"):
            owner.finish(status="completed", tree_gone=True)

    @unittest.skipUnless(sys.platform == 'linux', 'directory replacement while gate inode remains open')
    def test_catalog_replacement_with_same_lock_inode_is_rejected(self):
        owner = self.lease()
        saved = self.base / 'original-catalog'
        self.catalog.rename(saved)
        self.catalog.mkdir()
        (saved / 'catalog.lock').rename(self.catalog / 'catalog.lock')
        with self.assertRaisesRegex(StorageError, 'gate replaced'):
            owner.finish(status='completed', tree_gone=True)
        self.assertTrue((saved / (owner.key + '.json')).exists())

    def test_purpose_slots_count_together_and_do_not_adopt_manual_sibling(self):
        sibling = self.base / "startup-caller-diagnostic"
        sibling.mkdir()
        (sibling / "unknown").write_bytes(b"keep")
        slots = []
        for purpose in (None, "startup-caller", "another-purpose"):
            owner = ManagedBuild(self.build, self.source, purpose=purpose, slots_limit=2).acquire()
            self.addCleanup(owner.guard.release)
            slots.append(owner.slot)
            owner.finish(exit_code=0, tree_gone=True)
        self.assertFalse(slots[0].exists())
        self.assertEqual((sibling / "unknown").read_bytes(), b"keep")
        report = inventory([self.base])
        self.assertFalse(report["deletion_performed"])
        self.assertEqual(report["whole_C_drive_or_VHD_physical_usage"], "unknown")

    def test_unknown_v1_and_same_boot_lease_preserved(self):
        owner = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(owner.guard.release)
        (owner.scratch / "in-use").write_bytes(b"keep")
        owner.guard.release()
        for legacy in (False, True):
            if legacy:
                for path in (owner.lock, owner.slot / "record.json"):
                    record = json.loads(path.read_text())
                    record.pop("lifetime")
                    record.pop("root_identity")
                    write_record(path, record)
            original = owner.lock.read_bytes()
            with self.assertRaises(StorageError):
                recover_build(self.build, pin(owner.lock))
            self.assertEqual(owner.lock.read_bytes(), original)
            self.assertEqual((owner.scratch / "in-use").read_bytes(), b"keep")

    def test_mocked_reboot_recovery_exact_owner_and_root_only(self):
        owner = ManagedBuild(self.build, self.source).acquire()
        self.addCleanup(owner.guard.release)
        (owner.scratch / "input").write_bytes(b"regenerable")
        (owner.target / "cache").write_bytes(b"regenerable")
        owner.guard.release()
        original = pin(owner.lock)
        current = {**owner.lifetime, "boot": "different-test-boot"}
        with patch("managed_storage.lifetime", return_value=current):
            with self.assertRaisesRegex(StorageError, "digest changed"):
                recover_build(self.build, "0" * 64)
            result = recover_build(self.build, original)
        self.assertTrue(result["recovered"])
        self.assertFalse(owner.lock.exists())
        self.assertFalse(owner.scratch.exists())
        self.assertFalse(owner.target.exists())
        reuse = ManagedBuild(self.build, self.source).acquire()
        reuse.finish(exit_code=0, tree_gone=True)

    def test_real_owner_crash_after_durable_tree_closure_is_recoverable(self):
        scripts = str(Path(__file__).resolve().parents[1])
        program = f"""import sys,os
sys.path.insert(0, {scripts!r})
from pathlib import Path
from unittest.mock import patch
from managed_storage import ManagedBuild, StorageError
owner=ManagedBuild(Path({str(self.build)!r}),Path({str(self.source)!r})).acquire()
(owner.scratch/'input').write_bytes(b'owned')
try:
    with patch('managed_storage.remove_tree',side_effect=StorageError('synthetic crash after closure')):
        owner.finish(exit_code=0,tree_gone=True)
except StorageError:
    os._exit(17)
"""
        child = subprocess.run([sys.executable, "-B", "-c", program], timeout=10, capture_output=True)
        self.assertEqual(child.returncode, 17, child.stderr)
        digest = pin(self.build / "owner.json")
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('second pathname byte read forbidden')):
            result = recover_build(self.build, digest)
        self.assertEqual(result["reason"], "original_terminal_closure_and_owner_dead")
        self.assertFalse((self.build / "owner.json").exists())

    def test_shared_runtime_reuse_corruption_and_quota_keep_originals(self):
        source = self.base / "source-runtime.so"
        source.write_bytes(b"synthetic runtime bytes")
        store = self.base / "shared-runtime"
        first, digest = publish_runtime(source, store, limit=10000)
        original_identity = (first.stat().st_dev, first.stat().st_ino)
        second, same_digest = publish_runtime(source, store, limit=10000)
        self.assertEqual((digest, first), (same_digest, second))
        self.assertEqual(original_identity, (second.stat().st_dev, second.stat().st_ino))
        corrupt = b"c" * source.stat().st_size
        first.write_bytes(corrupt)
        with self.assertRaisesRegex(StorageError, "digest mismatch"):
            publish_runtime(source, store, limit=10000)
        self.assertEqual(first.read_bytes(), corrupt)
        first.write_bytes(corrupt + b'oversized')
        real_open = Path.open
        def forbid_target_read(path, *args, **kwargs):
            if path == first:
                raise AssertionError('oversized target must not be opened')
            return real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', forbid_target_read):
            with self.assertRaisesRegex(StorageError, 'size mismatch'):
                publish_runtime(source, store, limit=10000)
        source.write_bytes(b"new runtime")
        with self.assertRaisesRegex(StorageError, "budget exhausted"):
            publish_runtime(source, store, limit=1)
        self.assertEqual(source.read_bytes(), b"new runtime")


if __name__ == "__main__":
    unittest.main()
