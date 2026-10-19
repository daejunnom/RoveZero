"""Real temporary-file/owner boundaries; no NN, GPU, Cargo or real data cleanup."""
import hashlib
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

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
        first = self.lease(reserve=12000, total_limit=18000)
        evidence = first.output / "unique.pgn"
        evidence.write_bytes(b"x" * 7000)
        first.finish(status="failed_evidence_preserved", tree_gone=True)
        with patch("managed_outputs.time.time_ns", return_value=10**30):
            with self.assertRaisesRegex(StorageError, "budget"):
                self.lease("next", reserve=12000, total_limit=18000)
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

    def test_terminal_metadata_growth_is_not_a_successful_finish(self):
        owner = self.lease(reserve=4096, total_limit=8192)
        with self.assertRaises(OutputBudgetError):
            owner.finish(status='diagnostic:' + 'x' * 2000, tree_gone=True)
        row = json.loads((self.catalog / (owner.key + '.json')).read_text())
        self.assertEqual(row['status'], 'failed_output_budget')
        self.assertIn('budget', row['budget_error'])

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


class CreatedDirectoryBindingTests(unittest.TestCase):
    """Windows-safe in-memory OS/catalog faults; never touch a real output tree."""
    def setUp(self):
        self.mocks = contextlib.ExitStack()
        self.addCleanup(self.mocks.close)
        self.base = Path(os.path.abspath("mock-created-output"))
        self.parent = self.base / "private"
        self.target = self.parent / "attempt"
        self.catalog = self.base / "catalog"
        self.life = {"host": "same-host", "boot": "original-boot",
                     "platform": "linux", "process_birth": "original-birth"}
        self.parent_info = self.info(20)
        self.named_parent, self.named_child = self.parent_info, None
        self.fd_info, self.events, self.size = {}, [], 0
        self.version, self.writes, self.mkdir_calls = 0, 0, 0
        self.write_error = self.sync_error = self.readback_error = None
        self.gate = MagicMock(spec=CatalogGuard)
        self.gate.__enter__.return_value = self.gate
        self.gate.assert_held.side_effect = self.assert_held
        self.held = True
        self.mocks.enter_context(patch("managed_outputs.CatalogGuard", return_value=self.gate))
        self.mocks.enter_context(patch("managed_outputs.checked", side_effect=lambda path: Path(path)))
        self.mocks.enter_context(patch("managed_outputs.sys", SimpleNamespace(platform="linux", exc_info=sys.exc_info)))
        self.fake_os = SimpleNamespace(path=os.path, O_RDONLY=os.O_RDONLY,
            O_DIRECTORY=0x10000, O_NOFOLLOW=0x20000, getpid=lambda: 123, geteuid=lambda: 1000,
            open=Mock(side_effect=self.open_fd), fstat=lambda fd: self.fd_info[fd],
            stat=self.stat_name, mkdir=self.mkdir, fsync=self.fsync,
            close=Mock(side_effect=lambda fd: self.events.append(("close", fd))),
            unlink=Mock(side_effect=AssertionError("no deletion")))
        self.mocks.enter_context(patch("managed_outputs.os", self.fake_os))
        self.mocks.enter_context(patch("managed_outputs.lifetime", side_effect=lambda: dict(self.life)))
        self.mocks.enter_context(patch.object(Path, "lstat", lambda path: self.lstat(path)))
        self.mocks.enter_context(patch.object(Path, "exists", lambda path: path == self.catalog or (path == self.target and self.named_child is not None)))
        self.mocks.enter_context(patch("managed_outputs.read_record_snapshot", side_effect=self.snapshot))
        self.mocks.enter_context(patch("managed_outputs._read", side_effect=lambda path: copy.deepcopy(self.disk)))
        self.mocks.enter_context(patch("managed_outputs.write_record", side_effect=self.write))
        self.mocks.enter_context(patch("managed_outputs.tree_size", side_effect=lambda path, **kwargs: self.size if path == self.target else 0))
        self.owner = OutputLease(self.target, self.base / "source", catalog=self.catalog,
                                 deferred=True, reserve=4096, total_limit=8192)
        self.owner.record = {"schema": "rovezero.managed-output.v1", "key": self.owner.key,
            "output": str(self.target), "source": str(self.owner.source), "identity": None,
            "token": "original-token", "deferred": True, "outer_supervisor": None,
            "pid": 123, "lifetime": dict(self.life), "state": "active",
            "reserved_bytes": 4096, "catalog_limit_bytes": 8192}
        self.disk = copy.deepcopy(self.owner.record)
        self.owner._rows = Mock(side_effect=lambda: [(self.row_path, copy.deepcopy(self.disk), self.target)])
        self.addCleanup(self.owner._close_created_directory)

    @property
    def row_path(self):
        return self.catalog / (self.owner.key + ".json")

    @staticmethod
    def info(inode, mode=None):
        return SimpleNamespace(st_dev=7, st_ino=inode, st_mode=mode or (stat.S_IFDIR | 0o700),
                               st_uid=1000, st_gid=1000, st_nlink=2)

    def assert_held(self):
        if not self.held:
            raise StorageError("catalog gate not held")

    def open_fd(self, path, flags, *, dir_fd=None):
        self.assert_held()
        self.assertTrue(flags & self.fake_os.O_DIRECTORY)
        self.assertTrue(flags & self.fake_os.O_NOFOLLOW)
        if dir_fd is None:
            self.assertEqual(path, self.parent)
            self.fd_info[101] = self.parent_info
            return 101
        self.assertEqual((path, dir_fd), ("attempt", 101))
        self.fd_info[102] = self.named_child
        return 102

    def lstat(self, path):
        self.assertEqual(path, self.parent)
        return self.named_parent

    def stat_name(self, name, *, dir_fd, follow_symlinks):
        self.assertEqual((name, dir_fd, follow_symlinks), ("attempt", 101, False))
        if self.named_child is None:
            raise FileNotFoundError(name)
        return self.named_child

    def mkdir(self, name, mode, *, dir_fd):
        self.assertEqual((name, mode, dir_fd), ("attempt", 0o700, 101))
        if self.named_child is not None:
            raise FileExistsError(name)
        self.mkdir_calls += 1
        self.named_child = self.info(30)
        self.events.append(("mkdir", name))

    def fsync(self, fd):
        self.events.append(("fsync", fd))
        if self.sync_error is not None:
            raise self.sync_error

    def snapshot(self, path, *, schema):
        self.assertEqual((path, schema), (self.row_path, "rovezero.managed-output.v1"))
        if self.writes and self.readback_error is not None:
            raise self.readback_error
        raw = (json.dumps(self.disk, sort_keys=True) + "\n").encode()
        return copy.deepcopy(self.disk), hashlib.sha256(raw).hexdigest(), (self.version,)

    def write(self, path, value):
        self.assert_held()
        self.assertEqual(path, self.row_path)
        self.writes += 1
        self.disk, self.version = copy.deepcopy(value), self.version + 1
        self.events.append(("catalog_write", self.writes))
        if self.write_error is not None:
            raise self.write_error  # Simulate replace succeeding before directory fsync fails.

    def assert_sticky_failure(self, primary):
        self.assertIs(self.owner._directory_binding_error, primary)
        count = self.writes
        with self.assertRaises(StorageError) as blocked:
            self.owner.finish(status="completed", tree_gone=True)
        self.assertIs(blocked.exception.__cause__, primary)
        self.assertEqual(self.writes, count)
        self.assertEqual(self.disk["state"], "active")
        self.assertFalse(self.gate.release.called)

    def test_seal_precedes_payload_and_only_identity_changes(self):
        original = copy.deepcopy(self.disk)
        self.assertEqual(self.owner.create_deferred_directory(), self.target)
        self.assertEqual(self.owner._directory_binding_stage, "sealed")
        self.assertEqual(self.disk, dict(original, identity=[7, 30]))
        self.assertEqual(self.writes, 1)
        self.assertEqual(self.events[:3], [("mkdir", "attempt"), ("fsync", 102), ("fsync", 101)])
        self.assertEqual(self.events[3], ("catalog_write", 1))
        self.size = 100
        self.owner.check_budget()
        self.owner.finish(status="provider_unknown", tree_gone=False)
        self.assertEqual(self.disk["identity"], [7, 30])
        self.assertEqual(self.disk["state"], "active")
        self.assertEqual(self.disk["physical_completion"], "not_inferred_from_output_lifecycle")
        self.assertEqual(self.events[-2:], [("close", 102), ("close", 101)])
        self.gate.release.assert_called_once()

    def test_existing_directory_regular_and_symlink_are_never_adopted(self):
        for mode in (stat.S_IFDIR | 0o700, stat.S_IFREG | 0o600, stat.S_IFLNK | 0o777):
            with self.subTest(mode=mode):
                self.owner._directory_binding_error = None
                self.owner._created_directory = None
                self.named_child = self.info(99, mode)
                with self.assertRaisesRegex(StorageError, "exclusive fresh") as failure:
                    self.owner.create_deferred_directory()
                self.assertEqual(self.named_child.st_ino, 99)
                self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
                self.assert_sticky_failure(failure.exception)
                self.owner._close_created_directory()

    def test_changed_token_process_lifetime_or_gate_cannot_bind(self):
        for change in ("token", "lifetime", "gate"):
            with self.subTest(change=change):
                self.owner._directory_binding_error = None
                if change == "token":
                    self.disk["token"] = "replacement"
                elif change == "lifetime":
                    self.disk = copy.deepcopy(self.owner.record)
                    self.life["boot"] = "other-boot"
                else:
                    self.life["boot"] = "original-boot"
                    self.held = False
                with self.assertRaises(StorageError):
                    self.owner.create_deferred_directory()
                self.assertEqual((self.mkdir_calls, self.writes), (0, 0))

    def test_directory_and_parent_replacement_block_budget_and_finish(self):
        self.owner.create_deferred_directory()
        original = copy.deepcopy(self.disk)
        self.named_child = self.info(99)
        for action in (self.owner.check_budget, lambda: self.owner.finish(status="completed", tree_gone=True)):
            with self.assertRaisesRegex(StorageError, "directory changed"):
                action()
        self.assertEqual(self.disk, original)
        self.named_child = self.fd_info[102]
        self.named_parent = self.info(98)
        with self.assertRaisesRegex(StorageError, "parent changed"):
            self.owner.check_budget()
        self.assertEqual(self.disk, original)

    def test_parent_fstat_failure_keeps_fd_owned_and_close_preserves_primary(self):
        primary = OSError("parent fstat failed")
        with patch("managed_outputs.os.fstat", side_effect=primary):
            with self.assertRaises(OSError) as failure:
                self.owner.create_deferred_directory()
        self.assertIs(failure.exception, primary)
        self.assertEqual(self.owner._created_directory.parent_fd, 101)
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
        self.assert_sticky_failure(primary)
        self.owner._close_created_directory(primary)
        self.assertEqual(self.events, [("close", 101)])

    def test_parent_identity_setup_failure_keeps_fd_owned_for_close(self):
        primary = MemoryError("parent identity setup failed")
        with patch("managed_outputs._CreatedOutputDirectory.directory_identity_of", side_effect=primary):
            with self.assertRaises(MemoryError) as failure:
                self.owner.create_deferred_directory()
        self.assertIs(failure.exception, primary)
        self.assertEqual(self.owner._created_directory.parent_fd, 101)
        self.owner._close_created_directory(primary)
        self.assertEqual(self.events, [("close", 101)])
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))

    def test_cap_initialization_failure_precedes_any_fd_open(self):
        primary = MemoryError("cap initialization failed")
        with patch("managed_outputs._CreatedOutputDirectory", side_effect=primary):
            with self.assertRaises(MemoryError) as failure:
                self.owner.create_deferred_directory()
        self.assertIs(failure.exception, primary)
        self.fake_os.open.assert_not_called()
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
        self.assert_sticky_failure(primary)

    def test_prepublication_sync_failure_preserves_null_row_and_directory(self):
        self.sync_error = OSError("created directory sync failed")
        with self.assertRaises(OSError) as failure:
            self.owner.create_deferred_directory()
        self.assertIs(failure.exception, self.sync_error)
        self.assertEqual((self.mkdir_calls, self.writes), (1, 0))
        self.assertIsNone(self.disk["identity"])
        self.assertEqual(self.named_child.st_ino, 30)
        self.assert_sticky_failure(failure.exception)

    def test_replace_then_fsync_failure_is_not_rolled_back_or_finished(self):
        self.write_error = OSError("catalog directory sync failed")
        with self.assertRaises(OSError) as failure:
            self.owner.create_deferred_directory()
        self.assertIs(failure.exception, self.write_error)
        self.assertEqual(self.owner._directory_binding_stage, "catalog_write_pending")
        self.assertEqual(self.disk["identity"], [7, 30])
        self.assertIsNone(self.owner.record["identity"])
        self.assert_sticky_failure(failure.exception)

    def test_readback_and_postwrite_budget_errors_never_release_payload(self):
        self.readback_error = StorageError("readback unavailable")
        with self.assertRaises(StorageError) as failure:
            self.owner.create_deferred_directory()
        self.assert_sticky_failure(failure.exception)
        self.assertEqual(self.owner._directory_binding_stage, "catalog_written_readback_pending")

    def test_postwrite_budget_overflow_keeps_identity_but_blocks_finish(self):
        original_write = self.write
        def grow_after_write(path, row):
            original_write(path, row)
            self.size = 5000
        with patch("managed_outputs.write_record", side_effect=grow_after_write):
            with self.assertRaises(OutputBudgetError) as failure:
                self.owner.create_deferred_directory()
        self.assertEqual(self.disk["identity"], [7, 30])
        self.assert_sticky_failure(failure.exception)

    def test_actual_catalog_encoding_bound_prevents_write_after_create(self):
        size = len((json.dumps(dict(self.owner.record, padding=""), sort_keys=True) + "\n").encode())
        self.owner.record["padding"] = "x" * (8191 - size)
        self.disk = copy.deepcopy(self.owner.record)
        with self.assertRaisesRegex(StorageError, "record exceeds bound") as failure:
            self.owner.create_deferred_directory()
        self.assertEqual((self.mkdir_calls, self.writes), (1, 0))
        self.assertIsNone(self.disk["identity"])
        self.assert_sticky_failure(failure.exception)

    def test_legacy_deferred_regular_file_and_absent_path_remain_compatible(self):
        with patch("managed_outputs.identity", return_value=[7, 88]):
            self.named_child = self.info(88, stat.S_IFREG | 0o600)
            self.owner.finish(status="completed", tree_gone=True)
        self.assertEqual(self.disk["identity"], [7, 88])
        self.assertEqual(self.writes, 1)
        self.assertIsNone(self.owner._created_directory)
        self.assertEqual(self.events, [("catalog_write", 1)])

    def test_legacy_deferred_absent_row_rejects_unrelated_later_path(self):
        self.owner.finish(status="not_started", tree_gone=True)
        self.assertIsNone(self.disk["identity"])
        with patch.object(Path, "iterdir", return_value=iter([self.row_path])):
            self.assertEqual(len(OutputLease._rows(self.owner)), 1)
        self.named_child = self.info(99, stat.S_IFREG | 0o600)
        with patch.object(Path, "iterdir", return_value=iter([self.row_path])):
            with self.assertRaisesRegex(StorageError, "custody changed"):
                OutputLease._rows(self.owner)
        self.assertEqual(self.writes, 1)
        self.assertEqual(self.named_child.st_ino, 99)

    def test_group_or_other_writable_parent_is_rejected_before_mkdir(self):
        self.parent_info.st_mode = stat.S_IFDIR | 0o777
        with self.assertRaisesRegex(StorageError, "without group/other writers") as failure:
            self.owner.create_deferred_directory()
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))
        self.assert_sticky_failure(failure.exception)

    def test_descriptor_close_failure_preserves_primary_write_error(self):
        self.owner.create_deferred_directory()
        self.write_error = OSError("original finish write failed")
        with patch("managed_outputs.os.close", side_effect=OSError("secondary FD close failed")):
            with self.assertRaises(OSError) as failure:
                self.owner.finish(status="completed", tree_gone=True)
        self.assertIs(failure.exception, self.write_error)
        self.assertIn("close also failed", failure.exception.__notes__[0])
        self.gate.release.assert_called_once()

    def test_descriptor_close_failure_does_not_mask_recorded_budget_error(self):
        self.owner.create_deferred_directory()
        self.size = 5000
        with patch("managed_outputs.os.close", side_effect=OSError("secondary FD close failed")):
            with self.assertRaises(OutputBudgetError) as failure:
                self.owner.finish(status="completed", tree_gone=True)
        self.assertIn("close also failed", failure.exception.__notes__[0])
        self.assertEqual(self.disk["status"], "failed_output_budget")
        self.gate.release.assert_called_once()

    def test_normal_descriptor_close_failure_is_not_success(self):
        self.owner.create_deferred_directory()
        with patch("managed_outputs.os.close", side_effect=OSError("FD close failed")):
            with self.assertRaisesRegex(OSError, "FD close failed"):
                self.owner.finish(status="completed", tree_gone=True)
        self.gate.release.assert_called_once()

    def test_unsupported_host_does_not_create_or_publish(self):
        with patch("managed_outputs.sys.platform", "win32"):
            with self.assertRaisesRegex(StorageError, "original active Linux"):
                self.owner.create_deferred_directory()
        self.assertEqual((self.mkdir_calls, self.writes), (0, 0))

    def test_recovery_does_not_fill_historical_null_identity(self):
        self.named_child = self.info(30)
        rebooted = dict(self.life, boot="new-boot")
        with patch("managed_outputs.lifetime", return_value=rebooted):
            with self.assertRaisesRegex(StorageError, "termination not proven"):
                recover_output(self.catalog, self.owner.key, self.snapshot(self.row_path, schema="rovezero.managed-output.v1")[1])
        self.assertEqual(self.writes, 0)
        self.assertIsNone(self.disk["identity"])

    def test_reboot_recovery_uses_sealed_identity_and_same_boot_stays_rejected(self):
        self.owner.create_deferred_directory()
        digest = self.snapshot(self.row_path, schema="rovezero.managed-output.v1")[1]
        with patch("managed_outputs.identity", return_value=[7, 30]):
            with self.assertRaisesRegex(StorageError, "termination not proven"):
                recover_output(self.catalog, self.owner.key, digest)
            with patch("managed_outputs.lifetime", return_value=dict(self.life, boot="new-boot", host="other-host")):
                with self.assertRaisesRegex(StorageError, "termination not proven"):
                    recover_output(self.catalog, self.owner.key, digest)
            with patch("managed_outputs.lifetime", return_value=dict(self.life, boot="new-boot")):
                with self.assertRaisesRegex(StorageError, "digest changed"):
                    recover_output(self.catalog, self.owner.key, "0" * 64)
                with patch("managed_outputs.identity", return_value=[7, 99]):
                    with self.assertRaisesRegex(StorageError, "termination not proven"):
                        recover_output(self.catalog, self.owner.key, digest)
                result = recover_output(self.catalog, self.owner.key, digest)
        self.assertEqual(result["state"], "interrupted_retained")
        self.assertEqual(result["identity"], [7, 30])
        self.assertNotIn("physical_completion", result)

    def test_active_bound_row_still_blocks_other_root_admission(self):
        self.owner.create_deferred_directory()
        other = OutputLease(self.base / "different", self.owner.source, catalog=self.catalog)
        other._rows = Mock(return_value=[(self.row_path, copy.deepcopy(self.disk), self.target)])
        with patch.object(Path, "mkdir") as make:
            with self.assertRaisesRegex(StorageError, "interrupted output"):
                other.acquire()
        make.assert_called_once()  # Existing catalog check only; no new output.
        self.assertEqual(self.writes, 1)
        self.assertEqual(self.disk["state"], "active")


@unittest.skipUnless(sys.platform == "linux", "real Linux directory descriptors and private temporary filesystem")
class CreatedDirectoryLinuxFilesystemTests(unittest.TestCase):
    """Only this test's private source/catalog/output; no production assets."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rovezero-created-directory-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.base.chmod(0o700)
        self.source = self.base / "source"
        self.source.mkdir(mode=0o700)
        self.owners = []
        self.addCleanup(self.close_owned_descriptors)

    def close_owned_descriptors(self):
        first = None
        for owner in reversed(self.owners):
            try:
                owner._close_created_directory()
            except OSError as error:
                if first is None:
                    first = error
            finally:
                owner.guard.release()
        if first is not None:
            raise first

    def lease(self, label):
        parent, catalog = self.base / (label + "-parent"), self.base / (label + "-catalog")
        parent.mkdir(mode=0o700)
        parent.chmod(0o700)
        catalog.mkdir(mode=0o700)
        owner = OutputLease(parent / "attempt", self.source, catalog=catalog,
                            deferred=True, reserve=4096, total_limit=8192).acquire()
        self.owners.append(owner)
        return owner, parent, catalog / (owner.key + ".json")

    def test_real_create_seals_before_payload_and_finish_closes_original_fds(self):
        owner, _, row_path = self.lease("positive")
        self.assertIsNone(json.loads(row_path.read_text())["identity"])
        self.assertFalse(owner.output.exists())
        self.assertEqual(owner.create_deferred_directory(), owner.output)
        opened = os.fstat(owner._created_directory.fd)
        original_identity = [opened.st_dev, opened.st_ino]
        self.assertEqual(json.loads(row_path.read_text())["identity"], original_identity)
        self.assertEqual([owner.output.stat().st_dev, owner.output.stat().st_ino], original_identity)
        self.assertEqual(stat.S_IMODE(owner.output.stat().st_mode), 0o700)
        self.assertFalse((owner.output / ".rovezero-output.json").exists())
        payload = owner.output / "payload.txt"
        with payload.open("xb") as stream:
            stream.write(b"private finite fixture")
        owner.check_budget()
        result = owner.finish(status="temporary_fixture_written", tree_gone=True)
        self.assertEqual(result["state"], "retained")
        self.assertEqual(result["identity"], original_identity)
        self.assertEqual(result["physical_completion"], "not_inferred_from_output_lifecycle")
        self.assertEqual(payload.read_bytes(), b"private finite fixture")
        self.assertIsNone(owner._created_directory.fd)
        self.assertIsNone(owner._created_directory.parent_fd)
        self.assertIsNone(owner.guard.stream)

    def test_real_target_and_parent_rename_replacement_never_rebinds(self):
        for replace_parent in (False, True):
            with self.subTest(replace_parent=replace_parent):
                label = "parent" if replace_parent else "target"
                owner, parent, row_path = self.lease(label)
                owner.create_deferred_directory()
                (owner.output / "original.txt").write_bytes(b"keep original")
                original_identity, original_pin = list(owner.record["identity"]), pin(row_path)
                if replace_parent:
                    moved_parent = self.base / "moved-parent"
                    parent.rename(moved_parent)
                    parent.mkdir(mode=0o700)
                    original_path = moved_parent / "attempt"
                else:
                    original_path = parent / "moved-attempt"
                    owner.output.rename(original_path)
                owner.output.mkdir(mode=0o700)
                (owner.output / "replacement.txt").write_bytes(b"keep replacement")
                message = "parent changed" if replace_parent else "directory changed"
                with self.assertRaisesRegex(StorageError, message):
                    owner.check_budget()
                with self.assertRaisesRegex(StorageError, message):
                    owner.finish(status="must_not_finish", tree_gone=True)
                self.assertEqual(pin(row_path), original_pin)
                self.assertEqual(json.loads(row_path.read_text())["state"], "active")
                opened = os.fstat(owner._created_directory.fd)
                self.assertEqual([opened.st_dev, opened.st_ino], original_identity)
                self.assertEqual((original_path / "original.txt").read_bytes(), b"keep original")
                self.assertEqual((owner.output / "replacement.txt").read_bytes(), b"keep replacement")
                # Cleanup closes only these original FDs/gate; it never calls finish.

    def test_real_same_boot_recovery_rejects_bound_active_row(self):
        owner, _, row_path = self.lease("same-boot")
        owner.create_deferred_directory()
        (owner.output / "payload.txt").write_bytes(b"keep same boot")
        digest = pin(row_path)
        owner._close_created_directory()
        owner.guard.release()  # Releasing an OS gate is not a reboot/closure witness.
        with self.assertRaisesRegex(StorageError, "termination not proven"):
            recover_output(owner.catalog, owner.key, digest)
        self.assertEqual(pin(row_path), digest)
        self.assertEqual(json.loads(row_path.read_text())["state"], "active")
        self.assertEqual((owner.output / "payload.txt").read_bytes(), b"keep same boot")

    def test_real_existing_unsealed_directory_keeps_null_identity_and_payload(self):
        owner, _, row_path = self.lease("unsealed")
        owner.output.mkdir(mode=0o700)
        (owner.output / "unsealed.txt").write_bytes(b"keep unsealed fixture")
        digest = pin(row_path)
        with self.assertRaisesRegex(StorageError, "exclusive fresh"):
            owner.create_deferred_directory()
        self.assertIsNone(json.loads(row_path.read_text())["identity"])
        owner._close_created_directory()
        owner.guard.release()
        with self.assertRaisesRegex(StorageError, "termination not proven"):
            recover_output(owner.catalog, owner.key, digest)
        self.assertEqual(pin(row_path), digest)
        self.assertIsNone(json.loads(row_path.read_text())["identity"])
        self.assertEqual((owner.output / "unsealed.txt").read_bytes(), b"keep unsealed fixture")


if __name__ == "__main__":
    unittest.main()
