import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from managed_storage import (ManagedBuild, StorageError, _record_stamp, checked,
                             read_record, read_record_snapshot, remove_tree, tree_size)
from managed_process import run_group
import managed_outputs as outputs


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rovezero-storage-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.root = self.base / "owned-build"
        # ManagedBuild.finish/recover_build call the real output finalizer.
        # Keep its default lookup in this fixture instead of the user's live
        # catalog; explicit catalog arguments remain unaffected.
        self.output_catalog = self.base / "outputs"
        self.output_catalog.mkdir()
        catalog_patch = patch("managed_outputs.canonical_catalog", return_value=self.output_catalog)
        self.default_catalog = catalog_patch.start()
        self.addCleanup(catalog_patch.stop)
        self.model = self.source / "original-model.bin"
        self.model.write_bytes(b"retained-original")

    def acquire(self, source=None, **limits):
        return ManagedBuild(self.root, source or self.source, **limits).acquire()

    def sampled_identity_fixture(self, expected):
        target = self.base / "retained-output"
        target.mkdir(exist_ok=True)
        payload = target / "evidence.txt"
        payload.write_bytes(b"original retained evidence")
        record = self.output_catalog / (("a" * 64) + ".json")
        row = {"schema": outputs.SCHEMA, "key": "a" * 64,
               "output": str(target), "identity": expected, "reserved_bytes": 4096}
        raw = json.dumps(row).encode()
        record.write_bytes(raw)
        return target, payload, record, raw

    def test_sampled_identity_failure_keeps_original_error_and_one_identity_call(self):
        expected, observed = [67, (1 << 100) + 11], [193, (1 << 100) + 11]
        target, payload, record, raw = self.sampled_identity_fixture(expected)
        with patch.object(outputs, "identity", return_value=observed) as identity_call:
            with patch.object(outputs, "tree_size") as size_call:
                with self.assertRaisesRegex(StorageError, "^live output identity changed$") as caught:
                    outputs.sample_output_budget(self.output_catalog)
        self.assertIs(type(caught.exception), StorageError)
        self.assertEqual(caught.exception.args, ("live output identity changed",))
        identity_call.assert_called_once_with(checked(target))
        size_call.assert_not_called()  # The original failure still precedes size sampling.
        observation = caught.exception.managed_output_identity_observation
        self.assertEqual(observation["record_key"], "a" * 64)
        self.assertEqual(observation["normalized_target"], str(checked(target)))
        self.assertEqual(observation["expected_identity"], expected)
        self.assertEqual(observation["observed_identity"], observed)
        self.assertIs(observation["read_only_observation"], True)
        for key in ("identity_equivalence_attested", "storage_lifetime_authority_issued",
                    "recovery_authorized", "deletion_authorized", "physical_completion_attested",
                    "performance_evidence"):
            self.assertIs(observation[key], False)
        self.assertEqual(record.read_bytes(), raw)
        self.assertEqual(payload.read_bytes(), b"original retained evidence")
        self.assertEqual(expected, [67, (1 << 100) + 11])
        self.assertEqual(observed, [193, (1 << 100) + 11])

    def test_sampled_identity_success_and_none_keep_original_identity_call_count(self):
        for expected in ([67, 11], None):
            with self.subTest(expected=expected):
                target, _payload, record, raw = self.sampled_identity_fixture(expected)
                with patch.object(outputs, "identity", return_value=expected) as identity_call:
                    with patch.object(outputs, "tree_size", return_value=23) as size_call:
                        self.assertEqual(outputs.sample_output_budget(self.output_catalog), 23)
                if expected is None:
                    identity_call.assert_not_called()
                else:
                    identity_call.assert_called_once_with(checked(target))
                size_call.assert_called_once_with(checked(target), allow_disappearing=True)
                self.assertEqual(record.read_bytes(), raw)

    def test_run_managed_preserves_budget_failure_after_original_supervisor_stringifies_it(self):
        import run_managed
        target, payload, record, raw = self.sampled_identity_fixture([67, 11])
        original_errors, finished = [], []

        def original_supervisor(_command, _source, _environment, _timeout, check_budget):
            try:
                check_budget()
            except StorageError as error:
                original_errors.append(error)
                # Existing run_group converts the original exception to its
                # message after attempting the original process-tree cleanup.
                return 125, False, str(error)
            self.fail("original sample guard did not reject the changed identity")

        def finish(**arguments):
            finished.append(arguments)
            return {"failure": arguments["supervisor_error"], "tree_gone": arguments["tree_gone"]}

        manager = SimpleNamespace(acquire=lambda: None, environment=lambda: {}, finish=finish,
                                  root=self.root, source=self.source, total_limit=12 << 30)
        error_stream = io.StringIO()
        with patch.object(run_managed, "ManagedBuild", return_value=manager):
            with patch.object(run_managed, "canonical_root", return_value=self.root):
                with patch.object(run_managed.shutil, "which", return_value=sys.executable):
                    with patch.object(run_managed, "run_group", side_effect=original_supervisor):
                        with patch.object(outputs, "identity", return_value=[193, 11]) as identity_call:
                            with patch.object(sys, "argv", ["run_managed.py", "--", sys.executable]):
                                with patch.object(sys, "stderr", error_stream):
                                    self.assertEqual(run_managed.main(), 125)
        identity_call.assert_called_once_with(checked(target))
        self.assertEqual(len(original_errors), 1)
        self.assertEqual(finished, [{"exit_code": 125, "tree_gone": False,
                                     "supervisor_error": "live output identity changed"}])
        lines = error_stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("managed_storage="))
        self.assertEqual(json.loads(lines[1]), original_errors[0].managed_output_identity_observation)
        self.assertEqual(record.read_bytes(), raw)
        self.assertEqual(payload.read_bytes(), b"original retained evidence")

    def test_storage_observation_write_failure_does_not_replace_original_error(self):
        import run_managed
        error = StorageError("live output identity changed")
        outputs._attach_sampled_identity_observation(error, "a" * 64, self.base, [67, 11], [193, 11])
        with patch.object(sys, "stderr", SimpleNamespace(write=lambda _value: (_ for _ in ()).throw(OSError("closed fixture pipe")))):
            run_managed._emit_storage_failure_observation(error)
        self.assertEqual(error.args, ("live output identity changed",))

    def test_finish_real_finalizer_uses_only_the_fixture_default_catalog(self):
        owner = self.acquire()
        (owner.scratch / "owned-remnant").write_bytes(b"fixture temporary")
        result = owner.finish(exit_code=0, tree_gone=True)
        self.default_catalog.assert_called_once_with()
        self.assertEqual({path.name for path in self.output_catalog.iterdir()}, {"catalog.lock"})
        self.assertTrue(result["tree_cleanup_verified"])
        self.assertTrue(result["temporary_removed"])
        self.assertFalse(owner.lock.exists())
        self.assertFalse(owner.scratch.exists())
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    def test_snapshot_preserves_raw_hash_and_original_fd_stamp(self):
        record = self.source / "owner.json"
        raw = b'{"schema":"rovezero.managed-build.v1","owner":"original"}\n'
        record.write_bytes(raw)
        with record.open("rb") as stream:
            original = _record_stamp(os.fstat(stream.fileno()))
        snapshot = read_record_snapshot(record)
        self.assertEqual(snapshot, (json.loads(raw), hashlib.sha256(raw).hexdigest(), original))
        self.assertEqual(read_record_snapshot(record), snapshot)

    def test_windows_cross_source_creation_time_does_not_ignore_file_identity(self):
        # The actual CPython 3.12+ path stat uses creation time for ctime,
        # whereas fstat uses ChangeTime. Model both without depending on OS.
        fields = dict(st_dev=7, st_ino=(1 << 100) + 11, st_mode=stat.S_IFREG | 0o666,
                      st_nlink=1, st_size=40, st_mtime_ns=1000,
                      st_ctime_ns=2000, st_birthtime_ns=900)
        opened = SimpleNamespace(**fields)
        named = SimpleNamespace(**{**fields, "st_ctime_ns": fields["st_birthtime_ns"]})
        self.assertNotEqual(_record_stamp(opened), _record_stamp(named))
        self.assertEqual(_record_stamp(opened, creation_time=True),
                         _record_stamp(named, creation_time=True))
        for field in ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                      "st_mtime_ns", "st_birthtime_ns"):
            with self.subTest(field=field):
                changed = SimpleNamespace(**{**vars(named), field: getattr(named, field) + 1})
                self.assertNotEqual(_record_stamp(opened, creation_time=True),
                                    _record_stamp(changed, creation_time=True))

    def test_snapshot_rejects_fd_change_time_mutation(self):
        record = self.source / "owner.json"
        record.write_bytes(b'{"schema":"rovezero.managed-build.v1"}\n')
        fstat = os.fstat
        calls = 0

        def changed_after_read(fd):
            nonlocal calls
            info = fstat(fd)
            calls += 1
            if calls == 1:
                return info
            fields = {name: getattr(info, name) for name in (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_mtime_ns", "st_ctime_ns")}
            fields["st_ctime_ns"] += 1
            if hasattr(info, "st_birthtime_ns"):
                fields["st_birthtime_ns"] = info.st_birthtime_ns
            return SimpleNamespace(**fields)

        with patch("managed_storage.os.fstat", side_effect=changed_after_read):
            with self.assertRaisesRegex(StorageError, "identity changed while reading"):
                read_record_snapshot(record)
        self.assertEqual(record.read_bytes(), b'{"schema":"rovezero.managed-build.v1"}\n')

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

    @unittest.skipUnless(os.name == "nt", "Windows CreateProcessW start boundary")
    def test_definite_windows_non_start_releases_only_its_owned_lease_and_scratch(self):
        owner = self.acquire()
        invalid_program = self.source / "invalid-managed-program.exe"
        invalid_program.write_bytes(b"not a Windows executable\n")
        (owner.target / "retained-cache").write_bytes(b"cache")
        (owner.scratch / "unstarted-remnant").write_bytes(b"owned temporary")
        with patch.dict(os.environ, {"CARGO_TARGET_DIR": str(owner.target)}):
            environment = owner.environment()
        code, gone, failure = run_group(
            [str(invalid_program)], self.source, environment, 5, lambda: None)
        self.assertEqual((code, gone), (125, True))
        self.assertEqual(failure["kind"], "windows_process_not_started")
        self.assertIn(failure["winerror"], (193, 216))
        self.assertIn(str(failure["winerror"]), failure["message"])
        receipt = owner.finish(exit_code=code, tree_gone=gone, supervisor_error=failure)
        self.assertTrue(receipt["temporary_removed"])
        self.assertFalse(owner.scratch.exists())
        self.assertFalse(owner.lock.exists())
        self.assertEqual(receipt["supervisor_error"], failure)
        self.assertEqual((owner.target / "retained-cache").read_bytes(), b"cache")
        self.assertEqual(self.model.read_bytes(), b"retained-original")

    def test_post_run_budget_reset_preserves_source(self):
        owner = self.acquire(slot_limit=16_384, total_limit=49_152)
        (owner.target / "too-large").write_bytes(b"x" * 20_000)
        receipt = owner.finish(exit_code=0, tree_gone=True)
        self.assertTrue(receipt["post_run_budget_reset"])
        self.assertFalse(owner.target.exists())
        self.assertLessEqual(tree_size(owner.slot), owner.slot_limit)
        self.assertTrue(self.model.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX directory deletion permissions")
    def test_readonly_owned_directories_are_removed_after_verified_finish(self):
        owner = self.acquire()
        nested = owner.scratch / "frozen-parent"
        nested.mkdir()
        retained = nested / "private-input"
        retained.write_bytes(b"owned regenerable input")
        retained.chmod(0o400)
        nested.chmod(0o500)
        owner.scratch.chmod(0o500)
        try:
            receipt = owner.finish(exit_code=0, tree_gone=True)
            self.assertTrue(receipt["temporary_removed"])
            self.assertFalse(owner.scratch.exists())
            self.assertFalse(owner.lock.exists())
            self.assertEqual(self.model.read_bytes(), b"retained-original")
        finally:
            # Only test-fixture cleanup if the regression is still present.
            for directory in (owner.scratch, nested):
                if directory.exists():
                    directory.chmod(0o700)

    @unittest.skipUnless(os.name == "posix", "POSIX directory deletion permissions")
    def test_readonly_parent_retry_repairs_directory_without_chmod_shared_file(self):
        owner = self.acquire()
        nested = owner.scratch / "frozen-parent"
        nested.mkdir()
        target = nested / "private-input"
        target.write_bytes(b"owned")
        target.chmod(0o400)
        nested.chmod(0o500)
        original = PermissionError("unlink requires writable parent")
        import managed_storage
        real_rmtree = managed_storage.shutil.rmtree
        retries = []

        def unlink_after_repair(name):
            retries.append(name)
            self.assertTrue(stat.S_IMODE(nested.stat().st_mode) & stat.S_IWUSR)
            self.assertTrue(stat.S_IMODE(nested.stat().st_mode) & stat.S_IXUSR)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
            os.unlink(name)

        def permission_failure(path, *, onerror):
            # Exercise the same failure even when CI runs as root and bypasses
            # normal permission checks; the retry checks the real directory mode.
            onerror(unlink_after_repair, str(target), (PermissionError, original, None))
            real_rmtree(path, onerror=onerror)

        try:
            with patch("managed_storage.shutil.rmtree", side_effect=permission_failure):
                remove_tree(owner.scratch, owner.slot)
            self.assertEqual(retries, [str(target)])
            self.assertFalse(owner.scratch.exists())
            self.assertEqual(self.model.read_bytes(), b"retained-original")
        finally:
            if nested.exists():
                nested.chmod(0o700)

    def test_cleanup_retry_preserves_original_permission_failure(self):
        owner = self.acquire()
        target = owner.scratch / "private-input"
        target.write_bytes(b"owned")
        original = PermissionError("initial unlink failure")
        retry = OSError("retry failed")

        def retry_failure(_name):
            raise retry

        def permission_failure(_path, *, onerror):
            onerror(retry_failure, str(target), (PermissionError, original, None))

        with patch("managed_storage.shutil.rmtree", side_effect=permission_failure):
            with self.assertRaises(OSError) as failure:
                remove_tree(owner.scratch, owner.slot)
        self.assertIs(failure.exception, retry)
        self.assertIs(failure.exception.__cause__, original)
        self.assertTrue(target.exists())
        self.assertTrue(owner.lock.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX no-follow permission repair")
    def test_cleanup_changed_parent_link_is_rejected_before_permission_repair(self):
        owner = self.acquire()
        nested = owner.scratch / "frozen-parent"
        nested.mkdir()
        target = nested / self.model.name
        target.write_bytes(b"owned regenerable copy")
        nested.chmod(0o500)
        saved = owner.scratch / "saved-parent"
        source_mode = stat.S_IMODE(self.source.stat().st_mode)
        original = PermissionError("initial unlink failure")

        def changed_link(_path, *, onerror):
            # The full tree was already preflighted. A changed link must never
            # give permission repair authority over the external source tree.
            nested.rename(saved)
            nested.symlink_to(self.source, target_is_directory=True)
            onerror(os.unlink, str(target), (PermissionError, original, None))

        try:
            with patch("managed_storage.shutil.rmtree", side_effect=changed_link):
                with self.assertRaises(StorageError) as failure:
                    remove_tree(owner.scratch, owner.slot)
            self.assertIs(failure.exception.__cause__, original)
            self.assertEqual(stat.S_IMODE(self.source.stat().st_mode), source_mode)
            self.assertEqual(self.model.read_bytes(), b"retained-original")
            self.assertTrue((saved / self.model.name).exists())
            self.assertTrue(owner.lock.exists())
        finally:
            if nested.is_symlink():
                nested.unlink()  # Test fixture cleanup only; do not follow it.
            if saved.exists():
                saved.rename(nested)
            if nested.exists():
                nested.chmod(0o700)

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
    @unittest.skipUnless(os.name == "nt", "Windows CreateProcessW start boundary")
    def test_invalid_windows_executable_is_definitely_not_started(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-invalid-process-") as directory:
            program = Path(directory) / "invalid.exe"
            program.write_bytes(b"not a Windows executable\n")
            admitted = []
            code, gone, failure = run_group(
                [str(program)], directory, os.environ.copy(), 5,
                lambda: admitted.append(True))
            self.assertEqual(code, 125)
            self.assertIs(gone, True)
            self.assertEqual(admitted, [])
            self.assertIsInstance(failure, dict)
            self.assertEqual(failure["kind"], "windows_process_not_started")
            self.assertEqual(failure["stage"], "CreateProcessW")
            self.assertIs(failure["process_created"], False)
            self.assertIn(failure["winerror"], (193, 216))
            self.assertIn(str(failure["winerror"]), failure["message"])
            self.assertEqual(json.loads(json.dumps(failure)), failure)

    @unittest.skipUnless(os.name == "nt", "Windows Job-owned stdlib child")
    def test_windows_stdlib_child_runs_and_keeps_success_result_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="rovezero-stdlib-process-") as directory:
            marker = Path(directory) / "completed"
            program = f"from pathlib import Path; Path({str(marker)!r}).write_text('ran',encoding='utf-8')"
            code, gone, failure = run_group(
                [sys.executable, "-I", "-S", "-c", program],
                directory, os.environ.copy(), 5, lambda: None)
            self.assertEqual(code, 0)
            self.assertIs(gone, True)
            self.assertIsNone(failure)
            self.assertEqual(marker.read_text(encoding="utf-8"), "ran")

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
