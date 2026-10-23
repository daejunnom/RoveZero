"""Private CPU fixtures, not actual reboot/source-history/physical evidence.

The existing source7 fixture supplies six preserved private corpora and actual
FD/hash/mount walks. Its historical-chain seam and this test's old boot string
are explicitly synthetic. No operational owner, catalog, model or GPU is used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_output_bound_recovery as recovery
import managed_output_unattributed_preservation as policy
import managed_outputs as outputs
from managed_storage import GiB, StorageError, read_record_snapshot, write_record
import test_managed_output_unattributed_preservation as prior


class BoundRecoveryPredicates(unittest.TestCase):
    def setUp(self):
        self.key, self.digest = "1" * 64, "2" * 64
        self.target = Path("/private/bound-fixture")
        self.now = {"host": "fixture-host", "platform": "linux", "boot": "fixture-new",
            "process_birth": "fixture-new-process"}
        self.row = {"schema": outputs.SCHEMA, "key": self.key, "output": str(self.target),
            "state": "active", "deferred": True, "identity": [7, 12], "pid": 123,
            "token": "original-fixture", "outer_supervisor": None, "reserved_bytes": 1 << 20,
            "catalog_limit_bytes": 18 * GiB, "physical_completion": "unknown",
            "lifetime": dict(self.now, boot="fixture-old", process_birth="fixture-old-process"),
            "error": "original fixture failure"}

    def eligible(self, *, row=None, now=None, digest=None):
        return recovery._eligible(self.row if row is None else row, self.key,
            self.digest if digest is None else digest, self.digest, self.target,
            self.now if now is None else now)

    def test_exact_bound_reboot_predicates_are_read_only(self):
        original = copy.deepcopy(self.row)
        self.assertIsNone(self.eligible())
        self.assertEqual(self.row, original)

    def test_same_boot_foreign_host_platform_or_missing_lifetime_cannot_prove_death(self):
        for changes in ({"boot": "fixture-old"}, {"host": "foreign"}, {"platform": "win32"}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                self.eligible(now=dict(self.now, **changes))
        with self.assertRaises(StorageError):
            self.eligible(row=dict(self.row, lifetime=None))

    def test_row_digest_supervisor_identity_and_physical_upgrades_reject(self):
        for changes in ({"identity": None}, {"identity": [True, 12]}, {"identity": [7, 0]},
                {"state": "retained"}, {"deferred": False}, {"outer_supervisor": {}},
                {"reserved_bytes": True}, {"catalog_limit_bytes": 18 * GiB + 1},
                {"tree_gone": True}, {"physical_completion": "confirmed"}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                self.eligible(row=dict(self.row, **changes))
        with self.assertRaisesRegex(StorageError, "digest changed"):
            self.eligible(digest="3" * 64)

    def test_typed_selection_exact_booleans_pins_and_finite_deadline_required(self):
        arguments = dict(unattributed_selection={}, research_archive_pin={}, root_exclusive_writer=True)
        for changes in ({"apply": 1}, {"root_exclusive_writer": 1}, {"maximum_seconds": True},
                {"maximum_seconds": 31}, {"maximum_seconds": float("nan")}, {}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                recovery.recover_bound_output_after_reboot(Path("/not-opened"), self.key,
                    self.digest, **dict(arguments, **changes))
        with patch.object(recovery.time, "monotonic", return_value=10):
            with self.assertRaisesRegex(StorageError, "deadline expired"):
                recovery._remaining(10)


@unittest.skipUnless(sys.platform == "linux", "private Linux source7/FD fixtures; Main executes")
class BoundRecoveryLinuxTests(unittest.TestCase):
    setUp = prior.UnattributedInventoryTests.setUp
    _archive = prior.UnattributedInventoryTests._archive
    selection = prior.UnattributedInventoryTests.selection
    _mock_history = prior.UnattributedInventoryTests._mock_history
    _history_patches = prior.UnattributedInventoryTests._history_patches

    def rawpin(self, path):
        raw = path.read_bytes()
        return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def fixture(self, *, marker=False):
        (self.catalog / "catalog.lock").touch(mode=0o600)
        self.target = self.root / "original-bound-output"
        self.target.mkdir(mode=0o700)
        (self.target / "empty-directory").mkdir(mode=0o700)
        (self.target / "failure.log").write_bytes(b"original failed run; physical completion unknown\n")
        now = recovery.lifetime()
        self.key = hashlib.sha256(os.path.normcase(str(self.target)).encode()).hexdigest()
        self.row = {"schema": outputs.SCHEMA, "key": self.key, "output": str(self.target),
            "identity": outputs.identity(self.target), "state": "active", "deferred": True,
            "source": str(self.source), "pid": 123, "token": "private-fixture-only",
            "outer_supervisor": None, "reserved_bytes": 1 << 20, "catalog_limit_bytes": 18 * GiB,
            "lifetime": dict(now, boot="injected-old-fixture-boot", process_birth="fixture-old-process"),
            "tree_gone": False, "physical_completion": "unknown", "status": "original_failure",
            "producer_error": "preserve original fixture error"}
        self.row_path = self.catalog / (self.key + ".json")
        write_record(self.row_path, self.row)
        self.original = self.row_path.read_bytes()
        if marker:
            (self.target / recovery.MARKER).write_bytes(self.original)
        self.original_pin = self.rawpin(self.row_path)
        self.research = self.root / "research-archives"
        self.research.mkdir(mode=0o700)
        self.archive_path = self.research / "original.zip"
        self.archive_bound()
        self.selection_value = self.selection()
        self.protected = {Path(item[name]["path"]): Path(item[name]["path"]).read_bytes()
            for item in self.items for name in ("catalog", "marker", "research_archive")}

    def archive_bound(self, *, extra=False):
        files = []
        directories = ["."] + sorted(path.relative_to(self.target).as_posix()
            for path in self.target.rglob("*") if path.is_dir())
        with zipfile.ZipFile(self.archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(recovery.ORIGINAL_MEMBER, self.row_path.read_bytes())
            for path in sorted(self.target.rglob("*")):
                if not path.is_file():
                    continue
                raw, relative = path.read_bytes(), path.relative_to(self.target).as_posix()
                member = "corpus/" + relative
                files.append({"relative": relative, "member": member, "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest()})
                archive.writestr(member, raw)
            marker = ({"state": "present", "member": "corpus/" + recovery.MARKER}
                if (self.target / recovery.MARKER).exists() else {"state": "absent"})
            document = {"schema": recovery.ARCHIVE_SCHEMA,
                "original_catalog_pin": self.rawpin(self.row_path),
                "original_catalog_member": recovery.ORIGINAL_MEMBER, "output": str(self.target),
                "marker": marker, "directories": directories, "files": files}
            archive.writestr("inventory.json", json.dumps(document, sort_keys=True))
            if extra:
                archive.writestr("unlisted.txt", b"not an original member")
        self.archive_pin = self.rawpin(self.archive_path)

    def recover(self, *, apply=False, key=None, digest=None):
        with self._history_patches():
            return recovery.recover_bound_output_after_reboot(self.catalog, key or self.key,
                digest or self.original_pin["sha256"], unattributed_selection=self.selection_value,
                research_archive_pin=self.archive_pin, root_exclusive_writer=True, apply=apply)

    def assert_originals(self):
        for path, raw in self.protected.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual((self.target / "failure.log").read_bytes(),
            b"original failed run; physical completion unknown\n")
        self.assertTrue((self.target / "empty-directory").is_dir())
        self.assertEqual(self.rawpin(self.archive_path), self.archive_pin)

    def test_dry_run_verifies_real_private_bytes_without_catalog_commit(self):
        self.fixture()
        with patch.object(recovery, "write_record", side_effect=AssertionError("dry run cannot commit")):
            result = self.recover()
        self.assertEqual(result["status"], "eligible_but_not_applied")
        self.assertFalse(result["applied"])
        self.assertTrue(result["archive_all_original_bytes_verified"])
        self.assertEqual(self.row_path.read_bytes(), self.original)
        self.assert_originals()

    def test_apply_absent_marker_preserves_failure_identity_reservation_and_physical_unknown(self):
        self.fixture()
        result = self.recover(apply=True)
        row = read_record_snapshot(self.row_path, schema=outputs.SCHEMA)[0]
        self.assertEqual(row["state"], "interrupted_retained")
        self.assertTrue(result["applied"])
        self.assertTrue(result["canonical_readback_verified"])
        self.assertEqual(result["marker"], {"state": "absent"})
        for name in self.row:
            if name != "state":
                self.assertEqual(row[name], self.row[name])
        self.assertFalse((self.target / recovery.MARKER).exists())
        self.assert_originals()

    def test_existing_matching_marker_is_preserved_as_original_not_rewritten(self):
        self.fixture(marker=True)
        result = self.recover(apply=True)
        self.assertTrue(result["applied"])
        self.assertEqual((self.target / recovery.MARKER).read_bytes(), self.original)
        self.assertNotEqual(self.row_path.read_bytes(), self.original)
        self.assert_originals()

    def test_selected_six_or_historical_keys_are_immutable(self):
        self.fixture()
        selected = self.items[0]["catalog"]
        with self.assertRaisesRegex(StorageError, "selected/history original rows are immutable"):
            self.recover(apply=True, key=Path(selected["path"]).stem, digest=selected["sha256"])
        self.assertEqual(self.row_path.read_bytes(), self.original)
        self.assert_originals()

    def test_recorded_inode_cannot_be_replaced_by_current_observation(self):
        self.fixture()
        row = dict(self.row, identity=[self.row["identity"][0], self.row["identity"][1] + 1])
        write_record(self.row_path, row)
        before = self.row_path.read_bytes()
        with self.assertRaises(StorageError):
            self.recover(apply=True, digest=self.rawpin(self.row_path)["sha256"])
        self.assertEqual(self.row_path.read_bytes(), before)
        self.assert_originals()

    def test_existing_marker_mismatch_rejects_without_rewriting_it(self):
        self.fixture()
        path = self.target / recovery.MARKER
        path.write_bytes(b"different original marker")
        self.archive_bound()
        with self.assertRaisesRegex(StorageError, "marker differs"):
            self.recover(apply=True)
        self.assertEqual(path.read_bytes(), b"different original marker")
        self.assertEqual(self.row_path.read_bytes(), self.original)
        self.assert_originals()

    def test_unlisted_or_nul_truncated_member_is_not_accepted_as_recovery_evidence(self):
        self.fixture()
        for hidden_suffix in (False, True):
            with self.subTest(hidden_suffix=hidden_suffix):
                self.archive_bound(extra=not hidden_suffix)
                if hidden_suffix:
                    with zipfile.ZipFile(self.archive_path) as archive:
                        originals = [(item, archive.read(item)) for item in archive.infolist()]
                    with zipfile.ZipFile(self.archive_path, "w") as archive:
                        for item, raw in originals:
                            if item.filename == "corpus/failure.log":
                                # Setting the raw writer name after construction
                                # avoids ZipInfo's constructor NUL truncation.
                                item = zipfile.ZipInfo(item.filename)
                                item.filename += "\x00hidden-suffix"
                            archive.writestr(item, raw)
                    self.archive_pin = self.rawpin(self.archive_path)
                message = "ZIP members required" if hidden_suffix else "extra members forbidden"
                with self.assertRaisesRegex(StorageError, message):
                    self.recover(apply=True)
                self.assertEqual(self.row_path.read_bytes(), self.original)
                self.assert_originals()

    def test_independent_archive_actual_bytes_are_additional_quota_charge(self):
        self.fixture()
        actual = policy.CatalogView.checked_usage
        def near_limit(view):
            rows, _ = actual(view)
            return rows, 18 * GiB
        with patch.object(policy.CatalogView, "checked_usage", new=near_limit):
            with self.assertRaisesRegex(StorageError, "aggregate catalog/archive"):
                self.recover(apply=True)
        self.assertEqual(self.row_path.read_bytes(), self.original)
        self.assert_originals()

    def test_commit_failure_before_replace_preserves_primary_and_records_uncertain_attempt(self):
        self.fixture()
        class HostileDiagnostics(OSError):
            def __setattr__(self, name, value):
                if name == "bound_recovery_receipt":
                    raise RuntimeError("injected diagnostic setattr failure")
                return super().__setattr__(name, value)
            def add_note(self, message):
                raise RuntimeError("injected diagnostic note failure")
        class HostileRepr(OSError):
            def __repr__(self):
                raise RuntimeError("injected secondary diagnostic repr failure")
        for diagnostic in ("normal", "setattr-note", "deepcopy"):
            with self.subTest(diagnostic=diagnostic):
                primary = (HostileDiagnostics if diagnostic == "setattr-note" else OSError)(
                    "injected original atomic write failure")
                diagnostic_patch = None
                def failed_write(path, row):
                    nonlocal diagnostic_patch
                    if diagnostic == "deepcopy":
                        # Only receipt diagnostics after the actual primary is
                        # raised are poisoned; ordinary execution still works.
                        diagnostic_patch = patch.object(recovery.copy, "deepcopy",
                            side_effect=KeyboardInterrupt("injected diagnostic copy failure"))
                        diagnostic_patch.start()
                    raise primary
                try:
                    with patch.object(recovery, "write_record", side_effect=failed_write):
                        with self.assertRaises(OSError) as caught:
                            self.recover(apply=True)
                finally:
                    if diagnostic_patch is not None:
                        diagnostic_patch.stop()
                self.assertIs(caught.exception, primary)
                if diagnostic == "normal":
                    receipt = caught.exception.bound_recovery_receipt
                    self.assertTrue(receipt["catalog_write_attempted"])
                    self.assertIsNone(receipt["applied"])
                    self.assertFalse(receipt["catalog_write_returned"])
                    self.assertFalse(receipt["canonical_readback_verified"])
                    self.assertEqual(receipt["status"], "commit_effects_unknown_preserved")
                events = []
                class Resource:
                    def __init__(self, name, fail=False):
                        self.name, self.fail = name, fail
                    def close(self, original):
                        self_outer.assertIs(original, primary)
                        events.append(self.name)
                        if self.fail:
                            raise HostileRepr("secondary fixture close failure")
                self_outer = self
                recovery._close([Resource("remaining"), Resource("secondary", True)], primary)
                self.assertEqual(events, ["secondary", "remaining"])
                self.assertEqual(self.row_path.read_bytes(), self.original)
                self.assert_originals()

    def test_atomic_replace_then_error_is_not_rolled_back_or_retried(self):
        self.fixture()
        primary, calls = OSError("injected post-replace fsync failure"), []
        def replace_then_fail(path, row):
            calls.append(path)
            write_record(path, row)
            raise primary
        with patch.object(recovery, "write_record", side_effect=replace_then_fail):
            with self.assertRaises(OSError) as caught:
                self.recover(apply=True)
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, [self.row_path])
        self.assertEqual(json.loads(self.row_path.read_bytes())["state"], "interrupted_retained")
        self.assertFalse(caught.exception.bound_recovery_receipt["catalog_write_returned"])
        self.assertIsNone(caught.exception.bound_recovery_receipt["applied"])
        self.assert_originals()

    def test_post_commit_readback_failure_keeps_new_row_and_original_zip_separate(self):
        self.fixture()
        actual, calls = recovery._snapshots, []
        def snapshot(rows, until):
            result = actual(rows, until)
            if json.loads(self.row_path.read_bytes())["state"] == "interrupted_retained":
                calls.append("post-commit")
                raise StorageError("injected canonical readback failure")
            return result
        with patch.object(recovery, "_snapshots", side_effect=snapshot):
            with self.assertRaisesRegex(StorageError, "canonical readback failure") as caught:
                self.recover(apply=True)
        self.assertEqual(calls, ["post-commit"])
        receipt = caught.exception.bound_recovery_receipt
        self.assertTrue(receipt["catalog_write_attempted"])
        self.assertTrue(receipt["catalog_write_returned"])
        self.assertFalse(receipt["canonical_readback_verified"])
        self.assertIsNone(receipt["applied"])
        self.assertEqual(json.loads(self.row_path.read_bytes())["state"], "interrupted_retained")
        with zipfile.ZipFile(self.archive_path) as archive:
            self.assertEqual(archive.read(recovery.ORIGINAL_MEMBER), self.original)
        self.assert_originals()


if __name__ == "__main__":
    unittest.main()
