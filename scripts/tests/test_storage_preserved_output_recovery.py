"""Private reboot/FD fixtures; mocked history is not real Source7 acceptance."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import storage_preserved_output_recovery as recovery
import managed_outputs as outputs
import managed_output_unattributed_preservation as policy
from managed_storage import StorageError, GiB
from storage_guard import lifetime
import test_managed_output_unattributed_preservation as fixtures


def dump(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode())
    path.chmod(0o600)
    return fixtures.rawpin(path)


class UnsupportedTests(unittest.TestCase):
    def test_windows_refuses_before_any_guard_or_metadata_open(self):
        with patch.object(recovery.sys, "platform", "win32"), patch.object(recovery, "CatalogGuard") as guard:
            with self.assertRaises(recovery.PreservedOutputRecoveryUnsupported):
                recovery.recover_preserved_output(None, None, None, None, None, None,
                    preserved_owner_pin=None, preserved_row_pin=None)
            guard.assert_not_called()


@unittest.skipUnless(sys.platform == "linux" and getattr(os, "geteuid", lambda: -1)() == 0,
                     "native UID0 private FD fixtures required; a skip is not acceptance")
class PreservedOutputRecoveryTests(unittest.TestCase):
    _archive = fixtures.UnattributedInventoryTests._archive
    selection = fixtures.UnattributedInventoryTests.selection
    _mock_history = fixtures.UnattributedInventoryTests._mock_history
    _history_patches = fixtures.UnattributedInventoryTests._history_patches

    def setUp(self):
        fixtures.UnattributedInventoryTests.setUp(self)
        self.build = self.root / "build"
        self.build.mkdir(mode=0o700)
        self.current = lifetime()
        old = dict(self.current, boot="prior-fixture-boot-" + self.current["boot"])
        purpose = "preserved-output-recovery-fixture"
        source_id = hashlib.sha256((os.path.normcase(str(self.source)) + "\0" + purpose).encode()).hexdigest()[:24]
        self.slot = self.build / "slots" / source_id
        self.slot.mkdir(parents=True, mode=0o700)
        self.owner = {"schema": recovery.storage.SCHEMA, "token": "a" * 32, "pid": os.getpid(),
                      "source_id": source_id, "source": str(self.source), "lifetime": old,
                      "root_identity": list(recovery._identity(self.build.stat())), "purpose": purpose}
        self.owner_path = self.build / "owner.json"
        self.target = self.root / "interrupted-deferred-output"
        self.target.mkdir(mode=0o700)
        (self.target / "payload.txt").write_bytes(b"original failed payload retained")
        self.key = hashlib.sha256(os.path.normcase(str(self.target)).encode()).hexdigest()
        self.row_path = self.catalog / (self.key + ".json")
        self.row = {"schema": outputs.SCHEMA, "key": self.key, "output": str(self.target),
                    "identity": outputs.identity(self.target), "token": "b" * 32,
                    "source": str(self.source), "pid": os.getpid(), "lifetime": old,
                    "state": "active", "deferred": True, "reserved_bytes": 1 << 20,
                    "catalog_limit_bytes": 18 * GiB, "status": "original_functional_failure",
                    "tree_gone": False, "physical_completion": "unknown_original",
                    "outer_supervisor": {"root": str(self.build), "owner": self.owner,
                                         "scope": "inherited_outer_process_group_or_job"}}
        copies = self.root / "independent-preservation"
        copies.mkdir(mode=0o700)
        self.owner_copy, self.row_copy = copies / "owner-original.json", copies / "row-original.json"
        self.refresh_originals()
        canonical = patch.object(recovery.storage, "canonical_root", return_value=self.build)
        canonical.start()
        self.addCleanup(canonical.stop)
        history = self._history_patches()
        self.addCleanup(history.close)

    def refresh_originals(self):
        self.owner_pin = dump(self.owner_path, self.owner)
        dump(self.slot / "record.json", {**self.owner, "state": "active", "finished_ns": 0})
        if self.row["outer_supervisor"] is not None:
            self.row["outer_supervisor"]["owner"] = copy.deepcopy(self.owner)
        self.row_pin = dump(self.row_path, self.row)
        self.owner_copy.write_bytes(self.owner_path.read_bytes())
        self.row_copy.write_bytes(self.row_path.read_bytes())
        self.owner_copy.chmod(0o600)
        self.row_copy.chmod(0o600)

    def invoke(self, apply=False, selection=None):
        return recovery.recover_preserved_output(self.build, self.catalog, self.owner_pin["sha256"],
            self.key, self.row_pin["sha256"], selection or self.selection(),
            preserved_owner_pin=fixtures.rawpin(self.owner_copy),
            preserved_row_pin=fixtures.rawpin(self.row_copy), apply=apply)

    def protected(self):
        paths = [self.owner_path, self.slot / "record.json", self.owner_copy, self.row_copy,
                 self.target / "payload.txt", *self.source_paths.values()]
        for item in self.items:
            paths.extend(Path(pin["path"]) for pin in item.values())
        return {path: path.read_bytes() for path in paths}

    def assert_protected(self, before):
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def refuse(self, pattern):
        before, row = self.protected(), self.row_path.read_bytes()
        with self.assertRaisesRegex(StorageError, pattern):
            self.invoke(True)
        self.assertEqual(self.row_path.read_bytes(), row)
        self.assert_protected(before)

    def test_default_inspect_is_readonly_and_no_token_or_raw_escapes(self):
        before, row = self.protected(), self.row_path.read_bytes()
        with patch.object(recovery.storage, "write_record", side_effect=AssertionError("inspect may not write")):
            result = self.invoke()
        self.assertEqual(result["status"], "ELIGIBLE_READ_ONLY")
        self.assertFalse(result["catalog_row_changed"])
        self.assertFalse(result["original_creation_process_attested"])
        self.assertEqual(result["original_row_outer_binding"], "matched_original_record")
        self.assertLessEqual(len(json.dumps(result).encode()), recovery.ARTIFACT_CAP)
        self.assertNotIn(self.owner["token"], json.dumps(result))
        self.assertNotIn(self.row["token"], json.dumps(result))
        self.assertEqual(self.row_path.read_bytes(), row)
        self.assert_protected(before)

    def test_apply_changes_exactly_one_row_and_preserves_failed_physical_fields(self):
        before = self.protected()
        with patch.object(recovery.storage, "write_record", wraps=recovery.storage.write_record) as write:
            result = self.invoke(True)
        self.assertEqual(write.call_count, 1)
        self.assertEqual(write.call_args.args[0], self.row_path)
        actual = outputs._read(self.row_path)
        for name, value in self.row.items():
            if name != "state":
                self.assertEqual(actual[name], value)
        self.assertEqual(actual["state"], "interrupted_retained")
        self.assertEqual(actual["recovery_reason"], "verified_same_host_reboot")
        self.assertEqual(actual["recovery_source_sha256"], self.row_pin["sha256"])
        self.assertEqual(result["recovered_row_sha256"], fixtures.rawpin(self.row_path)["sha256"])
        self.assertFalse(result["physical_completion_attested"])
        self.assertFalse((self.target / ".rovezero-output.json").exists())
        self.assert_protected(before)
        with self.assertRaisesRegex(StorageError, "digest changed"):
            self.invoke(True)

    def test_same_boot_pid_absence_is_not_reboot_proof(self):
        self.owner["lifetime"] = dict(self.current)
        self.owner["pid"] = 999999999
        self.refresh_originals()
        self.refuse("same-host build owner reboot")

    def test_original_absent_outer_and_independent_source_stay_unbound(self):
        separate_source = self.root / "independent-row-source"
        separate_source.mkdir(mode=0o700)
        self.row["outer_supervisor"] = None
        self.row["source"] = str(separate_source)
        self.row["pid"] = self.owner["pid"] + 17
        self.row["lifetime"] = dict(self.row["lifetime"], process_birth="separate-original-birth",
                                    boot="independent-prior-fixture-boot")
        self.refresh_originals()
        before = self.protected()
        result = self.invoke(True)
        actual = outputs._read(self.row_path)
        self.assertIsNone(actual["outer_supervisor"])
        self.assertEqual(actual["source"], str(separate_source))
        self.assertEqual(actual["pid"], self.row["pid"])
        self.assertEqual(actual["lifetime"], self.row["lifetime"])
        self.assertEqual(result["original_row_outer_binding"], "absent_original")
        self.assertFalse(result["row_build_relationship_attested"])
        self.assertFalse(result["original_creation_process_attested"])
        self.assert_protected(before)

    def test_absent_outer_still_requires_independent_row_reboot(self):
        self.row["outer_supervisor"] = None
        self.row["lifetime"] = dict(self.current)
        self.refresh_originals()
        self.refuse("same-host original output reboot")

    def test_dictionary_selection_cannot_authorize_any_guard_or_write(self):
        with patch.object(recovery, "CatalogGuard") as guard:
            with self.assertRaisesRegex(StorageError, "exact typed Source7"):
                self.invoke(True, {"not": "a typed live selection"})
            guard.assert_not_called()

    def test_foreign_host_owner_is_never_recovered(self):
        self.owner["lifetime"] = dict(self.owner["lifetime"], host="0" * 64)
        self.refresh_originals()
        self.refuse("same-host build owner reboot")

    def test_row_requires_its_own_same_host_reboot_proof(self):
        self.row["lifetime"] = dict(self.current)
        self.refresh_originals()
        self.refuse("same-host original output reboot")

    def test_outer_owner_and_original_slot_must_match_exactly(self):
        dump(self.slot / "record.json", {**self.owner, "token": "c" * 32,
                                        "state": "active", "finished_ns": 0})
        self.refuse("slot and exact build owner")

    def test_outer_scope_is_not_replaced_by_unbound_owner_metadata(self):
        self.row["outer_supervisor"]["scope"] = "unknown_original_scope"
        self.refresh_originals()
        self.refuse("exact original outer build owner/source/scope")

    def test_identity_null_nondeferred_and_nonactive_are_refused(self):
        original = copy.deepcopy(self.row)
        for change in ({"identity": None}, {"deferred": False}, {"state": "retained"}):
            with self.subTest(change=change):
                self.row = {**copy.deepcopy(original), **change}
                self.refresh_originals()
                self.refuse("active identity-bound deferred")

    def test_current_directory_is_not_adopted_after_identity_replacement(self):
        displaced = self.root / "original-output-preserved"
        self.target.rename(displaced)
        self.target.mkdir(mode=0o700)
        with self.assertRaisesRegex(StorageError, "directory identity differs"):
            self.invoke(True)
        self.assertTrue((displaced / "payload.txt").is_file())
        self.assertEqual(outputs._read(self.row_path), self.row)

    def test_nonprivate_output_and_symlink_are_refused(self):
        self.target.chmod(0o755)
        self.refuse("private/control directory|private 0700")
        self.target.chmod(0o700)
        moved = self.root / "moved-private-output"
        self.target.rename(moved)
        self.target.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(StorageError, "symlink|junction|aliases"):
            self.invoke(True)
        self.assertTrue((moved / "payload.txt").exists())

    def test_preserved_copy_must_be_independent_and_exact(self):
        self.row_copy.write_bytes(b"{}\n")
        self.refuse("original raw bytes")
        self.row_copy.unlink()
        os.link(self.row_path, self.row_copy)
        with self.assertRaisesRegex(StorageError, "single-link"):
            self.invoke(True)

    def test_selected_or_historical_key_never_receives_recovery_exception(self):
        original = policy.observe_catalog
        def observe(*args, **kwargs):
            view = original(*args, **kwargs)
            view.keys = view.keys | {self.key}
            return view
        with patch.object(policy, "observe_catalog", side_effect=observe):
            self.refuse("selected or historical preserved")

    def test_physical_alias_of_preserved_scope_is_refused(self):
        original = policy._physical_scope
        def physical(path, table):
            return original(self.targets[0] if Path(path) == self.target else path, table)
        with patch.object(policy, "_physical_scope", side_effect=physical):
            self.refuse("aliases preserved/control physical scope")

    def test_unknown_final_history_check_does_not_publish_a_row(self):
        with patch.object(policy.CatalogView, "checked_usage",
                          side_effect=StorageError("fixture final history unknown")):
            self.refuse("fixture final history unknown")

    def test_original_samebytes_row_name_replacement_is_refused(self):
        original = policy.observe_catalog
        def observe(*args, **kwargs):
            view = original(*args, **kwargs)
            swap = self.catalog / "private-swap"
            swap.write_bytes(self.row_path.read_bytes())
            os.replace(swap, self.row_path)
            return view
        with patch.object(policy, "observe_catalog", side_effect=observe):
            self.refuse("original deferred row changed during preserved observation")

    def test_changed_current_source_after_final_fullcheck_stops_before_write(self):
        original = policy.CatalogView.checked_usage
        before = self.row_path.read_bytes()
        def checked(view):
            result = original(view)
            self.source_paths["guard"].write_bytes(b"# changed source after fullcheck\n")
            return result
        with patch.object(policy.CatalogView, "checked_usage", new=checked):
            with self.assertRaisesRegex(StorageError, "Source7"):
                self.invoke(True)
        self.assertEqual(self.row_path.read_bytes(), before)

    def test_postreplace_write_error_keeps_actual_row_and_first_error_without_retry(self):
        before = self.protected()
        original = recovery.storage.write_record
        primary = OSError("fixture directory fsync after replace failed")
        def failed(path, value):
            original(path, value)
            raise primary
        original_close = policy.CatalogView.close
        def close(view, error=None):
            original_close(view, error)
            raise OSError("fixture secondary view close failure")
        with patch.object(recovery.storage, "write_record", side_effect=failed) as write, \
                patch.object(policy.CatalogView, "close", new=close):
            with self.assertRaises(OSError) as caught:
                self.invoke(True)
        self.assertIs(caught.exception, primary)
        self.assertEqual(write.call_count, 1)
        evidence = primary.preserved_output_recovery
        self.assertTrue(evidence["catalog_write_attempted"])
        self.assertFalse(evidence["catalog_write_returned"])
        self.assertFalse(evidence["automatic_retry_authorized"])
        self.assertEqual(outputs._read(self.row_path)["state"], "interrupted_retained")
        self.assert_protected(before)
        self.assertTrue(any("secondary recovery observation close" in note for note in primary.__notes__))

    def test_close_failure_after_valid_publication_never_returns_success(self):
        original_close = policy.CatalogView.close
        def close(view, error=None):
            original_close(view, error)
            raise OSError("fixture final close failed")
        with patch.object(policy.CatalogView, "close", new=close):
            with self.assertRaisesRegex(OSError, "final close failed") as caught:
                self.invoke(True)
        evidence = caught.exception.preserved_output_recovery
        self.assertTrue(evidence["catalog_write_attempted"])
        self.assertTrue(evidence["catalog_write_returned"])
        self.assertEqual(outputs._read(self.row_path)["state"], "interrupted_retained")

    def test_expired_original_window_is_not_refreshed(self):
        selected = self.selection()
        with patch.object(policy.time, "monotonic", return_value=selected.original_hard_until):
            with self.assertRaisesRegex(StorageError, "expired"):
                self.invoke(True, selected)
        self.assertEqual(outputs._read(self.row_path), self.row)

    def test_plain_default_recovery_still_refuses_custody_registry(self):
        registry = policy.custody.registry_path(self.catalog)
        registry.mkdir(mode=0o700)
        with self.assertRaisesRegex(StorageError, "custody registry"):
            outputs.recover_output(self.catalog, self.key, self.row_pin["sha256"])
        self.assertEqual(outputs._read(self.row_path), self.row)


if __name__ == "__main__":
    unittest.main()
