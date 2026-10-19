"""CPU/private-file recovery contracts; injected boots never prove a real reboot.

No production catalog, subprocess, model, network or accelerator is used.  Root
executes these tests; their synthetic lifetime changes are fixture inputs only.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_output_quarantine as custody
import managed_output_quarantine_transition as transitions
import managed_output_recovery as recovery
import managed_outputs as outputs
from managed_storage import GiB, StorageError, read_record_snapshot, write_record
import test_output_quarantine_transition as prior


class SealedRecoveryPredicates(unittest.TestCase):
    def setUp(self):
        self.key, self.digest = "1" * 64, "2" * 64
        self.target = Path("/private/fixture-output")
        self.now = {"host": "fixture-host", "boot": "fixture-new-boot",
                    "platform": "linux", "process_birth": "fixture-new-process"}
        self.row = {"schema": outputs.SCHEMA, "key": self.key, "output": str(self.target),
            "state": "active", "deferred": True, "identity": [7, 10],
            "pid": 123, "token": "original-fixture-token", "reserved_bytes": 1 << 20,
            "catalog_limit_bytes": 18 * GiB, "tree_gone": False,
            "lifetime": dict(self.now, boot="fixture-old-boot", process_birth="fixture-old-process"),
            "error": "original failure unchanged", "physical_completion": "unknown"}

    def eligible(self, *, row=None, digest=None, now=None):
        recovery._eligible(self.row if row is None else row, self.key,
            self.digest if digest is None else digest, self.digest, self.target,
            self.now if now is None else now)

    def test_only_original_bound_deferred_reboot_predicates_admit_without_mutation(self):
        original = copy.deepcopy(self.row)
        self.eligible()
        self.assertEqual(self.row, original)
        self.assertIsNone(recovery._eligible(self.row, self.key, self.digest,
                                             self.digest, self.target, self.now))

    def test_same_boot_foreign_host_platform_and_missing_birth_are_not_dead_proofs(self):
        for now in (dict(self.now, boot=self.row["lifetime"]["boot"]),
                    dict(self.now, host="foreign-host"), dict(self.now, platform="win32"),
                    {key: value for key, value in self.now.items() if key != "process_birth"}):
            with self.subTest(now=now), self.assertRaises(StorageError):
                self.eligible(now=now)

    def test_unbound_non_deferred_retained_digest_and_physical_upgrades_reject(self):
        for changes in ({"identity": None}, {"identity": [True, 10]}, {"identity": [7, 0]},
                        {"deferred": 1}, {"deferred": False}, {"state": "retained"},
                        {"tree_gone": True}, {"physical_completion": "confirmed"},
                        {"reserved_bytes": True}, {"catalog_limit_bytes": 18 * GiB + 1}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                self.eligible(row=dict(self.row, **changes))
        with self.assertRaisesRegex(StorageError, "digest changed"):
            self.eligible(digest="3" * 64)

    def test_exact_booleans_explicit_pins_and_one_total_window_are_mandatory(self):
        arguments = dict(quarantine_transition_receipts=[], current_source_pins={},
                         root_exclusive_writer=True)
        for changes in ({"root_exclusive_writer": 1}, {"apply": 1},
                        {"maximum_seconds": True}, {"maximum_seconds": 31},
                        {"maximum_seconds": float("nan")}, {"current_source_pins": None}):
            with self.subTest(changes=changes), self.assertRaises(StorageError):
                recovery.recover_sealed_output_after_reboot(Path("/not-opened"), self.key,
                    self.digest, **dict(arguments, **changes))
        with patch.object(recovery.time, "monotonic", return_value=10):
            with self.assertRaisesRegex(StorageError, "deadline expired"):
                recovery._remaining(10)


@unittest.skipUnless(sys.platform == "linux", "Linux original private FD/mount fixtures; Root executes")
class SealedRecoveryLinuxTests(unittest.TestCase):
    """Original OutputLease seals the row; public transitions validate the view.

    The prior fixture supplies private real directories/source copies, original
    FD/hash/mount walkers and a deliberately injected lifetime.  A fixture reboot
    is not evidence for a real operator reboot or any physical GPU completion.
    """
    pins = prior.TransitionLinuxTests.pins
    anchor = prior.TransitionLinuxTests.anchor
    move = prior.TransitionLinuxTests.move

    def setUp(self):
        prior.TransitionLinuxTests.setUp(self)
        self.stack.enter_context(patch.object(recovery, "lifetime", side_effect=lambda: dict(self.now)))
        self.first = self.move(self.anchor())["receipt"]
        owner = outputs.OutputLease(self.parent / "sealed", self.source, catalog=self.catalog,
            reserve=1 << 20, deferred=True, quarantine_transition_receipts=[self.first]).acquire()
        try:
            owner.create_deferred_directory()
            (owner.output / "failure.log").write_bytes(b"original failed-run evidence; physical unknown\n")
            owner.check_budget()
            self.sealed_target, self.sealed_key = owner.output, owner.key
            self.sealed_path = self.catalog / (owner.key + ".json")
            terminal = owner.finish(status="failed-before-child-fixture", tree_gone=False)
            self.assertEqual(terminal["state"], "active")
            self.assertIs(terminal["tree_gone"], False)
        finally:
            # Original finish(False) leaves an active, bound row and explicitly
            # makes no physical inference.  Close any descriptors still held
            # if the original producer setup itself raised; do not edit its row.
            try:
                owner._close_created_directory()
            finally:
                try:
                    owner._close_quarantine_view()
                finally:
                    owner.guard.release()
        self.original = read_record_snapshot(self.sealed_path, schema=outputs.SCHEMA)
        self.original_raw = self.sealed_path.read_bytes()
        self.payload = (self.sealed_target / "failure.log").read_bytes()
        self.parent_identity = outputs.identity(self.sealed_target.parent)
        self.now.update(boot="fixture-later-boot", process_birth="fixture-later-maintainer")
        self.tip = transitions.publish_custody_transition(self.catalog, self.row_pin, self.first,
            [self.first], self.pins(), reason="reboot_reobserve", attempt_id="2" * 32,
            root_exclusive_writer=True, apply=True)["receipt"]
        self.preserved = {path: path.read_bytes() for path in self.registry.iterdir()}
        self.preserved[self.row_path] = self.raw

    def recover(self, *, apply=False, **changes):
        arguments = dict(quarantine_transition_receipts=[self.tip], current_source_pins=self.pins(),
                         root_exclusive_writer=True, apply=apply)
        arguments.update(changes)
        return recovery.recover_sealed_output_after_reboot(self.catalog, self.sealed_key,
                                                           self.original[1], **arguments)

    def assert_retained_evidence_unchanged(self):
        self.assertEqual((self.sealed_target / "failure.log").read_bytes(), self.payload)
        self.assertEqual(outputs.identity(self.sealed_target), self.original[0]["identity"])
        self.assertEqual(outputs.identity(self.sealed_target.parent), self.parent_identity)
        self.assertEqual(sorted(path.name for path in self.sealed_target.iterdir()), ["failure.log"])
        self.assertFalse((self.sealed_target / ".rovezero-output.json").exists())
        for path, raw in self.preserved.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_dry_run_then_apply_only_one_row_and_keep_unknown_originals_and_error(self):
        result = self.recover()
        self.assertEqual(result["status"], "eligible_but_not_applied")
        self.assertIs(result["applied"], False)
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()
        result = self.recover(apply=True)
        actual, _, _ = read_record_snapshot(self.sealed_path, schema=outputs.SCHEMA)
        self.assertEqual(result["status"], "interrupted_retained")
        self.assertEqual(result["row"], actual)
        self.assertEqual(actual["recovery_reason"], "verified_same_host_reboot")
        self.assertEqual(actual["recovery_source_sha256"], self.original[1])
        self.assertEqual(actual["physical_completion"], "not_inferred")
        self.assertIs(actual["tree_gone"], False)
        self.assertEqual(actual["status"], "failed-before-child-fixture")
        changed = {"state", "recovery_reason", "recovered_ns", "recovery_source_sha256", "physical_completion"}
        self.assertEqual({key: value for key, value in actual.items() if key not in changed},
                         {key: value for key, value in self.original[0].items() if key not in changed})
        self.assert_retained_evidence_unchanged()

    def test_same_boot_foreign_host_stale_tip_and_unknown_row_do_not_write(self):
        good = dict(self.now)
        for changes in ({"boot": self.original[0]["lifetime"]["boot"]}, {"host": "foreign-host"}):
            self.now.update(changes)
            try:
                with self.subTest(changes=changes), self.assertRaises(StorageError):
                    self.recover(apply=True)
            finally:
                self.now.clear()
                self.now.update(good)
        with self.assertRaises(StorageError):
            self.recover(apply=True, quarantine_transition_receipts=[self.first])
        with self.assertRaises(StorageError):
            recovery.recover_sealed_output_after_reboot(self.catalog, self.key, self.row_pin["sha256"],
                quarantine_transition_receipts=[self.tip], current_source_pins=self.pins(),
                root_exclusive_writer=True, apply=True)
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()

    def test_original_readable_catalog_accepts_0755_and_rejects_foreign_write_permissions(self):
        # Exercise the whole public path with the original root-owned catalog
        # permission policy, rather than duplicating its permission expression.
        self.catalog.chmod(0o755)
        result = self.recover()
        self.assertEqual(result["status"], "eligible_but_not_applied")
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        for mode in (0o775, 0o757, 0o777):
            self.catalog.chmod(mode)
            with self.subTest(mode=oct(mode)), patch.object(recovery, "write_record") as commit:
                with self.assertRaisesRegex(StorageError, "owner-controlled original catalog"):
                    self.recover(apply=True)
                commit.assert_not_called()
            self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.catalog.chmod(0o755)
        self.assertEqual(self.recover(apply=True)["status"], "interrupted_retained")
        self.assert_retained_evidence_unchanged()

    def test_digest_and_bound_directory_replacement_are_refused_before_commit(self):
        with self.assertRaisesRegex(StorageError, "digest changed"):
            recovery.recover_sealed_output_after_reboot(self.catalog, self.sealed_key, "0" * 64,
                quarantine_transition_receipts=[self.tip], current_source_pins=self.pins(),
                root_exclusive_writer=True, apply=True)
        moved = self.parent / "original-kept"
        self.sealed_target.rename(moved)
        self.sealed_target.mkdir(mode=0o700)
        with patch.object(recovery, "write_record") as commit:
            with self.assertRaises(StorageError):
                self.recover(apply=True)
            commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assertEqual((moved / "failure.log").read_bytes(), self.payload)

    def test_current_source_raw_change_and_live_view_mount_change_refuse(self):
        pins = self.pins()
        path = self.source_paths["storage"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# explicit fixture source mismatch\n")
            with patch.object(recovery, "write_record") as commit:
                with self.assertRaises(StorageError):
                    self.recover(apply=True, current_source_pins=pins)
                commit.assert_not_called()
        finally:
            path.write_bytes(original)
        original_mounts = custody._mounts
        calls = [0]
        def changed_mounts(*args):
            value = original_mounts(*args)
            calls[0] += 1
            # The initial transition history sees the original observation;
            # a later actual-view check must not tolerate a changed namespace.
            if calls[0] > 2:
                value = copy.deepcopy(value)
                value["namespace"][1] += 1
            return value
        with patch.object(custody, "_mounts", side_effect=changed_mounts):
            with patch.object(recovery, "write_record") as commit:
                with self.assertRaises(StorageError):
                    self.recover(apply=True)
                commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()

    def test_original_reservation_overflow_is_not_recovered_or_recharged(self):
        (self.sealed_target / "over-budget.log").write_bytes(b"x" * (1 << 20))
        with patch.object(recovery, "write_record") as commit:
            with self.assertRaisesRegex(StorageError, "reservation"):
                self.recover(apply=True)
            commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assertEqual((self.sealed_target / "failure.log").read_bytes(), self.payload)
        self.assertEqual((self.sealed_target / "over-budget.log").stat().st_size, 1 << 20)
        for path, raw in self.preserved.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_precommit_failure_keeps_all_raw_and_first_error(self):
        with patch.object(recovery, "write_record", side_effect=OSError("fixture original commit failure")):
            with self.assertRaisesRegex(OSError, "original commit failure") as caught:
                self.recover(apply=True)
        self.assertTrue(any("write_record returned=False" in note
                            for note in getattr(caught.exception, "__notes__", [])))
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()

    def test_postreplace_failure_preserves_actual_row_and_error_without_rollback(self):
        def fail_after_replace(path, row):
            write_record(path, row)
            raise OSError("fixture post-replace durability failure")
        with patch.object(recovery, "write_record", side_effect=fail_after_replace):
            with self.assertRaisesRegex(OSError, "post-replace durability failure") as caught:
                self.recover(apply=True)
        self.assertTrue(any("commit attempted" in note for note in getattr(caught.exception, "__notes__", [])))
        actual = read_record_snapshot(self.sealed_path, schema=outputs.SCHEMA)[0]
        self.assertEqual(actual["state"], "interrupted_retained")
        self.assertEqual(actual["physical_completion"], "not_inferred")
        self.assertIs(actual["tree_gone"], False)
        self.assert_retained_evidence_unchanged()

    def test_inner_validation_cannot_renew_the_one_outer_window(self):
        original = transitions.validate_transitions
        real_clock = recovery.time.monotonic
        expired = [False]
        def clock():
            return real_clock() + (31 if expired[0] else 0)
        def expire_after_validation(*args, **kwargs):
            self.assertGreater(kwargs["maximum_seconds"], 0)
            self.assertLessEqual(kwargs["maximum_seconds"], 30)
            view = original(*args, **kwargs)
            expired[0] = True
            return view
        with patch.object(recovery.time, "monotonic", side_effect=clock):
            with patch.object(transitions, "validate_transitions", side_effect=expire_after_validation):
                with patch.object(recovery, "write_record") as commit:
                    with self.assertRaisesRegex(StorageError, "deadline expired"):
                        self.recover(apply=True)
                    commit.assert_not_called()
        self.assertEqual(self.sealed_path.read_bytes(), self.original_raw)
        self.assert_retained_evidence_unchanged()


if __name__ == "__main__":
    unittest.main()
