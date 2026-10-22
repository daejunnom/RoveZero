"""Focused source7 poll regressions; Main alone imports and runs this file.

Linux cases reuse tiny private archive/catalog fixtures and an explicit mocked
historical-chain seam. They exercise the actual live view/guard/journal checks,
not a canonical user catalog, old recovery authority or product acceptance.
The portable cases do not claim Linux FD validation on another platform.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_output_unattributed_preservation as policy
import managed_outputs as outputs
from managed_storage import StorageError
import test_managed_output_unattributed_preservation as fixtures


class PollDispatchTests(unittest.TestCase):
    def test_ordinary_lease_keeps_default_full_path(self):
        owner = object.__new__(outputs.OutputLease)
        owner.unattributed_preservation = None
        with patch.object(owner, "check_budget", return_value="full") as full:
            self.assertEqual(owner.poll_budget(), "full")
        full.assert_called_once_with()

    def test_existing_failure_is_preserved_before_any_poll(self):
        owner = object.__new__(outputs.OutputLease)
        owner.unattributed_preservation = object()
        primary = KeyboardInterrupt("original failure")
        owner._preservation_error = primary
        with self.assertRaises(StorageError) as caught:
            owner.poll_budget()
        self.assertIs(caught.exception.__cause__, primary)
        self.assertIs(owner._preservation_error, primary)


@unittest.skipUnless(sys.platform == "linux", "requires actual private Linux held FD/mount observations")
class SealedBudgetPollTests(unittest.TestCase):
    def setUp(self):
        # Composition avoids inheriting and rerunning the older suite's tests.
        self.fixture = fixtures.UnattributedInventoryTests()
        self.fixture.setUp()
        self.history = self.fixture._history_patches()
        self.history.__enter__()
        self.addCleanup(self.history.close)
        self.addCleanup(self.fixture.doCleanups)
        self.owner = self.fixture._journal_bound_owner("sealed-poll")
        if self._testMethodName != "test_acquisition_without_budget_binding_uses_one_full_observation_first":
            self.owner.check_budget()

    def _reject_sticky(self, change):
        owner = self.owner
        row = (owner.catalog / (owner.key + ".json")).read_bytes()
        journal = owner._preservation_journal
        entries = journal.entries
        change()
        with patch.object(policy, "observe_fresh_output_budget",
                          side_effect=AssertionError("poll failure must not be retried")):
            with self.assertRaises(BaseException) as caught:
                owner.poll_budget()
        # AssertionError from a forbidden full retry must never satisfy this test.
        self.assertNotIsInstance(caught.exception, AssertionError)
        self.assertIs(owner._preservation_error, caught.exception)
        self.assertEqual(journal.entries, entries)
        self.fixture._assert_poison_blocks_budget_and_finish(owner, caught.exception, row)
        return caught.exception

    def _rewrite_same_size(self, path):
        before = path.stat()
        raw = path.read_bytes()
        self.assertTrue(raw)
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        self.assertEqual(path.stat().st_size, len(raw))

    def test_repeated_poll_keeps_original_view_fds_journal_and_deadlines(self):
        owner, view, journal = self.owner, self.owner._quarantine_view, self.owner._preservation_journal
        original = (view, journal.fd, journal.directory, journal.size, journal.entries,
                    journal.head, copy.deepcopy(journal.stamp), view._budget_binding)
        raw = (owner.output / journal.name).read_bytes()
        with patch.object(policy, "_inventory", side_effect=AssertionError("no new archive proof")), \
                patch.object(policy._Held, "digest", side_effect=AssertionError("no payload rehash")), \
                patch.object(journal, "append", side_effect=AssertionError("read-only poll")):
            owner.poll_budget()
            owner.poll_budget()
        self.assertEqual((owner._quarantine_view, journal.fd, journal.directory, journal.size,
                          journal.entries, journal.head, journal.stamp, view._budget_binding), original)
        self.assertEqual((owner.output / journal.name).read_bytes(), raw)
        self.assertTrue(view._sealed)
        self.assertIsNone(owner._preservation_error)

    def test_acquisition_without_budget_binding_uses_one_full_observation_first(self):
        owner = self.owner
        before, journal = owner._quarantine_view, owner._preservation_journal
        entries = journal.entries
        self.assertIsNone(before._budget_binding)
        original = policy.observe_fresh_output_budget
        with patch.object(policy, "observe_fresh_output_budget", wraps=original) as full:
            owner.poll_budget()
            sealed = owner._quarantine_view
            owner.poll_budget()
        self.assertEqual(full.call_count, 1)
        self.assertIsNot(sealed, before)
        self.assertTrue(before._closed)
        self.assertIs(owner._quarantine_view, sealed)
        self.assertEqual(journal.entries, entries + 1)

    def test_expired_before_poll_entry_uses_fresh_full_view_without_renewing_hard_clock(self):
        owner, old = self.owner, self.owner._quarantine_view
        hard = owner.unattributed_preservation.original_hard_until
        entries = owner._preservation_journal.entries
        clock = min(old._budget_binding[6]) + 0.001
        self.assertLess(clock, hard)
        original = policy.observe_fresh_output_budget
        with patch.object(time, "monotonic", return_value=clock), \
                patch.object(policy, "observe_fresh_output_budget", wraps=original) as full:
            owner.poll_budget()
        full.assert_called_once()
        self.assertIsNot(owner._quarantine_view, old)
        self.assertTrue(old._closed)
        self.assertEqual(owner._preservation_journal.entries, entries + 1)
        self.assertEqual(owner.unattributed_preservation.original_hard_until, hard)
        self.assertLessEqual(owner._quarantine_view.inventory.until, hard)

    def test_expiry_after_poll_entry_is_sticky_and_never_retried(self):
        view = self.owner._quarantine_view
        original = view.inventory.checked_usage
        now = [time.monotonic()]
        def expired():
            now[0] = min(view._budget_binding[6]) + 0.001
            return original()
        with patch.object(time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(view.inventory, "checked_usage", side_effect=expired):
            error = self._reject_sticky(lambda: None)
        self.assertIsInstance(error, StorageError)
        self.assertIn("expired", str(error))
        self.assertTrue(view.inventory.failed)

    def test_archive_replacement_with_identical_bytes_is_rejected(self):
        path = Path(self.fixture.items[0]["research_archive"]["path"])
        def replace():
            other = path.with_name("same-archive.zip")
            other.write_bytes(path.read_bytes())
            other.replace(path)
        self._reject_sticky(replace)

    def test_same_size_archive_write_is_rejected_by_original_full_stamp(self):
        path = Path(self.fixture.items[0]["research_archive"]["path"])
        self._reject_sticky(lambda: self._rewrite_same_size(path))

    def test_same_size_corpus_write_is_rejected(self):
        path = self.fixture.targets[0] / "result.txt"
        self._reject_sticky(lambda: self._rewrite_same_size(path))

    def test_corpus_added_entry_is_rejected(self):
        self._reject_sticky(lambda: (self.fixture.targets[0] / "unexpected.txt").write_bytes(b"new"))

    def test_corpus_deleted_entry_is_rejected(self):
        self._reject_sticky(lambda: (self.fixture.targets[0] / "result.txt").unlink())

    def test_source_bytes_changed_in_place_are_rejected(self):
        path = self.fixture.source_paths["guard"]
        self._reject_sticky(lambda: self._rewrite_same_size(path))

    def test_small_source_blob_hash_is_rechecked_even_if_stat_checks_pass(self):
        held = self.owner._quarantine_view.inventory.held
        source = str(self.fixture.source_paths["guard"])
        fd = next(fd for fd, _, pin in held.blobs if pin["path"] == source)
        original = held.bytes
        # Fault injection at the same held FD read, leaving all name/stat tests
        # real. Payload archives are not rehashed by this bounded poll.
        with patch.object(held, "bytes", side_effect=lambda current, size:
                          b"!" * size if current == fd else original(current, size)):
            error = self._reject_sticky(lambda: None)
        self.assertIsInstance(error, StorageError)
        self.assertIn("metadata/source", str(error))

    def test_source_replacement_with_identical_bytes_is_rejected(self):
        path = self.fixture.source_paths["guard"]
        def replace():
            other = path.with_name("same-source.py")
            other.write_bytes(path.read_bytes())
            other.replace(path)
        self._reject_sticky(replace)

    def test_original_marker_change_is_rejected(self):
        path = Path(self.fixture.items[0]["marker"]["path"])
        self._reject_sticky(lambda: self._rewrite_same_size(path))

    def test_journal_external_append_is_rejected_without_repair(self):
        path = self.owner.output / self.owner._preservation_journal.name
        def append():
            with path.open("ab") as stream:
                stream.write(b"partial external write\n")
        self._reject_sticky(append)
        self.assertTrue(path.read_bytes().endswith(b"partial external write\n"))

    def test_journal_identical_name_replacement_is_rejected(self):
        path = self.owner.output / self.owner._preservation_journal.name
        def replace():
            other = path.with_name("replacement.jsonl")
            other.write_bytes(path.read_bytes())
            os.chmod(other, 0o600)
            other.replace(path)
        self._reject_sticky(replace)

    def test_different_journal_object_with_same_fds_is_rejected(self):
        journal = self.owner._preservation_journal
        other = copy.copy(journal)
        try:
            self._reject_sticky(lambda: setattr(self.owner, "_preservation_journal", other))
        finally:
            self.owner._preservation_journal = journal

    def test_different_journal_fd_even_for_same_open_description_is_rejected(self):
        journal = self.owner._preservation_journal
        original, duplicate = journal.fd, os.dup(journal.fd)
        try:
            self._reject_sticky(lambda: setattr(journal, "fd", duplicate))
        finally:
            journal.fd = original
            os.close(duplicate)

    def test_journal_stamp_mutation_cannot_mutate_the_sealed_snapshot(self):
        journal = self.owner._preservation_journal
        before = self.owner._quarantine_view._budget_binding[7][-1]
        self._reject_sticky(lambda: setattr(journal, "stamp", (*journal.stamp[:-1], journal.stamp[-1] + 1)))
        self.assertEqual(self.owner._quarantine_view._budget_binding[7][-1], before)

    def test_view_deadline_change_is_rejected(self):
        view = self.owner._quarantine_view
        self._reject_sticky(lambda: setattr(view, "until", view.until + 1))

    def test_original_history_deadline_change_is_rejected(self):
        held = self.owner._quarantine_view.inventory.history.held
        self._reject_sticky(lambda: setattr(held, "until", held.until + 1))

    def test_same_selection_rows_mutation_is_rejected(self):
        selection = self.owner.unattributed_preservation
        self._reject_sticky(lambda: selection.rows[0]["research_archive"].__setitem__("sha256", "0" * 64))

    def test_same_selection_sources_mutation_is_rejected(self):
        selection = self.owner.unattributed_preservation
        self._reject_sticky(lambda: selection.sources["guard"].__setitem__("sha256", "0" * 64))

    def test_same_selection_tips_mutation_is_rejected(self):
        selection = self.owner.unattributed_preservation
        self._reject_sticky(lambda: selection.tips[0].__setitem__("sha256", "1" * 64))

    def test_same_selection_maximum_seconds_mutation_is_rejected(self):
        selection = self.owner.unattributed_preservation
        self._reject_sticky(lambda: object.__setattr__(selection, "maximum_seconds", 29))

    def test_recomputed_selection_digest_cannot_replace_original_binding(self):
        selection = self.owner.unattributed_preservation
        saved = self.owner._quarantine_view._budget_binding[8][1]
        def change():
            selection.sources["guard"]["sha256"] = "0" * 64
            digest = policy._sha(policy.custody._encode({"rows": selection.rows, "sources": selection.sources,
                "tips": selection.tips, "maximum_seconds": selection.maximum_seconds}))
            object.__setattr__(selection, "_configuration_digest", digest)
            self.assertNotEqual(digest, saved)
        self._reject_sticky(change)

    def test_original_selection_hard_deadline_cannot_be_extended(self):
        selection = self.owner.unattributed_preservation
        self._reject_sticky(lambda: object.__setattr__(selection, "_bound_deadline", selection.original_hard_until + 1))

    def test_reservation_change_is_rejected_before_poll(self):
        self._reject_sticky(lambda: setattr(self.owner, "reserve", self.owner.reserve + 1))

    def test_catalog_limit_change_is_rejected_before_poll(self):
        self._reject_sticky(lambda: setattr(self.owner, "total_limit", self.owner.total_limit + 1))

    def test_changed_original_producer_token_is_rejected_before_poll(self):
        self._reject_sticky(lambda: self.owner.record.__setitem__("token", "different-original-token"))

    def test_closed_view_does_not_fall_back_to_full(self):
        self.owner._quarantine_view.close()
        self._reject_sticky(lambda: None)

    def test_missing_journal_cannot_trigger_new_observation(self):
        journal = self.owner._preservation_journal
        try:
            row = (self.owner.catalog / (self.owner.key + ".json")).read_bytes()
            self.owner._preservation_journal = None
            with patch.object(policy, "observe_fresh_output_budget", side_effect=AssertionError("no retry")):
                with self.assertRaises(StorageError) as caught:
                    self.owner.poll_budget()
            self.assertIs(self.owner._preservation_error, caught.exception)
            self.assertEqual((self.owner.catalog / (self.owner.key + ".json")).read_bytes(), row)
        finally:
            self.owner._preservation_journal = journal

    def test_current_new_payload_is_sampled_without_journal_append(self):
        journal = self.owner._preservation_journal
        view = self.owner._quarantine_view
        entries = journal.entries
        path = self.owner.output / "new-payload.bin"
        path.write_bytes(b"bounded current new payload")
        original = policy.tree_size
        with patch.object(policy, "tree_size", wraps=original) as sizes:
            self.owner.poll_budget()
        self.assertTrue(any(call.args[0] == self.owner.output for call in sizes.call_args_list))
        self.assertIs(self.owner._quarantine_view, view)
        self.assertEqual(journal.entries, entries)
        self.assertIsNone(self.owner._preservation_error)

    def test_real_guard_loss_is_sticky_without_full_retry_or_finish(self):
        owner = self.owner
        row = (owner.catalog / (owner.key + ".json")).read_bytes()
        journal = owner._preservation_journal
        entries = journal.entries
        owner.guard.release()
        with patch.object(policy, "observe_fresh_output_budget", side_effect=AssertionError("no retry")):
            with self.assertRaises(StorageError) as caught:
                owner.poll_budget()
            self.assertIs(owner._preservation_error, caught.exception)
            with self.assertRaises(StorageError) as refused:
                owner.finish(status="must_not_finish", tree_gone=True)
        self.assertIs(refused.exception.__cause__, caught.exception)
        self.assertEqual(journal.entries, entries)
        self.assertEqual((owner.catalog / (owner.key + ".json")).read_bytes(), row)
        self.assertEqual(owner.record["state"], "active")

    def test_payload_growth_after_usage_sample_keeps_known_budget_error_distinct(self):
        owner, view = self.owner, self.owner._quarantine_view
        journal = owner._preservation_journal
        entries = journal.entries
        payload = owner.output / "growing-current-payload.bin"
        original = view.inventory.checked_usage
        def grow_after_sample():
            result = original()
            payload.write_bytes(b"x" * (owner.reserve + 1))
            return result
        try:
            with patch.object(view.inventory, "checked_usage", side_effect=grow_after_sample), \
                    patch.object(policy, "observe_fresh_output_budget", side_effect=AssertionError("no retry")):
                with self.assertRaises(outputs.OutputBudgetError):
                    owner.poll_budget()
            self.assertIsNone(owner._preservation_error)
            self.assertFalse(view.inventory.failed)
            self.assertEqual(journal.entries, entries)
            self.assertIs(owner._quarantine_view, view)
        finally:
            payload.unlink(missing_ok=True)
        owner.poll_budget()

    def test_current_producer_lifetime_change_is_sticky(self):
        changed = dict(self.owner._quarantine_view.now)
        changed["boot"] = "different-original-boot"
        with patch.object(policy, "lifetime", return_value=changed):
            self._reject_sticky(lambda: None)

    def test_default_check_and_finish_still_use_full_observations(self):
        original = policy.observe_fresh_output_budget
        with patch.object(policy, "observe_fresh_output_budget", wraps=original) as full:
            self.owner.check_budget()
            self.owner.finish(status="fixture_full_finish", tree_gone=True)
        self.assertGreaterEqual(full.call_count, 2)
        self.assertEqual(self.owner.record["state"], "retained")


if __name__ == "__main__":
    unittest.main()
