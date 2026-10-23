"""SOURCE ONLY: reject an unregistered fresh-budget composition candidate.

Main owns import/execution. These closed object fixtures call the production
fresh entry, catalog builder, ledger and one-shot observer. Inventory, historical
decoding and the new-output constructor are explicit fault-injection seams; no
real catalog, owner, archive, journal FD or recovery authority is created.
The opt-in is injected only by this test and is not a supported runtime profile.
"""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_output_unattributed_preservation as policy
from managed_storage import GiB, StorageError


class FreshInitialCandidateTests(unittest.TestCase):
    def probe(self, *, compose, history_error=None, selected_charge=64,
              archive_charge=0, current_amount=2, current_reserve=1):
        catalog = Path(os.path.abspath("unit-fresh-catalog"))
        output = Path(os.path.abspath("unit-fresh-output"))
        source = Path(os.path.abspath("unit-fresh-source"))
        selected = Path(os.path.abspath("unit-fresh-selected"))
        hard_until = time.monotonic() + 60
        selection = object.__new__(policy.UnattributedSelection)
        selection.maximum_seconds, selection.tips = 30, ()
        selection._bound_deadline = hard_until
        guard = types.SimpleNamespace(path=catalog / "catalog.lock", assert_held=Mock())
        rows = [(catalog / "selected.json", {"key": "selected", "reserved_bytes": selected_charge}, selected),
                (catalog / "current.json", {"key": "current", "state": "active",
                 "reserved_bytes": current_reserve}, output)]
        held_objects, observers, events = [], [], []

        class UnitHeld:
            def __init__(self, maximum_seconds, original_until=None):
                self.until = min(time.monotonic() + maximum_seconds,
                                 hard_until if original_until is None else original_until)
                self.trees = []
                self.close = Mock()
                self.tick = Mock()
                self.check = Mock()
                held_objects.append(self)

        def inventory(held, actual_catalog, actual_selection):
            self.assertIs(actual_selection, selection)
            self.assertEqual(actual_catalog, catalog)
            tree = object.__new__(policy._Tree)
            tree.target, tree.held, tree.payload, tree.identity = selected, held, 1, [2, 3]
            held.trees.append(tree)
            return rows, {"selected": {"output": str(selected), "charge_bytes": selected_charge,
                "observed_bytes": 1, "current_observed_identity": [2, 3]}}, archive_charge

        def measured(target, **options):
            self.assertEqual(options, {"allow_disappearing": True})
            events.append(("measure", target))
            return 1 if target == selected else current_amount

        def historical(held, actual_catalog, actual_rows, tips, actual_guard, *, _initial_usage=None):
            self.assertIs(actual_rows, rows)
            history = policy.transitions.TransitionValidation(held, actual_catalog, actual_guard,
                {}, {}, 0, (), (), {"unit-only": {"generation": 8}}, None)
            def check():
                events.append(("history-check", history))
                if history_error is not None:
                    raise history_error
            history.check = check
            # This is the production _history tail, with decoding isolated above.
            if _initial_usage is None:
                history.usage(actual_rows)
            else:
                observers.append(_initial_usage)
                _initial_usage.consume(history, actual_catalog, actual_rows, actual_guard, held)
            history.check()
            return history

        original_builder = policy._new_catalog_view
        seen_profiles = []
        def builder(*args, **kwargs):
            self.assertNotIn("compose_initial_usage", kwargs)
            seen_profiles.append(compose)
            if compose:
                kwargs["compose_initial_usage"] = True
            return original_builder(*args, **kwargs)

        journal = object.__new__(policy.legacy._EvidenceJournal)
        journal.append = Mock()
        constructor_error = StorageError("unit original new-output constructor failure")
        with ExitStack() as stack:
            stack.enter_context(patch.object(policy, "checked", side_effect=lambda value: value))
            stack.enter_context(patch.object(policy, "_Held", UnitHeld))
            stack.enter_context(patch.object(policy.custody, "_Held", UnitHeld))
            stack.enter_context(patch.object(policy, "_inventory", side_effect=inventory))
            stack.enter_context(patch.object(policy.transitions, "_history", side_effect=historical))
            stack.enter_context(patch.object(policy, "_rows", return_value=rows))
            stack.enter_context(patch.object(policy, "tree_size", side_effect=measured))
            stack.enter_context(patch.object(policy.custody, "tree_size", side_effect=measured))
            stack.enter_context(patch.object(policy.CatalogView, "selection_document", return_value={}))
            stack.enter_context(patch.object(policy, "_new_catalog_view", side_effect=builder))
            constructor = stack.enter_context(patch.object(policy, "FreshOutputView",
                                                            side_effect=constructor_error))
            with self.assertRaises(StorageError) as caught:
                policy.observe_fresh_output_budget(catalog, selection, guard, output=output,
                    source=source, producer_record={"unit-only": True}, journal=journal,
                    reserve=current_reserve, total_limit=18 * GiB)
        journal.append.assert_not_called()
        self.assertEqual(seen_profiles, [compose])
        self.assertEqual(selection.maximum_seconds, 30)
        self.assertEqual(selection.original_hard_until, hard_until)
        self.assertEqual(len(held_objects), 2)
        for held in held_objects:
            self.assertLessEqual(held.until, hard_until)
            held.close.assert_called_once_with(caught.exception)
        guard.assert_held.assert_called()
        return caught.exception, constructor_error, constructor, observers, events

    def test_history_failure_and_reservation_failure_reverse_first_error_under_candidate(self):
        original = StorageError("unit original history check failure")
        legacy, _, old_constructor, _, _ = self.probe(compose=False, history_error=original)
        candidate, _, new_constructor, observers, events = self.probe(compose=True,
                                                                     history_error=original)
        self.assertEqual(str(legacy), "managed output exceeds reservation")
        self.assertIs(candidate, original)
        old_constructor.assert_not_called()
        new_constructor.assert_not_called()
        self.assertFalse(any(event[0] == "measure" for event in events))
        self.assertEqual(len(observers), 1)
        self.assertTrue(observers[0].consumed)
        self.assertIsNone(observers[0].consumer)
        with self.assertRaisesRegex(StorageError, "already consumed"):
            observer = observers[0]
            observer.consume(observer.history, observer.catalog, observer.rows, observer.guard, observer.held)

    def test_conservative_failure_would_precede_original_constructor_failure(self):
        args = dict(selected_charge=18 * GiB, archive_charge=1,
                    current_amount=1, current_reserve=64)
        legacy, primary, old_constructor, _, _ = self.probe(compose=False, **args)
        candidate, _, new_constructor, _, _ = self.probe(compose=True, **args)
        self.assertIs(legacy, primary)
        old_constructor.assert_called_once()
        self.assertIn("conservative corpus/archives/registry/new reservations exceed18GiB", str(candidate))
        new_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
