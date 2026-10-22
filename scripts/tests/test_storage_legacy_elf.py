"""Small synthetic ELF fixtures; no fixture is an executable or runtime proof."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from managed_storage import StorageError
import storage_legacy_elf as legacy


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fd_pin(path):
    info = path.stat()
    return {**pin(path), "identity": [info.st_dev, info.st_ino], "mtime_ns": info.st_mtime_ns}


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)
    return pin(path)


class UnsupportedTests(unittest.TestCase):
    def test_windows_is_typed_unsupported_before_opening_inputs(self):
        with patch.object(legacy.sys, "platform", "win32"), patch.object(legacy.core, "_Inputs") as inputs:
            with self.assertRaises(legacy.LegacyElfUnsupported):
                legacy.maintain_legacy_elf({})
            inputs.assert_not_called()


@unittest.skipUnless(sys.platform == "linux" and getattr(os, "geteuid", lambda: -1)() == 0,
                     "UID0 native Linux FD maintenance fixtures required; skip is not acceptance")
class LegacyElfTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "validation-assets" / legacy.ROOT_NAME
        self.build = self.base / "build" / "managed-v1"
        self.catalog = self.base / "storage" / "outputs-v1"
        self.maintenance = self.base / "maintenance"
        for path in (self.root / "inputs/binaries", self.root / "inputs/original-binaries-build-141",
                     self.build, self.catalog, self.maintenance):
            path.mkdir(parents=True, mode=0o700)
        self.candidate = self.root / "inputs/binaries/rz-rovent-native-stripped-202"
        self.keeper = self.root / "inputs/binaries/rz-rovent-native-stripped-209"
        self.candidate.write_bytes(b"\x7fELF synthetic duplicate never executed")
        self.keeper.write_bytes(self.candidate.read_bytes())
        self.summary_path = self.maintenance / "summary.json"
        self.summary = self.make_summary()
        summary_pin = dump(self.summary_path, self.summary)
        self.plan_path = self.maintenance / "plan.json"
        self.plan = {"schema": legacy.PLAN_SCHEMA,
                     "root": {"path": str(self.root), "identity": list(legacy.core._inode(self.root.stat()))},
                     "catalog": {"path": str(self.catalog), "identity": list(legacy.core._inode(self.catalog.stat()))},
                     "journal": {"path": str(self.maintenance / "legacy-elf-20261022-001"),
                                 "reserve_bytes": 256 << 10, "directory_limit_bytes": 1 << 20},
                     "files": [{"mode": "duplicate", "candidate": fd_pin(self.candidate),
                                "keeper": fd_pin(self.keeper), "provenance": summary_pin}]}
        for target, value in (("canonical_root", self.build), ("canonical_catalog", self.catalog)):
            patched = patch.object(legacy, target, return_value=value)
            patched.start()
            self.addCleanup(patched.stop)
        # Success fixtures isolate unlink/custody/journal mechanics. The actual
        # census has a separate real-reference case and is reused unchanged.
        self.census = patch.object(legacy.core, "_process_census", return_value={
            "complete": True, "namespace": {"scope": "synthetic_fixture_only"}})
        self.census.start()
        self.addCleanup(self.census.stop)

    def make_summary(self):
        originals, derivatives, pairs = [], [], []
        for name in sorted(legacy.ELF_NAMES):
            original_path = self.root / "inputs/original-binaries-build-141" / name
            raw = b"\x7fELF original fixture " + name.encode()
            original_pin = {"path": str(original_path), "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest()}
            original = {**original_pin, "source": legacy.OBSOLETE_SOURCE,
                        "dependency_file": {"path": str(original_path) + ".d", "bytes": 1, "sha256": "1" * 64}}
            originals.append(original)
            derivatives.append({"path": str(self.root / "inputs/binaries" / (name + "-stripped-142")),
                                "bytes": 20, "sha256": "2" * 64, "original": original,
                                "source": legacy.OBSOLETE_SOURCE, "transformation": "recorded GNU strip"})
            if name == "rz-rovent-native":
                candidate, keeper = pin(self.candidate), pin(self.keeper)
            else:
                candidate = {"path": str(self.root / "inputs/binaries" / (name + "-stripped-202")),
                             "bytes": 20, "sha256": "3" * 64}
                keeper = {**candidate, "path": str(self.root / "inputs/binaries" / (name + "-stripped-209"))}
            pairs.append({"candidate": candidate, "keeper": keeper, "source": legacy.DUPLICATE_SOURCE})
        reports = {
            "native-functional-build-141.json": (37859, "fd6965310452eff2f9dcc788db2ca2cee8159d6883e6e29ce88bfad863c3c278"),
            "functional-binaries-stripped-142.json": (5039, "8feb25522192539d9468211b2249e98f86f0ea35e9bc1b6798f56bf7c35ca4c0"),
            "functional-binaries-stripped-202.json": (5039, "96b273e952690ce91d17d2e3a9bb61b52db9e5474bc89ae781227a0a657748d1"),
            "functional-binaries-stripped-209.json": (5039, "4daf0ddd417df6c044951d009ea68044b4309a8c46a5342f599d3dd68a857b07")}
        return {"schema": legacy.PROVENANCE_SCHEMA, "authority": "fixture, not execution authority",
                "current_required_inputs_retired": False, "duplicate_pairs": pairs, "gpu_execution": False,
                "not_authority_for_registered_outputs_or_unknown_physical_material": True,
                "old_results_promoted_to_current_acceptance": False, "optimizer_updates": 0,
                "purpose": "user_authorized_obsolete_code_and_exact_duplicate_retirement",
                "provenance": {name: {"path": "/preserved-reports/" + name, "bytes": amount, "sha256": digest}
                               for name, (amount, digest) in reports.items()},
                "obsolete_build": {"source": legacy.OBSOLETE_SOURCE, "status": "built_not_execution_admitted",
                    "toolchain": {"rustc": {"argv": ["rustc", "-vV"], "output": "recorded rustc"},
                                  "cargo": {"argv": ["cargo", "-Vv"], "output": "recorded cargo"},
                                  "executables": [{"path": "/preserved-toolchain/rustc", "bytes": 1, "sha256": "4" * 64},
                                                  {"path": "/preserved-toolchain/cargo", "bytes": 1, "sha256": "5" * 64}],
                                  "sysroot": "/preserved-toolchain"},
                    "build_argv": [["cargo", "build", "--locked", "--bin", "rz-rovent-native"]],
                    "original_build_target": "/original-target", "original_resource_policy": {"cpu_quota_percent": 200},
                    "artifact_retention": "recorded original preservation", "files": originals,
                    "retained_stripped_derivatives": derivatives, "exact_byte_rebuild_guaranteed": False,
                    "rebuild_description": "Recorded source/toolchain/locked command; exact bytes not promised."}}

    def invoke(self, apply=False):
        return legacy.maintain_legacy_elf(dump(self.plan_path, self.plan), apply=apply, seconds=30)

    def refuse(self, pattern):
        before = self.candidate.read_bytes()
        with self.assertRaisesRegex(StorageError, pattern):
            self.invoke(True)
        self.assertEqual(self.candidate.read_bytes(), before)

    def add_row(self, target, state="active"):
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()
        row = {"schema": legacy.OUTPUT_SCHEMA, "key": key, "output": str(target), "state": state,
               "identity": list(legacy.core._inode(target.stat()))}
        path = self.catalog / (key + ".json")
        dump(path, row)
        dump(target / ".rovezero-output.json", row)
        return path

    def test_default_audit_leaves_every_elf_summary_and_no_journal(self):
        before = {path: path.read_bytes() for path in (self.candidate, self.keeper, self.summary_path)}
        result = self.invoke()
        self.assertEqual(result["status"], "LEGACY_ELF_AUDITED")
        self.assertFalse(result["deletion_performed"])
        self.assertEqual(result["logical_removed_bytes"], 0)
        self.assertFalse(Path(self.plan["journal"]["path"]).exists())
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_apply_exact_duplicate_preserves_keeper_and_durable_summary(self):
        before, amount = self.summary_path.read_bytes(), self.candidate.stat().st_size
        result = self.invoke(True)
        self.assertFalse(self.candidate.exists())
        self.assertEqual(self.keeper.read_bytes(), b"\x7fELF synthetic duplicate never executed")
        self.assertEqual(result["logical_removed_bytes"], amount)
        self.assertEqual(result["physical_space_reclaimed"], "unknown")
        self.assertFalse(result["owner_lease_or_adoption_issued"])
        self.assertEqual(pin(Path(result["journal"]["path"])), result["journal"])
        events = [json.loads(line) for line in Path(result["journal"]["path"]).read_bytes().splitlines()]
        self.assertEqual(events[0]["provenance"][0]["raw_utf8"].encode(), before)
        self.assertEqual([event["event"] for event in events],
                         ["preflight_complete", "unlink_intent", "unlinked_directory_fsynced", "complete"])

    def test_obsolete_preserves_provenance_without_exact_restore_claim(self):
        path = self.root / "inputs/original-binaries-build-141/rz-rovent-native"
        path.write_bytes(b"\x7fELF original fixture rz-rovent-native")
        self.plan["files"] = [{"mode": "obsolete", "candidate": fd_pin(path), "keeper": None,
                               "provenance": pin(self.summary_path)}]
        result = self.invoke(True)
        self.assertFalse(path.exists())
        self.assertFalse(result["byte_exact_obsolete_restoration_attested"])
        self.assertTrue(self.candidate.exists())
        self.assertTrue(self.keeper.exists())

    def test_build_owner_even_empty_is_never_recovered_or_replaced(self):
        owner = self.build / "owner.json"
        owner.write_bytes(b"original unknown owner")
        self.refuse("owner is present")
        self.assertEqual(owner.read_bytes(), b"original unknown owner")

    def test_registered_ancestor_scope_is_refused_even_retained(self):
        self.add_row(self.candidate.parent, state="retained")
        self.refuse("registered/protected.*scope")

    def test_unrelated_active_row_and_registry_are_preserved_without_adoption(self):
        row = self.add_row(self.base / "unrelated-output")
        registry = self.catalog.with_name(self.catalog.name + ".interrupted-unbound-custody-v1")
        registry.mkdir(mode=0o700)
        sentinel = registry / "unknown-custody"
        sentinel.write_bytes(b"must never be interpreted or modified")
        before = row.read_bytes(), sentinel.read_bytes()
        self.invoke()
        self.assertEqual((row.read_bytes(), sentinel.read_bytes()), before)

    def test_unapproved_model_name_is_never_an_elf_candidate(self):
        model = self.root / "inputs/binaries/model-stripped-202"
        model.write_bytes(self.candidate.read_bytes())
        self.plan["files"][0]["candidate"] = fd_pin(model)
        self.refuse("explicit legacy build-202")
        self.assertTrue(model.exists())

    def test_keeper_different_stripped_bytes_is_refused(self):
        self.keeper.write_bytes(b"\x7fELF different stripped bytes")
        self.plan["files"][0]["keeper"] = fd_pin(self.keeper)
        self.refuse("identical ELF bytes")

    def test_same_bytes_inode_replacement_does_not_match_plan_fd_identity(self):
        before = self.candidate.read_bytes()
        swap = self.candidate.with_name("temporary-replacement")
        swap.write_bytes(before)
        os.replace(swap, self.candidate)
        self.refuse("FD identity/mtime changed")

    def test_hardlink_and_symlink_are_refused_without_unlink(self):
        alias = self.base / "hardlink"
        os.link(self.candidate, alias)
        self.refuse("FD/type/identity")
        alias.unlink()
        original = self.candidate.with_name("preserved-original")
        self.candidate.rename(original)
        self.candidate.symlink_to(original)
        with self.assertRaisesRegex(StorageError, "symlink|junction|alias"):
            self.invoke(True)
        self.assertTrue(original.exists())

    def test_unknown_census_refuses_before_journal_or_unlink(self):
        with patch.object(legacy.core, "_process_census", side_effect=StorageError("fixture census unknown")):
            self.refuse("fixture census unknown")
        self.assertFalse(Path(self.plan["journal"]["path"]).exists())

    def test_real_extra_held_candidate_fd_is_in_use(self):
        self.census.stop()
        with self.candidate.open("rb"):
            self.refuse("census|in use|open.*FD|referenc")

    def test_summary_wrong_source_is_not_obsolete_permission(self):
        self.summary["obsolete_build"]["source"] = "0" * 40
        self.plan["files"][0]["provenance"] = dump(self.summary_path, self.summary)
        self.refuse("original build-141 source")

    def test_external_journal_budget_and_existing_attempt_refuse_before_unlink(self):
        self.plan["journal"]["directory_limit_bytes"] = 256 << 10
        self.refuse("directory budget")
        self.plan["journal"]["directory_limit_bytes"] = 1 << 20
        path = Path(self.plan["journal"]["path"])
        path.write_bytes(b"original partial attempt")
        with self.assertRaises(FileExistsError):
            self.invoke(True)
        self.assertEqual(path.read_bytes(), b"original partial attempt")
        self.assertTrue(self.candidate.exists())

    def test_partial_unlink_preserves_first_error_and_actual_prefix(self):
        second = self.root / "inputs/original-binaries-build-141/rz-rovent-native"
        second.write_bytes(b"\x7fELF original fixture rz-rovent-native")
        self.plan["files"].append({"mode": "obsolete", "candidate": fd_pin(second), "keeper": None,
                                   "provenance": pin(self.summary_path)})
        original_unlink = os.unlink
        first = OSError("fixture second unlink failure")
        def unlink(name, *args, **kwargs):
            if name == second.name:
                raise first
            return original_unlink(name, *args, **kwargs)
        original_append, original_close = legacy.core._Journal.append, legacy.core._Journal.close
        def append(journal, event):
            if event["event"] == "failed_or_partial_preserved":
                raise OSError("fixture failure journal append also failed")
            return original_append(journal, event)
        def close(journal):
            original_close(journal)
            raise OSError("fixture journal close also failed")
        with patch.object(legacy.os, "unlink", side_effect=unlink), \
                patch.object(legacy.core._Journal, "append", new=append), \
                patch.object(legacy.core._Journal, "close", new=close):
            with self.assertRaises(OSError) as caught:
                self.invoke(True)
        self.assertIs(caught.exception, first)
        evidence = caught.exception.legacy_elf_maintenance
        self.assertEqual(evidence["retired_files"], [str(self.candidate)])
        self.assertEqual(evidence["logical_removed_bytes"], self.plan["files"][0]["candidate"]["bytes"])
        self.assertTrue(evidence["partial"])
        self.assertTrue(evidence["last_attempt"]["unlink_attempted"])
        self.assertFalse(evidence["last_attempt"]["unlink_returned"])
        self.assertFalse(self.candidate.exists())
        self.assertTrue(second.exists())
        self.assertTrue(self.keeper.exists())
        self.assertEqual(pin(Path(evidence["journal"]["path"])), evidence["journal"])
        self.assertTrue(any("journal append" in note for note in first.__notes__))
        self.assertTrue(any("journal close" in note for note in first.__notes__))

    def test_close_failure_after_complete_never_returns_success(self):
        original_close = legacy.core._Journal.close
        def close(journal):
            original_close(journal)
            raise OSError("fixture completed journal close failed")
        with patch.object(legacy.core._Journal, "close", new=close):
            with self.assertRaisesRegex(OSError, "completed journal close failed") as caught:
                self.invoke(True)
        evidence = caught.exception.legacy_elf_maintenance
        self.assertFalse(self.candidate.exists())
        self.assertTrue(evidence["partial"])
        self.assertEqual(evidence["retired_files"], [str(self.candidate)])


if __name__ == "__main__":
    unittest.main()
