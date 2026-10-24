"""Pure manifest/sequence fixtures. No mounts, owners or actual caller process."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).with_name("control_input_snapshot.py")
SPEC = importlib.util.spec_from_file_location("control_input_snapshot_fixture_subject", SOURCE)
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


R = "/mnt/c/public-reports/rovent-unit"
W = "/mnt/c/public-worktree"
WRAPPER = W + "/scripts/control_input_snapshot.py"
MANIFEST = R + "/snapshot-manifest.json"


def pin(path, size=1):
    return {"path": path, "bytes": size, "sha256": hashlib.sha256(b"x").hexdigest()}


def manifest():
    caller, request = pin(R + "/run-original-source-003.py"), pin(R + "/original-request-004.json")
    argv = ["/usr/bin/python3", "-I", "-S", "-B", caller["path"], "--action", "run",
        "--self-pin", caller["path"], str(caller["bytes"]), caller["sha256"],
        "--request-pin", request["path"], str(request["bytes"]), request["sha256"],
        "--result", R + "/new-original-result.json"]
    return {"schema": subject.SCHEMA, "profile": subject.PROFILE, "report_root": R,
        "tooling_root": W, "setup_receipt_path": R + "/new-setup-receipt.json",
        "maximum_bytes": subject.LIMIT, "setup_seconds": 60,
        "inputs": [{"kind": "report_python", "pin": caller}, {"kind": "report_json", "pin": request}],
        "caller_argv": argv}


def plan(value=None, literal=None):
    value = manifest() if value is None else value
    return subject.validate_manifest(value, value["caller_argv"] if literal is None else literal,
        manifest_path=MANIFEST, wrapper_path=WRAPPER)


class FakeOps:
    """Records only abstract syscall boundaries; never calls the native class."""
    def __init__(self, fail_at=None, failure=None, cleanup_error=None):
        self.started, self.until = 10.0, 130.0
        self.events = []
        self.fail_at = fail_at
        self.failure = RuntimeError("original fake syscall failure") if failure is None else failure
        self.cleanup_error = cleanup_error
        self.counter = 0

    def event(self, name):
        self.events.append(name)
        if name == self.fail_at:
            raise self.failure

    def tick(self):
        self.event("tick")

    def environment(self):
        self.event("environment")
        return {"synthetic": True, "physical_authority": False}

    def make_tmpfs(self):
        self.event("make_tmpfs")
        return "/tmp/synthetic-private-snapshot"

    def open_source(self, rawpin):
        self.event("open:" + rawpin["path"])
        self.counter += 1
        return {"fd": self.counter, "path": rawpin["path"], "stamp": tuple(range(9))}

    def copy_source(self, directory, index, item, rawpin):
        self.event("copy:" + rawpin["path"])
        return {"path": directory + "/input-%03d" % index, "stamp": tuple(range(10, 19))}

    def recheck_source(self, item):
        self.event("recheck:" + item["path"])

    def close_source(self, item):
        self.event("close:" + item["path"])

    def bind_readonly(self, copied, target):
        self.event("bind:" + target)
        return {"logical_path": target, "readonly": True, "synthetic": True}

    def seal_tmpfs(self, directory):
        self.event("seal")
        return {"readonly": True, "synthetic": True}

    def verify_binding(self, copied, target):
        self.event("verify:" + target)

    def close_setup_fds(self):
        self.event("close_setup_fds")
        return []

    def cleanup(self):
        self.event("cleanup")
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return {"errors": [], "setup_scope_closed": True, "product_physical_completion_attested": False}


class FakeDirectoryCalls:
    def __init__(self):
        self.nodes = {"/": self.info(1), "/reports": self.info(2), "/reports/unit": self.info(3),
            "/elsewhere": self.info(4)}
        self.next_fd = 100
        self.held, self.events = {}, []

    @staticmethod
    def info(inode, mode=stat.S_IFDIR | 0o755):
        return SimpleNamespace(st_dev=1, st_ino=inode, st_mode=mode, st_nlink=2, st_uid=0, st_gid=0)

    def open_root(self):
        return self.open_path("/")

    def open_path(self, path):
        self.events.append(("open_directory", path))
        node = self.nodes[path]
        if not stat.S_ISDIR(node.st_mode):
            raise subject.SnapshotError("fake O_NOFOLLOW directory rejection")
        self.next_fd += 1
        self.held[self.next_fd] = (path, node)
        return self.next_fd

    def open_directory(self, parent, name):
        return self.open_path(self.held[parent][0].rstrip("/") + "/" + name)

    def root_stat(self):
        return self.nodes["/"]

    def stat_at(self, parent, name):
        return self.nodes[self.held[parent][0].rstrip("/") + "/" + name]

    def fstat(self, fd):
        return self.held[fd][1]

    def close(self, fd):
        self.events.append(("close", fd))
        del self.held[fd]


class ManifestTests(unittest.TestCase):
    def test_literal_original_argv_and_both_raw3_flags_are_preserved(self):
        value = manifest()
        result = plan(value)
        self.assertEqual(tuple(value["caller_argv"]), result.caller_argv)
        self.assertEqual(1, result.caller_argv.count("--self-pin"))
        self.assertEqual(1, result.caller_argv.count("--request-pin"))
        self.assertEqual(2, len(result.inputs))
        self.assertEqual(60, result.setup_seconds)

    def test_original_flag_duplicate_or_literal_rewrite_is_rejected(self):
        value = manifest()
        changed = value["caller_argv"] + ["--self-pin", "extra", "1", "f" * 64]
        with self.assertRaises(subject.SnapshotError):
            plan(value, literal=changed)
        value["caller_argv"] = changed
        with self.assertRaises(subject.SnapshotError):
            plan(value)

    def test_both_explicit_python_paths_are_preserved_without_aliasing(self):
        for binary in ("/usr/bin/python3", "/usr/bin/python3.12"):
            value = manifest()
            value["caller_argv"][0] = binary
            self.assertEqual(binary, plan(value).caller_argv[0])
        for binary in ("/usr/bin/python3.13", "/usr/bin/env", "/bin/sh", "python3"):
            value = manifest()
            value["caller_argv"][0] = binary
            with self.assertRaises(subject.SnapshotError):
                plan(value)

    def test_exact_mutable_names_and_heavy_or_secret_paths_are_rejected(self):
        for path in (R + "/owner.json", R + "/.rovezero-output.json", R + "/catalog.json",
                R + "/.rovezero-preservation-session.jsonl", R + "/models/weights.json",
                R + "/.env.json", R + "/credential-public.json", R + "/research.zip"):
            with self.subTest(path=path):
                value = manifest()
                value["inputs"].append({"kind": "report_json", "pin": pin(path)})
                with self.assertRaises(subject.SnapshotError):
                    plan(value)

    def test_explicit_historical_owner_readback_is_public_json(self):
        value = manifest()
        value["inputs"].append({"kind": "report_json", "pin": pin(
            R + "/rovent-n05-005-postfailure-owner-readback-Main-20261023-001.json")})
        self.assertEqual(3, len(plan(value).inputs))

    def test_only_public_git_pointer_and_exact_tooling9_are_supported(self):
        value = manifest()
        value["inputs"].append({"kind": "tooling_git_pointer", "pin": pin(W + "/.git")})
        for name in sorted(subject.SOURCE9):
            value["inputs"].append({"kind": "tooling_source9", "pin": pin(W + "/scripts/" + name)})
        self.assertEqual(12, len(plan(value).inputs))
        for path, kind in ((W + "/.git/HEAD", "tooling_git_pointer"),
                (W + "/scripts/other.py", "tooling_source9"), (W + "/scripts/storage_guard.py", "report_python")):
            invalid = manifest()
            invalid["inputs"].append({"kind": kind, "pin": pin(path)})
            with self.assertRaises(subject.SnapshotError):
                plan(invalid)

    def test_total16MiB_boundary_and_duplicate_pin_rejected_before_io(self):
        value = manifest()
        for index in range(16):
            size = subject.FILE_LIMIT if index < 15 else subject.FILE_LIMIT - 2
            value["inputs"].append({"kind": "report_json", "pin": pin(R + "/control-%02d.json" % index, size)})
        self.assertEqual(subject.LIMIT, sum(item.bytes for item in plan(value).inputs))
        value["inputs"][-1]["pin"]["bytes"] += 1
        with self.assertRaises(subject.SnapshotError):
            plan(value)
        value = manifest()
        value["inputs"].append(deepcopy(value["inputs"][0]))
        with self.assertRaises(subject.SnapshotError):
            plan(value)

    def test_receipt_and_original_result_remain_writable_noninputs(self):
        value = manifest()
        value["setup_receipt_path"] = value["inputs"][1]["pin"]["path"]
        with self.assertRaises(subject.SnapshotError):
            plan(value)
        value = manifest()
        value["caller_argv"][-1] = value["inputs"][1]["pin"]["path"]
        with self.assertRaises(subject.SnapshotError):
            plan(value)

    def test_duplicate_manifest_fields_and_recursive_unknown_fields_rejected(self):
        with self.assertRaises(subject.SnapshotError):
            subject.strict_json(b'{"schema":1,"schema":2}')
        value = manifest()
        value["collect_from_json_recursively"] = True
        with self.assertRaises(subject.SnapshotError):
            plan(value)


class IsolationTests(unittest.TestCase):
    def proof(self, **overrides):
        rows = subject.mount_rows("10 1 0:1 / / rw - ext4 /dev/fake rw\n"
            "11 10 0:1 /tmp/systemd-private-example.service-abc/tmp /tmp rw - ext4 /dev/fake rw\n")
        data = {"uid": 0, "gid": 0, "cap_effective": 1 << 21, "current_namespace": "mnt:[2]",
            "init_namespace": "mnt:[1]", "rows": rows,
            "tmp_info": SimpleNamespace(st_mode=stat.S_IFDIR | 0o1777, st_uid=0, st_gid=0)}
        data.update(overrides)
        return subject.isolation_proof(**data)

    def test_actual_private_tmp_geometry_is_required_not_a_boolean_claim(self):
        self.assertTrue(self.proof()["propagation_private"])
        for change in ({"cap_effective": 0}, {"uid": 1000}, {"current_namespace": "mnt:[1]"},
                {"tmp_info": SimpleNamespace(st_mode=stat.S_IFDIR | 0o1777, st_uid=1000, st_gid=1000)},
                {"rows": subject.mount_rows("10 1 0:1 / / rw - ext4 /dev/fake rw\n")},
                {"rows": subject.mount_rows("11 1 0:1 /tmp/systemd-private-unit/tmp /tmp rw shared:2 - ext4 /dev/fake rw\n")}):
            with self.subTest(change=change):
                with self.assertRaises(subject.SnapshotError):
                    self.proof(**change)


class ParentCustodyTests(unittest.TestCase):
    def test_parent_replacement_is_rejected_before_any_target_operation(self):
        calls = FakeDirectoryCalls()
        parent = subject.ParentChain("/reports/unit", calls)
        calls.nodes["/reports"] = calls.info(900)
        target_operations = []
        try:
            with self.assertRaises(subject.SnapshotError):
                parent.recheck()
                target_operations.append("mount_or_publish")
            self.assertEqual([], target_operations)
        finally:
            self.assertEqual([], parent.close())
        self.assertEqual({}, calls.held)

    def test_middle_symlink_is_not_followed_and_already_opened_root_is_closed(self):
        calls = FakeDirectoryCalls()
        calls.nodes["/reports"] = calls.info(901, stat.S_IFLNK | 0o777)
        with self.assertRaises(subject.SnapshotError):
            subject.ParentChain("/reports/unit", calls)
        self.assertEqual({}, calls.held)
        self.assertNotIn(("open_directory", "/reports/unit"), calls.events)

    def test_receipt_cannot_use_another_parent_capability(self):
        calls = FakeDirectoryCalls()
        wrong_parent = subject.ParentChain("/elsewhere", calls)
        try:
            with self.assertRaises(subject.SnapshotError):
                subject.write_receipt("/reports/unit/new.json", {"synthetic": True}, parent=wrong_parent)
            self.assertEqual(2, len(calls.held))  # No file-open/write path was reached.
        finally:
            self.assertEqual([], wrong_parent.close())

    def test_own_directory_linkcount_change_does_not_hide_identity_replacement(self):
        calls = FakeDirectoryCalls()
        parent = subject.ParentChain("/reports/unit", calls)
        calls.nodes["/reports/unit"].st_nlink += 1
        parent.recheck()  # Own mkdir is allowed; the actual held inode stays fixed.
        calls.nodes["/reports/unit"] = calls.info(902)
        try:
            with self.assertRaises(subject.SnapshotError):
                parent.recheck()
        finally:
            self.assertEqual([], parent.close())


class ReceiptOwnerTests(unittest.TestCase):
    def parent(self, uid=1000, gid=1000):
        calls = FakeDirectoryCalls()
        calls.nodes["/reports/unit"].st_uid = uid
        calls.nodes["/reports/unit"].st_gid = gid
        calls.nodes["/reports/unit"].st_mode = stat.S_IFDIR | 0o777
        return subject.ParentChain("/reports/unit", calls), calls

    @staticmethod
    def receipt(uid=1000, gid=1000):
        return SimpleNamespace(st_dev=1, st_ino=500, st_mode=stat.S_IFREG | 0o777,
            st_nlink=1, st_uid=uid, st_gid=gid)

    def test_windows_receipt_matches_held_parent_owner_and_records_actual_mode(self):
        parent, calls = self.parent()
        try:
            observed = subject.receipt_owner_observation("/reports/unit/new.json", parent,
                calls.fstat(parent.fd), self.receipt())
            self.assertEqual(subject.RECEIPT_OWNER_POLICY, observed["policy"])
            self.assertEqual([1000, 1000], observed["parent_uid_gid"])
            self.assertEqual([1000, 1000], observed["receipt_uid_gid"])
            self.assertEqual("0777", observed["receipt_mode"])
            self.assertFalse(observed["arbitrary_owner_fallback_permitted"])
            self.assertFalse(observed["private_image_owner_policy_changed"])
        finally:
            self.assertEqual([], parent.close())

    def test_root_or_other_owner_is_not_an_alternative_to_the_held_parent_owner(self):
        parent, calls = self.parent()
        try:
            for uid, gid in ((0, 0), (1001, 1000), (1000, 1001)):
                with self.subTest(uid=uid, gid=gid):
                    with self.assertRaises(subject.SnapshotError):
                        subject.receipt_owner_observation("/reports/unit/new.json", parent,
                            calls.fstat(parent.fd), self.receipt(uid, gid))
            calls.nodes["/reports/unit"].st_uid = 1001
            with self.assertRaises(subject.SnapshotError):
                subject.receipt_owner_observation("/reports/unit/new.json", parent,
                    calls.fstat(parent.fd), self.receipt(1001, 1000))
        finally:
            self.assertEqual([], parent.close())

    def test_root_parent_still_requires_its_exact_owner_and_single_regular_receipt(self):
        parent, calls = self.parent(0, 0)
        try:
            self.assertEqual([0, 0], subject.receipt_owner_observation("/reports/unit/new.json", parent,
                calls.fstat(parent.fd), self.receipt(0, 0))["receipt_uid_gid"])
            for changed in (self.receipt(), self.receipt(0, 0)):
                if changed.st_uid == 0:
                    changed.st_nlink = 2
                with self.assertRaises(subject.SnapshotError):
                    subject.receipt_owner_observation("/reports/unit/new.json", parent,
                        calls.fstat(parent.fd), changed)
        finally:
            self.assertEqual([], parent.close())


class SetupSequenceTests(unittest.TestCase):
    def test_all_raw_readback_and_original_rechecks_precede_first_bind(self):
        ops, current = FakeOps(), plan()
        result = subject.prepare_snapshot(current, ops)
        first_bind = min(i for i, row in enumerate(ops.events) if row.startswith("bind:"))
        for item in current.inputs:
            self.assertLess(ops.events.index("copy:" + item.path), first_bind)
            self.assertLess(ops.events.index("recheck:" + item.path), first_bind)
            self.assertGreater(ops.events.index("verify:" + item.path), ops.events.index("seal"))
            self.assertGreater(ops.events.index("close:" + item.path), ops.events.index("verify:" + item.path))
        self.assertEqual(70.0, ops.until)  # Original start10 +60, not now+60.
        self.assertEqual(list(current.caller_argv), result["caller_argv"])
        self.assertFalse(result["normal_finish_or_product_authority_issued"])
        self.assertFalse(result["physical_completion_attested"])
        self.assertNotIn("cleanup", ops.events)
        self.assertIn("pending", result["exec_intent"])

    def test_setup_failure_preserves_exact_primary_and_attempts_only_own_cleanup(self):
        current = plan()
        for stage in ("environment", "make_tmpfs", "copy:" + current.inputs[0].path,
                "bind:" + current.inputs[0].path, "seal", "verify:" + current.inputs[1].path):
            with self.subTest(stage=stage):
                primary = KeyboardInterrupt("original setup failure")
                ops = FakeOps(stage, primary)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    subject.prepare_snapshot(current, ops)
                self.assertIs(primary, caught.exception)
                self.assertEqual(1, ops.events.count("cleanup"))
                self.assertFalse(primary.snapshot_cleanup["product_physical_completion_attested"])

    def test_cleanup_secondary_cannot_replace_primary_or_create_closure(self):
        primary = RuntimeError("copy failed")
        ops = FakeOps("make_tmpfs", primary, cleanup_error=SystemExit("cleanup interrupted"))
        with self.assertRaises(RuntimeError) as caught:
            subject.prepare_snapshot(plan(), ops)
        self.assertIs(primary, caught.exception)
        self.assertFalse(primary.snapshot_cleanup["setup_scope_closed"])
        self.assertIn("SystemExit", primary.snapshot_cleanup["errors"][0])


if __name__ == "__main__":
    unittest.main()
