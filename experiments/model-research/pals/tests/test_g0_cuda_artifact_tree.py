"""NN·payload 읽기 없이 G0 용량 관측과 일반 경로의 엄격함을 검사한다."""
import builtins
from contextlib import contextmanager
import copy
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch


@contextmanager
def no_neural_imports():
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime", "scipy", "transformers"}:
            raise AssertionError("용량 관측이 신경망 framework를 import함: " + name)
        level = kwargs.get("level", args[3] if len(args) > 3 else 0)
        if name == "rz_pals_model.model" or (level and name == "model"):
            raise AssertionError("용량 관측이 모델을 import함")
        return original(name, *args, **kwargs)
    with patch("builtins.__import__", checked):
        yield


@unittest.skipUnless(sys.platform == "linux", "CUDA 파일 관측은 Linux 전용이며 Windows에서는 실제 미실행")
class G0CudaArtifactTreeTests(unittest.TestCase):
    def setUp(self):
        self.guard = no_neural_imports(); self.guard.__enter__()
        self.addCleanup(self.guard.__exit__, None, None, None)
        self.module = importlib.import_module("rz_pals_model.g0_cuda_artifact_tree")
        self.generic = importlib.import_module("rz_pals_model.generation_training_cli")

    @contextmanager
    def fixture(self, *, aliases=True):
        with tempfile.TemporaryDirectory(prefix="rz-g0-cuda-tree-text-fixture-") as temporary:
            root = Path(temporary).resolve()
            parent = root / "runtime-cache"; parent.mkdir(mode=0o700)
            entry = parent / "registered-final-entry"; entry.mkdir(mode=0o700)
            for name in self.module._REGULAR:
                (entry / name).write_bytes(b"authored non-executable fixture; no native library is loaded")
                (entry / name).chmod(0o400)
            if aliases:
                for name, target in self.module._ALIASES.items():
                    (entry / name).symlink_to(target)
            entry.chmod(0o500)
            def directory(path):
                info = path.stat()
                return {"path": str(path), "device": info.st_dev, "inode": info.st_ino,
                        "mode": stat.S_IMODE(info.st_mode)}
            def row(name, alias=False):
                info = (entry / name).lstat()
                result = {"filename": name, "device": info.st_dev, "inode": info.st_ino,
                    "bytes": info.st_size, "mode": stat.S_IMODE(info.st_mode), "nlink": info.st_nlink,
                    "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}
                if alias: result["target"] = self.module._ALIASES[name]
                return result
            proof = {"schema": self.module.SCHEMA, "parent": directory(parent), "directory": directory(entry),
                "files": [row(name) for name in sorted(self.module._REGULAR)],
                "aliases": [row(name, True) for name in sorted(self.module._ALIASES)] if aliases else []}
            # 실제 부모의 opaque cap도 모든 원 inode를 보유한다. 이 보유가
            # 없는 fixture는 unlink 직후 inode 번호 재사용을 허용해 버린다.
            parent_pins = [os.open(entry / item["filename"], os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
                           for item in proof["files"] + proof["aliases"]]
            (root / "ordinary.txt").write_bytes(b"ordinary")
            registration = {"staging_root": str(root), "max_output_bytes": 64 * 1024**2,
                            "receipt_path": str(root / "terminal.json"), "prepared_cuda_tree": proof}
            try:
                yield root, parent, entry, registration
            finally:
                for fd in parent_pins: os.close(fd)
                # 테스트가 직접 만든 이 root 아래 directory만 해제한다.
                for current, directories, _ in os.walk(root, followlinks=False):
                    Path(current).chmod(0o700)
                    for name in directories:
                        target = Path(current) / name
                        if not target.is_symlink(): target.chmod(0o700)

    def test_exact_v1_and_v2_rosters_charge_all_bytes_without_payload_read(self):
        for aliases in (False, True):
            with self.fixture(aliases=aliases) as (_, _, _, registration):
                proof = registration["prepared_cuda_tree"]
                expected = len(b"ordinary") + sum(row["bytes"] for row in proof["files"] + proof["aliases"])
                self.assertEqual(self.module.validate_prepared_cuda_tree(proof, registration["staging_root"]), proof)
                with patch("os.read", side_effect=AssertionError("payload read 금지")), patch("hashlib.sha256", side_effect=AssertionError("payload hash 금지")):
                    for _ in range(3): self.assertEqual(self.module.staging_bytes(registration), expected)
                    self.assertEqual(self.module.output_credit(registration), registration["max_output_bytes"] - expected - self.generic.MAX_JSON_BYTES)

    def test_generic_training_path_does_not_accept_prepared_metadata_as_permission(self):
        with self.fixture() as (_, _, _, registration):
            with self.assertRaises(ValueError): self.generic._staging_bytes(registration)
            with self.assertRaises(ValueError): self.generic._output_credit(registration)
            with self.assertRaises(ValueError): self.generic._terminal_receipt(registration, {"status": "fixture_only"})
        with self.fixture(aliases=False) as (_, _, _, registration):
            ordinary = {key: value for key, value in registration.items() if key != "prepared_cuda_tree"}
            self.assertEqual(self.module.staging_bytes(ordinary), self.generic._staging_bytes(ordinary))

    def test_closed_declaration_rejects_unknown_partial_wrong_target_and_fabricated_identity(self):
        with self.fixture() as (_, _, _, registration):
            proof = registration["prepared_cuda_tree"]
            mutations = [lambda p: p.update(extra=True), lambda p: p["aliases"].pop(),
                lambda p: p["aliases"][0].update(target="../elsewhere"),
                lambda p: p["aliases"][0].update(target="/absolute"),
                lambda p: p["files"][0].update(filename="not-registered.so"),
                lambda p: p["files"][0].update(mode=0o600),
                lambda p: p["files"][0].update(nlink=2),
                lambda p: p["files"][1].update(device=p["files"][0]["device"], inode=p["files"][0]["inode"])]
            for mutate in mutations:
                changed = copy.deepcopy(proof); mutate(changed)
                with self.assertRaises(ValueError): self.module.validate_prepared_cuda_tree(changed, registration["staging_root"])
            fabricated = copy.deepcopy(registration)
            fabricated["prepared_cuda_tree"]["files"][0]["inode"] += 1
            with self.assertRaises(ValueError): self.module.staging_bytes(fabricated)

    def test_actual_metadata_drift_and_unregistered_links_fail_closed(self):
        for kind in ("extra", "missing", "writable", "alias-target", "replacement", "outside-link"):
            with self.fixture() as (root, _, entry, registration):
                if kind == "outside-link":
                    (root / "unregistered-link").symlink_to(entry)
                elif kind == "writable":
                    (entry / sorted(self.module._REGULAR)[0]).chmod(0o600)
                else:
                    entry.chmod(0o700)
                    path = entry / sorted(self.module._REGULAR)[0]
                    if kind == "extra": (entry / "extra.fixture").write_bytes(b"extra")
                    elif kind == "missing": path.unlink()
                    elif kind == "replacement":
                        raw = path.read_bytes(); path.unlink(); path.write_bytes(raw); path.chmod(0o400)
                    elif kind == "alias-target":
                        path = entry / sorted(self.module._ALIASES)[0]; path.unlink(); path.symlink_to("libonnxruntime.so.1.22.0")
                    entry.chmod(0o500)
                with self.assertRaises((ValueError, OSError), msg=kind): self.module.staging_bytes(registration)

    def test_missing_or_replaced_delegated_root_and_output_cap_are_not_ignored(self):
        with self.fixture() as (_, parent, entry, registration):
            initial = self.module.staging_bytes(registration)
            too_small = dict(registration, max_output_bytes=initial - 1)
            with self.assertRaises(ValueError): self.module.staging_bytes(too_small)
            with self.assertRaises(ValueError): self.module.output_credit(registration, reserved_bytes=64 * 1024**2)
            entry.rename(parent / "moved-original")
            with self.assertRaises((ValueError, OSError)): self.module.staging_bytes(registration)
            entry.mkdir(mode=0o500)
            with self.assertRaises((ValueError, OSError)): self.module.staging_bytes(registration)

    def test_terminal_receipt_wrapper_and_repeated_scans_close_original_fds(self):
        with self.fixture() as (_, _, _, registration):
            before = len(os.listdir("/proc/self/fd"))
            for _ in range(8): self.module.staging_bytes(registration)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)
            result = self.module.terminal_receipt(registration, {"status": "source_fixture_only", "NN": False})
            self.assertEqual(json.loads(Path(result["path"]).read_text()), {"status": "source_fixture_only", "NN": False})
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_both_g0_cli_registrations_select_optional_observer_and_preserve_legacy_shape(self):
        from test_g0_numerical_cli import registration as collection_registration
        from test_g0_deployment_numerical_cli import registration as deployment_registration
        for name, make in (("g0_numerical_cli", collection_registration), ("g0_deployment_numerical_cli", deployment_registration)):
            cli = importlib.import_module("rz_pals_model." + name)
            with self.fixture() as (root, _, _, observed):
                value = make(cli, root)
                # Deployment fixture의 원 staging은 별도 child directory다. 이
                # 검사는 관측 tree가 존재하는 동일 managed root를 명시한다.
                if name == "g0_deployment_numerical_cli": value["staging_root"] = str(root)
                legacy = cli.validate_registration(value)
                self.assertNotIn("prepared_cuda_tree", legacy)
                value["prepared_cuda_tree"] = observed["prepared_cuda_tree"]
                selected = cli.validate_registration(value)
                self.assertIn("prepared_cuda_tree", selected)
                self.assertEqual(cli._staging_bytes(selected), self.module.staging_bytes(selected))
                selected["prepared_cuda_tree"]["future_success_flag"] = True
                with self.assertRaises(ValueError): cli.validate_registration(selected)


if __name__ == "__main__": unittest.main()
