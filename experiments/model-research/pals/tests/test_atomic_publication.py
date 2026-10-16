"""Small stdlib publication checks; no model, NN library or optimizer import."""
import builtins
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


_original_import = builtins.__import__


def _without_neural_imports(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"} or name == "model" or name.endswith(".model"):
        raise AssertionError("publication checks must not import a model or neural library")
    return _original_import(name, *args, **kwargs)


with mock.patch("builtins.__import__", side_effect=_without_neural_imports):
    from rz_pals_model import artifacts


@unittest.skipUnless(sys.platform == "linux" or os.name == "nt", "native no-replace publication supports Linux and Windows only")
class AtomicPublicationTests(unittest.TestCase):
    def setUp(self):
        blocker = mock.patch("builtins.__import__", side_effect=_without_neural_imports)
        blocker.start()
        self.addCleanup(blocker.stop)
        temporary = tempfile.TemporaryDirectory(prefix="rz-atomic-publication-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def partial(self, name="owned.partial", raw=b"small owned ASCII publication\n"):
        path = self.root / name
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def test_move_is_complete_single_link_and_never_calls_link_or_replace(self):
        source = self.partial()
        destination = self.root / "final.txt"
        identity = source.stat()
        with mock.patch.object(os, "link", side_effect=AssertionError("hard links forbidden")), \
                mock.patch.object(os, "replace", side_effect=AssertionError("overwrite forbidden")):
            artifacts.publish_noreplace(source, destination)
        final = destination.stat()
        self.assertEqual((final.st_dev, final.st_ino, final.st_size), (identity.st_dev, identity.st_ino, identity.st_size))
        self.assertEqual(final.st_nlink, 1)
        self.assertEqual(destination.read_bytes(), b"small owned ASCII publication\n")
        self.assertFalse(source.exists())

    def test_existing_winner_keeps_its_bytes_and_loser_keeps_partial(self):
        source = self.partial()
        destination = self.root / "final.txt"
        destination.write_bytes(b"already registered\n")
        with self.assertRaises(FileExistsError) as failure:
            artifacts.publish_noreplace(source, destination)
        self.assertEqual(failure.exception.publication_state, "not_published")
        self.assertEqual(destination.read_bytes(), b"already registered\n")
        self.assertEqual(source.read_bytes(), b"small owned ASCII publication\n")
        self.assertEqual((source.stat().st_nlink, destination.stat().st_nlink), (1, 1))

    def test_concurrent_publishers_have_one_winner_and_observer_sees_no_hard_links(self):
        sources = [self.partial(f"writer-{index}.partial", f"writer {index}\n".encode("ascii")) for index in range(12)]
        destination = self.root / "final.txt"
        barrier = threading.Barrier(len(sources))
        stop, scanned = threading.Event(), threading.Event()
        outcomes, errors, observed = [], [], []

        def observe():
            try:
                while not stop.is_set():
                    for path in self.root.iterdir():
                        try:
                            details = path.lstat()
                        except FileNotFoundError:
                            continue  # A rename consumed that partial name.
                        observed.append(details.st_nlink)
                        if details.st_nlink != 1:
                            raise AssertionError("artifact observer saw a hard link")
                    scanned.set()
                    stop.wait(0.001)
            except BaseException as error:
                errors.append(error)

        def publish(source):
            try:
                barrier.wait(timeout=5)
                artifacts.publish_noreplace(source, destination)
                outcomes.append((source, "won"))
            except FileExistsError:
                outcomes.append((source, "lost"))
            except BaseException as error:
                errors.append(error)

        observer = threading.Thread(target=observe, daemon=True)
        workers = [threading.Thread(target=publish, args=(source,), daemon=True) for source in sources]
        observer.start()
        try:
            self.assertTrue(scanned.wait(5), "observer did not scan before publication")
            for worker in workers:
                worker.start()
            deadline = time.monotonic() + 5
            for worker in workers:
                worker.join(max(0, deadline - time.monotonic()))
            self.assertFalse(any(worker.is_alive() for worker in workers), "bounded publication did not finish")
        finally:
            stop.set()
            observer.join(5)
        self.assertFalse(observer.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(observed)
        winners = [source for source, result in outcomes if result == "won"]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(outcomes), len(sources))
        self.assertFalse(winners[0].exists())
        self.assertEqual(destination.stat().st_nlink, 1)
        self.assertEqual(len(list(self.root.glob("*.partial"))), len(sources) - 1)

    def test_old_link_unlink_barrier_exposes_two_links(self):
        # Mechanism reproduction only; this cannot identify a past failing path.
        source = self.partial()
        destination = self.root / "old-final.txt"
        linked, release = threading.Event(), threading.Event()
        errors = []

        def old_publish():
            try:
                os.link(source, destination)
                linked.set()
                if not release.wait(5):
                    raise AssertionError("old publisher barrier timed out")
                source.unlink()
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=old_publish, daemon=True)
        worker.start()
        try:
            self.assertTrue(linked.wait(5))
            self.assertEqual((source.stat().st_nlink, destination.stat().st_nlink), (2, 2))
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(destination.stat().st_nlink, 1)

    def test_preexisting_hardlink_and_different_parent_are_refused(self):
        source = self.partial()
        os.link(source, self.root / "second-name.txt")
        with self.assertRaises(artifacts.AtomicPublicationError):
            artifacts.publish_noreplace(source, self.root / "final.txt")
        self.assertFalse((self.root / "final.txt").exists())
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(ValueError):
            artifacts.publish_noreplace(source, other / "final.txt")

    def test_native_failure_keeps_primary_and_partial_without_fallback(self):
        source = self.partial()
        failure = OSError(errno.EIO, "injected native publication failure")
        owner, method = ((artifacts, "_linux_rename_noreplace") if sys.platform == "linux" else (os, "rename"))
        with mock.patch.object(owner, method, side_effect=failure), \
                mock.patch.object(os, "link", side_effect=AssertionError("no fallback")), \
                mock.patch.object(os, "replace", side_effect=AssertionError("no fallback")):
            with self.assertRaises(OSError) as observed:
                artifacts.publish_noreplace(source, self.root / "final.txt")
        self.assertIs(observed.exception, failure)
        self.assertEqual(failure.publication_state, "unknown")
        self.assertTrue(source.exists())
        self.assertFalse((self.root / "final.txt").exists())

    def test_post_rename_identity_error_keeps_final_and_does_not_report_success(self):
        source = self.partial()
        destination = self.root / "final.txt"
        original_identity = artifacts._publication_identity

        def identity(details, *, directory=False):
            observed = original_identity(details, directory=directory)
            if not directory and destination.exists():
                return observed[:-1] + (observed[-1] + 1,)
            return observed

        with mock.patch.object(artifacts, "_publication_identity", side_effect=identity):
            with self.assertRaises(artifacts.AtomicPublicationError) as failure:
                artifacts.publish_noreplace(source, destination)
        self.assertEqual(failure.exception.publication_state, "published")
        self.assertTrue(destination.exists())
        self.assertFalse(source.exists())

    @unittest.skipUnless(sys.platform == "linux", "retained directory FD is Linux-only")
    def test_directory_fsync_and_close_failures_keep_original_error(self):
        source = self.partial()
        failure = OSError(errno.EIO, "injected parent fsync failure")
        close = os.close

        def close_then_fail(descriptor):
            close(descriptor)
            raise OSError(errno.EIO, "injected close failure")

        with mock.patch.object(os, "fsync", side_effect=failure), mock.patch.object(os, "close", side_effect=close_then_fail):
            with self.assertRaises(OSError) as observed:
                artifacts.publish_noreplace(source, self.root / "final.txt")
        self.assertIs(observed.exception, failure)
        self.assertEqual(failure.publication_state, "published")
        self.assertIn("injected close failure", str(failure.publication_cleanup_error))
        self.assertTrue((self.root / "final.txt").exists())

    def test_pure_writer_seams_publish_only_small_ascii_bytes(self):
        from rz_pals_model import generation_export, initialization_evidence, numeric_capture, onnx_virtual_review
        numeric = self.root / "numeric.txt"
        pin = numeric_capture._publish({"source_only": True}, numeric, 0)
        self.assertEqual(pin["sha256"], hashlib.sha256(numeric.read_bytes()).hexdigest())
        initialization_evidence._atomic_publish_bytes(self.root, b'{"source_only":true}\n')
        generation_export._atomic_json(self.root / "generation.txt", {"source_only": True})

        class SmallDocument:
            def SerializeToString(self):
                return b"ASCII serialization fixture, no ONNX model\n"

        graph = self.root / "serialization.txt"
        onnx_virtual_review._save(SmallDocument(), {"source_only": True}, graph)
        self.assertEqual(graph.read_bytes(), SmallDocument().SerializeToString())
        self.assertFalse(list(self.root.glob("*.partial")))
        self.assertTrue(all(path.stat().st_nlink == 1 for path in self.root.iterdir()))

    def checkpoint_fixture(self):
        from rz_pals_model import generation_training
        temporary_names = []

        def ascii_serializer(payload, stream):
            temporary_names.append(Path(stream.stream.name))
            stream.write(b"ASCII checkpoint I/O fixture only\n")

        return generation_training, temporary_names, ascii_serializer

    def test_checkpoint_success_preserves_a_recreated_temporary_name(self):
        training, names, serializer = self.checkpoint_fixture()
        destination = self.root / "checkpoint-io.txt"
        owner, method = ((artifacts, "_linux_rename_noreplace") if sys.platform == "linux" else (os, "rename"))
        rename = getattr(owner, method)

        def publish_then_recreate(*args):
            rename(*args)
            names[-1].write_bytes(b"different owner after successful rename\n")

        with mock.patch.object(training, "_serialize_checkpoint", side_effect=serializer), \
                mock.patch.object(owner, method, side_effect=publish_then_recreate), \
                mock.patch.object(os, "unlink", side_effect=AssertionError("old temp name must not be deleted")) as unlink:
            result = training.atomic_checkpoint(destination, {"ledger": {"cost": {"output_bytes": 0}}}, max_file_bytes=128)
            unlink.assert_not_called()
        self.assertEqual(result, destination)
        self.assertEqual(destination.read_bytes(), b"ASCII checkpoint I/O fixture only\n")
        self.assertEqual(names[-1].read_bytes(), b"different owner after successful rename\n")

    def test_checkpoint_known_published_error_preserves_recreated_name(self):
        training, names, serializer = self.checkpoint_fixture()
        destination = self.root / "checkpoint-io.txt"
        owner, method = ((artifacts, "_linux_rename_noreplace") if sys.platform == "linux" else (os, "rename"))
        rename = getattr(owner, method)
        original_identity = artifacts._publication_identity

        def publish_then_recreate(*args):
            rename(*args)
            names[-1].write_bytes(b"different owner after published failure\n")

        def identity(details, *, directory=False):
            observed = original_identity(details, directory=directory)
            return observed[:-1] + (observed[-1] + 1,) if not directory and destination.exists() else observed

        with mock.patch.object(training, "_serialize_checkpoint", side_effect=serializer), \
                mock.patch.object(owner, method, side_effect=publish_then_recreate), \
                mock.patch.object(artifacts, "_publication_identity", side_effect=identity), \
                mock.patch.object(os, "unlink", side_effect=AssertionError("known published name must be preserved")) as unlink:
            with self.assertRaises(artifacts.AtomicPublicationError) as failure:
                training.atomic_checkpoint(destination, {"ledger": {"cost": {"output_bytes": 0}}}, max_file_bytes=128)
            unlink.assert_not_called()
        self.assertEqual(failure.exception.publication_state, "published")
        self.assertEqual(names[-1].read_bytes(), b"different owner after published failure\n")
        self.assertTrue(destination.exists())

    def test_checkpoint_unknown_native_error_preserves_new_name_and_original_error(self):
        training, names, serializer = self.checkpoint_fixture()
        destination = self.root / "checkpoint-io.txt"
        owner, method = ((artifacts, "_linux_rename_noreplace") if sys.platform == "linux" else (os, "rename"))
        rename = os.rename
        failure = OSError(errno.EIO, "original native failure, publication unknown")

        def replace_owned_name_then_fail(*args):
            # Move only this test's owned ASCII partial, then simulate another
            # inode at its old name while the native operation reports failure.
            rename(names[-1], self.root / "owned-before-native-error.txt")
            names[-1].write_bytes(b"different owner after unknown failure\n")
            raise failure

        with mock.patch.object(training, "_serialize_checkpoint", side_effect=serializer), \
                mock.patch.object(owner, method, side_effect=replace_owned_name_then_fail), \
                mock.patch.object(os, "unlink", side_effect=OSError(errno.EIO, "cleanup must not mask primary")) as unlink:
            with self.assertRaises(OSError) as observed:
                training.atomic_checkpoint(destination, {"ledger": {"cost": {"output_bytes": 0}}}, max_file_bytes=128)
            unlink.assert_not_called()
        self.assertIs(observed.exception, failure)
        self.assertEqual(failure.publication_state, "unknown")
        self.assertEqual(names[-1].read_bytes(), b"different owner after unknown failure\n")
        self.assertFalse(destination.exists())

    def test_checkpoint_serializer_error_keeps_owned_partial_and_primary(self):
        training, names, _ = self.checkpoint_fixture()
        failure = OSError(errno.EIO, "original serializer failure")

        def failed_serializer(payload, stream):
            names.append(Path(stream.stream.name))
            stream.write(b"incomplete owned ASCII bytes\n")
            raise failure

        with mock.patch.object(training, "_serialize_checkpoint", side_effect=failed_serializer), \
                mock.patch.object(os, "unlink", side_effect=OSError(errno.EIO, "cleanup must not mask primary")) as unlink:
            with self.assertRaises(OSError) as observed:
                training.atomic_checkpoint(self.root / "checkpoint-io.txt", {"ledger": {"cost": {"output_bytes": 0}}}, max_file_bytes=128)
            unlink.assert_not_called()
        self.assertIs(observed.exception, failure)
        self.assertEqual(names[-1].read_bytes(), b"incomplete owned ASCII bytes\n")
        self.assertFalse((self.root / "checkpoint-io.txt").exists())


class LinuxPublicationBindingTests(unittest.TestCase):
    def test_renameat2_abi_and_typed_unsupported_never_fall_back(self):
        calls = []

        class Rename:
            error = 0

            def __call__(self, *args):
                calls.append(args)
                ctypes.set_errno(self.error)
                return -1 if self.error else 0

        rename = Rename()
        library = type("LibcFixture", (), {"renameat2": rename})()
        with mock.patch.object(ctypes, "CDLL", return_value=library):
            artifacts._linux_rename_noreplace(91, "owned.partial", "final.txt")
            self.assertEqual(calls[-1], (91, b"owned.partial", 91, b"final.txt", 1))
            self.assertEqual(rename.restype, ctypes.c_int)
            self.assertEqual(len(rename.argtypes), 5)
            for error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                rename.error = error
                with self.assertRaises(artifacts.AtomicPublicationUnsupported) as failure:
                    artifacts._linux_rename_noreplace(91, "owned.partial", "final.txt")
                self.assertEqual(failure.exception.native_errno, error)
                self.assertEqual(failure.exception.code, "artifact.publication.unsupported")
            rename.error = errno.EEXIST
            with self.assertRaises(FileExistsError):
                artifacts._linux_rename_noreplace(91, "owned.partial", "final.txt")
        with mock.patch.object(ctypes, "CDLL", return_value=object()):
            with self.assertRaises(artifacts.AtomicPublicationUnsupported):
                artifacts._linux_rename_noreplace(91, "owned.partial", "final.txt")


if __name__ == "__main__":
    unittest.main()
