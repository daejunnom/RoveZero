"""원 오류 보존 회귀: 실제 함수에 메모리 syscall/close 실패만 주입한다.

관리 owner/lease/커널 자원은 만들지 않는다. 실행은 Main이 담당한다.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import managed_storage as storage
import managed_output_unattributed_preservation as preservation


class UnprintableFailure(BaseException):
    def __str__(self):
        raise AssertionError("error formatting must not control cleanup")

    def add_note(self, note):
        raise AssertionError("diagnostic failure must not replace the primary")


class StorageFailurePreservationTests(unittest.TestCase):
    def write_record_with_failures(self, errors=None):
        errors = {} if errors is None else errors
        events, writes, unlinks = [], [], []
        path = Path("failure-preservation-fixture") / "owner.json"
        temporary = path.with_name("owner.tmp")
        file_fd, directory_fd = 17, 23

        def operation(stage, result=None):
            events.append(stage)
            if stage in errors:
                raise errors[stage]
            return result

        class Stream:
            name = str(temporary)

            def write(self, raw):
                writes.append(raw)
                return operation("write", len(raw))

            def flush(self):
                return operation("flush")

            def fileno(self):
                return file_fd

            def close(self):
                return operation("stream_close")

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

        stream = Stream()

        def factory(**kwargs):
            self.assertEqual(kwargs, {"dir": path.parent, "delete": False})
            return operation("temporary", stream)

        def fsync(fd):
            self.assertIn(fd, (file_fd, directory_fd))
            return operation("file_fsync" if fd == file_fd else "directory_fsync")

        def unlink(target, *, missing_ok=False):
            unlinks.append((target, missing_ok))
            return operation("temporary_unlink")

        fake_os = SimpleNamespace(
            name="posix", O_RDONLY=0, O_DIRECTORY=0, O_NOFOLLOW=0,
            fsync=fsync,
            replace=Mock(side_effect=lambda *_: operation("replace")),
            open=Mock(side_effect=lambda *_: operation("directory_open", directory_fd)),
            close=Mock(side_effect=lambda *_: operation("directory_close")),
        )
        failure = None
        # Keep input validation outside this fault-injection seam. All actual
        # write_record publication/close/cleanup code runs; no filesystem opens.
        with patch.object(storage, "checked", side_effect=lambda value: value), \
                patch.object(storage.tempfile, "NamedTemporaryFile", side_effect=factory), \
                patch.object(storage, "os", fake_os), \
                patch.object(storage.Path, "unlink", new=unlink):
            try:
                storage.write_record(path, {"z": 2, "a": "value"})
            except BaseException as error:
                failure = error
        return SimpleNamespace(error=failure, events=events, writes=writes,
                               unlinks=unlinks, temporary=temporary)

    def assert_cleanup_once(self, result):
        self.assertEqual(result.events.count("stream_close"), 1)
        self.assertEqual(result.events.count("temporary_unlink"), 1)
        self.assertEqual(result.unlinks, [(result.temporary, True)])

    def test_write_failure_preserves_original_and_cleans_temporary(self):
        original, close, unlink = BaseException("write"), OSError("close"), OSError("unlink")
        result = self.write_record_with_failures(
            {"write": original, "stream_close": close, "temporary_unlink": unlink})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertNotIn("flush", result.events)
        self.assertNotIn("replace", result.events)
        self.assertEqual(result.events.count("write"), 1)
        self.assertTrue(any("stream close" in note for note in original.__notes__))
        self.assertTrue(any("temporary unlink" in note for note in original.__notes__))

    def test_flush_failure_preserves_original_and_close_note(self):
        original = RuntimeError("flush")
        result = self.write_record_with_failures(
            {"flush": original, "stream_close": OSError("close")})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertNotIn("file_fsync", result.events)
        self.assertNotIn("replace", result.events)
        self.assertTrue(any("stream close" in note for note in original.__notes__))

    def test_file_fsync_failure_preserves_original_and_close_note(self):
        original = OSError("file fsync")
        result = self.write_record_with_failures(
            {"file_fsync": original, "stream_close": OSError("close")})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertEqual(result.events.count("file_fsync"), 1)
        self.assertNotIn("replace", result.events)
        self.assertTrue(any("stream close" in note for note in original.__notes__))

    def test_replace_failure_preserves_original_and_unlink_note(self):
        original = OSError("replace")
        result = self.write_record_with_failures(
            {"replace": original, "temporary_unlink": OSError("unlink")})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertEqual(result.events.count("replace"), 1)
        self.assertNotIn("directory_open", result.events)
        self.assertTrue(any("temporary unlink" in note for note in original.__notes__))

    def test_directory_fsync_failure_preserves_original_and_close_note(self):
        original = OSError("directory fsync")
        result = self.write_record_with_failures(
            {"directory_fsync": original, "directory_close": OSError("directory close")})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertEqual(result.events.count("replace"), 1)
        self.assertEqual(result.events.count("directory_fsync"), 1)
        self.assertEqual(result.events.count("directory_close"), 1)
        self.assertTrue(any("directory close" in note for note in original.__notes__))

    def test_stream_close_failure_is_not_success(self):
        original = OSError("stream close")
        result = self.write_record_with_failures({"stream_close": original})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertNotIn("replace", result.events)

    def test_directory_close_failure_is_not_success(self):
        original = OSError("directory close")
        result = self.write_record_with_failures({"directory_close": original})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertEqual(result.events.count("directory_close"), 1)

    def test_temporary_unlink_failure_is_not_success(self):
        original = OSError("temporary unlink")
        result = self.write_record_with_failures({"temporary_unlink": original})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertIn("replace", result.events)

    def test_directory_open_failure_still_cleans_temporary(self):
        original = OSError("directory open")
        result = self.write_record_with_failures({"directory_open": original})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertIn("replace", result.events)
        self.assertNotIn("directory_close", result.events)

    def test_diagnostic_failure_cannot_stop_cleanup_or_replace_original(self):
        original = UnprintableFailure()
        result = self.write_record_with_failures(
            {"write": original, "stream_close": OSError("close"),
             "temporary_unlink": OSError("unlink")})
        self.assertIs(result.error, original)
        self.assert_cleanup_once(result)
        self.assertNotIn("replace", result.events)

    def test_normal_write_keeps_encoding_and_operation_order(self):
        result = self.write_record_with_failures()
        self.assertIsNone(result.error)
        self.assertEqual(result.writes, [(json.dumps({"z": 2, "a": "value"}, sort_keys=True) + "\n").encode()])
        self.assert_cleanup_once(result)
        self.assertEqual(result.events, ["temporary", "write", "flush", "file_fsync", "stream_close",
                                         "replace", "directory_open", "directory_fsync", "directory_close",
                                         "temporary_unlink"])

    def catalog_view(self, history_error=None, held_error=None, reenter=False):
        view = object.__new__(preservation.CatalogView)
        view.closed = False
        events = []

        def history_close(primary):
            events.append(("history", primary))
            if reenter:
                view.close(primary)
            if history_error is not None:
                raise history_error

        def held_close(primary):
            events.append(("held", primary))
            if held_error is not None:
                raise held_error

        view.history = SimpleNamespace(close=history_close)
        view.held = SimpleNamespace(close=held_close)
        return view, events

    def test_catalog_close_preserves_first_close_error(self):
        original = BaseException("history close")
        view, events = self.catalog_view(original, OSError("held close"))
        with self.assertRaises(BaseException) as caught:
            view.close()
        self.assertIs(caught.exception, original)
        self.assertEqual(events, [("history", None), ("held", original)])
        self.assertTrue(view.closed)
        self.assertTrue(any("held inputs" in note for note in original.__notes__))
        view.close()
        self.assertEqual(len(events), 2)

    def test_catalog_close_keeps_supplied_primary_and_runs_both(self):
        original = BaseException("existing operation failure")
        view, events = self.catalog_view(OSError("history close"), OSError("held close"))
        observed = None
        try:
            try:
                raise original
            except BaseException as primary:
                view.close(primary)
                raise
        except BaseException as error:
            observed = error
        self.assertIs(observed, original)
        self.assertEqual(events, [("history", original), ("held", original)])
        self.assertTrue(view.closed)
        self.assertTrue(any("history" in note for note in original.__notes__))
        self.assertTrue(any("held inputs" in note for note in original.__notes__))

    def test_catalog_close_note_failure_keeps_first_and_runs_both(self):
        original = UnprintableFailure()
        view, events = self.catalog_view(original, OSError("held close"))
        with self.assertRaises(UnprintableFailure) as caught:
            view.close()
        self.assertIs(caught.exception, original)
        self.assertEqual(events, [("history", None), ("held", original)])
        self.assertTrue(view.closed)

    def test_catalog_close_normal_and_reentry_once(self):
        view, events = self.catalog_view(reenter=True)
        self.assertIsNone(view.close())
        self.assertTrue(view.closed)
        self.assertIsNone(view.close())
        self.assertEqual(events, [("history", None), ("held", None)])

    def test_catalog_close_second_failure_propagates_when_first_succeeds(self):
        original = OSError("held close")
        view, events = self.catalog_view(held_error=original)
        with self.assertRaises(OSError) as caught:
            view.close()
        self.assertIs(caught.exception, original)
        self.assertEqual(events, [("history", None), ("held", None)])
        self.assertTrue(view.closed)


if __name__ == "__main__":
    unittest.main()
