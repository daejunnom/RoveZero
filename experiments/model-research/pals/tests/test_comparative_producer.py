"""Synthetic OS/process source fixtures, not actual child/checker evidence."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rz_pals_model import comparative_producer as producer
from rz_pals_model import comparative_training as candidate
from test_comparative_training import CandidateFixture
from test_training import frozen_fixture_bytes as raw, sha


class _OutputPipe:
    def __init__(self, process, name):
        self.process, self.name, self.stream = process, name, None

    def read(self, count):
        self.process.ready.wait()
        if self.stream is None:
            self.stream = io.BytesIO(self.process.outputs[self.name])
        return self.stream.read(count)

    def close(self):
        if self.stream is not None:
            self.stream.close()


class _InputPipe:
    def __init__(self, process):
        self.process, self.data = process, bytearray()

    def write(self, view):
        self.data.extend(view)
        return len(view)

    def flush(self):
        pass

    def close(self):
        if not self.process.ready.is_set():
            self.process.finish(bytes(self.data))


class SyntheticProcess:
    """Simulated process/pipe timing. No executable or chess engine launches."""
    instances = {}

    def __init__(self, fixture, *, mode="completed"):
        self.fixture, self.mode = fixture, mode
        self.pid, self.returncode = 4242, None
        self.group_alive = True
        SyntheticProcess.instances[self.pid] = self
        self.ready, self.killed = threading.Event(), False
        self.outputs = {"stdout": b"", "stderr": b""}
        self.stdin, self.stdout, self.stderr = _InputPipe(self), _OutputPipe(self, "stdout"), _OutputPipe(self, "stderr")
        self.exit_at = float("inf")

    def finish(self, request_raw):
        if self.killed:
            self.ready.set()
            return
        request = candidate._json(request_raw)
        template = json.loads(self.fixture.executions[request["task_id"]]["receipt"])
        for name in ("schema", "task_id", "captured_input_sha256", "checker_namespace_sha256", "before_result_anchor_sha256",
                     "frozen_epoch", "input_revision", "context_sha256", "cpu_binary_sha256"):
            template[name] = request[name]
        conditions = template["report"]["conditions"]
        conditions["resource_policy"] = {"max_wall_time_ms": request["max_wall_time_ms"], "max_checks": 1,
                                         "search_deadline_reserve_ms": min(request["max_wall_time_ms"] // 10, 1000)}
        template["report"]["conditions_sha256"] = candidate.wire_digest(conditions)
        if self.mode == "partial":
            template["status"] = "partial"
            template["report"].update(completion="node_limit", completed_depth=1)
        if self.mode == "nonzero":
            self.outputs = {"stdout": b'{"unknown_work":true}', "stderr": b'{"known_nodes":42}'}
        elif self.mode == "missing":
            self.outputs = {"stdout": b"", "stderr": b"no successful response"}
        else:
            self.outputs["stdout"] = raw(template)
        if self.mode == "overflow":
            self.outputs["stdout"] += b"x" * (request["max_output_bytes"] + 1)
        self.exit_at = time.monotonic() + (999 if self.mode == "timeout" else 0.02)
        self.ready.set()

    def wait(self, timeout):
        if self.killed:
            self.returncode = -9
            return -9
        until = time.monotonic() + timeout
        self.ready.wait(timeout=max(0, timeout))
        left = min(until - time.monotonic(), self.exit_at - time.monotonic())
        if left > 0:
            time.sleep(left)
        if time.monotonic() < self.exit_at:
            raise subprocess.TimeoutExpired("synthetic candidate", timeout)
        self.returncode = 1 if self.mode == "nonzero" else 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.ready.set()

    def kill_group(self):
        self.group_alive = False
        if self.returncode is None:
            self.kill()


def _synthetic_killpg(pid, requested_signal):
    """Never send an OS signal to a real PID, including on POSIX test hosts."""
    process = SyntheticProcess.instances.get(pid)
    if process is None or not process.group_alive:
        raise ProcessLookupError("synthetic owned group absent")
    if requested_signal:
        process.kill_group()


def _image_observation(expected):
    return {"status": "checked", "artifact": expected, "scope": "caller_registered_path_bytes",
            "device": "1", "inode": "2", "mtime_ns": "3", "ctime_ns": "4"}


def _arguments(fixture):
    files = {"checker_registration": fixture.registration, "checker_source": fixture.source, "cpu_binary": fixture.binary,
             "criterion": fixture.criterion,
             "selections": raw({"schema": producer.SELECTION_SCHEMA, "pairs": [
                 {name: pair[name] for name in ("pair_id", "input_sha256", "kind", "candidates")}
                 | {"task_ids": [check["task_id"] for check in pair["checks"]]} for pair in fixture.body["pairs"]]})}
    arguments = {"strict_parent_options": fixture.parent_options, "output": fixture.root / "producer-bank",
                 "max_pairs": 2, "max_nodes": 400000, "max_wall_time_ms": 10000,
                 "max_output_bytes": 16 * 1024 * 1024, "max_input_bytes": 32 * 1024 * 1024}
    for name, content in files.items():
        path = fixture.root / ("registered-" + name + ".fixture")
        path.write_bytes(content)
        arguments[name + "_path"] = path
        arguments[name + "_sha256"] = candidate.byte_pin(content)["sha256"]
    return arguments


class ComparativeProducerTests(unittest.TestCase):
    def setUp(self):
        SyntheticProcess.instances.clear()
        patcher = patch.object(producer.os, "killpg", side_effect=_synthetic_killpg, create=True)
        self.group_signals = patcher.start()
        self.addCleanup(patcher.stop)

    def test_durable_before_spawn_and_saved_pc_bank_reloads_through_strict_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary, history=True)
            arguments = _arguments(fixture)
            calls = []

            def popen(command, **options):
                bank = arguments["output"]
                self.assertTrue((bank / "pair-plan.json").is_file())
                self.assertEqual((bank / "ordinal-criterion.json").read_bytes(), fixture.criterion)
                self.assertTrue(all((bank / f"task-{slot}-request.json").is_file() for slot in range(4)))
                self.assertEqual(command, [str(arguments["cpu_binary_path"].resolve()), "--candidate-only"])
                self.assertFalse(options["shell"])
                self.assertEqual(options["start_new_session"], producer.os.name == "posix")
                calls.append(command)
                return SyntheticProcess(fixture)

            with patch.object(producer.sys, "platform", "win32"), patch.object(producer.subprocess, "Popen", side_effect=popen):
                receipt = producer.produce_candidate_bank(temporary, **arguments)
            self.assertIsNone(receipt["failure"])
            self.assertEqual(receipt["known_pairs"], {"critic": 1, "proposer": 1})
            self.assertEqual(receipt["children_spawned"], 4)
            self.assertEqual(len(calls), 4)
            self.assertFalse(receipt["model_executed"])
            self.assertFalse(receipt["actual_training_executed"])
            self.assertFalse(receipt["timing_comparison_admitted"])
            self.assertLessEqual(receipt["elapsed_ms"], arguments["max_wall_time_ms"])
            output = arguments["output"]
            pins = json.loads((output / "expected-pins.json").read_bytes())
            expected_metadata = json.loads((output / "metadata-pins.json").read_bytes())
            reloaded = candidate.load_persisted_comparative_pairs(temporary, output,
                       expected_bank_sha256=receipt["bank_artifact"]["sha256"], strict_parent_options=fixture.parent_options,
                       independent_pins=pins, expected_metadata_pins=expected_metadata)
            self.assertEqual({target.role for target in reloaded.targets}, {"proposer", "critic"})
            self.assertTrue(all(target.mask for target in reloaded.targets))
            self.assertEqual(len(reloaded.parents.records), 3)
            self.assertTrue(all(observation["receipt"] for observation in reloaded.raw_executions().values()))
            for slot in range(4):
                process = json.loads((output / f"task-{slot}-process.json").read_bytes())
                self.assertEqual(process["loaded_executable"]["status"], "unknown")
                self.assertEqual(process["loaded_executable"]["scope"], "platform_path_hash_only")
                self.assertTrue(process["reaped"])
                self.assertTrue(process["pipes_finished"])
                self.assertTrue(process["binary_path_stable"])
                self.assertTrue(process["source_path_stable"])
                self.assertEqual(process["pre_source"]["artifact"], candidate.byte_pin(fixture.source))
                self.assertEqual(process["post_source"]["artifact"], candidate.byte_pin(fixture.source))
                if producer.os.name == "posix":
                    self.assertEqual(process["process_supervision_scope"], "posix_owned_process_group")
                    self.assertTrue(process["owned_group_absent"])
                else:
                    self.assertEqual(process["process_supervision_scope"], "windows_direct_child_only_no_job_object")
                    self.assertIsNone(process["owned_group_absent"])

    def test_node_output_and_prior_pin_failures_happen_before_dispatch(self):
        for changed in ("nodes", "output", "binary_pin"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                fixture = CandidateFixture(temporary)
                arguments = _arguments(fixture)
                if changed == "nodes":
                    arguments["max_nodes"] = 1
                elif changed == "output":
                    arguments["max_output_bytes"] = 1024
                else:
                    arguments["cpu_binary_sha256"] = sha("wrong independent binary pin")
                with patch.object(producer.sys, "platform", "win32"), patch.object(producer.subprocess, "Popen") as launch:
                    with self.assertRaises(ValueError):
                        producer.produce_candidate_bank(temporary, **arguments)
                    launch.assert_not_called()

    def test_registered_inputs_and_registration_set_are_charged_before_parent_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary)
            arguments = _arguments(fixture)
            _, _, training, preparation = producer._modules()
            set_raw = raw({"version": preparation.REGISTRATION_SET_SCHEMA, "producer_registrations": [
                {**entry, "registration_path": str(entry["registration_path"]),
                 "checked_source_path": str(entry["checked_source_path"])}
                for entry in fixture.parent_options["producer_registrations"]]})
            arguments["_registration_set_pin"] = candidate.byte_pin(set_raw)
            registered_bytes = sum(arguments[name + "_path"].stat().st_size for name in
                                   ("checker_registration", "checker_source", "cpu_binary", "criterion", "selections"))
            observed = []

            def stop_before_read(directory, **options):
                observed.append(options["max_input_bytes"])
                raise ValueError("synthetic observed remaining allowance before strict read")

            with patch.object(producer.sys, "platform", "win32"), \
                    patch.object(training, "load_frozen_collected_dataset", side_effect=stop_before_read), \
                    patch.object(producer.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "remaining allowance before strict read"):
                    producer.produce_candidate_bank(temporary, **arguments)
                launch.assert_not_called()
            self.assertEqual(observed, [arguments["max_input_bytes"] - registered_bytes - len(set_raw)])

    def test_remaining_parent_and_exact_reread_allowances_reject_before_allocation(self):
        for boundary in ("parent_first_read", "prepared_reread"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temporary:
                fixture = CandidateFixture(temporary)
                arguments = _arguments(fixture)
                _, _, training, preparation = producer._modules()
                registered_bytes = sum(arguments[name + "_path"].stat().st_size for name in
                                       ("checker_registration", "checker_source", "cpu_binary", "criterion", "selections"))
                receipt_bytes = (fixture.root / "receipt.json").stat().st_size
                parent_bytes = preparation._frozen_input_bytes(fixture.root, fixture.parent_options["producer_registrations"],
                                                               fixture.parents.frozen_admission)
                # First case cannot allocate receipt bytes. Second admits the
                # parent, then denies an additional retained receipt buffer.
                arguments["max_input_bytes"] = registered_bytes + receipt_bytes - 1
                if boundary == "prepared_reread":
                    arguments["max_input_bytes"] += parent_bytes
                with patch.object(producer.sys, "platform", "win32"), patch.object(producer.subprocess, "Popen") as launch:
                    with self.assertRaisesRegex(ValueError, "allocation budget"):
                        producer.produce_candidate_bank(temporary, **arguments)
                    launch.assert_not_called()
                self.assertFalse(arguments["output"].exists())

    def test_source_current_path_drift_after_publication_prevents_dispatch_and_preserves_pinned_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary)
            arguments = _arguments(fixture)
            ordinary_write = producer._Bank.write

            def source_drift(bank, name, content):
                result = ordinary_write(bank, name, content)
                if name == "pair-plan.json":
                    arguments["checker_source_path"].write_bytes(fixture.source + b" ")
                return result

            with patch.object(producer.sys, "platform", "win32"), patch.object(producer._Bank, "write", source_drift), \
                    patch.object(producer.subprocess, "Popen") as launch:
                receipt = producer.produce_candidate_bank(temporary, **arguments)
                launch.assert_not_called()
            self.assertEqual(receipt["status"], "rejected")
            self.assertEqual(receipt["children_spawned"], 0)
            self.assertEqual(receipt["known_pairs"], {})
            self.assertEqual((arguments["output"] / "checker-source.json").read_bytes(), fixture.source)
            process = json.loads((arguments["output"] / "task-0-process.json").read_bytes())
            self.assertFalse(process["source_path_stable"])
            self.assertEqual(process["post_source"]["status"], "failed")

    def test_source_current_path_drift_during_child_retains_raw_and_blocks_later_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary)
            arguments = _arguments(fixture)
            calls = []

            def popen(*args, **options):
                process = SyntheticProcess(fixture)
                ordinary_finish = process.finish

                def finish(request_raw):
                    ordinary_finish(request_raw)
                    arguments["checker_source_path"].write_bytes(fixture.source + b" ")

                process.finish = finish
                calls.append(process)
                return process

            with patch.object(producer.sys, "platform", "win32"), patch.object(producer.subprocess, "Popen", side_effect=popen):
                receipt = producer.produce_candidate_bank(temporary, **arguments)
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipt["status"], "rejected")
            self.assertEqual(receipt["known_pairs"], {})
            self.assertTrue((arguments["output"] / "task-0-stdout.bin").is_file())
            self.assertEqual((arguments["output"] / "checker-source.json").read_bytes(), fixture.source)
            process = json.loads((arguments["output"] / "task-0-process.json").read_bytes())
            self.assertEqual(process["pre_source"]["status"], "checked")
            self.assertEqual(process["post_source"]["status"], "failed")
            self.assertFalse(process["source_path_stable"])

    def test_partial_nonzero_and_missing_preserve_raw_and_mask_targets(self):
        for mode in ("partial", "nonzero", "missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = CandidateFixture(temporary, roles=("proposer",))
                arguments = _arguments(fixture)
                with patch.object(producer.sys, "platform", "win32"), patch.object(producer.subprocess, "Popen",
                                  side_effect=lambda *args, **kwargs: SyntheticProcess(fixture, mode=mode)):
                    receipt = producer.produce_candidate_bank(temporary, **arguments)
                self.assertIsNone(receipt["failure"])
                self.assertEqual(receipt["status"], "masked_only")
                self.assertEqual(receipt["known_pairs"], {})
                self.assertTrue((arguments["output"] / "task-0-stderr.bin").is_file())
                if mode != "missing":
                    self.assertTrue((arguments["output"] / "task-0-stdout.bin").is_file())
                targets = json.loads((arguments["output"] / "pair-targets.json").read_bytes())["targets"]
                self.assertEqual(len(targets), 1)
                self.assertFalse(targets[0]["mask"])

    def test_overflow_timeout_kill_reap_and_pipe_prefix_limits(self):
        for mode in ("overflow", "timeout"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = CandidateFixture(temporary, roles=("proposer",))
                binary = Path(temporary) / "synthetic-image"
                binary.write_bytes(fixture.binary)
                source = Path(temporary) / "synthetic-source"
                source.write_bytes(fixture.source)
                request = json.loads(fixture.executions["proposer-0"]["request"])
                request["max_wall_time_ms"] = 100
                proc = SyntheticProcess(fixture, mode=mode)
                with patch.object(producer.subprocess, "Popen", return_value=proc):
                    execution = producer._launch_candidate(binary, raw(request), expected_binary=candidate.byte_pin(fixture.binary),
                                      source_path=source, expected_source=candidate.byte_pin(fixture.source),
                                      platform="windows", overall_deadline=time.monotonic() + 1, cancel_file=None)
                self.assertTrue(proc.killed)
                self.assertTrue(execution["diagnostic"]["reaped"])
                self.assertTrue(execution["diagnostic"]["pipes_finished"])
                self.assertLessEqual(len(execution["receipt"] or b""), request["max_output_bytes"])
                self.assertLess(execution["diagnostic"]["elapsed_ms"], 500)  # No fixed 2-second grace.
                self.assertTrue(execution["diagnostic"]["overflow"] if mode == "overflow" else execution["diagnostic"]["timed_out"])

    def test_posix_owned_group_is_killed_after_leader_exit_before_pipe_join(self):
        if producer.os.name != "posix":
            self.skipTest("POSIX-only owned process-group source fixture")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary, roles=("proposer",))
            arguments = _arguments(fixture)
            proc = SyntheticProcess(fixture)
            descendant_gone = threading.Event()
            ordinary_group_kill = proc.kill_group

            def kill_group():
                ordinary_group_kill()
                descendant_gone.set()

            proc.kill_group = kill_group
            for pipe in (proc.stdout, proc.stderr):
                ordinary_read = pipe.read

                def inherited_pipe_read(count, read=ordinary_read):
                    block = read(count)
                    if not block:
                        # A synthetic descendant holds EOF until group kill.
                        if not descendant_gone.wait(timeout=1):
                            raise OSError("synthetic inherited pipe not closed")
                    return block

                pipe.read = inherited_pipe_read
            with patch.object(producer.subprocess, "Popen", return_value=proc) as launch:
                execution = producer._launch_candidate(arguments["cpu_binary_path"], fixture.executions["proposer-0"]["request"],
                      expected_binary=candidate.byte_pin(fixture.binary), source_path=arguments["checker_source_path"],
                      expected_source=candidate.byte_pin(fixture.source), platform="windows",
                      overall_deadline=time.monotonic() + 2, cancel_file=None)
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            self.group_signals.assert_any_call(proc.pid, producer.signal.SIGKILL)
            self.assertTrue(descendant_gone.is_set())
            self.assertFalse(proc.killed)  # The leader had already returned 0.
            self.assertEqual(execution["exit_code"], 0)
            self.assertTrue(execution["diagnostic"]["reaped"])
            self.assertTrue(execution["diagnostic"]["pipes_finished"])
            self.assertTrue(execution["diagnostic"]["owned_group_absent"])
            self.assertIsNone(execution["diagnostic"]["transport_failure"])

    def test_windows_direct_child_supervision_never_claims_owned_group_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary, roles=("proposer",))
            proc = SyntheticProcess(fixture, mode="timeout")
            request = json.loads(fixture.executions["proposer-0"]["request"])
            request["max_wall_time_ms"] = 100
            with patch.object(producer, "os", SimpleNamespace(name="nt")), \
                    patch.object(producer, "_hash_image", side_effect=lambda path, expected, deadline: _image_observation(expected)), \
                    patch.object(producer.subprocess, "Popen", return_value=proc) as launch:
                execution = producer._launch_candidate(Path(temporary) / "synthetic-image", raw(request),
                      expected_binary=candidate.byte_pin(fixture.binary), source_path=Path(temporary) / "synthetic-source",
                      expected_source=candidate.byte_pin(fixture.source), platform="windows",
                      overall_deadline=time.monotonic() + 1, cancel_file=None)
            self.assertFalse(launch.call_args.kwargs["start_new_session"])
            self.group_signals.assert_not_called()
            self.assertTrue(proc.killed)
            self.assertTrue(execution["diagnostic"]["reaped"])
            self.assertEqual(execution["diagnostic"]["process_supervision_scope"], "windows_direct_child_only_no_job_object")
            self.assertIsNone(execution["diagnostic"]["owned_group_absent"])

    def test_cancel_is_safe_existence_only_and_never_spawns(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary, roles=("proposer",))
            arguments = _arguments(fixture)
            cancel = Path(temporary) / "cancel.flag"
            cancel.touch()
            arguments["cancel_file"] = cancel
            with patch.object(producer.sys, "platform", "win32"), patch.object(producer.subprocess, "Popen") as launch:
                with self.assertRaises(InterruptedError):
                    producer.produce_candidate_bank(temporary, **arguments)
                launch.assert_not_called()
            arguments["cancel_file"] = Path(temporary) / ".env"
            with self.assertRaisesRegex(ValueError, "secret"):
                producer.produce_candidate_bank(temporary, **arguments)

    def test_linux_actual_pid_observation_failure_does_not_use_rust_self_assertion(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary, roles=("proposer",))
            proc = SyntheticProcess(fixture)
            seen = []

            def hash_image(path, expected, deadline, *, loaded_pid=None):
                seen.append(loaded_pid)
                if loaded_pid is not None:
                    raise OSError("synthetic proc inode unavailable")
                return _image_observation(expected)

            with patch.object(producer.subprocess, "Popen", return_value=proc), patch.object(producer, "_hash_image", side_effect=hash_image):
                execution = producer._launch_candidate(Path(temporary) / "synthetic-image", fixture.executions["proposer-0"]["request"],
                        expected_binary=candidate.byte_pin(fixture.binary), source_path=Path(temporary) / "synthetic-source",
                        expected_source=candidate.byte_pin(fixture.source), platform="linux",
                        overall_deadline=time.monotonic() + 2, cancel_file=None)
            self.assertIn(proc.pid, seen)
            self.assertEqual(execution["diagnostic"]["loaded_executable"]["status"], "failed")
            self.assertTrue(execution["diagnostic"]["reaped"])
            self.assertIsNone(execution["receipt"])

    def test_plan_fsync_failure_prevents_any_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary)
            arguments = _arguments(fixture)
            with patch.object(producer.sys, "platform", "win32"), patch.object(producer.os, "fsync", side_effect=OSError("synthetic fsync failure")), \
                    patch.object(producer.subprocess, "Popen") as launch:
                with self.assertRaises(OSError):
                    producer.produce_candidate_bank(temporary, **arguments)
                launch.assert_not_called()

    def test_whole_original_allowance_covers_strict_input_before_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary)
            arguments = _arguments(fixture)
            with patch.object(producer.subprocess, "Popen") as launch:
                with self.assertRaises(TimeoutError):
                    producer.produce_candidate_bank(temporary, _started=time.monotonic() - 100, **arguments)
                launch.assert_not_called()

    def test_unknown_child_cleanup_blocks_every_later_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary)
            arguments = _arguments(fixture)
            calls = []

            def unresolved(binary, request_raw, **options):
                calls.append(request_raw)
                task = candidate._json(request_raw)["task_id"]
                return {"request": request_raw, "receipt": None, "stderr": b"", "spawned": True, "exit_code": None,
                        "diagnostic": {"schema": producer.SCHEMA, "task_id": task, "pid": 4242,
                                       "reaped": False, "pipes_finished": False, "transport_failure": "synthetic cleanup unknown",
                                       "post_binary": {"status": "checked"}, "loaded_executable": {"status": "unknown"},
                                       "elapsed_ms": 12, "timed_out": True, "canceled": False}}

            with patch.object(producer.sys, "platform", "win32"), patch.object(producer, "_launch_candidate", side_effect=unresolved):
                receipt = producer.produce_candidate_bank(temporary, **arguments)
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipt["status"], "rejected")
            self.assertEqual(receipt["failure"], "child_or_pipe_cleanup_unknown")
            self.assertEqual(receipt["known_pairs"], {})
            self.assertTrue((arguments["output"] / "task-0-process.json").is_file())

    def test_group_absence_unknown_cannot_publish_a_completed_positive_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CandidateFixture(temporary)
            arguments = _arguments(fixture)
            calls = []

            def unresolved_group(binary, request_raw, **options):
                task = candidate._json(request_raw)["task_id"]
                calls.append(task)
                return {"request": request_raw, "receipt": fixture.executions[task]["receipt"], "stderr": b"",
                        "spawned": True, "exit_code": 0,
                        "diagnostic": {"schema": producer.SCHEMA, "task_id": task, "pid": 4242,
                                       "reaped": True, "pipes_finished": True, "owned_group_absent": False,
                                       "process_supervision_scope": "posix_owned_process_group",
                                       "transport_failure": None, "post_binary": {"status": "checked"},
                                       "post_source": {"status": "checked"}, "binary_path_stable": True,
                                       "source_path_stable": True, "loaded_executable": {"status": "unknown"},
                                       "elapsed_ms": 12, "timed_out": False, "canceled": False}}

            with patch.object(producer.sys, "platform", "win32"), \
                    patch.object(producer, "_launch_candidate", side_effect=unresolved_group):
                receipt = producer.produce_candidate_bank(temporary, **arguments)
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipt["status"], "rejected")
            self.assertEqual(receipt["failure"], "child_or_pipe_cleanup_unknown")
            self.assertEqual(receipt["known_pairs"], {})
            overlay = json.loads((arguments["output"] / "comparative-overlay.json").read_bytes())["overlay"]
            self.assertTrue(all(pair["preference"] == "masked" for pair in overlay["pairs"]))
            self.assertTrue(all(check["completion"] != "completed" for pair in overlay["pairs"] for check in pair["checks"]))
            self.assertTrue((arguments["output"] / "task-0-stdout.bin").is_file())


if __name__ == "__main__":
    unittest.main()
