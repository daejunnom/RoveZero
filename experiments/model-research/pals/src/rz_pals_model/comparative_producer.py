"""Bounded, independently pinned Own CPU candidate child producer.

This is a functional CPU evidence path, not a speed/strength/model/training
benchmark. The caller supplies prior registration/source/binary/criterion pins
and explicit current P/C selections. No producer is inferred from a CPU teacher.
Before any fresh child, file fsync publishes the fixed criterion and result-free
plan. Actual raw pipes and caller observations feed the existing strict receipt
adapter; metadata declarations alone cannot admit an ordinal preference.

Only stdlib is imported at startup. Strict parent/receipt modules are imported
inside the original overall allowance. No model, optimizer, backward or GPU is
invoked. Tests may mock OS process observations; they are not actual CPU proof.
POSIX supervision covers the newly owned process group. Windows supervision is
limited to the direct child here; no Job Object or descendant-group claim is made.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

SCHEMA = "rz-pals-owned-candidate-producer/1"
SELECTION_SCHEMA = "rz-pals-owned-candidate-pair-selections/1"
MAX_PAIRS = 64
MAX_BYTES = 128 * 1024 * 1024
POST_RESERVE = 8 * 1024 * 1024
TASK_RESERVE = 16384


def _modules():
    from . import comparative_training as candidate
    from . import frozen_producer as frozen
    from . import training
    from . import preparation_check as preparation
    return candidate, frozen, training, preparation


def _raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _positive(value, maximum, name):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("finite producer limit: " + name)


class _Budget:
    def __init__(self, *, started, max_wall_time_ms, max_output_bytes, cancel_file):
        _positive(max_wall_time_ms, 300000, "wall")
        _positive(max_output_bytes, MAX_BYTES, "output")
        self.started, self.deadline = started, started + max_wall_time_ms / 1000
        # Child cleanup has its own prepaid slice inside dispatch_deadline;
        # durable raw output/final admission spends the remaining original
        # overall allowance, never a new grace deadline.
        self.publication_reserve_ms = min(max_wall_time_ms // 10, 5000)
        self.dispatch_deadline = self.deadline - self.publication_reserve_ms / 1000
        self.maximum, self.reserved = max_output_bytes, 0
        self.cancel_file = cancel_file

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("original producer wall allowance expired")
        if self.cancel_file is not None and self.cancel_file.exists():
            raise InterruptedError("producer canceled")

    def reserve(self, count):
        if type(count) is not int or count < 0 or self.reserved + count > self.maximum:
            raise ValueError("producer output reservation before dispatch exceeds allowance")
        self.reserved += count


class _Bank:
    def __init__(self, output, budget):
        _, _, _, preparation = _modules()
        self.root = preparation._registered_path(output)
        source = Path(__file__).resolve()
        repository = next((path for path in source.parents if (path / ".git").exists()), None)
        if repository is not None and (self.root == repository or repository in self.root.parents):
            raise ValueError("candidate evidence bank must be outside Git")
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise ValueError("candidate evidence bank must be fresh; no overwrite/resume")
        self.budget, self.written, self.assets = budget, 0, {}
        self.directory_sync = "directory_fsync" if os.name == "posix" else "file_fsync_directory_sync_unavailable"

    def write(self, name, raw):
        candidate, _, _, _ = _modules()
        if not isinstance(raw, bytes) or Path(name).name != name or "/" in name or "\\" in name or name in self.assets:
            raise ValueError("unique direct-child candidate artifact required")
        if self.written + len(raw) > self.budget.reserved:
            raise ValueError("candidate artifact exceeds prepaid output allowance")
        if time.monotonic() >= self.budget.deadline:
            raise TimeoutError("original allowance expired before durable artifact")
        self.written += len(raw)  # Partial writes stay charged; never retry.
        with (self.root / name).open("xb", buffering=0) as stream:
            view = memoryview(raw)
            while view:
                if time.monotonic() >= self.budget.deadline:
                    raise TimeoutError("original allowance expired during bounded durable write")
                count = stream.write(view[:65536])
                if not count:
                    raise OSError("candidate durable write made no progress")
                view = view[count:]
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix":
            descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self.assets[name] = candidate.byte_pin(raw)
        if time.monotonic() >= self.budget.deadline:
            raise TimeoutError("original allowance expired during artifact fsync")
        return self.assets[name]


def _hash_image(path, expected, deadline, *, loaded_pid=None):
    """Stable regular-file hash; proc symlink is allowed only for this child PID."""
    candidate, _, _, preparation = _modules()
    if loaded_pid is None:
        path = preparation._registered_path(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("registered executable must be a regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    else:
        if type(loaded_pid) is not int or loaded_pid <= 0 or not sys.platform.startswith("linux"):
            raise ValueError("loaded executable observation requires an actual Linux child PID")
        path, before, flags = Path("/proc") / str(loaded_pid) / "exe", None, os.O_RDONLY
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if not stat.S_ISREG(opened.st_mode) or not 1 <= opened.st_size <= MAX_BYTES or (before is not None and identity(before) != identity(opened)):
            raise ValueError("executable changed or exceeds bounded regular-file admission")
        digest, total = hashlib.sha256(), 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("original allowance expired hashing registered executable")
            block = os.read(descriptor, min(65536, opened.st_size - total + 1))
            if not block:
                break
            total += len(block)
            if total > opened.st_size:
                raise ValueError("executable grew while hashing")
            digest.update(block)
        after = os.fstat(descriptor)
        if identity(after) != identity(opened) or total != opened.st_size or (before is not None and identity(path.lstat()) != identity(opened)):
            raise ValueError("executable changed while hashing")
        pin = {"bytes": total, "sha256": digest.hexdigest()}
        if pin != expected:
            raise ValueError("actual executable differs from independently registered byte pin")
        return {"status": "checked", "artifact": pin, "device": str(opened.st_dev), "inode": str(opened.st_ino),
                "mtime_ns": str(opened.st_mtime_ns), "ctime_ns": str(opened.st_ctime_ns),
                "scope": "caller_child_loaded_inode" if loaded_pid is not None else "caller_registered_path_bytes"}
    finally:
        os.close(descriptor)


def _same_file_window(before, after):
    return (isinstance(before, dict) and isinstance(after, dict) and before.get("status") == after.get("status") == "checked"
            and all(before.get(name) == after.get(name) for name in ("artifact", "device", "inode", "mtime_ns", "ctime_ns")))


def _launch_candidate(binary, request_raw, *, expected_binary, source_path, expected_source,
                      platform, overall_deadline, cancel_file):
    """One fresh child; concurrent capped pipes, prepaid cleanup, no grace reset."""
    candidate, _, _, _ = _modules()
    request = candidate._json(request_raw)
    started = time.monotonic()
    deadline = min(overall_deadline, started + request["max_wall_time_ms"] / 1000)
    remaining = max(0, deadline - started)
    cleanup_reserve = min(request["max_wall_time_ms"] / 10000, 1.0, remaining / 4)
    transport_deadline = deadline - cleanup_reserve
    cap = request["max_output_bytes"]
    capture = {"stdout": bytearray(), "stderr": bytearray()}
    counts = {"stdout": 0, "stderr": 0}
    errors, overflow = [], threading.Event()
    proc, workers, exit_code = None, [], None
    diagnostic = {"schema": SCHEMA, "task_id": request["task_id"], "pid": None,
                  "loaded_executable": {"status": "unknown", "scope": "not_observed"},
                  "pre_binary": None, "post_binary": None, "pre_source": None, "post_source": None, "transport_failure": None,
                  "process_supervision_scope": "posix_owned_process_group" if os.name == "posix" else "windows_direct_child_only_no_job_object",
                  "group_kill_status": "not_attempted", "owned_group_absent": None,
                  "timed_out": False, "canceled": False, "overflow": False,
                  "reaped": False, "pipes_finished": False, "cleanup_reserve_ms": math.floor(cleanup_reserve * 1000)}

    def drain(pipe, name):
        try:
            while True:
                block = pipe.read(32768)
                if not block:
                    return
                counts[name] += len(block)
                keep = min(len(block), cap - len(capture[name]))
                capture[name].extend(block[:keep])
                if keep != len(block):
                    overflow.set()
        except (OSError, ValueError) as error:
            errors.append(type(error).__name__ + ":" + str(error)[:160])
        finally:
            pipe.close()

    def send():
        try:
            view = memoryview(request_raw)
            while view:
                count = proc.stdin.write(view)
                if not count:
                    raise OSError("candidate stdin made no progress")
                view = view[count:]
            proc.stdin.flush()
        except (OSError, ValueError) as error:
            errors.append(type(error).__name__ + ":" + str(error)[:160])
        finally:
            proc.stdin.close()

    try:
        if time.monotonic() >= transport_deadline:
            raise TimeoutError("insufficient original allowance before child")
        diagnostic["pre_binary"] = _hash_image(binary, expected_binary, transport_deadline)
        diagnostic["pre_source"] = _hash_image(source_path, expected_source, transport_deadline)
        if cancel_file is not None and cancel_file.exists():
            raise InterruptedError("candidate canceled before spawn")
        proc = subprocess.Popen([str(binary), "--candidate-only"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0, shell=False,
                                start_new_session=os.name == "posix",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        diagnostic["pid"] = proc.pid
        # Rust is waiting for EOF on stdin. Observe this actual child image
        # before releasing the request; never substitute Rust's self assertion.
        if platform == "linux":
            try:
                diagnostic["loaded_executable"] = _hash_image(None, expected_binary, transport_deadline, loaded_pid=proc.pid)
            except (OSError, ValueError, TimeoutError) as error:
                diagnostic["loaded_executable"] = {"status": "failed", "scope": "caller_child_loaded_inode", "error": str(error)[:256]}
                raise ValueError("independent loaded child identity unavailable") from error
        else:
            diagnostic["loaded_executable"] = {"status": "unknown", "scope": "platform_path_hash_only"}
        workers = [threading.Thread(target=drain, args=(proc.stdout, "stdout"), daemon=True),
                   threading.Thread(target=drain, args=(proc.stderr, "stderr"), daemon=True),
                   threading.Thread(target=send, daemon=True)]
        for worker in workers:
            worker.start()
        while True:
            if overflow.is_set():
                raise ValueError("candidate pipe output overflow")
            if cancel_file is not None and cancel_file.exists():
                raise InterruptedError("candidate canceled")
            left = transport_deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("candidate transport allowance expired with prepaid cleanup")
            try:
                exit_code = proc.wait(timeout=min(left, 0.01))
                diagnostic["reaped"] = True
                break
            except subprocess.TimeoutExpired:
                pass
    except (OSError, ValueError, TimeoutError, InterruptedError) as error:
        diagnostic["transport_failure"] = type(error).__name__ + ":" + str(error)[:256]
        diagnostic["timed_out"] = isinstance(error, TimeoutError)
        diagnostic["canceled"] = isinstance(error, InterruptedError)
    finally:
        if proc is not None:
            # The child created a new POSIX session, so this pgid belongs to
            # this launch. Kill the entire owned group even if the leader has
            # already exited: a descendant may still hold a pipe. Windows has
            # no Job Object here and explicitly claims direct-child scope only.
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    diagnostic["group_kill_status"] = "signal_sent"
                except ProcessLookupError:
                    diagnostic["group_kill_status"] = "already_absent"
                except OSError as error:
                    diagnostic["group_kill_status"] = "failed"
                    errors.append("owned_group_kill:" + type(error).__name__)
            if not diagnostic["reaped"]:
                try:
                    if os.name != "posix" or diagnostic["group_kill_status"] == "failed":
                        proc.kill()
                    left = max(0, deadline - time.monotonic())
                    exit_code = proc.wait(timeout=left)
                    diagnostic["reaped"] = True
                except (OSError, subprocess.TimeoutExpired) as error:
                    errors.append("kill_reap:" + type(error).__name__)
            # Every join spends only the original remaining allowance. A
            # surviving worker is explicit rejected evidence, never success.
            for worker in workers:
                worker.join(timeout=max(0, deadline - time.monotonic()))
            diagnostic["pipes_finished"] = all(not worker.is_alive() for worker in workers)
            if os.name == "posix":
                while True:
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        diagnostic["owned_group_absent"] = True
                        break
                    except OSError as error:
                        errors.append("owned_group_probe:" + type(error).__name__)
                        diagnostic["owned_group_absent"] = False
                        break
                    left = deadline - time.monotonic()
                    if left <= 0:
                        diagnostic["owned_group_absent"] = False
                        break
                    time.sleep(min(left, 0.005))
            if not workers:
                for pipe in (proc.stdin, proc.stdout, proc.stderr):
                    pipe.close()
        try:
            diagnostic["post_binary"] = _hash_image(binary, expected_binary, deadline)
        except (OSError, ValueError, TimeoutError) as error:
            diagnostic["post_binary"] = {"status": "failed", "error": str(error)[:256]}
        try:
            diagnostic["post_source"] = _hash_image(source_path, expected_source, deadline)
        except (OSError, ValueError, TimeoutError) as error:
            diagnostic["post_source"] = {"status": "failed", "error": str(error)[:256]}
        diagnostic["binary_path_stable"] = _same_file_window(diagnostic["pre_binary"], diagnostic["post_binary"])
        diagnostic["source_path_stable"] = _same_file_window(diagnostic["pre_source"], diagnostic["post_source"])
        diagnostic["overflow"] = overflow.is_set()
        diagnostic["errors"] = errors[:8]
        diagnostic["pipe_bytes_observed"] = counts
        diagnostic["pipe_prefix_bytes_retained"] = {name: len(data) for name, data in capture.items()}
        diagnostic["elapsed_ms"] = math.ceil(max(0, time.monotonic() - started) * 1000)
    return {"request": request_raw, "receipt": bytes(capture["stdout"]) or None,
            "stderr": bytes(capture["stderr"]), "diagnostic": diagnostic, "exit_code": exit_code,
            "spawned": proc is not None}


def _context(parents, parent_artifacts, selections, checker, criterion_pin, recipe_id):
    candidate, frozen, training, _ = _modules()
    admission = parents.frozen_admission
    parent = {"receipt_artifact": admission["receipt"], "raw_dataset_sha256": admission["raw_dataset_sha256"],
              "split_sha256": admission["split_sha256"], "current_view_sha256": parents.current_view.sha256,
              "producer_roster_sha256": frozen.load_roster(parent_artifacts["producer-roster.json"])["sha256"],
              "producer_envelope_sha256": frozen.load_envelope(parent_artifacts["producer-envelope.json"])["sha256"]}
    by_input = {parents.records[index]["input"]["sha256"]: index for index in parents.current_view.current_indices}
    journals = {journal["sha256"]: journal["prepared"] for journal in
                (training._unique_json(raw.decode("utf-8")) for raw in parent_artifacts["producer-prepared.jsonl"].splitlines())}
    bindings = {entry["binding"]["input_sha256"]: entry for entry in admission["inputs"]}
    prepared, pairs, used = [], [], set()
    for selection in selections:
        training._fields(selection, ("pair_id", "input_sha256", "kind", "candidates", "task_ids"), "explicit current candidate selection")
        identity = training._sha(selection["input_sha256"])
        if identity not in by_input:
            raise ValueError("explicit candidate selection is not a checked current input")
        row = parents.records[by_input[identity]]
        snapshot = row["input"]["snapshot"]
        current = {"input_sha256": identity, "label_sha256": training.label_digest(row),
                   **{name: snapshot[name] for name in ("game_id", "role", "rules_state_sha256", "rules_history_sha256",
                      "encoding_sha256", "source", "frozen_epoch", "input_revision", "legal_moves")},
                   "side_to_move": "white" if snapshot["white_to_move"] else "black"}
        if identity not in used:
            entry = bindings[identity]
            journal = journals[entry["prepared_evidence_sha256"]]
            prepared.append({"current": current, "producer_id": entry["binding"]["producer_id"],
                             "producer_registration_sha256": entry["producer"]["registration_sha256"],
                             "capture_evidence_sha256": entry["prepared_evidence_sha256"],
                             **{name: journal[name] for name in ("input_json", "tensor_sidecar_json", "lineage_json")}})
            used.add(identity)
        tasks = selection["task_ids"]
        if not isinstance(tasks, list) or len(tasks) != 2:
            raise ValueError("two explicit candidate task IDs required")
        checks = [{"candidate": move, "task_id": task, "conditions": candidate.metadata.task_conditions(current, checker),
                   "restriction": {"kind": "candidate_only", "root_moves": [move]}, "completion": "unknown",
                   "completed_depth": 0, "evidence_id": None, "evidence_artifact": None}
                  for move, task in zip(selection["candidates"], tasks)]
        pairs.append({name: selection[name] for name in ("pair_id", "input_sha256", "kind", "candidates")}
                     | {"checks": checks, "preference": "masked"})
    body = candidate.metadata.normalize_overlay({"version": candidate.metadata.COMPARATIVE_DOMAIN, "parent": parent,
                                                "criterion": {"recipe_id": recipe_id, "recipe_sha256": criterion_pin["sha256"]},
                                                "checker": checker, "prepared_inputs": prepared, "pairs": pairs})
    candidate._parent_context(parents, parent_artifacts, body)
    all_tasks = [check["task_id"] for pair in body["pairs"] for check in pair["checks"]]
    if len(set(all_tasks)) != len(all_tasks):
        raise ValueError("candidate producer requires globally unique fresh task IDs")
    return body, by_input


def produce_candidate_bank(collection, *, strict_parent_options, checker_registration_path, checker_registration_sha256,
                           checker_source_path, checker_source_sha256, cpu_binary_path, cpu_binary_sha256,
                           criterion_path, criterion_sha256, selections_path, selections_sha256, output,
                           max_pairs, max_nodes, max_wall_time_ms, max_output_bytes, max_input_bytes,
                           per_check_wall_time_ms=10000, per_pipe_bytes=65536, cancel_file=None, _started=None,
                           _registration_set_pin=None):
    """Independent owner entry point; serial fresh children, no legacy fallback.

    Input pins must predate this invocation. The emitted expected-pins and bank
    receipt are the actual launch owner's observations; a later caller must pin
    them independently. Supplied source/build facts are registered declarations,
    not an inferred build attestation. Plan fsync is checked before each child.
    Output names are logical; no private machine path is embedded in source.
    """
    started = time.monotonic() if _started is None else _started
    budget = _Budget(started=started, max_wall_time_ms=max_wall_time_ms, max_output_bytes=max_output_bytes, cancel_file=None)
    candidate, _, training, preparation = _modules()
    budget.cancel_file = preparation._registered_path(cancel_file) if cancel_file is not None else None
    for name, value, maximum in (("pairs", max_pairs, MAX_PAIRS), ("nodes", max_nodes, (1 << 63) - 1),
                                 ("input", max_input_bytes, MAX_BYTES), ("child wall", per_check_wall_time_ms, 300000),
                                 ("pipe", per_pipe_bytes, 1048576)):
        _positive(value, maximum, name)
    if per_pipe_bytes < 1024:
        raise ValueError("candidate pipe allowance must support a bounded receipt")
    registration_set_bytes = 0
    if _registration_set_pin is not None:
        training._fields(_registration_set_pin, ("bytes", "sha256"), "actual registration set bytes")
        registration_set_bytes = training._uint(_registration_set_pin["bytes"], max_input_bytes)
        training._sha(_registration_set_pin["sha256"])
    budget.check()
    if registration_set_bytes >= max_input_bytes:
        raise ValueError("no remaining input allowance after prior registration set")
    reader = training._CollectionArtifactReader(max_input_bytes - registration_set_bytes)

    def asset(path, expected, maximum=candidate.MAX_JSON):
        budget.check()
        path = preparation._registered_path(path)
        raw = reader.read(path, maximum=maximum)
        if candidate.byte_pin(raw)["sha256"] != training._sha(expected):
            raise ValueError("actual producer input differs from independent prior SHA")
        budget.check()
        return path, raw

    _, registration_raw = asset(checker_registration_path, checker_registration_sha256)
    source_path, source_raw = asset(checker_source_path, checker_source_sha256)
    binary, binary_raw = asset(cpu_binary_path, cpu_binary_sha256, MAX_BYTES)
    _, criterion_raw = asset(criterion_path, criterion_sha256)
    _, selection_raw = asset(selections_path, selections_sha256)
    registration_value = candidate._json(registration_raw)
    checker = {**registration_value["checker"], "registration_sha256": checker_registration_sha256}
    prior_pins = {"registration": candidate.byte_pin(registration_raw), "source": candidate.byte_pin(source_raw), "binary": candidate.byte_pin(binary_raw)}
    registration = candidate._registered_checker(registration_raw, source_raw, binary_raw, prior_pins, checker)
    criterion_pin = candidate.byte_pin(criterion_raw)
    criterion = candidate._criterion(criterion_raw, criterion_pin)
    actual_platform = "linux" if sys.platform.startswith("linux") else "windows" if sys.platform == "win32" else "macos" if sys.platform == "darwin" else "unsupported"
    if registration["platform"] != actual_platform:
        raise ValueError("registered child platform differs from actual launcher")
    selections = training._fields(candidate._json(selection_raw), ("schema", "pairs"), "explicit pinned candidate selections")
    if selections["schema"] != SELECTION_SCHEMA or not isinstance(selections["pairs"], list) or not 1 <= len(selections["pairs"]) <= max_pairs:
        raise ValueError("candidate selection schema/finite pair count")
    options = dict(strict_parent_options)
    remaining_input = max_input_bytes - registration_set_bytes - reader.consumed
    if remaining_input <= 0:
        raise ValueError("no remaining input allowance before strict parent read")
    options["max_input_bytes"] = remaining_input
    parents = training.load_frozen_collected_dataset(collection, **options)
    budget.check()
    parent_bytes = preparation._frozen_input_bytes(Path(collection), options["producer_registrations"], parents.frozen_admission)
    if parent_bytes + reader.consumed + registration_set_bytes > max_input_bytes:
        raise ValueError("combined strict parent/registered checker input byte allowance")
    # This adapter needs exact parent byte buffers in addition to the strict
    # loader's retained records. Charge these additional reads/allocations from
    # the remaining combined allowance before touching any of those files.
    reread_allowance = remaining_input - parent_bytes
    if reread_allowance <= 0:
        raise ValueError("no remaining allocation allowance for exact prepared parent bytes")
    parent_reader = training._CollectionArtifactReader(reread_allowance)
    names = ("receipt.json", "producer-roster.json", "producer-envelope.json", "producer-prepared.jsonl",
             "inputs.jsonl", "native-inputs.jsonl", "input-lineage.jsonl")
    parent_artifacts = {name: parent_reader.read(Path(collection) / name, parents.frozen_admission["receipt"] if name == "receipt.json"
                                              else parents.frozen_admission["artifacts"][name]) for name in names}
    body, by_input = _context(parents, parent_artifacts, selections["pairs"], checker, criterion_pin, criterion["recipe_id"])
    if len(body["pairs"]) * 2 * checker["node_budget"] > max_nodes:
        raise ValueError("candidate node budget must be reserved before any dispatch")
    plan = {"schema": candidate.PLAN_SCHEMA, **{name: body[name] for name in ("parent", "criterion", "prepared_inputs")},
            "checker_namespace_sha256": candidate.metadata.checker_namespace(checker),
            "pairs": [{name: pair[name] for name in ("pair_id", "input_sha256", "kind", "candidates")}
                      | {"task_ids": [check["task_id"] for check in pair["checks"]]} for pair in body["pairs"]]}
    plan_raw = candidate.canonical_wire(plan)
    plan_pin = candidate.byte_pin(plan_raw)
    requests = {}
    for pair in body["pairs"]:
        snapshot = parents.records[by_input[pair["input_sha256"]]]["input"]["snapshot"]
        for check in pair["checks"]:
            request = {"schema": candidate.REQUEST_SCHEMA, "task_id": check["task_id"], "captured_input_sha256": pair["input_sha256"],
                       "checker_namespace_sha256": candidate.metadata.checker_namespace(checker), "before_result_anchor_sha256": plan_pin["sha256"],
                       **{name: snapshot[name] for name in ("frozen_epoch", "input_revision", "position_command", "rules_state_sha256",
                          "rules_history_sha256", "white_to_move")}, "expected_board_fen": snapshot["board_fen"],
                       "expected_legal_moves": snapshot["legal_moves"], "candidate": check["candidate"],
                       "cpu_binary_sha256": cpu_binary_sha256, "cpu_profile_sha256": registration["profile_sha256"],
                       "horizon": checker["horizon"], "node_budget": checker["node_budget"],
                       "tt_entries": registration["tt_entries"], "quiescence_ply": registration["quiescence_ply"],
                       "max_wall_time_ms": per_check_wall_time_ms, "max_output_bytes": per_pipe_bytes}
            request["context_sha256"] = candidate.wire_digest([candidate.REQUEST_SCHEMA, request])
            encoded = candidate.canonical_wire(request)
            candidate._request(encoded, parents.records[by_input[pair["input_sha256"]]], check["candidate"], check["task_id"], checker, registration, plan_pin["sha256"])
            requests[check["task_id"]] = encoded
    static = {"checker-registration.json": registration_raw, "checker-source.json": source_raw, "checker-binary.image": binary_raw,
              "ordinal-criterion.json": criterion_raw, "pair-plan.json": plan_raw, "candidate-selections.json": selection_raw}
    budget.reserve(sum(map(len, static.values())) + sum(map(len, requests.values()))
                   + len(requests) * (2 * per_pipe_bytes + TASK_RESERVE) + POST_RESERVE)
    budget.check()
    bank = _Bank(output, budget)
    for name, raw in static.items():
        bank.write(name, raw)
    for slot, (task, request_raw) in enumerate(requests.items()):
        bank.write(f"task-{slot}-request.json", request_raw)
    budget.check()  # All pre-result files completed fsync before first child.
    executions, execution_pins, execution_files, reasons = {}, {}, {}, {}
    failure, dispatch_blocked = None, False
    for slot, (task, request_raw) in enumerate(requests.items()):
        # Re-read exact durable anchors within the original allowance before
        # dispatch; no true flag substitutes for failed publication/byte pins.
        anchor_reader = training._CollectionArtifactReader(len(plan_raw) + len(criterion_raw) + len(request_raw))
        try:
            budget.check()
            if dispatch_blocked:
                raise ValueError("previous child cleanup unfinished; additional dispatch refused")
            if time.monotonic() >= budget.dispatch_deadline:
                raise TimeoutError("prepaid overall publication slice; no remaining dispatch allowance")
            anchor_reader.read(bank.root / "pair-plan.json", plan_pin)
            anchor_reader.read(bank.root / "ordinal-criterion.json", criterion_pin)
            anchor_reader.read(bank.root / f"task-{slot}-request.json", candidate.byte_pin(request_raw))
            execution = _launch_candidate(binary, request_raw, expected_binary=prior_pins["binary"], platform=actual_platform,
                                           source_path=source_path, expected_source=prior_pins["source"],
                                           overall_deadline=budget.dispatch_deadline, cancel_file=budget.cancel_file)
        except (ValueError, OSError, TimeoutError, InterruptedError) as error:
            execution = {"request": request_raw, "receipt": None, "stderr": b"", "exit_code": None, "spawned": False,
                         "diagnostic": {"schema": SCHEMA, "task_id": task, "pid": None, "reaped": False, "pipes_finished": False,
                                        "pre_binary": None, "post_binary": None, "pre_source": None, "post_source": None,
                                        "loaded_executable": {"status": "unknown"},
                                        "transport_failure": type(error).__name__ + ":" + str(error)[:256], "elapsed_ms": 0,
                                        "timed_out": isinstance(error, TimeoutError), "canceled": isinstance(error, InterruptedError)}}
        diagnostic = execution["diagnostic"]
        if execution["spawned"] and (not diagnostic["reaped"] or not diagnostic["pipes_finished"] or diagnostic.get("owned_group_absent") is False):
            dispatch_blocked = True
            failure = failure or "child_or_pipe_cleanup_unknown"
        observed_pins = {"request": candidate.byte_pin(request_raw), "receipt": None if execution["receipt"] is None else candidate.byte_pin(execution["receipt"]),
                         "stderr": candidate.byte_pin(execution["stderr"])}
        files = {"request": f"task-{slot}-request.json", "receipt": f"task-{slot}-stdout.bin" if execution["receipt"] is not None else None,
                 "stderr": f"task-{slot}-stderr.bin", "launch_observation": f"task-{slot}-launch.json"}
        if execution["receipt"] is not None:
            bank.write(files["receipt"], execution["receipt"])
        bank.write(files["stderr"], execution["stderr"])
        reason = diagnostic["transport_failure"]
        if diagnostic.get("overflow") or diagnostic.get("errors") or not diagnostic["pipes_finished"]:
            reason = reason or "pipe_completion_or_output_failure"
        if diagnostic.get("owned_group_absent") is False:
            reason = reason or "owned_process_group_cleanup_unknown"
        if diagnostic["post_binary"] is None or diagnostic["post_binary"].get("status") != "checked":
            reason = reason or "post_binary_identity_failed"
        if execution["spawned"] and (not diagnostic.get("binary_path_stable") or not diagnostic.get("source_path_stable")):
            reason = reason or "registered_binary_or_source_path_changed"
            failure = failure or reason
            dispatch_blocked = True
        if (diagnostic.get("post_source") or {}).get("status") == "failed":
            reason = reason or "registered_source_path_identity_failed"
            failure = failure or reason
            dispatch_blocked = True
        if execution["exit_code"] != 0 or not diagnostic["reaped"] or not execution["spawned"]:
            reason = reason or "child_failed_or_not_reaped"
        if reason and execution["exit_code"] == 0:
            # Preserve the actual exit code. A transport/binary failure cannot
            # be hidden by inventing a nonzero exit or a partial Rust receipt.
            failure = failure or reason
        observation = {"schema": candidate.LAUNCH_SCHEMA, "task_id": task, "registration_sha256": checker_registration_sha256,
                       "binary_sha256": cpu_binary_sha256, "platform": actual_platform, "binary_pin_scope": registration["binary_pin_scope"],
                       "before_result_anchor_sha256": plan_pin["sha256"], **observed_pins,
                       "assurance_scope": "independently_pinned_caller_observation", "anchor_durable_before_spawn": True,
                       "criterion_fixed_before_spawn": True, "spawned": execution["spawned"], "reaped": diagnostic["reaped"],
                       "exit_code": execution["exit_code"], "elapsed_ms": diagnostic["elapsed_ms"],
                       "timed_out": bool(diagnostic["timed_out"] or diagnostic.get("canceled"))}
        observation_raw = candidate.canonical_wire(observation)
        bank.write(files["launch_observation"], observation_raw)
        bank.write(f"task-{slot}-process.json", candidate.canonical_wire(diagnostic))
        executions[task] = {name: execution[name] for name in ("request", "receipt", "stderr")} | {"launch_observation": observation_raw}
        execution_pins[task] = observed_pins | {"launch_observation": candidate.byte_pin(observation_raw)}
        execution_files[task] = files
        request = candidate._json(request_raw)
        report = None
        if reason is None and execution["receipt"] is not None:
            try:
                report, reason = candidate._receipt(execution["receipt"], request, checker, registration,
                                                     caller_elapsed_ms=diagnostic["elapsed_ms"])
            except (ValueError, TypeError, KeyError) as error:
                reason, failure = "receipt_admission_failed", failure or str(error)[:256]
        reasons[task] = (report, reason or ("missing" if report is None else None))
        for pair in body["pairs"]:
            for check in pair["checks"]:
                if check["task_id"] != task:
                    continue
                if observed_pins["receipt"] is not None:
                    check.update(evidence_id=task, evidence_artifact=observed_pins["receipt"])
                if report is not None and reason is None:
                    check.update(completion="completed", completed_depth=report["completed_depth"])
                elif report is not None and reason in ("partial", "canceled"):
                    check.update(completion="partial" if reason == "partial" else "cancelled", completed_depth=report["completed_depth"])
                elif execution["receipt"] is None:
                    check.update(completion="missing", completed_depth=0)
    for pair in body["pairs"]:
        checked = [reasons[check["task_id"]] for check in pair["checks"]]
        if any(reason is not None for _, reason in checked):
            continue
        left, right = [report["raw_score"] for report, _ in checked]
        if (all(abs(score) <= criterion["score_limit"] and abs(score) < criterion["mate_threshold"] for score in (left, right))
                and abs(left - right) >= criterion["minimum_margin"]):
            pair["preference"] = "left" if left > right else "right"
    overlay_raw = candidate.canonical_wire(candidate.metadata.seal_overlay(body))
    bank.write("comparative-overlay.json", overlay_raw)
    expected_metadata = {name: body[name] for name in ("parent", "criterion", "checker", "prepared_inputs")}
    pins = {"overlay": candidate.byte_pin(overlay_raw), "criterion": criterion_pin, **prior_pins, "plan": plan_pin, "executions": execution_pins}
    manifest = {"schema": candidate.BANK_SCHEMA,
                "files": {"overlay": "comparative-overlay.json", "criterion": "ordinal-criterion.json", "registration": "checker-registration.json",
                          "source": "checker-source.json", "binary": "checker-binary.image", "plan": "pair-plan.json"}, "executions": execution_files}
    manifest_raw = candidate.canonical_wire(manifest)
    bank.write("candidate-bank.json", manifest_raw)
    bank.write("expected-pins.json", candidate.canonical_wire(pins))
    bank.write("metadata-pins.json", candidate.canonical_wire(expected_metadata))
    known, targets = {}, []
    try:
        budget.check()
        checked = candidate.admit_candidate_pairs(parents=parents, parent_artifacts=parent_artifacts, overlay_bytes=overlay_raw,
                   expected_metadata_pins=expected_metadata, criterion_bytes=criterion_raw, registration_bytes=registration_raw,
                   source_bytes=source_raw, binary_bytes=binary_raw, plan_bytes=plan_raw, executions=executions, independent_pins=pins)
        if failure is not None:
            raise ValueError("launch transport/identity failed: " + failure)
        targets = [asdict(target) for target in checked.targets]
        for target in checked.targets:
            if target.mask:
                known[target.role] = known.get(target.role, 0) + 1
    except (ValueError, TimeoutError, InterruptedError) as error:
        failure = failure or type(error).__name__ + ":" + str(error)[:512]
    bank.write("pair-targets.json", _raw({"schema": candidate.ADMISSION_SCHEMA, "targets": targets, "failure": failure}))
    receipt = {"schema": SCHEMA, "scope": "owned_cpu_candidate_functional_only",
               "status": "rejected" if failure else "completed_with_targets" if known else "masked_only",
               "failure": failure, "bank_artifact": candidate.byte_pin(manifest_raw), "parent": body["parent"],
               "pairs": len(body["pairs"]), "children_requested": len(requests),
               "children_spawned": sum(candidate._json(executions[task]["launch_observation"])["spawned"] for task in executions),
               "known_pairs": known, "reserved_nodes": len(requests) * checker["node_budget"],
               "reserved_output_bytes": budget.reserved, "written_output_bytes_before_receipt": bank.written,
               "publication_reserve_ms": budget.publication_reserve_ms,
               "elapsed_ms": math.ceil(max(0, time.monotonic() - started) * 1000),
               "artifacts": dict(bank.assets), "directory_sync_scope": bank.directory_sync,
               "independent_launch_pins": "produced_by_this_launch_owner; later_caller_must_pin_actual_receipt",
               "source_build_attestation": "registered_declaration_only_not_inferred_from_binary_hash",
               "rules_validation": "registered_rust_receipt_pv_attestation_not_python_rules_replay",
               "binary_pin_scope": registration["binary_pin_scope"], "platform": actual_platform,
               "process_supervision_scope": "posix_owned_process_group" if os.name == "posix" else "windows_direct_child_only_no_job_object",
               "producer_registration_set_artifact": _registration_set_pin,
               "actual_training_executed": False, "model_executed": False, "gpu_executed": False,
               "timing_comparison_admitted": False, "critic_repair_validity_admitted": False}
    encoded = _raw(receipt)
    bank.write("producer-receipt.json", encoded)
    return receipt


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection", "parent-receipt-sha256", "producer-registration-set", "producer-registration-set-sha256",
                 "checker-registration", "checker-registration-sha256", "checker-source", "checker-source-sha256",
                 "cpu-binary", "cpu-binary-sha256", "criterion", "criterion-sha256", "selections", "selections-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    for name in ("max-pairs", "max-nodes", "max-wall-time-ms", "max-output-bytes", "max-input-bytes"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--per-check-wall-time-ms", type=int, default=10000)
    parser.add_argument("--per-pipe-bytes", type=int, default=65536)
    parser.add_argument("--cancel-file")
    args = parser.parse_args(argv)
    _, _, _, preparation = _modules()
    registrations, set_pin = preparation._registration_set(args.producer_registration_set, args.producer_registration_set_sha256, args.max_input_bytes)
    receipt = produce_candidate_bank(args.collection, strict_parent_options={"expected_receipt_sha256": args.parent_receipt_sha256,
                   "producer_registrations": registrations}, checker_registration_path=args.checker_registration,
                   checker_registration_sha256=args.checker_registration_sha256, checker_source_path=args.checker_source,
                   checker_source_sha256=args.checker_source_sha256, cpu_binary_path=args.cpu_binary, cpu_binary_sha256=args.cpu_binary_sha256,
                   criterion_path=args.criterion, criterion_sha256=args.criterion_sha256, selections_path=args.selections,
                   selections_sha256=args.selections_sha256, output=args.output, max_pairs=args.max_pairs, max_nodes=args.max_nodes,
                   max_wall_time_ms=args.max_wall_time_ms, max_output_bytes=args.max_output_bytes, max_input_bytes=args.max_input_bytes,
                   per_check_wall_time_ms=args.per_check_wall_time_ms, per_pipe_bytes=args.per_pipe_bytes,
                   cancel_file=args.cancel_file, _started=started, _registration_set_pin=set_pin)
    print(_raw(receipt).decode("utf-8"))
    return 1 if receipt["failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
