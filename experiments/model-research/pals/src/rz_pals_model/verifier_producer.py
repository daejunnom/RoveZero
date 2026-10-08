"""Training-private frozen V selection and bounded own Rust CPU_T evidence.

Native/product V remains unsupported. This tool reuses verified native PUBLIC
board/record features under a distinct private query encoding, seals the fresh
V input before dispatch, and preserves CPU results as future evidence. Random
V logits select a task; they are never targets. One observed action supplies no
comparative preference rank, so seven-task loss remains explicitly masked.
Only current P/C parents are selected; all admitted raw parent history remains
preserved and checked independently of this selection view.
No optimizer is created, no backward runs, and no weights are updated.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
import time

import torch

from .artifacts import load_checkpoint
from .config import TASKS, matmul_flops
from .preparation_check import _parameter_digest
from .training import (EncodedSnapshot, TaskContext, ValidatedDataset,
                       load_collected_dataset, masked_losses, seal_snapshot)

SCHEMA = "rz-pals-private-verifier-producer/1"
CPU_SCHEMA = "rz-pals-private-cpu-task/1"
QUERY_SCHEMA = "rz-pals-private-verifier-query/1"
GAIN_SCHEMA = "rz-pals-observed-scope-progress/1"
QUERY_FIELDS = (
    *["eligible_" + name for name in TASKS], "baseline_depth_div_64",
    "requested_depth_div_64", "node_budget_log2_div_64", "wall_ms_div_300000",
    "budget_bucket_div_16", "public_records_div_128", "legal_moves_div_256",
    "unresolved_before_cpu", "reserved_zero",
)
PROFILE = "cpu-plan-assisted-conservative-v1"
RECHECK_PROFILE = "cpu-independent-conservative-v1"
CPU_SEARCH = "rz-cpu-pvs/0.1"
CPU_VALUE = "bootstrap-material-pst-v1"
LEGACY_ORDERING = "legacy_mvv_lva_v1"
SEE_ORDERING = "legal_see_v1"
SEE_ORDERING_IDENTITY = "cpu-ordering-rules-legal-target-exchange-v1"
SEE_CPU_SCHEMA = "rz-pals-private-cpu-task-legal-see/1"
SEE_CONDITIONS_SCHEMA = "rz-pals-private-cpu-conditions-legal-see/1"
SEE_CPU_SEARCH = "rz-cpu-pvs-legal-see/0.1"
SEE_CLI_ARGUMENT = "--cpu-ordering=legal-see-v1"
SEE_CONDITIONS_SUFFIX = (";ordering=" + SEE_ORDERING_IDENTITY +
    ";exchange:all-Rules-legal-recaptures-on-target,optional-stop-zero,material-only,P100-N320-B330-R500-Q900-K0,no-pruning,max-plies32,max-positions-per-order4096,typed-fail-on-exhaustion"
    ";ordering-priority:TT,nonnegative-exchange,quiet-history,negative-exchange"
    ";node-work:search+qsearch+exchange;exchange-cancel-deadline:original")
# Current generated gain scalars, seven-task/provenance label, and masked loss
# fields have separate finite allowances; append rechecks every sealed bound.
CPU_GAIN_OUTPUT_RESERVE = 4096
CPU_LABEL_OUTPUT_RESERVE = 8192
CPU_LOSS_OUTPUT_RESERVE = 4096
# Source declaration of the exact own PVS namespace, not a learned claim.
CPU_CONDITIONS = "iterative-deepening:1..requested;root-window:full-first,aspiration40-following,full-when-mate-or-fail-inclusive;pvs:first-full,following-zero-window,strict-interior-research;qsearch:tactical-capture-ep-promotion,all-check-evasions,no-check-standpat;q-limit:checked-abort;tt:direct-mapped,full-history-value-profile,equal-remaining-depth,completed-nodes-only;selectivity:no-reductions-no-nullmove;ties:Rules-order;score:side-to-move-raw"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(domain, value):
    return hashlib.sha256(_json([domain, value])).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256 identity")
    return value


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} exceeds finite integer admission")
    return value


def _parse(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _path(path):
    lexical = Path(path).expanduser().absolute()
    for part in lexical.parts:
        lower = part.lower()
        if lower == ".env" or lower.startswith(".env.") or any(marker in lower for marker in (
                "credential", "service-account", "service_account", "service account", "api_key", "api-key", "id_rsa", "id_ed25519")) or lower.endswith((".key", ".pem")):
            raise ValueError("secret source paths are prohibited")
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError("source/output links and junctions are prohibited")
    return lexical.resolve()


def _read_bounded(path, maximum):
    path = _path(path)
    if not path.is_file() or not 1 <= path.stat().st_size <= maximum:
        raise ValueError("source asset must be a bounded regular file")
    content = bytearray()
    with path.open("rb") as stream:
        while len(content) <= maximum:
            block = stream.read(min(65536, maximum - len(content) + 1))
            if not block:
                break
            content.extend(block)
    if len(content) > maximum:
        raise ValueError("source asset grew beyond registered byte admission")
    return path, bytes(content)


def _file(path, expected_sha256, maximum):
    path, content = _read_bounded(path, maximum)
    if hashlib.sha256(content).hexdigest() != _sha(expected_sha256):
        raise ValueError("independently registered source asset SHA mismatch")
    return path, content


def cpu_profile(tt_entries, requested_depth, quiescence_ply, profile=PROFILE):
    """Bridge and producer share this integer/string-only source declaration."""
    return {"domain": CPU_SCHEMA, "search": CPU_SEARCH, "evaluator": CPU_VALUE,
            "profile": profile, "tt_entries": tt_entries, "max_depth": requested_depth,
            "quiescence_ply": quiescence_ply, "selective_reductions": False}


def profile_sha256(configuration):
    return hashlib.sha256(_json(configuration)).hexdigest()


def _ordering_policy(value):
    if type(value) is not str or value not in (LEGACY_ORDERING, SEE_ORDERING):
        raise ValueError("unsupported explicit CPU ordering policy")
    return value


def cpu_profile_with_ordering(tt_entries, requested_depth, quiescence_ply, profile=PROFILE,
                              *, ordering_policy=LEGACY_ORDERING):
    """Opt-in S namespace; unchanged value semantics and legacy declaration."""
    selected = _ordering_policy(ordering_policy)
    value = cpu_profile(tt_entries, requested_depth, quiescence_ply, profile)
    if selected == SEE_ORDERING:
        value.update(domain=SEE_CPU_SCHEMA, search=SEE_CPU_SEARCH,
                     ordering_policy=SEE_ORDERING_IDENTITY)
    return value


def cpu_conditions_with_ordering(tt_entries, requested_depth, quiescence_ply, profile=PROFILE,
                                 *, ordering_policy=LEGACY_ORDERING):
    selected = _ordering_policy(ordering_policy)
    legacy = CPU_CONDITIONS + f";profile={profile};max_depth={requested_depth};q_plies={quiescence_ply};tt_entries={tt_entries}"
    return legacy + SEE_CONDITIONS_SUFFIX if selected == SEE_ORDERING else legacy


def _cpu_request_domain(request, ordering_policy):
    selected = _ordering_policy(ordering_policy)
    if selected == LEGACY_ORDERING:
        if request.get("schema") != CPU_SCHEMA or "ordering_policy" in request:
            raise ValueError("legacy CPU task requires original domain and omitted ordering marker")
        return CPU_SCHEMA
    if request.get("schema") != SEE_CPU_SCHEMA or request.get("ordering_policy") != SEE_ORDERING:
        raise ValueError("selected CPU CLI policy, task domain and sealed ordering marker differ")
    for key, profile in (("cpu_profile_sha256", PROFILE), ("recheck_profile_sha256", RECHECK_PROFILE)):
        expected = cpu_profile_with_ordering(request["tt_entries"], request["requested_depth"],
                    request["quiescence_ply"], profile, ordering_policy=selected)
        if request[key] != profile_sha256(expected):
            raise ValueError("selected CPU profile digest differs from fixed SEE declaration")
    raw = dict(request)
    context = raw.pop("context_sha256")
    if context != _hash(SEE_CPU_SCHEMA, raw):
        raise ValueError("selected CPU request context is not sealed in SEE domain")
    return SEE_CPU_SCHEMA


def cpu_bridge_arguments(binary, request, *, ordering_policy=LEGACY_ORDERING):
    """Literal argv policy agrees with the sealed request; no auto-detection."""
    _cpu_request_domain(request, ordering_policy)
    return [str(binary), SEE_CLI_ARGUMENT] if ordering_policy == SEE_ORDERING else [str(binary)]


def eligible_tasks(allowed, control, legal_moves=None, *, require_available=True):
    if not isinstance(allowed, (tuple, list)) or not allowed or len(set(allowed)) != len(allowed) or any(v not in TASKS for v in allowed):
        raise ValueError("explicit unique V task allowlist required")
    if set(control) != {"prefix", "root_moves"}:
        raise ValueError("private branch controls require exact prefix/root_moves fields")
    from .training import move_components
    for key in ("prefix", "root_moves"):
        if not isinstance(control[key], list) or len(control[key]) > (64 if key == "prefix" else 256):
            raise ValueError("private CPU control exceeds actual Rust capability")
        for movement in control[key]:
            move_components(movement)
        if key == "root_moves" and len(set(control[key])) != len(control[key]):
            raise ValueError("duplicate restricted root move")
    if legal_moves is not None and not control["prefix"] and any(move not in legal_moves for move in control["root_moves"]):
        raise ValueError("restricted root control not in captured Rules legal moves")
    widen = bool(not control["prefix"] and control["root_moves"] and legal_moves is not None and
                 set(control["root_moves"]) < set(legal_moves))
    empty = not control["prefix"] and not control["root_moves"]
    support = {"defend_response": bool(control["prefix"] and control["root_moves"]),
               "attack_repair": bool(control["prefix"] and not control["root_moves"]), "widen_responses": widen,
               "lower_selectivity": False, "resume_task": empty,
               "cross_profile_recheck": empty, "defer": empty}
    mask = [name in allowed and support[name] for name in TASKS]
    if require_available and not any(mask):
        raise ValueError("allowlist has no supported task under sealed controls")
    reasons = {name: ("not_allowed" if name not in allowed else
                     "reductions_already_disabled" if name == "lower_selectivity" else
                     "no_strict_root_response_subset" if name == "widen_responses" else
                     "missing_explicit_branch_controls") for name in TASKS if not support[name] or name not in allowed}
    return mask, reasons


def private_query(mask, snapshot, *, baseline_depth, requested_depth, max_nodes_per_check,
                  max_wall_time_ms, budget_bucket):
    """Versioned pre-result private task controls, never labels/hash features."""
    return tuple(float(v) for v in mask) + (
        baseline_depth / 64, requested_depth / 64,
        math.log2(max_nodes_per_check + 1) / 64, max_wall_time_ms / 300000,
        budget_bucket / 16, len(snapshot["public_records"]) / 128,
        len(snapshot["legal_moves"]) / 256, 1.0, 0.0,
    )


def select_task(logits, mask):
    if logits.shape != (1, len(TASKS)) or logits.device.type != "cpu" or logits.dtype != torch.float32 or not torch.all(torch.isfinite(logits)):
        raise ValueError("V task head must supply seven finite CPU FP32 logits")
    if len(mask) != len(TASKS) or any(type(v) is not bool for v in mask) or not any(mask):
        raise ValueError("task capability mask mismatch")
    selected = int(logits[0].masked_fill(~torch.tensor(mask, dtype=torch.bool), -torch.inf).argmax())
    return TASKS[selected]  # Declared vocabulary order only breaks exact ties.


def verifier_input(parent, encoding, source, context, query, encoding_sha256, frozen_epoch):
    snapshot = copy.deepcopy(parent["input"]["snapshot"])
    snapshot.update(role="verifier", source=copy.deepcopy(source),
                    encoding_sha256=encoding_sha256, frozen_epoch=frozen_epoch)
    sealed = {"snapshot": snapshot, "sha256": seal_snapshot(snapshot)}
    prepared = EncodedSnapshot(sealed["sha256"], encoding_sha256, encoding.board,
                               encoding.metadata, encoding.public_records, query, (), ((context, query),))
    return {"input": sealed, "future_label": None, "verifier_private": None}, prepared


class Limits:
    def __init__(self, *, max_games, max_steps, max_nodes, max_wall_time_ms, max_output_bytes,
                 max_forward_flops, cancel_file=None, clock=time.monotonic):
        for name, value, cap in (("games", max_games, 65536), ("steps", max_steps, 65536),
                                 ("nodes", max_nodes, 2**63 - 1), ("wall", max_wall_time_ms, 300000),
                                 ("output", max_output_bytes, 64 * 1024 * 1024),
                                 ("forward FLOPs", max_forward_flops, 2**63 - 1)):
            _integer(value, 1, cap, name)
        if max_output_bytes < 32768:
            raise ValueError("private producer needs bounded failure receipt reserve")
        self.maximum = {"games": max_games, "steps": max_steps, "nodes": max_nodes,
                        "output_bytes": max_output_bytes, "forward_matmul_flops": max_forward_flops}
        self.usage = dict.fromkeys(self.maximum, 0)
        self.max_wall_time_ms, self.clock = max_wall_time_ms, clock
        self.started = clock()
        self.cancel_file = _path(cancel_file) if cancel_file else None
        self.games = set()

    def check(self):
        elapsed = self.clock() - self.started
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed * 1000 >= self.max_wall_time_ms:
            raise TimeoutError("private verifier producer wall limit")
        if self.cancel_file is not None and self.cancel_file.exists():
            raise InterruptedError("private verifier producer canceled")
        return math.ceil(elapsed * 1000)

    def remaining_ms(self):
        return max(1, self.max_wall_time_ms - self.check())

    def charge(self, name, value):
        self.charge_many({name: value})

    def charge_many(self, charges):
        # Admit all dispatch costs before updating any counter or spawning.
        for name, value in charges.items():
            _integer(value, 0, 2**63 - 1, name)
            if name not in self.usage or self.usage[name] + value > self.maximum[name]:
                raise ValueError("private verifier producer resource limit: " + name)
        for name, value in charges.items():
            self.usage[name] += value

    def step(self, game):
        self.check()
        if game not in self.games:
            self.charge("games", 1)
            self.games.add(game)
        self.charge("steps", 1)


class _OutputCredit:
    """Finite output bytes already charged by one atomic admission."""
    def __init__(self, limits, maximum):
        self.limits, self.maximum, self.remaining = limits, maximum, maximum

    def consume(self, value):
        _integer(value, 0, self.remaining, "reserved output bytes")
        self.remaining -= value  # Keep attempted/partial writes charged.

    def release(self):
        self.limits.usage["output_bytes"] -= self.remaining
        self.remaining = 0


class PrivateBank:
    """Append-only separate pre-result input and post-result private artifacts."""
    FILES = ("inputs.jsonl", "encodings.jsonl", "decisions.jsonl", "cpu-evidence.jsonl",
             "future-labels.jsonl", "private-records.jsonl")
    FAILURE_FILES = ("failed-cpu-stdout.bin", "failed-cpu-stderr.bin")

    def __init__(self, directory, limits):
        self.root = _path(directory)
        source = Path(__file__).resolve()
        repository = next((p for p in source.parents if (p / ".git").exists()), None)
        if any((p / ".git").exists() for p in (self.root, *self.root.parents)) or (repository is not None and (self.root == repository or repository in self.root.parents)):
            raise ValueError("private training bank must be outside repository")
        if self.root.exists():
            raise FileExistsError("private bank must use a fresh directory")
        limits.check()
        self.root.mkdir(parents=True, exist_ok=False)
        self.limits, self.artifacts, self.hashers, self.file_identities = limits, {}, {}, {}
        # Reserve final receipt bytes; failed/partial inputs are never admitted.
        self.limits.charge("output_bytes", 16384)

    def reserve_cpu(self, request, row, private, context):
        """Reserve pipe, raw recovery and capped post-result files together."""
        self.limits.check()
        cap = _integer(request["max_output_bytes"], 1024, 1024 * 1024, "CPU response output")
        # Known envelopes are sized exactly. The only future payloads are the
        # response (capped below), bounded gain/label fields, and masked losses.
        envelopes = (
            {"evidence_sha256": "0" * 64, "input_sha256": row["input"]["sha256"],
             "context": asdict(context), "request": request, "response": None, "observed_information": None},
            {"input_sha256": row["input"]["sha256"], "future_label": None,
             "observed_information_evidence_sha256": "0" * 64},
            {"record": dict(row, verifier_private=private), "evidence_sha256": "0" * 64,
             "task_context": asdict(context), "loss": None, "comparative_training_target": False},
        )
        bounds = dict(zip(("cpu-evidence.jsonl", "future-labels.jsonl", "private-records.jsonl"),
                          (len(_json(envelopes[0])) + 1 + cap + CPU_GAIN_OUTPUT_RESERVE,
                           len(_json(envelopes[1])) + 1 + CPU_LABEL_OUTPUT_RESERVE,
                           len(_json(envelopes[2])) + 1 + CPU_LABEL_OUTPUT_RESERVE + CPU_LOSS_OUTPUT_RESERVE)))
        artifact_bytes = sum(bounds.values())
        self.limits.charge_many({"output_bytes": artifact_bytes + 2 * cap,
                                 "nodes": 2 * request["max_nodes_per_check"]})
        evidence, recovery, pipe = (_OutputCredit(self.limits, value) for value in (artifact_bytes, cap, cap))
        evidence.bounds = bounds
        return evidence, recovery, pipe

    def append(self, name, value, *, reservation=None):
        if name not in self.FILES:
            raise ValueError("unknown private artifact")
        self.limits.check()
        content = _json(value) + b"\n"
        if reservation is not None and len(content) > reservation.bounds.get(name, 0):
            raise ValueError("private CPU artifact exceeds sealed output reservation")
        self._append_bytes(name, content, reservation=reservation)

    def failed_cpu_bytes(self, name, content, *, reservation=None):
        if name not in self.FAILURE_FILES or not isinstance(content, bytes):
            raise ValueError("unknown private CPU failure artifact")
        # Failure recovery can retain bounded observed bytes after a deadline;
        # it can never produce an admitted complete receipt or task label.
        self._append_bytes(name, content, reservation=reservation)

    def _append_bytes(self, name, content, *, reservation=None):
        if reservation is None:
            self.limits.charge("output_bytes", len(content))
        else:
            if reservation.limits is not self.limits:
                raise ValueError("private output reservation belongs to another run")
            reservation.consume(len(content))
        path = self.root / name
        _path(path)
        first = name not in self.artifacts
        if not first and (not path.is_file() or path.stat().st_ino != self.file_identities[name] or path.stat().st_size != self.artifacts[name]["bytes"]):
            raise ValueError("private artifact ownership or bytes changed before append")
        with path.open("xb" if first else "ab") as stream:
            stream.write(content)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        hasher = self.hashers.setdefault(name, hashlib.sha256())
        hasher.update(content)
        self.file_identities[name] = path.stat().st_ino
        self.artifacts[name] = {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}

    def finish(self, receipt):
        receipt = dict(receipt, schema=SCHEMA, artifacts=self.artifacts,
                       actual_training_executed=False, backward_executed=False,
                       optimizer_created=False, optimizer_steps=0, product_verifier_enabled=False,
                       private_bank_only=True, external_teacher_used=False)
        content = _json(receipt) + b"\n"
        if len(content) > 16384:
            raise ValueError("private bank receipt exceeds reserved output bytes")
        with (self.root / "receipt.json").open("xb") as stream:
            stream.write(content)
        return {"receipt_sha256": hashlib.sha256(content).hexdigest(),
                "output": str(self.root), "complete": receipt["complete"]}


class CpuBridgeFailure(RuntimeError):
    def __init__(self, cause, capture):
        super().__init__(f"CPU_T {type(cause).__name__}: {str(cause)[:512]}")
        self.capture = {key: bytes(capture.get(key, b"")) for key in ("stdout", "stderr")}
        self.details = {"cause_type": type(cause).__name__, "exit_code": capture.get("exit_code"),
                        "spawned": capture.get("spawned", False),
                        "reaped": capture.get("reaped", False),
                        "stdout_bytes": len(self.capture["stdout"]), "stderr_bytes": len(self.capture["stderr"]),
                        "stdout_sha256": hashlib.sha256(self.capture["stdout"]).hexdigest(),
                        "stderr_sha256": hashlib.sha256(self.capture["stderr"]).hexdigest()}


def run_cpu_bridge(binary, request, limits, *, capture=None, pipe_reservation=None,
                   ordering_policy=LEGACY_ORDERING):
    capture = {} if capture is None else capture
    try:
        return _run_cpu_bridge(binary, request, limits, capture, pipe_reservation, ordering_policy=ordering_policy)
    except BaseException as error:
        raise CpuBridgeFailure(error, capture) from error


def _run_cpu_bridge(binary, request, limits, capture, pipe_reservation=None, *, ordering_policy=LEGACY_ORDERING):
    """One own no-child Rust worker; capped pipes, timeout/cancel kill and reap."""
    arguments = cpu_bridge_arguments(binary, request, ordering_policy=ordering_policy)
    raw = _json(request) + b"\n"
    if len(raw) > 512 * 1024:
        raise ValueError("CPU_T request exceeds finite stdin admission")
    limits.check()
    cap = request["max_output_bytes"]
    _integer(cap, 1024, 1024 * 1024, "CPU response output")
    fallback_recovery = None
    if pipe_reservation is None:
        # Preserve direct callers' existing two-cap pre-spawn admission. The
        # producer passes separate recovery credit that survives this return.
        limits.charge("output_bytes", 2 * cap)
        pipe_reservation = _OutputCredit(limits, cap)
        fallback_recovery = _OutputCredit(limits, cap)
    if pipe_reservation.limits is not limits or pipe_reservation.remaining != cap:
        raise ValueError("CPU_T pipe reservation differs from registered bound")
    started = time.monotonic()
    try:
        proc = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except BaseException:
        pipe_reservation.release()  # A rejected spawn produced no pipe bytes.
        if fallback_recovery is not None:
            fallback_recovery.release()
        raise
    capture["spawned"] = True
    output, errors, failures, lock = bytearray(), bytearray(), [], threading.Lock()
    capture.update(stdout=output, stderr=errors)

    def read(pipe, target):
        try:
            while True:
                block = pipe.read(4096)
                if not block:
                    return
                with lock:
                    if len(output) + len(errors) + len(block) > cap:
                        failures.append(ValueError("CPU_T output exceeded registered bound"))
                        return
                    target.extend(block)
        except BaseException as error:
            with lock:
                failures.append(error)

    def write():
        try:
            proc.stdin.write(raw)
            proc.stdin.close()
        except BaseException as error:
            with lock:
                failures.append(error)

    workers = [threading.Thread(target=read, args=(proc.stdout, output), daemon=True),
               threading.Thread(target=read, args=(proc.stderr, errors), daemon=True),
               threading.Thread(target=write, daemon=True)]
    response = None
    try:
        for worker in workers:
            worker.start()
        while proc.poll() is None:
            limits.check()
            if (time.monotonic() - started) * 1000 >= request["max_wall_time_ms"]:
                raise TimeoutError("CPU_T subprocess exceeded registered wall bound")
            with lock:
                if failures:
                    raise failures[0]
            time.sleep(0.005)
        for worker in workers:
            worker.join(timeout=1)
        if any(worker.is_alive() for worker in workers):
            raise RuntimeError("CPU_T pipes did not close after process completion")
        if failures:
            raise failures[0]
        if proc.returncode != 0:
            raise RuntimeError(f"CPU_T failed with exit {proc.returncode}; stderr_sha256=" + hashlib.sha256(errors).hexdigest())
        if errors:
            raise ValueError("CPU_T emitted unexpected diagnostic output")
        response = _parse(bytes(output))
        if len(_json(response)) > cap:
            raise ValueError("CPU_T canonical response exceeds registered output bound")
        limits.check()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)
        capture.update(exit_code=proc.returncode, reaped=True)
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=1)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            pipe.close()
        retained = len(output) + len(errors) if not failures and not any(worker.is_alive() for worker in workers) else cap
        pipe_reservation.consume(retained)  # Unknown pipe cost stays reserved.
        pipe_reservation.release()
        if fallback_recovery is not None:
            fallback_recovery.release()
    if (time.monotonic() - started) * 1000 >= request["max_wall_time_ms"]:
        raise TimeoutError("CPU_T task including spawn/parse/cleanup exceeded wall bound")
    limits.check()
    return response


def _record_producer_failure(bank, error, *, counts, checkpoint_sha256, cpu_binary_sha256,
                             capture=None, reservations=(), reserved_nodes=0, stage=None):
    """Best-effort bounded recovery must never replace the primary exception."""
    failure = {"type": type(error).__name__, "message": str(error)[:512]}
    evidence, recovery, pipe = reservations if reservations else (None, None, None)
    if isinstance(error, CpuBridgeFailure):
        # The wrapper froze the streams together with these hashes. A late
        # reader must not change the bytes later copied into failure evidence.
        capture = error.capture
    if capture is not None:
        failure["stage"] = stage
        failure["cpu_observation"] = (error.details if isinstance(error, CpuBridgeFailure)
                                      else CpuBridgeFailure(error, capture).details)
        if reserved_nodes and not failure["cpu_observation"]["spawned"]:
            bank.limits.usage["nodes"] -= reserved_nodes
        if pipe is not None and pipe.remaining and failure["cpu_observation"]["spawned"]:
            pipe.consume(pipe.remaining)  # Cleanup did not confirm pipe usage.
        retention_errors = []
        for stream in ("stdout", "stderr"):
            content = bytes(capture.get(stream, b""))
            if content:
                try:
                    bank.failed_cpu_bytes("failed-cpu-" + stream + ".bin", content, reservation=recovery)
                except BaseException as retention_error:
                    retention_errors.append({"stream": stream, "type": type(retention_error).__name__,
                                             "message": str(retention_error)[:512]})
        failure["cpu_raw_output_preserved"] = not retention_errors
        if retention_errors:
            failure["retention_failures"] = retention_errors
            error.__dict__.setdefault("preservation_failures", []).extend(
                dict(item, stage="cpu_raw_retention") for item in retention_errors)
    for reservation in (evidence, recovery, pipe):
        if reservation is not None:
            reservation.release()
    try:
        bank.finish({"complete": False, "failure": failure, "counts": counts,
                     "parameters_unchanged": None, "checkpoint_sha256": checkpoint_sha256,
                     "cpu_binary_sha256": cpu_binary_sha256,
                     "resources": {"maximum": bank.limits.maximum, "usage": bank.limits.usage}})
    except BaseException as receipt_error:
        # Keep a bounded secondary diagnostic on the exception even when disk
        # failure prevents publishing the failure receipt itself.
        diagnostic = {"stage": "failure_receipt", "type": type(receipt_error).__name__,
                      "message": str(receipt_error)[:512]}
        error.__dict__.setdefault("preservation_failures", []).append(diagnostic)
        error.__dict__["producer_failure"] = {"complete": False, "failure": failure,
                                               "counts": copy.deepcopy(counts),
                                               "resources": {"maximum": dict(bank.limits.maximum),
                                                             "usage": dict(bank.limits.usage)}}
        if hasattr(error, "add_note"):
            error.add_note("failure receipt could not be preserved: " + type(receipt_error).__name__)
    return failure


def _report(value, request, profile, *, after=False):
    """Historical legacy validation, preserved for feedback/utility consumers."""
    expected = cpu_conditions_with_ordering(request["tt_entries"], request["requested_depth"],
               request["quiescence_ply"], PROFILE if profile == request["cpu_profile_sha256"] else RECHECK_PROFILE)
    return _report_impl(value, request, profile, after=after,
                        conditions_schema="rz-pals-private-cpu-conditions/1",
                        search_version=CPU_SEARCH, expected_search=expected)


def report_with_ordering(value, request, profile, *, after=False, ordering_policy=LEGACY_ORDERING):
    _cpu_request_domain(request, ordering_policy)
    if ordering_policy == LEGACY_ORDERING:
        return _report(value, request, profile, after=after)
    if profile not in (request["cpu_profile_sha256"], request["recheck_profile_sha256"]):
        raise ValueError("unknown selected CPU profile")
    expected = cpu_conditions_with_ordering(request["tt_entries"], request["requested_depth"],
               request["quiescence_ply"], PROFILE if profile == request["cpu_profile_sha256"] else RECHECK_PROFILE,
               ordering_policy=ordering_policy)
    return _report_impl(value, request, profile, after=after,
                        conditions_schema=SEE_CONDITIONS_SCHEMA,
                        search_version=SEE_CPU_SEARCH, expected_search=expected)


def _report_impl(value, request, profile, *, after, conditions_schema, search_version, expected_search):
    required = {"profile_sha256", "conditions_sha256", "score_scope", "completed_depth", "requested_depth",
                "nodes", "quiescence_nodes", "completion", "root_restricted", "raw_score", "pv",
                "white_to_move", "elapsed_ms", "reused_completed_depth", "score_provenance",
                "conditions", "pv_rules_validated"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("CPU_T raw report fields differ from private bridge contract")
    if value["profile_sha256"] != profile:
        raise ValueError("CPU_T profile identity mismatch")
    conditions = value["conditions"]
    fields = {"schema", "rules_state_sha256", "rules_history_sha256", "board_fen", "profile_sha256",
              "value_identity", "search_version", "quiescence_ply", "root_moves", "white_to_move",
              "legal_moves", "root_order", "root_order_sha256", "search_conditions", "root_selection", "resource_policy"}
    if not isinstance(conditions, dict) or set(conditions) != fields or conditions["schema"] != conditions_schema:
        raise ValueError("CPU actual conditions descriptor missing")
    if hashlib.sha256(_json(conditions)).hexdigest() != _sha(value["conditions_sha256"]):
        raise ValueError("CPU actual conditions descriptor SHA mismatch")
    if conditions["profile_sha256"] != profile or conditions["search_version"] != search_version or conditions["quiescence_ply"] != request["quiescence_ply"] or conditions["value_identity"] != {"semantics": CPU_VALUE, "weights_sha256": None, "training": {"kind": "bootstrap"}}:
        raise ValueError("CPU search/evaluator/configuration conditions mismatch")
    if conditions["search_conditions"] != expected_search:
        raise ValueError("CPU actual PVS namespace differs from registered source semantics")
    if conditions["resource_policy"] != {"max_wall_time_ms": request["max_wall_time_ms"],
                                          "max_nodes_per_check": request["max_nodes_per_check"],
                                          "max_checks": 2,
                                          "search_deadline_reserve_ms": min(request["max_wall_time_ms"] // 10, 1000)}:
        raise ValueError("CPU actual resource policy differs from request admission")
    _sha(conditions["rules_state_sha256"])
    _sha(conditions["rules_history_sha256"])
    if not request["prefix"] and any(conditions[key] != request[key] for key in ("rules_state_sha256", "rules_history_sha256")):
        raise ValueError("CPU actual Rules branch differs from captured root")
    if not request["prefix"] and conditions["board_fen"] != request["expected_board_fen"]:
        raise ValueError("CPU branch FEN differs from captured root")
    expected_white = (request["expected_board_fen"].split()[1] == "w") != bool(len(request["prefix"]) % 2)
    if conditions["white_to_move"] is not expected_white or value["white_to_move"] is not expected_white:
        raise ValueError("CPU raw score viewpoint differs from actual prefix")
    from .training import move_components
    for key in ("legal_moves", "root_order"):
        moves = conditions[key]
        if not isinstance(moves, list) or len(moves) > 256 or len(set(moves)) != len(moves):
            raise ValueError("CPU ordered root moves invalid")
        for movement in moves:
            move_components(movement)
    if _hash("rz-pals-private-cpu-root-order/1", conditions["root_order"]) != conditions["root_order_sha256"]:
        raise ValueError("CPU root order digest mismatch")
    restricted = request["task"] == "defend_response" or (request["task"] == "widen_responses" and not after)
    if restricted:
        roots = conditions["root_moves"]
        if not isinstance(roots, list) or not roots or set(roots) != set(request["root_moves"]) or len(roots) != len(request["root_moves"]) or roots != conditions["root_order"] or roots != [m for m in conditions["legal_moves"] if m in roots]:
            raise ValueError("CPU actual root restriction differs from admitted task")
    elif conditions["root_moves"] is not None or conditions["root_order"] != conditions["legal_moves"]:
        raise ValueError("CPU task unexpectedly restricted root responses")
    if value["root_restricted"] is not restricted or value["pv_rules_validated"] is not True:
        raise ValueError("CPU root/PV Rules attestation missing")
    if conditions["root_selection"] != ("explicit_registered_controls" if restricted else "unrestricted"):
        raise ValueError("CPU actual root selection differs from sealed controls")
    scope = value["score_scope"]
    if scope not in ("frontier_only", "completed_iteration", "rules_terminal"):
        raise ValueError("unknown CPU score scope")
    depth = _integer(value["completed_depth"], 0, 64, "completed CPU depth")
    if (scope == "completed_iteration") != (depth > 0):
        raise ValueError("CPU scope/depth mismatch")
    _integer(value["requested_depth"], 1, 64, "CPU requested depth")
    if depth > value["requested_depth"]:
        raise ValueError("CPU completed depth exceeds request")
    _integer(value["nodes"], 0, request["max_nodes_per_check"], "CPU nodes")
    _integer(value["quiescence_nodes"], 0, value["nodes"], "CPU qnodes")
    _integer(value["elapsed_ms"], 0, request["max_wall_time_ms"], "CPU elapsed")
    _integer(value["reused_completed_depth"], 0, depth, "CPU resume depth")
    _integer(value["raw_score"], -(2**31), 2**31 - 1, "CPU raw scalar")
    if type(value["root_restricted"]) is not bool or type(value["white_to_move"]) is not bool:
        raise ValueError("CPU scope viewpoint/root flags")
    if value["completion"] not in ("depth_limit", "node_limit", "deadline", "canceled", "quiescence_limit", "rules_terminal"):
        raise ValueError("CPU completion unknown")
    if (scope == "rules_terminal") != (value["completion"] == "rules_terminal"):
        raise ValueError("Rules terminal is exclusive to actual Rules completion")
    if value["score_provenance"] != ("rz-position/rules-terminal" if scope == "rules_terminal" else CPU_VALUE):
        raise ValueError("CPU score provenance missing")
    if not isinstance(value["pv"], list) or len(value["pv"]) > value["requested_depth"] + request["quiescence_ply"]:
        raise ValueError("CPU PV exceeds bounded evidence")
    for move in value["pv"]:
        move_components(move)


def observed_gain(request, response):
    """Only new conditional completed scope/depth; no score/mate/nodes reward."""
    return _observed_gain_impl(request, response, _report)


def observed_gain_with_ordering(request, response, *, ordering_policy=LEGACY_ORDERING):
    """Fixed-ordering conditional scope progress; never an ordering A/B reward."""
    _cpu_request_domain(request, ordering_policy)
    if ordering_policy == LEGACY_ORDERING:
        return observed_gain(request, response)
    def validate(value, req, profile, *, after=False):
        return report_with_ordering(value, req, profile, after=after, ordering_policy=ordering_policy)
    return _observed_gain_impl(request, response, validate)


def _observed_gain_impl(request, response, report_validator):
    required = {"schema", "task", "context_sha256", "cpu_binary_sha256", "rules_state_sha256",
                "rules_history_sha256", "board_fen", "branch_sha256", "status", "reason",
                "baseline", "after", "nodes", "elapsed_ms", "resume_kind", "product_verifier_enabled", "deadline_exceeded"}
    if not isinstance(response, dict) or set(response) != required:
        raise ValueError("CPU_T bridge response fields mismatch")
    for key in ("schema", "task", "context_sha256", "cpu_binary_sha256", "rules_state_sha256", "rules_history_sha256"):
        expected = request[key] if key in request else None
        if response[key] != expected:
            raise ValueError("CPU_T verified request identity mismatch: " + key)
    if response["board_fen"] != request["expected_board_fen"] or response["branch_sha256"] != request["branch_sha256"] or response["product_verifier_enabled"] is not False:
        raise ValueError("CPU_T state/branch/private boundary mismatch")
    if response["status"] not in ("observed", "deferred", "unavailable"):
        raise ValueError("CPU_T execution status invalid")
    if type(response["deadline_exceeded"]) is not bool or response["deadline_exceeded"]:
        raise TimeoutError("CPU_T absolute deadline was exceeded")
    if (request["task"] == "defer") != (response["status"] == "deferred"):
        raise ValueError("defer must preserve no-action status")
    if request["task"] == "lower_selectivity" and response["status"] != "unavailable":
        raise ValueError("CPU without reductions cannot supply lower-selectivity observation")
    if response["status"] == "unavailable" and (not isinstance(response["reason"], str) or not response["reason"] or len(response["reason"]) > 256):
        raise ValueError("unavailable CPU task requires bounded actual reason")
    if response["resume_kind"] not in (None, "completed_iteration"):
        raise ValueError("CPU_T cannot claim stack restoration")
    _integer(response["elapsed_ms"], 0, request["max_wall_time_ms"], "CPU total elapsed")
    before, after = response["baseline"], response["after"]
    _integer(response["nodes"], 0, 2 * request["max_nodes_per_check"], "CPU total nodes")
    if response["status"] != "observed":
        if after is not None:
            raise ValueError("deferred/unavailable task must not invent after observation")
        if before is not None:
            if response["status"] != "unavailable":
                raise ValueError("only unavailability may preserve a baseline")
            report_validator(before, request, request["cpu_profile_sha256"])
            if before["requested_depth"] != request["baseline_depth"] or response["nodes"] != before["nodes"]:
                raise ValueError("unavailable resume baseline accounting mismatch")
        elif response["nodes"] != 0:
            raise ValueError("no-action task invented node cost")
        return {"schema": GAIN_SCHEMA, "status": response["status"], "baseline_observed": before is not None, "same_conditions": False,
                "new_completed_depth": 0, "uncertainty_before":
                ("finite_depth_estimate" if before and before["completed_depth"] > 0 else
                 "rules_terminal_current_state" if before and before["score_scope"] == "rules_terminal" else
                 "unresolved_frontier" if before else "unobserved"), "uncertainty_after": "unobserved",
                "comparative_preference_available": False, "preference_rank": None}
    base_profile = request["cpu_profile_sha256"]
    next_profile = (request["recheck_profile_sha256"] if request["task"] == "cross_profile_recheck" else base_profile)
    report_validator(before, request, base_profile)
    report_validator(after, request, next_profile, after=True)
    if before["reused_completed_depth"] != 0:
        raise ValueError("fresh baseline cannot claim existing resume state")
    if before["requested_depth"] != request["baseline_depth"] or after["requested_depth"] != request["requested_depth"]:
        raise ValueError("CPU_T report not from admitted depth request")
    if response["nodes"] != before["nodes"] + after["nodes"] or response["nodes"] > 2 * request["max_nodes_per_check"]:
        raise ValueError("CPU_T actual node accounting mismatch")
    if request["task"] == "resume_task" and response["resume_kind"] != "completed_iteration":
        raise ValueError("resume must use Rust-owned completed-iteration token")
    same = (before["conditions_sha256"] == after["conditions_sha256"] and
            before["profile_sha256"] == after["profile_sha256"] and
            before["white_to_move"] == after["white_to_move"] and
            before["root_restricted"] == after["root_restricted"])
    if request["task"] == "resume_task":
        if not same or before["completed_depth"] < 1 or after["reused_completed_depth"] != before["completed_depth"]:
            raise ValueError("CPU resume did not reuse its same-condition completed iteration")
    elif response["resume_kind"] is not None or before["reused_completed_depth"] != 0 or after["reused_completed_depth"] != 0:
        raise ValueError("fresh CPU task cannot claim reused resume state")
    completed_gain = max(0, after["completed_depth"] - before["completed_depth"]) if same else 0
    uncertainty = lambda v: ("rules_terminal_current_state" if v["score_scope"] == "rules_terminal" else
                             "finite_depth_estimate" if v["completed_depth"] else "unresolved_frontier")
    return {"schema": GAIN_SCHEMA, "status": "observed", "same_conditions": same,
            "new_completed_depth": completed_gain, "new_completed_profile_observation":
            bool(not same and next_profile != base_profile and after["completed_depth"] > 0),
            "new_unrestricted_completed_observation": bool(before["root_restricted"] and not after["root_restricted"] and after["completed_depth"] > 0),
            "root_choices_before": len(before["conditions"]["root_order"]), "root_choices_after": len(after["conditions"]["root_order"]),
            "uncertainty_before": uncertainty(before), "uncertainty_after": uncertainty(after),
            "actual_question_complete": after["score_scope"] == "rules_terminal" or
            (after["completion"] == "depth_limit" and after["completed_depth"] >= request["requested_depth"]),
            "raw_score_interpretation": "side_to_move_uncalibrated_scalar",
            "all_defenses_mate_certified": False, "wdl_inferred": False,
            "comparative_preference_available": False, "preference_rank": None}


def future_label(snapshot, context, task, request, response, evidence_sha256):
    after = response["after"]
    if response["status"] != "observed" or after["completed_depth"] < 1 or after["completion"] == "canceled":
        return None  # /2 owned_cpu provenance requires a completed iteration.
    return {"observed_sequence": snapshot["capture_sequence"] + 1,
            "provenance": {"source": "owned_cpu", "engine_sha256": request["cpu_binary_sha256"],
                           "profile_sha256": after["profile_sha256"],
                           "task_sha256": request["context_sha256"], "completed_depth": after["completed_depth"],
                           "nodes": response["nodes"], "raw_evidence_sha256": evidence_sha256},
            "policy": None, "value_wdl": None, "white_to_move": snapshot["white_to_move"],
            "counterexample": None,
            "verifier_tasks": [{"task": name, **asdict(context), "preference_rank": None,
                                "information_gain_evidence_sha256": evidence_sha256 if name == task else None}
                               for name in TASKS], "supersedes_label_sha256": None}


def _encoded_json(encoding, parent_sha256, context, derivation):
    value = asdict(encoding)
    value["task_queries"] = [{"context": asdict(context), "query": list(encoding.query)}]
    payload = {"schema": QUERY_SCHEMA, "parent_input_sha256": parent_sha256,
               "encoding": value, "derivation": derivation, "query_feature_names": list(QUERY_FIELDS),
               "query_captured_before_cpu": True, "native_verifier_encoded": False}
    return {"payload_sha256": _hash(QUERY_SCHEMA, payload), **payload}


def load_private_verifier_bank(directory, *, expected_receipt_sha256, parents,
                               expected_checkpoint_sha256, expected_cpu_binary_sha256,
                               expected_private_encoder_source_sha256,
                               max_input_bytes=64 * 1024 * 1024,
                               ordering_policy=LEGACY_ORDERING):
    """Reload an independently accepted zero-step bank into ValidatedDataset.

    The owner confirms the CLI's final passed status/exit 0 before supplying
    the receipt SHA. A receipt records checks before final write; it alone is
    not evidence that launch, final I/O and cleanup met the execution budget.
    Parent native data is independently admitted and never silently recreated.
    SEE is never inferred from a receipt: the caller must select its namespace.
    """
    selected_ordering = _ordering_policy(ordering_policy)
    if not isinstance(parents, ValidatedDataset):
        raise ValueError("independently admitted native public parent dataset required")
    root = _path(directory)
    paths = [_path(root / name) for name in ("receipt.json", *PrivateBank.FILES)]
    _integer(max_input_bytes, 1, 1024 * 1024 * 1024, "private bank input bytes")
    if any(not p.is_file() or p.stat().st_size < 1 for p in paths) or sum(p.stat().st_size for p in paths) > max_input_bytes:
        raise ValueError("private bank exceeds aggregate input admission")
    _, raw = _file(paths[0], expected_receipt_sha256, paths[0].stat().st_size)
    receipt = _parse(raw)
    if ((selected_ordering == LEGACY_ORDERING and "cpu_ordering_policy" in receipt)
            or (selected_ordering == SEE_ORDERING and receipt.get("cpu_ordering_policy") != SEE_ORDERING)):
        raise ValueError("private bank ordering differs from independently selected namespace")
    if receipt.get("schema") != SCHEMA or receipt.get("complete") is not True or receipt.get("failure") is not None or receipt.get("status") != "checks_passed_before_receipt_write":
        raise ValueError("failed or incomplete private bank cannot enter preparation")
    for name in ("actual_training_executed", "backward_executed", "optimizer_created", "product_verifier_enabled", "external_teacher_used", "native_verifier_encoded", "trained"):
        if receipt.get(name) is not False:
            raise ValueError("private bank crossed zero-step/private-only admission")
    if type(receipt.get("optimizer_steps")) is not int or receipt["optimizer_steps"] != 0 or receipt.get("private_bank_only") is not True or receipt.get("no_comparative_training_target") is not True:
        raise ValueError("private bank learning provenance mismatch")
    declaration = receipt["query_encoding"]
    if ((selected_ordering == LEGACY_ORDERING and "cpu_ordering_policy" in declaration)
            or (selected_ordering == SEE_ORDERING and declaration.get("cpu_ordering_policy") != SEE_ORDERING_IDENTITY)):
        raise ValueError("private query declaration differs from selected CPU ordering")
    if declaration["private_encoder_source_sha256"] != _sha(expected_private_encoder_source_sha256) or receipt["checkpoint_sha256"] != _sha(expected_checkpoint_sha256):
        raise ValueError("private encoder/own frozen weight authority mismatch")
    if _hash(QUERY_SCHEMA, declaration) != receipt["encoding_schema_sha256"]:
        raise ValueError("private encoding schema seal mismatch")
    authority = receipt["source_registry"]
    if authority["cpu_binary_sha256"] != [_sha(expected_cpu_binary_sha256)] or len(authority["input_sources"]) != 1:
        raise ValueError("private own CPU/source registry differs from owner authority")
    source = authority["input_sources"][0]
    if source != {"kind": "own_pals", "model_configuration_sha256": declaration["model_configuration_sha256"],
                  "model_weights_sha256": expected_checkpoint_sha256}:
        raise ValueError("private fresh V source does not match actual own model")
    rows = {}
    for path in paths[1:]:
        artifact = receipt["artifacts"].get(path.name)
        if not isinstance(artifact, dict) or set(artifact) != {"bytes", "sha256"} or type(artifact["bytes"]) is not int or path.stat().st_size != artifact["bytes"]:
            raise ValueError("private artifact length missing or changed")
        _, raw = _file(path, artifact["sha256"], artifact["bytes"])
        parsed = [_parse(line) for line in raw.splitlines()]
        if not 1 <= len(parsed) <= 65536:
            raise ValueError("private bank row allocation limit")
        rows[path.name] = parsed
    count = len(rows["inputs.jsonl"])
    if any(len(values) != count for values in rows.values()) or receipt["counts"]["selections"] != count:
        raise ValueError("private bank files do not preserve one row per selection")
    parent_map = {row["input"]["sha256"]: row for row in parents.records}
    encodings, records, games = {}, [], {}
    for current, sidecar, decision, evidence, future, joined in zip(*(rows[name] for name in PrivateBank.FILES)):
        frozen, context = current["input"], TaskContext(**current["context"])
        identity = frozen["sha256"]
        parent_id = current["parent_input_sha256"]
        parent = parent_map.get(parent_id)
        if parent is None or identity in encodings:
            raise ValueError("missing native parent or duplicate V input")
        native = parents.encodings.get(parent_id)
        if native is None:
            raise ValueError("missing admitted parent native features")
        value = dict(sidecar)
        if value.pop("payload_sha256") != _hash(QUERY_SCHEMA, value) or sidecar["parent_input_sha256"] != parent_id or sidecar["schema"] != QUERY_SCHEMA or sidecar["query_captured_before_cpu"] is not True or sidecar["native_verifier_encoded"] is not False:
            raise ValueError("private pre-result query sidecar seal mismatch")
        derivation, encoded = sidecar["derivation"], sidecar["encoding"]
        if derivation["encoding_schema_sha256"] != receipt["encoding_schema_sha256"] or derivation["parent_input_sha256"] != parent_id or derivation["context"] != asdict(context) or _hash(QUERY_SCHEMA, derivation) != frozen["snapshot"]["encoding_sha256"]:
            raise ValueError("private query differs from sealed V input/context")
        controls = {name: derivation[name] for name in ("prefix", "root_moves")}
        mask, _ = eligible_tasks(derivation["allowed_tasks"], controls, parent["input"]["snapshot"]["legal_moves"])
        query = private_query(mask, parent["input"]["snapshot"],
                              baseline_depth=derivation["baseline_depth"], requested_depth=derivation["requested_depth"],
                              max_nodes_per_check=derivation["max_nodes_per_check"],
                              max_wall_time_ms=derivation["max_task_wall_time_ms"], budget_bucket=context.budget_bucket)
        if derivation["query"] != list(query) or derivation["eligible_tasks"] != mask or encoded["query"] != list(query):
            raise ValueError("private query was re-encoded from a future result")
        prepared, encoding = verifier_input(parent, native, source, context, query,
                                            frozen["snapshot"]["encoding_sha256"], frozen["snapshot"]["frozen_epoch"])
        if frozen != prepared["input"] or _json(encoded) != _json(_encoded_json(encoding, parent_id, context, derivation)["encoding"]):
            raise ValueError("fresh V input/public features differ from admitted native parent")
        request, response = evidence["request"], evidence["response"]
        request_domain = _cpu_request_domain(request, selected_ordering)
        pre_request = dict(request)
        request_seal = pre_request.pop("context_sha256")
        if request_seal != _hash(request_domain, pre_request) or request["branch_sha256"] != context.branch_sha256 or request["cpu_profile_sha256"] != context.cpu_profile_sha256 or request["parent_input_sha256"] != parent_id or request["cpu_binary_sha256"] != expected_cpu_binary_sha256:
            raise ValueError("CPU future evidence not bound to selected private context")
        if request["branch_sha256"] != _hash("rz-pals-private-cpu-branch/1", {"parent_input_sha256": parent_id, **controls}):
            raise ValueError("CPU branch controls changed after V selection")
        if request != decision["request"] or decision["input_sha256"] != identity or evidence["input_sha256"] != identity or evidence["context"] != asdict(context):
            raise ValueError("CPU future evidence attached to another V decision")
        if decision["task"] != request["task"] or not mask[TASKS.index(decision["task"])] or decision["verifier_private"]["task_kind"] != decision["task"] or decision["verifier_private"]["control_sha256"] != request_seal:
            raise ValueError("private task/latent controls changed after selection")
        actual_gain = observed_gain_with_ordering(request, response, ordering_policy=selected_ordering)
        expected_evidence = dict(evidence)
        evidence_sha = expected_evidence.pop("evidence_sha256")
        if evidence["observed_information"] != actual_gain or _hash(GAIN_SCHEMA, expected_evidence) != evidence_sha:
            raise ValueError("CPU observed information evidence changed")
        label = future_label(frozen["snapshot"], context, decision["task"], request, response, evidence_sha)
        record = {"input": frozen, "future_label": label, "verifier_private": decision["verifier_private"]}
        if future != {"input_sha256": identity, "future_label": label, "observed_information_evidence_sha256": evidence_sha} or joined["record"] != record or joined["evidence_sha256"] != evidence_sha:
            raise ValueError("future label rewrote historical private input")
        if any(value["preference_rank"] is not None for value in (label or {}).get("verifier_tasks", [])):
            raise ValueError("single dispatch cannot manufacture comparative ranks")
        game = frozen["snapshot"]["game_id"]
        if current["split"] != parents.split[game]:
            raise ValueError("private bank changed immutable native game split")
        games[game] = current["split"]
        records.append(record)
        encodings[identity] = encoding
    return ValidatedDataset(records, {"games": games}, authority, encodings)


def run_verifier_producer(*, collection, receipt_sha256, encoder_source_sha256, checkpoint,
                          checkpoint_sha256, cpu_binary, cpu_binary_sha256, output,
                          max_games, max_steps, max_nodes, max_wall_time_ms, max_output_bytes,
                          max_forward_flops, baseline_depth=1, requested_depth=2,
                          max_nodes_per_check=4096, max_task_wall_time_ms=5000,
                          max_cpu_output_bytes=65536,
                          budget_bucket=0, tt_entries=8192, quiescence_ply=8,
                          frozen_epoch=1, max_input_bytes=256 * 1024 * 1024,
                          allowed_tasks=("widen_responses", "resume_task", "cross_profile_recheck", "defer"),
                          controls=None, controls_sha256=None, cancel_file=None,
                          ordering_policy=LEGACY_ORDERING):
    """Finite collection steps are selections/dispatches; optimizer steps stay 0."""
    selected_ordering = _ordering_policy(ordering_policy)
    cpu_domain = SEE_CPU_SCHEMA if selected_ordering == SEE_ORDERING else CPU_SCHEMA
    limits = Limits(max_games=max_games, max_steps=max_steps, max_nodes=max_nodes,
                    max_wall_time_ms=max_wall_time_ms, max_output_bytes=max_output_bytes,
                    max_forward_flops=max_forward_flops, cancel_file=cancel_file)
    _integer(baseline_depth, 1, 63, "baseline depth")
    _integer(requested_depth, baseline_depth + 1, 64, "task depth")
    _integer(max_nodes_per_check, 1, 2**32 - 1, "per-check nodes")
    _integer(max_task_wall_time_ms, 1, 300000, "per-task wall")
    _integer(max_cpu_output_bytes, 1024, 1048576, "per-task CPU output")
    _integer(budget_bucket, 0, 16, "explicit budget bucket")
    _integer(tt_entries, 0, 1048576, "TT entries")
    _integer(quiescence_ply, 0, 32, "quiescence ply")
    _integer(frozen_epoch, 1, 2**64 - 1, "frozen model epoch")
    _integer(max_input_bytes, 1, 1024 * 1024 * 1024, "input bytes")
    initial_threads = torch.get_num_threads()
    bank = None
    cpu_capture, cpu_reservations, cpu_reserved_nodes, cpu_stage = None, (), 0, None
    try:
        torch.set_num_threads(2)
        checkpoint, binary = _path(checkpoint), _path(cpu_binary)
        collection = _path(collection)
        metadata_path = _path(checkpoint.with_name("checkpoint.json"))
        source_paths = [checkpoint, binary, metadata_path, *[
            _path(collection / name) for name in ("receipt.json", "records.jsonl", "native-inputs.jsonl", "source-registry.jsonl", "split.jsonl")]]
        if controls is not None:
            source_paths.append(_path(controls))
        if any(not p.is_file() or p.stat().st_size < 1 for p in source_paths):
            raise ValueError("all registered source assets must be nonempty regular files")
        source_sizes = {p: p.stat().st_size for p in source_paths}
        if sum(source_sizes.values()) > max_input_bytes or source_sizes[metadata_path] > 65536:
            raise ValueError("source assets exceed aggregate input admission before reading")
        checkpoint, checkpoint_raw = _file(checkpoint, checkpoint_sha256, source_sizes[checkpoint])
        binary, binary_raw = _file(binary, cpu_binary_sha256, source_sizes[binary])
        _, metadata_raw = _read_bounded(metadata_path, source_sizes[metadata_path])
        metadata_digest = hashlib.sha256(metadata_raw).hexdigest()
        model_meta_bytes = source_sizes[metadata_path]
        remaining = max_input_bytes - len(checkpoint_raw) - len(binary_raw) - model_meta_bytes
        del checkpoint_raw, binary_raw
        control_map = {}
        if controls is not None:
            _, raw = _file(controls, controls_sha256, source_sizes[_path(controls)])
            remaining -= len(raw)
            control_map = _parse(raw)
            if not isinstance(control_map, dict) or len(control_map) > 65536:
                raise ValueError("private pre-result controls must be bounded input SHA map")
            for identity, control in control_map.items():
                _sha(identity)
                # Branch legality is verified by Rust; this checks shape only.
                eligible_tasks(allowed_tasks, control, require_available=False)
        elif controls_sha256 is not None:
            raise ValueError("controls digest without controls file")
        if remaining <= 0:
            raise ValueError("source assets exceed total input admission")
        data = load_collected_dataset(collection, expected_receipt_sha256=receipt_sha256,
                                      expected_encoder_source_sha256=encoder_source_sha256,
                                      max_input_bytes=remaining)
        if any(row["input"]["snapshot"]["role"] not in ("proposer", "critic") for row in data.records):
            raise ValueError("V private producer requires admitted public P/C parents")
        data._verify_raw_integrity()
        current_parent_indices = data.current_view.current_indices
        parent_identity = _hash("rz-pals-private-immutable-parents/1", data.records)
        if any(identity not in {row["input"]["sha256"] for row in data.records} for identity in control_map):
            raise ValueError("control refers to an unobserved parent input")
        model, metadata = load_checkpoint(checkpoint)
        if metadata.get("trained") is not False or metadata.get("training_steps") != 0 or "validator" not in model.experts:
            raise ValueError("private producer requires explicitly own untrained zero-step V")
        if metadata["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("V checkpoint differs from independently admitted asset")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        model.eval()
        before = _parameter_digest(model)
        source = {"kind": "own_pals", "model_configuration_sha256": _hash("rz-pals-private-model-configuration/1", model.config.to_dict()),
                  "model_weights_sha256": checkpoint_sha256}
        profile = cpu_profile_with_ordering(tt_entries, requested_depth, quiescence_ply, ordering_policy=selected_ordering)
        profile_sha = profile_sha256(profile)
        recheck_sha = profile_sha256(cpu_profile_with_ordering(tt_entries, requested_depth, quiescence_ply, RECHECK_PROFILE,
                                                              ordering_policy=selected_ordering))
        producer_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        query_encoding = {"schema": QUERY_SCHEMA, "native_encoding_sha256": data.records[0]["input"]["snapshot"]["encoding_sha256"],
                          "encoder_source_sha256": encoder_source_sha256, "query_fields": list(QUERY_FIELDS),
                          "private_encoder_source_sha256": producer_source_sha,
                          "model_configuration_sha256": source["model_configuration_sha256"], "private_only": True}
        if selected_ordering == SEE_ORDERING:
            query_encoding["cpu_ordering_policy"] = SEE_ORDERING_IDENTITY
        if any(row["input"]["snapshot"]["encoding_sha256"] != query_encoding["native_encoding_sha256"] for row in data.records):
            raise ValueError("one private bank cannot mix native encoding revisions")
        encoding_sha = _hash(QUERY_SCHEMA, query_encoding)
        authority = {"cpu_binary_sha256": [cpu_binary_sha256], "input_sources": [source]}
        bank = PrivateBank(output, limits)
        counts = {"selections": 0, "public_encoder_forwards": 0, "verifier_forwards": 0,
                  "cpu_dispatches": 0, "cpu_checks_observed": 0,
                  "completed_questions": 0, "deferred": 0, "unavailable": 0,
                  "future_labels": 0, "ranked_task_targets": 0, "task_loss_rows": 0}
        # No inherited CPU report is treated as a completed check under this
        # freshly registered profile/context; baseline is executed by Rust.
        with torch.inference_mode():
            for selection_index, parent_index in enumerate(current_parent_indices):
                if selection_index >= max_steps:
                    break
                parent = data.records[parent_index]
                snapshot = parent["input"]["snapshot"]
                if snapshot["game_id"] not in limits.games and len(limits.games) >= max_games:
                    break
                limits.step(snapshot["game_id"])
                control = control_map.get(parent["input"]["sha256"],
                                          {"prefix": [], "root_moves": []})
                mask, reasons = eligible_tasks(allowed_tasks, control, snapshot["legal_moves"])
                branch = _hash("rz-pals-private-cpu-branch/1", {"parent_input_sha256": parent["input"]["sha256"], **control})
                context = TaskContext(branch, profile_sha, budget_bucket)
                query = private_query(mask, snapshot, baseline_depth=baseline_depth, requested_depth=requested_depth,
                                      max_nodes_per_check=max_nodes_per_check, max_wall_time_ms=max_task_wall_time_ms,
                                      budget_bucket=budget_bucket)
                derivation = {"encoding_schema_sha256": encoding_sha,
                              "parent_input_sha256": parent["input"]["sha256"], "context": asdict(context),
                              "prefix": control["prefix"], "root_moves": control["root_moves"],
                              "allowed_tasks": list(allowed_tasks), "eligible_tasks": mask, "query": list(query),
                              "baseline_depth": baseline_depth, "requested_depth": requested_depth,
                              "max_nodes_per_check": max_nodes_per_check, "max_task_wall_time_ms": max_task_wall_time_ms}
                private_encoding_sha = _hash(QUERY_SCHEMA, derivation)
                row, encoding = verifier_input(parent, data.encodings[parent["input"]["sha256"]], source,
                                               context, query, private_encoding_sha, frozen_epoch)
                split = {"games": {snapshot["game_id"]: data.split[snapshot["game_id"]]}}
                prepared = ValidatedDataset([row], split, authority, {encoding.input_sha256: encoding})
                batch = prepared.collate([0], "verifier", split=data.split[snapshot["game_id"]], task_contexts=[context])
                analytic = matmul_flops(model.config, "validator", len(encoding.public_records), len(snapshot["legal_moves"]))
                limits.charge("forward_matmul_flops", analytic["cold_forward_matmul_flops"])
                # Write the fresh /2 seal and private query BEFORE V and CPU.
                bank.append("inputs.jsonl", {"parent_input_sha256": parent["input"]["sha256"], "input": row["input"],
                                              "context": asdict(context), "split": data.split[snapshot["game_id"]]})
                bank.append("encodings.jsonl", _encoded_json(encoding, parent["input"]["sha256"], context, derivation))
                memory = model.public_encoder(*batch.inputs.public_args())
                actual = model.role_graph("validator")(*batch.inputs.role_args(memory))
                counts["public_encoder_forwards"] += 1
                counts["verifier_forwards"] += 1
                if any(value.requires_grad or not torch.all(torch.isfinite(value)) for value in (*memory[:2], *actual)):
                    raise ValueError("private V forward must be finite frozen inference")
                task = select_task(actual[3], mask)
                counts["selections"] += 1
                wall_budget = min(max_task_wall_time_ms, limits.remaining_ms())
                request = {"schema": cpu_domain, "task": task,
                           "parent_input_sha256": parent["input"]["sha256"],
                           "position_command": snapshot["position_command"], "expected_board_fen": snapshot["board_fen"],
                           "rules_state_sha256": snapshot["rules_state_sha256"], "rules_history_sha256": snapshot["rules_history_sha256"],
                           "cpu_binary_sha256": cpu_binary_sha256, "branch_sha256": branch,
                           "prefix": control["prefix"], "root_moves": control["root_moves"],
                           "baseline_depth": baseline_depth, "requested_depth": requested_depth,
                           "max_nodes_per_check": max_nodes_per_check, "max_wall_time_ms": wall_budget,
                           "max_output_bytes": max_cpu_output_bytes,
                           "tt_entries": tt_entries, "quiescence_ply": quiescence_ply,
                           "cpu_profile_sha256": profile_sha, "recheck_profile_sha256": recheck_sha}
                if selected_ordering == SEE_ORDERING:
                    request["ordering_policy"] = SEE_ORDERING
                request["context_sha256"] = _hash(cpu_domain, request)
                private = {"task_kind": task, "control_sha256": request["context_sha256"],
                           "private_latent": actual[2][0].reshape(-1).tolist()}
                decision = {"input_sha256": row["input"]["sha256"], "parent_input_sha256": parent["input"]["sha256"],
                            "request": request, "task": task, "task_logits": actual[3][0].tolist(),
                            "eligible_tasks": mask, "unavailable_reasons": reasons,
                            "conditional_capabilities": {"resume_task": "Rust baseline must issue an owned completed-iteration token"},
                            "root_control_source": "none" if parent["input"]["sha256"] not in control_map else "explicit_registered_controls",
                            "selection_weights": "own_random_initialization_untrained", "trained": False,
                            "checkpoint_sha256": checkpoint_sha256, "verifier_private": private}
                bank.append("decisions.jsonl", decision)
                # Admit nodes plus pipe, all post-result files and raw recovery
                # atomically. Keep recovery credit until the entire row commits.
                cpu_stage = "dispatch_admission"
                cpu_reservations = bank.reserve_cpu(request, row, private, context)
                evidence_reservation, recovery_reservation, pipe_reservation = cpu_reservations
                cpu_reserved_nodes = 2 * max_nodes_per_check
                counts["cpu_dispatches"] += 1
                cpu_capture = {}
                cpu_stage = "cpu_bridge"
                # Preserve the legacy bridge call boundary. Only the explicitly
                # selected SEE lane supplies its new ordering keyword.
                bridge_options = {} if selected_ordering == LEGACY_ORDERING else {
                    "ordering_policy": selected_ordering}
                response = run_cpu_bridge(binary, request, limits, capture=cpu_capture,
                                          pipe_reservation=pipe_reservation, **bridge_options)
                cpu_stage = "response_validation"
                try:
                    gain = observed_gain_with_ordering(request, response, ordering_policy=selected_ordering)
                except BaseException as protocol_error:
                    raise CpuBridgeFailure(protocol_error, cpu_capture) from protocol_error
                # On failure retain the upper reservation: actual child usage
                # is unknown, never silently recorded as zero. Only validated
                # successful responses reconcile it to observed node count.
                limits.usage["nodes"] -= 2 * max_nodes_per_check - response["nodes"]
                cpu_reserved_nodes = 0
                if response["status"] == "observed":
                    counts["cpu_checks_observed"] += 2
                    counts["completed_questions"] += int(gain["actual_question_complete"])
                else:
                    counts[response["status"]] += 1
                    counts["cpu_checks_observed"] += int(response["baseline"] is not None)
                evidence = {"input_sha256": row["input"]["sha256"], "context": asdict(context),
                            "request": request, "response": response, "observed_information": gain}
                evidence_sha = _hash(GAIN_SCHEMA, evidence)
                cpu_stage = "cpu-evidence.jsonl"
                bank.append("cpu-evidence.jsonl", {"evidence_sha256": evidence_sha, **evidence},
                            reservation=evidence_reservation)
                cpu_stage = "future-labels.jsonl"
                label = future_label(row["input"]["snapshot"], context, task, request, response, evidence_sha)
                bank.append("future-labels.jsonl", {"input_sha256": row["input"]["sha256"], "future_label": label,
                                                     "observed_information_evidence_sha256": evidence_sha},
                            reservation=evidence_reservation)
                cpu_stage = "target_validation"
                row = dict(row, future_label=label, verifier_private=private)
                ready = ValidatedDataset([row], split, authority, {encoding.input_sha256: encoding})
                target_batch = ready.collate([0], "verifier", split=data.split[snapshot["game_id"]], task_contexts=[context])
                # Reuse pre-result frozen output with post-result targets; do
                # not re-encode the historical query after observing CPU.
                losses = masked_losses(actual, target_batch)
                if target_batch.task_mask.any() or any(value.requires_grad or not torch.isfinite(value) or float(value) != 0.0 for value in losses.values()):
                    raise ValueError("single action must preserve unresolved masked V loss")
                cpu_stage = "private-records.jsonl"
                bank.append("private-records.jsonl", {"record": row, "evidence_sha256": evidence_sha,
                                                       "task_context": asdict(context), "loss": {k: float(v) for k, v in losses.items()},
                                                        "comparative_training_target": False}, reservation=evidence_reservation)
                counts["future_labels"] += int(label is not None)
                cpu_stage = "row_completion"
                limits.check()
                for reservation in cpu_reservations:
                    reservation.release()
                cpu_capture, cpu_reservations, cpu_stage = None, (), None
        after = _parameter_digest(model)
        if before != after or any(value.requires_grad or value.grad is not None for value in model.parameters()):
            raise ValueError("private verifier producer changed parameters or gradients")
        data._verify_raw_integrity()
        if _hash("rz-pals-private-immutable-parents/1", data.records) != parent_identity:
            raise ValueError("historical P/C parents changed during private production")
        _file(binary, cpu_binary_sha256, max_input_bytes)
        _file(checkpoint, checkpoint_sha256, source_sizes[checkpoint])
        _file(metadata_path, metadata_digest, source_sizes[metadata_path])
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != producer_source_sha:
            raise ValueError("private encoder source changed during production")
        elapsed = limits.check()
        result = bank.finish({"complete": True, "status": "checks_passed_before_receipt_write", "failure": None, "counts": counts,
                              **({"cpu_ordering_policy": SEE_ORDERING} if selected_ordering == SEE_ORDERING else {}),
                              "source_registry": authority, "query_encoding": query_encoding,
                              "encoding_schema_sha256": encoding_sha, "checkpoint_sha256": checkpoint_sha256,
                              "collection_receipt_sha256": receipt_sha256, "controls_sha256": controls_sha256,
                              "parent_records_available": len(data.records), "parent_records_selected": counts["selections"],
                              "parent_records_sha256": parent_identity,
                              "current_parent_records_available": len(current_parent_indices),
                              "parent_current_view_sha256": data.current_view.sha256,
                              "parameters_sha256_before": before, "parameters_sha256_after": after,
                              "parameters_unchanged": True, "parent_inputs_unchanged": True,
                              "trained": False, "selection_weights": "own_random_initialization_untrained",
                              "no_comparative_training_target": True, "native_verifier_encoded": False,
                              "scope": "finite_conditional_cpu_evidence_only", "cpu_threads": 2,
                              "worker_contract": "independently_registered_own_rust_no_descendants",
                              "resources": {"maximum": limits.maximum, "usage": limits.usage,
                                            "nodes_counter_semantics": "observed_cost_plus_upper_reservations_for_unresolved_children",
                                            "wall_time_ms_before_receipt_write": elapsed,
                                            "max_wall_time_ms": max_wall_time_ms, "max_nodes_per_check": max_nodes_per_check,
                                            "max_task_wall_time_ms": max_task_wall_time_ms},
                              "flops": {"method": "analytic_matrix_products_fma_2",
                                        "excluded": analytic["excluded_operations"], "backward": "not_executed"}})
        limits.check()
        reloaded = load_private_verifier_bank(bank.root, expected_receipt_sha256=result["receipt_sha256"],
                                              parents=data, expected_checkpoint_sha256=checkpoint_sha256,
                                              expected_cpu_binary_sha256=cpu_binary_sha256,
                                              expected_private_encoder_source_sha256=producer_source_sha,
                                              max_input_bytes=max_output_bytes, ordering_policy=selected_ordering)
        if len(reloaded.records) != counts["selections"]:
            raise ValueError("private bank reload lost admitted V selections")
        for reloaded_row in reloaded.records:
            if any(target["preference_rank"] is not None for target in (reloaded_row["future_label"] or {}).get("verifier_tasks", [])):
                raise ValueError("private bank reload exposed invented comparative target")
        result.update(status="passed", counts=counts, optimizer_steps=0, parameters_unchanged=True,
                      private_bank_reloaded=True, reloaded_records=len(reloaded.records))
    except BaseException as error:
        if bank is not None and not (bank.root / "receipt.json").exists():
            _record_producer_failure(bank, error, counts=locals().get("counts"),
                                     checkpoint_sha256=checkpoint_sha256, cpu_binary_sha256=cpu_binary_sha256,
                                     capture=cpu_capture, reservations=cpu_reservations,
                                     reserved_nodes=cpu_reserved_nodes, stage=cpu_stage)
        raise
    finally:
        torch.set_num_threads(initial_threads)
    result["wall_time_ms"] = limits.check()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection", "receipt-sha256", "encoder-source-sha256", "checkpoint", "checkpoint-sha256",
                 "cpu-binary", "cpu-binary-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    for name in ("max-games", "max-steps", "max-nodes", "max-wall-time-ms", "max-output-bytes", "max-forward-flops"):
        parser.add_argument("--" + name, type=int, required=True)
    for name, default in (("baseline-depth", 1), ("requested-depth", 2), ("max-nodes-per-check", 4096),
                          ("max-task-wall-time-ms", 5000), ("max-cpu-output-bytes", 65536), ("budget-bucket", 0), ("tt-entries", 8192),
                          ("quiescence-ply", 8), ("frozen-epoch", 1), ("max-input-bytes", 256 * 1024 * 1024)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--allowed-task", choices=TASKS, action="append", dest="allowed_tasks")
    parser.add_argument("--controls")
    parser.add_argument("--controls-sha256")
    parser.add_argument("--cancel-file")
    parser.add_argument("--ordering-policy", choices=(LEGACY_ORDERING, SEE_ORDERING), default=LEGACY_ORDERING)
    arguments = vars(parser.parse_args(argv))
    if arguments["allowed_tasks"] is None:
        del arguments["allowed_tasks"]
    print(json.dumps(run_verifier_producer(**arguments), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
