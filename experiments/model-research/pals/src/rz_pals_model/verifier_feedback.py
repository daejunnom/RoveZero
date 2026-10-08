"""Finite, private V coverage feedback; the legacy default wire is preserved.

Actual CPU receipts precede a NEW immutable semantic input and cold V call.
Only snapshot-relative completed-iteration coverage conditions that input.
Raw history/PV/CP is retained, never converted to WDL, strategic truth, reward
or a comparative rank. This is a narrow preparation seam, not the complete
strategic V feedback algorithm. No optimizer, backward, V warm seed, persistent
CPU stack, product V, checkpoint load or model export is implemented here.

The caller owns checkpoint reload and the ORIGINAL absolute deadline. Source,
build and parameter observations are independently pinned caller assurances;
metadata declarations alone do not establish that a registered worker ran.
The producer observes actual Linux loaded images/owned groups and persists raw
bytes. A reload requires the owner's separately accepted final receipt pin.
"""
import copy
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time

from . import semantic_verifier as semantic
from . import verifier_producer as legacy
from . import verifier_utility as utility
from .comparative_producer import _hash_image, _same_file_window
from .config import TASKS, ModelConfig, matmul_flops
from .model import PalsModel
from .training import ValidatedDataset

PLAN_SCHEMA = "rz-pals-private-v-feedback-plan/1"
BEFORE_SCHEMA = "rz-pals-private-v-feedback-before/1"
EVIDENCE_SCHEMA = "rz-pals-private-v-feedback-evidence/1"
RECEIPT_SCHEMA = "rz-pals-private-v-feedback-bank/1"
PROCESS_SCHEMA = "rz-pals-private-v-feedback-process/1"
SCOPE = "declared_snapshot_relative_completed_iteration_coverage_feedback_only"
SUPPORTED = ("resume_task", "cross_profile_recheck", "defer")
MAX_BYTES = 64 << 20
FINAL_BYTES = 32768
INDEX_BYTES = 1 << 20


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("feedback exact fields required")
    return value


def _int(value, low, high):
    return legacy._integer(value, low, high, "feedback integer")


def _raw(value):
    return semantic.canonical(value)


def _pin(raw):
    return semantic.byte_pin(raw)


def _actual(raw, expected, maximum=MAX_BYTES, empty=False):
    _fields(expected, ("bytes", "sha256"))
    _int(expected["bytes"], 0 if empty else 1, maximum)
    semantic._sha(expected["sha256"])
    if type(raw) is not bytes or _pin(raw) != expected:
        raise ValueError("independent feedback actual byte pin mismatch")
    return raw


def _unhex(value, maximum):
    if type(value) is not str or len(value) > 2 * maximum or len(value) % 2 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("bounded lowercase raw evidence hex required")
    return bytes.fromhex(value)


def _parent_pins(parents):
    if type(parents) is not ValidatedDataset or parents._frozen_admission_identity is None:
        raise ValueError("feedback requires existing strict frozen current parents")
    parents._verify_raw_integrity()
    return {"frozen_admission_sha256": parents._frozen_admission_identity,
            "current_view_sha256": parents.current_view.sha256,
            "receipt": copy.deepcopy(parents.frozen_admission["receipt"]),
            "raw_dataset_sha256": parents.frozen_admission["raw_dataset_sha256"]}


def _plan(raw, parents, expected_pin):
    _actual(raw, expected_pin, 1 << 20)
    plan = _fields(legacy._parse(raw), ("schema", "scope", "recipe_id", "parent", "parent_inputs", "limits", "stages"))
    if (plan["schema"] != PLAN_SCHEMA or plan["scope"] != SCOPE or plan["parent"] != _parent_pins(parents)
            or type(plan["recipe_id"]) is not str or not 1 <= len(plan["recipe_id"]) <= 128
            or plan["recipe_id"] != plan["recipe_id"].strip() or any(ord(c) < 32 for c in plan["recipe_id"])):
        raise ValueError("fixed feedback plan/current parent scope")
    inputs = plan["parent_inputs"]
    if type(inputs) is not list or not 1 <= len(inputs) <= 64 or any(type(x) is not str for x in inputs) or len(set(inputs)) != len(inputs):
        raise ValueError("unique explicitly selected current P/C parents required")
    current = {parents.records[i]["input"]["sha256"]: i for i in parents.current_view.current_indices}
    for identity in inputs:
        semantic._sha(identity)
        if identity not in current or parents.records[current[identity]]["input"]["snapshot"]["role"] not in ("proposer", "critic"):
            raise ValueError("feedback input must be a checked current P/C leaf")
        if not parents.records[current[identity]]["input"]["snapshot"]["legal_moves"]:
            raise ValueError("first coverage feedback scope requires a nonterminal captured legal root")
    limits = _fields(plan["limits"], ("max_games", "max_steps", "max_nodes", "max_wall_time_ms", "max_output_bytes",
        "max_forward_flops", "max_semantic_bytes", "cleanup_reserve_ms", "max_child_output_bytes", "max_input_bytes"))
    for name, low, high in (("max_games", 1, 65536), ("max_steps", 1, 65536), ("max_nodes", 1, (1 << 63) - 1),
        ("max_wall_time_ms", 1, 300000), ("max_output_bytes", FINAL_BYTES + INDEX_BYTES, MAX_BYTES),
        ("max_forward_flops", 1, (1 << 63) - 1), ("max_semantic_bytes", 1, semantic.MAX_SEMANTIC_BYTES),
        ("cleanup_reserve_ms", 1, 30000), ("max_child_output_bytes", 1024, 1 << 20), ("max_input_bytes", 1, 128 << 20)):
        _int(limits[name], low, high)
    if limits["cleanup_reserve_ms"] >= limits["max_wall_time_ms"]:
        raise ValueError("cleanup is inside the original feedback deadline")
    stages = plan["stages"]
    if type(stages) is not list or not 1 <= len(stages) <= 16:
        raise ValueError("finite per-parent feedback rounds are 1..16")
    fixed = None
    for stage in stages:
        _fields(stage, ("allowed_tasks", "baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "tt_entries", "quiescence_ply"))
        _int(stage["baseline_depth"], 1, 63)
        _int(stage["requested_depth"], stage["baseline_depth"] + 1, 64)
        _int(stage["max_nodes_per_check"], 1, (1 << 32) - 1)
        _int(stage["max_wall_time_ms"], 1, 300000)
        _int(stage["tt_entries"], 0, 1048576)
        _int(stage["quiescence_ply"], 0, 32)
        allowed = stage["allowed_tasks"]
        if type(allowed) is not list or not allowed or any(type(x) is not str for x in allowed) or len(set(allowed)) != len(allowed) or any(name not in SUPPORTED for name in allowed):
            raise ValueError("first coverage-feedback profile supports empty-root Resume/Recheck/Defer only")
        profile = legacy.cpu_profile(stage["tt_entries"], stage["requested_depth"], stage["quiescence_ply"])
        if fixed is not None and profile != fixed:
            raise ValueError("first feedback profile holds H/TT/Q fixed; no silent new registered profile")
        fixed = profile
    return copy.deepcopy(plan)


class FeedbackBudget:
    """All rounds share one caller clock; reservation never creates a grace."""
    def __init__(self, limits, deadline, cancel_file=None):
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("original caller absolute deadline required")
        self.deadline = float(deadline)
        self.started = self.deadline - limits["max_wall_time_ms"] / 1000
        self.work_deadline = self.deadline - limits["cleanup_reserve_ms"] / 1000
        self.limits = copy.deepcopy(limits)
        self.used = {"steps": 0, "nodes": 0, "output_bytes": FINAL_BYTES + INDEX_BYTES, "forward_flops": 0, "semantic_bytes": 0}
        self.games = set()
        self.cancel_file = legacy._path(cancel_file) if cancel_file is not None else None
        self.check()

    def check(self, cleanup=False):
        now = time.monotonic()
        if now < self.started or now >= (self.deadline if cleanup else self.work_deadline):
            raise TimeoutError("original feedback caller deadline exhausted")
        # Explicit safe-path existence only; cancellation contents are unread.
        if not cleanup and self.cancel_file is not None and self.cancel_file.exists():
            raise InterruptedError("feedback canceled")

    def reserve(self, game, *, nodes, flops, output_bytes, semantic_bytes):
        self.check()
        additions = {"steps": 1, "nodes": nodes, "forward_flops": flops, "output_bytes": output_bytes, "semantic_bytes": semantic_bytes}
        if not _reserve_usage(self.limits, self.used, self.games, game, additions):
            return None
        return _Credit(self, output_bytes)

    def can_dispatch(self, wall_ms, calls=1):
        """No lower-wall request may masquerade as the fixed coverage scope."""
        self.check()
        _int(wall_ms, 1, 300000)
        _int(calls, 1, 2)
        return self.work_deadline - time.monotonic() >= calls * wall_ms / 1000


class _Credit:
    def __init__(self, budget, size):
        self.budget, self.remaining = budget, size

    def take(self, size):
        _int(size, 0, self.remaining)
        self.remaining -= size  # Failed/partial writes remain charged.

    def release(self):
        self.budget.used["output_bytes"] -= self.remaining
        self.remaining = 0


def _reserve_usage(limits, used, games, game, additions):
    """Pure quota transaction shared by live execution and historical replay."""
    _fields(additions, ("steps", "nodes", "forward_flops", "output_bytes", "semantic_bytes"))
    if game not in games and len(games) >= limits["max_games"]:
        return False
    for name, value in additions.items():
        _int(value, 0, (1 << 63) - 1)
        if used[name] + value > limits["max_" + name]:
            return False
    for name, value in additions.items():
        used[name] += value
    games.add(game)
    return True


def _round_cost(stage, limits, config, encoding, snapshot):
    return {"steps": 1, "nodes": 2 * stage["max_nodes_per_check"],
        "forward_flops": forward_flops_upper(config, len(encoding.public_records), len(snapshot["legal_moves"])),
        "output_bytes": 6 * limits["max_child_output_bytes"] + (2 << 20),
        "semantic_bytes": semantic.reservation_bytes(1, 66 + max(1, len(encoding.public_records)), limits["max_semantic_bytes"])}


def _stage_capabilities(plan, capabilities):
    """The same registered pre-dispatch gate applies at execution AND reload."""
    for stage in plan["stages"]:
        for key, capability in (("requested_depth", "max_depth"), ("max_nodes_per_check", "max_nodes_per_check"), ("max_wall_time_ms", "max_wall_time_ms")):
            maximum = {"max_depth": 64, "max_nodes_per_check": (1 << 32) - 1, "max_wall_time_ms": 300000}[capability]
            if stage[key] > _int(capabilities[capability], 1, maximum):
                raise ValueError("fixed feedback schedule exceeds independently registered actual capability")
    if plan["limits"]["max_child_output_bytes"] > _int(capabilities["max_response_bytes"], 1, 1 << 20):
        raise ValueError("feedback child output admission exceeds actual capability")


def _episode_action(task, number, round_count):
    if task == "defer" and number != round_count - 1:
        raise ValueError("explicit defer terminates its feedback episode; later round forbidden")


def forward_flops_upper(config, records, candidates):
    """FMA=2 conservative bound including private semantic K/V and attention.

The legacy counter omits the 1394 semantic slots; charging it alone would be
incorrect. Every semantic slot is conservatively charged as a move projection.
Bias/norm/softmax/lookup/transfer/CPU search are explicitly excluded operations.
"""
    base = matmul_flops(config, "validator", records, candidates)
    t, w, kv, n = semantic.TOKEN_COUNT, config.width, config.kv_heads * config.head_dimension, config.latent_slots
    extra = 4 * t * w * kv + 4 * n * t * w * config.recurrent_blocks * config.iterations + 6 * t * w * w
    return base["cold_forward_matmul_flops"] + extra


def _scope(request, profile_sha, legal_order):
    # Wall/nodes/order/config differences are NEVER same-scope coverage.
    return {name: request[name] for name in ("parent_input_sha256", "rules_state_sha256", "rules_history_sha256",
        "expected_board_fen", "position_command", "prefix", "root_moves", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "tt_entries", "quiescence_ply")} | {
        "profile_sha256": profile_sha, "cpu_binary_sha256": request["cpu_binary_sha256"],
        "legal_order": list(legal_order), "perspective": "captured_side_to_move",
        "value_identity": {"semantics": legacy.CPU_VALUE, "weights_sha256": None, "training": {"kind": "bootstrap"}}}


def _request(snapshot, parent_id, stage, task, binary_sha):
    profile = legacy.cpu_profile(stage["tt_entries"], stage["requested_depth"], stage["quiescence_ply"])
    value = {"schema": legacy.CPU_SCHEMA, "task": task, "parent_input_sha256": parent_id,
        "position_command": snapshot["position_command"], "expected_board_fen": snapshot["board_fen"],
        "rules_state_sha256": snapshot["rules_state_sha256"], "rules_history_sha256": snapshot["rules_history_sha256"],
        "cpu_binary_sha256": binary_sha, "branch_sha256": legacy._hash("rz-pals-private-cpu-branch/1", {"parent_input_sha256": parent_id, "prefix": [], "root_moves": []}),
        "prefix": [], "root_moves": [], **{name: stage[name] for name in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "tt_entries", "quiescence_ply")},
        "cpu_profile_sha256": legacy.profile_sha256(profile),
        "recheck_profile_sha256": legacy.profile_sha256(legacy.cpu_profile(stage["tt_entries"], stage["requested_depth"], stage["quiescence_ply"], legacy.RECHECK_PROFILE))}
    return value


class FeedbackLedger:
    """Immutable raw observations, checked anew before every next query."""
    def __init__(self):
        self._entries = ()
        self._seal = semantic.digest(EVIDENCE_SCHEMA, [])

    def verify(self):
        if type(self._entries) is not tuple or len(self._entries) > 16:
            raise ValueError("finite immutable feedback history required")
        checked = []
        for ordinal, raw in enumerate(self._entries):
            if type(raw) is not bytes or len(raw) > 5 << 20:
                raise ValueError("bounded raw feedback history required")
            event = _fields(legacy._parse(raw), ("schema", "scope", "input_sha256", "ordinal", "previous_ledger_sha256",
                "request_hex", "receipt_hex", "process_hex", "legal_order", "observed_information", "preference_rank",
                "wdl_inferred", "strategic_truth_admitted", "cpu_stack_continued_across_children"))
            if (event["schema"] != EVIDENCE_SCHEMA or event["scope"] != SCOPE or type(event["ordinal"]) is not int
                    or event["ordinal"] != ordinal or event["previous_ledger_sha256"] != semantic.digest(EVIDENCE_SCHEMA, [_pin(x) for x in checked])
                    or event["preference_rank"] is not None or any(event[x] is not False for x in
                    ("wdl_inferred", "strategic_truth_admitted", "cpu_stack_continued_across_children"))):
                raise ValueError("immutable feedback evidence chain/unknown targets differ")
            semantic._sha(event["input_sha256"])
            request_raw, receipt_raw, process_raw = (_unhex(event[x], 1 << 20) for x in ("request_hex", "receipt_hex", "process_hex"))
            gain = _cpu_observation(request_raw, receipt_raw, process_raw, event["legal_order"])
            if gain != event["observed_information"]:
                raise ValueError("feedback observation does not match retained CPU bytes")
            checked.append(raw)
        if semantic.digest(EVIDENCE_SCHEMA, [_pin(x) for x in checked]) != self._seal:
            raise ValueError("feedback raw history mutated")
        return self

    @property
    def sha256(self):
        return self.verify()._seal

    def entries(self):
        self.verify()
        return tuple(self._entries)

    def known_depth(self, request, profile_sha, legal_order):
        self.verify()
        result = 0
        for raw in self._entries:
            event = legacy._parse(raw)
            prior = legacy._parse(bytes.fromhex(event["request_hex"]))
            response = legacy._parse(bytes.fromhex(event["receipt_hex"]))
            legacy.observed_gain(prior, response)
            if _scope(prior, profile_sha, event["legal_order"]) != _scope(request, profile_sha, legal_order):
                continue
            for report in (response["baseline"], response["after"]):
                if (report is not None and report["profile_sha256"] == profile_sha
                        and report["score_scope"] == "completed_iteration" and report["completion"] == "depth_limit"
                        and report["completed_depth"] == report["requested_depth"] and report["pv_rules_validated"] is True):
                    result = max(result, report["completed_depth"])
        return result

    def append(self, *, input_sha256, request_raw, receipt_raw, process_raw, legal_order, gain):
        self.verify()
        if len(self._entries) >= 16:
            raise ValueError("finite per-parent feedback evidence limit")
        if _cpu_observation(request_raw, receipt_raw, process_raw, legal_order) != gain:
            raise ValueError("feedback gain must come from actual validated raw CPU evidence")
        event = {"schema": EVIDENCE_SCHEMA, "scope": SCOPE, "input_sha256": semantic._sha(input_sha256),
            "ordinal": len(self._entries), "previous_ledger_sha256": self._seal,
            "request_hex": request_raw.hex(), "receipt_hex": receipt_raw.hex(), "process_hex": process_raw.hex(), "legal_order": list(legal_order),
            "observed_information": gain, "preference_rank": None, "wdl_inferred": False,
            "strategic_truth_admitted": False, "cpu_stack_continued_across_children": False}
        raw = _raw(event)
        self._entries += (raw,)
        self._seal = semantic.digest(EVIDENCE_SCHEMA, [_pin(value) for value in self._entries])
        return raw


def next_common_query(parents, parent_index, stage, ledger, *, round_index, maximum_rounds, remaining_global_steps, binary_sha256):
    _parent_pins(parents)
    if parent_index not in parents.current_view.current_indices:
        raise ValueError("feedback parent is superseded")
    _int(maximum_rounds, 1, 16)
    _int(round_index, 0, maximum_rounds - 1)
    _int(remaining_global_steps, 0, 65536)
    if type(ledger) is not FeedbackLedger:
        raise ValueError("actual immutable feedback ledger required")
    opportunities = min(maximum_rounds - round_index, remaining_global_steps)
    if opportunities == 0:
        return None
    row = parents.records[parent_index]
    snapshot, identity = row["input"]["snapshot"], row["input"]["sha256"]
    template = _request(snapshot, identity, stage, "resume_task", semantic._sha(binary_sha256))
    h = stage["requested_depth"]
    known = ledger.known_depth(template, template["cpu_profile_sha256"], snapshot["legal_moves"])
    recheck = ledger.known_depth(template, template["recheck_profile_sha256"], snapshot["legal_moves"])
    allowed = [task for task in stage["allowed_tasks"] if not (task == "resume_task" and known >= h)
               and not (task == "cross_profile_recheck" and recheck >= h)]
    if not allowed:
        return None
    # This real, predeclared opportunity count accounts for BOTH per-parent and
    # global remaining steps. It is not a feature invented to remint an input.
    bucket = 16 * opportunities // maximum_rounds
    return {"schema": semantic.COMMON_SCHEMA, "parent_input_sha256": identity,
        "current_view_sha256": parents.current_view.sha256, "question": "unrestricted_recheck",
        "prefix": [], "root_moves": [], "claimed_line": [], "allowed_tasks": allowed,
        **{name: stage[name] for name in ("baseline_depth", "requested_depth", "max_nodes_per_check", "max_wall_time_ms")},
        "budget_bucket": bucket, "cpu_profile_sha256": template["cpu_profile_sha256"], "known_completed_depth": known}


class _Bank:
    def __init__(self, path, budget):
        self.root = legacy._path(path)
        if any((p / ".git").exists() for p in (self.root, *self.root.parents)):
            raise ValueError("feedback bank must be outside Git")
        budget.check()
        self.root.mkdir(parents=True, exist_ok=False)
        self.budget, self.artifacts = budget, {}

    def write(self, name, raw, credit=None, final=False):
        self.budget.check(cleanup=final)
        if type(raw) is not bytes or not 0 <= len(raw) <= MAX_BYTES or Path(name).name != name or name in self.artifacts:
            raise ValueError("bounded fresh feedback artifact required")
        if credit is not None:
            credit.take(len(raw))
        fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix":
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self.artifacts[name] = _pin(raw)
        self.budget.check(cleanup=final)
        return self.artifacts[name]


def _capture(binary, arguments, request_raw, *, assets, budget):
    """Actual Linux fresh child, bounded concurrent pipes, no added grace.

Registered own Rust has no child-spawning API. This proves the owned group,
not arbitrary process trees or registration/build independence.
"""
    request = legacy._parse(request_raw)
    _int(len(request_raw), 1, 512 << 10)
    started = time.monotonic()
    deadline = min(budget.work_deadline, started + request["max_wall_time_ms"] / 1000)
    reserve = min(.2, max(0, deadline - started) / 10)
    work = deadline - reserve
    cap = request["max_output_bytes"]
    _int(cap, 1024, 1 << 20)
    output, errors = bytearray(), bytearray()
    proc, selector = None, None
    seen = {"schema": PROCESS_SCHEMA, "pid": None, "spawned": False, "reaped": False, "pipes_finished": False,
        "owned_group_absent": False, "exit_code": None, "failure": None, "cleanup_error": None,
        "elapsed_ms": None, "original_deadline_met": False, "loaded_image_before_stdin": False,
        "loaded_image_sha256": None, "loaded_executable": None, "process_supervision_scope": "posix_owned_process_group"}
    def cleanup_error(error):
        detail = type(error).__name__ + ": " + str(error)[:256]
        seen["cleanup_error"] = detail if seen["cleanup_error"] is None else seen["cleanup_error"] + "; " + detail
    try:
        budget.check()
        if sys.platform != "linux" or work <= time.monotonic():
            raise ValueError("first feedback mode requires Linux actual loaded-inode/prepaid cleanup")
        seen["binary_before"] = _hash_image(binary, assets["binary_pin"], work)
        seen["source_before"] = _hash_image(assets["source_path"], assets["source_pin"], work)
        selector = selectors.DefaultSelector()
        proc = subprocess.Popen([str(binary), *arguments], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, shell=False, bufsize=0)
        seen.update(spawned=True, pid=proc.pid)
        loaded = _hash_image(None, assets["binary_pin"], work, loaded_pid=proc.pid)
        seen.update(loaded_image_sha256=loaded["artifact"]["sha256"], loaded_image_before_stdin=True, loaded_executable=loaded)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        sent = 0
        while True:
            budget.check()
            if time.monotonic() >= work:
                raise TimeoutError("original feedback child work allowance exhausted")
            exited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None and not selector.get_map():
                seen["pipes_finished"] = True
                break
            for key, _ in selector.select(min(.005, max(0, work - time.monotonic()))):
                if key.data == "stdin":
                    sent += os.write(key.fd, request_raw[sent:sent + 4096])
                    if sent == len(request_raw):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                else:
                    block = os.read(key.fd, 4096)
                    if not block:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    else:
                        remaining = cap - len(output) - len(errors)
                        target = output if key.data == "stdout" else errors
                        target.extend(block[:max(0, remaining)])
                        if len(block) > remaining:
                            raise ValueError("feedback aggregate stdout/stderr cap exceeded")
    except BaseException as error:
        seen["failure"] = type(error).__name__ + ": " + str(error)[:512]
    finally:
        if proc is not None:
            try:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        pass
                proc.wait(timeout=max(0, deadline - time.monotonic()))
                seen.update(reaped=True, exit_code=proc.returncode)
                while time.monotonic() < deadline:
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        seen["owned_group_absent"] = True
                        break
                    time.sleep(min(.002, max(0, deadline - time.monotonic())))
            except BaseException as error:
                cleanup_error(error)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except BaseException as error:
                    cleanup_error(error)
        if selector is not None:
            try:
                selector.close()
            except BaseException as error:
                cleanup_error(error)
        try:
            seen["binary_after"] = _hash_image(binary, assets["binary_pin"], deadline)
            seen["source_after"] = _hash_image(assets["source_path"], assets["source_pin"], deadline)
            seen["binary_path_stable"] = _same_file_window(seen["binary_before"], seen["binary_after"])
            seen["source_path_stable"] = _same_file_window(seen["source_before"], seen["source_after"])
        except BaseException as error:
            cleanup_error(error)
        seen.update(elapsed_ms=math.ceil((time.monotonic() - started) * 1000), original_deadline_met=time.monotonic() < deadline)
    return bytes(output), bytes(errors), _raw(seen)


def _clean_process(raw, binary_pin, wall_ms):
    value = legacy._parse(raw)
    if (value["schema"] != PROCESS_SCHEMA or value["failure"] is not None or value["cleanup_error"] is not None
            or any(value[name] is not True for name in ("spawned", "reaped", "pipes_finished", "owned_group_absent", "original_deadline_met", "loaded_image_before_stdin", "binary_path_stable", "source_path_stable"))
            or type(value["exit_code"]) is not int or value["exit_code"] != 0
            or value["loaded_image_sha256"] != binary_pin["sha256"] or value["process_supervision_scope"] != "posix_owned_process_group"):
        raise ValueError("feedback actual child identity/exit/pipes/group/cleanup failed")
    _int(value["pid"], 1, (1 << 31) - 1)
    _int(value["elapsed_ms"], 0, wall_ms - 1)
    for name in ("binary_before", "binary_after", "source_before", "source_after", "loaded_executable"):
        facts = _fields(value[name], ("status", "artifact", "device", "inode", "mtime_ns", "ctime_ns", "scope"))
        _fields(facts["artifact"], ("bytes", "sha256"))
        _int(facts["artifact"]["bytes"], 1, MAX_BYTES)
        semantic._sha(facts["artifact"]["sha256"])
        if (facts["status"] != "checked" or facts["scope"] != ("caller_child_loaded_inode" if name == "loaded_executable" else "caller_registered_path_bytes")
                or any(type(facts[x]) is not str or not facts[x].isdigit() or len(facts[x]) > 32 for x in ("device", "inode", "mtime_ns", "ctime_ns"))):
            raise ValueError("actual registered-path/loaded-inode observation required")
        if name.startswith("binary") or name == "loaded_executable":
            if facts["artifact"] != binary_pin:
                raise ValueError("actual child binary differs from independent pin")
    if not _same_file_window(value["binary_before"], value["binary_after"]) or not _same_file_window(value["source_before"], value["source_after"]):
        raise ValueError("current registered paths changed during child")
    return value


def _implementation_sha():
    # Bounded source read; never scan other files or generated trees.
    _, raw = legacy._read_bounded(__file__, 1 << 20)
    return hashlib.sha256(raw).hexdigest()


def _implementation_sources():
    """Exact imported source files, not a generated-tree or environment scan."""
    names = ("verifier_feedback.py", "verifier_producer.py", "verifier_utility.py", "semantic_verifier.py", "comparative_producer.py",
        "config.py", "model.py", "training.py", "preparation_check.py")
    directory = Path(__file__).parent
    return {name: _pin(legacy._read_bounded(directory / name, 1 << 20)[1]) for name in names}


def _cpu_observation(request_raw, receipt_raw, process_raw, legal_order):
    for raw in (request_raw, receipt_raw, process_raw):
        if type(raw) is not bytes or not 1 <= len(raw) <= 1 << 20:
            raise ValueError("bounded immutable CPU evidence bytes required")
    request = _fields(legacy._parse(request_raw), ("schema", "task", "parent_input_sha256", "position_command", "expected_board_fen",
        "rules_state_sha256", "rules_history_sha256", "cpu_binary_sha256", "branch_sha256", "prefix", "root_moves", "baseline_depth",
        "requested_depth", "max_nodes_per_check", "max_wall_time_ms", "tt_entries", "quiescence_ply", "cpu_profile_sha256",
        "recheck_profile_sha256", "max_output_bytes", "context_sha256"))
    if (request["schema"] != legacy.CPU_SCHEMA or request["task"] not in SUPPORTED or request["prefix"] != [] or request["root_moves"] != []
            or request["context_sha256"] != legacy._hash(legacy.CPU_SCHEMA, {k: v for k, v in request.items() if k != "context_sha256"})
            or request["branch_sha256"] != legacy._hash("rz-pals-private-cpu-branch/1", {"parent_input_sha256": request["parent_input_sha256"], "prefix": [], "root_moves": []})):
        raise ValueError("feedback CPU exact empty branch/request context differs")
    for name in ("parent_input_sha256", "rules_state_sha256", "rules_history_sha256", "cpu_binary_sha256"):
        semantic._sha(request[name])
    _int(request["baseline_depth"], 1, 63)
    _int(request["requested_depth"], request["baseline_depth"] + 1, 64)
    _int(request["max_nodes_per_check"], 1, (1 << 32) - 1)
    _int(request["max_wall_time_ms"], 1, 300000)
    _int(request["tt_entries"], 0, 1048576)
    _int(request["quiescence_ply"], 0, 32)
    _int(request["max_output_bytes"], 1024, 1 << 20)
    for name, profile in (("cpu_profile_sha256", legacy.PROFILE), ("recheck_profile_sha256", legacy.RECHECK_PROFILE)):
        if request[name] != legacy.profile_sha256(legacy.cpu_profile(request["tt_entries"], request["requested_depth"], request["quiescence_ply"], profile)):
            raise ValueError("registered fixed feedback profile differs")
    if type(legal_order) is not list or not 1 <= len(legal_order) <= 256 or any(type(x) is not int for x in legal_order) or len(set(legal_order)) != len(legal_order):
        raise ValueError("actual captured Rules legal order required")
    from .training import move_components
    for movement in legal_order:
        move_components(movement)
    process = legacy._parse(process_raw)
    binary_pin = process["binary_before"]["artifact"]
    if binary_pin["sha256"] != request["cpu_binary_sha256"]:
        raise ValueError("CPU receipt/actual launch identity mismatch")
    seen = _clean_process(process_raw, binary_pin, request["max_wall_time_ms"])
    response = legacy._parse(receipt_raw)
    gain = legacy.observed_gain(request, response)
    _int(response["elapsed_ms"], 0, seen["elapsed_ms"])
    for report in (response["baseline"], response["after"]):
        if report is not None:
            if report["conditions"]["legal_moves"] != legal_order or report["conditions"]["root_order"] != legal_order:
                raise ValueError("CPU report actual ordered Rules moves differ from captured root")
            _int(report["elapsed_ms"], 0, response["elapsed_ms"])
    return gain


def _assets(paths, expected, stage, deadline, remaining_input):
    semantic._deadline(deadline)  # Exact int/float, finite, not bool; before any I/O.
    names = ("registration", "source", "capabilities", "semantic_registration", "semantic_source")
    _fields(paths, (*names, "binary"))
    _fields(expected, (*names, "binary"))
    total = sum(_int(_fields(expected[x], ("bytes", "sha256"))["bytes"], 1, MAX_BYTES if x == "binary" else 4 << 20) for x in (*names, "binary"))
    if total > remaining_input:
        raise ValueError("aggregate registered input allowance denied BEFORE reads/allocation")
    values, actual_paths = {}, {}
    for name in (*names, "binary"):
        semantic._deadline(deadline)
        maximum = MAX_BYTES if name == "binary" else 4 << 20
        _fields(expected[name], ("bytes", "sha256"))
        _int(expected[name]["bytes"], 1, maximum)
        actual_paths[name], values[name] = legacy._file(paths[name], expected[name]["sha256"], expected[name]["bytes"])
        _actual(values[name], expected[name], maximum)
        semantic._deadline(deadline)
    profile = legacy.cpu_profile(stage["tt_entries"], stage["requested_depth"], stage["quiescence_ply"])
    registration, capabilities = utility._registered(values["registration"], values["source"], values["binary"], values["capabilities"],
        {name: expected[name] for name in ("registration", "source", "binary", "capabilities")},
        {"profile": profile, "obligation": {"required_horizon": stage["requested_depth"]}})
    semantic_registration = legacy._parse(values["semantic_registration"])
    _fields(semantic_registration, ("schema", "semantic_schema", "implementation", "source_artifact", "binary_sha256", "rules_version", "platform", "accepted_binary_pin_scope", "assurance_scope"))
    if (registration["platform"] != "linux" or semantic_registration["platform"] != "linux"
            or semantic_registration["binary_sha256"] != expected["binary"]["sha256"]
            or semantic_registration["schema"] != semantic.REGISTRATION_SCHEMA or semantic_registration["semantic_schema"] != semantic.SEMANTIC_SCHEMA
            or semantic_registration["implementation"] != semantic.IMPLEMENTATION
            or semantic_registration["source_artifact"] != expected["semantic_source"]
            or semantic_registration["assurance_scope"] != "independently_registered_caller_source"
            or semantic_registration["accepted_binary_pin_scope"] != "linux_loaded_executable_inode"):
        raise ValueError("same independently registered Linux CPU and Rules worker required")
    semantic._deadline(deadline)
    return values, actual_paths, registration, capabilities


def _semantic_step(parents, index, common, assets, paths, bank, credit, stem, before_feedback_raw):
    common_raw = _raw(common)
    before_raw = _raw({"schema": semantic.BEFORE_SCHEMA, "parent_input_sha256": common["parent_input_sha256"],
        "current_view_sha256": common["current_view_sha256"], "common_query_sha256": _pin(common_raw)["sha256"],
        "registration_sha256": _pin(assets["semantic_registration"])["sha256"]})
    snapshot = parents.records[index]["input"]["snapshot"]
    request = {"schema": semantic.SEMANTIC_SCHEMA, "question": common["question"], "parent_input_sha256": common["parent_input_sha256"],
        "before_result_anchor_sha256": _pin(before_raw)["sha256"], "position_command": snapshot["position_command"], "expected_board_fen": snapshot["board_fen"],
        "rules_state_sha256": snapshot["rules_state_sha256"], "rules_history_sha256": snapshot["rules_history_sha256"],
        "prefix": [], "root_moves": [], "claimed_line": [], "current_binary_sha256": _pin(assets["binary"])["sha256"],
        "max_wall_time_ms": common["max_wall_time_ms"], "max_output_bytes": bank.budget.limits["max_child_output_bytes"]}
    request["context_sha256"] = semantic.digest(semantic.SEMANTIC_SCHEMA, request)
    raws = {"common_query": common_raw, "before_result": before_raw, "request": _raw(request),
        "source": assets["semantic_source"], "registration": assets["semantic_registration"]}
    for name in ("common_query", "before_result", "request"):
        bank.write(stem + "-" + name + ".json", raws[name], credit)
    dispatch_raw = _raw({"feedback_before": _pin(before_feedback_raw), "semantic_before": _pin(before_raw),
        "common_query": _pin(common_raw), "durable_before_dispatch": True})
    bank.write(stem + "-Rules-dispatch.json", dispatch_raw, credit)
    transport = {"binary_pin": _pin(assets["binary"]), "source_path": paths["semantic_source"], "source_pin": _pin(assets["semantic_source"])}
    out, err, process = _capture(paths["binary"], ["--prepare-semantic"], raws["request"], assets=transport, budget=bank.budget)
    for name, raw in (("receipt", out), ("stderr", err), ("process", process)):
        bank.write(stem + "-Rules-" + name + ".bin", raw, credit, final=True)
    seen = _clean_process(process, transport["binary_pin"], common["max_wall_time_ms"])
    if not out or err:
        raise ValueError("successful feedback Rules child requires typed stdout/empty stderr")
    receipt = legacy._parse(out)
    _int(receipt["elapsed_ms"], 0, seen["elapsed_ms"])
    raws["receipt"] = out
    raws["launch_observation"] = _raw({"schema": semantic.LAUNCH_SCHEMA, "registration_sha256": _pin(raws["registration"])["sha256"],
        "request": _pin(raws["request"]), "receipt": _pin(out), "binary_sha256": transport["binary_pin"]["sha256"],
        "platform": "linux", "binary_pin_scope": "linux_loaded_executable_inode", "before_result_anchor_sha256": request["before_result_anchor_sha256"],
        "assurance_scope": "independently_pinned_caller_observation", "anchor_durable_before_spawn": True,
        "spawned": True, "reaped": True, "exit_code": 0, "elapsed_ms": seen["elapsed_ms"], "timed_out": False})
    bank.write(stem + "-Rules-launch.json", raws["launch_observation"], credit)
    pins = {name: _pin(raw) for name, raw in raws.items()}
    pins.update(parent_input_sha256=common["parent_input_sha256"], current_view_sha256=parents.current_view.sha256,
        frozen_admission_sha256=parents._frozen_admission_identity,
        encoding_sha256=parents.encodings[common["parent_input_sha256"]].encoding_sha256)
    checked = semantic.admit_semantic_input(parent=parents, parent_index=index,
        **{name + "_bytes": raw for name, raw in raws.items()}, expected_pins=pins)
    return checked, raws, pins


def run_verifier_feedback(*, parents, model, plan_bytes, expected_plan_pin, registered_paths, expected_registered_pins,
                          parameter_observation_bytes, expected_parameter_observation_pin, output, deadline, cancel_file=None):
    """Fresh Rules→V→CPU rounds; caller reload/final process supervision is external.

The exact prior plan chooses parents, finite budget schedule and supported task
set. V chooses TASK only. Coverage disables already covered actions, preserving
ties via vocabulary order for ACTION selection but all supervision ranks stay
unknown. Partial/canceled/terminal observations never become positive targets.
"""
    plan = _plan(plan_bytes, parents, expected_plan_pin)
    budget = FeedbackBudget(plan["limits"], deadline, cancel_file)
    _actual(parameter_observation_bytes, expected_parameter_observation_pin, 8192)
    if type(model) is not PalsModel:
        raise ValueError("actual existing frozen PalsModel required")
    frontend = semantic.FrozenSemanticVerifier(model, parameter_observation_bytes=parameter_observation_bytes,
        expected_observation_sha256=expected_parameter_observation_pin["sha256"])
    if len(plan_bytes) + len(parameter_observation_bytes) > plan["limits"]["max_input_bytes"]:
        raise ValueError("aggregate caller input bytes exceed fixed feedback allowance")
    remaining_input = plan["limits"]["max_input_bytes"] - len(plan_bytes) - len(parameter_observation_bytes)
    assets, paths, registration, capabilities = _assets(registered_paths, expected_registered_pins, plan["stages"][0], budget.work_deadline, remaining_input)
    _stage_capabilities(plan, capabilities)
    # Parent loaders/checkpoint loads are already caller-owned. Read allocations
    # here are independently bounded by the supplied small asset pins plus one
    # <=64MiB binary. Their combined byte count is declared explicitly.
    bank, episodes = _Bank(output, budget), []
    setup_size = len(plan_bytes) + len(parameter_observation_bytes) + sum(len(raw) for name, raw in assets.items() if name != "binary")
    if setup_size + FINAL_BYTES + INDEX_BYTES > plan["limits"]["max_output_bytes"]:
        raise ValueError("feedback setup exceeds output allowance before dispatch")
    budget.used["output_bytes"] += setup_size
    bank.write("plan.json", plan_bytes)
    bank.write("parameter-observation.json", parameter_observation_bytes)
    for name, raw in assets.items():
        if name != "binary":
            bank.write(name + ".json", raw)
    parent_before = _parent_pins(parents)
    source_before = _implementation_sha()
    implementation_before = _implementation_sources()
    counts = {"selections": 0, "semantic_preparations": 0, "verifier_cold_forwards": 0,
        "CPU_dispatches": 0, "CPU_checks": 0, "feedback_transitions": 0, "ranked_targets": 0}
    failure, stop = None, "fixed_round_schedule_exhausted"
    try:
        indices = {parents.records[i]["input"]["sha256"]: i for i in parents.current_view.current_indices}
        for identity in plan["parent_inputs"]:
            index, ledger = indices[identity], FeedbackLedger()
            episode = {"parent_input_sha256": identity, "parent_index": index, "rounds": [], "stop": None}
            episodes.append(episode)
            for number, stage in enumerate(plan["stages"]):
                budget.check()
                remaining_steps = plan["limits"]["max_steps"] - budget.used["steps"]
                if remaining_steps == 0 or not budget.can_dispatch(stage["max_wall_time_ms"], calls=2):
                    episode["stop"] = stop = "pre_dispatch_resource_limit_unresolved"
                    break
                common = next_common_query(parents, index, stage, ledger, round_index=number,
                    maximum_rounds=len(plan["stages"]), remaining_global_steps=remaining_steps,
                    binary_sha256=expected_registered_pins["binary"]["sha256"])
                if common is None:
                    episode["stop"] = "declared_snapshot_scope_covered;no_global_truth_or_optimality"
                    break
                snapshot = parents.records[index]["input"]["snapshot"]
                encoded = parents.encodings[identity]
                cost = _round_cost(stage, plan["limits"], model.config, encoded, snapshot)
                credit = budget.reserve(snapshot["game_id"], nodes=cost["nodes"], flops=cost["forward_flops"],
                    output_bytes=cost["output_bytes"], semantic_bytes=cost["semantic_bytes"])
                if credit is None:
                    episode["stop"] = stop = "pre_dispatch_resource_limit_unresolved"
                    break
                stem = "parent-" + str(index) + "-round-" + str(number)
                before = {"schema": BEFORE_SCHEMA, "scope": SCOPE, "plan": expected_plan_pin,
                    "parent_input_sha256": identity, "round": number, "previous_ledger_sha256": ledger.sha256,
                    "previous_evidence_pins": [_pin(raw) for raw in ledger.entries()], "common_query": _pin(_raw(common)),
                    "remaining_global_steps": remaining_steps, "remaining_parent_rounds": len(plan["stages"]) - number,
                    "budget_bucket_meaning": "min_fixed_remaining_parent_rounds_and_global_steps_div_maximum_parent_rounds",
                    "budget_policy": "registered_schedule_not_V_budget_head",
                    "warm_private_latent": False, "cpu_stack_continued_across_children": False}
                before_raw = _raw(before)
                bank.write(stem + "-feedback-before.json", before_raw, credit)
                checked, raw_semantic, semantic_pins = _semantic_step(parents, index, common, assets, paths, bank, credit, stem, before_raw)
                counts["semantic_preparations"] += 1
                checked.verify_parent(parents)
                if checked.common_query()["known_completed_depth"] != ledger.known_depth(
                        _request(snapshot, identity, stage, "resume_task", expected_registered_pins["binary"]["sha256"]), common["cpu_profile_sha256"], snapshot["legal_moves"]):
                    raise ValueError("actual evidence coverage differs from new V semantic input")
                task_logits, latent, forward_audit = frontend.forward([checked], deadline=budget.work_deadline,
                    byte_limit=plan["limits"]["max_semantic_bytes"])
                counts["verifier_cold_forwards"] += 1
                if number:
                    counts["feedback_transitions"] += 1
                mask = checked.common_query()["eligible_tasks"]
                task = legacy.select_task(task_logits, mask)
                request = _request(snapshot, identity, stage, task, expected_registered_pins["binary"]["sha256"])
                request["max_output_bytes"] = plan["limits"]["max_child_output_bytes"]
                request["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, request)
                request_raw = _raw(request) + b"\n"
                private = {"task_kind": task, "control_sha256": request["context_sha256"], "private_latent": latent[0].reshape(-1).tolist()}
                decision = {"semantic_input_sha256": checked.sha256, "input_sha256": checked.derived_input()["input"]["sha256"],
                    "feedback_before": _pin(before_raw), "task": task, "task_logits": task_logits[0].tolist(), "private": private,
                    "forward_audit": forward_audit, "preference_rank": None, "warm_continuation": False,
                    "budget_head_executed": False, "request": _pin(request_raw)}
                bank.write(stem + "-decision.json", _raw(decision), credit)
                if not budget.can_dispatch(stage["max_wall_time_ms"]):
                    bank.write(stem + "-unresolved.json", _raw({"reason": "fixed_task_wall_no_longer_available_before_CPU",
                        "unknown_observation": True, "positive_training_target": False}), credit, final=True)
                    episode["stop"] = stop = "pre_dispatch_resource_limit_unresolved"
                    episode["rounds"].append({"stem": stem, "semantic_pins": semantic_pins,
                        "semantic_input_sha256": checked.sha256, "derived_input_sha256": decision["input_sha256"],
                        "CPU_dispatched": False, "ledger_after_sha256": ledger.sha256})
                    counts["selections"] += 1
                    credit.release()
                    break
                bank.write(stem + "-CPU-request.json", request_raw, credit)
                transport = {"binary_pin": expected_registered_pins["binary"], "source_path": paths["source"], "source_pin": expected_registered_pins["source"]}
                out, err, process_raw = _capture(paths["binary"], [], request_raw, assets=transport, budget=budget)
                counts["CPU_dispatches"] += 1
                for name, raw in (("stdout", out), ("stderr", err), ("process", process_raw)):
                    bank.write(stem + "-CPU-" + name + ".bin", raw, credit, final=True)
                seen = _clean_process(process_raw, transport["binary_pin"], stage["max_wall_time_ms"])
                if not out or err:
                    raise ValueError("actual feedback CPU success requires receipt and empty stderr")
                response = legacy._parse(out)
                gain = _cpu_observation(request_raw, out, process_raw, snapshot["legal_moves"])
                budget.used["nodes"] -= 2 * stage["max_nodes_per_check"] - response["nodes"]
                counts["CPU_checks"] += int(response["baseline"] is not None) + int(response["after"] is not None)
                evidence_raw = ledger.append(input_sha256=decision["input_sha256"], request_raw=request_raw,
                    receipt_raw=out, process_raw=process_raw, legal_order=snapshot["legal_moves"], gain=gain)
                bank.write(stem + "-feedback-evidence.json", evidence_raw, credit)
                label = legacy.future_label(checked.derived_input()["input"]["snapshot"],
                    legacy.TaskContext(**checked.common_query()["context"]), task, request, response, _pin(evidence_raw)["sha256"])
                # A provenance label with every rank masked is not an admitted
                # utility/comparative target or a nonzero training loss.
                bank.write(stem + "-masked-label.json", _raw({"future_label": label, "task_rank_masked": True,
                    "ordinary_admission": False, "positive_training_target": False}), credit)
                episode["rounds"].append({"stem": stem, "semantic_pins": semantic_pins,
                    "semantic_input_sha256": checked.sha256, "derived_input_sha256": decision["input_sha256"],
                    "CPU_dispatched": True, "ledger_after_sha256": ledger.sha256})
                counts["selections"] += 1
                budget.check()
                credit.release()
                if task == "defer":
                    episode["stop"] = "actual_explicit_defer;no_new_CPU_coverage"
                    break
            if episode["stop"] is None:
                episode["stop"] = "finite_per_parent_rounds_exhausted_unresolved"
            if stop == "pre_dispatch_resource_limit_unresolved":
                break
        frontend._check_model()
        if _parent_pins(parents) != parent_before or _implementation_sha() != source_before or _implementation_sources() != implementation_before:
            raise ValueError("historical parents or feedback implementation mutated")
        for name in ("registration", "source", "capabilities", "semantic_registration", "semantic_source", "binary"):
            _, raw = legacy._file(paths[name], expected_registered_pins[name]["sha256"], expected_registered_pins[name]["bytes"])
            _actual(raw, expected_registered_pins[name])
        budget.check()
    except BaseException as error:
        failure = type(error).__name__ + ": " + str(error)[:1024]
    # Large private logits/latents are in their own reserved decision files.
    # The final receipt is a small pinned index, not an unbounded copy of them.
    index_raw = _raw({"schema": RECEIPT_SCHEMA, "episodes": episodes})
    _int(len(index_raw), 1, INDEX_BYTES)
    bank.write("episodes.json", index_raw, final=True)
    report = {"schema": RECEIPT_SCHEMA, "scope": SCOPE, "status": "failed" if failure else "checks_passed_before_final_receipt_publication",
        "failure": failure, "stop": stop, "counts": counts, "episodes": _pin(index_raw), "plan": expected_plan_pin,
        "parent": parent_before, "registered_pins": expected_registered_pins, "parameter_observation": expected_parameter_observation_pin,
        "producer_source_sha256": source_before, "resources": {"limits": plan["limits"], "used": budget.used},
        "implementation_sources": implementation_before,
        "artifacts": copy.deepcopy(bank.artifacts), "private_only": True, "historical_pre_result_inputs_unchanged": True,
        "budget_schedule_is_fixed_not_V_head": True, "cold_recomputation_only": True, "private_warm_supported": False,
        "CPU_stack_resume_across_children": False, "full_strategic_V_feedback_complete": False, "strategic_truth_admitted": False,
        "wdl_inferred": False, "comparative_ranks_admitted": False, "positive_training_target": False,
        "product_verifier_enabled": False, "training_steps": 0, "optimizer_created": False, "backward_executed": False,
        "registered_build_scope": "independently_pinned_caller_registration;source_to_binary_build_mapping_not_proved_here",
        "semantic_byte_scope": "sum_of_prepaid_per_round_semantic_owner_reservations;per_forward_peak_also_bounded",
        "flops_scope": "conservative_FMA2_matrix_product_bound_including_private_semantic_KV_attention_max_move_projection;norm_softmax_lookup_transfer_CPU_excluded",
        "wall_scope": "caller_absolute_deadline_including_caller_reload;receipt_final_publication_exit_cleanup_require_external_owner_observation"}
    report_raw = _raw(report)
    _int(len(report_raw), 1, FINAL_BYTES)
    bank.write("receipt.json", report_raw, final=True)
    budget.check(cleanup=True)
    if failure:
        error = RuntimeError("private V feedback failed; raw bank retained: " + failure)
        error.feedback_receipt_sha256 = _pin(report_raw)["sha256"]
        raise error
    return {**report, "receipt_sha256": _pin(report_raw)["sha256"], "status": "returned_before_external_exit_supervision"}


@dataclass(frozen=True)
class CheckedFeedbackHistory:
    """Conditional replayed raw bank; no training dataset/admission capability.

The independent owner's accepted receipt pin covers producer runtime/model
observations. Reload rechecks bytes, evidence chronology and semantic inputs;
it does not execute a child/model or establish caller independence.
"""
    _parent: object
    _raws: tuple
    _receipt_pin: bytes
    _registered_pins: bytes
    _parameter_pin: bytes
    _source_sha256: str

    def verify(self):
        return _replay(self._parent, dict(self._raws), legacy._parse(self._receipt_pin),
                       legacy._parse(self._registered_pins), legacy._parse(self._parameter_pin), self._source_sha256)

    def summary(self):
        return copy.deepcopy(self.verify())


def _replay(parents, raws, receipt_pin, registered_pins, parameter_pin, source_sha):
    receipt_raw = _actual(raws["receipt.json"], receipt_pin, FINAL_BYTES)
    report = legacy._parse(receipt_raw)
    if (report["schema"] != RECEIPT_SCHEMA or report["scope"] != SCOPE or report["failure"] is not None
            or report["status"] != "checks_passed_before_final_receipt_publication" or report["parent"] != _parent_pins(parents)
            or report["registered_pins"] != registered_pins or report["parameter_observation"] != parameter_pin
            or report["producer_source_sha256"] != semantic._sha(source_sha) or source_sha != _implementation_sha()
            or report["implementation_sources"] != _implementation_sources()
            or report["private_only"] is not True or report["cold_recomputation_only"] is not True
            or any(report[x] is not False for x in ("private_warm_supported", "CPU_stack_resume_across_children", "full_strategic_V_feedback_complete",
                "strategic_truth_admitted", "wdl_inferred", "comparative_ranks_admitted", "positive_training_target", "product_verifier_enabled",
                "optimizer_created", "backward_executed")) or type(report["training_steps"]) is not int or report["training_steps"] != 0):
        raise ValueError("accepted coverage-feedback receipt/current/source/no-learning scope differs")
    if set(raws) != set(report["artifacts"]) | {"receipt.json"}:
        raise ValueError("feedback bank immutable artifact inventory mismatch")
    for name, pin in report["artifacts"].items():
        _actual(raws[name], pin, MAX_BYTES, empty=True)
    plan = _plan(raws["plan.json"], parents, report["plan"])
    _stage_capabilities(plan, legacy._parse(raws["capabilities.json"]))
    _fields(report["resources"], ("limits", "used"))
    if report["resources"]["limits"] != plan["limits"]:
        raise ValueError("feedback fixed resource policy mismatch")
    for name in ("registration", "source", "capabilities", "semantic_registration", "semantic_source"):
        _actual(raws[name + ".json"], registered_pins[name], 4 << 20)
    _actual(raws["parameter-observation.json"], parameter_pin, 8192)
    parameter = _fields(legacy._parse(raws["parameter-observation.json"]), ("schema", "checkpoint_sha256", "parameter_sha256", "assurance_scope"))
    if parameter["schema"] != semantic.PARAMETER_SCHEMA or parameter["assurance_scope"] != "independently_pinned_caller_parameter_observation":
        raise ValueError("feedback parameter observation scope")
    semantic._sha(parameter["checkpoint_sha256"])
    semantic._sha(parameter["parameter_sha256"])
    index = _fields(legacy._parse(_actual(raws["episodes.json"], report["episodes"], INDEX_BYTES)), ("schema", "episodes"))
    if index["schema"] != RECEIPT_SCHEMA or type(index["episodes"]) is not list or len(index["episodes"]) > len(plan["parent_inputs"]):
        raise ValueError("bounded feedback episode index")
    steps = 0
    transitions = 0
    CPU_dispatches = 0
    CPU_checks = 0
    nodes = 0
    semantic_bytes = 0
    flops = 0
    games = set()
    setup_names = ("plan.json", "parameter-observation.json", "registration.json", "source.json", "capabilities.json", "semantic_registration.json", "semantic_source.json")
    reserved_usage = {"steps": 0, "nodes": 0, "forward_flops": 0, "semantic_bytes": 0,
        "output_bytes": FINAL_BYTES + INDEX_BYTES + sum(len(raws[name]) for name in setup_names)}
    if reserved_usage["output_bytes"] > plan["limits"]["max_output_bytes"]:
        raise ValueError("historical feedback setup reservation denied before any round")
    reserved_games = set()
    inventory = {"receipt.json", "plan.json", "parameter-observation.json", "registration.json", "source.json", "capabilities.json",
        "semantic_registration.json", "semantic_source.json", "episodes.json"}
    current = {parents.records[i]["input"]["sha256"]: i for i in parents.current_view.current_indices}
    for episode_number, episode in enumerate(index["episodes"]):
        _fields(episode, ("parent_input_sha256", "parent_index", "rounds", "stop"))
        identity = plan["parent_inputs"][episode_number]
        if episode["parent_input_sha256"] != identity or type(episode["parent_index"]) is not int or episode["parent_index"] != current[identity]:
            raise ValueError("feedback episode/current input order mismatch")
        parent_index = current[identity]
        snapshot = parents.records[parent_index]["input"]["snapshot"]
        ledger = FeedbackLedger()
        if type(episode["rounds"]) is not list or len(episode["rounds"]) > len(plan["stages"]):
            raise ValueError("finite indexed feedback rounds")
        for number, round_info in enumerate(episode["rounds"]):
            _fields(round_info, ("stem", "semantic_pins", "semantic_input_sha256", "derived_input_sha256", "CPU_dispatched", "ledger_after_sha256"))
            stage = plan["stages"][number]
            stem = "parent-" + str(parent_index) + "-round-" + str(number)
            if round_info["stem"] != stem or type(round_info["CPU_dispatched"]) is not bool:
                raise ValueError("feedback round ordinal/type mismatch")
            remaining = plan["limits"]["max_steps"] - reserved_usage["steps"]
            encoding = parents.encodings[identity]
            cost = _round_cost(stage, plan["limits"], ModelConfig(), encoding, snapshot)
            if not _reserve_usage(plan["limits"], reserved_usage, reserved_games, snapshot["game_id"], cost):
                raise ValueError("historical feedback reservation denied BEFORE round dispatch")
            common = next_common_query(parents, parent_index, stage, ledger, round_index=number, maximum_rounds=len(plan["stages"]),
                                      remaining_global_steps=remaining, binary_sha256=registered_pins["binary"]["sha256"])
            if common is None or _raw(common) != raws[stem + "-common_query.json"]:
                raise ValueError("next semantic input differs from actual previous CPU coverage/schedule")
            before = {"schema": BEFORE_SCHEMA, "scope": SCOPE, "plan": report["plan"], "parent_input_sha256": identity,
                "round": number, "previous_ledger_sha256": ledger.sha256, "previous_evidence_pins": [_pin(raw) for raw in ledger.entries()],
                "common_query": _pin(_raw(common)), "remaining_global_steps": remaining, "remaining_parent_rounds": len(plan["stages"]) - number,
                "budget_bucket_meaning": "min_fixed_remaining_parent_rounds_and_global_steps_div_maximum_parent_rounds",
                "budget_policy": "registered_schedule_not_V_budget_head", "warm_private_latent": False, "cpu_stack_continued_across_children": False}
            if raws[stem + "-feedback-before.json"] != _raw(before):
                raise ValueError("feedback immutable before-result evidence/schedule changed")
            inventory.update(stem + suffix for suffix in ("-feedback-before.json", "-common_query.json", "-before_result.json", "-request.json",
                "-Rules-dispatch.json", "-Rules-receipt.bin", "-Rules-stderr.bin", "-Rules-process.bin", "-Rules-launch.json", "-decision.json"))
            dispatch = {"feedback_before": _pin(_raw(before)), "semantic_before": _pin(raws[stem + "-before_result.json"]),
                "common_query": _pin(_raw(common)), "durable_before_dispatch": True}
            if legacy._parse(raws[stem + "-Rules-dispatch.json"]) != dispatch:
                raise ValueError("Rules dispatch was not bound to durable pre-result plan and current coverage")
            sem_raw = {"source": raws["semantic_source.json"], "registration": raws["semantic_registration.json"],
                "common_query": raws[stem + "-common_query.json"], "before_result": raws[stem + "-before_result.json"],
                "request": raws[stem + "-request.json"], "receipt": raws[stem + "-Rules-receipt.bin"], "launch_observation": raws[stem + "-Rules-launch.json"]}
            seen = _clean_process(raws[stem + "-Rules-process.bin"], registered_pins["binary"], stage["max_wall_time_ms"])
            if raws[stem + "-Rules-stderr.bin"] or seen["source_before"]["artifact"] != registered_pins["semantic_source"]:
                raise ValueError("feedback Rules actual stderr/source differs")
            _int(legacy._parse(sem_raw["receipt"])["elapsed_ms"], 0, seen["elapsed_ms"])
            launch = legacy._parse(sem_raw["launch_observation"])
            if launch["elapsed_ms"] != seen["elapsed_ms"]:
                raise ValueError("Rules launch total accounting differs from raw process observation")
            checked = semantic.admit_semantic_input(parent=parents, parent_index=parent_index,
                **{name + "_bytes": raw for name, raw in sem_raw.items()}, expected_pins=round_info["semantic_pins"])
            if (checked.sha256 != round_info["semantic_input_sha256"] or checked.derived_input()["input"]["sha256"] != round_info["derived_input_sha256"]
                    or checked.verify_parent(parents) != parent_index):
                raise ValueError("feedback semantic/derived/current input identity mismatch")
            decision = legacy._parse(raws[stem + "-decision.json"])
            if (decision["semantic_input_sha256"] != checked.sha256 or decision["input_sha256"] != round_info["derived_input_sha256"]
                    or decision["feedback_before"] != _pin(_raw(before)) or decision["preference_rank"] is not None
                    or decision["warm_continuation"] is not False or decision["budget_head_executed"] is not False):
                raise ValueError("feedback decision cold/non-training scope differs")
            logits = decision["task_logits"]
            private = decision["private"]
            if (type(logits) is not list or len(logits) != len(TASKS) or type(private["private_latent"]) is not list or len(private["private_latent"]) != 16 * 384
                    or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3.4028234663852886e38 for x in logits + private["private_latent"])):
                raise ValueError("retained finite CPU FP32 task logits/private latent required")
            eligible = checked.common_query()["eligible_tasks"]
            chosen = max((i for i, flag in enumerate(eligible) if flag), key=lambda i: (logits[i], -i))
            task = TASKS[chosen]
            _episode_action(task, number, len(episode["rounds"]))
            request = _request(snapshot, identity, stage, task, registered_pins["binary"]["sha256"])
            request["max_output_bytes"] = plan["limits"]["max_child_output_bytes"]
            request["context_sha256"] = legacy._hash(legacy.CPU_SCHEMA, request)
            request_raw = _raw(request) + b"\n"
            if (decision["task"] != task or private["task_kind"] != task or private["control_sha256"] != request["context_sha256"]
                    or decision["request"] != _pin(request_raw)):
                raise ValueError("actual V selection/request differs from stored pre-dispatch decision")
            audit = decision["forward_audit"]
            if (audit["parameter_sha256"] != parameter["parameter_sha256"] or audit["input_sha256"] != [checked.sha256]
                    or audit["private_tokens"] != semantic.TOKEN_COUNT or audit["actual_training_executed"] is not False
                    or audit["product_verifier_enabled"] is not False):
                raise ValueError("producer forward observation is not pinned frozen semantic scope")
            steps += 1
            transitions += int(number > 0)
            games.add(snapshot["game_id"])
            semantic_bytes += cost["semantic_bytes"]
            flops += cost["forward_flops"]
            if round_info["CPU_dispatched"]:
                inventory.update(stem + suffix for suffix in ("-CPU-request.json", "-CPU-stdout.bin", "-CPU-stderr.bin", "-CPU-process.bin",
                    "-feedback-evidence.json", "-masked-label.json"))
                if request_raw != raws[stem + "-CPU-request.json"] or raws[stem + "-CPU-stderr.bin"]:
                    raise ValueError("actual CPU request/stderr differed")
                process_raw = raws[stem + "-CPU-process.bin"]
                process = _clean_process(process_raw, registered_pins["binary"], stage["max_wall_time_ms"])
                if process["source_before"]["artifact"] != registered_pins["source"]:
                    raise ValueError("actual CPU registered source differed")
                out = raws[stem + "-CPU-stdout.bin"]
                gain = _cpu_observation(request_raw, out, process_raw, snapshot["legal_moves"])
                evidence = ledger.append(input_sha256=round_info["derived_input_sha256"], request_raw=request_raw,
                    receipt_raw=out, process_raw=process_raw, legal_order=snapshot["legal_moves"], gain=gain)
                if evidence != raws[stem + "-feedback-evidence.json"]:
                    raise ValueError("persisted CPU feedback evidence/raw chain changed")
                response = legacy._parse(out)
                nodes += response["nodes"]
                reserved_usage["nodes"] -= cost["nodes"] - response["nodes"]
                expected_label = {"future_label": legacy.future_label(checked.derived_input()["input"]["snapshot"],
                    legacy.TaskContext(**checked.common_query()["context"]), task, request, response, _pin(evidence)["sha256"]),
                    "task_rank_masked": True, "ordinary_admission": False, "positive_training_target": False}
                if legacy._parse(raws[stem + "-masked-label.json"]) != expected_label:
                    raise ValueError("feedback label/rank mask differs from raw CPU evidence")
                CPU_dispatches += 1
                CPU_checks += int(response["baseline"] is not None) + int(response["after"] is not None)
            else:
                inventory.add(stem + "-unresolved.json")
                nodes += 2 * stage["max_nodes_per_check"]  # Undispatched prepaid work stays conservatively charged.
                if number != len(episode["rounds"]) - 1 or stem + "-CPU-request.json" in raws:
                    raise ValueError("unresolved no-dispatch round cannot supply later coverage")
                if legacy._parse(raws[stem + "-unresolved.json"]) != {"reason": "fixed_task_wall_no_longer_available_before_CPU",
                        "unknown_observation": True, "positive_training_target": False}:
                    raise ValueError("missing unresolved-stop evidence")
            if round_info["ledger_after_sha256"] != ledger.sha256:
                raise ValueError("retained feedback ledger identity changed")
            actual_round_bytes = sum(len(raw) for name, raw in raws.items() if name.startswith(stem + "-"))
            _int(actual_round_bytes, 0, cost["output_bytes"])
            reserved_usage["output_bytes"] -= cost["output_bytes"] - actual_round_bytes
    expected_counts = {"selections": steps, "semantic_preparations": steps, "verifier_cold_forwards": steps,
        "CPU_dispatches": CPU_dispatches, "CPU_checks": CPU_checks, "feedback_transitions": transitions, "ranked_targets": 0}
    if set(raws) != inventory:
        raise ValueError("unexpected/missing feedback artifact outside indexed current history")
    _fields(report["counts"], expected_counts)
    if any(type(x) is not int for x in report["counts"].values()) or report["counts"] != expected_counts:
        raise ValueError("feedback physical cold-forward/CPU/transition accounting mismatch")
    expected_used = {"steps": steps, "nodes": nodes, "semantic_bytes": semantic_bytes, "forward_flops": flops,
        "output_bytes": FINAL_BYTES + INDEX_BYTES + sum(len(raw) for name, raw in raws.items() if name not in ("receipt.json", "episodes.json"))}
    _fields(report["resources"]["used"], expected_used)
    if report["resources"]["used"] != expected_used or reserved_usage != expected_used:
        raise ValueError("feedback prepaid/observed resource accounting differs")
    if len(games) > plan["limits"]["max_games"]:
        raise ValueError("feedback admitted games exceed pre-result allowance")
    for name, used in report["resources"]["used"].items():
        _int(used, 0, plan["limits"]["max_" + name])
    return {"scope": SCOPE, "receipt_sha256": receipt_pin["sha256"], "counts": expected_counts,
        "training_steps": 0, "ordinary_admission": False, "positive_training_target": False,
        "scope_of_replay": "accepted_caller_producer_observation_and_raw_bytes_rechecked;no_new_child_model_or_training_execution"}


def load_verifier_feedback_bank(directory, *, parents, expected_receipt_pin, expected_registered_pins,
        expected_parameter_observation_pin, expected_producer_source_sha256, registered_paths, deadline, max_input_bytes=128 << 20):
    """Separate strict raw bank replay. The legacy bank reader is unchanged.

An external owner must independently accept actual producer exit/cleanup/time
and provide the receipt pin. No fabricated observation object/factory callback
opens this reader. It never returns an ordinary training ValidatedDataset.
"""
    semantic._deadline(deadline)  # Reject invalid/expired caller clocks BEFORE paths/reads.
    _int(max_input_bytes, 1, 128 << 20)
    root = legacy._path(directory)
    if any((p / ".git").exists() for p in (root, *root.parents)):
        raise ValueError("feedback bank must be outside Git")
    remaining = max_input_bytes
    def read(name, expected, empty=False):
        nonlocal remaining
        if type(name) is not str or Path(name).name != name or name in (".", ".."):
            raise ValueError("feedback artifact basename required")
        _fields(expected, ("bytes", "sha256"))
        size = _int(expected["bytes"], 0 if empty else 1, min(remaining, FINAL_BYTES if name == "receipt.json" else MAX_BYTES))
        remaining -= size  # Deduct BEFORE allocation/read.
        semantic._deadline(deadline)
        path = legacy._path(root / name)
        if size == 0:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_size != 0:
                raise ValueError("pinned empty feedback artifact differs")
            raw = b""
        else:
            _, raw = legacy._file(path, expected["sha256"], size)
        semantic._deadline(deadline)
        return _actual(raw, expected, MAX_BYTES, empty=empty)
    raws = {"receipt.json": read("receipt.json", expected_receipt_pin)}
    report = legacy._parse(raws["receipt.json"])
    artifacts = report["artifacts"]
    if type(artifacts) is not dict or len(artifacts) > 64 * 16 * 18 + 16 or "receipt.json" in artifacts:
        raise ValueError("bounded acyclic feedback artifact inventory required")
    for name, pin in artifacts.items():
        raws[name] = read(name, pin, empty=True)
    plan = _plan(raws["plan.json"], parents, report["plan"])
    assets, _, _, capabilities = _assets(registered_paths, expected_registered_pins, plan["stages"][0], deadline, remaining)
    semantic._deadline(deadline)
    _stage_capabilities(plan, capabilities)
    for name, raw in assets.items():
        if name != "binary" and raw != raws[name + ".json"]:
            raise ValueError("actual independent registration paths differ from accepted bank")
    result = CheckedFeedbackHistory(parents, tuple(sorted(raws.items())), _raw(expected_receipt_pin),
        _raw(expected_registered_pins), _raw(expected_parameter_observation_pin), semantic._sha(expected_producer_source_sha256))
    semantic._deadline(deadline)
    result.verify()
    semantic._deadline(deadline)
    return result
