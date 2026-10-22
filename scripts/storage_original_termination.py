"""Pure replay of a caller-pinned, original run003b termination observation.

This module consumes bytes, never paths, handles, callbacks or approval flags.
Pins must come from the caller's independently preserved original source/evidence
roster. Matching caller-supplied hashes establish byte identity, not trust in an
arbitrary replacement producer. The supported producer declarations are the
original run003b graph; its four source roles are read as text, never imported.

An accepted observation is NOT recovery authority. It issues no closure, ACK,
owner adoption, deletion permission, current reboot proof or product/GPU success.
The original Windows Job report and Linux PG report remain reports by their
original pinned actors. The Linux originally live-bound pidfd/cgroup raw is also
replayed. Arbitrarily migrated descendants and an independent Linux host identity
are outside that capture's scope. Original failure and physical unknown survive.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re


NAMESPACE = "archive-build151-originals-N05-run003"
PRODUCT_SOURCE = "b5671299f2a62fe660c57877d57e15fe10bc0730"
BUILD_SOURCE = "315722cf98936b95018acc7c2c468473fc2bf4d9"
CONTROL_SOURCE = "212c57b65d5002d66274b291299176cdd8bc7112"
SOURCE_NAMES = {
    "windows_parent": "archive-build151-originals-N05-run003b-Windows-parent-source.py",
    "windows_driver": "archive-build151-originals-N05-run003b-Windows-driver-source.py",
    "linux_capture": "archive-build151-originals-N05-run003b-linux-capture-source.py",
    "linux_parent": "archive-build151-originals-N05-run003-linux-outer-source.py",
}
EVIDENCE_NAMES = {
    "windows_parent": "build151-N05-run003-Windows-parent-result.json",
    "windows_supervision": "build151-N05-run003-Windows-supervision.json",
    "linux_parent": "build151-N05-run003-linux-parent-result.json",
    "linux_capture": "build151-N05-run003-linux-capture-result.json",
    "transcript": "build151-N05-run003-systemd-transcript.log",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_BOOT = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")
_SHOW_PROPERTIES = frozenset(("Id", "Description", "LoadState", "ActiveState", "SubState",
    "InvocationID", "ControlGroup", "MainPID", "ControlPID", "ExecMainPID",
    "ExecMainCode", "ExecMainStatus", "Result", "Job"))
_SCHEMAS = {
    "windows_parent": "rovent.build151-retirement-original-Windows-parent/2",
    "windows_supervision": "rovent.build151-retirement-Windows-supervision/2",
    "linux_parent": "rovent.build151-retirement-linux-outer/2",
    "linux_capture": "rovent.build151-retirement-linux-capture/2",
}
# Closed producer field vocabulary; every consumed field is required separately.
_FIELDS = {
    "windows_parent": frozenset("Linux_writer_closure_verified Windows_Job_tree_gone Windows_affinity driver_exit_code exit_code failures namespace no_submission_unproven original_Job_supervisor_error original_Windows_owner original_manager_finish_called original_parent_cleanup_state original_parent_dispatch_attempted physical_completion pre_driver_dispatch_clock_gate pre_manager_acquire_clock_gate pre_owner_clock_gate prior_owner_adopted schema status unfinished_original_owner_preserved unit".split()),
    "windows_supervision": frozenset("Linux_writer_closure_verified Windows_Job_closure_issued Windows_affinity admission_rawpin capture_client_exit_code capture_command capture_registration clock_observation exit_code failures functional_exact3_success namespace native_intake_complete original_systemd_command original_windows_parent output output_descriptor output_original_owner physical_completion pre_Linux_admission pre_first_WSL_reservation_clock_gate pre_output_lease_clock_gate pre_owner_clock_gate prior_owner_adopted registration schema source_descriptors_closed source_registry status unit".split()),
    "linux_parent": frozenset("actual_original_owner actual_original_owner_raw_sha256 admission_rawpin child_argv dispatch_rawpin exit_code failures finish_refusal finished_monotonic first_error journal_copy_complete manager_finish_called namespace native_Windows_ack native_registry_rawpin old007_closure_issued old_records_1_to_8_modified optimizer_updates original_child_exit original_child_supervisor_error original_child_tree_gone original_cleanup_until original_managed_terminal original_owner_and_scratch_preserved original_work_until physical_reclamation private_mount_observation product_execution ready_pin registration schema source_descriptors_closed status unit".split()),
    "linux_capture": frozenset("admission_rawpin cancel_observations cancellation_requested capture_descriptor_closed capture_scope cgroup_descriptors_closed config controller_source exit_code failure_rows_omitted failures manager_finish_allowed namespace native_registry_rawpin observed_capture_finish observed_capture_start old007_closure_issued optimizer_updates original_cleanup_until original_process_descriptors_closed original_stream_eof original_systemd_client_pid original_systemd_exit_code original_systemd_wait_completed original_transcript original_unit original_window_start original_work_until outer_source performance_evidence primary_failure product_execution registration result_cap_bytes retirement_accepted schema source_descriptors_closed source_only_proposal_tested systemd_command transcript_cap_bytes transcript_overflow unit unit_preflight unit_product_result_metadata unit_product_result_status unit_terminal unit_terminal_verified unrecorded_stream_bytes".split()),
}
# A missing nullable observation is unknown, not the producer's explicit null.
_REQUIRED_NULLABLE = {
    "windows_parent": frozenset(("original_Job_supervisor_error",)),
    "windows_supervision": frozenset(),
    "linux_parent": frozenset(("original_child_supervisor_error", "ready_pin",
                               "native_Windows_ack", "original_managed_terminal")),
    "linux_capture": frozenset(("unit_product_result_metadata",)),
}


class RejectionCode(str, Enum):
    INPUT = "InputShape"
    PIN = "RawPinMismatch"
    JSON = "InvalidJson"
    SOURCE = "SourceGraphMismatch"
    SCHEMA = "UnsupportedSchema"
    JOIN = "OriginalCrosslinkMismatch"
    KERNEL = "OriginalKernelEvidenceMismatch"
    TRANSCRIPT = "OriginalTranscriptMismatch"


class OriginalTerminationError(ValueError):
    """A bounded, value-free reason; owner tokens never enter diagnostics."""

    def __init__(self, code: RejectionCode, detail: str):
        self.code = code
        super().__init__(code.value + ": " + detail)


def _need(condition, code, detail):
    if condition is not True:
        raise OriginalTerminationError(code, detail)


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class PinnedBytes:
    """An immutable caller-declared original raw, without filesystem authority."""

    name: str
    size: int
    sha256: str
    raw: bytes = field(repr=False)

    def _checked(self, name, cap):
        _need(type(self) is PinnedBytes and type(self.name) is str and self.name == name
              and type(self.size) is int and 0 < self.size <= cap
              and type(self.sha256) is str and _HEX.fullmatch(self.sha256) is not None
              and type(self.raw) is bytes, RejectionCode.INPUT, "bounded immutable original bytes")
        _need(len(self.raw) == self.size and hashlib.sha256(self.raw).hexdigest() == self.sha256,
              RejectionCode.PIN, "original byte count and SHA-256")
        return self.raw

    def _public(self):
        return {"name": self.name, "bytes": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class OriginalTerminationObservation:
    """Static original evidence only; this object is not a live capability."""

    namespace: str
    terminal_branch: str
    unit_product_result_status: str
    _sources: tuple[PinnedBytes, ...] = field(repr=False)
    _evidence: tuple[PinnedBytes, ...] = field(repr=False)
    _windows_owner_raw: bytes = field(repr=False)
    _linux_owner_raw: bytes = field(repr=False)
    _output_owner_raw: bytes = field(repr=False)
    _summary_raw: bytes = field(repr=False)

    def redacted_report(self):
        """Return an independent, explicit projection; never dump owner documents."""
        result = json.loads(self._summary_raw)
        result.update({
            "schema": "rovezero.storage-original-termination-observation.v1",
            "namespace": self.namespace,
            "source_pins": [value._public() for value in self._sources],
            "evidence_pins": [value._public() for value in self._evidence],
            "terminal_branch": self.terminal_branch,
            "unit_product_result_status": self.unit_product_result_status,
            "recovery_authorized": False,
            "closure_issued": False,
            "owner_adopted": False,
            "deletion_authorized": False,
            "current_owner_death_observed": False,
            "current_linux_reboot_observed": False,
            "current_linux_same_host_reboot_requires_separate_live_probe": True,
            "independent_linux_host_identity": "unknown",
            "owner_json_raw_replayed": False,
            "source_registry_raw_replayed": False,
            "windows_job_observation_scope": "original_pinned_parent_report_only",
            "linux_pg_observation_scope": "original_pinned_linux_parent_report_only",
            "linux_kernel_replay_scope": "original_live_bound_main_pidfd_and_original_cgroup_only",
            "arbitrarily_migrated_descendants_closed": False,
        })
        return result


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, RejectionCode.JSON, "duplicate JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise OriginalTerminationError(RejectionCode.JSON, "nonfinite JSON number")

    try:
        value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
                           parse_constant=constant)
        stack, count = [(value, 0)], 0
        while stack:
            item, depth = stack.pop()
            count += 1
            _need(count <= 12000 and depth <= 40, RejectionCode.JSON, "bounded JSON tree")
            if type(item) is dict:
                stack.extend((child, depth + 1) for child in item.values())
            elif type(item) is list:
                stack.extend((child, depth + 1) for child in item)
            elif type(item) is float:
                _need(math.isfinite(item), RejectionCode.JSON, "finite JSON number")
        return value
    except OriginalTerminationError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as error:
        raise OriginalTerminationError(RejectionCode.JSON, "strict bounded UTF-8 JSON") from error


def _source_literals(raw, wanted):
    """Inspect only constant declarations; never import/execute original sources."""
    try:
        tree = ast.parse(raw.decode("utf-8", "strict"))
        values = {}

        def literal(node):
            if isinstance(node, ast.Constant) and type(node.value) in (str, int):
                return node.value
            if isinstance(node, ast.Name) and node.id in values:
                return values[node.id]
            if isinstance(node, ast.Tuple):
                return tuple(literal(item) for item in node.elts)
            if isinstance(node, ast.Dict):
                result = {}
                for key, value in zip(node.keys, node.values):
                    key = literal(key)
                    _need(type(key) is str and key not in result, RejectionCode.SOURCE,
                          "unique literal source declaration key")
                    result[key] = literal(value)
                return result
            if isinstance(node, ast.Subscript):
                return literal(node.value)[literal(node.slice)]
            raise OriginalTerminationError(RejectionCode.SOURCE, "nonliteral source declaration")

        for statement in tree.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id in wanted:
                    _need(target.id not in values, RejectionCode.SOURCE, "one source declaration")
                    values[target.id] = literal(statement.value)
        _need(set(values) == set(wanted), RejectionCode.SOURCE, "complete source declarations")
        return values
    except OriginalTerminationError:
        raise
    except (UnicodeError, SyntaxError, ValueError, TypeError, KeyError, IndexError,
            RecursionError, OverflowError) as error:
        raise OriginalTerminationError(RejectionCode.SOURCE, "original source declaration parse") from error


def _check_sources(sources):
    declarations = {
        "windows_parent": {"NS", "DRIVER"},
        "windows_driver": {"NS", "SOURCE", "BUILD_SOURCE", "CONTROL_COMMIT", "PROTOCOL", "ROLE_NAMES"},
        "linux_capture": {"NAMESPACE", "UNIT", "CAPTURE_SOURCE_NAME", "LINUX_OUTER_SOURCE_NAME",
                          "BUILD_SOURCE", "STORAGE_SOURCE", "SOURCE_ROLE_NAMES"},
        "linux_parent": {"NS", "SOURCE_FILENAME", "CURRENT", "BUILD_SOURCE"},
    }
    values = {role: _source_literals(sources[role].raw, declarations[role]) for role in declarations}
    parent, driver = values["windows_parent"], values["windows_driver"]
    capture, outer = values["linux_capture"], values["linux_parent"]
    roles = {
        "windows_outer": SOURCE_NAMES["windows_parent"],
        "windows_driver": SOURCE_NAMES["windows_driver"],
        "capture": SOURCE_NAMES["linux_capture"], "linux_outer": SOURCE_NAMES["linux_parent"],
        "bridge": "archive-build151-originals-N05-run003-native-bridge-source.py",
        "consumer": "archive-build151-originals-N05-run003-consumer-source.py",
        "context": "storage-current-source7-context-001.py",
        "source7_config": "currentb567-plain-source7-configuration-main-001.json",
        "protocol": "old738-original-pre-linux-admission-protocol-003.py",
    }
    _need(parent == {"NS": NAMESPACE, "DRIVER": SOURCE_NAMES["windows_driver"]}
          and driver["NS"] == capture["NAMESPACE"] == capture["UNIT"] == outer["NS"] == NAMESPACE
          and driver["SOURCE"] == capture["STORAGE_SOURCE"] == outer["CURRENT"] == PRODUCT_SOURCE
          and driver["BUILD_SOURCE"] == capture["BUILD_SOURCE"] == outer["BUILD_SOURCE"] == BUILD_SOURCE
          and driver["CONTROL_COMMIT"] == CONTROL_SOURCE
          and driver["ROLE_NAMES"] == capture["SOURCE_ROLE_NAMES"] == roles
          and driver["PROTOCOL"] == (roles["protocol"], 18383,
              "49176633611c2e242f47ee8f741e9da7a81259d6023103e3cbacfeccf58a7912")
          and capture["CAPTURE_SOURCE_NAME"] == SOURCE_NAMES["linux_capture"]
          and capture["LINUX_OUTER_SOURCE_NAME"] == outer["SOURCE_FILENAME"] == SOURCE_NAMES["linux_parent"],
          RejectionCode.SOURCE, "original 3b source-role graph and distinct product/control domains")


def _rawpin(value, name=None):
    _need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
          and type(value["path"]) is str and value["path"].startswith("/mnt/c/")
          and "\\" not in value["path"] and ".." not in value["path"].split("/")
          and type(value["bytes"]) is int and 0 < value["bytes"] <= 512 << 10
          and type(value["sha256"]) is str and _HEX.fullmatch(value["sha256"]) is not None,
          RejectionCode.JOIN, "closed original embedded raw pin")
    if name is not None:
        _need(value["path"].rsplit("/", 1)[-1] == name, RejectionCode.JOIN,
              "exact original embedded basename")
    return value


def _pin_matches(value, pinned):
    _rawpin(value, pinned.name)
    _need(value["bytes"] == pinned.size and value["sha256"] == pinned.sha256,
          RejectionCode.JOIN, "embedded source/evidence raw pin equals supplied original bytes")


def _owner(value, platform):
    _need(type(value) is dict and set(value) == ({"schema", "pid", "purpose", "root_identity",
          "source", "source_id", "token", "lifetime"} | ({"state", "finished_ns"} if platform == "linux" else set()))
          and value.get("schema") == "rovezero.managed-build.v1"
          and type(value.get("pid")) is int and value["pid"] > 0
          and type(value.get("token")) is str and re.fullmatch("[0-9a-f]{32}", value["token"]) is not None
          and type(value.get("root_identity")) is list and len(value["root_identity"]) == 2
          and all(type(n) is int and n > 0 for n in value["root_identity"])
          and type(value.get("source")) is str and bool(value["source"])
          and type(value.get("source_id")) is str and re.fullmatch("[0-9a-f]{24}", value["source_id"]) is not None,
          RejectionCode.JOIN, "original managed-build owner shape")
    life = value.get("lifetime")
    _need(type(life) is dict and set(life) == {"host", "boot", "platform", "process_birth"}
          and life["platform"] == platform and type(life["host"]) is str
          and _HEX.fullmatch(life["host"]) is not None and type(life["process_birth"]) is str
          and life["process_birth"].isascii() and life["process_birth"].isdecimal()
          and int(life["process_birth"]) > 0, RejectionCode.JOIN, "original lifetime shape")
    if platform == "linux":
        _need(type(life["boot"]) is str and _BOOT.fullmatch(life["boot"]) is not None
              and value["state"] == "active" and type(value["finished_ns"]) is int
              and value["finished_ns"] == 0, RejectionCode.JOIN, "original Linux owner lifetime")
    else:
        _need(type(life["boot"]) is str
              and re.fullmatch(r"windows-boot-counter-v1:[1-9][0-9]{0,9}", life["boot"]) is not None,
              RejectionCode.JOIN, "original Windows boot-counter domain")
    return value


def _identity(value):
    return (type(value) is list and len(value) == 6
            and all(type(n) is int and n >= 0 for n in value)
            and value[0] > 0 and value[1] > 0 and value[2] > 0)


def _show(evidence):
    _need(type(evidence) is dict and set(evidence) == {"command", "exit_code", "stdout", "stderr", "properties"}
          and type(evidence.get("exit_code")) is int and type(evidence.get("stderr")) is str
          and len(evidence["stderr"].encode()) <= 64 << 10 and type(evidence.get("stdout")) is str
          and 0 < len(evidence["stdout"].encode()) <= 64 << 10
          and type(evidence.get("command")) is list and len(evidence["command"]) == 5
          and evidence["command"][:3] == ["/usr/bin/systemctl", "show", "--no-pager"]
          and type(evidence["command"][3]) is str and evidence["command"][3].startswith("--property=")
          and evidence["command"][-1] == NAMESPACE + ".service",
          RejectionCode.KERNEL, "bounded exact-unit raw systemctl show")
    requested = evidence["command"][3][len("--property="):].split(",")
    _need(len(requested) == len(_SHOW_PROPERTIES) and set(requested) == _SHOW_PROPERTIES,
          RejectionCode.KERNEL, "exact original requested systemctl property set")
    values = {}
    for line in evidence["stdout"].splitlines():
        _need("=" in line, RejectionCode.KERNEL, "whole systemctl property line")
        key, value = line.split("=", 1)
        _need(key and key not in values, RejectionCode.KERNEL, "unique systemctl property")
        values[key] = value
    _need(set(values) == _SHOW_PROPERTIES and _same(values, evidence["properties"])
          and values.get("Id") == NAMESPACE + ".service",
          RejectionCode.KERNEL, "independent raw systemctl property replay")
    return values


def _parent_projection(value):
    _need(type(value) is dict and set(value) == {"path", "identity", "name_identity", "identity_at_bind",
          "nlink_used_as_identity", "readlink"}
          and value["path"] == value["readlink"] == "/sys/fs/cgroup/system.slice"
          and value["nlink_used_as_identity"] is False and _identity(value["identity"])
          and _identity(value["identity_at_bind"]) and _same(value["identity"], value["name_identity"])
          and value["identity"][:5] == value["identity_at_bind"][:5],
          RejectionCode.KERNEL, "original held cgroup parent FD/name core")
    return (value["path"], value["readlink"], value["identity"][:5], value["identity_at_bind"])


def _kernel_replay(cap, linux_owner):
    before, after = cap.get("original_unit"), cap.get("unit_terminal")
    _need(type(before) is dict and type(after) is dict, RejectionCode.KERNEL, "original unit binding and terminal")
    description = NAMESPACE + " admission=" + cap["admission_rawpin"]["sha256"] + " registration=" + cap["registration"]["sha256"]
    group = "/system.slice/" + NAMESPACE + ".service"
    group_path = "/sys/fs/cgroup" + group
    invocation = before.get("InvocationID")
    _need(type(invocation) is str and re.fullmatch("[0-9a-f]{32}", invocation) is not None
          and before.get("ControlGroup") == after.get("ControlGroup") == group
          and before.get("Description") == after.get("Description") == description
          and after.get("InvocationID") == invocation, RejectionCode.KERNEL, "same original invocation and Description")
    first, confirm = _show(before.get("show")), _show(before.get("confirmation_show"))
    for raw, values in ((before["show"], first), (before["confirmation_show"], confirm)):
        _need(raw["exit_code"] == 0 and raw["stderr"] == "" and values.get("LoadState") == "loaded"
              and values.get("InvocationID") == invocation and values.get("Description") == description
              and values.get("ControlGroup") == group and values.get("MainPID") == first.get("MainPID")
              and type(values.get("MainPID")) is str and values["MainPID"].isascii()
              and values["MainPID"].isdecimal() and int(values["MainPID"]) > 0,
              RejectionCode.KERNEL, "original live MainPID observations around held pidfd")
    disc = after.get("terminal_discriminator")
    _need(type(disc) is dict and disc.get("schema") == "rovent.build151-original-cgroup-terminal-discriminator/3"
          and "product_result_metadata" in disc
          and disc.get("branch") in ("live_cgroup_empty", "removed_original_cgroup")
          and disc.get("client_exit_used_as_closure_proof") is False
          and disc.get("unit_name_absence_used_as_closure_proof") is False,
          RejectionCode.KERNEL, "explicit original cgroup discriminator")
    bound, terminal = disc.get("original_binding"), disc.get("terminal_readback")
    _need(type(bound) is dict and type(terminal) is dict and bound.get("InvocationID") == invocation
          and bound.get("ControlGroup") == group and bound.get("Description") == description
          and _same(bound.get("main_process"), before.get("main_process")),
          RejectionCode.KERNEL, "discriminator joins originally live-bound process")
    process, dead = bound.get("main_process"), terminal.get("main_process")
    _need(type(process) is dict and type(dead) is dict and type(process.get("pid")) is int
          and process["pid"] == int(first["MainPID"]) == linux_owner["pid"]
          and type(process.get("starttime_ticks")) is int and process["starttime_ticks"] > 0
          and str(process["starttime_ticks"]) == linux_owner["lifetime"]["process_birth"]
          and process.get("proc_cgroup") == "0::" + group
          and type(process.get("pidfd_open_flags")) is int and process["pidfd_open_flags"] == 0
          and process.get("pidfd_dead_at_bind") is False,
          RejectionCode.KERNEL, "MainPID and real birth join original Linux managed owner")
    birth = {"starttime_ticks": process["starttime_ticks"], "proc_cgroup": process["proc_cgroup"]}
    _need(_same(process.get("proc_observation_before_pidfd_open"), birth)
          and _same(process.get("proc_observation_after_pidfd_open"), birth)
          and _identity(process.get("proc_directory_identity")) and _identity(process.get("pidfd_identity"))
          and _same(process.get("pidfd_poll_at_bind"), {"pidfd_dead": False, "poll_mask": 0, "poll_events": []}),
          RejectionCode.KERNEL, "held pidfd bracketed by identical live birth and cgroup")
    info = process.get("pidfd_fdinfo_at_bind")
    _need(type(info) is dict and info.get("Pid") == str(process["pid"])
          and type(info.get("NSpid")) is str and bool(info["NSpid"].split())
          and info["NSpid"].split()[0] == str(process["pid"])
          and all(x.isascii() and x.isdecimal() and int(x) > 0 for x in info["NSpid"].split()),
          RejectionCode.KERNEL, "original live pidfd fdinfo process namespace")
    mask = dead.get("poll_mask")
    _need(dead.get("pidfd_dead") is True and type(mask) is int and mask > 0 and mask & ~17 == 0
          and dead.get("poll_events") == [name for flag, name in ((1, "POLLIN"), (16, "POLLHUP")) if mask & flag]
          and type(dead.get("pid")) is int and dead["pid"] == process["pid"]
          and type(dead.get("starttime_ticks")) is int and dead["starttime_ticks"] == process["starttime_ticks"]
          and _same(dead.get("pidfd_identity"), process["pidfd_identity"])
          and dead.get("birth_reconstructed_after_death") is False,
          RejectionCode.KERNEL, "same originally live-bound pidfd death without PID reopening")
    original, final_group = bound.get("cgroup"), terminal.get("cgroup")
    _need(type(original) is dict and type(final_group) is dict and _same(final_group, after.get("cgroup"))
          and final_group.get("InvocationID") == invocation and final_group.get("ControlGroup") == group
          and _identity(original.get("directory_identity"))
          and _same(original["directory_identity"], before["cgroup"].get("directory_identity"))
          and _same(original["directory_identity"], final_group.get("directory_identity"))
          and type(original.get("file_identities")) is dict
          and set(original["file_identities"]) == {"cgroup.events", "cgroup.procs"}
          and all(_identity(v) for v in original["file_identities"].values())
          and _same(original["file_identities"], before["cgroup"].get("file_identities"))
          and _same(original["file_identities"], final_group.get("file_identities"))
          and original["file_identities"]["cgroup.events"][:2] != original["file_identities"]["cgroup.procs"][:2],
          RejectionCode.KERNEL, "same original cgroup directory and two distinct held metadata FDs")
    parent = _parent_projection(original.get("parent"))
    _need(_same(_parent_projection(before["cgroup"].get("parent")), parent)
          and _same(_parent_projection(final_group.get("parent")), parent),
          RejectionCode.KERNEL, "original cgroup parent core across observations")
    mount = original.get("mount_context")
    _need(type(mount) is dict and _same(mount, before["cgroup"].get("mount_context"))
          and _same(mount, final_group.get("mount_context")) and mount.get("boot_id") == linux_owner["lifetime"]["boot"]
          and type(mount.get("mount")) is dict and mount["mount"].get("filesystem") == "cgroup2"
          and mount["mount"].get("mount_point") == "/sys/fs/cgroup" and mount["mount"].get("root") == "/"
          and type(mount["mount"].get("mount_id")) is int and mount["mount"]["mount_id"] > 0
          and type(mount.get("fd_mount_ids")) is dict
          and set(mount["fd_mount_ids"]) == {"parent", "directory", "cgroup.events", "cgroup.procs"}
          and all(type(v) is int and v == mount["mount"]["mount_id"] for v in mount["fd_mount_ids"].values()),
          RejectionCode.KERNEL, "original owner boot and actual held cgroup2 mount context")
    final = _show(after.get("show"))
    loaded = (after["show"]["exit_code"] == 0 and after["show"]["stderr"] == "" and final.get("LoadState") == "loaded"
              and final.get("InvocationID") == invocation and final.get("Description") == description
              and final.get("ControlGroup") in (group, "") and final.get("ActiveState") in ("inactive", "failed")
              and final.get("SubState") in ("dead", "failed") and final.get("MainPID") == final.get("ControlPID") == "0"
              and final.get("Job") == "")
    absent = (final.get("LoadState") == "not-found" and final.get("InvocationID") == ""
              and final.get("ControlGroup") == "" and final.get("MainPID") == final.get("ControlPID") == "0"
              and final.get("Job") == "")
    if disc["branch"] == "live_cgroup_empty":
        _need(loaded and final_group.get("directory_readlink") == group_path
              and _same(final_group.get("old_name"), {"state": "same_original", "identity": original["directory_identity"]})
              and type(final_group.get("events")) is dict and type(final_group["events"].get("populated")) is int
              and final_group["events"]["populated"] == 0 and final_group.get("procs") == [],
              RejectionCode.KERNEL, "original live cgroup empty and same process dead")
    else:
        _need((loaded or absent) and final_group.get("directory_readlink") == group_path + " (deleted)"
              and _same(final_group.get("old_name"), {"state": "ENOENT", "errno": 2})
              and final_group.get("link_count_used_as_removal_proof") is False
              and type(final_group.get("metadata")) is dict and set(final_group["metadata"]) == {"cgroup.events", "cgroup.procs"}
              and _same(_parent_projection(final_group.get("parent_final")), parent),
              RejectionCode.KERNEL, "held original deleted cgroup and absent original name")
        for name in ("cgroup.events", "cgroup.procs"):
            _need(_same(final_group["metadata"][name], {"identity": original["file_identities"][name],
                  "read_state": "ENODEV", "errno": 19}), RejectionCode.KERNEL, "both original held metadata FDs report ENODEV19")
    status, metadata = disc.get("product_result_status"), disc.get("product_result_metadata")
    _need(status == after.get("product_result_status") == cap.get("unit_product_result_status")
          and _same(metadata, cap.get("unit_product_result_metadata")), RejectionCode.KERNEL, "separate exact product metadata")
    if loaded:
        _need(all(key in final and type(final[key]) is str for key in ("Result", "ExecMainCode", "ExecMainStatus")),
              RejectionCode.KERNEL, "loaded terminal contains all three original product result fields")
        actual = {key: final[key] for key in ("Result", "ExecMainCode", "ExecMainStatus")}
        expected = "observed_success" if actual == {"Result": "success", "ExecMainCode": "1", "ExecMainStatus": "0"} else "observed_failure"
        _need(_same(metadata, actual) and status == expected, RejectionCode.KERNEL, "actual product result retained without success fabrication")
    else:
        _need(disc["branch"] == "removed_original_cgroup" and status == "unknown" and metadata is None,
              RejectionCode.KERNEL, "garbage-collected product metadata remains unknown")
    _need(after.get("original_systemd_wait_completed") is True and type(after.get("original_systemd_exit_code")) is int
          and after["original_systemd_exit_code"] == cap["original_systemd_exit_code"]
          and _finite(process.get("bound_monotonic")) and _finite(dead.get("readback_monotonic"))
          and _finite(final_group.get("readback_monotonic"))
          and 0 < process["bound_monotonic"] <= final_group["readback_monotonic"] <= dead["readback_monotonic"],
          RejectionCode.KERNEL, "original wait and ordered live-bind/terminal readback clocks")
    return disc["branch"], status


def _commands(native, cap, sources, linux_owner):
    command = native.get("capture_command")
    _need(type(command) is list and all(type(v) is str for v in command)
          and command[:10] == ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--", "/usr/bin/python3.12", "-I", "-S", "-B"]
          and len(command) == 21, RejectionCode.JOIN, "original exact isolated WSL capture command")
    pin = cap["controller_source"]
    _need(command[10:] == [pin["path"], "--config", cap["config"]["path"], "--config-bytes", str(cap["config"]["bytes"]),
          "--config-sha256", cap["config"]["sha256"], "--self-bytes", str(pin["bytes"]), "--self-sha256", pin["sha256"]],
          RejectionCode.JOIN, "original capture command source/config exact pins")
    systemd = native.get("original_systemd_command")
    _need(type(systemd) is list and all(type(v) is str for v in systemd) and _same(systemd, cap.get("systemd_command"))
          and systemd[:4] == ["/usr/bin/systemd-run", "--wait", "--pipe", "--unit=" + NAMESPACE],
          RejectionCode.JOIN, "same original systemd invocation")
    description = NAMESPACE + " admission=" + cap["admission_rawpin"]["sha256"] + " registration=" + cap["registration"]["sha256"]
    _need(systemd[4:15] == ["--property=Description=" + description, "--property=CPUQuota=200%",
          "--property=AllowedCPUs=0,2", "--property=MemoryHigh=6G", "--property=MemoryMax=12G",
          "--property=RuntimeMaxSec=660", "--property=TimeoutStopSec=20", "--property=TasksMax=256",
          "--property=LimitNOFILE=131072", "--property=PrivateMounts=yes", "--property=UMask=0077"]
          and len(systemd) == 34 and systemd[15] == "--working-directory=" + linux_owner["source"]
          and systemd[16:25] == ["/usr/bin/unshare", "--mount", "--propagation", "private", "--",
                               "/usr/bin/python3.12", "-I", "-S", "-B"]
          and systemd[25:] == [cap["outer_source"]["path"], "--registration", cap["registration"]["path"],
                               str(cap["registration"]["bytes"]), cap["registration"]["sha256"],
                               "--self-bytes", str(sources["linux_parent"].size), "--self-sha256", sources["linux_parent"].sha256],
          RejectionCode.JOIN, "original resources and outer source/runtime command pins")


def _verify(sources, evidence):
    _need(type(sources) is dict and len(sources) == len(SOURCE_NAMES)
          and all(type(key) is str for key in sources) and set(sources) == set(SOURCE_NAMES)
          and type(evidence) is dict and len(evidence) == len(EVIDENCE_NAMES)
          and all(type(key) is str for key in evidence) and set(evidence) == set(EVIDENCE_NAMES),
          RejectionCode.INPUT, "closed four-source/five-evidence roster")
    for role, name in SOURCE_NAMES.items():
        _need(type(sources[role]) is PinnedBytes, RejectionCode.INPUT, "typed source bytes")
        sources[role]._checked(name, 128 << 10)
    for role, name in EVIDENCE_NAMES.items():
        _need(type(evidence[role]) is PinnedBytes, RejectionCode.INPUT, "typed evidence bytes")
        evidence[role]._checked(name, (64 if role == "transcript" else 128) << 10)
    _check_sources(sources)
    documents = {role: _strict_json(evidence[role].raw) for role in _SCHEMAS}
    for role, value in documents.items():
        _need(type(value) is dict and _REQUIRED_NULLABLE[role] <= set(value) <= _FIELDS[role]
              and value.get("schema") == _SCHEMAS[role]
              and value.get("namespace") == value.get("unit") == NAMESPACE,
              RejectionCode.SCHEMA, "supported original producer schema and field vocabulary")
    win, native, linux, cap = (documents[r] for r in ("windows_parent", "windows_supervision", "linux_parent", "linux_capture"))
    original = win.get("original_Windows_owner")
    _need(type(original) is dict and set(original) == {"root", "owner"} and type(original.get("root")) is str
          and bool(original["root"]), RejectionCode.JOIN, "original Windows manager root and owner")
    win_owner = _owner(original["owner"], "win32")
    native_parent = native.get("original_windows_parent")
    _need(type(native_parent) is dict and set(native_parent) == {"root", "owner", "owner_raw_sha256",
          "Linux_writer_closure_issued", "exact_Job_owned_by_child_observation"}
          and native_parent["root"] == original["root"] and _same(native_parent["owner"], win_owner)
          and type(native_parent.get("owner_raw_sha256")) is str and _HEX.fullmatch(native_parent["owner_raw_sha256"]) is not None
          and native_parent["Linux_writer_closure_issued"] is False
          and native_parent["exact_Job_owned_by_child_observation"] is False,
          RejectionCode.JOIN, "native child reports same original parent, without Job ownership promotion")
    output = native.get("output_original_owner")
    _need(type(output) is dict and output.get("schema") == "rovezero.managed-output.v1"
          and output.get("state") == "active" and type(output.get("token")) is str
          and re.fullmatch("[0-9a-f]{32}", output["token"]) is not None
          and type(output.get("pid")) is int and output["pid"] > 0
          and type(output.get("key")) is str and _HEX.fullmatch(output["key"]) is not None
          and output.get("output") == native.get("output") and output.get("source") == win_owner["source"]
          and _same(output.get("outer_supervisor"), dict(original, scope="inherited_outer_process_group_or_job"))
          and type(output.get("lifetime")) is dict and output["lifetime"].get("platform") == "win32"
          and output["lifetime"].get("host") == win_owner["lifetime"]["host"]
          and output["lifetime"].get("boot") == win_owner["lifetime"]["boot"],
          RejectionCode.JOIN, "original output lease is bound to this original Windows manager")
    linux_owner = _owner(linux.get("actual_original_owner"), "linux")
    _need(type(linux.get("actual_original_owner_raw_sha256")) is str
          and _HEX.fullmatch(linux["actual_original_owner_raw_sha256"]) is not None,
          RejectionCode.JOIN, "original Linux owner raw hash reference")
    for field in ("admission_rawpin", "registration"):
        for value in (native, linux, cap):
            _rawpin(value.get(field))
        _need(_same(native[field], linux[field]) and _same(linux[field], cap[field]),
              RejectionCode.JOIN, "same original admission and runtime registration pins")
    _rawpin(native.get("source_registry"))
    _need(_same(native["source_registry"], linux.get("native_registry_rawpin"))
          and _same(native["source_registry"], cap.get("native_registry_rawpin")),
          RejectionCode.JOIN, "same original source registry pin reference")
    _rawpin(native.get("capture_registration"))
    _need(_same(native["capture_registration"], cap.get("config")), RejectionCode.JOIN, "same capture configuration pin")
    _pin_matches(cap.get("controller_source"), sources["linux_capture"])
    _pin_matches(cap.get("outer_source"), sources["linux_parent"])
    _pin_matches(cap.get("original_transcript"), evidence["transcript"])
    _commands(native, cap, sources, linux_owner)
    _need(win.get("Windows_Job_tree_gone") is True and win.get("original_Job_supervisor_error") is None
          and win.get("original_parent_dispatch_attempted") is True
          and _same(win.get("original_parent_cleanup_state"), {
              "acquire_returned": True, "constructor_attempted": True,
              "pre_Linux_constructor_returned": True, "pre_Linux_close_returned": True,
              "pre_Linux_close_failed_monotonic": False, "source_close_failed_monotonic": False})
          and linux.get("original_child_tree_gone") is True and linux.get("original_child_supervisor_error") is None,
          RejectionCode.JOIN, "original actor Job/PG terminal reports, not a detached treegone flag")
    for value, fields in ((win, ("exit_code", "driver_exit_code")),
                          (native, ("exit_code", "capture_client_exit_code")),
                          (linux, ("exit_code", "original_child_exit")),
                          (cap, ("exit_code", "original_systemd_exit_code"))):
        _need(all(type(value.get(k)) is int and value[k] == 2 for k in fields),
              RejectionCode.JOIN, "original run003b nonzero outcome preserved")
    _need(win.get("status") == "FAILED_OR_UNKNOWN_PRESERVED_NO_OWNER_REISSUE"
          and native.get("status") == linux.get("status") == "FAILED_OR_UNKNOWN_PRESERVED"
          and win.get("physical_completion") == native.get("physical_completion") == "unknown_unchanged"
          and linux.get("physical_reclamation") == "unknown"
          and win.get("original_manager_finish_called") is False and linux.get("manager_finish_called") is False
          and win.get("prior_owner_adopted") is False and native.get("prior_owner_adopted") is False
          and win.get("unfinished_original_owner_preserved") is True
          and linux.get("original_owner_and_scratch_preserved") is True
          and native.get("native_intake_complete") is False and native.get("functional_exact3_success") is False
          and native.get("Windows_Job_closure_issued") is False
          and win.get("Linux_writer_closure_verified") is False and native.get("Linux_writer_closure_verified") is False
          and linux.get("ready_pin") is None and linux.get("native_Windows_ack") is None
          and linux.get("original_managed_terminal") is None and linux.get("journal_copy_complete") is False,
          RejectionCode.JOIN, "original failed intake, missing ACK and physical unknown remain unchanged")
    for value, names in ((native, ("source_descriptors_closed",)),
                         (linux, ("source_descriptors_closed",)),
                         (cap, ("capture_descriptor_closed", "source_descriptors_closed", "cgroup_descriptors_closed",
                                "original_process_descriptors_closed", "original_stream_eof", "original_systemd_wait_completed"))):
        _need(all(value.get(k) is True for k in names), RejectionCode.JOIN, "original source/capture descriptor close observations")
    _need(cap.get("transcript_overflow") is False and type(cap.get("unrecorded_stream_bytes")) is int
          and cap["unrecorded_stream_bytes"] == 0 and type(cap.get("failure_rows_omitted")) is int
          and cap["failure_rows_omitted"] == 0 and cap.get("manager_finish_allowed") is False
          and cap.get("retirement_accepted") is False and cap.get("performance_evidence") is False,
          RejectionCode.JOIN, "complete original bounded stream without retrospective acceptance")
    for field in ("original_work_until", "original_cleanup_until"):
        _need(_finite(linux.get(field)) and _same(linux[field], cap.get(field)),
              RejectionCode.JOIN, "one original Linux work/cleanup clock")
    _need(0 < linux["original_work_until"] < linux["original_cleanup_until"]
          and _finite(linux.get("finished_monotonic")) and linux["finished_monotonic"] < linux["original_cleanup_until"],
          RejectionCode.JOIN, "original finite work and cleanup boundaries")
    transcript = evidence["transcript"].raw.decode("utf-8", "strict")
    _need("\r" not in transcript and transcript.endswith("\n"), RejectionCode.TRANSCRIPT, "complete LF original transcript")
    rows = [line[len("build151_outer="):] for line in transcript.splitlines() if line.startswith("build151_outer=")]
    _need(len(rows) == 1 and _same(_strict_json(rows[0].encode("utf-8")), linux)
          and not any(line.startswith("managed_storage=") for line in transcript.splitlines()),
          RejectionCode.TRANSCRIPT, "one exact original Linux parent raw, no invented finish footer")
    branch, product = _kernel_replay(cap, linux_owner)
    failures = {}
    for role, document in documents.items():
        rows = document.get("failures")
        _need(type(rows) is list and 0 < len(rows) <= 16 and all(type(row) is dict for row in rows),
              RejectionCode.SCHEMA, "preserved bounded original failure rows")
        failures[role] = len(rows)  # Details/notes may contain sensitive input; do not publish them.
    summary = {
        "product_source": PRODUCT_SOURCE, "build_source": BUILD_SOURCE, "control_source": CONTROL_SOURCE,
        "original_windows_owner": {"pid": win_owner["pid"], "lifetime": win_owner["lifetime"], "token": "[REDACTED]"},
        "original_linux_owner": {"pid": linux_owner["pid"], "lifetime": linux_owner["lifetime"], "token": "[REDACTED]"},
        "original_output": {"key": output["key"], "pid": output["pid"], "token": "[REDACTED]"},
        "original_windows_job_tree_gone_reported": True,
        "original_linux_pg_tree_gone_reported": True,
        "original_linux_pidfd_death_replayed": True,
        "original_linux_cgroup_terminal_replayed": True,
        "original_exit_code": 2, "functional_success": False,
        "native_intake_complete": False, "original_ack_present": False,
        "original_windows_manager_finish_called": False, "original_linux_manager_finish_called": False,
        "original_physical_completion": "unknown_unchanged", "original_linux_physical_reclamation": "unknown",
        "failure_counts": failures,
    }
    return OriginalTerminationObservation(NAMESPACE, branch, product,
        tuple(sources[role] for role in SOURCE_NAMES), tuple(evidence[role] for role in EVIDENCE_NAMES),
        _encoded(original), _encoded(linux_owner), _encoded(output), _encoded(summary))


def verify_original_termination(*, sources: dict[str, PinnedBytes], evidence: dict[str, PinnedBytes]):
    """Validate supplied originals without touching storage, OS state or authority.

    No filesystem writer or live recovery API exists here. A result cannot prove
    current owner death, a current reboot, cross-boot same-host identity, an ACK,
    or physical product/GPU completion. Missing/contradictory original crosslinks
    raise a typed rejection; they are never replaced with a user approval flag.
    """
    try:
        return _verify(sources, evidence)
    except OriginalTerminationError:
        raise
    except (UnicodeError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as error:
        raise OriginalTerminationError(RejectionCode.SCHEMA, "malformed original evidence structure") from error
