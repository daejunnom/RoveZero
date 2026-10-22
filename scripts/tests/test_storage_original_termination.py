"""In-memory contract cases. Synthetic pins never represent a live owner.

No fixture acquires a lease, starts a process or writes a closure. Main separately
provides the preserved actual run003b source/evidence bytes for real-data replay.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage_original_termination as original


def dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pinned(name, raw):
    return original.PinnedBytes(name, len(raw), hashlib.sha256(raw).hexdigest(), raw)


def rawpin(name, raw):
    return {"path": "/mnt/c/fixture-reports/" + name, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def reference(name, marker):
    return rawpin(name, marker.encode())


def source_fixture():
    # Small, inert declarations exercise source graph parsing without executing
    # a copied operational actor. They are explicitly synthetic caller pins.
    names, namespace = original.SOURCE_NAMES, original.NAMESPACE
    roles = {"windows_outer": names["windows_parent"], "windows_driver": names["windows_driver"],
             "capture": names["linux_capture"], "linux_outer": names["linux_parent"],
             "bridge": "archive-build151-originals-N05-run003-native-bridge-source.py",
             "consumer": "archive-build151-originals-N05-run003-consumer-source.py",
             "context": "storage-current-source7-context-001.py",
             "source7_config": "currentb567-plain-source7-configuration-main-001.json",
             "protocol": "old738-original-pre-linux-admission-protocol-003.py"}
    protocol = (roles["protocol"], 18383,
                "49176633611c2e242f47ee8f741e9da7a81259d6023103e3cbacfeccf58a7912")
    text = {
        "windows_parent": f"NS={namespace!r}\nDRIVER={names['windows_driver']!r}\n",
        "windows_driver": (f"NS={namespace!r}\nSOURCE={original.PRODUCT_SOURCE!r}\n"
            f"BUILD_SOURCE={original.BUILD_SOURCE!r}\nCONTROL_COMMIT={original.CONTROL_SOURCE!r}\n"
            f"PROTOCOL={protocol!r}\nROLE_NAMES={roles!r}\n"),
        "linux_capture": (f"NAMESPACE={namespace!r}\nUNIT=NAMESPACE\n"
            f"CAPTURE_SOURCE_NAME={names['linux_capture']!r}\nLINUX_OUTER_SOURCE_NAME={names['linux_parent']!r}\n"
            f"BUILD_SOURCE={original.BUILD_SOURCE!r}\nSTORAGE_SOURCE={original.PRODUCT_SOURCE!r}\nSOURCE_ROLE_NAMES={roles!r}\n"),
        "linux_parent": (f"NS={namespace!r}\nSOURCE_FILENAME={names['linux_parent']!r}\n"
            f"CURRENT={original.PRODUCT_SOURCE!r}\nBUILD_SOURCE={original.BUILD_SOURCE!r}\n"),
    }
    return {role: pinned(names[role], value.encode()) for role, value in text.items()}


def show(properties):
    requested = "Id,Description,LoadState,ActiveState,SubState,InvocationID,ControlGroup,MainPID,ControlPID,ExecMainPID,ExecMainCode,ExecMainStatus,Result,Job"
    return {"command": ["/usr/bin/systemctl", "show", "--no-pager", "--property=" + requested,
                        original.NAMESPACE + ".service"],
            "exit_code": 0, "stderr": "", "stdout": "".join(k + "=" + v + "\n" for k, v in properties.items()),
            "properties": deepcopy(properties)}


def owner(platform):
    result = {"schema": "rovezero.managed-build.v1", "pid": 393 if platform == "linux" else 20984,
              "purpose": "storage-validation" if platform == "linux" else "build151-retirement-native-control-run003",
              "root_identity": [20, 40], "source": "/mnt/c/fixture-source" if platform == "linux" else "C:/fixture-source",
              "source_id": "a" * 24, "token": ("b" if platform == "linux" else "c") * 32,
              "lifetime": {"host": ("d" if platform == "linux" else "e") * 64,
                           "boot": "342be99e-cfc5-419a-a631-0e307b1feff9" if platform == "linux" else "windows-boot-counter-v1:59",
                           "platform": platform, "process_birth": "2070" if platform == "linux" else "134371328394385188"}}
    if platform == "linux":
        result.update(state="active", finished_ns=0)
    return result


def kernel_fixture(admission, registration, linux_owner):
    namespace = original.NAMESPACE
    group = "/system.slice/" + namespace + ".service"
    invocation = "9a05239b92614bf8a2a15d9333bc08ca"
    description = namespace + " admission=" + admission["sha256"] + " registration=" + registration["sha256"]
    parent = {"path": "/sys/fs/cgroup/system.slice", "readlink": "/sys/fs/cgroup/system.slice",
              "identity": [23, 60, 16877, 0, 0, 16], "name_identity": [23, 60, 16877, 0, 0, 16],
              "identity_at_bind": [23, 60, 16877, 0, 0, 18], "nlink_used_as_identity": False}
    mount = {"boot_id": linux_owner["lifetime"]["boot"],
             "mount": {"filesystem": "cgroup2", "mount_point": "/sys/fs/cgroup", "root": "/", "mount_id": 120},
             "fd_mount_ids": {key: 120 for key in ("parent", "directory", "cgroup.events", "cgroup.procs")}}
    files = {"cgroup.events": [23, 2694, 33060, 0, 0, 1], "cgroup.procs": [23, 2690, 33188, 0, 0, 1]}
    before_group = {"ControlGroup": group, "InvocationID": invocation,
                    "directory_identity": [23, 2688, 16877, 0, 0, 2], "file_identities": files,
                    "parent": deepcopy(parent), "mount_context": deepcopy(mount),
                    "directory_readlink": "/sys/fs/cgroup" + group, "events": {"populated": 1}, "procs": ["393"],
                    "old_name": {"state": "same_original", "identity": [23, 2688, 16877, 0, 0, 2]},
                    "readback_monotonic": 20.8}
    process = {"pid": 393, "starttime_ticks": 2070, "proc_cgroup": "0::" + group,
               "pidfd_open_flags": 0, "pidfd_dead_at_bind": False, "pidfd_identity": [15, 42, 384, 0, 0, 1],
               "proc_directory_identity": [59, 7561, 16749, 0, 0, 9], "bound_monotonic": 20.7,
               "pidfd_poll_at_bind": {"pidfd_dead": False, "poll_mask": 0, "poll_events": []},
               "pidfd_fdinfo_at_bind": {"Pid": "393", "NSpid": "393"},
               "proc_observation_before_pidfd_open": {"starttime_ticks": 2070, "proc_cgroup": "0::" + group},
               "proc_observation_after_pidfd_open": {"starttime_ticks": 2070, "proc_cgroup": "0::" + group}}
    properties = {"Id": namespace + ".service", "InvocationID": invocation, "ControlGroup": group,
                  "Description": description, "LoadState": "loaded", "MainPID": "393", "ControlPID": "0",
                  "ActiveState": "active", "SubState": "running", "Job": "", "Result": "success",
                  "ExecMainPID": "393", "ExecMainCode": "0", "ExecMainStatus": "0"}
    before = {"InvocationID": invocation, "Description": description, "ControlGroup": group,
              "show": show(properties), "confirmation_show": show(properties),
              "main_process": deepcopy(process), "cgroup": deepcopy(before_group)}
    final_group = deepcopy(before_group)
    final_group.pop("events")
    final_group.pop("procs")
    final_group.update(directory_readlink="/sys/fs/cgroup" + group + " (deleted)",
                       old_name={"state": "ENOENT", "errno": 2}, link_count_used_as_removal_proof=False,
                       metadata={name: {"identity": value, "read_state": "ENODEV", "errno": 19} for name, value in files.items()},
                       parent_final=deepcopy(parent), readback_monotonic=86.2)
    dead = {"pid": 393, "starttime_ticks": 2070, "pidfd_identity": [15, 42, 384, 0, 0, 1],
            "pidfd_dead": True, "poll_mask": 1, "poll_events": ["POLLIN"],
            "birth_reconstructed_after_death": False, "readback_monotonic": 86.3}
    terminal_properties = dict(properties, MainPID="0", ControlPID="0", ControlGroup="", ActiveState="failed",
                               SubState="failed", Result="exit-code", ExecMainCode="1", ExecMainStatus="2")
    metadata = {key: terminal_properties[key] for key in ("Result", "ExecMainCode", "ExecMainStatus")}
    bound = {"InvocationID": invocation, "ControlGroup": group, "Description": description,
             "main_process": deepcopy(process), "cgroup": deepcopy(before_group)}
    discriminator = {"schema": "rovent.build151-original-cgroup-terminal-discriminator/3",
                     "branch": "removed_original_cgroup", "client_exit_used_as_closure_proof": False,
                     "unit_name_absence_used_as_closure_proof": False, "original_binding": bound,
                     "terminal_readback": {"main_process": dead, "cgroup": deepcopy(final_group)},
                     "product_result_status": "observed_failure", "product_result_metadata": metadata}
    after = {"InvocationID": invocation, "Description": description, "ControlGroup": group,
             "show": show(terminal_properties), "cgroup": final_group, "terminal_discriminator": discriminator,
             "original_systemd_wait_completed": True, "original_systemd_exit_code": 2,
             "product_result_status": "observed_failure"}
    return before, after, metadata


def fixture():
    sources, ns = source_fixture(), original.NAMESPACE
    admission = reference("original-admission.json", "admission")
    registration = reference("build151-N05-run003-linux-registration.json", "registration")
    registry = reference("original-registry.json", "registry")
    config = reference("build151-N05-run003-capture-registration.json", "config")
    failure = [{"stage": "original", "type": "RuntimeError", "detail": "c" * 32, "cleanup_notes": []}]
    common = {"namespace": ns, "unit": ns, "exit_code": 2, "failures": failure}
    win_owner, linux_owner = owner("win32"), owner("linux")
    outer = {"root": "C:/fixture/build/managed-v1", "owner": win_owner}
    win = dict(common, schema="rovent.build151-retirement-original-Windows-parent/2",
        status="FAILED_OR_UNKNOWN_PRESERVED_NO_OWNER_REISSUE", original_Windows_owner=outer,
        Windows_Job_tree_gone=True, original_Job_supervisor_error=None, original_parent_dispatch_attempted=True,
        original_parent_cleanup_state={"acquire_returned": True, "constructor_attempted": True,
            "pre_Linux_constructor_returned": True, "pre_Linux_close_returned": True,
            "pre_Linux_close_failed_monotonic": False, "source_close_failed_monotonic": False},
        driver_exit_code=2, physical_completion="unknown_unchanged", original_manager_finish_called=False,
        prior_owner_adopted=False, unfinished_original_owner_preserved=True, Linux_writer_closure_verified=False)
    output = {"schema": "rovezero.managed-output.v1", "state": "active", "pid": 15504,
              "key": "f" * 64, "token": "a" * 32, "source": win_owner["source"],
              "output": "C:/fixture/reports/build151-N05-run003-Windows-evidence",
              "outer_supervisor": dict(outer, scope="inherited_outer_process_group_or_job"),
              "lifetime": dict(win_owner["lifetime"], process_birth="134371328495052382")}
    native = dict(common, schema="rovent.build151-retirement-Windows-supervision/2", status="FAILED_OR_UNKNOWN_PRESERVED",
        original_windows_parent=dict(outer, owner_raw_sha256="1" * 64, Linux_writer_closure_issued=False,
                                     exact_Job_owned_by_child_observation=False),
        output_original_owner=output, output=output["output"], source_registry=registry,
        admission_rawpin=admission, registration=registration, capture_registration=config,
        capture_client_exit_code=2, native_intake_complete=False, functional_exact3_success=False,
        Windows_Job_closure_issued=False, Linux_writer_closure_verified=False,
        prior_owner_adopted=False, physical_completion="unknown_unchanged", source_descriptors_closed=True)
    linux = dict(common, schema="rovent.build151-retirement-linux-outer/2", status="FAILED_OR_UNKNOWN_PRESERVED",
        actual_original_owner=linux_owner, actual_original_owner_raw_sha256="2" * 64,
        admission_rawpin=admission, registration=registration, native_registry_rawpin=registry,
        original_child_exit=2, original_child_tree_gone=True, original_child_supervisor_error=None,
        original_owner_and_scratch_preserved=True, source_descriptors_closed=True, manager_finish_called=False,
        original_managed_terminal=None, journal_copy_complete=False, ready_pin=None, native_Windows_ack=None,
        original_work_until=605.7, original_cleanup_until=635.7, finished_monotonic=86.1, physical_reclamation="unknown")
    before, after, metadata = kernel_fixture(admission, registration, linux_owner)
    cap = dict(common, schema="rovent.build151-retirement-linux-capture/2", admission_rawpin=admission,
        registration=registration, native_registry_rawpin=registry, config=config,
        controller_source=rawpin(sources["linux_capture"].name, sources["linux_capture"].raw),
        outer_source=rawpin(sources["linux_parent"].name, sources["linux_parent"].raw),
        original_work_until=605.7, original_cleanup_until=635.7, original_unit=before, unit_terminal=after,
        original_systemd_exit_code=2, original_systemd_wait_completed=True,
        original_stream_eof=True, capture_descriptor_closed=True, source_descriptors_closed=True,
        cgroup_descriptors_closed=True, original_process_descriptors_closed=True,
        transcript_overflow=False, unrecorded_stream_bytes=0, failure_rows_omitted=0,
        manager_finish_allowed=False, retirement_accepted=False, performance_evidence=False,
        unit_product_result_status="observed_failure", unit_product_result_metadata=metadata)
    command = ["/usr/bin/systemd-run", "--wait", "--pipe", "--unit=" + ns,
        "--property=Description=" + before["Description"], "--property=CPUQuota=200%", "--property=AllowedCPUs=0,2",
        "--property=MemoryHigh=6G", "--property=MemoryMax=12G", "--property=RuntimeMaxSec=660",
        "--property=TimeoutStopSec=20", "--property=TasksMax=256", "--property=LimitNOFILE=131072",
        "--property=PrivateMounts=yes", "--property=UMask=0077", "--working-directory=" + linux_owner["source"],
        "/usr/bin/unshare", "--mount", "--propagation", "private", "--", "/usr/bin/python3.12", "-I", "-S", "-B",
        cap["outer_source"]["path"], "--registration", registration["path"], str(registration["bytes"]), registration["sha256"],
        "--self-bytes", str(sources["linux_parent"].size), "--self-sha256", sources["linux_parent"].sha256]
    native["original_systemd_command"] = command
    cap["systemd_command"] = command
    native["capture_command"] = ["wsl.exe", "-d", "Ubuntu", "-u", "root", "--", "/usr/bin/python3.12", "-I", "-S", "-B",
        cap["controller_source"]["path"], "--config", config["path"], "--config-bytes", str(config["bytes"]),
        "--config-sha256", config["sha256"], "--self-bytes", str(sources["linux_capture"].size), "--self-sha256", sources["linux_capture"].sha256]
    return sources, {"windows_parent": win, "windows_supervision": native, "linux_parent": linux, "linux_capture": cap}


def pack(documents):
    documents = deepcopy(documents)
    transcript = b"build151_outer=" + dump(documents["linux_parent"]) + b"\n"
    documents["linux_capture"]["original_transcript"] = rawpin(original.EVIDENCE_NAMES["transcript"], transcript)
    evidence = {role: pinned(original.EVIDENCE_NAMES[role], dump(value)) for role, value in documents.items()}
    evidence["transcript"] = pinned(original.EVIDENCE_NAMES["transcript"], transcript)
    return evidence


class OriginalTerminationTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.documents = fixture()

    def verify(self, documents=None, sources=None):
        return original.verify_original_termination(sources=self.sources if sources is None else sources,
            evidence=pack(self.documents if documents is None else documents))

    def reject(self, documents=None, sources=None, code=None):
        with self.assertRaises(original.OriginalTerminationError) as caught:
            self.verify(documents=documents, sources=sources)
        if code is not None:
            self.assertEqual(caught.exception.code, code)

    def test_original_nonzero_missing_ack_can_replay_without_recovery_authority(self):
        result = self.verify()
        report = result.redacted_report()
        self.assertEqual(result.terminal_branch, "removed_original_cgroup")
        self.assertEqual(report["original_exit_code"], 2)
        self.assertFalse(report["original_ack_present"])
        self.assertFalse(report["functional_success"])
        self.assertEqual(report["original_physical_completion"], "unknown_unchanged")
        for key in ("recovery_authorized", "closure_issued", "deletion_authorized", "owner_adopted",
                    "current_owner_death_observed", "current_linux_reboot_observed"):
            self.assertIs(report[key], False)

    def test_report_and_repr_redact_all_three_owner_tokens_and_failure_detail(self):
        result = self.verify()
        text = json.dumps(result.redacted_report()) + repr(result)
        for token in ("a" * 32, "b" * 32, "c" * 32):
            self.assertNotIn(token, text)
        self.assertNotIn("detail", result.redacted_report())
        self.assertIn(b'"token":"' + b"c" * 32 + b'"', result._windows_owner_raw)

    def test_public_projection_mutation_cannot_change_original_result(self):
        result = self.verify()
        report = result.redacted_report()
        report["original_windows_owner"]["lifetime"]["boot"] = "changed"
        self.documents["windows_parent"]["original_Windows_owner"]["owner"]["token"] = "f" * 32
        self.assertEqual(result.redacted_report()["original_windows_owner"]["lifetime"]["boot"], "windows-boot-counter-v1:59")
        self.assertIn(b"c" * 32, result._windows_owner_raw)

    def test_wrong_hash_size_or_role_roster_rejected_before_replay(self):
        for change in ({"sha256": "0" * 64}, {"size": 1}, {"raw": bytearray(b"x")}):
            with self.subTest(change=list(change)):
                sources = dict(self.sources)
                sources["windows_driver"] = replace(sources["windows_driver"], **change)
                self.reject(sources=sources)
        sources = dict(self.sources)
        sources["extra"] = sources["windows_driver"]
        self.reject(sources=sources, code=original.RejectionCode.INPUT)

    def test_nonexact_string_names_and_roster_keys_never_call_user_comparison(self):
        calls = []

        class CallbackString(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                calls.append("eq")
                return str.__eq__(self, other)

            def __str__(self):
                calls.append("str")
                return str.__str__(self)

        sources = dict(self.sources)
        prior = sources["windows_parent"]
        sources["windows_parent"] = replace(prior, name=CallbackString(prior.name))
        self.reject(sources=sources, code=original.RejectionCode.INPUT)
        self.assertEqual(calls, [])
        for which in ("source", "evidence"):
            sources, evidence = dict(self.sources), pack(self.documents)
            roster = sources if which == "source" else evidence
            value = roster.pop("windows_parent")
            roster[CallbackString("windows_parent")] = value
            calls.clear()  # Dictionary construction is outside the validator.
            with self.assertRaises(original.OriginalTerminationError) as caught:
                original.verify_original_termination(sources=sources, evidence=evidence)
            self.assertEqual(caught.exception.code, original.RejectionCode.INPUT)
            self.assertEqual(calls, [])

    def test_source_role_rebinding_or_control_product_confusion_is_rejected(self):
        for old, new in ((original.CONTROL_SOURCE, original.PRODUCT_SOURCE),
                         (original.SOURCE_NAMES["linux_capture"], "other-capture.py")):
            sources = dict(self.sources)
            prior = sources["windows_driver"]
            sources["windows_driver"] = pinned(prior.name, prior.raw.replace(old.encode(), new.encode()))
            self.reject(sources=sources, code=original.RejectionCode.SOURCE)

    def test_nonliteral_source_declaration_is_not_executed(self):
        sources = dict(self.sources)
        prior = sources["windows_parent"]
        sources["windows_parent"] = pinned(prior.name, b"NS=__import__('must_never_execute')\nDRIVER='x'\n")
        self.reject(sources=sources, code=original.RejectionCode.SOURCE)

    def test_duplicate_nonfinite_and_overflow_numbers_are_typed_rejections(self):
        for raw in (b'{"schema":1,"schema":2}', b'{"value":NaN}', b'{"value":1e999}'):
            evidence = pack(self.documents)
            evidence["windows_parent"] = pinned(original.EVIDENCE_NAMES["windows_parent"], raw)
            with self.assertRaises(original.OriginalTerminationError) as caught:
                original.verify_original_termination(sources=self.sources, evidence=evidence)
            self.assertEqual(caught.exception.code, original.RejectionCode.JSON)

    def test_unknown_schema_field_and_boolean_exit_do_not_pass(self):
        for change in ({"schema": "future/3"}, {"unexpected": True}, {"exit_code": True}):
            documents = deepcopy(self.documents)
            documents["windows_parent"].update(change)
            self.reject(documents)

    def test_missing_nullable_observation_is_not_an_explicit_original_null(self):
        for role, key in (("windows_parent", "original_Job_supervisor_error"),
                          ("linux_parent", "original_child_supervisor_error"),
                          ("linux_parent", "ready_pin"),
                          ("linux_parent", "native_Windows_ack"),
                          ("linux_parent", "original_managed_terminal"),
                          ("linux_capture", "unit_product_result_metadata")):
            with self.subTest(role=role, key=key):
                documents = deepcopy(self.documents)
                del documents[role][key]
                self.reject(documents, code=original.RejectionCode.SCHEMA)
        documents = deepcopy(self.documents)
        del documents["linux_capture"]["unit_terminal"]["terminal_discriminator"]["product_result_metadata"]
        self.reject(documents, code=original.RejectionCode.KERNEL)

    def test_detached_treegone_flag_without_original_owner_crosslink_rejected(self):
        documents = deepcopy(self.documents)
        documents["windows_supervision"]["original_windows_parent"]["owner"] = deepcopy(owner("win32"))
        documents["windows_supervision"]["original_windows_parent"]["owner"]["token"] = "f" * 32
        self.reject(documents, code=original.RejectionCode.JOIN)

    def test_original_job_and_pg_error_or_sticky_close_failure_are_rejected(self):
        for role, key, value in (("windows_parent", "Windows_Job_tree_gone", False),
                                  ("windows_parent", "original_Job_supervisor_error", "unknown"),
                                  ("linux_parent", "original_child_tree_gone", False),
                                  ("linux_parent", "original_child_supervisor_error", "unknown")):
            documents = deepcopy(self.documents)
            documents[role][key] = value
            self.reject(documents, code=original.RejectionCode.JOIN)
        documents = deepcopy(self.documents)
        documents["windows_parent"]["original_parent_cleanup_state"]["source_close_failed_monotonic"] = True
        self.reject(documents, code=original.RejectionCode.JOIN)

    def test_admission_registry_or_controller_pin_substitution_rejected(self):
        for key in ("admission_rawpin", "native_registry_rawpin", "controller_source"):
            documents = deepcopy(self.documents)
            documents["linux_capture"][key] = dict(documents["linux_capture"][key], sha256="0" * 64)
            self.reject(documents, code=original.RejectionCode.JOIN)

    def test_command_and_resource_substitution_rejected(self):
        for index, replacement in ((0, "fake-systemd"), (5, "--property=CPUQuota=400%"), (33, "0" * 64)):
            documents = deepcopy(self.documents)
            documents["linux_capture"]["systemd_command"][index] = replacement
            documents["windows_supervision"]["original_systemd_command"] = deepcopy(documents["linux_capture"]["systemd_command"])
            self.reject(documents, code=original.RejectionCode.JOIN)

    def test_mainpid_birth_boot_and_platform_must_join_linux_original_owner(self):
        for key, value in (("pid", 394), ("process_birth", "2071"),
                           ("boot", "342be99e-cfc5-419a-a631-0e307b1feff8"), ("platform", "win32")):
            documents = deepcopy(self.documents)
            target = documents["linux_parent"]["actual_original_owner"]
            (target if key == "pid" else target["lifetime"])[key] = value
            self.reject(documents)

    def test_pidfd_reopen_reconstructed_birth_or_missing_live_binding_rejected(self):
        for key, value in (("pidfd_dead", False), ("birth_reconstructed_after_death", True), ("pid", 394),
                           ("pidfd_identity", [15, 43, 384, 0, 0, 1]), ("poll_mask", True)):
            documents = deepcopy(self.documents)
            dead = documents["linux_capture"]["unit_terminal"]["terminal_discriminator"]["terminal_readback"]["main_process"]
            dead[key] = value
            self.reject(documents, code=original.RejectionCode.KERNEL)

    def test_removed_cgroup_needs_both_original_metadata_fds_not_name_absence(self):
        documents = deepcopy(self.documents)
        after = documents["linux_capture"]["unit_terminal"]
        after["cgroup"]["metadata"]["cgroup.procs"]["read_state"] = "ENOENT"
        after["terminal_discriminator"]["terminal_readback"]["cgroup"] = deepcopy(after["cgroup"])
        self.reject(documents, code=original.RejectionCode.KERNEL)

    def test_raw_systemctl_must_replay_properties_and_same_invocation(self):
        documents = deepcopy(self.documents)
        documents["linux_capture"]["unit_terminal"]["show"]["stdout"] += "MainPID=0\n"
        self.reject(documents, code=original.RejectionCode.KERNEL)
        documents = deepcopy(self.documents)
        documents["linux_capture"]["unit_terminal"]["InvocationID"] = "0" * 32
        self.reject(documents, code=original.RejectionCode.KERNEL)

    def test_loaded_result_missing_fields_or_none_metadata_is_not_observed_failure(self):
        for key in ("Result", "ExecMainCode", "ExecMainStatus"):
            documents = deepcopy(self.documents)
            after = documents["linux_capture"]["unit_terminal"]
            properties = dict(after["show"]["properties"])
            del properties[key]
            after["show"] = show(properties)
            after["terminal_discriminator"]["product_result_metadata"][key] = None
            documents["linux_capture"]["unit_product_result_metadata"][key] = None
            self.reject(documents, code=original.RejectionCode.KERNEL)
        documents = deepcopy(self.documents)
        documents["linux_capture"]["unit_terminal"]["terminal_discriminator"]["product_result_metadata"] = None
        documents["linux_capture"]["unit_product_result_metadata"] = None
        self.reject(documents, code=original.RejectionCode.KERNEL)
        documents = deepcopy(self.documents)
        documents["linux_capture"]["unit_terminal"]["show"]["command"][3] = "--property=Id,MainPID"
        self.reject(documents, code=original.RejectionCode.KERNEL)

    def test_invented_ack_finish_or_physical_success_cannot_upgrade_original(self):
        for role, key, value in (("linux_parent", "native_Windows_ack", {}),
                                 ("linux_parent", "manager_finish_called", True),
                                 ("windows_parent", "physical_completion", "closed"),
                                 ("windows_supervision", "functional_exact3_success", True)):
            documents = deepcopy(self.documents)
            documents[role][key] = value
            self.reject(documents, code=original.RejectionCode.JOIN)

    def test_transcript_parent_substitution_or_invented_finish_footer_rejected(self):
        for raw in (b'build151_outer={}\n', b'build151_outer={}\nbuild151_outer={}\n',
                    b'build151_outer={}\nmanaged_storage={}\n'):
            evidence = pack(self.documents)
            evidence["transcript"] = pinned(original.EVIDENCE_NAMES["transcript"], raw)
            cap = json.loads(evidence["linux_capture"].raw)
            cap["original_transcript"] = rawpin(original.EVIDENCE_NAMES["transcript"], raw)
            evidence["linux_capture"] = pinned(original.EVIDENCE_NAMES["linux_capture"], dump(cap))
            with self.assertRaises(original.OriginalTerminationError) as caught:
                original.verify_original_termination(sources=self.sources, evidence=evidence)
            self.assertEqual(caught.exception.code, original.RejectionCode.TRANSCRIPT)

    def test_product_metadata_unknown_after_unit_gc_stays_unknown(self):
        documents = deepcopy(self.documents)
        cap = documents["linux_capture"]
        after = cap["unit_terminal"]
        properties = dict(after["show"]["properties"], LoadState="not-found", InvocationID="", ControlGroup="")
        after["show"] = show(properties)
        after["product_result_status"] = "unknown"
        after["terminal_discriminator"]["product_result_status"] = "unknown"
        after["terminal_discriminator"]["product_result_metadata"] = None
        cap["unit_product_result_status"], cap["unit_product_result_metadata"] = "unknown", None
        result = self.verify(documents)
        self.assertEqual(result.unit_product_result_status, "unknown")
        self.assertFalse(result.redacted_report()["functional_success"])

    def test_live_empty_branch_preserves_original_binding_and_no_recovery(self):
        documents = deepcopy(self.documents)
        cap = documents["linux_capture"]
        after = cap["unit_terminal"]
        live = deepcopy(cap["original_unit"]["cgroup"])
        live.update(events={"populated": 0}, procs=[], readback_monotonic=86.2)
        after["cgroup"] = live
        after["terminal_discriminator"]["branch"] = "live_cgroup_empty"
        after["terminal_discriminator"]["terminal_readback"]["cgroup"] = deepcopy(live)
        result = self.verify(documents)
        self.assertEqual(result.terminal_branch, "live_cgroup_empty")
        self.assertFalse(result.redacted_report()["recovery_authorized"])


if __name__ == "__main__":
    unittest.main()
