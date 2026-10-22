"""Bounded research/archive and regenerable-code audits; no retirement authority.

The original output/catalog and native-custody handlers remain responsible for
deletion. These readers never recover leases, interpret expiry as permission,
signal processes, adopt quarantine records, or modify catalog rows.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import zipfile

from ci_native_bundle import ELF_CAP, ELF_PAYLOAD_CAP, FUNCTIONAL_SPECS, pinned_file, same_file_identity
from managed_storage import StorageError, checked, read_record_snapshot
from managed_outputs import _catalog_rows, _refuse_custody_registry, identity
from storage_guard import CatalogGuard, lifetime, owner_proven_dead

PLAN_SCHEMA = "rovezero.code-cache-audit.v1"
RESEARCH_CAP = 32 << 20
INVENTORY_CAP = 4 << 20
ARCHIVE_CAP = 512 << 20
MAX_FILES = 512
CODE_NAMES = {spec[2] for spec in FUNCTIONAL_SPECS}
HEX = re.compile(r"[0-9a-f]{64}\Z")


def need(value, message):
    if value is not True:
        raise StorageError(message)


def fields(value, names):
    need(type(value) is dict and set(value) == set(names), "closed audit fields required")


def no_secret(path):
    for part in Path(path).parts:
        name = part.lower()
        need(not (name.startswith(".env") or name.endswith((".key", ".pem"))
                  or any(tag in name for tag in ("credential", "service-account", "service_account",
                                                 "id_rsa", "id_ed25519", "api_key", "api-key"))),
             "secret paths are never audit inputs")


def member_name(value):
    need(type(value) is str and 0 < len(value) <= 512 and "\\" not in value and ":" not in value,
         "bounded relative archive member required")
    p = PurePosixPath(value)
    need(not p.is_absolute() and p.as_posix() == value and all(v not in (".", "..") for v in p.parts),
         "archive member must not escape or normalize")
    no_secret(value)
    return value


def raw_pin(value, maximum):
    fields(value, ("path", "bytes", "sha256"))
    need(type(value["path"]) is str and Path(value["path"]).is_absolute(), "absolute explicit input required")
    need(type(value["bytes"]) is int and 0 < value["bytes"] <= maximum
         and type(value["sha256"]) is str and HEX.fullmatch(value["sha256"]) is not None,
         "bounded raw byte/hash pin required")
    no_secret(value["path"])
    return value


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise StorageError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


class Inputs:
    def __init__(self, seconds):
        need(type(seconds) is int and 1 <= seconds <= 600, "finite <=600s audit budget required")
        self.until = time.monotonic() + seconds
        self.held = []
        self.stack = contextlib.ExitStack()

    def tick(self):
        need(time.monotonic() < self.until, "audit time budget exceeded; originals preserved")

    def open(self, value, maximum):
        self.tick()
        raw_pin(value, maximum)
        path = checked(Path(value["path"]))
        try:
            stream, actual, before, _ = pinned_file(path, maximum)
        except RuntimeError as error:
            raise StorageError("audit FD/type/identity failure: " + str(error)) from error
        self.stack.callback(stream.close)
        need(actual == {k: value[k] for k in ("bytes", "sha256")}, "original audit input bytes changed")
        self.held.append((path, stream, before))
        self.tick()
        return stream

    def check(self):
        self.tick()
        for path, stream, before in self.held:
            checked(path)
            need(same_file_identity(before, os.fstat(stream.fileno()), path.lstat()),
                 "held audit input identity changed")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        # A close failure must not erase the original audit failure.
        try:
            self.stack.close()
        except OSError:
            if value is None:
                raise
            if hasattr(value, "add_note"):
                value.add_note("audit descriptor close also failed")
        return False


def zip_entries(z, maximum_total):
    infos = z.infolist()
    need(0 < len(infos) <= MAX_FILES + 1 and len({v.filename for v in infos}) == len(infos),
         "bounded unique ZIP inventory required")
    total = 0
    for item in infos:
        member_name(item.filename)
        # ZIP has no dedicated symlink predicate. Unix producers declare the
        # file type in the upper external-attribute bits; DOS producers often
        # leave those bits zero. Permissions without a type also occur in
        # Python ZIPs. Preserve those ordinary files, but never accept a link,
        # special inode, DOS directory/volume label, or an unknown file type.
        # Check declared Unix types even when create_system names another OS:
        # a mismatched host marker must not turn a symlink into a plain file.
        file_type = stat.S_IFMT(item.external_attr >> 16)
        need(not item.is_dir() and file_type in (0, stat.S_IFREG)
             and not item.external_attr & (0x10 | 0x08)
             and item.file_size >= 0 and not item.flag_bits & 1,
             "plain unencrypted ZIP files required")
        total += item.file_size
        need(total <= maximum_total, "ZIP uncompressed budget exceeded")
    return {item.filename: item for item in infos}


def research(inputs, pin):
    stream = inputs.open(pin, RESEARCH_CAP)
    with zipfile.ZipFile(stream) as z:
        entries = zip_entries(z, RESEARCH_CAP + INVENTORY_CAP)
        need("inventory.json" in entries and entries["inventory.json"].file_size <= INVENTORY_CAP,
             "bounded research inventory.json required")
        rows = decode(z.read("inventory.json"))
        need(type(rows) is list and 0 < len(rows) <= MAX_FILES, "bounded research members required")
        selected = set()
        for row in rows:
            inputs.tick()
            need(type(row) is dict and {"member", "bytes", "sha256"} <= set(row), "research member pin required")
            name = member_name(row["member"])
            need(name != "inventory.json" and name not in selected, "research member aliases are forbidden")
            need(type(row["bytes"]) is int and 0 <= row["bytes"] <= RESEARCH_CAP
                 and type(row["sha256"]) is str and HEX.fullmatch(row["sha256"]) is not None,
                 "bounded research member bytes/hash required")
            need(name in entries and entries[name].file_size == row["bytes"], "research member size differs")
            raw = z.read(name)  # CRC checked by ZipExtFile; already bounded by inventory.
            need((len(raw), hashlib.sha256(raw).hexdigest()) == (row["bytes"], row["sha256"]),
                 "research member hash differs")
            selected.add(name)
        need(set(entries) == selected | {"inventory.json"}, "unlisted ZIP evidence is not silently ignored")
    inputs.check()
    return {"archive": pin, "files": len(rows), "crc_and_every_member_verified": True}


def verify_research(pin, *, seconds=180):
    with Inputs(seconds) as inputs:
        result = research(inputs, pin)
        inputs.check()
        return {"schema": "rovezero.research-audit.v1", "status": "RESEARCH_ARCHIVE_VERIFIED",
                **result, "deletion_performed": False, "physical_completion_attested": False}


def canonical_code(inputs, pin, archive_member, expected):
    stream = inputs.open(pin, ARCHIVE_CAP)
    found = set()
    with zipfile.ZipFile(stream) as z:
        entries = zip_entries(z, 1 << 30)
        need(archive_member in entries, "registered nested code archive missing")
        with z.open(archive_member) as nested, gzip.GzipFile(fileobj=nested) as tar:
            count = total = 0
            def read(size):
                nonlocal total
                inputs.tick()
                need(0 <= size <= 1 << 20 and total + size <= 2 << 30, "canonical TAR stream budget exceeded")
                raw = tar.read(size)
                need(len(raw) == size, "truncated canonical TAR stream")
                total += size
                return raw
            while True:
                header = read(512)
                if header == bytes(512):
                    need(read(512) == bytes(512), "two canonical TAR end blocks required")
                    while True:
                        inputs.tick()
                        raw = tar.read(65536)
                        if not raw:
                            break
                        total += len(raw)
                        need(total <= 2 << 30 and not any(raw), "nonzero or oversized canonical TAR tail")
                    break
                item = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
                member_name(item.name)
                count += 1
                need(count <= 256 and item.size >= 0 and item.isfile() and not item.issparse(),
                     "canonical producer supports only bounded plain files, no links/PAX/sparse metadata")
                selected = item.name in expected
                if selected:
                    need(item.name not in found, "canonical ELF alias")
                    need(item.size == expected[item.name]["bytes"], "canonical ELF member size differs")
                h = hashlib.sha256()
                remaining = item.size
                while remaining:
                    block = read(min(1 << 20, remaining))
                    if selected:
                        h.update(block)
                    remaining -= len(block)
                need(not any(read((-item.size) % 512)), "nonzero canonical TAR padding")
                if selected:
                    need(h.hexdigest() == expected[item.name]["sha256"], "canonical ELF member hash differs")
                    found.add(item.name)
    need(found == set(expected), "all canonical ELF members required")
    inputs.check()


def read_code_plan(inputs, plan_pin):
    """Validate the existing closed plan; this issues no retirement authority."""
    plan = decode(inputs.open(plan_pin, INVENTORY_CAP).read(INVENTORY_CAP + 1))
    fields(plan, ("schema", "catalog", "key", "record_sha256", "artifact", "archive_member", "code", "research"))
    need(plan["schema"] == PLAN_SCHEMA and type(plan["catalog"]) is str
         and Path(plan["catalog"]).is_absolute() and type(plan["key"]) is str
         and HEX.fullmatch(plan["key"]) is not None and type(plan["record_sha256"]) is str
         and HEX.fullmatch(plan["record_sha256"]) is not None, "exact registered catalog/output required")
    no_secret(plan["catalog"])
    catalog = checked(Path(plan["catalog"]))
    need(catalog.is_dir() and (catalog / "catalog.lock").is_file(), "existing original catalog gate required")
    code = plan["code"]
    need(type(code) is list and 1 <= len(code) <= 32, "bounded explicit code list required")
    expected = {}
    paths = set()
    for row in code:
        fields(row, ("relative", "tar_member", "bytes", "sha256"))
        relative = member_name(row["relative"])
        name = PurePosixPath(relative)
        need(len(name.parts) == 2 and name.parts[0] in ("selected4", "private-elf")
             and name.name in CODE_NAMES and relative not in paths, "only explicit product code-cache leaves allowed")
        raw_pin({"path": str(Path(os.path.abspath(relative))), "bytes": row["bytes"], "sha256": row["sha256"]}, ELF_CAP)
        canonical = member_name(row["tar_member"])
        need(canonical not in expected, "canonical ELF alias")
        expected[canonical] = row
        paths.add(relative)
    need(sum(row["bytes"] for row in code) <= ELF_PAYLOAD_CAP, "aggregate ELF budget exceeded")
    archive_member = member_name(plan["archive_member"])
    raw_pin(plan["research"], RESEARCH_CAP)
    raw_pin(plan["artifact"], ARCHIVE_CAP)
    return plan, catalog, code, expected, archive_member


def audit_code_cache(plan_pin, *, seconds=300):
    """Only audit. Even a matching report cannot authorize unlink or lease recovery."""
    with Inputs(seconds) as inputs:
        plan, catalog, code, expected, archive_member = read_code_plan(inputs, plan_pin)
        with CatalogGuard(catalog / "catalog.lock") as guard:
            _refuse_custody_registry(catalog)  # Existing special custody must use its original validator.
            rows = _catalog_rows(catalog)
            matches = [row for row in rows if row[1]["key"] == plan["key"]]
            need(len(matches) == 1, "exact original output row missing")
            record_path, row, output = matches[0]
            original = read_record_snapshot(record_path, schema="rovezero.managed-output.v1")
            need(original[1] == plan["record_sha256"] and original[0] == row, "original output row raw pin changed")
            need(row["state"] == "retained" and row.get("tree_gone") is True and row.get("budget_error") is None
                 and row.get("deferred") is False, "only original completed nondeferred output can be audited")
            need(owner_proven_dead(row.get("pid"), row.get("lifetime"), lifetime()),
                 "original owner death not proven; no adoption")
            verified_research = research(inputs, plan["research"])
            for target in code:
                path = checked(output / target["relative"])
                need(path.is_relative_to(output) and identity(output) == row["identity"], "original output identity changed")
                stream = inputs.open({"path": str(path), "bytes": target["bytes"], "sha256": target["sha256"]}, ELF_CAP)
                need(stream.read(4) == b"\x7fELF", "candidate must contain actual ELF bytes")
            canonical_code(inputs, plan["artifact"], archive_member, expected)
            guard.assert_held()
            _refuse_custody_registry(catalog)
            need(read_record_snapshot(record_path, schema="rovezero.managed-output.v1") == original
                 and identity(output) == row["identity"] and _catalog_rows(catalog) == rows,
                 "original catalog/row/output changed during audit")
            inputs.check()
        return {"schema": "rovezero.code-cache-audit-result.v1", "status": "CODE_CACHE_AND_RESEARCH_AUDIT_MATCHED",
                "plan": plan_pin, "code_files": len(code), "code_bytes": sum(v["bytes"] for v in code),
                "research": verified_research, "original_row_unchanged": True, "deletion_performed": False,
                "retirement_authority_issued": False, "physical_completion_attested": False,
                "still_required_for_retirement": ["original_custody_revalidation", "current_exe_fd_maps_census",
                                                 "held_name_and_inode_check_at_unlink", "durable_retirement_journal"]}

