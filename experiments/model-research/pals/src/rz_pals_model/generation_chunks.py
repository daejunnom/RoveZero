"""Pinned JSONL generation corpus with bounded rows and a disposable disk index.

These are logical byte/entry limits, not a promise about Python process RSS.
No neural framework is imported, no whole-corpus fallback exists, and closing
the reader removes only its owned transient index, never collected evidence.
"""
import copy
import hashlib
import json
import math
import os
import sqlite3
import stat
import struct
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from .generation_data import (ACTUAL_SOURCE, GenerationDataset, admit_phase, digest, fields,
                              identifier, sha, strict_json, trajectory_digest, uint, validate_record,
                              _count_target_coverage, _admit_coverage, validate_evidence_receipt)

MANIFEST_DOMAIN = "rz-pals-generation-chunk-manifest/1"
MAX_ROW_BYTES = 1024 * 1024
MAX_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHUNK_ROWS = 128
MAX_CORPUS_BYTES = 18 * 1024**3
MAX_INDEX_BYTES = 512 * 1024**2
MAX_RECORDS = 262144
MAX_EVIDENCE = 1048576
MAX_TRAJECTORIES = 8192
MAX_CHUNKS = 32768
MAX_CACHE_BYTES = 16 * 1024**2
SPLITS = ("train", "validation", "holdout")
KINDS = ("inputs", "evidence", "trajectories", "records")
NATIVE_DOMAINS = ("rz-pals-generation-submission-evidence/1", "rz-pals-generation-native-completion-evidence/1",
                  "rz-pals-generation-native-retirement-evidence/1")


def _native_ticket(ticket):
    """Validate the same bounded causal identity for accepted and failed work."""
    fields(ticket, ("epoch", "request_sequence", "game", "root", "question_sha256", "source_view_sha256", "view_sequence",
                    "context_policy_sha256", "representation_input_sha256", "representation_history_sha256",
                    "representation_model_sha256", "representation_encoding_sha256", "frozen_epoch", "record_revision",
                    "deadline_tick"), "native causal ticket")
    for name in ("epoch", "request_sequence", "game", "root", "view_sequence", "frozen_epoch", "record_revision", "deadline_tick"):
        uint(ticket[name])
    for name in ("question_sha256", "source_view_sha256", "context_policy_sha256", "representation_input_sha256",
                 "representation_history_sha256", "representation_model_sha256", "representation_encoding_sha256"):
        sha(ticket[name])
    return ticket


def _native_retirement(retired):
    """Retain only a complete, unaccepted pre-submission null tuple as raw data."""
    fields(retired, ("schema", "input_sha256", "submission_sha256", "completion_sha256", "ticket", "accepted"), "native retirement")
    if retired["schema"] != NATIVE_DOMAINS[2] or type(retired["accepted"]) is not bool:
        raise ValueError("native retirement domain/acceptance type mismatch")
    _native_ticket(retired["ticket"])
    identities = (retired["input_sha256"], retired["submission_sha256"], retired["completion_sha256"])
    if any(identity is None for identity in identities):
        if all(identity is None for identity in identities) and retired["accepted"] is False:
            return False
        raise ValueError("native retirement requires a complete unaccepted null tuple or complete receipt identities")
    for identity in identities:
        sha(identity)
    return True


def semantic_sample_seal(record, submission):
    """Identify actual model input independently of publication/request IDs.

    This pure identity does not admit evidence: the caller must first validate
    the native submission/completion/retirement chain for the publication row.
    """
    validate_record(record)
    snapshot = record["input"]
    ticket = _native_ticket(submission["ticket"])
    return digest("rz-pals-generation-semantic-sample/1", [
        snapshot["role"], sha(submission["native_model_epoch_sha256"]),
        sha(submission["native_input_sha256"]), sha(ticket["representation_history_sha256"]),
        snapshot["action_support_mask"] if snapshot["role"] == "validator" else None,
    ])


def _native_chain(record, receipts, retirement_sha256, *, with_details=False):
    """Join original submission, physical completion and accepted retirement."""
    input_seal = record["input_sha256"]
    retired = receipts.get(retirement_sha256)
    if not _native_retirement(retired):
        raise ValueError("pre-submission native retirement cannot admit a generation row")
    completed = receipts.get(sha(retired["completion_sha256"]))
    fields(completed, ("schema", "input_sha256", "submission_sha256", "ticket", "physical_completion", "execution",
                       "completed_nn_inputs", "measurement_complete", "physical_fresh", "invocation_key_sha256",
                       "accepted_seed_sha256", "output", "admissible_actual_nn"), "native completion")
    submitted = receipts.get(sha(retired["submission_sha256"]))
    fields(submitted, ("schema", "input_sha256", "ticket", "source_view_seal_sha256", "physical_nn_admitted",
                       "native_input_sha256", "native_model_epoch_sha256", "native_backend_sha256",
                       "native_anchor_history_sha256", "generation_view_sha256", "game_id", "opening_id", "role",
                       "model_epoch", "context_epoch", "capture_sequence", "worker_id", "job_id", "slice_id",
                       "actual_root_anchor", "query_anchor", "source_view_content_sha256", "native_endpoint_history_sha256"), "native submission")
    for receipt, domain in zip((submitted, completed, retired), NATIVE_DOMAINS):
        if receipt["schema"] != domain or receipt["input_sha256"] != input_seal:
            raise ValueError("native receipt has another input/domain")
    if (submitted["physical_nn_admitted"] is not False or completed["submission_sha256"] != retired["submission_sha256"]
            or submitted["ticket"] != completed["ticket"] or completed["ticket"] != retired["ticket"]):
        raise ValueError("native submission/completion/retirement ticket chain mismatch")
    ticket = _native_ticket(submitted["ticket"])
    for name in ("source_view_seal_sha256", "native_input_sha256", "native_model_epoch_sha256", "native_backend_sha256",
                 "generation_view_sha256", "source_view_content_sha256", "native_endpoint_history_sha256"):
        sha(submitted[name])
    history = submitted["native_anchor_history_sha256"]
    if type(history) is not list or len(history) != len(record["input"]["selected_view"]["anchors"]):
        raise ValueError("native submission anchor history allocation mismatch")
    for identity in history:
        sha(identity)
    snapshot = record["input"]
    for name in ("game_id", "opening_id", "role", "model_epoch", "context_epoch", "capture_sequence", "worker_id", "job_id", "slice_id"):
        if type(submitted[name]) is not type(snapshot[name]) or submitted[name] != snapshot[name]:
            raise ValueError("native submission changed original source context")
    from .virtual_review import input_digest
    if (ticket["source_view_sha256"] != submitted["source_view_seal_sha256"]
            or ticket["representation_model_sha256"] != submitted["native_model_epoch_sha256"]
            or ticket["question_sha256"] != record["input"]["question_id"]
            or ticket["context_policy_sha256"] != snapshot["selected_view"]["context_policy_sha256"]
            or ticket["representation_history_sha256"] != submitted["native_endpoint_history_sha256"]
            or submitted["generation_view_sha256"] != record["input"]["view_sha256"]
            or submitted["actual_root_anchor"] != snapshot["selected_view"]["anchors"][0]
            or submitted["query_anchor"] != snapshot["rules_anchor"]
            or submitted["native_input_sha256"] != input_digest(record["input"]["encoded_model_input"])):
        raise ValueError("native representation differs from its original submission")
    fields(completed["execution"], ("epoch", "sequence"), "known physical execution")
    for name in ("epoch", "sequence"):
        uint(completed["execution"][name])
    if completed["execution"]["epoch"] != ticket["epoch"]:
        raise ValueError("physical execution belongs to another request epoch")
    if (completed["physical_completion"] != "confirmed" or completed["measurement_complete"] is not True
            or completed["admissible_actual_nn"] is not True or not uint(completed["completed_nn_inputs"])
            or type(completed["physical_fresh"]) is not bool or type(completed["output"]) is not dict
            or not completed["output"] or retired["accepted"] is not True):
        raise ValueError("actual generation input lacks accepted confirmed physical NN evidence")
    sha(completed["invocation_key_sha256"])
    if completed["physical_fresh"]:
        if completed["accepted_seed_sha256"] is not None:
            raise ValueError("Fresh physical invocation cannot claim a Warm seed")
    else:
        sha(completed["accepted_seed_sha256"])
    _native_output(snapshot, completed["output"])
    semantic_seal = semantic_sample_seal(record, submitted)
    if with_details:
        return {"semantic_sample_sha256": semantic_seal, "physical_fresh": completed["physical_fresh"]}
    return semantic_seal


def _native_output(snapshot, output):
    role = snapshot["role"]
    expected = {"proposer": ("proposal", ("kind", "candidate_logits_f32", "wdl_f32")),
                "critic": ("counterexample", ("kind", "candidate_logits_f32", "divergence_logits_f32", "wdl_f32")),
                "validator": ("task_ranking", ("kind", "task_logits_f32"))}[role]
    fields(output, expected[1], "actual native role output")
    if output["kind"] != expected[0]:
        raise ValueError("actual native output role mismatch")
    def values(name, count):
        raw = output[name]
        if type(raw) is not list or len(raw) != count:
            raise ValueError("actual native output allocation mismatch")
        decoded = []
        for value in raw:
            if type(value) is not str or len(value) != 8 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("actual native output needs canonical f32 bit strings")
            number = struct.unpack(">f", bytes.fromhex(value))[0]
            if not math.isfinite(number):
                raise ValueError("actual native output is nonfinite")
            decoded.append(number)
        return decoded
    if role == "validator":
        values("task_logits_f32", 7)
    else:
        values("candidate_logits_f32", len(snapshot["legal_moves"]))
        if role == "critic":
            values("divergence_logits_f32", len(snapshot["divergence_record_ids"]))
        wdl = values("wdl_f32", 3)
        if any(v < 0 or v > 1 for v in wdl) or abs(sum(wdl) - 1) > 1e-4:
            raise ValueError("actual native WDL is not a normalized probability")


def _safe_path(root, relative):
    if type(relative) is not str or "\\" in relative:
        raise ValueError("chunk path must be relative POSIX JSONL")
    parts = PurePosixPath(relative)
    if (parts.is_absolute() or not parts.parts or any(p in ("", ".", "..") for p in parts.parts)
            or str(parts) != relative or parts.suffix != ".jsonl" or ":" in relative):
        raise ValueError("chunk path escapes registered corpus")
    target = root.joinpath(*parts.parts)
    # Walk only the declared root and descendants, including the final file.
    # A parent-chain loop can reach root.parent before checking target itself.
    current = root
    for depth in range(len(parts.parts) + 1):
        if current.is_symlink() or current.exists() and getattr(current.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("linked/reparse generation chunk is forbidden")
        if depth < len(parts.parts):
            current = current / parts.parts[depth]
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("chunk outside immutable corpus root")
    return target


def validate_manifest(manifest):
    fields(manifest, ("schema", "source_kind", "producer", "opening_splits", "chunks", "total_bytes", "total_records"), "chunk manifest")
    if manifest["schema"] != MANIFEST_DOMAIN or manifest["source_kind"] != ACTUAL_SOURCE:
        raise ValueError("unknown generation chunk origin")
    producer, splits = manifest["producer"], manifest["opening_splits"]
    fields(producer, ("schema", "source_kind", "source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256",
                      "teacher_profile_sha256", "opening_split_sha256", "registered_before_outcomes", "actual_game_ids"), "chunk producer authority")
    if (producer["schema"] != "rz-pals-actual-collection-authority/2" or producer["source_kind"] != ACTUAL_SOURCE
            or producer["registered_before_outcomes"] is not True):
        raise ValueError("actual chunk producer authority required")
    for name in ("source_sha256", "model_sha256", "runtime_sha256", "teacher_binary_sha256", "teacher_profile_sha256", "opening_split_sha256"):
        sha(producer[name])
    if type(splits) is not dict or not 1 <= len(splits) <= 8192 or any(v not in SPLITS for v in splits.values()):
        raise ValueError("bounded preregistered opening splits required")
    for key in splits:
        identifier(key)
    if producer["opening_split_sha256"] != digest("rz-pals-opening-split/1", splits):
        raise ValueError("outcome-selected or changed opening split")
    games = producer["actual_game_ids"]
    if type(games) is not list or len(games) > MAX_TRAJECTORIES or len({identifier(g) for g in games}) != len(games):
        raise ValueError("bounded unique actual game publications required")
    chunks = manifest["chunks"]
    if type(chunks) is not list or not 1 <= len(chunks) <= MAX_CHUNKS:
        raise ValueError("bounded nonempty chunk manifest required")
    total_bytes, total_records, paths = 0, 0, set()
    for index, chunk in enumerate(chunks):
        fields(chunk, ("index", "kind", "path", "sha256", "bytes", "rows"), "generation chunk")
        if uint(chunk["index"]) != index or chunk["kind"] not in KINDS:
            raise ValueError("unknown or reordered chunk")
        identifier(chunk["path"])
        if chunk["path"] in paths:
            raise ValueError("duplicate chunk path")
        paths.add(chunk["path"])
        sha(chunk["sha256"])
        if not 1 <= uint(chunk["bytes"], MAX_CHUNK_BYTES) or not 1 <= uint(chunk["rows"], MAX_CHUNK_ROWS):
            raise ValueError("empty or oversized generation chunk")
        total_bytes += chunk["bytes"]
        total_records += chunk["rows"] if chunk["kind"] == "records" else 0
    if (uint(manifest["total_bytes"], MAX_CORPUS_BYTES) != total_bytes
            or uint(manifest["total_records"], MAX_RECORDS) != total_records):
        raise ValueError("generation corpus byte/record totals mismatch")
    return digest(MANIFEST_DOMAIN, manifest)


class _Lookup:
    def __init__(self, owner, table, value_field):
        self.owner, self.table, self.value_field = owner, table, value_field

    def __contains__(self, key):
        return self.owner._db.execute(f"SELECT 1 FROM {self.table} WHERE seal=?", (key,)).fetchone() is not None

    def get(self, key, default=None):
        found = self.owner._db.execute(f"SELECT chunk,offset,length FROM {self.table} WHERE seal=?", (key,)).fetchone()
        if found is None:
            return default
        row = self.owner._read_row(*found)
        if row["sha256"] != key:
            raise ValueError("sealed authority/evidence index mismatch")
        return row[self.value_field]


class ChunkRows(Sequence):
    """Immutable row references; random access materializes at most one row."""
    def __init__(self, owner, split):
        self.owner, self.split = owner, split
        self.count = owner._db.execute("SELECT COUNT(*) FROM selected WHERE split=?", (split,)).fetchone()[0]

    def __len__(self):
        self.owner._assert_open()
        return self.count

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError("bounded generation row access requires one integer, not a whole-corpus slice")
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.owner._check_index()
        reference = self.owner._db.execute("SELECT chunk,offset,length,row_seal FROM selected WHERE split=? AND ordinal=?", (self.split, index)).fetchone()
        result = self.owner._read_row(*reference[:3])
        if digest("rz-pals-generation-chunk-row/1", result) != reference[3]:
            raise ValueError("immutable generation row changed")
        validate_record(result)
        return result

    def __iter__(self):
        self.owner._check_index()
        references = self.owner._db.execute("SELECT chunk,offset,length,row_seal FROM selected WHERE split=? ORDER BY ordinal", (self.split,))
        active, stream = None, None
        try:
            for chunk, offset, length, row_seal in references:
                if chunk != active:
                    if stream is not None:
                        stream.close()
                    stream = self.owner._open_verified(chunk)
                    active = chunk
                stream.seek(offset)
                raw = stream.read(length)
                result = strict_json(raw.decode("utf-8", "strict"))
                if digest("rz-pals-generation-chunk-row/1", result) != row_seal:
                    raise ValueError("immutable streamed generation row changed")
                validate_record(result)
                yield result
        finally:
            if stream is not None:
                stream.close()

    def role_pool(self, role):
        """Only bounded indices and input identities, never all input payloads."""
        self.owner._check_index()
        found = self.owner._db.execute("SELECT ordinal,input_seal FROM selected WHERE split=? AND role=? ORDER BY ordinal", (self.split, role))
        indices, hasher, first = [], hashlib.sha256(), True
        # Canonical [domain,[manifest, [input seals...]]] without a seal list copy.
        hasher.update(b"[" + json.dumps("rz-pals-generation-chunk-sampler-pool/1").encode()
                      + b",[" + json.dumps(self.owner._seal).encode() + b",[")
        for ordinal, seal in found:
            indices.append(ordinal)
            hasher.update((b"" if first else b",") + json.dumps(seal).encode())
            first = False
        hasher.update(b"]]]")
        return indices, hasher.hexdigest()


class ChunkGenerationDataset:
    """Global admission over pinned chunks using bounded disposable SQLite.

An external manifest pin is mandatory. Receipt claims still require recovery
from the actual native producer; hashes alone do not prove an executed game.
The index is regenerated in an explicitly supplied transient directory.
"""
    def __init__(self, root, manifest, *, expected_manifest_sha256, index_root, max_index_bytes=MAX_INDEX_BYTES):
        self._closed, self._sealed, self._temporary = False, False, None
        self._cache, self._cache_bytes = OrderedDict(), 0
        self._seal = validate_manifest(manifest)  # Reject capacity before copying caller data.
        self._manifest = copy.deepcopy(manifest)
        if validate_manifest(self._manifest) != self._seal:
            raise ValueError("manifest changed while forming immutable snapshot")
        if sha(expected_manifest_sha256) != self._seal:
            raise ValueError("generation manifest differs from externally registered pin")
        self._root = Path(root).resolve(strict=True)
        if Path(root).is_symlink() or not self._root.is_dir():
            raise ValueError("explicit unlinked generation corpus directory required")
        self._paths = [_safe_path(self._root, c["path"]) for c in self._manifest["chunks"]]
        self._identities = [None] * len(self._paths)
        self._max_index_bytes = uint(max_index_bytes, MAX_INDEX_BYTES)
        if self._max_index_bytes < 65536:
            raise ValueError("generation index capacity is too small")
        parent = Path(index_root).resolve(strict=True)
        if not parent.is_dir() or Path(index_root).is_symlink():
            raise ValueError("explicit unlinked transient index directory required")
        self._index_parent = parent
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="rz-generation-index-", dir=parent)
            self._index_path = Path(self._temporary.name) / "verified.sqlite3"
            self._db = sqlite3.connect(self._index_path)
            self._db.execute("PRAGMA journal_mode=OFF")
            self._db.execute("PRAGMA temp_store=FILE")
            self._db.execute("PRAGMA cache_size=-2048")
            self._db.execute(f"PRAGMA max_page_count={self._max_index_bytes // 4096}")
            self._build_index()
            self._db.commit()
            self._db.execute("PRAGMA query_only=ON")
            self._index_seal = self._index_digest()
            self._index_stamp = self._file_stamp(self._index_path)
            self._sealed = True
        except sqlite3.Error as error:
            self.close()
            raise ValueError("GenerationIndexCapacityOrIntegrity: bounded SQLite index refused data") from error
        except Exception:
            self.close()
            raise

    def _assert_open(self):
        if self._closed:
            raise ValueError("generation chunk dataset is closed")

    def _open_verified(self, index):
        self._assert_open()
        chunk, path = self._manifest["chunks"][index], self._paths[index]
        if _safe_path(self._root, chunk["path"]) != path:
            raise ValueError("immutable chunk path changed")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        stream = os.fdopen(os.open(path, flags), "rb")
        try:
            info = os.fstat(stream.fileno())
            identity = (info.st_dev, info.st_ino, info.st_size)
            if not stat.S_ISREG(info.st_mode) or info.st_size != chunk["bytes"] or info.st_nlink != 1:
                raise ValueError("chunk type/size/link identity mismatch")
            if self._identities[index] is not None and self._identities[index] != identity:
                raise ValueError("immutable chunk was replaced")
            hasher, total = hashlib.sha256(), 0
            while True:
                block = stream.read(65536)
                if not block:
                    break
                total += len(block)
                if total > MAX_CHUNK_BYTES:
                    raise ValueError("chunk byte capacity exceeded")
                hasher.update(block)
            if total != chunk["bytes"] or hasher.hexdigest() != chunk["sha256"]:
                raise ValueError("generation chunk byte hash mismatch")
            current = path.stat()
            if (current.st_dev, current.st_ino, current.st_size) != identity:
                raise ValueError("chunk changed during verified open")
            self._identities[index] = identity
            stream.seek(0)
            return stream
        except Exception:
            stream.close()
            raise

    def _chunk_rows(self, index):
        with self._open_verified(index) as stream:
            count = 0
            while True:
                offset = stream.tell()
                raw = stream.readline(MAX_ROW_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_ROW_BYTES or not raw.endswith(b"\n"):
                    raise ValueError("bounded complete UTF-8 JSONL row required")
                count += 1
                if count > MAX_CHUNK_ROWS:
                    raise ValueError("chunk row capacity exceeded")
                yield offset, len(raw), strict_json(raw.decode("utf-8", "strict"))
            if count != self._manifest["chunks"][index]["rows"]:
                raise ValueError("chunk row count mismatch")

    def _read_row(self, chunk, offset, length):
        uint(length, MAX_ROW_BYTES)
        # Cached bytes are an immutable verified snapshot, not parsed corpus
        # objects. Full source hashes are checked again by sha256/admission;
        # callers never receive or mutate the cache backing this row.
        path = _safe_path(self._root, self._manifest["chunks"][chunk]["path"])
        info = path.stat()
        stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        cached = self._cache.pop(chunk, None)
        if cached is not None and cached[0] != stamp:
            self._cache_bytes -= len(cached[1])
            cached = None
        if cached is None:
            with self._open_verified(chunk) as stream:
                content = stream.read(MAX_CHUNK_BYTES + 1)
            if len(content) != self._manifest["chunks"][chunk]["bytes"] or hashlib.sha256(content).hexdigest() != self._manifest["chunks"][chunk]["sha256"]:
                raise ValueError("chunk changed while forming immutable cache")
            while self._cache and self._cache_bytes + len(content) > MAX_CACHE_BYTES:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= len(evicted[1])
            cached = (stamp, content)
            self._cache_bytes += len(content)
        self._cache[chunk] = cached
        raw = cached[1][offset:offset + length]
        if len(raw) != length or not raw.endswith(b"\n"):
            raise ValueError("sealed row offset/length mismatch")
        return strict_json(raw.decode("utf-8", "strict"))

    def _index_digest(self):
        size = self._index_path.stat().st_size
        if size > self._max_index_bytes or size + self._manifest["total_bytes"] > MAX_CORPUS_BYTES:
            raise ValueError("GenerationIndexCapacity: corpus plus transient index exceeds storage budget")
        hasher = hashlib.sha256()
        with self._index_path.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()

    @staticmethod
    def _file_stamp(path):
        info = path.stat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _check_index(self):
        self._assert_open()
        # This private query-only index has one owner and is never externally
        # supplied. Any observed replacement/write triggers full verification;
        # corpus identity/admission still rehash every source chunk's bytes.
        if self._sealed and self._file_stamp(self._index_path) != self._index_stamp:
            raise ValueError("immutable generation manifest/index changed")

    def _build_index(self):
        self._db.executescript("""
            CREATE TABLE inputs(seal TEXT PRIMARY KEY);
            CREATE TABLE evidence(seal TEXT PRIMARY KEY,chunk INTEGER,offset INTEGER,length INTEGER);
            CREATE TABLE trajectories(seal TEXT PRIMARY KEY,chunk INTEGER,offset INTEGER,length INTEGER);
            CREATE TABLE native(input_seal TEXT,domain TEXT,seal TEXT,PRIMARY KEY(input_seal,domain,seal));
            CREATE TABLE records(position INTEGER PRIMARY KEY,chunk INTEGER,offset INTEGER,length INTEGER,row_seal TEXT,
                                 input_seal TEXT UNIQUE,semantic_seal TEXT,valid_target INTEGER,physical_fresh INTEGER,role TEXT,split TEXT,
                                 excluded INTEGER DEFAULT 0);
            CREATE INDEX by_semantic_choice ON records(semantic_seal,excluded,physical_fresh,valid_target,position);
            CREATE TABLE domains(kind TEXT,identity TEXT,mask INTEGER,PRIMARY KEY(kind,identity));
            CREATE TABLE record_domains(position INTEGER,kind TEXT,identity TEXT);
            CREATE INDEX by_record_domain ON record_domains(position);
            CREATE TABLE selected(split TEXT,ordinal INTEGER,chunk INTEGER,offset INTEGER,length INTEGER,row_seal TEXT,
                                  input_seal TEXT,role TEXT,PRIMARY KEY(split,ordinal));
        """)
        counts = {kind: 0 for kind in KINDS}
        # All authorities are indexed before any supervised record is joined.
        for chunk in self._manifest["chunks"]:
            kind, index = chunk["kind"], chunk["index"]
            if kind == "records":
                continue
            for offset, length, row in self._chunk_rows(index):
                counts[kind] += 1
                cap = {"inputs": MAX_RECORDS, "evidence": MAX_EVIDENCE, "trajectories": MAX_TRAJECTORIES}[kind]
                if counts[kind] > cap:
                    raise ValueError("generation authority entry capacity exceeded")
                if kind == "inputs":
                    fields(row, ("input_sha256",), "published chunk input")
                    self._db.execute("INSERT INTO inputs VALUES(?)", (sha(row["input_sha256"]),))
                else:
                    name = "receipt" if kind == "evidence" else "trajectory"
                    fields(row, ("sha256", name), "chunk authority entry")
                    seal, value = sha(row["sha256"]), row[name]
                    if kind == "evidence":
                        try:
                            validate_evidence_receipt(seal, value)
                        except ValueError as error:
                            raise ValueError("chunk full receipt seal/domain mismatch") from error
                        if value["schema"] in NATIVE_DOMAINS:
                            _native_ticket(value.get("ticket"))
                            if value["schema"] == NATIVE_DOMAINS[2] and not _native_retirement(value):
                                pass  # Complete pre-submission failure is retained, never a row.
                            else:
                                self._db.execute("INSERT INTO native VALUES(?,?,?)", (sha(value.get("input_sha256")), value["schema"], seal))
                    elif trajectory_digest(value) != seal:
                        raise ValueError("chunk trajectory seal mismatch")
                    self._db.execute(f"INSERT INTO {kind} VALUES(?,?,?,?)", (seal, index, offset, length))
        producer, splits = self._manifest["producer"], self._manifest["opening_splits"]
        published, receipts = _Lookup(self, "inputs", None), _Lookup(self, "evidence", "receipt")
        trajectories, games = _Lookup(self, "trajectories", "trajectory"), set(producer["actual_game_ids"])
        duplicate_rows, position = 0, 0
        for chunk in self._manifest["chunks"]:
            if chunk["kind"] != "records":
                continue
            for offset, length, record in self._chunk_rows(chunk["index"]):
                counts["records"] += 1
                GenerationDataset._validate_published_record(record, producer, published, games, trajectories, receipts)
                accepted = []
                for (retirement_sha256,) in self._db.execute("SELECT seal FROM native WHERE input_seal=? AND domain=?", (record["input_sha256"], NATIVE_DOMAINS[2])):
                    if receipts.get(retirement_sha256).get("accepted") is True:
                        accepted.append(retirement_sha256)
                        if len(accepted) > 1:
                            raise ValueError("duplicate accepted native chains for one input seal")
                if len(accepted) != 1:
                    raise ValueError("published input has no accepted native retirement")
                chain = _native_chain(record, receipts, accepted[0], with_details=True)
                semantic_seal, physical_fresh = chain["semantic_sample_sha256"], chain["physical_fresh"]
                targets = validate_record(record)
                snapshot = record["input"]
                valid_target = (bool(targets["task"] and targets["task"]["pairs"])
                                if snapshot["role"] == "validator"
                                else any(targets[head] is not None for head in ("policy", "wdl", "divergence")))
                if snapshot["opening_id"] not in splits:
                    raise ValueError("unregistered opening in generation chunks")
                row_seal = digest("rz-pals-generation-chunk-row/1", record)
                found = self._db.execute("SELECT row_seal FROM records WHERE input_seal=?", (record["input_sha256"],)).fetchone()
                if found:
                    if found[0] != row_seal:
                        raise ValueError("conflicting targets/provenance for identical generation input seal")
                    duplicate_rows += 1
                    continue
                split = splits[snapshot["opening_id"]]
                self._db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,0)",
                                 (position, chunk["index"], offset, length, row_seal, record["input_sha256"],
                                  semantic_seal, int(valid_target), int(physical_fresh), snapshot["role"], split))
                for kind, identity in (("genealogy", snapshot["line_genealogy_id"]), ("transposition", snapshot["transposition_sha256"]),
                                       ("rules_state", snapshot["rules_anchor"]["rules_state_sha256"]), ("game", snapshot["game_id"]),
                                       ("sample", semantic_seal)):
                    bit = 1 << SPLITS.index(split)
                    self._db.execute("INSERT INTO domains VALUES(?,?,?) ON CONFLICT(kind,identity) DO UPDATE SET mask=mask|excluded.mask", (kind, identity, bit))
                    self._db.execute("INSERT INTO record_domains VALUES(?,?,?)", (position, kind, identity))
                position += 1
        if counts["records"] != self._manifest["total_records"]:
            raise ValueError("manifest record count differs from parsed corpus")
        if self._db.execute("SELECT COUNT(*) FROM inputs WHERE seal NOT IN (SELECT input_seal FROM records)").fetchone()[0]:
            raise ValueError("published input seal absent from collected record corpus")
        # Incremental empty-table indexes and a correlated EXISTS avoid a
        # corpus-sized sort/IN-result temp file outside max_page_count.
        self._db.execute("UPDATE records SET excluded=1 WHERE EXISTS (SELECT 1 FROM record_domains rd JOIN domains d USING(kind,identity) WHERE rd.position=records.position AND d.mask NOT IN (1,2,4))")
        self._exclusions = {"cross_split_records_excluded": self._db.execute("SELECT COUNT(*) FROM records WHERE excluded=1").fetchone()[0],
                           "cross_split_identities": self._db.execute("SELECT COUNT(*) FROM domains WHERE mask NOT IN (1,2,4)").fetchone()[0],
                           "duplicate_inputs_excluded": duplicate_rows,
                           "warm_seed_inputs_excluded": self._db.execute("SELECT COUNT(*) FROM records WHERE physical_fresh=0").fetchone()[0]}
        for split in SPLITS:
            # Only confirmed Fresh rows enter actual training; Warm source,
            # output and seeds remain raw evidence and are never rewritten.
            # Only target availability is compared among Fresh repetitions.
            # Prefer earliest valid, or earliest masked if every one is masked.
            # Raw publication rows stay in records and original chunks.
            # Two indexed EXISTS probes avoid GROUP BY or a temp sorting pool.
            source = self._db.execute("""
                SELECT r.chunk,r.offset,r.length,r.row_seal,r.input_seal,r.role
                FROM records r NOT INDEXED WHERE r.excluded=0 AND r.physical_fresh=1 AND r.split=?
                  AND (r.valid_target=1 OR NOT EXISTS (
                    SELECT 1 FROM records v INDEXED BY by_semantic_choice WHERE v.semantic_seal=r.semantic_seal
                      AND v.excluded=0 AND v.physical_fresh=1 AND v.valid_target=1))
                  AND NOT EXISTS (
                    SELECT 1 FROM records earlier INDEXED BY by_semantic_choice WHERE earlier.semantic_seal=r.semantic_seal
                      AND earlier.excluded=0 AND earlier.physical_fresh=1 AND earlier.valid_target=r.valid_target
                      AND earlier.position<r.position)
                ORDER BY r.position
            """, (split,))
            for ordinal, reference in enumerate(source):
                self._db.execute("INSERT INTO selected VALUES(?,?,?,?,?,?,?,?)", (split, ordinal, *reference))
        self._exclusions["duplicate_semantic_inputs_excluded"] = (
            self._db.execute("SELECT COUNT(*) FROM records WHERE excluded=0 AND physical_fresh=1").fetchone()[0]
            - self._db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
        )

    def rows(self, split, *, optimization=False):
        self._check_index()
        if split not in SPLITS or optimization and split != "train":
            raise ValueError("holdout/validation corpus cannot be an optimization pool")
        return ChunkRows(self, split)

    def admit(self, phase):
        # SQL selected already removes semantic repeats and publication conflicts;
        # no corpus-sized Python set or full record collection is needed here.
        self.sha256
        if (self._exclusions["warm_seed_inputs_excluded"]
                and not self._db.execute("SELECT 1 FROM records WHERE physical_fresh=1 LIMIT 1").fetchone()):
            raise ValueError("UnsupportedWarmTrainingNamespace: eligible_fresh_inputs=0")
        counts = Counter()
        for record in self.rows("train", optimization=True):
            _count_target_coverage(counts, record, validate_record(record))
        return _admit_coverage(dict(counts), phase)

    @property
    def exclusions(self):
        self._check_index()
        return dict(self._exclusions)

    @property
    def exclusion_reasons(self):
        self._check_index()
        return {"warm_seed_inputs_excluded": "UnsupportedWarmTrainingNamespace"}

    @property
    def sha256(self):
        self._check_index()
        if validate_manifest(self._manifest) != self._seal or self._index_digest() != self._index_seal:
            raise ValueError("immutable generation manifest/index changed")
        # A dataset identity never survives unnoticed source chunk changes.
        for index in range(len(self._paths)):
            with self._open_verified(index):
                pass
        return self._seal

    @property
    def authority(self):
        self._check_index()
        return copy.deepcopy(self._manifest["producer"])

    @property
    def opening_splits(self):
        self._check_index()
        return copy.deepcopy(self._manifest["opening_splits"])

    def close(self):
        try:
            if not self._closed:
                self._closed = True
                if hasattr(self, "_db"):
                    self._db.close()
            if self._temporary is not None:
                target = Path(self._temporary.name)
                if target.exists() and (target.is_symlink() or not target.resolve(strict=True).is_relative_to(self._index_parent)
                                         or getattr(target.stat(), "st_file_attributes", 0) & 0x400):
                    raise ValueError("owned transient index cleanup target changed; quarantined")
                self._temporary.cleanup()
                self._temporary = None
                self.cleanup_failure = None
        except OSError as error:
            self.cleanup_failure = {"error_type": type(error).__name__, "owned_index_root": self._temporary.name}
            raise ValueError("GenerationIndexCleanupFailed: " + self._temporary.name) from error
        finally:
            self._cache.clear()
            self._cache_bytes = 0

    def __enter__(self):
        self._assert_open()
        return self

    def __exit__(self, *_):
        self.close()
