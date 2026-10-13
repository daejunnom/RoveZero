//! Exclusive bounded JSONL storage. Hashes are read back from actual files;
//! failed or unrecovered output is never deleted by this owner.
use super::*;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

pub const GENERATION_CHUNK_DOMAIN: &str = "rz-pals-generation-chunk-manifest/1";
pub const GENERATION_CHUNK_BYTES: u64 = 8 * 1024 * 1024;
pub const GENERATION_CHUNK_ROWS: u32 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationChunkKindV3 {
    Inputs,
    Records,
    Evidence,
    Trajectories,
}
impl GenerationChunkKindV3 {
    fn label(self) -> &'static str {
        match self {
            Self::Inputs => "inputs",
            Self::Records => "records",
            Self::Evidence => "evidence",
            Self::Trajectories => "trajectories",
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationChunkPinV3 {
    pub index: u32,
    pub kind: GenerationChunkKindV3,
    pub path: String,
    pub bytes: u64,
    pub rows: u32,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationChunkManifestV3 {
    pub schema: String,
    pub source_kind: String,
    pub producer: serde_json::Value,
    pub opening_splits: BTreeMap<String, GenerationSplitV3>,
    pub chunks: Vec<GenerationChunkPinV3>,
    pub total_bytes: u64,
    pub total_records: u64,
}
impl GenerationChunkManifestV3 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == GENERATION_CHUNK_DOMAIN
                && self.source_kind == "actual_selfplay"
                && self.total_bytes <= GENERATION_OUTPUT_CAP
                && self.total_records <= GENERATION_MAX_RECORDS as u64
                && self.chunks.len() <= 16_384,
            "generation chunk manifest bound/domain",
        )?;
        let mut names = BTreeSet::new();
        let mut total = 0u64;
        let mut rows = 0u64;
        for (index, pin) in self.chunks.iter().enumerate() {
            check(
                pin.index as usize == index
                    && safe_name(&pin.path)
                    && names.insert(&pin.path)
                    && generation_sha(&pin.sha256)
                    && pin.bytes > 0
                    && pin.bytes <= GENERATION_CHUNK_BYTES
                    && pin.rows > 0
                    && pin.rows <= GENERATION_CHUNK_ROWS,
                "generation immutable chunk pin",
            )?;
            total = total
                .checked_add(pin.bytes)
                .ok_or_else(|| invalid("generation bytes overflow"))?;
            if pin.kind == GenerationChunkKindV3::Records {
                rows += u64::from(pin.rows);
            }
        }
        check(
            total == self.total_bytes
                && rows == self.total_records
                && self.producer["schema"] == "rz-pals-actual-collection-authority/2"
                && self.producer["opening_split_sha256"]
                    == generation_digest("rz-pals-opening-split/1", &self.opening_splits)?
                && self.producer.get("input_seals").is_none(),
            "generation chunk accounting/inline producer",
        )
    }
}
fn safe_name(name: &str) -> bool {
    ["inputs-", "records-", "evidence-", "trajectories-"]
        .iter()
        .any(|prefix| {
            name.strip_prefix(prefix).is_some_and(|rest| {
                rest.strip_suffix(".jsonl").is_some_and(|digits| {
                    digits.len() == 8 && digits.bytes().all(|b| b.is_ascii_digit())
                })
            })
        })
}
fn is_link(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return true;
        }
    }
    false
}
fn io(error: std::io::Error) -> ArenaError {
    ArenaError::Io(error.to_string())
}

struct ActiveChunk {
    kind: GenerationChunkKindV3,
    name: String,
    file: File,
    bytes: u64,
    rows: u32,
    hash: Sha256,
}
pub struct GenerationChunkWriterV3 {
    directory: PathBuf,
    manifest: GenerationChunkManifestV3,
    /// At most one open file for each of four kinds. Alternating record and
    /// receipt writes cannot create an otherwise unbounded forest of files.
    active: BTreeMap<GenerationChunkKindV3, ActiveChunk>,
    sequence: u64,
    maximum_output_bytes: u64,
    poisoned: bool,
    sealed: bool,
    recovery: BTreeSet<String>,
}
impl GenerationChunkWriterV3 {
    pub fn create(
        directory: &Path,
        run_id: String,
        maximum_output_bytes: u64,
    ) -> Result<Self, ArenaError> {
        check(
            directory.is_absolute()
                && generation_identifier(&run_id)
                && maximum_output_bytes > GENERATION_CHUNK_BYTES
                && maximum_output_bytes <= GENERATION_OUTPUT_CAP,
            "owned absolute generation directory/output budget",
        )?;
        let parent = directory
            .parent()
            .ok_or_else(|| invalid("generation output parent"))?;
        for ancestor in parent.ancestors() {
            let metadata = fs::symlink_metadata(ancestor).map_err(io)?;
            check(
                metadata.is_dir() && !is_link(&metadata) && !ancestor.join(".git").exists(),
                "generation output must be outside Git and link/junction ancestors",
            )?;
        }
        fs::create_dir(directory).map_err(io)?;
        let canonical = fs::canonicalize(directory).map_err(io)?;
        let marker = serde_json::json!({"schema":"rz-pals-generation-owned-output/1","run_id":run_id,"maximum_output_bytes":maximum_output_bytes,"cleanup":"preserve-until-recovery-verified"});
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(canonical.join("owner.json"))
            .map_err(io)?;
        let mut bytes = serde_json::to_vec(&marker).map_err(|e| invalid(e.to_string()))?;
        bytes.push(b'\n');
        file.write_all(&bytes).map_err(io)?;
        file.sync_all().map_err(io)?;
        Ok(Self {
            directory: canonical,
            manifest: GenerationChunkManifestV3 {
                schema: GENERATION_CHUNK_DOMAIN.into(),
                source_kind: "actual_selfplay".into(),
                producer: serde_json::Value::Null,
                opening_splits: BTreeMap::new(),
                chunks: vec![],
                total_bytes: 0,
                total_records: 0,
            },
            maximum_output_bytes,
            active: BTreeMap::new(),
            sequence: 0,
            poisoned: false,
            sealed: false,
            recovery: BTreeSet::new(),
        })
    }
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn append_record(&mut self, record: &GenerationRecordV3) -> Result<(), ArenaError> {
        record.validate()?;
        check(
            self.manifest.total_records < GENERATION_MAX_RECORDS as u64,
            "generation record cap",
        )?;
        self.append(GenerationChunkKindV3::Records, record)
    }
    pub fn append_evidence(&mut self, receipt: &serde_json::Value) -> Result<String, ArenaError> {
        let schema = receipt
            .get("schema")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid("typed generation evidence schema"))?;
        check(
            [
                "rz-pals-generation-head-evidence/1",
                "rz-pals-generation-causal-evidence/1",
                "rz-pals-generation-witness-evidence/1",
                "rz-pals-rules-outcome-evidence/1",
                "rz-pals-generation-submission-evidence/1",
                "rz-pals-generation-native-completion-evidence/1",
                "rz-pals-generation-native-retirement-evidence/1",
                "rz-pals-generation-runtime-audit-evidence/1",
                "rz-pals-generation-action-authority-evidence/1",
                "rz-pals-generation-game-pgn-evidence/1",
                "rz-pals-generation-game-failure-evidence/1",
                "rz-pals-generation-teacher-submission-evidence/1",
                "rz-pals-generation-teacher-completion-evidence/1",
                "rz-pals-generation-teacher-input-binding-evidence/1",
            ]
            .contains(&schema),
            "unsupported generation evidence domain",
        )?;
        let sha = generation_digest(schema, receipt)?;
        self.append(
            GenerationChunkKindV3::Evidence,
            &serde_json::json!({"sha256":sha,"receipt":receipt}),
        )?;
        Ok(sha)
    }
    pub fn append_trajectory(
        &mut self,
        authority: &GenerationTrajectoryAuthorityV3,
    ) -> Result<String, ArenaError> {
        let sha = authority.sha256()?;
        self.append(
            GenerationChunkKindV3::Trajectories,
            &serde_json::json!({"sha256":sha,"trajectory":authority}),
        )?;
        Ok(sha)
    }
    pub fn publish_authority(
        &mut self,
        authority: &GenerationProducerAuthorityV3,
        splits: &BTreeMap<String, GenerationSplitV3>,
    ) -> Result<(), ArenaError> {
        authority.validate(splits)?;
        check(
            self.manifest.producer.is_null(),
            "duplicate actual producer publication",
        )?;
        // Input seals can exceed the whole-JSON reader's 16MiB ceiling. The V2
        // authority stays inline; one immutable input seal occupies each row.
        // Do not first serialize the potentially large input seal vector just
        // to remove it. The authority's exact inline subset is bounded pins and
        // game IDs; each actual input is written separately below.
        let header = serde_json::json!({"schema":"rz-pals-actual-collection-authority/2",
            "source_kind":authority.source_kind,"source_sha256":authority.source_sha256,
            "model_sha256":authority.model_sha256,"runtime_sha256":authority.runtime_sha256,
            "teacher_binary_sha256":authority.teacher_binary_sha256,"teacher_profile_sha256":authority.teacher_profile_sha256,
            "opening_split_sha256":authority.opening_split_sha256,"registered_before_outcomes":authority.registered_before_outcomes,
            "actual_game_ids":authority.actual_game_ids});
        for seal in &authority.input_seals {
            self.append(
                GenerationChunkKindV3::Inputs,
                &serde_json::json!({"input_sha256":seal}),
            )?;
        }
        self.manifest.producer = header;
        self.manifest.opening_splits = splits.clone();
        Ok(())
    }
    fn append<T: Serialize>(
        &mut self,
        kind: GenerationChunkKindV3,
        value: &T,
    ) -> Result<(), ArenaError> {
        check(
            !self.poisoned && !self.sealed,
            "generation writer failed/sealed",
        )?;
        let mut value = serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?;
        value.sort_all_objects();
        let mut row = serde_json::to_vec(&value).map_err(|e| invalid(e.to_string()))?;
        row.push(b'\n');
        check(
            row.len() <= GENERATION_MAX_ROW_BYTES,
            "generation JSONL row cap",
        )?;
        let next = self
            .manifest
            .total_bytes
            .checked_add(row.len() as u64)
            .and_then(|n| n.checked_add(self.active.values().map(|a| a.bytes).sum::<u64>()))
            .ok_or_else(|| invalid("generation output overflow"))?;
        check(
            next + self.manifest.chunks.len() as u64 * 256 + 4 * 1024 * 1024
                <= self.maximum_output_bytes,
            "generation output quota preserves final manifest reserve",
        )?;
        if self.active.get(&kind).is_some_and(|a| {
            a.rows >= GENERATION_CHUNK_ROWS || a.bytes + row.len() as u64 > GENERATION_CHUNK_BYTES
        }) {
            self.flush_chunk(kind)?;
        }
        if !self.active.contains_key(&kind) {
            check(
                self.manifest.chunks.len() + self.active.len() < 16_384,
                "generation chunk count quota before file creation",
            )?;
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or_else(|| invalid("generation chunk sequence overflow"))?;
            let name = format!("{}-{:08}.jsonl", kind.label(), self.sequence);
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.directory.join(&name))
                .map_err(io)?;
            self.active.insert(
                kind,
                ActiveChunk {
                    kind,
                    name,
                    file,
                    bytes: 0,
                    rows: 0,
                    hash: Sha256::new(),
                },
            );
        }
        let active = self.active.get_mut(&kind).unwrap();
        if let Err(error) = active.file.write_all(&row) {
            self.poisoned = true;
            return Err(io(error));
        }
        active.hash.update(&row);
        active.bytes += row.len() as u64;
        active.rows += 1;
        if kind == GenerationChunkKindV3::Records {
            self.manifest.total_records += 1;
        }
        Ok(())
    }
    fn flush_chunk(&mut self, kind: GenerationChunkKindV3) -> Result<(), ArenaError> {
        if self.active.contains_key(&kind) {
            check(
                self.manifest.chunks.len() < 16_384,
                "generation chunk count quota before finalizing file",
            )?;
        }
        if let Some(mut chunk) = self.active.remove(&kind) {
            if let Err(error) = chunk.file.flush().and_then(|_| chunk.file.sync_all()) {
                self.poisoned = true;
                return Err(io(error));
            }
            drop(chunk.file);
            let sha = format!("{:x}", chunk.hash.finalize());
            let pin = GenerationChunkPinV3 {
                index: self.manifest.chunks.len() as u32,
                kind: chunk.kind,
                path: chunk.name,
                bytes: chunk.bytes,
                rows: chunk.rows,
                sha256: sha,
            };
            if let Err(error) = verify_chunk(&self.directory, &pin) {
                self.poisoned = true;
                return Err(error);
            }
            self.manifest.total_bytes += pin.bytes;
            self.manifest.chunks.push(pin);
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<GenerationChunkManifestV3, ArenaError> {
        check(
            !self.poisoned && !self.sealed,
            "generation writer failed/sealed",
        )?;
        for kind in self.active.keys().copied().collect::<Vec<_>>() {
            self.flush_chunk(kind)?;
        }
        check(
            !self.manifest.producer.is_null() && !self.manifest.opening_splits.is_empty(),
            "actual producer/split authority not published",
        )?;
        self.manifest.validate()?;
        let mut value = serde_json::to_value(&self.manifest).map_err(|e| invalid(e.to_string()))?;
        value.sort_all_objects();
        let mut bytes = serde_json::to_vec(&value).map_err(|e| invalid(e.to_string()))?;
        bytes.push(b'\n');
        check(
            self.manifest.total_bytes + bytes.len() as u64 + 1024 <= self.maximum_output_bytes,
            "generation final manifest quota",
        )?;
        self.write_manifest_bytes(&bytes, "manifest")?;
        self.sealed = true;
        Ok(self.manifest.clone())
    }
    /// Failure evidence is durable under a distinct, non-training manifest.
    /// It is never exposed as a completed generation by changing a pass flag.
    pub fn finish_partial(
        &mut self,
        reason: &str,
    ) -> Result<GenerationPartialArchiveV3, ArenaError> {
        check(
            !self.poisoned && !self.sealed && !reason.is_empty() && reason.len() <= 4096,
            "partial archive failed/sealed/reason bound",
        )?;
        for kind in self.active.keys().copied().collect::<Vec<_>>() {
            self.flush_chunk(kind)?;
        }
        check(
            !self.manifest.producer.is_null() && !self.manifest.opening_splits.is_empty(),
            "partial source pins not published",
        )?;
        self.manifest.validate()?;
        let mut value = serde_json::to_value(&self.manifest).map_err(|e| invalid(e.to_string()))?;
        value["schema"] =
            serde_json::Value::String("rz-pals-generation-partial-chunk-manifest/1".into());
        value["source_kind"] = serde_json::Value::String("actual_selfplay_partial".into());
        value["incomplete_reason"] = serde_json::Value::String(reason.into());
        value.sort_all_objects();
        let mut bytes = serde_json::to_vec(&value).map_err(|e| invalid(e.to_string()))?;
        bytes.push(b'\n');
        check(
            self.manifest.total_bytes + bytes.len() as u64 + 1024 <= self.maximum_output_bytes,
            "partial archive manifest quota",
        )?;
        let sha = self.write_manifest_bytes(&bytes, "manifest.partial")?;
        self.sealed = true;
        Ok(GenerationPartialArchiveV3 {
            directory: self.directory.clone(),
            manifest_path: self.directory.join("manifest.partial.json"),
            manifest_sha256: sha,
            chunks: self.manifest.chunks.clone(),
            reason: reason.into(),
        })
    }
    fn write_manifest_bytes(&self, bytes: &[u8], stem: &str) -> Result<String, ArenaError> {
        let temporary = self.directory.join(format!("{stem}.pending.json"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io)?;
        file.write_all(bytes).map_err(io)?;
        file.sync_all().map_err(io)?;
        drop(file);
        let path = self.directory.join(format!("{stem}.json"));
        fs::rename(temporary, &path).map_err(io)?;
        let metadata = fs::symlink_metadata(&path).map_err(io)?;
        check(
            metadata.is_file() && !is_link(&metadata) && metadata.len() == bytes.len() as u64,
            "final actual manifest type/size",
        )?;
        let mut actual = vec![];
        File::open(&path)
            .map_err(io)?
            .take(bytes.len() as u64 + 1)
            .read_to_end(&mut actual)
            .map_err(io)?;
        check(actual == bytes, "actual manifest readback differs")?;
        Ok(format!("{:x}", Sha256::digest(&actual)))
    }
    /// Verification reads actual recovered bytes. A remote URI, success flag or
    /// declared pin does not permit cleanup. No automatic deletion is provided.
    pub fn observe_recovered_chunk(
        &mut self,
        pin: &GenerationChunkPinV3,
        recovered: &Path,
    ) -> Result<(), ArenaError> {
        check(
            self.sealed && self.manifest.chunks.contains(pin),
            "recovery requires sealed owned chunk",
        )?;
        check(
            recovered.is_absolute()
                && !fs::canonicalize(recovered)
                    .map_err(io)?
                    .starts_with(&self.directory),
            "recovery must be a separate verified copy, not the original output",
        )?;
        verify_chunk_file(recovered, pin)?;
        self.recovery.insert(pin.path.clone());
        Ok(())
    }
    pub fn all_chunks_recovered(&self) -> bool {
        self.sealed && self.recovery.len() == self.manifest.chunks.len()
    }
}

#[derive(Clone, Debug)]
pub struct GenerationPartialArchiveV3 {
    pub directory: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub chunks: Vec<GenerationChunkPinV3>,
    pub reason: String,
}
pub fn verify_chunk(directory: &Path, pin: &GenerationChunkPinV3) -> Result<(), ArenaError> {
    check(
        directory.is_absolute() && safe_name(&pin.path),
        "chunk read path",
    )?;
    verify_chunk_file(&directory.join(&pin.path), pin)
}
fn verify_chunk_file(path: &Path, pin: &GenerationChunkPinV3) -> Result<(), ArenaError> {
    let metadata = fs::symlink_metadata(path).map_err(io)?;
    check(
        metadata.is_file()
            && !is_link(&metadata)
            && metadata.len() == pin.bytes
            && pin.bytes <= GENERATION_CHUNK_BYTES,
        "chunk actual file type/size",
    )?;
    let file = File::open(path).map_err(io)?;
    let mut reader = BufReader::new(file);
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut rows = 0u32;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = reader
            .by_ref()
            .take((GENERATION_MAX_ROW_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .map_err(io)?;
        if read == 0 {
            break;
        }
        check(
            line.len() <= GENERATION_MAX_ROW_BYTES && line.last() == Some(&b'\n'),
            "bounded complete generation JSONL row",
        )?;
        total += read as u64;
        rows += 1;
        check(
            total <= pin.bytes && rows <= GENERATION_CHUNK_ROWS,
            "chunk actual byte/row cap",
        )?;
        serde_json::from_slice::<serde_json::Value>(&line).map_err(|e| invalid(e.to_string()))?;
        hash.update(&line);
    }
    check(
        total == pin.bytes && rows == pin.rows && format!("{:x}", hash.finalize()) == pin.sha256,
        "chunk readback digest/row count",
    )
}
