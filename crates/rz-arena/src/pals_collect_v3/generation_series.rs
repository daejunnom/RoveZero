//! Ordered whole-game archive aggregation. Only opaque verified OS-game owners
//! enter this producer. The final index references their retained original
//! chunks; it neither reconstructs search authority nor runs inference.

use super::*;
#[cfg(target_os = "linux")]
use sha2::{Digest as _, Sha256};
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub const GENERATION_SERIES_INDEX_V3: &str = "rz-pals-whole-game-series-index/1";
pub const GENERATION_SERIES_INDEX_FILE_V3: &str = "generation-series-index.json";
pub const GENERATION_SERIES_PARTIAL_FILE_V3: &str = "generation-series-partial-index.json";
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EVIDENCE: usize = 1_048_576;
const MAX_CHUNKS: usize = 16_384;

#[derive(Clone, Copy, Debug)]
pub struct GenerationSeriesLimitsV3 {
    pub max_output_bytes: u64,
    pub max_index_bytes: u64,
    pub max_read_bytes: u64,
}
impl GenerationSeriesLimitsV3 {
    fn validate(&self, plan: &SelfplayCollectionPlanV3) -> Result<(), ArenaError> {
        check(
            self.max_output_bytes > 0
                && self.max_output_bytes <= plan.maximum_output_bytes
                && self.max_output_bytes <= GENERATION_OUTPUT_CAP
                && self.max_index_bytes > 0
                && self.max_index_bytes <= MAX_INDEX_BYTES
                && self.max_read_bytes > 0
                && self.max_read_bytes <= 128 * 4 * GENERATION_OUTPUT_CAP,
            "whole-game series output/index/read bounds",
        )
    }
}

/// The publication deadline is fixed at construction. Earlier child execution
/// and closure receipts retain their original, already verified ceilings.
pub struct GenerationSeriesCollectorV3 {
    plan: SelfplayCollectionPlanV3,
    root_path: PathBuf,
    // Retain the original directory owner in the public collector type; only
    // Linux can construct the aggregation capability and inspect this FD.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    root_file: File,
    #[cfg(target_os = "linux")]
    root: cap_std::fs::Dir,
    limits: GenerationSeriesLimitsV3,
    publication_until: Instant,
    accepted_games: usize,
    archives: Vec<Arc<VerifiedGenerationGameArchiveV3>>,
    child_names: BTreeSet<String>,
    record_seals: BTreeSet<[u8; 32]>,
    evidence_seals: BTreeSet<[u8; 32]>,
    run_pins_sha256: Option<String>,
    total_bytes: u64,
    total_artifact_bytes: u64,
    total_records: u64,
    total_evidence_rows: u64,
    total_chunks: usize,
    halted: bool,
    failure: Option<Box<GenerationGameProcessFailureV3>>,
    admission_error: Option<String>,
    // Keep failed publication FD custody with its collector. Windows refuses
    // construction rather than exposing another diagnostic publication path.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    diagnostic_files: Vec<File>,
    diagnostic_reserved_bytes: u64,
    partial_publication_ceiling: Option<u64>,
    publication_read_bytes: u64,
}

impl GenerationSeriesCollectorV3 {
    pub fn new(
        plan: SelfplayCollectionPlanV3,
        run_root_path: PathBuf,
        held_run_root: File,
        limits: GenerationSeriesLimitsV3,
        publication_until: Instant,
    ) -> Result<Self, ArenaError> {
        plan.validate()?;
        limits.validate(&plan)?;
        check(
            Instant::now() < publication_until && run_root_path.is_absolute(),
            "whole-game series original publication/root scope",
        )?;
        #[cfg(not(target_os = "linux"))]
        {
            let _ = held_run_root;
            Err(invalid(
                "whole-game retained archive aggregation requires Linux",
            ))
        }
        #[cfg(target_os = "linux")]
        {
            let root = {
                let root = open_root(&run_root_path)?;
                verify_root(&held_run_root, &root)?;
                for name in [
                    GENERATION_SERIES_INDEX_FILE_V3,
                    GENERATION_SERIES_PARTIAL_FILE_V3,
                ] {
                    check(
                        root.symlink_metadata(name)
                            .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
                        "whole-game series publication names already occupied",
                    )?;
                }
                root
            };
            Ok(Self {
                plan,
                root_path: run_root_path,
                root_file: held_run_root,
                root,
                limits,
                publication_until,
                accepted_games: 0,
                archives: vec![],
                child_names: BTreeSet::new(),
                record_seals: BTreeSet::new(),
                evidence_seals: BTreeSet::new(),
                run_pins_sha256: None,
                total_bytes: 0,
                total_artifact_bytes: 0,
                total_records: 0,
                total_evidence_rows: 0,
                total_chunks: 0,
                halted: false,
                failure: None,
                admission_error: None,
                diagnostic_files: vec![],
                diagnostic_reserved_bytes: 0,
                partial_publication_ceiling: None,
                publication_read_bytes: 0,
            })
        }
    }
    pub fn accepted_games(&self) -> usize {
        self.accepted_games
    }
    pub fn halted(&self) -> bool {
        self.halted
    }
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }
    pub fn halt_with_admission_error(&mut self, error: String) {
        self.halted = true;
        self.admission_error.get_or_insert(error);
    }

    /// Restrict subsequent diagnostic writes to the caller's remaining byte
    /// quota. This does not verify an observed tree size or issue completion
    /// authority. Repeated calls can only narrow the original quota.
    pub fn reserve_partial_publication_capacity_v3(
        &mut self,
        max_additional_bytes: u64,
    ) -> Result<(), ArenaError> {
        self.halted = true;
        check(
            max_additional_bytes <= self.limits.max_output_bytes,
            "whole-game partial additional publication quota",
        )?;
        let ceiling = self
            .diagnostic_reserved_bytes
            .checked_add(max_additional_bytes)
            .ok_or_else(|| invalid("whole-game partial publication quota overflow"))?;
        self.partial_publication_ceiling = Some(
            self.partial_publication_ceiling
                .map_or(ceiling, |original| original.min(ceiling)),
        );
        Ok(())
    }

    /// On an admission error the original compact archive remains in this
    /// collector, including its OS owner and every original artifact handle.
    /// The caller stops launching games and preserves this collector for its
    /// partial receipt or owning finish error.
    pub fn accept(
        &mut self,
        game: VerifiedGenerationGameProcessV3,
        cancel: Option<&AtomicBool>,
    ) -> Result<(), ArenaError> {
        let archive = game.into_archive();
        // The actual process series emits at most 128 owners. Retain one extra
        // misuse observation without permitting unbounded continued intake.
        if self.archives.len() > 128 {
            self.halted = true;
            return Err(invalid("whole-game series intake already stopped"));
        }
        // Count every retained owner, including an archive rejected during
        // admission. Its original files still exist and consume staging.
        self.total_artifact_bytes = archive_output_bytes(&archive)
            .ok()
            .and_then(|bytes| self.total_artifact_bytes.checked_add(bytes))
            .unwrap_or(u64::MAX);
        self.archives.push(Arc::new(archive));
        let result = self.admit_last(cancel);
        if let Err(error) = &result {
            self.halted = true;
            self.admission_error
                .get_or_insert_with(|| error.to_string());
        }
        result
    }

    fn admit_last(&mut self, cancel: Option<&AtomicBool>) -> Result<(), ArenaError> {
        self.live(cancel)?;
        check(
            !self.halted
                && self.accepted_games < 128
                && self.archives.len() == self.accepted_games + 1,
            "whole-game series halted or unexpected owner intake",
        )?;
        let archive = self.archives.last().expect("retained incoming archive");
        let request = archive.request();
        request.validate()?;
        let assignment = request.assignment()?;
        let expected = self
            .plan
            .games()?
            .into_iter()
            .nth(self.accepted_games)
            .ok_or_else(|| invalid("whole-game series assigned ordinal absent"))?;
        check(
            request.game_ordinal as usize == self.accepted_games
                && serde_json::to_value(&request.plan).map_err(|e| invalid(e.to_string()))?
                    == serde_json::to_value(&self.plan).map_err(|e| invalid(e.to_string()))?
                && serde_json::to_value(&assignment).map_err(|e| invalid(e.to_string()))?
                    == serde_json::to_value(&expected).map_err(|e| invalid(e.to_string()))?,
            "whole-game series original ordered plan/assignment",
        )?;
        let pins = run_pins(request)?;
        check(
            self.run_pins_sha256
                .as_ref()
                .is_none_or(|existing| existing == &pins),
            "whole-game series changed phase/model/runtime/source/clock/config",
        )?;
        let child = child_name(&self.root_path, archive.cwd_path())?;
        check(
            !self.child_names.contains(&child),
            "whole-game series reused child cwd owner",
        )?;
        check(
            archive.process().transport_complete(),
            "whole-game original OS closure lost",
        )?;
        archive.process().checked_loaded_image()?;
        archive.process().checked_timing()?;
        archive.manifest().validate()?;
        let summary = archive.summary();
        let pgn = summary.pgn_evidence();
        check(
            pgn["game_id"] == assignment.game_id
                && pgn["opening_id"] == assignment.opening_id
                && pgn["engine_a_white"] == assignment.engine_a_white
                && generation_digest("rz-pals-generation-game-pgn-evidence/1", pgn)?
                    == summary.pgn_receipt_sha256(),
            "whole-game series original PGN identity/seal",
        )?;
        verify_result(
            summary.result(),
            summary.termination(),
            summary.rules_terminated(),
        )?;
        check(
            archive.manifest().total_records == archive.record_seals().len() as u64,
            "whole-game compact actual record membership",
        )?;
        let bytes = self
            .total_bytes
            .checked_add(archive.manifest().total_bytes)
            .ok_or_else(|| invalid("whole-game series byte accounting overflow"))?;
        let artifact_bytes = self.total_artifact_bytes;
        let evidence_rows = archive
            .manifest()
            .chunks
            .iter()
            .filter(|p| p.kind == GenerationChunkKindV3::Evidence)
            .try_fold(self.total_evidence_rows, |total, pin| {
                total.checked_add(u64::from(pin.rows))
            })
            .ok_or_else(|| invalid("whole-game evidence occurrence accounting overflow"))?;
        let records = self
            .total_records
            .checked_add(archive.manifest().total_records)
            .ok_or_else(|| invalid("whole-game series record accounting overflow"))?;
        let chunks = self
            .total_chunks
            .checked_add(archive.manifest().chunks.len())
            .ok_or_else(|| invalid("whole-game series chunk accounting overflow"))?;
        check(
            artifact_bytes <= self.limits.max_output_bytes
                && evidence_rows <= MAX_EVIDENCE as u64
                && records <= GENERATION_MAX_RECORDS as u64
                && chunks <= MAX_CHUNKS,
            "whole-game series corpus capacity",
        )?;
        for seal in archive.record_seals() {
            self.live(cancel)?;
            check(
                self.record_seals.len() < GENERATION_MAX_RECORDS && self.record_seals.insert(*seal),
                "whole-game series duplicate/over-limit original record",
            )?;
        }
        for seal in archive.evidence_seals() {
            self.live(cancel)?;
            // Every original row remains pinned in its child chunk. Identical
            // content-domain digests share a definition, never an occurrence.
            if !self.evidence_seals.contains(seal) {
                check(
                    self.evidence_seals.len() < MAX_EVIDENCE,
                    "whole-game series distinct evidence capacity",
                )?;
                self.evidence_seals.insert(*seal);
            }
        }
        self.run_pins_sha256.get_or_insert(pins);
        self.child_names.insert(child);
        self.total_bytes = bytes;
        self.total_records = records;
        self.total_evidence_rows = evidence_rows;
        self.total_chunks = chunks;
        self.accepted_games += 1;
        Ok(())
    }

    /// Preserve the actual failed child owner. No later ordinal can make a
    /// stopped run into formal 128-game completion.
    pub fn halt_with_process_failure(
        &mut self,
        failure: Box<GenerationGameProcessFailureV3>,
    ) -> Result<(), Box<GenerationGameProcessFailureV3>> {
        self.halted = true;
        if self.failure.is_some() {
            return Err(failure);
        }
        self.failure = Some(failure);
        Ok(())
    }

    fn live(&self, cancel: Option<&AtomicBool>) -> Result<(), ArenaError> {
        check(
            Instant::now() < self.publication_until
                && !cancel.is_some_and(|flag| flag.load(Ordering::Acquire)),
            "whole-game original series publication deadline/cancellation",
        )
    }

    /// Metadata only. This never grants a finished-run or native authority.
    pub fn partial_receipt(&self) -> Result<serde_json::Value, ArenaError> {
        let mut games = Vec::with_capacity(self.archives.len());
        for (index, archive) in self.archives.iter().enumerate() {
            games.push(
                match game_index(&self.root_path, archive, index < self.accepted_games) {
                    Ok(value) => value,
                    Err(error) => serde_json::json!({"game_ordinal":archive.request().game_ordinal,
                    "accepted_into_series":false,"diagnostic_error":error.to_string(),
                    "original_process_receipt":archive.receipt(),
                    "pgn_receipt_sha256":archive.summary().pgn_receipt_sha256()}),
                },
            );
        }
        Ok(
            serde_json::json!({"schema":"rz-pals-whole-game-series-partial/1",
            "run_id":self.plan.run_id,"accepted_games":self.accepted_games,
            "required_games":128,"halted":self.halted,"games":games,
            "admission_error":self.admission_error,
            "process_failure":self.failure.as_ref().map(|failure|serde_json::json!({
                "error":failure.error.to_string(),"request":failure.request(),
                "original_process":failure.process().map(|p|&p.process().receipt)})),
            "assurance_scope":"incomplete_original_owners_preserved; no_128game_or_training_acceptance"}),
        )
    }

    /// Cancellation halts execution but does not discard original failure
    /// bytes. This bounded preservation uses the initial publication deadline
    /// and never grants full-series or native completion authority.
    pub fn publish_partial(&mut self) -> Result<GenerationSeriesPartialPublicationV3, ArenaError> {
        self.halted = true;
        self.live(None)?;
        let mut read_left = self
            .limits
            .max_read_bytes
            .checked_sub(self.publication_read_bytes)
            .ok_or_else(|| invalid("whole-game partial publication read credit"))?;
        let mut extra_bytes = 0u64;
        let mut pins = Vec::new();
        let failure = self.failure.take();
        let result: Result<(), ArenaError> = (|| {
            if let Some(process) = failure.as_ref().and_then(|f| f.process()) {
                for (name, bytes) in [
                    (
                        "generation-series-failed-child-stdout.bin",
                        process.process().stdout.as_slice(),
                    ),
                    (
                        "generation-series-failed-child-stderr.bin",
                        process.process().stderr.as_slice(),
                    ),
                ] {
                    self.live(None)?;
                    extra_bytes = extra_bytes
                        .checked_add(bytes.len() as u64)
                        .ok_or_else(|| invalid("whole-game failed output byte overflow"))?;
                    check(
                        self.total_artifact_bytes
                            .checked_add(extra_bytes)
                            .is_some_and(|n| n <= self.limits.max_output_bytes),
                        "whole-game series original failed output retention budget",
                    )?;
                    let sha = self.write_pinned(name, bytes, None, &mut read_left)?;
                    pins.push(serde_json::json!({"file":name,"sha256":sha,"bytes":bytes.len()}));
                }
            }
            Ok(())
        })();
        self.failure = failure;
        result?;
        let mut value = self.partial_receipt()?;
        value["original_failed_output_files"] = serde_json::json!(pins);
        let mut canonical = value.clone();
        canonical.sort_all_objects();
        let mut bytes = serde_json::to_vec(&canonical).map_err(|e| invalid(e.to_string()))?;
        bytes.push(b'\n');
        check(
            bytes.len() as u64 <= self.limits.max_index_bytes
                && self
                    .total_artifact_bytes
                    .checked_add(extra_bytes)
                    .and_then(|n| n.checked_add(bytes.len() as u64))
                    .is_some_and(|n| n <= self.limits.max_output_bytes),
            "whole-game partial raw/index total publication bound",
        )?;
        let sha = self.write_pinned(
            GENERATION_SERIES_PARTIAL_FILE_V3,
            &bytes,
            None,
            &mut read_left,
        )?;
        self.publication_read_bytes = self.limits.max_read_bytes - read_left;
        Ok(GenerationSeriesPartialPublicationV3 {
            path: self.root_path.join(GENERATION_SERIES_PARTIAL_FILE_V3),
            file_sha256: sha,
        })
    }

    pub fn finish(
        mut self,
        cancel: Option<&AtomicBool>,
    ) -> Result<GenerationVerifiedSeriesArchiveV3, Box<GenerationSeriesFailureV3>> {
        let result = self.finish_index(cancel);
        match result {
            Ok((index, dataset_manifest, dataset_sha256, file_sha256)) => {
                Ok(GenerationVerifiedSeriesArchiveV3 {
                    owner: self,
                    index,
                    dataset_manifest,
                    dataset_sha256,
                    file_sha256,
                })
            }
            Err(error) => {
                self.halted = true;
                Err(Box::new(GenerationSeriesFailureV3 { error, owner: self }))
            }
        }
    }

    fn finish_index(
        &mut self,
        cancel: Option<&AtomicBool>,
    ) -> Result<(serde_json::Value, serde_json::Value, String, String), ArenaError> {
        self.live(cancel)?;
        check(
            !self.halted
                && self.failure.is_none()
                && self.accepted_games == 128
                && self.archives.len() == 128
                && self.child_names.len() == 128,
            "whole-game series requires 128 actual closed ordered game owners",
        )?;
        let assignments = self.plan.games()?;
        let splits = self.plan.opening_splits();
        let mut pairs = BTreeSet::new();
        let mut counts = BTreeMap::new();
        for pair in assignments.chunks_exact(2) {
            self.live(cancel)?;
            check(
                pair[0].pair == pair[1].pair
                    && pair[0].opening_id == pair[1].opening_id
                    && pair[0].engine_a_white
                    && !pair[1].engine_a_white
                    && pair[0].split == pair[1].split
                    && pairs.insert(pair[0].pair),
                "whole-game series exact exchanged-color opening pair",
            )?;
            *counts.entry(pair[0].split).or_insert(0usize) += 1;
        }
        check(
            pairs.len() == 64
                && counts.get(&GenerationSplitV3::Train) == Some(&48)
                && counts.get(&GenerationSplitV3::Validation) == Some(&8)
                && counts.get(&GenerationSplitV3::Holdout) == Some(&8),
            "whole-game series registered 64 opening / 48-8-8 split completion",
        )?;
        #[cfg(target_os = "linux")]
        {
            verify_root(&self.root_file, &open_root(&self.root_path)?)?;
        }
        let mut read_bytes = 0u64;
        // Keep one bounded index readback for the final attempt and another
        // for partial diagnostics, including a failed final attempt. These
        // credits are never offered to an archive whose failed readback has
        // no exact consumption report.
        let archive_read_ceiling = self
            .limits
            .max_read_bytes
            .saturating_sub(self.limits.max_index_bytes.saturating_mul(2));
        let mut chunks = Vec::with_capacity(self.total_chunks);
        let mut games = Vec::with_capacity(128);
        for archive in &self.archives {
            self.live(cancel)?;
            let remaining = archive_read_ceiling
                .checked_sub(read_bytes)
                .ok_or_else(|| invalid("whole-game series read credit exhausted"))?;
            let read =
                match archive.reverify_owned_artifacts(self.publication_until, cancel, remaining) {
                    Ok(read) => read,
                    Err(error) => {
                        // An unsuccessful member readback has no public exact
                        // consumption report. Keep its offered credit consumed;
                        // later diagnostics cannot mint a second read allowance.
                        self.publication_read_bytes = archive_read_ceiling;
                        return Err(error);
                    }
                };
            read_bytes = read_bytes
                .checked_add(read)
                .ok_or_else(|| invalid("whole-game series read accounting overflow"))?;
            check(
                read_bytes <= archive_read_ceiling,
                "whole-game series declared revalidation read budget",
            )?;
            self.publication_read_bytes = read_bytes;
            let child = child_name(&self.root_path, archive.cwd_path())?;
            for pin in &archive.manifest().chunks {
                self.live(cancel)?;
                let mut pin = pin.clone();
                pin.index = u32::try_from(chunks.len())
                    .map_err(|_| invalid("whole-game series chunk index overflow"))?;
                pin.path = format!(
                    "{}/{}/{}",
                    child,
                    archive.request().archive_directory,
                    pin.path
                );
                chunks.push(pin);
            }
            games.push(game_index(&self.root_path, archive, true)?);
        }
        let request = self.archives[0].request();
        let mut producer =
            serde_json::to_value(&request.producer).map_err(|e| invalid(e.to_string()))?;
        let producer_fields = producer
            .as_object_mut()
            .ok_or_else(|| invalid("whole-game series producer fields"))?;
        producer_fields.insert(
            "schema".into(),
            serde_json::json!("rz-pals-actual-collection-authority/2"),
        );
        producer_fields.remove("input_seals");
        producer_fields.insert(
            "actual_game_ids".into(),
            serde_json::json!(assignments.iter().map(|a| &a.game_id).collect::<Vec<_>>()),
        );
        let dataset_manifest = serde_json::json!({"schema":GENERATION_CHUNK_DOMAIN,"source_kind":"actual_selfplay",
            "producer":producer,"opening_splits":splits,"chunks":chunks,
            "total_bytes":self.total_bytes,"total_records":self.total_records});
        let dataset_sha256 = generation_digest(GENERATION_CHUNK_DOMAIN, &dataset_manifest)?;
        let index = serde_json::json!({"schema":GENERATION_SERIES_INDEX_V3,"run_id":self.plan.run_id,
            "producer_phase":request.producer_phase,"run_pins_sha256":self.run_pins_sha256,
            "plan":self.plan,"games":games,"dataset_manifest":dataset_manifest,
            "dataset_manifest_sha256":dataset_sha256,"distinct_input_seals":self.record_seals.len(),
            "distinct_evidence_seals":self.evidence_seals.len(),"original_evidence_occurrences":self.total_evidence_rows,
            "original_game_artifact_bytes":self.total_artifact_bytes,"original_archive_revalidation_read_bytes":read_bytes,
            "assurance_scope":"128_original_OS_game_owners_64_exchanged_color_pairs_raw_artifact_index_only; no_training_acceptance"});
        let mut remaining = self
            .limits
            .max_read_bytes
            .checked_sub(read_bytes)
            .ok_or_else(|| invalid("whole-game series index read credit"))?;
        let file_sha256 = self.publish(
            GENERATION_SERIES_INDEX_FILE_V3,
            &index,
            cancel,
            &mut remaining,
        )?;
        self.publication_read_bytes = self.limits.max_read_bytes - remaining;
        self.live(cancel)?;
        Ok((index, dataset_manifest, dataset_sha256, file_sha256))
    }

    fn publish(
        &mut self,
        name: &str,
        value: &serde_json::Value,
        cancel: Option<&AtomicBool>,
        read_left: &mut u64,
    ) -> Result<String, ArenaError> {
        self.live(cancel)?;
        let mut canonical = value.clone();
        canonical.sort_all_objects();
        let mut bytes = serde_json::to_vec(&canonical).map_err(|e| invalid(e.to_string()))?;
        bytes.push(b'\n');
        check(
            bytes.len() as u64 <= self.limits.max_index_bytes
                && self
                    .total_artifact_bytes
                    .checked_add(bytes.len() as u64)
                    .is_some_and(|n| n <= self.limits.max_output_bytes),
            "whole-game series bounded total corpus/index publication",
        )?;
        self.write_pinned(name, &bytes, cancel, read_left)
    }
    #[cfg(target_os = "linux")]
    fn write_pinned(
        &mut self,
        name: &str,
        bytes: &[u8],
        cancel: Option<&AtomicBool>,
        read_left: &mut u64,
    ) -> Result<String, ArenaError> {
        use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
        use cap_std::fs::OpenOptionsExt;
        use std::io::Write;
        use std::os::unix::fs::{FileExt, MetadataExt};
        self.live(cancel)?;
        check(
            bytes.len() as u64 <= *read_left,
            "whole-game series original publication read credit",
        )?;
        let reserved_bytes = self
            .diagnostic_reserved_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("whole-game retained publication byte overflow"))?;
        check(
            self.total_artifact_bytes
                .checked_add(reserved_bytes)
                .is_some_and(|total| total <= self.limits.max_output_bytes),
            "whole-game original archives and all publication files exceed staging",
        )?;
        check(
            self.partial_publication_ceiling
                .is_none_or(|ceiling| reserved_bytes <= ceiling),
            "whole-game partial caller-restricted publication quota exhausted",
        )?;
        verify_root(&self.root_file, &open_root(&self.root_path)?)?;
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
        self.diagnostic_files.push(
            self.root
                .open_with(name, &options)
                .map_err(|_| invalid("whole-game series exclusive index publication"))?
                .into_std(),
        );
        // Reserve the full intended length before any write. A short/failed
        // write retains this conservative charge and the original file owner.
        self.diagnostic_reserved_bytes = reserved_bytes;
        let file = self
            .diagnostic_files
            .last_mut()
            .expect("retained series publication file");
        for chunk in bytes.chunks(65_536) {
            check(
                Instant::now() < self.publication_until
                    && !cancel.is_some_and(|c| c.load(Ordering::Acquire)),
                "whole-game series original publication write deadline",
            )?;
            file.write_all(chunk)
                .map_err(|_| invalid("whole-game series publication write"))?;
        }
        file.sync_all()
            .map_err(|_| invalid("whole-game series durable index publication"))?;
        let expected = format!("{:x}", Sha256::digest(bytes));
        let before = file
            .metadata()
            .map_err(|_| invalid("whole-game series index metadata"))?;
        check(
            before.is_file() && before.nlink() == 1 && before.len() == bytes.len() as u64,
            "whole-game series original index file identity",
        )?;
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        let mut buffer = [0u8; 65_536];
        while offset < bytes.len() as u64 {
            check(
                Instant::now() < self.publication_until
                    && !cancel.is_some_and(|c| c.load(Ordering::Acquire)),
                "whole-game series index original readback deadline",
            )?;
            let remaining_bytes = bytes.len() as u64 - offset;
            let take = usize::try_from(remaining_bytes.min(buffer.len() as u64))
                .map_err(|_| invalid("whole-game publication read window"))?;
            let count = file
                .read_at(&mut buffer[..take], offset)
                .map_err(|_| invalid("whole-game series index readback"))?;
            if count == 0 {
                break;
            }
            *read_left = read_left
                .checked_sub(count as u64)
                .ok_or_else(|| invalid("whole-game series publication read budget exhausted"))?;
            self.publication_read_bytes = self.limits.max_read_bytes - *read_left;
            offset += count as u64;
            check(
                offset <= bytes.len() as u64,
                "whole-game series index grew during readback",
            )?;
            hash.update(&buffer[..count]);
        }
        let after = file
            .metadata()
            .map_err(|_| invalid("whole-game series index final metadata"))?;
        check(
            format!("{:x}", hash.finalize()) == expected
                && offset == bytes.len() as u64
                && before.dev() == after.dev()
                && before.ino() == after.ino()
                && before.mtime() == after.mtime()
                && before.mtime_nsec() == after.mtime_nsec()
                && before.ctime() == after.ctime()
                && before.ctime_nsec() == after.ctime_nsec()
                && after.nlink() == 1
                && after.len() == before.len(),
            "whole-game series original index bytes changed",
        )?;
        check(
            self.root
                .symlink_metadata(name)
                .is_ok_and(|m| m.file_type().is_file()),
            "whole-game original publication name replaced/linked",
        )?;
        let mut read_options = cap_std::fs::OpenOptions::new();
        read_options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
        let named = self
            .root
            .open_with(name, &read_options)
            .map_err(|_| invalid("whole-game original publication name missing"))?
            .into_std()
            .metadata()
            .map_err(|_| invalid("whole-game named publication metadata"))?;
        check(
            named.is_file()
                && named.dev() == after.dev()
                && named.ino() == after.ino()
                && named.len() == after.len()
                && named.nlink() == 1,
            "whole-game original publication pathname/held inode mismatch",
        )?;
        verify_root(&self.root_file, &open_root(&self.root_path)?)?;
        self.live(cancel)?;
        Ok(expected)
    }
    #[cfg(not(target_os = "linux"))]
    fn write_pinned(
        &mut self,
        _name: &str,
        _bytes: &[u8],
        _cancel: Option<&AtomicBool>,
        _read_left: &mut u64,
    ) -> Result<String, ArenaError> {
        Err(invalid(
            "whole-game series index requires Linux pinned files",
        ))
    }
}

/// Finish failures retain the collector, failed process and any partial index
/// file. Deserialization can never create this owner or completed-run evidence.
pub struct GenerationSeriesFailureV3 {
    pub error: ArenaError,
    owner: GenerationSeriesCollectorV3,
}
impl GenerationSeriesFailureV3 {
    pub fn collector(&self) -> &GenerationSeriesCollectorV3 {
        &self.owner
    }
    pub fn collector_mut(&mut self) -> &mut GenerationSeriesCollectorV3 {
        &mut self.owner
    }
    pub fn into_collector(self) -> GenerationSeriesCollectorV3 {
        self.owner
    }
    pub fn publish_partial(&mut self) -> Result<GenerationSeriesPartialPublicationV3, ArenaError> {
        self.owner.publish_partial()
    }
    pub fn partial_receipt(&self) -> Result<serde_json::Value, ArenaError> {
        self.owner.partial_receipt()
    }
}

/// A saved diagnostic file, without completed-run authority.
pub struct GenerationSeriesPartialPublicationV3 {
    path: PathBuf,
    file_sha256: String,
}
impl GenerationSeriesPartialPublicationV3 {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
}
impl std::fmt::Debug for GenerationSeriesFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationSeriesFailureV3")
            .field("error", &self.error)
            .field("accepted_games", &self.owner.accepted_games)
            .finish_non_exhaustive()
    }
}

pub struct GenerationVerifiedSeriesArchiveV3 {
    owner: GenerationSeriesCollectorV3,
    index: serde_json::Value,
    dataset_manifest: serde_json::Value,
    dataset_sha256: String,
    file_sha256: String,
}
impl GenerationVerifiedSeriesArchiveV3 {
    pub fn index(&self) -> &serde_json::Value {
        &self.index
    }
    pub fn dataset_manifest(&self) -> &serde_json::Value {
        &self.dataset_manifest
    }
    pub fn dataset_manifest_sha256(&self) -> &str {
        &self.dataset_sha256
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub fn verified_read_bytes(&self) -> u64 {
        self.owner.publication_read_bytes
    }
    pub fn path(&self) -> PathBuf {
        self.owner.root_path.join(GENERATION_SERIES_INDEX_FILE_V3)
    }
    pub fn archives(&self) -> &[Arc<VerifiedGenerationGameArchiveV3>] {
        &self.owner.archives
    }
    /// Share the same original OS-game custody while retaining this completed
    /// series for trainer resume. No files or authority are reconstructed.
    pub fn shared_archives(&self) -> Vec<Arc<VerifiedGenerationGameArchiveV3>> {
        self.owner.archives.clone()
    }
    /// Transfer original per-game owners to the saved-root consumer without
    /// rebuilding them from a serialized index or issuing another execution.
    pub fn into_archives(self) -> Vec<Arc<VerifiedGenerationGameArchiveV3>> {
        self.owner.archives
    }
}

fn run_pins(request: &GenerationWholeGameRequestV3) -> Result<String, ArenaError> {
    generation_digest("rz-pals-whole-game-run-pins/1",&serde_json::json!({
        "plan":request.plan,"producer":request.producer,"producer_phase":request.producer_phase,
        "model_epoch":request.model_epoch,"context_epoch":request.context_epoch,"clock":request.clock,
        "startup_configs":request.startup_configs})).map_err(Into::into)
}
fn archive_output_bytes(archive: &VerifiedGenerationGameArchiveV3) -> Result<u64, ArenaError> {
    let manifest_bytes = archive.receipt()["manifest"]["bytes"]
        .as_u64()
        .ok_or_else(|| invalid("whole-game original manifest byte receipt absent"))?;
    archive
        .manifest()
        .total_bytes
        .checked_add(manifest_bytes)
        .and_then(|n| n.checked_add(archive.process().process().receipt.stdout_bytes))
        .ok_or_else(|| invalid("whole-game retained original output byte overflow"))
}
fn child_name(root: &Path, cwd: &Path) -> Result<String, ArenaError> {
    let relative = cwd
        .strip_prefix(root)
        .map_err(|_| invalid("whole-game child outside series root"))?;
    let mut parts = relative.components();
    let Some(Component::Normal(name)) = parts.next() else {
        return Err(invalid("whole-game child direct relative cwd absent"));
    };
    check(
        parts.next().is_none(),
        "whole-game child cwd must be direct series-root child",
    )?;
    let name = name
        .to_str()
        .ok_or_else(|| invalid("whole-game child directory encoding"))?;
    check(
        !name.is_empty()
            && name.len() <= 128
            && !name.starts_with('.')
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_')),
        "whole-game stable child directory name",
    )?;
    Ok(name.to_owned())
}
fn verify_result(
    result: CollectionPgnResultV3,
    termination: CollectionPgnTerminationV3,
    rules: bool,
) -> Result<(), ArenaError> {
    let valid = match termination {
        CollectionPgnTerminationV3::RulesAutomatic | CollectionPgnTerminationV3::RulesClaim(_) => {
            rules && result != CollectionPgnResultV3::Incomplete
        }
        CollectionPgnTerminationV3::TimeForfeit { loser }
        | CollectionPgnTerminationV3::IllegalMove { loser }
        | CollectionPgnTerminationV3::EngineCrash { loser } => {
            !rules
                && result
                    == if loser == Color::White {
                        CollectionPgnResultV3::BlackWin
                    } else {
                        CollectionPgnResultV3::WhiteWin
                    }
        }
        CollectionPgnTerminationV3::PlyLimit => {
            !rules && result == CollectionPgnResultV3::Incomplete
        }
        CollectionPgnTerminationV3::Infrastructure | CollectionPgnTerminationV3::Cancelled => false,
    };
    check(valid, "whole-game series PGN result/forfeit/Rules scope")
}
fn game_index(
    root: &Path,
    archive: &VerifiedGenerationGameArchiveV3,
    accepted: bool,
) -> Result<serde_json::Value, ArenaError> {
    let request = archive.request();
    let summary = archive.summary();
    let result = match summary.result() {
        CollectionPgnResultV3::WhiteWin => "1-0",
        CollectionPgnResultV3::BlackWin => "0-1",
        CollectionPgnResultV3::Draw => "1/2-1/2",
        CollectionPgnResultV3::Incomplete => "*",
    };
    Ok(
        serde_json::json!({"game_ordinal":request.game_ordinal,"assignment":request.assignment()?,
        "child_directory":child_name(root,archive.cwd_path())?,"archive_directory":request.archive_directory,
        "accepted_into_series":accepted,"original_process_receipt":archive.receipt(),
        "original_manifest":archive.manifest(),"pgn_receipt_sha256":summary.pgn_receipt_sha256(),
        "pgn_result":result,"termination":format!("{:?}",summary.termination()),
        "rules_terminated":summary.rules_terminated(),"engine_names":summary.engine_names(),
        "actual_decision_count":summary.decision_count(),"record_count":archive.record_seals().len(),
        "distinct_evidence_count":archive.evidence_seals().len()}),
    )
}

#[cfg(target_os = "linux")]
fn open_root(path: &Path) -> Result<cap_std::fs::Dir, ArenaError> {
    use cap_fs_ext::DirExt;
    check(path.is_absolute(), "whole-game series absolute root")?;
    let mut current = cap_std::fs::Dir::open_ambient_dir("/", cap_std::ambient_authority())
        .map_err(|_| invalid("whole-game series filesystem root pin"))?;
    for part in path.components() {
        match part {
            Component::RootDir => {}
            Component::Normal(name) => {
                current = current
                    .open_dir_nofollow(name)
                    .map_err(|_| invalid("whole-game series nofollow root component"))?
            }
            _ => return Err(invalid("whole-game series normalized root required")),
        }
    }
    Ok(current)
}
#[cfg(target_os = "linux")]
fn verify_root(file: &File, root: &cap_std::fs::Dir) -> Result<(), ArenaError> {
    use std::os::unix::fs::MetadataExt;
    let held = file
        .metadata()
        .map_err(|_| invalid("whole-game series held root metadata"))?;
    let actual = root
        .open(".")
        .map_err(|_| invalid("whole-game series root metadata pin"))?
        .into_std()
        .metadata()
        .map_err(|_| invalid("whole-game series actual root metadata"))?;
    check(
        held.is_dir()
            && actual.is_dir()
            && held.dev() == actual.dev()
            && held.ino() == actual.ino()
            && actual.uid() == nix::unistd::geteuid().as_raw()
            && actual.mode() & 0o777 == 0o700,
        "whole-game series same owned private root",
    )
}
