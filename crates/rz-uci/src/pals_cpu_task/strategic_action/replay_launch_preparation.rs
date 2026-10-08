//! Result-free, caller-bound durable preparation of a frozen CPU Fresh launch.
//!
//! Original action, whole semantic receipt, new replay registration and profile
//! bytes remain distinct. This boundary prepares files; it does not load assets,
//! create providers/models/owners, spawn a process or certify historical callers.

use super::ArtifactPin;
use super::native_replay::{
    AdmissionFault, CpuFreshAssetError, CpuFreshAssetProfile, check_cpu_fresh_asset_profile,
    replay_output_requirements,
};
use super::replay_inputs::{
    PreparedReplayRequest, ReplayAuthorities, ReplayBindingPins, ReplayConstructionBudget,
    ReplayExpectedPins, ReplayInputError, ReplayInputMode, ReplayOriginals, ReplayParentPins,
    ReplayPreparationDeclaration, ReplayPreparationError, ReplayRegisteredArtifacts,
    SemanticReceiptProducerScope, check_replay_inputs_with_semantic_scope, prepare_replay_request,
};
use rz_search::pals::engine::replay::ReplayError;
#[cfg(test)]
use rz_search::pals::engine::replay::repair_replay_requirements;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::TryReserveError;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const EXPECTED_TRANSPORT_V2_SCHEMA: &str = "rz-pals-frozen-replay-cli-expected/2";
pub const LAUNCH_ASSETS_SCHEMA: &str = "rz-pals-frozen-replay-cli-assets/1";
pub const PREPARATION_MANIFEST_SCHEMA: &str = "rz-pals-frozen-replay-launch-preparation/1";
pub const MAX_EXPECTED_BYTES: usize = 64 * 1024;
pub const MAX_LAUNCH_ASSETS_BYTES: usize = 128 * 1024;
pub const MAX_PATH_BYTES: usize = 4096;
pub const MAX_PUBLICATION_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PUBLICATION_BACKING_BYTES: usize = 16 * 1024 * 1024;
const MANIFEST_CAP: usize = 64 * 1024;
const METADATA_CREDIT: usize = 2 * 64 * 1024;
const FILE_COUNT: usize = 8;
const PENDING_MANIFEST: &str = "preparation-manifest.pending.json";
const FINAL_MANIFEST: &str = "preparation-manifest.json";
const FAILED_MANIFEST: &str = "preparation-manifest.failed.json";

/// Closed shared codec of independently supplied expected declarations. This is
/// not a constructor of a checked Query/semantic capability or loaded image.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedPinsWire {
    pub registration_artifact: ArtifactPin,
    pub prepared_action_artifact: ArtifactPin,
    pub semantic_receipt_artifact: ArtifactPin,
    pub parent: ReplayParentPins,
    pub binding: ReplayBindingPins,
    pub registered_artifacts: ReplayRegisteredArtifacts,
    pub legacy_cpu_profile_sha256: String,
    pub provider_factory_id: String,
    pub semantic_binary_sha256: String,
}

impl ExpectedPinsWire {
    pub fn into_expected(self) -> ReplayExpectedPins {
        ReplayExpectedPins {
            registration_artifact: self.registration_artifact,
            prepared_action_artifact: self.prepared_action_artifact,
            semantic_receipt_artifact: self.semantic_receipt_artifact,
            parent: self.parent,
            binding: self.binding,
            registered_artifacts: self.registered_artifacts,
            legacy_cpu_profile_sha256: self.legacy_cpu_profile_sha256,
            provider_factory_id: self.provider_factory_id,
            semantic_binary_sha256: self.semantic_binary_sha256,
        }
    }
    fn from_expected(expected: &ReplayExpectedPins) -> Self {
        Self {
            registration_artifact: expected.registration_artifact.clone(),
            prepared_action_artifact: expected.prepared_action_artifact.clone(),
            semantic_receipt_artifact: expected.semantic_receipt_artifact.clone(),
            parent: expected.parent.clone(),
            binding: expected.binding.clone(),
            registered_artifacts: expected.registered_artifacts.clone(),
            legacy_cpu_profile_sha256: expected.legacy_cpu_profile_sha256.clone(),
            provider_factory_id: expected.provider_factory_id.clone(),
            semantic_binary_sha256: expected.semantic_binary_sha256.clone(),
        }
    }
}

/// Existing CLI /2 wire. Its scope is an independent expectation for the
/// historical semantic producer; replay_binary_pin_scope is a separate domain.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedTransportV2 {
    pub schema: String,
    pub request_artifact: ArtifactPin,
    pub replay_expected: ExpectedPinsWire,
    pub launch_asset_artifact: ArtifactPin,
    pub cpu_fresh_profile_artifact: ArtifactPin,
    pub cpu_fresh_profile: CpuFreshAssetProfile,
    pub whole_wall_ms: u64,
    pub cleanup_reserve_ms: u64,
    pub output_bytes: usize,
    pub replay_binary_pin_scope: String,
    pub expected_semantic_receipt_producer_scope: SemanticReceiptProducerScope,
}

/// Existing asset transport contains descriptor paths and the exact original
/// profile string. No referenced manifest/runtime contents are read here.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchAssets {
    pub schema: String,
    pub export_manifest_path: String,
    pub runtime_library_path: String,
    pub runtime_cache_root: String,
    pub cpu_fresh_profile_raw: String,
}

#[derive(Clone, Copy, Debug)]
pub struct ReplayLaunchOriginals<'a> {
    pub replay: ReplayOriginals<'a>,
    pub cpu_fresh_profile: &'a [u8],
}
#[derive(Clone, Copy, Debug)]
pub struct ReplayLaunchAssetPaths<'a> {
    pub export_manifest: &'a Path,
    pub runtime_library: &'a Path,
    pub runtime_cache_root: &'a Path,
}
#[derive(Clone, Copy, Debug)]
pub struct ReplayLaunchDeclaration<'a> {
    pub replay: ReplayPreparationDeclaration<'a>,
    pub expected_semantic_receipt_producer_scope: SemanticReceiptProducerScope,
    pub assets: ReplayLaunchAssetPaths<'a>,
    /// Existing caller-owned directory. Only a new fixed child is created.
    pub output_root: &'a Path,
    pub bundle_directory_name: &'a str,
    /// Independent expectation for the eventual replay executable, not the
    /// historical semantic scope or an observation made by preparation.
    pub replay_binary_pin_scope: &'a str,
}
#[derive(Clone, Copy, Debug)]
pub struct ReplayLaunchExpected<'a> {
    pub replay: &'a ReplayExpectedPins,
    pub cpu_fresh_profile_artifact: &'a ArtifactPin,
    pub cpu_fresh_profile: &'a CpuFreshAssetProfile,
}
#[derive(Clone, Copy, Debug)]
pub struct ReplayLaunchBudget {
    pub construction: ReplayConstructionBudget,
    /// Cumulative requested write + readback bytes, including manifest maximum.
    pub max_publication_bytes: usize,
    /// Separate policy for copied originals, serialized transports, manifest
    /// backing and bounded metadata. Not native output or allocator/RSS peak.
    pub max_publication_backing_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct LaunchClock {
    started: Instant,
    execution: Instant,
    whole: Instant,
    whole_wall_ms: u64,
    cleanup_reserve_ms: u64,
    output_limit: usize,
}
impl LaunchClock {
    fn declared(started: Instant, declaration: ReplayPreparationDeclaration<'_>) -> Option<Self> {
        let resources = declaration.resources;
        if !(1..=super::super::MAX_WALL_TIME_MS).contains(&resources.whole_wall_ms) {
            return None;
        }
        let whole = started.checked_add(Duration::from_millis(resources.whole_wall_ms))?;
        let output_valid =
            (1024..=super::replay_inputs::MAX_OUTPUT_BYTES).contains(&resources.output_bytes);
        let execution = if output_valid
            && resources.cleanup_reserve_ms > 0
            && resources.cleanup_reserve_ms < resources.whole_wall_ms
        {
            started.checked_add(Duration::from_millis(
                resources.whole_wall_ms - resources.cleanup_reserve_ms,
            ))?
        } else {
            whole
        };
        Some(Self {
            started,
            execution,
            whole,
            whole_wall_ms: resources.whole_wall_ms,
            cleanup_reserve_ms: resources.cleanup_reserve_ms,
            output_limit: if output_valid {
                resources.output_bytes
            } else {
                1024
            },
        })
    }
    fn check(self, cancel: &AtomicBool) -> Result<(), ReplayLaunchCause> {
        if cancel.load(Ordering::Acquire) {
            return Err(ReplayLaunchCause::Refusal("preparation canceled"));
        }
        let now = Instant::now();
        if self.started > now {
            return Err(ReplayLaunchCause::Refusal(
                "original start is in the future",
            ));
        }
        if now >= self.execution {
            return Err(ReplayLaunchCause::Refusal(
                "original execution partition expired",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ReplayLaunchCause {
    Construction(Box<ReplayPreparationError>),
    Input(Box<ReplayInputError>),
    Profile(Box<CpuFreshAssetError>),
    Output(Box<AdmissionFault>),
    Io(io::Error),
    Reserve(TryReserveError),
    Json(serde_json::Error),
    Utf8(std::str::Utf8Error),
    Replay(Box<ReplayError>),
    Integer(std::num::TryFromIntError),
    Cpu(Box<super::super::CpuTaskError>),
    Refusal(&'static str),
}
impl fmt::Display for ReplayLaunchCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refusal(reason) => f.write_str(reason),
            _ => f.write_str("typed launch preparation cause retained"),
        }
    }
}
impl std::error::Error for ReplayLaunchCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Construction(e) => Some(e.as_ref()),
            Self::Input(e) => Some(e.as_ref()),
            Self::Profile(e) => match e.source.as_deref() {
                Some(super::native_replay::CpuFreshAssetSourceError::Json(source)) => Some(source),
                Some(super::native_replay::CpuFreshAssetSourceError::Io(source)) => Some(source),
                None => Some(&e.fault),
            },
            Self::Io(e) => Some(e),
            Self::Reserve(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Utf8(e) => Some(e),
            Self::Output(e) => Some(e.as_ref()),
            Self::Integer(e) => Some(e),
            Self::Cpu(e) => Some(e.as_ref()),
            Self::Replay(_) | Self::Refusal(_) => None,
        }
    }
}

// The prepared request includes its bounded audit metadata. Keep its ownership
// behind one pointer so every ordinary payload row does not reserve that larger
// inline layout; the original byte backing itself is moved, never copied.
enum PayloadBacking {
    Bytes(Vec<u8>),
    Request(Box<PreparedReplayRequest>),
}
pub struct ReplayLaunchPayload {
    name: &'static str,
    artifact: ArtifactPin,
    backing: PayloadBacking,
}
impl ReplayLaunchPayload {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn artifact(&self) -> &ArtifactPin {
        &self.artifact
    }
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            PayloadBacking::Bytes(bytes) => bytes,
            PayloadBacking::Request(request) => request.as_bytes(),
        }
    }
}
impl fmt::Debug for ReplayLaunchPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplayLaunchPayload")
            .field("name", &self.name)
            .field("artifact", &self.artifact)
            .finish()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayPublishedFile {
    pub name: String,
    pub artifact: ArtifactPin,
    pub created: bool,
    /// Confirmed successful write return counts, not unknown kernel side effects.
    pub written_bytes: usize,
    pub sync_completed: bool,
    /// Confirmed read returns, including a mismatching/trailing byte if observed.
    pub readback_bytes: usize,
    pub readback_verified: bool,
    pub close_result_observed: bool,
}
#[derive(Debug, Serialize)]
pub struct ReplayLaunchDiagnostics {
    pub expected_semantic_receipt_producer_scope: SemanticReceiptProducerScope,
    pub elapsed_ms: u64,
    pub deadline_exceeded: bool,
    pub execution_deadline_exceeded: bool,
    pub original_whole_deadline_retained: bool,
    pub whole_wall_ms: Option<u64>,
    pub cleanup_reserve_ms: Option<u64>,
    pub output_limit: usize,
    pub canceled: bool,
    pub normal_manifest_published: bool,
    pub normal_manifest_quarantined: bool,
    pub normal_manifest_absence_verified: bool,
    pub directory_created: bool,
    pub blocking_io_hard_timeout_supported: bool,
    #[serde(skip)]
    clock: Option<LaunchClock>,
    #[serde(skip)]
    original_started: Instant,
}
impl ReplayLaunchDiagnostics {
    pub fn deadline(&self) -> Option<Instant> {
        self.clock.map(|c| c.whole)
    }
    pub fn execution_deadline(&self) -> Option<Instant> {
        self.clock.map(|c| c.execution)
    }
    pub fn original_started(&self) -> Instant {
        self.original_started
    }
}
#[derive(Debug)]
pub struct ReplayLaunchPreparationError {
    pub stage: &'static str,
    pub cause: Box<ReplayLaunchCause>,
    pub diagnostics: Box<ReplayLaunchDiagnostics>,
    pub evidence: Box<ReplayLaunchFailureEvidence>,
    pub quarantine_error: Option<Box<ReplayLaunchCause>>,
    pub bundle_directory: Option<PathBuf>,
}
pub struct ReplayLaunchFailureEvidence {
    publication: Vec<ReplayPublishedFile>,
    retained_payloads: Vec<ReplayLaunchPayload>,
    retained_request: Option<PreparedReplayRequest>,
    partial_serialization: Vec<u8>,
}
impl ReplayLaunchFailureEvidence {
    pub fn publication(&self) -> &[ReplayPublishedFile] {
        &self.publication
    }
    pub fn retained_payloads(&self) -> &[ReplayLaunchPayload] {
        &self.retained_payloads
    }
    pub fn retained_request(&self) -> Option<&PreparedReplayRequest> {
        self.retained_request.as_ref()
    }
    pub fn partial_serialization(&self) -> &[u8] {
        &self.partial_serialization
    }
}
impl fmt::Debug for ReplayLaunchFailureEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplayLaunchFailureEvidence")
            .field("publication", &self.publication)
            .field("retained_payloads", &self.retained_payloads)
            .field("retained_request", &self.retained_request)
            .field(
                "partial_serialization_bytes",
                &self.partial_serialization.len(),
            )
            .finish()
    }
}
impl fmt::Display for ReplayLaunchPreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "launch preparation failed at {}", self.stage)
    }
}
impl std::error::Error for ReplayLaunchPreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayLaunchManifest {
    pub schema: String,
    pub scope: String,
    pub files: Vec<ReplayPublishedFile>,
    pub mode: ReplayInputMode,
    pub expected_semantic_receipt_producer_scope: SemanticReceiptProducerScope,
    pub whole_wall_ms: u64,
    pub cleanup_reserve_ms: u64,
    pub elapsed_before_manifest_ms: u64,
    pub publication_io_credit_bytes: usize,
    pub publication_backing_policy_bytes: usize,
    pub inherited_request_backing_bytes: usize,
    pub required_native_output_bytes: usize,
    pub original_clock_transferable_to_other_process: bool,
    pub blocking_io_hard_timeout_supported: bool,
    pub hostile_path_race_protection_proved: bool,
    pub allocator_peak_observed: bool,
    pub rss_peak_observed: bool,
    pub spawn_observed: bool,
    pub native_result_observed: bool,
    pub directory_metadata_persistence_observed: bool,
    pub authorities: ReplayAuthorities,
    pub context_sha256: String,
}
/// Owns prepared bytes and observed file identities only. Drop closes Rust-owned
/// buffers, never deletes durable or failed evidence. Later dispatch must recheck
/// pins, independent registration and caller's same original deadline itself.
pub struct PreparedReplayLaunchBundle {
    directory: PathBuf,
    payloads: Vec<ReplayLaunchPayload>,
    publication: Vec<ReplayPublishedFile>,
    manifest: ReplayLaunchManifest,
    manifest_artifact: ArtifactPin,
    clock: LaunchClock,
    directory_sync_after_publication: Option<bool>,
}
impl fmt::Debug for PreparedReplayLaunchBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedReplayLaunchBundle")
            .field("manifest_artifact", &self.manifest_artifact)
            .finish_non_exhaustive()
    }
}
impl PreparedReplayLaunchBundle {
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn payloads(&self) -> &[ReplayLaunchPayload] {
        &self.payloads
    }
    pub fn publication(&self) -> &[ReplayPublishedFile] {
        &self.publication
    }
    pub fn manifest(&self) -> &ReplayLaunchManifest {
        &self.manifest
    }
    pub fn manifest_artifact(&self) -> &ArtifactPin {
        &self.manifest_artifact
    }
    pub fn deadline(&self) -> Instant {
        self.clock.whole
    }
    pub fn execution_deadline(&self) -> Instant {
        self.clock.execution
    }
    pub fn original_started(&self) -> Instant {
        self.clock.started
    }
    pub fn directory_sync_after_publication(&self) -> Option<bool> {
        self.directory_sync_after_publication
    }
}

struct WireWriter<'a> {
    backing: Option<&'a mut Vec<u8>>,
    written: usize,
    limit: usize,
    clock: LaunchClock,
    cancel: &'a AtomicBool,
    hasher: Sha256,
}
impl Write for WireWriter<'_> {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        self.clock
            .check(self.cancel)
            .map_err(|_| io::Error::other("original launch clock/cancel refused serialization"))?;
        let next = self
            .written
            .checked_add(input.len())
            .ok_or_else(|| io::Error::other("launch wire count overflow"))?;
        if next > self.limit
            || self
                .backing
                .as_ref()
                .is_some_and(|bytes| next > bytes.capacity())
        {
            return Err(io::Error::other(
                "launch wire finite byte/backing limit exceeded",
            ));
        }
        if let Some(backing) = self.backing.as_mut() {
            backing.extend_from_slice(input);
        }
        self.hasher.update(input);
        self.written = next;
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn count_wire<T: Serialize>(
    value: &T,
    limit: usize,
    clock: LaunchClock,
    cancel: &AtomicBool,
) -> Result<ArtifactPin, ReplayLaunchCause> {
    clock.check(cancel)?;
    let mut writer = WireWriter {
        backing: None,
        written: 0,
        limit,
        clock,
        cancel,
        hasher: Sha256::new(),
    };
    serde_json::to_writer(&mut writer, value).map_err(ReplayLaunchCause::Json)?;
    clock.check(cancel)?;
    Ok(ArtifactPin {
        bytes: writer.written as u64,
        sha256: format!("{:x}", writer.hasher.finalize()),
    })
}
fn reserve_bytes(amount: usize) -> Result<Vec<u8>, ReplayLaunchCause> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(amount)
        .map_err(ReplayLaunchCause::Reserve)?;
    if bytes.capacity() != amount {
        return Err(ReplayLaunchCause::Refusal(
            "retained publication backing exceeds exact credit",
        ));
    }
    Ok(bytes)
}
fn wire_into<T: Serialize>(
    value: &T,
    backing: &mut Vec<u8>,
    limit: usize,
    clock: LaunchClock,
    cancel: &AtomicBool,
) -> Result<(), ReplayLaunchCause> {
    clock.check(cancel)?;
    let mut writer = WireWriter {
        backing: Some(backing),
        written: 0,
        limit,
        clock,
        cancel,
        hasher: Sha256::new(),
    };
    serde_json::to_writer(&mut writer, value).map_err(ReplayLaunchCause::Json)?;
    clock.check(cancel)
}
fn owned_original(
    name: &'static str,
    raw: &[u8],
) -> Result<ReplayLaunchPayload, ReplayLaunchCause> {
    let mut bytes = reserve_bytes(raw.len())?;
    bytes.extend_from_slice(raw);
    Ok(ReplayLaunchPayload {
        name,
        artifact: super::pin(raw),
        backing: PayloadBacking::Bytes(bytes),
    })
}
fn descriptor(path: &Path) -> Result<&str, ReplayLaunchCause> {
    let text = path
        .to_str()
        .ok_or(ReplayLaunchCause::Refusal("descriptor path must be UTF-8"))?;
    if !path.is_absolute()
        || text.is_empty()
        || text.len() > MAX_PATH_BYTES
        || text.contains('\0')
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(ReplayLaunchCause::Refusal(
            "bounded absolute descriptor path without traversal required",
        ));
    }
    Ok(text)
}
fn safe_directory_name(name: &str) -> Result<(), ReplayLaunchCause> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        return Err(ReplayLaunchCause::Refusal(
            "single bounded bundle directory component required",
        ));
    }
    let upper = name.to_ascii_uppercase();
    if matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && matches!(upper.as_bytes()[3], b'1'..=b'9'))
    {
        return Err(ReplayLaunchCause::Refusal(
            "reserved Windows directory component refused",
        ));
    }
    Ok(())
}
fn metadata_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_type().is_symlink() || metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}
fn plain_directory(path: &Path) -> Result<(), ReplayLaunchCause> {
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor).map_err(ReplayLaunchCause::Io)?;
        if metadata_link(&metadata) || !metadata.is_dir() {
            return Err(ReplayLaunchCause::Refusal(
                "output root/ancestor must be plain directory without links or reparse points",
            ));
        }
    }
    Ok(())
}
fn absent(path: &Path) -> Result<(), ReplayLaunchCause> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ReplayLaunchCause::Io(error)),
        Ok(_) => Err(ReplayLaunchCause::Refusal(
            "owned publication target already exists",
        )),
    }
}
fn fixed_child(directory: &Path, name: &'static str) -> Result<PathBuf, ReplayLaunchCause> {
    if name.contains('/') || name.contains('\\') {
        return Err(ReplayLaunchCause::Refusal(
            "fixed file name is not one component",
        ));
    }
    let path = directory.join(name);
    if path.parent() != Some(directory)
        || path.to_str().is_none_or(|text| text.len() > MAX_PATH_BYTES)
    {
        return Err(ReplayLaunchCause::Refusal(
            "publication target must remain inside checked bundle directory",
        ));
    }
    Ok(path)
}
fn directory_sync(path: &Path) -> Result<Option<bool>, ReplayLaunchCause> {
    #[cfg(unix)]
    {
        File::open(path)
            .and_then(|file| file.sync_all())
            .map(|()| Some(true))
            .map_err(ReplayLaunchCause::Io)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(None)
    }
}

fn publish_file(
    directory: &Path,
    physical_name: &'static str,
    payload: &ReplayLaunchPayload,
    evidence: &mut ReplayPublishedFile,
    clock: LaunchClock,
    cancel: &AtomicBool,
) -> Result<(), ReplayLaunchCause> {
    clock.check(cancel)?;
    plain_directory(directory)?;
    clock.check(cancel)?;
    let path = fixed_child(directory, physical_name)?;
    // create_new is the actual collision gate. No old file is opened to replace.
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(ReplayLaunchCause::Io)?;
    evidence.created = true;
    clock.check(cancel)?;
    while evidence.written_bytes < payload.as_bytes().len() {
        clock.check(cancel)?;
        let written = file
            .write(&payload.as_bytes()[evidence.written_bytes..])
            .map_err(ReplayLaunchCause::Io)?;
        if written == 0 {
            return Err(ReplayLaunchCause::Io(io::Error::new(
                io::ErrorKind::WriteZero,
                "owned payload write returned zero",
            )));
        }
        evidence.written_bytes = evidence
            .written_bytes
            .checked_add(written)
            .ok_or(ReplayLaunchCause::Refusal("write accounting overflow"))?;
        clock.check(cancel)?;
    }
    clock.check(cancel)?;
    file.sync_all().map_err(ReplayLaunchCause::Io)?;
    evidence.sync_completed = true;
    clock.check(cancel)?;
    verify_readback(&mut file, payload.as_bytes(), evidence, clock, cancel)
}
fn verify_readback(
    file: &mut File,
    original: &[u8],
    evidence: &mut ReplayPublishedFile,
    clock: LaunchClock,
    cancel: &AtomicBool,
) -> Result<(), ReplayLaunchCause> {
    clock.check(cancel)?;
    file.seek(SeekFrom::Start(0))
        .map_err(ReplayLaunchCause::Io)?;
    clock.check(cancel)?;
    let mut chunk = [0_u8; 4096];
    while evidence.readback_bytes < original.len() {
        clock.check(cancel)?;
        let count = (original.len() - evidence.readback_bytes).min(chunk.len());
        let got = file
            .read(&mut chunk[..count])
            .map_err(ReplayLaunchCause::Io)?;
        if got == 0 {
            return Err(ReplayLaunchCause::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "owned payload readback ended early",
            )));
        }
        let start = evidence.readback_bytes;
        let end = start
            .checked_add(got)
            .ok_or(ReplayLaunchCause::Refusal("readback accounting overflow"))?;
        evidence.readback_bytes = end;
        if chunk[..got] != original[start..end] {
            return Err(ReplayLaunchCause::Refusal(
                "readback bytes differ from original prepared payload",
            ));
        }
        clock.check(cancel)?;
    }
    clock.check(cancel)?;
    let trailing = file.read(&mut chunk[..1]).map_err(ReplayLaunchCause::Io)?;
    evidence.readback_bytes =
        evidence
            .readback_bytes
            .checked_add(trailing)
            .ok_or(ReplayLaunchCause::Refusal(
                "trailing read accounting overflow",
            ))?;
    if trailing != 0 {
        return Err(ReplayLaunchCause::Refusal(
            "readback has undeclared trailing bytes",
        ));
    }
    evidence.readback_verified = true;
    clock.check(cancel)?;
    // File Drop's close result and crash persistence are not observed here.
    Ok(())
}

fn evidence(payload: &ReplayLaunchPayload) -> ReplayPublishedFile {
    ReplayPublishedFile {
        name: payload.name.into(),
        artifact: payload.artifact.clone(),
        created: false,
        written_bytes: 0,
        sync_completed: false,
        readback_bytes: 0,
        readback_verified: false,
        close_result_observed: false,
    }
}
struct LaunchFailureEvidence {
    files: Vec<ReplayPublishedFile>,
    payloads: Vec<ReplayLaunchPayload>,
    directory: Option<PathBuf>,
    request: Option<PreparedReplayRequest>,
    partial_serialization: Vec<u8>,
}
fn launch_error(
    stage: &'static str,
    cause: ReplayLaunchCause,
    clock: Option<LaunchClock>,
    scope: SemanticReceiptProducerScope,
    cancel: &AtomicBool,
    original_started: Instant,
    evidence: LaunchFailureEvidence,
) -> ReplayLaunchPreparationError {
    let LaunchFailureEvidence {
        files,
        payloads,
        directory,
        request,
        partial_serialization,
    } = evidence;
    let now = Instant::now();
    ReplayLaunchPreparationError {
        stage,
        cause: Box::new(cause),
        diagnostics: Box::new(ReplayLaunchDiagnostics {
            expected_semantic_receipt_producer_scope: scope,
            elapsed_ms: super::super::milliseconds(now.saturating_duration_since(original_started)),
            deadline_exceeded: clock.is_some_and(|c| now >= c.whole),
            execution_deadline_exceeded: clock.is_some_and(|c| now >= c.execution),
            original_whole_deadline_retained: clock.is_some(),
            whole_wall_ms: clock.map(|c| c.whole_wall_ms),
            cleanup_reserve_ms: clock.map(|c| c.cleanup_reserve_ms),
            output_limit: clock.map_or(1024, |c| c.output_limit),
            canceled: cancel.load(Ordering::Acquire),
            normal_manifest_published: false,
            normal_manifest_quarantined: false,
            normal_manifest_absence_verified: false,
            directory_created: directory.is_some(),
            blocking_io_hard_timeout_supported: false,
            clock,
            original_started,
        }),
        evidence: Box::new(ReplayLaunchFailureEvidence {
            publication: files,
            retained_payloads: payloads,
            retained_request: request,
            partial_serialization,
        }),
        quarantine_error: None,
        bundle_directory: directory,
    }
}
fn quarantine_manifest(
    error: &mut ReplayLaunchPreparationError,
    clock: Option<LaunchClock>,
) -> Result<(), ReplayLaunchCause> {
    let Some(directory) = error.bundle_directory.as_ref() else {
        return Ok(());
    };
    if let Some(clock) = clock {
        if Instant::now() >= clock.whole {
            return Err(ReplayLaunchCause::Refusal(
                "original whole deadline expired before manifest quarantine",
            ));
        }
    }
    plain_directory(directory)?;
    let normal = fixed_child(directory, FINAL_MANIFEST)?;
    if error.diagnostics.normal_manifest_published {
        let failed = fixed_child(directory, FAILED_MANIFEST)?;
        absent(&failed)?;
        fs::rename(normal, failed).map_err(ReplayLaunchCause::Io)?;
        error.diagnostics.normal_manifest_quarantined = true;
    }
    absent(&fixed_child(directory, FINAL_MANIFEST)?)?;
    error.diagnostics.normal_manifest_absence_verified = true;
    Ok(())
}

/// Result-free file preparation. All asset paths remain descriptors. The original
/// process-local S/E/W survive in the returned bundle, not in a transferable
/// Instant serialized to another process. Actual supervision/dispatch is later.
pub fn prepare_replay_launch_bundle(
    originals: ReplayLaunchOriginals<'_>,
    declaration: ReplayLaunchDeclaration<'_>,
    expected: ReplayLaunchExpected<'_>,
    budget: ReplayLaunchBudget,
    started: Instant,
    cancel: &AtomicBool,
) -> Result<PreparedReplayLaunchBundle, ReplayLaunchPreparationError> {
    let mut clock = LaunchClock::declared(started, declaration.replay);
    let mut stage = "declaration";
    let mut payloads = Vec::new();
    let mut files = Vec::new();
    let mut directory = None;
    let mut normal_manifest_published = false;
    let mut pending_request = None;
    let mut serializing = Vec::new();
    let result =
        (|| -> Result<(ReplayLaunchManifest, ArtifactPin, Option<bool>), ReplayLaunchCause> {
            if !(1..=MAX_PUBLICATION_BYTES).contains(&budget.max_publication_bytes)
                || !(1..=MAX_PUBLICATION_BACKING_BYTES)
                    .contains(&budget.max_publication_backing_bytes)
            {
                return Err(ReplayLaunchCause::Refusal(
                    "finite publication IO/backing budget required",
                ));
            }
            if budget.max_publication_backing_bytes < METADATA_CREDIT {
                return Err(ReplayLaunchCause::Refusal(
                    "bounded metadata bootstrap policy credit is insufficient",
                ));
            }
            let scope = declaration.replay_binary_pin_scope;
            if !matches!(
                scope,
                "linux_loaded_executable_inode" | "current_exe_path_hash"
            ) {
                return Err(ReplayLaunchCause::Refusal(
                    "independent eventual replay executable scope required",
                ));
            }
            let export_path = descriptor(declaration.assets.export_manifest)?;
            let runtime_path = descriptor(declaration.assets.runtime_library)?;
            let cache_path = descriptor(declaration.assets.runtime_cache_root)?;
            descriptor(declaration.output_root)?;
            safe_directory_name(declaration.bundle_directory_name)?;
            let declared_clock = clock.ok_or(ReplayLaunchCause::Refusal(
                "original finite wall is unknown",
            ))?;
            declared_clock.check(cancel)?;
            stage = "profile";
            let profile = check_cpu_fresh_asset_profile(
                originals.cpu_fresh_profile,
                expected.cpu_fresh_profile_artifact,
                expected.cpu_fresh_profile,
            )
            .map_err(|error| ReplayLaunchCause::Profile(Box::new(error)))?;
            let profile_text = std::str::from_utf8(originals.cpu_fresh_profile)
                .map_err(ReplayLaunchCause::Utf8)?;
            stage = "construction";
            pending_request = Some(
                prepare_replay_request(
                    originals.replay,
                    declaration.replay,
                    expected.replay,
                    declaration.expected_semantic_receipt_producer_scope,
                    budget.construction,
                    started,
                )
                .map_err(|error| ReplayLaunchCause::Construction(Box::new(error)))?,
            );
            let request = pending_request.as_ref().ok_or(ReplayLaunchCause::Refusal(
                "missing own constructed request",
            ))?;
            let admitted_clock = LaunchClock {
                execution: request.execution_deadline(),
                whole: request.deadline(),
                ..declared_clock
            };
            clock = Some(admitted_clock);
            admitted_clock.check(cancel)?;
            stage = "native_output_reservation";
            let checked = check_replay_inputs_with_semantic_scope(
                request.as_bytes(),
                expected.replay,
                declaration.expected_semantic_receipt_producer_scope,
                started,
            )
            .map_err(|error| ReplayLaunchCause::Input(Box::new(error)))?;
            let (config, root, plan, limits) = checked.into_owner_args();
            let required = super::replay_inputs::replay_mode_requirements(
                declaration.replay.mode,
                config.line_plies,
                plan.prefix.len(),
                plan.nodes_per_check,
            )
            .map_err(|error| ReplayLaunchCause::Replay(Box::new(error)))?;
            let roles = usize::try_from(required.actual_role_calls())
                .map_err(ReplayLaunchCause::Integer)?;
            let checks = required.cpu_checks();
            drop((config, root, plan, limits));
            let output_required = replay_output_requirements(roles, checks)
                .map_err(|error| ReplayLaunchCause::Output(Box::new(error)))?;
            output_required
                .check_output_limit(declaration.replay.resources.output_bytes)
                .map_err(|error| ReplayLaunchCause::Output(Box::new(error)))?;
            stage = "publication_credit";
            // Expected fields and profile facts were bounded by existing checkers
            // before these finite metadata clones. This separate credit is not the
            // inherited request construction credit or native output reservation.
            let assets = LaunchAssets {
                schema: LAUNCH_ASSETS_SCHEMA.into(),
                export_manifest_path: export_path.into(),
                runtime_library_path: runtime_path.into(),
                runtime_cache_root: cache_path.into(),
                cpu_fresh_profile_raw: profile_text.into(),
            };
            let assets_pin = count_wire(&assets, MAX_LAUNCH_ASSETS_BYTES, admitted_clock, cancel)?;
            let assets_len =
                usize::try_from(assets_pin.bytes).map_err(ReplayLaunchCause::Integer)?;
            let transport = ExpectedTransportV2 {
                schema: EXPECTED_TRANSPORT_V2_SCHEMA.into(),
                request_artifact: request.artifact().clone(),
                replay_expected: ExpectedPinsWire::from_expected(expected.replay),
                launch_asset_artifact: assets_pin.clone(),
                cpu_fresh_profile_artifact: expected.cpu_fresh_profile_artifact.clone(),
                cpu_fresh_profile: profile,
                whole_wall_ms: declaration.replay.resources.whole_wall_ms,
                cleanup_reserve_ms: declaration.replay.resources.cleanup_reserve_ms,
                output_bytes: declaration.replay.resources.output_bytes,
                replay_binary_pin_scope: scope.into(),
                expected_semantic_receipt_producer_scope: declaration
                    .expected_semantic_receipt_producer_scope,
            };
            let expected_pin = count_wire(&transport, MAX_EXPECTED_BYTES, admitted_clock, cancel)?;
            let expected_len =
                usize::try_from(expected_pin.bytes).map_err(ReplayLaunchCause::Integer)?;
            let original_bytes = [
                originals.replay.registration,
                originals.replay.prepared_action,
                originals.replay.semantic_receipt,
                originals.cpu_fresh_profile,
            ]
            .iter()
            .try_fold(0_usize, |total, raw| total.checked_add(raw.len()))
            .ok_or(ReplayLaunchCause::Refusal(
                "original publication byte sum overflow",
            ))?;
            let backing_credit = original_bytes
                .checked_add(assets_len)
                .and_then(|sum| sum.checked_add(expected_len))
                .and_then(|sum| sum.checked_add(MANIFEST_CAP))
                .and_then(|sum| sum.checked_add(METADATA_CREDIT))
                .ok_or(ReplayLaunchCause::Refusal(
                    "publication backing arithmetic overflow",
                ))?;
            let io_credit = original_bytes
                .checked_add(request.as_bytes().len())
                .and_then(|sum| sum.checked_add(assets_len))
                .and_then(|sum| sum.checked_add(expected_len))
                .and_then(|sum| sum.checked_add(MANIFEST_CAP))
                .and_then(|sum| sum.checked_mul(2))
                .and_then(|sum| sum.checked_add(FILE_COUNT))
                .ok_or(ReplayLaunchCause::Refusal(
                    "publication IO arithmetic overflow",
                ))?;
            if backing_credit > budget.max_publication_backing_bytes
                || io_credit > budget.max_publication_bytes
            {
                return Err(ReplayLaunchCause::Refusal(
                    "declared publication IO/backing credit is insufficient",
                ));
            }
            payloads
                .try_reserve_exact(FILE_COUNT)
                .map_err(ReplayLaunchCause::Reserve)?;
            files
                .try_reserve_exact(FILE_COUNT)
                .map_err(ReplayLaunchCause::Reserve)?;
            if payloads.capacity() != FILE_COUNT || files.capacity() != FILE_COUNT {
                return Err(ReplayLaunchCause::Refusal(
                    "retained publication row capacity exceeds credit",
                ));
            }
            stage = "assets_serialization";
            serializing = reserve_bytes(assets_len)?;
            wire_into(
                &assets,
                &mut serializing,
                assets_len,
                admitted_clock,
                cancel,
            )?;
            if super::pin(&serializing) != assets_pin {
                return Err(ReplayLaunchCause::Refusal(
                    "counted assets pin differs from actual serialization",
                ));
            }
            payloads.push(ReplayLaunchPayload {
                name: "launch-assets.json",
                artifact: assets_pin,
                backing: PayloadBacking::Bytes(std::mem::take(&mut serializing)),
            });
            stage = "expected_serialization";
            serializing = reserve_bytes(expected_len)?;
            wire_into(
                &transport,
                &mut serializing,
                expected_len,
                admitted_clock,
                cancel,
            )?;
            if super::pin(&serializing) != expected_pin {
                return Err(ReplayLaunchCause::Refusal(
                    "counted expected pin differs from actual serialization",
                ));
            }
            payloads.push(ReplayLaunchPayload {
                name: "replay-expected.json",
                artifact: expected_pin,
                backing: PayloadBacking::Bytes(std::mem::take(&mut serializing)),
            });
            for (name, raw) in [
                ("replay-registration.json", originals.replay.registration),
                ("prepared-action.json", originals.replay.prepared_action),
                ("semantic-receipt.json", originals.replay.semantic_receipt),
                ("cpu-fresh-profile.json", originals.cpu_fresh_profile),
            ] {
                payloads.push(owned_original(name, raw)?);
            }
            let inherited_request_backing_bytes = request.as_bytes().len();
            let request = pending_request
                .take()
                .ok_or(ReplayLaunchCause::Refusal("own request already consumed"))?;
            payloads.push(ReplayLaunchPayload {
                name: "replay-input.json",
                artifact: request.artifact().clone(),
                backing: PayloadBacking::Request(Box::new(request)),
            });
            serializing = reserve_bytes(MANIFEST_CAP)?;
            stage = "output_root";
            admitted_clock.check(cancel)?;
            plain_directory(declaration.output_root)?;
            let caller_root =
                fs::canonicalize(declaration.output_root).map_err(ReplayLaunchCause::Io)?;
            plain_directory(&caller_root)?;
            let bundle_root = caller_root.join(declaration.bundle_directory_name);
            if bundle_root.parent() != Some(caller_root.as_path())
                || bundle_root
                    .to_str()
                    .is_none_or(|text| text.len() > MAX_PATH_BYTES)
            {
                return Err(ReplayLaunchCause::Refusal(
                    "new bundle directory must remain in checked caller root",
                ));
            }
            fs::create_dir(&bundle_root).map_err(ReplayLaunchCause::Io)?;
            directory = Some(bundle_root.clone());
            admitted_clock.check(cancel)?;
            stage = "payload_publication";
            for payload in &payloads {
                files.push(evidence(payload));
                let last = files
                    .last_mut()
                    .ok_or(ReplayLaunchCause::Refusal("missing own publication row"))?;
                publish_file(
                    &bundle_root,
                    payload.name,
                    payload,
                    last,
                    admitted_clock,
                    cancel,
                )?;
            }
            stage = "manifest_preparation";
            admitted_clock.check(cancel)?;
            let _ = directory_sync(&bundle_root)?;
            admitted_clock.check(cancel)?;
            let mut manifest = ReplayLaunchManifest {
                schema: PREPARATION_MANIFEST_SCHEMA.into(),
                scope: "original_bytes_to_result_free_durable_launch_bundle_only".into(),
                files: files.clone(),
                mode: declaration.replay.mode,
                expected_semantic_receipt_producer_scope: declaration
                    .expected_semantic_receipt_producer_scope,
                whole_wall_ms: admitted_clock.whole_wall_ms,
                cleanup_reserve_ms: admitted_clock.cleanup_reserve_ms,
                elapsed_before_manifest_ms: super::super::milliseconds(started.elapsed()),
                publication_io_credit_bytes: io_credit,
                publication_backing_policy_bytes: backing_credit,
                inherited_request_backing_bytes,
                required_native_output_bytes: output_required.required_output_bytes(),
                original_clock_transferable_to_other_process: false,
                blocking_io_hard_timeout_supported: false,
                hostile_path_race_protection_proved: false,
                allocator_peak_observed: false,
                rss_peak_observed: false,
                spawn_observed: false,
                native_result_observed: false,
                directory_metadata_persistence_observed: false,
                authorities: ReplayAuthorities::default(),
                context_sha256: String::new(),
            };
            let mut body = serde_json::to_value(&manifest).map_err(ReplayLaunchCause::Json)?;
            body.as_object_mut()
                .ok_or(ReplayLaunchCause::Refusal("manifest body is not object"))?
                .remove("context_sha256");
            manifest.context_sha256 =
                super::super::json_digest(&serde_json::json!([PREPARATION_MANIFEST_SCHEMA, body]))
                    .map_err(|error| ReplayLaunchCause::Cpu(Box::new(error)))?;
            wire_into(
                &manifest,
                &mut serializing,
                MANIFEST_CAP,
                admitted_clock,
                cancel,
            )?;
            let manifest_artifact = super::pin(&serializing);
            payloads.push(ReplayLaunchPayload {
                name: FINAL_MANIFEST,
                artifact: manifest_artifact.clone(),
                backing: PayloadBacking::Bytes(std::mem::take(&mut serializing)),
            });
            let payload = payloads
                .last()
                .ok_or(ReplayLaunchCause::Refusal("missing own manifest payload"))?;
            files.push(evidence(payload));
            stage = "manifest_pending_publication";
            publish_file(
                &bundle_root,
                PENDING_MANIFEST,
                payload,
                files
                    .last_mut()
                    .ok_or(ReplayLaunchCause::Refusal("missing manifest row"))?,
                admitted_clock,
                cancel,
            )?;
            stage = "manifest_commit";
            admitted_clock.check(cancel)?;
            plain_directory(&bundle_root)?;
            let pending = fixed_child(&bundle_root, PENDING_MANIFEST)?;
            let final_path = fixed_child(&bundle_root, FINAL_MANIFEST)?;
            absent(&final_path)?;
            fs::rename(&pending, &final_path).map_err(ReplayLaunchCause::Io)?;
            normal_manifest_published = true;
            admitted_clock.check(cancel)?;
            stage = "directory_sync_after_commit";
            let synced = directory_sync(&bundle_root)?;
            admitted_clock.check(cancel)?;
            Ok((manifest, manifest_artifact, synced))
        })();
    match result {
        Ok((manifest, manifest_artifact, synced)) => {
            let Some(clock) = clock else {
                return Err(launch_error(
                    "publication",
                    ReplayLaunchCause::Refusal("completed preparation lost original clock"),
                    None,
                    declaration.expected_semantic_receipt_producer_scope,
                    cancel,
                    started,
                    LaunchFailureEvidence {
                        files,
                        payloads,
                        directory,
                        request: pending_request,
                        partial_serialization: serializing,
                    },
                ));
            };
            let Some(directory) = directory else {
                return Err(launch_error(
                    "publication",
                    ReplayLaunchCause::Refusal("completed preparation lost owned directory"),
                    Some(clock),
                    declaration.expected_semantic_receipt_producer_scope,
                    cancel,
                    started,
                    LaunchFailureEvidence {
                        files,
                        payloads,
                        directory: None,
                        request: pending_request,
                        partial_serialization: serializing,
                    },
                ));
            };
            Ok(PreparedReplayLaunchBundle {
                directory,
                payloads,
                publication: files,
                manifest,
                manifest_artifact,
                clock,
                directory_sync_after_publication: synced,
            })
        }
        Err(cause) => {
            let mut error = launch_error(
                stage,
                cause,
                clock,
                declaration.expected_semantic_receipt_producer_scope,
                cancel,
                started,
                LaunchFailureEvidence {
                    files,
                    payloads,
                    directory,
                    request: pending_request,
                    partial_serialization: serializing,
                },
            );
            error.diagnostics.normal_manifest_published = normal_manifest_published;
            if let Err(secondary) = quarantine_manifest(&mut error, clock) {
                error.quarantine_error = Some(Box::new(secondary));
            }
            // Quarantine never creates a fresh grace/drain window. The original
            // primary, partial bytes and original clock remain owned even if it
            // cannot remove the normal name. File existence grants no authority.
            let now = Instant::now();
            if let Some(clock) = clock {
                error.diagnostics.elapsed_ms =
                    super::super::milliseconds(now.saturating_duration_since(clock.started));
                error.diagnostics.deadline_exceeded = now >= clock.whole;
                error.diagnostics.execution_deadline_exceeded = now >= clock.execution;
            }
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super as cpu_task;
    use super::super::replay_inputs::{
        BINARY_DECLARATION_SCOPE, MAX_CONSTRUCTION_CREDIT_BYTES, MAX_OUTPUT_BYTES,
        MAX_REQUEST_BYTES, MAX_SEMANTIC_RECEIPT_BYTES, REGISTRATION_SCHEMA, ReplayConfigInput,
        ReplayConsumerRegistration, ReplayResourceDeclaration,
    };
    use super::*;
    use crate::engine::{OwnerRegistry, RulesUciPort};
    use crate::{Command, ParserLimits, PositionPort, parse};
    use cpu_task::semantic::{self, SemanticReceipt, SemanticRequest};
    use rz_position::{BoardMove, PositionLimits};
    use rz_search::cpu::CpuProfile;
    use std::sync::{Arc, atomic::AtomicU64};

    const WALL: u64 = 20_000;
    const CLEANUP: u64 = 500;
    const OLD_BINARY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const NEW_BINARY: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SEMANTIC_BINARY: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    // Controlled filesystem fixtures own one new canonical temp child. They do
    // not load a runtime/model or run a CPU task, child process or native owner.
    struct TempRoot {
        base: PathBuf,
        path: PathBuf,
    }
    impl TempRoot {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let base = fs::canonicalize(std::env::temp_dir()).unwrap();
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let name = format!(
                "rz-replay-launch-test-{}-{stamp}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let path = base.join(name);
            assert_eq!(path.parent(), Some(base.as_path()));
            fs::create_dir(&path).unwrap();
            Self { base, path }
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            // Only this fixture's already-created exact child may be removed.
            if self.path.parent() == Some(self.base.as_path())
                && self
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("rz-replay-launch-test-"))
            {
                let _ = fs::remove_dir_all(&self.path);
            }
        }
    }
    struct Fixture {
        registration: Vec<u8>,
        action: Vec<u8>,
        semantic: Vec<u8>,
        profile_raw: Vec<u8>,
        expected: ReplayExpectedPins,
        profile: CpuFreshAssetProfile,
        profile_pin: ArtifactPin,
        config: ReplayConfigInput,
        resources: ReplayResourceDeclaration,
        mode: ReplayInputMode,
    }
    fn bits(text: &str) -> u16 {
        cpu_task::pack_moves(&[BoardMove::from_uci(text).unwrap()]).unwrap()[0]
    }
    fn artifact(sha: &str, bytes: u64) -> ArtifactPin {
        ArtifactPin {
            bytes,
            sha256: sha.into(),
        }
    }
    fn seal<T: Serialize>(schema: &str, value: &T) -> String {
        let mut body = serde_json::to_value(value).unwrap();
        body.as_object_mut().unwrap().remove("context_sha256");
        cpu_task::json_digest(&serde_json::json!([schema, body])).unwrap()
    }
    fn fixture(mode: ReplayInputMode) -> Fixture {
        let command = "position startpos";
        let Command::Position(spec) = parse(command, ParserLimits::default()).unwrap() else {
            panic!("position fixture");
        };
        let owners = Arc::new(OwnerRegistry::default());
        let prepared = RulesUciPort::new(Arc::clone(&owners), PositionLimits::default())
            .prepare(&spec)
            .unwrap();
        let root = prepared.snapshot.rules_position();
        let prefix = vec![bits("e2e4")];
        let restriction = vec![bits("e7e5"), bits("c7c5")];
        let mut semantic_request = SemanticRequest {
            schema: semantic::SEMANTIC_SCHEMA.into(),
            question: semantic::SemanticQuestion::RestrictedResponse,
            parent_input_sha256: "d".repeat(64),
            before_result_anchor_sha256: "e".repeat(64),
            position_command: command.into(),
            expected_board_fen: root.to_fen(),
            rules_state_sha256: cpu_task::state_sha(root, &owners).unwrap(),
            rules_history_sha256: cpu_task::history_sha(root).unwrap(),
            prefix: prefix.clone(),
            root_moves: restriction.clone(),
            claimed_line: Vec::new(),
            current_binary_sha256: SEMANTIC_BINARY.into(),
            max_wall_time_ms: WALL,
            max_output_bytes: MAX_SEMANTIC_RECEIPT_BYTES,
            context_sha256: String::new(),
        };
        semantic_request.context_sha256 =
            semantic::request_context_sha256(&semantic_request).unwrap();
        // Actual existing Rules preparation; not a fabricated receipt or CPU run.
        let mut semantic = semantic::prepare_started(
            &serde_json::to_vec(&semantic_request).unwrap(),
            SEMANTIC_BINARY,
            Instant::now(),
        )
        .unwrap();
        semantic.extend_from_slice(b" \n");
        let receipt: SemanticReceipt = serde_json::from_slice(&semantic).unwrap();
        let mut cpu = cpu_task::Request {
            schema: cpu_task::CPU_TASK_SCHEMA.into(),
            ordering_policy: None,
            task: cpu_task::TaskKind::DefendResponse,
            parent_input_sha256: semantic_request.parent_input_sha256.clone(),
            position_command: command.into(),
            expected_board_fen: semantic_request.expected_board_fen,
            rules_state_sha256: semantic_request.rules_state_sha256,
            rules_history_sha256: semantic_request.rules_history_sha256,
            cpu_binary_sha256: OLD_BINARY.into(),
            branch_sha256: String::new(),
            prefix,
            root_moves: restriction,
            baseline_depth: 1,
            requested_depth: 2,
            max_nodes_per_check: 64,
            max_wall_time_ms: WALL,
            max_output_bytes: 65536,
            tt_entries: 32,
            quiescence_ply: 4,
            cpu_profile_sha256: String::new(),
            recheck_profile_sha256: String::new(),
            context_sha256: String::new(),
        };
        cpu.cpu_profile_sha256 =
            cpu_task::json_digest(&cpu_task::profile(&cpu, CpuProfile::PlanAssisted)).unwrap();
        cpu.recheck_profile_sha256 =
            cpu_task::json_digest(&cpu_task::profile(&cpu, CpuProfile::Independent)).unwrap();
        cpu.branch_sha256 = cpu_task::json_digest(&serde_json::json!([cpu_task::BRANCH_DOMAIN, {
            "parent_input_sha256": cpu.parent_input_sha256, "prefix": cpu.prefix, "root_moves": cpu.root_moves,
        }])).unwrap();
        cpu.context_sha256 = seal(cpu_task::CPU_TASK_SCHEMA, &cpu);
        let cpu_raw = serde_json::to_string(&cpu).unwrap() + " \n";
        let binding = ReplayBindingPins {
            query_sha256: "f".repeat(64),
            catalogue_artifact: super::super::pin(b"caller catalogue original"),
            before_result_artifact: super::super::pin(b"caller Query before original"),
            prior_ledger_sha256: "1".repeat(64),
            semantic_input_sha256: "2".repeat(64),
            semantic_context_sha256: receipt.context_sha256.clone(),
            semantic_branch_meaning_sha256: receipt.branch_meaning_sha256.clone(),
            semantic_before_result_anchor_sha256: receipt.before_result_anchor_sha256.clone(),
            cpu_request_artifact: super::super::pin(cpu_raw.as_bytes()),
        };
        let mut action = super::super::Request {
            schema: super::super::SCHEMA.into(),
            query_sha256: binding.query_sha256.clone(),
            catalogue_artifact: binding.catalogue_artifact.clone(),
            before_result_artifact: binding.before_result_artifact.clone(),
            prior_ledger_sha256: binding.prior_ledger_sha256.clone(),
            action: super::super::StrategicAction {
                slot: 0,
                task: super::super::StrategicTaskKind::DefendResponse,
                semantic_input_sha256: binding.semantic_input_sha256.clone(),
                profile_registration: 0,
                baseline_depth: 1,
                requested_depth: 2,
                max_nodes_per_check: 64,
                max_wall_time_ms: WALL,
                max_output_bytes: 65536,
                budget_bucket: 0,
            },
            remaining: super::super::Remaining {
                steps: 1,
                nodes: 192,
                wall_ms: WALL,
                output_bytes: 4 * 1024 * 1024,
            },
            cpu_request_raw: cpu_raw,
            cpu_request_artifact: binding.cpu_request_artifact.clone(),
            context_sha256: String::new(),
        };
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        let action = (serde_json::to_string(&action).unwrap() + "\r\n").into_bytes();
        let artifacts = ReplayRegisteredArtifacts {
            legacy_cpu_binary: artifact(OLD_BINARY, 1234),
            replay_binary: artifact(NEW_BINARY, 2345),
            engine_source: artifact(&"3".repeat(64), 344097),
            wrapper_source: artifact(&"4".repeat(64), 12345),
            replay_source: artifact(&"5".repeat(64), 115515),
            provider_factory_source: artifact(&"6".repeat(64), 23456),
            opponent_recheck_source: None,
        };
        let mut registration = ReplayConsumerRegistration {
            schema: REGISTRATION_SCHEMA.into(),
            binary_pin_scope: BINARY_DECLARATION_SCOPE.into(),
            artifacts: artifacts.clone(),
            legacy_cpu_max_checks: 2,
            legacy_cpu_profile_sha256: cpu.cpu_profile_sha256.clone(),
            provider_factory_id: "caller-registered-factory-fixture-v1".into(),
            authorities: ReplayAuthorities::default(),
            context_sha256: String::new(),
        };
        registration.context_sha256 = seal(REGISTRATION_SCHEMA, &registration);
        let registration_raw = (serde_json::to_string(&registration).unwrap() + " \n").into_bytes();
        let expected = ReplayExpectedPins {
            registration_artifact: super::super::pin(&registration_raw),
            prepared_action_artifact: super::super::pin(&action),
            semantic_receipt_artifact: super::super::pin(&semantic),
            parent: ReplayParentPins {
                parent_input_sha256: cpu.parent_input_sha256,
                current_view_sha256: "7".repeat(64),
                frozen_admission_sha256: "8".repeat(64),
                encoding_sha256: "9".repeat(64),
            },
            binding,
            registered_artifacts: artifacts,
            legacy_cpu_profile_sha256: cpu.cpu_profile_sha256,
            provider_factory_id: registration.provider_factory_id,
            semantic_binary_sha256: SEMANTIC_BINARY.into(),
        };
        let required = repair_replay_requirements(4, 1, 64).unwrap();
        let config = ReplayConfigInput {
            beam_width: 1,
            line_plies: 4,
            max_nodes: 257,
            max_records: 4,
            max_role_calls: required.role_calls(),
            cpu_nodes_per_task: 64,
        };
        let resources = ReplayResourceDeclaration {
            cpu_nodes: match mode {
                ReplayInputMode::ReplyOnly2n => 128,
                ReplayInputMode::RepairEndpoint3n => 192,
                ReplayInputMode::RepairOpponent4n => 256,
            },
            role_calls: required.role_calls(),
            store_nodes: 257,
            store_records: 4,
            required_observations: required.observations(),
            required_line_chunks: required.line_chunks(),
            required_stages: required.stages(),
            max_rounds: 1,
            output_bytes: 4 * 1024 * 1024,
            whole_wall_ms: WALL,
            cleanup_reserve_ms: CLEANUP,
        };
        let mut profile = CpuFreshAssetProfile {
            schema: super::super::native_replay::ASSET_PROFILE_SCHEMA.into(),
            domain: "cpu_fresh".into(),
            provider: "cpu".into(),
            export_manifest: artifact(&"0".repeat(64), 4),
            runtime_library: artifact(&"1".repeat(64), 4),
            model_epoch: "2".repeat(64),
            encoding_semantic_sha256: String::new(),
            intra_threads: 1,
            cache_public_memory: false,
            device_public_memory: false,
            context_sha256: String::new(),
        };
        profile.encoding_semantic_sha256 =
            crate::pals_native::pals_fresh_encoding_semantic_digest()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
        profile.context_sha256 =
            super::super::native_replay::asset_profile_context(&profile).unwrap();
        let profile_raw = (serde_json::to_string(&profile).unwrap() + "\r\n").into_bytes();
        let profile_pin = super::super::pin(&profile_raw);
        Fixture {
            registration: registration_raw,
            action,
            semantic,
            profile_raw,
            expected,
            profile,
            profile_pin,
            config,
            resources,
            mode,
        }
    }
    fn budget() -> ReplayLaunchBudget {
        ReplayLaunchBudget {
            construction: ReplayConstructionBudget {
                max_outer_bytes: MAX_REQUEST_BYTES,
                max_construction_credit_bytes: MAX_CONSTRUCTION_CREDIT_BYTES,
            },
            max_publication_bytes: MAX_PUBLICATION_BYTES,
            max_publication_backing_bytes: MAX_PUBLICATION_BACKING_BYTES,
        }
    }

    fn opponent_fixture() -> Fixture {
        let mut f = fixture(ReplayInputMode::RepairEndpoint3n);
        f.mode = ReplayInputMode::RepairOpponent4n;
        let mut registration: ReplayConsumerRegistration =
            serde_json::from_slice(&f.registration).unwrap();
        registration.schema = f.mode.registration_schema().into();
        registration.artifacts.opponent_recheck_source = Some(artifact(&"a".repeat(64), 4096));
        registration.context_sha256 = seal(&registration.schema, &registration);
        f.registration = (serde_json::to_string(&registration).unwrap() + "\n").into_bytes();
        f.expected.registration_artifact = super::super::pin(&f.registration);
        f.expected.registered_artifacts = registration.artifacts;
        let mut action: super::super::Request = serde_json::from_slice(&f.action).unwrap();
        // The fresh controlled original declares 4N. No production allowance
        // is extended in request construction or durable launch preparation.
        action.remaining.nodes = 256;
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        f.action = (serde_json::to_string(&action).unwrap() + "\r\n").into_bytes();
        f.expected.prepared_action_artifact = super::super::pin(&f.action);
        let required =
            rz_search::pals::engine::replay::repair_opponent_replay_requirements(4, 1, 64).unwrap();
        f.config.max_records = required.records();
        f.config.max_role_calls = required.role_calls();
        f.resources.cpu_nodes = required.cpu_nodes();
        f.resources.role_calls = required.role_calls();
        f.resources.store_records = required.records();
        f.resources.required_observations = required.observations();
        f.resources.required_line_chunks = required.line_chunks();
        f.resources.required_stages = required.stages();
        f
    }

    #[test]
    fn durable_opponent_v2_preserves_originals_and_independent_source_pin() {
        let f = opponent_fixture();
        let root = TempRoot::new();
        let started = Instant::now();
        let bundle = prepare(
            &f,
            &root.path,
            "four-n",
            budget(),
            started,
            &AtomicBool::new(false),
        )
        .unwrap();
        let request: super::replay_inputs::ReplayInputRequest =
            serde_json::from_slice(payload(&bundle, "replay-input.json").as_bytes()).unwrap();
        assert_eq!(request.schema, super::replay_inputs::OPPONENT_SCHEMA);
        assert_eq!(request.mode, ReplayInputMode::RepairOpponent4n);
        assert_eq!(request.registration_raw.as_bytes(), f.registration);
        assert_eq!(request.prepared_action_raw.as_bytes(), f.action);
        assert_eq!(request.semantic_receipt_raw.as_bytes(), f.semantic);
        assert_eq!(request.resources.cpu_nodes, 256);
        assert_eq!(request.resources.required_stages, 4);
        let transport: ExpectedTransportV2 =
            serde_json::from_slice(payload(&bundle, "replay-expected.json").as_bytes()).unwrap();
        assert_eq!(transport.schema, EXPECTED_TRANSPORT_V2_SCHEMA);
        assert_eq!(
            transport
                .replay_expected
                .into_expected()
                .registered_artifacts,
            f.expected.registered_artifacts
        );
        assert_eq!(bundle.original_started(), started);
        assert_eq!(bundle.deadline(), started + Duration::from_millis(WALL));
        assert_eq!(
            bundle.execution_deadline(),
            started + Duration::from_millis(WALL - CLEANUP)
        );
        let checked = check_replay_inputs_with_semantic_scope(
            payload(&bundle, "replay-input.json").as_bytes(),
            &f.expected,
            SemanticReceiptProducerScope::LibraryDispatcherArgument,
            started,
        )
        .unwrap();
        assert_eq!(checked.audit().authorities, ReplayAuthorities::default());
        assert_eq!(checked.audit().actual_utility_groups, 0);
        assert!(
            checked
                .audit()
                .registered_artifacts
                .opponent_recheck_source
                .is_some()
        );
    }

    #[test]
    fn opponent_launch_output_shortage_refuses_before_any_publication() {
        let mut f = opponent_fixture();
        let required = super::replay_inputs::replay_mode_requirements(f.mode, 4, 1, 64).unwrap();
        let output = replay_output_requirements(
            required.actual_role_calls() as usize,
            required.cpu_checks(),
        )
        .unwrap();
        f.resources.output_bytes = output.required_output_bytes() - 1;
        let root = TempRoot::new();
        let error = prepare(
            &f,
            &root.path,
            "shortage",
            budget(),
            Instant::now(),
            &AtomicBool::new(false),
        )
        .err()
        .unwrap();
        assert_eq!(error.stage, "native_output_reservation");
        assert!(!root.path.join("shortage").exists());
    }
    fn prepare(
        f: &Fixture,
        root: &Path,
        name: &str,
        b: ReplayLaunchBudget,
        started: Instant,
        cancel: &AtomicBool,
    ) -> Result<PreparedReplayLaunchBundle, ReplayLaunchPreparationError> {
        prepare_scoped(
            f,
            root,
            name,
            b,
            started,
            cancel,
            SemanticReceiptProducerScope::LibraryDispatcherArgument,
        )
    }
    fn prepare_scoped(
        f: &Fixture,
        root: &Path,
        name: &str,
        b: ReplayLaunchBudget,
        started: Instant,
        cancel: &AtomicBool,
        expected_scope: SemanticReceiptProducerScope,
    ) -> Result<PreparedReplayLaunchBundle, ReplayLaunchPreparationError> {
        let export = root.join("unopened-export.json");
        let runtime = root.join("unopened-runtime-library");
        let cache = root.join("unopened-runtime-cache");
        prepare_replay_launch_bundle(
            ReplayLaunchOriginals {
                replay: ReplayOriginals {
                    registration: &f.registration,
                    prepared_action: &f.action,
                    semantic_receipt: &f.semantic,
                },
                cpu_fresh_profile: &f.profile_raw,
            },
            ReplayLaunchDeclaration {
                replay: ReplayPreparationDeclaration {
                    mode: f.mode,
                    config: &f.config,
                    resources: &f.resources,
                },
                expected_semantic_receipt_producer_scope: expected_scope,
                assets: ReplayLaunchAssetPaths {
                    export_manifest: &export,
                    runtime_library: &runtime,
                    runtime_cache_root: &cache,
                },
                output_root: root,
                bundle_directory_name: name,
                replay_binary_pin_scope: "current_exe_path_hash",
            },
            ReplayLaunchExpected {
                replay: &f.expected,
                cpu_fresh_profile_artifact: &f.profile_pin,
                cpu_fresh_profile: &f.profile,
            },
            b,
            started,
            cancel,
        )
    }
    fn payload<'a>(bundle: &'a PreparedReplayLaunchBundle, name: &str) -> &'a ReplayLaunchPayload {
        bundle
            .payloads()
            .iter()
            .find(|payload| payload.name() == name)
            .unwrap()
    }
    fn empty_evidence(directory: Option<PathBuf>) -> LaunchFailureEvidence {
        LaunchFailureEvidence {
            files: Vec::new(),
            payloads: Vec::new(),
            directory,
            request: None,
            partial_serialization: Vec::new(),
        }
    }
    fn test_clock(started: Instant) -> LaunchClock {
        LaunchClock {
            started,
            execution: started + Duration::from_millis(WALL - CLEANUP),
            whole: started + Duration::from_millis(WALL),
            whole_wall_ms: WALL,
            cleanup_reserve_ms: CLEANUP,
            output_limit: MAX_OUTPUT_BYTES,
        }
    }

    #[test]
    fn durable_reply_and_repair_preserve_original_bytes_and_seal_manifest_last() {
        for mode in [
            ReplayInputMode::ReplyOnly2n,
            ReplayInputMode::RepairEndpoint3n,
        ] {
            let mut f = fixture(mode);
            if matches!(mode, ReplayInputMode::RepairEndpoint3n) {
                // Explicit public-cache true is a distinct exact profile fact;
                // durable preparation must not invent an unsupported-false rule.
                f.profile.cache_public_memory = true;
                f.profile.context_sha256 =
                    super::super::native_replay::asset_profile_context(&f.profile).unwrap();
                f.profile_raw = (serde_json::to_string(&f.profile).unwrap() + "\r\n").into_bytes();
                f.profile_pin = super::super::pin(&f.profile_raw);
            }
            let root = TempRoot::new();
            let started = Instant::now();
            let bundle = prepare(
                &f,
                &root.path,
                "complete",
                budget(),
                started,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(bundle.original_started(), started);
            assert_eq!(bundle.deadline(), started + Duration::from_millis(WALL));
            assert_eq!(
                bundle.execution_deadline(),
                started + Duration::from_millis(WALL - CLEANUP)
            );
            assert_eq!(bundle.payloads().len(), FILE_COUNT);
            assert_eq!(bundle.publication().len(), FILE_COUNT);
            for (name, raw) in [
                ("replay-registration.json", f.registration.as_slice()),
                ("prepared-action.json", f.action.as_slice()),
                ("semantic-receipt.json", f.semantic.as_slice()),
                ("cpu-fresh-profile.json", f.profile_raw.as_slice()),
            ] {
                let entry = payload(&bundle, name);
                assert_eq!(entry.as_bytes(), raw);
                assert_eq!(entry.artifact(), &super::super::pin(raw));
                assert_eq!(fs::read(bundle.directory().join(name)).unwrap(), raw);
            }
            let assets: LaunchAssets =
                serde_json::from_slice(payload(&bundle, "launch-assets.json").as_bytes()).unwrap();
            assert_eq!(assets.cpu_fresh_profile_raw.as_bytes(), f.profile_raw);
            for path in [
                &assets.export_manifest_path,
                &assets.runtime_library_path,
                &assets.runtime_cache_root,
            ] {
                assert!(!Path::new(path).exists());
            }
            let transport: ExpectedTransportV2 =
                serde_json::from_slice(payload(&bundle, "replay-expected.json").as_bytes())
                    .unwrap();
            assert_eq!(
                transport.request_artifact,
                *payload(&bundle, "replay-input.json").artifact()
            );
            assert_eq!(
                transport.launch_asset_artifact,
                *payload(&bundle, "launch-assets.json").artifact()
            );
            assert_eq!(transport.cpu_fresh_profile_artifact, f.profile_pin);
            assert_eq!(
                transport.replay_expected.into_expected().binding,
                f.expected.binding
            );
            let checked = check_replay_inputs_with_semantic_scope(
                payload(&bundle, "replay-input.json").as_bytes(),
                &f.expected,
                SemanticReceiptProducerScope::LibraryDispatcherArgument,
                started,
            )
            .unwrap();
            assert_eq!(checked.resources().cpu_nodes, f.resources.cpu_nodes);
            assert_eq!(checked.resources().whole_wall_ms, WALL);
            assert_eq!(checked.resources().cleanup_reserve_ms, CLEANUP);
            for row in bundle.publication() {
                assert!(row.created && row.sync_completed && row.readback_verified);
                assert_eq!(row.written_bytes as u64, row.artifact.bytes);
                assert_eq!(row.readback_bytes, row.written_bytes);
                assert!(!row.close_result_observed);
            }
            assert_eq!(bundle.payloads().last().unwrap().name(), FINAL_MANIFEST);
            assert_eq!(
                super::super::pin(&fs::read(bundle.directory().join(FINAL_MANIFEST)).unwrap()),
                *bundle.manifest_artifact()
            );
            assert!(!bundle.directory().join(PENDING_MANIFEST).exists());
            assert!(!bundle.directory().join(FAILED_MANIFEST).exists());
            assert_eq!(bundle.manifest().files.len(), FILE_COUNT - 1);
            assert_eq!(bundle.manifest().authorities, ReplayAuthorities::default());
            assert!(!bundle.manifest().spawn_observed && !bundle.manifest().native_result_observed);
            assert!(
                !bundle
                    .manifest()
                    .original_clock_transferable_to_other_process
                    && !bundle.manifest().hostile_path_race_protection_proved
            );
            assert!(
                !bundle.manifest().allocator_peak_observed && !bundle.manifest().rss_peak_observed
            );
            assert!(bundle.manifest().required_native_output_bytes <= f.resources.output_bytes);
            assert_eq!(
                bundle.manifest().context_sha256,
                seal(PREPARATION_MANIFEST_SCHEMA, bundle.manifest())
            );
            #[cfg(unix)]
            assert_eq!(bundle.directory_sync_after_publication(), Some(true));
            #[cfg(not(unix))]
            assert_eq!(bundle.directory_sync_after_publication(), None);
        }
    }

    #[test]
    fn original_resource_shortages_are_not_automatically_expanded() {
        let root = TempRoot::new();
        for kind in 0..6 {
            let mut f = fixture(ReplayInputMode::RepairEndpoint3n);
            match kind {
                0 => f.resources.cpu_nodes = 128,
                1 => f.resources.role_calls = 0,
                2 => f.resources.store_nodes = 256,
                3 => f.resources.store_records = 3,
                4 => f.resources.required_observations = 0,
                _ => f.resources.output_bytes = 1024,
            }
            let error = prepare(
                &f,
                &root.path,
                "short",
                budget(),
                Instant::now(),
                &AtomicBool::new(false),
            )
            .unwrap_err();
            assert!(matches!(
                error.cause.as_ref(),
                ReplayLaunchCause::Construction(_) | ReplayLaunchCause::Output(_)
            ));
            assert_eq!(error.diagnostics.whole_wall_ms, Some(WALL));
            assert!(!root.path.join("short").exists());
            assert!(!error.diagnostics.normal_manifest_published);
        }
        let mut f = fixture(ReplayInputMode::RepairEndpoint3n);
        let mut action: super::super::Request = serde_json::from_slice(&f.action).unwrap();
        action.remaining.nodes = 128;
        action.context_sha256 = seal(super::super::SCHEMA, &action);
        f.action = serde_json::to_vec(&action).unwrap();
        f.expected.prepared_action_artifact = super::super::pin(&f.action);
        let error = prepare(
            &f,
            &root.path,
            "old-two-n",
            budget(),
            Instant::now(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(
            error.cause.as_ref(),
            ReplayLaunchCause::Construction(_)
        ));
        assert!(!root.path.join("old-two-n").exists());
        // The caller's explicit 2N declaration is a distinct mode, not a 3N clip.
        f.mode = ReplayInputMode::ReplyOnly2n;
        f.resources.cpu_nodes = 128;
        assert!(
            prepare(
                &f,
                &root.path,
                "explicit-two-n",
                budget(),
                Instant::now(),
                &AtomicBool::new(false)
            )
            .is_ok()
        );
    }

    #[test]
    fn early_clock_cancel_and_invalid_output_keep_original_deadline_and_scope() {
        let mut f = fixture(ReplayInputMode::ReplyOnly2n);
        let root = TempRoot::new();
        let expired = Instant::now()
            .checked_sub(Duration::from_millis(WALL + 1000))
            .unwrap();
        f.resources.output_bytes = 0;
        let error = prepare(
            &f,
            &root.path,
            "expired",
            budget(),
            expired,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            error.diagnostics.deadline_exceeded
                && error.diagnostics.original_whole_deadline_retained
        );
        assert_eq!(
            error.diagnostics.deadline(),
            Some(expired + Duration::from_millis(WALL))
        );
        assert_eq!(error.diagnostics.output_limit, 1024);
        assert_eq!(
            error.diagnostics.expected_semantic_receipt_producer_scope,
            SemanticReceiptProducerScope::LibraryDispatcherArgument
        );
        assert_eq!(error.diagnostics.original_started(), expired);
        f.resources.output_bytes = MAX_OUTPUT_BYTES;
        let started = Instant::now();
        let error = prepare(
            &f,
            &root.path,
            "cancel",
            budget(),
            started,
            &AtomicBool::new(true),
        )
        .unwrap_err();
        assert!(error.diagnostics.canceled);
        assert_eq!(
            error.diagnostics.execution_deadline(),
            Some(started + Duration::from_millis(WALL - CLEANUP))
        );
        let future = Instant::now() + Duration::from_secs(60);
        let error = prepare(
            &f,
            &root.path,
            "future",
            budget(),
            future,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(
            error.cause.as_ref(),
            ReplayLaunchCause::Refusal("original start is in the future")
        ));
        assert_eq!(
            error.diagnostics.deadline(),
            Some(future + Duration::from_millis(WALL))
        );
        assert!(!error.diagnostics.directory_created);
    }

    #[test]
    fn profile_pin_and_json_causes_are_retained_before_any_filesystem_publication() {
        let root = TempRoot::new();
        let mut f = fixture(ReplayInputMode::ReplyOnly2n);
        f.profile_raw.push(b' '); // Original raw identity changes; independent pin does not.
        let error = prepare(
            &f,
            &root.path,
            "pin",
            budget(),
            Instant::now(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(
            error.cause.as_ref(),
            ReplayLaunchCause::Profile(_)
        ));
        assert!(!root.path.join("pin").exists());
        f.profile_raw = b"{".to_vec();
        f.profile_pin = super::super::pin(&f.profile_raw);
        let error = prepare(
            &f,
            &root.path,
            "json",
            budget(),
            Instant::now(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        let ReplayLaunchCause::Profile(profile) = error.cause.as_ref() else {
            panic!("typed profile cause");
        };
        assert!(matches!(
            profile.source.as_deref(),
            Some(super::super::native_replay::CpuFreshAssetSourceError::Json(
                _
            ))
        ));
        assert!(std::error::Error::source(error.cause.as_ref()).is_some());
        assert!(error.evidence.publication().is_empty());
    }

    #[test]
    fn publication_credit_rejects_before_new_backing_or_directory_and_keeps_constructed_outer() {
        let f = fixture(ReplayInputMode::ReplyOnly2n);
        let root = TempRoot::new();
        let mut small = budget();
        small.max_publication_backing_bytes = METADATA_CREDIT - 1;
        let early = prepare(
            &f,
            &root.path,
            "bootstrap",
            small,
            Instant::now(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(early.evidence.retained_request().is_none());
        assert!(early.evidence.retained_payloads().is_empty());
        small = budget();
        small.max_publication_bytes = 1;
        let started = Instant::now();
        let error = prepare(
            &f,
            &root.path,
            "io-credit",
            small,
            started,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert_eq!(error.stage, "publication_credit");
        let request = error.evidence.retained_request().unwrap();
        assert_eq!(request.artifact(), &super::super::pin(request.as_bytes()));
        assert!(
            error.evidence.retained_payloads().is_empty()
                && error.evidence.partial_serialization().is_empty()
        );
        assert_eq!(
            error.diagnostics.deadline(),
            Some(started + Duration::from_millis(WALL))
        );
        assert!(!root.path.join("io-credit").exists());
    }

    #[test]
    fn create_new_collision_preserves_previous_bundle_and_does_not_quarantine_it() {
        let root = TempRoot::new();
        let f = fixture(ReplayInputMode::ReplyOnly2n);
        let bundle = prepare(
            &f,
            &root.path,
            "collision",
            budget(),
            Instant::now(),
            &AtomicBool::new(false),
        )
        .unwrap();
        let before = fs::read(bundle.directory().join(FINAL_MANIFEST)).unwrap();
        let error = prepare(
            &f,
            &root.path,
            "collision",
            budget(),
            Instant::now(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            matches!(error.cause.as_ref(), ReplayLaunchCause::Io(cause) if cause.kind() == io::ErrorKind::AlreadyExists)
        );
        assert!(error.bundle_directory.is_none());
        assert!(
            !error.diagnostics.directory_created && !error.diagnostics.normal_manifest_quarantined
        );
        assert_eq!(
            fs::read(bundle.directory().join(FINAL_MANIFEST)).unwrap(),
            before
        );
        assert!(!bundle.directory().join(FAILED_MANIFEST).exists());
    }

    #[test]
    fn descriptor_and_single_child_rules_reject_traversal_and_oversized_or_reserved_names() {
        assert!(descriptor(Path::new("relative-model.json")).is_err());
        let root = TempRoot::new();
        // Preserve the raw parent component: PathBuf::push normalizes it when
        // the Windows canonical root has a verbatim prefix.
        let mut lexical = root.path.as_os_str().to_owned();
        lexical.push(std::path::MAIN_SEPARATOR_STR);
        lexical.push("..");
        lexical.push(std::path::MAIN_SEPARATOR_STR);
        lexical.push("other");
        let traversing = PathBuf::from(lexical);
        assert!(
            traversing
                .components()
                .any(|part| matches!(part, Component::ParentDir))
        );
        assert!(descriptor(&traversing).is_err());
        assert!(descriptor(&root.path.join("x".repeat(MAX_PATH_BYTES))).is_err());
        for name in ["", "../other", "child/file", "CON", "lpt9", "a.b"] {
            assert!(safe_directory_name(name).is_err());
        }
        assert!(safe_directory_name(&"x".repeat(129)).is_err());
        assert!(safe_directory_name("own-fixed_01").is_ok());
        assert!(fixed_child(&root.path, "../other").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let invalid = root.path.join(std::ffi::OsString::from_vec(vec![0xff]));
            assert!(descriptor(&invalid).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn linked_output_root_is_rejected_without_touching_its_target() {
        let root = TempRoot::new();
        let target = root.path.join("target");
        fs::create_dir(&target).unwrap();
        let link = root.path.join("linked");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let error = prepare(
            &fixture(ReplayInputMode::ReplyOnly2n),
            &link,
            "bundle",
            budget(),
            Instant::now(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(
            error.cause.as_ref(),
            ReplayLaunchCause::Refusal(_)
        ));
        assert!(!target.join("bundle").exists());
        assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
    }

    #[test]
    fn readback_mismatch_eof_and_trailing_bytes_preserve_actual_counts_and_original_cause() {
        let root = TempRoot::new();
        let payload = owned_original("check.json", b"expected").unwrap();
        let clock = test_clock(Instant::now());
        let cancel = AtomicBool::new(false);
        for (name, raw, count, eof) in [
            ("different", b"changed!".as_slice(), 8, false),
            ("short", b"exp".as_slice(), 3, true),
            ("trailing", b"expected!".as_slice(), 9, false),
        ] {
            let path = root.path.join(name);
            fs::write(&path, raw).unwrap();
            let mut file = File::open(path).unwrap();
            let mut row = evidence(&payload);
            row.created = true;
            row.written_bytes = raw.len();
            row.sync_completed = true;
            let cause = verify_readback(&mut file, payload.as_bytes(), &mut row, clock, &cancel)
                .unwrap_err();
            assert_eq!(row.readback_bytes, count);
            assert!(!row.readback_verified && !row.close_result_observed);
            if eof {
                assert!(
                    matches!(cause, ReplayLaunchCause::Io(cause) if cause.kind() == io::ErrorKind::UnexpectedEof)
                );
            } else {
                assert!(matches!(cause, ReplayLaunchCause::Refusal(_)));
            }
        }
        let path = root.path.join("existing.json");
        fs::write(&path, b"preserved old data").unwrap();
        let mut row = evidence(&payload);
        assert!(matches!(
            publish_file(
                &root.path,
                "existing.json",
                &payload,
                &mut row,
                clock,
                &cancel
            ),
            Err(ReplayLaunchCause::Io(_))
        ));
        assert!(!row.created && row.written_bytes == 0 && row.readback_bytes == 0);
        assert_eq!(fs::read(path).unwrap(), b"preserved old data");
    }

    #[test]
    fn failed_commit_quarantine_preserves_primary_and_never_claims_unobserved_absence() {
        let root = TempRoot::new();
        let clock = test_clock(Instant::now());
        fs::write(root.path.join(FINAL_MANIFEST), b"failed preparation bytes").unwrap();
        let mut error = launch_error(
            "directory_sync_after_commit",
            ReplayLaunchCause::Io(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "original sync failure",
            )),
            Some(clock),
            SemanticReceiptProducerScope::CurrentExePathHash,
            &AtomicBool::new(false),
            clock.started,
            empty_evidence(Some(root.path.clone())),
        );
        error.diagnostics.normal_manifest_published = true;
        quarantine_manifest(&mut error, Some(clock)).unwrap();
        assert!(
            matches!(error.cause.as_ref(), ReplayLaunchCause::Io(cause) if cause.kind() == io::ErrorKind::PermissionDenied)
        );
        assert!(
            error.diagnostics.normal_manifest_quarantined
                && error.diagnostics.normal_manifest_absence_verified
        );
        assert_eq!(
            fs::read(root.path.join(FAILED_MANIFEST)).unwrap(),
            b"failed preparation bytes"
        );
        fs::write(root.path.join(FINAL_MANIFEST), b"late normal name").unwrap();
        error.diagnostics.normal_manifest_absence_verified = false;
        error.diagnostics.normal_manifest_quarantined = false;
        let secondary = quarantine_manifest(&mut error, Some(clock)).unwrap_err();
        error.quarantine_error = Some(Box::new(secondary));
        assert!(
            !error.diagnostics.normal_manifest_absence_verified
                && !error.diagnostics.normal_manifest_quarantined
        );
        assert_eq!(
            fs::read(root.path.join(FINAL_MANIFEST)).unwrap(),
            b"late normal name"
        );
        assert_eq!(
            fs::read(root.path.join(FAILED_MANIFEST)).unwrap(),
            b"failed preparation bytes"
        );
        let expired = LaunchClock {
            whole: Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
            ..clock
        };
        assert!(quarantine_manifest(&mut error, Some(expired)).is_err());
        assert!(!error.diagnostics.normal_manifest_absence_verified);
    }

    #[test]
    fn escaped_wire_count_and_partial_serialization_use_exact_finite_backing() {
        let cancel = AtomicBool::new(false);
        let clock = test_clock(Instant::now());
        let value = "original \"quote\"\n\t\\\u{0} 한글";
        let expected = serde_json::to_vec(value).unwrap();
        let counted = count_wire(&value, expected.len(), clock, &cancel).unwrap();
        assert_eq!(counted, super::super::pin(&expected));
        let mut exact = reserve_bytes(expected.len()).unwrap();
        wire_into(&value, &mut exact, expected.len(), clock, &cancel).unwrap();
        assert_eq!(exact, expected);
        assert!(matches!(
            count_wire(&value, expected.len() - 1, clock, &cancel),
            Err(ReplayLaunchCause::Json(_))
        ));
        let mut partial = reserve_bytes(expected.len() - 1).unwrap();
        let cause =
            wire_into(&value, &mut partial, expected.len() - 1, clock, &cancel).unwrap_err();
        assert!(matches!(&cause, ReplayLaunchCause::Json(_)));
        assert!(!partial.is_empty() && partial.len() < expected.len());
        let error = launch_error(
            "expected_serialization",
            cause,
            Some(clock),
            SemanticReceiptProducerScope::LinuxLoadedExecutableInode,
            &cancel,
            clock.started,
            LaunchFailureEvidence {
                partial_serialization: partial,
                ..empty_evidence(None)
            },
        );
        assert!(!error.evidence.partial_serialization().is_empty());
        assert_eq!(
            error.diagnostics.expected_semantic_receipt_producer_scope,
            SemanticReceiptProducerScope::LinuxLoadedExecutableInode
        );
        assert_eq!(error.diagnostics.deadline(), Some(clock.whole));
        assert!(std::error::Error::source(error.cause.as_ref()).is_some());
    }

    #[test]
    fn shared_closed_codecs_keep_required_fields_and_distinct_scope_domains() {
        let f = fixture(ReplayInputMode::ReplyOnly2n);
        let wire = ExpectedPinsWire::from_expected(&f.expected);
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded: ExpectedPinsWire = serde_json::from_slice(&bytes).unwrap();
        let expected = decoded.into_expected();
        assert_eq!(expected.parent, f.expected.parent);
        assert_eq!(
            expected.registered_artifacts,
            f.expected.registered_artifacts
        );
        assert_eq!(
            expected.registration_artifact,
            f.expected.registration_artifact
        );
        let mut value = serde_json::to_value(&wire).unwrap();
        value["unknown"] = true.into();
        assert!(serde_json::from_value::<ExpectedPinsWire>(value).is_err());
        let text = std::str::from_utf8(&bytes).unwrap();
        let duplicate = format!("{{\"provider_factory_id\":\"duplicate\",{}", &text[1..]);
        assert!(serde_json::from_str::<ExpectedPinsWire>(&duplicate).is_err());
        let mut value = serde_json::to_value(&wire).unwrap();
        value["semantic_binary_sha256"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<ExpectedPinsWire>(value).is_err());
        let mut value = serde_json::to_value(&wire).unwrap();
        value["registration_artifact"]["bytes"] = serde_json::json!(1.5);
        assert!(serde_json::from_value::<ExpectedPinsWire>(value).is_err());
        let transport = ExpectedTransportV2 {
            schema: EXPECTED_TRANSPORT_V2_SCHEMA.into(),
            request_artifact: artifact(NEW_BINARY, 10),
            replay_expected: wire,
            launch_asset_artifact: artifact(NEW_BINARY, 11),
            cpu_fresh_profile_artifact: f.profile_pin,
            cpu_fresh_profile: f.profile,
            whole_wall_ms: WALL,
            cleanup_reserve_ms: CLEANUP,
            output_bytes: MAX_OUTPUT_BYTES,
            replay_binary_pin_scope: "current_exe_path_hash".into(),
            expected_semantic_receipt_producer_scope:
                SemanticReceiptProducerScope::LibraryDispatcherArgument,
        };
        let value = serde_json::to_value(transport).unwrap();
        assert_eq!(value["replay_binary_pin_scope"], "current_exe_path_hash");
        assert_eq!(
            value["expected_semantic_receipt_producer_scope"],
            "dispatcher_compared_verified_argument"
        );
    }

    #[test]
    fn independent_semantic_scope_mismatch_keeps_checker_cause_and_never_publishes() {
        let f = fixture(ReplayInputMode::ReplyOnly2n);
        let root = TempRoot::new();
        let started = Instant::now();
        let scope = SemanticReceiptProducerScope::LinuxLoadedExecutableInode;
        let error = prepare_scoped(
            &f,
            &root.path,
            "scope",
            budget(),
            started,
            &AtomicBool::new(false),
            scope,
        )
        .unwrap_err();
        assert!(matches!(
            error.cause.as_ref(),
            ReplayLaunchCause::Construction(_)
        ));
        assert_eq!(
            error.diagnostics.expected_semantic_receipt_producer_scope,
            scope
        );
        assert_eq!(
            error.diagnostics.deadline(),
            Some(started + Duration::from_millis(WALL))
        );
        assert_eq!(
            error.diagnostics.execution_deadline(),
            Some(started + Duration::from_millis(WALL - CLEANUP))
        );
        assert!(std::error::Error::source(error.cause.as_ref()).is_some());
        assert!(error.evidence.publication().is_empty());
        assert!(
            !error.diagnostics.directory_created && !error.diagnostics.normal_manifest_published
        );
        assert!(!root.path.join("scope").exists());
    }
}
