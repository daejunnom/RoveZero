//! Explicit original-process generation training transport.
//!
//! JSON declares work; it cannot manufacture a game, frozen P/C, or process
//! capability. This owner retains the actual series/action archives, original
//! process output and read-only checkpoint descriptors. It does not dispatch
//! training as a side effect of parsing, importing, or preparing a command.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(target_os = "linux")]
use super::hex;
use super::{GenerationVerifiedSeriesArchiveV3, VerifiedActionOutcomeArchiveV3, check, invalid};
use crate::ArenaError;
use crate::process::{OriginalProcessOutput, ProcessLimits};
#[cfg(target_os = "linux")]
use rz_experiments::pals_data_v3::generation_identifier;
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(target_os = "linux")]
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Component, Path, PathBuf};
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, atomic::AtomicBool};
#[cfg(target_os = "linux")]
use std::time::Duration;
use std::time::Instant;

const REGISTRATION: &str = "rz-pals-generation-training-window-registration/1";
const STARTUP: &str = "rz-pals-original-generation-trainer-startup/1";
const RESULT: &str = "rz-pals-generation-training-cli-window/1";
const DATA_SCOPE: &str = "external_original_archive_pin_native_chain_readback_only";
const CHECKPOINT: &str = "rz-pals-actual-generation-checkpoint/1";
const MODEL: &str = "rovezero.pals-model-semantics.v3";
const PACKAGE: &str = "experiments/model-research/pals/src";
const CATALOG: &str = "crates/rz-experiments/src/pals_source_inventory_v5.json";
const COMPONENTS: [&str; 7] = [
    "trainer",
    "checkpoint",
    "loss_recipe",
    "model_reference",
    "model_encoding",
    "data_v3",
    "collector",
];
const MAX_JSON: u64 = 16 * 1024 * 1024;
const MAX_CHECKPOINT: u64 = 1024 * 1024 * 1024;
const MAX_SOURCE: u64 = 4 * 1024 * 1024;
const MAX_SOURCES: u64 = 32 * 1024 * 1024;
const MAX_STAGING: u64 = 18 * 1024 * 1024 * 1024;
/// Disk residency and repeated original-corpus I/O are distinct limits. This
/// caps two complete per-game readbacks plus bounded control/output passes.
pub const GENERATION_TRAINING_MAX_READ_BYTES_V3: u64 = 260 * MAX_STAGING;

/// Exact file declaration, verified against a held read-only descriptor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationTrainingFilePinV3 {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
impl GenerationTrainingFilePinV3 {
    pub(crate) fn validate(&self, maximum: u64) -> Result<(), ArenaError> {
        check(
            absolute_safe(&self.path)
                && generation_sha(&self.sha256)
                && self.bytes > 0
                && self.bytes <= maximum,
            "training bounded nonsecret absolute file pin",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationTrainingPythonRuntimeV3 {
    pub version: String,
    pub executable: GenerationTrainingFilePinV3,
}

/// A separately registered, finite launch. The interpreter and registration
/// descriptors are original inputs; the immutable startup envelope is produced
/// from those exact bytes and original deadlines, then pinned before spawning.
/// Dependency roots are explicitly trusted installed libraries. Their inode
/// readback is not an attestation of every Torch/CUDA library in those roots.
pub struct GenerationTrainingLaunchV3 {
    pub interpreter: File,
    pub python_runtime: GenerationTrainingPythonRuntimeV3,
    pub registration: File,
    pub registration_pin: GenerationTrainingFilePinV3,
    pub registration_sha256: String,
    pub execution_id: String,
    pub dependency_directories: Vec<PathBuf>,
    /// Explicit trusted runtime module files (including imported torch/__init__
    /// bytes). Readback covers these files and reported versions/device only;
    /// it does not claim a complete mapped CUDA/cuDNN library inventory.
    pub runtime_module_pins: Vec<GenerationTrainingFilePinV3>,
    pub started: Instant,
    pub execution_deadline: Instant,
    pub whole_deadline: Instant,
    pub limits: ProcessLimits,
    /// Counts preparation and post-closure readback, including original corpus
    /// owners. An interrupted read is conservatively charged, never refunded.
    pub max_read_bytes: u64,
}

/// No successful constructor or deserializer is exposed. This establishes
/// source-owned optimizer dispatch, original process closure and output bytes;
/// it is not neural numerical, arena, strength, or full runtime acceptance.
pub struct VerifiedGenerationTrainingWindowV3 {
    registration: TrainingRegistration,
    receipt: Value,
    parent_receipt: Value,
    process: OriginalProcessOutput,
    series: Arc<GenerationVerifiedSeriesArchiveV3>,
    actions: Option<Arc<VerifiedActionOutcomeArchiveV3>>,
    previous: Option<Arc<VerifiedGenerationTrainingWindowV3>>,
    launch: GenerationTrainingLaunchV3,
    #[cfg(target_os = "linux")]
    held: LinuxHeld,
}
impl VerifiedGenerationTrainingWindowV3 {
    pub fn receipt(&self) -> &Value {
        &self.receipt
    }
    pub fn parent_receipt(&self) -> &Value {
        &self.parent_receipt
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        &self.process
    }
    pub fn series(&self) -> &Arc<GenerationVerifiedSeriesArchiveV3> {
        &self.series
    }
    pub fn actions(&self) -> Option<&Arc<VerifiedActionOutcomeArchiveV3>> {
        self.actions.as_ref()
    }
    pub fn previous(&self) -> Option<&Arc<VerifiedGenerationTrainingWindowV3>> {
        self.previous.as_ref()
    }
    pub fn registration_value(&self) -> Result<Value, ArenaError> {
        serde_json::to_value(&self.registration).map_err(|e| invalid(e.to_string()))
    }
    pub fn registration_sha256(&self) -> &str {
        &self.launch.registration_sha256
    }
    pub fn registration_file_pin(&self) -> &GenerationTrainingFilePinV3 {
        &self.launch.registration_pin
    }
    pub fn execution_id(&self) -> &str {
        &self.launch.execution_id
    }
    pub fn receipt_file_pin(&self) -> Option<&GenerationTrainingFilePinV3> {
        #[cfg(target_os = "linux")]
        {
            self.held.receipt.as_ref().map(|held| &held.pin)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
    #[cfg(target_os = "linux")]
    pub fn held_receipt_file(&self) -> Option<(&GenerationTrainingFilePinV3, &File)> {
        self.held
            .receipt
            .as_ref()
            .map(|held| (&held.pin, &held.file))
    }
    pub(super) fn registration(&self) -> &TrainingRegistration {
        &self.registration
    }
    pub(super) fn launch(&self) -> &GenerationTrainingLaunchV3 {
        &self.launch
    }
    /// Only actually read-back artifacts are exposed. Their tensor/optimizer
    /// semantics remain the source-owned child's scope, rather than a new
    /// checkpoint deserializer in the arena.
    #[cfg(target_os = "linux")]
    pub fn checkpoint_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.held
            .checkpoints
            .iter()
            .map(|held| (&held.pin, &held.file))
    }
    /// Original pinned input descriptors. Consumers retain this opaque trainer
    /// owner and copy descriptor bytes; a same-named reopen is not custody.
    #[cfg(target_os = "linux")]
    pub fn held_input_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.held.inputs.iter().map(|held| (&held.pin, &held.file))
    }
    /// Read-only duplicate of the original staging directory descriptor. This
    /// preserves its inode owner; it does not admit a newly named directory.
    #[cfg(target_os = "linux")]
    pub(crate) fn held_staging_directory_file(&self) -> Result<File, ArenaError> {
        directory_file(&self.held.staging)
    }
    pub fn durable_resume_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            !self.held.checkpoints.is_empty() && self.receipt["durable_checkpoint_boundary"] == true
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

/// Every post-spawn failure preserves the original process/group owner, partial
/// receipt, charged work and archive/checkpoint custody. Inspect pending-child
/// custody through the existing supervisor before reclamation. No retry, GPU
/// release or usage refund is implicit in this type.
pub struct GenerationTrainingFailureV3 {
    pub error: ArenaError,
    pub process: Option<OriginalProcessOutput>,
    pub receipt: Option<Value>,
    pub series: Arc<GenerationVerifiedSeriesArchiveV3>,
    pub actions: Option<Arc<VerifiedActionOutcomeArchiveV3>>,
    pub previous: Option<Arc<VerifiedGenerationTrainingWindowV3>>,
    pub launch: GenerationTrainingLaunchV3,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    /// Raw observed output custody only. This is not a durable checkpoint,
    /// completed optimizer ledger, physical completion or resume capability.
    pub artifact_preservation: Option<Value>,
    // Preserve original descriptor/source/checkpoint custody for the entire
    // failure owner's lifetime. No inspection, retry or release is implicit.
    #[cfg(target_os = "linux")]
    _held: Option<LinuxHeld>,
}
impl std::fmt::Debug for GenerationTrainingFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationTrainingFailureV3")
            .field("error", &self.error)
            .field("process", &self.process)
            .field("receipt_preserved", &self.receipt.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DatasetPin {
    pub(super) manifest: GenerationTrainingFilePinV3,
    pub(super) manifest_sha256: String,
    pub(super) corpus_root: PathBuf,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DatasetPins {
    pub(super) pc: DatasetPin,
    pub(super) v: Option<DatasetPin>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CheckpointPin {
    pub(super) artifact: GenerationTrainingFilePinV3,
    pub(super) domain: String,
    pub(super) mode: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrainingRuntime {
    pub(super) torch: String,
    pub(super) cuda: String,
    pub(super) cudnn: u64,
    pub(super) device_name: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrainingRegistration {
    pub(super) schema: String,
    pub(super) window_id: String,
    pub(super) spec: GenerationTrainingFilePinV3,
    pub(super) phase: String,
    pub(super) datasets: DatasetPins,
    pub(super) checkpoint: CheckpointPin,
    pub(super) source_components: BTreeMap<String, String>,
    pub(super) runtime_sha256: String,
    pub(super) model_sha256: String,
    pub(super) frozen_pc_sha256: Option<String>,
    pub(super) model_epoch: Option<u64>,
    pub(super) training_runtime: TrainingRuntime,
    pub(super) staging_root: PathBuf,
    pub(super) index_root: PathBuf,
    pub(super) checkpoint_directory: PathBuf,
    pub(super) receipt_path: PathBuf,
    pub(super) max_wall_ns: u64,
    pub(super) max_updates: u64,
    pub(super) max_output_bytes: u64,
}
impl TrainingRegistration {
    fn validate(&self) -> Result<(), ArenaError> {
        self.spec.validate(MAX_JSON)?;
        self.datasets.pc.manifest.validate(MAX_JSON)?;
        self.checkpoint.artifact.validate(MAX_CHECKPOINT)?;
        let names = self
            .source_components
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let compiled =
            rz_experiments::compiled_source_components_v5().map_err(|e| invalid(e.to_string()))?;
        check(
            self.schema == REGISTRATION
                && plain_name(&self.window_id)
                && ["pc", "v"].contains(&self.phase.as_str())
                && names == BTreeSet::from(COMPONENTS)
                && self
                    .source_components
                    .iter()
                    .all(|(name, hash)| compiled.get(name) == Some(hash))
                && generation_sha(&self.runtime_sha256)
                && generation_sha(&self.model_sha256)
                && self.training_runtime.torch == "2.8.0"
                && !self.training_runtime.cuda.is_empty()
                && self.training_runtime.cuda.len() <= 64
                && (1..=1_000_000_000).contains(&self.training_runtime.cudnn)
                && self.training_runtime.device_name == "NVIDIA L4"
                && (1..=3_600_000_000_000).contains(&self.max_wall_ns)
                && self.max_updates == 250
                && self.max_output_bytes > MAX_JSON
                && self.max_output_bytes <= MAX_STAGING,
            "training exact source/runtime/profile/update/output registration",
        )?;
        check(
            [
                &self.staging_root,
                &self.index_root,
                &self.checkpoint_directory,
                &self.receipt_path,
            ]
            .into_iter()
            .all(|p| absolute_safe(p))
                && [
                    &self.index_root,
                    &self.checkpoint_directory,
                    &self.receipt_path,
                    &self.checkpoint.artifact.path,
                    &self.datasets.pc.corpus_root,
                ]
                .into_iter()
                .all(|p| p != &self.staging_root && p.starts_with(&self.staging_root))
                && self.receipt_path.extension().is_some_and(|x| x == "json")
                && self.spec.path.extension().is_some_and(|x| x == "json")
                && self
                    .checkpoint
                    .artifact
                    .path
                    .extension()
                    .is_some_and(|x| x == "pt")
                && generation_sha(&self.datasets.pc.manifest_sha256),
            "training managed canonical artifact roots",
        )?;
        if self.phase == "pc" {
            check(
                self.datasets.v.is_none()
                    && self.frozen_pc_sha256.is_none()
                    && self.model_epoch.is_none(),
                "P/C cannot adopt a V corpus/freeze epoch",
            )?;
        } else {
            let v = self
                .datasets
                .v
                .as_ref()
                .ok_or_else(|| invalid("V requires actual action corpus"))?;
            v.manifest.validate(MAX_JSON)?;
            check(
                absolute_safe(&v.corpus_root)
                    && v.corpus_root.starts_with(&self.staging_root)
                    && v.corpus_root != self.staging_root
                    && generation_sha(&v.manifest_sha256)
                    && self
                        .frozen_pc_sha256
                        .as_ref()
                        .is_some_and(|x| generation_sha(x))
                    && self.model_epoch.is_some_and(|x| x > 0),
                "V newly frozen actual corpus pin",
            )?;
        }
        check(
            match self.checkpoint.mode.as_str() {
                "initialize" => {
                    self.phase == "pc"
                        && self.checkpoint.domain == MODEL
                        && self.checkpoint.artifact.sha256 == self.model_sha256
                }
                "resume" => self.checkpoint.domain == CHECKPOINT,
                _ => false,
            },
            "training original checkpoint mode/domain",
        )
    }
    fn digest(&self) -> Result<String, ArenaError> {
        generation_digest(REGISTRATION, self).map_err(|e| invalid(e.to_string()))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildSummary {
    schema: String,
    status: String,
    receipt: GenerationTrainingFilePinV3,
    actual_training_executed: bool,
    physical_completion: String,
    arena_eligible: bool,
}
pub(super) fn decode<T: serde::de::DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    let raw = std::str::from_utf8(raw).map_err(|_| invalid("training strict UTF-8 metadata"))?;
    rz_experiments::decode_json(raw).map_err(|e| invalid(e.to_string()))
}
pub(super) fn plain_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .enumerate()
            .all(|(i, x)| x.is_ascii_alphanumeric() || i > 0 && [b'_', b'-'].contains(&x))
}
fn nonsecret(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    !name.starts_with(".env")
        && ![".key", ".pem", ".p12"]
            .iter()
            .any(|suffix| name.ends_with(*suffix))
        && ![
            "credential",
            "service-account",
            "service_account",
            "serviceaccount",
            "ssh-key",
            "sshkey",
            "api-key",
            "api_key",
            "apikey",
            "client_secret",
            "access_token",
            "refresh_token",
        ]
        .iter()
        .any(|s| name.contains(*s))
}
pub(super) fn absolute_safe(path: &Path) -> bool {
    path.is_absolute()
        && path.components().count() <= 64
        && path.components().all(|part| match part {
            Component::RootDir => true,
            Component::Normal(name) => name
                .to_str()
                .is_some_and(|s| nonsecret(s) && !s.chars().any(char::is_control)),
            _ => false,
        })
}

/// Explicit opt-in transport. The integration owner must hold the sole GPU
/// execution lease throughout this call and retain it with a failure owner if
/// physical completion/group custody is unknown. No GPU authority is inferred
/// from a device name, JSON flag, checkpoint, or successful CPU CI.
pub fn run_generation_training_original_v3(
    launch: GenerationTrainingLaunchV3,
    series: Arc<GenerationVerifiedSeriesArchiveV3>,
    actions: Option<Arc<VerifiedActionOutcomeArchiveV3>>,
    previous: Option<Arc<VerifiedGenerationTrainingWindowV3>>,
    cancel: &AtomicBool,
) -> Result<VerifiedGenerationTrainingWindowV3, Box<GenerationTrainingFailureV3>> {
    #[cfg(target_os = "linux")]
    {
        run_linux(launch, series, actions, previous, cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cancel;
        Err(Box::new(GenerationTrainingFailureV3 {
            error: invalid("training original transport requires Linux"),
            process: None,
            receipt: None,
            series,
            actions,
            previous,
            launch,
            charged_read_bytes: 0,
            read_usage_uncertain: false,
            artifact_preservation: None,
        }))
    }
}

// The fixed isolated bootstrap does not consume stdin or evaluate site/.pth.
// The module receives the original source-owned argv and exact stdin+EOF.
// cwd remains the complete owned staging root for the existing tree supervisor.
const BOOTSTRAP: &str = "import sys,runpy; package=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,package); sys.argv[0]=module; runpy.run_module(module,run_name='__main__',alter_sys=True)";

#[cfg(target_os = "linux")]
pub(super) struct LinuxHeld {
    pub(super) staging: cap_std::fs::Dir,
    pub(super) source_root: PathBuf,
    pub(super) directories: Vec<(PathBuf, File, FileIdentity)>,
    pub(super) inputs: Vec<HeldFile>,
    pub(super) sources: Vec<HeldFile>,
    pub(super) checkpoints: Vec<HeldFile>,
    pub(super) receipt: Option<HeldFile>,
    pub(super) startup: Option<HeldFile>,
    pub(super) consumed_read_bytes: u64,
    // Keep partial/uncertain output descriptors separate from admitted input,
    // receipt and checkpoint files. No success getter exposes these owners.
    _failure_artifacts: Vec<FailureArtifact>,
}
#[cfg(target_os = "linux")]
struct FailureArtifact {
    _path: PathBuf,
    _file: File,
    _identity: Option<FileIdentity>,
    _raw_pin: Option<GenerationTrainingFilePinV3>,
}
#[cfg(target_os = "linux")]
pub(super) struct HeldFile {
    pub(super) file: File,
    pub(super) pin: GenerationTrainingFilePinV3,
    pub(super) identity: FileIdentity,
}
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct FileIdentity {
    pub(super) device: u64,
    pub(super) inode: u64,
    bytes: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}
#[cfg(target_os = "linux")]
impl FileIdentity {
    pub(crate) fn from(meta: &std::fs::Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;
        Self {
            device: meta.dev(),
            inode: meta.ino(),
            bytes: meta.len(),
            mtime: meta.mtime(),
            mtime_ns: meta.mtime_nsec(),
            ctime: meta.ctime(),
            ctime_ns: meta.ctime_nsec(),
        }
    }
    pub(crate) fn same_inode(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }
}
#[cfg(target_os = "linux")]
pub(crate) struct ReadAccounting {
    pub(crate) charged: AtomicU64,
    pub(crate) uncertain: AtomicBool,
}
#[cfg(target_os = "linux")]
impl ReadAccounting {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            charged: AtomicU64::new(0),
            uncertain: AtomicBool::new(false),
        })
    }
}
#[cfg(target_os = "linux")]
pub(crate) struct ReadCredit<'a> {
    pub(crate) remaining: u64,
    pub(crate) maximum: u64,
    pub(crate) until: Instant,
    pub(crate) cancel: &'a AtomicBool,
    pub(crate) account: Arc<ReadAccounting>,
}
#[cfg(target_os = "linux")]
impl ReadCredit<'_> {
    pub(crate) fn take(&mut self, count: u64) -> Result<(), ArenaError> {
        // Remaining credit may only decrease from its original registration;
        // this check never renews credit or creates a larger read allowance.
        check(
            self.remaining <= self.maximum,
            "training remaining read credit exceeds registered maximum",
        )?;
        check(
            Instant::now() < self.until && !self.cancel.load(Ordering::Acquire),
            "training original read deadline/cancel",
        )?;
        self.remaining = self
            .remaining
            .checked_sub(count)
            .ok_or_else(|| invalid("training registered read credit exhausted"))?;
        self.account.charged.fetch_add(count, Ordering::AcqRel);
        Ok(())
    }
    pub(super) fn external(&mut self, result: Result<u64, ArenaError>) -> Result<(), ArenaError> {
        match result {
            Ok(read) => self.take(read),
            // An interrupted owner pass cannot quantify all partial reads.
            // Preserve the failure and consume its entire remaining allowance.
            Err(error) => {
                self.account
                    .charged
                    .fetch_add(self.remaining, Ordering::AcqRel);
                self.account.uncertain.store(true, Ordering::Release);
                self.remaining = 0;
                Err(error)
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn io_error(stage: &str) -> ArenaError {
    ArenaError::Io(format!("training_process.{stage}"))
}
#[cfg(target_os = "linux")]
pub(crate) fn open_directory(path: &Path) -> Result<cap_std::fs::Dir, ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::OpenOptionsExt;
    check(absolute_safe(path), "training safe absolute directory")?;
    let root = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open("/")
        .map_err(|_| io_error("directory_root"))?;
    let mut directory = cap_std::fs::Dir::from_std_file(root);
    for part in path.components() {
        if let Component::Normal(part) = part {
            directory = directory
                .open_dir_nofollow(part)
                .map_err(|_| io_error("directory_nofollow"))?;
        }
    }
    Ok(directory)
}
#[cfg(target_os = "linux")]
pub(crate) fn directory_file(directory: &cap_std::fs::Dir) -> Result<File, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    directory
        .open_with(".", &options)
        .map(|f| f.into_std())
        .map_err(|_| io_error("directory_fd"))
}
#[cfg(target_os = "linux")]
pub(crate) fn open_file(path: &Path) -> Result<File, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    check(absolute_safe(path), "training safe absolute file")?;
    let parent = open_directory(
        path.parent()
            .ok_or_else(|| invalid("training file lacks parent"))?,
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
    parent
        .open_with(
            path.file_name()
                .ok_or_else(|| invalid("training file lacks basename"))?,
            &options,
        )
        .map(|f| f.into_std())
        .map_err(|_| io_error("file_nofollow"))
}
#[cfg(target_os = "linux")]
pub(crate) fn same_named(
    pin: &GenerationTrainingFilePinV3,
    file: &File,
) -> Result<FileIdentity, ArenaError> {
    let identity = FileIdentity::from(&file.metadata().map_err(|_| io_error("held_metadata"))?);
    let current = open_file(&pin.path)?;
    check(
        identity
            == FileIdentity::from(&current.metadata().map_err(|_| io_error("named_metadata"))?),
        "training held/named original file identity differs",
    )?;
    Ok(identity)
}
#[cfg(target_os = "linux")]
pub(crate) fn verify_bytes(
    file: &File,
    pin: &GenerationTrainingFilePinV3,
    retain: bool,
    credit: &mut ReadCredit<'_>,
) -> Result<Vec<u8>, ArenaError> {
    use std::os::unix::fs::{FileExt, MetadataExt};
    let before = file.metadata().map_err(|_| io_error("read_metadata"))?;
    check(
        before.is_file() && before.nlink() == 1 && before.len() == pin.bytes,
        "training exact regular single-link file",
    )?;
    let identity = same_named(pin, file)?;
    let mut hash = Sha256::new();
    let mut bytes = if retain {
        Vec::with_capacity(
            usize::try_from(pin.bytes).map_err(|_| invalid("training metadata allocation"))?,
        )
    } else {
        Vec::new()
    };
    let mut block = [0u8; 65536];
    let mut offset = 0;
    while offset < pin.bytes {
        let count = (pin.bytes - offset).min(block.len() as u64) as usize;
        credit.take(count as u64)?;
        let read = file
            .read_at(&mut block[..count], offset)
            .map_err(|_| io_error("read_at"))?;
        check(read > 0, "training exact pin premature EOF")?;
        hash.update(&block[..read]);
        if retain {
            bytes.extend_from_slice(&block[..read]);
        }
        offset += read as u64;
    }
    check(
        identity
            == FileIdentity::from(&file.metadata().map_err(|_| io_error("readback_metadata"))?)
            && identity == same_named(pin, file)?
            && pin.sha256 == hex(&hash.finalize()),
        "training original file content/identity changed",
    )?;
    credit.take(0)?;
    Ok(bytes)
}
#[cfg(target_os = "linux")]
pub(super) fn hold_input(
    pin: GenerationTrainingFilePinV3,
    retain: bool,
    held: &mut LinuxHeld,
    credit: &mut ReadCredit<'_>,
) -> Result<Vec<u8>, ArenaError> {
    let file = open_file(&pin.path)?;
    let raw = verify_bytes(&file, &pin, retain, credit)?;
    let identity = FileIdentity::from(&file.metadata().map_err(|_| io_error("input_identity"))?);
    held.inputs.push(HeldFile {
        file,
        pin,
        identity,
    });
    Ok(raw)
}
#[cfg(target_os = "linux")]
pub(super) fn retain_directory(path: PathBuf, held: &mut LinuxHeld) -> Result<Value, ArenaError> {
    use std::os::unix::fs::MetadataExt;
    let directory = open_directory(&path)?;
    let file = directory_file(&directory)?;
    let metadata = file
        .metadata()
        .map_err(|_| io_error("directory_identity"))?;
    let identity = FileIdentity::from(&metadata);
    check(metadata.is_dir(), "training directory type")?;
    let declaration =
        serde_json::json!({"path":path,"device":metadata.dev(),"inode":metadata.ino()});
    held.directories.push((path, file, identity));
    Ok(declaration)
}
#[cfg(target_os = "linux")]
pub(super) fn prepare_root(
    registration: &TrainingRegistration,
    execution_id: &str,
) -> Result<LinuxHeld, ArenaError> {
    prepare_original_root(
        &registration.staging_root,
        registration
            .staging_root
            .join(format!("trainer-source-{execution_id}")),
        &registration.receipt_path,
        &[&registration.index_root, &registration.checkpoint_directory],
    )
}

/// Original-process directory custody shared with the non-training G0 path.
/// This creates no trainer, checkpoint or generation capability.
#[cfg(target_os = "linux")]
pub(super) fn prepare_original_root(
    staging_root: &Path,
    source_root: PathBuf,
    receipt_path: &Path,
    output_directories: &[&Path],
) -> Result<LinuxHeld, ArenaError> {
    use std::os::unix::fs::MetadataExt;
    let staging = open_directory(staging_root)?;
    let metadata = directory_file(&staging)?
        .metadata()
        .map_err(|_| io_error("staging_metadata"))?;
    check(
        metadata.mode() & 0o7777 == 0o700
            && metadata.uid() == nix::unistd::Uid::effective().as_raw(),
        "training exclusive current-owner 0700 staging root",
    )?;
    for ancestor in staging_root.ancestors() {
        check(
            open_directory(ancestor)?.symlink_metadata(".git").is_err(),
            "training staging and generated artifacts must remain outside Git",
        )?;
    }
    check(
        !source_root.exists()
            && !receipt_path.exists()
            && output_directories
                .iter()
                .all(|path| source_root.as_path() != *path),
        "training unique source/receipt paths; no overwrite/retry",
    )?;
    let mut held = LinuxHeld {
        staging,
        source_root,
        directories: Vec::new(),
        inputs: Vec::new(),
        sources: Vec::new(),
        checkpoints: Vec::new(),
        receipt: None,
        startup: None,
        consumed_read_bytes: 0,
        _failure_artifacts: Vec::new(),
    };
    retain_directory(staging_root.to_path_buf(), &mut held)?;
    for path in output_directories {
        retain_directory(path.to_path_buf(), &mut held)?;
    }
    Ok(held)
}
#[cfg(target_os = "linux")]
pub(super) fn write_owned_new(path: &Path, raw: &[u8]) -> Result<HeldFile, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    use std::io::Write;
    let parent = open_directory(
        path.parent()
            .ok_or_else(|| invalid("training owned file parent"))?,
    )?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .mode(0o400)
        .follow(FollowSymlinks::No);
    let mut file = parent
        .open_with(
            path.file_name()
                .ok_or_else(|| invalid("training owned file basename"))?,
            &options,
        )
        .map_err(|_| io_error("create_owned_new"))?;
    file.write_all(raw)
        .map_err(|_| io_error("write_owned_new"))?;
    file.sync_all().map_err(|_| io_error("sync_owned_new"))?;
    drop(file);
    let file = open_file(path)?;
    let identity = FileIdentity::from(&file.metadata().map_err(|_| io_error("owned_identity"))?);
    Ok(HeldFile {
        file,
        identity,
        pin: GenerationTrainingFilePinV3 {
            path: path.into(),
            sha256: hex(&Sha256::digest(raw)),
            bytes: raw.len() as u64,
        },
    })
}
#[cfg(target_os = "linux")]
pub(super) fn install_sources(
    held: &mut LinuxHeld,
    credit: &mut ReadCredit<'_>,
) -> Result<(), ArenaError> {
    install_sources_for_components(held, credit, &COMPONENTS)
}
#[cfg(target_os = "linux")]
pub(super) fn install_sources_for_components(
    held: &mut LinuxHeld,
    credit: &mut ReadCredit<'_>,
    components: &[&str],
) -> Result<(), ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::PermissionsExt;
    let available =
        rz_experiments::compiled_source_components_v5().map_err(|e| invalid(e.to_string()))?;
    let selected = components.iter().copied().collect::<BTreeSet<_>>();
    check(
        !components.is_empty()
            && components.len() <= 64
            && selected.len() == components.len()
            && components.iter().all(|name| available.contains_key(*name)),
        "training/export exact known unique source component selection",
    )?;
    let mut source = BTreeMap::<String, &'static [u8]>::new();
    for (component, files) in rz_experiments::SOURCE_COMPONENT_FILES_V5 {
        if selected.contains(component) {
            for (path, bytes) in *files {
                rz_experiments::validate_source_path_v5(path)
                    .map_err(|e| invalid(e.to_string()))?;
                check(
                    !bytes.is_empty() && bytes.len() as u64 <= MAX_SOURCE,
                    "training compiled source per-file cap",
                )?;
                if let Some(previous) = source.insert((*path).into(), *bytes) {
                    check(
                        previous == *bytes,
                        "training conflicting compiled source bytes",
                    )?;
                }
            }
        }
    }
    source.insert(
        CATALOG.into(),
        rz_experiments::PALS_SOURCE_INVENTORY_V5_JSON.as_bytes(),
    );
    check(
        source.len() <= 2048
            && source.values().map(|b| b.len() as u64).sum::<u64>() <= MAX_SOURCES
            && source.contains_key(&format!(
                "{PACKAGE}/rz_pals_model/generation_training_cli.py"
            ))
            && source.contains_key(&format!("{PACKAGE}/rz_pals_model/__init__.py")),
        "training complete compiled registry/module source union",
    )?;
    let root_name = held
        .source_root
        .file_name()
        .ok_or_else(|| invalid("training source root basename"))?;
    held.staging
        .create_dir(root_name)
        .map_err(|_| io_error("source_root_create"))?;
    let source_dir = held
        .staging
        .open_dir_nofollow(root_name)
        .map_err(|_| io_error("source_root_pin"))?;
    directory_file(&source_dir)?
        .set_permissions(std::fs::Permissions::from_mode(0o700))
        .map_err(|_| io_error("source_root_mode"))?;
    let mut directories = BTreeSet::from([held.source_root.clone()]);
    for (logical, bytes) in source {
        credit.take(0)?;
        let mut current = held.source_root.clone();
        let logical = Path::new(&logical);
        for part in logical
            .parent()
            .ok_or_else(|| invalid("training logical source parent"))?
            .components()
        {
            let Component::Normal(part) = part else {
                return Err(invalid("training source path traversal"));
            };
            current.push(part);
            if directories.insert(current.clone()) {
                let parent = open_directory(current.parent().expect("source parent"))?;
                parent
                    .create_dir(part)
                    .map_err(|_| io_error("source_dir_create"))?;
                directory_file(&open_directory(&current)?)?
                    .set_permissions(std::fs::Permissions::from_mode(0o700))
                    .map_err(|_| io_error("source_dir_mode"))?;
            }
        }
        let source = write_owned_new(&held.source_root.join(logical), bytes)?;
        verify_bytes(&source.file, &source.pin, false, credit)?;
        held.sources.push(source);
    }
    // The root/module bytes and path list are immutable for the child. Source
    // publication is a private copy of compiled bytes, never an arbitrary script.
    for path in directories.iter().rev() {
        directory_file(&open_directory(path)?)?
            .set_permissions(std::fs::Permissions::from_mode(0o500))
            .map_err(|_| io_error("source_dir_seal"))?;
    }
    retain_directory(held.source_root.clone(), held)?;
    Ok(())
}
#[cfg(target_os = "linux")]
pub(super) fn revalidate(held: &LinuxHeld, credit: &mut ReadCredit<'_>) -> Result<(), ArenaError> {
    for (path, file, identity) in &held.directories {
        credit.take(0)?;
        let named = directory_file(&open_directory(path)?)?;
        check(
            identity.same_inode(FileIdentity::from(
                &file.metadata().map_err(|_| io_error("held_dir_recheck"))?,
            )) && identity.same_inode(FileIdentity::from(
                &named
                    .metadata()
                    .map_err(|_| io_error("named_dir_recheck"))?,
            )),
            "training retained directory inode changed",
        )?;
    }
    for file in held
        .inputs
        .iter()
        .chain(&held.sources)
        .chain(&held.checkpoints)
        .chain(held.receipt.iter())
        .chain(held.startup.iter())
    {
        check(
            file.identity == same_named(&file.pin, &file.file)?,
            "training retained file identity changed",
        )?;
        verify_bytes(&file.file, &file.pin, false, credit)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub(super) fn reverify_archives(
    series: &GenerationVerifiedSeriesArchiveV3,
    actions: Option<&VerifiedActionOutcomeArchiveV3>,
    credit: &mut ReadCredit<'_>,
) -> Result<(), ArenaError> {
    for game in series.archives() {
        credit.take(0)?;
        let result =
            game.reverify_owned_artifacts(credit.until, Some(credit.cancel), credit.remaining);
        credit.external(result)?;
    }
    if let Some(actions) = actions {
        credit.take(0)?;
        let result = actions.reverify_owned_artifacts(
            credit.until,
            credit.cancel,
            credit.remaining.min(2 * MAX_STAGING),
        );
        credit.external(result)?;
    }
    Ok(())
}
#[cfg(target_os = "linux")]
fn join_archive_inputs(
    registration: &TrainingRegistration,
    series: &Arc<GenerationVerifiedSeriesArchiveV3>,
    actions: Option<&Arc<VerifiedActionOutcomeArchiveV3>>,
    previous: Option<&Arc<VerifiedGenerationTrainingWindowV3>>,
    held: &mut LinuxHeld,
    credit: &mut ReadCredit<'_>,
) -> Result<Value, ArenaError> {
    let series_path = series.path();
    check(
        series.archives().len() == 128
            && series_path.parent() == Some(registration.datasets.pc.corpus_root.as_path())
            && series.dataset_manifest_sha256() == registration.datasets.pc.manifest_sha256,
        "training P/C dataset requires original complete series and exact corpus root",
    )?;
    let raw = hold_input(
        registration.datasets.pc.manifest.clone(),
        true,
        held,
        credit,
    )?;
    let manifest: Value = decode(&raw)?;
    check(
        &manifest == series.dataset_manifest(),
        "training supplied chunk manifest differs from original opaque series",
    )?;
    let index = open_file(&series_path)?;
    let index_pin = GenerationTrainingFilePinV3 {
        path: series_path,
        sha256: series.file_sha256().into(),
        bytes: index
            .metadata()
            .map_err(|_| io_error("series_index_metadata"))?
            .len(),
    };
    index_pin.validate(MAX_JSON)?;
    let raw = hold_input(index_pin, true, held, credit)?;
    let value: Value = decode(&raw)?;
    check(
        &value == series.index(),
        "training original series index raw/semantic identity",
    )?;
    let spec_raw = hold_input(registration.spec.clone(), true, held, credit)?;
    let spec: Value = decode(&spec_raw)?;
    check(
        spec["schema"] == "rz-pals-train-generation/1"
            && spec["dataset_sha256"] == registration.datasets.pc.manifest_sha256
            && spec["model_sha256"] == registration.model_sha256
            && spec["runtime_sha256"] == registration.runtime_sha256
            && spec["device"] == "cuda:0"
            && spec["precision"] == "fp32"
            && spec["tf32"] == false
            && spec["cpu_threads"] == 2
            && spec["microbatch"] == 1
            && spec["accumulation"] == 8
            && spec["checkpoint_interval"] == 250,
        "training source-owned generation recipe/input pins",
    )?;
    check(
        manifest["producer"]["model_sha256"] == registration.model_sha256
            && manifest["producer"]["runtime_sha256"] == registration.runtime_sha256,
        "training original native collection model/runtime meaning",
    )?;
    match (registration.phase.as_str(), actions) {
        ("pc", None) => {}
        ("v", Some(actions)) => {
            let v = registration
                .datasets
                .v
                .as_ref()
                .ok_or_else(|| invalid("training missing V dataset"))?;
            actions.plan().validate()?;
            let expected =
                serde_json::to_value(actions.manifest()).map_err(|e| invalid(e.to_string()))?;
            let semantic = generation_digest(super::GENERATION_CHUNK_DOMAIN, &expected)
                .map_err(|e| invalid(e.to_string()))?;
            check(
                v.corpus_root == actions.directory()
                    && v.manifest.path == actions.directory().join("manifest.json")
                    && v.manifest.sha256 == actions.manifest_file_sha256()
                    && v.manifest_sha256 == semantic
                    && registration.frozen_pc_sha256.as_deref()
                        == Some(actions.plan().frozen_pc_weights_sha256.as_str())
                    && registration.model_epoch == Some(actions.plan().frozen_pc_epoch)
                    && actions.source_game_archives().len() == 128,
                "training V exact original action manifest/frozen epoch/source game owners",
            )?;
            let raw = hold_input(v.manifest.clone(), true, held, credit)?;
            let actual: Value = decode(&raw)?;
            check(
                actual == expected,
                "training V manifest differs from actual collector publication",
            )?;
            let frozen = actions.frozen_intake_receipt()?;
            check(
                frozen["named_shared_pc_parameter_sha256"]
                    == actions.plan().frozen_pc_weights_sha256,
                "training V must retain actual named frozen P/C intake",
            )?;
            let originals = series
                .archives()
                .iter()
                .map(|archive| (archive.request().game_ordinal, archive))
                .collect::<BTreeMap<_, _>>();
            for archive in actions.source_game_archives() {
                check(
                    originals
                        .get(&archive.request().game_ordinal)
                        .is_some_and(|original| Arc::ptr_eq(original, archive)),
                    "training V saved roots must join same original game process evidence",
                )?;
            }
        }
        _ => return Err(invalid("training phase and opaque V action owner mismatch")),
    }
    if registration.checkpoint.mode == "resume" {
        let prior = previous
            .ok_or_else(|| invalid("resume requires previous opaque original trainer owner"))?;
        check(
            Arc::ptr_eq(&prior.series, series)
                && prior.registration.spec == registration.spec
                && prior.registration.model_sha256 == registration.model_sha256
                && prior.registration.runtime_sha256 == registration.runtime_sha256
                && prior.registration.source_components == registration.source_components
                && prior.durable_resume_available()
                && prior.process.transport_complete(),
            "training resume requires same series/spec/source/runtime durable previous window",
        )?;
        check(
            prior
                .held
                .checkpoints
                .last()
                .is_some_and(|file| file.pin == registration.checkpoint.artifact),
            "training resume must use the exact last original read-back checkpoint",
        )?;
        if prior.registration.phase == "v" {
            check(
                prior
                    .actions
                    .as_ref()
                    .zip(actions)
                    .is_some_and(|(a, b)| Arc::ptr_eq(a, b))
                    && prior.registration.frozen_pc_sha256 == registration.frozen_pc_sha256
                    && prior.registration.model_epoch == registration.model_epoch,
                "training V resume frozen/action authority mismatch",
            )?;
        } else if registration.phase == "v" {
            check(
                prior.receipt["window"]["ledger"]["phase"] == "v"
                    && prior.receipt["window"]["ledger"]["completed"]["validator"] == 0,
                "training V starts only from actual completed P/C freeze boundary",
            )?;
        }
        for checkpoint in &prior.held.checkpoints {
            verify_bytes(&checkpoint.file, &checkpoint.pin, false, credit)?;
        }
    } else {
        check(
            previous.is_none(),
            "initial generation cannot silently replace/resume an earlier window",
        )?;
    }
    hold_input(
        registration.checkpoint.artifact.clone(),
        false,
        held,
        credit,
    )?;
    Ok(spec)
}

#[cfg(target_os = "linux")]
pub(super) fn original_window_ticks(
    started: Instant,
    execution: Instant,
    whole: Instant,
) -> Result<(u64, u64, u64), ArenaError> {
    let tick = nix::time::clock_gettime(nix::time::ClockId::CLOCK_MONOTONIC)
        .map_err(|_| io_error("monotonic_clock"))?;
    let tick = u64::try_from(tick.tv_sec())
        .ok()
        .and_then(|s| s.checked_mul(1_000_000_000))
        .and_then(|s| {
            u64::try_from(tick.tv_nsec())
                .ok()
                .and_then(|n| s.checked_add(n))
        })
        .ok_or_else(|| invalid("training monotonic sample overflow"))?;
    let sampled = Instant::now();
    let to_ns = |duration: Duration| {
        u64::try_from(duration.as_nanos()).map_err(|_| invalid("training duration tick overflow"))
    };
    let since_start = sampled
        .checked_duration_since(started)
        .ok_or_else(|| invalid("training original start is in future"))?;
    let start = tick
        .checked_sub(to_ns(since_start)?)
        .ok_or_else(|| invalid("training original start tick underflow"))?;
    let future = |deadline: Instant| -> Result<u64, ArenaError> {
        tick.checked_add(to_ns(
            deadline
                .checked_duration_since(sampled)
                .ok_or_else(|| invalid("training original deadline expired before dispatch"))?,
        )?)
        .ok_or_else(|| invalid("training absolute original deadline overflow"))
    };
    Ok((start, future(execution)?, future(whole)?))
}

#[cfg(target_os = "linux")]
pub(super) fn read_observed_output(
    pin: GenerationTrainingFilePinV3,
    maximum: u64,
    credit: &mut ReadCredit<'_>,
    retain: bool,
) -> Result<(HeldFile, Vec<u8>), ArenaError> {
    pin.validate(maximum)?;
    let file = open_file(&pin.path)?;
    let raw = verify_bytes(&file, &pin, retain, credit)?;
    let identity = same_named(&pin, &file)?;
    Ok((
        HeldFile {
            file,
            pin,
            identity,
        },
        raw,
    ))
}

/// Failure custody observes only original, fixed output names. A raw pin here
/// describes bytes read from a retained descriptor; it never admits a model,
/// checkpoint, optimizer ledger, physical completion or future resume.
#[cfg(target_os = "linux")]
pub(super) struct FailureOutputPreservation {
    pub(super) report: Value,
    pub(super) error: Option<ArenaError>,
    pub(super) receipt_raw: Option<Vec<u8>>,
}

#[cfg(target_os = "linux")]
fn checkpoint_temporary_name(name: &str, checkpoint: &str) -> bool {
    name.strip_prefix(checkpoint)
        .and_then(|suffix| suffix.strip_prefix('.'))
        .and_then(|suffix| suffix.strip_suffix(".tmp"))
        .is_some_and(|random| {
            random.len() == 8
                && random
                    .bytes()
                    .all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == b'_')
        })
}

#[cfg(target_os = "linux")]
fn retained_output_directory(
    path: &Path,
    held: &mut LinuxHeld,
) -> Result<Option<cap_std::fs::Dir>, ArenaError> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(io_error("preserve_directory_metadata")),
    };
    check(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "training preservation directory cannot be linked or non-directory",
    )?;
    if !held.directories.iter().any(|(named, _, _)| named == path) {
        retain_directory(path.to_path_buf(), held)?;
    }
    let (_, original, identity) = held
        .directories
        .iter()
        .find(|(named, _, _)| named == path)
        .ok_or_else(|| invalid("training preservation directory owner absent"))?;
    let named = directory_file(&open_directory(path)?)?;
    check(
        identity.same_inode(FileIdentity::from(&metadata))
            && identity.same_inode(FileIdentity::from(
                &original
                    .metadata()
                    .map_err(|_| io_error("preserve_held_directory"))?,
            ))
            && identity.same_inode(FileIdentity::from(
                &named
                    .metadata()
                    .map_err(|_| io_error("preserve_named_directory"))?,
            )),
        "training preservation original directory inode changed",
    )?;
    Ok(Some(cap_std::fs::Dir::from_std_file(
        original
            .try_clone()
            .map_err(|_| io_error("preserve_directory_clone"))?,
    )))
}

#[cfg(target_os = "linux")]
fn observe_failure_output(
    path: &Path,
    maximum: u64,
    retain_bytes: bool,
    held: &mut LinuxHeld,
    credit: &mut ReadCredit<'_>,
) -> Result<Option<(GenerationTrainingFilePinV3, Vec<u8>)>, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    use std::os::unix::fs::{FileExt, MetadataExt};
    credit.take(0)?;
    let staging = held
        .directories
        .first()
        .ok_or_else(|| invalid("training preservation original staging owner absent"))?;
    check(
        absolute_safe(path)
            && path.starts_with(&staging.0)
            && path != staging.0.as_path()
            && maximum > 0
            && maximum <= MAX_CHECKPOINT,
        "training preservation bounded original nonsecret path",
    )?;
    let parent = retained_output_directory(
        path.parent()
            .ok_or_else(|| invalid("training preservation parent"))?,
        held,
    )?;
    let Some(parent) = parent else {
        return Ok(None);
    };
    let name = path
        .file_name()
        .ok_or_else(|| invalid("training preservation basename"))?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
    let file = match parent.open_with(name, &options) {
        Ok(file) => file.into_std(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(io_error("preserve_output_nofollow")),
    };
    let custody = held._failure_artifacts.len();
    // Retain before size/type/hash/deadline checks. A zero-byte or two-link
    // interrupted atomic publication remains opaque custody, never HeldFile.
    held._failure_artifacts.push(FailureArtifact {
        _path: path.to_path_buf(),
        _file: file,
        _identity: None,
        _raw_pin: None,
    });
    let before = held._failure_artifacts[custody]
        ._file
        .metadata()
        .map_err(|_| {
            credit.account.uncertain.store(true, Ordering::Release);
            io_error("preserve_output_metadata")
        })?;
    let identity = FileIdentity::from(&before);
    held._failure_artifacts[custody]._identity = Some(identity);
    check(
        before.is_file()
            && before.uid() == nix::unistd::Uid::effective().as_raw()
            && before.len() <= maximum,
        "training failure output bounded regular current-owner file",
    )?;
    let file = &held._failure_artifacts[custody]._file;
    let mut hash = Sha256::new();
    let mut raw = if retain_bytes {
        Vec::with_capacity(
            usize::try_from(before.len())
                .map_err(|_| invalid("training preservation metadata allocation"))?,
        )
    } else {
        Vec::new()
    };
    let mut block = [0u8; 65536];
    let mut offset = 0;
    while offset < before.len() {
        let count = (before.len() - offset).min(block.len() as u64) as usize;
        credit.take(count as u64)?;
        let count = match file.read_at(&mut block[..count], offset) {
            Ok(count) if count > 0 => count,
            _ => {
                credit.account.uncertain.store(true, Ordering::Release);
                return Err(io_error("preserve_output_read_at"));
            }
        };
        hash.update(&block[..count]);
        if retain_bytes {
            raw.extend_from_slice(&block[..count]);
        }
        offset += count as u64;
    }
    credit.take(0)?;
    let pin = GenerationTrainingFilePinV3 {
        path: path.to_path_buf(),
        sha256: hex(&hash.finalize()),
        bytes: before.len(),
    };
    let after = FileIdentity::from(
        &file
            .metadata()
            .map_err(|_| io_error("preserve_output_after"))?,
    );
    if identity != after || identity != same_named(&pin, file)? {
        credit.account.uncertain.store(true, Ordering::Release);
        return Err(invalid(
            "training failure output changed during raw readback",
        ));
    }
    held._failure_artifacts[custody]._raw_pin = Some(pin.clone());
    Ok(Some((pin, raw)))
}

#[cfg(target_os = "linux")]
pub(super) fn preserve_original_outputs(
    targets: &[(PathBuf, u64)],
    checkpoint_temporary: Option<&Path>,
    receipt_path: &Path,
    held: &mut LinuxHeld,
    credit: &mut ReadCredit<'_>,
    original: &OriginalProcessOutput,
) -> FailureOutputPreservation {
    let mut targets = targets.to_vec();
    let mut errors = Vec::new();
    let mut first_error = None;
    let mut receipt_raw = None;
    let mut observed = Vec::new();
    let temporary_result = (|| -> Result<(), ArenaError> {
        check(
            targets.len() <= 128,
            "training preservation fixed target count",
        )?;
        if let Some(checkpoint) = checkpoint_temporary {
            credit.take(0)?;
            let parent = checkpoint
                .parent()
                .ok_or_else(|| invalid("checkpoint temporary parent"))?;
            let name = checkpoint
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| invalid("checkpoint temporary basename"))?;
            if let Some(directory) = retained_output_directory(parent, held)? {
                let entries = directory
                    .entries()
                    .map_err(|_| io_error("preserve_checkpoint_directory"))?;
                for (index, entry) in entries.enumerate() {
                    credit.take(0)?;
                    check(
                        index < 4096 && targets.len() < 128,
                        "training bounded temporary name scan",
                    )?;
                    let entry = entry.map_err(|_| io_error("preserve_checkpoint_entry"))?;
                    let entry_name = entry.file_name();
                    if entry_name
                        .to_str()
                        .is_some_and(|temporary| checkpoint_temporary_name(temporary, name))
                    {
                        targets.push((parent.join(entry_name), MAX_CHECKPOINT));
                    }
                }
            }
        }
        Ok(())
    })();
    if let Err(error) = temporary_result {
        errors.push(error.to_string());
        first_error = Some(error);
    }
    let mut unique = BTreeSet::new();
    for (path, maximum) in targets.into_iter().take(128) {
        if !unique.insert(path.clone()) {
            continue;
        }
        let held_before = held._failure_artifacts.len();
        match observe_failure_output(&path, maximum, path == receipt_path, held, credit) {
            Ok(Some((pin, raw))) => {
                if path == receipt_path {
                    receipt_raw = Some(raw);
                }
                observed.push(
                    serde_json::json!({"path":path,"status":"raw_bytes_readback_only",
                    "descriptor_retained":true,"pin":pin,"admission":"none","quarantined":true}),
                );
            }
            Ok(None) => observed.push(serde_json::json!({"path":path,"status":"not_observed",
                "descriptor_retained":false,"pin":null,"reason":"not_found","admission":"none"})),
            Err(error) => {
                errors.push(error.to_string());
                observed.push(serde_json::json!({"path":path,"status":"readback_unknown",
                    "descriptor_retained":held._failure_artifacts.len()>held_before,"pin":null,
                    "error":error.to_string(),"admission":"none","quarantined":true}));
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    FailureOutputPreservation {
        report: serde_json::json!({"schema":"rz-pals-original-output-failure-custody/1",
            "scope":"registered_names_only_raw_descriptor_custody_not_resume_or_success",
            "observed":observed,"errors":errors,"physical_completion":"unknown",
            "group_cleanup":original.process().receipt.group_cleanup,
            "pending_child_retained":original.process().pending_child.is_some(),
            "transport":original.observation(),"charged_read_bytes":credit.account.charged.load(Ordering::Acquire),
            "read_usage_uncertain":credit.account.uncertain.load(Ordering::Acquire),
            "automatic_retry":false,"durable_resume_available":false}),
        error: first_error,
        receipt_raw,
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservedFile {
    pub(super) path: PathBuf,
    pub(super) sha256: String,
    pub(super) bytes: u64,
    pub(super) device: u64,
    pub(super) inode: u64,
}
impl ObservedFile {
    pub(super) fn pin(&self) -> GenerationTrainingFilePinV3 {
        GenerationTrainingFilePinV3 {
            path: self.path.clone(),
            sha256: self.sha256.clone(),
            bytes: self.bytes,
        }
    }
}
#[cfg(target_os = "linux")]
pub(super) fn readback_matches(observed: &ObservedFile, file: &HeldFile) -> bool {
    observed.pin() == file.pin
        && observed.device == file.identity.device
        && observed.inode == file.identity.inode
}
#[cfg(target_os = "linux")]
pub(super) fn validate_child_readback(
    receipt: &Value,
    registration: &TrainingRegistration,
    launch: &GenerationTrainingLaunchV3,
    held: &LinuxHeld,
    dependencies: &Value,
) -> Result<(), ArenaError> {
    let source = &receipt["active_source"];
    check(
        source["source_components"]
            == serde_json::to_value(&registration.source_components)
                .map_err(|e| invalid(e.to_string()))?,
        "training active original source component readback",
    )?;
    let modules = source["source_modules"]
        .as_array()
        .ok_or_else(|| invalid("training source module readback absent"))?;
    check(
        !modules.is_empty() && modules.len() <= 128,
        "training bounded actual imported module inventory",
    )?;
    let mut names = BTreeSet::new();
    for module in modules {
        let name = module["module"]
            .as_str()
            .ok_or_else(|| invalid("training actual module name"))?;
        let object = module
            .as_object()
            .ok_or_else(|| invalid("training actual module readback object"))?;
        check(
            object.len() == 6
                && (name == "__main__"
                    || name == "rz_pals_model"
                    || name.starts_with("rz_pals_model."))
                && names.insert(name),
            "training original actual module namespace/duplicate",
        )?;
        let observed: ObservedFile = serde_json::from_value(serde_json::json!({"path":module["path"],
            "sha256":module["sha256"],"bytes":module["bytes"],"device":module["device"],"inode":module["inode"]}))
            .map_err(|e| invalid(e.to_string()))?;
        check(
            held.sources
                .iter()
                .any(|file| readback_matches(&observed, file))
                && observed
                    .path
                    .starts_with(held.source_root.join(PACKAGE).join("rz_pals_model")),
            "training actual imported module differs from original compiled held source",
        )?;
        if name == "__main__" {
            check(
                ["generation_training_cli.py", "generation_export_cli.py"]
                    .into_iter()
                    .any(|file| {
                        observed.path
                            == held
                                .source_root
                                .join(PACKAGE)
                                .join("rz_pals_model")
                                .join(file)
                    }),
                "training/export __main__ must be an exact original compiled child entry point",
            )?;
        }
    }
    check(
        [
            "__main__",
            "rz_pals_model.generation_training",
            "rz_pals_model.generation_data",
            "rz_pals_model.generation_chunks",
            "rz_pals_model.generation_export",
            "rz_pals_model.v5_source_registry",
        ]
        .into_iter()
        .all(|name| names.contains(name)),
        "training required loaded source modules absent",
    )?;
    let python = &receipt["interpreter_readback"];
    let observed: ObservedFile =
        serde_json::from_value(python["executable"].clone()).map_err(|e| invalid(e.to_string()))?;
    let mapped = same_named(&launch.python_runtime.executable, &launch.interpreter)?;
    check(
        observed.pin() == launch.python_runtime.executable
            && observed.device == mapped.device
            && observed.inode == mapped.inode
            && python["version"].as_str().is_some_and(|v| {
                v.split_whitespace().next() == Some(launch.python_runtime.version.as_str())
            })
            && python["mapped_executable_identity"]["device"] == mapped.device
            && python["mapped_executable_identity"]["inode"] == mapped.inode
            && python["os"] == "posix"
            && python["platform"] == "linux"
            && python["cwd"] == registration.staging_root.to_string_lossy().as_ref()
            && python["isolated"] == true
            && python["no_site"] == true
            && python["bytecode_disabled"] == true
            && python["dependency_pth_executed"] == false
            && python["dependency_directories"] == *dependencies,
        "training original mapped Python/cwd/isolated dependency readback",
    )?;
    let runtime = &receipt["runtime_readback"];
    let torch = runtime["torch"]
        .as_str()
        .ok_or_else(|| invalid("training actual Torch version absent"))?;
    let torch_path = runtime["torch_module_path"]
        .as_str()
        .ok_or_else(|| invalid("training actual Torch module path absent"))?;
    check(
        torch.split('+').next() == Some(registration.training_runtime.torch.as_str())
            && runtime["cuda"] == registration.training_runtime.cuda
            && runtime["cudnn"] == registration.training_runtime.cudnn
            && runtime["device_name"] == registration.training_runtime.device_name
            && runtime["device_count"] == 1
            && runtime["device"] == "cuda:0"
            && runtime["precision"] == "fp32"
            && runtime["tf32_requested"] == false
            && runtime["cuda_matmul_tf32"] == false
            && runtime["cudnn_tf32"] == false
            && runtime["cpu_threads"] == 2
            && runtime["interop_threads"] == 1
            && runtime["runtime_registration_checked"] == true
            && launch
                .runtime_module_pins
                .iter()
                .any(|pin| pin.path == Path::new(torch_path)),
        "training actual versions/device/thread/precision and pinned Torch source readback",
    )
}

fn role_counts(value: &Value) -> Result<[u64; 3], ArenaError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("training ledger role counters absent"))?;
    check(
        object.len() == 3
            && ["proposer", "critic", "validator"]
                .into_iter()
                .all(|name| object.contains_key(name)),
        "training ledger exact role counters",
    )?;
    let get = |name: &str| {
        value[name]
            .as_u64()
            .ok_or_else(|| invalid("training ledger integer counter"))
    };
    Ok([get("proposer")?, get("critic")?, get("validator")?])
}
fn validate_ledger(
    receipt: &Value,
    spec: &Value,
    previous: Option<&Value>,
) -> Result<u64, ArenaError> {
    let window = &receipt["window"];
    let ledger = &window["ledger"];
    let completed = role_counts(&ledger["completed"])?;
    let dispatched = role_counts(&ledger["dispatched"])?;
    let prior_completed = previous
        .map(|value| role_counts(&value["window"]["ledger"]["completed"]))
        .transpose()?
        .unwrap_or([0; 3]);
    let prior_dispatched = previous
        .map(|value| role_counts(&value["window"]["ledger"]["dispatched"]))
        .transpose()?
        .unwrap_or([0; 3]);
    let checked_sum = |counts: &[u64; 3]| {
        counts
            .iter()
            .try_fold(0u64, |sum, count| sum.checked_add(*count))
            .ok_or_else(|| invalid("training external ledger counter overflow"))
    };
    let completed_total = checked_sum(&completed)?;
    let previous_total = checked_sum(&prior_completed)?;
    let dispatched_total = checked_sum(&dispatched)?;
    let previous_dispatched_total = checked_sum(&prior_dispatched)?;
    check(
        window["schema"] == "rz-pals-generation-window/1"
            && ledger["spec"] == *spec
            && window["completed_updates"] == ledger["completed"]
            && window["usage"] == ledger["cost"]
            && ledger["pending_microbatches"] == 0
            && completed[0] <= 2000
            && completed[1] <= 2000
            && completed[2] <= 1000
            && completed_total >= previous_total
            && completed_total - previous_total <= 250
            && completed
                .iter()
                .zip(dispatched)
                .all(|(done, sent)| *done <= sent)
            && completed
                .iter()
                .zip(prior_completed)
                .all(|(now, old)| *now >= old)
            && dispatched
                .iter()
                .zip(prior_dispatched)
                .all(|(now, old)| *now >= old)
            && dispatched_total
                .checked_sub(previous_dispatched_total)
                .is_some_and(|n| n <= 250)
            && window["actual_training_executed"].as_bool()
                == Some(completed_total > previous_total),
        "training actual charged ledger/update delta/source recipe",
    )?;
    let phase = ledger["phase"]
        .as_str()
        .ok_or_else(|| invalid("training ledger phase absent"))?;
    check(
        match phase {
            "pc" => {
                completed[2] == 0
                    && completed[1] < 2000
                    && completed[0] >= completed[1]
                    && completed[0] <= completed[1] + 1
                    && ledger["next_role"]
                        == if completed[0] == completed[1] {
                            "proposer"
                        } else {
                            "critic"
                        }
            }
            "v" => {
                completed[0] == 2000
                    && completed[1] == 2000
                    && completed[2] < 1000
                    && ledger["next_role"] == "validator"
            }
            "complete" => completed == [2000, 2000, 1000] && ledger["next_role"] == "validator",
            _ => false,
        },
        "training actual P/C alternation/freeze/V terminal scope",
    )?;
    let cost = ledger["cost"]
        .as_object()
        .ok_or_else(|| invalid("training actual ledger costs absent"))?;
    let names = [
        "forward_calls",
        "backward_calls",
        "samples",
        "wall_time_ns",
        "partial_accumulation_ns",
        "replay_ns",
        "masked_groups",
        "output_bytes",
    ];
    check(
        cost.len() == names.len()
            && names
                .into_iter()
                .all(|name| cost.get(name).is_some_and(|v| v.as_u64().is_some())),
        "training bounded nonnegative actual cost counters",
    )?;
    if let Some(previous) = previous {
        for name in names {
            check(
                ledger["cost"][name].as_u64()
                    >= previous["window"]["ledger"]["cost"][name].as_u64(),
                "training resume cannot refund previously charged work",
            )?;
        }
    }
    Ok(completed_total)
}

#[cfg(target_os = "linux")]
fn preserve_checkpoints(
    receipt: &Value,
    registration: &TrainingRegistration,
    held: &mut LinuxHeld,
    credit: &mut ReadCredit<'_>,
) -> Result<(), ArenaError> {
    let checkpoints = receipt["checkpoint_pins"]
        .as_array()
        .ok_or_else(|| invalid("training output checkpoint pin inventory absent"))?;
    check(
        checkpoints.len() <= 1,
        "one finite 250-update window cannot publish multiple checkpoint boundaries",
    )?;
    for checkpoint in checkpoints {
        let observed: ObservedFile =
            serde_json::from_value(checkpoint.clone()).map_err(|e| invalid(e.to_string()))?;
        check(
            observed.path.parent() == Some(registration.checkpoint_directory.as_path())
                && observed.path.extension().is_some_and(|x| x == "pt")
                && observed.path != registration.checkpoint.artifact.path,
            "training new checkpoint path cannot alias/overwrite original input",
        )?;
        let (file, _) = read_observed_output(observed.pin(), MAX_CHECKPOINT, credit, false)?;
        check(
            readback_matches(&observed, &file),
            "training actual original checkpoint inode readback differs",
        )?;
        held.checkpoints.push(file);
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn failure_checkpoint_target(
    generation_id: &str,
    phase: &str,
    prior: Option<&Value>,
    directory: &Path,
) -> Result<PathBuf, ArenaError> {
    let counts = prior
        .map(|prior| role_counts(&prior["window"]["ledger"]["completed"]))
        .transpose()?
        .unwrap_or([0; 3]);
    let total = counts
        .iter()
        .try_fold(0u64, |sum, count| sum.checked_add(*count))
        .ok_or_else(|| invalid("training original failure checkpoint counter overflow"))?;
    let ceiling = match phase {
        "pc" => 4000,
        "v" => 5000,
        _ => return Err(invalid("training original failure checkpoint phase")),
    };
    check(
        plain_name(generation_id)
            && total % 250 == 0
            && total < ceiling
            && counts[0] <= 2000
            && counts[1] <= 2000
            && counts[2] <= 1000
            && (phase != "pc" || counts[2] == 0)
            && (phase != "v" || counts[0] == 2000 && counts[1] == 2000),
        "training failure checkpoint must follow original phase and 250-update boundary",
    )?;
    Ok(directory.join(format!("{generation_id}-{:05}.pt", total + 250)))
}

#[cfg(target_os = "linux")]
fn run_linux(
    launch: GenerationTrainingLaunchV3,
    series: Arc<GenerationVerifiedSeriesArchiveV3>,
    actions: Option<Arc<VerifiedActionOutcomeArchiveV3>>,
    previous: Option<Arc<VerifiedGenerationTrainingWindowV3>>,
    cancel: &AtomicBool,
) -> Result<VerifiedGenerationTrainingWindowV3, Box<GenerationTrainingFailureV3>> {
    use crate::process::{
        MAX_ORIGINAL_INPUT_BYTES, OriginalProcessInput, OriginalProcessWindow,
        OwnedArtifactTreeWatch, supervise_verified_input_in_original_window,
    };
    use std::ffi::OsString;
    let mut held = None;
    let mut process = None;
    let mut receipt = None;
    let mut artifact_preservation = None;
    let account = ReadAccounting::new();
    // Cancellation closes work admission. Read-only terminal preservation may
    // still use its original cleanup reserve; it never dispatches another job.
    let preservation_cancel = AtomicBool::new(false);
    let result = (|| -> Result<(TrainingRegistration, Value), ArenaError> {
        launch.registration_pin.validate(MAX_JSON)?;
        launch
            .python_runtime
            .executable
            .validate(256 * 1024 * 1024)?;
        check(
            generation_sha(&launch.registration_sha256)
                && plain_name(&launch.execution_id)
                && generation_identifier(&launch.execution_id)
                && launch.python_runtime.version.starts_with("3.")
                && launch.python_runtime.version.len() <= 16
                && (1..=8).contains(&launch.dependency_directories.len())
                && (1..=64).contains(&launch.runtime_module_pins.len())
                && (1..=3_600_000).contains(&launch.limits.wall_ms)
                && (1..=15_000).contains(&launch.limits.shutdown_grace_ms)
                && (1..=128 * 1024).contains(&launch.limits.max_output_bytes)
                && launch.limits.max_child_processes == 1
                && launch.max_read_bytes > 0
                && launch.max_read_bytes <= GENERATION_TRAINING_MAX_READ_BYTES_V3,
            "training finite exact launch/interpreter/read/process limits",
        )?;
        let window = OriginalProcessWindow::new(
            launch.started,
            launch.execution_deadline,
            launch.whole_deadline,
        )?;
        check(
            launch.execution_deadline.duration_since(launch.started)
                <= Duration::from_millis(launch.limits.wall_ms)
                && launch
                    .whole_deadline
                    .duration_since(launch.execution_deadline)
                    <= Duration::from_secs(30)
                && launch
                    .whole_deadline
                    .duration_since(launch.execution_deadline)
                    <= Duration::from_millis(launch.limits.shutdown_grace_ms * 2),
            "training original execution and cleanup remain finite before preparation",
        )?;
        let mut credit = ReadCredit {
            remaining: launch.max_read_bytes,
            maximum: launch.max_read_bytes,
            until: launch.execution_deadline,
            cancel,
            account: Arc::clone(&account),
        };
        let raw = verify_bytes(
            &launch.registration,
            &launch.registration_pin,
            true,
            &mut credit,
        )?;
        let registration: TrainingRegistration = decode(&raw)?;
        registration.validate()?;
        check(
            registration.digest()? == launch.registration_sha256
                && launch
                    .execution_deadline
                    .duration_since(launch.started)
                    .as_nanos()
                    <= u128::from(registration.max_wall_ns),
            "training separately registered exact window semantic digest/deadlines",
        )?;
        verify_bytes(
            &launch.interpreter,
            &launch.python_runtime.executable,
            false,
            &mut credit,
        )?;
        held = Some(prepare_root(&registration, &launch.execution_id)?);
        let owner = held.as_mut().expect("original staging owner retained");
        let mut dependencies = Vec::new();
        let mut paths = BTreeSet::new();
        let mut inodes = BTreeSet::new();
        for path in &launch.dependency_directories {
            check(paths.insert(path), "training duplicate dependency root")?;
            let pin = retain_directory(path.clone(), owner)?;
            check(
                inodes.insert((pin["device"].as_u64(), pin["inode"].as_u64())),
                "training dependency root inode alias",
            )?;
            dependencies.push(pin);
        }
        let dependencies = Value::Array(dependencies);
        let mut runtime_names = BTreeSet::new();
        for pin in &launch.runtime_module_pins {
            pin.validate(MAX_CHECKPOINT)?;
            check(
                runtime_names.insert(&pin.path)
                    && launch
                        .dependency_directories
                        .iter()
                        .any(|root| pin.path.starts_with(root)),
                "training registered runtime module belongs to trusted pinned dependency root",
            )?;
            hold_input(pin.clone(), false, owner, &mut credit)?;
        }
        let spec = join_archive_inputs(
            &registration,
            &series,
            actions.as_ref(),
            previous.as_ref(),
            owner,
            &mut credit,
        )?;
        let next_checkpoint = failure_checkpoint_target(
            spec["generation_id"]
                .as_str()
                .ok_or_else(|| invalid("training registered generation identifier absent"))?,
            &registration.phase,
            previous.as_ref().map(|previous| &previous.receipt),
            &registration.checkpoint_directory,
        )?;
        check(
            next_checkpoint != registration.checkpoint.artifact.path,
            "training next output boundary cannot alias original input checkpoint",
        )?;
        reverify_archives(&series, actions.as_deref(), &mut credit)?;
        install_sources(owner, &mut credit)?;
        let (start_ns, execution_ns, whole_ns) = original_window_ticks(
            launch.started,
            launch.execution_deadline,
            launch.whole_deadline,
        )?;
        let startup = serde_json::json!({"schema":STARTUP,"execution_id":launch.execution_id,
            "registration_sha256":launch.registration_sha256,"registration":registration,
            "original_started_ns":start_ns,"execution_until_ns":execution_ns,"whole_until_ns":whole_ns,
            "dataset_pin_scope":DATA_SCOPE,"dependency_directories":dependencies,"python_runtime":launch.python_runtime});
        let raw = serde_json::to_vec(&startup).map_err(|e| invalid(e.to_string()))?;
        check(
            !raw.is_empty() && raw.len() <= MAX_ORIGINAL_INPUT_BYTES,
            "training source-owned original stdin credit",
        )?;
        let startup_pin = write_owned_new(
            &registration
                .staging_root
                .join(format!("trainer-startup-{}.json", launch.execution_id)),
            &raw,
        )?;
        let startup_sha256 = startup_pin.pin.sha256.clone();
        owner.startup = Some(startup_pin);
        revalidate(owner, &mut credit)?;
        let args = [
            OsString::from("-I"),
            OsString::from("-S"),
            OsString::from("-B"),
            OsString::from("-c"),
            OsString::from(BOOTSTRAP),
            owner.source_root.join(PACKAGE).into_os_string(),
            OsString::from("rz_pals_model.generation_training_cli"),
            OsString::from("--execute"),
            OsString::from("--original-trainer-child"),
            OsString::from("--startup-sha256"),
            OsString::from(&startup_sha256),
            OsString::from("--startup-bytes"),
            OsString::from(raw.len().to_string()),
            OsString::from("--execution-until-ns"),
            OsString::from(execution_ns.to_string()),
            OsString::from("--whole-until-ns"),
            OsString::from(whole_ns.to_string()),
        ];
        let watch = OwnedArtifactTreeWatch {
            max_total_bytes: registration.max_output_bytes,
            max_file_bytes: MAX_CHECKPOINT,
            max_files: 4096,
            max_depth: 32,
        };
        // The mapped image is observed by the original supervisor before first
        // stdin bytes. A child-reported sys.executable is not a replacement.
        let image_read_reservation = 2 * launch.python_runtime.executable.bytes;
        credit.take(image_read_reservation)?;
        let mut loaded_credit = ReadCredit {
            remaining: image_read_reservation,
            maximum: image_read_reservation,
            until: launch.execution_deadline,
            cancel,
            account: ReadAccounting::new(),
        };
        let mut loaded = |image: &File| {
            verify_bytes(
                image,
                &launch.python_runtime.executable,
                false,
                &mut loaded_credit,
            )
            .map(|_| ())
            .map_err(|_| ())
        };
        let cwd = directory_file(&owner.staging)?;
        process = Some(supervise_verified_input_in_original_window(
            &launch.interpreter,
            &args,
            &cwd,
            launch.limits,
            Some(cancel),
            &watch,
            OriginalProcessInput {
                bytes: &raw,
                window,
            },
            &mut loaded,
        )?);
        let original = process.as_ref().expect("original process output retained");
        credit.until = launch.whole_deadline;
        credit.cancel = &preservation_cancel;
        // Preserve fixed original output names before parsing stdout. The child
        // can die after an atomic checkpoint write, or while printing summary.
        // All of these owners remain separate from admitted checkpoint custody.
        let preservation = preserve_original_outputs(
            &[
                (registration.receipt_path.clone(), MAX_JSON),
                (
                    registration.receipt_path.with_file_name(format!(
                        "{}.partial",
                        registration
                            .receipt_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .ok_or_else(|| invalid("training receipt original basename"))?
                    )),
                    MAX_JSON,
                ),
                (next_checkpoint.clone(), MAX_CHECKPOINT),
            ],
            Some(&next_checkpoint),
            &registration.receipt_path,
            owner,
            &mut credit,
            original,
        );
        if let Some(raw) = preservation.receipt_raw.as_ref() {
            receipt = decode::<Value>(raw).ok();
        }
        artifact_preservation = Some(preservation.report);
        // Preserve the original stdout/process/status error as the primary
        // failure even if bounded artifact preservation independently failed.
        let summary: ChildSummary = decode(&original.process().stdout)?;
        summary.receipt.validate(MAX_JSON)?;
        check(
            summary.schema == RESULT
                && summary.receipt.path == registration.receipt_path
                && !summary.arena_eligible,
            "training original bounded terminal summary path/scope",
        )?;
        original.checked_loaded_image()?;
        let timing = original.checked_timing()?;
        check(
            timing.original_started() == launch.started
                && timing.execution_deadline() == launch.execution_deadline
                && timing.whole_deadline() == launch.whole_deadline
                && original.transport_complete(),
            "training original process/input/EOF/group/reap/finite timing incomplete",
        )?;
        check(
            [
                "window_complete",
                "window_expired",
                "await_actual_v_collection",
                "generation_updates_complete",
            ]
            .contains(&summary.status.as_str())
                && summary.physical_completion == "confirmed",
            "training original child status/physical completion failed",
        )?;
        if let Some(error) = preservation.error {
            return Err(error);
        }
        let (terminal, raw) =
            read_observed_output(summary.receipt.clone(), MAX_JSON, &mut credit, true)?;
        let decoded: Value = decode(&raw)?;
        owner.receipt = Some(terminal);
        receipt = Some(decoded);
        let decoded = receipt.as_ref().expect("original child receipt retained");
        preserve_checkpoints(decoded, &registration, owner, &mut credit)?;
        check(
            [
                "window_complete",
                "window_expired",
                "await_actual_v_collection",
                "generation_updates_complete",
            ]
            .contains(&summary.status.as_str())
                && summary.physical_completion == "confirmed"
                && decoded["schema"] == RESULT
                && decoded["status"] == summary.status
                && decoded["window_id"] == registration.window_id
                && decoded["registration_sha256"] == launch.registration_sha256
                && decoded["execution_id"] == launch.execution_id
                && decoded["dataset_origin_scope"] == DATA_SCOPE
                && decoded["original_resume_checkpoint"] == startup["registration"]["checkpoint"]
                && decoded["original_execution_until_ns"] == execution_ns
                && decoded["original_whole_until_ns"] == whole_ns
                && decoded["physical_completion"] == "confirmed"
                && decoded["native_game_authority_issued"] == false
                && decoded["original_process_authority_issued"] == false
                && decoded["parent_must_verify_process_and_artifacts"] == true
                && decoded["arena_eligible"] == false
                && decoded["data_readback"]["original_game_process_verified_here"] == false
                && decoded["data_readback"]["head_provenance_and_native_chain_bytes_checked"]
                    == true,
            "training terminal exact original scope/physical completion/registration",
        )?;
        let dataset = if registration.phase == "pc" {
            &registration.datasets.pc
        } else {
            registration
                .datasets
                .v
                .as_ref()
                .expect("V registration validated")
        };
        check(
            decoded["window"]["dataset_sha256"] == dataset.manifest_sha256
                && decoded["data_readback"]["dataset_sha256"] == dataset.manifest_sha256
                && decoded["data_readback"]["phase"] == registration.phase
                && decoded["window"]["status"] == summary.status,
            "training actual phase data readback/window joins original manifest",
        )?;
        let total = validate_ledger(decoded, &spec, previous.as_ref().map(|old| &old.receipt))?;
        let complete = &decoded["window"]["actual_training_executed"];
        check(
            complete.as_bool() == Some(summary.actual_training_executed)
                && decoded["durable_checkpoint_boundary"].as_bool()
                    == Some(!owner.checkpoints.is_empty()),
            "training summary work/durable checkpoint boundary differs",
        )?;
        if let Some(checkpoint) = owner.checkpoints.last() {
            let generation_id = spec["generation_id"]
                .as_str()
                .ok_or_else(|| invalid("training generation id absent"))?;
            check(
                plain_name(generation_id)
                    && total > 0
                    && total % 250 == 0
                    && checkpoint.pin.path.file_name().and_then(|x| x.to_str())
                        == Some(format!("{generation_id}-{total:05}.pt").as_str())
                    && decoded["window"]["checkpoints"] == serde_json::json!([checkpoint.pin.path]),
                "training original checkpoint filename/actual charged optimizer boundary",
            )?;
        } else {
            check(
                decoded["window"]["checkpoints"] == serde_json::json!([]),
                "training unverified checkpoint cannot become durable resume",
            )?;
        }
        validate_child_readback(decoded, &registration, &launch, owner, &dependencies)?;
        check(
            decoded["active_source"]["source_modules"]
                .as_array()
                .is_some_and(|modules| {
                    modules.iter().any(|module| {
                        module["module"] == "__main__"
                            && module["path"]
                                == owner
                                    .source_root
                                    .join(PACKAGE)
                                    .join("rz_pals_model/generation_training_cli.py")
                                    .to_string_lossy()
                                    .as_ref()
                    })
                }),
            "training exact original executed main module readback",
        )?;
        verify_bytes(
            &launch.registration,
            &launch.registration_pin,
            false,
            &mut credit,
        )?;
        verify_bytes(
            &launch.interpreter,
            &launch.python_runtime.executable,
            false,
            &mut credit,
        )?;
        reverify_archives(&series, actions.as_deref(), &mut credit)?;
        revalidate(owner, &mut credit)?;
        owner.consumed_read_bytes = account.charged.load(Ordering::Acquire);
        let mapped = original.checked_loaded_image()?;
        let parent_receipt = serde_json::json!({"schema":"rz-pals-original-generation-training-parent/1",
            "execution_id":launch.execution_id,"registration_sha256":launch.registration_sha256,
            "registration_file":launch.registration_pin,"startup":owner.startup.as_ref().expect("startup held").pin,
            "launcher_sha256":hex(&Sha256::digest(BOOTSTRAP.as_bytes())),"child_receipt":summary.receipt,
            "interpreter":launch.python_runtime,"mapped_interpreter":{"pid":mapped.process_id(),"device":mapped.file_device(),"inode":mapped.file_inode(),"scope":mapped.assurance_scope()},
            "process":original.process().receipt,"transport":original.observation(),"charged_read_bytes":owner.consumed_read_bytes,
            "read_scope":"conservative_reserved_read_credit; mapped_image_allowance_reserved_before_spawn; not_measured_kernel_io",
            "series_index_file_sha256":series.file_sha256(),"pc_dataset_sha256":series.dataset_manifest_sha256(),
            "v_original_action_owner_retained":actions.is_some(),"previous_original_window_retained":previous.is_some(),
            "scope":"source_owned_optimizer_work_original_process_closure_and_pinned_output_bytes_only",
            "runtime_scope":"reported_versions_device_thread_settings_and_registered_module_files_only",
            "full_mapped_cuda_runtime_attestation":false,"sole_gpu_lease_observed_here":false,
            "execution_owner_policy":"integration_owner_single_gpu_execution; retain_failure_until_physical_and_group_custody_resolved",
            "numerical_acceptance":false,"arena_eligible":false,"automatic_retry":false});
        Ok((registration, parent_receipt))
    })();
    if let Some(owner) = held.as_mut() {
        owner.consumed_read_bytes = account.charged.load(Ordering::Acquire);
    }
    match result {
        Ok((registration, parent_receipt)) => Ok(VerifiedGenerationTrainingWindowV3 {
            registration,
            receipt: receipt.expect("successful original receipt retained"),
            parent_receipt,
            process: process.expect("successful original process retained"),
            series,
            actions,
            previous,
            launch,
            held: held.expect("successful original held artifacts retained"),
        }),
        Err(error) => Err(Box::new(GenerationTrainingFailureV3 {
            error,
            process,
            receipt,
            series,
            actions,
            previous,
            launch,
            _held: held,
            charged_read_bytes: account.charged.load(Ordering::Acquire),
            read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
            artifact_preservation,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declarations_do_not_admit_secret_paths_or_unknown_summary_fields() {
        let owned = std::env::temp_dir().join("owned");
        let baseline = owned.join("data.json");
        assert!(baseline.is_absolute());
        for name in [".env", "client_secret.json", "ssh-key.key"] {
            let path = owned.join(name);
            assert!(path.is_absolute());
            assert!(!nonsecret(name));
            assert!(!absolute_safe(&path));
        }
        let parent = owned.join("a").join("..").join("data.json");
        assert!(parent.is_absolute());
        assert!(!absolute_safe(&parent));
        // Actual training/process admission is Linux-native. A Windows drive
        // prefix remains unsupported even when its fixture path is absolute.
        #[cfg(windows)]
        assert!(!absolute_safe(&baseline));
        #[cfg(not(windows))]
        assert!(absolute_safe(&baseline));
        let extra = br#"{"schema":"x","status":"failed","receipt":{"path":"/tmp/owned/receipt.json","sha256":"x","bytes":1},"actual_training_executed":false,"physical_completion":"unknown_quarantined","arena_eligible":false,"native_authority":true}"#;
        assert!(decode::<ChildSummary>(extra).is_err());
    }
    #[test]
    fn cumulative_role_accounting_rejects_negative_float_and_missing_owner_counters() {
        assert!(role_counts(&serde_json::json!({"proposer":0,"critic":0,"validator":0})).is_ok());
        for counts in [
            serde_json::json!({"proposer":-1,"critic":0,"validator":0}),
            serde_json::json!({"proposer":0.0,"critic":0,"validator":0}),
            serde_json::json!({"proposer":0,"critic":0}),
        ] {
            assert!(role_counts(&counts).is_err());
        }
    }
    #[test]
    fn overflowing_external_dispatch_ledger_is_rejected_without_wrapping() {
        let receipt = serde_json::json!({"window":{"ledger":{
            "completed":{"proposer":0,"critic":0,"validator":0},
            "dispatched":{"proposer":u64::MAX,"critic":u64::MAX,"validator":u64::MAX}
        }}});
        assert!(validate_ledger(&receipt, &serde_json::json!({}), None).is_err());
    }
    #[test]
    fn isolated_fixed_bootstrap_preserves_original_stdin_for_the_child() {
        assert!(!BOOTSTRAP.contains("stdin"));
        assert!(!BOOTSTRAP.contains("site"));
        assert!(!BOOTSTRAP.contains("eval"));
        assert!(BOOTSTRAP.contains("runpy.run_module"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failure_checkpoint_names_use_only_the_original_next_registered_boundary() {
        let directory = Path::new("/tmp/owned-training/checkpoints");
        assert_eq!(
            failure_checkpoint_target("g0", "pc", None, directory).unwrap(),
            directory.join("g0-00250.pt")
        );
        let prior = serde_json::json!({"window":{"ledger":{"completed":{
            "proposer":2000,"critic":2000,"validator":0}}}});
        assert_eq!(
            failure_checkpoint_target("g0", "v", Some(&prior), directory).unwrap(),
            directory.join("g0-04250.pt")
        );
        assert!(failure_checkpoint_target("g0", "pc", Some(&prior), directory).is_err());
        assert!(failure_checkpoint_target("../foreign", "pc", None, directory).is_err());
        let overflow = serde_json::json!({"window":{"ledger":{"completed":{
            "proposer":u64::MAX,"critic":u64::MAX,"validator":0}}}});
        assert!(failure_checkpoint_target("g0", "pc", Some(&overflow), directory).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failure_temporary_selector_does_not_scan_other_boundaries_or_unrelated_files() {
        let name = "g0-00250.pt";
        assert!(checkpoint_temporary_name("g0-00250.pt.ab12_cd3.tmp", name));
        for unrelated in [
            "g0-00500.pt.ab12_cd3.tmp",
            "g0-00250.pt.partial",
            "g0-00250.pt.longer_token.tmp",
            "g0-00250.pt.AB12_cd3.tmp",
            "secret.key",
        ] {
            assert!(!checkpoint_temporary_name(unrelated, name));
        }
    }

    #[cfg(target_os = "linux")]
    fn failure_custody_fixture() -> (PathBuf, LinuxHeld) {
        use std::os::unix::fs::DirBuilderExt;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rz-training-failure-custody-{}-{stamp}",
            std::process::id()
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let mut held = LinuxHeld {
            staging: open_directory(&root).unwrap(),
            source_root: root.join("unused-source"),
            directories: Vec::new(),
            inputs: Vec::new(),
            sources: Vec::new(),
            checkpoints: Vec::new(),
            receipt: None,
            startup: None,
            consumed_read_bytes: 0,
            _failure_artifacts: Vec::new(),
        };
        retain_directory(root.clone(), &mut held).unwrap();
        (root, held)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn truncated_stdout_keeps_zero_byte_and_two_link_outputs_out_of_success_custody() {
        let (root, mut held) = failure_custody_fixture();
        let checkpoint = root.join("g0-00250.pt");
        let linked_partial = root.join("g0-00250.pt.ab12_cd3.tmp");
        let empty_partial = root.join("receipt.json.partial");
        std::fs::write(&checkpoint, b"opaque-checkpoint-fixture").unwrap();
        std::fs::hard_link(&checkpoint, &linked_partial).unwrap();
        std::fs::write(&empty_partial, []).unwrap();
        let cancel = AtomicBool::new(false);
        let mut credit = ReadCredit {
            remaining: 1024,
            maximum: 1024,
            until: Instant::now() + Duration::from_secs(5),
            cancel: &cancel,
            account: ReadAccounting::new(),
        };
        for path in [&checkpoint, &linked_partial, &empty_partial] {
            assert!(
                observe_failure_output(path, 1024, false, &mut held, &mut credit)
                    .unwrap()
                    .is_some()
            );
        }
        assert!(decode::<ChildSummary>(b"{\"schema\":").is_err());
        assert_eq!(held._failure_artifacts.len(), 3);
        assert!(held.checkpoints.is_empty() && held.receipt.is_none());
        assert_eq!(
            held._failure_artifacts[2]._raw_pin.as_ref().unwrap().bytes,
            0
        );
        assert!(
            held._failure_artifacts[2]
                ._raw_pin
                .as_ref()
                .unwrap()
                .validate(MAX_JSON)
                .is_err()
        );
        drop(held);
        assert!(
            root.starts_with(std::env::temp_dir())
                && root
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("rz-training-failure-custody-")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn insufficient_read_credit_retains_descriptor_without_inventing_a_raw_pin() {
        let (root, mut held) = failure_custody_fixture();
        let path = root.join("receipt.json.partial");
        std::fs::write(&path, b"unfinished").unwrap();
        let cancel = AtomicBool::new(false);
        let mut credit = ReadCredit {
            remaining: 1,
            maximum: 1,
            until: Instant::now() + Duration::from_secs(5),
            cancel: &cancel,
            account: ReadAccounting::new(),
        };
        assert!(observe_failure_output(&path, 1024, false, &mut held, &mut credit).is_err());
        assert_eq!(held._failure_artifacts.len(), 1);
        assert!(held._failure_artifacts[0]._raw_pin.is_none() && held.checkpoints.is_empty());
        assert_eq!(credit.remaining, 1); // No read occurred and no credit was renewed.
        drop(held);
        assert!(
            root.starts_with(std::env::temp_dir())
                && root
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("rz-training-failure-custody-")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
