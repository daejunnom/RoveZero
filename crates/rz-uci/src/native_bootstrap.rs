//! Explicit, single-session Maia ONNX CPU composition.
//!
//! Assets and the native runtime are loaded once, before protocol service. A
//! search receives only a lightweight adapter to the process-owned physical
//! worker. Normal cancellation audits retain counts and first/last receipts;
//! they are deliberately not a complete request journal. Fatal native causes
//! remain owned alongside B's common-contract failures, including after quit.

use crate::{
    bootstrap::MAX_EVALUATIONS,
    engine::{
        EngineError, EvaluatorFactory, EvaluatorProfile, ManagedEvaluator, OwnerRegistry,
        ProcessClock, SearchAuthority,
    },
};
use rz_contracts::*;
use rz_encoding::classical::HistoryFill;
use rz_eval::{
    asset::{self, MaiaAsset},
    contracts::{
        ClassicalProjection, HOST_BYTES_PER_ITEM, MaiaBinding, PhysicalFailure, encoding_manifest,
    },
    error::{BackendError, CauseCode, FailureKind, FailureStage},
    native_runtime_bridge::{
        NATIVE_RUNTIME_OVERHEAD_BYTES, NativeDiagnosticBatch, NativeDiagnosticKind,
        NativeDiagnosticReceipt, NativeRuntimeBackend, NativeWorkerOrigin, NativeWorkerOwner,
    },
    onnx::{BackendConfig, IO_BYTES_PER_ITEM, OnnxBackend, OrtRuntime},
    runtime_pin::RuntimeLibraryPin,
};
use rz_position::contracts::RulesState;
use rz_runtime::{
    Clock, DrainState, Limits, Resources,
    contracts::{ContractClock, ContractEvaluator, ContractsAdapter, SharedScope},
};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, TryLockError},
    thread,
    time::{Duration, Instant},
};

const MAX_PATH_BYTES: usize = 4096;
const MAX_FATAL_RECEIPTS: usize = 32;
const NATIVE_DIAGNOSTIC_CAPACITY: usize = 32;
const FINAL_COLLECTION_LIMIT: Duration = Duration::from_secs(2);

fn error(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}

/// Private local paths are never rendered by Debug, Display, or parser errors.
#[derive(Clone)]
pub struct NativeConfig {
    source_weights: PathBuf,
    onnx_model: PathBuf,
    export_manifest: PathBuf,
    manifest_sha256: [u8; 32],
    ort_library: PathBuf,
    ort_sha256: [u8; 32],
    output_root: PathBuf,
}
impl fmt::Debug for NativeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeConfig")
            .field("manifest_sha256", &self.manifest_sha256)
            .field("ort_sha256", &self.ort_sha256)
            .finish_non_exhaustive()
    }
}
impl NativeConfig {
    pub fn parse(
        arguments: impl IntoIterator<Item = String>,
    ) -> Result<Self, NativeBootstrapError> {
        let mut native = false;
        let mut source_weights = None;
        let mut onnx_model = None;
        let mut export_manifest = None;
        let mut manifest_sha256 = None;
        let mut ort_library = None;
        let mut ort_sha256 = None;
        let mut output_root = None;
        for argument in arguments {
            if argument == "--onnx-cpu" {
                if std::mem::replace(&mut native, true) {
                    return Err(NativeBootstrapError::Config(
                        "duplicate native provider flag",
                    ));
                }
                continue;
            }
            let (name, value) = argument
                .split_once('=')
                .ok_or(NativeBootstrapError::Config(
                    "native CPU arguments require named asset/hash values",
                ))?;
            match name {
                "--source-weights" => set_path(&mut source_weights, value)?,
                "--onnx-model" => set_path(&mut onnx_model, value)?,
                "--export-manifest" => set_path(&mut export_manifest, value)?,
                "--manifest-sha256" => set_hash(&mut manifest_sha256, value)?,
                "--ort-library" => set_path(&mut ort_library, value)?,
                "--ort-sha256" => set_hash(&mut ort_sha256, value)?,
                "--output-root" => set_path(&mut output_root, value)?,
                _ => {
                    return Err(NativeBootstrapError::Config(
                        "unsupported or mixed native provider argument",
                    ));
                }
            }
        }
        if !native {
            return Err(NativeBootstrapError::Config(
                "explicit --onnx-cpu selection is required",
            ));
        }
        let missing = || {
            NativeBootstrapError::Config(
                "native CPU requires all asset, manifest, runtime hash and private output-root arguments",
            )
        };
        Ok(Self {
            source_weights: source_weights.ok_or_else(missing)?,
            onnx_model: onnx_model.ok_or_else(missing)?,
            export_manifest: export_manifest.ok_or_else(missing)?,
            manifest_sha256: manifest_sha256.ok_or_else(missing)?,
            ort_library: ort_library.ok_or_else(missing)?,
            ort_sha256: ort_sha256.ok_or_else(missing)?,
            output_root: output_root.ok_or_else(missing)?,
        })
    }
}
fn set_path(slot: &mut Option<PathBuf>, value: &str) -> Result<(), NativeBootstrapError> {
    if slot.is_some() {
        return Err(NativeBootstrapError::Config(
            "duplicate native asset argument",
        ));
    }
    if value.is_empty() || value.len() > MAX_PATH_BYTES || value.chars().any(char::is_control) {
        return Err(NativeBootstrapError::Config(
            "empty, oversized or invalid native asset path",
        ));
    }
    *slot = Some(PathBuf::from(value));
    Ok(())
}
fn set_hash(slot: &mut Option<[u8; 32]>, value: &str) -> Result<(), NativeBootstrapError> {
    if slot.is_some() {
        return Err(NativeBootstrapError::Config(
            "duplicate native hash argument",
        ));
    }
    *slot = Some(asset::parse_sha256(value).map_err(|_| {
        NativeBootstrapError::Config(
            "native hash must contain exactly 64 lowercase hexadecimal characters",
        )
    })?);
    Ok(())
}

#[derive(Debug)]
pub enum NativeBootstrapError {
    Config(&'static str),
    Backend(Box<BackendError>),
    Physical(Box<PhysicalFailure>),
    Contract(ContractError),
}
impl fmt::Display for NativeBootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(detail) => f.write_str(detail),
            Self::Backend(error) => fmt::Display::fmt(error, f),
            Self::Physical(error) => match &error.backend {
                Some(backend) => fmt::Display::fmt(backend, f),
                None => fmt::Display::fmt(&error.contract, f),
            },
            Self::Contract(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for NativeBootstrapError {}
impl From<BackendError> for NativeBootstrapError {
    fn from(error: BackendError) -> Self {
        Self::Backend(Box::new(error))
    }
}
impl From<PhysicalFailure> for NativeBootstrapError {
    fn from(error: PhysicalFailure) -> Self {
        Self::Physical(Box::new(error))
    }
}
impl From<ContractError> for NativeBootstrapError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}

/// A bounded aggregate: middle normal audit receipts are not a request journal.
#[derive(Debug, Default)]
pub struct NativeAuditAggregate {
    pub count: u64,
    pub first: Option<NativeDiagnosticReceipt>,
    pub last: Option<NativeDiagnosticReceipt>,
}
impl NativeAuditAggregate {
    fn accept(&mut self, receipt: NativeDiagnosticReceipt) -> Option<NativeDiagnosticReceipt> {
        let Some(count) = self.count.checked_add(1) else {
            return Some(receipt);
        };
        if self.first.is_none() {
            self.first = Some(receipt.clone());
        }
        self.last = Some(receipt);
        self.count = count;
        None
    }
}

/// Typed process evidence. Canceled/expired prelaunch audits are count+first/last
/// aggregates, not full request provenance. Fatal receipts preserve C's bounded
/// original BackendError; ordinary Display never renders its raw native text.
#[derive(Debug)]
pub struct NativeRunReport {
    pub model_manifest: Digest,
    pub backend: Digest,
    pub origin: NativeWorkerOrigin,
    /// Completed results accepted by D, not total physical native invocations.
    pub completed_by_runtime: u64,
    pub first_completed: Option<NativeCompletedReceipt>,
    pub last_completed: Option<NativeCompletedReceipt>,
    pub canceled: NativeAuditAggregate,
    pub expired: NativeAuditAggregate,
    pub failures: Vec<NativeDiagnosticReceipt>,
    pub overflow: Option<NativeDiagnosticReceipt>,
    pub boundary_error: Option<ContractError>,
    pub poison_error: Option<ContractError>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCompletedReceipt {
    pub context: CompletionContext,
    pub actual: ActualCompute,
}
impl NativeRunReport {
    fn new(profile: &EvaluatorProfile, origin: NativeWorkerOrigin) -> Result<Self, ContractError> {
        let mut failures = Vec::new();
        failures
            .try_reserve_exact(MAX_FATAL_RECEIPTS)
            .map_err(|_| {
                error(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "cannot reserve bounded native evidence store",
                )
            })?;
        Ok(Self {
            model_manifest: profile.model.handle().manifest,
            backend: profile.backend,
            origin,
            completed_by_runtime: 0,
            first_completed: None,
            last_completed: None,
            canceled: NativeAuditAggregate::default(),
            expired: NativeAuditAggregate::default(),
            failures,
            overflow: None,
            boundary_error: None,
            poison_error: None,
        })
    }
    pub fn has_failure(&self) -> bool {
        !self.failures.is_empty()
            || self.overflow.is_some()
            || self.boundary_error.is_some()
            || self.poison_error.is_some()
    }
    fn remaining_receipts(&self) -> usize {
        if self.overflow.is_some() {
            0
        } else {
            MAX_FATAL_RECEIPTS + 1 - self.failures.len()
        }
    }
    fn accept(&mut self, receipt: NativeDiagnosticReceipt) -> Option<ContractError> {
        let normal = receipt.kind == NativeDiagnosticKind::DispatchRefused
            && receipt.failure.backend.is_none();
        let aggregate = match (normal, receipt.failure.contract.code) {
            (true, ErrorCode::Canceled) => Some(&mut self.canceled),
            (true, ErrorCode::Expired) => Some(&mut self.expired),
            _ => None,
        };
        if let Some(aggregate) = aggregate {
            if let Some(receipt) = aggregate.accept(receipt) {
                let overflow = error(
                    ErrorCode::ResourceExhausted,
                    Stage::Output,
                    "native normal-audit count exhausted; original overflow receipt retained",
                );
                self.overflow = Some(receipt);
                self.boundary_error.get_or_insert(overflow);
                return Some(overflow);
            }
            return None;
        }
        let failure = receipt.failure.contract;
        if self.failures.len() < MAX_FATAL_RECEIPTS {
            self.failures.push(receipt);
        } else {
            debug_assert!(
                self.overflow.is_none(),
                "collector takes only remaining receipt slots"
            );
            self.overflow = Some(receipt);
            self.boundary_error.get_or_insert(error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native fatal evidence limit reached; first overflow original retained",
            ));
        }
        Some(failure)
    }
}
impl fmt::Display for NativeRunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ONNX CPU evidence: origin={:?} completed_by_runtime={} (fresh fullsteps=1 FP32 Computed; physical invocation count is separate), canceled={} expired={} (count+first/last; no full request journal), fatal={} overflow={} boundary={} poison={}",
            self.origin,
            self.completed_by_runtime,
            self.canceled.count,
            self.expired.count,
            self.failures.len(),
            self.overflow.is_some(),
            self.boundary_error.is_some(),
            self.poison_error.is_some()
        )?;
        if let Some(first) = self.first_completed {
            write!(
                f,
                "; first_completed={:?}/{:?}/{:?}/game={:?}/root={:?}",
                first.context.execution,
                first.context.request.request,
                first.context.request.selection,
                first.context.request.game,
                first.context.request.root
            )?;
        }
        if let Some(last) = self.last_completed {
            write!(
                f,
                "; last_completed={:?}/{:?}/{:?}/game={:?}/root={:?}",
                last.context.execution,
                last.context.request.request,
                last.context.request.selection,
                last.context.request.game,
                last.context.request.root
            )?;
        }
        if let Some(receipt) = self.failures.first().or(self.overflow.as_ref()) {
            write!(f, "; first fatal {:?} {:?}: ", receipt.ticket, receipt.kind)?;
            match &receipt.failure.backend {
                Some(backend) => fmt::Display::fmt(backend, f)?,
                None => fmt::Display::fmt(&receipt.failure.contract, f)?,
            }
        }
        Ok(())
    }
}

struct EvidenceState {
    report: Option<NativeRunReport>,
    closed: Option<ContractError>,
}
/// Retained on final collection timeout, so concurrent late receipts remain
/// owned and may be inspected explicitly instead of disappearing with an error.
#[derive(Clone)]
pub struct NativeEvidenceHandle(Arc<Mutex<EvidenceState>>);
impl NativeEvidenceHandle {
    fn try_state(&self) -> Result<Option<MutexGuard<'_, EvidenceState>>, ContractError> {
        match self.0.try_lock() {
            Ok(state) => Ok(Some(state)),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut state = poisoned.into_inner();
                let boundary = error(
                    ErrorCode::BackendFailure,
                    Stage::Output,
                    "native evidence mutex poisoned; admission closed and originals retained",
                );
                state.closed.get_or_insert(boundary);
                if let Some(report) = state.report.as_mut() {
                    report.poison_error.get_or_insert(boundary);
                }
                Ok(Some(state))
            }
        }
    }
    pub fn try_take_report(&self) -> Result<Option<NativeRunReport>, ContractError> {
        Ok(self.try_state()?.and_then(|mut state| state.report.take()))
    }
    fn admission_error(&self) -> Result<Option<ContractError>, ContractError> {
        self.try_state()?.map(|state| state.closed).ok_or_else(|| {
            error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "native evidence consumer busy; request not admitted",
            )
        })
    }
    fn record_completed(&self, output: &EvalOutput) -> Result<(), ContractError> {
        let Some(mut state) = self.try_state()? else {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native completion evidence consumer busy",
            ));
        };
        let report = state.report.as_mut().ok_or_else(|| {
            error(
                ErrorCode::Canceled,
                Stage::Output,
                "native completion evidence already transferred",
            )
        })?;
        if report.origin != NativeWorkerOrigin::CpuOnnx {
            return Ok(());
        }
        if output.actual.precision != PrecisionProfile::Fp32
            || output.actual.steps != 1
            || !output.actual.full
            || output.actual.provenance != CacheProvenance::Computed
            || output.actual.backend != report.backend
            || output.actual.execution.is_none()
        {
            let boundary = error(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "D-completed native output differs from fresh single-step CPU profile",
            );
            report.boundary_error.get_or_insert(boundary);
            state.closed.get_or_insert(boundary);
            return Err(boundary);
        }
        let receipt = NativeCompletedReceipt {
            context: CompletionContext {
                request: output.context,
                execution: output.actual.execution,
            },
            actual: output.actual,
        };
        if report.first_completed.is_none() {
            report.first_completed = Some(receipt);
        }
        report.last_completed = Some(receipt);
        let count = report.completed_by_runtime.checked_add(1).ok_or_else(|| {
            error(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "native completed-by-runtime count exhausted; last original receipt retained",
            )
        });
        match count {
            Ok(count) => {
                report.completed_by_runtime = count;
                Ok(())
            }
            Err(boundary) => {
                report.boundary_error.get_or_insert(boundary);
                state.closed.get_or_insert(boundary);
                Err(boundary)
            }
        }
    }
    fn collect(&self, owner: &NativeWorkerOwner) -> Result<bool, ContractError> {
        let Some(mut state) = self.try_state()? else {
            return Ok(false);
        };
        if let Some(closed) = state.closed {
            owner.try_close_admission(closed)?;
        }
        let Some(report) = state.report.as_mut() else {
            return Err(error(
                ErrorCode::Canceled,
                Stage::Output,
                "native evidence report already transferred; original owner retained",
            ));
        };
        // One transfer per poll keeps the consumer's first overflow original
        // exclusive. After overflow no further original is removed from C.
        let batch = match owner.try_take_diagnostics(report.remaining_receipts().min(1)) {
            Ok(batch) => batch,
            Err(boundary) => {
                report.boundary_error.get_or_insert(boundary);
                state.closed.get_or_insert(boundary);
                owner.try_close_admission(boundary)?;
                return Err(boundary);
            }
        };
        let Some(NativeDiagnosticBatch {
            entries,
            boundary_error,
            poison_error,
        }) = batch
        else {
            return Ok(false);
        };
        let mut fatal = None;
        for receipt in entries {
            if let Some(error) = report.accept(receipt) {
                fatal.get_or_insert(error);
            }
        }
        if let Some(boundary) = boundary_error {
            report.boundary_error.get_or_insert(boundary);
            fatal.get_or_insert(boundary);
        }
        if let Some(poison) = poison_error {
            report.poison_error.get_or_insert(poison);
            fatal.get_or_insert(poison);
        }
        if let Some(fatal) = fatal {
            state.closed.get_or_insert(fatal);
        }
        if let Some(closed) = state.closed {
            owner.try_close_admission(closed)?;
        }
        Ok(true)
    }
}

/// Both service and native evidence survive compound failure. If collection
/// could not finish within its bound, the original owner and shared evidence
/// capability are retained rather than claiming that physical work drained.
pub struct NativeRunError {
    pub service: Option<Box<EngineError>>,
    pub report: Option<Box<NativeRunReport>>,
    pub collection_error: Option<ContractError>,
    pub retained_evidence: NativeEvidenceHandle,
    retained_owner: NativeWorkerOwner,
}
impl NativeRunError {
    pub fn retained_owner(&self) -> &NativeWorkerOwner {
        &self.retained_owner
    }
}
impl fmt::Debug for NativeRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeRunError")
            .field("service", &self.service)
            .field("report", &self.report)
            .field("collection_error", &self.collection_error)
            .finish_non_exhaustive()
    }
}
impl fmt::Display for NativeRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(service) = &self.service {
            write!(f, "protocol service failed: {service}; ")?;
        }
        if let Some(report) = &self.report {
            fmt::Display::fmt(report, f)?;
        } else {
            f.write_str("native evidence remains owned pending bounded collection")?;
        }
        if let Some(error) = &self.collection_error {
            write!(f, "; collection: {error}")?;
        }
        Ok(())
    }
}
impl std::error::Error for NativeRunError {}

/// One verified session/worker for the executable process. No inference-count
/// lifetime quota: identity ledgers are constant-space and every root has B's
/// own finite 129-request/execution range. Fatal errors close future admission.
pub struct NativeCpuFactory {
    owner: NativeWorkerOwner,
    profile: EvaluatorProfile,
    evidence: NativeEvidenceHandle,
    _runtime: Option<OrtRuntime>,
}
impl NativeCpuFactory {
    pub fn load(
        owners: &OwnerRegistry,
        config: &NativeConfig,
    ) -> Result<Self, NativeBootstrapError> {
        let asset = MaiaAsset::load(
            &config.source_weights,
            &config.onnx_model,
            &config.export_manifest,
        )?;
        if asset.manifest_digest() != config.manifest_sha256 {
            return Err(BackendError::new(
                FailureKind::IdentityMismatch,
                FailureStage::Asset,
                "export manifest file digest differs from explicit bootstrap pin",
            )
            .into());
        }
        let output_root = private_output_root(&config.output_root)?;
        let expected_ort: String = config
            .ort_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let pin =
            RuntimeLibraryPin::copy_verified(&config.ort_library, &output_root, &expected_ort)?;
        let runtime = OrtRuntime::load(&pin)?;
        let mut backend_config = BackendConfig::cpu();
        backend_config.max_batch = 1;
        backend_config.intra_threads = 1;
        backend_config.host_io_bytes = IO_BYTES_PER_ITEM;
        let backend = OnnxBackend::load(&runtime, &asset, backend_config)?;
        let fill = HistoryFill::No;
        let encoding = EncodingHandle {
            owner: owners.allocate()?,
            slot: 0,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(fill),
        };
        let model = ModelHandle {
            owner: owners.allocate()?,
            slot: 0,
            generation: SlotGeneration(1),
            manifest: Digest(asset.manifest_digest()),
        };
        let projection =
            ClassicalProjection::new(MaiaBinding::for_backend(&backend, model, encoding, fill)?);
        let owner = NativeWorkerOwner::from_onnx(backend, projection, NATIVE_DIAGNOSTIC_CAPACITY)?;
        Self::from_owner(owner, Some(runtime)).map_err(Into::into)
    }
    fn from_owner(
        owner: NativeWorkerOwner,
        runtime: Option<OrtRuntime>,
    ) -> Result<Self, ContractError> {
        let profile = EvaluatorProfile {
            model: Arc::clone(owner.projection().model()),
            precision: PrecisionProfile::Fp32,
            backend: owner.projection().backend(),
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            bytes: ByteBudget {
                host: HOST_BYTES_PER_ITEM,
                device: 0,
                pinned: 0,
            },
        };
        let evidence = NativeEvidenceHandle(Arc::new(Mutex::new(EvidenceState {
            report: Some(NativeRunReport::new(&profile, owner.origin())?),
            closed: None,
        })));
        Ok(Self {
            owner,
            profile,
            evidence,
            _runtime: runtime,
        })
    }
    /// Called for serve success and failure. A busy physical owner/evidence lock
    /// is queried without blocking; a bounded failure keeps both capabilities.
    pub fn finish(
        &self,
        served: Result<(), EngineError>,
    ) -> Result<NativeRunReport, NativeRunError> {
        let until = Instant::now() + FINAL_COLLECTION_LIMIT;
        let mut collection_error = None;
        loop {
            match self.evidence.collect(&self.owner) {
                Ok(true) => match self.owner.try_status() {
                    Ok(Some(status))
                        if !status.active
                            && status.reserved_diagnostics == 0
                            && status.occupied_diagnostics == 0 =>
                    {
                        break;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        collection_error = Some(error);
                        break;
                    }
                },
                Ok(false) => {}
                Err(error) => {
                    collection_error = Some(error);
                    break;
                }
            }
            if Instant::now() >= until {
                collection_error = Some(error(
                    ErrorCode::Expired,
                    Stage::Output,
                    "native final collection deadline reached; owner and evidence retained",
                ));
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        // If a worker/owner is still active, leave the report in its shared
        // store. A late collector must not encounter a moved-out destination.
        let report = if collection_error.is_none() {
            match self.evidence.try_take_report() {
                Ok(report) => report,
                Err(error) => {
                    collection_error.get_or_insert(error);
                    None
                }
            }
        } else {
            None
        };
        let service = served.err().map(Box::new);
        if service.is_none()
            && collection_error.is_none()
            && report.as_ref().is_some_and(|report| !report.has_failure())
        {
            return Ok(report.expect("checked report presence"));
        }
        Err(NativeRunError {
            service,
            report: report.map(Box::new),
            collection_error,
            retained_evidence: self.evidence.clone(),
            retained_owner: self.owner.clone(),
        })
    }
}

fn private_output_root(path: &Path) -> Result<PathBuf, BackendError> {
    let boundary = |detail| BackendError::new(FailureKind::Io, FailureStage::Asset, detail);
    let metadata = fs::symlink_metadata(path).map_err(|cause| {
        boundary("native output root is unavailable")
            .with_external_cause(CauseCode::RuntimePath, &cause)
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(boundary(
            "native output root must be an existing real private directory",
        ));
    }
    let canonical = path.canonicalize().map_err(|cause| {
        boundary("cannot resolve native output root")
            .with_external_cause(CauseCode::RuntimePath, &cause)
    })?;
    for ancestor in canonical.ancestors() {
        match fs::symlink_metadata(ancestor.join(".git")) {
            Ok(_) => return Err(boundary("native output root must be outside Git checkouts")),
            Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => {}
            Err(cause) => {
                return Err(
                    boundary("cannot verify native output-root ownership boundary")
                        .with_external_cause(CauseCode::RuntimePath, &cause),
                );
            }
        }
    }
    Ok(canonical)
}

#[derive(Clone)]
struct RuntimeClock(ProcessClock);
impl Clock for RuntimeClock {
    type Tick = MonotonicTick;
    fn now(&self) -> MonotonicTick {
        self.0.now().unwrap_or(MonotonicTick(u64::MAX))
    }
    fn elapsed(&self, earlier: MonotonicTick, later: MonotonicTick) -> Duration {
        Duration::from_nanos(later.0.saturating_sub(earlier.0))
    }
}
impl ContractClock for RuntimeClock {
    fn domain(&self) -> ClockDomain {
        self.0.domain()
    }
}
impl EvaluatorFactory for NativeCpuFactory {
    fn profile(&self) -> EvaluatorProfile {
        self.profile.clone()
    }
    fn input_key(&self, state: &RulesState, legal: &[Move]) -> Result<EvalInputKey, ContractError> {
        self.owner.projection().input_key(state, legal)
    }
    fn create(
        &self,
        clock: ProcessClock,
        authority: SearchAuthority,
    ) -> Result<Box<dyn ManagedEvaluator>, ContractError> {
        self.evidence.collect(&self.owner)?;
        if let Some(error) = self.evidence.admission_error()? {
            return Err(error);
        }
        let high_water = clock.reserve_executions(MAX_EVALUATIONS)?;
        let runtime_clock = RuntimeClock(clock.clone());
        let scope = SharedScope::new(authority.current_scope()?);
        let adapter = ContractsAdapter::with_execution_high_water(
            scope.clone(),
            runtime_clock.clone(),
            1,
            high_water,
        )?;
        let backend = NativeRuntimeBackend::new(self.owner.clone(), runtime_clock)?;
        let evaluator = ContractEvaluator::new(
            adapter,
            backend,
            Limits {
                max_requests: 1,
                max_batch_items: 1,
                max_executions: 1,
                max_batch_wait: Duration::ZERO,
                max_queue_age: Duration::from_secs(30),
                deadline_reserve: Duration::ZERO,
                memory: Resources {
                    host_bytes: HOST_BYTES_PER_ITEM + NATIVE_RUNTIME_OVERHEAD_BYTES,
                    device_bytes: 0,
                    pinned_bytes: 0,
                },
            },
            256,
        )?;
        Ok(Box::new(NativeCpuRuntime {
            evaluator,
            scope,
            authority,
            clock,
            owner: self.owner.clone(),
            evidence: self.evidence.clone(),
            submissions: 0,
            pending: None,
        }))
    }
}
struct NativeCpuRuntime {
    evaluator: ContractEvaluator<RulesState, NativeRuntimeBackend<RuntimeClock>, RuntimeClock>,
    scope: SharedScope,
    authority: SearchAuthority,
    clock: ProcessClock,
    owner: NativeWorkerOwner,
    evidence: NativeEvidenceHandle,
    submissions: u64,
    pending: Option<EvalContext>,
}
impl NativeCpuRuntime {
    fn refresh(&self) -> Result<(), ContractError> {
        self.scope.update(self.authority.current_scope()?);
        Ok(())
    }
    fn drain(&mut self, until: Instant) -> Result<(), ContractError> {
        // Closing D admission never revokes queued UCI natural-completion output.
        self.evaluator.begin_shutdown(self.clock.deadline(until)?)?;
        loop {
            self.refresh()?;
            while self.evaluator.poll().is_some() {}
            self.evidence.collect(&self.owner)?;
            let snapshot = self.evaluator.shutdown_snapshot();
            match snapshot.drain {
                DrainState::Drained if snapshot.state.reserved_requests == 0 => return Ok(()),
                DrainState::TimedOut { .. } => {
                    return Err(error(
                        ErrorCode::BackendFailure,
                        Stage::Backend,
                        "native CPU physical drain timed out; lease remains pinned",
                    ));
                }
                _ if Instant::now() >= until => {
                    return Err(error(
                        ErrorCode::Expired,
                        Stage::Backend,
                        "native CPU shutdown deadline reached; completion unconfirmed",
                    ));
                }
                _ => thread::sleep(Duration::from_millis(1)),
            }
        }
    }
}
impl Evaluator<RulesState> for NativeCpuRuntime {
    fn submit(&mut self, request: Arc<EvalRequest<RulesState>>) -> Result<(), ContractError> {
        self.refresh()?;
        self.evidence.collect(&self.owner)?;
        if let Some(error) = self.evidence.admission_error()? {
            return Err(error);
        }
        if self.submissions >= MAX_EVALUATIONS {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "per-root native execution range limit reached",
            ));
        }
        self.submissions += 1;
        self.pending = Some(request.context());
        let result = self.evaluator.submit(request);
        if result.is_err() {
            self.pending = None;
        }
        result
    }
    fn poll(&mut self) -> Option<EvalResult> {
        if let Err(error) = self.refresh() {
            self.authority.cancel();
            return self.pending.take().map(|request| {
                EvalResult::Failed(EvalFailure {
                    context: CompletionContext {
                        request,
                        execution: None,
                    },
                    error,
                    recovery: RecoveryOutcome::NotAttempted,
                })
            });
        }
        let result = self.evaluator.poll();
        let collection = self.evidence.collect(&self.owner);
        if let Some(EvalResult::Completed(output)) = &result
            && let Err(error) = self.evidence.record_completed(output)
        {
            self.pending = None;
            return Some(EvalResult::Failed(EvalFailure {
                context: CompletionContext {
                    request: output.context,
                    execution: output.actual.execution,
                },
                error,
                recovery: RecoveryOutcome::Failed,
            }));
        }
        if result.is_some() {
            self.pending = None;
            return result;
        }
        if let Err(error) = collection {
            return self.pending.take().map(|request| {
                EvalResult::Failed(EvalFailure {
                    context: CompletionContext {
                        request,
                        execution: None,
                    },
                    error,
                    recovery: RecoveryOutcome::NotAttempted,
                })
            });
        }
        None
    }
    fn cancel(&mut self, request: RequestId) -> Result<(), ContractError> {
        self.evaluator.cancel(request)
    }
}
impl ManagedEvaluator for NativeCpuRuntime {
    fn shutdown(&mut self, until: Instant) -> Result<(), ContractError> {
        let drained = self.drain(until);
        let collected = self.evidence.collect(&self.owner);
        // Common cleanup failure is retained by B; C's original cause remains in
        // the native evidence owner even when both operations fail.
        drained.and(collected.map(|_| ()))
    }
}

#[cfg(test)]
#[path = "native_bootstrap_tests.rs"]
mod tests;

#[cfg(test)]
mod physical_owner_tests {
    use super::*;
    use crate::{
        EngineIdentity, Event, PositionPort, PositionSpec,
        engine::{self, RulesUciPort},
    };
    use rz_eval::{RawOutput, contracts::PreparedBatch, worker::SingleWorker};
    use std::{
        io::Write,
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
    };

    const WAIT: Duration = Duration::from_secs(5);

    fn projection(owners: &OwnerRegistry) -> ClassicalProjection {
        ClassicalProjection::new(
            MaiaBinding::new(
                ModelHandle {
                    owner: owners.allocate().unwrap(),
                    slot: 0,
                    generation: SlotGeneration(1),
                    manifest: Digest([51; 32]),
                },
                EncodingHandle {
                    owner: owners.allocate().unwrap(),
                    slot: 0,
                    generation: SlotGeneration(1),
                    manifest: encoding_manifest(HistoryFill::No),
                },
                HistoryFill::No,
                Digest([52; 32]),
                1,
            )
            .unwrap(),
        )
    }
    fn raw() -> RawOutput {
        RawOutput {
            policy_logits: vec![0.0; rz_encoding::POLICY_SIZE],
            wdl: vec![0.4, 0.3, 0.3],
        }
    }
    fn request(
        factory: &NativeCpuFactory,
        owners: &Arc<OwnerRegistry>,
        clock: &ProcessClock,
        sequence: u64,
    ) -> Arc<EvalRequest<RulesState>> {
        let position =
            RulesUciPort::new(Arc::clone(owners), rz_position::PositionLimits::default())
                .prepare(&PositionSpec::default())
                .unwrap()
                .snapshot;
        let profile = factory.profile();
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(clock.epoch(), sequence),
            selection: SelectionId::new(clock.epoch(), sequence),
            game: GameGeneration(1),
            root: RootGeneration(1),
            state: position.state().snapshot().identity(),
            legal_order: position.state().legal_moves().order(),
            input: factory
                .input_key(
                    position.state().rules(),
                    position.state().legal_moves().moves(),
                )
                .unwrap(),
            model: profile.model.handle(),
            encoding: profile.model.encoding().handle,
            precision: profile.precision,
            compute: profile.compute,
            backend: profile.backend,
        };
        Arc::new(
            EvalRequest::try_new(
                context,
                position.state().snapshot().clone(),
                position.state().legal_moves().clone(),
                profile.model,
                clock.deadline(Instant::now() + WAIT).unwrap(),
                CancelToken::new(),
                profile.bytes,
            )
            .unwrap(),
        )
    }
    fn authority(context: EvalContext) -> SearchAuthority {
        SearchAuthority::isolated(AcceptanceScope {
            game: context.game,
            root: context.root,
            model: context.model,
            encoding: context.encoding,
            backend: context.backend,
        })
    }

    #[test]
    fn native_physical_failure_keeps_original_cause_and_service_cleanup_independently() {
        let owners = Arc::new(OwnerRegistry::default());
        let source = BackendError::new(
            FailureKind::BackendFailure,
            FailureStage::Backend,
            "injected physical failure",
        )
        .with_external_cause(CauseCode::OrtRun, &"private-native-cause")
        .with_diagnostic("InjectedNative", "private-native-cause");
        let expected = source.clone();
        let worker = SingleWorker::spawn(move |_: &PreparedBatch<RulesState>| {
            Err(PhysicalFailure::from(source.clone()))
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let clock = ProcessClock::new(ProcessEpoch(301));
        let request = request(&factory, &owners, &clock, 1);
        let context = request.context();
        let mut runtime = factory.create(clock.clone(), authority(context)).unwrap();
        runtime.submit(request).unwrap();
        let until = Instant::now() + WAIT;
        let result = loop {
            if let Some(result) = runtime.poll() {
                break result;
            }
            assert!(
                Instant::now() < until,
                "bounded physical failure publication"
            );
            thread::yield_now();
        };
        let EvalResult::Failed(failure) = result else {
            panic!("native physical failure cannot become a completed evaluation");
        };
        assert_eq!(failure.context.request, context);
        assert!(failure.context.execution.is_some());
        runtime.shutdown(Instant::now() + WAIT).unwrap();
        assert_eq!(
            factory
                .create(clock, authority(context))
                .err()
                .unwrap()
                .code,
            ErrorCode::BackendFailure
        );
        let cleanup = error(
            ErrorCode::Expired,
            Stage::Backend,
            "independent injected outer cleanup failure",
        );
        let failed = factory
            .finish(Err(EngineError::Contract(cleanup)))
            .unwrap_err();
        assert!(
            matches!(failed.service.as_deref(), Some(EngineError::Contract(actual)) if *actual == cleanup)
        );
        let report = failed.report.as_ref().unwrap();
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(report.completed_by_runtime, 0);
        assert_eq!(report.failures.len(), 1);
        let receipt = &report.failures[0];
        assert_eq!(receipt.kind, NativeDiagnosticKind::PhysicalFailure);
        assert_eq!(receipt.context, failure.context);
        assert_eq!(receipt.failure.backend.as_ref(), Some(&expected));
        assert!(!failed.to_string().contains("private-native-cause"));
        assert!(!format!("{failed:?}").contains("private-native-cause"));
    }

    #[test]
    fn periodic_consumer_accepts_normal_cancel_audits_without_lifetime_exhaustion() {
        use rz_runtime::{Backend, contracts::RuntimeRequest};
        let owners = Arc::new(OwnerRegistry::default());
        let worker = SingleWorker::spawn(
            |_: &PreparedBatch<RulesState>| -> Result<Vec<EvalOutput>, PhysicalFailure> {
                panic!("a prelaunch canceled request cannot run native code")
            },
        )
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let factory = NativeCpuFactory::from_owner(owner, None).unwrap();
        let clock = ProcessClock::new(ProcessEpoch(302));
        let mut backend =
            NativeRuntimeBackend::new(factory.owner.clone(), RuntimeClock(clock.clone())).unwrap();
        for sequence in 1..=64 {
            let eval = request(&factory, &owners, &clock, sequence);
            eval.cancel_token().cancel();
            let refused = backend.dispatch(
                &ExecutionId::new(clock.epoch(), sequence),
                &[Arc::new(RuntimeRequest::new(eval))],
            );
            assert_eq!(refused.err().unwrap().code, ErrorCode::Canceled);
            assert!(factory.evidence.collect(&factory.owner).unwrap());
        }
        let report = factory.finish(Ok(())).unwrap();
        assert_eq!(report.canceled.count, 64);
        assert_eq!(report.canceled.first.as_ref().unwrap().ticket.1.sequence, 1);
        assert_eq!(report.canceled.last.as_ref().unwrap().ticket.1.sequence, 64);
        assert!(report.failures.is_empty());
        assert!(!report.has_failure());
    }

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let mut output = self.0.lock().unwrap();
            if output.len() + bytes.len() > 16 * 1024 {
                return Err(std::io::Error::other("bounded test output exhausted"));
            }
            output.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }
    struct CountingFactory {
        native: Arc<NativeCpuFactory>,
        created: AtomicUsize,
    }
    impl EvaluatorFactory for CountingFactory {
        fn profile(&self) -> EvaluatorProfile {
            self.native.profile()
        }
        fn input_key(
            &self,
            state: &RulesState,
            legal: &[Move],
        ) -> Result<EvalInputKey, ContractError> {
            self.native.input_key(state, legal)
        }
        fn create(
            &self,
            clock: ProcessClock,
            authority: SearchAuthority,
        ) -> Result<Box<dyn ManagedEvaluator>, ContractError> {
            self.created.fetch_add(1, Ordering::AcqRel);
            self.native.create(clock, authority)
        }
    }
    struct ServiceGuard {
        events: mpsc::SyncSender<Event>,
        release: Option<mpsc::SyncSender<()>>,
        service: Option<thread::JoinHandle<Result<(), EngineError>>>,
    }
    impl Drop for ServiceGuard {
        fn drop(&mut self) {
            if let Some(release) = self.release.take() {
                let _ = release.try_send(());
            }
            let _ = self.events.try_send(Event::Line("quit".into()));
        }
    }

    fn assert_blocked_physical_owner_active(owner: &NativeWorkerOwner, stage: &'static str) {
        let until = Instant::now() + WAIT;
        loop {
            match owner.try_status() {
                Ok(Some(status)) => {
                    assert!(
                        status.active,
                        "{stage}: blocked physical owner is inactive: {status:?}"
                    );
                    return;
                }
                Err(error) => panic!("{stage}: native owner status query failed: {error:?}"),
                Ok(None) => {
                    // None means owner-lock contention, not an inactive worker.
                    // Retry the actual observation without claiming synchronization.
                    assert!(
                        Instant::now() < until,
                        "{stage}: native owner status remained busy until the bounded observation deadline"
                    );
                    thread::yield_now();
                }
            }
        }
    }

    #[test]
    fn root_replacement_while_the_single_physical_worker_is_blocked_returns_current_legal_fallback_once()
     {
        let owners = Arc::new(OwnerRegistry::default());
        let (entered, entering) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let worker = SingleWorker::spawn(move |batch: &PreparedBatch<RulesState>| {
            entered.send(()).unwrap();
            if released.recv_timeout(WAIT).is_err() {
                return Err(BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Backend,
                    "blocked injected worker release timed out",
                )
                .into());
            }
            batch
                .requests()
                .iter()
                .map(|item| item.physical_output(&raw(), batch.execution()))
                .collect()
        })
        .unwrap();
        let owner = NativeWorkerOwner::from_worker(worker, projection(&owners), 1).unwrap();
        let native = Arc::new(NativeCpuFactory::from_owner(owner, None).unwrap());
        let factory = Arc::new(CountingFactory {
            native: Arc::clone(&native),
            created: AtomicUsize::new(0),
        });
        let process = engine::EngineProcess::new(
            factory.clone(),
            owners,
            ProcessClock::new(ProcessEpoch(303)),
        )
        .with_identity(EngineIdentity {
            name: "RoveZero injected single physical worker fixture".into(),
            author: "RoveZero tests".into(),
        });
        let settings = engine::EngineSettings {
            max_workers: 1,
            ..engine::EngineSettings::default()
        };
        let (events, receiver) = engine::event_channel();
        let sender = events.clone();
        let mut protocol = Capture::default();
        let protocol_view = protocol.clone();
        let mut diagnostics = Capture::default();
        let diagnostics_view = diagnostics.clone();
        let service = thread::spawn(move || {
            engine::serve(
                receiver,
                sender,
                &mut protocol,
                &mut diagnostics,
                process,
                settings,
            )
        });
        let mut guard = ServiceGuard {
            events,
            release: Some(release),
            service: Some(service),
        };
        for line in ["uci", "position startpos", "go nodes 128 movetime 30000"] {
            guard.events.send(Event::Line(line.into())).unwrap();
        }
        entering
            .recv_timeout(WAIT)
            .expect("first root reached the actual physical worker");
        assert_blocked_physical_owner_active(&native.owner, "after physical worker entry");
        for line in ["position startpos moves e2e4", "go nodes 1 movetime 30000"] {
            guard.events.send(Event::Line(line.into())).unwrap();
        }
        let until = Instant::now() + WAIT;
        let bestmove = loop {
            if let Some(line) = protocol_view
                .text()
                .lines()
                .find(|line| line.starts_with("bestmove "))
            {
                break line.to_owned();
            }
            assert!(
                Instant::now() < until,
                "current root received a bounded legal fallback"
            );
            thread::sleep(Duration::from_millis(1));
        };
        let mut black = rz_position::Position::startpos();
        black.make_uci("e2e4").unwrap();
        let movement =
            rz_position::BoardMove::from_uci(bestmove.strip_prefix("bestmove ").unwrap()).unwrap();
        assert!(black.legal_moves().contains(&movement));
        assert_eq!(
            factory.created.load(Ordering::Acquire),
            1,
            "busy root fallback cannot reload/create another provider session"
        );
        assert_blocked_physical_owner_active(
            &native.owner,
            "after current-root legal fallback before physical release",
        );
        guard.release.take().unwrap().send(()).unwrap();
        guard.events.send(Event::Line("quit".into())).unwrap();
        let until = Instant::now() + WAIT;
        while !guard.service.as_ref().unwrap().is_finished() {
            assert!(
                Instant::now() < until,
                "bounded quit and old-root physical drain"
            );
            thread::sleep(Duration::from_millis(1));
        }
        let served = guard.service.take().unwrap().join().unwrap();
        let report = native.finish(served).unwrap();
        assert_eq!(report.origin, NativeWorkerOrigin::Injected);
        assert_eq!(
            report.completed_by_runtime, 0,
            "an injected worker is not native ONNX execution evidence"
        );
        assert_eq!(
            protocol_view
                .text()
                .lines()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
        assert!(diagnostics_view.text().contains("WorkerLimit"));
        assert!(!report.has_failure());
    }
}
