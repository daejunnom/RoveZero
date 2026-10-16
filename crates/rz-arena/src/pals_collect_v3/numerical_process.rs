//! 학습된 원래 export를 독립 Torch → 실제 native 수치 캡처와 연결한다.
//! JSON은 실행 계획이며 opaque export/prepared/process owner를 복원하지 않는다.
//! 실패·취소는 원래 파일, partial 출력과 자식 수명을 보존하고 재시도하지 않는다.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(target_os = "linux")]
use super::GenerationExportKindV3;
use super::{
    GenerationTrainingFilePinV3, GenerationTrainingPythonRuntimeV3,
    PreparedFrozenActionStartupInputsV3, VerifiedGenerationExportV3, check, invalid,
};
use crate::{ArenaError, process::OriginalProcessOutput};
use rz_experiments::{
    ArtifactRef, PalsAdaptiveStartupConfigV5, PalsInputLockV5, PalsNumericalAdmissionV5,
    PalsNumericalBindingsV5, PalsStartupAssetPurposeV5 as Purpose, PalsStartupAssetV5,
};
#[cfg(target_os = "linux")]
use rz_experiments::{
    PalsNumericalAcceptanceV5, PalsNumericalInvocationV5, PalsNumericalRunPairV5,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(target_os = "linux")]
use std::time::Duration;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

const LAUNCH: &str = "rz-pals-generation-numerical-launch/1";
const REGISTRATION: &str = "rz-pals-generation-numerical-registration/1";
const STARTUP: &str = "rz-pals-original-generation-numerical-startup/1";
const RESULT: &str = "rz-pals-generation-numerical-cli-window/1";
const MAX_JSON: u64 = 4 * 1024 * 1024;
const MAX_FILE: u64 = 1024 * 1024 * 1024;
const MAX_RAW: u64 = 64 * 1024 * 1024;
const MAX_STAGING: u64 = 18 * 1024 * 1024 * 1024;
const PACKAGE: &str = "experiments/model-research/pals/src";
const BOOTSTRAP: &str = "import sys,runpy; package=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,package); sys.argv[0]=module; runpy.run_module(module,run_name='__main__',alter_sys=True)";
const COMPONENTS: [&str; 5] = [
    "model_native",
    "model_encoding",
    "nn_runtime",
    "model_reference",
    "model_export",
];

/// 미래의 학습 가중치 SHA를 요구하지 않는다. 원래 export owner에서 실제
/// checkpoint/manifest/graph를 얻고, 미리 등록한 case 내용만 그대로 보존한다.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationNumericalLaunchRegistrationV3 {
    pub schema: String,
    pub numerical_id: String,
    pub template_startup: GenerationTrainingFilePinV3,
    pub case_plan: GenerationTrainingFilePinV3,
    pub native_capture_config: GenerationTrainingFilePinV3,
    pub native_executable: GenerationTrainingFilePinV3,
    pub python_runtime: GenerationTrainingPythonRuntimeV3,
    pub dependency_directories: Vec<PathBuf>,
    pub runtime_module_pins: Vec<GenerationTrainingFilePinV3>,
    pub reference_output_root: PathBuf,
    pub native_output_root: PathBuf,
    pub receipt_path: PathBuf,
    pub acceptance_path: PathBuf,
    pub max_wall_ns: u64,
    pub max_output_bytes: u64,
    pub max_read_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OriginalEmptyNativeOutputRoot {
    pub(super) schema: String,
    pub(super) device: u64,
    pub(super) inode: u64,
    pub(super) uid: u32,
    pub(super) gid: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeCaptureConfig {
    pub(super) case_plan: PathBuf,
    pub(super) case_plan_sha256: String,
    pub(super) export_manifest: PathBuf,
    pub(super) runtime_library: PathBuf,
    pub(super) runtime_cache: PathBuf,
    pub(super) output_root: PathBuf,
    /// Original G0 parent descriptor readback, never a caller-issued capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) output_root_admission: Option<OriginalEmptyNativeOutputRoot>,
    pub(super) provider: String,
    pub(super) cuda_bundle: Option<PathBuf>,
    pub(super) cuda_inventory: Option<PathBuf>,
    pub(super) cuda_inventory_sha256: Option<String>,
    /// Omission preserves the historical implicit exhaustive session policy.
    #[serde(default)]
    pub(super) cudnn_conv_search: Option<String>,
    pub(super) gpu_device_id: i32,
    pub(super) gpu_arena_bytes: usize,
    pub(super) deadline_seconds: u64,
    pub(super) cleanup_seconds: u64,
}

/// 두 실제 캡처와 원래 raw 파일을 보존한 성공 owner. 공개 생성자나 decoder는
/// 없다. 이는 수치·완료 증거이며 Rules 합법성·학습 기력·GPU 독점의 증거가 아니다.
pub struct VerifiedFrozenNumericalV3 {
    prepared: Arc<PreparedFrozenActionStartupInputsV3>,
    core: VerifiedNumericalCoreV3,
}
impl VerifiedFrozenNumericalV3 {
    pub fn prepared(&self) -> &Arc<PreparedFrozenActionStartupInputsV3> {
        &self.prepared
    }
    pub fn frozen(&self) -> &Arc<VerifiedGenerationExportV3> {
        self.prepared.frozen()
    }
    pub fn startup_config(&self) -> &PalsAdaptiveStartupConfigV5 {
        &self.core.config
    }
    pub fn lock(&self) -> &PalsInputLockV5 {
        &self.core.lock
    }
    pub fn admission(&self) -> &PalsNumericalAdmissionV5 {
        &self.core.admission
    }
    pub fn acceptance_pin(&self) -> &GenerationTrainingFilePinV3 {
        &self.core.acceptance
    }
    pub fn processes(&self) -> &[OriginalProcessOutput] {
        &self.core.processes
    }
    pub fn parent_receipt(&self) -> &Value {
        &self.core.parent_receipt
    }
    #[cfg(target_os = "linux")]
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.core.held.checkpoints.iter().map(|f| (&f.pin, &f.file))
    }
    #[cfg(not(target_os = "linux"))]
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        std::iter::empty()
    }
}
struct VerifiedNumericalCoreV3 {
    config: PalsAdaptiveStartupConfigV5,
    lock: PalsInputLockV5,
    admission: PalsNumericalAdmissionV5,
    acceptance: GenerationTrainingFilePinV3,
    processes: Vec<OriginalProcessOutput>,
    parent_receipt: Value,
    #[cfg(target_os = "linux")]
    held: super::training_process::LinuxHeld,
}
pub struct VerifiedG1NumericalV3 {
    prepared: Arc<crate::pals_adaptive_launch::PreparedG1EvaluationStartupV5>,
    core: VerifiedNumericalCoreV3,
}
impl VerifiedG1NumericalV3 {
    pub fn prepared(&self) -> &Arc<crate::pals_adaptive_launch::PreparedG1EvaluationStartupV5> {
        &self.prepared
    }
    pub fn g1(&self) -> &Arc<VerifiedGenerationExportV3> {
        self.prepared.g1()
    }
    pub fn startup_config(&self) -> &PalsAdaptiveStartupConfigV5 {
        &self.core.config
    }
    pub fn lock(&self) -> &PalsInputLockV5 {
        &self.core.lock
    }
    pub fn admission(&self) -> &PalsNumericalAdmissionV5 {
        &self.core.admission
    }
    pub fn acceptance_pin(&self) -> &GenerationTrainingFilePinV3 {
        &self.core.acceptance
    }
    pub fn processes(&self) -> &[OriginalProcessOutput] {
        &self.core.processes
    }
    pub fn parent_receipt(&self) -> &Value {
        &self.core.parent_receipt
    }
    #[cfg(target_os = "linux")]
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.core.held.checkpoints.iter().map(|f| (&f.pin, &f.file))
    }
    #[cfg(not(target_os = "linux"))]
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        std::iter::empty()
    }
}
pub struct G1NumericalFailureV3 {
    pub error: ArenaError,
    pub processes: Vec<OriginalProcessOutput>,
    pub reference_receipt: Option<Value>,
    pub prepared: Arc<crate::pals_adaptive_launch::PreparedG1EvaluationStartupV5>,
    pub registration_pin: GenerationTrainingFilePinV3,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    pub receipt_preservation_errors: Vec<String>,
    // 실패·완료 불명 자료의 원래 FD를 failure owner와 함께 격리 보존한다.
    // 인수용 읽기 권한을 발급하지 않으며 직접 조회하지 않아도 수명은 유지한다.
    #[cfg(target_os = "linux")]
    _held: Option<super::training_process::LinuxHeld>,
    // 원래 receipt/raw/pending을 성공용 raw/인수 목록과 분리해 보존한다.
    #[cfg(target_os = "linux")]
    _receipt_files: Vec<FailureOnlyOutputFileV3>,
}
impl std::fmt::Debug for G1NumericalFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G1NumericalFailureV3")
            .field("error", &self.error)
            .field("processes", &self.processes)
            .field(
                "receipt_preservation_errors",
                &self.receipt_preservation_errors,
            )
            .field(
                "reference_receipt_preserved",
                &self.reference_receipt.is_some(),
            )
            .finish_non_exhaustive()
    }
}

/// 같은 수치 producer를 사용하되 actual5000 export와 V-free Deployment
/// prepared owner를 유지한다. frozen4000의 raw/admission을 재사용하지 않는다.
pub fn run_g1_numerical_original_v3(
    registration_pin: GenerationTrainingFilePinV3,
    prepared: Arc<crate::pals_adaptive_launch::PreparedG1EvaluationStartupV5>,
    until: Instant,
    cleanup_until: Instant,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedG1NumericalV3, Box<G1NumericalFailureV3>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, cleanup_until, cancel);
        Err(Box::new(G1NumericalFailureV3 {
            error: invalid("G1 original numerical requires Linux"),
            processes: Vec::new(),
            reference_receipt: None,
            prepared,
            registration_pin,
            charged_read_bytes: 0,
            read_usage_uncertain: false,
            receipt_preservation_errors: Vec::new(),
        }))
    }
    #[cfg(target_os = "linux")]
    {
        use super::training_process::{self as common, ReadAccounting, ReadCredit};
        use std::sync::atomic::Ordering;
        let started = Instant::now();
        let account = ReadAccounting::new();
        let mut held = None;
        let mut processes = Vec::new();
        let mut reference_receipt = None;
        let mut receipt_files = Vec::new();
        let mut receipt_preservation_errors = Vec::new();
        let result = (|| -> Result<VerifiedNumericalCoreV3, ArenaError> {
            check(
                prepared.g1().kind() == GenerationExportKindV3::G1ActualGeneration,
                "G1 numerical requires original actual5000 export",
            )?;
            registration_pin.validate(MAX_JSON)?;
            check(
                started < until
                    && until < cleanup_until
                    && cleanup_until.duration_since(until) <= Duration::from_secs(30),
                "G1 numerical original finite window order",
            )?;
            let mut credit = ReadCredit {
                remaining: 4 * MAX_STAGING,
                maximum: 4 * MAX_STAGING,
                until,
                cancel: &cancel,
                account: Arc::clone(&account),
            };
            let file = common::open_file(&registration_pin.path)?;
            let raw = common::verify_bytes(&file, &registration_pin, true, &mut credit)?;
            let registration: GenerationNumericalLaunchRegistrationV3 = common::decode(&raw)?;
            check(
                registration.template_startup == *prepared.original().template_pin(),
                "G1 numerical must retain the same preregistered evaluation template",
            )?;
            let files = prepared
                .held_files()
                .chain(
                    prepared
                        .original()
                        .held_files()
                        .filter(|(pin, _)| *pin == prepared.original().template_pin()),
                )
                .collect::<Vec<_>>();
            run_captures(
                &registration,
                &registration_pin,
                &file,
                prepared.g1(),
                prepared.config(),
                prepared.lock(),
                prepared.private_cwd(),
                &files,
                started,
                until,
                cleanup_until,
                &cancel,
                &account,
                &mut held,
                &mut processes,
                &mut reference_receipt,
                &mut receipt_files,
                &mut receipt_preservation_errors,
            )
        })();
        match result {
            Ok(core) => Ok(VerifiedG1NumericalV3 { prepared, core }),
            Err(error) => Err(Box::new(G1NumericalFailureV3 {
                error,
                processes,
                reference_receipt,
                prepared,
                registration_pin,
                charged_read_bytes: account.charged.load(Ordering::Acquire),
                read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
                receipt_preservation_errors,
                _held: held,
                _receipt_files: receipt_files,
            })),
        }
    }
}
pub struct FrozenNumericalFailureV3 {
    pub error: ArenaError,
    pub processes: Vec<OriginalProcessOutput>,
    pub reference_receipt: Option<Value>,
    pub prepared: Option<Arc<PreparedFrozenActionStartupInputsV3>>,
    pub preparation_failure: Option<Box<super::FrozenActionPreparationFailureV3>>,
    pub frozen: Arc<VerifiedGenerationExportV3>,
    pub registration_pin: GenerationTrainingFilePinV3,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    pub receipt_preservation_errors: Vec<String>,
    // 실패·완료 불명 자료의 원래 FD를 failure owner와 함께 격리 보존한다.
    // 인수용 읽기 권한을 발급하지 않으며 직접 조회하지 않아도 수명은 유지한다.
    #[cfg(target_os = "linux")]
    _held: Option<super::training_process::LinuxHeld>,
    // 원래 receipt/raw/pending을 성공용 raw/인수 목록과 분리해 보존한다.
    #[cfg(target_os = "linux")]
    _receipt_files: Vec<FailureOnlyOutputFileV3>,
}
impl std::fmt::Debug for FrozenNumericalFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenNumericalFailureV3")
            .field("error", &self.error)
            .field("processes", &self.processes)
            .field(
                "receipt_preservation_errors",
                &self.receipt_preservation_errors,
            )
            .field(
                "reference_receipt_preserved",
                &self.reference_receipt.is_some(),
            )
            .field("charged_read_bytes", &self.charged_read_bytes)
            .finish_non_exhaustive()
    }
}

pub fn run_frozen_numerical_original_v3(
    registration_pin: GenerationTrainingFilePinV3,
    frozen: Arc<VerifiedGenerationExportV3>,
    until: Instant,
    cleanup_until: Instant,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedFrozenNumericalV3, Box<FrozenNumericalFailureV3>> {
    #[cfg(target_os = "linux")]
    {
        run_frozen_linux(registration_pin, frozen, until, cleanup_until, cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, cleanup_until, cancel);
        Err(Box::new(FrozenNumericalFailureV3 {
            error: invalid("original generation numerical capture requires Linux"),
            processes: Vec::new(),
            reference_receipt: None,
            prepared: None,
            preparation_failure: None,
            frozen,
            registration_pin,
            charged_read_bytes: 0,
            read_usage_uncertain: false,
            receipt_preservation_errors: Vec::new(),
        }))
    }
}

pub(super) fn bindings(
    config: &PalsAdaptiveStartupConfigV5,
) -> Result<PalsNumericalBindingsV5, ArenaError> {
    let m = &config.endpoint.model;
    let asset = |p| config.asset(p).map_err(|e| invalid(e.to_string()));
    let source_components = COMPONENTS
        .into_iter()
        .map(|name| {
            config
                .source_components
                .get(name)
                .map(|sha| (name.into(), sha.clone()))
                .ok_or_else(|| invalid("numerical exact source component absent"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let compiled =
        rz_experiments::compiled_source_components_v5().map_err(|e| invalid(e.to_string()))?;
    check(
        source_components
            .iter()
            .all(|(name, sha)| compiled.get(name) == Some(sha)),
        "numerical source differs from compiled producer roster",
    )?;
    let mut graphs = BTreeMap::new();
    for (name, p) in [
        ("public", Purpose::PublicGraph),
        ("shared_pc", Purpose::SharedPcGraph),
        ("proposer", Purpose::ProposerGraph),
        ("critic", Purpose::CriticGraph),
        ("validator", Purpose::ValidatorGraph),
    ] {
        let values: Vec<_> = config.assets.iter().filter(|a| a.purpose == p).collect();
        check(values.len() <= 1, "ambiguous numerical graph role")?;
        if let Some(a) = values.first() {
            graphs.insert(name.into(), a.artifact.sha256.clone());
        }
    }
    Ok(PalsNumericalBindingsV5 {
        model_profile: m.profile.clone(),
        model_schema: m.model_schema.clone(),
        model_semantics: m.model_semantics.clone(),
        encoding_schema: m.encoding_schema.clone(),
        model_semantic_sha256: m.semantic_sha256.clone(),
        encoding_sha256: m.encoding_sha256.clone(),
        model_epoch: m.epoch,
        weights_sha256: m.weights.sha256.clone(),
        export_manifest_sha256: asset(Purpose::ExportManifest)?.artifact.sha256.clone(),
        graphs,
        runtime_library_sha256: asset(Purpose::RuntimeLibrary)?.artifact.sha256.clone(),
        cuda_libraries: config
            .assets
            .iter()
            .filter(|a| a.purpose == Purpose::CudaLibrary)
            .map(|a| a.artifact.sha256.clone())
            .collect(),
        cuda_manifest_sha256: config
            .assets
            .iter()
            .find(|a| a.purpose == Purpose::CudaManifest)
            .map(|a| a.artifact.sha256.clone()),
        backend: m.backend,
        precision: m.precision,
        tf32: m.tf32,
        source_components,
    })
}

#[cfg(target_os = "linux")]
fn run_frozen_linux(
    registration_pin: GenerationTrainingFilePinV3,
    frozen: Arc<VerifiedGenerationExportV3>,
    until: Instant,
    cleanup_until: Instant,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedFrozenNumericalV3, Box<FrozenNumericalFailureV3>> {
    use super::training_process::{self as common, ReadAccounting, ReadCredit};
    use std::sync::atomic::Ordering;
    let started = Instant::now();
    let account = ReadAccounting::new();
    let mut prepared = None;
    let mut preparation_failure = None;
    let mut held = None;
    let mut processes = Vec::new();
    let mut reference_receipt = None;
    let mut receipt_files = Vec::new();
    let mut receipt_preservation_errors = Vec::new();
    let result = (|| -> Result<VerifiedNumericalCoreV3, ArenaError> {
        check(
            frozen.kind() == GenerationExportKindV3::FrozenPcCollection,
            "frozen numerical requires original durable4000 collection export",
        )?;
        registration_pin.validate(MAX_JSON)?;
        check(
            started < until
                && until < cleanup_until
                && cleanup_until.duration_since(until) <= Duration::from_secs(30),
            "numerical original finite window order",
        )?;
        let mut credit = ReadCredit {
            remaining: 4 * MAX_STAGING,
            maximum: 4 * MAX_STAGING,
            until,
            cancel: &cancel,
            account: Arc::clone(&account),
        };
        let file = common::open_file(&registration_pin.path)?;
        let raw = common::verify_bytes(&file, &registration_pin, true, &mut credit)?;
        let registration: GenerationNumericalLaunchRegistrationV3 = common::decode(&raw)?;
        let inputs = match super::prepare_frozen_action_startup_inputs_owned_v3(
            registration.template_startup.clone(),
            Arc::clone(&frozen),
            until,
            &cancel,
        ) {
            Ok(inputs) => inputs,
            Err(mut failure) => {
                // 첫 실패의 원래 분류/원인과 하위 custody를 함께 보존한다.
                let error = std::mem::replace(
                    &mut failure.error,
                    invalid(
                        "original preparation error is retained by enclosing numerical failure",
                    ),
                );
                preparation_failure = Some(failure);
                return Err(error);
            }
        };
        prepared = Some(Arc::new(inputs));
        run_captures(
            &registration,
            &registration_pin,
            &file,
            &frozen,
            prepared
                .as_ref()
                .expect("original prepared inputs")
                .config(),
            prepared.as_ref().expect("original prepared inputs").lock(),
            prepared
                .as_ref()
                .expect("original prepared inputs")
                .private_cwd(),
            &prepared
                .as_ref()
                .expect("original prepared inputs")
                .held_files()
                .collect::<Vec<_>>(),
            started,
            until,
            cleanup_until,
            &cancel,
            &account,
            &mut held,
            &mut processes,
            &mut reference_receipt,
            &mut receipt_files,
            &mut receipt_preservation_errors,
        )
    })();
    match result {
        Ok(core) => Ok(VerifiedFrozenNumericalV3 {
            prepared: prepared.expect("successful original inputs retained"),
            core,
        }),
        Err(error) => Err(Box::new(FrozenNumericalFailureV3 {
            error,
            processes,
            reference_receipt,
            prepared,
            preparation_failure,
            frozen,
            registration_pin,
            charged_read_bytes: account.charged.load(Ordering::Acquire),
            read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
            receipt_preservation_errors,
            _held: held,
            _receipt_files: receipt_files,
        })),
    }
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_captures(
    registration: &GenerationNumericalLaunchRegistrationV3,
    registration_pin: &GenerationTrainingFilePinV3,
    registration_file: &File,
    export: &Arc<VerifiedGenerationExportV3>,
    input_config: &PalsAdaptiveStartupConfigV5,
    lock: &PalsInputLockV5,
    private_cwd: &Path,
    prepared_files: &[(&GenerationTrainingFilePinV3, &File)],
    started: Instant,
    until: Instant,
    cleanup_until: Instant,
    cancel: &AtomicBool,
    account: &Arc<super::training_process::ReadAccounting>,
    held: &mut Option<super::training_process::LinuxHeld>,
    processes: &mut Vec<OriginalProcessOutput>,
    reference_receipt: &mut Option<Value>,
    receipt_files: &mut Vec<FailureOnlyOutputFileV3>,
    receipt_preservation_errors: &mut Vec<String>,
) -> Result<VerifiedNumericalCoreV3, ArenaError> {
    use super::training_process::{self as common, ReadCredit};
    use crate::process::{
        OriginalNumericalEnvironment, OriginalProcessInput, OriginalProcessWindow,
        OwnedArtifactTreeWatch, ProcessLimits, supervise_numerical_input_in_original_window,
        supervise_verified_input_in_original_window,
    };
    use std::{ffi::OsString, sync::atomic::Ordering};
    let preserve_cancel = AtomicBool::new(false);
    let original = export.trainer();
    check(
        registration.python_runtime == original.launch().python_runtime
            && registration.dependency_directories == original.launch().dependency_directories
            && registration.runtime_module_pins == original.launch().runtime_module_pins,
        "numerical original interpreter/trusted installed runtime pins differ from actual trainer",
    )?;
    let mut loading = original.registration().clone();
    validate_launch(
        registration,
        &loading.staging_root,
        private_cwd,
        started,
        until,
    )?;
    loading.window_id = registration.numerical_id.clone();
    loading.receipt_path = registration.receipt_path.clone();
    loading.max_wall_ns = registration.max_wall_ns;
    check(
        registration.max_output_bytes == loading.max_output_bytes,
        "numerical retains original managed staging ceiling",
    )?;
    let already_charged = account.charged.load(Ordering::Acquire);
    let remaining = registration
        .max_read_bytes
        .checked_sub(already_charged)
        .ok_or_else(|| invalid("numerical preparation consumed original read credit"))?;
    let mut credit = ReadCredit {
        remaining,
        maximum: registration.max_read_bytes,
        until,
        cancel,
        account: Arc::clone(account),
    };
    *held = Some(common::prepare_root(&loading, &registration.numerical_id)?);
    let owner = held.as_mut().expect("original numerical file custody");
    let root = common::open_directory(&loading.staging_root)?;
    let staging_cap = registration.max_output_bytes;
    super::selfplay_cli::scan_staging(&root, staging_cap, until, cancel)?;
    common::hold_input(registration_pin.clone(), false, owner, &mut credit)?;
    let mut dependencies = Vec::new();
    for path in &registration.dependency_directories {
        dependencies.push(common::retain_directory(path.clone(), owner)?);
    }
    let dependencies = Value::Array(dependencies);
    for pin in &registration.runtime_module_pins {
        check(
            registration
                .dependency_directories
                .iter()
                .any(|path| pin.path.starts_with(path)),
            "numerical runtime module outside trusted installed roots",
        )?;
        common::hold_input(pin.clone(), false, owner, &mut credit)?;
    }
    for (pin, file) in prepared_files
        .iter()
        .copied()
        .chain(export.held_files())
        .chain(original.checkpoint_files())
    {
        common::verify_bytes(file, pin, false, &mut credit)?;
        common::hold_input(pin.clone(), false, owner, &mut credit)?;
    }
    common::hold_input(
        registration.python_runtime.executable.clone(),
        false,
        owner,
        &mut credit,
    )?;
    common::hold_input(
        registration.native_executable.clone(),
        false,
        owner,
        &mut credit,
    )?;
    let python = common::open_file(&registration.python_runtime.executable.path)?;
    let native = common::open_file(&registration.native_executable.path)?;
    let original_checkpoint = original
        .checkpoint_files()
        .last()
        .ok_or_else(|| invalid("numerical original actual checkpoint not held"))?
        .0
        .clone();
    let manifest = match export.kind() {
        GenerationExportKindV3::FrozenPcCollection => export.collection_manifest_pin(),
        GenerationExportKindV3::G1ActualGeneration => export
            .deployment_manifest_pin()
            .ok_or_else(|| invalid("G1 original V-free deployment manifest not held"))?,
    };
    let manifest_file = export
        .held_files()
        .find(|(pin, _)| *pin == manifest)
        .ok_or_else(|| invalid("numerical exact export descriptor absent"))?
        .1;
    let manifest_json: Value = common::decode(&common::verify_bytes(
        manifest_file,
        manifest,
        true,
        &mut credit,
    )?)?;
    let expected = bindings(input_config)?;
    check(
        expected.weights_sha256 == original_checkpoint.sha256
            && expected.export_manifest_sha256 == manifest.sha256
            && expected.model_epoch == export.model_epoch(),
        "numerical bindings differ from original actual export",
    )?;
    let graph_pins = match export.kind() {
        GenerationExportKindV3::FrozenPcCollection => {
            export.collection_graph_pins().cloned().collect::<Vec<_>>()
        }
        GenerationExportKindV3::G1ActualGeneration => {
            export.deployment_graph_pins().cloned().collect::<Vec<_>>()
        }
    };
    let graph_rows=manifest_json["graphs"].as_array().ok_or_else(||invalid("actual original graph descriptor list absent"))?.iter().map(|descriptor| {
        let pin=graph_pins.iter().find(|pin|descriptor["sha256"]==pin.sha256)
            .ok_or_else(||invalid("actual export graph descriptor has no original held pin"))?;
        Ok(serde_json::json!({"role":descriptor["role"],"file":descriptor["file"],"path":pin.path,"sha256":pin.sha256,"bytes":pin.bytes}))
    }).collect::<Result<Vec<_>,ArenaError>>()?;
    check(
        graph_rows
            .iter()
            .filter_map(|g| g["role"].as_str())
            .collect::<BTreeSet<_>>()
            == expected.graphs.keys().map(String::as_str).collect(),
        "original numerical graph role scope differs",
    )?;
    let template_raw =
        common::hold_input(registration.case_plan.clone(), true, owner, &mut credit)?;
    let mut plan: Value = common::decode(&template_raw)?;
    validate_plan(&plan, &expected)?;
    let declared_groups = plan["groups"].clone();
    plan["bindings"] = serde_json::to_value(&expected).map_err(|e| invalid(e.to_string()))?;
    check(
        plan["groups"] == declared_groups,
        "preregistered numerical cases changed",
    )?;
    let plan_raw = serde_json::to_vec(&plan).map_err(|e| invalid(e.to_string()))?;
    ensure_output_credit(&root, staging_cap, plan_raw.len() as u64, until, cancel)?;
    let actual_plan = common::write_owned_new(
        &loading
            .staging_root
            .join(format!("numerical-plan-{}.json", registration.numerical_id)),
        &plan_raw,
    )?;
    let actual_plan_pin = actual_plan.pin.clone();
    owner.checkpoints.push(actual_plan);
    let capture_raw = common::hold_input(
        registration.native_capture_config.clone(),
        true,
        owner,
        &mut credit,
    )?;
    let mut capture: NativeCaptureConfig = common::decode(&capture_raw)?;
    validate_capture_inputs(&capture, input_config, private_cwd, &loading.staging_root)?;
    for (pin, file) in prepared_files {
        common::verify_bytes(file, pin, false, &mut credit)?;
    }
    let semantic = serde_json::json!({"kind":export.kind(),"config":manifest_json["config"],
        "checkpoint_sha256":original_checkpoint.sha256,"trained":manifest_json["trained"],"training_steps":manifest_json["training_steps"],
        "validator_present":export.kind()==GenerationExportKindV3::FrozenPcCollection,
        "model_epoch":export.model_epoch(),"frozen_pc_sha256":export.frozen_pc_sha256()});
    let expected_updates = if export.kind() == GenerationExportKindV3::FrozenPcCollection {
        4000
    } else {
        5000
    };
    check(
        manifest_json["trained"] == true && manifest_json["training_steps"] == expected_updates,
        "numerical requires actual export's completed training provenance",
    )?;
    let child_registration = serde_json::json!({"schema":REGISTRATION,"numerical_id":registration.numerical_id,
        "template_startup":registration.template_startup,"case_plan":actual_plan_pin,"checkpoint":original_checkpoint,
        "export_manifest":manifest,"export_graphs":graph_rows,"source_components":expected.source_components,
        "frozen_pc_sha256":export.frozen_pc_sha256(),"model_epoch":export.model_epoch(),"generation_export_semantic":semantic,
        "staging_root":loading.staging_root,"reference_output_root":registration.reference_output_root,
        "receipt_path":registration.receipt_path,"max_wall_ns":registration.max_wall_ns,"max_output_bytes":registration.max_output_bytes});
    let registration_sha256 =
        rz_experiments::pals_data_v3::generation_digest(REGISTRATION, &child_registration)
            .map_err(|e| invalid(e.to_string()))?;
    // 기존 source overlay의 최대 32MiB와 inventory를 쓰기 전에 예약한다.
    ensure_output_credit(&root, staging_cap, 36 * 1024 * 1024, until, cancel)?;
    common::install_sources_for_components(owner, &mut credit, &COMPONENTS)?;
    check(
        owner.sources.iter().any(|file| {
            file.pin
                .path
                .ends_with("rz_pals_model/generation_numerical_cli.py")
        }),
        "original numerical child absent from compiled source overlay",
    )?;
    let (start_ns, execution_ns, whole_ns) =
        common::original_window_ticks(started, until, cleanup_until)?;
    let startup = serde_json::json!({"schema":STARTUP,"execution_id":registration.numerical_id,"registration_sha256":registration_sha256,
        "registration":child_registration,"original_started_ns":start_ns,"execution_until_ns":execution_ns,"whole_until_ns":whole_ns,
        "dependency_directories":dependencies,"python_runtime":registration.python_runtime,"runtime_module_pins":registration.runtime_module_pins});
    let stdin = serde_json::to_vec(&startup).map_err(|e| invalid(e.to_string()))?;
    check(
        stdin.len() <= crate::process::MAX_ORIGINAL_INPUT_BYTES,
        "numerical bounded original startup",
    )?;
    ensure_output_credit(&root, staging_cap, stdin.len() as u64, until, cancel)?;
    owner.startup = Some(common::write_owned_new(
        &loading.staging_root.join(format!(
            "numerical-startup-{}.json",
            registration.numerical_id
        )),
        &stdin,
    )?);
    let mut args = vec![
        OsString::from("-I"),
        OsString::from("-S"),
        OsString::from("-B"),
        OsString::from("-c"),
        OsString::from(BOOTSTRAP),
        owner.source_root.join(PACKAGE).into_os_string(),
        OsString::from("rz_pals_model.generation_numerical_cli"),
        OsString::from("--execute"),
        OsString::from("--original-numerical-child"),
        OsString::from("--startup-sha256"),
        OsString::from(
            &owner
                .startup
                .as_ref()
                .expect("held original startup")
                .pin
                .sha256,
        ),
        OsString::from("--startup-bytes"),
        OsString::from(stdin.len().to_string()),
        OsString::from("--execution-until-ns"),
        OsString::from(execution_ns.to_string()),
        OsString::from("--whole-until-ns"),
        OsString::from(whole_ns.to_string()),
    ];
    let duration = until.duration_since(started);
    let limits = ProcessLimits {
        wall_ms: u64::try_from(duration.as_nanos().div_ceil(1_000_000))
            .map_err(|_| invalid("numerical wall overflow"))?
            .max(1),
        shutdown_grace_ms: 15_000,
        max_output_bytes: 128 * 1024,
        max_child_processes: 1,
    };
    let window = OriginalProcessWindow::new(started, until, cleanup_until)?;
    let watch = OwnedArtifactTreeWatch::bounded_output(staging_cap, MAX_FILE);
    common::revalidate(owner, &mut credit)?;
    let mut mapped_credit = reserve_image(
        &registration.python_runtime.executable,
        &mut credit,
        until,
        cancel,
    )?;
    let mut verify_python = |file: &File| {
        common::verify_bytes(
            file,
            &registration.python_runtime.executable,
            false,
            &mut mapped_credit,
        )
        .map(|_| ())
        .map_err(|_| ())
    };
    let cwd = common::directory_file(&owner.staging)?;
    processes.push(supervise_verified_input_in_original_window(
        &python,
        &args,
        &cwd,
        limits,
        Some(cancel),
        &watch,
        OriginalProcessInput {
            bytes: &stdin,
            window,
        },
        &mut verify_python,
    )?);
    credit.until = cleanup_until;
    credit.cancel = &preserve_cancel;
    // stdout가 깨졌거나 receipt가 partial이어도 첫 오류를 해석하기 전에
    // 원래 두 경로의 FD를 보존한다. 회수 오류는 아래 원래 decode/status
    // 오류를 덮지 않고 별도로 기록하며 인수 권한을 발급하지 않는다.
    retain_reference_receipt_failure_files(
        receipt_files,
        receipt_preservation_errors,
        &registration.receipt_path,
        &root,
        staging_cap,
        &mut credit,
    );
    let raw_preservation = retain_partial_raw_outputs(
        receipt_files,
        &plan,
        &registration.reference_output_root,
        false,
        &mut credit,
    );
    if let Err(error) = &raw_preservation {
        account.uncertain.store(true, Ordering::Release);
        receipt_preservation_errors.push(format!("reference_raw_failure_readback: {error}"));
    }
    let reference = processes
        .last()
        .expect("actual original reference process retained");
    let summary: Value = common::decode(&reference.process().stdout)?;
    let pin: GenerationTrainingFilePinV3 =
        serde_json::from_value(summary["receipt"].clone()).map_err(|e| invalid(e.to_string()))?;
    check(
        pin.path == registration.receipt_path,
        "reference exact terminal path",
    )?;
    let (receipt_file, raw) = common::read_observed_output(pin, MAX_JSON, &mut credit, true)?;
    owner.receipt = Some(receipt_file);
    *reference_receipt = Some(common::decode(&raw)?);
    let receipt = reference_receipt
        .as_ref()
        .expect("actual reference receipt preserved");
    verify_reference_receipt(
        receipt,
        &summary,
        reference,
        owner,
        registration,
        &child_registration,
        &registration_sha256,
        &dependencies,
        start_ns,
        execution_ns,
        whole_ns,
    )?;
    // 원래 stdout/receipt 해석 또는 child/status 실패가 있으면 회수 오류가
    // 이를 대체하지 않는다. 원래 검증이 성공해도 raw 회수 실패는 인수를 막는다.
    raw_preservation?;
    check(
        receipt_preservation_errors.is_empty(),
        "reference receipt preservation incomplete; original decode/status passed but readback remains uncertain",
    )?;
    check(
        receipt["runtime_readback"]["torch"]
            .as_str()
            .and_then(|v| v.split('+').next())
            == Some(original.registration().training_runtime.torch.as_str()),
        "reference Torch version differs from original actual trainer runtime",
    )?;
    let references = hold_reference_outputs(
        &registration.reference_output_root,
        receipt,
        &plan,
        owner,
        &mut credit,
    )?;
    // 두 실행은 같은 원래 창을 공유한다. reference 준비 비용 이후 새 시계를
    // 발급하지 않고 남은 정수 초만 native 내부 cooperative 기한으로 전달한다.
    credit.until = until;
    credit.cancel = cancel;
    credit.take(0)?;
    capture.case_plan = actual_plan_pin.path.clone();
    capture.case_plan_sha256 = actual_plan_pin.sha256.clone();
    capture.export_manifest = manifest.path.clone();
    capture.output_root = registration.native_output_root.clone();
    capture.deadline_seconds = until.saturating_duration_since(Instant::now()).as_secs();
    capture.cleanup_seconds = cleanup_until.duration_since(until).as_secs().min(30);
    check(
        capture.deadline_seconds > 0 && capture.cleanup_seconds > 0,
        "original numerical remaining window exhausted",
    )?;
    let native_raw = serde_json::to_vec(&capture).map_err(|e| invalid(e.to_string()))?;
    ensure_output_credit(&root, staging_cap, native_raw.len() as u64, until, cancel)?;
    let native_config = common::write_owned_new(
        &loading.staging_root.join(format!(
            "numerical-native-{}.json",
            registration.numerical_id
        )),
        &native_raw,
    )?;
    let native_config_pin = native_config.pin.clone();
    owner.checkpoints.push(native_config);
    args = vec![native_config_pin.path.clone().into_os_string()];
    common::revalidate(owner, &mut credit)?;
    let mut mapped_credit =
        reserve_image(&registration.native_executable, &mut credit, until, cancel)?;
    let mut verify_native = |file: &File| {
        common::verify_bytes(
            file,
            &registration.native_executable,
            false,
            &mut mapped_credit,
        )
        .map(|_| ())
        .map_err(|_| ())
    };
    processes.push(supervise_numerical_input_in_original_window(
        &native,
        &args,
        &cwd,
        limits,
        Some(cancel),
        &watch,
        OriginalProcessInput {
            bytes: &native_raw,
            window,
        },
        &mut verify_native,
        OriginalNumericalEnvironment::Fp32Tf32Off,
    )?);
    credit.until = cleanup_until;
    credit.cancel = &preserve_cancel;
    let native_raw_preservation = retain_partial_raw_outputs(
        receipt_files,
        &plan,
        &registration.native_output_root,
        true,
        &mut credit,
    );
    if let Err(error) = &native_raw_preservation {
        account.uncertain.store(true, Ordering::Release);
        receipt_preservation_errors.push(format!("native_raw_failure_readback: {error}"));
    }
    let actual_native = processes.last().expect("original native process retained");
    actual_native.checked_loaded_image()?;
    actual_native.checked_timing()?;
    check(
        actual_native.transport_complete()
            && actual_native.observation().numerical_environment()
                == Some("NVIDIA_TF32_OVERRIDE=0"),
        "native original closure or fixed precision environment absent",
    )?;
    let stdout = std::str::from_utf8(&actual_native.process().stdout)
        .map_err(|_| invalid("native status UTF-8"))?;
    let status: Value = common::decode(
        stdout
            .trim()
            .strip_prefix("RZ_NUMERIC_CAPTURE_STATUS=")
            .ok_or_else(|| invalid("native original status absent"))?
            .as_bytes(),
    )?;
    check(
        status
            == serde_json::json!({"status":"captured","numerical_acceptance":false,"rules_legality_certified":false,"native_worker_joined":true}),
        "native physical join/status scope differs",
    )?;
    // 원래 loader/closure/timing/stdout/physical-status 오류를 먼저 보존한다.
    // 이 검증이 성공해도 raw 회수가 실패한 상태에서 수치 인수를 진행하지 않는다.
    native_raw_preservation?;
    let groups = plan["groups"]
        .as_array()
        .expect("validated original groups");
    check(
        references.len() == groups.len(),
        "reference group coverage differs from original case plan",
    )?;
    let mut finalized = input_config.clone();
    finalized.assets.retain(|a| {
        !matches!(
            a.purpose,
            Purpose::NumericalAcceptance | Purpose::NumericalRawEvidence
        )
    });
    let mut pairs = Vec::new();
    let mut total = 0u64;
    for group in groups {
        let id = group["id"].as_str().expect("validated original group ID");
        let reference = references
            .get(&format!("reference.{id}.json"))
            .ok_or_else(|| invalid("original reference group absent"))?;
        let native_path = registration
            .native_output_root
            .join(format!("native-{id}.json"));
        let native_pin = observe_raw(&native_path, &mut credit)?;
        let mut artifacts = Vec::new();
        for pin in [reference, &native_pin] {
            total = total
                .checked_add(pin.bytes)
                .filter(|n| *n <= MAX_RAW)
                .ok_or_else(|| invalid("numerical combined raw cap exceeded"))?;
            let actual = common::open_file(&pin.path)?;
            let raw = common::verify_bytes(&actual, pin, true, &mut credit)?;
            ensure_output_credit(
                &root,
                staging_cap,
                raw.len() as u64,
                cleanup_until,
                &preserve_cancel,
            )?;
            let target_name = format!(
                "numerical-{}-{}",
                registration.numerical_id,
                pin.path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| invalid("numerical raw target name"))?
            );
            let copy = common::write_owned_new(&private_cwd.join(&target_name), &raw)?;
            let asset = raw_asset(&copy.pin, target_name, Purpose::NumericalRawEvidence);
            artifacts.push(asset.artifact.clone());
            finalized.assets.push(asset);
            owner.checkpoints.push(copy);
        }
        let (file, _) = common::read_observed_output(native_pin, MAX_JSON, &mut credit, false)?;
        owner.checkpoints.push(file);
        pairs.push(PalsNumericalRunPairV5 {
            reference: artifacts.remove(0),
            native: artifacts.remove(0),
        });
    }
    let acceptance = PalsNumericalAcceptanceV5 {
        schema_version: 5,
        domain: rz_experiments::PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN.into(),
        bindings: expected,
        invocation: PalsNumericalInvocationV5::Fresh,
        batch: 1,
        runs: pairs,
    };
    let raw = serde_json::to_vec(&acceptance).map_err(|e| invalid(e.to_string()))?;
    check(raw.len() <= 256 * 1024, "numerical bounded acceptance")?;
    ensure_output_credit(
        &root,
        staging_cap,
        raw.len() as u64,
        cleanup_until,
        &preserve_cancel,
    )?;
    let file = common::write_owned_new(&registration.acceptance_path, &raw)?;
    let acceptance_pin = file.pin.clone();
    let relative = registration
        .acceptance_path
        .strip_prefix(private_cwd)
        .map_err(|_| invalid("acceptance must reside in original private cwd"))?
        .to_str()
        .ok_or_else(|| invalid("acceptance relative UTF-8"))?
        .replace('\\', "/");
    finalized
        .assets
        .push(raw_asset(&file.pin, relative, Purpose::NumericalAcceptance));
    owner.checkpoints.push(file);
    let admission = {
        let mut read = |artifact: &ArtifactRef,
                        limit: usize|
         -> Result<Vec<u8>, rz_experiments::ManifestError> {
            let asset = finalized
                .assets
                .iter()
                .find(|a| &a.artifact == artifact)
                .ok_or_else(|| {
                    rz_experiments::ManifestError::Artifact(
                        "numerical undeclared original artifact".into(),
                    )
                })?;
            let path = private_cwd.join(&asset.snapshot_path);
            let found = owner
                .checkpoints
                .iter()
                .map(|f| (&f.pin, &f.file))
                .chain(prepared_files.iter().copied())
                .find(|(pin, _)| {
                    pin.path == path && pin.sha256 == artifact.sha256 && pin.bytes == artifact.bytes
                })
                .ok_or_else(|| {
                    rz_experiments::ManifestError::Artifact(
                        "numerical original readonly descriptor absent".into(),
                    )
                })?;
            check(
                found.0.bytes <= limit as u64,
                "numerical verifier read limit",
            )
            .map_err(|e| rz_experiments::ManifestError::Artifact(e.to_string()))?;
            common::verify_bytes(found.1, found.0, true, &mut credit)
                .map_err(|e| rz_experiments::ManifestError::Artifact(e.to_string()))
        };
        rz_experiments::validate_pals_numerical_acceptance_v5(&finalized, &raw, &mut read)
            .map_err(|e| invalid(e.to_string()))?
    };
    finalized
        .validate_against(lock)
        .map_err(|e| invalid(e.to_string()))?;
    common::verify_bytes(registration_file, registration_pin, false, &mut credit)?;
    common::revalidate(owner, &mut credit)?;
    super::selfplay_cli::scan_staging(&root, staging_cap, cleanup_until, &preserve_cancel)?;
    owner.consumed_read_bytes = account.charged.load(Ordering::Acquire);
    let parent_receipt = serde_json::json!({"schema":"rz-pals-original-generation-numerical-parent/1","numerical_id":registration.numerical_id,
        "launch_registration":registration_pin,"actual_export_kind":export.kind(),"actual_export_receipt":export.receipt_file_pin(),
        "checkpoint":original_checkpoint,"case_plan":actual_plan_pin,"native_config":native_config_pin,
        "acceptance":acceptance_pin,"reference_process":processes[0].process().receipt,"reference_transport":processes[0].observation(),
        "native_process":processes[1].process().receipt,"native_transport":processes[1].observation(),
        "native_fixed_environment":"NVIDIA_TF32_OVERRIDE=0","admission":admission,
        "original_export_prepared_and_process_owners_retained":true,"charged_read_bytes":owner.consumed_read_bytes,
        "staging_cap_bytes":staging_cap,"automatic_retry":false,"rules_legality_certified":false,
        "arena_accepted":false,"sole_gpu_lease_observed_here":false});
    Ok(VerifiedNumericalCoreV3 {
        config: finalized,
        lock: lock.clone(),
        admission,
        acceptance: acceptance_pin,
        processes: std::mem::take(processes),
        parent_receipt,
        held: held.take().expect("verified original numerical custody"),
    })
}

pub(super) fn raw_asset(
    pin: &GenerationTrainingFilePinV3,
    snapshot_path: String,
    purpose: Purpose,
) -> PalsStartupAssetV5 {
    PalsStartupAssetV5 {
        artifact: ArtifactRef {
            path: format!("evidence/{snapshot_path}"),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
            source: "original generation numerical capture; readonly exact output".into(),
            license: "MIT".into(),
        },
        snapshot_path,
        purpose,
    }
}

#[cfg(target_os = "linux")]
pub(super) fn ensure_output_credit(
    root: &cap_std::fs::Dir,
    cap: u64,
    additional: u64,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<(), ArenaError> {
    let observed = super::selfplay_cli::scan_staging(root, cap, until, cancel)?;
    check(
        observed
            .unique_inode_bytes
            .checked_add(additional)
            .is_some_and(|bytes| bytes <= cap),
        "numerical original staging credit must be available before publication",
    )
}

fn validate_launch(
    r: &GenerationNumericalLaunchRegistrationV3,
    root: &Path,
    cwd: &Path,
    started: Instant,
    until: Instant,
) -> Result<(), ArenaError> {
    use super::training_process::{absolute_safe, plain_name};
    for (pin, max) in [
        (&r.template_startup, 8 * 1024 * 1024),
        (&r.case_plan, MAX_JSON),
        (&r.native_capture_config, 64 * 1024),
        (&r.native_executable, 256 * 1024 * 1024),
        (&r.python_runtime.executable, 256 * 1024 * 1024),
    ] {
        pin.validate(max)?;
    }
    check(
        r.schema == LAUNCH
            && plain_name(&r.numerical_id)
            && !r.python_runtime.version.is_empty()
            && (1..=8).contains(&r.dependency_directories.len())
            && (1..=64).contains(&r.runtime_module_pins.len())
            && r.dependency_directories
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                == r.dependency_directories.len()
            && r.runtime_module_pins
                .iter()
                .map(|p| &p.path)
                .collect::<BTreeSet<_>>()
                .len()
                == r.runtime_module_pins.len()
            && (1..=MAX_STAGING).contains(&r.max_output_bytes)
            && (1..=4 * MAX_STAGING).contains(&r.max_read_bytes)
            && (1..=3_600_000_000_000).contains(&r.max_wall_ns)
            && until.duration_since(started).as_nanos() <= u128::from(r.max_wall_ns)
            && cwd.starts_with(root)
            && cwd != root
            && r.acceptance_path.starts_with(cwd)
            && [
                &r.reference_output_root,
                &r.native_output_root,
                &r.receipt_path,
                &r.acceptance_path,
            ]
            .into_iter()
            .all(|p| absolute_safe(p) && p.starts_with(root) && p != root)
            && r.reference_output_root != r.native_output_root
            && r.receipt_path != r.acceptance_path
            && r.receipt_path
                .extension()
                .is_some_and(|value| value == "json"),
        "numerical exact finite original managed launch",
    )?;
    for pin in &r.runtime_module_pins {
        pin.validate(MAX_FILE)?;
    }
    Ok(())
}

pub(super) fn validate_plan(
    plan: &Value,
    expected: &PalsNumericalBindingsV5,
) -> Result<(), ArenaError> {
    check(
        plan.as_object().is_some_and(|o| o.len() == 6)
            && plan["schema_version"] == 5
            && plan["domain"] == "rz-pals-numerical-case-plan-v5/1"
            && plan["invocation"] == "fresh"
            && plan["batch"] == 1
            && expected.precision == rz_experiments::PalsPrecisionV3::Fp32
            && !expected.tf32,
        "original Fresh FP32 TF32-off B1 numerical plan",
    )?;
    let old: PalsNumericalBindingsV5 =
        serde_json::from_value(plan["bindings"].clone()).map_err(|e| invalid(e.to_string()))?;
    let mut a = old.clone();
    let mut b = expected.clone();
    for binding in [&mut a, &mut b] {
        binding.weights_sha256.clear();
        binding.export_manifest_sha256.clear();
        binding.graphs.clear();
        binding.model_epoch = 0;
    }
    check(
        a == b,
        "numerical plan may change only actual weight/export/graph/epoch bindings",
    )?;
    let groups = plan["groups"]
        .as_array()
        .ok_or_else(|| invalid("original numerical groups absent"))?;
    check(
        !groups.is_empty() && groups.len() <= 18,
        "numerical bounded original groups",
    )?;
    let mut ids = BTreeSet::new();
    let mut cases = 0usize;
    let mut roles = BTreeSet::new();
    for group in groups {
        let id = group["id"]
            .as_str()
            .ok_or_else(|| invalid("numerical group ID"))?;
        let role = group["role"]
            .as_str()
            .ok_or_else(|| invalid("numerical role"))?;
        roles.insert(role);
        check(
            super::training_process::plain_name(id) && ids.insert(id),
            "numerical original group identity",
        )?;
        let rows = group["cases"]
            .as_array()
            .ok_or_else(|| invalid("numerical original case list"))?;
        check(
            !rows.is_empty() && rows.len() <= 10,
            "numerical original case group bound",
        )?;
        cases += rows.len();
    }
    let expected_roles = if expected.graphs.contains_key("validator") {
        BTreeSet::from(["proposer", "critic", "validator"])
    } else {
        BTreeSet::from(["proposer", "critic"])
    };
    check(
        cases <= 256 && roles == expected_roles,
        "original numerical case role capability scope",
    )
}

#[cfg(target_os = "linux")]
pub(super) fn validate_capture_inputs(
    capture: &NativeCaptureConfig,
    config: &PalsAdaptiveStartupConfigV5,
    cwd: &Path,
    root: &Path,
) -> Result<(), ArenaError> {
    let path = |p| {
        config
            .asset(p)
            .map(|a| cwd.join(&a.snapshot_path))
            .map_err(|e| invalid(e.to_string()))
    };
    let cuda = config.endpoint.model.backend == rz_experiments::PalsModelBackendV3::OrtCuda;
    check(
        capture
            .cudnn_conv_search
            .as_deref()
            .unwrap_or(rz_experiments::PALS_CUDNN_IMPLICIT_EXHAUSTIVE_V5)
            == config.native_options.cudnn_conv_search,
        "native numerical cuDNN selection differs from original startup registration",
    )?;
    check(
        capture.runtime_library == path(Purpose::RuntimeLibrary)?
            && capture.runtime_cache.starts_with(root)
            && capture.runtime_cache != root
            && super::training_process::absolute_safe(&capture.runtime_cache)
            && capture.provider == if cuda { "cuda-control" } else { "cpu" }
            && !config.native_options.approximate_warm
            && !config.native_options.device_public_memory
            && capture.output_root_admission.is_none(),
        "native numerical original runtime/provider/Fresh capability",
    )?;
    if cuda {
        let inventory = config
            .asset(Purpose::CudaControlInventory)
            .map_err(|e| invalid(e.to_string()))?;
        check(
            capture.cuda_bundle == Some(path(Purpose::CudaManifest)?)
                && capture.cuda_inventory == Some(cwd.join(&inventory.snapshot_path))
                && capture.cuda_inventory_sha256.as_deref()
                    == Some(inventory.artifact.sha256.as_str())
                && Some(capture.gpu_device_id) == config.native_options.cuda_device_id
                && u64::try_from(capture.gpu_arena_bytes).ok()
                    == config.native_options.cuda_arena_bytes,
            "native numerical original CUDA/control/device/arena registrations",
        )?;
    } else {
        check(
            capture.cuda_bundle.is_none()
                && capture.cuda_inventory.is_none()
                && capture.cuda_inventory_sha256.is_none(),
            "CPU numerical cannot inherit CUDA registrations",
        )?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub(super) fn reserve_image<'a>(
    pin: &GenerationTrainingFilePinV3,
    credit: &mut super::training_process::ReadCredit<'_>,
    until: Instant,
    cancel: &'a AtomicBool,
) -> Result<super::training_process::ReadCredit<'a>, ArenaError> {
    let amount = pin
        .bytes
        .checked_mul(2)
        .ok_or_else(|| invalid("numerical mapped image credit overflow"))?;
    credit.take(amount)?;
    Ok(super::training_process::ReadCredit {
        remaining: amount,
        maximum: amount,
        until,
        cancel,
        account: super::training_process::ReadAccounting::new(),
    })
}

#[cfg(target_os = "linux")]
pub(super) fn observe_raw(
    path: &Path,
    credit: &mut super::training_process::ReadCredit<'_>,
) -> Result<GenerationTrainingFilePinV3, ArenaError> {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::FileExt;
    let file = super::training_process::open_file(path)?;
    let before = super::training_process::FileIdentity::from(
        &file
            .metadata()
            .map_err(|_| invalid("native raw metadata"))?,
    );
    let bytes = file
        .metadata()
        .map_err(|_| invalid("native raw bytes"))?
        .len();
    check(
        bytes > 0 && bytes <= MAX_JSON,
        "native bounded completed raw",
    )?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut offset = 0u64;
    while offset < bytes {
        let amount = (bytes - offset).min(buffer.len() as u64) as usize;
        credit.take(amount as u64)?;
        let got = file
            .read_at(&mut buffer[..amount], offset)
            .map_err(|_| invalid("native raw positional read"))?;
        check(got == amount, "native raw short read")?;
        digest.update(&buffer[..got]);
        offset += got as u64;
    }
    let pin = GenerationTrainingFilePinV3 {
        path: path.into(),
        sha256: format!("{:x}", digest.finalize()),
        bytes,
    };
    check(
        super::training_process::same_named(&pin, &file)? == before,
        "native raw identity changed during original observation",
    )?;
    Ok(pin)
}

/// 이 타입은 실패 자료의 원래 descriptor만 소유한다. malformed/빈 partial의
/// 관측 pin도 성공용 LinuxHeld나 수치 admission에 넣지 않는다.
#[cfg(target_os = "linux")]
struct FailureOnlyOutputFileV3 {
    _path: PathBuf,
    _file: File,
    _identity: Option<super::training_process::FileIdentity>,
    _observed_pin: Option<GenerationTrainingFilePinV3>,
}

/// 등록한 terminal receipt와 정확한 `.partial` 두 경로만 해석 전 보존한다.
/// 읽기 실패는 별도 기록하고 original stdout/receipt/status 오류를 바꾸지 않는다.
#[cfg(target_os = "linux")]
fn retain_reference_receipt_failure_files(
    files: &mut Vec<FailureOnlyOutputFileV3>,
    errors: &mut Vec<String>,
    receipt_path: &Path,
    staging: &cap_std::fs::Dir,
    staging_cap: u64,
    credit: &mut super::training_process::ReadCredit<'_>,
) {
    use super::training_process::{self as common, FileIdentity};
    use std::sync::atomic::Ordering;
    if let Err(error) =
        super::selfplay_cli::scan_staging(staging, staging_cap, credit.until, credit.cancel)
    {
        credit.account.uncertain.store(true, Ordering::Release);
        errors.push(format!("reference_receipt_staging_readback: {error}"));
        return;
    }
    let Some(name) = receipt_path.file_name() else {
        credit.account.uncertain.store(true, Ordering::Release);
        errors.push("reference_receipt_path_has_no_basename".into());
        return;
    };
    let mut partial_name = name.to_os_string();
    partial_name.push(".partial");
    for path in [
        receipt_path.to_path_buf(),
        receipt_path.with_file_name(partial_name),
    ] {
        let observed = (|| -> Result<(), ArenaError> {
            credit.take(0)?;
            match std::fs::symlink_metadata(&path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(_) => return Err(invalid("reference receipt named metadata unavailable")),
                Ok(metadata) => check(
                    metadata.is_file() && !metadata.file_type().is_symlink(),
                    "reference receipt must be an original regular file",
                )?,
            }
            let file = common::open_file(&path)?;
            // 첫 open 직후 보존한다. 이후 metadata/hash가 실패해도 이 FD를
            // 버리거나 실패를 빈 receipt/성공으로 바꾸지 않는다.
            files.push(FailureOnlyOutputFileV3 {
                _path: path.clone(),
                _file: file,
                _identity: None,
                _observed_pin: None,
            });
            let retained = files
                .last_mut()
                .expect("original receipt descriptor retained");
            let metadata = retained
                ._file
                .metadata()
                .map_err(|_| invalid("reference receipt held metadata unavailable"))?;
            check(metadata.is_file(), "reference held receipt must be regular")?;
            retained._identity = Some(FileIdentity::from(&metadata));
            check(
                metadata.len() <= MAX_JSON,
                "reference failure receipt exceeds raw cap",
            )?;
            retained._observed_pin = Some(observe_failure_output_bytes(
                &path,
                &retained._file,
                metadata.len(),
                credit,
            )?);
            Ok(())
        })();
        if let Err(error) = observed {
            credit.account.uncertain.store(true, Ordering::Release);
            errors.push(format!(
                "reference_receipt_failure_readback {}: {error}",
                path.display()
            ));
        }
    }
}

/// 완료 권한 없는 byte 관측이다. JSON을 해석하지 않고 빈 partial도 hash하며,
/// 원래 FD의 cursor/내용/identity는 변경하지 않는다.
#[cfg(target_os = "linux")]
fn observe_failure_output_bytes(
    path: &Path,
    file: &File,
    bytes: u64,
    credit: &mut super::training_process::ReadCredit<'_>,
) -> Result<GenerationTrainingFilePinV3, ArenaError> {
    use super::training_process::{self as common, FileIdentity};
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::FileExt;
    credit.take(0)?;
    let before = FileIdentity::from(
        &file
            .metadata()
            .map_err(|_| invalid("original failure output metadata"))?,
    );
    let mut pin = GenerationTrainingFilePinV3 {
        path: path.into(),
        sha256: String::new(),
        bytes,
    };
    check(
        common::same_named(&pin, file)? == before,
        "original failure output identity before read",
    )?;
    let mut digest = Sha256::new();
    let mut block = [0u8; 64 * 1024];
    let mut offset = 0;
    while offset < bytes {
        let amount = (bytes - offset).min(block.len() as u64) as usize;
        credit.take(amount as u64)?;
        let read = file
            .read_at(&mut block[..amount], offset)
            .map_err(|_| invalid("original failure output positional read unavailable"))?;
        check(read > 0, "original failure output premature EOF")?;
        digest.update(&block[..read]);
        offset += read as u64;
    }
    pin.sha256 = format!("{:x}", digest.finalize());
    check(
        common::same_named(&pin, file)? == before
            && FileIdentity::from(
                &file
                    .metadata()
                    .map_err(|_| invalid("original failure output final metadata"))?,
            ) == before,
        "original failure output identity changed while observing bytes",
    )?;
    credit.take(0)?;
    Ok(pin)
}

/// 성공 여부를 해석하기 전에 원래 child의 알려진 raw/pending 경로를
/// 보존한다. 빈 partial도 descriptor를 남기며 인수용 유효 pin으로 사용하지
/// 않는다. 실패 후의 읽기 전용 회수는 원래 whole deadline 안에서만 수행한다.
#[cfg(target_os = "linux")]
fn retain_partial_raw_outputs(
    files: &mut Vec<FailureOnlyOutputFileV3>,
    plan: &Value,
    root: &Path,
    native: bool,
    credit: &mut super::training_process::ReadCredit<'_>,
) -> Result<(), ArenaError> {
    use super::training_process::{self as common, FileIdentity};
    use std::os::unix::fs::MetadataExt;
    let mut names = Vec::new();
    for group in plan["groups"]
        .as_array()
        .ok_or_else(|| invalid("partial original group custody"))?
    {
        let id = group["id"]
            .as_str()
            .ok_or_else(|| invalid("partial original group identity"))?;
        if native {
            names.extend([
                format!("native-{id}.json"),
                format!("native-{id}.pending.json"),
            ]);
        } else {
            names.extend([
                format!("reference.{id}.json"),
                format!("reference.{id}.json.partial"),
            ]);
        }
    }
    if native {
        names.extend([
            "actual-stage-counters.json".into(),
            "actual-stage-counters.pending.json".into(),
            "actual-virtual-cuda-placement.json".into(),
        ]);
    }
    for name in names {
        credit.take(0)?;
        let path = root.join(name);
        match std::fs::symlink_metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => return Err(invalid("original partial raw named metadata")),
            Ok(metadata) => check(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "original partial raw must be regular",
            )?,
        }
        if files.iter().any(|file| file._path == path) {
            continue;
        }
        credit.take(0)?;
        let file = common::open_file(&path)?;
        // 첫 open부터 failure 전용 owner에 넣는다. 이후 metadata/sha/credit
        // 실패에도 이 원래 FD는 유지하며 성공용 HeldFile로 이동하지 않는다.
        files.push(FailureOnlyOutputFileV3 {
            _path: path.clone(),
            _file: file,
            _identity: None,
            _observed_pin: None,
        });
        let retained = files.last_mut().expect("original raw descriptor retained");
        let metadata = retained
            ._file
            .metadata()
            .map_err(|_| invalid("original partial raw held metadata"))?;
        retained._identity = Some(FileIdentity::from(&metadata));
        check(
            metadata.is_file()
                && metadata.uid() == nix::unistd::Uid::effective().as_raw()
                && metadata.len() <= MAX_JSON,
            "original partial raw bounded regular current-owner file",
        )?;
        // empty/two-link/pending도 관측만 한다. 실제 success raw는 아래 원래
        // strict read_observed_output/verify_bytes/revalidation 경로가 별도 검증한다.
        retained._observed_pin = Some(observe_failure_output_bytes(
            &path,
            &retained._file,
            metadata.len(),
            credit,
        )?);
    }
    Ok(())
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn verify_reference_receipt(
    receipt: &Value,
    summary: &Value,
    process: &OriginalProcessOutput,
    held: &super::training_process::LinuxHeld,
    r: &GenerationNumericalLaunchRegistrationV3,
    child: &Value,
    sha: &str,
    dependencies: &Value,
    start_ns: u64,
    execution_ns: u64,
    whole_ns: u64,
) -> Result<(), ArenaError> {
    use super::training_process::{self as common, ObservedFile};
    process.checked_loaded_image()?;
    process.checked_timing()?;
    check(
        process.transport_complete()
            && summary.as_object().is_some_and(|o| o.len() == 5)
            && summary["schema"] == RESULT
            && summary["status"] == "captured"
            && summary["actual_reference_executed"] == true
            && receipt["schema"] == RESULT
            && receipt["status"] == "captured"
            && summary["before_original_execution_deadline"] == true
            && receipt["before_original_execution_deadline"] == true
            && receipt["execution_id"] == r.numerical_id
            && receipt["registration_sha256"] == sha
            && receipt["original_execution_until_ns"] == execution_ns
            && receipt["original_whole_until_ns"] == whole_ns
            && receipt["optimizer_updates_executed"] == 0
            && receipt["native_numeric_authority_issued"] == false
            && receipt["parent_must_verify_process_and_artifacts"] == true
            && receipt["physical_completion"] == "reference_completed_cpu"
            && receipt["original_model_bits_preserved"] == true
            && receipt["reference_dispatch_started"] == true
            && receipt["capture_result"]["cases"] == receipt["reference_neural_forwards_executed"]
            && receipt["reference_neural_forwards_executed"]
                .as_u64()
                .is_some_and(|n| (1..=256).contains(&n)),
        "reference original transport/CPU/scope/exact export agreement",
    )?;
    let timing = process.checked_timing()?;
    check(
        timing.supervisor_entry_ns() <= timing.exit_observed_ns() && start_ns > 0,
        "reference original timing source",
    )?;
    check(
        receipt["active_source"]["source_components"] == child["source_components"],
        "reference original numerical source roster",
    )?;
    let modules = receipt["active_source"]["source_modules"]
        .as_array()
        .ok_or_else(|| invalid("reference actual imported source inventory"))?;
    check(
        !modules.is_empty() && modules.len() <= 128,
        "reference imported source bound",
    )?;
    let mut names = BTreeSet::new();
    for module in modules {
        let name = module["module"]
            .as_str()
            .ok_or_else(|| invalid("reference module name"))?;
        check(
            names.insert(name)
                && (name == "__main__"
                    || name == "rz_pals_model"
                    || name.starts_with("rz_pals_model.")),
            "reference source namespace/duplicate",
        )?;
        let observed: ObservedFile = serde_json::from_value(
            serde_json::json!({"path":module["path"],"sha256":module["sha256"],
            "bytes":module["bytes"],"device":module["device"],"inode":module["inode"]}),
        )
        .map_err(|e| invalid(e.to_string()))?;
        check(
            held.sources
                .iter()
                .any(|file| common::readback_matches(&observed, file)),
            "reference actual imported module is not held compiled source",
        )?;
        if name == "__main__" {
            check(
                observed.path
                    == held
                        .source_root
                        .join(PACKAGE)
                        .join("rz_pals_model/generation_numerical_cli.py"),
                "reference fixed original numerical entry point",
            )?;
        }
    }
    check(
        [
            "__main__",
            "rz_pals_model.numeric_capture",
            "rz_pals_model.model",
            "rz_pals_model.virtual_review",
        ]
        .into_iter()
        .all(|name| names.contains(name)),
        "reference required loaded source module absent",
    )?;
    let python = &receipt["interpreter_readback"];
    let observed: ObservedFile =
        serde_json::from_value(python["executable"].clone()).map_err(|e| invalid(e.to_string()))?;
    let original_python = held
        .inputs
        .iter()
        .find(|file| file.pin == r.python_runtime.executable)
        .ok_or_else(|| invalid("reference original interpreter descriptor absent"))?;
    check(
        common::readback_matches(&observed, original_python)
            && python["version"].as_str().is_some_and(|s| {
                s.split_whitespace().next() == Some(r.python_runtime.version.as_str())
            })
            && python["mapped_executable_identity"]["device"] == observed.device
            && python["mapped_executable_identity"]["inode"] == observed.inode
            && python["os"] == "posix"
            && python["platform"] == "linux"
            && python["isolated"] == true
            && python["no_site"] == true
            && python["bytecode_disabled"] == true
            && python["dependency_pth_executed"] == false
            && python["dependency_directories"] == *dependencies
            && python["cwd"] == child["staging_root"],
        "reference original interpreter/isolated dependency readback",
    )?;
    let runtime = &receipt["runtime_readback"];
    check(
        runtime["device"] == "cpu"
            && runtime["precision"] == "fp32"
            && runtime["cuda_matmul_tf32"] == false
            && runtime["cudnn_tf32"] == false
            && runtime["cpu_threads"]
                .as_u64()
                .is_some_and(|n| (1..=2).contains(&n))
            && runtime["interop_threads"]
                .as_u64()
                .is_some_and(|n| (1..=2).contains(&n))
            && r.runtime_module_pins
                .iter()
                .any(|pin| runtime["torch_module_path"] == pin.path.to_string_lossy().as_ref()),
        "reference actual CPU/thread/precision/module readback",
    )?;
    for stage in [
        "runtime_module_readbacks_before",
        "runtime_module_readbacks_after",
    ] {
        let readbacks = receipt[stage]
            .as_array()
            .ok_or_else(|| invalid("reference explicit runtime file readbacks absent"))?;
        check(
            readbacks.len() == r.runtime_module_pins.len(),
            "reference runtime file roster differs",
        )?;
        let mut paths = BTreeSet::new();
        for value in readbacks {
            let observed: ObservedFile =
                serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
            check(
                paths.insert(observed.path.clone())
                    && r.runtime_module_pins.contains(&observed.pin())
                    && held
                        .inputs
                        .iter()
                        .any(|file| common::readback_matches(&observed, file)),
                "reference original runtime file inode/readback differs",
            )?;
        }
    }
    let inputs = &receipt["original_input_readbacks"];
    for name in [
        "template_startup",
        "case_plan",
        "checkpoint",
        "export_manifest",
    ] {
        let observed: ObservedFile =
            serde_json::from_value(inputs[name].clone()).map_err(|e| invalid(e.to_string()))?;
        let pin: GenerationTrainingFilePinV3 =
            serde_json::from_value(child[name].clone()).map_err(|e| invalid(e.to_string()))?;
        check(
            observed.pin() == pin
                && held
                    .inputs
                    .iter()
                    .chain(&held.checkpoints)
                    .any(|file| common::readback_matches(&observed, file)),
            "reference original input descriptor differs",
        )?;
    }
    let graph_readbacks = inputs["export_graphs"]
        .as_array()
        .ok_or_else(|| invalid("reference original graph readback absent"))?;
    let declared = child["export_graphs"]
        .as_array()
        .ok_or_else(|| invalid("reference original graph declaration absent"))?;
    check(
        graph_readbacks.len() == declared.len(),
        "reference original graph count differs",
    )?;
    for value in graph_readbacks {
        let observed: ObservedFile =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        check(
            declared.iter().any(|pin| {
                pin["path"] == observed.path.to_string_lossy().as_ref()
                    && pin["sha256"] == observed.sha256
                    && pin["bytes"] == observed.bytes
            }) && held
                .inputs
                .iter()
                .any(|file| common::readback_matches(&observed, file)),
            "reference exact original graph inode/readback differs",
        )?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
const MAX_REFERENCE_OWNER_BYTES: u64 = 1024;

#[cfg(target_os = "linux")]
fn verify_reference_owner_marker(
    file: &File,
    pin: &GenerationTrainingFilePinV3,
    credit: &mut super::training_process::ReadCredit<'_>,
) -> Result<(), ArenaError> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct OwnerMarker {
        schema: String,
        owner: String,
    }
    pin.validate(MAX_REFERENCE_OWNER_BYTES)?;
    let raw = super::training_process::verify_bytes(file, pin, true, credit)?;
    let marker: OwnerMarker = super::training_process::decode(&raw)?;
    check(
        marker.schema == "rovezero.pals-model.v1" && marker.owner == "rz-pals-model",
        "numerical reference original artifact owner marker differs",
    )
}

#[cfg(target_os = "linux")]
struct ReferenceOutputRoster<'a> {
    owner: &'a Value,
    groups: Vec<(String, &'a Value)>,
}

#[cfg(target_os = "linux")]
fn reference_output_roster<'a>(
    root: &Path,
    receipt: &'a Value,
    plan: &Value,
) -> Result<ReferenceOutputRoster<'a>, ArenaError> {
    let rows = receipt["artifact_pins"]
        .as_array()
        .ok_or_else(|| invalid("numerical reference original raw roster absent"))?;
    let groups = plan["groups"]
        .as_array()
        .ok_or_else(|| invalid("numerical original groups absent"))?;
    // The capture producer publishes an ownership marker alongside its raw
    // groups. It is retained metadata, not another numerical group.
    check(
        groups.len().checked_add(1) == Some(rows.len()),
        "numerical reference exact group coverage",
    )?;
    let owner_path = root.join("pals-artifact-owner.json");
    let owner = rows
        .iter()
        .find(|row| row["path"] == owner_path.to_string_lossy().as_ref())
        .ok_or_else(|| invalid("numerical reference original owner marker absent"))?;
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::with_capacity(groups.len());
    for group in groups {
        let id = group["id"]
            .as_str()
            .ok_or_else(|| invalid("numerical original group ID absent"))?;
        let name = format!("reference.{id}.json");
        check(
            seen.insert(name.clone()),
            "duplicate numerical reference group",
        )?;
        let path = root.join(&name);
        let row = rows
            .iter()
            .find(|row| row["path"] == path.to_string_lossy().as_ref())
            .ok_or_else(|| invalid("numerical original raw output group absent"))?;
        ordered.push((name, row));
    }
    // Exact count plus all distinct expected paths excludes duplicate,
    // missing and unknown rows without changing the submitted group order.
    Ok(ReferenceOutputRoster {
        owner,
        groups: ordered,
    })
}

#[cfg(target_os = "linux")]
fn retain_reference_output_row(
    row: &Value,
    path: &Path,
    cap: u64,
    owner: &mut super::training_process::LinuxHeld,
    credit: &mut super::training_process::ReadCredit<'_>,
) -> Result<GenerationTrainingFilePinV3, ArenaError> {
    use super::training_process::{self as common, ObservedFile};
    let observed: ObservedFile =
        serde_json::from_value(row.clone()).map_err(|e| invalid(e.to_string()))?;
    check(
        observed.path == path,
        "numerical reference exact output path differs",
    )?;
    let (file, _) = common::read_observed_output(observed.pin(), cap, credit, false)?;
    check(
        common::readback_matches(&observed, &file),
        "numerical reference original inode changed",
    )?;
    let pin = file.pin.clone();
    owner.checkpoints.push(file);
    Ok(pin)
}

#[cfg(target_os = "linux")]
pub(super) fn hold_reference_outputs(
    root: &Path,
    receipt: &Value,
    plan: &Value,
    owner: &mut super::training_process::LinuxHeld,
    credit: &mut super::training_process::ReadCredit<'_>,
) -> Result<BTreeMap<String, GenerationTrainingFilePinV3>, ArenaError> {
    let roster = reference_output_roster(root, receipt, plan)?;
    let owner_pin = retain_reference_output_row(
        roster.owner,
        &root.join("pals-artifact-owner.json"),
        MAX_REFERENCE_OWNER_BYTES,
        owner,
        credit,
    )?;
    let marker = owner
        .checkpoints
        .last()
        .ok_or_else(|| invalid("numerical reference original owner descriptor absent"))?;
    check(
        marker.pin == owner_pin,
        "numerical reference original owner descriptor differs",
    )?;
    verify_reference_owner_marker(&marker.file, &marker.pin, credit)?;
    let mut references = BTreeMap::new();
    for (name, row) in roster.groups {
        let pin = retain_reference_output_row(row, &root.join(&name), MAX_JSON, owner, credit)?;
        check(
            references.insert(name, pin).is_none(),
            "duplicate numerical reference group",
        )?;
    }
    Ok(references)
}

#[cfg(all(test, target_os = "linux"))]
mod reference_roster_tests {
    use super::super::training_process::{self as common, ReadAccounting, ReadCredit};
    use super::*;

    #[test]
    fn reference_roster_keeps_plan_order_and_exact_group_plus_owner_coverage() {
        let root = Path::new("/owned/reference");
        let plan = serde_json::json!({"groups":[
            {"id":"proposer_L8"}, {"id":"critic_L8"}
        ]});
        let receipt = serde_json::json!({"artifact_pins":[
            {"path":"/owned/reference/pals-artifact-owner.json"},
            {"path":"/owned/reference/reference.critic_L8.json"},
            {"path":"/owned/reference/reference.proposer_L8.json"}
        ]});
        let frozen_plan = plan.clone();
        let roster = reference_output_roster(root, &receipt, &plan).unwrap();
        assert_eq!(
            roster.owner["path"],
            "/owned/reference/pals-artifact-owner.json"
        );
        assert_eq!(
            roster
                .groups
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            ["reference.proposer_L8.json", "reference.critic_L8.json"]
        );
        assert_eq!(plan, frozen_plan);

        let mut extra = receipt.clone();
        extra["artifact_pins"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"path":"/owned/reference/unexpected.json"}));
        let mut missing_raw = receipt.clone();
        missing_raw["artifact_pins"].as_array_mut().unwrap().pop();
        let mut missing_owner = receipt.clone();
        missing_owner["artifact_pins"][0]["path"] =
            serde_json::json!("/unowned/pals-artifact-owner.json");
        let mut duplicate_raw = receipt.clone();
        duplicate_raw["artifact_pins"][1] = duplicate_raw["artifact_pins"][2].clone();
        let mut duplicate_owner = receipt.clone();
        duplicate_owner["artifact_pins"][1] = duplicate_owner["artifact_pins"][0].clone();
        for bad in [
            extra,
            missing_raw,
            missing_owner,
            duplicate_raw,
            duplicate_owner,
        ] {
            assert!(reference_output_roster(root, &bad, &plan).is_err());
        }
        let duplicate_plan = serde_json::json!({"groups":[
            {"id":"proposer_L8"}, {"id":"proposer_L8"}
        ]});
        assert!(reference_output_roster(root, &receipt, &duplicate_plan).is_err());
    }
    #[test]
    fn owner_marker_content_uses_original_fd_credit_and_strict_two_field_schema() {
        use std::os::unix::fs::DirBuilderExt;
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        struct MarkerDirectory(PathBuf);
        impl Drop for MarkerDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(self.0.join("pals-artifact-owner.json"));
                let _ = std::fs::remove_dir(&self.0);
            }
        }
        let new_directory = || {
            let parent = std::env::temp_dir().canonicalize().unwrap();
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = parent.join(format!(
                "rz-g0-owner-marker-{}-{}-{nonce}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&path)
                .unwrap();
            MarkerDirectory(path)
        };
        let cancel = AtomicBool::new(false);
        let mut credit = ReadCredit {
            remaining: 16 * MAX_REFERENCE_OWNER_BYTES,
            maximum: 16 * MAX_REFERENCE_OWNER_BYTES,
            until: Instant::now() + Duration::from_secs(5),
            cancel: &cancel,
            account: ReadAccounting::new(),
        };
        let directory = new_directory();
        let path = directory.0.join("pals-artifact-owner.json");
        let valid = br#"{"schema":"rovezero.pals-model.v1","owner":"rz-pals-model"}"#;
        let original = common::write_owned_new(&path, valid).unwrap();
        let before = credit.remaining;
        verify_reference_owner_marker(&original.file, &original.pin, &mut credit).unwrap();
        assert_eq!(before - credit.remaining, valid.len() as u64);
        assert_eq!(
            credit.account.charged.load(Ordering::Acquire),
            valid.len() as u64
        );
        credit.remaining = 0;
        assert!(verify_reference_owner_marker(&original.file, &original.pin, &mut credit).is_err());
        // A replacement pathname never authorizes the retained original FD.
        std::fs::remove_file(&path).unwrap();
        let replacement = common::write_owned_new(&path, valid).unwrap();
        let mut remaining = ReadCredit {
            remaining: 16 * MAX_REFERENCE_OWNER_BYTES,
            maximum: 16 * MAX_REFERENCE_OWNER_BYTES,
            until: Instant::now() + Duration::from_secs(5),
            cancel: &cancel,
            account: ReadAccounting::new(),
        };
        assert!(
            verify_reference_owner_marker(&original.file, &original.pin, &mut remaining).is_err()
        );
        drop(replacement);
        for raw in [
            br#"{"schema":"rovezero.pals-model.v3","owner":"rz-pals-model"}"#.as_slice(),
            br#"{"schema":"rovezero.pals-model.v1","owner":"other"}"#,
            br#"{"schema":"rovezero.pals-model.v1","owner":"rz-pals-model","extra":true}"#,
            br#"{"schema":"rovezero.pals-model.v1","owner":"rz-pals-model","owner":"other"}"#,
            br#"{"schema":"rovezero.pals-model.v1"}"#,
            b"not-json",
        ] {
            let directory = new_directory();
            let held = common::write_owned_new(&directory.0.join("pals-artifact-owner.json"), raw)
                .unwrap();
            assert!(verify_reference_owner_marker(&held.file, &held.pin, &mut remaining).is_err());
        }
        let oversized = new_directory();
        let held = common::write_owned_new(
            &oversized.0.join("pals-artifact-owner.json"),
            &vec![b' '; MAX_REFERENCE_OWNER_BYTES as usize + 1],
        )
        .unwrap();
        assert!(verify_reference_owner_marker(&held.file, &held.pin, &mut remaining).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    #[test]
    fn raw_failure_custody_survives_two_links_hash_rejection_and_exhausted_credit() {
        use super::super::training_process::{self as common, ReadAccounting, ReadCredit};
        use std::io::{Seek, SeekFrom};
        use std::os::unix::fs::{FileExt, MetadataExt};
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        struct RawDirectory(PathBuf);
        impl Drop for RawDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(self.0.join("reference.case.json"));
                let _ = std::fs::remove_file(self.0.join("reference.case.json.partial"));
                let _ = std::fs::remove_dir(&self.0);
            }
        }
        let directory = RawDirectory(std::env::temp_dir().join(format!(
            "rz-numerical-raw-custody-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        )));
        std::fs::create_dir(&directory.0).unwrap();
        let path = directory.0.join("reference.case.json");
        let pending = directory.0.join("reference.case.json.partial");
        let bytes = b"{\"status\":\"unfinished\"}";
        std::fs::write(&path, bytes).unwrap();
        std::fs::hard_link(&path, &pending).unwrap();
        let plan = serde_json::json!({"groups": [{"id": "case"}]});
        let cancel = AtomicBool::new(false);
        let mut credit = ReadCredit {
            remaining: 1024,
            maximum: 1024,
            until: Instant::now() + Duration::from_secs(5),
            cancel: &cancel,
            account: ReadAccounting::new(),
        };
        let mut files = Vec::new();
        retain_partial_raw_outputs(&mut files, &plan, &directory.0, false, &mut credit).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0]._file.metadata().unwrap().nlink(), 2);
        let pin = files[0]._observed_pin.as_ref().unwrap().clone();
        // 이 관측은 실패 자료 보존일 뿐이며 two-link raw의 strict 인수를 허용하지 않는다.
        assert!(common::read_observed_output(pin.clone(), MAX_JSON, &mut credit, false).is_err());
        let mut retained = &files[0]._file;
        retained.seek(SeekFrom::Start(2)).unwrap();
        observe_failure_output_bytes(&path, retained, bytes.len() as u64, &mut credit).unwrap();
        assert_eq!(retained.stream_position().unwrap(), 2);

        std::fs::remove_file(&pending).unwrap();
        assert_eq!(retained.metadata().unwrap().nlink(), 1);
        let mut wrong_hash = pin;
        wrong_hash.sha256 = "0".repeat(64);
        assert!(common::read_observed_output(wrong_hash, MAX_JSON, &mut credit, false).is_err());
        let mut original = vec![0; bytes.len()];
        assert_eq!(retained.read_at(&mut original, 0).unwrap(), bytes.len());
        assert_eq!(original, bytes);
        assert_eq!(retained.stream_position().unwrap(), 2);

        let mut failed_files = Vec::new();
        credit.remaining = 0;
        assert!(
            retain_partial_raw_outputs(&mut failed_files, &plan, &directory.0, false, &mut credit)
                .is_err()
        );
        assert_eq!(failed_files.len(), 1);
        assert!(failed_files[0]._observed_pin.is_none());
        assert!(failed_files[0]._identity.is_some());
        assert_eq!(
            failed_files[0]._file.metadata().unwrap().len(),
            bytes.len() as u64
        );

        let mut expired_files = Vec::new();
        credit.until = Instant::now();
        assert!(
            retain_partial_raw_outputs(&mut expired_files, &plan, &directory.0, false, &mut credit)
                .is_err()
        );
        assert!(expired_files.is_empty());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn malformed_and_empty_receipts_are_failure_only_and_read_failure_keeps_the_fd() {
        use super::super::training_process::{self as common, ReadAccounting, ReadCredit};
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        struct ReceiptDirectory(PathBuf);
        impl Drop for ReceiptDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(self.0.join("receipt.json"));
                let _ = std::fs::remove_file(self.0.join("receipt.json.partial"));
                let _ = std::fs::remove_dir(&self.0);
            }
        }
        let directory = ReceiptDirectory(std::env::temp_dir().join(format!(
            "rz-numerical-receipt-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        )));
        std::fs::create_dir(&directory.0).unwrap();
        let path = directory.0.join("receipt.json");
        let partial = directory.0.join("receipt.json.partial");
        let malformed = b"{unfinished";
        std::fs::write(&path, malformed).unwrap();
        std::fs::write(&partial, []).unwrap();
        let staging = common::open_directory(&directory.0).unwrap();
        let cancel = AtomicBool::new(false);
        let account = ReadAccounting::new();
        let mut credit = ReadCredit {
            remaining: 128,
            maximum: 128,
            until: Instant::now() + Duration::from_secs(5),
            cancel: &cancel,
            account: Arc::clone(&account),
        };
        let mut files = Vec::new();
        let mut errors = Vec::new();
        retain_reference_receipt_failure_files(
            &mut files,
            &mut errors,
            &path,
            &staging,
            128,
            &mut credit,
        );
        assert!(errors.is_empty());
        assert_eq!(files.len(), 2);
        assert_eq!(
            files[0]._observed_pin.as_ref().unwrap().bytes,
            malformed.len() as u64
        );
        let empty_partial = files[1]._observed_pin.as_ref().unwrap();
        assert_eq!(empty_partial.bytes, 0);
        assert!(empty_partial.validate(MAX_JSON).is_err());
        assert!(common::decode::<Value>(malformed).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), malformed);
        assert!(std::fs::read(&partial).unwrap().is_empty());

        let mut failed_files = Vec::new();
        let mut failed_errors = Vec::new();
        credit.remaining = 0;
        retain_reference_receipt_failure_files(
            &mut failed_files,
            &mut failed_errors,
            &path,
            &staging,
            128,
            &mut credit,
        );
        assert!(!failed_errors.is_empty());
        assert!(account.uncertain.load(Ordering::Acquire));
        assert_eq!(failed_files.len(), 2);
        assert!(failed_files[0]._observed_pin.is_none());
        assert_eq!(
            failed_files[0]._file.metadata().unwrap().len(),
            malformed.len() as u64
        );
    }

    fn pin(name: &str) -> GenerationTrainingFilePinV3 {
        GenerationTrainingFilePinV3 {
            path: PathBuf::from(format!("/metadata/{name}")),
            sha256: "a".repeat(64),
            bytes: 1,
        }
    }
    fn launch() -> GenerationNumericalLaunchRegistrationV3 {
        GenerationNumericalLaunchRegistrationV3 {
            schema: LAUNCH.into(),
            numerical_id: "math-1".into(),
            template_startup: pin("template.json"),
            case_plan: pin("cases.json"),
            native_capture_config: pin("native.json"),
            native_executable: pin("native"),
            python_runtime: GenerationTrainingPythonRuntimeV3 {
                version: "3.12.3".into(),
                executable: pin("python"),
            },
            dependency_directories: vec![PathBuf::from("/trusted")],
            runtime_module_pins: vec![pin("torch.py")],
            reference_output_root: PathBuf::from("/managed/reference"),
            native_output_root: PathBuf::from("/managed/native"),
            receipt_path: PathBuf::from("/managed/reference-receipt.json"),
            acceptance_path: PathBuf::from("/managed/private/acceptance.json"),
            max_wall_ns: 1_000_000_000,
            max_output_bytes: MAX_STAGING,
            max_read_bytes: MAX_STAGING,
        }
    }
    #[test]
    fn startup_environment_cannot_be_added_to_preregistered_launch() {
        let mut value = serde_json::to_value(launch()).unwrap();
        assert!(
            super::super::training_process::decode::<GenerationNumericalLaunchRegistrationV3>(
                &serde_json::to_vec(&value).unwrap()
            )
            .is_ok()
        );
        value["env"] = serde_json::json!({"CUDA_VISIBLE_DEVICES":"1"});
        assert!(
            super::super::training_process::decode::<GenerationNumericalLaunchRegistrationV3>(
                &serde_json::to_vec(&value).unwrap()
            )
            .is_err()
        );
    }
    #[test]
    fn numerical_rebinding_changes_weights_but_rejects_runtime_and_teacher_role() {
        let old: PalsNumericalBindingsV5 = serde_json::from_value(serde_json::json!({
            "model_profile":"virtual_review_v1","model_schema":"rovezero.pals-model.v3",
            "model_semantics":"rovezero.pals-model-semantics.v3","encoding_schema":"review-v3",
            "model_semantic_sha256":"a".repeat(64),"encoding_sha256":"b".repeat(64),"model_epoch":1,
            "weights_sha256":"c".repeat(64),"export_manifest_sha256":"d".repeat(64),
            "graphs":{"public":"e".repeat(64),"shared_pc":"f".repeat(64)},
            "runtime_library_sha256":"1".repeat(64),"cuda_libraries":[],"cuda_manifest_sha256":null,
            "backend":"ort_cpu","precision":"fp32","tf32":false,"source_components":{}
        }))
        .unwrap();
        let plan = serde_json::json!({"schema_version":5,"domain":"rz-pals-numerical-case-plan-v5/1",
            "bindings":old,"invocation":"fresh","batch":1,"groups":[
                {"id":"p-8","role":"proposer","line_bucket":8,"cases":[{"id":"p-case"}]},
                {"id":"c-8","role":"critic","line_bucket":8,"cases":[{"id":"c-case"}]}]});
        let mut new = old.clone();
        new.weights_sha256 = "2".repeat(64);
        new.model_epoch = 2;
        assert!(validate_plan(&plan, &new).is_ok());
        new.runtime_library_sha256 = "3".repeat(64);
        assert!(validate_plan(&plan, &new).is_err());
        let mut tf32_expected = old.clone();
        tf32_expected.tf32 = true;
        let mut tf32_plan = plan.clone();
        tf32_plan["bindings"]["tf32"] = Value::Bool(true);
        assert!(validate_plan(&tf32_plan, &tf32_expected).is_err());
        let mut fp16_expected = old.clone();
        fp16_expected.precision = rz_experiments::PalsPrecisionV3::Fp16;
        let mut fp16_plan = plan.clone();
        fp16_plan["bindings"]["precision"] = Value::String("fp16".into());
        assert!(validate_plan(&fp16_plan, &fp16_expected).is_err());
        let mut wrong_role = plan.clone();
        wrong_role["groups"][1]["role"] = Value::String("validator".into());
        assert!(validate_plan(&wrong_role, &old).is_err());
    }
}
