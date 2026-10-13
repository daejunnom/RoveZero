//! Original, non-training G0 initialization → export → independent numerical
//! capture. JSON describes the execution; only this live process/FD owner can
//! issue numerical-only completion. It cannot issue startup content admission,
//! collection, optimizer or arena authority. A later startup must independently
//! admit the retained original file roster with the existing content guard.
use super::{
    GenerationTrainingFilePinV3 as Pin, GenerationTrainingPythonRuntimeV3, check, invalid,
};
use crate::{ArenaError, process::OriginalProcessOutput};
use rz_experiments::{PalsAdaptiveStartupConfigV5, PalsInputLockV5, PalsNumericalAdmissionV5};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(target_os = "linux")]
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
use std::{
    collections::BTreeSet,
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

pub const G0_NUMERICAL_LAUNCH_V5: &str = "rz-pals-g0-numerical-launch/1";
pub const G0_NUMERICAL_TEMPLATE_V5: &str = "rz-pals-g0-numerical-startup-template/1";
const MAX_JSON: u64 = 4 << 20;
const MAX_FILE: u64 = 1 << 30;
const MAX_STAGING: u64 = 18 << 30;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalInputPinV5 {
    pub snapshot_path: String,
    pub source: Pin,
}
/// Only unchanged runtime/Rules/source snapshots are original inputs. The
/// template's old model pins are never assertions about a future G0 output.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalStartupTemplateV5 {
    pub schema: String,
    pub config: PalsAdaptiveStartupConfigV5,
    pub lock: PalsInputLockV5,
    pub private_cwd: PathBuf,
    pub input_pins: Vec<G0NumericalInputPinV5>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0NumericalLaunchRegistrationV5 {
    pub schema: String,
    pub numerical_id: String,
    pub source_commit: String,
    pub template_startup: Pin,
    pub case_plan: Pin,
    pub rules_profile: Pin,
    pub native_capture_config: Pin,
    pub native_executable: Pin,
    pub python_runtime: GenerationTrainingPythonRuntimeV3,
    pub dependency_directories: Vec<PathBuf>,
    pub runtime_module_pins: Vec<Pin>,
    pub staging_root: PathBuf,
    pub initialization_directory: PathBuf,
    pub export_directory: PathBuf,
    pub reference_output_root: PathBuf,
    pub native_output_root: PathBuf,
    pub prepare_receipt_path: PathBuf,
    pub reference_receipt_path: PathBuf,
    pub acceptance_path: PathBuf,
    pub parent_receipt_path: PathBuf,
    pub cudnn_conv_search: String,
    pub max_wall_ns: u64,
    pub max_output_bytes: u64,
    pub max_read_bytes: u64,
}
impl G0NumericalLaunchRegistrationV5 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        use super::training_process::{absolute_safe, plain_name};
        check(
            self.schema == G0_NUMERICAL_LAUNCH_V5
                && plain_name(&self.numerical_id)
                && self.numerical_id.len() <= 96
                && self.source_commit.len() == 40
                && self
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                && (1..=3_600_000_000_000).contains(&self.max_wall_ns)
                && (1..=MAX_STAGING).contains(&self.max_output_bytes)
                && (1..=4 * MAX_STAGING).contains(&self.max_read_bytes)
                && matches!(
                    self.cudnn_conv_search.as_str(),
                    "legacy_implicit_exhaustive" | "experimental_default"
                )
                && !self.python_runtime.version.is_empty(),
            "G0 finite explicit non-training registration",
        )?;
        for (pin, cap) in [
            (&self.template_startup, 8 << 20),
            (&self.case_plan, MAX_JSON),
            (&self.rules_profile, MAX_JSON),
            (&self.native_capture_config, 64 << 10),
            (&self.native_executable, 256 << 20),
            (&self.python_runtime.executable, 256 << 20),
        ] {
            pin.validate(cap)?;
        }
        check(
            (1..=8).contains(&self.dependency_directories.len())
                && (1..=64).contains(&self.runtime_module_pins.len())
                && self.dependency_directories.iter().all(|p| absolute_safe(p))
                && self
                    .dependency_directories
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.dependency_directories.len()
                && self
                    .runtime_module_pins
                    .iter()
                    .map(|p| &p.path)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.runtime_module_pins.len(),
            "G0 unique explicit trusted installed dependencies",
        )?;
        for pin in &self.runtime_module_pins {
            pin.validate(MAX_FILE)?;
            check(
                self.dependency_directories
                    .iter()
                    .any(|p| pin.path.starts_with(p)),
                "G0 runtime pin outside registered roots",
            )?;
        }
        check(
            absolute_safe(&self.staging_root),
            "G0 original nonsecret staging root",
        )?;
        let directories = [
            &self.initialization_directory,
            &self.export_directory,
            &self.reference_output_root,
            &self.native_output_root,
        ];
        let receipts = [
            &self.prepare_receipt_path,
            &self.reference_receipt_path,
            &self.acceptance_path,
            &self.parent_receipt_path,
        ];
        let paths = directories.into_iter().chain(receipts).collect::<Vec<_>>();
        check(
            paths.iter().all(|p| {
                absolute_safe(p) && p.starts_with(&self.staging_root) && **p != self.staging_root
            }) && paths.iter().collect::<BTreeSet<_>>().len() == paths.len()
                && directories
                    .iter()
                    .all(|a| paths.iter().all(|b| *a == *b || !b.starts_with(a)))
                && receipts
                    .iter()
                    .all(|p| p.extension().is_some_and(|s| s == "json")),
            "G0 outputs are separate finite owned children, never aliases",
        )?;
        for input in [
            &self.template_startup,
            &self.case_plan,
            &self.rules_profile,
            &self.native_capture_config,
            &self.native_executable,
            &self.python_runtime.executable,
        ] {
            check(
                !directories.iter().any(|p| input.path.starts_with(p))
                    && !receipts.iter().any(|p| input.path == **p),
                "G0 original input cannot alias a generated output namespace",
            )?;
        }
        Ok(())
    }
}

/// Actual G0 raw comparison only. Source pins, original children,
/// initializer/export files and final raw descriptors stay alive with this
/// non-deserializable owner. This is not `PalsStartupContentAdmissionV5` and does
/// not authorize startup, collection, training or arena execution.
pub struct VerifiedG0NumericalV5 {
    config: PalsAdaptiveStartupConfigV5,
    lock: PalsInputLockV5,
    admission: PalsNumericalAdmissionV5,
    acceptance_pin: Pin,
    parent_receipt: Value,
    processes: Vec<OriginalProcessOutput>,
    // Retain the exact first descriptor opened before declaration metadata/hash,
    // in addition to the subsequently revalidated input roster descriptor.
    #[cfg(target_os = "linux")]
    _registration: File,
    #[cfg(target_os = "linux")]
    held: super::training_process::LinuxHeld,
}
impl VerifiedG0NumericalV5 {
    pub fn startup_config(&self) -> &PalsAdaptiveStartupConfigV5 {
        &self.config
    }
    pub fn lock(&self) -> &PalsInputLockV5 {
        &self.lock
    }
    pub fn admission(&self) -> &PalsNumericalAdmissionV5 {
        &self.admission
    }
    pub fn acceptance_pin(&self) -> &Pin {
        &self.acceptance_pin
    }
    pub fn parent_receipt(&self) -> &Value {
        &self.parent_receipt
    }
    pub fn processes(&self) -> &[OriginalProcessOutput] {
        &self.processes
    }
    pub fn held_files(&self) -> impl Iterator<Item = (&Pin, &File)> {
        #[cfg(target_os = "linux")]
        {
            self.held
                .checkpoints
                .iter()
                .chain(self.held.inputs.iter())
                .map(|f| (&f.pin, &f.file))
        }
        #[cfg(not(target_os = "linux"))]
        {
            std::iter::empty()
        }
    }
}
pub struct G0NumericalFailureV5 {
    pub error: ArenaError,
    pub processes: Vec<OriginalProcessOutput>,
    pub child_receipts: Vec<Value>,
    pub preservation_reports: Vec<Value>,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    pub registration_pin: Option<Pin>,
    // The original registration survives failures before staging preparation.
    _registration: Option<File>,
    #[cfg(target_os = "linux")]
    _held: Option<super::training_process::LinuxHeld>,
}
impl std::fmt::Debug for G0NumericalFailureV5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G0NumericalFailureV5")
            .field("error", &self.error)
            .field("processes", &self.processes)
            .field("child_receipts", &self.child_receipts)
            .field("charged_read_bytes", &self.charged_read_bytes)
            .field("read_usage_uncertain", &self.read_usage_uncertain)
            .finish_non_exhaustive()
    }
}

/// Default callers do not initialize a model; only the explicit thin CLI calls
/// this original transport. No diagnostic/trainer owner can be supplied here.
pub fn run_g0_numerical_original_v5(
    path: &Path,
    sha256: &str,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedG0NumericalV5, Box<G0NumericalFailureV5>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, sha256, cancel);
        Err(Box::new(G0NumericalFailureV5 {
            error: invalid("G0 original numerical requires Linux"),
            processes: Vec::new(),
            child_receipts: Vec::new(),
            preservation_reports: Vec::new(),
            charged_read_bytes: 0,
            read_usage_uncertain: false,
            registration_pin: None,
            _registration: None,
        }))
    }
    #[cfg(target_os = "linux")]
    {
        linux::run(path, sha256, &cancel)
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::super::{numerical_process as numerical, training_process as common};
    use super::*;
    use crate::process::{
        OriginalNumericalEnvironment, OriginalProcessInput, OriginalProcessWindow,
        OwnedArtifactTreeWatch, ProcessLimits, supervise_numerical_input_in_original_window,
        supervise_verified_input_in_original_window,
    };
    use common::{LinuxHeld, ObservedFile, ReadAccounting, ReadCredit};
    use rz_experiments::{
        ArtifactRef, PalsArtifactDomainV5, PalsEngineModeV5, PalsModelBackendV3,
        PalsNumericalAcceptanceV5, PalsNumericalInvocationV5, PalsNumericalRunPairV5,
        PalsStartupAssetPurposeV5 as Purpose, PalsStartupAssetV5,
    };
    use std::{ffi::OsString, sync::atomic::Ordering};
    const REGISTRATION: &str = "rz-pals-g0-numerical-registration/1";
    const STARTUP: &str = "rz-pals-original-g0-numerical-startup/1";
    const RESULT: &str = "rz-pals-g0-numerical-cli-window/1";
    const PACKAGE: &str = "experiments/model-research/pals/src";
    const MODULE: &str = "rz_pals_model.g0_numerical_cli";
    const BOOTSTRAP: &str = "import sys,runpy; package=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,package); sys.argv[0]=module; runpy.run_module(module,run_name='__main__',alter_sys=True)";
    const COMPONENTS: [&str; 5] = [
        "model_native",
        "model_encoding",
        "nn_runtime",
        "model_reference",
        "model_export",
    ];
    const INIT_SOURCES: [&str; 8] = [
        "__init__.py",
        "config.py",
        "model.py",
        "artifacts.py",
        "controlled_initialization.py",
        "virtual_review.py",
        "onnx_shared.py",
        "initialization_evidence.py",
    ];

    struct Running {
        registration_pin: Option<Pin>,
        registration: Option<File>,
        held: Option<LinuxHeld>,
        processes: Vec<OriginalProcessOutput>,
        receipts: Vec<Value>,
        preservation: Vec<Value>,
        account: Arc<ReadAccounting>,
    }
    pub(super) fn run(
        path: &Path,
        sha256: &str,
        cancel: &AtomicBool,
    ) -> Result<VerifiedG0NumericalV5, Box<G0NumericalFailureV5>> {
        let started = Instant::now();
        let mut running = Running {
            registration_pin: None,
            registration: None,
            held: None,
            processes: Vec::new(),
            receipts: Vec::new(),
            preservation: Vec::new(),
            account: ReadAccounting::new(),
        };
        let result = execute(path, sha256, started, cancel, &mut running);
        match result {
            Ok((config, lock, admission, acceptance_pin, parent_receipt)) => {
                Ok(VerifiedG0NumericalV5 {
                    config,
                    lock,
                    admission,
                    acceptance_pin,
                    parent_receipt,
                    processes: running.processes,
                    _registration: running
                        .registration
                        .expect("successful G0 retains first original declaration descriptor"),
                    held: running
                        .held
                        .expect("successful G0 retains original custody"),
                })
            }
            Err(error) => Err(Box::new(G0NumericalFailureV5 {
                error,
                processes: running.processes,
                child_receipts: running.receipts,
                preservation_reports: running.preservation,
                charged_read_bytes: running.account.charged.load(Ordering::Acquire),
                read_usage_uncertain: running.account.uncertain.load(Ordering::Acquire),
                registration_pin: running.registration_pin,
                _registration: running.registration,
                _held: running.held,
            })),
        }
    }

    type Completed = (
        PalsAdaptiveStartupConfigV5,
        PalsInputLockV5,
        PalsNumericalAdmissionV5,
        Pin,
        Value,
    );
    fn execute(
        path: &Path,
        sha256: &str,
        started: Instant,
        cancel: &AtomicBool,
        running: &mut Running,
    ) -> Result<Completed, ArenaError> {
        // Even declaration reads consume the original clock. This initial
        // ceiling is never restarted when its actual finite limit is decoded.
        let mut credit = ReadCredit {
            remaining: 4 * MAX_STAGING,
            maximum: 4 * MAX_STAGING,
            until: started + Duration::from_secs(60),
            cancel,
            account: Arc::clone(&running.account),
        };
        let file = common::open_file(path)?;
        running.registration = Some(file);
        let file = running
            .registration
            .as_ref()
            .expect("original declaration FD");
        let pin = Pin {
            path: path.into(),
            sha256: sha256.into(),
            bytes: file
                .metadata()
                .map_err(|_| invalid("G0 declaration metadata"))?
                .len(),
        };
        running.registration_pin = Some(pin.clone());
        pin.validate(MAX_JSON)?;
        let registration: G0NumericalLaunchRegistrationV5 =
            common::decode(&common::verify_bytes(file, &pin, true, &mut credit)?)?;
        registration.validate()?;
        let until = started
            .checked_add(Duration::from_nanos(registration.max_wall_ns))
            .ok_or_else(|| invalid("G0 deadline overflow"))?;
        let cleanup_until = until + Duration::from_secs(30);
        let charged = running.account.charged.load(Ordering::Acquire);
        credit.remaining = registration
            .max_read_bytes
            .checked_sub(charged)
            .ok_or_else(|| invalid("G0 declaration consumed original read credit"))?;
        credit.maximum = registration.max_read_bytes;
        credit.until = until;
        credit.take(0)?;
        running.held = Some(common::prepare_original_root(
            &registration.staging_root,
            registration
                .staging_root
                .join(format!("g0-source-{}", registration.numerical_id)),
            &registration.prepare_receipt_path,
            &[
                &registration.initialization_directory,
                &registration.export_directory,
                &registration.reference_output_root,
                &registration.native_output_root,
            ],
        )?);
        let held = running.held.as_mut().expect("original G0 staging");
        for directory in [
            &registration.initialization_directory,
            &registration.export_directory,
            &registration.reference_output_root,
            &registration.native_output_root,
        ] {
            check(
                common::open_directory(directory)?
                    .entries()
                    .map_err(|_| invalid("G0 output directory entries"))?
                    .next()
                    .is_none(),
                "G0 requires fresh empty output directories; no checkpoint relabelling",
            )?;
        }
        for output in [
            &registration.reference_receipt_path,
            &registration.acceptance_path,
            &registration.parent_receipt_path,
        ] {
            check(!output.exists(), "G0 refuses an existing terminal output")?;
        }
        common::hold_input(pin.clone(), false, held, &mut credit)?;
        let template_raw = common::hold_input(
            registration.template_startup.clone(),
            true,
            held,
            &mut credit,
        )?;
        let template: G0NumericalStartupTemplateV5 = common::decode(&template_raw)?;
        validate_template(&template, &registration)?;
        common::retain_directory(template.private_cwd.clone(), held)?;
        for input in &template.input_pins {
            common::hold_input(input.source.clone(), false, held, &mut credit)?;
        }
        for input in [
            &registration.case_plan,
            &registration.rules_profile,
            &registration.native_capture_config,
            &registration.native_executable,
            &registration.python_runtime.executable,
        ] {
            common::hold_input(input.clone(), false, held, &mut credit)?;
        }
        let dependencies = registration
            .dependency_directories
            .iter()
            .map(|p| common::retain_directory(p.clone(), held))
            .collect::<Result<Vec<_>, _>>()?;
        for input in &registration.runtime_module_pins {
            common::hold_input(input.clone(), false, held, &mut credit)?;
        }
        numerical::ensure_output_credit(
            &held.staging,
            registration.max_output_bytes,
            36 << 20,
            until,
            cancel,
        )?;
        common::install_sources_for_components(held, &mut credit, &COMPONENTS)?;
        check(
            held.sources
                .iter()
                .any(|f| f.pin.path.ends_with("rz_pals_model/g0_numerical_cli.py")),
            "G0 compiled child source absent",
        )?;
        let sources = COMPONENTS
            .into_iter()
            .map(|name| {
                template
                    .config
                    .source_components
                    .get(name)
                    .cloned()
                    .map(|pin| (name.to_owned(), pin))
                    .ok_or_else(|| invalid("G0 source component absent"))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        check(
            sources.iter().all(|(name, sha)| {
                rz_experiments::compiled_source_components_v5()
                    .ok()
                    .and_then(|c| c.get(name).cloned())
                    .as_ref()
                    == Some(sha)
            }),
            "G0 compiled source component differs",
        )?;
        let mut initializer_pins = BTreeMap::new();
        for name in INIT_SOURCES {
            let source = held
                .sources
                .iter()
                .find(|f| {
                    f.pin.path
                        == held
                            .source_root
                            .join(PACKAGE)
                            .join("rz_pals_model")
                            .join(name)
                })
                .ok_or_else(|| invalid("G0 exact initializer compiled source absent"))?;
            initializer_pins.insert(name, source.pin.sha256.clone());
        }
        let initialization_sources = serde_json::json!({"schema":"rz-pals-g0-initialization-sources/1","source_commit":registration.source_commit,"pins":initializer_pins});
        let prepare_inputs = serde_json::json!({"rules_profile":registration.rules_profile,"case_plan_template":registration.case_plan});
        let prepare_receipt = child(
            &registration,
            "prepare",
            prepare_inputs,
            &sources,
            &initialization_sources,
            &Value::Array(dependencies.clone()),
            started,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )?;
        let mut config = template.config.clone();
        let outputs = prepare_receipt
            .get("outputs")
            .ok_or_else(|| invalid("G0 prepare output roster absent"))?;
        rebind_outputs(
            &mut config,
            &template.private_cwd,
            outputs,
            &registration,
            running.held.as_mut().expect("G0 custody"),
            &mut credit,
        )?;
        let mut manifest = template.lock.manifest.clone();
        let endpoint = manifest
            .engines
            .iter_mut()
            .find(|e| **e == template.config.endpoint)
            .ok_or_else(|| invalid("G0 original endpoint absent"))?;
        *endpoint = config.endpoint.clone();
        let lock = manifest.lock()?;
        config.semantic_lock_sha256 = lock.canonical_sha256.clone();
        config.validate_against(&lock)?;
        let expected = numerical::bindings(&config)?;
        let held = running.held.as_mut().expect("G0 custody");
        let original_plan =
            common::hold_input(registration.case_plan.clone(), true, held, &mut credit)?;
        let mut plan: Value = common::decode(&original_plan)?;
        numerical::validate_plan(&plan, &expected)?;
        let groups = plan["groups"].clone();
        plan["bindings"] = serde_json::to_value(&expected).map_err(|e| invalid(e.to_string()))?;
        check(
            plan["groups"] == groups,
            "G0 must retain preregistered numerical cases",
        )?;
        let plan_raw = serde_json::to_vec(&plan).map_err(|e| invalid(e.to_string()))?;
        numerical::ensure_output_credit(
            &held.staging,
            registration.max_output_bytes,
            plan_raw.len() as u64,
            until,
            cancel,
        )?;
        let actual_plan = common::write_owned_new(
            &registration
                .staging_root
                .join(format!("g0-plan-{}.json", registration.numerical_id)),
            &plan_raw,
        )?;
        let actual_plan_pin = actual_plan.pin.clone();
        held.checkpoints.push(actual_plan);
        let reference_inputs = serde_json::json!({"case_plan":actual_plan_pin,"checkpoint":pin_value(&outputs["checkpoint"])?,
            "checkpoint_metadata":pin_value(&outputs["checkpoint_metadata"])?,"initialization_evidence":pin_value(&outputs["initialization_evidence"])?,
            "export_manifest":pin_value(&outputs["export_manifest"])?,"export_graphs":graph_pin_values(&outputs["export_graphs"])?,
            "cuda_control_inventory":pin_value(&outputs["cuda_control_inventory"])?});
        let reference_receipt = child(
            &registration,
            "reference",
            reference_inputs,
            &sources,
            &initialization_sources,
            &Value::Array(dependencies),
            started,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )?;
        let references = hold_reference_outputs(
            &registration,
            &reference_receipt,
            &plan,
            running.held.as_mut().expect("G0 custody"),
            &mut credit,
        )?;
        run_native(
            &registration,
            &config,
            &template.private_cwd,
            &actual_plan_pin,
            &plan,
            started,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )?;
        finalize(
            &registration,
            config,
            lock,
            &template.private_cwd,
            &plan,
            &references,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )
    }

    fn pin_value(value: &Value) -> Result<Value, ArenaError> {
        let observed: ObservedFile =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        serde_json::to_value(observed.pin()).map_err(|e| invalid(e.to_string()))
    }
    fn graph_pin_values(value: &Value) -> Result<Value, ArenaError> {
        let graphs = value
            .as_array()
            .ok_or_else(|| invalid("G0 exact graph outputs absent"))?;
        graphs.iter().map(|g| { let mut pin = pin_value(&serde_json::json!({"path":g["path"],"sha256":g["sha256"],"bytes":g["bytes"],"device":g["device"],"inode":g["inode"]}))?;
            pin["role"] = g["role"].clone(); pin["file"] = g["file"].clone(); Ok(pin) }).collect::<Result<Vec<_>, _>>().map(Value::Array)
    }

    fn validate_template(
        t: &G0NumericalStartupTemplateV5,
        r: &G0NumericalLaunchRegistrationV5,
    ) -> Result<(), ArenaError> {
        t.config.validate_against(&t.lock)?;
        check(
            t.schema == G0_NUMERICAL_TEMPLATE_V5
                && common::absolute_safe(&t.private_cwd)
                && t.private_cwd.starts_with(&r.staging_root)
                && t.private_cwd != r.staging_root
                && [&r.initialization_directory, &r.export_directory]
                    .into_iter()
                    .all(|p| p.starts_with(&t.private_cwd) && *p != t.private_cwd)
                && r.acceptance_path.parent() == Some(t.private_cwd.as_path())
                && t.config.endpoint.mode == PalsEngineModeV5::Collection
                && t.config.endpoint.model.artifact_domain == PalsArtifactDomainV5::Untrained
                && t.config.endpoint.model.profile == "virtual_review_v1"
                && t.config.endpoint.model.provenance.is_none()
                && t.config.endpoint.model.checkpoint_domain.is_none()
                && t.config.endpoint.source_commit == r.source_commit
                && !t.lock.manifest.training_authorized
                && !t.config.native_options.approximate_warm
                && !t.config.native_options.device_public_memory
                && t.config.native_options.cudnn_conv_search == r.cudnn_conv_search,
            "G0 template retains explicit untrained Collection/Fresh/runtime/source identity",
        )?;
        let retained = |p: Purpose| {
            !matches!(
                p,
                Purpose::Weights
                    | Purpose::Provenance
                    | Purpose::ExportManifest
                    | Purpose::PublicGraph
                    | Purpose::SharedPcGraph
                    | Purpose::ProposerGraph
                    | Purpose::CriticGraph
                    | Purpose::ValidatorGraph
                    | Purpose::InitializationEvidence
                    | Purpose::InitializationCheckpointMetadata
                    | Purpose::CudaControlInventory
                    | Purpose::NumericalAcceptance
                    | Purpose::NumericalRawEvidence
            )
        };
        let required = t
            .config
            .assets
            .iter()
            .filter(|a| retained(a.purpose))
            .collect::<Vec<_>>();
        check(
            t.input_pins.len() == required.len()
                && t.input_pins.len() <= 96
                && t.input_pins
                    .iter()
                    .map(|p| &p.snapshot_path)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == t.input_pins.len(),
            "G0 template exact unchanged input roster",
        )?;
        for input in &t.input_pins {
            input.source.validate(MAX_STAGING)?;
            check(
                required.iter().any(|a| {
                    a.snapshot_path == input.snapshot_path
                        && a.artifact.sha256 == input.source.sha256
                        && a.artifact.bytes == input.source.bytes
                }) && input.source.path == t.private_cwd.join(&input.snapshot_path),
                "G0 unchanged original snapshot differs",
            )?;
        }
        let rules = t.config.asset(Purpose::RulesDescriptor)?;
        check(
            t.input_pins
                .iter()
                .any(|p| p.snapshot_path == rules.snapshot_path && p.source == r.rules_profile),
            "G0 export Rules descriptor must be the unchanged original snapshot",
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "One original execution owns all phase clocks and file/process custody"
    )]
    fn child(
        r: &G0NumericalLaunchRegistrationV5,
        phase: &str,
        inputs: Value,
        sources: &BTreeMap<String, String>,
        initialization_sources: &Value,
        dependencies: &Value,
        started: Instant,
        until: Instant,
        cleanup_until: Instant,
        cancel: &AtomicBool,
        running: &mut Running,
        credit: &mut ReadCredit<'_>,
    ) -> Result<Value, ArenaError> {
        check(
            matches!(phase, "prepare" | "reference"),
            "G0 closed child phase",
        )?;
        credit.until = until;
        credit.take(0)?;
        let receipt_path = if phase == "prepare" {
            &r.prepare_receipt_path
        } else {
            &r.reference_receipt_path
        };
        let registration = serde_json::json!({"schema":REGISTRATION,"numerical_id":r.numerical_id,"phase":phase,
            "source_components":sources,"source_commit":r.source_commit,"initialization_sources":initialization_sources,
            "staging_root":r.staging_root,"initialization_directory":r.initialization_directory,"export_directory":r.export_directory,
            "reference_output_root":r.reference_output_root,"receipt_path":receipt_path,
            "max_wall_ns":r.max_wall_ns,"max_output_bytes":r.max_output_bytes,"inputs":inputs});
        let sha = rz_experiments::pals_data_v3::generation_digest(REGISTRATION, &registration)
            .map_err(|e| invalid(e.to_string()))?;
        let (start_ns, execution_ns, whole_ns) =
            common::original_window_ticks(started, until, cleanup_until)?;
        let startup = serde_json::json!({"schema":STARTUP,"execution_id":r.numerical_id,"phase":phase,
            "registration_sha256":sha,"registration":registration,"original_started_ns":start_ns,"execution_until_ns":execution_ns,
            "whole_until_ns":whole_ns,"dependency_directories":dependencies,"python_runtime":r.python_runtime,"runtime_module_pins":r.runtime_module_pins});
        let stdin = serde_json::to_vec(&startup).map_err(|e| invalid(e.to_string()))?;
        check(
            stdin.len() <= crate::process::MAX_ORIGINAL_INPUT_BYTES,
            "G0 bounded original startup",
        )?;
        let held = running.held.as_mut().expect("G0 original phase custody");
        numerical::ensure_output_credit(
            &held.staging,
            r.max_output_bytes,
            stdin.len() as u64,
            until,
            cancel,
        )?;
        let startup_file = common::write_owned_new(
            &r.staging_root
                .join(format!("g0-{phase}-startup-{}.json", r.numerical_id)),
            &stdin,
        )?;
        let startup_pin = startup_file.pin.clone();
        held.checkpoints.push(startup_file);
        let args = [
            OsString::from("-I"),
            OsString::from("-S"),
            OsString::from("-B"),
            OsString::from("-c"),
            OsString::from(BOOTSTRAP),
            held.source_root.join(PACKAGE).into_os_string(),
            OsString::from(MODULE),
            OsString::from("--execute"),
            OsString::from("--original-g0-child"),
            OsString::from("--startup-sha256"),
            OsString::from(&startup_pin.sha256),
            OsString::from("--startup-bytes"),
            OsString::from(stdin.len().to_string()),
            OsString::from("--execution-until-ns"),
            OsString::from(execution_ns.to_string()),
            OsString::from("--whole-until-ns"),
            OsString::from(whole_ns.to_string()),
        ];
        common::revalidate(held, credit)?;
        let program = common::open_file(&r.python_runtime.executable.path)?;
        let cwd = common::directory_file(&held.staging)?;
        let mut mapped =
            numerical::reserve_image(&r.python_runtime.executable, credit, until, cancel)?;
        let mut image = |file: &File| {
            common::verify_bytes(file, &r.python_runtime.executable, false, &mut mapped)
                .map(|_| ())
                .map_err(|_| ())
        };
        running
            .processes
            .push(supervise_verified_input_in_original_window(
                &program,
                &args,
                &cwd,
                limits(started, until)?,
                Some(cancel),
                &watch(r),
                OriginalProcessInput {
                    bytes: &stdin,
                    window: OriginalProcessWindow::new(started, until, cleanup_until)?,
                },
                &mut image,
            )?);
        // Pin every registered output before trusting even the stdout status.
        // Hash/metadata failures leave the original failure-only FD quarantined.
        let preserve_cancel = AtomicBool::new(false);
        let mut cleanup_credit = ReadCredit {
            remaining: credit.remaining,
            maximum: credit.maximum,
            until: cleanup_until,
            cancel: &preserve_cancel,
            account: Arc::clone(&credit.account),
        };
        let result = (|| -> Result<Value, ArenaError> {
            let credit = &mut cleanup_credit;
            let original = running.processes.last().expect("actual original G0 child");
            let mut targets = vec![
                (receipt_path.clone(), MAX_JSON),
                (receipt_path.with_extension("json.partial"), MAX_JSON),
            ];
            if phase == "prepare" {
                for (directory, names) in [
                    (
                        &r.initialization_directory,
                        &[
                            "untrained.pt",
                            "untrained.pt.partial",
                            "checkpoint.json",
                            "checkpoint.json.partial",
                            "g0-initialization-evidence.json",
                            "g0-initialization-evidence.json.partial",
                            "pals-artifact-owner.json",
                        ][..],
                    ),
                    (
                        &r.export_directory,
                        &[
                            "public_memory.onnx",
                            "public_memory.onnx.partial",
                            "shared_pc_if.onnx",
                            "shared_pc_if.onnx.partial",
                            "role_validator.onnx",
                            "role_validator.onnx.partial",
                            "export.json",
                            "export.json.partial",
                            "cuda-control-inventory.json",
                            "cuda-control-inventory.json.partial",
                            "pals-artifact-owner.json",
                        ][..],
                    ),
                ] {
                    targets.extend(names.iter().map(|name| (directory.join(name), MAX_FILE)));
                }
            } else {
                let plan_pin: Pin = serde_json::from_value(inputs["case_plan"].clone())
                    .map_err(|e| invalid(e.to_string()))?;
                let plan_file = common::open_file(&plan_pin.path)?;
                let plan: Value =
                    common::decode(&common::verify_bytes(&plan_file, &plan_pin, true, credit)?)?;
                for group in plan["groups"]
                    .as_array()
                    .ok_or_else(|| invalid("G0 registered reference groups"))?
                {
                    let id = group["id"]
                        .as_str()
                        .ok_or_else(|| invalid("G0 registered group ID"))?;
                    targets.push((
                        r.reference_output_root.join(format!("reference.{id}.json")),
                        MAX_JSON,
                    ));
                    targets.push((
                        r.reference_output_root
                            .join(format!("reference.{id}.json.partial")),
                        MAX_JSON,
                    ));
                }
            }
            let checkpoint_temporary =
                (phase == "prepare").then(|| r.initialization_directory.join("untrained.pt"));
            let preservation = common::preserve_original_outputs(
                &targets,
                checkpoint_temporary.as_deref(),
                receipt_path,
                running.held.as_mut().expect("G0 original output custody"),
                credit,
                original,
            );
            running.preservation.push(preservation.report);
            let summary: Value = common::decode(&original.process().stdout)?;
            let receipt_pin: Pin = serde_json::from_value(summary["receipt"].clone())
                .map_err(|e| invalid(e.to_string()))?;
            check(
                receipt_pin.path == *receipt_path,
                "G0 exact original terminal receipt path",
            )?;
            let (receipt_file, raw) =
                common::read_observed_output(receipt_pin, MAX_JSON, credit, true)?;
            running
                .held
                .as_mut()
                .expect("G0 receipt custody")
                .checkpoints
                .push(receipt_file);
            let receipt: Value = common::decode(&raw)?;
            running.receipts.push(receipt.clone());
            verify_child(
                r,
                phase,
                &summary,
                &receipt,
                &registration,
                &sha,
                dependencies,
                execution_ns,
                whole_ns,
                original,
                running.held.as_ref().expect("G0 original source custody"),
            )?;
            if let Some(error) = preservation.error {
                return Err(error);
            }
            Ok(receipt)
        })();
        credit.remaining = cleanup_credit.remaining;
        result
    }

    fn limits(started: Instant, until: Instant) -> Result<ProcessLimits, ArenaError> {
        Ok(ProcessLimits {
            wall_ms: u64::try_from(until.duration_since(started).as_nanos().div_ceil(1_000_000))
                .map_err(|_| invalid("G0 wall milliseconds overflow"))?
                .max(1),
            shutdown_grace_ms: 15_000,
            max_output_bytes: 128 << 10,
            max_child_processes: 1,
        })
    }
    fn watch(r: &G0NumericalLaunchRegistrationV5) -> OwnedArtifactTreeWatch {
        OwnedArtifactTreeWatch {
            max_total_bytes: r.max_output_bytes,
            max_file_bytes: MAX_FILE,
            max_files: 65_536,
            max_depth: 32,
        }
    }
    fn observed_matches(value: &Value, held: &LinuxHeld) -> Result<(), ArenaError> {
        let observed: ObservedFile =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        check(
            held.inputs
                .iter()
                .chain(held.checkpoints.iter())
                .chain(held.sources.iter())
                .any(|f| common::readback_matches(&observed, f)),
            "G0 original readback has no matching held descriptor",
        )
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "Original child verification binds exact envelope, source and live process evidence"
    )]
    fn verify_child(
        r: &G0NumericalLaunchRegistrationV5,
        phase: &str,
        summary: &Value,
        receipt: &Value,
        registration: &Value,
        sha: &str,
        dependencies: &Value,
        execution_ns: u64,
        whole_ns: u64,
        process: &OriginalProcessOutput,
        held: &LinuxHeld,
    ) -> Result<(), ArenaError> {
        process.checked_loaded_image()?;
        process.checked_timing()?;
        let status = if phase == "prepare" {
            "prepared"
        } else {
            "captured"
        };
        check(
            process.transport_complete()
                && summary["schema"] == RESULT
                && summary["status"] == status
                && receipt["schema"] == RESULT
                && receipt["status"] == status
                && receipt["phase"] == phase
                && receipt["execution_id"] == r.numerical_id
                && receipt["registration_sha256"] == sha
                && receipt["original_execution_until_ns"] == execution_ns
                && receipt["original_whole_until_ns"] == whole_ns
                && summary["before_original_execution_deadline"] == true
                && receipt["before_original_execution_deadline"] == true
                && receipt["optimizer_updates_executed"] == 0
                && receipt["native_numeric_authority_issued"] == false
                && receipt["parent_must_verify_process_and_artifacts"] == true
                && receipt["arena_accepted"] == false,
            "G0 original phase closure/deadline/non-training scope differs",
        )?;
        check(
            receipt["active_source"]["source_components"] == registration["source_components"],
            "G0 actual source roster differs",
        )?;
        let mut names = BTreeSet::new();
        let modules = receipt["active_source"]["source_modules"]
            .as_array()
            .ok_or_else(|| invalid("G0 actual imported modules absent"))?;
        check(
            !modules.is_empty() && modules.len() <= 128,
            "G0 imported source bounded inventory",
        )?;
        for module in modules {
            let name = module["module"]
                .as_str()
                .ok_or_else(|| invalid("G0 imported source namespace"))?;
            check(
                names.insert(name)
                    && (name == "__main__"
                        || name == "rz_pals_model"
                        || name.starts_with("rz_pals_model.")),
                "G0 source namespace/duplicate",
            )?;
            let pin = serde_json::json!({"path":module["path"],"sha256":module["sha256"],"bytes":module["bytes"],"device":module["device"],"inode":module["inode"]});
            let observed: ObservedFile =
                serde_json::from_value(pin.clone()).map_err(|e| invalid(e.to_string()))?;
            check(
                held.sources
                    .iter()
                    .any(|f| common::readback_matches(&observed, f)),
                "G0 imported module is not original compiled source",
            )?;
            if name == "__main__" {
                check(
                    observed.path
                        == held
                            .source_root
                            .join(PACKAGE)
                            .join("rz_pals_model/g0_numerical_cli.py"),
                    "G0 fixed child module differs",
                )?;
            }
        }
        check(
            [
                "__main__",
                "rz_pals_model.model",
                "rz_pals_model.virtual_review",
            ]
            .into_iter()
            .all(|name| names.contains(name)),
            "G0 required loaded source absent",
        )?;
        check(
            names.contains(if phase == "prepare" {
                "rz_pals_model.initialization_evidence"
            } else {
                "rz_pals_model.numeric_capture"
            }),
            "G0 phase producer source absent",
        )?;
        let python = &receipt["interpreter_readback"];
        observed_matches(&python["executable"], held)?;
        let py: ObservedFile = serde_json::from_value(python["executable"].clone())
            .map_err(|e| invalid(e.to_string()))?;
        check(
            py.pin() == r.python_runtime.executable
                && python["version"].as_str().is_some_and(|s| {
                    s.split_whitespace().next() == Some(r.python_runtime.version.as_str())
                })
                && python["mapped_executable_identity"]["device"] == py.device
                && python["mapped_executable_identity"]["inode"] == py.inode
                && python["os"] == "posix"
                && python["platform"] == "linux"
                && python["isolated"] == true
                && python["no_site"] == true
                && python["bytecode_disabled"] == true
                && python["dependency_pth_executed"] == false
                && python["dependency_directories"] == *dependencies
                && python["cwd"] == registration["staging_root"],
            "G0 actual isolated interpreter differs",
        )?;
        for stage in [
            "runtime_module_readbacks_before",
            "runtime_module_readbacks_after",
        ] {
            let rows = receipt[stage]
                .as_array()
                .ok_or_else(|| invalid("G0 runtime readback absent"))?;
            let mut paths = BTreeSet::new();
            check(
                rows.len() == r.runtime_module_pins.len(),
                "G0 trusted runtime roster differs",
            )?;
            for value in rows {
                observed_matches(value, held)?;
                let observed: ObservedFile =
                    serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
                check(
                    paths.insert(observed.path.clone())
                        && r.runtime_module_pins.contains(&observed.pin()),
                    "G0 trusted runtime file identity differs",
                )?;
            }
        }
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
                    .any(|p| runtime["torch_module_path"] == p.path.to_string_lossy().as_ref()),
            "G0 actual CPU reference runtime differs",
        )?;
        let reads = receipt["original_input_readbacks"]
            .as_object()
            .ok_or_else(|| invalid("G0 original input readbacks absent"))?;
        let inputs = registration["inputs"]
            .as_object()
            .ok_or_else(|| invalid("G0 original inputs absent"))?;
        check(
            reads.len() == inputs.len(),
            "G0 original input roster differs",
        )?;
        for (name, pin) in inputs {
            let row = reads
                .get(name)
                .ok_or_else(|| invalid("G0 phase input missing"))?;
            if name == "export_graphs" {
                let rows = row
                    .as_array()
                    .ok_or_else(|| invalid("G0 graph input readbacks"))?;
                let pins = pin
                    .as_array()
                    .ok_or_else(|| invalid("G0 registered graph input list"))?;
                check(rows.len() == pins.len(), "G0 original graph input count")?;
                for (observed, registered) in rows.iter().zip(pins) {
                    observed_matches(observed, held)?;
                    let actual: ObservedFile = serde_json::from_value(observed.clone())
                        .map_err(|e| invalid(e.to_string()))?;
                    check(
                        actual.path.as_path()
                            == Path::new(
                                registered["path"]
                                    .as_str()
                                    .ok_or_else(|| invalid("G0 graph pin path"))?,
                            )
                            && registered["sha256"] == actual.sha256
                            && registered["bytes"] == actual.bytes,
                        "G0 original graph input changed",
                    )?;
                }
            } else {
                observed_matches(row, held)?;
                check(pin_value(row)? == *pin, "G0 original input pin changed")?;
            }
        }
        if phase == "reference" {
            check(
                receipt["original_model_bits_preserved"] == true
                    && receipt["actual_reference_executed"] == true
                    && receipt["physical_completion"] == "reference_completed_cpu"
                    && receipt["reference_dispatch_started"] == true
                    && receipt["reference_neural_forwards_executed"]
                        .as_u64()
                        .is_some_and(|n| (1..=256).contains(&n))
                    && receipt["capture_result"]["cases"]
                        == receipt["reference_neural_forwards_executed"],
                "G0 original CPU reference capture absent",
            )?;
        } else {
            check(
                receipt["reference_neural_forwards_executed"] == 0
                    && receipt["actual_reference_executed"] == false
                    && receipt["actual_initialization_executed"] == true
                    && receipt["actual_export_executed"] == true
                    && summary["actual_initialization_executed"] == true
                    && summary["actual_export_executed"] == true,
                "G0 prepare cannot substitute for reference or optimizer execution",
            )?;
        }
        Ok(())
    }

    fn hold_observed(
        value: &Value,
        expected_path: &Path,
        cap: u64,
        held: &mut LinuxHeld,
        credit: &mut ReadCredit<'_>,
    ) -> Result<Pin, ArenaError> {
        let observed: ObservedFile =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        check(
            observed.path == expected_path,
            "G0 output path differs from the original fixed target",
        )?;
        let (file, _) = common::read_observed_output(observed.pin(), cap, credit, false)?;
        check(
            common::readback_matches(&observed, &file),
            "G0 original output inode changed",
        )?;
        let pin = file.pin.clone();
        held.checkpoints.push(file);
        Ok(pin)
    }
    fn startup_asset(
        pin: &Pin,
        cwd: &Path,
        purpose: Purpose,
        prototype: &ArtifactRef,
    ) -> Result<PalsStartupAssetV5, ArenaError> {
        let relative = pin
            .path
            .strip_prefix(cwd)
            .map_err(|_| invalid("G0 actual asset outside original private cwd"))?
            .to_str()
            .ok_or_else(|| invalid("G0 asset UTF-8 path"))?
            .replace('\\', "/");
        check(
            rz_experiments::pals_private_snapshot_path_v5(&relative),
            "G0 safe actual snapshot path",
        )?;
        Ok(PalsStartupAssetV5 {
            artifact: ArtifactRef {
                path: format!("g0/{relative}"),
                sha256: pin.sha256.clone(),
                bytes: pin.bytes,
                source: prototype.source.clone(),
                license: prototype.license.clone(),
            },
            snapshot_path: relative,
            purpose,
        })
    }
    fn rebind_outputs(
        config: &mut PalsAdaptiveStartupConfigV5,
        cwd: &Path,
        outputs: &Value,
        r: &G0NumericalLaunchRegistrationV5,
        held: &mut LinuxHeld,
        credit: &mut ReadCredit<'_>,
    ) -> Result<(), ArenaError> {
        check(
            outputs.as_object().is_some_and(|o| o.len() == 6),
            "G0 exact prepare output structure",
        )?;
        let prototype = config.endpoint.model.weights.clone();
        let mut replaced = Vec::new();
        for (name, path, purpose, cap) in [
            (
                "checkpoint",
                r.initialization_directory.join("untrained.pt"),
                Purpose::Weights,
                MAX_FILE,
            ),
            (
                "checkpoint_metadata",
                r.initialization_directory.join("checkpoint.json"),
                Purpose::InitializationCheckpointMetadata,
                64 << 10,
            ),
            (
                "initialization_evidence",
                r.initialization_directory
                    .join("g0-initialization-evidence.json"),
                Purpose::InitializationEvidence,
                MAX_JSON,
            ),
            (
                "export_manifest",
                r.export_directory.join("export.json"),
                Purpose::ExportManifest,
                256 << 10,
            ),
        ] {
            let pin = hold_observed(&outputs[name], &path, cap, held, credit)?;
            replaced.push(startup_asset(&pin, cwd, purpose, &prototype)?);
        }
        let graphs = outputs["export_graphs"]
            .as_array()
            .ok_or_else(|| invalid("G0 export graph roster absent"))?;
        check(
            graphs.len() == 3,
            "G0 Collection needs exact public/shared_pc/validator graphs",
        )?;
        for ((role, file, purpose), graph) in [
            ("public", "public_memory.onnx", Purpose::PublicGraph),
            ("shared_pc", "shared_pc_if.onnx", Purpose::SharedPcGraph),
            ("validator", "role_validator.onnx", Purpose::ValidatorGraph),
        ]
        .into_iter()
        .zip(graphs)
        {
            check(
                graph["role"] == role && graph["file"] == file,
                "G0 graph order, role or basename differs",
            )?;
            let observed = serde_json::json!({"path":graph["path"],"sha256":graph["sha256"],"bytes":graph["bytes"],"device":graph["device"],"inode":graph["inode"]});
            let pin = hold_observed(
                &observed,
                &r.export_directory.join(file),
                MAX_FILE,
                held,
                credit,
            )?;
            replaced.push(startup_asset(&pin, cwd, purpose, &prototype)?);
        }
        let inventory = hold_observed(
            &outputs["cuda_control_inventory"],
            &r.export_directory.join("cuda-control-inventory.json"),
            MAX_JSON,
            held,
            credit,
        )?;
        if config.endpoint.model.backend == PalsModelBackendV3::OrtCuda {
            replaced.push(startup_asset(
                &inventory,
                cwd,
                Purpose::CudaControlInventory,
                &prototype,
            )?);
        }
        config.assets.retain(|a| {
            !matches!(
                a.purpose,
                Purpose::Weights
                    | Purpose::Provenance
                    | Purpose::ExportManifest
                    | Purpose::PublicGraph
                    | Purpose::SharedPcGraph
                    | Purpose::ProposerGraph
                    | Purpose::CriticGraph
                    | Purpose::ValidatorGraph
                    | Purpose::InitializationEvidence
                    | Purpose::InitializationCheckpointMetadata
                    | Purpose::CudaControlInventory
                    | Purpose::GenerationSpecification
            )
        });
        config.assets.extend(replaced);
        config.endpoint.model.weights = config.asset(Purpose::Weights)?.artifact.clone();
        check(
            config.endpoint.model.artifact_domain == PalsArtifactDomainV5::Untrained
                && config.endpoint.model.provenance.is_none()
                && config.endpoint.model.checkpoint_domain.is_none(),
            "G0 cannot acquire trained/diagnostic provenance",
        )?;
        let mut read = |artifact: &ArtifactRef, limit: usize| {
            read_asset(config, cwd, held, artifact, limit, credit)
        };
        let export = read(&config.asset(Purpose::ExportManifest)?.artifact, 256 << 10)?;
        crate::pals_adaptive_launch::validate_export_content_v5(config, &export)?;
        let rules = read(&config.asset(Purpose::RulesDescriptor)?.artifact, 4 << 20)?;
        crate::pals_adaptive_launch::validate_export_rules_content_v5(config, &export, &rules)?;
        let admission =
            rz_experiments::validate_pals_artifact_domain_v5(config, &export, &mut read)?;
        check(
            admission.initializer_source_commit.as_deref() == Some(r.source_commit.as_str()),
            "G0 initializer source commit differs",
        )?;
        for (name, sha) in admission.initializer_source_pins {
            check(
                held.sources.iter().any(|f| {
                    f.pin.path
                        == held
                            .source_root
                            .join(PACKAGE)
                            .join("rz_pals_model")
                            .join(name.as_str())
                        && f.pin.sha256 == sha
                }),
                "G0 initializer evidence source bytes do not join compiled originals",
            )?;
        }
        Ok(())
    }
    fn read_asset(
        config: &PalsAdaptiveStartupConfigV5,
        cwd: &Path,
        held: &LinuxHeld,
        artifact: &ArtifactRef,
        limit: usize,
        credit: &mut ReadCredit<'_>,
    ) -> Result<Vec<u8>, rz_experiments::ManifestError> {
        let fail = |s: &str| rz_experiments::ManifestError::Artifact(s.into());
        let asset = config
            .assets
            .iter()
            .find(|a| &a.artifact == artifact)
            .ok_or_else(|| fail("G0 read requested an undeclared asset"))?;
        let path = cwd.join(&asset.snapshot_path);
        let original = held
            .checkpoints
            .iter()
            .chain(held.inputs.iter())
            .find(|f| {
                f.pin.path == path
                    && f.pin.sha256 == artifact.sha256
                    && f.pin.bytes == artifact.bytes
            })
            .ok_or_else(|| fail("G0 original descriptor absent"))?;
        check(
            original.pin.bytes <= limit as u64,
            "G0 bounded artifact reader",
        )
        .map_err(|e| fail(&e.to_string()))?;
        common::verify_bytes(&original.file, &original.pin, true, credit)
            .map_err(|e| fail(&e.to_string()))
    }
    fn hold_reference_outputs(
        r: &G0NumericalLaunchRegistrationV5,
        receipt: &Value,
        plan: &Value,
        held: &mut LinuxHeld,
        credit: &mut ReadCredit<'_>,
    ) -> Result<BTreeMap<String, Pin>, ArenaError> {
        let rows = receipt["artifact_pins"]
            .as_array()
            .ok_or_else(|| invalid("G0 reference original raw roster absent"))?;
        let groups = plan["groups"]
            .as_array()
            .ok_or_else(|| invalid("G0 original groups absent"))?;
        check(
            rows.len() == groups.len(),
            "G0 reference exact group coverage",
        )?;
        let mut result = BTreeMap::new();
        for group in groups {
            let id = group["id"]
                .as_str()
                .ok_or_else(|| invalid("G0 original group ID absent"))?;
            let name = format!("reference.{id}.json");
            let path = r.reference_output_root.join(&name);
            let row = rows
                .iter()
                .find(|row| row["path"] == path.to_string_lossy().as_ref())
                .ok_or_else(|| invalid("G0 original raw output group absent"))?;
            let pin = hold_observed(row, &path, MAX_JSON, held, credit)?;
            check(
                result.insert(name, pin).is_none(),
                "G0 duplicate reference group",
            )?;
        }
        Ok(result)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Native physical lifetime stays under the original G0 execution deadline and credit owner"
    )]
    fn run_native(
        r: &G0NumericalLaunchRegistrationV5,
        config: &PalsAdaptiveStartupConfigV5,
        cwd_path: &Path,
        actual_plan: &Pin,
        plan: &Value,
        started: Instant,
        until: Instant,
        cleanup_until: Instant,
        cancel: &AtomicBool,
        running: &mut Running,
        credit: &mut ReadCredit<'_>,
    ) -> Result<(), ArenaError> {
        credit.until = until;
        credit.take(0)?;
        let held = running.held.as_mut().expect("G0 original native custody");
        let raw = common::hold_input(r.native_capture_config.clone(), true, held, credit)?;
        let mut capture: numerical::NativeCaptureConfig = common::decode(&raw)?;
        check(
            capture
                .cudnn_conv_search
                .as_deref()
                .unwrap_or("legacy_implicit_exhaustive")
                == r.cudnn_conv_search,
            "G0 capture cannot silently change registered cuDNN selection",
        )?;
        capture.case_plan = actual_plan.path.clone();
        capture.case_plan_sha256 = actual_plan.sha256.clone();
        capture.export_manifest =
            cwd_path.join(&config.asset(Purpose::ExportManifest)?.snapshot_path);
        capture.output_root = r.native_output_root.clone();
        capture.cudnn_conv_search = Some(r.cudnn_conv_search.clone());
        if config.endpoint.model.backend == PalsModelBackendV3::OrtCuda {
            let inventory = config.asset(Purpose::CudaControlInventory)?;
            capture.cuda_inventory = Some(cwd_path.join(&inventory.snapshot_path));
            capture.cuda_inventory_sha256 = Some(inventory.artifact.sha256.clone());
        }
        capture.deadline_seconds = until.saturating_duration_since(Instant::now()).as_secs();
        capture.cleanup_seconds = cleanup_until.duration_since(until).as_secs();
        check(
            capture.deadline_seconds > 0,
            "G0 original native deadline exhausted",
        )?;
        numerical::validate_capture_inputs(&capture, config, cwd_path, &r.staging_root)?;
        let raw = serde_json::to_vec(&capture).map_err(|e| invalid(e.to_string()))?;
        numerical::ensure_output_credit(
            &held.staging,
            r.max_output_bytes,
            raw.len() as u64,
            until,
            cancel,
        )?;
        let file = common::write_owned_new(
            &r.staging_root
                .join(format!("g0-native-config-{}.json", r.numerical_id)),
            &raw,
        )?;
        let file_pin = file.pin.clone();
        held.checkpoints.push(file);
        common::revalidate(held, credit)?;
        let native = common::open_file(&r.native_executable.path)?;
        let cwd = common::directory_file(&held.staging)?;
        let mut mapped = numerical::reserve_image(&r.native_executable, credit, until, cancel)?;
        let mut image = |file: &File| {
            common::verify_bytes(file, &r.native_executable, false, &mut mapped)
                .map(|_| ())
                .map_err(|_| ())
        };
        running
            .processes
            .push(supervise_numerical_input_in_original_window(
                &native,
                &[file_pin.path.into_os_string()],
                &cwd,
                limits(started, until)?,
                Some(cancel),
                &watch(r),
                OriginalProcessInput {
                    bytes: &raw,
                    window: OriginalProcessWindow::new(started, until, cleanup_until)?,
                },
                &mut image,
                OriginalNumericalEnvironment::Fp32Tf32Off,
            )?);
        let preserve_cancel = AtomicBool::new(false);
        let mut cleanup_credit = ReadCredit {
            remaining: credit.remaining,
            maximum: credit.maximum,
            until: cleanup_until,
            cancel: &preserve_cancel,
            account: Arc::clone(&credit.account),
        };
        let result = (|| -> Result<(), ArenaError> {
            let credit = &mut cleanup_credit;
            let mut targets = vec![(
                r.native_output_root
                    .join("actual-cudnn-execution-policy.json"),
                16 << 10,
            )];
            targets.push((
                r.native_output_root.join("actual-stage-counters.json"),
                MAX_JSON,
            ));
            targets.push((
                r.native_output_root
                    .join("actual-stage-counters.pending.json"),
                MAX_JSON + 64,
            ));
            for group in plan["groups"]
                .as_array()
                .ok_or_else(|| invalid("G0 native groups absent"))?
            {
                let id = group["id"]
                    .as_str()
                    .ok_or_else(|| invalid("G0 native group ID"))?;
                targets.push((
                    r.native_output_root.join(format!("native-{id}.json")),
                    MAX_JSON,
                ));
                targets.push((
                    r.native_output_root
                        .join(format!("native-{id}.pending.json")),
                    MAX_JSON + 64,
                ));
                targets.push((
                    r.native_output_root
                        .join(format!("native-{id}.json.partial")),
                    MAX_JSON + 64,
                ));
            }
            let original = running
                .processes
                .last()
                .expect("G0 original native process");
            let preserved = common::preserve_original_outputs(
                &targets,
                None,
                &r.parent_receipt_path,
                running.held.as_mut().expect("G0 native output custody"),
                credit,
                original,
            );
            running.preservation.push(preserved.report);
            original.checked_loaded_image()?;
            original.checked_timing()?;
            check(
                original.transport_complete()
                    && original.observation().numerical_environment()
                        == Some("NVIDIA_TF32_OVERRIDE=0"),
                "G0 original native closure/fixed precision environment absent",
            )?;
            let stdout = std::str::from_utf8(&original.process().stdout)
                .map_err(|_| invalid("G0 native status UTF-8"))?;
            let status: Value = common::decode(
                stdout
                    .trim()
                    .strip_prefix("RZ_NUMERIC_CAPTURE_STATUS=")
                    .ok_or_else(|| invalid("G0 native status absent"))?
                    .as_bytes(),
            )?;
            check(
                status
                    == serde_json::json!({"status":"captured","numerical_acceptance":false,"rules_legality_certified":false,"native_worker_joined":true}),
                "G0 native physical completion or worker join absent",
            )?;
            if let Some(error) = preserved.error {
                return Err(error);
            }
            let policy_path = r
                .native_output_root
                .join("actual-cudnn-execution-policy.json");
            let policy_pin = numerical::observe_raw(&policy_path, credit)?;
            policy_pin.validate(16 << 10)?;
            let (policy_file, policy_raw) =
                common::read_observed_output(policy_pin, 16 << 10, credit, true)?;
            let policy: Value = common::decode(&policy_raw)?;
            check(
                policy["domain"] == "rz-pals-numerical-capture-execution-policy/1"
                    && policy["loaded"]["requested_cudnn_conv_search"] == r.cudnn_conv_search
                    && policy["captured_physical_completions_known"] == true
                    && policy["native_worker_joined"] == true
                    && policy["actual_cudnn_plan"] == "unknown"
                    && policy["tf32_actual"] == "unknown"
                    && policy["numerical_acceptance"] == false
                    && policy["performance_improvement_claim"] == false,
                "G0 actual loaded policy/physical join scope differs",
            )?;
            running
                .held
                .as_mut()
                .expect("G0 actual native policy custody")
                .checkpoints
                .push(policy_file);
            Ok(())
        })();
        credit.remaining = cleanup_credit.remaining;
        result
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Final admission binds both original numerical producers and held G0 assets"
    )]
    fn finalize(
        r: &G0NumericalLaunchRegistrationV5,
        mut config: PalsAdaptiveStartupConfigV5,
        lock: PalsInputLockV5,
        cwd: &Path,
        plan: &Value,
        references: &BTreeMap<String, Pin>,
        until: Instant,
        cleanup_until: Instant,
        cancel: &AtomicBool,
        running: &mut Running,
        credit: &mut ReadCredit<'_>,
    ) -> Result<Completed, ArenaError> {
        // Final bounded raw readback is part of mandatory cleanup, not a new NN
        // execution window. Physical completion was required before this point.
        let preserve_cancel = AtomicBool::new(false);
        check(
            !cancel.load(Ordering::Acquire),
            "G0 final admission cancelled",
        )?;
        let mut cleanup_credit = ReadCredit {
            remaining: credit.remaining,
            maximum: credit.maximum,
            until: cleanup_until,
            cancel: &preserve_cancel,
            account: Arc::clone(&credit.account),
        };
        let result = (|| -> Result<Completed, ArenaError> {
            let credit = &mut cleanup_credit;
            let held = running.held.as_mut().expect("original G0 final custody");
            config.assets.retain(|a| {
                !matches!(
                    a.purpose,
                    Purpose::NumericalAcceptance | Purpose::NumericalRawEvidence
                )
            });
            let mut pairs = Vec::new();
            let mut total = 0u64;
            for group in plan["groups"]
                .as_array()
                .ok_or_else(|| invalid("G0 final groups absent"))?
            {
                let id = group["id"]
                    .as_str()
                    .ok_or_else(|| invalid("G0 final group ID"))?;
                let reference = references
                    .get(&format!("reference.{id}.json"))
                    .ok_or_else(|| invalid("G0 final reference absent"))?;
                let native_path = r.native_output_root.join(format!("native-{id}.json"));
                let native = numerical::observe_raw(&native_path, credit)?;
                let mut artifacts = Vec::new();
                for (label, pin) in [("reference", reference), ("native", &native)] {
                    total = total
                        .checked_add(pin.bytes)
                        .filter(|n| *n <= 64 << 20)
                        .ok_or_else(|| invalid("G0 combined raw payload cap"))?;
                    let file = common::open_file(&pin.path)?;
                    let raw = common::verify_bytes(&file, pin, true, credit)?;
                    numerical::ensure_output_credit(
                        &held.staging,
                        r.max_output_bytes,
                        raw.len() as u64,
                        cleanup_until,
                        &preserve_cancel,
                    )?;
                    let name = format!("g0-numerical-{label}-{id}.json");
                    let copy = common::write_owned_new(&cwd.join(&name), &raw)?;
                    let asset =
                        numerical::raw_asset(&copy.pin, name, Purpose::NumericalRawEvidence);
                    artifacts.push(asset.artifact.clone());
                    config.assets.push(asset);
                    held.checkpoints.push(copy);
                }
                let (file, _) = common::read_observed_output(native, MAX_JSON, credit, false)?;
                held.checkpoints.push(file);
                pairs.push(PalsNumericalRunPairV5 {
                    reference: artifacts.remove(0),
                    native: artifacts.remove(0),
                });
            }
            let acceptance = PalsNumericalAcceptanceV5 {
                schema_version: 5,
                domain: rz_experiments::PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN.into(),
                bindings: numerical::bindings(&config)?,
                invocation: PalsNumericalInvocationV5::Fresh,
                batch: 1,
                runs: pairs,
            };
            let raw = serde_json::to_vec(&acceptance).map_err(|e| invalid(e.to_string()))?;
            check(raw.len() <= 256 << 10, "G0 bounded V5 acceptance")?;
            numerical::ensure_output_credit(
                &held.staging,
                r.max_output_bytes,
                raw.len() as u64,
                cleanup_until,
                &preserve_cancel,
            )?;
            let file = common::write_owned_new(&r.acceptance_path, &raw)?;
            let acceptance_pin = file.pin.clone();
            let relative = r
                .acceptance_path
                .strip_prefix(cwd)
                .map_err(|_| invalid("G0 acceptance original cwd"))?
                .to_str()
                .ok_or_else(|| invalid("G0 acceptance UTF-8"))?
                .replace('\\', "/");
            config.assets.push(numerical::raw_asset(
                &acceptance_pin,
                relative,
                Purpose::NumericalAcceptance,
            ));
            held.checkpoints.push(file);
            config.validate_against(&lock)?;
            let mut read = |artifact: &ArtifactRef, limit: usize| {
                read_asset(&config, cwd, held, artifact, limit, credit)
            };
            let admission =
                rz_experiments::validate_pals_numerical_acceptance_v5(&config, &raw, &mut read)?;
            let export = read(&config.asset(Purpose::ExportManifest)?.artifact, 256 << 10)?;
            let domain =
                rz_experiments::validate_pals_artifact_domain_v5(&config, &export, &mut read)?;
            check(
                domain.artifact_domain == PalsArtifactDomainV5::Untrained
                    && !domain.tensor_optimizer_content_verified
                    && !domain.arena_accepted,
                "G0 numerical cannot mint trained/game authority",
            )?;
            common::revalidate(held, credit)?;
            super::super::selfplay_cli::scan_staging(
                &held.staging,
                r.max_output_bytes,
                cleanup_until,
                &preserve_cancel,
            )?;
            check(
                running.processes.len() == 3,
                "G0 exact original prepare/reference/native process chain",
            )?;
            let parent = serde_json::json!({"schema":"rz-pals-original-g0-numerical-parent/1","numerical_id":r.numerical_id,
            "launch_registration":running.registration_pin,"acceptance":acceptance_pin,"admission":admission,
            "artifact_domain":domain,"initialization_seed":23,"trained":false,"optimizer_updates_executed":0,
            "processes":running.processes.iter().map(|p|serde_json::json!({"receipt":p.process().receipt,"transport":p.observation()})).collect::<Vec<_>>(),
            "child_receipts":running.receipts,"charged_read_bytes":running.account.charged.load(Ordering::Acquire),
            "read_usage_uncertain":running.account.uncertain.load(Ordering::Acquire),"staging_cap_bytes":r.max_output_bytes,
            "original_g0_initializer_export_process_and_fds_retained":true,
            "first_original_registration_descriptor_retained":true,"physical_completion":"confirmed_native_join",
            "cudnn_conv_search_requested":r.cudnn_conv_search,"cuda_actual_plan":"unknown","actual_tf32":"unknown",
            "original_deadline_preserved":true,"automatic_retry":false,"collection_authority_issued":false,
            "numerical_only":true,"startup_admission_not_issued":true,
            "training_authority_issued":false,"arena_accepted":false,"sole_gpu_lease_observed_here":false});
            let raw = serde_json::to_vec(&parent).map_err(|e| invalid(e.to_string()))?;
            check(raw.len() as u64 <= MAX_JSON, "G0 bounded parent receipt")?;
            numerical::ensure_output_credit(
                &held.staging,
                r.max_output_bytes,
                raw.len() as u64,
                cleanup_until,
                &preserve_cancel,
            )?;
            held.checkpoints
                .push(common::write_owned_new(&r.parent_receipt_path, &raw)?);
            Ok((config, lock, admission, acceptance_pin, parent))
        })();
        credit.remaining = cleanup_credit.remaining;
        credit.until = until;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_g0_launch_rejects_unknown_authority_fields() {
        let raw = br#"{"schema":"rz-pals-g0-numerical-launch/1","training_authorized":true}"#;
        assert!(
            super::super::training_process::decode::<G0NumericalLaunchRegistrationV5>(raw).is_err()
        );
    }
    #[cfg(target_os = "linux")]
    fn metadata_launch() -> G0NumericalLaunchRegistrationV5 {
        let pin = |name: &str| Pin {
            path: PathBuf::from(format!("/original/{name}")),
            sha256: "a".repeat(64),
            bytes: 1,
        };
        G0NumericalLaunchRegistrationV5 {
            schema: G0_NUMERICAL_LAUNCH_V5.into(),
            numerical_id: "g0-math".into(),
            source_commit: "b".repeat(40),
            template_startup: pin("template.json"),
            case_plan: pin("cases.json"),
            rules_profile: pin("rules.json"),
            native_capture_config: pin("native.json"),
            native_executable: pin("native"),
            python_runtime: GenerationTrainingPythonRuntimeV3 {
                version: "3.12.3".into(),
                executable: pin("python"),
            },
            dependency_directories: vec![PathBuf::from("/trusted")],
            runtime_module_pins: vec![Pin {
                path: PathBuf::from("/trusted/torch.py"),
                sha256: "c".repeat(64),
                bytes: 1,
            }],
            staging_root: PathBuf::from("/managed"),
            initialization_directory: PathBuf::from("/managed/private/init"),
            export_directory: PathBuf::from("/managed/private/export"),
            reference_output_root: PathBuf::from("/managed/reference"),
            native_output_root: PathBuf::from("/managed/native"),
            prepare_receipt_path: PathBuf::from("/managed/prepare.json"),
            reference_receipt_path: PathBuf::from("/managed/reference.json"),
            acceptance_path: PathBuf::from("/managed/private/acceptance.json"),
            parent_receipt_path: PathBuf::from("/managed/parent.json"),
            cudnn_conv_search: "legacy_implicit_exhaustive".into(),
            max_wall_ns: 1_000_000_000,
            max_output_bytes: MAX_STAGING,
            max_read_bytes: MAX_STAGING,
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn g0_registration_refuses_output_aliases_credit_expansion_and_implicit_policy() {
        let valid = metadata_launch();
        assert!(valid.validate().is_ok());
        let mut alias = valid.clone();
        alias.native_output_root = alias.reference_output_root.clone();
        assert!(alias.validate().is_err());
        let mut oversized = valid.clone();
        oversized.max_read_bytes = 4 * MAX_STAGING + 1;
        assert!(oversized.validate().is_err());
        let mut implicit = valid.clone();
        implicit.cudnn_conv_search.clear();
        assert!(implicit.validate().is_err());
        let mut input_alias = valid;
        input_alias.case_plan.path = input_alias.export_directory.join("cases.json");
        assert!(input_alias.validate().is_err());
    }
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn platform_refusal_neither_opens_inputs_nor_creates_processes() {
        let failed = run_g0_numerical_original_v5(
            Path::new("never-open"),
            "",
            Arc::new(AtomicBool::new(false)),
        )
        .err()
        .unwrap();
        assert!(failed.processes.is_empty());
        assert!(failed.registration_pin.is_none());
        assert_eq!(failed.charged_read_bytes, 0);
    }
}
