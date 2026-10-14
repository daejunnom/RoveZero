//! Original, V-free numerical admission of the same live untrained G0.
//!
//! This stage cannot be reconstructed from a Collection receipt. Its caller
//! transfers a live Collection owner; the returned owner (or failure) retains
//! that owner and its original checkpoint descriptors. Only the new P/C export
//! and its independent raw comparison are issued here, never startup authority.
#[cfg(target_os = "linux")]
use super::g0_bootstrap_process::G0NumericalBootstrapRegistrationV5;
use super::{
    GenerationTrainingFilePinV3 as Pin, check,
    g0_bootstrap_process::VerifiedG0BootstrapNumericalV5, invalid,
};
use crate::{ArenaError, process::OriginalProcessOutput};
use rz_experiments::{PalsNumericalAdmissionV5, PalsNumericalScopeV5};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

pub const G0_DEPLOYMENT_NUMERICAL_LAUNCH_V5: &str = "rz-pals-g0-deployment-numerical-launch/1";
pub const G0_DEPLOYMENT_NUMERICAL_INTENT_V5: &str = "rz-pals-g0-deployment-numerical-intent/1";
#[cfg(target_os = "linux")]
const MAX_JSON: u64 = 4 << 20;
const MAX_STAGING: u64 = 18 << 30;
#[cfg(target_os = "linux")]
const MAX_INTENT: u64 = 64 << 10;
#[cfg(target_os = "linux")]
const SEAL_READ_BYTES: u64 = 16 << 20;
#[cfg(target_os = "linux")]
const SEAL_WALL_NS: u64 = 30_000_000_000;

/// Fresh output names and a new bounded execution, not a serialized G0 owner.
/// Runtime, Rules, source, inputs and actual cases come from the live owner.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0DeploymentNumericalRegistrationV5 {
    pub schema: String,
    pub numerical_id: String,
    pub collection_acceptance: Pin,
    pub export_directory: PathBuf,
    pub reference_output_root: PathBuf,
    pub native_output_root: PathBuf,
    pub export_receipt_path: PathBuf,
    pub reference_receipt_path: PathBuf,
    pub acceptance_path: PathBuf,
    pub parent_receipt_path: PathBuf,
    pub max_wall_ns: u64,
    pub max_output_bytes: u64,
    pub max_read_bytes: u64,
}
impl G0DeploymentNumericalRegistrationV5 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        use super::training_process::{absolute_safe, plain_name};
        check(
            self.schema == G0_DEPLOYMENT_NUMERICAL_LAUNCH_V5
                && plain_name(&self.numerical_id)
                && self.numerical_id.len() <= 96
                && (1..=3_600_000_000_000).contains(&self.max_wall_ns)
                && (1..=MAX_STAGING).contains(&self.max_output_bytes)
                && (1..=4 * MAX_STAGING).contains(&self.max_read_bytes),
            "G0 Deployment fresh finite registration",
        )?;
        self.collection_acceptance.validate(256 << 10)?;
        let directories = [
            &self.export_directory,
            &self.reference_output_root,
            &self.native_output_root,
        ];
        let receipts = [
            &self.export_receipt_path,
            &self.reference_receipt_path,
            &self.acceptance_path,
            &self.parent_receipt_path,
        ];
        let paths = directories.into_iter().chain(receipts).collect::<Vec<_>>();
        check(
            paths.iter().all(|p| absolute_safe(p))
                && paths.iter().collect::<BTreeSet<_>>().len() == paths.len()
                && directories
                    .iter()
                    .all(|a| paths.iter().all(|b| *a == *b || !b.starts_with(a)))
                && receipts
                    .iter()
                    .all(|p| p.extension().is_some_and(|s| s == "json")),
            "G0 Deployment disjoint output paths",
        )
    }
}

/// Predeclare output names and finite work without a future acceptance digest.
/// The live owner supplies that digest only after Collection actually succeeds.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G0DeploymentNumericalIntentV5 {
    pub schema: String,
    pub numerical_id: String,
    pub registration_path: PathBuf,
    pub export_directory: PathBuf,
    pub reference_output_root: PathBuf,
    pub native_output_root: PathBuf,
    pub export_receipt_path: PathBuf,
    pub reference_receipt_path: PathBuf,
    pub acceptance_path: PathBuf,
    pub parent_receipt_path: PathBuf,
    pub max_wall_ns: u64,
    pub max_output_bytes: u64,
    pub max_read_bytes: u64,
}
impl G0DeploymentNumericalIntentV5 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        use super::training_process::{absolute_safe, plain_name};
        check(
            self.schema == G0_DEPLOYMENT_NUMERICAL_INTENT_V5
                && plain_name(&self.numerical_id)
                && self.numerical_id.len() <= 96
                && (1..=3_600_000_000_000).contains(&self.max_wall_ns)
                && (1..=MAX_STAGING).contains(&self.max_output_bytes)
                && (1..=4 * MAX_STAGING).contains(&self.max_read_bytes),
            "G0 Deployment intent finite output names, no future acceptance",
        )?;
        let directories = [
            &self.export_directory,
            &self.reference_output_root,
            &self.native_output_root,
        ];
        let receipts = [
            &self.registration_path,
            &self.export_receipt_path,
            &self.reference_receipt_path,
            &self.acceptance_path,
            &self.parent_receipt_path,
        ];
        let paths = directories.into_iter().chain(receipts).collect::<Vec<_>>();
        check(
            paths.iter().all(|p| absolute_safe(p))
                && paths.iter().collect::<BTreeSet<_>>().len() == paths.len()
                && directories
                    .iter()
                    .all(|a| paths.iter().all(|b| *a == *b || !b.starts_with(a)))
                && receipts
                    .iter()
                    .all(|p| p.extension().is_some_and(|s| s == "json")),
            "G0 Deployment intent disjoint safe output and registration names",
        )
    }
    #[cfg(any(target_os = "linux", test))]
    fn actual_registration(&self, acceptance: &Pin) -> G0DeploymentNumericalRegistrationV5 {
        G0DeploymentNumericalRegistrationV5 {
            schema: G0_DEPLOYMENT_NUMERICAL_LAUNCH_V5.into(),
            numerical_id: self.numerical_id.clone(),
            collection_acceptance: acceptance.clone(),
            export_directory: self.export_directory.clone(),
            reference_output_root: self.reference_output_root.clone(),
            native_output_root: self.native_output_root.clone(),
            export_receipt_path: self.export_receipt_path.clone(),
            reference_receipt_path: self.reference_receipt_path.clone(),
            acceptance_path: self.acceptance_path.clone(),
            parent_receipt_path: self.parent_receipt_path.clone(),
            max_wall_ns: self.max_wall_ns,
            max_output_bytes: self.max_output_bytes,
            max_read_bytes: self.max_read_bytes,
        }
    }
}

/// Metadata seal only. It retains the first original intent descriptor and the
/// same create-new registration descriptor across write, sync and readback.
/// It issues no neural, startup, training or arena capability.
pub struct SealedG0DeploymentRegistrationV5 {
    pin: Pin,
    intent_pin: Pin,
    observation: Value,
    #[cfg(target_os = "linux")]
    _intent: File,
    #[cfg(target_os = "linux")]
    held: super::training_process::LinuxHeld,
}

/// Strict metadata preflight before Collection may allocate a model. Keeping
/// this opaque owner alive preserves the original intent FD until the seal
/// joins its own first intent descriptor to that same inode.
pub struct PreflightedG0DeploymentIntentV5 {
    pin: Pin,
    observation: Value,
    #[cfg(target_os = "linux")]
    intent: File,
}
impl PreflightedG0DeploymentIntentV5 {
    pub fn pin(&self) -> &Pin {
        &self.pin
    }
    pub fn observation(&self) -> &Value {
        &self.observation
    }
}

pub struct G0DeploymentIntentPreflightFailureV5 {
    pub error: ArenaError,
    pub intent_pin: Option<Pin>,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    #[cfg(target_os = "linux")]
    _intent: Option<File>,
}
impl std::fmt::Debug for G0DeploymentIntentPreflightFailureV5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G0DeploymentIntentPreflightFailureV5")
            .field("error", &self.error)
            .field("intent_pin", &self.intent_pin)
            .field("charged_read_bytes", &self.charged_read_bytes)
            .field("read_usage_uncertain", &self.read_usage_uncertain)
            .finish_non_exhaustive()
    }
}

/// This checks only the already existing intent bytes and static declarations.
/// It does not inspect models, initialize a model or issue execution authority.
pub fn preflight_g0_deployment_intent_original_v5(
    path: &Path,
    sha256: &str,
    cancel: Arc<AtomicBool>,
) -> Result<PreflightedG0DeploymentIntentV5, Box<G0DeploymentIntentPreflightFailureV5>> {
    #[cfg(target_os = "linux")]
    {
        linux::preflight(path, sha256, &cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, sha256, cancel);
        Err(Box::new(G0DeploymentIntentPreflightFailureV5 {
            error: invalid("original G0 Deployment intent preflight requires Linux"),
            intent_pin: None,
            charged_read_bytes: 0,
            read_usage_uncertain: false,
        }))
    }
}
impl SealedG0DeploymentRegistrationV5 {
    pub fn pin(&self) -> &Pin {
        &self.pin
    }
    pub fn intent_pin(&self) -> &Pin {
        &self.intent_pin
    }
    pub fn observation(&self) -> &Value {
        &self.observation
    }
    #[cfg(target_os = "linux")]
    pub fn registration_file(&self) -> &File {
        &self
            .held
            .checkpoints
            .iter()
            .find(|f| f.pin == self.pin)
            .expect("sealed original registration FD")
            .file
    }
}

pub struct G0DeploymentIntentSealFailureV5 {
    pub error: ArenaError,
    pub intent_pin: Option<Pin>,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    #[cfg(target_os = "linux")]
    _intent: Option<File>,
    #[cfg(target_os = "linux")]
    _held: Option<super::training_process::LinuxHeld>,
}
impl std::fmt::Debug for G0DeploymentIntentSealFailureV5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G0DeploymentIntentSealFailureV5")
            .field("error", &self.error)
            .field("intent_pin", &self.intent_pin)
            .field("charged_read_bytes", &self.charged_read_bytes)
            .field("read_usage_uncertain", &self.read_usage_uncertain)
            .finish_non_exhaustive()
    }
}

pub fn seal_g0_deployment_intent_original_v5(
    intent_path: &Path,
    intent_sha256: &str,
    collection: &VerifiedG0BootstrapNumericalV5,
    cancel: Arc<AtomicBool>,
) -> Result<SealedG0DeploymentRegistrationV5, Box<G0DeploymentIntentSealFailureV5>> {
    #[cfg(target_os = "linux")]
    {
        linux::seal(intent_path, intent_sha256, collection, &cancel, None)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (intent_path, intent_sha256, collection, cancel);
        Err(Box::new(G0DeploymentIntentSealFailureV5 {
            error: invalid("original G0 Deployment intent seal requires Linux"),
            intent_pin: None,
            charged_read_bytes: 0,
            read_usage_uncertain: false,
        }))
    }
}

/// Seal after Collection using the exact preflighted intent inode. An identical
/// replacement pathname cannot stand in for the descriptor retained here.
pub fn seal_g0_deployment_preflighted_intent_original_v5(
    preflight: &PreflightedG0DeploymentIntentV5,
    collection: &VerifiedG0BootstrapNumericalV5,
    cancel: Arc<AtomicBool>,
) -> Result<SealedG0DeploymentRegistrationV5, Box<G0DeploymentIntentSealFailureV5>> {
    #[cfg(target_os = "linux")]
    {
        linux::seal(
            &preflight.pin.path,
            &preflight.pin.sha256,
            collection,
            &cancel,
            Some(preflight),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        seal_g0_deployment_intent_original_v5(
            &preflight.pin.path,
            &preflight.pin.sha256,
            collection,
            cancel,
        )
    }
}

/// No decoder or public constructor. The Collection owner remains alive even
/// after its old numerical execution window ended; it supplies no new time or
/// read credit. This stage has its own original finite execution and ledger.
pub struct VerifiedG0DeploymentNumericalV5 {
    collection: Arc<VerifiedG0BootstrapNumericalV5>,
    scope: PalsNumericalScopeV5,
    admission: PalsNumericalAdmissionV5,
    acceptance_pin: Pin,
    parent_receipt: Value,
    processes: Vec<OriginalProcessOutput>,
    #[cfg(target_os = "linux")]
    _registration: File,
    #[cfg(target_os = "linux")]
    held: super::training_process::LinuxHeld,
}
impl VerifiedG0DeploymentNumericalV5 {
    pub fn collection(&self) -> &VerifiedG0BootstrapNumericalV5 {
        &self.collection
    }
    pub fn scope(&self) -> &PalsNumericalScopeV5 {
        &self.scope
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
                .chain(&self.held.inputs)
                .map(|f| (&f.pin, &f.file))
        }
        #[cfg(not(target_os = "linux"))]
        {
            std::iter::empty()
        }
    }
}

/// A failed or physically uncertain new stage keeps both original owners.
pub struct G0DeploymentNumericalFailureV5 {
    pub error: Box<super::g0_numerical_process::G0NumericalFailureV5>,
    collection: Arc<VerifiedG0BootstrapNumericalV5>,
}
impl G0DeploymentNumericalFailureV5 {
    pub fn collection(&self) -> &VerifiedG0BootstrapNumericalV5 {
        &self.collection
    }
}
impl std::fmt::Debug for G0DeploymentNumericalFailureV5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G0DeploymentNumericalFailureV5")
            .field("error", &self.error.error)
            .finish_non_exhaustive()
    }
}

pub fn run_g0_deployment_numerical_original_v5(
    path: &Path,
    sha256: &str,
    collection: Arc<VerifiedG0BootstrapNumericalV5>,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedG0DeploymentNumericalV5, Box<G0DeploymentNumericalFailureV5>> {
    #[cfg(target_os = "linux")]
    {
        linux::run(path, sha256, collection, &cancel, None)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, sha256, cancel);
        Err(Box::new(G0DeploymentNumericalFailureV5 {
            error: Box::new(super::g0_numerical_process::G0NumericalFailureV5 {
                error: invalid("original G0 Deployment numerical stage requires Linux"),
                processes: Vec::new(),
                child_receipts: Vec::new(),
                preservation_reports: Vec::new(),
                charged_read_bytes: 0,
                read_usage_uncertain: false,
                registration_pin: None,
                _registration: None,
            }),
            collection,
        }))
    }
}

/// Same-process CLI handoff: the stage must join its first registration FD to
/// the original create-new FD retained by this seal before decoding or dispatch.
pub fn run_g0_deployment_numerical_sealed_original_v5(
    sealed: &SealedG0DeploymentRegistrationV5,
    collection: Arc<VerifiedG0BootstrapNumericalV5>,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedG0DeploymentNumericalV5, Box<G0DeploymentNumericalFailureV5>> {
    #[cfg(target_os = "linux")]
    {
        linux::run(
            &sealed.pin.path,
            &sealed.pin.sha256,
            collection,
            &cancel,
            Some(sealed),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        run_g0_deployment_numerical_original_v5(
            &sealed.pin.path,
            &sealed.pin.sha256,
            collection,
            cancel,
        )
    }
}

/// Project the already consumed Collection inputs, preserving every P/C byte
/// and group order. This produces no future graph/weights digest.
#[cfg(any(target_os = "linux", test))]
fn deployment_groups(
    plan: &Value,
    collection_bindings: &rz_experiments::PalsNumericalBindingsV5,
) -> Result<Value, ArenaError> {
    use super::g0_bootstrap_process::{
        G0_CASE_SPEC_V5, G0NumericalCaseSpecV5, G0NumericalImmutableBindingsV5,
    };
    super::numerical_process::validate_plan(plan, collection_bindings)?;
    let original = G0NumericalCaseSpecV5 {
        schema: G0_CASE_SPEC_V5.into(),
        immutable_bindings: G0NumericalImmutableBindingsV5::from_actual(collection_bindings),
        groups: plan["groups"].clone(),
    };
    original.validate()?;
    let groups = original
        .groups
        .as_array()
        .expect("validated original group array")
        .iter()
        .filter(|g| matches!(g["role"].as_str(), Some("proposer" | "critic")))
        .cloned()
        .collect::<Vec<_>>();
    let count = groups
        .iter()
        .map(|g| g["cases"].as_array().map_or(0, Vec::len))
        .sum::<usize>();
    check(
        groups.len() == 12 && count == 100,
        "G0 Deployment exact P/C twelve groups and hundred cases",
    )?;
    Ok(Value::Array(groups))
}

#[cfg(any(target_os = "linux", test))]
fn validate_deployment_rebinding(
    collection: &rz_experiments::PalsNumericalBindingsV5,
    deployment: &rz_experiments::PalsNumericalBindingsV5,
) -> Result<(), ArenaError> {
    check(
        collection
            .graphs
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == BTreeSet::from(["public", "shared_pc", "validator"])
            && deployment
                .graphs
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == BTreeSet::from(["public", "shared_pc"]),
        "Deployment exact graph-role transition",
    )?;
    let mut expected = collection.clone();
    expected.graphs = deployment.graphs.clone();
    expected.export_manifest_sha256 = deployment.export_manifest_sha256.clone();
    check(
        expected == *deployment,
        "Deployment may only change the actual V-free export graph/manifest bindings",
    )
}

#[cfg(target_os = "linux")]
mod linux {
    use super::super::{
        g0_bootstrap_process::{
            self as bootstrap, G0NativeCaptureRecipeV5, G0NumericalPreInitScopeV5,
        },
        g0_numerical_process::linux as original,
        numerical_process as numerical, training_process as common,
    };
    use super::*;
    use common::{HeldFile, LinuxHeld, ReadAccounting, ReadCredit};
    use original::{OriginalG0Scope, Running};
    use rz_experiments::{
        ArtifactRef, PalsArtifactDomainV5, PalsEngineModeV5, PalsStartupAssetPurposeV5 as Purpose,
    };
    use std::{
        collections::BTreeMap,
        sync::atomic::Ordering,
        time::{Duration, Instant},
    };
    const COMPONENTS: [&str; 5] = [
        "model_native",
        "model_encoding",
        "nn_runtime",
        "model_reference",
        "model_export",
    ];
    const INITIALIZERS: [&str; 8] = [
        "__init__.py",
        "config.py",
        "model.py",
        "artifacts.py",
        "controlled_initialization.py",
        "virtual_review.py",
        "onnx_shared.py",
        "initialization_evidence.py",
    ];
    type Completed = (PalsNumericalScopeV5, PalsNumericalAdmissionV5, Pin, Value);

    pub(super) fn preflight(
        path: &Path,
        sha: &str,
        cancel: &AtomicBool,
    ) -> Result<PreflightedG0DeploymentIntentV5, Box<G0DeploymentIntentPreflightFailureV5>> {
        let started = Instant::now();
        let account = ReadAccounting::new();
        let mut credit = ReadCredit {
            remaining: SEAL_READ_BYTES,
            maximum: SEAL_READ_BYTES,
            until: started + Duration::from_nanos(SEAL_WALL_NS),
            cancel,
            account: Arc::clone(&account),
        };
        let mut original = None;
        let mut pin = None;
        let result = (|| -> Result<Value, ArenaError> {
            use std::os::unix::fs::MetadataExt;
            credit.take(0)?;
            original = Some(common::open_file(path)?);
            let file = original
                .as_ref()
                .expect("first original preflight intent FD");
            let metadata = file
                .metadata()
                .map_err(|_| invalid("Deployment preflight original intent metadata"))?;
            pin = Some(Pin {
                path: path.into(),
                sha256: sha.into(),
                bytes: metadata.len(),
            });
            let pin = pin.as_ref().expect("original preflight intent pin");
            pin.validate(MAX_INTENT)?;
            check(
                metadata.uid() == nix::unistd::Uid::effective().as_raw(),
                "Deployment preflight current-owner intent",
            )?;
            let intent: G0DeploymentNumericalIntentV5 =
                common::decode(&common::verify_bytes(file, pin, true, &mut credit)?)?;
            intent.validate()?;
            common::verify_bytes(file, pin, false, &mut credit)?;
            common::check_original_g0_issue(&mut credit, cancel)?;
            let elapsed_ns = u64::try_from(started.elapsed().as_nanos())
                .map_err(|_| invalid("Deployment preflight elapsed overflow"))?;
            let observation = serde_json::json!({"schema":"rz-pals-g0-deployment-intent-preflight-observation/1",
                "status":"intent_preflighted_metadata_only","intent":pin,"numerical_id":intent.numerical_id,
                "maximum_wall_ns":SEAL_WALL_NS,"elapsed_ns":elapsed_ns,"maximum_read_bytes":SEAL_READ_BYTES,
                "charged_read_bytes":account.charged.load(Ordering::Acquire),"read_usage_uncertain":account.uncertain.load(Ordering::Acquire),
                "original_intent_fd_retained":true,"static_intent_structure_verified":true,
                "collection_execution_started":false,"numerical_stage_clock_started":false,
                "preflight_cost_separate_from_collection_and_numerical_stage":true,
                "actual_initialization_executed":false,"actual_reference_executed":false,"native_status":"not_run",
                "optimizer_updates_executed":0,"numerical_authority_issued":false,"startup_authority_issued":false,"arena_accepted":false});
            common::check_original_g0_issue(&mut credit, cancel)?;
            Ok(observation)
        })();
        match result {
            Ok(observation) => Ok(PreflightedG0DeploymentIntentV5 {
                pin: pin.expect("preflight actual intent pin"),
                observation,
                intent: original.expect("preflight first original intent descriptor"),
            }),
            Err(error) => Err(Box::new(G0DeploymentIntentPreflightFailureV5 {
                error,
                intent_pin: pin,
                charged_read_bytes: account.charged.load(Ordering::Acquire),
                read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
                _intent: original,
            })),
        }
    }

    pub(super) fn seal(
        path: &Path,
        sha: &str,
        collection: &VerifiedG0BootstrapNumericalV5,
        cancel: &AtomicBool,
        preflight: Option<&PreflightedG0DeploymentIntentV5>,
    ) -> Result<SealedG0DeploymentRegistrationV5, Box<G0DeploymentIntentSealFailureV5>> {
        let started = Instant::now();
        let until = started + Duration::from_nanos(SEAL_WALL_NS);
        let account = ReadAccounting::new();
        let mut credit = ReadCredit {
            remaining: SEAL_READ_BYTES,
            maximum: SEAL_READ_BYTES,
            until,
            cancel,
            account: Arc::clone(&account),
        };
        let mut first_intent = None;
        let mut intent_pin = None;
        let mut held = None;
        let result = (|| -> Result<(Pin, Value), ArenaError> {
            use std::os::unix::fs::MetadataExt;
            credit.take(0)?;
            first_intent = Some(common::open_file(path)?);
            let file = first_intent.as_ref().expect("first original intent FD");
            let metadata = file
                .metadata()
                .map_err(|_| invalid("Deployment intent original metadata"))?;
            let pin = Pin {
                path: path.into(),
                sha256: sha.into(),
                bytes: metadata.len(),
            };
            intent_pin = Some(pin.clone());
            pin.validate(MAX_INTENT)?;
            if let Some(prior) = preflight {
                check(
                    pin == *prior.pin(),
                    "seal intent exact preflight path/hash/length differs",
                )?;
                check_original_file_join(
                    &pin,
                    file,
                    &prior.intent,
                    "seal intent must join the original preflight inode/device/length",
                )?;
            }
            check(
                metadata.uid() == nix::unistd::Uid::effective().as_raw(),
                "Deployment intent current-owner input",
            )?;
            let intent: G0DeploymentNumericalIntentV5 =
                common::decode(&common::verify_bytes(file, &pin, true, &mut credit)?)?;
            intent.validate()?;
            check(
                collection.scope().mode == PalsEngineModeV5::Collection
                    && collection.scope().model.artifact_domain == PalsArtifactDomainV5::Untrained
                    && !collection.scope().training_executed,
                "Deployment intent requires the live untrained Collection owner",
            )?;
            collection.scope().validate_pre_acceptance()?;
            let old_pin: Pin =
                serde_json::from_value(collection.parent_receipt()["launch_registration"].clone())
                    .map_err(|e| invalid(e.to_string()))?;
            let old: G0NumericalBootstrapRegistrationV5 =
                common::decode(&read_original(collection, &old_pin, MAX_JSON, &mut credit)?)?;
            old.validate()?;
            let input: G0NumericalPreInitScopeV5 = common::decode(&read_original(
                collection,
                &old.pre_init_scope,
                8 << 20,
                &mut credit,
            )?)?;
            check(
                input.source_commit == collection.scope().source_commit
                    && input.source_components == collection.scope().source_components,
                "Deployment intent live original source differs",
            )?;
            let native: G0NativeCaptureRecipeV5 = common::decode(&read_original(
                collection,
                &old.native_recipe,
                64 << 10,
                &mut credit,
            )?)?;
            let actual = intent.actual_registration(collection.acceptance_pin());
            actual.validate()?;
            let mut execution = old.clone();
            execution.numerical_id = actual.numerical_id.clone();
            execution.export_directory = actual.export_directory.clone();
            execution.reference_output_root = actual.reference_output_root.clone();
            execution.native_output_root = actual.native_output_root.clone();
            execution.prepare_receipt_path = actual.export_receipt_path.clone();
            execution.reference_receipt_path = actual.reference_receipt_path.clone();
            execution.acceptance_path = actual.acceptance_path.clone();
            execution.parent_receipt_path = actual.parent_receipt_path.clone();
            execution.max_wall_ns = actual.max_wall_ns;
            execution.max_output_bytes = actual.max_output_bytes;
            execution.max_read_bytes = actual.max_read_bytes;
            execution.validate()?;
            check(
                actual.numerical_id != old.numerical_id
                    && actual.max_output_bytes <= old.max_output_bytes
                    && actual.max_read_bytes <= old.max_read_bytes
                    && actual
                        .export_directory
                        .starts_with(input.private_cwd.join("inputs"))
                    && actual.export_directory != input.private_cwd.join("inputs")
                    && actual.acceptance_path.parent()
                        == Some(input.private_cwd.join("inputs").as_path())
                    && intent.registration_path.parent() == Some(old.staging_root.as_path()),
                "Deployment seal retains original private root/caps and original staging parent",
            )?;
            for output in [
                &actual.export_directory,
                &actual.reference_output_root,
                &actual.native_output_root,
                &actual.export_receipt_path,
                &actual.reference_receipt_path,
                &actual.acceptance_path,
                &actual.parent_receipt_path,
                &intent.registration_path,
            ] {
                check(
                    !path.starts_with(output)
                        && !collection
                            .held_files()
                            .any(|(p, _)| p.path.starts_with(output))
                        && !output.starts_with(&old.initialization_directory)
                        && !output.starts_with(&old.export_directory)
                        && !output.starts_with(&old.reference_output_root)
                        && !output.starts_with(&old.native_output_root)
                        && !output.starts_with(&native.runtime_cache)
                        && !output.starts_with(
                            old.staging_root
                                .join(format!("g0-bootstrap-source-{}", old.numerical_id)),
                        ),
                    "Deployment intent cannot alias original inputs/evidence/cache/source",
                )?;
            }
            // The write parent is precisely the directory retained by the
            // Collection owner, not an unverified pathname created by the CLI.
            let original_root = collection
                .held_directory_file(&old.staging_root)
                .ok_or_else(|| invalid("live Collection original staging directory FD missing"))?;
            let original_identity = common::FileIdentity::from(
                &original_root
                    .metadata()
                    .map_err(|_| invalid("Collection original staging identity"))?,
            );
            let named_root = common::open_directory(&old.staging_root)?;
            let named_file = common::directory_file(&named_root)?;
            check(
                original_identity.same_inode(common::FileIdentity::from(
                    &named_file
                        .metadata()
                        .map_err(|_| invalid("Collection named staging identity"))?,
                )),
                "Deployment seal original staging parent was replaced",
            )?;
            held = Some(common::prepare_original_root(
                &old.staging_root,
                old.staging_root.join(format!(
                    "g0-deployment-seal-unused-source-{}",
                    actual.numerical_id
                )),
                &intent.registration_path,
                &[],
            )?);
            let custody = held
                .as_mut()
                .expect("intent seal original directory custody");
            // Keep original Collection root descriptor continuity in the new
            // seal owner. The original owner itself remains borrowed/alive.
            let root_clone = original_root
                .try_clone()
                .map_err(|_| invalid("Collection original staging FD clone"))?;
            custody.staging = cap_std::fs::Dir::from_std_file(root_clone);
            let directory_clone = original_root
                .try_clone()
                .map_err(|_| invalid("Collection original staging retention"))?;
            custody.directories[0] = (old.staging_root.clone(), directory_clone, original_identity);
            common::verify_bytes(
                original_pin(collection, collection.acceptance_pin())?,
                collection.acceptance_pin(),
                false,
                &mut credit,
            )?;
            let raw = serde_json::to_vec(&actual).map_err(|e| invalid(e.to_string()))?;
            check(
                !raw.is_empty() && raw.len() as u64 <= MAX_INTENT,
                "Deployment bounded actual sealed registration",
            )?;
            numerical::ensure_output_credit(
                &custody.staging,
                actual.max_output_bytes,
                raw.len() as u64,
                until,
                cancel,
            )?;
            common::revalidate(custody, &mut credit)?;
            let sealed_pin = common::write_original_owned_new(
                &intent.registration_path,
                &raw,
                custody,
                &mut credit,
            )?;
            common::revalidate(custody, &mut credit)?;
            common::verify_bytes(file, &pin, false, &mut credit)?;
            let final_root = original_root
                .metadata()
                .map_err(|_| invalid("Deployment sealed original staging metadata"))?;
            check(
                final_root.mode() & 0o7777 == 0o700
                    && final_root.uid() == nix::unistd::Uid::effective().as_raw(),
                "Deployment sealed original staging owner/mode changed",
            )?;
            common::check_original_g0_issue(&mut credit, cancel)?;
            let elapsed_ns = u64::try_from(started.elapsed().as_nanos())
                .map_err(|_| invalid("Deployment seal elapsed overflow"))?;
            let observation = serde_json::json!({"schema":"rz-pals-g0-deployment-intent-seal-observation/1",
                "status":"registration_sealed_metadata_only","intent":pin,"registration":sealed_pin,
                "collection_acceptance":collection.acceptance_pin(),"maximum_wall_ns":SEAL_WALL_NS,"elapsed_ns":elapsed_ns,
                "maximum_read_bytes":SEAL_READ_BYTES,"charged_read_bytes":account.charged.load(Ordering::Acquire),
                "read_usage_uncertain":account.uncertain.load(Ordering::Acquire),"staging_cap_bytes":actual.max_output_bytes,
                "original_intent_and_create_new_registration_fds_retained":true,"same_original_collection_staging_parent":true,
                "original_preflight_intent_inode_joined":preflight.is_some(),
                "preparation_cost_separate_from_numerical_stage":true,"numerical_stage_clock_started":false,
                "actual_initialization_executed":false,"actual_reference_executed":false,"native_status":"not_run",
                "optimizer_updates_executed":0,"numerical_authority_issued":false,"startup_authority_issued":false,"arena_accepted":false});
            common::check_original_g0_issue(&mut credit, cancel)?;
            Ok((sealed_pin, observation))
        })();
        match result {
            Ok((pin, observation)) => Ok(SealedG0DeploymentRegistrationV5 {
                pin,
                intent_pin: intent_pin.expect("original intent pin"),
                observation,
                _intent: first_intent.expect("first original intent descriptor"),
                held: held.expect("same original registration custody"),
            }),
            Err(error) => Err(Box::new(G0DeploymentIntentSealFailureV5 {
                error,
                intent_pin,
                charged_read_bytes: account.charged.load(Ordering::Acquire),
                read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
                _intent: first_intent,
                _held: held,
            })),
        }
    }

    pub(super) fn run(
        path: &Path,
        sha: &str,
        collection: Arc<VerifiedG0BootstrapNumericalV5>,
        cancel: &AtomicBool,
        seal: Option<&SealedG0DeploymentRegistrationV5>,
    ) -> Result<VerifiedG0DeploymentNumericalV5, Box<G0DeploymentNumericalFailureV5>> {
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
        match execute(path, sha, &collection, started, cancel, seal, &mut running) {
            Ok((scope, admission, acceptance_pin, parent_receipt)) => {
                Ok(VerifiedG0DeploymentNumericalV5 {
                    collection,
                    scope,
                    admission,
                    acceptance_pin,
                    parent_receipt,
                    processes: running.processes,
                    _registration: running
                        .registration
                        .expect("original Deployment registration"),
                    held: running.held.expect("original Deployment file custody"),
                })
            }
            Err(error) => Err(Box::new(G0DeploymentNumericalFailureV5 {
                collection,
                error: Box::new(super::super::g0_numerical_process::G0NumericalFailureV5 {
                    error,
                    processes: running.processes,
                    child_receipts: running.receipts,
                    preservation_reports: running.preservation,
                    charged_read_bytes: running.account.charged.load(Ordering::Acquire),
                    read_usage_uncertain: running.account.uncertain.load(Ordering::Acquire),
                    registration_pin: running.registration_pin,
                    _registration: running.registration,
                    _held: running.held,
                }),
            })),
        }
    }

    fn original_pin<'a>(
        collection: &'a VerifiedG0BootstrapNumericalV5,
        pin: &Pin,
    ) -> Result<&'a File, ArenaError> {
        let mut files = collection.held_files().filter(|(p, _)| *p == pin);
        let (_, first) = files
            .next()
            .ok_or_else(|| invalid("live Collection original input FD missing"))?;
        let identity = common::same_named(pin, first)?;
        for (_, duplicate) in files {
            check(
                common::same_named(pin, duplicate)? == identity,
                "ambiguous Collection original input identity",
            )?;
        }
        Ok(first)
    }
    fn read_original(
        collection: &VerifiedG0BootstrapNumericalV5,
        pin: &Pin,
        cap: u64,
        credit: &mut ReadCredit<'_>,
    ) -> Result<Vec<u8>, ArenaError> {
        pin.validate(cap)?;
        common::verify_bytes(original_pin(collection, pin)?, pin, true, credit)
    }
    fn retain_original(
        collection: &VerifiedG0BootstrapNumericalV5,
        pin: &Pin,
        held: &mut LinuxHeld,
        credit: &mut ReadCredit<'_>,
    ) -> Result<(), ArenaError> {
        if held.inputs.iter().any(|f| f.pin == *pin) {
            return Ok(());
        }
        let file = original_pin(collection, pin)?;
        let identity = common::same_named(pin, file)?;
        let clone = file
            .try_clone()
            .map_err(|_| invalid("Collection original FD duplication"))?;
        // Put the original descriptor into custody before any new read fails.
        held.inputs.push(HeldFile {
            file: clone,
            pin: pin.clone(),
            identity,
        });
        common::verify_bytes(
            &held.inputs.last().expect("retained original").file,
            pin,
            false,
            credit,
        )?;
        Ok(())
    }
    fn scope_pin(
        collection: &VerifiedG0BootstrapNumericalV5,
        cwd: &Path,
        purpose: Purpose,
    ) -> Result<Pin, ArenaError> {
        let a = collection.scope().asset(purpose)?;
        let pin = Pin {
            path: cwd.join(&a.snapshot_path),
            sha256: a.artifact.sha256.clone(),
            bytes: a.artifact.bytes,
        };
        original_pin(collection, &pin)?;
        Ok(pin)
    }

    fn check_sealed_registration_join(
        pin: &Pin,
        opened: &File,
        original: &File,
    ) -> Result<(), ArenaError> {
        check_original_file_join(
            pin,
            opened,
            original,
            "stage registration must join the original sealed file inode/device/length",
        )
    }

    fn check_original_file_join(
        pin: &Pin,
        opened: &File,
        original: &File,
        mismatch: &str,
    ) -> Result<(), ArenaError> {
        use std::os::unix::fs::MetadataExt;
        let original_meta = original
            .metadata()
            .map_err(|_| invalid("handoff original metadata"))?;
        let opened_meta = opened
            .metadata()
            .map_err(|_| invalid("handoff opened metadata"))?;
        let original_identity = common::FileIdentity::from(&original_meta);
        check(
            original_meta.is_file()
                && opened_meta.is_file()
                && original_meta.nlink() == 1
                && opened_meta.nlink() == 1
                && original_meta.uid() == nix::unistd::Uid::effective().as_raw()
                && opened_meta.uid() == original_meta.uid()
                && original_meta.len() == pin.bytes
                && opened_meta.len() == pin.bytes
                && original_identity.same_inode(common::FileIdentity::from(&opened_meta))
                && original_identity == common::same_named(pin, original)?,
            mismatch,
        )
    }

    fn execute(
        path: &Path,
        sha: &str,
        collection: &VerifiedG0BootstrapNumericalV5,
        started: Instant,
        cancel: &AtomicBool,
        seal: Option<&SealedG0DeploymentRegistrationV5>,
        running: &mut Running,
    ) -> Result<Completed, ArenaError> {
        let mut credit = ReadCredit {
            remaining: 4 * MAX_STAGING,
            maximum: 4 * MAX_STAGING,
            until: started + Duration::from_secs(60),
            cancel,
            account: Arc::clone(&running.account),
        };
        running.registration = Some(common::open_file(path)?);
        let file = running
            .registration
            .as_ref()
            .expect("first original Deployment registration");
        let pin = Pin {
            path: path.into(),
            sha256: sha.into(),
            bytes: file
                .metadata()
                .map_err(|_| invalid("Deployment registration metadata"))?
                .len(),
        };
        running.registration_pin = Some(pin.clone());
        pin.validate(MAX_JSON)?;
        if let Some(sealed) = seal {
            check(
                pin == *sealed.pin(),
                "stage registration exact sealed path/hash/length differs",
            )?;
            check_sealed_registration_join(&pin, file, sealed.registration_file())?;
        }
        let d: G0DeploymentNumericalRegistrationV5 =
            common::decode(&common::verify_bytes(file, &pin, true, &mut credit)?)?;
        d.validate()?;
        check(
            d.collection_acceptance == *collection.acceptance_pin(),
            "Deployment registration requires the supplied live Collection owner",
        )?;
        check(
            collection.scope().mode == PalsEngineModeV5::Collection
                && collection.scope().model.artifact_domain == PalsArtifactDomainV5::Untrained
                && !collection.scope().training_executed,
            "only live untrained Collection G0 can enter Deployment stage",
        )?;
        collection.scope().validate_pre_acceptance()?;
        let until = started
            .checked_add(Duration::from_nanos(d.max_wall_ns))
            .ok_or_else(|| invalid("Deployment deadline overflow"))?;
        let cleanup_until = until + Duration::from_secs(30);
        credit.remaining = d
            .max_read_bytes
            .checked_sub(running.account.charged.load(Ordering::Acquire))
            .ok_or_else(|| invalid("Deployment registration consumed read credit"))?;
        credit.maximum = d.max_read_bytes;
        credit.until = until;
        credit.take(0)?;
        let old_registration: Pin =
            serde_json::from_value(collection.parent_receipt()["launch_registration"].clone())
                .map_err(|e| invalid(e.to_string()))?;
        let original_r: G0NumericalBootstrapRegistrationV5 = common::decode(&read_original(
            collection,
            &old_registration,
            MAX_JSON,
            &mut credit,
        )?)?;
        original_r.validate()?;
        check(
            d.numerical_id != original_r.numerical_id,
            "Deployment stage requires a distinct original execution identifier",
        )?;
        let i: G0NumericalPreInitScopeV5 = common::decode(&read_original(
            collection,
            &original_r.pre_init_scope,
            8 << 20,
            &mut credit,
        )?)?;
        check(
            i.source_commit == collection.scope().source_commit
                && i.source_components == collection.scope().source_components,
            "Deployment cannot replace Collection source",
        )?;
        let native: G0NativeCaptureRecipeV5 = common::decode(&read_original(
            collection,
            &original_r.native_recipe,
            64 << 10,
            &mut credit,
        )?)?;
        check(
            native.native_options == i.native_options,
            "Deployment native options differ from Collection",
        )?;
        let mut r = original_r.clone();
        r.numerical_id = d.numerical_id.clone();
        r.export_directory = d.export_directory.clone();
        r.reference_output_root = d.reference_output_root.clone();
        r.native_output_root = d.native_output_root.clone();
        r.prepare_receipt_path = d.export_receipt_path.clone();
        r.reference_receipt_path = d.reference_receipt_path.clone();
        r.acceptance_path = d.acceptance_path.clone();
        r.parent_receipt_path = d.parent_receipt_path.clone();
        r.max_wall_ns = d.max_wall_ns;
        r.max_output_bytes = d.max_output_bytes;
        r.max_read_bytes = d.max_read_bytes;
        r.validate()?;
        check(
            d.max_output_bytes <= original_r.max_output_bytes
                && d.max_read_bytes <= original_r.max_read_bytes
                && d.export_directory.starts_with(i.private_cwd.join("inputs"))
                && d.export_directory != i.private_cwd.join("inputs")
                && d.acceptance_path.parent() == Some(i.private_cwd.join("inputs").as_path()),
            "Deployment retains Collection private snapshot root and managed caps",
        )?;
        for output in [
            &d.export_directory,
            &d.reference_output_root,
            &d.native_output_root,
            &d.export_receipt_path,
            &d.reference_receipt_path,
            &d.acceptance_path,
            &d.parent_receipt_path,
        ] {
            check(
                !collection
                    .held_files()
                    .any(|(p, _)| p.path.starts_with(output))
                    && !output.starts_with(&original_r.initialization_directory)
                    && !output.starts_with(&original_r.export_directory)
                    && !output.starts_with(&original_r.reference_output_root)
                    && !output.starts_with(&original_r.native_output_root)
                    && !output.starts_with(&native.runtime_cache),
                "Deployment fresh namespace cannot alias retained Collection evidence or cache",
            )?;
        }
        running.held = Some(common::prepare_original_root(
            &r.staging_root,
            r.staging_root
                .join(format!("g0-deployment-source-{}", d.numerical_id)),
            &d.export_receipt_path,
            &[
                &d.export_directory,
                &d.reference_output_root,
                &d.native_output_root,
            ],
        )?);
        let held = running.held.as_mut().expect("Deployment directory custody");
        for directory in [
            &d.export_directory,
            &d.reference_output_root,
            &d.native_output_root,
        ] {
            check(
                common::open_directory(directory)?
                    .entries()
                    .map_err(|_| invalid("Deployment output entries"))?
                    .next()
                    .is_none(),
                "Deployment requires fresh empty output directories",
            )?;
        }
        for output in [
            &d.reference_receipt_path,
            &d.acceptance_path,
            &d.parent_receipt_path,
        ] {
            check(
                !output.exists(),
                "Deployment refuses existing terminal output",
            )?;
        }
        if seal.is_some() {
            // Continue custody from the exact first FD joined to the seal;
            // reopening an identical replacement pathname is not a handoff.
            let first = running
                .registration
                .as_ref()
                .expect("joined original stage registration");
            let identity = common::same_named(&pin, first)?;
            let original = first
                .try_clone()
                .map_err(|_| invalid("joined registration original FD clone"))?;
            held.inputs.push(HeldFile {
                file: original,
                pin: pin.clone(),
                identity,
            });
            common::verify_bytes(
                &held
                    .inputs
                    .last()
                    .expect("retained joined registration")
                    .file,
                &pin,
                false,
                &mut credit,
            )?;
        } else {
            common::hold_input(pin, false, held, &mut credit)?;
        }
        common::retain_directory(i.private_cwd.clone(), held)?;
        common::retain_directory(native.runtime_cache.clone(), held)?;
        for input in [
            &old_registration,
            &original_r.pre_init_scope,
            &original_r.native_recipe,
            collection.acceptance_pin(),
            &r.rules_profile,
            &r.native_executable,
            &r.python_runtime.executable,
        ] {
            retain_original(collection, input, held, &mut credit)?;
        }
        let inherited = collection
            .scope()
            .assets
            .iter()
            .filter(|a| {
                matches!(
                    a.purpose,
                    Purpose::Weights
                        | Purpose::InitializationEvidence
                        | Purpose::InitializationCheckpointMetadata
                        | Purpose::RuntimeLibrary
                        | Purpose::RulesDescriptor
                        | Purpose::CudaManifest
                        | Purpose::CudaLibrary
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        for a in &inherited {
            retain_original(
                collection,
                &Pin {
                    path: i.private_cwd.join(&a.snapshot_path),
                    sha256: a.artifact.sha256.clone(),
                    bytes: a.artifact.bytes,
                },
                held,
                &mut credit,
            )?;
        }
        let dependencies = r
            .dependency_directories
            .iter()
            .map(|p| common::retain_directory(p.clone(), held))
            .collect::<Result<Vec<_>, _>>()?;
        for input in &r.runtime_module_pins {
            retain_original(collection, input, held, &mut credit)?;
        }
        common::install_sources_for_components(held, &mut credit, &COMPONENTS)?;
        let sources = COMPONENTS
            .into_iter()
            .map(|key| {
                i.source_components
                    .get(key)
                    .map(|v| (key.into(), v.clone()))
                    .ok_or_else(|| invalid("Deployment numerical source missing"))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let mut initializers = BTreeMap::new();
        for name in INITIALIZERS {
            let source = held
                .sources
                .iter()
                .find(|f| {
                    f.pin.path
                        == held
                            .source_root
                            .join("experiments/model-research/pals/src/rz_pals_model")
                            .join(name)
                })
                .ok_or_else(|| invalid("Deployment original initializer source absent"))?;
            initializers.insert(name, source.pin.sha256.clone());
        }
        let initialization_sources = serde_json::json!({"schema":"rz-pals-g0-initialization-sources/1","source_commit":r.source_commit,"pins":initializers});
        let plan_path = original_r.staging_root.join(format!(
            "g0-bootstrap-plan-{}.json",
            original_r.numerical_id
        ));
        let plan_pin = collection
            .held_files()
            .find(|(p, _)| p.path == plan_path)
            .map(|(p, _)| p.clone())
            .ok_or_else(|| invalid("live Collection submitted plan FD missing"))?;
        let plan: Value = common::decode(&read_original(
            collection,
            &plan_pin,
            MAX_JSON,
            &mut credit,
        )?)?;
        let collection_bindings = rz_experiments::numerical_scope_bindings_v5(collection.scope())?;
        check(
            plan["bindings"]
                == serde_json::to_value(&collection_bindings)
                    .map_err(|e| invalid(e.to_string()))?,
            "Collection plan exact original bindings differ",
        )?;
        let groups = deployment_groups(&plan, &collection_bindings)?;
        let group_seal = serde_json::to_vec(&groups).map_err(|e| invalid(e.to_string()))?;
        let checkpoint = scope_pin(collection, &i.private_cwd, Purpose::Weights)?;
        let metadata = scope_pin(
            collection,
            &i.private_cwd,
            Purpose::InitializationCheckpointMetadata,
        )?;
        let evidence = scope_pin(collection, &i.private_cwd, Purpose::InitializationEvidence)?;
        let execution = OriginalG0Scope {
            numerical_id: r.numerical_id.clone(),
            source_commit: r.source_commit.clone(),
            python_runtime: r.python_runtime.clone(),
            runtime_module_pins: r.runtime_module_pins.clone(),
            native_executable: r.native_executable.clone(),
            staging_root: r.staging_root.clone(),
            initialization_directory: r.initialization_directory.clone(),
            export_directory: r.export_directory.clone(),
            reference_output_root: r.reference_output_root.clone(),
            native_output_root: r.native_output_root.clone(),
            prepare_receipt_path: r.prepare_receipt_path.clone(),
            reference_receipt_path: r.reference_receipt_path.clone(),
            parent_receipt_path: r.parent_receipt_path.clone(),
            cudnn_conv_search: native.native_options.cudnn_conv_search.clone(),
            max_wall_ns: r.max_wall_ns,
            max_output_bytes: r.max_output_bytes,
        };
        let export = original::deployment_child(
            &execution,
            "export",
            serde_json::json!({"checkpoint":checkpoint,
            "checkpoint_metadata":metadata,"initialization_evidence":evidence,"rules_profile":r.rules_profile}),
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
        let outputs = export
            .get("outputs")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("Deployment export outputs absent"))?;
        check(
            outputs.len() == 3
                && ["export_manifest", "export_graphs", "cuda_control_inventory"]
                    .into_iter()
                    .all(|k| outputs.contains_key(k)),
            "Deployment export cannot issue a new checkpoint or initializer",
        )?;
        let mut scope = prepared_scope(
            collection.scope(),
            &i,
            &r,
            &export["outputs"],
            inherited,
            running.held.as_mut().expect("Deployment export custody"),
            &mut credit,
        )?;
        bootstrap::linux::validate_prepared(
            &scope,
            &i.private_cwd,
            &r,
            running.held.as_ref().expect("Deployment prepared custody"),
            &mut credit,
        )?;
        let bindings = rz_experiments::numerical_scope_bindings_v5(&scope)?;
        validate_deployment_rebinding(&collection_bindings, &bindings)?;
        let plan = serde_json::json!({"schema_version":5,"domain":"rz-pals-numerical-case-plan-v5/1","invocation":"fresh","batch":1,"bindings":bindings,"groups":groups});
        numerical::validate_plan(&plan, &bindings)?;
        check(
            serde_json::to_vec(&plan["groups"]).map_err(|e| invalid(e.to_string()))? == group_seal,
            "Deployment P/C group order/content changed",
        )?;
        let raw = serde_json::to_vec(&plan).map_err(|e| invalid(e.to_string()))?;
        check(
            raw.len() as u64 <= MAX_JSON,
            "Deployment materialized plan byte cap",
        )?;
        let held = running.held.as_mut().expect("Deployment plan custody");
        numerical::ensure_output_credit(
            &held.staging,
            r.max_output_bytes,
            raw.len() as u64,
            until,
            cancel,
        )?;
        let actual_plan = common::write_original_owned_new(
            &r.staging_root
                .join(format!("g0-deployment-plan-{}.json", r.numerical_id)),
            &raw,
            held,
            &mut credit,
        )?;
        let reference = original::deployment_child(
            &execution,
            "reference",
            serde_json::json!({"case_plan":actual_plan,
            "checkpoint":checkpoint,"checkpoint_metadata":metadata,"initialization_evidence":evidence,
            "export_manifest":original::pin_value(&export["outputs"]["export_manifest"])?,
            "export_graphs":original::graph_pin_values(&export["outputs"]["export_graphs"])?,
            "cuda_control_inventory":original::pin_value(&export["outputs"]["cuda_control_inventory"])?}),
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
        let references = original::hold_reference_outputs(
            &execution,
            &reference,
            &plan,
            running.held.as_mut().expect("Deployment reference custody"),
            &mut credit,
        )?;
        let capture = bootstrap::linux::materialize_native(
            &native,
            &scope,
            &i.private_cwd,
            &r,
            &actual_plan,
            until,
            cleanup_until,
        )?;
        original::execute_native(
            &execution,
            capture,
            &plan,
            started,
            until,
            cleanup_until,
            cancel,
            running,
            &mut credit,
        )?;
        finalize(
            &d,
            &r,
            &i,
            collection,
            &mut scope,
            &plan,
            &plan_pin,
            &references,
            cleanup_until,
            cancel,
            seal.is_some(),
            running,
            &mut credit,
        )
    }

    fn prepared_scope(
        original_scope: &PalsNumericalScopeV5,
        i: &G0NumericalPreInitScopeV5,
        r: &G0NumericalBootstrapRegistrationV5,
        outputs: &Value,
        mut assets: Vec<rz_experiments::PalsStartupAssetV5>,
        held: &mut LinuxHeld,
        credit: &mut ReadCredit<'_>,
    ) -> Result<PalsNumericalScopeV5, ArenaError> {
        let manifest = original::hold_observed(
            &outputs["export_manifest"],
            &r.export_directory.join("export.json"),
            256 << 10,
            held,
            credit,
        )?;
        assets.push(bootstrap::linux::asset(
            &manifest,
            &i.private_cwd,
            Purpose::ExportManifest,
            &i.model.origin,
        )?);
        let graphs = outputs["export_graphs"]
            .as_array()
            .ok_or_else(|| invalid("Deployment export graphs absent"))?;
        check(
            graphs.len() == 2,
            "Deployment must omit the validator graph",
        )?;
        for (row, (role, file, purpose)) in graphs.iter().zip([
            ("public", "public_memory.onnx", Purpose::PublicGraph),
            ("shared_pc", "shared_pc_if.onnx", Purpose::SharedPcGraph),
        ]) {
            check(
                row["role"] == role && row["file"] == file,
                "Deployment exact ordered P/C graph roster",
            )?;
            let mut value = row.clone();
            value.as_object_mut().expect("graph object").remove("role");
            value.as_object_mut().expect("graph object").remove("file");
            let pin = original::hold_observed(
                &value,
                &r.export_directory.join(file),
                1 << 30,
                held,
                credit,
            )?;
            assets.push(bootstrap::linux::asset(
                &pin,
                &i.private_cwd,
                purpose,
                &i.model.origin,
            )?);
        }
        let inventory = original::hold_observed(
            &outputs["cuda_control_inventory"],
            &r.export_directory.join("cuda-control-inventory.json"),
            MAX_JSON,
            held,
            credit,
        )?;
        if original_scope.model.backend == rz_experiments::PalsModelBackendV3::OrtCuda {
            assets.push(bootstrap::linux::asset(
                &inventory,
                &i.private_cwd,
                Purpose::CudaControlInventory,
                &i.model.origin,
            )?);
        }
        let mut model = original_scope.model.clone();
        model.exported_roles = BTreeSet::from([
            rz_experiments::PalsRoleV3::Proposer,
            rz_experiments::PalsRoleV3::Critic,
        ]);
        let scope = PalsNumericalScopeV5 {
            mode: PalsEngineModeV5::Deployment,
            model,
            source_commit: original_scope.source_commit.clone(),
            source_components: original_scope.source_components.clone(),
            assets,
            training_executed: false,
        };
        scope.validate_pre_acceptance()?;
        check(
            scope.model.weights == original_scope.model.weights,
            "Deployment cannot replace original G0 weights",
        )?;
        Ok(scope)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "One original deployment window retains collection and new raw owners"
    )]
    fn finalize(
        d: &G0DeploymentNumericalRegistrationV5,
        r: &G0NumericalBootstrapRegistrationV5,
        i: &G0NumericalPreInitScopeV5,
        collection: &VerifiedG0BootstrapNumericalV5,
        scope: &mut PalsNumericalScopeV5,
        plan: &Value,
        collection_plan: &Pin,
        references: &BTreeMap<String, Pin>,
        cleanup_until: Instant,
        cancel: &AtomicBool,
        sealed_registration_joined: bool,
        running: &mut Running,
        credit: &mut ReadCredit<'_>,
    ) -> Result<Completed, ArenaError> {
        check(
            !cancel.load(Ordering::Acquire),
            "Deployment numerical admission cancelled",
        )?;
        let preserve_cancel = AtomicBool::new(false);
        let mut cleanup = ReadCredit {
            remaining: credit.remaining,
            maximum: credit.maximum,
            until: cleanup_until,
            cancel: &preserve_cancel,
            account: Arc::clone(&credit.account),
        };
        let result = (|| {
            let credit = &mut cleanup;
            let held = running.held.as_mut().expect("Deployment final custody");
            let mut pairs = Vec::new();
            let mut total = 0u64;
            for group in plan["groups"]
                .as_array()
                .ok_or_else(|| invalid("Deployment final groups"))?
            {
                let id = group["id"]
                    .as_str()
                    .ok_or_else(|| invalid("Deployment group ID"))?;
                let reference = references
                    .get(&format!("reference.{id}.json"))
                    .ok_or_else(|| invalid("Deployment original reference missing"))?;
                let native = numerical::observe_raw(
                    &r.native_output_root.join(format!("native-{id}.json")),
                    credit,
                )?;
                let mut artifacts = Vec::new();
                for (label, pin) in [("reference", reference), ("native", &native)] {
                    total = total
                        .checked_add(pin.bytes)
                        .filter(|n| *n <= 64 << 20)
                        .ok_or_else(|| invalid("Deployment total raw payload cap"))?;
                    let original_file = common::open_file(&pin.path)?;
                    let raw = common::verify_bytes(&original_file, pin, true, credit)?;
                    numerical::ensure_output_credit(
                        &held.staging,
                        r.max_output_bytes,
                        raw.len() as u64,
                        cleanup_until,
                        &preserve_cancel,
                    )?;
                    let copy = common::write_original_owned_new(
                        &i.private_cwd.join(format!(
                            "inputs/g0-deployment-{}-{label}-{id}.json",
                            r.numerical_id
                        )),
                        &raw,
                        held,
                        credit,
                    )?;
                    let asset = bootstrap::linux::asset(
                        &copy,
                        &i.private_cwd,
                        Purpose::NumericalRawEvidence,
                        &i.model.origin,
                    )?;
                    artifacts.push(asset.artifact.clone());
                    scope.assets.push(asset);
                }
                let (file, _) = common::read_observed_output(native, MAX_JSON, credit, false)?;
                held.checkpoints.push(file);
                pairs.push(rz_experiments::PalsNumericalRunPairV5 {
                    reference: artifacts.remove(0),
                    native: artifacts.remove(0),
                });
            }
            let acceptance = rz_experiments::PalsNumericalAcceptanceV5 {
                schema_version: 5,
                domain: rz_experiments::PALS_NUMERICAL_ACCEPTANCE_V5_DOMAIN.into(),
                bindings: rz_experiments::numerical_scope_bindings_v5(scope)?,
                invocation: rz_experiments::PalsNumericalInvocationV5::Fresh,
                batch: 1,
                runs: pairs,
            };
            let raw = serde_json::to_vec(&acceptance).map_err(|e| invalid(e.to_string()))?;
            check(raw.len() <= 256 << 10, "Deployment bounded acceptance")?;
            numerical::ensure_output_credit(
                &held.staging,
                r.max_output_bytes,
                raw.len() as u64,
                cleanup_until,
                &preserve_cancel,
            )?;
            let acceptance_pin =
                common::write_original_owned_new(&d.acceptance_path, &raw, held, credit)?;
            scope.assets.push(bootstrap::linux::asset(
                &acceptance_pin,
                &i.private_cwd,
                Purpose::NumericalAcceptance,
                &i.model.origin,
            )?);
            let mut read = |a: &ArtifactRef, n: usize| {
                original::read_assets(&scope.assets, &i.private_cwd, held, a, n, credit)
            };
            let admission = rz_experiments::validate_pals_numerical_scope_acceptance_v5(
                scope, &raw, &mut read,
            )?;
            let export = read(&scope.asset(Purpose::ExportManifest)?.artifact, 256 << 10)?;
            let domain =
                rz_experiments::validate_pals_scope_artifact_domain_v5(scope, &export, &mut read)?;
            check(
                domain.artifact_domain == PalsArtifactDomainV5::Untrained
                    && !domain.tensor_optimizer_content_verified
                    && !domain.arena_accepted,
                "Deployment numerical cannot issue startup or training/game authority",
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
                "Deployment exact original export/reference/native chain",
            )?;
            let parent = serde_json::json!({"schema":"rz-pals-original-g0-deployment-numerical-parent/1","numerical_id":d.numerical_id,
                "launch_registration":running.registration_pin,"collection_acceptance":collection.acceptance_pin(),"collection_plan":collection_plan,
                "scope":scope,"admission":admission,"artifact_domain":domain,"acceptance":acceptance_pin,
                "trained":false,"optimizer_updates_executed":0,"actual_initialization_executed":false,
                "numerical_only":true,"startup_admission_not_issued":true,"startup_config_or_lock_issued":false,
                "training_authority_issued":false,"collection_authority_issued":false,"arena_accepted":false,
                "live_collection_owner_retained":true,"same_original_checkpoint":true,"initial_model_namespace_epoch":1,
                "original_sealed_registration_inode_joined":sealed_registration_joined,
                "original_collection_clock_not_extended":true,"deployment_stage_has_separate_original_clock_and_read_credit":true,
                "resources_declared":i.resources,"resources_applied":"unknown","staging_cap_bytes":r.max_output_bytes,
                "managed_staging_includes_retained_collection_artifacts":true,"groups":12,"cases":100,
                "processes":running.processes.iter().map(|p|serde_json::json!({"receipt":p.process().receipt,"transport":p.observation()})).collect::<Vec<_>>(),
                "child_receipts":running.receipts,"charged_read_bytes":running.account.charged.load(Ordering::Acquire),
                "charged_read_bytes_scope":"before_parent_receipt_publication","parent_receipt_readback_counted_in_live_owner":true,
                "read_usage_uncertain":running.account.uncertain.load(Ordering::Acquire),"physical_completion":"confirmed_native_join","automatic_retry":false});
            let raw = serde_json::to_vec(&parent).map_err(|e| invalid(e.to_string()))?;
            check(
                raw.len() as u64 <= MAX_JSON,
                "Deployment bounded parent receipt",
            )?;
            numerical::ensure_output_credit(
                &held.staging,
                r.max_output_bytes,
                raw.len() as u64,
                cleanup_until,
                &preserve_cancel,
            )?;
            common::write_original_owned_new(&d.parent_receipt_path, &raw, held, credit)?;
            common::check_original_g0_issue(credit, cancel)?;
            Ok((scope.clone(), admission, acceptance_pin, parent))
        })();
        credit.remaining = cleanup.remaining;
        result
    }
    #[cfg(test)]
    mod seal_join_tests {
        use super::*;
        use std::{
            io::Write,
            os::unix::fs::{DirBuilderExt, OpenOptionsExt},
            sync::atomic::AtomicU64,
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        struct Fixture(PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                for name in ["registration.json", "copy.json", "prior.json"] {
                    let _ = std::fs::remove_file(self.0.join(name));
                }
                let _ = std::fs::remove_dir(&self.0);
            }
        }
        #[test]
        fn sealed_stage_join_refuses_same_bytes_from_another_inode_or_replaced_name() {
            let root = std::env::temp_dir().join(format!(
                "g0-seal-join-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&root)
                .unwrap();
            let fixture = Fixture(root);
            let path = fixture.0.join("registration.json");
            let copy_path = fixture.0.join("copy.json");
            let mut original = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o400)
                .open(&path)
                .unwrap();
            original.write_all(b"g0\n").unwrap();
            original.sync_all().unwrap();
            let mut copy = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o400)
                .open(&copy_path)
                .unwrap();
            copy.write_all(b"g0\n").unwrap();
            copy.sync_all().unwrap();
            // Only inode handoff is under test; this fixture never parses a
            // registration, dispatches a child, or creates an admitted owner.
            let pin = Pin {
                path: path.clone(),
                sha256: "a".repeat(64),
                bytes: 3,
            };
            check_sealed_registration_join(&pin, &common::open_file(&path).unwrap(), &original)
                .expect("the exact sealed inode may hand off to a new read FD");
            assert!(check_sealed_registration_join(&pin, &copy, &original).is_err());
            std::fs::rename(&path, fixture.0.join("prior.json")).unwrap();
            std::fs::rename(&copy_path, &path).unwrap();
            assert!(
                check_sealed_registration_join(&pin, &common::open_file(&path).unwrap(), &original)
                    .is_err()
            );
        }

        #[test]
        fn intent_preflight_checks_sha_static_shape_and_cap_without_a_collection_owner() {
            use sha2::{Digest, Sha256};
            let root = std::env::temp_dir().join(format!(
                "g0-preflight-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&root)
                .unwrap();
            let fixture = Fixture(root);
            let path = fixture.0.join("registration.json");
            let intent = G0DeploymentNumericalIntentV5 {
                schema: G0_DEPLOYMENT_NUMERICAL_INTENT_V5.into(),
                numerical_id: "preflight_metadata_only".into(),
                registration_path: fixture.0.join("new-registration.json"),
                export_directory: fixture.0.join("export"),
                reference_output_root: fixture.0.join("reference"),
                native_output_root: fixture.0.join("native"),
                export_receipt_path: fixture.0.join("export-receipt.json"),
                reference_receipt_path: fixture.0.join("reference-receipt.json"),
                acceptance_path: fixture.0.join("acceptance.json"),
                parent_receipt_path: fixture.0.join("parent.json"),
                max_wall_ns: 1_000_000_000,
                max_output_bytes: MAX_STAGING,
                max_read_bytes: 4 * MAX_STAGING,
            };
            let raw = serde_json::to_vec(&intent).unwrap();
            let sha = |bytes: &[u8]| {
                Sha256::digest(bytes)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            };
            let write = |path: &Path, bytes: &[u8]| {
                let mut file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .mode(0o400)
                    .open(path)
                    .unwrap();
                file.write_all(bytes).unwrap();
                file.sync_all().unwrap();
                file
            };
            let original = write(&path, &raw);
            let cancel = Arc::new(AtomicBool::new(false));
            let mut changed_sha = sha(&raw);
            let replacement = if changed_sha.starts_with('a') {
                "b"
            } else {
                "a"
            };
            changed_sha.replace_range(..1, replacement);
            let wrong_sha = match preflight_g0_deployment_intent_original_v5(
                &path,
                &changed_sha,
                Arc::clone(&cancel),
            ) {
                Ok(_) => panic!("wrong intent SHA cannot preflight"),
                Err(failure) => failure,
            };
            assert!(wrong_sha._intent.is_some());
            assert!(wrong_sha.charged_read_bytes > 0);
            let preflight =
                preflight_g0_deployment_intent_original_v5(&path, &sha(&raw), Arc::clone(&cancel))
                    .unwrap();
            assert_eq!(preflight.pin().bytes, raw.len() as u64);
            assert_eq!(
                preflight.observation()["status"],
                "intent_preflighted_metadata_only"
            );
            assert_eq!(preflight.observation()["numerical_authority_issued"], false);
            check_original_file_join(
                preflight.pin(),
                &original,
                &preflight.intent,
                "fixture original intent join",
            )
            .unwrap();
            let bad_path = fixture.0.join("copy.json");
            let bad_raw = b"{\"schema\":\"wrong-static-domain\"}";
            let _bad = write(&bad_path, bad_raw);
            let malformed = match preflight_g0_deployment_intent_original_v5(
                &bad_path,
                &sha(bad_raw),
                Arc::clone(&cancel),
            ) {
                Ok(_) => panic!("invalid static intent cannot preflight"),
                Err(failure) => failure,
            };
            assert!(malformed._intent.is_some());
            let large_path = fixture.0.join("prior.json");
            let _large = write(&large_path, &vec![b' '; MAX_INTENT as usize + 1]);
            let oversized = match preflight_g0_deployment_intent_original_v5(
                &large_path,
                &"a".repeat(64),
                cancel,
            ) {
                Ok(_) => panic!("oversized metadata cannot preflight"),
                Err(failure) => failure,
            };
            assert!(oversized._intent.is_some());
            assert_eq!(oversized.charged_read_bytes, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_experiments::{PalsModelBackendV3, PalsNumericalBindingsV5, PalsPrecisionV3};
    use std::collections::BTreeMap;

    fn bindings() -> PalsNumericalBindingsV5 {
        PalsNumericalBindingsV5 {
            model_profile: "virtual_review_v1".into(),
            model_schema: rz_experiments::PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: rz_experiments::PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: rz_experiments::PALS_V5_ENCODING_SCHEMA.into(),
            model_semantic_sha256: "a".repeat(64),
            encoding_sha256: "b".repeat(64),
            model_epoch: 1,
            weights_sha256: "c".repeat(64),
            export_manifest_sha256: "d".repeat(64),
            graphs: BTreeMap::from([
                ("public".into(), "e".repeat(64)),
                ("shared_pc".into(), "f".repeat(64)),
                ("validator".into(), "0".repeat(64)),
            ]),
            runtime_library_sha256: "1".repeat(64),
            cuda_libraries: BTreeSet::new(),
            cuda_manifest_sha256: None,
            backend: PalsModelBackendV3::OrtCpu,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            source_components: [
                "model_native",
                "model_encoding",
                "nn_runtime",
                "model_reference",
                "model_export",
            ]
            .into_iter()
            .map(|k| (k.into(), "2".repeat(64)))
            .collect(),
        }
    }
    // Typed input data only: no filesystem/process/framework or model execution.
    fn original_plan() -> Value {
        use rz_eval::pals_model::{
            PalsCandidateToken, PalsRole,
            virtual_review::{
                PalsVirtualReviewInput, ReviewAnchorInput, ReviewQueryInput, ReviewQueryPurpose,
            },
        };
        let mut groups = Vec::new();
        for (name, role) in [
            ("proposer", PalsRole::Proposer),
            ("critic", PalsRole::Critic),
            ("validator", PalsRole::Validator),
        ] {
            for bucket in [8, 16, 32, 64, 128, 256] {
                let mut rows = Vec::new();
                for anchors in 1..=4 {
                    for white in [false, true] {
                        let mut metadata = [0.; 16];
                        metadata[0] = f32::from(u8::from(white));
                        let candidate = PalsCandidateToken {
                            from: 1,
                            to: 18,
                            promotion: 0,
                        };
                        let input = PalsVirtualReviewInput {
                            schema: rz_eval::pals_model::virtual_review::INPUT_SCHEMA.into(),
                            role,
                            anchors: vec![
                                ReviewAnchorInput {
                                    board: vec![0; 64],
                                    metadata
                                };
                                anchors
                            ],
                            records: Vec::new(),
                            query: ReviewQueryInput {
                                purpose: if role == PalsRole::Validator {
                                    ReviewQueryPurpose::TaskPolicy
                                } else {
                                    ReviewQueryPurpose::ReplyPolicy
                                },
                                perspective: if white { 1 } else { -1 },
                                prefix: Vec::new(),
                                proposal: vec![candidate; bucket],
                                counter: Vec::new(),
                                counter_present: false,
                                anchors: [0, 0, 0],
                            },
                            candidates: vec![candidate],
                            divergences: Vec::new(),
                            token_budget: 512,
                        };
                        rows.push(serde_json::json!({"id":format!("{name}_L{bucket}_A{anchors}_W{}", u8::from(white)),
                            "input":{"model_input_kind":"virtual_review","input":input}}));
                    }
                }
                if bucket == 8 {
                    for label in ["empty", "order_reversed"] {
                        let mut extra = rows[0].clone();
                        extra["id"] = format!("{name}_{label}").into();
                        rows.push(extra);
                    }
                }
                groups.push(serde_json::json!({"id":format!("{name}_L{bucket}"),"role":name,"line_bucket":bucket,"cases":rows}));
            }
        }
        serde_json::json!({"schema_version":5,"domain":"rz-pals-numerical-case-plan-v5/1","invocation":"fresh","batch":1,"bindings":bindings(),"groups":groups})
    }
    #[test]
    fn deployment_preserves_original_pc_inputs_and_order_without_weakening_collection() {
        let collection = original_plan();
        let before = serde_json::to_vec(&collection).unwrap();
        let projected = deployment_groups(&collection, &bindings())
            .expect("actual full Collection input coverage");
        let expected = collection["groups"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|g| g["role"] != "validator")
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(projected, Value::Array(expected));
        assert_eq!(projected.as_array().unwrap().len(), 12);
        assert_eq!(
            projected
                .as_array()
                .unwrap()
                .iter()
                .map(|g| g["cases"].as_array().unwrap().len())
                .sum::<usize>(),
            100
        );
        assert_eq!(serde_json::to_vec(&collection).unwrap(), before);
        let mut incomplete = collection.clone();
        incomplete["groups"].as_array_mut().unwrap().pop();
        assert!(deployment_groups(&incomplete, &bindings()).is_err());
        let mut wrong_role = collection;
        wrong_role["groups"][0]["cases"][0]["input"]["input"]["role"] = "validator".into();
        assert!(deployment_groups(&wrong_role, &bindings()).is_err());
    }
    #[test]
    fn deployment_rebinding_never_substitutes_weights_runtime_source_or_epoch() {
        let original = bindings();
        let mut pc = original.clone();
        pc.graphs.remove("validator");
        pc.export_manifest_sha256 = "3".repeat(64);
        validate_deployment_rebinding(&original, &pc)
            .expect("same original checkpoint with actual P/C export");
        for mutation in 0..6 {
            let mut changed = pc.clone();
            match mutation {
                0 => changed.weights_sha256 = "4".repeat(64),
                1 => changed.runtime_library_sha256 = "4".repeat(64),
                2 => changed.model_epoch += 1,
                3 => {
                    changed.source_components.clear();
                }
                4 => changed.encoding_sha256 = "4".repeat(64),
                _ => {
                    changed.graphs.insert("validator".into(), "4".repeat(64));
                }
            }
            assert!(validate_deployment_rebinding(&original, &changed).is_err());
        }
    }
    #[test]
    fn deployment_registration_is_not_a_checkpoint_or_owner_decoder() {
        let root = std::env::temp_dir().join("g0-deployment-metadata-fixture");
        let registration = G0DeploymentNumericalRegistrationV5 {
            schema: G0_DEPLOYMENT_NUMERICAL_LAUNCH_V5.into(),
            numerical_id: "metadata_only".into(),
            collection_acceptance: Pin {
                path: root.join("original.json"),
                sha256: "a".repeat(64),
                bytes: 1,
            },
            export_directory: root.join("export"),
            reference_output_root: root.join("reference"),
            native_output_root: root.join("native"),
            export_receipt_path: root.join("export-receipt.json"),
            reference_receipt_path: root.join("reference-receipt.json"),
            acceptance_path: root.join("acceptance.json"),
            parent_receipt_path: root.join("parent.json"),
            max_wall_ns: 1_000_000_000,
            max_output_bytes: MAX_STAGING,
            max_read_bytes: 4 * MAX_STAGING,
        };
        #[cfg(not(windows))]
        registration
            .validate()
            .expect("metadata-only finite disjoint POSIX names");
        #[cfg(windows)]
        assert!(
            registration.validate().is_err(),
            "Windows path prefixes cannot enter the Linux original stage"
        );
        for field in ["checkpoint", "collection_owner", "scope", "trained"] {
            let mut value = serde_json::to_value(&registration).unwrap();
            value[field] = Value::Null;
            assert!(serde_json::from_value::<G0DeploymentNumericalRegistrationV5>(value).is_err());
        }
        let mut alias = registration.clone();
        alias.native_output_root = alias.export_directory.clone();
        assert!(alias.validate().is_err());
        let mut unbounded = registration;
        unbounded.max_output_bytes = MAX_STAGING + 1;
        assert!(unbounded.validate().is_err());
    }

    fn intent() -> G0DeploymentNumericalIntentV5 {
        let root = std::env::temp_dir().join("g0-deployment-intent-metadata-fixture");
        G0DeploymentNumericalIntentV5 {
            schema: G0_DEPLOYMENT_NUMERICAL_INTENT_V5.into(),
            numerical_id: "metadata_only".into(),
            registration_path: root.join("deployment-registration.json"),
            export_directory: root.join("export"),
            reference_output_root: root.join("reference"),
            native_output_root: root.join("native"),
            export_receipt_path: root.join("export-receipt.json"),
            reference_receipt_path: root.join("reference-receipt.json"),
            acceptance_path: root.join("acceptance.json"),
            parent_receipt_path: root.join("parent.json"),
            max_wall_ns: 1_000_000_000,
            max_output_bytes: MAX_STAGING,
            max_read_bytes: 4 * MAX_STAGING,
        }
    }
    #[test]
    fn deployment_intent_contains_no_future_acceptance_or_owner_and_injects_only_actual_pin() {
        let intent = intent();
        let raw = serde_json::to_value(&intent).unwrap();
        assert!(raw.get("collection_acceptance").is_none());
        for field in [
            "collection_acceptance",
            "collection_owner",
            "weights",
            "numerical_admission",
        ] {
            let mut changed = raw.clone();
            changed[field] = Value::Null;
            assert!(serde_json::from_value::<G0DeploymentNumericalIntentV5>(changed).is_err());
        }
        // This is pure registration metadata; no fixture can construct a live
        // Collection or sealed owner, start a child or issue numerical authority.
        let observed = Pin {
            path: std::env::temp_dir().join("observed-acceptance.json"),
            sha256: "a".repeat(64),
            bytes: 1,
        };
        let actual = intent.actual_registration(&observed);
        assert_eq!(actual.collection_acceptance, observed);
        assert_eq!(actual.export_directory, intent.export_directory);
        assert_eq!(actual.max_wall_ns, intent.max_wall_ns);
        assert_eq!(actual.max_read_bytes, intent.max_read_bytes);
        assert_eq!(actual.schema, G0_DEPLOYMENT_NUMERICAL_LAUNCH_V5);
        assert_ne!(actual.schema, intent.schema);
    }
    #[test]
    fn deployment_intent_refuses_registration_alias_or_unbounded_predeclared_work() {
        let original = intent();
        #[cfg(not(windows))]
        original
            .validate()
            .expect("bounded disjoint POSIX metadata intent");
        #[cfg(windows)]
        assert!(original.validate().is_err());
        let mut alias = original.clone();
        alias.registration_path = alias.parent_receipt_path.clone();
        assert!(alias.validate().is_err());
        let mut contained = original.clone();
        contained.registration_path = contained.export_directory.join("registration.json");
        assert!(contained.validate().is_err());
        let mut unbounded = original;
        unbounded.max_wall_ns = 0;
        assert!(unbounded.validate().is_err());
    }
}
