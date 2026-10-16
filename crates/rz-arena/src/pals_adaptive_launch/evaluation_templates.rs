//! Original-owner final evaluation preparation, without executing a model/game.
//! The preregistered System templates are pending G1 declarations, never results.
//! Only held actual G1 exporter/checkpoint and independent numerical owners may
//! replace their explicitly registered future model fields.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]
use super::evaluation_series::registration_pair;
#[cfg(target_os = "linux")]
use super::evaluation_series::validate_series;
use super::*;
#[cfg(target_os = "linux")]
use crate::pals_collect_v3::training_process::{self as io, ReadAccounting, ReadCredit};
use crate::pals_collect_v3::{
    GenerationTrainingFilePinV3, VerifiedG1NumericalV3, VerifiedGenerationExportV3,
};
use std::{
    fs::File,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

pub const PALS_G1_EVALUATION_TEMPLATES_V5_DOMAIN: &str = "rz-pals-g1-evaluation-templates-v5/1";
const MAX_METADATA: u64 = 16 << 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsG1EvaluationTemplatesV5 {
    pub schema: String,
    pub managed_root: PathBuf,
    pub source_root: PathBuf,
    pub private_cwd: PathBuf,
    pub templates: Vec<PalsArenaLaunchV5>,
    pub input_pins: Vec<GenerationTrainingFilePinV3>,
}
struct Held {
    pin: GenerationTrainingFilePinV3,
    file: File,
}
pub struct RegisteredPalsEvaluationTemplatesV5 {
    registration: Arc<PalsEvaluationOpeningRegistrationV5>,
    template_pin: GenerationTrainingFilePinV3,
    template_file: File,
    declaration: PalsG1EvaluationTemplatesV5,
    originals: Vec<Held>,
    #[cfg(target_os = "linux")]
    root: cap_std::fs::Dir,
    #[cfg(target_os = "linux")]
    cwd: cap_std::fs::Dir,
    #[cfg(target_os = "linux")]
    source_dir: cap_std::fs::Dir,
}
impl RegisteredPalsEvaluationTemplatesV5 {
    pub fn registration(&self) -> &Arc<PalsEvaluationOpeningRegistrationV5> {
        &self.registration
    }
    pub fn template_pin(&self) -> &GenerationTrainingFilePinV3 {
        &self.template_pin
    }
    pub fn managed_root(&self) -> &Path {
        &self.declaration.managed_root
    }
    pub fn source_root(&self) -> &Path {
        &self.declaration.source_root
    }
    pub fn private_cwd(&self) -> &Path {
        &self.declaration.private_cwd
    }
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        std::iter::once((&self.template_pin, &self.template_file))
            .chain(self.originals.iter().map(|h| (&h.pin, &h.file)))
    }
}
pub struct PreparedG1EvaluationStartupV5 {
    original: Arc<RegisteredPalsEvaluationTemplatesV5>,
    g1: Arc<VerifiedGenerationExportV3>,
    config: PalsAdaptiveStartupConfigV5,
    lock: PalsInputLockV5,
    files: Vec<Held>,
}
impl PreparedG1EvaluationStartupV5 {
    pub fn original(&self) -> &Arc<RegisteredPalsEvaluationTemplatesV5> {
        &self.original
    }
    pub fn g1(&self) -> &Arc<VerifiedGenerationExportV3> {
        &self.g1
    }
    pub fn config(&self) -> &PalsAdaptiveStartupConfigV5 {
        &self.config
    }
    pub fn lock(&self) -> &PalsInputLockV5 {
        &self.lock
    }
    pub fn managed_root(&self) -> &Path {
        self.original.managed_root()
    }
    pub fn source_root(&self) -> &Path {
        self.original.source_root()
    }
    pub fn private_cwd(&self) -> &Path {
        self.original.private_cwd()
    }
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.files.iter().map(|h| (&h.pin, &h.file))
    }
    #[cfg(target_os = "linux")]
    pub fn private_cwd_file(&self) -> Result<File, ArenaError> {
        io::directory_file(&self.original.cwd)
    }
}
pub struct PreparedG1EvaluationLaunchesV5 {
    prepared: Arc<PreparedG1EvaluationStartupV5>,
    numerical: Arc<VerifiedG1NumericalV3>,
    launches: Vec<LockedPalsArenaLaunchV5>,
    files: Vec<Held>,
    serial_evaluation_issued: AtomicBool,
}
impl PreparedG1EvaluationLaunchesV5 {
    pub fn prepared(&self) -> &Arc<PreparedG1EvaluationStartupV5> {
        &self.prepared
    }
    pub fn numerical(&self) -> &Arc<VerifiedG1NumericalV3> {
        &self.numerical
    }
    pub fn launches(&self) -> &[LockedPalsArenaLaunchV5] {
        &self.launches
    }
    pub fn source_root(&self) -> &Path {
        self.prepared.source_root()
    }
    pub fn private_cwd(&self) -> &Path {
        self.prepared.private_cwd()
    }
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.files.iter().map(|h| (&h.pin, &h.file))
    }
    pub(super) fn claim_serial_evaluation(&self) -> Result<(), ArenaError> {
        claim_serial_once(&self.serial_evaluation_issued)
    }
}
fn claim_serial_once(issued: &AtomicBool) -> Result<(), ArenaError> {
    require(
        issued
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok(),
        "the original final32 owner has already issued its one registered serial evaluation",
    )
}
/// Partial original files, root capabilities and actual exporter/numerical
/// ownership survive every preparation error. This is not retry authority.
pub struct G1EvaluationPreparationFailureV5 {
    pub error: ArenaError,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    registration: Arc<PalsEvaluationOpeningRegistrationV5>,
    original: Option<Arc<RegisteredPalsEvaluationTemplatesV5>>,
    g1: Option<Arc<VerifiedGenerationExportV3>>,
    prepared: Option<Arc<PreparedG1EvaluationStartupV5>>,
    numerical: Option<Arc<VerifiedG1NumericalV3>>,
    files: Vec<Held>,
    partial_files: Vec<File>,
    #[cfg(target_os = "linux")]
    directories: Vec<cap_std::fs::Dir>,
}
impl std::fmt::Debug for G1EvaluationPreparationFailureV5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G1EvaluationPreparationFailureV5")
            .field("error", &self.error)
            .field("charged_read_bytes", &self.charged_read_bytes)
            .field("read_usage_uncertain", &self.read_usage_uncertain)
            .field("retained_files", &self.files.len())
            .field("partial_files", &self.partial_files.len())
            .field("actual_export_retained", &self.g1.is_some())
            .field("numerical_retained", &self.numerical.is_some())
            .field("registration_id", &self.registration.registration_id())
            .field("original_templates_retained", &self.original.is_some())
            .field("prepared_startup_retained", &self.prepared.is_some())
            .finish_non_exhaustive()
    }
}
impl G1EvaluationPreparationFailureV5 {
    pub fn registration(&self) -> &Arc<PalsEvaluationOpeningRegistrationV5> {
        &self.registration
    }
    pub fn original(&self) -> Option<&Arc<RegisteredPalsEvaluationTemplatesV5>> {
        self.original.as_ref()
    }
    pub fn g1(&self) -> Option<&Arc<VerifiedGenerationExportV3>> {
        self.g1.as_ref()
    }
    pub fn prepared(&self) -> Option<&Arc<PreparedG1EvaluationStartupV5>> {
        self.prepared.as_ref()
    }
    pub fn numerical(&self) -> Option<&Arc<VerifiedG1NumericalV3>> {
        self.numerical.as_ref()
    }
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.files.iter().map(|held| (&held.pin, &held.file))
    }
    pub fn partial_files(&self) -> &[File] {
        &self.partial_files
    }
    #[cfg(target_os = "linux")]
    pub fn retained_directory_count(&self) -> usize {
        self.directories.len()
    }
}
fn failure(
    registration: Arc<PalsEvaluationOpeningRegistrationV5>,
    error: ArenaError,
) -> G1EvaluationPreparationFailureV5 {
    G1EvaluationPreparationFailureV5 {
        error,
        charged_read_bytes: 0,
        read_usage_uncertain: false,
        registration,
        original: None,
        g1: None,
        prepared: None,
        numerical: None,
        files: Vec::new(),
        partial_files: Vec::new(),
        #[cfg(target_os = "linux")]
        directories: Vec::new(),
    }
}
fn same_model_condition(g0: &PalsEngineV5, g1: &PalsEngineV5) -> bool {
    let mut left = g0.clone();
    left.id = g1.id.clone();
    left.model.weights = g1.model.weights.clone();
    left.model.provenance = g1.model.provenance.clone();
    left.model.artifact_domain = g1.model.artifact_domain;
    left.model.checkpoint_domain = g1.model.checkpoint_domain.clone();
    left.model.epoch = g1.model.epoch;
    left == *g1
}
fn validate_templates(
    declaration: &PalsG1EvaluationTemplatesV5,
    opening: &PalsEvaluationOpeningRegistrationV5,
) -> Result<(), ArenaError> {
    require(
        declaration.schema == PALS_G1_EVALUATION_TEMPLATES_V5_DOMAIN
            && declaration.templates.len() == 32
            && declaration.input_pins.len() <= 4096,
        "exact precollection final template domain/count/input bound",
    )?;
    let first = &declaration.templates[0];
    let base = &first.semantic_lock.manifest;
    let mut runs = BTreeSet::new();
    let mut windows = BTreeSet::new();
    for (index, t) in declaration.templates.iter().enumerate() {
        t.validate()?;
        let m = &t.semantic_lock.manifest;
        let mut pending = m.engines[1].clone();
        pending.id = m.engines[0].id.clone();
        require(
            m.comparison == PalsComparisonV5::System
                && m.purpose == PalsRunPurposeV5::EvaluationPair
                && m.generation.as_ref() == Some(&PalsGenerationRecipeV5::default())
                && m.engines[0] == *opening.g0_deployment()
                && pending == m.engines[0]
                && m.engines[0].id != m.engines[1].id,
            "future G1 is a pending same-G0 model template, not preclaimed weights or execution evidence",
        )?;
        let registered = &opening.openings()[index];
        require(
            t.opening_artifact == registered.opening_artifact
                && m.pair.as_ref()
                    == Some(&registration_pair(
                        registered,
                        &m.engines[0].id,
                        &m.engines[1].id,
                    )),
            "registered final opening/color/Fischer/seed differs",
        )?;
        let mut window = m.window.clone();
        window.window_id = base.window.window_id.clone();
        require(
            runs.insert(&m.run_id)
                && windows.insert(&m.window.window_id)
                && m.engines == base.engines
                && m.source_pins == base.source_pins
                && m.rules_profile == base.rules_profile
                && m.generation == base.generation
                && m.window.wall_time_max_ms <= 20 * 60 * 1000
                && m.window.cleanup_max_ms == 30_000
                && window == base.window
                && t.runner == first.runner
                && t.budget == first.budget
                && t.storage == first.storage
                && t.source_files == first.source_files,
            "preregistered final cohort changed source/resources/runner/options/window or reused identities",
        )?;
        for (candidate, baseline) in t.endpoints.iter().zip(&first.endpoints) {
            let mut config = candidate.configuration.clone();
            config.semantic_lock_sha256 = baseline.configuration.semantic_lock_sha256.clone();
            require(
                config == baseline.configuration
                    && candidate.expected_uci_name == baseline.expected_uci_name
                    && candidate.version == baseline.version
                    && candidate.handshake_max_ms == baseline.handshake_max_ms
                    && candidate.environment == baseline.environment
                    && candidate.cuda_bundle == baseline.cuda_bundle
                    && candidate.startup_snapshot_path == baseline.startup_snapshot_path,
                "preregistered cohort native endpoint settings differ",
            )?;
        }
    }
    Ok(())
}

pub fn register_pals_evaluation_templates_v5(
    pin: GenerationTrainingFilePinV3,
    registration: Arc<PalsEvaluationOpeningRegistrationV5>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<RegisteredPalsEvaluationTemplatesV5, Box<G1EvaluationPreparationFailureV5>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pin, until, cancel);
        Err(Box::new(failure(
            registration,
            invalid("original evaluation templates require Linux descriptors"),
        )))
    }
    #[cfg(target_os = "linux")]
    {
        let mut failed = failure(
            Arc::clone(&registration),
            invalid("evaluation template preparation incomplete"),
        );
        let account = ReadAccounting::new();
        let mut credit = read_credit(until, cancel, &account);
        let result=(||->Result<(PalsG1EvaluationTemplatesV5,File,cap_std::fs::Dir,cap_std::fs::Dir,cap_std::fs::Dir),ArenaError>{
            pin.validate(MAX_METADATA)?;let file=io::open_file(&pin.path)?;
            let raw=io::verify_bytes(&file,&pin,true,&mut credit)?;
            let declaration:PalsG1EvaluationTemplatesV5=crate::decode_json(std::str::from_utf8(&raw).map_err(|_|invalid("template UTF8"))?)?;
            failed.files.push(Held{pin:pin.clone(),file:file.try_clone().map_err(|_|invalid("original template descriptor clone"))?});
            validate_templates(&declaration,&registration)?;
            require(declaration.source_root.starts_with(&declaration.managed_root)&&declaration.private_cwd.starts_with(&declaration.source_root)
                &&declaration.private_cwd!=declaration.managed_root&&pin.path.starts_with(&declaration.managed_root),"original final roots outside managed storage")?;
            let root=io::open_directory(&declaration.managed_root)?;crate::pals_collect_v3::selfplay_cli::private_root(&root)?;
            let cwd=io::open_directory(&declaration.private_cwd)?;crate::pals_collect_v3::selfplay_cli::private_root(&cwd)?;
            failed.directories.push(root.try_clone().map_err(|_|invalid("root custody clone"))?);
            failed.directories.push(cwd.try_clone().map_err(|_|invalid("cwd custody clone"))?);
            let source_dir=io::open_directory(&declaration.source_root)?;crate::pals_collect_v3::selfplay_cli::private_root(&source_dir)?;
            failed.directories.push(source_dir.try_clone().map_err(|_|invalid("source root custody clone"))?);
            crate::pals_collect_v3::selfplay_cli::scan_staging(&root,PALS_STAGING_BYTES_V5,until,cancel)?;
            let mut artifacts=BTreeMap::new();
            for a in declaration.templates.iter().flat_map(PalsArenaLaunchV5::declared_artifacts) {
                if let Some(previous)=artifacts.insert(declaration.source_root.join(&a.path),a) { require(previous==a,"contradictory preregistered artifact")?; }
            }
            require(artifacts.len()==declaration.input_pins.len(),"complete original template artifact descriptor set required")?;
            let mut seen=BTreeSet::new();
            for input in &declaration.input_pins {
                input.validate(PALS_STAGING_BYTES_V5)?;let a=artifacts.get(&input.path).ok_or_else(||invalid("unregistered original template descriptor"))?;
                require(seen.insert(&input.path)&&a.sha256==input.sha256&&a.bytes==input.bytes,"template input pin duplicated or differs")?;
                let file=io::open_file(&input.path)?;io::verify_bytes(&file,input,false,&mut credit)?;
                failed.files.push(Held{pin:input.clone(),file});
            }
            crate::pals_collect_v3::selfplay_cli::scan_staging(&root,PALS_STAGING_BYTES_V5,until,cancel)?;
            Ok((declaration,file,root,cwd,source_dir))
        })();
        match result {
            Ok((declaration, template_file, root, cwd, source_dir)) => {
                failed.files.remove(0);
                Ok(RegisteredPalsEvaluationTemplatesV5 {
                    registration,
                    template_pin: pin,
                    template_file,
                    declaration,
                    originals: std::mem::take(&mut failed.files),
                    root,
                    cwd,
                    source_dir,
                })
            }
            Err(error) => {
                set_error(&mut failed, error, &account);
                Err(Box::new(failed))
            }
        }
    }
}

pub fn prepare_original_g1_evaluation_startup_v5(
    original: Arc<RegisteredPalsEvaluationTemplatesV5>,
    g1: Arc<VerifiedGenerationExportV3>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<PreparedG1EvaluationStartupV5, Box<G1EvaluationPreparationFailureV5>> {
    let mut failed = failure(
        Arc::clone(original.registration()),
        invalid("G1 original startup preparation incomplete"),
    );
    failed.original = Some(Arc::clone(&original));
    failed.g1 = Some(Arc::clone(&g1));
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, cancel);
        failed.error = invalid("G1 preparation requires Linux original descriptors");
        Err(Box::new(failed))
    }
    #[cfg(target_os = "linux")]
    {
        let account = ReadAccounting::new();
        let mut credit = read_credit(until, cancel, &account);
        let result = (|| -> Result<(PalsAdaptiveStartupConfigV5, PalsInputLockV5), ArenaError> {
            check_original_roots(&original)?;
            validate_templates(&original.declaration, original.registration())?;
            io::verify_bytes(
                &original.template_file,
                &original.template_pin,
                false,
                &mut credit,
            )?;
            require(
                g1.kind() == crate::pals_collect_v3::GenerationExportKindV3::G1ActualGeneration
                    && g1.semantic_receipt()["eligibility"]
                        == "evaluation_candidate_pending_gpu_acceptance"
                    && g1.semantic_receipt()["numeric_status"] == "not_run"
                    && g1.semantic_receipt()["cuda_status"] == "not_run"
                    && g1.semantic_receipt()["arena_accepted"] == false,
                "only original actual5000 export prepares pending G1 evaluation",
            )?;
            let template = &original.declaration.templates[0];
            let mut config = template.endpoints[1].configuration.clone();
            let mut manifest = template.semantic_lock.manifest.clone();
            let sha = g1.semantic_receipt()["checkpoint"]["sha256"]
                .as_str()
                .ok_or_else(|| invalid("original G1 checkpoint SHA absent"))?;
            let checkpoint = g1
                .trainer()
                .checkpoint_files()
                .find(|(p, _)| {
                    p.sha256 == sha && g1.semantic_receipt()["checkpoint"]["bytes"] == p.bytes
                })
                .ok_or_else(|| invalid("original final checkpoint descriptor missing"))?;
            let receipt_pin = g1.semantic_receipt_file_pin();
            let receipt = g1
                .held_files()
                .find(|(p, _)| *p == receipt_pin)
                .ok_or_else(|| invalid("original semantic receipt descriptor missing"))?;
            let export_pin = g1
                .deployment_manifest_pin()
                .ok_or_else(|| invalid("original V-free export absent"))?;
            let export = g1
                .held_files()
                .find(|(p, _)| *p == export_pin)
                .ok_or_else(|| invalid("original V-free export descriptor absent"))?;
            let trainer_registration = g1.trainer().registration_value()?;
            require(
                trainer_registration["staging_root"].as_str() == original.managed_root().to_str(),
                "G1 preparation must use the original trainer/collector staging root and18GiB owner",
            )?;
            let spec_pin: GenerationTrainingFilePinV3 =
                serde_json::from_value(trainer_registration["spec"].clone())
                    .map_err(|_| invalid("original trainer specification pin"))?;
            let spec = g1
                .trainer()
                .held_input_files()
                .find(|(p, _)| *p == &spec_pin)
                .ok_or_else(|| invalid("original trainer specification descriptor absent"))?;
            let source = config.endpoint.model.weights.source.clone();
            let license = config.endpoint.model.weights.license.clone();
            let mut replacements = Vec::<(
                PalsStartupAssetPurposeV5,
                &GenerationTrainingFilePinV3,
                &File,
            )>::new();
            use PalsStartupAssetPurposeV5 as P;
            replacements.extend([
                (P::Weights, checkpoint.0, checkpoint.1),
                (P::Provenance, receipt.0, receipt.1),
                (P::ExportManifest, export.0, export.1),
                (P::GenerationSpecification, spec.0, spec.1),
            ]);
            for (purpose, name) in [
                (P::PublicGraph, "public_memory.onnx"),
                (P::SharedPcGraph, "shared_pc_if.onnx"),
            ] {
                let path = g1.directory().join("deployment").join(name);
                let actual = g1
                    .held_files()
                    .find(|(p, _)| p.path == path)
                    .ok_or_else(|| {
                        invalid("original compact Deployment graph descriptor absent")
                    })?;
                require(
                    g1.semantic_receipt()["deployment"]["graphs"]
                        .as_array()
                        .is_some_and(|gs| {
                            gs.iter().any(|g| {
                                g["sha256"] == actual.0.sha256 && g["bytes"] == actual.0.bytes
                            })
                        }),
                    "G1 actual graph descriptor differs",
                )?;
                replacements.push((purpose, actual.0, actual.1));
            }
            if config.endpoint.model.backend == PalsModelBackendV3::OrtCuda {
                let pin = g1.deployment_control_inventory_pin().ok_or_else(|| {
                    invalid("actual G1 CUDA inventory was not generated by original export child")
                })?;
                let actual = g1
                    .held_files()
                    .find(|(p, _)| *p == pin)
                    .ok_or_else(|| invalid("actual G1 CUDA inventory descriptor missing"))?;
                replacements.push((P::CudaControlInventory, actual.0, actual.1));
            }
            config.assets.retain(|a| {
                !matches!(
                    a.purpose,
                    P::InitializationEvidence
                        | P::InitializationCheckpointMetadata
                        | P::ProposerGraph
                        | P::CriticGraph
                        | P::ValidatorGraph
                )
            });
            let mut sources = BTreeMap::<String, (&GenerationTrainingFilePinV3, &File)>::new();
            for (purpose, pin, file) in replacements {
                io::verify_bytes(file, pin, false, &mut credit)?;
                let relative = pin
                    .path
                    .strip_prefix(original.source_root())
                    .map_err(|_| invalid("actual G1 asset outside preregistered source root"))?
                    .to_str()
                    .ok_or_else(|| invalid("G1 logical artifact UTF8"))?
                    .replace('\\', "/");
                let leaf = match purpose {
                    P::Weights => "checkpoint.pt",
                    P::Provenance => "actual-generation.json",
                    P::ExportManifest => "export.json",
                    P::GenerationSpecification => "generation-spec.json",
                    P::PublicGraph => "public_memory.onnx",
                    P::SharedPcGraph => "shared_pc_if.onnx",
                    P::CudaControlInventory => "cuda-control-inventory.json",
                    _ => unreachable!(),
                };
                let snapshot_path = format!("inputs/g1/{leaf}");
                let artifact = ArtifactRef {
                    path: relative,
                    sha256: pin.sha256.clone(),
                    bytes: pin.bytes,
                    source: source.clone(),
                    license: license.clone(),
                };
                artifact.validate()?;
                config.assets.retain(|a| a.purpose != purpose);
                config.assets.push(PalsStartupAssetV5 {
                    artifact,
                    snapshot_path: snapshot_path.clone(),
                    purpose,
                });
                require(
                    sources.insert(snapshot_path, (pin, file)).is_none(),
                    "G1 snapshot source collision",
                )?;
            }
            config.endpoint.model.weights = config.asset(P::Weights)?.artifact.clone();
            config.endpoint.model.provenance = Some(config.asset(P::Provenance)?.artifact.clone());
            config.endpoint.model.artifact_domain = PalsArtifactDomainV5::RealGeneration;
            config.endpoint.model.checkpoint_domain =
                Some("rz-pals-actual-generation-checkpoint/1".into());
            config.endpoint.model.epoch = g1.model_epoch();
            require(
                same_model_condition(original.registration().g0_deployment(), &config.endpoint),
                "G1 replaced more than model weights/provenance/domain/epoch",
            )?;
            manifest.engines[1] = config.endpoint.clone();
            manifest.comparison = PalsComparisonV5::InternalWeights;
            let lock = manifest.lock()?;
            config.semantic_lock_sha256 = lock.canonical_sha256.clone();
            config.validate_against(&lock)?;
            for a in &config.assets {
                if sources.contains_key(&a.snapshot_path) {
                    continue;
                }
                let path = original.source_root().join(&a.artifact.path);
                let held = original
                    .originals
                    .iter()
                    .find(|h| {
                        h.pin.path == path
                            && h.pin.sha256 == a.artifact.sha256
                            && h.pin.bytes == a.artifact.bytes
                    })
                    .ok_or_else(|| invalid("unchanged G1 input lost preregistered descriptor"))?;
                sources.insert(a.snapshot_path.clone(), (&held.pin, &held.file));
            }
            let observed = crate::pals_collect_v3::selfplay_cli::scan_staging(
                &original.root,
                PALS_STAGING_BYTES_V5,
                until,
                cancel,
            )?;
            let copy_bytes = config.assets.iter().try_fold(0u64, |sum, a| {
                let input = sources
                    .get(&a.snapshot_path)
                    .ok_or_else(|| invalid("G1 snapshot reservation lacks original source"))?;
                sum.checked_add(
                    if input.0.path == original.private_cwd().join(&a.snapshot_path) {
                        0
                    } else {
                        a.artifact.bytes
                    },
                )
                .ok_or_else(|| invalid("G1 snapshot size overflow"))
            })?;
            require(
                observed
                    .unique_inode_bytes
                    .checked_add(copy_bytes)
                    .and_then(|v| v.checked_add(MAX_METADATA))
                    .is_some_and(|v| v <= PALS_STAGING_BYTES_V5),
                "actual G1 prepared snapshots exceed managed18GiB staging ceiling",
            )?;
            for a in &config.assets {
                let input = *sources
                    .get(&a.snapshot_path)
                    .ok_or_else(|| invalid("G1 snapshot lacks original source"))?;
                if input.0.path == original.private_cwd().join(&a.snapshot_path) {
                    // Earlier stages retain these same private runtime/G0 evidence
                    // descriptors. Reuse only the exact original named inode;
                    // another existing pathname never permits a copy overwrite.
                    io::same_named(input.0, input.1)?;
                    io::verify_bytes(input.1, input.0, false, &mut credit)?;
                    failed.files.push(Held {
                        pin: input.0.clone(),
                        file: input
                            .1
                            .try_clone()
                            .map_err(|_| invalid("original private G1 input custody"))?,
                    });
                    continue;
                }
                copy_new(
                    &original.cwd,
                    original.private_cwd(),
                    &a.snapshot_path,
                    input.0,
                    input.1,
                    &mut credit,
                    &mut failed,
                )?;
            }
            let export_raw = io::verify_bytes(export.1, export.0, true, &mut credit)?;
            validate_domain_held(&config, &export_raw, &failed.files, &mut credit)?;
            crate::pals_collect_v3::selfplay_cli::scan_staging(
                &original.root,
                PALS_STAGING_BYTES_V5,
                until,
                cancel,
            )?;
            check_original_roots(&original)?;
            Ok((config, lock))
        })();
        match result {
            Ok((config, lock)) => Ok(PreparedG1EvaluationStartupV5 {
                original,
                g1,
                config,
                lock,
                files: std::mem::take(&mut failed.files),
            }),
            Err(error) => {
                set_error(&mut failed, error, &account);
                Err(Box::new(failed))
            }
        }
    }
}

pub fn finalize_original_g1_evaluation_launches_v5(
    prepared: Arc<PreparedG1EvaluationStartupV5>,
    numerical: Arc<VerifiedG1NumericalV3>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<PreparedG1EvaluationLaunchesV5, Box<G1EvaluationPreparationFailureV5>> {
    let mut failed = failure(
        Arc::clone(prepared.original.registration()),
        invalid("G1 final launch preparation incomplete"),
    );
    failed.original = Some(Arc::clone(&prepared.original));
    failed.g1 = Some(Arc::clone(&prepared.g1));
    failed.prepared = Some(Arc::clone(&prepared));
    failed.numerical = Some(Arc::clone(&numerical));
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, cancel);
        failed.error = invalid("G1 final launch publication requires Linux");
        Err(Box::new(failed))
    }
    #[cfg(target_os = "linux")]
    {
        let account = ReadAccounting::new();
        let mut credit = read_credit(until, cancel, &account);
        let result = (|| -> Result<Vec<LockedPalsArenaLaunchV5>, ArenaError> {
            require(
                Arc::ptr_eq(&prepared, numerical.prepared())
                    && Arc::ptr_eq(prepared.g1(), numerical.g1()),
                "G1 final publication requires exact original prepared/export numerical owners",
            )?;
            check_original_roots(&prepared.original)?;
            let completed = numerical.startup_config().clone();
            completed.validate_against(numerical.lock())?;
            let mut expected = prepared.config.clone();
            let mut actual = completed.clone();
            for config in [&mut expected, &mut actual] {
                config.assets.retain(|a| {
                    !matches!(
                        a.purpose,
                        PalsStartupAssetPurposeV5::NumericalAcceptance
                            | PalsStartupAssetPurposeV5::NumericalRawEvidence
                    )
                });
            }
            require(
                expected == actual && numerical.lock() == prepared.lock(),
                "G1 numerical producer changed original model/options outside actual numerical/control evidence",
            )?;
            let mut all_files = Vec::new();
            for (pin, file) in numerical.held_files().chain(prepared.held_files()) {
                io::verify_bytes(file, pin, false, &mut credit)?;
                all_files.push(Held {
                    pin: pin.clone(),
                    file: file
                        .try_clone()
                        .map_err(|_| invalid("original numerical FD clone"))?,
                });
            }
            // Acceptance validates exact G1 bindings/reference/raw physical completion;
            // G0 data cannot be relabelled by matching only a JSON status.
            let read = |artifact: &ArtifactRef,
                        limit: usize,
                        credit: &mut ReadCredit<'_>|
             -> Result<Vec<u8>, rz_experiments::ManifestError> {
                let asset = completed
                    .assets
                    .iter()
                    .find(|a| &a.artifact == artifact)
                    .ok_or_else(|| {
                        rz_experiments::ManifestError::Artifact("G1 final metadata absent".into())
                    })?;
                let path = prepared.private_cwd().join(&asset.snapshot_path);
                let held = all_files
                    .iter()
                    .find(|h| {
                        h.pin.path == path
                            && h.pin.sha256 == artifact.sha256
                            && h.pin.bytes == artifact.bytes
                    })
                    .ok_or_else(|| {
                        rz_experiments::ManifestError::Artifact(
                            "original numerical metadata FD absent".into(),
                        )
                    })?;
                require(
                    held.pin.bytes <= limit as u64,
                    "G1 bounded final metadata read",
                )
                .map_err(|e| rz_experiments::ManifestError::Artifact(e.to_string()))?;
                io::verify_bytes(&held.file, &held.pin, true, credit)
                    .map_err(|e| rz_experiments::ManifestError::Artifact(e.to_string()))
            };
            let acceptance = read(
                &completed
                    .asset(PalsStartupAssetPurposeV5::NumericalAcceptance)?
                    .artifact,
                256 << 10,
                &mut credit,
            )?;
            rz_experiments::validate_pals_numerical_acceptance_v5(
                &completed,
                &acceptance,
                |a, limit| read(a, limit, &mut credit),
            )?;
            let copy_bytes = completed
                .assets
                .iter()
                .filter(|a| {
                    matches!(
                        a.purpose,
                        PalsStartupAssetPurposeV5::NumericalAcceptance
                            | PalsStartupAssetPurposeV5::NumericalRawEvidence
                    )
                })
                .try_fold(0u64, |sum, a| {
                    let held = all_files
                        .iter()
                        .find(|h| {
                            h.pin.path == prepared.private_cwd().join(&a.snapshot_path)
                                && h.pin.sha256 == a.artifact.sha256
                                && h.pin.bytes == a.artifact.bytes
                        })
                        .ok_or_else(|| {
                            invalid("actual numerical reservation lacks original descriptor")
                        })?;
                    sum.checked_add(
                        if held.pin.path == prepared.source_root().join(&a.artifact.path) {
                            0
                        } else {
                            a.artifact.bytes
                        },
                    )
                    .ok_or_else(|| invalid("actual numerical reservation overflow"))
                })?;
            let mut metadata_bytes = 0u64;
            for template in &prepared.original.declaration.templates {
                let mut manifest = template.semantic_lock.manifest.clone();
                manifest.engines[1] = completed.endpoint.clone();
                manifest.comparison = PalsComparisonV5::InternalWeights;
                let lock = manifest.lock()?;
                let raw = serde_json::to_vec(&lock)
                    .map_err(|_| invalid("actual semantic lock reservation encoding"))?;
                metadata_bytes = metadata_bytes
                    .checked_add(raw.len() as u64)
                    .ok_or_else(|| invalid("metadata reservation overflow"))?;
                for configuration in [&template.endpoints[0].configuration, &completed] {
                    let mut config = configuration.clone();
                    config.semantic_lock_sha256 = lock.canonical_sha256.clone();
                    let raw = config.canonical_bytes(&lock)?;
                    metadata_bytes = metadata_bytes
                        .checked_add(raw.len() as u64)
                        .ok_or_else(|| invalid("metadata reservation overflow"))?;
                }
            }
            let observed = crate::pals_collect_v3::selfplay_cli::scan_staging(
                &prepared.original.root,
                PALS_STAGING_BYTES_V5,
                until,
                cancel,
            )?;
            require(
                observed
                    .unique_inode_bytes
                    .checked_add(copy_bytes)
                    .and_then(|v| v.checked_add(metadata_bytes))
                    .is_some_and(|v| v <= PALS_STAGING_BYTES_V5),
                "final G1 metadata/raw publication exceeds original18GiB storage ceiling",
            )?;
            // Preserve artifact identities used inside the original acceptance
            // run references. Copy held numerical bytes to their registered
            // logical source paths rather than rewriting references or proofs.
            let source_dir = io::open_directory(prepared.source_root())?;
            for a in &completed.assets {
                if matches!(
                    a.purpose,
                    PalsStartupAssetPurposeV5::NumericalAcceptance
                        | PalsStartupAssetPurposeV5::NumericalRawEvidence
                ) {
                    let held = all_files
                        .iter()
                        .find(|h| {
                            h.pin.path == prepared.private_cwd().join(&a.snapshot_path)
                                && h.pin.sha256 == a.artifact.sha256
                                && h.pin.bytes == a.artifact.bytes
                        })
                        .ok_or_else(|| invalid("actual numerical source descriptor missing"))?;
                    if held.pin.path != prepared.source_root().join(&a.artifact.path) {
                        copy_new(
                            &source_dir,
                            prepared.source_root(),
                            &a.artifact.path,
                            &held.pin,
                            &held.file,
                            &mut credit,
                            &mut failed,
                        )?;
                    }
                }
            }
            let mut launches = Vec::with_capacity(32);
            for (index, template) in prepared.original.declaration.templates.iter().enumerate() {
                credit.take(0)?;
                let mut input = template.clone();
                let mut manifest = input.semantic_lock.manifest.clone();
                manifest.engines[1] = completed.endpoint.clone();
                manifest.comparison = PalsComparisonV5::InternalWeights;
                let lock = manifest.lock()?;
                input.semantic_lock = lock.clone();
                input.endpoints[1].configuration = completed.clone();
                for endpoint in &mut input.endpoints {
                    endpoint.configuration.semantic_lock_sha256 = lock.canonical_sha256.clone();
                    let raw = endpoint.configuration.canonical_bytes(&lock)?;
                    let path = format!(
                        "evaluation-metadata/pair-{index:02}-{}.json",
                        endpoint.configuration.endpoint.id
                    );
                    let pin = publish_new(
                        &prepared.original.cwd,
                        prepared.private_cwd(),
                        &path,
                        &raw,
                        &mut credit,
                        &mut failed,
                    )?;
                    endpoint.startup_config =
                        artifact_from_pin(&pin, prepared.source_root(), &endpoint.startup_config)?;
                }
                let lock_raw = serde_json::to_vec(&lock)
                    .map_err(|_| invalid("actual G1 semantic lock encoding"))?;
                let pin = publish_new(
                    &prepared.original.cwd,
                    prepared.private_cwd(),
                    &format!("evaluation-metadata/pair-{index:02}-lock.json"),
                    &lock_raw,
                    &mut credit,
                    &mut failed,
                )?;
                input.semantic_lock_artifact =
                    artifact_from_pin(&pin, prepared.source_root(), &input.semantic_lock_artifact)?;
                input.validate()?;
                launches.push(input.lock()?);
            }
            crate::pals_collect_v3::selfplay_cli::scan_staging(
                &prepared.original.root,
                PALS_STAGING_BYTES_V5,
                until,
                cancel,
            )?;
            validate_series(
                prepared.original.registration(),
                &prepared.g1,
                &launches,
                prepared.source_root(),
            )?;
            check_original_roots(&prepared.original)?;
            failed.files.extend(all_files);
            Ok(launches)
        })();
        match result {
            Ok(launches) => Ok(PreparedG1EvaluationLaunchesV5 {
                prepared,
                numerical,
                launches,
                files: std::mem::take(&mut failed.files),
                serial_evaluation_issued: AtomicBool::new(false),
            }),
            Err(error) => {
                set_error(&mut failed, error, &account);
                Err(Box::new(failed))
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn read_credit<'a>(
    until: Instant,
    cancel: &'a AtomicBool,
    account: &Arc<ReadAccounting>,
) -> ReadCredit<'a> {
    ReadCredit {
        remaining: 8 * PALS_STAGING_BYTES_V5,
        maximum: 8 * PALS_STAGING_BYTES_V5,
        until,
        cancel,
        account: Arc::clone(account),
    }
}
#[cfg(target_os = "linux")]
fn set_error(
    failed: &mut G1EvaluationPreparationFailureV5,
    error: ArenaError,
    account: &ReadAccounting,
) {
    failed.error = error;
    failed.charged_read_bytes = account.charged.load(Ordering::Acquire);
    failed.read_usage_uncertain = account.uncertain.load(Ordering::Acquire);
}
#[cfg(target_os = "linux")]
fn check_original_roots(original: &RegisteredPalsEvaluationTemplatesV5) -> Result<(), ArenaError> {
    for (path, held) in [
        (original.managed_root(), &original.root),
        (original.private_cwd(), &original.cwd),
        (original.source_root(), &original.source_dir),
    ] {
        let named = io::open_directory(path)?;
        let a = io::directory_file(held)?;
        let b = io::directory_file(&named)?;
        require(
            io::FileIdentity::from(&a.metadata().map_err(|_| invalid("held root metadata"))?)
                .same_inode(io::FileIdentity::from(
                    &b.metadata().map_err(|_| invalid("named root metadata"))?,
                )),
            "original final managed/private root inode replaced",
        )?;
        crate::pals_collect_v3::selfplay_cli::private_root(&named)?;
    }
    Ok(())
}
#[cfg(target_os = "linux")]
fn ensure_parent(cwd: &cap_std::fs::Dir, relative: &str) -> Result<cap_std::fs::Dir, ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::PermissionsExt;
    require(
        !Path::new(relative).is_absolute()
            && Path::new(relative)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "safe owned output relative path",
    )?;
    let mut parent = cwd.try_clone().map_err(|_| invalid("output cwd clone"))?;
    for part in Path::new(relative).parent().unwrap().components() {
        let std::path::Component::Normal(name) = part else {
            return Err(invalid("output parent component"));
        };
        match parent.create_dir(name) {
            Ok(()) => parent
                .set_permissions(
                    name,
                    cap_std::fs::Permissions::from_std(std::fs::Permissions::from_mode(0o700)),
                )
                .map_err(|_| invalid("owned output directory mode"))?,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return Err(invalid("owned output directory create")),
        }
        parent = parent
            .open_dir_nofollow(name)
            .map_err(|_| invalid("owned output directory linked"))?;
        crate::pals_collect_v3::selfplay_cli::private_root(&parent)?;
    }
    Ok(parent)
}
#[cfg(target_os = "linux")]
fn copy_new(
    cwd: &cap_std::fs::Dir,
    base: &Path,
    relative: &str,
    pin: &GenerationTrainingFilePinV3,
    source: &File,
    credit: &mut ReadCredit<'_>,
    failed: &mut G1EvaluationPreparationFailureV5,
) -> Result<GenerationTrainingFilePinV3, ArenaError> {
    use std::{io::Write, os::unix::fs::FileExt};
    let identity = io::same_named(pin, source)?;
    let parent = ensure_parent(cwd, relative)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut target = parent
        .open_with(Path::new(relative).file_name().unwrap(), &options)
        .map_err(|_| invalid("exclusive original G1 snapshot create"))?
        .into_std();
    failed.partial_files.push(
        target
            .try_clone()
            .map_err(|_| invalid("partial snapshot descriptor custody"))?,
    );
    let mut offset = 0;
    let mut block = [0u8; 65536];
    while offset < pin.bytes {
        let count = (pin.bytes - offset).min(block.len() as u64) as usize;
        credit.take(count as u64)?;
        let n = source
            .read_at(&mut block[..count], offset)
            .map_err(|_| invalid("original snapshot read"))?;
        require(n > 0, "original snapshot premature EOF")?;
        target
            .write_all(&block[..n])
            .map_err(|_| invalid("original snapshot write"))?;
        offset += n as u64;
    }
    target
        .sync_all()
        .map_err(|_| invalid("original snapshot sync"))?;
    require(
        identity == io::same_named(pin, source)?,
        "original source changed while copying",
    )?;
    let pin = GenerationTrainingFilePinV3 {
        path: base.join(relative),
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
    };
    let file = io::open_file(&pin.path)?;
    io::verify_bytes(&file, &pin, false, credit)?;
    failed.files.push(Held {
        pin: pin.clone(),
        file,
    });
    failed.partial_files.pop();
    Ok(pin)
}
#[cfg(target_os = "linux")]
fn publish_new(
    cwd: &cap_std::fs::Dir,
    base: &Path,
    relative: &str,
    raw: &[u8],
    credit: &mut ReadCredit<'_>,
    failed: &mut G1EvaluationPreparationFailureV5,
) -> Result<GenerationTrainingFilePinV3, ArenaError> {
    use std::io::Write;
    require(
        !raw.is_empty() && raw.len() as u64 <= MAX_METADATA,
        "bounded actual G1 metadata",
    )?;
    credit.take(0)?;
    let parent = ensure_parent(cwd, relative)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = parent
        .open_with(Path::new(relative).file_name().unwrap(), &options)
        .map_err(|_| invalid("exclusive G1 metadata create"))?
        .into_std();
    failed.partial_files.push(
        file.try_clone()
            .map_err(|_| invalid("partial G1 metadata custody"))?,
    );
    file.write_all(raw)
        .map_err(|_| invalid("actual G1 metadata write"))?;
    file.sync_all()
        .map_err(|_| invalid("actual G1 metadata sync"))?;
    let pin = GenerationTrainingFilePinV3 {
        path: base.join(relative),
        sha256: format!("{:x}", Sha256::digest(raw)),
        bytes: raw.len() as u64,
    };
    let held = io::open_file(&pin.path)?;
    io::verify_bytes(&held, &pin, false, credit)?;
    failed.files.push(Held {
        pin: pin.clone(),
        file: held,
    });
    failed.partial_files.pop();
    Ok(pin)
}
#[cfg(target_os = "linux")]
fn artifact_from_pin(
    pin: &GenerationTrainingFilePinV3,
    source_root: &Path,
    original: &ArtifactRef,
) -> Result<ArtifactRef, ArenaError> {
    let relative = pin
        .path
        .strip_prefix(source_root)
        .map_err(|_| invalid("actual final artifact outside registered source root"))?
        .to_str()
        .ok_or_else(|| invalid("final artifact UTF8"))?
        .replace('\\', "/");
    let artifact = ArtifactRef {
        path: relative,
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
        source: original.source.clone(),
        license: original.license.clone(),
    };
    artifact.validate()?;
    Ok(artifact)
}
#[cfg(target_os = "linux")]
fn validate_domain_held(
    config: &PalsAdaptiveStartupConfigV5,
    export_raw: &[u8],
    files: &[Held],
    credit: &mut ReadCredit<'_>,
) -> Result<(), ArenaError> {
    rz_experiments::validate_pals_artifact_domain_v5(config, export_raw, |a, limit| {
        let held = files
            .iter()
            .find(|h| h.pin.sha256 == a.sha256 && h.pin.bytes == a.bytes)
            .ok_or_else(|| {
                rz_experiments::ManifestError::Artifact(
                    "actual G1 metadata descriptor absent".into(),
                )
            })?;
        require(
            held.pin.bytes <= limit as u64,
            "actual G1 metadata allocation bound",
        )
        .map_err(|e| rz_experiments::ManifestError::Artifact(e.to_string()))?;
        io::verify_bytes(&held.file, &held.pin, true, credit)
            .map_err(|e| rz_experiments::ManifestError::Artifact(e.to_string()))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::claim_serial_once;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    // Authored CPU control-state tests; no original exporter or numerical
    // owner is fabricated, and no native evaluation is admitted by them.
    #[test]
    fn serial_claim_cannot_be_reissued_after_an_attempt() {
        let issued = AtomicBool::new(false);
        assert!(claim_serial_once(&issued).is_ok());
        assert!(claim_serial_once(&issued).is_err());
        assert!(issued.load(Ordering::Acquire));
    }
    #[test]
    fn concurrent_consumers_can_claim_only_one_registered_series() {
        let issued = Arc::new(AtomicBool::new(false));
        let other = Arc::clone(&issued);
        let first = std::thread::spawn(move || claim_serial_once(&other).is_ok());
        let second = claim_serial_once(&issued).is_ok();
        assert_eq!(usize::from(first.join().unwrap()) + usize::from(second), 1);
        assert!(claim_serial_once(&issued).is_err());
    }
}
