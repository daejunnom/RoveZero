//! Prepare a distinct frozen-P/C ActionCollection startup from the original
//! export owner. Metadata preparation neither executes training/NN nor grants
//! native numerical or loaded-tensor authority. Exact independent numerical
//! evidence is still required; G0 evidence is never relabelled for new weights.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]
use super::{
    ActionCliDeclarationV3, GenerationTrainingFilePinV3, VerifiedGenerationExportV3, invalid,
};
#[cfg(target_os = "linux")]
use super::{GenerationExportKindV3, check, compiled_frozen_pc_verifier_pins_v3};
use crate::ArenaError;
#[cfg(target_os = "linux")]
use rz_experiments::{
    ArtifactRef, PalsArtifactDomainV5, PalsEngineModeV5, PalsModelBackendV3, PalsRunPurposeV5,
    PalsStartupAssetPurposeV5 as Purpose, PalsStartupAssetV5,
};
use rz_experiments::{PalsAdaptiveStartupConfigV5, PalsInputLockV5};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(target_os = "linux")]
use std::collections::BTreeMap;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

pub const FROZEN_ACTION_STARTUP_TEMPLATE_V3_DOMAIN: &str =
    "rz-pals-frozen-action-startup-template/1";
const MAX_TEMPLATE: u64 = 8 * 1024 * 1024;
const MAX_JSON: u64 = 4 * 1024 * 1024;
const MAX_STAGING: u64 = 18 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenStartupInputPinV3 {
    pub snapshot_path: String,
    pub source: GenerationTrainingFilePinV3,
}
/// A preregistered ActionCollection template. Its future checkpoint/model pins
/// are replaced only from the supplied opaque actual export. Other assets
/// remain exact original inputs. Numerical declarations are pending until the
/// separate actual numerical owner replaces them before action publication.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenActionStartupTemplateV3 {
    pub schema: String,
    pub declaration: ActionCliDeclarationV3,
    pub config: PalsAdaptiveStartupConfigV5,
    pub lock: PalsInputLockV5,
    pub content_request: Value,
    pub input_pins: Vec<FrozenStartupInputPinV3>,
}

struct RetainedFile {
    pin: GenerationTrainingFilePinV3,
    file: File,
}
pub struct PreparedFrozenActionStartupInputsV3 {
    declaration_path: PathBuf,
    declaration_file_sha256: String,
    config: PalsAdaptiveStartupConfigV5,
    lock: PalsInputLockV5,
    frozen: Arc<VerifiedGenerationExportV3>,
    template_pin: GenerationTrainingFilePinV3,
    template: File,
    files: Vec<RetainedFile>,
    #[cfg(target_os = "linux")]
    cwd: cap_std::fs::Dir,
    preparation_until: Instant,
    declaration: ActionCliDeclarationV3,
}
impl PreparedFrozenActionStartupInputsV3 {
    pub fn pending_input_template_path(&self) -> &Path {
        &self.declaration_path
    }
    pub fn pending_input_template_file_sha256(&self) -> &str {
        &self.declaration_file_sha256
    }
    pub fn frozen(&self) -> &Arc<VerifiedGenerationExportV3> {
        &self.frozen
    }
    pub fn config(&self) -> &PalsAdaptiveStartupConfigV5 {
        &self.config
    }
    pub fn lock(&self) -> &PalsInputLockV5 {
        &self.lock
    }
    pub fn template_pin(&self) -> &GenerationTrainingFilePinV3 {
        &self.template_pin
    }
    pub fn preparation_until(&self) -> Instant {
        self.preparation_until
    }
    pub fn private_cwd(&self) -> &Path {
        &self.declaration.private_cwd
    }
    /// These descriptors establish original metadata/file custody only.
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        std::iter::once((&self.template_pin, &self.template))
            .chain(self.files.iter().map(|held| (&held.pin, &held.file)))
    }
    #[cfg(target_os = "linux")]
    pub fn private_cwd_file(&self) -> Result<File, ArenaError> {
        super::training_process::directory_file(&self.cwd)
    }
}

/// Final declaration publication retains both original input and actual
/// independent numerical owners. It does not establish first-Run readiness,
/// frozen tensor intake, an action witness or optimizer execution.
pub struct PreparedFrozenActionCollectionV3 {
    declaration_path: PathBuf,
    declaration_file_sha256: String,
    config: PalsAdaptiveStartupConfigV5,
    lock: PalsInputLockV5,
    prepared: Arc<PreparedFrozenActionStartupInputsV3>,
    numerical: Arc<super::VerifiedFrozenNumericalV3>,
    files: Vec<RetainedFile>,
}
impl PreparedFrozenActionCollectionV3 {
    pub fn declaration_path(&self) -> &Path {
        &self.declaration_path
    }
    pub fn declaration_file_sha256(&self) -> &str {
        &self.declaration_file_sha256
    }
    pub fn config(&self) -> &PalsAdaptiveStartupConfigV5 {
        &self.config
    }
    pub fn lock(&self) -> &PalsInputLockV5 {
        &self.lock
    }
    pub fn prepared(&self) -> &Arc<PreparedFrozenActionStartupInputsV3> {
        &self.prepared
    }
    pub fn numerical(&self) -> &Arc<super::VerifiedFrozenNumericalV3> {
        &self.numerical
    }
    pub fn frozen(&self) -> &Arc<VerifiedGenerationExportV3> {
        self.prepared.frozen()
    }
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.files.iter().map(|held| (&held.pin, &held.file))
    }
}

pub fn prepare_frozen_action_collection_owned_v3(
    prepared: Arc<PreparedFrozenActionStartupInputsV3>,
    numerical: Arc<super::VerifiedFrozenNumericalV3>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<PreparedFrozenActionCollectionV3, Box<FrozenActionPreparationFailureV3>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, cancel);
        Err(Box::new(FrozenActionPreparationFailureV3 {
            error: invalid("frozen action publication requires Linux original descriptors"),
            frozen: Arc::clone(prepared.frozen()),
            template_pin: prepared.template_pin.clone(),
            charged_read_bytes: 0,
            read_usage_uncertain: false,
            prepared: Some(prepared),
            numerical: Some(numerical),
            template: None,
            files: Vec::new(),
        }))
    }
    #[cfg(target_os = "linux")]
    {
        finalize_linux(prepared, numerical, until, cancel)
    }
}

#[cfg(target_os = "linux")]
fn finalize_linux(
    prepared: Arc<PreparedFrozenActionStartupInputsV3>,
    numerical: Arc<super::VerifiedFrozenNumericalV3>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<PreparedFrozenActionCollectionV3, Box<FrozenActionPreparationFailureV3>> {
    use super::training_process::{
        FileIdentity, ReadAccounting, ReadCredit, directory_file, open_directory, verify_bytes,
    };
    use std::sync::atomic::Ordering;
    let account = ReadAccounting::new();
    let mut files = Vec::new();
    let mut credit = ReadCredit {
        remaining: 2 * MAX_STAGING,
        maximum: 2 * MAX_STAGING,
        until,
        cancel,
        account: Arc::clone(&account),
    };
    let result=(||->Result<(PathBuf,String,PalsAdaptiveStartupConfigV5,PalsInputLockV5),ArenaError>{
        check(Arc::ptr_eq(&prepared,numerical.prepared()) && Arc::ptr_eq(prepared.frozen(),numerical.frozen()),
            "frozen action publication must retain the exact original prepared/export numerical owners")?;
        let config=numerical.startup_config().clone();let lock=numerical.lock().clone();
        config.validate_against(&lock)?;
        let mut source=config.clone();let mut original=prepared.config.clone();
        for item in [&mut source,&mut original] {
            item.assets.retain(|a|!matches!(a.purpose,Purpose::NumericalAcceptance|Purpose::NumericalRawEvidence));
        }
        check(source==original && lock==prepared.lock,
            "actual frozen numerical completion may replace numerical artifacts only")?;
        let mut read=|artifact:&ArtifactRef,limit:usize|->Result<Vec<u8>,rz_experiments::ManifestError>{
            let declared=config.assets.iter().find(|a|&a.artifact==artifact).ok_or_else(||
                rz_experiments::ManifestError::Artifact("frozen finalized metadata missing".into()))?;
            let path=prepared.private_cwd().join(&declared.snapshot_path);
            let held=numerical.held_files().chain(prepared.held_files()).find(|(pin,_)|pin.path==path
                &&pin.sha256==artifact.sha256&&pin.bytes==artifact.bytes).ok_or_else(||
                rz_experiments::ManifestError::Artifact("frozen finalized original descriptor missing".into()))?;
            check(held.0.bytes<=limit as u64,"frozen finalized metadata limit").map_err(|e|
                rz_experiments::ManifestError::Artifact(e.to_string()))?;
            verify_bytes(held.1,held.0,true,&mut credit).map_err(|e|rz_experiments::ManifestError::Artifact(e.to_string()))
        };
        let export=read(&config.asset(Purpose::ExportManifest)?.artifact,256<<10)?;
        rz_experiments::validate_pals_artifact_domain_v5(&config,&export,&mut read)?;
        let acceptance=read(&config.asset(Purpose::NumericalAcceptance)?.artifact,256<<10)?;
        rz_experiments::validate_pals_numerical_acceptance_v5(&config,&acceptance,&mut read)?;
        let mut declaration=prepared.declaration.clone();
        let named=open_directory(prepared.private_cwd())?;
        let held_cwd=directory_file(&prepared.cwd)?;let named_cwd=directory_file(&named)?;
        check(FileIdentity::from(&held_cwd.metadata().map_err(|_|invalid("frozen final held cwd metadata"))?)
            .same_inode(FileIdentity::from(&named_cwd.metadata().map_err(|_|invalid("frozen final named cwd metadata"))?)),
            "frozen final private cwd was replaced")?;
        let managed=open_directory(&declaration.managed_root)?;
        validate_original_managed_scope(
            &prepared.frozen().trainer().registration().staging_root,
            prepared.frozen().trainer().registration().max_output_bytes,
            &declaration.managed_root,
            &declaration.private_cwd,
            declaration.max_staging_bytes,
        )?;
        validate_original_staging_owner(prepared.frozen().trainer(),&managed)?;
        super::selfplay_cli::private_root(&managed)?;
        let observed=super::selfplay_cli::scan_staging(&managed,declaration.max_staging_bytes,until,cancel)?;
        declaration.startup_config=prepared.private_cwd().join("frozen-startup-config-v5.json");
        declaration.startup_lock=prepared.private_cwd().join("frozen-startup-lock-v5.json");
        let config_raw=config.canonical_bytes(&lock)?;
        use sha2::{Digest, Sha256};
        declaration.startup_config_sha256=format!("{:x}",Sha256::digest(&config_raw));
        let lock_raw=serde_json::to_vec(&lock).map_err(|e|invalid(e.to_string()))?;
        declaration.validate()?;
        let declaration_raw=serde_json::to_vec(&declaration).map_err(|e|invalid(e.to_string()))?;
        let mut publication_bytes=0u64;
        for raw in [&config_raw,&lock_raw,&declaration_raw] {
            check(!raw.is_empty()&&raw.len() as u64<=MAX_JSON,"frozen final metadata publication bound")?;
            publication_bytes=publication_bytes.checked_add(raw.len() as u64).ok_or_else(||invalid("frozen final output size overflow"))?;
        }
        check(observed.unique_inode_bytes.checked_add(publication_bytes)
            .is_some_and(|sum|sum<=declaration.max_staging_bytes&&sum<=MAX_STAGING),
            "frozen final publication must fit the original staging cap before any output is created")?;
        publish_json(&prepared.cwd,prepared.private_cwd(),"frozen-startup-config-v5.json",&config_raw,&mut credit,&mut files)?;
        publish_json(&prepared.cwd,prepared.private_cwd(),"frozen-startup-lock-v5.json",&lock_raw,&mut credit,&mut files)?;
        let pin=publish_json(&prepared.cwd,prepared.private_cwd(),"frozen-action-declaration-v3.json",&declaration_raw,&mut credit,&mut files)?;
        let path=pin.path;let sha=pin.sha256;
        super::selfplay_cli::scan_staging(&managed,declaration.max_staging_bytes,until,cancel)?;
        Ok((path,sha,config,lock))
    })();
    match result {
        Ok((declaration_path, declaration_file_sha256, config, lock)) => {
            Ok(PreparedFrozenActionCollectionV3 {
                declaration_path,
                declaration_file_sha256,
                config,
                lock,
                prepared,
                numerical,
                files,
            })
        }
        Err(error) => Err(Box::new(FrozenActionPreparationFailureV3 {
            error,
            frozen: Arc::clone(prepared.frozen()),
            template_pin: prepared.template_pin.clone(),
            charged_read_bytes: account.charged.load(Ordering::Acquire),
            read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
            prepared: Some(prepared),
            numerical: Some(numerical),
            template: None,
            files,
            cwd: None,
        })),
    }
}

pub struct FrozenActionPreparationFailureV3 {
    pub error: ArenaError,
    pub frozen: Arc<VerifiedGenerationExportV3>,
    pub template_pin: GenerationTrainingFilePinV3,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    pub prepared: Option<Arc<PreparedFrozenActionStartupInputsV3>>,
    pub numerical: Option<Arc<super::VerifiedFrozenNumericalV3>>,
    template: Option<File>,
    files: Vec<RetainedFile>,
    #[cfg(target_os = "linux")]
    cwd: Option<cap_std::fs::Dir>,
}
impl std::fmt::Debug for FrozenActionPreparationFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("FrozenActionPreparationFailureV3");
        debug
            .field("error", &self.error)
            .field("retained_files", &self.files.len())
            .field("template_retained", &self.template.is_some())
            .field("charged_read_bytes", &self.charged_read_bytes)
            .field("read_usage_uncertain", &self.read_usage_uncertain);
        #[cfg(target_os = "linux")]
        debug.field("cwd_retained", &self.cwd.is_some());
        debug.finish_non_exhaustive()
    }
}

pub fn prepare_frozen_action_startup_inputs_owned_v3(
    template_file_pin: GenerationTrainingFilePinV3,
    frozen: Arc<VerifiedGenerationExportV3>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<PreparedFrozenActionStartupInputsV3, Box<FrozenActionPreparationFailureV3>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (until, cancel);
        Err(Box::new(FrozenActionPreparationFailureV3 {
            error: invalid("frozen action startup preparation requires Linux original descriptors"),
            frozen,
            template_pin: template_file_pin,
            charged_read_bytes: 0,
            read_usage_uncertain: false,
            prepared: None,
            numerical: None,
            template: None,
            files: Vec::new(),
        }))
    }
    #[cfg(target_os = "linux")]
    {
        prepare_linux(template_file_pin, frozen, until, cancel)
    }
}

#[cfg(target_os = "linux")]
fn prepare_linux(
    template_pin: GenerationTrainingFilePinV3,
    frozen: Arc<VerifiedGenerationExportV3>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<PreparedFrozenActionStartupInputsV3, Box<FrozenActionPreparationFailureV3>> {
    use super::training_process::{
        ReadAccounting, ReadCredit, open_directory, open_file, verify_bytes,
    };
    use std::sync::atomic::Ordering;
    let account = ReadAccounting::new();
    let mut credit = ReadCredit {
        remaining: 4 * MAX_STAGING,
        maximum: 4 * MAX_STAGING,
        until,
        cancel,
        account: Arc::clone(&account),
    };
    let mut template_file = None;
    let mut files = Vec::new();
    let mut cwd = None;
    let result=(||->Result<(PathBuf,String,PalsAdaptiveStartupConfigV5,PalsInputLockV5,ActionCliDeclarationV3),ArenaError>{
        template_pin.validate(MAX_TEMPLATE)?;
        check(frozen.kind()==GenerationExportKindV3::FrozenPcCollection,
            "only the original 4000-update frozen export owner can prepare action collection")?;
        let file=open_file(&template_pin.path)?;
        let raw=verify_bytes(&file,&template_pin,true,&mut credit)?;
        template_file=Some(file);
        let template:FrozenActionStartupTemplateV3=serde_json::from_slice(&raw).map_err(|e|invalid(e.to_string()))?;
        check(template.schema==FROZEN_ACTION_STARTUP_TEMPLATE_V3_DOMAIN,
            "frozen action template has its own preregistered domain")?;
        template.declaration.validate()?;
        validate_original_managed_scope(
            &frozen.trainer().registration().staging_root,
            frozen.trainer().registration().max_output_bytes,
            &template.declaration.managed_root,
            &template.declaration.private_cwd,
            template.declaration.max_staging_bytes,
        )?;
        template.lock.verify()?;
        template.config.validate_against(&template.lock)?;
        check(template.lock.manifest.purpose==PalsRunPurposeV5::ActionCollection
            && template.lock.manifest.engines.len()==1
            && template.config.endpoint.mode==PalsEngineModeV5::Collection
            && template.config.endpoint.model.profile=="virtual_review_v1"
            && !template.config.native_options.approximate_warm && !template.lock.manifest.training_authorized,
            "frozen template is explicit Fresh ActionCollection, never optimizer or arena admission")?;
        let mut declaration=template.declaration;
        let mut config=template.config;
        let mut manifest=template.lock.manifest;
        let registered_model_source=config.endpoint.model.weights.source.clone();
        let registered_model_license=config.endpoint.model.weights.license.clone();
        let p=frozen.semantic_receipt();
        check(p["schema"]==rz_experiments::PALS_FROZEN_PC_COLLECTION_EXPORT_V5_DOMAIN
            && p["eligibility"]=="frozen_pc_collection_only" && p["arena_eligible"]==false,
            "original frozen export semantic scope differs")?;
        let checkpoint_sha=p["checkpoint"]["sha256"].as_str().ok_or_else(||invalid("frozen original checkpoint pin"))?;
        let spec_pin=frozen.trainer().registration().spec.clone();
        let checkpoint_source=frozen.trainer().checkpoint_files().find(|(pin,_)|pin.sha256==checkpoint_sha)
            .ok_or_else(||invalid("frozen export checkpoint lacks the retained original trainer artifact"))?;
        let provenance_path=frozen.directory().join("frozen-pc-collection-v2.json");
        let provenance_source=frozen.held_files().find(|(pin,_)|pin.path==provenance_path)
            .ok_or_else(||invalid("frozen strong receipt has no original read-back descriptor"))?;
        let export_path=frozen.directory().join("collection/export.json");
        let export_source=frozen.held_files().find(|(pin,_)|pin.path==export_path)
            .ok_or_else(||invalid("frozen collection export has no original read-back descriptor"))?;
        export_source.0.validate(256<<10)?;
        let export_raw=verify_bytes(export_source.1,export_source.0,true,&mut credit)?;
        let export:Value=serde_json::from_slice(&export_raw).map_err(|e|invalid(e.to_string()))?;
        let mut replaced=BTreeMap::<String,RetainedFile>::new();
        let requires_cuda_inventory=config.endpoint.model.backend==PalsModelBackendV3::OrtCuda;
        let mut add=|purpose:Purpose,relative:&str,pin:&GenerationTrainingFilePinV3,file:File|->Result<(),ArenaError>{
            verify_bytes(&file,pin,false,&mut credit)?;
            let snapshot_path=format!("inputs/frozen/{relative}");
            let artifact=ArtifactRef {path:format!("generation/frozen-pc/{relative}"),sha256:pin.sha256.clone(),bytes:pin.bytes,
                source:registered_model_source.clone(),license:registered_model_license.clone()};
            config.assets.retain(|a|a.purpose!=purpose);
            config.assets.push(PalsStartupAssetV5 {artifact,snapshot_path:snapshot_path.clone(),purpose});
            check(replaced.insert(snapshot_path,RetainedFile {pin:pin.clone(),file}).is_none(),"frozen snapshot destination duplicated")
        };
        add(Purpose::Weights,"checkpoint.pt",checkpoint_source.0,checkpoint_source.1.try_clone().map_err(|_|invalid("frozen checkpoint custody clone"))?)?;
        add(Purpose::Provenance,"frozen-pc-collection-v2.json",provenance_source.0,provenance_source.1.try_clone().map_err(|_|invalid("frozen receipt custody clone"))?)?;
        add(Purpose::ExportManifest,"collection/export.json",export_source.0,export_source.1.try_clone().map_err(|_|invalid("frozen export custody clone"))?)?;
        let spec_file=frozen.trainer().held_input_files().find(|(pin,_)|*pin==&spec_pin)
            .ok_or_else(||invalid("frozen recipe lacks the original retained trainer descriptor"))?
            .1.try_clone().map_err(|_|invalid("frozen recipe custody clone"))?;
        add(Purpose::GenerationSpecification,"generation-spec.json",&spec_pin,spec_file)?;
        for (role,name,purpose) in [("public","public_memory.onnx",Purpose::PublicGraph),
            ("shared_pc","shared_pc_if.onnx",Purpose::SharedPcGraph),("validator","role_validator.onnx",Purpose::ValidatorGraph)] {
            let path=frozen.directory().join("collection").join(name);
            let source=frozen.held_files().find(|(pin,_)|pin.path==path).ok_or_else(||invalid("frozen actual graph custody missing"))?;
            check(p["collection"]["graphs"].as_array().is_some_and(|rows|rows.iter().any(|row|
                row["role"]==role&&row["sha256"]==source.0.sha256&&row["bytes"]==source.0.bytes)),
                "frozen graph descriptor and actual export pin differ")?;
            add(purpose,&format!("collection/{name}"),source.0,source.1.try_clone().map_err(|_|invalid("frozen graph custody clone"))?)?;
        }
        if requires_cuda_inventory {
            let inventory=frozen.collection_control_inventory_pin();
            let source=frozen.held_files().find(|(pin,_)|*pin==inventory)
                .ok_or_else(||invalid("frozen CUDA control inventory lacks the actual export descriptor"))?;
            add(Purpose::CudaControlInventory,"collection/cuda-control-inventory.json",source.0,
                source.1.try_clone().map_err(|_|invalid("frozen CUDA control inventory custody clone"))?)?;
        }
        config.assets.retain(|a|!matches!(a.purpose,Purpose::InitializationEvidence|Purpose::InitializationCheckpointMetadata));
        config.endpoint.model.weights=config.asset(Purpose::Weights)?.artifact.clone();
        config.endpoint.model.provenance=Some(config.asset(Purpose::Provenance)?.artifact.clone());
        config.endpoint.model.artifact_domain=PalsArtifactDomainV5::FrozenPcCollection;
        config.endpoint.model.checkpoint_domain=Some("rz-pals-actual-generation-checkpoint/1".into());
        config.endpoint.model.epoch=frozen.model_epoch();
        declaration.plan.frozen_pc_epoch=frozen.model_epoch();
        declaration.plan.frozen_pc_weights_sha256=frozen.frozen_pc_sha256().to_owned();
        declaration.producer.model_sha256=frozen.frozen_pc_sha256().to_owned();
        check(p["source_sha256"]==declaration.producer.source_sha256
            && p["runtime_sha256"]==declaration.producer.runtime_sha256
            && p["checkpoint"]["context_epoch"]==declaration.context_epoch,
            "frozen action source/runtime/original context was changed")?;
        manifest.engines[0]=config.endpoint.clone();
        let lock=manifest.lock()?;
        config.semantic_lock_sha256=lock.canonical_sha256.clone();
        config.validate_against(&lock)?;
        let mut originals=BTreeMap::<String,RetainedFile>::new();
        for input in template.input_pins {
            input.source.validate(MAX_STAGING)?;
            check(config.assets.iter().any(|a|a.snapshot_path==input.snapshot_path&&a.artifact.sha256==input.source.sha256
                &&a.artifact.bytes==input.source.bytes)&&!replaced.contains_key(&input.snapshot_path),
                "template input is not one unchanged startup artifact")?;
            check(input.source.path==declaration.private_cwd.join(&input.snapshot_path),
                "unchanged startup artifact must retain its actual private-cwd descriptor")?;
            let file=open_file(&input.source.path)?;
            verify_bytes(&file,&input.source,false,&mut credit)?;
            check(originals.insert(input.snapshot_path,RetainedFile {pin:input.source,file}).is_none(),
                "unchanged source snapshot duplicated")?;
        }
        check(originals.len()+replaced.len()==config.assets.len(),"frozen preparation lacks exact complete original asset custody")?;
        let mut metadata_reader=|artifact:&ArtifactRef,limit:usize|->Result<Vec<u8>,rz_experiments::ManifestError>{
            let asset=config.assets.iter().find(|a|&a.artifact==artifact).ok_or_else(||
                rz_experiments::ManifestError::Artifact("frozen registered metadata missing".into()))?;
            let held=replaced.get(&asset.snapshot_path).or_else(||originals.get(&asset.snapshot_path)).ok_or_else(||
                rz_experiments::ManifestError::Artifact("frozen original metadata descriptor missing".into()))?;
            check(held.pin.bytes<=limit as u64,"frozen bounded metadata read").map_err(|e|
                rz_experiments::ManifestError::Artifact(e.to_string()))?;
            verify_bytes(&held.file,&held.pin,true,&mut credit).map_err(|e|
                rz_experiments::ManifestError::Artifact(e.to_string()))
        };
        rz_experiments::validate_pals_artifact_domain_v5(&config,&export_raw,&mut metadata_reader)?;
        // The template's numerical declarations remain explicitly pending.
        // This owner grants no numerical admission. A separate actual numeric
        // producer must retain these exact inputs and supply its own completed
        // evidence before the final action declaration can be published.
        let managed=open_directory(&declaration.managed_root)?;
        validate_original_staging_owner(frozen.trainer(),&managed)?;
        super::selfplay_cli::private_root(&managed)?;
        let observed=super::selfplay_cli::scan_staging(&managed,declaration.max_staging_bytes,until,cancel)?;
        let copy_bytes=replaced.values().try_fold(0u64,|sum,held|sum.checked_add(held.pin.bytes)
            .ok_or_else(||invalid("frozen preparation output size overflow")))?;
        check(observed.unique_inode_bytes.checked_add(copy_bytes).and_then(|sum|sum.checked_add(16*MAX_JSON))
            .is_some_and(|sum|sum<=declaration.max_staging_bytes&&sum<=MAX_STAGING),"frozen preparation independent resident output cap")?;
        check(declaration.private_cwd.starts_with(&declaration.managed_root),"frozen private cwd outside original managed root")?;
        let cwd_owner=open_directory(&declaration.private_cwd)?;
        super::selfplay_cli::private_root(&cwd_owner)?;
        cwd=Some(cwd_owner);
        let owner=cwd.as_ref().expect("private cwd retained");
        create_frozen_directories(owner)?;
        for (relative,held) in replaced {
            copy_new(owner,&declaration.private_cwd,&relative,&held,&mut credit,&mut files)?;
            files.push(held);
        }
        files.extend(originals.into_values());
        declaration.startup_config=declaration.private_cwd.join("frozen-input-config-v5.json");
        declaration.startup_lock=declaration.private_cwd.join("frozen-input-lock-v5.json");
        let config_bytes=config.canonical_bytes(&lock)?;
        let config_pin=publish_json(owner,&declaration.private_cwd,"frozen-input-config-v5.json",&config_bytes,&mut credit,&mut files)?;
        declaration.startup_config_sha256=config_pin.sha256.clone();
        let lock_bytes=serde_json::to_vec(&lock).map_err(|e|invalid(e.to_string()))?;
        publish_json(owner,&declaration.private_cwd,"frozen-input-lock-v5.json",&lock_bytes,&mut credit,&mut files)?;
        let mut request=template.content_request;
        check(request.is_object(),"frozen actual content request template must be an object")?;
        request["schema"]=serde_json::json!("rz-pals-frozen-pc-content-request/1");
        request["source_sha256"]=serde_json::json!(declaration.producer.source_sha256);
        request["model_epoch"]=serde_json::json!(frozen.model_epoch()); request["config"]=export["config"].clone();
        request["checkpoint"]=serde_json::json!({"file":"checkpoint.pt","sha256":config.endpoint.model.weights.sha256,
            "bytes":config.endpoint.model.weights.bytes,"domain":"rz-pals-actual-generation-checkpoint/1"});
        request["graphs"]=Value::Array(p["collection"]["graphs"].as_array().ok_or_else(||invalid("frozen graph descriptors missing"))?
            .iter().filter(|g|g["role"]!="validator").cloned().collect());
        request["export_manifest"]=p["collection"]["manifest"].clone();
        request["verifier_sources"]=serde_json::to_value(compiled_frozen_pc_verifier_pins_v3()?).map_err(|e|invalid(e.to_string()))?;
        request["expected_frozen_pc_sha256"]=serde_json::json!(frozen.frozen_pc_sha256());
        let request_raw=serde_json::to_vec(&request).map_err(|e|invalid(e.to_string()))?;
        check(request_raw.len()<=128<<10,"frozen actual content request exceeds original bound")?;
        let request_pin=publish_json(owner,&declaration.private_cwd,"frozen-pc-content-request.json",&request_raw,&mut credit,&mut files)?;
        declaration.intake.request_root=declaration.private_cwd.clone();
        declaration.intake.request=ArtifactRef {path:"frozen-pc-content-request.json".into(),sha256:request_pin.sha256.clone(),bytes:request_pin.bytes,
            source:"original-source-owned-generation-export".into(),license:registered_model_license};
        declaration.intake.artifact_directory=declaration.private_cwd.join("inputs/frozen");
        declaration.validate()?;
        let declaration_raw=serde_json::to_vec(&declaration).map_err(|e|invalid(e.to_string()))?;
        let pin=publish_json(owner,&declaration.private_cwd,"frozen-action-input-template-v3.json",&declaration_raw,&mut credit,&mut files)?;
        let path=pin.path.clone();let sha=pin.sha256.clone();
        super::selfplay_cli::scan_staging(&managed,declaration.max_staging_bytes,until,cancel)?;
        Ok((path,sha,config,lock,declaration))
    })();
    match result {
        Ok((declaration_path, declaration_file_sha256, config, lock, declaration)) => {
            Ok(PreparedFrozenActionStartupInputsV3 {
                declaration_path,
                declaration_file_sha256,
                config,
                lock,
                frozen,
                template_pin,
                template: template_file.expect("successful original template retained"),
                files,
                cwd: cwd.expect("successful original cwd retained"),
                preparation_until: until,
                declaration,
            })
        }
        Err(error) => Err(Box::new(FrozenActionPreparationFailureV3 {
            error,
            frozen,
            template_pin,
            charged_read_bytes: account.charged.load(Ordering::Acquire),
            read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
            prepared: None,
            numerical: None,
            template: template_file,
            files,
            cwd,
        })),
    }
}

#[cfg(target_os = "linux")]
fn validate_original_managed_scope(
    original_root: &Path,
    original_cap: u64,
    declared_root: &Path,
    private_cwd: &Path,
    declared_cap: u64,
) -> Result<(), ArenaError> {
    use super::training_process::absolute_safe;
    check(
        original_root == declared_root
            && original_cap == declared_cap
            && (1..=MAX_STAGING).contains(&original_cap)
            && absolute_safe(declared_root)
            && absolute_safe(private_cwd)
            && private_cwd != original_root
            && private_cwd.starts_with(original_root),
        "frozen preparation must retain the original trainer staging root, cap and private child scope before any write",
    )
}

#[cfg(target_os = "linux")]
fn validate_original_staging_owner(
    trainer: &super::VerifiedGenerationTrainingWindowV3,
    named: &cap_std::fs::Dir,
) -> Result<(), ArenaError> {
    use super::training_process::{FileIdentity, directory_file};
    let original = trainer.held_staging_directory_file()?;
    let current = directory_file(named)?;
    check(
        FileIdentity::from(
            &original
                .metadata()
                .map_err(|_| invalid("frozen original staging identity"))?,
        )
        .same_inode(FileIdentity::from(
            &current
                .metadata()
                .map_err(|_| invalid("frozen named staging identity"))?,
        )),
        "frozen staging name no longer belongs to the original trainer directory owner",
    )
}

#[cfg(all(test, target_os = "linux"))]
mod managed_scope_tests {
    use super::*;
    #[test]
    fn foreign_root_cap_or_private_cwd_is_rejected_before_preparation_writes() {
        // Pure path/limit checks only. These declarations never construct a
        // trainer, directory capability, native admission or export owner.
        let original = Path::new("/managed/original");
        let private = Path::new("/managed/original/source/private");
        validate_original_managed_scope(original, MAX_STAGING, original, private, MAX_STAGING)
            .unwrap();
        for (root, cwd, cap) in [
            (
                Path::new("/managed/foreign"),
                Path::new("/managed/foreign/private"),
                MAX_STAGING,
            ),
            (original, private, MAX_STAGING - 1),
            (original, private, MAX_STAGING + 1),
            (original, Path::new("/managed/foreign/private"), MAX_STAGING),
            (original, original, MAX_STAGING),
            (
                original,
                Path::new("/managed/original/../foreign/private"),
                MAX_STAGING,
            ),
        ] {
            assert!(
                validate_original_managed_scope(original, MAX_STAGING, root, cwd, cap).is_err()
            );
        }
    }
}

#[cfg(target_os = "linux")]
fn create_frozen_directories(cwd: &cap_std::fs::Dir) -> Result<(), ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::PermissionsExt;
    let inputs = cwd
        .open_dir_nofollow("inputs")
        .map_err(|_| invalid("frozen inputs owner absent"))?;
    inputs
        .create_dir("frozen")
        .map_err(|_| invalid("frozen snapshot namespace already exists or creation failed"))?;
    let frozen = inputs
        .open_dir_nofollow("frozen")
        .map_err(|_| invalid("frozen snapshot namespace linked"))?;
    super::training_process::directory_file(&frozen)?
        .set_permissions(std::fs::Permissions::from_mode(0o700))
        .map_err(|_| invalid("frozen snapshot mode"))?;
    frozen
        .create_dir("collection")
        .map_err(|_| invalid("frozen collection namespace creation"))?;
    Ok(())
}
#[cfg(target_os = "linux")]
fn copy_new(
    cwd: &cap_std::fs::Dir,
    base: &Path,
    relative: &str,
    source: &RetainedFile,
    credit: &mut super::training_process::ReadCredit<'_>,
    files: &mut Vec<RetainedFile>,
) -> Result<GenerationTrainingFilePinV3, ArenaError> {
    use super::training_process::{FileIdentity, same_named};
    use cap_std::fs::OpenOptionsExt;
    use std::io::Write;
    use std::os::unix::fs::FileExt;
    let before = same_named(&source.pin, &source.file)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    let target = cwd
        .open_with(relative, &options)
        .map_err(|_| invalid("frozen source snapshot exclusive create"))?
        .into_std();
    let pin = GenerationTrainingFilePinV3 {
        path: base.join(relative),
        sha256: source.pin.sha256.clone(),
        bytes: source.pin.bytes,
    };
    // Retain the created descriptor before the first byte is written. On
    // failure its pin is an expected identity, never a verified publication.
    files.push(RetainedFile {
        pin: pin.clone(),
        file: target,
    });
    let retained_index = files.len() - 1;
    let target = &mut files[retained_index].file;
    let mut block = [0u8; 65536];
    let mut offset = 0;
    while offset < source.pin.bytes {
        let count = (source.pin.bytes - offset).min(block.len() as u64) as usize;
        credit.take(count as u64)?;
        let read = source
            .file
            .read_at(&mut block[..count], offset)
            .map_err(|_| invalid("frozen source snapshot read"))?;
        check(read > 0, "frozen source snapshot premature EOF")?;
        target
            .write_all(&block[..read])
            .map_err(|_| invalid("frozen source snapshot write"))?;
        offset += read as u64;
    }
    target
        .sync_all()
        .map_err(|_| invalid("frozen source snapshot sync"))?;
    check(
        before
            == FileIdentity::from(
                &source
                    .file
                    .metadata()
                    .map_err(|_| invalid("frozen source readback metadata"))?,
            )
            && before == same_named(&source.pin, &source.file)?,
        "frozen original source changed during snapshot",
    )?;
    retain_readonly_output(files, retained_index, &pin)?;
    super::training_process::verify_bytes(&files[retained_index].file, &pin, false, credit)?;
    Ok(pin)
}
#[cfg(target_os = "linux")]
fn publish_json(
    cwd: &cap_std::fs::Dir,
    base: &Path,
    name: &str,
    raw: &[u8],
    credit: &mut super::training_process::ReadCredit<'_>,
    files: &mut Vec<RetainedFile>,
) -> Result<GenerationTrainingFilePinV3, ArenaError> {
    use cap_std::fs::OpenOptionsExt;
    use sha2::{Digest, Sha256};
    use std::io::Write;
    check(
        !raw.is_empty() && raw.len() as u64 <= MAX_JSON,
        "frozen metadata publication bound",
    )?;
    credit.take(0)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    let file = cwd
        .open_with(name, &options)
        .map_err(|_| invalid("frozen metadata exclusive create"))?
        .into_std();
    let pin = GenerationTrainingFilePinV3 {
        path: base.join(name),
        sha256: format!("{:x}", Sha256::digest(raw)),
        bytes: raw.len() as u64,
    };
    // Partial outputs belong to the failure owner too. A failed write/readback
    // cannot drop the only open descriptor and later pass as a successful pin.
    files.push(RetainedFile {
        pin: pin.clone(),
        file,
    });
    let retained_index = files.len() - 1;
    let file = &mut files[retained_index].file;
    file.write_all(raw)
        .map_err(|_| invalid("frozen metadata publication write"))?;
    file.sync_all()
        .map_err(|_| invalid("frozen metadata publication sync"))?;
    retain_readonly_output(files, retained_index, &pin)?;
    super::training_process::verify_bytes(&files[retained_index].file, &pin, false, credit)?;
    Ok(pin)
}

#[cfg(target_os = "linux")]
fn retain_readonly_output(
    files: &mut [RetainedFile],
    index: usize,
    pin: &GenerationTrainingFilePinV3,
) -> Result<(), ArenaError> {
    use super::training_process::{FileIdentity, open_file};
    use std::os::unix::fs::PermissionsExt;
    files[index]
        .file
        .set_permissions(std::fs::Permissions::from_mode(0o400))
        .map_err(|_| invalid("frozen output read-only sealing"))?;
    let original = FileIdentity::from(
        &files[index]
            .file
            .metadata()
            .map_err(|_| invalid("frozen created output identity"))?,
    );
    let readonly = open_file(&pin.path)?;
    check(
        original.same_inode(FileIdentity::from(
            &readonly
                .metadata()
                .map_err(|_| invalid("frozen read-only output identity"))?,
        )),
        "frozen output name no longer refers to the created descriptor",
    )?;
    files[index].file = readonly;
    Ok(())
}
