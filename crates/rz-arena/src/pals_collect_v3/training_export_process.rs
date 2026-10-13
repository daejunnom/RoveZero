//! Original actual-checkpoint export transport, separate from training and GPU
//! acceptance. A parsed registration never reconstructs the original trainer.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use super::training_process::{
    CheckpointPin, GenerationTrainingFilePinV3, GenerationTrainingLaunchV3, TrainingRegistration,
    VerifiedGenerationTrainingWindowV3,
};
use super::{check, invalid};
use crate::{ArenaError, process::OriginalProcessOutput};
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(target_os = "linux")]
use std::sync::atomic::Ordering;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

pub type GenerationExportLaunchV3 = GenerationTrainingLaunchV3;
const REGISTRATION: &str = "rz-pals-generation-export-registration/1";
const STARTUP: &str = "rz-pals-original-generation-export-startup/1";
const RESULT: &str = "rz-pals-generation-export-cli-window/1";
const CHECKPOINT: &str = "rz-pals-actual-generation-checkpoint/1";
const PACKAGE: &str = "experiments/model-research/pals/src";
const COMPONENTS: [&str; 8] = [
    "trainer",
    "checkpoint",
    "loss_recipe",
    "model_reference",
    "model_encoding",
    "data_v3",
    "collector",
    "model_export",
];
const MAX_JSON: u64 = 16 * 1024 * 1024;
const MAX_FILE: u64 = 1024 * 1024 * 1024;
const BOOTSTRAP: &str = "import sys,runpy; package=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0,package); sys.argv[0]=module; runpy.run_module(module,run_name='__main__',alter_sys=True)";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GenerationExportKindV3 {
    #[serde(rename = "frozen_pc_collection")]
    FrozenPcCollection,
    #[serde(rename = "g1_actual_generation")]
    G1ActualGeneration,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExportRegistration {
    schema: String,
    export_id: String,
    kind: GenerationExportKindV3,
    training_registration: TrainingRegistration,
    original_training_registration_sha256: String,
    trainer_receipt: GenerationTrainingFilePinV3,
    original_training_execution_id: String,
    checkpoint: GenerationTrainingFilePinV3,
    rules_profile: GenerationTrainingFilePinV3,
    export_source_components: BTreeMap<String, String>,
    model_epoch: u64,
    index_root: PathBuf,
    output_directory: PathBuf,
    receipt_path: PathBuf,
    max_wall_ns: u64,
    max_output_bytes: u64,
}
impl ExportRegistration {
    fn loading(&self) -> TrainingRegistration {
        let mut value = self.training_registration.clone();
        value.window_id = self.export_id.clone();
        value.index_root = self.index_root.clone();
        value.receipt_path = self.receipt_path.clone();
        value.max_wall_ns = self.max_wall_ns;
        value.checkpoint = CheckpointPin {
            artifact: self.checkpoint.clone(),
            domain: CHECKPOINT.into(),
            mode: "resume".into(),
        };
        value
    }
    fn validate(&self, trainer: &VerifiedGenerationTrainingWindowV3) -> Result<(), ArenaError> {
        use super::training_process::{absolute_safe, plain_name};
        for (pin, maximum) in [
            (&self.trainer_receipt, MAX_JSON),
            (&self.rules_profile, MAX_JSON),
            (&self.checkpoint, MAX_FILE),
        ] {
            pin.validate(maximum)?;
        }
        let source =
            rz_experiments::compiled_source_components_v5().map_err(|e| invalid(e.to_string()))?;
        let root = &self.training_registration.staging_root;
        check(
            self.schema == REGISTRATION
                && plain_name(&self.export_id)
                && self.training_registration == *trainer.registration()
                && self.original_training_registration_sha256 == trainer.registration_sha256()
                && self.original_training_execution_id == trainer.execution_id()
                && trainer.receipt_file_pin() == Some(&self.trainer_receipt)
                && trainer.durable_resume_available()
                && self
                    .export_source_components
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    == BTreeSet::from(COMPONENTS)
                && self
                    .export_source_components
                    .iter()
                    .all(|(name, pin)| source.get(name) == Some(pin))
                && self
                    .training_registration
                    .source_components
                    .iter()
                    .all(|(name, pin)| self.export_source_components.get(name) == Some(pin))
                && (1..=3_600_000_000_000).contains(&self.max_wall_ns)
                && self.model_epoch > 0
                && self.max_output_bytes == self.training_registration.max_output_bytes
                && [&self.index_root, &self.output_directory, &self.receipt_path]
                    .into_iter()
                    .all(|path| absolute_safe(path) && path != root && path.starts_with(root))
                && self.receipt_path != self.training_registration.receipt_path
                && self.receipt_path.extension().is_some_and(|x| x == "json")
                && self.checkpoint.path.parent()
                    == Some(self.training_registration.checkpoint_directory.as_path()),
            "export exact original trainer/source/managed input and output registration",
        )?;
        let expected = if self.kind == GenerationExportKindV3::FrozenPcCollection {
            [2000, 2000, 0]
        } else {
            [2000, 2000, 1000]
        };
        let ledger = &trainer.receipt()["window"]["ledger"];
        check(
            ["proposer", "critic", "validator"]
                .iter()
                .zip(expected)
                .all(|(name, count)| ledger["completed"][*name] == count)
                && ledger["pending_microbatches"] == 0,
            "export requires exact original durable4000PC or5000G1 terminal boundary",
        )?;
        match self.kind {
            GenerationExportKindV3::FrozenPcCollection => check(
                self.training_registration.phase == "pc"
                    && self.training_registration.datasets.v.is_none()
                    && trainer.receipt()["status"] == "await_actual_v_collection",
                "frozen collection original P/C boundary",
            ),
            GenerationExportKindV3::G1ActualGeneration => check(
                self.training_registration.phase == "v"
                    && self.training_registration.datasets.v.is_some()
                    && trainer.actions().is_some()
                    && self.training_registration.model_epoch == Some(self.model_epoch)
                    && trainer.receipt()["status"] == "generation_updates_complete",
                "G1 original V terminal corpus and epoch",
            ),
        }
    }
}

/// No public constructor/decoder. The original child and all actual output
/// descriptors remain owned together with the original opaque trainer chain.
pub struct VerifiedGenerationExportV3 {
    registration: ExportRegistration,
    receipt: Value,
    parent_receipt: Value,
    process: OriginalProcessOutput,
    trainer: Arc<VerifiedGenerationTrainingWindowV3>,
    // Keep the original launch and its retained registration/interpreter owners
    // alive until this verified export owner is dropped.
    _launch: GenerationExportLaunchV3,
    #[cfg(target_os = "linux")]
    held: super::training_process::LinuxHeld,
}
impl VerifiedGenerationExportV3 {
    pub fn receipt(&self) -> &Value {
        &self.receipt
    }
    pub fn semantic_receipt(&self) -> &Value {
        &self.receipt["semantic_receipt"]
    }
    pub fn control_inventory_seals(&self) -> &Value {
        &self.receipt["control_inventory_seals"]
    }
    pub fn parent_receipt(&self) -> &Value {
        &self.parent_receipt
    }
    pub fn process(&self) -> &OriginalProcessOutput {
        &self.process
    }
    pub fn trainer(&self) -> &Arc<VerifiedGenerationTrainingWindowV3> {
        &self.trainer
    }
    pub fn directory(&self) -> &Path {
        &self.registration.output_directory
    }
    pub fn kind(&self) -> GenerationExportKindV3 {
        self.registration.kind
    }
    pub fn frozen_pc_sha256(&self) -> &str {
        self.receipt["frozen_pc_sha256"]
            .as_str()
            .expect("verified named frozen P/C digest")
    }
    pub fn model_epoch(&self) -> u64 {
        self.registration.model_epoch
    }
    pub fn receipt_file_pin(&self) -> Option<&GenerationTrainingFilePinV3> {
        #[cfg(target_os = "linux")]
        {
            self.held.receipt.as_ref().map(|file| &file.pin)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
    #[cfg(target_os = "linux")]
    pub fn held_files(&self) -> impl Iterator<Item = (&GenerationTrainingFilePinV3, &File)> {
        self.held
            .checkpoints
            .iter()
            .map(|file| (&file.pin, &file.file))
    }
    #[cfg(target_os = "linux")]
    pub fn collection_manifest_pin(&self) -> &GenerationTrainingFilePinV3 {
        &self
            .held
            .checkpoints
            .iter()
            .find(|file| file.pin.path == self.directory().join("collection/export.json"))
            .expect("verified original collection manifest")
            .pin
    }
    #[cfg(target_os = "linux")]
    pub fn collection_graph_pins(&self) -> impl Iterator<Item = &GenerationTrainingFilePinV3> {
        self.held
            .checkpoints
            .iter()
            .filter(|file| {
                file.pin.path.parent() == Some(self.directory().join("collection").as_path())
                    && file.pin.path.extension().is_some_and(|x| x == "onnx")
            })
            .map(|file| &file.pin)
    }
    #[cfg(target_os = "linux")]
    pub fn semantic_receipt_file_pin(&self) -> &GenerationTrainingFilePinV3 {
        let name = if self.kind() == GenerationExportKindV3::FrozenPcCollection {
            "frozen-pc-collection-v2.json"
        } else {
            "actual-generation.json"
        };
        &self
            .held
            .checkpoints
            .iter()
            .find(|file| file.pin.path == self.directory().join(name))
            .expect("verified original semantic receipt artifact")
            .pin
    }
    #[cfg(target_os = "linux")]
    pub fn deployment_manifest_pin(&self) -> Option<&GenerationTrainingFilePinV3> {
        self.held
            .checkpoints
            .iter()
            .find(|file| file.pin.path == self.directory().join("deployment/export.json"))
            .map(|file| &file.pin)
    }
    #[cfg(target_os = "linux")]
    pub fn deployment_graph_pins(&self) -> impl Iterator<Item = &GenerationTrainingFilePinV3> {
        self.held
            .checkpoints
            .iter()
            .filter(|file| {
                file.pin.path.parent() == Some(self.directory().join("deployment").as_path())
                    && file.pin.path.extension().is_some_and(|x| x == "onnx")
            })
            .map(|file| &file.pin)
    }
    #[cfg(target_os = "linux")]
    pub fn collection_control_inventory_pin(&self) -> &GenerationTrainingFilePinV3 {
        &self
            .held
            .checkpoints
            .iter()
            .find(|file| {
                file.pin.path
                    == self
                        .directory()
                        .join("collection/cuda-control-inventory.json")
            })
            .expect("verified original collection control inventory")
            .pin
    }
    #[cfg(target_os = "linux")]
    pub fn deployment_control_inventory_pin(&self) -> Option<&GenerationTrainingFilePinV3> {
        self.held
            .checkpoints
            .iter()
            .find(|file| {
                file.pin.path
                    == self
                        .directory()
                        .join("deployment/cuda-control-inventory.json")
            })
            .map(|file| &file.pin)
    }
}

pub struct GenerationExportFailureV3 {
    pub error: ArenaError,
    pub process: Option<OriginalProcessOutput>,
    pub receipt: Option<Value>,
    pub trainer: Arc<VerifiedGenerationTrainingWindowV3>,
    pub launch: GenerationExportLaunchV3,
    pub charged_read_bytes: u64,
    pub read_usage_uncertain: bool,
    /// Raw fixed-name descriptor custody; never a verified export, numerical
    /// acceptance, physical completion or resume capability.
    pub artifact_preservation: Option<Value>,
    #[cfg(target_os = "linux")]
    // Failed/uncertain exports retain all original descriptors and partial
    // artifacts until the caller releases this failure custody.
    _held: Option<super::training_process::LinuxHeld>,
}
impl std::fmt::Debug for GenerationExportFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationExportFailureV3")
            .field("error", &self.error)
            .field("process", &self.process)
            .field("receipt_preserved", &self.receipt.is_some())
            .finish_non_exhaustive()
    }
}

/// Integration owns a single GPU execution lease. Failure custody is retained
/// until original group and physical completion are resolved; no implicit retry.
pub fn run_generation_export_original_v3(
    launch: GenerationExportLaunchV3,
    trainer: Arc<VerifiedGenerationTrainingWindowV3>,
    cancel: &AtomicBool,
) -> Result<VerifiedGenerationExportV3, Box<GenerationExportFailureV3>> {
    #[cfg(target_os = "linux")]
    {
        run_linux(launch, trainer, cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cancel;
        Err(Box::new(GenerationExportFailureV3 {
            error: invalid("original export requires Linux"),
            process: None,
            receipt: None,
            trainer,
            launch,
            charged_read_bytes: 0,
            read_usage_uncertain: false,
            artifact_preservation: None,
        }))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildSummary {
    schema: String,
    status: String,
    receipt: GenerationTrainingFilePinV3,
    actual_export_executed: bool,
    physical_completion: String,
    optimizer_updates_executed: u64,
    arena_accepted: bool,
}

#[cfg(target_os = "linux")]
fn failure_output_targets(
    registration: &ExportRegistration,
    execution_id: &str,
) -> Result<Vec<(PathBuf, u64)>, ArenaError> {
    let mut targets = Vec::new();
    let mut json_target = |path: PathBuf| -> Result<(), ArenaError> {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("export original output basename"))?;
        let partial = path.with_file_name(format!("{name}.partial"));
        targets.push((path, MAX_JSON));
        targets.push((partial, MAX_JSON));
        Ok(())
    };
    json_target(registration.receipt_path.clone())?;
    for name in ["generation-export-owner.json", "export-incomplete.json"] {
        json_target(registration.output_directory.join(name))?;
    }
    match registration.kind {
        GenerationExportKindV3::FrozenPcCollection => {
            for name in ["frozen-pc-collection.json", "frozen-pc-collection-v2.json"] {
                json_target(registration.output_directory.join(name))?;
            }
        }
        GenerationExportKindV3::G1ActualGeneration => {
            json_target(registration.output_directory.join("actual-generation.json"))?;
        }
    }
    let kinds: &[&str] = if registration.kind == GenerationExportKindV3::FrozenPcCollection {
        &["collection"]
    } else {
        &["collection", "deployment"]
    };
    for kind in kinds {
        for name in ["pals-artifact-owner.json", "export.json"] {
            json_target(registration.output_directory.join(kind).join(name))?;
        }
    }
    // Direct exclusive writers have no separately named partial publication.
    targets.push((
        registration
            .training_registration
            .staging_root
            .join(format!("export-rules-{execution_id}.json")),
        MAX_JSON,
    ));
    for kind in kinds {
        targets.push((
            registration
                .output_directory
                .join(kind)
                .join("cuda-control-inventory.json"),
            MAX_JSON,
        ));
        for name in ["public_memory.onnx", "shared_pc_if.onnx"] {
            targets.push((
                registration.output_directory.join(kind).join(name),
                MAX_FILE,
            ));
            targets.push((
                registration
                    .output_directory
                    .join(kind)
                    .join(format!("{name}.partial")),
                MAX_FILE,
            ));
        }
    }
    for name in ["role_validator.onnx", "role_validator.onnx.partial"] {
        targets.push((
            registration.output_directory.join("collection").join(name),
            MAX_FILE,
        ));
    }
    if registration.kind == GenerationExportKindV3::G1ActualGeneration {
        for name in ["checkpoint.pt", "checkpoint.pt.partial"] {
            targets.push((registration.output_directory.join(name), MAX_FILE));
        }
    }
    check(
        targets.len() <= 128,
        "export bounded original known output names",
    )?;
    Ok(targets)
}

#[cfg(target_os = "linux")]
fn semantic_files(
    receipt: &Value,
    registration: &ExportRegistration,
    held: &super::training_process::LinuxHeld,
) -> Result<(), ArenaError> {
    let semantic = &receipt["semantic_receipt"];
    let checkpoint = &semantic["checkpoint"];
    check(
        checkpoint["sha256"] == registration.checkpoint.sha256
            && checkpoint["bytes"] == registration.checkpoint.bytes
            && checkpoint["pending_microbatches"] == 0
            && checkpoint["frozen_pc_sha256"] == receipt["frozen_pc_sha256"]
            && checkpoint["frozen_pc_model_epoch"] == registration.model_epoch
            && semantic["numeric_status"] == "not_run"
            && semantic["cuda_status"] == "not_run",
        "export actual checkpoint provenance is separate from numerical or GPU acceptance",
    )?;
    match registration.kind {
        GenerationExportKindV3::FrozenPcCollection => check(
            semantic["schema"] == "rz-pals-frozen-pc-collection-export/2"
                && semantic["eligibility"] == "frozen_pc_collection_only"
                && checkpoint["phase"] == "v"
                && checkpoint["completed_updates"]
                    == serde_json::json!({"proposer":2000,"critic":2000,"validator":0})
                && semantic["validator_present"] == true
                && semantic["validator_training_updates"] == 0
                && semantic["deployment_exported"] == false
                && semantic["native_loaded_assets_verified"] == false
                && semantic["arena_eligible"] == false,
            "frozen4000 cannot masquerade as G1 or a loaded native model",
        )?,
        GenerationExportKindV3::G1ActualGeneration => check(
            semantic["schema"] == "rz-pals-actual-generation-export/1"
                && checkpoint["phase"] == "complete"
                && checkpoint["completed_updates"]
                    == serde_json::json!({"proposer":2000,"critic":2000,"validator":1000})
                && semantic["deployment"]["validator_present"] == false
                && semantic["arena_accepted"] == false,
            "G1 export retains real5000 provenance and V-free deployment without arena acceptance",
        )?,
    }
    let kinds: &[&str] = if registration.kind == GenerationExportKindV3::FrozenPcCollection {
        &["collection"]
    } else {
        &["collection", "deployment"]
    };
    for kind in kinds {
        let descriptors = &semantic[*kind];
        let manifest = &descriptors["manifest"];
        check(
            manifest["file"] == format!("{kind}/export.json"),
            "original export exact manifest relative path",
        )?;
        let graphs = descriptors["graphs"]
            .as_array()
            .ok_or_else(|| invalid("export graph pin inventory"))?;
        check(
            (2..=4).contains(&graphs.len()),
            "original export bounded graph bank",
        )?;
        let mut names = BTreeSet::new();
        for descriptor in std::iter::once(manifest).chain(graphs) {
            let logical = descriptor["file"]
                .as_str()
                .ok_or_else(|| invalid("original export artifact logical path"))?;
            let relative = Path::new(logical);
            check(
                !relative.is_absolute()
                    && relative
                        .components()
                        .all(|part| matches!(part, std::path::Component::Normal(_)))
                    && relative.starts_with(*kind)
                    && names.insert(logical),
                "original export bounded relative artifact path",
            )?;
            let path = registration.output_directory.join(relative);
            check(
                held.checkpoints.iter().any(|file| {
                    file.pin.path == path
                        && descriptor["sha256"] == file.pin.sha256
                        && descriptor["bytes"] == file.pin.bytes
                }),
                "original export semantic descriptor joins actual held artifact",
            )?;
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn control_inventory_files(
    receipt: &Value,
    registration: &ExportRegistration,
    held: &super::training_process::LinuxHeld,
    credit: &mut super::training_process::ReadCredit<'_>,
) -> Result<(), ArenaError> {
    use super::training_process as common;
    let expected: &[&str] = if registration.kind == GenerationExportKindV3::FrozenPcCollection {
        &["collection"]
    } else {
        &["collection", "deployment"]
    };
    let seals = receipt["control_inventory_seals"]
        .as_object()
        .ok_or_else(|| invalid("original export control inventory seals absent"))?;
    check(
        seals.keys().map(String::as_str).collect::<BTreeSet<_>>()
            == expected.iter().copied().collect(),
        "exact original actual export control inventory set",
    )?;
    let producer = held
        .sources
        .iter()
        .find(|file| {
            file.pin.path
                == held
                    .source_root
                    .join(PACKAGE)
                    .join("rz_pals_model/virtual_control_inventory.py")
        })
        .ok_or_else(|| invalid("original control inventory producer source not held"))?;
    for kind in expected {
        let seal = &seals[*kind];
        let descriptors = &receipt["semantic_receipt"][*kind];
        let relative = format!("{kind}/cuda-control-inventory.json");
        let file = held
            .checkpoints
            .iter()
            .find(|file| file.pin.path == registration.output_directory.join(&relative))
            .ok_or_else(|| invalid("actual newly exported graph control inventory not held"))?;
        check(
            file.pin.bytes <= 4 * 1024 * 1024
                && seal.as_object().is_some_and(|object| object.len() == 11)
                && seal["schema"] == "rz-pals-actual-export-control-inventory/1"
                && seal["kind"] == *kind
                && seal["manifest"] == descriptors["manifest"]
                && seal["graphs"] == descriptors["graphs"]
                && seal["inventory"]
                    == serde_json::json!({"file":relative,"sha256":file.pin.sha256,"bytes":file.pin.bytes})
                && seal["producer_source_sha256"] == producer.pin.sha256
                && seal["inventory_schema"] == "rovezero.pals-virtual-static-control-inventory.v1"
                && seal["scope"] == "read_only_serialized_graph_metadata_no_runtime"
                && seal["actual_provider_placement"] == "not_observed"
                && seal["cpu_allowlist"] == "not_created",
            "actual inventory seal joins original producer/source/manifest/graph/raw file pins",
        )?;
        let raw: Value =
            common::decode(&common::verify_bytes(&file.file, &file.pin, true, credit)?)?;
        check(
            raw["schema"] == seal["inventory_schema"]
                && raw["scope"] == seal["scope"]
                && raw["artifact_kind"] == "fresh"
                && raw["manifest_sha256"] == descriptors["manifest"]["sha256"]
                && raw["producer_source_sha256"] == producer.pin.sha256
                && raw["actual_provider_placement"] == "not_observed"
                && raw["cpu_allowlist"] == "not_created"
                && raw["total_nodes"] == seal["total_nodes"]
                && raw["total_nodes"]
                    .as_u64()
                    .is_some_and(|nodes| (1..=12000).contains(&nodes)),
            "actual serialized control inventory is not provider placement or CPU execution permission",
        )?;
        let graphs = raw["graphs"]
            .as_array()
            .ok_or_else(|| invalid("actual control inventory graphs absent"))?;
        let pins = descriptors["graphs"]
            .as_array()
            .ok_or_else(|| invalid("actual export graph descriptors absent"))?;
        check(
            graphs.len() == pins.len(),
            "actual inventory graph count differs from exact export",
        )?;
        for (graph, pin) in graphs.iter().zip(pins) {
            let logical = pin["file"]
                .as_str()
                .ok_or_else(|| invalid("exact actual export graph logical file"))?;
            let name = Path::new(logical)
                .file_name()
                .ok_or_else(|| invalid("exact actual graph basename"))?;
            check(
                graph["role"] == pin["role"]
                    && graph["sha256"] == pin["sha256"]
                    && graph["file"] == name.to_string_lossy().as_ref(),
                "actual inventory ordered graph source pins differ",
            )?;
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn run_linux(
    launch: GenerationExportLaunchV3,
    trainer: Arc<VerifiedGenerationTrainingWindowV3>,
    cancel: &AtomicBool,
) -> Result<VerifiedGenerationExportV3, Box<GenerationExportFailureV3>> {
    use super::training_process::{
        self as common, LinuxHeld, ObservedFile, ReadAccounting, ReadCredit,
    };
    use crate::process::{
        MAX_ORIGINAL_INPUT_BYTES, OriginalProcessInput, OriginalProcessWindow,
        OwnedArtifactTreeWatch, supervise_verified_input_in_original_window,
    };
    use sha2::{Digest as _, Sha256};
    use std::ffi::OsString;
    let mut held: Option<LinuxHeld> = None;
    let mut process = None;
    let mut receipt = None;
    let mut artifact_preservation = None;
    let account = ReadAccounting::new();
    let preserve_cancel = AtomicBool::new(false);
    let result = (|| -> Result<(ExportRegistration, Value), ArenaError> {
        launch.registration_pin.validate(MAX_JSON)?;
        launch
            .python_runtime
            .executable
            .validate(256 * 1024 * 1024)?;
        check(
            generation_sha(&launch.registration_sha256)
                && common::plain_name(&launch.execution_id)
                && (1..=8).contains(&launch.dependency_directories.len())
                && (1..=64).contains(&launch.runtime_module_pins.len())
                && (1..=3_600_000).contains(&launch.limits.wall_ms)
                && (1..=15_000).contains(&launch.limits.shutdown_grace_ms)
                && (1..=128 * 1024).contains(&launch.limits.max_output_bytes)
                && launch.limits.max_child_processes == 1
                && launch.max_read_bytes > 0
                && launch.max_read_bytes
                    <= super::training_process::GENERATION_TRAINING_MAX_READ_BYTES_V3,
            "export finite original launch/read/process registration",
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
                    <= Duration::from_secs(30),
            "export original finite execution and cleanup window",
        )?;
        let mut credit = ReadCredit {
            remaining: launch.max_read_bytes,
            maximum: launch.max_read_bytes,
            until: launch.execution_deadline,
            cancel,
            account: Arc::clone(&account),
        };
        let raw = common::verify_bytes(
            &launch.registration,
            &launch.registration_pin,
            true,
            &mut credit,
        )?;
        let registration: ExportRegistration = common::decode(&raw)?;
        registration.validate(&trainer)?;
        check(
            generation_digest(REGISTRATION, &registration).map_err(|e| invalid(e.to_string()))?
                == launch.registration_sha256
                && launch
                    .execution_deadline
                    .duration_since(launch.started)
                    .as_nanos()
                    <= u128::from(registration.max_wall_ns),
            "export exact original semantic startup and deadline",
        )?;
        let preservation_targets = failure_output_targets(&registration, &launch.execution_id)?;
        // Re-read the same original checkpoint descriptor, never locate a new
        // file and treat its JSON as the missing original trainer owner.
        let (checkpoint_pin, checkpoint_file) = trainer
            .checkpoint_files()
            .last()
            .ok_or_else(|| invalid("export original trainer has no held actual checkpoint"))?;
        check(
            *checkpoint_pin == registration.checkpoint,
            "export latest actual checkpoint owner/pin differs",
        )?;
        common::verify_bytes(checkpoint_file, checkpoint_pin, false, &mut credit)?;
        let (terminal_pin, terminal_file) = trainer
            .held_receipt_file()
            .ok_or_else(|| invalid("export original trainer terminal not held"))?;
        check(
            *terminal_pin == registration.trainer_receipt,
            "export original terminal owner pin differs",
        )?;
        let original_terminal: Value = common::decode(&common::verify_bytes(
            terminal_file,
            terminal_pin,
            true,
            &mut credit,
        )?)?;
        check(
            original_terminal == *trainer.receipt(),
            "export original held trainer receipt changed",
        )?;
        common::verify_bytes(
            &launch.interpreter,
            &launch.python_runtime.executable,
            false,
            &mut credit,
        )?;
        let loading = registration.loading();
        held = Some(common::prepare_root(&loading, &launch.execution_id)?);
        let owner = held.as_mut().expect("original export staging custody");
        let mut dependencies = Vec::new();
        let mut paths = BTreeSet::new();
        for path in &launch.dependency_directories {
            check(paths.insert(path), "export duplicate dependency root")?;
            dependencies.push(common::retain_directory(path.clone(), owner)?);
        }
        let dependencies = Value::Array(dependencies);
        for pin in &launch.runtime_module_pins {
            pin.validate(MAX_FILE)?;
            check(
                launch
                    .dependency_directories
                    .iter()
                    .any(|root| pin.path.starts_with(root)),
                "export actual trusted runtime module root",
            )?;
            common::hold_input(pin.clone(), false, owner, &mut credit)?;
        }
        for pin in [
            &loading.spec,
            &loading.datasets.pc.manifest,
            &registration.checkpoint,
            &registration.trainer_receipt,
            &registration.rules_profile,
        ] {
            common::hold_input(pin.clone(), false, owner, &mut credit)?;
        }
        if let Some(v) = &loading.datasets.v {
            common::hold_input(v.manifest.clone(), false, owner, &mut credit)?;
        }
        common::reverify_archives(
            trainer.series(),
            trainer.actions().map(Arc::as_ref),
            &mut credit,
        )?;
        common::install_sources_for_components(owner, &mut credit, &COMPONENTS)?;
        check(
            owner.sources.iter().any(|file| {
                file.pin
                    .path
                    .ends_with("rz_pals_model/generation_export_cli.py")
            }),
            "export fixed compiled source child is absent",
        )?;
        let (start_ns, execution_ns, whole_ns) = common::original_window_ticks(
            launch.started,
            launch.execution_deadline,
            launch.whole_deadline,
        )?;
        let startup = serde_json::json!({"schema":STARTUP,"execution_id":launch.execution_id,
            "registration_sha256":launch.registration_sha256,"registration":registration,
            "original_started_ns":start_ns,"execution_until_ns":execution_ns,"whole_until_ns":whole_ns,
            "dataset_pin_scope":"original_trainer_held_checkpoint_readback_only",
            "dependency_directories":dependencies,"python_runtime":launch.python_runtime});
        let raw = serde_json::to_vec(&startup).map_err(|e| invalid(e.to_string()))?;
        check(
            !raw.is_empty() && raw.len() <= MAX_ORIGINAL_INPUT_BYTES,
            "export exact bounded original stdin",
        )?;
        owner.startup = Some(common::write_owned_new(
            &loading
                .staging_root
                .join(format!("export-startup-{}.json", launch.execution_id)),
            &raw,
        )?);
        let startup_sha = owner
            .startup
            .as_ref()
            .expect("held startup")
            .pin
            .sha256
            .clone();
        common::revalidate(owner, &mut credit)?;
        let args = [
            OsString::from("-I"),
            OsString::from("-S"),
            OsString::from("-B"),
            OsString::from("-c"),
            OsString::from(BOOTSTRAP),
            owner.source_root.join(PACKAGE).into_os_string(),
            OsString::from("rz_pals_model.generation_export_cli"),
            OsString::from("--execute"),
            OsString::from("--original-export-child"),
            OsString::from("--startup-sha256"),
            OsString::from(startup_sha),
            OsString::from("--startup-bytes"),
            OsString::from(raw.len().to_string()),
            OsString::from("--execution-until-ns"),
            OsString::from(execution_ns.to_string()),
            OsString::from("--whole-until-ns"),
            OsString::from(whole_ns.to_string()),
        ];
        let watch = OwnedArtifactTreeWatch {
            max_total_bytes: registration.max_output_bytes,
            max_file_bytes: MAX_FILE,
            max_files: 4096,
            max_depth: 32,
        };
        let image_credit = 2 * launch.python_runtime.executable.bytes;
        credit.take(image_credit)?;
        let mut mapped_credit = ReadCredit {
            remaining: image_credit,
            maximum: image_credit,
            until: launch.execution_deadline,
            cancel,
            account: ReadAccounting::new(),
        };
        let mut loaded = |file: &File| {
            common::verify_bytes(
                file,
                &launch.python_runtime.executable,
                false,
                &mut mapped_credit,
            )
            .map(|_| ())
            .map_err(|_| ())
        };
        let cwd = common::directory_file(&owner.staging)?;
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
        credit.until = launch.whole_deadline;
        credit.cancel = &preserve_cancel;
        let original = process.as_ref().expect("original export process preserved");
        let preservation = common::preserve_original_outputs(
            &preservation_targets,
            None,
            &registration.receipt_path,
            owner,
            &mut credit,
            original,
        );
        if let Some(raw) = preservation.receipt_raw.as_ref() {
            receipt = common::decode::<Value>(raw).ok();
        }
        artifact_preservation = Some(preservation.report);
        // Original stdout/closure/status remain the primary error. Failed raw
        // preservation is recorded separately and cannot hide the actual child.
        let summary: ChildSummary = common::decode(&original.process().stdout)?;
        summary.receipt.validate(MAX_JSON)?;
        check(
            summary.schema == RESULT
                && summary.receipt.path == registration.receipt_path
                && !summary.arena_accepted
                && summary.optimizer_updates_executed == 0,
            "export original bounded terminal summary",
        )?;
        let timing = original.checked_timing()?;
        original.checked_loaded_image()?;
        check(
            original.transport_complete()
                && timing.original_started() == launch.started
                && timing.execution_deadline() == launch.execution_deadline
                && timing.whole_deadline() == launch.whole_deadline
                && summary.status == "exported"
                && summary.actual_export_executed
                && summary.physical_completion == "confirmed",
            "export original child closure/status/physical completion failed",
        )?;
        if let Some(error) = preservation.error {
            return Err(error);
        }
        let (file, raw) =
            common::read_observed_output(summary.receipt, MAX_JSON, &mut credit, true)?;
        owner.receipt = Some(file);
        receipt = Some(common::decode::<Value>(&raw)?);
        let decoded = receipt.as_ref().expect("original export receipt preserved");
        let observed = decoded["artifact_pins"]
            .as_array()
            .ok_or_else(|| invalid("export actual artifact readback missing"))?;
        check(
            !observed.is_empty() && observed.len() <= 128,
            "export finite actual output artifact set",
        )?;
        let mut names = BTreeSet::new();
        for artifact in observed {
            let pin: ObservedFile =
                serde_json::from_value(artifact.clone()).map_err(|e| invalid(e.to_string()))?;
            check(
                pin.path.starts_with(&registration.output_directory)
                    && pin.path != registration.output_directory
                    && names.insert(pin.path.clone()),
                "export original artifact path/duplicate",
            )?;
            let (file, _) = common::read_observed_output(pin.pin(), MAX_FILE, &mut credit, false)?;
            check(
                common::readback_matches(&pin, &file),
                "export actual output inode/readback differs",
            )?;
            owner.checkpoints.push(file);
        }
        let rules_snapshot: ObservedFile =
            serde_json::from_value(decoded["rules_profile_snapshot"].clone())
                .map_err(|e| invalid(e.to_string()))?;
        check(
            rules_snapshot.path
                == registration
                    .training_registration
                    .staging_root
                    .join(format!("export-rules-{}.json", launch.execution_id))
                && rules_snapshot.sha256 == registration.rules_profile.sha256
                && rules_snapshot.bytes == registration.rules_profile.bytes,
            "exact pinned read-only original Rules snapshot",
        )?;
        let (rules, _) =
            common::read_observed_output(rules_snapshot.pin(), MAX_JSON, &mut credit, false)?;
        check(
            common::readback_matches(&rules_snapshot, &rules)
                && rules
                    .file
                    .metadata()
                    .map_err(|_| invalid("Rules snapshot metadata"))?
                    .permissions()
                    .readonly(),
            "original Rules snapshot inode or read-only mode differs",
        )?;
        owner.inputs.push(rules);
        check(
            original.transport_complete()
                && timing.original_started() == launch.started
                && timing.execution_deadline() == launch.execution_deadline
                && timing.whole_deadline() == launch.whole_deadline
                && summary.status == "exported"
                && summary.actual_export_executed
                && summary.physical_completion == "confirmed"
                && decoded["schema"] == RESULT
                && decoded["status"] == "exported"
                && decoded["actual_export_executed"] == true
                && decoded["kind"]
                    == serde_json::to_value(registration.kind)
                        .map_err(|e| invalid(e.to_string()))?
                && decoded["registration_sha256"] == launch.registration_sha256
                && decoded["execution_id"] == launch.execution_id
                && decoded["original_training_execution_id"] == trainer.execution_id()
                && decoded["original_training_registration_sha256"]
                    == trainer.registration_sha256()
                && decoded["original_trainer_receipt"]
                    == serde_json::to_value(&registration.trainer_receipt)
                        .map_err(|e| invalid(e.to_string()))?
                && decoded["original_checkpoint"]
                    == serde_json::to_value(&registration.checkpoint)
                        .map_err(|e| invalid(e.to_string()))?
                && decoded["physical_completion"] == "confirmed"
                && decoded["optimizer_updates_executed"] == 0
                && decoded["neural_forwards_executed"] == 0
                && decoded["native_game_authority_issued"] == false
                && decoded["original_process_authority_issued"] == false
                && decoded["numeric_status"] == "not_run"
                && decoded["cuda_status"] == "not_run"
                && decoded["arena_accepted"] == false
                && decoded["parent_must_verify_process_and_artifacts"] == true
                && decoded["original_execution_until_ns"] == execution_ns
                && decoded["original_whole_until_ns"] == whole_ns
                && generation_sha(decoded["frozen_pc_sha256"].as_str().unwrap_or(""))
                && decoded["model_epoch"] == registration.model_epoch,
            "export original closure/checkpoint/no-optimizer/physical/scope agreement",
        )?;
        check(
            decoded["active_source"]["source_components"]
                == serde_json::to_value(&registration.export_source_components)
                    .map_err(|e| invalid(e.to_string()))?,
            "export independent complete eight-component source readback",
        )?;
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
                                    .join("rz_pals_model/generation_export_cli.py")
                                    .to_string_lossy()
                                    .as_ref()
                    })
                }),
            "export actual compiled original child entry point readback absent",
        )?;
        check(
            decoded["active_source"]["source_modules"]
                .as_array()
                .is_some_and(|modules| {
                    [
                        "rz_pals_model.virtual_control_inventory",
                        "rz_pals_model.cuda_control_inventory",
                    ]
                    .into_iter()
                    .all(|name| modules.iter().any(|module| module["module"] == name))
                }),
            "actual export control inventory original producer modules absent",
        )?;
        let mut common_readback = decoded.clone();
        common_readback["active_source"]["source_components"] =
            serde_json::to_value(&loading.source_components).map_err(|e| invalid(e.to_string()))?;
        common::validate_child_readback(&common_readback, &loading, &launch, owner, &dependencies)?;
        let semantic_name = if registration.kind == GenerationExportKindV3::FrozenPcCollection {
            "frozen-pc-collection-v2.json"
        } else {
            "actual-generation.json"
        };
        let semantic_file = owner
            .checkpoints
            .iter()
            .find(|file| file.pin.path == registration.output_directory.join(semantic_name))
            .ok_or_else(|| invalid("original semantic export receipt not held"))?;
        check(
            semantic_file.pin.bytes <= MAX_JSON,
            "bounded original semantic export JSON",
        )?;
        let semantic: Value = common::decode(&common::verify_bytes(
            &semantic_file.file,
            &semantic_file.pin,
            true,
            &mut credit,
        )?)?;
        check(
            semantic == decoded["semantic_receipt"],
            "original semantic export receipt bytes differ from child readback",
        )?;
        check(
            owner.checkpoints.iter().all(|file| {
                !file.pin.path.file_name().is_some_and(|name| {
                    name.to_string_lossy().ends_with(".partial") || name == "export-incomplete.json"
                })
            }),
            "partial export artifacts cannot enter verified export custody",
        )?;
        semantic_files(decoded, &registration, owner)?;
        control_inventory_files(decoded, &registration, owner, &mut credit)?;
        common::verify_bytes(
            &launch.registration,
            &launch.registration_pin,
            false,
            &mut credit,
        )?;
        common::verify_bytes(
            &launch.interpreter,
            &launch.python_runtime.executable,
            false,
            &mut credit,
        )?;
        common::verify_bytes(checkpoint_file, checkpoint_pin, false, &mut credit)?;
        common::verify_bytes(terminal_file, terminal_pin, false, &mut credit)?;
        common::reverify_archives(
            trainer.series(),
            trainer.actions().map(Arc::as_ref),
            &mut credit,
        )?;
        common::revalidate(owner, &mut credit)?;
        owner.consumed_read_bytes = account.charged.load(Ordering::Acquire);
        let parent = serde_json::json!({"schema":"rz-pals-original-generation-export-parent/1", "execution_id":launch.execution_id,
            "registration_sha256":launch.registration_sha256,"registration_file":launch.registration_pin,
            "startup":owner.startup.as_ref().expect("original startup").pin,"launcher_sha256":super::hex(&Sha256::digest(BOOTSTRAP.as_bytes())),
            "process":original.process().receipt,"transport":original.observation(),"original_trainer_execution_id":trainer.execution_id(),
            "original_trainer_and_game_action_owners_retained":true,"charged_read_bytes":owner.consumed_read_bytes,
            "read_scope":"conservative_original_reserved_read_credit","numeric_acceptance":false,"arena_accepted":false,
            "full_mapped_cuda_runtime_attestation":false,"sole_gpu_lease_observed_here":false,"automatic_retry":false});
        Ok((registration, parent))
    })();
    if let Some(owner) = held.as_mut() {
        owner.consumed_read_bytes = account.charged.load(Ordering::Acquire);
    }
    match result {
        Ok((registration, parent_receipt)) => Ok(VerifiedGenerationExportV3 {
            registration,
            receipt: receipt.expect("successful export receipt"),
            parent_receipt,
            process: process.expect("original export process"),
            trainer,
            _launch: launch,
            held: held.expect("successful original artifact custody"),
        }),
        Err(error) => Err(Box::new(GenerationExportFailureV3 {
            error,
            process,
            receipt,
            trainer,
            launch,
            _held: held,
            charged_read_bytes: account.charged.load(Ordering::Acquire),
            read_usage_uncertain: account.uncertain.load(Ordering::Acquire),
            artifact_preservation,
        })),
    }
}
