//! Explicit historical training input admission. A recovered publication is
//! durable data evidence; it never recreates PublishedRoventRecord, a native
//! ticket/run witness, OriginalProcessOutput or a past physical completion.
//! This module is intentionally separate from the existing live /1 codecs.

use super::*;
use crate::pals_collect_v3::training_process::{
    FileIdentity, GenerationTrainingFilePinV3, ReadAccounting, ReadCredit, open_file, verify_bytes,
};
use crate::rovent_collect::{
    RoventCorpusPublication, RoventPublicationFile, verify_historical_record_evidence,
};
use rz_experiments::rovent_corpus::{MAX_CHUNK_BYTES, RoventCorpusReader};
use std::{
    path::{Component, Path},
    sync::atomic::{AtomicU64, Ordering},
};

pub const ROVENT_HISTORICAL_DATASET: &str = "rz-rovent-historical-dataset-admission/1";
pub const ROVENT_RECOVERY_REGISTRATION: &str = "rz-rovent-recovered-training-registration/1";
pub const ROVENT_RECOVERY_STARTUP: &str = "rz-rovent-recovered-training-startup/1";
pub const ROVENT_RECOVERED_CHECKPOINT: &str = "rz-rovent-recovered-checkpoint-admission/1";
pub const ROVENT_RECOVERY_SOURCE_KIND: &str = "historical_exact_pins";
const MAX_METADATA_BYTES: u64 = 16 << 20;

pub struct RoventRecoveryReadBudget {
    maximum: u64,
    charged: AtomicU64,
    uncertain: AtomicBool,
    until: Instant,
    cancel: Arc<AtomicBool>,
}
impl RoventRecoveryReadBudget {
    pub fn new(maximum: u64, until: Instant, cancel: Arc<AtomicBool>) -> Result<Self, ArenaError> {
        check(
            maximum > 0 && maximum <= MAX_TRAINER_OUTPUT_BYTES * 260 && Instant::now() < until,
            "finite original recovery read credit/deadline required",
        )?;
        Ok(Self {
            maximum,
            charged: AtomicU64::new(0),
            uncertain: AtomicBool::new(false),
            until,
            cancel,
        })
    }
    pub(super) fn reserve(&self, bytes: u64) -> Result<(), ArenaError> {
        check(
            Instant::now() < self.until && !self.cancel.load(Ordering::Acquire),
            "historical admission cancelled/expired",
        )?;
        self.charged
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(bytes).filter(|next| *next <= self.maximum)
            })
            .map_err(|_| {
                invalid("historical read credit exhausted; failed reservations are not refunded")
            })?;
        Ok(())
    }
    pub fn observation(&self) -> Value {
        json!({"charged_bytes":self.charged.load(Ordering::Acquire),"charged_bytes_scope":"conservative_reserved_upper_bound","unknown":self.uncertain.load(Ordering::Acquire),"maximum_bytes":self.maximum})
    }
    pub(super) fn read(
        &self,
        pin: &RoventTrainingFilePin,
        maximum: u64,
        retain: bool,
    ) -> Result<(HeldHistoricalFile, Vec<u8>), ArenaError> {
        let legacy = GenerationTrainingFilePinV3 {
            path: pin.path.clone(),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        legacy.validate(maximum)?;
        self.reserve(pin.bytes)?;
        let file = open_file(&pin.path).inspect_err(|_| {
            self.uncertain.store(true, Ordering::Release);
        })?;
        let raw = self.verify_reserved(&file, pin, retain)?;
        let identity = FileIdentity::from(&file.metadata().map_err(|_| {
            self.uncertain.store(true, Ordering::Release);
            invalid("historical held file metadata")
        })?);
        Ok((
            HeldHistoricalFile {
                file,
                pin: pin.clone(),
                identity,
            },
            raw,
        ))
    }
    pub(super) fn verify_held(
        &self,
        file: &File,
        pin: &RoventTrainingFilePin,
        retain: bool,
    ) -> Result<Vec<u8>, ArenaError> {
        GenerationTrainingFilePinV3 {
            path: pin.path.clone(),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        }
        .validate(MAX_TRAINER_OUTPUT_BYTES)?;
        self.reserve(pin.bytes)?;
        self.verify_reserved(file, pin, retain)
    }
    fn verify_reserved(
        &self,
        file: &File,
        pin: &RoventTrainingFilePin,
        retain: bool,
    ) -> Result<Vec<u8>, ArenaError> {
        let legacy = GenerationTrainingFilePinV3 {
            path: pin.path.clone(),
            sha256: pin.sha256.clone(),
            bytes: pin.bytes,
        };
        let account = ReadAccounting::new();
        let mut credit = ReadCredit {
            remaining: pin.bytes,
            maximum: pin.bytes,
            until: self.until,
            cancel: &self.cancel,
            account: Arc::clone(&account),
        };
        let result = verify_bytes(&file, &legacy, retain, &mut credit);
        if result.is_err() || account.uncertain.load(Ordering::Acquire) {
            self.uncertain.store(true, Ordering::Release);
        }
        let raw = result?;
        if account.charged.load(Ordering::Acquire) != pin.bytes {
            self.uncertain.store(true, Ordering::Release);
            return Err(invalid("historical full file read accounting mismatch"));
        }
        Ok(raw)
    }
}
pub(super) struct HeldHistoricalFile {
    pub(super) file: File,
    pub(super) pin: RoventTrainingFilePin,
    pub(super) identity: FileIdentity,
}
fn decode<T: serde::de::DeserializeOwned>(raw: &[u8]) -> Result<T, ArenaError> {
    let text = std::str::from_utf8(raw).map_err(|_| invalid("historical metadata UTF8"))?;
    rz_experiments::decode_json(text).map_err(|e| invalid(e.to_string()))
}
pub(super) fn archive_pin(
    root: &Path,
    item: &RoventPublicationFile,
) -> Result<RoventTrainingFilePin, ArenaError> {
    item.validate()?;
    let relative = Path::new(&item.path);
    check(
        root.is_absolute()
            && !item.path.is_empty()
            && !relative.is_absolute()
            && !item.path.contains('\\')
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "historical artifact escapes registered archive root",
    )?;
    Ok(RoventTrainingFilePin {
        path: root.join(relative),
        sha256: item.sha256.clone(),
        bytes: item.bytes,
    })
}

/// Caller registration is a request to validate exact bytes, not a native run
/// capability. Collection and trainer/runtime/initial weights are separate.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalRoleBinding {
    pub model_semantics_sha256: String,
    pub model_identity_sha256: String,
    pub weights_sha256: String,
    pub encoding_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalDatasetRegistration {
    pub root: PathBuf,
    pub publication: RoventTrainingFilePin,
    pub collection_source_commit: String,
    pub collection_source_sha256: String,
    pub collection_runtime_library_sha256: String,
    pub model_semantics_sha256: String,
    pub model_epoch: u64,
    pub context_epoch: u64,
    pub recipe_sha256: String,
    pub stage_id: String,
    pub role_bindings: BTreeMap<String, RoventHistoricalRoleBinding>,
}

/// No Deserialize or public success constructor. Every retained row has been
/// reread from the externally pinned publication and replayed through Rules.
/// These rows are training data, never a reconstruction of native authority.
pub struct VerifiedRoventDataset {
    registration: RoventHistoricalDatasetRegistration,
    publication: RoventCorpusPublication,
    manifest: RoventCorpusManifest,
    index: RoventCorpusIndex,
    rows: BTreeMap<String, NativeRowAuthority>,
    held: Vec<HeldHistoricalFile>,
    ancestors: Vec<Arc<Self>>,
}
impl VerifiedRoventDataset {
    pub fn open(
        registration: RoventHistoricalDatasetRegistration,
        ancestors: Vec<Arc<Self>>,
        budget: &RoventRecoveryReadBudget,
    ) -> Result<Self, ArenaError> {
        check(
            ancestors.len() <= 256
                && registration.publication.path == registration.root.join("publication.json")
                && generation_identifier(&registration.stage_id)
                && generation_sha(&registration.collection_source_sha256)
                && generation_sha(&registration.collection_runtime_library_sha256)
                && generation_sha(&registration.model_semantics_sha256)
                && generation_sha(&registration.recipe_sha256)
                && !registration.role_bindings.is_empty()
                && registration.role_bindings.len() <= 3,
            "strict historical dataset registration",
        )?;
        for (role, binding) in &registration.role_bindings {
            check(
                ROLES.contains(&role.as_str())
                    && [
                        &binding.model_semantics_sha256,
                        &binding.model_identity_sha256,
                        &binding.weights_sha256,
                        &binding.encoding_sha256,
                    ]
                    .iter()
                    .all(|s| generation_sha(s)),
                "historical per-role model/weights/encoding binding",
            )?;
        }
        let (mut held, published_raw) = budget
            .read(&registration.publication, MAX_METADATA_BYTES, true)
            .map(|(f, r)| (vec![f], r))?;
        let publication: RoventCorpusPublication = decode(&published_raw)?;
        publication.validate()?;
        check(
            publication.source_commit == registration.collection_source_commit
                && publication.source_sha256 == registration.collection_source_sha256
                && publication.runtime_sha256 == registration.collection_runtime_library_sha256,
            "historical publication changed registered collection source/runtime",
        )?;
        let (manifest_file, manifest_raw) = budget.read(
            &archive_pin(&registration.root, &publication.manifest)?,
            MAX_METADATA_BYTES,
            true,
        )?;
        held.push(manifest_file);
        let (index_file, index_raw) = budget.read(
            &archive_pin(&registration.root, &publication.index)?,
            MAX_METADATA_BYTES,
            true,
        )?;
        held.push(index_file);
        let manifest: RoventCorpusManifest = decode(&manifest_raw)?;
        let index: RoventCorpusIndex = decode(&index_raw)?;
        check(
            manifest.sha256().map_err(|e| invalid(e.to_string()))?
                == publication.canonical_manifest_sha256
                && index.sha256().map_err(|e| invalid(e.to_string()))?
                    == publication.canonical_index_sha256
                && manifest.source_commit == registration.collection_source_commit
                && manifest.source_sha256 == registration.collection_source_sha256
                && manifest.runtime_sha256 == registration.collection_runtime_library_sha256
                && manifest.model_semantics_sha256 == registration.model_semantics_sha256
                && manifest.model_epoch == registration.model_epoch
                && manifest.context_epoch == registration.context_epoch
                && manifest.recipe_sha256 == registration.recipe_sha256
                && manifest.stage_id == registration.stage_id
                && index.record_count == publication.record_count
                && index.chunks.len() == publication.chunks.len(),
            "historical publication/manifest/index source and canonical seals differ",
        )?;
        let ancestor_indices = ancestors
            .iter()
            .map(|owner| owner.index.clone())
            .collect::<Vec<_>>();
        let reader = RoventCorpusReader::new_rules_replay_v2(
            manifest.clone(),
            index.clone(),
            &ancestor_indices,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let mut rows = BTreeMap::new();
        let mut admissions = BTreeSet::new();
        let mut total = publication
            .manifest
            .bytes
            .checked_add(publication.index.bytes)
            .ok_or_else(|| invalid("historical bytes overflow"))?;
        let mut lock_files = BTreeMap::new();
        let mut bundle_files = BTreeMap::new();
        let mut admission_locks = BTreeMap::new();
        for admitted in &publication.admissions {
            let lock = if let Some((_, owner)) = lock_files.get(&admitted.input_lock.path) {
                Arc::clone(owner)
            } else {
                let (file, raw) = budget.read(
                    &archive_pin(&registration.root, &admitted.input_lock)?,
                    MAX_METADATA_BYTES,
                    true,
                )?;
                let lock = Arc::new(
                    rz_experiments::RoventInputLock::from_json(
                        std::str::from_utf8(&raw)
                            .map_err(|_| invalid("historical input lock UTF8"))?,
                    )
                    .map_err(|e| invalid(e.to_string()))?,
                );
                total = total
                    .checked_add(admitted.input_lock.bytes)
                    .ok_or_else(|| invalid("historical lock bytes overflow"))?;
                lock_files.insert(
                    admitted.input_lock.path.clone(),
                    (admitted.input_lock.clone(), Arc::clone(&lock)),
                );
                held.push(file);
                lock
            };
            check(
                lock.canonical_sha256 == admitted.canonical_input_lock_sha256,
                "historical full lock canonical seal differs from publication",
            )?;
            let bundle = if let Some(pin) = &admitted.runtime_bundle_manifest {
                let spec = if let Some((_, spec)) = bundle_files.get(&pin.path) {
                    Arc::clone(spec)
                } else {
                    let (file, raw) =
                        budget.read(&archive_pin(&registration.root, pin)?, 64 << 10, true)?;
                    let spec = Arc::new(
                        rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(
                            std::str::from_utf8(&raw)
                                .map_err(|_| invalid("historical runtime bundle UTF8"))?,
                        )
                        .map_err(|error| invalid(error.to_string()))?,
                    );
                    total = total
                        .checked_add(pin.bytes)
                        .ok_or_else(|| invalid("historical bundle bytes overflow"))?;
                    bundle_files.insert(pin.path.clone(), (pin.clone(), Arc::clone(&spec)));
                    held.push(file);
                    spec
                };
                Some((pin.clone(), spec))
            } else {
                None
            };
            admission_locks.insert(admitted.native_admission_sha256.clone(), (lock, bundle));
        }
        for (slot, (declared, pin)) in index.chunks.iter().zip(&publication.chunks).enumerate() {
            check(
                declared.path == pin.path
                    && declared.sha256 == pin.sha256
                    && declared.bytes == pin.bytes,
                "publication chunk exact bytes changed",
            )?;
            total = total
                .checked_add(pin.bytes)
                .ok_or_else(|| invalid("historical total bytes overflow"))?;
            let (file, bytes) = budget.read(
                &archive_pin(&registration.root, pin)?,
                MAX_CHUNK_BYTES as u64,
                true,
            )?;
            held.push(file);
            for row in reader
                .read_chunk(slot, &bytes)
                .map_err(|e| invalid(e.to_string()))?
            {
                let evidence = verify_historical_record_evidence(&row, &manifest)?;
                let role = role_name(row.role);
                let binding = registration
                    .role_bindings
                    .get(role)
                    .ok_or_else(|| invalid("historical row role was not registered"))?;
                check(
                    evidence.collection_source_commit == registration.collection_source_commit
                        && evidence.collection_source_sha256
                            == registration.collection_source_sha256
                        && evidence.collection_runtime_library_sha256
                            == registration.collection_runtime_library_sha256
                        && evidence.model_semantics_sha256 == binding.model_semantics_sha256
                        && evidence.model_identity_sha256 == binding.model_identity_sha256
                        && evidence.weights_sha256 == binding.weights_sha256
                        && evidence.encoding_sha256 == binding.encoding_sha256,
                    "historical row/root/role model source differs from caller registration",
                )?;
                let (lock, bundle) = admission_locks
                    .get(&evidence.native_admission_sha256)
                    .ok_or_else(|| {
                        invalid("historical row admission is missing its full input lock")
                    })?;
                join_historical_input_lock(
                    lock,
                    bundle.as_ref().map(|(pin, spec)| (pin, spec.as_ref())),
                    &row,
                    &evidence,
                )?;
                admissions.insert(evidence.native_admission_sha256);
                let data = row_authority(&row)?;
                check(
                    rows.insert(row.origin_sha256.clone(), data).is_none(),
                    "duplicate historical origin",
                )?;
            }
        }
        check(
            total == publication.total_payload_bytes
                && total <= MAX_TRAINER_OUTPUT_BYTES
                && rows.len() as u64 == publication.record_count
                && admissions.into_iter().collect::<Vec<_>>()
                    == publication.native_admission_sha256,
            "historical full payload/row/admission set did not close",
        )?;
        Ok(Self {
            registration,
            publication,
            manifest,
            index,
            rows,
            held,
            ancestors,
        })
    }
    pub fn registration(&self) -> &RoventHistoricalDatasetRegistration {
        &self.registration
    }
    pub fn observation(&self) -> Value {
        json!({"schema":ROVENT_HISTORICAL_DATASET,"source_kind":ROVENT_RECOVERY_SOURCE_KIND,
        "publication_sha256":self.registration.publication.sha256,"record_count":self.rows.len(),"past_native_authority_reissued":false,
        "past_physical_completion_reissued":false,"historical_evidence_only":true,"execution_authorized":false})
    }
    pub fn revalidate(&self, budget: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        for retained in &self.held {
            let (new_file, _) = budget.read(&retained.pin, retained.pin.bytes, false)?;
            check(
                new_file.identity == retained.identity,
                "historical dataset file identity changed while retained",
            )?;
        }
        Ok(())
    }
}
fn join_historical_input_lock(
    lock: &rz_experiments::RoventInputLock,
    bundle: Option<(
        &RoventPublicationFile,
        &rz_eval::runtime_pin::CudaRuntimeBundleSpec,
    )>,
    row: &rz_experiments::rovent_corpus::RoventTrainingRecord,
    evidence: &crate::rovent_collect::RoventHistoricalRecordEvidence,
) -> Result<(), ArenaError> {
    evidence
        .root_descriptor
        .verify_against(lock)
        .map_err(|error| invalid(error.to_string()))?;
    let manifest = &lock.manifest;
    let engine = &manifest.engine;
    let model = match row.role {
        GenerationRoleV3::Proposer => &engine.pc.proposer,
        GenerationRoleV3::Critic => &engine.pc.critic,
        GenerationRoleV3::Validator => engine
            .pc
            .validator
            .as_ref()
            .ok_or_else(|| invalid("historical V row is absent from the full loaded lock"))?,
    };
    let origin = &row.native.raw_retirement["root_admission"]["native_origin"];
    check(
        manifest.purpose == rz_experiments::RoventPurpose::Collection
            && lock.canonical_sha256 == evidence.collection_input_lock_sha256
            && engine.source_commit == evidence.collection_source_commit
            && engine.source_components == evidence.collection_source_components
            && model.weights.sha256 == evidence.weights_sha256
            && origin["model_implementation_sha256"] == model.implementation_sha256
            && origin["model_semantics"] == model.semantics
            && origin["encoding"] == model.encoding
            && engine.pc.backend_configuration_sha256
                == evidence.collection_native_configuration_sha256
            && engine
                .pc
                .runtime_artifacts
                .iter()
                .any(|file| file.sha256 == evidence.collection_runtime_library_sha256)
            && origin["teacher_profiles"]
                == serde_json::to_value(&engine.teachers).map_err(|e| invalid(e.to_string()))?,
        "historical raw origin differs from the full registered source/model/runtime/teacher lock",
    )?;
    // Mode's public wire spelling is defined by the lock; the native owner uses
    // the shared Rust enum's Debug spelling. Compare the actual common value.
    check(
        origin["mode"] == format!("{:?}", engine.mode),
        "historical root mode differs from its full lock",
    )?;
    match (bundle, &evidence.collection_runtime_bundle_sha256) {
        (None, None) => check(
            matches!(engine.pc.device, rz_experiments::RoventDevice::Cpu),
            "CUDA history is missing its original runtime bundle manifest",
        ),
        (Some((pin, spec)), Some(digest)) => {
            let actual = spec.digest().map_err(|error| invalid(error.to_string()))?;
            check(
                !matches!(engine.pc.device, rz_experiments::RoventDevice::Cpu)
                    && actual
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                        == *digest
                    && manifest
                        .provenance
                        .iter()
                        .any(|file| file.sha256 == pin.sha256 && file.bytes == pin.bytes)
                    && spec.files.len() == engine.pc.runtime_artifacts.len()
                    && spec.files.iter().all(|expected| {
                        engine
                            .pc
                            .runtime_artifacts
                            .iter()
                            .filter(|actual| {
                                Path::new(&actual.path)
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    == Some(expected.filename.as_str())
                                    && actual.sha256 == expected.sha256
                                    && actual.bytes == expected.bytes
                            })
                            .count()
                            == 1
                    }),
                "historical runtime bundle semantics/aliases/roster/provenance do not match the original lock",
            )
        }
        _ => Err(invalid(
            "historical runtime bundle presence disagrees with original runtime evidence",
        )),
    }
}
fn row_authority(
    row: &rz_experiments::rovent_corpus::RoventTrainingRecord,
) -> Result<NativeRowAuthority, ArenaError> {
    let mut valid = BTreeMap::from(HEADS.map(|name| (name.to_owned(), 0)));
    if row.role == GenerationRoleV3::Validator {
        if let Some(task) = &row.targets.task {
            valid.insert(
                "task".into(),
                generation_derive_tasks(&row.input, &task.outcomes)
                    .map_err(|e| invalid(e.to_string()))?
                    .pairs
                    .len() as u64,
            );
        }
    } else {
        valid.insert("policy".into(), u64::from(row.targets.policy.is_some()));
        valid.insert("wdl".into(), u64::from(row.targets.wdl.is_some()));
        valid.insert(
            "divergence".into(),
            u64::from(row.role == GenerationRoleV3::Critic && row.targets.divergence.is_some()),
        );
        valid.insert("nnuec_wdl".into(), u64::from(row.nnuec_wdl.is_some()));
    }
    Ok(NativeRowAuthority {
        origin: RoventCorpusOrigin::from_record(row),
        role: role_name(row.role).into(),
        retirement_sha256: row.native.raw_retirement_sha256.clone(),
        head_raw_sha256: row
            .head_sources
            .iter()
            .map(|(name, source)| (name.clone(), source.raw_receipt_sha256.clone()))
            .collect(),
        valid_heads: valid,
    })
}

mod checkpoint;
pub use checkpoint::*;
mod registration;
pub use registration::*;
mod reclamation;
pub use reclamation::*;
pub mod cli;
mod runtime;
pub use runtime::*;
mod entry;
pub use entry::*;
mod checkpoint_publication;
pub use checkpoint_publication::*;
mod r_auxiliary;
pub use r_auxiliary::*;
mod r_dataset;
pub use r_dataset::*;
mod r_auxiliary_ledger;
use r_auxiliary_ledger::*;
#[cfg(test)]
pub(crate) mod metadata_wire_tests;
