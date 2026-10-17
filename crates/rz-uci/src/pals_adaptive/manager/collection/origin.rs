//! 원 loaded native source와 실제 Frontier 부착을 함께 보존하는 수집 권한.
//! 이 파일은 wire DTO가 아니며 JSON/복사한 receipt에서 권한을 복원하지 않는다.
use super::*;
use rz_contracts::{Deadline, Digest};
use rz_position::contracts::ContractState;
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::rovent_engine::RootMoveDomain;
use rz_search::pals::rovent_frontier::RoventFrontier;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use serde_json::{Value, json};
use std::sync::Arc;

pub const ROVENT_COLLECTION_NATIVE_ORIGIN: &str = "rz-rovent-collection-native-origin/1";
pub const ROVENT_COLLECTION_ROOT_ADMISSION: &str = "rz-rovent-collection-root-admission/1";

fn reject(message: impl Into<String>) -> AdaptiveCollectionError {
    AdaptiveCollectionError::new(
        AdaptiveCollectionStage::Accepted,
        "RoventRootAdmission",
        message,
    )
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
fn text_sha(value: Digest) -> String {
    value.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct LoadedOrigin {
    // Retains the original pinned source descriptors without copying weights.
    _source_custody: Arc<dyn Send + Sync>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    sources: Arc<crate::rovent_bootstrap::RoventPinnedNativeSources>,
    process_epoch: u64,
    mode: rz_contracts::rovent::ExecutionMode,
    input_lock: Digest,
    namespace: AdaptiveModelNamespace,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw: Value,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw_sha256: String,
}

/// 발급은 PreparedRoventNative의 원 성공 owner에서만 가능하다.
/// public constructor·Deserialize·wire에서의 복원은 제공하지 않는다.
#[derive(Clone)]
pub struct RoventNativeCollectionOrigin {
    original: Arc<LoadedOrigin>,
}
impl RoventNativeCollectionOrigin {
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn from_loaded(
        snapshot: &crate::rovent_bootstrap::RoventNativeStartupSnapshot,
        sources: Arc<crate::rovent_bootstrap::RoventPinnedNativeSources>,
    ) -> Result<Self, AdaptiveCollectionError> {
        let lock = sources.lock();
        let manifest = &lock.manifest;
        let pc = &manifest.engine.pc;
        if manifest.purpose != rz_experiments::RoventPurpose::Collection
            || pc.proposer.diagnostic
            || pc.critic.diagnostic
            || pc.validator.is_some()
            || snapshot.validator_loaded
            || snapshot.process_epoch == 0
            || snapshot.input_lock != sources.lock_digest()
            || snapshot.mode != manifest.engine.mode
            || snapshot.weights.0 != sources.weights_digest()
            || text_sha(snapshot.export_manifest) != sources.export_digest_text()
            || snapshot.neural_service_executed
        {
            return Err(reject(
                "loaded source is not the original nondiagnostic Rovent collection startup",
            ));
        }
        let namespace = snapshot
            .model_namespace()
            .map_err(|error| reject(error.to_string()))?;
        let source_sha256 = rz_experiments::rovent_source_aggregate_sha256(
            &manifest.engine.source_commit,
            &manifest.engine.source_components,
        )
        .map_err(|error| reject(error.to_string()))?;
        let raw = json!({
            "schema":ROVENT_COLLECTION_NATIVE_ORIGIN,
            "purpose":"collection",
            "process_epoch":snapshot.process_epoch,
            "mode":format!("{:?}",snapshot.mode),
            "input_lock_sha256":text_sha(snapshot.input_lock),
            "source_commit":manifest.engine.source_commit,
            "source_components":manifest.engine.source_components,
            "source_sha256":source_sha256,
            "weights_sha256":text_sha(snapshot.weights),
            "export_manifest_sha256":text_sha(snapshot.export_manifest),
            "backend_sha256":text_sha(snapshot.backend),
            "runtime_library_sha256":text_sha(snapshot.runtime_library),
            "runtime_bundle_sha256":snapshot.runtime_bundle.map(text_sha),
            "native_configuration_sha256":text_sha(snapshot.native_configuration),
            "model_semantics":pc.proposer.semantics,
            "model_semantics_sha256":text_sha(Digest(rz_eval::pals_model::PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())),
            "encoding":pc.proposer.encoding,
            "encoding_sha256":text_sha(Digest({use sha2::Digest as _; sha2::Sha256::digest(rz_eval::pals_model::virtual_review::ENCODING.as_bytes()).into()})),
            "encoding_registry_sha256":text_sha(Digest(rz_eval::pals_model::virtual_review::registered_encoding_sha256())),
            "model_implementation_sha256":pc.proposer.implementation_sha256,
            "representation_model_domain":"loaded_checkpoint_sha256",
            "teacher_profiles":manifest.engine.teachers,
            "frozen_epoch":1,
            "runtime_sha256_domain":"loaded_runtime_library_file_sha256",
            "runtime_sha256":text_sha(snapshot.runtime_library),
        });
        let raw_sha256 =
            rz_experiments::pals_data_v3::generation_digest(ROVENT_COLLECTION_NATIVE_ORIGIN, &raw)
                .map_err(|error| reject(error.to_string()))?;
        let process_epoch = snapshot.process_epoch;
        let mode = snapshot.mode;
        let input_lock = snapshot.input_lock;
        Ok(Self {
            original: Arc::new(LoadedOrigin {
                _source_custody: Arc::clone(&sources) as Arc<dyn Send + Sync>,
                sources,
                process_epoch,
                mode,
                input_lock,
                namespace,
                raw,
                raw_sha256,
            }),
        })
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_receipt(&self) -> &Value {
        &self.original.raw
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_sha256(&self) -> &str {
        &self.original.raw_sha256
    }
    /// Original immutable parsed lock, retained by the actual loaded source.
    /// Serializing this snapshot preserves meaning, not original whitespace.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn original_input_lock(&self) -> &rz_experiments::RoventInputLock {
        self.original.sources.lock()
    }
    /// Original metadata bytes only; neither this read nor historical replay
    /// grants runtime, native, physical or training execution authority.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn runtime_bundle_publication_bytes(
        &self,
    ) -> Result<Option<(rz_experiments::ArtifactRef, Vec<u8>)>, AdaptiveCollectionError> {
        let original = self
            .original
            .sources
            .runtime_bundle_publication_bytes()
            .map_err(|error| reject(error.to_string()))?;
        match &original {
            Some((_, bytes)) => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| reject("original bundle metadata is not UTF-8"))?;
                let spec = rz_eval::runtime_pin::CudaRuntimeBundleSpec::from_json(text)
                    .map_err(|error| reject(error.to_string()))?;
                let digest = text_sha(Digest(
                    spec.digest().map_err(|error| reject(error.to_string()))?,
                ));
                if self.original.raw["runtime_bundle_sha256"] != digest {
                    return Err(reject(
                        "original bundle file bytes do not join loaded binary-codec digest",
                    ));
                }
            }
            None if !self.original.raw["runtime_bundle_sha256"].is_null() => {
                return Err(reject(
                    "loaded CUDA bundle lacks original retained metadata bytes",
                ));
            }
            None => {}
        }
        Ok(original)
    }
    pub(crate) fn verify_profile(
        &self,
        mode: rz_contracts::rovent::ExecutionMode,
        input_lock: Digest,
        namespace: &AdaptiveModelNamespace,
    ) -> Result<(), AdaptiveCollectionError> {
        if self.original.mode != mode
            || self.original.input_lock != input_lock
            || &self.original.namespace != namespace
        {
            return Err(reject(
                "collection profile differs from the original loaded source identity",
            ));
        }
        Ok(())
    }
}

/// 원 attach 성공 제어흐름에서 발급하는 root-scoped 권한이다. 원 Rules와
/// 제한된 root move domain을 Arc로 보존하므로 복사한 digest는 이를 대신하지 못한다.
#[derive(Clone)]
pub struct RoventCollectionRootAdmission {
    original: RoventNativeCollectionOrigin,
    authority: SearchAuthority,
    deadline: Deadline,
    rules: Arc<ContractState>,
    root_moves: Arc<RootMoveDomain>,
    root_state: rz_contracts::StateIdentity,
    root_budget: Digest,
    selective_policy: Digest,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw: Arc<Value>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    raw_sha256: Arc<str>,
}
impl RoventCollectionRootAdmission {
    /// Call only immediately after the actual port's attach_rovent_frontier
    /// has checked independent Rules/Store/namespace/root-domain/ledger owners.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn from_attached_frontier(
        original: &RoventNativeCollectionOrigin,
        frontier: &RoventFrontier,
        input_lock: Digest,
        root_budget: Digest,
        selective_policy: Digest,
    ) -> Result<Self, AdaptiveCollectionError> {
        let registration = frontier.registration();
        let root_state = registration.root_rules.snapshot().identity();
        let loaded = &original.original;
        if input_lock != loaded.input_lock
            || registration.mode != loaded.mode
            || registration.authority.epoch.0 != loaded.process_epoch
            || registration.root_budget != root_budget
            || registration.selective_policy != selective_policy
            || registration.model_namespace.as_ref() != &loaded.namespace
            || registration.root_rules.snapshot().identity()
                != registration.root_moves.root_identity()
            || registration.mode.has_native_r() != registration.native_ledger.is_some()
        {
            return Err(reject(
                "actual attached frontier differs from original loaded collection scope",
            ));
        }
        let raw = json!({
            "schema":ROVENT_COLLECTION_ROOT_ADMISSION,
            "native_origin":original.raw_receipt(),
            "native_origin_sha256":original.raw_sha256(),
            "process_epoch":registration.authority.epoch.0,
            "game_generation":registration.authority.game.0,
            "root_generation":registration.authority.root.0,
            "implementation_sha256":text_sha(registration.authority.implementation),
            "deadline_clock":registration.original_deadline.clock.0.0,
            "deadline_tick":registration.original_deadline.at.0,
            "root_budget_sha256":text_sha(root_budget),
            "selective_policy_sha256":text_sha(selective_policy),
            "root_state":{"owner":root_state.owner.0,
                "revision":root_state.revision.0,"semantic":text_sha(root_state.semantic)},
            "store_root_state":registration.root_state.0,
            "root_moves":registration.root_moves.allowed().iter().map(|movement|movement.to_string()).collect::<Vec<_>>(),
            "depth_plies":registration.goal.depth_plies.map(|value|value.get()),
            "mate_moves":registration.goal.mate_moves.map(|value|value.get()),
            "requested_nodes":registration.nodes,
        });
        let raw_sha256 =
            rz_experiments::pals_data_v3::generation_digest(ROVENT_COLLECTION_ROOT_ADMISSION, &raw)
                .map_err(|error| reject(error.to_string()))?;
        Ok(Self {
            original: original.clone(),
            authority: registration.authority,
            deadline: registration.original_deadline,
            rules: Arc::clone(&registration.root_rules),
            root_moves: Arc::clone(&registration.root_moves),
            root_state,
            root_budget,
            selective_policy,
            raw: Arc::new(raw),
            raw_sha256: raw_sha256.into(),
        })
    }
    #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
    pub(crate) fn from_attached_frontier(
        _: &RoventNativeCollectionOrigin,
        _: &RoventFrontier,
        _: Digest,
        _: Digest,
        _: Digest,
    ) -> Result<Self, AdaptiveCollectionError> {
        Err(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Accepted,
            "RoventCollectionUnsupported",
            "actual native collection requires pals-adaptive and onnx-cpu",
        ))
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn validates_registration(
        &self,
        authority: SearchAuthority,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: Digest,
        deadline: Deadline,
    ) -> bool {
        self.authority == authority
            && self.original.original.mode == mode
            && self.root_budget == root_budget
            && self.deadline == deadline
    }
    pub fn selective_policy(&self) -> Digest {
        self.selective_policy
    }
    pub fn rules(&self) -> &Arc<ContractState> {
        &self.rules
    }
    pub fn root_moves(&self) -> &Arc<RootMoveDomain> {
        &self.root_moves
    }
    pub fn native_origin(&self) -> &RoventNativeCollectionOrigin {
        &self.original
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_receipt(&self) -> &Value {
        &self.raw
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }
    pub fn validate_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveCollectionError> {
        if ticket.authority != self.authority
            || ticket.original_deadline != self.deadline
            || ticket.view.root_state != self.root_state
            || ticket.role == Role::Validator
        {
            return Err(reject(
                "native ticket is outside the original admitted Rovent P/C root",
            ));
        }
        Ok(())
    }
    pub fn validate_output(
        &self,
        ticket: &RoleTicketEnvelope,
        output: &rz_contracts::pals::RoleOutput,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(ticket)?;
        let namespace = &self.original.original.namespace;
        namespace
            .verify(ticket, output, Some(namespace.value_identity()))
            .map_err(|error| reject(error.to_string()))
    }
    pub(super) fn validate_ranking(
        &self,
        ranking: &ProposerRankingPublished,
    ) -> Result<(), AdaptiveCollectionError> {
        self.validate_ticket(&ranking.source().ticket)?;
        if ranking.source().ticket.role != Role::Proposer
            || !std::ptr::eq(ranking.rules(), self.rules.as_ref())
        {
            return Err(reject(
                "initial P publication is not from the original admitted Rules owner",
            ));
        }
        Ok(())
    }
}
