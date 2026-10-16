//! Registered V5 loader for the original FullLineInteractionV2 model.
//!
//! Original manifest, graph names, fixed-L ABI, P/C-only guard and namespace
//! remain legacy identities. This is not a VirtualReview model conversion and
//! preserves V2Full and V2Selected as distinct input routes. All native work happens inside
//! the ticket owner's one factory, with the caller's original process clock.
use crate::pals_adaptive_startup::{PinnedPalsStartupV5, PinnedStartupAssetV5};
use crate::pals_native::ticketed::NativeFreshTicketBackend;
use crate::pals_native::ticketed::NativeLegacyInputModeV5;
use crate::pals_native::{
    NativeOwnerOptions, NativeRoleFinishHandle, NativeRoleModel, NativeRoleReceipt,
    NativeRoleSourceIdentity, pals_fresh_encoding_semantic_digest_for_config,
    validate_pals_legacy_rules_descriptor_bytes,
};
#[cfg(target_os = "linux")]
use cap_fs_ext::DirExt;
use rz_eval::asset::parse_sha256;
use rz_eval::error::{BackendError, FailureKind as K, FailureStage as S};
use rz_eval::onnx::{NativeLoadingProfile, OrtRuntime, Provider};
use rz_eval::pals_model::PalsModelProfile;
use rz_eval::pals_onnx::virtual_review::{
    LegacyPalsExportManifestSummary, VirtualReviewExecutionMode, VirtualReviewProfileDirectory,
    validate_pals_legacy_export_manifest,
};
use rz_eval::pals_onnx::{
    PalsCuDnnConvSearch, PalsCudaControlPolicy, PalsOnnxBackend, PalsOnnxConfig, PalsOwnedSources,
};
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeLibraryPin};
use rz_experiments::{
    PalsArtifactDomainV5, PalsEngineModeV5, PalsModelBackendV3,
    PalsStartupAssetPurposeV5 as Purpose, PalsStartupContentAdmissionV5, PalsViewSelectionV5,
    verify_compiled_source_components_v5,
};
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::pals::coordinator::RoleFailure;
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::engine::{RoleError, RoleModel};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::{
    fs::File,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

pub struct NativeLegacySelectedStartupResourcesV5 {
    pub content_admission: PalsStartupContentAdmissionV5,
    pub runtime_pin: RuntimeLibraryPin,
    /// New exclusive child under this endpoint's held runtime directory.
    pub cuda_profile_root: Option<PathBuf>,
    /// Original absolute admission deadline. No stage creates a new window.
    pub until: Instant,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeLegacySelectedStartupSnapshotV5 {
    pub schema: &'static str,
    pub process_epoch: u64,
    pub model_profile: String,
    pub selected_view: bool,
    pub model_semantic_sha256: [u8; 32],
    pub encoding_sha256: [u8; 32],
    pub model_implementation_sha256: [u8; 32],
    pub content_admission_config_sha256: [u8; 32],
    pub content_admission_semantic_lock_sha256: [u8; 32],
    pub numerical_acceptance_sha256: [u8; 32],
    pub tensor_optimizer_content_verified: bool,
    pub registered_model_epoch: u64,
    pub backend_sha256: [u8; 32],
    pub configured_provider: &'static str,
    pub configured_intra_threads: usize,
    pub configured_inter_threads: usize,
    pub configured_spinning: bool,
    pub configured_tf32: bool,
    pub configured_cache_public_memory: bool,
    pub configured_approximate_warm: bool,
    pub validator_loaded: bool,
    pub physical_workers_created: usize,
    /// Actual source and execution declarations from this retained model.
    pub source: NativeRoleSourceIdentity,
    /// Actual probe counters, mapping and reset. This is not final shutdown.
    /// CPU constructor metadata remains distinct from a first inference.
    pub startup_owner_receipt: NativeRoleReceipt,
    #[serde(skip)]
    namespace: AdaptiveModelNamespace,
}
impl NativeLegacySelectedStartupSnapshotV5 {
    pub fn adaptive_model_namespace(&self) -> AdaptiveModelNamespace {
        self.namespace.clone()
    }
    pub fn role_model_identity(&self) -> &str {
        &self.namespace.value_identity().model
    }
    pub fn model_value_identity(&self) -> &rz_search::pals::value::ModelValueIdentity {
        self.namespace.value_identity()
    }
}
pub struct LoadedNativeLegacySelectedStartupV5 {
    pub backend: NativeFreshTicketBackend,
    pub snapshot: NativeLegacySelectedStartupSnapshotV5,
}

/// Retained cleanup opportunity on a partially constructed actual owner. The
/// caller must finish with its already registered cleanup deadline and preserve
/// unknown completion; this error never claims a worker was released.
pub struct NativeLegacyStartupFailureV5 {
    pub stage: &'static str,
    pub cause: RoleFailure,
    pub finish: Option<NativeRoleFinishHandle>,
    pub startup_owner_receipt: Option<Box<NativeRoleReceipt>>,
}
impl std::fmt::Debug for NativeLegacyStartupFailureV5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeLegacyStartupFailureV5")
            .field("stage", &self.stage)
            .field("cause", &self.cause)
            .field("finish_handle_available", &self.finish.is_some())
            .field("startup_owner_receipt", &self.startup_owner_receipt)
            .finish()
    }
}
impl NativeLegacyStartupFailureV5 {
    fn backend(stage: &'static str, error: BackendError) -> Self {
        Self {
            stage,
            cause: RoleFailure::Backend(error.to_string()),
            finish: None,
            startup_owner_receipt: None,
        }
    }
    fn owner(
        stage: &'static str,
        error: RoleError,
        finish: Option<NativeRoleFinishHandle>,
    ) -> Self {
        let receipt = finish.as_ref().map(|owner| Box::new(owner.receipt()));
        Self {
            stage,
            cause: role_failure(error),
            finish,
            startup_owner_receipt: receipt,
        }
    }
    fn backend_with_owner(
        stage: &'static str,
        error: BackendError,
        finish: &NativeRoleFinishHandle,
    ) -> Self {
        Self {
            stage,
            cause: RoleFailure::Backend(error.to_string()),
            finish: Some(finish.clone()),
            startup_owner_receipt: Some(Box::new(finish.receipt())),
        }
    }
}

struct PinnedLegacySourcesV5 {
    owner: Arc<PinnedPalsStartupV5>,
    // The physical worker keeps this object through its native owner. ORT's
    // CUDA profile writes therefore cannot outlive their directory capability.
    profile: Option<VirtualReviewProfileDirectory>,
}
impl PalsOwnedSources for PinnedLegacySourcesV5 {
    fn read_manifest(&self, maximum: usize) -> Result<Vec<u8>, BackendError> {
        read_pinned(
            self.owner
                .asset(Purpose::ExportManifest)
                .map_err(transport)?,
            maximum,
        )
    }
    fn read_graph(
        &self,
        role: &str,
        file: &str,
        expected: [u8; 32],
        maximum: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let graph = self.owner.asset(graph_purpose(role)?).map_err(transport)?;
        let manifest = self
            .owner
            .asset(Purpose::ExportManifest)
            .map_err(transport)?;
        exact_graph_snapshot_join(
            &manifest.declaration.snapshot_path,
            &graph.declaration.snapshot_path,
            file,
        )?;
        if parse_sha256(&graph.declaration.artifact.sha256)? != expected {
            return Err(fail(
                K::IdentityMismatch,
                "legacy graph declaration differs from its original export",
            ));
        }
        read_pinned(graph, maximum)
    }
}

pub fn load_native_legacy_selected_startup_v5(
    owner: Arc<PinnedPalsStartupV5>,
    process_clock: crate::engine::ProcessClock,
    clock: ContractSystemClock,
    resources: NativeLegacySelectedStartupResourcesV5,
) -> Result<LoadedNativeLegacySelectedStartupV5, NativeLegacyStartupFailureV5> {
    if owner.config.endpoint.context.selection != PalsViewSelectionV5::V2Selected {
        return Err(NativeLegacyStartupFailureV5::backend(
            "selected_context",
            fail(
                K::UnsupportedModel,
                "selected legacy startup cannot relabel the full context route",
            ),
        ));
    }
    load_native_legacy_startup_v5(owner, process_clock, clock, resources)
}

/// The original V2 model is loaded once; full versus selected records affect
/// the retained input route, never the graph ABI or the loaded model identity.
pub fn load_native_legacy_startup_v5(
    owner: Arc<PinnedPalsStartupV5>,
    process_clock: crate::engine::ProcessClock,
    clock: ContractSystemClock,
    resources: NativeLegacySelectedStartupResourcesV5,
) -> Result<LoadedNativeLegacySelectedStartupV5, NativeLegacyStartupFailureV5> {
    validate_process_clock(&process_clock, &clock)
        .map_err(|error| NativeLegacyStartupFailureV5::backend("process_clock", error))?;
    let admitted = admit(&owner, &clock, &resources)
        .map_err(|error| NativeLegacyStartupFailureV5::backend("admission", error))?;
    let native = &owner.config.native_options;
    let profile = if let Some(path) = &resources.cuda_profile_root {
        Some(
            create_owned_profile_directory(&owner, path)
                .map_err(|error| NativeLegacyStartupFailureV5::backend("profile_owner", error))?,
        )
    } else {
        None
    };
    let sources = Arc::new(PinnedLegacySourcesV5 {
        owner: Arc::clone(&owner),
        profile,
    });
    let runtime = match admitted.loading {
        Some(profile) => {
            OrtRuntime::load_with_cuda_loading_profile(&resources.runtime_pin, profile)
        }
        None => OrtRuntime::load(&resources.runtime_pin),
    }
    .map_err(|error| NativeLegacyStartupFailureV5::backend("runtime_load", error))?;
    check_until(resources.until)
        .map_err(|error| NativeLegacyStartupFailureV5::backend("runtime_load", error))?;
    let export_sha256 = &owner
        .asset(Purpose::ExportManifest)
        .map_err(|error| NativeLegacyStartupFailureV5::backend("backend_load", transport(error)))?
        .declaration
        .artifact
        .sha256;
    let retained_sources: Arc<dyn PalsOwnedSources> = sources.clone();
    let backend = match admitted.policy {
        Some(policy) => PalsOnnxBackend::load_owned_sources_with_cuda_control_policy(
            Arc::clone(&retained_sources),
            export_sha256,
            runtime,
            admitted.config,
            policy,
            sources.profile.as_ref().ok_or_else(|| {
                NativeLegacyStartupFailureV5::backend(
                    "backend_load",
                    fail(K::InvalidInput, "legacy CUDA profile owner missing"),
                )
            })?,
        ),
        None => PalsOnnxBackend::load_owned_sources(
            Arc::clone(&retained_sources),
            export_sha256,
            runtime,
            admitted.config,
        ),
    }
    .map_err(|error| NativeLegacyStartupFailureV5::backend("backend_load", error))?;
    check_until(resources.until)
        .map_err(|error| NativeLegacyStartupFailureV5::backend("backend_load", error))?;
    let backend_sha256 = legacy_backend_digest(&backend);
    // Ownership, namespace and clock are established together, before runtime
    // admission. No post-construction epoch override is performed.
    let mut model = NativeRoleModel::new_registered_fresh_owner(
        backend,
        NativeOwnerOptions {
            drain_limit: Duration::from_millis(native.drain_ms),
            host_record_pages: None,
        },
        clock.clone(),
        retained_sources,
    )
    .map_err(|(error, finish)| {
        NativeLegacyStartupFailureV5::owner("owner_construction", error, finish)
    })?;
    let finish = model.finish_handle();
    model
        .enable_followup_execution_evidence()
        .map_err(|error| {
            NativeLegacyStartupFailureV5::owner("followup_evidence", error, Some(finish.clone()))
        })?;
    // Existing CUDA startup performs actual first P/C Run, dependency mapping,
    // placement and reset. It is not skipped or replaced with a selected flag.
    model.prepare_startup(resources.until).map_err(|error| {
        NativeLegacyStartupFailureV5::owner("startup_probe", error, Some(finish.clone()))
    })?;
    check_until(resources.until).map_err(|error| {
        let mut failure = NativeLegacyStartupFailureV5::backend("post_startup", error);
        failure.startup_owner_receipt = Some(Box::new(finish.receipt()));
        failure.finish = Some(finish.clone());
        failure
    })?;
    let source = model.source_identity();
    let namespace = model.registered_adaptive_namespace().map_err(|error| {
        NativeLegacyStartupFailureV5::owner("model_namespace", error, Some(finish.clone()))
    })?;
    if model.identity() != namespace.value_identity().model
        || model.value_identity() != Some(namespace.value_identity())
        || source.checkpoint_sha256 != admitted.manifest.checkpoint_sha256
        || source.encoding_semantic_sha256 != admitted.encoding
        || model.ticketed_clock().domain() != clock.domain()
        || model.ticketed_clock().origin() != clock.origin()
    {
        return Err(NativeLegacyStartupFailureV5::owner(
            "model_namespace",
            RoleError::InvalidOutput,
            Some(finish),
        ));
    }
    let selected_view = owner.config.endpoint.context.selection == PalsViewSelectionV5::V2Selected;
    let snapshot = NativeLegacySelectedStartupSnapshotV5 {
        schema: if selected_view {
            "rz-pals-native-legacy-selected-startup-snapshot-v5/1"
        } else {
            "rz-pals-native-legacy-full-startup-snapshot-v5/1"
        },
        process_epoch: clock.domain().0.0,
        model_profile: source.model_configuration.profile.as_str().into(),
        selected_view,
        model_semantic_sha256: source
            .model_configuration
            .profile
            .registered_semantics_sha256(),
        encoding_sha256: source.encoding_semantic_sha256,
        model_implementation_sha256: parse_sha256(
            &owner.config.endpoint.model.implementation_sha256,
        )
        .map_err(|error| {
            NativeLegacyStartupFailureV5::backend_with_owner("snapshot", error, &finish)
        })?,
        content_admission_config_sha256: parse_sha256(resources.content_admission.config_sha256())
            .map_err(|error| {
                NativeLegacyStartupFailureV5::backend_with_owner("snapshot", error, &finish)
            })?,
        content_admission_semantic_lock_sha256: parse_sha256(
            resources.content_admission.semantic_lock_sha256(),
        )
        .map_err(|error| {
            NativeLegacyStartupFailureV5::backend_with_owner("snapshot", error, &finish)
        })?,
        numerical_acceptance_sha256: parse_sha256(
            &resources.content_admission.numerical().acceptance.sha256,
        )
        .map_err(|error| {
            NativeLegacyStartupFailureV5::backend_with_owner("snapshot", error, &finish)
        })?,
        tensor_optimizer_content_verified: resources
            .content_admission
            .tensor_optimizer_content_verified(),
        registered_model_epoch: owner.config.endpoint.model.epoch,
        backend_sha256,
        configured_provider: if matches!(admitted.config.provider, Provider::Cpu) {
            "CPUExecutionProvider"
        } else {
            "CUDAExecutionProvider"
        },
        configured_intra_threads: admitted.config.intra_threads,
        configured_inter_threads: 1,
        configured_spinning: false,
        configured_tf32: false,
        configured_cache_public_memory: admitted.config.cache_public_memory,
        configured_approximate_warm: false,
        validator_loaded: false,
        physical_workers_created: 1,
        startup_owner_receipt: finish.receipt(),
        source,
        namespace,
    };
    let input_mode = if selected_view {
        NativeLegacyInputModeV5::SelectedV2
    } else {
        NativeLegacyInputModeV5::FullV2
    };
    let backend = NativeFreshTicketBackend::new_with_input_mode(model, resources.until, input_mode)
        .map_err(|cause| NativeLegacyStartupFailureV5 {
            stage: "ticket_owner",
            cause,
            startup_owner_receipt: Some(Box::new(finish.receipt())),
            finish: Some(finish),
        })?;
    Ok(LoadedNativeLegacySelectedStartupV5 { backend, snapshot })
}

struct LegacyAdmission {
    manifest: LegacyPalsExportManifestSummary,
    config: PalsOnnxConfig,
    encoding: [u8; 32],
    loading: Option<NativeLoadingProfile>,
    policy: Option<PalsCudaControlPolicy>,
}
fn validate_process_clock(
    process: &crate::engine::ProcessClock,
    clock: &ContractSystemClock,
) -> Result<(), BackendError> {
    let original = process.runtime_clock();
    if process.domain() != clock.domain() || original.origin() != clock.origin() {
        return Err(fail(
            K::IdentityMismatch,
            "legacy runtime clock differs from the original process epoch or origin",
        ));
    }
    Ok(())
}
fn admit(
    owner: &PinnedPalsStartupV5,
    clock: &ContractSystemClock,
    resources: &NativeLegacySelectedStartupResourcesV5,
) -> Result<LegacyAdmission, BackendError> {
    if owner.metadata_preflight || clock.domain().0.0 == 0 || clock.origin() > Instant::now() {
        return Err(fail(
            K::InvalidInput,
            "legacy native startup requires an actual registered process clock and model execution",
        ));
    }
    check_until(resources.until)?;
    owner
        .config
        .validate_against(&owner.lock)
        .map_err(transport)?;
    verify_compiled_source_components_v5(&owner.config.source_components).map_err(transport)?;
    let bytes = owner
        .config
        .canonical_bytes(&owner.lock)
        .map_err(transport)?;
    if hex(Sha256::digest(bytes).into()) != owner.config_sha256
        || owner.assets().len() != owner.config.assets.len()
        || owner
            .assets()
            .iter()
            .zip(&owner.config.assets)
            .any(|(p, a)| &p.declaration != a)
    {
        return Err(fail(
            K::IdentityMismatch,
            "legacy startup declarations changed after admission",
        ));
    }
    let held = owner
        .assets()
        .iter()
        .map(|asset| (&asset.declaration, &asset.file))
        .collect::<Vec<_>>();
    resources
        .content_admission
        .verify_binding(&owner.config, &owner.lock, &held)
        .map_err(transport)?;
    let endpoint = &owner.config.endpoint;
    let specification = &endpoint.model;
    let native = &owner.config.native_options;
    if !matches!(
        endpoint.context.selection,
        PalsViewSelectionV5::V2Selected | PalsViewSelectionV5::V2Full
    ) || endpoint.mode != PalsEngineModeV5::Deployment
        || native.approximate_warm
        || native.device_public_memory
        || endpoint.resources.ort_intra_threads != 1
        || endpoint.resources.ort_inter_threads != 1
        || endpoint.resources.ort_spinning
        || specification.profile != PalsModelProfile::FullLineInteractionV2.as_str()
        || specification.semantic_sha256
            != hex(PalsModelProfile::FullLineInteractionV2.registered_semantics_sha256())
        || owner.config.source_components.get("model_native")
            != Some(&specification.implementation_sha256)
    {
        return Err(fail(
            K::UnsupportedModel,
            "legacy loader requires explicitly selected V2 full/selected Fresh P/C Deployment and registered host ORT configuration; compact and V cannot be relabeled",
        ));
    }
    let manifest_bytes = read_pinned(
        owner.asset(Purpose::ExportManifest).map_err(transport)?,
        256 * 1024,
    )?;
    let manifest = validate_pals_legacy_export_manifest(
        &manifest_bytes,
        VirtualReviewExecutionMode::Deployment,
    )?;
    let descriptor = validate_pals_legacy_rules_descriptor_bytes(
        &read_pinned(
            owner.asset(Purpose::RulesDescriptor).map_err(transport)?,
            64 * 1024,
        )?,
        &manifest.config,
    )?;
    let encoding = pals_fresh_encoding_semantic_digest_for_config(&manifest.config);
    if manifest.checkpoint_sha256 != parse_sha256(&specification.weights.sha256)?
        || manifest.model_profile != PalsModelProfile::FullLineInteractionV2
        || specification.encoding_sha256 != hex(encoding)
        || manifest.config != descriptor.config
        || manifest.rules_input_semantic_sha256 != Some(descriptor.semantic_sha256)
        || manifest.rules_input_profile.as_deref()
            != Some(crate::pals_native::pals_rules_profile_for_config(
                &manifest.config,
            ))
        || manifest.rules_encoder_source_sha256 != Some(descriptor.encoder_source_sha256)
        || manifest.rules_profile_descriptor_sha256 != Some(descriptor.descriptor_sha256())
        || manifest.validator_present
    {
        return Err(fail(
            K::IdentityMismatch,
            "legacy loaded weights, encoding and original Rules descriptor differ",
        ));
    }
    match specification.artifact_domain {
        PalsArtifactDomainV5::Untrained if !manifest.trained && manifest.training_steps == 0 => {}
        _ => {
            return Err(fail(
                K::UnsupportedModel,
                "registered legacy comparison requires the original untrained baseline domain; diagnostic and G1 cannot be relabeled",
            ));
        }
    }
    for graph in &manifest.graphs {
        let pinned = owner
            .asset(graph_purpose(&graph.role)?)
            .map_err(transport)?;
        exact_graph_snapshot_join(
            &owner
                .asset(Purpose::ExportManifest)
                .map_err(transport)?
                .declaration
                .snapshot_path,
            &pinned.declaration.snapshot_path,
            &graph.file,
        )?;
        if parse_sha256(&pinned.declaration.artifact.sha256)? != graph.sha256 {
            return Err(fail(
                K::IdentityMismatch,
                "legacy retained graph differs before ORT session creation",
            ));
        }
    }
    if resources.runtime_pin.binary_digest()
        != parse_sha256(
            &owner
                .asset(Purpose::RuntimeLibrary)
                .map_err(transport)?
                .declaration
                .artifact
                .sha256,
        )?
    {
        return Err(fail(
            K::IdentityMismatch,
            "legacy runtime pin differs from the registered retained library",
        ));
    }
    let cudnn = match native.cudnn_conv_search.as_str() {
        "legacy_implicit_exhaustive" => PalsCuDnnConvSearch::LegacyImplicitExhaustive,
        "experimental_default" => PalsCuDnnConvSearch::ExperimentalDefault,
        _ => {
            return Err(fail(
                K::UnsupportedModel,
                "legacy host cuDNN option is not registered",
            ));
        }
    };
    let (provider, loading, policy) = match specification.backend {
        PalsModelBackendV3::OrtCpu => {
            if resources.runtime_pin.bundle_digest().is_some()
                || native.native_loading_profile.is_some()
                || native.cuda_device_id.is_some()
                || native.cuda_arena_bytes.is_some()
                || resources.cuda_profile_root.is_some()
                || cudnn != PalsCuDnnConvSearch::LegacyImplicitExhaustive
            {
                return Err(fail(
                    K::IdentityMismatch,
                    "legacy CPU cannot inherit CUDA runtime options",
                ));
            }
            (Provider::Cpu, None, None)
        }
        PalsModelBackendV3::OrtCuda => {
            if manifest.layout != "shared_pc_if" {
                return Err(fail(
                    K::UnsupportedModel,
                    "registered legacy CUDA metadata-control has the original shared P/C graph capability only",
                ));
            }
            verify_runtime_bundle(owner, &resources.runtime_pin)?;
            let loading = match native.native_loading_profile.as_deref() {
                Some("eager-cuda12-cudnn9-v1") => NativeLoadingProfile::EagerCuda12Cudnn9V1,
                _ => {
                    return Err(fail(
                        K::UnsupportedModel,
                        "legacy CUDA loading profile differs from the exact registered V5 selection",
                    ));
                }
            };
            let device_id = native
                .cuda_device_id
                .ok_or_else(|| fail(K::InvalidInput, "legacy CUDA device missing"))?;
            let arena_bytes = usize::try_from(
                native
                    .cuda_arena_bytes
                    .ok_or_else(|| fail(K::InvalidInput, "legacy CUDA arena missing"))?,
            )
            .map_err(|_| {
                fail(
                    K::ResourceExhausted,
                    "legacy CUDA arena cannot fit host address space",
                )
            })?;
            let inventory = owner
                .asset(Purpose::CudaControlInventory)
                .map_err(transport)?;
            let policy = policy_from_retained_inventory(inventory)?;
            if resources.cuda_profile_root.is_none() {
                return Err(fail(
                    K::InvalidInput,
                    "legacy CUDA requires an owned profile child",
                ));
            }
            (
                Provider::Cuda {
                    device_id,
                    arena_bytes,
                },
                Some(loading),
                Some(policy),
            )
        }
        _ => {
            return Err(fail(
                K::UnsupportedModel,
                "legacy native loader cannot substitute a mock/Rust implementation for ORT",
            ));
        }
    };
    let config = PalsOnnxConfig {
        provider,
        intra_threads: 1,
        cache_public_memory: native.cache_public_memory,
        device_public_memory: false,
        cudnn_conv_search: cudnn,
    };
    config.validate_execution_selection(loading, policy.is_some())?;
    check_until(resources.until)?;
    Ok(LegacyAdmission {
        manifest,
        config,
        encoding,
        loading,
        policy,
    })
}

fn legacy_backend_digest(backend: &PalsOnnxBackend) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-legacy-selected-actual-backend-fp32/1");
    hash.update(backend.residency().layout.as_bytes());
    hash.update(backend.runtime_binary_digest());
    for optional in [
        backend.runtime_bundle_digest(),
        backend.native_loading_profile_digest(),
        backend.cuda_control_inventory_digest(),
    ] {
        match optional {
            Some(digest) => {
                hash.update([1]);
                hash.update(digest);
            }
            None => hash.update([0]),
        }
    }
    let config = backend.config();
    match config.provider {
        Provider::Cpu => hash.update([0]),
        Provider::Cuda {
            device_id,
            arena_bytes,
        } => {
            hash.update([1]);
            hash.update(device_id.to_le_bytes());
            hash.update((arena_bytes as u64).to_le_bytes());
        }
    }
    hash.update((config.intra_threads as u64).to_le_bytes());
    hash.update([
        1,
        0,
        0,
        u8::from(config.cache_public_memory),
        u8::from(config.device_public_memory),
    ]);
    hash.update([match config.cudnn_conv_search {
        PalsCuDnnConvSearch::LegacyImplicitExhaustive => 0,
        PalsCuDnnConvSearch::ExperimentalDefault => 1,
        PalsCuDnnConvSearch::ExperimentalDefaultDeviceWarm => 2,
    }]);
    hash.finalize().into()
}
fn graph_purpose(role: &str) -> Result<Purpose, BackendError> {
    match role {
        "public" => Ok(Purpose::PublicGraph),
        "shared_pc" => Ok(Purpose::SharedPcGraph),
        "proposer" => Ok(Purpose::ProposerGraph),
        "critic" => Ok(Purpose::CriticGraph),
        _ => Err(fail(
            K::UnsupportedModel,
            "unregistered legacy P/C graph role",
        )),
    }
}
fn exact_graph_snapshot_join(manifest: &str, graph: &str, file: &str) -> Result<(), BackendError> {
    if !matches!(
        Path::new(file).components().collect::<Vec<_>>().as_slice(),
        [Component::Normal(_)]
    ) || Path::new(file).file_name().and_then(|name| name.to_str()) != Some(file)
        || Path::new(manifest)
            .parent()
            .map(|parent| parent.join(file))
            .as_deref()
            != Some(Path::new(graph))
    {
        return Err(fail(
            K::IdentityMismatch,
            "legacy graph must retain its original manifest-relative basename",
        ));
    }
    Ok(())
}
fn verify_runtime_bundle(
    owner: &PinnedPalsStartupV5,
    pin: &RuntimeLibraryPin,
) -> Result<(), BackendError> {
    let raw = read_pinned(
        owner.asset(Purpose::CudaManifest).map_err(transport)?,
        64 * 1024,
    )?;
    let declared = CudaRuntimeBundleSpec::from_json(
        std::str::from_utf8(&raw)
            .map_err(|_| fail(K::InvalidInput, "legacy CUDA manifest is not UTF-8"))?,
    )?;
    if pin.bundle_spec().as_ref() != Some(&declared)
        || pin.bundle_digest() != Some(declared.digest()?)
    {
        return Err(fail(
            K::IdentityMismatch,
            "legacy retained CUDA bundle differs from its declaration",
        ));
    }
    for library in &declared.files {
        let matches = owner
            .assets()
            .iter()
            .filter(|a| {
                matches!(
                    a.declaration.purpose,
                    Purpose::RuntimeLibrary | Purpose::CudaLibrary
                ) && Path::new(&a.declaration.snapshot_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    == Some(library.filename.as_str())
            })
            .collect::<Vec<_>>();
        if matches.len() != 1
            || matches[0].declaration.artifact.sha256 != library.sha256
            || matches[0].declaration.artifact.bytes != library.bytes
        {
            return Err(fail(
                K::IdentityMismatch,
                "legacy CUDA library roster differs from actual retained assets",
            ));
        }
    }
    if owner
        .assets()
        .iter()
        .filter(|a| {
            matches!(
                a.declaration.purpose,
                Purpose::RuntimeLibrary | Purpose::CudaLibrary
            )
        })
        .count()
        != declared.files.len()
    {
        return Err(fail(
            K::IdentityMismatch,
            "legacy CUDA library roster contains an unregistered extra",
        ));
    }
    Ok(())
}
fn policy_from_retained_inventory(
    asset: &PinnedStartupAssetV5,
) -> Result<PalsCudaControlPolicy, BackendError> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // Legacy policy's path validator consumes this exact live descriptor,
        // not the mutable snapshot parent. The complete original inventory pin
        // is checked again by that parser; graph/manifest pins are not rewritten.
        let _verified = read_pinned(asset, 4 * 1024 * 1024)?;
        PalsCudaControlPolicy::from_inventory(
            &PathBuf::from(format!("/proc/self/fd/{}", asset.file.as_raw_fd())),
            &asset.declaration.artifact.sha256,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = asset;
        Err(fail(
            K::UnsupportedModel,
            "retained legacy CUDA inventory requires Linux descriptor access",
        ))
    }
}
fn profile_child_name(
    owner: &PinnedPalsStartupV5,
    requested: &Path,
) -> Result<String, BackendError> {
    profile_name(
        owner.private_cwd(),
        &owner.config.runtime_directory,
        requested,
    )
}
fn profile_name(
    private_cwd: &Path,
    runtime_directory: &str,
    requested: &Path,
) -> Result<String, BackendError> {
    let name = requested
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| fail(K::InvalidInput, "legacy CUDA profile needs a named child"))?;
    if !private_cwd.is_absolute()
        || !requested.is_absolute()
        || requested
            .components()
            .any(|p| matches!(p, Component::ParentDir | Component::CurDir))
        || requested.parent() != Some(private_cwd.join(runtime_directory).as_path())
        || !requested.to_str().is_some_and(|path| {
            !path
                .split(['/', '\\'])
                .any(|part| matches!(part, "." | ".."))
        })
        || name.is_empty()
        || name.len() > 96
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(fail(
            K::InvalidInput,
            "legacy CUDA profile must be a direct bounded runtime child",
        ));
    }
    Ok(name.to_owned())
}
fn create_owned_profile_directory(
    owner: &PinnedPalsStartupV5,
    requested: &Path,
) -> Result<VirtualReviewProfileDirectory, BackendError> {
    let name = profile_child_name(owner, requested)?;
    #[cfg(target_os = "linux")]
    {
        let runtime = owner.runtime_directory_capability().map_err(transport)?;
        runtime.create_dir(&name).map_err(|_| {
            fail(
                K::InvalidInput,
                "legacy profile child must be new and exclusive",
            )
        })?;
        let profile = runtime.open_dir_nofollow(&name).map_err(|_| {
            fail(
                K::InvalidInput,
                "legacy profile child could not be retained without links",
            )
        })?;
        VirtualReviewProfileDirectory::from_owned_directory(profile.into_std_file())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = name;
        Err(fail(
            K::UnsupportedModel,
            "owned legacy CUDA profile is Linux-only",
        ))
    }
}
fn read_pinned(asset: &PinnedStartupAssetV5, maximum: usize) -> Result<Vec<u8>, BackendError> {
    let before = asset
        .file
        .metadata()
        .map_err(|_| fail(K::InvalidInput, "legacy pinned metadata unavailable"))?;
    let length = usize::try_from(asset.declaration.artifact.bytes)
        .map_err(|_| fail(K::ResourceExhausted, "legacy pinned byte count overflow"))?;
    if !before.is_file() || length == 0 || length > maximum || before.len() != length as u64 {
        return Err(fail(
            K::ResourceExhausted,
            "legacy pinned asset exceeds the exact bounded declaration",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| fail(K::ResourceExhausted, "legacy input allocation failed"))?;
    bytes.resize(length, 0);
    #[cfg(windows)]
    let reader =
        rz_native_loader::pinned_reader::independent_read_handle(&asset.file).map_err(|error| {
            let mut failure = BackendError::new(
                K::InvalidInput,
                S::Admission,
                "legacy pinned reader unavailable",
            )
            .with_external_cause(rz_eval::error::CauseCode::ModelLoad, &error);
            failure.native = Some(Box::new(rz_eval::error::NativeDiagnostic {
                code: "PinnedReadHandle".into(),
                message: format!("raw_os_error={:?}", error.raw_os_error()),
                truncated: false,
            }));
            failure
        })?;
    #[cfg(windows)]
    let read_file = &reader;
    #[cfg(not(windows))]
    let read_file = &asset.file;
    let mut at = 0;
    while at < length {
        let count = positional_read(read_file, &mut bytes[at..], at as u64)
            .map_err(|_| fail(K::InvalidInput, "legacy pinned positional read failed"))?;
        if count == 0 {
            return Err(fail(
                K::IdentityMismatch,
                "legacy pinned asset ended before its declared length",
            ));
        }
        at += count;
    }
    let after = asset
        .file
        .metadata()
        .map_err(|_| fail(K::InvalidInput, "legacy pinned metadata disappeared"))?;
    if !same_file_metadata(&before, &after)
        || <[u8; 32]>::from(Sha256::digest(&bytes))
            != parse_sha256(&asset.declaration.artifact.sha256)?
    {
        return Err(fail(
            K::IdentityMismatch,
            "legacy pinned asset changed during read",
        ));
    }
    Ok(bytes)
}
fn same_file_metadata(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
            && before.nlink() == after.nlink()
    }
    #[cfg(not(unix))]
    {
        true
    }
}
#[cfg(unix)]
fn positional_read(file: &File, bytes: &mut [u8], at: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(bytes, at)
}
#[cfg(windows)]
fn positional_read(file: &File, bytes: &mut [u8], at: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(bytes, at)
}
#[cfg(not(any(unix, windows)))]
fn positional_read(_: &File, _: &mut [u8], _: u64) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "legacy pinned positional reads unsupported",
    ))
}
fn check_until(until: Instant) -> Result<(), BackendError> {
    if Instant::now() < until {
        Ok(())
    } else {
        Err(fail(
            K::BackendFailure,
            "legacy original startup window ended",
        ))
    }
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn fail(kind: K, message: &'static str) -> BackendError {
    BackendError::new(kind, S::Admission, message)
}
fn transport(_: rz_experiments::ManifestError) -> BackendError {
    fail(
        K::IdentityMismatch,
        "legacy typed startup declaration failed validation",
    )
}
fn role_failure(error: RoleError) -> RoleFailure {
    match error {
        RoleError::Canceled => RoleFailure::Canceled,
        RoleError::Deadline => RoleFailure::Expired,
        RoleError::InvalidOutput => RoleFailure::InvalidOutput,
        RoleError::PhysicalCompletionUnknown => RoleFailure::PhysicalUnknown,
        RoleError::Unsupported(capability) => RoleFailure::Unsupported(capability),
        other => RoleFailure::Backend(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn legacy_graph_role_and_original_manifest_location_cannot_be_relabelled() {
        for (role, purpose) in [
            ("public", Purpose::PublicGraph),
            ("shared_pc", Purpose::SharedPcGraph),
            ("proposer", Purpose::ProposerGraph),
            ("critic", Purpose::CriticGraph),
        ] {
            assert_eq!(graph_purpose(role).unwrap(), purpose);
        }
        for role in ["validator", "shared_warm", "unknown"] {
            assert!(graph_purpose(role).is_err());
        }
        exact_graph_snapshot_join(
            "inputs/candidate-model/export.json",
            "inputs/candidate-model/public_memory.onnx",
            "public_memory.onnx",
        )
        .unwrap();
        for (graph, file) in [
            ("inputs/other/public_memory.onnx", "public_memory.onnx"),
            (
                "inputs/candidate-model/public_memory.onnx",
                "../public_memory.onnx",
            ),
            (
                "inputs/candidate-model/role_critic.onnx",
                "role_proposer.onnx",
            ),
        ] {
            assert!(
                exact_graph_snapshot_join("inputs/candidate-model/export.json", graph, file)
                    .is_err()
            );
        }
    }
    #[test]
    fn legacy_fresh_model_mapping_keeps_manifest_and_checkpoint_distinct() {
        let config = rz_eval::pals_model::PalsModelConfig::for_profile(
            PalsModelProfile::FullLineInteractionV2,
        );
        let encoding = pals_fresh_encoding_semantic_digest_for_config(&config);
        assert_ne!(
            encoding,
            <[u8; 32]>::from(Sha256::digest(config.profile.encoding_schema().as_bytes()))
        );
        assert_ne!(
            encoding,
            rz_eval::pals_model::virtual_review::registered_encoding_sha256()
        );
    }
    #[test]
    fn registered_clock_clone_preserves_both_domain_and_origin() {
        let origin = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        let clock = ContractSystemClock::with_origin(rz_contracts::ProcessEpoch(71), origin);
        let retained = clock.clone();
        assert_eq!(retained.domain(), clock.domain());
        assert_eq!(retained.origin(), origin);
        // Equal epochs alone cannot authorize a newly sampled deadline origin.
        assert_ne!(
            ContractSystemClock::new(rz_contracts::ProcessEpoch(71)).origin(),
            origin
        );
        let process = crate::engine::ProcessClock::new(rz_contracts::ProcessEpoch(89));
        validate_process_clock(&process, &process.runtime_clock()).unwrap();
        assert!(
            validate_process_clock(&process, &ContractSystemClock::new(process.epoch())).is_err()
        );
        assert!(
            validate_process_clock(
                &process,
                &ContractSystemClock::with_origin(
                    rz_contracts::ProcessEpoch(90),
                    process.runtime_clock().origin()
                )
            )
            .is_err()
        );
    }
    #[test]
    fn legacy_profile_is_one_new_child_under_registered_runtime() {
        let root = std::env::temp_dir().join("rz-legacy-profile-name-only");
        assert_eq!(
            profile_name(
                &root,
                "candidate-runtime",
                &root.join("candidate-runtime/legacy-profiles")
            )
            .unwrap(),
            "legacy-profiles"
        );
        for path in [
            root.join("legacy-profiles"),
            root.join("candidate-runtime/sub/name"),
            root.join("candidate-runtime/../outside"),
            root.join("candidate-runtime/./name"),
        ] {
            assert!(profile_name(&root, "candidate-runtime", &path).is_err());
        }
    }
    #[test]
    fn legacy_reads_retain_cursor_and_verify_original_bytes() {
        let path = std::env::temp_dir().join(format!(
            "rz-legacy-source-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bytes = b"readonly legacy model source fixture";
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.seek(SeekFrom::Start(7)).unwrap();
        let mut asset = PinnedStartupAssetV5 {
            declaration: rz_experiments::PalsStartupAssetV5 {
                artifact: rz_experiments::ArtifactRef {
                    path: "logical/fixture.bin".into(),
                    sha256: hex(Sha256::digest(bytes).into()),
                    bytes: bytes.len() as u64,
                    source: "MIT synthetic source".into(),
                    license: "MIT".into(),
                },
                snapshot_path: "inputs/candidate-model/fixture.bin".into(),
                purpose: Purpose::PublicGraph,
            },
            file,
        };
        assert_eq!(read_pinned(&asset, bytes.len()).unwrap(), bytes);
        assert_eq!(asset.file.stream_position().unwrap(), 7);
        std::thread::scope(|scope| {
            let first = scope.spawn(|| read_pinned(&asset, bytes.len()));
            let second = scope.spawn(|| read_pinned(&asset, bytes.len()));
            assert_eq!(first.join().unwrap().unwrap(), bytes);
            assert_eq!(second.join().unwrap().unwrap(), bytes);
        });
        assert_eq!(asset.file.stream_position().unwrap(), 7);
        assert!(read_pinned(&asset, bytes.len() - 1).is_err());
        asset.declaration.artifact.sha256 = "11".repeat(32);
        assert!(read_pinned(&asset, bytes.len()).is_err());
        assert_eq!(asset.file.stream_position().unwrap(), 7);
        drop(asset);
        std::fs::remove_file(path).unwrap();
    }
}
