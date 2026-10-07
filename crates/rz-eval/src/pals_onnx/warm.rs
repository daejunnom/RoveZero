//! Explicit CPU private Warm graph domain. A loaded capability is not seed
//! eligibility or search acceptance: the Rules/search owner supplies those.
//! The ignored Fresh seed is still a complete, finite, bounded tensor.
use super::*;
use crate::pals_private::{
    PrivateInvocation, PrivateModelIdentity, PrivatePrecision, PrivateSeedLease,
    PRIVATE_WARM_SEMANTICS,
};
use sha2::{Digest as _, Sha256};
use std::fs::{Metadata, OpenOptions};
use std::io::Read;

pub const PRIVATE_WARM_SCHEMA: &str = "rovezero.pals-private-warm.v1";
const WARM_LAYOUT: &str = "shared_pc_if_approx_warm_v1";
pub const PRIVATE_WARM_GRAPH_SEMANTICS: &str =
    "rz-pals-private-native-warm/1;accepted-final-latent;no-query-double-add;fixed-2-iterations;private-self-kv-recomputed;fp32;pc-only";
/// This is the consumer's opt-in query meaning, not a change to legacy input.
/// It requires the first actual 16 FP32 query bits to be consumed unchanged in
/// later Warm calls. Actual record revision and original controls remain live.
pub const FROZEN_QUERY_SEMANTICS_V1: &str =
    "rz-pals-private-frozen-query/1;first-actual-query-fp32-bits;logical-game-search-situation-prefix-focus-role-isolated;records-revision-current;original-deadline-cancel-fixed;value-fresh";
const LATENT_ELEMENTS: usize = 16 * 384;
const MANIFEST_LIMIT: usize = 64 * 1024;
const GRAPH_LIMIT: usize = 256 * 1024 * 1024;
const PAYLOAD_LIMIT: u64 = 1024 * 1024;

/// Issued only after this domain's two actual CPU sessions have been loaded and
/// their interfaces/metadata checked. No public constructor accepts declarations
/// as a loaded witness. Numerical/physical/accepted-output gates are separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsWarmCapability {
    identity: PrivateModelIdentity,
    rules: [u8; 32],
    rules_source: [u8; 32],
    token: [u8; 32],
}
impl PalsWarmCapability {
    pub fn private_model_identity(&self) -> PrivateModelIdentity {
        self.identity
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.identity.model
    }
    pub fn model_epoch(&self) -> [u8; 32] {
        self.identity.model_epoch
    }
    pub fn rules_semantic_digest(&self) -> [u8; 32] {
        self.rules
    }
    pub fn rules_source_digest(&self) -> [u8; 32] {
        self.rules_source
    }
    pub fn query_semantics_digest(&self) -> [u8; 32] {
        asset::sha256(FROZEN_QUERY_SEMANTICS_V1.as_bytes())
    }
    pub fn implementation_digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-private-warm-implementation/1");
        for source in [
            include_bytes!("warm.rs").as_slice(),
            include_bytes!("../pals_onnx.rs").as_slice(),
        ] {
            hash.update((source.len() as u64).to_le_bytes());
            hash.update(source);
        }
        hash.finalize().into()
    }
    pub const fn schema(&self) -> &'static str {
        PRIVATE_WARM_SCHEMA
    }
    pub const fn query_semantics(&self) -> &'static str {
        FROZEN_QUERY_SEMANTICS_V1
    }
}

/// Owned physical input. The original bank lease must also stay with the actual
/// physical owner until Ready; copying a seed never certifies its completion.
/// The mode has no public setter, and a Warm seed comes only from a bank lease.
pub struct PalsWarmInput {
    input: PalsModelInput,
    prepared: PreparedWarmIdentity,
    initial_bits: Vec<u32>,
    invocation: PrivateInvocation,
    capability: PalsWarmCapability,
    owned_host_bytes: u64,
}

/// Sealed during Fresh preflight, while validation and canonical-key allocation
/// are still allowed. Neither the owned input nor this fixed-size authority has
/// a public mutation path. Binding sees this value, not `PalsModelInput`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedWarmIdentity {
    input_key: [u8; 32],
    role: PalsRole,
    model_epoch: [u8; 32],
}
impl PreparedWarmIdentity {
    /// Allocation-free by construction: only fixed-size identities, borrowed
    /// seed slices and static-literal errors are available at this boundary.
    /// Full input validation remains a separate worker-side check.
    fn validate_binding(
        self,
        prepared_capability: PalsWarmCapability,
        capability: PalsWarmCapability,
        current: PrivateInvocation,
        initial_bits: &[u32],
        lease: &PrivateSeedLease,
    ) -> Result<PrivateInvocation, BackendError> {
        if prepared_capability != capability
            || self.model_epoch != capability.identity.model_epoch
            || self.role == PalsRole::Validator
            || initial_bits.len() != LATENT_ELEMENTS
        {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Admission,
                "PALS Warm prepared capability or shape differs",
            ));
        }
        if initial_bits
            .iter()
            .any(|bits| !f32::from_bits(*bits).is_finite())
        {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "PALS Warm prepared seed contains a nonfinite FP32 value",
            ));
        }
        if current
            != (PrivateInvocation::Fresh {
                input_key: self.input_key,
            })
        {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Admission,
                "PALS Warm payload is already bound or its prepared key differs",
            ));
        }
        // Fresh has no seed provenance. Its complete admitted model identity
        // must therefore be retained by the lease and checked in both modes.
        if lease.model_identity() != capability.identity {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Admission,
                "PALS Warm lease model differs from loaded capability",
            ));
        }
        let invocation = lease.invocation();
        let expected = match invocation {
            PrivateInvocation::Fresh { input_key }
            | PrivateInvocation::ApproxWarmV1 { input_key, .. } => input_key,
        };
        if self.input_key != expected {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Admission,
                "PALS Warm lease and prepared input differ",
            ));
        }
        if let PrivateInvocation::ApproxWarmV1 { .. } = invocation {
            let source = lease.seed_provenance().ok_or_else(|| {
                BackendError::new(
                    K::IdentityMismatch,
                    S::Admission,
                    "PALS Warm lease has no accepted seed provenance",
                )
            })?;
            if source.model != capability.identity || source.role != self.role {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Admission,
                    "PALS Warm seed model or role differs from loaded capability",
                ));
            }
        }
        if matches!(invocation, PrivateInvocation::ApproxWarmV1 { .. })
            != lease.seed_bits().is_some()
        {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Admission,
                "PALS Warm lease mode and seed presence differ",
            ));
        }
        if let Some(bits) = lease.seed_bits() {
            if bits.len() != LATENT_ELEMENTS || bits.iter().any(|b| !f32::from_bits(*b).is_finite())
            {
                return Err(BackendError::new(
                    K::InvalidInput,
                    S::Admission,
                    "PALS Warm seed must contain all 6144 finite FP32 values",
                ));
            }
        }
        Ok(invocation)
    }
}
impl PalsWarmInput {
    /// Explicit bank-free Fresh execution for value/startup. Its full zero seed
    /// is ignored by the graph's false branch; it cannot authorize a Warm call.
    pub fn fresh(
        capability: &PalsWarmCapability,
        input: PalsModelInput,
    ) -> Result<Self, BackendError> {
        Self::make(capability, input, None)
    }
    pub fn from_lease(
        capability: &PalsWarmCapability,
        input: PalsModelInput,
        lease: &PrivateSeedLease,
    ) -> Result<Self, BackendError> {
        let mut result = Self::fresh(capability, input)?;
        result.bind_lease(capability, lease)?;
        Ok(result)
    }
    /// Bind an admitted lease into already-owned full storage without allocating.
    /// Prepare Fresh before bank.begin so a preflight allocation cannot abandon
    /// an active bank lease. All checks precede mutation; previous payload survives
    /// refusal. The caller still owns the lease until actual physical Ready.
    pub fn bind_lease(
        &mut self,
        capability: &PalsWarmCapability,
        lease: &PrivateSeedLease,
    ) -> Result<(), BackendError> {
        let invocation = self.prepared.validate_binding(
            self.capability,
            *capability,
            self.invocation,
            &self.initial_bits,
            lease,
        )?;
        if let Some(bits) = lease.seed_bits() {
            self.initial_bits.copy_from_slice(bits);
        }
        self.invocation = invocation;
        Ok(())
    }
    fn make(
        capability: &PalsWarmCapability,
        input: PalsModelInput,
        bits: Option<&[u32]>,
    ) -> Result<Self, BackendError> {
        validate_input(capability, &input)?;
        // Preflight caller-owned backing and full seed bytes before allocating.
        owned_bytes(&input, LATENT_ELEMENTS)?;
        if bits.is_some_and(|v| {
            v.len() != LATENT_ELEMENTS || v.iter().any(|b| !f32::from_bits(*b).is_finite())
        }) {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "PALS Warm seed must contain all 6144 finite FP32 values",
            ));
        }
        let input_key = input
            .canonical_input_key(&PalsModelConfig::baseline())
            .map_err(model_input)?;
        let mut initial_bits = Vec::new();
        initial_bits
            .try_reserve_exact(LATENT_ELEMENTS)
            .map_err(|e| allocation(&e))?;
        match bits {
            Some(bits) => initial_bits.extend_from_slice(bits),
            None => initial_bits.resize(LATENT_ELEMENTS, 0),
        }
        let owned_host_bytes = owned_bytes(&input, initial_bits.capacity())?;
        let prepared = PreparedWarmIdentity {
            input_key,
            role: input.role,
            model_epoch: input.model_epoch,
        };
        Ok(Self {
            input,
            prepared,
            initial_bits,
            invocation: PrivateInvocation::Fresh { input_key },
            capability: *capability,
            owned_host_bytes,
        })
    }
    pub fn input(&self) -> &PalsModelInput {
        &self.input
    }
    pub fn invocation(&self) -> PrivateInvocation {
        self.invocation
    }
    /// Entire payload and known Vec capacities, not observed process/native RSS.
    pub fn owned_host_bytes(&self) -> u64 {
        self.owned_host_bytes
    }
    pub(super) fn validate_for(&self, capability: &PalsWarmCapability) -> Result<(), BackendError> {
        validate_input(capability, &self.input)?;
        if &self.capability != capability
            || self.initial_bits.len() != LATENT_ELEMENTS
            || self
                .initial_bits
                .iter()
                .any(|b| !f32::from_bits(*b).is_finite())
            || owned_bytes(&self.input, self.initial_bits.capacity())? != self.owned_host_bytes
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "PALS Warm physical payload identity or shape differs",
            ));
        }
        let canonical = self
            .input
            .canonical_input_key(&PalsModelConfig::baseline())
            .map_err(model_input)?;
        let input_key = match self.invocation {
            PrivateInvocation::Fresh { input_key }
            | PrivateInvocation::ApproxWarmV1 { input_key, .. } => input_key,
        };
        if input_key != canonical
            || self.prepared.input_key != canonical
            || self.prepared.role != self.input.role
            || self.prepared.model_epoch != self.input.model_epoch
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "PALS Warm invocation does not bind its actual canonical input",
            ));
        }
        Ok(())
    }
    pub(super) fn prepare(&self) -> Result<ActiveWarmInputs, BackendError> {
        self.validate_for(&self.capability)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(LATENT_ELEMENTS)
            .map_err(|e| allocation(&e))?;
        values.extend(self.initial_bits.iter().map(|b| f32::from_bits(*b)));
        let host_bytes = (values.capacity() as u64)
            .checked_mul(4)
            .and_then(|v| v.checked_add(std::mem::size_of::<ActiveWarmInputs>() as u64 + 1))
            .ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "PALS Warm tensor byte overflow",
                )
            })?;
        let tensor_error = |e| {
            native(
                CauseCode::TensorCreate,
                "cannot own full PALS Warm seed tensor",
                e,
            )
        };
        Ok(ActiveWarmInputs {
            initial_latent: Tensor::from_array(([1, 16, 384], values)).map_err(tensor_error)?,
            warm_start: Tensor::from_array((
                Vec::<usize>::new(),
                vec![matches!(
                    self.invocation,
                    PrivateInvocation::ApproxWarmV1 { .. }
                )],
            ))
            .map_err(tensor_error)?,
            host_bytes,
        })
    }
}
pub(super) struct ActiveWarmInputs {
    pub(super) initial_latent: Tensor<f32>,
    pub(super) warm_start: Tensor<bool>,
    pub(super) host_bytes: u64,
}
fn allocation(error: &impl std::fmt::Display) -> BackendError {
    fail(
        K::ResourceExhausted,
        S::Admission,
        "PALS Warm owned allocation failed",
    )
    .with_external_cause(CauseCode::InputAllocation, error)
}
fn owned_bytes(input: &PalsModelInput, seeds: usize) -> Result<u64, BackendError> {
    let mut bytes = std::mem::size_of::<PalsWarmInput>() as u64;
    for (capacity, width) in [
        (input.board.capacity(), 1),
        (
            input.records.capacity(),
            std::mem::size_of::<crate::pals_model::PalsRecordToken>(),
        ),
        (input.required_critical_records.capacity(), 8),
        (
            input.candidates.capacity(),
            std::mem::size_of::<crate::pals_model::PalsCandidateToken>(),
        ),
        (input.divergence_features.capacity(), 32),
        (seeds, 4),
    ] {
        bytes = bytes
            .checked_add((capacity as u64).checked_mul(width as u64).ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "PALS Warm payload byte overflow",
                )
            })?)
            .ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "PALS Warm payload byte overflow",
                )
            })?;
    }
    if bytes > PAYLOAD_LIMIT {
        return Err(fail(
            K::ResourceExhausted,
            S::Admission,
            "PALS Warm payload exceeds its 1 MiB host reservation",
        ));
    }
    Ok(bytes)
}
fn validate_input(
    capability: &PalsWarmCapability,
    input: &PalsModelInput,
) -> Result<(), BackendError> {
    input
        .validate(&PalsModelConfig::baseline())
        .map_err(model_input)?;
    if input.role == PalsRole::Validator {
        return Err(fail(
            K::UnsupportedModel,
            S::Admission,
            "PALS CPU private Warm domain has no Validator",
        ));
    }
    if input.model_epoch != capability.model_epoch() {
        return Err(fail(
            K::IdentityMismatch,
            S::Admission,
            "PALS Warm input checkpoint differs from loaded session",
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WarmManifest {
    schema: String,
    model_semantics: String,
    layout: String,
    layout_revision: u32,
    checkpoint_sha256: String,
    config: PalsModelConfig,
    trained: bool,
    training_steps: u64,
    precision: String,
    tf32: bool,
    approximate: bool,
    roles: Vec<String>,
    validator_present: bool,
    task_names: Vec<String>,
    warm_graph_semantics: String,
    private_seed_policy: String,
    batch_mode: String,
    native_seed_owner_support: String,
    graphs: Vec<GraphManifest>,
    ownership: WarmOwnership,
    numeric_status: String,
    cuda: String,
    torch_version: String,
    expected_ort: String,
    rust_ort_crate: String,
    rules_input_profile: String,
    rules_input_semantic_sha256: String,
    rules_encoder_source_sha256: String,
    rules_profile_descriptor_sha256: String,
    rules_profile_canonical_sha256: String,
    rules_input_declaration: RulesDeclaration,
    learned_input_compatibility: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WarmOwnership {
    scope: String,
    layout_revision: u32,
    legacy_graph_sha256: String,
    legacy_ownership: ReaderInitializerBank,
    recursive_role_if_routes: usize,
    warm_mode_if_routes: usize,
    branch_local_initializers: usize,
    seed_values_per_batch_row: usize,
    fresh_initialization: String,
    warm_initialization: String,
    private_self_kv: String,
    accepted_seed_and_rules_context: String,
    inactive_private_execution: String,
    ort_prepack_copies: String,
    device_residency_sharing: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RulesDeclaration {
    schema: String,
    rules_input_profile: String,
    rules_input_semantic_sha256: String,
    rules_encoder_source_sha256: String,
    encoder_source: String,
    semantic_digest_algorithm: String,
    rules_base_semantics: String,
    semantic_fields: Vec<String>,
    learned_input_compatibility: String,
    metadata_features: Vec<String>,
    record_features: Vec<String>,
    query_features: Vec<String>,
    divergence_features: Vec<String>,
    config: PalsModelConfig,
}

fn warm_inputs() -> Vec<Interface> {
    let mut inputs = shared_inputs();
    inputs.extend([
        spec("initial_latent", TensorElementType::Float32, &[-1, 16, 384]),
        spec("warm_start", TensorElementType::Bool, &[]),
    ]);
    inputs
}

impl PalsOnnxBackend {
    /// Separate strict manifest. Existing Fresh load(), CLI and keys are untouched.
    /// CPU/host only; loading does not certify a numerical/physical seed acceptance.
    pub fn load_cpu_private_warm(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        validate_cpu(config, runtime.bundle_digest())?;
        if !path.is_absolute() {
            return Err(fail(
                K::InvalidInput,
                S::Asset,
                "PALS Warm manifest requires an absolute path",
            ));
        }
        let manifest_bytes = read_owned(
            path,
            MANIFEST_LIMIT,
            strict_digest(expected_manifest_sha256)?,
        )?;
        let manifest_digest = asset::sha256(&manifest_bytes);
        let manifest: WarmManifest = serde_json::from_slice(&manifest_bytes).map_err(|e| {
            fail(
                K::UnsupportedModel,
                S::Asset,
                "cannot decode strict PALS Warm manifest",
            )
            .with_external_cause(CauseCode::ModelLoad, &e)
        })?;
        validate_manifest(&manifest)?;
        let parent = path.parent().ok_or_else(|| {
            fail(
                K::InvalidInput,
                S::Asset,
                "PALS Warm manifest parent is absent",
            )
        })?;
        // Both bounded immutable graph vectors are verified before native loading.
        // No later pathname read supplies execution bytes, and combined admission
        // is 256 MiB rather than two independent 256 MiB allowances.
        let mut graph_bytes = Vec::new();
        let mut identities = Vec::new();
        let mut total = 0usize;
        for graph in &manifest.graphs {
            let remaining = GRAPH_LIMIT.checked_sub(total).ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "PALS Warm graph byte overflow",
                )
            })?;
            let bytes = read_owned(
                &parent.join(&graph.file),
                remaining,
                parse_sha256(&graph.sha256)?,
            )?;
            total = total.checked_add(bytes.len()).ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "PALS Warm graph byte overflow",
                )
            })?;
            identities.push(PalsGraphIdentity {
                role: graph.role.clone(),
                sha256: asset::sha256(&bytes),
                serialized_bytes: bytes.len() as u64,
            });
            graph_bytes.push(bytes);
        }
        let mut sessions = BTreeMap::new();
        for (graph, bytes) in manifest.graphs.iter().zip(graph_bytes) {
            let (session, _) = load_session(bytes, config, &graph.role, None)?;
            validate_warm_session(&session, graph, &manifest)?;
            sessions.insert(graph.role.as_str(), session);
        }
        let rules = parse_sha256(&manifest.rules_input_semantic_sha256)?;
        let model_epoch = parse_sha256(&manifest.checkpoint_sha256)?;
        let rules_source = parse_sha256(&manifest.rules_encoder_source_sha256)?;
        let mut hash = Sha256::new();
        for field in [
            b"rz-pals-private-warm-encoding/1".as_slice(),
            PALS_ENCODING_SCHEMA.as_bytes(),
            &rules,
            PRIVATE_WARM_GRAPH_SEMANTICS.as_bytes(),
            PRIVATE_WARM_SEMANTICS.as_bytes(),
            FROZEN_QUERY_SEMANTICS_V1.as_bytes(),
        ] {
            hash.update((field.len() as u64).to_le_bytes());
            hash.update(field);
        }
        let identity = PrivateModelIdentity {
            model: manifest_digest,
            encoding: hash.finalize().into(),
            precision: PrivatePrecision::Fp32,
            model_epoch,
            frozen_epoch: 1,
        };
        let token = capability_token(identity, rules, rules_source);
        let capability = PalsWarmCapability {
            identity,
            rules,
            rules_source,
            token,
        };
        let tokens = manifest
            .config
            .public_memory_tokens(manifest.config.max_records)
            .map_err(model_input)?;
        let memory = MemoryBank::new(
            1,
            public_owner_bytes(2 * tokens * 64, 2 * tokens * 64, tokens),
        )
        .map_err(public_bank_error)?;
        Ok(Self {
            public: sessions.remove("public"),
            proposer: None,
            critic: None,
            shared_pc: sessions.remove("shared_pc_warm"),
            layout: Layout::SharedPcIf,
            private_warm: Some(capability),
            runtime,
            config,
            model_config: manifest.config,
            epoch: model_epoch,
            manifest_digest,
            trained: manifest.trained,
            rules_input_profile: Some(manifest.rules_input_profile),
            rules_input_semantic_sha256: Some(rules),
            residency: PalsSessionResidency {
                graphs: identities,
                native_sessions: 2,
                layout: WARM_LAYOUT.into(),
                reader_initializer_bank: Some(manifest.ownership.legacy_ownership),
                role_reader_weights_shared: None,
                native_resident_parameter_bytes: None,
                vram_peak_bytes: None,
            },
            stats: PalsBackendStats::default(),
            cuda_mapping_audit: RefCell::default(),
            cuda_control_audit: None,
            memory: Some(memory),
            cached_memory_key: None,
            active_memory: None,
            record_pages: None,
            game_generation: 0,
            active: None,
            quarantine: None,
            startup_stage_probe: None,
            public_encodes: 0,
            public_cache_hits: 0,
            #[cfg(feature = "experimental-io-binding")]
            device_memory: None,
            #[cfg(feature = "experimental-io-binding")]
            device_role: None,
        })
    }
    pub fn private_warm_capability(&self) -> Option<&PalsWarmCapability> {
        self.private_warm.as_ref()
    }
    pub fn run_private_warm(
        &mut self,
        input: &PalsWarmInput,
    ) -> Result<PalsRawOutput, BackendError> {
        if self
            .startup_stage_probe
            .as_ref()
            .is_some_and(PalsStartupStageProbe::closed)
        {
            self.startup_stage_probe = None;
        }
        let trace = self
            .startup_stage_probe
            .as_ref()
            .and_then(|p| p.begin_role(input.input.role));
        let result = self.run_inner(&input.input, trace.as_ref(), Some(input));
        if let Some(trace) = trace {
            trace.finish(result.is_ok());
            if trace.probe.closed() {
                self.startup_stage_probe = None;
            }
        }
        result
    }
}

fn capability_token(identity: PrivateModelIdentity, rules: [u8; 32], source: [u8; 32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-private-warm-loaded-capability/1");
    hash.update(identity.model);
    hash.update(identity.encoding);
    hash.update(identity.model_epoch);
    hash.update(identity.frozen_epoch.to_le_bytes());
    hash.update(rules);
    hash.update(source);
    hash.finalize().into()
}
fn validate_cpu(config: PalsOnnxConfig, bundle: Option<[u8; 32]>) -> Result<(), BackendError> {
    config.validate()?;
    if config.provider != Provider::Cpu || config.device_public_memory || bundle.is_some() {
        return Err(fail(K::UnsupportedModel, S::Admission, "PALS private Warm v1 requires explicit CPU and host public memory; CUDA/device/V unsupported"));
    }
    Ok(())
}

fn validate_manifest(m: &WarmManifest) -> Result<(), BackendError> {
    m.config.validate().map_err(model_input)?;
    if m.schema != PRIVATE_WARM_SCHEMA
        || m.model_semantics != PALS_MODEL_SCHEMA
        || m.layout != WARM_LAYOUT
        || m.layout_revision != 1
        || m.precision != "fp32"
        || m.tf32
        || !m.approximate
        || m.warm_graph_semantics != PRIVATE_WARM_GRAPH_SEMANTICS
        || m.private_seed_policy != PRIVATE_WARM_SEMANTICS
        || m.batch_mode != "one_scalar_role_and_one_scalar_mode_per_physical_batch"
        || m.native_seed_owner_support != "not_registered_by_python_export"
        || m.roles != ["proposer", "critic"]
        || m.validator_present
        || !m.task_names.iter().map(String::as_str).eq(V_TASK_NAMES)
        || m.expected_ort != "1.22.0"
        || m.rust_ort_crate != "2.0.0-rc.10"
        || m.numeric_status != "not_run"
        || m.cuda != "not_run"
        || m.torch_version.is_empty()
        || m.torch_version.len() > 128
        || m.torch_version.chars().any(char::is_control)
        || m.trained != (m.training_steps > 0)
        || m.graphs.len() != 2
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "expected explicit approximate CPU P/C-only Warm export",
        ));
    }
    for digest in [
        &m.checkpoint_sha256,
        &m.rules_input_semantic_sha256,
        &m.rules_encoder_source_sha256,
        &m.rules_profile_descriptor_sha256,
        &m.rules_profile_canonical_sha256,
    ] {
        strict_digest(digest)?;
    }
    validate_rules(m)?;
    let o = &m.ownership;
    strict_digest(&o.legacy_graph_sha256)?;
    validate_reader_bank(&o.legacy_ownership)?;
    if o.scope != "onnx_serialized_initializers_only"
        || o.layout_revision != 1
        || o.recursive_role_if_routes != 6
        || o.warm_mode_if_routes != 1
        || o.branch_local_initializers != 0
        || o.seed_values_per_batch_row != LATENT_ELEMENTS
        || o.fresh_initialization != "unchanged_legacy_role_if"
        || o.warm_initialization != "complete_final_latent_identity_no_query_addition"
        || o.private_self_kv != "recomputed_by_unchanged_readers"
        || o.accepted_seed_and_rules_context != "requires_native_owner_validation"
        || o.inactive_private_execution != "requires_runtime_profile"
        || o.ort_prepack_copies != "unknown"
        || o.device_residency_sharing != "unknown"
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "PALS Warm serialized ownership declaration differs",
        ));
    }
    for (index, (role, filename)) in [
        ("public", "public_memory.onnx"),
        ("shared_pc_warm", "shared_pc_if_approx_warm_v1.onnx"),
    ]
    .into_iter()
    .enumerate()
    {
        let graph = &m.graphs[index];
        if graph.role != role || graph.file != filename || graph.opset != 17 {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS Warm ordered graph names or opset differ",
            ));
        }
        strict_digest(&graph.sha256)?;
        validate_declared_interface(
            &graph.inputs,
            &if index == 0 {
                public_inputs()
            } else {
                warm_inputs()
            },
        )?;
        validate_declared_interface(
            &graph.outputs,
            &if index == 0 {
                memory_outputs()
            } else {
                shared_outputs()
            },
        )?;
    }
    Ok(())
}
fn strict_digest(text: &str) -> Result<[u8; 32], BackendError> {
    if text.len() != 64
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS Warm digest must be 64 lowercase hexadecimal characters",
        ));
    }
    parse_sha256(text)
}
fn validate_rules(m: &WarmManifest) -> Result<(), BackendError> {
    let d = &m.rules_input_declaration;
    if m.rules_input_profile != "rz-pals-rules-fields-v1"
        || m.learned_input_compatibility != "unverified_declaration_only"
        || d.schema != "rovezero.pals-rules-descriptor.v1"
        || d.rules_input_profile != m.rules_input_profile
        || d.rules_input_semantic_sha256 != m.rules_input_semantic_sha256
        || d.rules_encoder_source_sha256 != m.rules_encoder_source_sha256
        || d.encoder_source != "crates/rz-uci/src/pals_native.rs"
        || d.semantic_digest_algorithm != "sha256_u64le_length_prefixed_utf8_fields"
        || d.learned_input_compatibility != m.learned_input_compatibility
        || d.config != m.config
        || d.rules_base_semantics.is_empty()
        || d.semantic_fields.len() != 58
        || d.metadata_features.len() != 16
        || d.record_features.len() != 16
        || d.query_features.len() != 16
        || d.divergence_features.len() != 8
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS Warm Rules source declaration differs",
        ));
    }
    let fields: Vec<&str> = std::iter::once(d.rules_input_profile.as_str())
        .chain(std::iter::once(d.rules_base_semantics.as_str()))
        .chain(
            d.metadata_features
                .iter()
                .chain(&d.record_features)
                .chain(&d.query_features)
                .chain(&d.divergence_features)
                .map(String::as_str),
        )
        .collect();
    let mut hash = Sha256::new();
    for (expected, actual) in fields.iter().zip(&d.semantic_fields) {
        if expected != &actual.as_str()
            || actual.is_empty()
            || actual.len() > 4096
            || actual.chars().any(char::is_control)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS Warm Rules feature vocabulary differs",
            ));
        }
        hash.update((actual.len() as u64).to_le_bytes());
        hash.update(actual.as_bytes());
    }
    let semantic: [u8; 32] = hash.finalize().into();
    if semantic != strict_digest(&m.rules_input_semantic_sha256)? {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS Warm Rules semantic digest differs",
        ));
    }
    let canonical = canonical_value(serde_json::to_value(d).map_err(model_input)?);
    let bytes = serde_json::to_vec(&canonical).map_err(model_input)?;
    if bytes.len() > MANIFEST_LIMIT
        || asset::sha256(&bytes) != strict_digest(&m.rules_profile_canonical_sha256)?
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS Warm Rules canonical declaration digest differs",
        ));
    }
    Ok(())
}
fn canonical_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: BTreeMap<_, _> = map.into_iter().collect();
            serde_json::Value::Object(
                sorted
                    .into_iter()
                    .map(|(k, v)| (k, canonical_value(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonical_value).collect())
        }
        other => other,
    }
}
fn validate_warm_session(
    session: &Session,
    graph: &GraphManifest,
    m: &WarmManifest,
) -> Result<(), BackendError> {
    let inputs = if graph.role == "public" {
        public_inputs()
    } else {
        warm_inputs()
    };
    let outputs = if graph.role == "public" {
        memory_outputs()
    } else {
        shared_outputs()
    };
    let valid = |name: &str, ty: &ValueType, expected: &[Interface]| {
        expected.iter().any(|v|
        name == v.name && matches!(ty, ValueType::Tensor { ty, shape, .. } if *ty == v.dtype && shape.as_ref() == v.shape))
    };
    if session.inputs.len() != inputs.len()
        || session.outputs.len() != outputs.len()
        || !session
            .inputs
            .iter()
            .all(|v| valid(&v.name, &v.input_type, &inputs))
        || !session
            .outputs
            .iter()
            .all(|v| valid(&v.name, &v.output_type, &outputs))
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "loaded PALS Warm tensor interface differs",
        ));
    }
    let metadata = session.metadata().map_err(|e| {
        native(
            CauseCode::ModelLoad,
            "cannot inspect PALS Warm graph metadata",
            e,
        )
    })?;
    let steps = m.training_steps.to_string();
    for (key, expected) in [
        ("schema", PRIVATE_WARM_SCHEMA),
        ("model_semantics", PALS_MODEL_SCHEMA),
        ("layout", WARM_LAYOUT),
        ("layout_revision", "1"),
        ("role", graph.role.as_str()),
        ("checkpoint_sha256", m.checkpoint_sha256.as_str()),
        ("precision", "fp32"),
        ("trained", if m.trained { "true" } else { "false" }),
        ("training_steps", steps.as_str()),
        ("expected_ort", "1.22.0"),
        ("approximate", "true"),
        ("warm_graph_semantics", PRIVATE_WARM_GRAPH_SEMANTICS),
        ("private_seed_policy", PRIVATE_WARM_SEMANTICS),
        ("public_kv", "shared_role_neutral"),
        ("rules_input_profile", m.rules_input_profile.as_str()),
        (
            "rules_input_semantic_sha256",
            m.rules_input_semantic_sha256.as_str(),
        ),
        (
            "rules_encoder_source_sha256",
            m.rules_encoder_source_sha256.as_str(),
        ),
    ] {
        if metadata
            .custom(key)
            .map_err(|e| {
                native(
                    CauseCode::ModelLoad,
                    "cannot read PALS Warm semantic metadata",
                    e,
                )
            })?
            .as_deref()
            != Some(expected)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "PALS Warm graph metadata differs from registered manifest",
            ));
        }
    }
    Ok(())
}

/// Snapshot read bound to one opened regular file. The owned bytes, never a
/// reopened path, supply native commit_from_memory. No shell/env/dynamic loader.
fn read_owned(path: &Path, limit: usize, digest: [u8; 32]) -> Result<Vec<u8>, BackendError> {
    if limit == 0 {
        return Err(fail(
            K::ResourceExhausted,
            S::Asset,
            "PALS Warm serialized byte budget exhausted",
        ));
    }
    let before = std::fs::symlink_metadata(path)
        .map_err(|_| fail(K::Io, S::Asset, "cannot inspect PALS Warm artifact"))?;
    if !before.is_file() || before.len() == 0 || before.len() > limit as u64 {
        return Err(fail(
            K::ResourceExhausted,
            S::Asset,
            "PALS Warm artifact must be a bounded nonempty regular file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Linux O_NOFOLLOW | O_NONBLOCK: prevent a replaced symlink/FIFO from
        // turning this bounded snapshot into ambient file IO or a blocking open.
        options.custom_flags(0x20000 | 0x800);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // OPEN_REPARSE_POINT and read-sharing only; writers/deleters cannot
        // replace the opened source while this immutable snapshot is read.
        options.custom_flags(0x0020_0000).share_mode(1);
    }
    let mut file = options
        .open(path)
        .map_err(|_| fail(K::Io, S::Asset, "cannot open PALS Warm artifact snapshot"))?;
    let opened = file
        .metadata()
        .map_err(|_| fail(K::Io, S::Asset, "cannot inspect opened PALS Warm artifact"))?;
    if !same_file_stamp(&before, &opened) {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS Warm artifact changed before snapshot read",
        ));
    }
    let expected = usize::try_from(opened.len()).map_err(|_| {
        fail(
            K::ResourceExhausted,
            S::Asset,
            "PALS Warm artifact byte count overflows",
        )
    })?;
    let reservation = expected.checked_add(1).ok_or_else(|| {
        fail(
            K::ResourceExhausted,
            S::Asset,
            "PALS Warm artifact byte reservation overflows",
        )
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(reservation)
        .map_err(|e| allocation(&e))?;
    (&mut file)
        .take(reservation as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| fail(K::Io, S::Asset, "cannot read PALS Warm artifact snapshot"))?;
    let after = file
        .metadata()
        .map_err(|_| fail(K::Io, S::Asset, "cannot inspect final PALS Warm artifact"))?;
    let path_after = std::fs::symlink_metadata(path)
        .map_err(|_| fail(K::Io, S::Asset, "PALS Warm artifact pathname disappeared"))?;
    if bytes.len() != expected
        || !same_file_stamp(&opened, &after)
        || !same_file_stamp(&opened, &path_after)
        || asset::sha256(&bytes) != digest
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "PALS Warm artifact snapshot size, identity or digest changed",
        ));
    }
    Ok(bytes)
}
fn same_file_stamp(a: &Metadata, b: &Metadata) -> bool {
    if !a.is_file()
        || !b.is_file()
        || a.len() != b.len()
        || !a
            .modified()
            .ok()
            .zip(b.modified().ok())
            .is_some_and(|(a, b)| a == b)
    {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        a.dev() == b.dev()
            && a.ino() == b.ino()
            && a.mode() == b.mode()
            && a.mtime() == b.mtime()
            && a.mtime_nsec() == b.mtime_nsec()
            && a.ctime() == b.ctime()
            && a.ctime_nsec() == b.ctime_nsec()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // Reject reparse points even if their target reports a file-shaped type.
        a.file_attributes() & 0x400 == 0
            && b.file_attributes() & 0x400 == 0
            && a.creation_time() == b.creation_time()
            && a.last_write_time() == b.last_write_time()
            && a.file_attributes() == b.file_attributes()
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_private::{
        PhysicalSeedCompletion, PrivateInvocationMode, PrivateRulesContext, PrivateSeedBank,
        PrivateSeedLimits, PrivateSeedRequest, PrivateSituationContext,
    };
    use crate::worker::{PhysicalPoll, PhysicalRun};
    use rz_contracts::{pals::SituationHandle, CancelToken};
    use rz_position::Position;
    use std::time::Duration;

    // Authored declarations/capabilities below are available only to this child
    // unit. They never load an ORT session or claim Native numerical acceptance.
    fn declaration() -> RulesDeclaration {
        let features = |n| {
            (0..n)
                .map(|i| format!("authored_field_{i}"))
                .collect::<Vec<_>>()
        };
        let mut d = RulesDeclaration {
            schema: "rovezero.pals-rules-descriptor.v1".into(),
            rules_input_profile: "rz-pals-rules-fields-v1".into(),
            rules_input_semantic_sha256: String::new(),
            rules_encoder_source_sha256: "07".repeat(32),
            encoder_source: "crates/rz-uci/src/pals_native.rs".into(),
            semantic_digest_algorithm: "sha256_u64le_length_prefixed_utf8_fields".into(),
            rules_base_semantics: "authored_rules_fields".into(),
            semantic_fields: Vec::new(),
            learned_input_compatibility: "unverified_declaration_only".into(),
            metadata_features: features(16),
            record_features: features(16),
            query_features: features(16),
            divergence_features: features(8),
            config: PalsModelConfig::baseline(),
        };
        d.semantic_fields = std::iter::once(d.rules_input_profile.clone())
            .chain(std::iter::once(d.rules_base_semantics.clone()))
            .chain(
                d.metadata_features
                    .iter()
                    .chain(&d.record_features)
                    .chain(&d.query_features)
                    .chain(&d.divergence_features)
                    .cloned(),
            )
            .collect();
        let mut h = Sha256::new();
        for f in &d.semantic_fields {
            h.update((f.len() as u64).to_le_bytes());
            h.update(f.as_bytes());
        }
        d.rules_input_semantic_sha256 = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        d
    }
    fn descriptors(values: Vec<Interface>) -> Vec<TensorManifest> {
        values
            .into_iter()
            .map(|v| TensorManifest {
                name: v.name.into(),
                dtype: match v.dtype {
                    TensorElementType::Float32 => "FLOAT",
                    TensorElementType::Int64 => "INT64",
                    TensorElementType::Bool => "BOOL",
                    _ => unreachable!(),
                }
                .into(),
                shape: v
                    .shape
                    .iter()
                    .map(|dim| {
                        if *dim == -1 {
                            serde_json::json!("batch")
                        } else {
                            serde_json::json!(dim)
                        }
                    })
                    .collect(),
            })
            .collect()
    }
    fn manifest() -> WarmManifest {
        let d = declaration();
        let canonical =
            serde_json::to_vec(&canonical_value(serde_json::to_value(&d).unwrap())).unwrap();
        WarmManifest {
            schema: PRIVATE_WARM_SCHEMA.into(),
            model_semantics: PALS_MODEL_SCHEMA.into(),
            layout: WARM_LAYOUT.into(),
            layout_revision: 1,
            checkpoint_sha256: "03".repeat(32),
            config: PalsModelConfig::baseline(),
            trained: false,
            training_steps: 0,
            precision: "fp32".into(),
            tf32: false,
            approximate: true,
            roles: vec!["proposer".into(), "critic".into()],
            validator_present: false,
            task_names: V_TASK_NAMES.iter().map(|v| v.to_string()).collect(),
            warm_graph_semantics: PRIVATE_WARM_GRAPH_SEMANTICS.into(),
            private_seed_policy: PRIVATE_WARM_SEMANTICS.into(),
            batch_mode: "one_scalar_role_and_one_scalar_mode_per_physical_batch".into(),
            native_seed_owner_support: "not_registered_by_python_export".into(),
            graphs: vec![
                GraphManifest {
                    file: "public_memory.onnx".into(),
                    sha256: "05".repeat(32),
                    role: "public".into(),
                    inputs: descriptors(public_inputs()),
                    outputs: descriptors(memory_outputs()),
                    opset: 17,
                },
                GraphManifest {
                    file: "shared_pc_if_approx_warm_v1.onnx".into(),
                    sha256: "06".repeat(32),
                    role: "shared_pc_warm".into(),
                    inputs: descriptors(warm_inputs()),
                    outputs: descriptors(shared_outputs()),
                    opset: 17,
                },
            ],
            ownership: WarmOwnership {
                scope: "onnx_serialized_initializers_only".into(),
                layout_revision: 1,
                legacy_graph_sha256: "08".repeat(32),
                legacy_ownership: ReaderInitializerBank {
                    scope: "onnx_serialized_initializers_only".into(),
                    layout_revision: 1,
                    shared_parameters: vec![SharedParameter {
                        parameter: "reader.weight".into(),
                        initializer: "reader_weight".into(),
                        bytes: 4,
                        sha256: "0a".repeat(32),
                        source_parameter_sha256: "0a".repeat(32),
                        shape: vec![1],
                    }],
                    unique_shared_parameters: 1,
                    shared_weight_bytes: 4,
                    if_routes: 6,
                    branch_local_initializers: 0,
                    branch_operations: (0..6)
                        .flat_map(|i| {
                            ["then_branch", "else_branch"].map(|branch| BranchOperations {
                                route: format!("route_{i}"),
                                branch: branch.into(),
                                nodes: 1,
                            })
                        })
                        .collect(),
                    inactive_private_execution: "requires_runtime_profile".into(),
                    ort_prepack_copies: "unknown".into(),
                    device_residency_sharing: "unknown".into(),
                },
                recursive_role_if_routes: 6,
                warm_mode_if_routes: 1,
                branch_local_initializers: 0,
                seed_values_per_batch_row: LATENT_ELEMENTS,
                fresh_initialization: "unchanged_legacy_role_if".into(),
                warm_initialization: "complete_final_latent_identity_no_query_addition".into(),
                private_self_kv: "recomputed_by_unchanged_readers".into(),
                accepted_seed_and_rules_context: "requires_native_owner_validation".into(),
                inactive_private_execution: "requires_runtime_profile".into(),
                ort_prepack_copies: "unknown".into(),
                device_residency_sharing: "unknown".into(),
            },
            numeric_status: "not_run".into(),
            cuda: "not_run".into(),
            torch_version: "authored-fixture".into(),
            expected_ort: "1.22.0".into(),
            rust_ort_crate: "2.0.0-rc.10".into(),
            rules_input_profile: d.rules_input_profile.clone(),
            rules_input_semantic_sha256: d.rules_input_semantic_sha256.clone(),
            rules_encoder_source_sha256: d.rules_encoder_source_sha256.clone(),
            rules_profile_descriptor_sha256: "09".repeat(32),
            rules_profile_canonical_sha256: asset::sha256(&canonical)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
            rules_input_declaration: d,
            learned_input_compatibility: "unverified_declaration_only".into(),
        }
    }
    fn capability() -> PalsWarmCapability {
        let identity = PrivateModelIdentity {
            model: [1; 32],
            encoding: [2; 32],
            precision: PrivatePrecision::Fp32,
            model_epoch: [3; 32],
            frozen_epoch: 1,
        };
        PalsWarmCapability {
            identity,
            rules: [4; 32],
            rules_source: [5; 32],
            token: capability_token(identity, [4; 32], [5; 32]),
        }
    }
    fn input(role: PalsRole) -> PalsModelInput {
        PalsModelInput {
            role,
            board: vec![0; 64],
            metadata: [0.; 16],
            records: vec![],
            required_critical_records: vec![],
            candidates: vec![],
            divergence_features: vec![],
            query: [0.; 16],
            situation_revision: 0,
            history_digest: [8; 32],
            model_epoch: [3; 32],
        }
    }
    #[test]
    fn strict_manifest_rejects_legacy_v_seed_mode_and_source_mismatches() {
        validate_manifest(&manifest()).unwrap();
        let mutations: [fn(&mut WarmManifest); 7] = [
            |m| m.schema = PALS_MODEL_SCHEMA.into(),
            |m| m.approximate = false,
            |m| m.validator_present = true,
            |m| m.ownership.seed_values_per_batch_row = 1,
            |m| m.graphs[1].file = "../private.onnx".into(),
            |m| m.rules_input_declaration.rules_encoder_source_sha256 = "ff".repeat(32),
            |m| m.rules_input_declaration.query_features[7] = "unregistered_time_meaning".into(),
        ];
        for mutation in mutations {
            let mut m = manifest();
            mutation(&mut m);
            assert!(validate_manifest(&m).is_err());
        }
        let mut m = manifest();
        m.graphs[1]
            .inputs
            .iter_mut()
            .find(|v| v.name == "initial_latent")
            .unwrap()
            .shape[2] = serde_json::json!(383);
        assert!(validate_manifest(&m).is_err());
        let mut m = manifest();
        m.graphs[1]
            .inputs
            .iter_mut()
            .find(|v| v.name == "warm_start")
            .unwrap()
            .dtype = "FLOAT".into();
        assert!(validate_manifest(&m).is_err());
    }
    #[test]
    fn strict_decode_rejects_duplicate_unknown_and_invalid_digest() {
        for (text, detail) in [
            (r#"{"schema":"x","schema":"y"}"#, "duplicate field"),
            (r#"{"unexpected":true}"#, "unknown field"),
        ] {
            let e = serde_json::from_str::<WarmManifest>(text).err().unwrap();
            assert!(e.to_string().contains(detail));
        }
        assert!(strict_digest(&"AA".repeat(32)).is_err());
        assert!(strict_digest(&"0".repeat(63)).is_err());
    }
    #[test]
    fn cpu_only_guard_refuses_device_and_cuda_domains() {
        validate_cpu(PalsOnnxConfig::cpu(), None).unwrap();
        let mut c = PalsOnnxConfig::cpu();
        c.device_public_memory = true;
        assert!(validate_cpu(c, None).is_err());
        let mut c = PalsOnnxConfig::cpu();
        c.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 1024 * 1024,
        };
        assert!(validate_cpu(c, Some([1; 32])).is_err());
        assert!(validate_cpu(PalsOnnxConfig::cpu(), Some([1; 32])).is_err());
    }
    #[test]
    fn bank_free_fresh_preserves_canonical_input_and_full_finite_ignored_seed() {
        let c = capability();
        let i = input(PalsRole::Proposer);
        let key = i.canonical_input_key(&PalsModelConfig::baseline()).unwrap();
        let p = PalsWarmInput::fresh(&c, i).unwrap();
        assert_eq!(p.invocation(), PrivateInvocation::Fresh { input_key: key });
        assert_eq!(p.prepared.input_key, key);
        assert_eq!(p.prepared.role, p.input().role);
        assert_eq!(p.prepared.model_epoch, c.model_epoch());
        assert_eq!(p.initial_bits.len(), LATENT_ELEMENTS);
        assert!(p.initial_bits.iter().all(|b| *b == 0));
        assert!(p.owned_host_bytes() >= (LATENT_ELEMENTS * 4) as u64);
        let mut other = c;
        other.token[0] ^= 1;
        assert!(p.validate_for(&other).is_err());
        assert!(PalsWarmInput::fresh(&c, input(PalsRole::Validator)).is_err());
    }
    #[test]
    fn fresh_lease_requires_the_full_admitted_model_and_preserves_prepared_payload() {
        fn fixed_copy<T: Copy>() {}
        fixed_copy::<PreparedWarmIdentity>();
        fixed_copy::<PrivateModelIdentity>();

        let c = capability();
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let i = input(PalsRole::Proposer);
        let request = PrivateSeedRequest {
            role: i.role,
            model: c.private_model_identity(),
            game_generation: 1,
            search_generation: 1,
            context: PrivateRulesContext::seal(
                &snapshot,
                &i,
                &PalsModelConfig::baseline(),
                PrivateSituationContext {
                    situation: SituationHandle {
                        slot: 0,
                        generation: 1,
                    },
                    prefix: [1; 32],
                    focus: [2; 32],
                },
            )
            .unwrap(),
        };
        let changes: [fn(&mut PrivateModelIdentity); 3] = [
            |model| model.model[0] ^= 1,
            |model| model.encoding[0] ^= 1,
            |model| model.frozen_epoch += 1,
        ];
        for change in changes {
            let bank = PrivateSeedBank::new(
                PrivateSeedLimits {
                    slots_per_role: 1,
                    max_bank_bytes: 256 * 1024,
                    max_transient_bytes: 256 * 1024,
                    latent_elements: LATENT_ELEMENTS,
                },
                1,
            )
            .unwrap();
            let mut admitted = request.clone();
            change(&mut admitted.model);
            let cancel = CancelToken::new();
            let until = Instant::now() + Duration::from_secs(2);
            let lease = bank
                .begin(
                    admitted.clone(),
                    &snapshot,
                    PrivateInvocationMode::Fresh,
                    &cancel,
                    until,
                )
                .unwrap();
            assert_eq!(lease.model_identity(), admitted.model);
            assert!(lease.seed_provenance().is_none());
            let mut p = PalsWarmInput::fresh(&c, i.clone()).unwrap();
            let pointer = p.initial_bits.as_ptr();
            let capacity = p.initial_bits.capacity();
            let prepared = p.prepared;
            let invocation = p.invocation;
            let error = p.bind_lease(&c, &lease).unwrap_err();
            assert_eq!(error.kind, K::IdentityMismatch);
            // These binding errors contain static literals only. No diagnostic
            // String/Box or external-cause capture is created on refusal.
            assert!(error.cause.is_none() && error.native.is_none());
            assert_eq!(p.initial_bits.as_ptr(), pointer);
            assert_eq!(p.initial_bits.capacity(), capacity);
            assert!(p.initial_bits.iter().all(|b| *b == 0));
            assert_eq!(p.prepared, prepared);
            assert_eq!(p.invocation, invocation);
            assert_eq!(lease.model_identity(), admitted.model);
            // No NN was submitted; dropping remains the conservative unknown
            // boundary. Refusal never manufactures a physical completion.
            drop(lease);
            assert!(bank.snapshot().unwrap().admission_closed);
        }
    }
    #[test]
    fn all_seed_bits_are_bound_without_allocation_after_real_fixture_worker_ready() {
        let c = capability();
        let i = input(PalsRole::Critic);
        let position = Position::startpos();
        let snapshot = position.snapshot();
        let request = PrivateSeedRequest {
            role: i.role,
            model: c.private_model_identity(),
            game_generation: 1,
            search_generation: 1,
            context: PrivateRulesContext::seal(
                &snapshot,
                &i,
                &PalsModelConfig::baseline(),
                PrivateSituationContext {
                    situation: SituationHandle {
                        slot: 0,
                        generation: 1,
                    },
                    prefix: [1; 32],
                    focus: [2; 32],
                },
            )
            .unwrap(),
        };
        let bank = PrivateSeedBank::new(
            PrivateSeedLimits {
                slots_per_role: 1,
                max_bank_bytes: 256 * 1024,
                max_transient_bytes: 256 * 1024,
                latent_elements: LATENT_ELEMENTS,
            },
            1,
        )
        .unwrap();
        let cancel = CancelToken::new();
        let until = Instant::now() + Duration::from_secs(2);
        let first = bank
            .begin(
                request.clone(),
                &snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                until,
            )
            .unwrap();
        let expected: Vec<f32> = (0..LATENT_ELEMENTS)
            .map(|i| (i as f32 - 3000.) / 1024.)
            .collect();
        // This is an actual bounded CPU fixture worker fence, not an ORT/GPU or
        // Native support claim. No complete_known call precedes Physical Ready.
        let output = expected.clone();
        let mut worker =
            SingleWorker::spawn_with_outcome(move |_: &()| PhysicalRun::Complete(output.clone()))
                .unwrap();
        let mut physical = worker.submit(()).unwrap();
        let latent = loop {
            match physical.poll() {
                PhysicalPoll::Ready(v) => break v,
                PhysicalPoll::Pending if Instant::now() < until => std::thread::yield_now(),
                _ => panic!("fixture physical completion was not confirmed"),
            }
        };
        first
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&latent)
            .unwrap()
            .commit(&request, &snapshot, &cancel, until)
            .unwrap();
        let mut p = PalsWarmInput::fresh(&c, i).unwrap();
        let pointer = p.initial_bits.as_ptr();
        let capacity = p.initial_bits.capacity();
        let prepared = p.prepared;
        let lease = bank
            .begin(
                request,
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &cancel,
                until,
            )
            .unwrap();
        assert_eq!(lease.model_identity(), c.private_model_identity());
        let seed_pointer = lease.seed_bits().unwrap().as_ptr();
        let provenance = *lease.seed_provenance().unwrap();
        let mut other = c;
        other.token[0] ^= 1;
        let error = p.bind_lease(&other, &lease).unwrap_err();
        assert_eq!(error.kind, K::IdentityMismatch);
        assert!(error.cause.is_none() && error.native.is_none());
        assert_eq!(p.prepared, prepared);
        assert_eq!(p.initial_bits.as_ptr(), pointer);
        assert!(p.initial_bits.iter().all(|b| *b == 0));
        p.initial_bits[LATENT_ELEMENTS - 1] = f32::NAN.to_bits();
        let error = p.bind_lease(&c, &lease).unwrap_err();
        assert_eq!(error.kind, K::InvalidInput);
        assert!(error.cause.is_none() && error.native.is_none());
        assert_eq!(p.initial_bits[LATENT_ELEMENTS - 1], f32::NAN.to_bits());
        p.initial_bits[LATENT_ELEMENTS - 1] = 0;
        assert_eq!(lease.seed_bits().unwrap().as_ptr(), seed_pointer);
        assert_eq!(*lease.seed_provenance().unwrap(), provenance);
        p.bind_lease(&c, &lease).unwrap();
        assert_eq!(p.initial_bits.as_ptr(), pointer);
        assert_eq!(p.initial_bits.capacity(), capacity);
        assert_eq!(
            p.initial_bits,
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(matches!(
            p.invocation(),
            PrivateInvocation::ApproxWarmV1 { .. }
        ));
        // Only this child test can corrupt the private immutable input/cache.
        // Worker validation still recomputes the actual canonical identity and
        // rejects such corruption before constructing native tensors.
        p.input.query[0] = f32::NAN;
        assert!(p.validate_for(&c).is_err());
        // The second invocation is deliberately never dispatched: dropping its
        // lease remains the existing conservative unknown-retention boundary.
        drop(lease);
        assert!(bank.snapshot().unwrap().admission_closed);
    }
    #[test]
    fn payload_preflight_charges_capacity_and_rejects_nonfinite_seed() {
        let c = capability();
        let mut i = input(PalsRole::Proposer);
        i.board.reserve_exact(PAYLOAD_LIMIT as usize);
        assert!(PalsWarmInput::fresh(&c, i).is_err());
        let mut bits = vec![0; LATENT_ELEMENTS];
        bits[6143] = f32::NAN.to_bits();
        assert!(PalsWarmInput::make(&c, input(PalsRole::Proposer), Some(&bits)).is_err());
        assert!(PalsWarmInput::make(&c, input(PalsRole::Proposer), Some(&bits[..6143])).is_err());
    }
    #[test]
    fn immutable_snapshot_rejects_digest_and_keeps_owned_bytes_after_source_change() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rz-warm-snapshot-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        struct Owned(std::path::PathBuf);
        impl Drop for Owned {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let _owned = Owned(path.clone());
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        f.write_all(b"authored-public-graph-bytes").unwrap();
        drop(f);
        assert!(read_owned(&path, 1024, [0; 32]).is_err());
        let bytes = read_owned(&path, 1024, asset::sha256(b"authored-public-graph-bytes")).unwrap();
        std::fs::write(&path, b"changed-source").unwrap();
        assert_eq!(bytes, b"authored-public-graph-bytes");
    }
}
