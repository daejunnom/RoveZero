//! Explicit CUDA/device Warm owner. Serialized declarations, loaded sessions,
//! physical completion and caller seed acceptance are separate boundaries.
//! The complete seed is transferred through the real private I/O binding;
//! public K/V stay in the existing device owner. No device/private allocator
//! sharing or VRAM saving is inferred from the graph or these types.
use super::*;
use crate::pals_model::PalsModelProfile;
use crate::pals_private::PRIVATE_CUDA_WARM_SEMANTICS;

pub const PRIVATE_CUDA_WARM_SCHEMA: &str = "rovezero.pals-private-cuda-warm.v2";
pub const PRIVATE_CUDA_WARM_EXECUTION_DOMAIN: &str = "cuda_device_io_binding_v2";
pub const PRIVATE_CUDA_WARM_GRAPH_SEMANTICS: &str =
    "rz-pals-private-native-cuda-warm/2;accepted-final-latent;no-query-double-add;fixed-2-iterations;private-self-kv-recomputed;fp32;pc-only;device-public-kv;io-binding-required";
pub const PRIVATE_CUDA_QUERY_SEMANTICS: &str =
    "rz-pals-private-cuda-query/2;same-actual-nonrecord-fp32-and-full-lines;logical-game-search-situation-prefix-focus-role-isolated;records-revision-current;original-deadline-cancel-fixed;value-fresh";
const CUDA_LAYOUT: &str = "shared_pc_if_approx_cuda_warm_v2";
const CUDA_GRAPH_FILE: &str = "shared_pc_if_approx_cuda_warm_v2.onnx";
const CUDA_PAYLOAD_LIMIT: u64 = 2 * 1024 * 1024;

/// No public declaration-based constructor. The witness is issued after both
/// exact CUDA sessions and their registered interfaces/metadata have loaded.
/// It does not certify native parameter sharing, numerics, placement, a seed,
/// or a physical invocation. Quarantined owners stop issuing this capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsCudaWarmCapability {
    identity: PrivateModelIdentity,
    rules: [u8; 32],
    rules_source: [u8; 32],
    profile: PalsModelProfile,
    device_id: i32,
    runtime_binary: [u8; 32],
    runtime_bundle: [u8; 32],
    token: [u8; 32],
}
impl PalsCudaWarmCapability {
    pub fn private_model_identity(&self) -> PrivateModelIdentity {
        self.identity
    }
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.identity.model
    }
    pub fn model_epoch(&self) -> [u8; 32] {
        self.identity.model_epoch
    }
    pub fn model_config(&self) -> PalsModelConfig {
        PalsModelConfig::for_profile(self.profile)
    }
    pub fn rules_semantic_digest(&self) -> [u8; 32] {
        self.rules
    }
    pub fn rules_source_digest(&self) -> [u8; 32] {
        self.rules_source
    }
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    pub fn query_semantics_digest(&self) -> [u8; 32] {
        asset::sha256(PRIVATE_CUDA_QUERY_SEMANTICS.as_bytes())
    }
    pub fn implementation_digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-private-cuda-warm-implementation/2");
        for source in [
            include_bytes!("cuda.rs").as_slice(),
            include_bytes!("../warm.rs").as_slice(),
            include_bytes!("../../pals_onnx.rs").as_slice(),
            include_bytes!("../../pals_private.rs").as_slice(),
        ] {
            hash.update((source.len() as u64).to_le_bytes());
            hash.update(source);
        }
        hash.finalize().into()
    }
    pub const fn schema(&self) -> &'static str {
        PRIVATE_CUDA_WARM_SCHEMA
    }
    pub const fn execution_domain(&self) -> &'static str {
        PRIVATE_CUDA_WARM_EXECUTION_DOMAIN
    }
    pub const fn query_semantics(&self) -> &'static str {
        PRIVATE_CUDA_QUERY_SEMANTICS
    }
}

/// Full immutable model input and 6144 FP32 seed bits. Mode is only bound from
/// the exact admitted bank lease. The bank lease stays with the caller's actual
/// physical owner until Ready, even though this payload owns a seed copy.
pub struct PalsCudaWarmInput {
    input: PalsModelInput,
    capability: PalsCudaWarmCapability,
    input_key: [u8; 32],
    invocation: PrivateInvocation,
    initial_bits: Vec<u32>,
    owned_host_bytes: u64,
}
impl PalsCudaWarmInput {
    /// Explicit bank-free Fresh, used for value/startup. The full finite zero
    /// seed is ignored by the graph false branch. Validator is never Warm.
    pub fn fresh(
        capability: &PalsCudaWarmCapability,
        input: PalsModelInput,
    ) -> Result<Self, BackendError> {
        validate_cuda_input(capability, &input)?;
        cuda_owned_bytes(&input, LATENT_ELEMENTS)?;
        let input_key = input
            .canonical_input_key(&capability.model_config())
            .map_err(model_input)?;
        let mut initial_bits = Vec::new();
        initial_bits
            .try_reserve_exact(LATENT_ELEMENTS)
            .map_err(|e| allocation(&e))?;
        initial_bits.resize(LATENT_ELEMENTS, 0);
        let owned_host_bytes = cuda_owned_bytes(&input, initial_bits.capacity())?;
        Ok(Self {
            input,
            capability: *capability,
            input_key,
            invocation: PrivateInvocation::Fresh { input_key },
            initial_bits,
            owned_host_bytes,
        })
    }
    pub fn from_lease(
        capability: &PalsCudaWarmCapability,
        input: PalsModelInput,
        lease: &PrivateSeedLease,
    ) -> Result<Self, BackendError> {
        let mut owned = Self::fresh(capability, input)?;
        owned.bind_lease(capability, lease)?;
        Ok(owned)
    }
    /// Allocation-free after caller preflight. Every check precedes copying and
    /// mode mutation; refusal preserves the existing full payload.
    pub fn bind_lease(
        &mut self,
        capability: &PalsCudaWarmCapability,
        lease: &PrivateSeedLease,
    ) -> Result<(), BackendError> {
        if self.capability != *capability
            || self.input.role == PalsRole::Validator
            || self.input.model_epoch != capability.model_epoch()
            || lease.model_identity() != capability.identity
            || self.initial_bits.len() != LATENT_ELEMENTS
            || self.invocation
                != (PrivateInvocation::Fresh {
                    input_key: self.input_key,
                })
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "PALS CUDA Warm prepared capability, model, role or mode differs",
            ));
        }
        let invocation = lease.invocation();
        let (input_key, approximate) = match invocation {
            PrivateInvocation::Fresh { input_key } => (input_key, false),
            PrivateInvocation::ApproxCudaWarmV2 { input_key, .. } => (input_key, true),
            PrivateInvocation::ApproxWarmV1 { .. } => {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Admission,
                    "CUDA PALS Warm v2 cannot bind a CPU Warm v1 invocation",
                ))
            }
        };
        if input_key != self.input_key || approximate != lease.seed_bits().is_some() {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "PALS CUDA Warm actual input, invocation and seed presence differ",
            ));
        }
        if approximate {
            let source = lease.seed_provenance().ok_or_else(|| {
                fail(
                    K::IdentityMismatch,
                    S::Admission,
                    "PALS CUDA Warm has no accepted seed provenance",
                )
            })?;
            if source.model != capability.identity || source.role != self.input.role {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Admission,
                    "PALS CUDA Warm accepted seed model or role differs",
                ));
            }
        }
        let bits = lease.seed_bits().unwrap_or(&self.initial_bits);
        if bits.len() != LATENT_ELEMENTS || bits.iter().any(|b| !f32::from_bits(*b).is_finite()) {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "PALS CUDA Warm requires every one of 6144 finite FP32 seed values",
            ));
        }
        if let Some(bits) = lease.seed_bits() {
            self.initial_bits.copy_from_slice(bits);
        }
        self.invocation = invocation;
        Ok(())
    }
    pub fn input(&self) -> &PalsModelInput {
        &self.input
    }
    pub fn invocation(&self) -> PrivateInvocation {
        self.invocation
    }
    pub fn owned_host_bytes(&self) -> u64 {
        self.owned_host_bytes
    }
    /// Diagnostic/reference input only. This borrows all actual seed bits and
    /// does not admit a seed, a physical fence, or a numerical success.
    pub fn initial_latent_bits(&self) -> &[u32] {
        &self.initial_bits
    }
    fn validate_for(&self, capability: &PalsCudaWarmCapability) -> Result<(), BackendError> {
        validate_cuda_input(capability, &self.input)?;
        let key = match self.invocation {
            PrivateInvocation::Fresh { input_key }
            | PrivateInvocation::ApproxCudaWarmV2 { input_key, .. } => input_key,
            PrivateInvocation::ApproxWarmV1 { .. } => {
                return Err(fail(
                    K::IdentityMismatch,
                    S::Admission,
                    "PALS CUDA Warm input is in the CPU invocation domain",
                ))
            }
        };
        if self.capability != *capability
            || key != self.input_key
            || self
                .input
                .canonical_input_key(&capability.model_config())
                .map_err(model_input)?
                != self.input_key
            || self.initial_bits.len() != LATENT_ELEMENTS
            || self
                .initial_bits
                .iter()
                .any(|b| !f32::from_bits(*b).is_finite())
            || cuda_owned_bytes(&self.input, self.initial_bits.capacity())? != self.owned_host_bytes
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "PALS CUDA Warm immutable input or seed shape/identity differs",
            ));
        }
        Ok(())
    }
    fn prepare(&self) -> Result<ActiveWarmInputs, BackendError> {
        self.validate_for(&self.capability)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(LATENT_ELEMENTS)
            .map_err(|e| allocation(&e))?;
        values.extend(self.initial_bits.iter().map(|b| f32::from_bits(*b)));
        let host_bytes = (values.capacity() as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add(std::mem::size_of::<ActiveWarmInputs>() as u64 + 1))
            .ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "CUDA Warm tensor bytes overflow",
                )
            })?;
        let error = |e| {
            native(
                CauseCode::TensorCreate,
                "cannot own full CUDA Warm seed tensors",
                e,
            )
        };
        Ok(ActiveWarmInputs {
            initial_latent: Tensor::from_array(([1, 16, 384], values)).map_err(error)?,
            warm_start: Tensor::from_array((
                Vec::<usize>::new(),
                vec![matches!(
                    self.invocation,
                    PrivateInvocation::ApproxCudaWarmV2 { .. }
                )],
            ))
            .map_err(error)?,
            host_bytes,
        })
    }
}
fn validate_cuda_input(
    capability: &PalsCudaWarmCapability,
    input: &PalsModelInput,
) -> Result<(), BackendError> {
    input
        .validate(&capability.model_config())
        .map_err(model_input)?;
    if input.role == PalsRole::Validator {
        return Err(fail(
            K::UnsupportedModel,
            S::Admission,
            "PALS CUDA Warm has no Validator graph; V must use its separate Fresh domain",
        ));
    }
    if input.model_epoch != capability.model_epoch() {
        return Err(fail(
            K::IdentityMismatch,
            S::Admission,
            "PALS CUDA Warm input differs from frozen model epoch",
        ));
    }
    Ok(())
}
fn cuda_owned_bytes(input: &PalsModelInput, seeds: usize) -> Result<u64, BackendError> {
    let mut bytes = owned_bytes(input, seeds)?;
    if let Some(full) = &input.full_line {
        let mut charge = |capacity: usize, width: usize| -> Result<(), BackendError> {
            bytes = (capacity as u64)
                .checked_mul(width as u64)
                .and_then(|n| bytes.checked_add(n))
                .ok_or_else(|| {
                    fail(
                        K::ResourceExhausted,
                        S::Admission,
                        "CUDA Warm full-line bytes overflow",
                    )
                })?;
            Ok(())
        };
        charge(
            full.records.capacity(),
            std::mem::size_of::<crate::pals_model::PalsRecordLine>(),
        )?;
        for line in &full.records {
            charge(
                line.moves.capacity(),
                std::mem::size_of::<crate::pals_model::PalsCandidateToken>(),
            )?;
        }
        for line in [
            &full.query_prefix,
            &full.query_proposal,
            &full.query_counter,
        ] {
            charge(
                line.capacity(),
                std::mem::size_of::<crate::pals_model::PalsCandidateToken>(),
            )?;
        }
    }
    bytes = bytes
        .checked_add(std::mem::size_of::<PalsCudaWarmInput>() as u64)
        .ok_or_else(|| {
            fail(
                K::ResourceExhausted,
                S::Admission,
                "CUDA Warm owner bytes overflow",
            )
        })?;
    if bytes > CUDA_PAYLOAD_LIMIT {
        return Err(fail(
            K::ResourceExhausted,
            S::Admission,
            "PALS CUDA Warm payload exceeds its 2 MiB host reservation",
        ));
    }
    Ok(bytes)
}

/// Wrap the established binding/output owner, adding the actual Warm tensor
/// inputs before its synchronous Run + synchronize_outputs fence. If any bind,
/// copy, Run or synchronization fails, the backend retains this entire owner,
/// all original inputs and its device public memory; no completion is invented.
pub(in crate::pals_onnx) struct DeviceWarmRole {
    role: device::DeviceRole,
}
impl DeviceWarmRole {
    fn new(session: &Session, input: &PalsModelInput) -> ort::Result<Self> {
        Ok(Self {
            role: device::DeviceRole::new(
                session,
                input.candidates.len().max(1),
                input.divergence_features.len().max(1),
                input.role == PalsRole::Critic,
                true,
            )?,
        })
    }
    fn run(
        &mut self,
        session: &mut Session,
        active: &ActiveInputs,
        memory: &device::DeviceMemory,
        stats: &mut PalsBackendStats,
    ) -> ort::Result<()> {
        let warm = active
            .private_warm
            .as_ref()
            .ok_or_else(|| ort::Error::new("CUDA Warm seed owner absent"))?;
        self.role
            .binding
            .bind_input("initial_latent", &warm.initial_latent)?;
        // If conditions have host input memory requirements. The bool scalar
        // stays owned on CPU and never masquerades as an uploaded condition.
        self.role
            .binding
            .bind_input("warm_start", &warm.warm_start)?;
        self.role.run(session, active, memory, stats)
    }
    fn extract(&self, input: &PalsModelInput) -> Result<PalsRawOutput, BackendError> {
        self.role.extract(input)
    }
}

fn cuda_warm_inputs(profile: PalsModelProfile) -> Vec<Interface> {
    let mut values = shared_inputs_for(profile);
    values.extend([
        spec("initial_latent", TensorElementType::Float32, &[-1, 16, 384]),
        spec("warm_start", TensorElementType::Bool, &[]),
    ]);
    values
}

impl PalsOnnxBackend {
    /// Separate CUDA manifest and invocation domain; CPU Warm v1 stays strict.
    /// No CPU model fallback or declared capability substitutes for this load.
    pub fn load_cuda_private_warm(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
    ) -> Result<Self, BackendError> {
        Self::load_cuda_warm_inner(path, expected_manifest_sha256, runtime, config, None)
    }
    /// An optional exact metadata-only CPU control policy. Device K/V remain
    /// mandatory; the existing native-loading and complete placement gates run.
    pub fn load_cuda_private_warm_with_control_policy(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        policy: PalsCudaControlPolicy,
        profile_root: &Path,
    ) -> Result<Self, BackendError> {
        let root = cuda_control::owned_profile_root(profile_root)?;
        Self::load_cuda_warm_inner(
            path,
            expected_manifest_sha256,
            runtime,
            config,
            Some((policy, root)),
        )
    }
    fn load_cuda_warm_inner(
        path: &Path,
        expected_manifest_sha256: &str,
        runtime: OrtRuntime,
        config: PalsOnnxConfig,
        audit: Option<(PalsCudaControlPolicy, std::path::PathBuf)>,
    ) -> Result<Self, BackendError> {
        let (device_id, runtime_bundle) = validate_cuda_domain(config, runtime.bundle_digest())?;
        validate_native_loading_path(runtime.native_loading_profile(), config, audit.is_some())?;
        if !path.is_absolute() {
            return Err(fail(
                K::InvalidInput,
                S::Asset,
                "PALS CUDA Warm manifest requires an absolute path",
            ));
        }
        let bytes = read_owned(
            path,
            MANIFEST_LIMIT,
            strict_digest(expected_manifest_sha256)?,
        )?;
        let manifest_digest = asset::sha256(&bytes);
        let manifest: WarmManifest = serde_json::from_slice(&bytes).map_err(|e| {
            fail(
                K::UnsupportedModel,
                S::Asset,
                "cannot decode strict CUDA Warm manifest",
            )
            .with_external_cause(CauseCode::ModelLoad, &e)
        })?;
        validate_cuda_manifest(&manifest)?;
        runtime.verify_cuda_dependencies()?;
        let parent = path.parent().ok_or_else(|| {
            fail(
                K::InvalidInput,
                S::Asset,
                "PALS CUDA Warm manifest has no parent",
            )
        })?;
        // Both graph snapshots are bounded/hash-verified before native loading.
        let mut graph_bytes = Vec::new();
        let mut identities = Vec::new();
        let mut total = 0usize;
        for graph in &manifest.graphs {
            if let Some((policy, _)) = &audit {
                policy.validate_pins(manifest_digest, graph)?;
            }
            let remaining = GRAPH_LIMIT.checked_sub(total).ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "CUDA Warm serialized graph bytes overflow",
                )
            })?;
            let owned = read_owned(
                &parent.join(&graph.file),
                remaining,
                strict_digest(&graph.sha256)?,
            )?;
            total = total.checked_add(owned.len()).ok_or_else(|| {
                fail(
                    K::ResourceExhausted,
                    S::Asset,
                    "CUDA Warm serialized graph bytes overflow",
                )
            })?;
            identities.push(PalsGraphIdentity {
                role: graph.role.clone(),
                sha256: asset::sha256(&owned),
                serialized_bytes: owned.len() as u64,
            });
            graph_bytes.push(owned);
        }
        let mut sessions = BTreeMap::new();
        let mut placements = Vec::new();
        for (graph, owned) in manifest.graphs.iter().zip(graph_bytes) {
            let (session, placement) = load_session(
                owned,
                config,
                &graph.role,
                audit
                    .as_ref()
                    .map(|(policy, root)| (policy, root.as_path())),
            )?;
            validate_cuda_session(&session, graph, &manifest)?;
            if let Some(placement) = placement {
                placements.push(placement);
            }
            sessions.insert(graph.role.clone(), session);
        }
        let rules = strict_digest(&manifest.rules_input_semantic_sha256)?;
        let rules_source = strict_digest(&manifest.rules_encoder_source_sha256)?;
        let epoch = strict_digest(&manifest.checkpoint_sha256)?;
        let runtime_binary = runtime.binary_digest();
        let mut hash = Sha256::new();
        for field in [
            b"rz-pals-private-cuda-warm-encoding/2".as_slice(),
            manifest.config.profile.encoding_schema().as_bytes(),
            manifest.config.profile.as_str().as_bytes(),
            &rules,
            PRIVATE_CUDA_WARM_GRAPH_SEMANTICS.as_bytes(),
            PRIVATE_CUDA_WARM_SEMANTICS.as_bytes(),
            PRIVATE_CUDA_QUERY_SEMANTICS.as_bytes(),
            PRIVATE_CUDA_WARM_EXECUTION_DOMAIN.as_bytes(),
            &runtime_binary,
            &runtime_bundle,
            &device_id.to_le_bytes(),
        ] {
            hash.update((field.len() as u64).to_le_bytes());
            hash.update(field);
        }
        // A loaded numerical domain also fixes native optimization, reviewed
        // placement controls and resource options. A different loaded owner
        // cannot inherit a seed merely because its checkpoint digest matches.
        hash.update(b"rz-pals-cuda-warm-native-options/2");
        hash.update((config.intra_threads as u64).to_le_bytes());
        if let Provider::Cuda { arena_bytes, .. } = config.provider {
            hash.update(arena_bytes.to_le_bytes());
        }
        hash.update([u8::from(audit.is_some())]);
        if let Some((policy, _)) = &audit {
            hash.update(policy.inventory);
        }
        let identity = PrivateModelIdentity {
            model: manifest_digest,
            encoding: hash.finalize().into(),
            precision: PrivatePrecision::Fp32,
            model_epoch: epoch,
            frozen_epoch: 1,
        };
        let token = cuda_capability_token(
            identity,
            rules,
            rules_source,
            manifest.config.profile,
            device_id,
            runtime_binary,
            runtime_bundle,
        );
        let capability = PalsCudaWarmCapability {
            identity,
            rules,
            rules_source,
            profile: manifest.config.profile,
            device_id,
            runtime_binary,
            runtime_bundle,
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
            shared_pc: sessions.remove("shared_pc"),
            layout: Layout::SharedPcIf,
            private_warm: None,
            private_cuda_warm: Some(capability),
            runtime,
            config,
            model_config: manifest.config,
            epoch,
            manifest_digest,
            trained: manifest.trained,
            rules_input_profile: Some(manifest.rules_input_profile),
            rules_input_semantic_sha256: Some(rules),
            residency: PalsSessionResidency {
                graphs: identities,
                native_sessions: 2,
                layout: CUDA_LAYOUT.into(),
                reader_initializer_bank: Some(manifest.ownership.legacy_ownership),
                role_reader_weights_shared: None,
                native_resident_parameter_bytes: None,
                vram_peak_bytes: None,
            },
            stats: PalsBackendStats::default(),
            cuda_mapping_audit: RefCell::default(),
            cuda_control_audit: audit.map(|(policy, profile_root)| CudaControlAudit {
                policy,
                profile_root,
                initialization: placements,
                proposer_seen: false,
                critic_seen: false,
                witness: None,
                rejected: None,
            }),
            memory: Some(memory),
            cached_memory_key: None,
            active_memory: None,
            record_pages: None,
            game_generation: 0,
            active: None,
            device_memory: None,
            device_role: None,
            device_warm_role: None,
            cuda_record_pages: None,
            quarantine: None,
            startup_stage_probe: None,
            public_encodes: 0,
            public_cache_hits: 0,
        })
    }
    pub fn private_cuda_warm_capability(&self) -> Option<&PalsCudaWarmCapability> {
        if self.quarantine.is_some()
            || self.has_active_physical_invocation()
            || self.cuda_mapping_audit.borrow().allow_run().is_err()
            || self
                .cuda_control_audit
                .as_ref()
                .is_some_and(|a| a.rejected.is_some())
        {
            None
        } else {
            self.private_cuda_warm.as_ref()
        }
    }
    /// Explicit Fresh or ApproxCudaWarmV2. Original payload, bound seed tensors,
    /// device public K/V and final output binding remain installed on unknown.
    pub fn run_cuda_private_warm(
        &mut self,
        input: &PalsCudaWarmInput,
    ) -> Result<PalsRawOutput, BackendError> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        let capability = self.private_cuda_warm.as_ref().ok_or_else(|| {
            fail(
                K::UnsupportedModel,
                S::Admission,
                "PALS backend is not the loaded CUDA Warm domain",
            )
        })?;
        input.validate_for(capability)?;
        validate_cuda_domain(self.config, self.runtime.bundle_digest())?;
        if input.capability.model_config() != self.model_config
            || self.private_warm.is_some()
            || self.cuda_record_pages.is_some()
            || self.record_pages.is_some()
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "CUDA Warm model or public-memory domain differs",
            ));
        }
        self.cuda_mapping_audit.borrow().allow_run()?;
        if let Some(cause) = self
            .cuda_control_audit
            .as_ref()
            .and_then(|a| a.rejected.as_ref())
        {
            return Err(cause.clone());
        }
        if self.has_active_physical_invocation() {
            return Err(fail(
                K::BackendFailure,
                S::Admission,
                "CUDA Warm physical owner is still active",
            ));
        }
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
        let result = self.run_cuda_warm_owned(input, trace.as_ref());
        if let Some(trace) = trace {
            trace.finish(result.is_ok());
            if trace.probe.closed() {
                self.startup_stage_probe = None;
            }
        }
        result
    }
    fn run_cuda_warm_owned(
        &mut self,
        input: &PalsCudaWarmInput,
        trace: Option<&StartupRoleTrace>,
    ) -> Result<PalsRawOutput, BackendError> {
        startup_enter(trace, PalsStartupBackendStage::InputPreparation);
        let tensors = input
            .input
            .prepare_tensors(&self.model_config)
            .map_err(model_input)?;
        let key = tensors.public_memory_key;
        // No physical transfer is possible until the entire active aggregate
        // has been installed. Partial input allocations are prelaunch failures.
        let mut active = ActiveInputs::new(tensors, input.input.role == PalsRole::Critic, None)?;
        active.private_warm = Some(input.prepare()?);
        self.active = Some(active);
        startup_return(trace, PalsStartupBackendStage::InputPreparation, true);
        self.stats.admitted_role_requests = self.stats.admitted_role_requests.saturating_add(1);
        let result = self.run_cuda_warm_device(&input.input, key, trace);
        if self.quarantine.is_none() {
            self.device_warm_role = None;
            self.active = None;
            if !self.config.cache_public_memory {
                self.device_memory = None;
            }
        }
        self.finish_role_run(&input.input, trace, result)
    }
    fn run_cuda_warm_device(
        &mut self,
        input: &PalsModelInput,
        key: [u8; 32],
        trace: Option<&StartupRoleTrace>,
    ) -> Result<PalsRawOutput, BackendError> {
        let device_id = self
            .private_cuda_warm
            .as_ref()
            .expect("CUDA domain preflight")
            .device_id;
        let cached = self.config.cache_public_memory
            && self.device_memory.as_ref().is_some_and(|m| m.key == key);
        if cached {
            self.stats.public_cache_hits = self.stats.public_cache_hits.saturating_add(1);
            self.public_cache_hits = self.public_cache_hits.saturating_add(1);
            if let Some(trace) = trace {
                trace.record(
                    PalsStartupBackendStage::PublicCacheHit,
                    PalsStartupStageBoundary::Observed,
                );
            }
        } else {
            self.stats.public_cache_misses = self.stats.public_cache_misses.saturating_add(1);
            let tokens = self
                .model_config
                .public_memory_tokens(input.records.len())
                .map_err(model_input)?;
            self.device_memory = Some(
                device::DeviceMemory::new(
                    self.public.as_ref().expect("public session"),
                    key,
                    tokens,
                    device_id,
                )
                .map_err(|e| {
                    native(
                        CauseCode::TensorCreate,
                        "cannot allocate CUDA Warm public K/V owner",
                        e,
                    )
                })?,
            );
            startup_enter(trace, PalsStartupBackendStage::PublicRun);
            let result = self
                .device_memory
                .as_mut()
                .expect("installed device public owner")
                .run(
                    self.public.as_mut().expect("public session"),
                    self.active.as_ref().expect("active inputs"),
                    &mut self.stats,
                );
            startup_return(trace, PalsStartupBackendStage::PublicRun, result.is_ok());
            if let Err(error) = result {
                let failure = native(
                    CauseCode::OrtRun,
                    "CUDA Warm public binding/copy/fence failed",
                    error,
                );
                self.quarantine = Some(failure.clone());
                return Err(failure);
            }
            if let Err(error) = self
                .device_memory
                .as_ref()
                .expect("completed public owner")
                .validate_mask(input.records.len())
            {
                self.device_memory = None; // Public fence succeeded; invalid cache cannot survive.
                return Err(error);
            }
            self.public_encodes = self.public_encodes.saturating_add(1);
        }
        self.device_warm_role = Some(
            DeviceWarmRole::new(self.shared_pc.as_ref().expect("CUDA Warm session"), input)
                .map_err(|e| {
                    native(
                        CauseCode::TensorCreate,
                        "cannot own bounded CUDA Warm private outputs",
                        e,
                    )
                })?,
        );
        startup_enter(trace, PalsStartupBackendStage::PrivateRun);
        let result = self
            .device_warm_role
            .as_mut()
            .expect("private output owner installed")
            .run(
                self.shared_pc.as_mut().expect("CUDA Warm session"),
                self.active.as_ref().expect("active seed inputs"),
                self.device_memory
                    .as_ref()
                    .expect("completed device public owner"),
                &mut self.stats,
            );
        startup_return(trace, PalsStartupBackendStage::PrivateRun, result.is_ok());
        if let Err(error) = result {
            let failure = native(
                CauseCode::OrtRun,
                "CUDA Warm private seed/binding/copy/fence failed",
                error,
            );
            self.quarantine = Some(failure.clone());
            return Err(failure);
        }
        startup_enter(trace, PalsStartupBackendStage::PrivateOutputValidation);
        let output = self
            .device_warm_role
            .as_ref()
            .expect("completed final output owner")
            .extract(input)?;
        output
            .decode(input, &self.model_config)
            .map_err(model_input)?;
        self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
        startup_return(
            trace,
            PalsStartupBackendStage::PrivateOutputValidation,
            true,
        );
        Ok(output)
    }
}

fn validate_cuda_domain(
    config: PalsOnnxConfig,
    bundle: Option<[u8; 32]>,
) -> Result<(i32, [u8; 32]), BackendError> {
    config.validate()?;
    match (config.provider, config.device_public_memory, bundle) {
        (Provider::Cuda { device_id, .. }, true, Some(bundle)) => Ok((device_id, bundle)),
        _ => Err(fail(K::BackendUnavailable, S::Admission,
            "CUDA Warm requires explicit CUDA, device public K/V, I/O binding and verified runtime bundle")),
    }
}
fn cuda_capability_token(
    identity: PrivateModelIdentity,
    rules: [u8; 32],
    source: [u8; 32],
    profile: PalsModelProfile,
    device_id: i32,
    runtime_binary: [u8; 32],
    runtime_bundle: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-private-cuda-warm-loaded-capability/2");
    for field in [
        identity.model.as_slice(),
        &identity.encoding,
        &identity.model_epoch,
        &rules,
        &source,
        profile.as_str().as_bytes(),
        &runtime_binary,
        &runtime_bundle,
    ] {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field);
    }
    hash.update(identity.frozen_epoch.to_le_bytes());
    hash.update(device_id.to_le_bytes());
    hash.finalize().into()
}

fn validate_cuda_manifest(m: &WarmManifest) -> Result<(), BackendError> {
    m.config.validate().map_err(model_input)?;
    if m.schema != PRIVATE_CUDA_WARM_SCHEMA
        || !m.config.profile.uses_full_line()
        || m.execution_domain.as_deref() != Some(PRIVATE_CUDA_WARM_EXECUTION_DOMAIN)
        || m.query_semantics.as_deref() != Some(PRIVATE_CUDA_QUERY_SEMANTICS)
        || m.encoding_schema.as_deref() != Some(m.config.profile.encoding_schema())
        || m.model_profile.as_deref() != Some(m.config.profile.as_str())
        || m.model_semantics != m.config.profile.model_semantics()
        || m.layout != CUDA_LAYOUT
        || m.layout_revision != 2
        || m.precision != "fp32"
        || m.tf32
        || !m.approximate
        || m.warm_graph_semantics != PRIVATE_CUDA_WARM_GRAPH_SEMANTICS
        || m.private_seed_policy != PRIVATE_CUDA_WARM_SEMANTICS
        || m.native_seed_owner_support != "not_registered_by_python_export"
        || m.batch_mode != "one_scalar_role_and_one_scalar_mode_per_physical_batch"
        || m.roles != ["proposer", "critic"]
        || m.validator_present
        || !m.task_names.iter().map(String::as_str).eq(V_TASK_NAMES)
        || m.trained != (m.training_steps > 0)
        || m.graphs.len() != 2
        || m.numeric_status != "not_run"
        || m.cuda != "not_run"
        || m.expected_ort != "1.22.0"
        || m.rust_ort_crate != "2.0.0-rc.10"
        || m.torch_version.is_empty()
        || m.torch_version.len() > 128
        || m.torch_version.chars().any(char::is_control)
    {
        return Err(fail(
            K::UnsupportedModel,
            S::Asset,
            "expected separately registered FP32 P/C CUDA/device Warm v2 export",
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
    validate_cuda_rules(m)?;
    let o = &m.ownership;
    strict_digest(&o.legacy_graph_sha256)?;
    // This audit remains serialized initializers only, never native or VRAM
    // sharing. The reader's six role routes and its old audit format remain.
    validate_reader_bank(&o.legacy_ownership)?;
    if o.scope != "onnx_serialized_initializers_only"
        || o.layout_revision != 2
        || o.recursive_role_if_routes != 6
        || o.warm_mode_if_routes != 1
        || o.branch_local_initializers != 0
        || o.seed_values_per_batch_row != LATENT_ELEMENTS
        || o.fresh_initialization != "registered_profile_role_if"
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
            "PALS CUDA Warm serialized ownership declaration differs",
        ));
    }
    for (index, (role, filename)) in [
        ("public", "public_memory.onnx"),
        ("shared_pc", CUDA_GRAPH_FILE),
    ]
    .into_iter()
    .enumerate()
    {
        let g = &m.graphs[index];
        if g.role != role || g.file != filename || g.opset != 17 {
            return Err(fail(
                K::UnsupportedModel,
                S::Asset,
                "PALS CUDA Warm registered graph names or opset differ",
            ));
        }
        strict_digest(&g.sha256)?;
        validate_declared_interface(
            &g.inputs,
            &if index == 0 {
                public_inputs_for(m.config.profile)
            } else {
                cuda_warm_inputs(m.config.profile)
            },
        )?;
        validate_declared_interface(
            &g.outputs,
            &if index == 0 {
                memory_outputs()
            } else {
                shared_outputs()
            },
        )?;
    }
    Ok(())
}
const FULL_LINE_RULE_FEATURES: [&str; 9] = [
    "full_line.record_tokens",
    "full_line.record_lengths",
    "full_line.record_mask",
    "full_line.query_tokens",
    "full_line.query_lengths",
    "full_line.query_mask",
    "full_line.record_kind",
    "full_line.parent_local_index",
    "full_line.supersedes_local_index",
];
fn validate_cuda_rules(m: &WarmManifest) -> Result<(), BackendError> {
    if !m.config.profile.uses_full_line() {
        return validate_rules(m);
    }
    let d = &m.rules_input_declaration;
    let bounds = d.bounds.as_ref();
    if m.rules_input_profile != "rz-pals-rules-full-line-v2"
        || m.learned_input_compatibility != "unverified_declaration_only"
        || d.schema != "rovezero.pals-rules-descriptor.v2"
        || d.rules_input_profile != m.rules_input_profile
        || d.rules_input_semantic_sha256 != m.rules_input_semantic_sha256
        || d.rules_encoder_source_sha256 != m.rules_encoder_source_sha256
        || d.encoder_source != "crates/rz-uci/src/pals_native.rs"
        || d.semantic_digest_algorithm != "sha256_u64le_length_prefixed_utf8_fields"
        || d.learned_input_compatibility != m.learned_input_compatibility
        || d.config != m.config
        || d.rules_base_semantics.is_empty()
        || d.metadata_features.len() != 16
        || d.record_features.len() != 16
        || d.query_features.len() != 16
        || d.divergence_features.len() != 8
        || d.semantic_fields.len() != 67
        || !d
            .full_line_features
            .as_ref()
            .is_some_and(|f| f.iter().map(String::as_str).eq(FULL_LINE_RULE_FEATURES))
        || !bounds.is_some_and(|b| {
            b.max_records == 128 && b.max_line_plies == 256 && b.max_candidates == 256
        })
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "CUDA Warm exact full-line Rules descriptor or bounds differ",
        ));
    }
    let fields = std::iter::once(d.rules_input_profile.as_str())
        .chain(std::iter::once(d.rules_base_semantics.as_str()))
        .chain(
            d.metadata_features
                .iter()
                .chain(&d.record_features)
                .chain(&d.query_features)
                .chain(&d.divergence_features)
                .chain(d.full_line_features.as_ref().expect("checked"))
                .map(String::as_str),
        );
    let mut hash = Sha256::new();
    for (expected, actual) in fields.zip(&d.semantic_fields) {
        if expected != actual.as_str()
            || actual.is_empty()
            || actual.len() > 4096
            || actual.chars().any(char::is_control)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "CUDA Warm Rules semantic vocabulary differs",
            ));
        }
        hash.update((actual.len() as u64).to_le_bytes());
        hash.update(actual.as_bytes());
    }
    if <[u8; 32]>::from(hash.finalize()) != strict_digest(&m.rules_input_semantic_sha256)? {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "CUDA Warm Rules semantic digest differs",
        ));
    }
    let canonical = serde_json::to_vec(&canonical_value(
        serde_json::to_value(d).map_err(model_input)?,
    ))
    .map_err(model_input)?;
    if canonical.len() > MANIFEST_LIMIT
        || asset::sha256(&canonical) != strict_digest(&m.rules_profile_canonical_sha256)?
    {
        return Err(fail(
            K::IdentityMismatch,
            S::Asset,
            "CUDA Warm Rules canonical descriptor differs",
        ));
    }
    Ok(())
}
fn validate_cuda_session(
    session: &Session,
    graph: &GraphManifest,
    m: &WarmManifest,
) -> Result<(), BackendError> {
    let inputs = if graph.role == "public" {
        public_inputs_for(m.config.profile)
    } else {
        cuda_warm_inputs(m.config.profile)
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
            "loaded CUDA Warm V2 tensor interface differs",
        ));
    }
    let metadata = session
        .metadata()
        .map_err(|e| native(CauseCode::ModelLoad, "cannot inspect CUDA Warm metadata", e))?;
    let steps = m.training_steps.to_string();
    let mut fields = vec![
        ("schema", PRIVATE_CUDA_WARM_SCHEMA),
        ("model_semantics", m.config.profile.model_semantics()),
        ("layout", CUDA_LAYOUT),
        ("layout_revision", "2"),
        ("role", graph.role.as_str()),
        ("checkpoint_sha256", m.checkpoint_sha256.as_str()),
        ("precision", "fp32"),
        ("trained", if m.trained { "true" } else { "false" }),
        ("training_steps", steps.as_str()),
        ("expected_ort", "1.22.0"),
        ("approximate", "true"),
        ("warm_graph_semantics", PRIVATE_CUDA_WARM_GRAPH_SEMANTICS),
        ("private_seed_policy", PRIVATE_CUDA_WARM_SEMANTICS),
        ("execution_domain", PRIVATE_CUDA_WARM_EXECUTION_DOMAIN),
        ("query_semantics", PRIVATE_CUDA_QUERY_SEMANTICS),
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
    ];
    if !m.config.profile.is_legacy() {
        fields.extend([
            ("encoding_schema", m.config.profile.encoding_schema()),
            ("model_profile", m.config.profile.as_str()),
        ]);
    }
    for (key, expected) in fields {
        if metadata
            .custom(key)
            .map_err(|e| {
                native(
                    CauseCode::ModelLoad,
                    "cannot read CUDA Warm semantic metadata",
                    e,
                )
            })?
            .as_deref()
            != Some(expected)
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Asset,
                "loaded CUDA Warm metadata differs from registered manifest",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::{PalsCandidateToken, PalsFullLineInput};
    use crate::pals_private::{
        PhysicalSeedCompletion, PrivateInvocationMode, PrivateRulesContext, PrivateSeedBank,
        PrivateSeedLimits, PrivateSeedRequest, PrivateSituationContext,
    };
    use rz_contracts::{pals::SituationHandle, CancelToken};
    use rz_position::{Position, PositionSnapshot};
    use std::time::Duration;

    // Private test-only authored witness. No ORT load, NN run, GPU completion,
    // caller seed adoption, placement or device-sharing evidence is asserted.
    fn capability() -> PalsCudaWarmCapability {
        PalsCudaWarmCapability {
            identity: PrivateModelIdentity {
                model: [1; 32],
                encoding: [2; 32],
                precision: PrivatePrecision::Fp32,
                model_epoch: [3; 32],
                frozen_epoch: 1,
            },
            rules: [4; 32],
            rules_source: [5; 32],
            profile: PalsModelProfile::FullLineInteractionV2,
            device_id: 0,
            runtime_binary: [6; 32],
            runtime_bundle: [7; 32],
            token: [8; 32],
        }
    }
    fn input() -> PalsModelInput {
        let movement = PalsCandidateToken {
            from: 12,
            to: 28,
            promotion: 0,
        };
        PalsModelInput {
            role: PalsRole::Proposer,
            board: vec![0; 64],
            metadata: [0.; 16],
            records: vec![],
            required_critical_records: vec![],
            candidates: vec![movement],
            divergence_features: vec![],
            query: [0.; 16],
            situation_revision: 0,
            history_digest: [9; 32],
            model_epoch: [3; 32],
            full_line: Some(PalsFullLineInput {
                records: vec![],
                query_prefix: vec![movement],
                query_proposal: vec![],
                query_counter: vec![],
            }),
        }
    }
    fn request(
        capability: &PalsCudaWarmCapability,
        snapshot: &PositionSnapshot,
        input: &PalsModelInput,
    ) -> PrivateSeedRequest {
        PrivateSeedRequest {
            role: input.role,
            model: capability.identity,
            game_generation: 1,
            search_generation: 2,
            context: PrivateRulesContext::seal(
                snapshot,
                input,
                &capability.model_config(),
                PrivateSituationContext {
                    situation: SituationHandle {
                        slot: 1,
                        generation: 1,
                    },
                    prefix: [10; 32],
                    focus: [11; 32],
                },
            )
            .unwrap(),
        }
    }
    fn bank() -> PrivateSeedBank {
        PrivateSeedBank::new(
            PrivateSeedLimits {
                slots_per_role: 1,
                max_bank_bytes: 256 * 1024,
                max_transient_bytes: 256 * 1024,
                latent_elements: LATENT_ELEMENTS,
            },
            1,
        )
        .unwrap()
    }
    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }
    fn admit_fixture_seed(
        bank: &PrivateSeedBank,
        req: &PrivateSeedRequest,
        snapshot: &PositionSnapshot,
    ) -> Vec<u32> {
        let cancel = CancelToken::new();
        let lease = bank
            .begin(
                req.clone(),
                snapshot,
                PrivateInvocationMode::Fresh,
                &cancel,
                deadline(),
            )
            .unwrap();
        let values: Vec<f32> = (0..LATENT_ELEMENTS)
            .map(|i| i as f32 * 0.000125 + 0.25)
            .collect();
        lease
            .complete_known(PhysicalSeedCompletion::Succeeded)
            .unwrap()
            .stage_output(&values)
            .unwrap()
            .commit(req, snapshot, &cancel, deadline())
            .unwrap();
        values.into_iter().map(f32::to_bits).collect()
    }
    fn manifest() -> WarmManifest {
        let mut m = super::super::tests::manifest();
        m.schema = PRIVATE_CUDA_WARM_SCHEMA.into();
        m.execution_domain = Some(PRIVATE_CUDA_WARM_EXECUTION_DOMAIN.into());
        m.query_semantics = Some(PRIVATE_CUDA_QUERY_SEMANTICS.into());
        m.config = PalsModelConfig::full_line_interaction_v2();
        m.encoding_schema = Some(m.config.profile.encoding_schema().into());
        m.model_profile = Some(m.config.profile.as_str().into());
        m.model_semantics = m.config.profile.model_semantics().into();
        m.layout = CUDA_LAYOUT.into();
        m.layout_revision = 2;
        m.warm_graph_semantics = PRIVATE_CUDA_WARM_GRAPH_SEMANTICS.into();
        m.private_seed_policy = PRIVATE_CUDA_WARM_SEMANTICS.into();
        m.ownership.layout_revision = 2;
        m.ownership.fresh_initialization = "registered_profile_role_if".into();
        m.graphs[0].inputs = super::super::tests::descriptors(public_inputs_for(m.config.profile));
        m.graphs[1].role = "shared_pc".into();
        m.graphs[1].file = CUDA_GRAPH_FILE.into();
        m.graphs[1].inputs = super::super::tests::descriptors(cuda_warm_inputs(m.config.profile));
        let d = &mut m.rules_input_declaration;
        d.schema = "rovezero.pals-rules-descriptor.v2".into();
        d.rules_input_profile = "rz-pals-rules-full-line-v2".into();
        d.config = m.config.clone();
        d.full_line_features = Some(
            FULL_LINE_RULE_FEATURES
                .iter()
                .map(|s| (*s).into())
                .collect(),
        );
        d.bounds = Some(RulesBounds {
            max_records: 128,
            max_line_plies: 256,
            max_candidates: 256,
        });
        d.semantic_fields[0] = d.rules_input_profile.clone();
        d.semantic_fields
            .extend(d.full_line_features.as_ref().unwrap().iter().cloned());
        let mut hash = Sha256::new();
        for field in &d.semantic_fields {
            hash.update((field.len() as u64).to_le_bytes());
            hash.update(field);
        }
        d.rules_input_semantic_sha256 =
            hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        m.rules_input_profile = d.rules_input_profile.clone();
        m.rules_input_semantic_sha256 = d.rules_input_semantic_sha256.clone();
        m.rules_profile_canonical_sha256 = asset::sha256(
            &serde_json::to_vec(&canonical_value(serde_json::to_value(d).unwrap())).unwrap(),
        )
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
        m
    }
    #[test]
    fn cuda_registration_rejects_host_legacy_partial_shape_and_unattested_sharing() {
        validate_cuda_manifest(&manifest()).unwrap();
        let mutations: [fn(&mut WarmManifest); 8] = [
            |m| m.schema = PRIVATE_WARM_SCHEMA.into(),
            |m| m.execution_domain = None,
            |m| m.query_semantics = Some(FROZEN_QUERY_SEMANTICS_V1.into()),
            |m| m.config.profile = PalsModelProfile::LegacySummaryV1,
            |m| m.graphs[1].inputs.retain(|i| i.name != "query_line_mask"),
            |m| m.ownership.seed_values_per_batch_row = 6143,
            |m| m.ownership.device_residency_sharing = "syntax_proves_sharing".into(),
            |m| {
                m.rules_input_declaration
                    .bounds
                    .as_mut()
                    .unwrap()
                    .max_line_plies = 128
            },
        ];
        for change in mutations {
            let mut m = manifest();
            change(&mut m);
            assert!(validate_cuda_manifest(&m).is_err());
        }
        assert!(validate_manifest(&manifest()).is_err()); // CPU v1 is not widened.
        assert!(validate_cuda_manifest(&super::super::tests::manifest()).is_err());
        assert!(validate_cuda_domain(PalsOnnxConfig::cpu(), None).is_err());
        let mut cfg = PalsOnnxConfig::cpu();
        cfg.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 64 * 1024 * 1024,
        };
        assert!(validate_cuda_domain(cfg, Some([1; 32])).is_err());
        cfg.device_public_memory = true;
        assert!(validate_cuda_domain(cfg, None).is_err());
        assert!(validate_cuda_domain(cfg, Some([1; 32])).is_ok());
    }
    #[test]
    fn actual_cuda_seed_is_complete_and_cpu_mode_refusal_is_atomic() {
        let cap = capability();
        let snapshot = Position::startpos().snapshot();
        let i = input();
        let request = request(&cap, &snapshot, &i);
        let bank = bank();
        let seed_bits = admit_fixture_seed(&bank, &request, &snapshot);
        let cpu = bank
            .begin(
                request.clone(),
                &snapshot,
                PrivateInvocationMode::ApproxWarmV1,
                &CancelToken::new(),
                deadline(),
            )
            .unwrap();
        let mut owned = PalsCudaWarmInput::fresh(&cap, i.clone()).unwrap();
        let backing = owned.initial_bits.as_ptr();
        let original = owned.initial_bits.clone();
        let mode = owned.invocation;
        let error = owned.bind_lease(&cap, &cpu).unwrap_err();
        assert_eq!(error.kind, K::IdentityMismatch);
        assert!(error.cause.is_none() && error.native.is_none());
        assert_eq!(owned.initial_bits, original);
        assert_eq!(owned.invocation, mode);
        assert_eq!(owned.initial_bits.as_ptr(), backing);
        drop(cpu.complete_known(PhysicalSeedCompletion::Failed).unwrap());
        let cuda = bank
            .begin(
                request,
                &snapshot,
                PrivateInvocationMode::ApproxCudaWarmV2,
                &CancelToken::new(),
                deadline(),
            )
            .unwrap();
        owned.bind_lease(&cap, &cuda).unwrap();
        assert_eq!(owned.initial_bits.as_ptr(), backing);
        assert_eq!(owned.initial_bits, seed_bits);
        assert!(matches!(
            owned.invocation,
            PrivateInvocation::ApproxCudaWarmV2 { .. }
        ));
        owned.validate_for(&cap).unwrap();
        drop(cuda.complete_known(PhysicalSeedCompletion::Failed).unwrap());
        let mut changed = cap;
        changed.runtime_bundle[0] ^= 1;
        assert!(owned.validate_for(&changed).is_err());
        let mut v = input();
        v.role = PalsRole::Validator;
        assert!(PalsCudaWarmInput::fresh(&cap, v).is_err());
    }
    #[test]
    fn payload_charges_full_line_capacity_and_requires_all_finite_seed_elements() {
        let cap = capability();
        let mut owned = PalsCudaWarmInput::fresh(&cap, input()).unwrap();
        assert_eq!(owned.initial_bits.len(), 6144);
        assert!(matches!(owned.invocation, PrivateInvocation::Fresh { .. }));
        owned.initial_bits[6143] = f32::INFINITY.to_bits();
        assert!(owned.validate_for(&cap).is_err());
        owned.initial_bits[6143] = 0;
        owned.initial_bits.pop();
        assert!(owned.validate_for(&cap).is_err());
        let mut i = input();
        i.full_line
            .as_mut()
            .unwrap()
            .query_counter
            .reserve(3 * 1024 * 1024);
        assert!(matches!(
            PalsCudaWarmInput::fresh(&cap, i),
            Err(BackendError {
                kind: K::ResourceExhausted,
                ..
            })
        ));
    }
}
