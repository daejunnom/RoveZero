//! Explicit FP32 ONNX Runtime backend. No implicit runtime download or CPU fallback.
//!
//! `run` is synchronous and requires exclusive access. It returns only after ORT
//! has synchronized and copied all outputs to owned host vectors. Runtime must
//! call it on its physical worker, retaining request leases until it returns;
//! logical cancellation cannot release those leases or terminate native work.

use crate::asset::{self, MaiaAsset, INPUT_NAME, POLICY_NAME, WDL_NAME};
use crate::error::{BackendError, FailureKind as K, FailureStage as S};
use crate::{output, RawOutput};
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use rz_encoding::classical::{EncodedInput, INPUT_VALUES};
use rz_encoding::POLICY_SIZE;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX_BATCH: usize = 16;
/// Input staging plus native and owned output copies. Excludes caller features,
/// model/session/ORT activation workspace: bootstrap must budget those too.
pub const IO_BYTES_PER_ITEM: usize = (INPUT_VALUES + 2 * (POLICY_SIZE + 3)) * 4;

#[derive(Clone, Debug)]
pub struct OrtRuntime {
    path: PathBuf,
    binary_digest: [u8; 32],
    build_info: String,
}

static RUNTIME: Mutex<Option<Result<OrtRuntime, BackendError>>> = Mutex::new(None);

impl OrtRuntime {
    /// Bootstrap must be the sole initializer of the process-global `ort`
    /// library. Refuse prior initialization, path changes and failed retries.
    /// Only load trusted native libraries; a hash is identity, not a sandbox.
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self, BackendError> {
        let unavailable = || {
            BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "ORT initialization failed",
            )
        };
        let path = path.canonicalize().map_err(|_| unavailable())?;
        let expected = asset::parse_sha256(expected_sha256)?;
        let mut guard = RUNTIME.lock().map_err(|_| unavailable())?;
        if let Some(result) = guard.as_ref() {
            let runtime = result.as_ref().map_err(|error| *error)?;
            return if runtime.path == path && runtime.binary_digest == expected {
                Ok(runtime.clone())
            } else {
                Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "ORT library is process-global and already pinned",
                ))
            };
        }
        let bytes = asset::read_bounded(&path, 512 * 1024 * 1024)?;
        if asset::sha256(&bytes) != expected {
            return Err(BackendError::new(
                K::IdentityMismatch,
                S::Backend,
                "ORT library digest differs",
            ));
        }
        drop(bytes);
        let path_text = path.to_str().ok_or_else(unavailable)?;
        // ort rc.10 panics on dlopen / API-version failure; translate that narrow
        // bootstrap boundary and latch failure. Native crashes are not recoverable.
        let result = std::panic::catch_unwind(|| {
            if !ort::init_from(path_text)
                .with_name("RoveZero-C")
                .commit()
                .map_err(|_| unavailable())?
            {
                return Err(BackendError::new(
                    K::IdentityMismatch,
                    S::Backend,
                    "ORT was initialized outside bootstrap",
                ));
            }
            let info = ort::info();
            if !info.contains("git-branch=rel-1.22.0,") {
                return Err(BackendError::new(
                    K::BackendUnavailable,
                    S::Backend,
                    "this profile requires ORT release 1.22.0",
                ));
            }
            Ok(Self {
                path: path.clone(),
                binary_digest: expected,
                build_info: info.to_owned(),
            })
        })
        .unwrap_or_else(|_| Err(unavailable()));
        *guard = Some(result.clone());
        result
    }

    pub fn binary_digest(&self) -> [u8; 32] {
        self.binary_digest
    }
    pub fn build_info(&self) -> &str {
        &self.build_info
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    Cpu,
    /// The arena cap is NOT a total VRAM cap. External process/device monitoring
    /// must account for cuDNN workspace, drivers and other resident sessions.
    Cuda {
        device_id: i32,
        arena_bytes: usize,
    },
}

#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub provider: Provider,
    pub max_batch: usize,
    pub host_io_bytes: usize,
    pub intra_threads: usize,
    /// Required for CUDA's one-shot actual-node-placement probe. Kept on disk.
    pub profiling_prefix: Option<PathBuf>,
}

impl BackendConfig {
    pub fn cpu() -> Self {
        Self {
            provider: Provider::Cpu,
            max_batch: 16,
            host_io_bytes: 16 * IO_BYTES_PER_ITEM,
            intra_threads: 1,
            profiling_prefix: None,
        }
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        if self.max_batch == 0
            || self.max_batch > MAX_BATCH
            || self.intra_threads == 0
            || self.intra_threads > 4
            || self.host_io_bytes < self.max_batch * IO_BYTES_PER_ITEM
        {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Admission,
                "invalid finite batch/thread/IO budget",
            ));
        }
        if let Provider::Cuda {
            device_id,
            arena_bytes,
        } = self.provider
        {
            if device_id < 0 || arena_bytes == 0 || self.profiling_prefix.is_none() {
                return Err(BackendError::new(
                    K::InvalidInput,
                    S::Admission,
                    "CUDA requires device, arena cap and profile path",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CudaEvidence {
    pub profile_path: PathBuf,
    pub executed_cuda_nodes: usize,
}

pub struct OnnxBackend {
    session: Session,
    config: BackendConfig,
    identity: [u8; 32],
    asset_identity: [u8; 32],
    cuda_evidence: Option<CudaEvidence>,
}

impl OnnxBackend {
    pub fn load(
        runtime: &OrtRuntime,
        asset: &MaiaAsset,
        config: BackendConfig,
    ) -> Result<Self, BackendError> {
        config.validate()?;
        let setup_error = |_| {
            BackendError::new(
                K::BackendUnavailable,
                S::Backend,
                "ORT session/provider setup failed",
            )
        };
        let mut builder = Session::builder()
            .map_err(setup_error)?
            .with_no_environment_execution_providers()
            .map_err(setup_error)?
            .with_intra_threads(config.intra_threads)
            .map_err(setup_error)?
            .with_inter_threads(1)
            .map_err(setup_error)?
            .with_parallel_execution(false)
            .map_err(setup_error)?
            .with_memory_pattern(false)
            .map_err(setup_error)?
            .with_intra_op_spinning(false)
            .map_err(setup_error)?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(setup_error)?;
        match config.provider {
            Provider::Cpu => {
                builder = builder
                    .with_execution_providers([CPUExecutionProvider::default()
                        .with_arena_allocator(false)
                        .build()
                        .error_on_failure()])
                    .map_err(setup_error)?;
            }
            Provider::Cuda {
                device_id,
                arena_bytes,
            } => {
                let cuda = CUDAExecutionProvider::default();
                if !cuda.is_available().map_err(setup_error)? {
                    return Err(BackendError::new(
                        K::BackendUnavailable,
                        S::Backend,
                        "CUDA EP is unavailable; CPU fallback is forbidden",
                    ));
                }
                builder = builder
                    .with_config_entry("session.disable_cpu_ep_fallback", "1")
                    .map_err(setup_error)?
                    .with_execution_providers([cuda
                        .with_device_id(device_id)
                        .with_memory_limit(arena_bytes)
                        .with_tf32(false)
                        .with_conv_max_workspace(false)
                        .build()
                        .error_on_failure()])
                    .map_err(setup_error)?;
            }
        }
        if let Some(prefix) = &config.profiling_prefix {
            builder = builder.with_profiling(prefix).map_err(setup_error)?;
        }
        let session = builder
            .commit_from_memory(asset.onnx_bytes())
            .map_err(|_| {
                BackendError::new(
                    K::BackendUnavailable,
                    S::Backend,
                    "ORT could not load the verified model with the requested provider",
                )
            })?;
        validate_interface(&session)?;
        // This is a versioned C backend identity, not a new global wire codec.
        let profile = format!("rz-maia-ort-v1;ort=1.22.0;wrapper=2.0.0-rc.10;runtime={:?};asset={:?};provider={:?};threads={};batch={};fp32;tf32=0;opt=1;sync;full=1;temp=1;sum=1e-5",
            runtime.binary_digest, asset.manifest_digest(), config.provider, config.intra_threads, config.max_batch);
        let mut result = Self {
            session,
            config,
            identity: asset::sha256(profile.as_bytes()),
            asset_identity: asset.manifest_digest(),
            cuda_evidence: None,
        };
        if matches!(result.config.provider, Provider::Cuda { .. }) {
            // A registered provider alone proves nothing. Synchronous physical
            // execution plus ORT kernel placement is required before returning.
            result.run_values(&[&vec![0.0; INPUT_VALUES]])?;
            let path = result.session.end_profiling().map_err(setup_error)?;
            let bytes = asset::read_bounded(Path::new(&path), 4 * 1024 * 1024)?;
            let executed_cuda_nodes = verify_cuda_profile(&bytes)?;
            result.cuda_evidence = Some(CudaEvidence {
                profile_path: path.into(),
                executed_cuda_nodes,
            });
        }
        Ok(result)
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn asset_identity(&self) -> [u8; 32] {
        self.asset_identity
    }
    pub fn config(&self) -> &BackendConfig {
        &self.config
    }
    pub fn cuda_evidence(&self) -> Option<&CudaEvidence> {
        self.cuda_evidence.as_ref()
    }

    pub fn run(&mut self, inputs: &[&EncodedInput]) -> Result<Vec<RawOutput>, BackendError> {
        if inputs.len() > self.config.max_batch {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Admission,
                "batch exceeds configured limit",
            ));
        }
        self.run_values(
            &inputs
                .iter()
                .map(|input| input.values())
                .collect::<Vec<_>>(),
        )
    }

    /// Dense entry point for independent reference fixtures, still fully checked.
    pub fn run_values(&mut self, inputs: &[&[f32]]) -> Result<Vec<RawOutput>, BackendError> {
        if inputs.is_empty() || inputs.len() > self.config.max_batch {
            return Err(BackendError::new(
                K::ResourceExhausted,
                S::Admission,
                "empty or oversized batch",
            ));
        }
        if inputs
            .iter()
            .any(|input| input.len() != INPUT_VALUES || input.iter().any(|v| !v.is_finite()))
        {
            return Err(BackendError::new(
                K::InvalidInput,
                S::Admission,
                "input must be finite FP32 [112,8,8]",
            ));
        }
        let mut dense = Vec::new();
        dense
            .try_reserve_exact(inputs.len() * INPUT_VALUES)
            .map_err(|_| {
                BackendError::new(
                    K::ResourceExhausted,
                    S::Admission,
                    "input staging allocation failed",
                )
            })?;
        for input in inputs {
            dense.extend_from_slice(input);
        }
        let input = Tensor::from_array(([inputs.len(), 112, 8, 8], dense)).map_err(|_| {
            BackendError::new(
                K::BackendFailure,
                S::Backend,
                "cannot create ORT input tensor",
            )
        })?;
        // No RunOptions enabling asynchronous EP execution or terminate-on-cancel.
        // On both success and error ORT's default synchronous Run has returned
        // before input/output storage is released. D keeps its lease throughout.
        let outputs = self
            .session
            .run(ort::inputs![INPUT_NAME => input])
            .map_err(|_| {
                BackendError::new(K::BackendFailure, S::Backend, "synchronous ORT Run failed")
            })?;
        let malformed = || {
            BackendError::new(
                K::NumericalFailure,
                S::Output,
                "ORT output shape/dtype differs",
            )
        };
        let (policy_shape, policy) = outputs
            .get(POLICY_NAME)
            .ok_or_else(malformed)?
            .try_extract_tensor::<f32>()
            .map_err(|_| malformed())?;
        let (wdl_shape, wdl) = outputs
            .get(WDL_NAME)
            .ok_or_else(malformed)?
            .try_extract_tensor::<f32>()
            .map_err(|_| malformed())?;
        if policy_shape.as_ref() != [inputs.len() as i64, POLICY_SIZE as i64]
            || wdl_shape.as_ref() != [inputs.len() as i64, 3]
        {
            return Err(malformed());
        }
        let mut result = Vec::with_capacity(inputs.len());
        for (policy, wdl) in policy.chunks_exact(POLICY_SIZE).zip(wdl.chunks_exact(3)) {
            let raw = RawOutput {
                policy_logits: policy.to_vec(),
                wdl: wdl.to_vec(),
            };
            // Validate every raw element and WDL, before a legal view is attached.
            output::validate_maia(&raw, &[0]).map_err(|_| {
                BackendError::new(
                    K::NumericalFailure,
                    S::Output,
                    "nonfinite or inadmissible model output",
                )
            })?;
            result.push(raw);
        }
        Ok(result)
    }
}

fn validate_interface(session: &Session) -> Result<(), BackendError> {
    let valid = |value: &ValueType, expected: &[i64]| {
        matches!(value,
        ValueType::Tensor { ty: TensorElementType::Float32, shape, .. } if shape.as_ref() == expected)
    };
    if session.inputs.len() != 1
        || session.outputs.len() != 2
        || session.inputs[0].name != INPUT_NAME
        || !valid(&session.inputs[0].input_type, &[-1, 112, 8, 8])
        || ![(POLICY_NAME, 1858), (WDL_NAME, 3)]
            .iter()
            .all(|(name, size)| {
                session
                    .outputs
                    .iter()
                    .any(|out| out.name == *name && valid(&out.output_type, &[-1, *size]))
            })
    {
        return Err(BackendError::new(
            K::UnsupportedModel,
            S::Asset,
            "expected dynamic-batch FP32 Maia input/logits/WDL interface",
        ));
    }
    Ok(())
}

/// Strict evidence check; an empty/registration-only profile never passes.
pub fn verify_cuda_profile(bytes: &[u8]) -> Result<usize, BackendError> {
    let fail = || {
        BackendError::new(
            K::BackendUnavailable,
            S::Backend,
            "profile does not prove exclusive CUDA node execution",
        )
    };
    if bytes.len() > 4 * 1024 * 1024 {
        return Err(fail());
    }
    let events: Vec<serde_json::Value> = serde_json::from_slice(bytes).map_err(|_| fail())?;
    let mut count = 0;
    for event in events {
        if event.get("cat").and_then(|v| v.as_str()) != Some("Node") {
            continue;
        }
        if let Some(provider) = event.pointer("/args/provider").and_then(|v| v.as_str()) {
            if provider != "CUDAExecutionProvider" {
                return Err(fail());
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err(fail());
    }
    Ok(count)
}
