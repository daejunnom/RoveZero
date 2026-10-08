//! Opt-in resident CUDA records, owned by the existing exclusive PALS worker.
//!
//! Body inspection, initialized provider-count admission, physical Run/sync,
//! finite output, and selected-slice certification are separate boundaries.
//! This file provides no caller completion flag or synthetic CUDA constructor.
//! Pending values are installed in backend state before native allocation/bind.
//! Unknown completion retains every bank/pin/value/binding/session until exit.

use super::super::device_packing_graph::{
    CheckedFixedPackingGraph, PackingGraphDataType, PackingGraphDimension,
    PackingGraphTensorDescriptor,
};
use super::super::{
    cuda_control, device, fail, model_input, native, startup_enter, startup_return, ActiveInputs,
    BackendError, CauseCode, Layout, PalsBackendStats, PalsOnnxBackend, PalsOnnxConfig,
    PalsStartupBackendStage, PreparedPalsTensors, Provider, StartupRoleTrace, K, S,
};
use super::*;
use ort::execution_providers::{ArenaExtendStrategy, CUDAExecutionProvider, ExecutionProvider};
use ort::io_binding::IoBinding;
use ort::logging::LogLevel;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use rz_native_loader::ort_binding::run_fixed_binding;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

type NativeResult<T> = std::result::Result<T, BackendError>;
const PACKING_NODES: usize = 275;
const MAX_INITIAL_LOG_BYTES: u64 = 1024 * 1024;
// The reused collector bounds 5,000 nodes plus 32 lines; both String vectors
// can retain rounded allocation capacities even when the strict gate rejects.
const MAX_INITIAL_LOG_STRING_HEADERS: u64 = 16_384;
const MAX_INITIAL_RECORD_BYTES: usize = 2 * 1024 * 1024;
const COUNT_SCOPE: &str = "initialized_provider_count";
const INITIAL_LOG_DIGEST_DOMAIN: &str = "rz-pals-fixed-packing-initialized-provider-count/1";

fn page_error(error: DevicePageError) -> BackendError {
    fail(
        K::ResourceExhausted,
        S::Admission,
        "resident CUDA page admission failed",
    )
    .with_external_cause(CauseCode::InputAllocation, &error)
}
fn body_error(error: impl std::fmt::Display) -> BackendError {
    fail(
        K::UnsupportedModel,
        S::Asset,
        "checked packing body descriptor failed",
    )
    .with_external_cause(CauseCode::ModelLoad, &error)
}
fn refused(detail: &'static str) -> BackendError {
    fail(K::UnsupportedModel, S::Admission, detail)
}
fn arithmetic(detail: &'static str) -> BackendError {
    fail(K::ResourceExhausted, S::Admission, detail)
}
fn add(a: u64, b: u64) -> NativeResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| arithmetic("resident CUDA byte arithmetic overflow"))
}
fn mul(a: usize, b: usize) -> NativeResult<u64> {
    let value = a
        .checked_mul(b)
        .ok_or_else(|| arithmetic("resident CUDA byte arithmetic overflow"))?;
    u64::try_from(value).map_err(|_| arithmetic("resident CUDA byte conversion overflow"))
}
fn alias<T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug>(
    tensor: &Tensor<T>,
) -> ort::Result<Tensor<T>> {
    // The same owned OrtValue is retained. Tensor::clone would copy data.
    tensor
        .view()
        .try_upgrade()
        .map_err(|_| ort::Error::new("resident owned tensor cannot upgrade view"))
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CudaRecordPageStats {
    pub admitted_views: u64,
    pub public_subset_runs_attempted: u64,
    pub public_subset_runs_completed: u64,
    pub packing_runs_attempted: u64,
    pub packing_runs_completed: u64,
    pub private_joined_runs_attempted: u64,
    pub private_joined_runs_completed: u64,
    pub certified_board_slices: u64,
    pub certified_record_slices: u64,
    pub published_blocks: u64,
    pub last_unique_whole_pins: usize,
    pub last_reserved_host_bytes: u64,
    pub last_reserved_device_bytes: u64,
}
fn bump(counter: &mut u64) -> NativeResult<()> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| arithmetic("resident CUDA counter overflow"))?;
    Ok(())
}

/// Observation DTO only; cloning it grants no Session, Run, or slice authority.
#[derive(Clone, Debug, Serialize)]
pub struct CudaRecordPageSnapshot {
    pub schema: &'static str,
    pub initialized_scope: &'static str,
    pub initialized_provider_count: usize,
    /// Actual held auxiliary Session count, separate from loaded model sessions.
    pub packing_native_sessions: usize,
    pub initialized_log_sha256: [u8; 32],
    pub initialized_log_digest_domain: &'static str,
    pub packing_graph_sha256: [u8; 32],
    pub process_epoch: u64,
    pub game_generation: u64,
    pub model_manifest_sha256: [u8; 32],
    pub model_epoch: [u8; 32],
    pub public_graph_sha256: [u8; 32],
    pub encoding_sha256: [u8; 32],
    pub runtime_sha256: [u8; 32],
    pub runtime_identity_scope: &'static str,
    pub device_id: u32,
    pub live_blocks: usize,
    pub certified_projections: usize,
    pub whole_owner_host_bytes: u64,
    pub whole_owner_device_bytes: u64,
    pub active_invocation: bool,
    pub physical_completion_unknown: bool,
    pub quarantined: bool,
    pub stats: CudaRecordPageStats,
}

// This capability is never constructed from an external placement DTO.
struct InitializedPackingCount {
    graph_sha256: [u8; 32],
    log_sha256: [u8; 32],
    count: usize,
}

/// The bank stores immutable values only. IoBinding is Send-only and belongs
/// to the active invocation. Allocator is Send-only: the sealed Mutex is solely
/// a lifetime keeper, with no public lock/reallocation/concurrent native API.
pub(in crate::pals_onnx) struct CudaPublicBacking {
    domain: DevicePageDomain,
    tokens: usize,
    key: Tensor<f32>,
    value: Tensor<f32>,
    _allocator: Mutex<Allocator>,
}
impl sealed::Backing for CudaPublicBacking {}
impl DevicePublicBacking for CudaPublicBacking {
    fn domain(&self) -> DevicePageDomain {
        self.domain
    }
    fn tokens(&self) -> usize {
        self.tokens
    }
    fn whole_payload(&self) -> Result<DevicePagePayload> {
        // Known tensor payload, not allocator arena/workspace or measured peak.
        Ok(DevicePagePayload {
            host: 0,
            device: kv_bytes(self.tokens)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Prepared,
    PublicStarted,
    PublicFenced,
    PackingStarted,
    PackingFenced,
    PrivateStarted,
    PrivateFenced,
}
impl Phase {
    fn unknown(self) -> bool {
        matches!(
            self,
            Self::PublicStarted | Self::PackingStarted | Self::PrivateStarted
        )
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Source {
    Hit(DeviceProjectionOffset),
    Pending(u16),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Port {
    source: Source,
    certification_required: bool,
}

struct PublicSubsetInputs {
    board: Tensor<i64>,
    metadata: Tensor<f32>,
    records: Tensor<f32>,
    mask: Tensor<bool>,
}
struct PendingCudaPublicBlock {
    descriptor: DevicePublicBlockDescriptor,
    inputs: Option<PublicSubsetInputs>,
    binding: Option<IoBinding>,
    key: Option<Tensor<f32>>,
    value: Option<Tensor<f32>>,
    mask: Option<Tensor<bool>>,
    // Allocator always follows values/binding on normal destruction.
    allocator: Option<Allocator>,
}
struct PackingAllocation {
    binding: Option<IoBinding>,
    key: Option<Tensor<f32>>,
    value: Option<Tensor<f32>>,
    finite: Option<Tensor<bool>>,
    mask: Option<Tensor<bool>>,
    offsets: Vec<Tensor<i64>>,
    stop: Option<Tensor<i64>>,
    allocator: Option<Allocator>,
}
struct ActiveCudaPageInvocation {
    phase: Phase,
    plan: DevicePagePlan<CudaPublicBacking>,
    reservation: DevicePageReservation,
    base: Port,
    ports: [Port; DEVICE_PAGE_RECORD_CAPACITY],
    certify_board: bool,
    certify_records: [bool; DEVICE_PAGE_RECORD_CAPACITY],
    pending: Option<PendingCudaPublicBlock>,
    packing: PackingAllocation,
    completion: Option<PackingCompletedSlices>,
    published_pin: Option<MemoryPin<DevicePublicBlock<CudaPublicBacking>>>,
}

// Only actual packing Run/sync/shape/finite success constructs this token.
// It is retained through the private read and is never Clone or public.
struct PackingCompletedSlices {
    instance_seal: RegistryInstanceSeal,
    namespace: DevicePageNamespace,
    full_input_key: [u8; 32],
    packing_graph: [u8; 32],
    pending_identity: Option<(MemoryKey, u64)>,
    board: bool,
    records: [bool; DEVICE_PAGE_RECORD_CAPACITY],
    publication_finished: bool,
}

/// Exclusive captured state. No public constructor/Run/borrow of its handles.
pub(in crate::pals_onnx) struct CudaRecordPages {
    registry: DevicePagesRegistry<CudaPublicBacking>,
    declaration: DevicePageInvocationDeclaration,
    graph: CheckedFixedPackingGraph,
    initial_count: Option<InitializedPackingCount>,
    initialization_log: Option<Arc<Mutex<cuda_control::PlacementLog>>>,
    profile_root: PathBuf,
    active: Option<ActiveCudaPageInvocation>,
    quarantine: Option<BackendError>,
    initializing: bool,
    stats: CudaRecordPageStats,
    // Sessions follow all dependent bindings/values on normal destruction.
    packing_session: Option<Session>,
}

impl CudaRecordPages {
    pub(in crate::pals_onnx) fn has_active_invocation(&self) -> bool {
        self.active.is_some()
    }
    pub(in crate::pals_onnx) fn physical_completion_unknown(&self) -> bool {
        self.initializing
            || self
                .active
                .as_ref()
                .is_some_and(|active| active.phase.unknown())
    }
    pub(in crate::pals_onnx) fn quarantine_cause(&self) -> Option<&BackendError> {
        self.quarantine.as_ref()
    }
    pub(in crate::pals_onnx) fn release_completed(&mut self) -> NativeResult<()> {
        if self.quarantine.is_some() || self.physical_completion_unknown() {
            return Err(refused(
                "resident CUDA completion cannot release unknown owners",
            ));
        }
        self.active = None;
        Ok(())
    }
    pub(in crate::pals_onnx) fn clear_for_new_game(&mut self, next_game: u64) -> NativeResult<()> {
        if self.active.is_some()
            || self.quarantine.is_some()
            || self.initializing
            || next_game <= self.registry.namespace.game_generation
        {
            return Err(refused(
                "resident CUDA newgame requires idle owner and increasing generation",
            ));
        }
        // Atomic bank refusal precedes every registry/namespace mutation.
        self.registry.clear().map_err(page_error)?;
        self.registry.namespace.game_generation = next_game;
        Ok(())
    }
    pub(in crate::pals_onnx) fn snapshot(&self) -> NativeResult<CudaRecordPageSnapshot> {
        let count = self
            .initial_count
            .as_ref()
            .ok_or_else(|| refused("resident packing initialization not admitted"))?;
        let payload = self
            .registry
            .whole_owner_reservation()
            .map_err(page_error)?;
        let namespace = self.registry.namespace;
        let DevicePageDomain::Cuda {
            device_id,
            runtime_sha256,
        } = namespace.domain
        else {
            return Err(refused("resident native registry has a non-CUDA namespace"));
        };
        Ok(CudaRecordPageSnapshot {
            schema: "rz-pals-resident-cuda-record-pages-observation/1",
            initialized_scope: COUNT_SCOPE,
            initialized_provider_count: count.count,
            packing_native_sessions: usize::from(self.packing_session.is_some()),
            initialized_log_sha256: count.log_sha256,
            packing_graph_sha256: count.graph_sha256,
            initialized_log_digest_domain: INITIAL_LOG_DIGEST_DOMAIN,
            process_epoch: namespace.process_epoch.0,
            game_generation: namespace.game_generation,
            model_manifest_sha256: namespace.model_manifest,
            model_epoch: namespace.model_epoch,
            public_graph_sha256: namespace.public_graph,
            encoding_sha256: namespace.encoding,
            runtime_sha256,
            runtime_identity_scope: "closed_cuda_library_bundle",
            device_id,
            live_blocks: self.registry.blocks.len(),
            certified_projections: self.registry.certified_projection_count(),
            whole_owner_host_bytes: payload.host,
            whole_owner_device_bytes: payload.device,
            active_invocation: self.active.is_some(),
            physical_completion_unknown: self.physical_completion_unknown(),
            quarantined: self.quarantine.is_some(),
            stats: self.stats.clone(),
        })
    }
    fn initialize(&mut self, config: PalsOnnxConfig) -> NativeResult<()> {
        let Provider::Cuda {
            device_id,
            arena_bytes,
        } = config.provider
        else {
            return Err(refused("packing requires explicit CUDA"));
        };
        let configure = |error| {
            native(
                CauseCode::SessionConfiguration,
                "packing session configuration failed",
                error,
            )
        };
        let cuda = CUDAExecutionProvider::default();
        if !cuda.is_available().map_err(configure)? {
            return Err(fail(
                K::BackendUnavailable,
                S::Backend,
                "packing CUDA provider unavailable; CPU fallback forbidden",
            ));
        }
        let log = Arc::new(Mutex::new(cuda_control::PlacementLog::default()));
        self.initialization_log = Some(Arc::clone(&log));
        let captured = Arc::clone(&log);
        let id = format!(
            "pals-resident-packing-{}-{}",
            self.registry.namespace.process_epoch.0, self.registry.instance_seal.0
        );
        let expected_id = id.clone();
        self.initializing = true;
        let builder = Session::builder()
            .map_err(configure)?
            .with_no_environment_execution_providers()
            .map_err(configure)?
            .with_intra_threads(config.intra_threads)
            .map_err(configure)?
            .with_inter_threads(1)
            .map_err(configure)?
            .with_parallel_execution(false)
            .map_err(configure)?
            .with_intra_op_spinning(false)
            .map_err(configure)?
            .with_inter_op_spinning(false)
            .map_err(configure)?
            .with_memory_pattern(false)
            .map_err(configure)?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(configure)?
            .with_config_entry("session.disable_cpu_ep_fallback", "1")
            .map_err(configure)?
            .with_execution_providers([cuda
                .with_device_id(device_id)
                .with_memory_limit(arena_bytes)
                .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                .with_tf32(false)
                .with_cuda_graph(false)
                .build()
                .error_on_failure()])
            .map_err(configure)?
            .with_log_id(id)
            .map_err(configure)?
            .with_log_level(LogLevel::Verbose)
            .map_err(configure)?
            .with_log_verbosity(1)
            .map_err(configure)?
            .with_logger(Box::new(move |level, _, id, location, message| {
                // No panic or policy acceptance in the native logger callback.
                if id != expected_id {
                    return;
                }
                if let Ok(mut log) = captured.lock() {
                    if level == LogLevel::Verbose
                        && location.contains("VerifyEachNodeIsAssignedToAnEp")
                    {
                        log.capture(message);
                    } else if location.contains("transformer_memcpy.cc")
                        && ((level == LogLevel::Info && location.contains("AddCopyNode"))
                            || (level == LogLevel::Warning && location.contains("ApplyImpl")))
                    {
                        log.capture_transfer(message);
                    }
                }
            }))
            .map_err(configure)?
            .with_profiling(self.profile_root.join("packing-kernels"))
            .map_err(configure)?;
        // No direct-reference initializer option; owned body bytes stay pinned.
        let loaded = builder.commit_from_memory(self.graph.graph_bytes());
        match loaded {
            Ok(session) => self.packing_session = Some(session),
            Err(error) => {
                // Constructor refusal supplied no native input/Run lease. Save
                // the bounded raw initialization log but preserve native cause.
                let failure = native(
                    CauseCode::ModelLoad,
                    "packing session initialization failed",
                    error,
                );
                let _ = self.record_initialization(false, None);
                self.initializing = false;
                return Err(failure);
            }
        }
        self.initializing = false;
        let interface = validate_packing_session(
            self.packing_session
                .as_ref()
                .expect("packing session installed"),
            &self.graph,
        );
        let count = if interface.is_ok() {
            let guard = log
                .lock()
                .map_err(|_| refused("packing placement collector poisoned"))?;
            count_initialization(&self.graph, &guard)
        } else {
            Err(interface.expect_err("interface rejected"))
        };
        let recorded = self.record_initialization(true, count.as_ref().ok());
        // Gate failure has priority; IO failure cannot bypass a successful gate.
        self.initial_count = Some(count?);
        recorded?;
        self.initialization_log = None;
        Ok(())
    }
    fn record_initialization(
        &self,
        loaded: bool,
        count: Option<&InitializedPackingCount>,
    ) -> NativeResult<()> {
        #[derive(Serialize)]
        struct Record<'a> {
            schema: &'static str,
            scope: &'static str,
            graph_sha256: [u8; 32],
            session_created: bool,
            count_gate_passed: bool,
            provider_count: Option<usize>,
            named_node_proof: bool,
            run_observed: bool,
            finite_observed: bool,
            fence_observed: bool,
            lines: &'a [String],
            transfer_lines_present: bool,
            collector_overflowed: bool,
            log_digest_domain: &'static str,
            log_sha256: Option<[u8; 32]>,
        }
        let guard = self
            .initialization_log
            .as_ref()
            .ok_or_else(|| refused("packing initialization collector missing"))?
            .lock()
            .map_err(|_| refused("packing initialization collector poisoned"))?;
        // A rejected/overflowed prefix remains diagnostics, never count proof.
        let lines = guard.diagnostic_lines();
        let record = Record {
            schema: "rz-pals-fixed-packing-initialization/1",
            scope: COUNT_SCOPE,
            graph_sha256: self.graph.registration().graph.sha256,
            session_created: loaded,
            count_gate_passed: count.is_some(),
            provider_count: count.map(|value| value.count),
            named_node_proof: false,
            run_observed: false,
            finite_observed: false,
            fence_observed: false,
            lines,
            transfer_lines_present: guard.has_transfer_lines(),
            collector_overflowed: guard.overflowed(),
            log_digest_domain: INITIAL_LOG_DIGEST_DOMAIN,
            log_sha256: count.map(|value| value.log_sha256),
        };
        let file =
            std::fs::File::create_new(self.profile_root.join("packing-initial-placement.json"))
                .map_err(|error| {
                    fail(
                        K::BackendFailure,
                        S::Backend,
                        "cannot preserve packing initialization",
                    )
                    .with_external_cause(CauseCode::ProfilingStart, &error)
                })?;
        // Stream escaping into the owned output. A rejected prefix is retained
        // without allocating an unbounded serialized JSON buffer.
        let mut output = BoundedInitializationRecord { file, written: 0 };
        serde_json::to_writer(&mut output, &record).map_err(|error| {
            fail(
                K::BackendFailure,
                S::Backend,
                "cannot preserve bounded packing initialization",
            )
            .with_external_cause(CauseCode::ProfilingStart, &error)
        })?;
        std::io::Write::flush(&mut output).map_err(|error| {
            fail(
                K::BackendFailure,
                S::Backend,
                "cannot flush packing initialization",
            )
            .with_external_cause(CauseCode::ProfilingStart, &error)
        })
    }
}

struct BoundedInitializationRecord {
    file: std::fs::File,
    written: usize,
}
impl std::io::Write for BoundedInitializationRecord {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .written
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("packing initialization byte overflow"))?;
        if next > MAX_INITIAL_RECORD_BYTES {
            return Err(std::io::Error::other(
                "packing initialization output bound exceeded",
            ));
        }
        let count = std::io::Write::write(&mut self.file, bytes)?;
        self.written += count;
        Ok(count)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        std::io::Write::flush(&mut self.file)
    }
}

fn validate_packing_session(
    session: &Session,
    graph: &CheckedFixedPackingGraph,
) -> NativeResult<()> {
    if session.inputs.len() != 387 || session.outputs.len() != 3 {
        return Err(refused("packing loaded port count differs"));
    }
    for (index, input) in session.inputs.iter().enumerate() {
        let expected = graph.input_descriptor(index).map_err(body_error)?;
        if input.name != expected.name.as_str().map_err(body_error)?
            || !interface_matches(&input.input_type, expected)
        {
            return Err(refused(
                "packing loaded input order/type/shape/symbol differs",
            ));
        }
    }
    for (index, output) in session.outputs.iter().enumerate() {
        let expected = graph.output_descriptor(index).map_err(body_error)?;
        if output.name != expected.name.as_str().map_err(body_error)?
            || !interface_matches(&output.output_type, expected)
        {
            return Err(refused(
                "packing loaded output order/type/shape/symbol differs",
            ));
        }
    }
    Ok(())
}
fn interface_matches(value: &ValueType, expected: PackingGraphTensorDescriptor) -> bool {
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols,
    } = value
    else {
        return false;
    };
    let dtype = match expected.dtype {
        PackingGraphDataType::Float32 => TensorElementType::Float32,
        PackingGraphDataType::Int64 => TensorElementType::Int64,
        PackingGraphDataType::Bool => TensorElementType::Bool,
    };
    let dimensions = expected.shape.dimensions();
    *ty == dtype
        && shape.len() == dimensions.len()
        && dimension_symbols.len() == dimensions.len()
        && dimensions
            .iter()
            .zip(shape.iter().zip(dimension_symbols.iter()))
            .all(|(dimension, (size, symbol))| match dimension {
                PackingGraphDimension::Number(expected) => {
                    i64::try_from(*expected).is_ok_and(|expected| expected == *size)
                        && symbol.is_empty()
                }
                PackingGraphDimension::Symbol(expected) => {
                    *size == -1 && expected.as_str().is_ok_and(|expected| symbol == expected)
                }
            })
}
fn count_initialization(
    graph: &CheckedFixedPackingGraph,
    log: &cuda_control::PlacementLog,
) -> NativeResult<InitializedPackingCount> {
    if graph.inspection().node_count != PACKING_NODES || log.has_transfer_lines() {
        return Err(refused(
            "packing initialization body count or transfer scope differs",
        ));
    }
    let lines = log.bounded_lines()?;
    let count = count_summary_lines(lines)?;
    let mut digest = Sha256::new();
    digest.update(INITIAL_LOG_DIGEST_DOMAIN.as_bytes());
    digest.update(graph.registration().graph.sha256);
    for line in lines {
        digest.update((line.len() as u64).to_le_bytes());
        digest.update(line.as_bytes());
    }
    Ok(InitializedPackingCount {
        graph_sha256: graph.registration().graph.sha256,
        log_sha256: digest.finalize().into(),
        count,
    })
}
fn count_summary_lines(lines: &[String]) -> NativeResult<usize> {
    let mut began = false;
    let mut summary = None;
    for line in lines {
        let line = line.trim();
        if line == "Node placements" {
            if began || summary.is_some() {
                return Err(refused("packing initialization section duplicated"));
            }
            began = true;
        } else if began {
            let Some((provider, count)) =
                cuda_control::placement_header(line, "All nodes placed on [")
            else {
                return Err(refused(
                    "packing initialization is not an unambiguous all-CUDA summary",
                ));
            };
            if provider != "CUDAExecutionProvider"
                || count != PACKING_NODES
                || summary.replace(count).is_some()
            {
                return Err(refused("packing initialized provider/count differs"));
            }
        } else {
            return Err(refused(
                "packing initialization contains unknown placement text",
            ));
        }
    }
    if !began {
        return Err(refused("packing initialization section missing"));
    }
    summary.ok_or_else(|| refused("packing initialization summary missing"))
}

/// Conservative known metadata reservation, not native allocator/VRAM peak.
/// Root can use this before install; no native handle or authority is returned.
pub fn resident_cuda_minimum_metadata_host_bytes(
    graph: &CheckedFixedPackingGraph,
) -> NativeResult<u64> {
    let mut total = graph.known_retained_host_bytes().map_err(body_error)?;
    for amount in [
        size_of::<CudaRecordPages>(),
        size_of::<ActiveCudaPageInvocation>(),
        size_of::<PendingCudaPublicBlock>(),
        size_of::<PublicSubsetInputs>(),
        size_of::<PackingAllocation>(),
    ] {
        total = add(total, amount as u64)?;
    }
    total = add(total, mul(129, size_of::<Tensor<i64>>())?)?;
    // Bounded binding names and initialization collector are conservative
    // metadata reservations, even after initialization scratch is released.
    total = add(total, mul(387, 64)?)?;
    total = add(total, MAX_INITIAL_LOG_BYTES)?;
    total = add(
        total,
        MAX_INITIAL_LOG_STRING_HEADERS
            .checked_mul(size_of::<String>() as u64)
            .ok_or_else(|| arithmetic("packing collector reservation overflow"))?,
    )?;
    // Descriptor features and subset Tensor backing overlap. The generic plan
    // charges the descriptor; this charges the additional actual subset input.
    add(total, (128 * 16 * 4 + 128 + 64 * 8 + 16 * 4) as u64)
}

impl PalsOnnxBackend {
    pub(in crate::pals_onnx) fn install_cuda_record_pages(
        &mut self,
        graph: CheckedFixedPackingGraph,
        namespace: DevicePageNamespace,
        limits: DevicePagesLimits,
        declaration: DevicePageInvocationDeclaration,
        profile_root: PathBuf,
    ) -> NativeResult<()> {
        let Provider::Cuda { device_id, .. } = self.config.provider else {
            return Err(refused("resident pages require explicit CUDA"));
        };
        if self.config.device_public_memory
            || !self.config.cache_public_memory
            || self.layout != Layout::SharedPcIf
            || self.private_warm.is_some()
            || self.record_pages.is_some()
            || self.active.is_some()
            || self.active_memory.is_some()
            || self.device_memory.is_some()
            || self.device_role.is_some()
            || self.cuda_record_pages.is_some()
            || self.quarantine.is_some()
            || self.stats.admitted_role_requests != 0
            || self.cached_memory_key.is_some()
        {
            return Err(refused("resident CUDA selection must precede admission and cannot combine legacy device/host/Warm policies"));
        }
        let audit = self.cuda_control_audit.as_ref().ok_or_else(|| {
            refused("resident learned graphs require their existing initialized CUDA control gate")
        })?;
        if audit.rejected.is_some()
            || audit.initialization.len() != 2
            || !["public", "shared_pc"]
                .iter()
                .all(|role| audit.initialization.iter().any(|item| item.role == *role))
        {
            return Err(refused(
                "resident learned initialization gates are absent or rejected",
            ));
        }
        let public = self
            .residency
            .graphs
            .iter()
            .find(|item| item.role == "public")
            .ok_or_else(|| refused("registered public producer is absent"))?;
        let raw_semantic = self.rules_input_semantic_sha256.ok_or_else(|| {
            refused("resident producer lacks registered Rules semantic input profile")
        })?;
        let mut encoding = Sha256::new();
        encoding.update(super::super::PALS_ENCODING_SCHEMA);
        encoding.update(raw_semantic);
        let expected_encoding: [u8; 32] = encoding.finalize().into();
        let runtime_bundle = self.runtime.bundle_digest().ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Admission,
                "resident CUDA requires the actual complete runtime bundle identity",
            )
        })?;
        if namespace.process_epoch.0 == 0
            || namespace.game_generation != self.game_generation
            || namespace.frozen_epoch == 0
            || namespace.model_manifest != self.manifest_digest
            || namespace.model_epoch != self.epoch
            || namespace.public_graph != public.sha256
            || namespace.encoding != expected_encoding
            || namespace.domain
                != (DevicePageDomain::Cuda {
                    device_id: u32::try_from(device_id)
                        .map_err(|_| refused("resident CUDA device is negative"))?,
                    runtime_sha256: runtime_bundle,
                })
        {
            return Err(refused(
                "resident namespace differs from this validated producer/runtime",
            ));
        }
        let metadata = declaration
            .additional_owner_metadata_payload
            .ok_or_else(|| refused("resident native metadata declaration missing"))?;
        if metadata.host < resident_cuda_minimum_metadata_host_bytes(&graph)?
            || declaration.packing_session_bytes
                != Some(
                    graph
                        .registered_artifacts()
                        .resources()
                        .declared_packing_session_bytes,
                )
        {
            return Err(arithmetic(
                "resident native metadata/session declaration is below registered backing",
            ));
        }
        // Validate every mandatory declaration and upper bound before Session
        // construction. Runtime invocation later adds actual bank/plan/input.
        let initial = DevicePageReservation::calculate(
            DevicePagePayload::default(),
            0,
            0,
            size_of::<DevicePublicBlock<CudaPublicBacking>>() as u64,
            namespace.domain,
            declaration,
        )
        .map_err(page_error)?;
        if initial.total.host > limits.max_invocation_host_bytes
            || initial.total.device > limits.max_invocation_device_bytes
        {
            return Err(arithmetic(
                "resident factory aggregate declaration exceeds limits",
            ));
        }
        let registry = DevicePagesRegistry::new(namespace, limits).map_err(page_error)?;
        // Registry construction charges its actual reserved Vec/bank capacities.
        // Include that retained container before any packing Session allocation;
        // the zero-container scalar preflight above is only an earlier rejection.
        let initial_with_container = DevicePageReservation::calculate(
            DevicePagePayload::default(),
            registry.container_host_reservation(),
            0,
            size_of::<DevicePublicBlock<CudaPublicBacking>>() as u64,
            namespace.domain,
            declaration,
        )
        .map_err(page_error)?;
        if initial_with_container.total.host > limits.max_invocation_host_bytes
            || initial_with_container.total.device > limits.max_invocation_device_bytes
        {
            return Err(arithmetic(
                "resident factory retained container and declaration exceed limits",
            ));
        }
        let root = cuda_control::owned_profile_root(&profile_root)?;
        // Owned state is captured before the first native Session operation.
        self.cuda_record_pages = Some(CudaRecordPages {
            registry,
            declaration,
            graph,
            initial_count: None,
            initialization_log: None,
            profile_root: root,
            active: None,
            quarantine: None,
            initializing: false,
            stats: CudaRecordPageStats::default(),
            packing_session: None,
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.cuda_record_pages
                .as_mut()
                .expect("resident owner installed")
                .initialize(self.config)
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                let owner = self
                    .cuda_record_pages
                    .as_mut()
                    .expect("resident owner installed");
                if owner.physical_completion_unknown() {
                    owner.quarantine = Some(error.clone());
                    self.quarantine = Some(error.clone());
                } else {
                    self.cuda_record_pages = None;
                }
                Err(error)
            }
            Err(payload) => {
                let cause = unwind_cause(payload.as_ref());
                self.cuda_record_pages
                    .as_mut()
                    .expect("resident owner installed")
                    .quarantine = Some(cause.clone());
                self.quarantine = Some(cause);
                std::panic::resume_unwind(payload)
            }
        }
    }

    pub(in crate::pals_onnx) fn run_cuda_record_pages(
        &mut self,
        input: &PalsModelInput,
        trace: Option<&StartupRoleTrace>,
    ) -> NativeResult<PalsRawOutput> {
        if let Some(cause) = &self.quarantine {
            return Err(cause.clone());
        }
        let owner = self
            .cuda_record_pages
            .as_mut()
            .ok_or_else(|| refused("resident CUDA was not selected"))?;
        if let Some(cause) = &owner.quarantine {
            return Err(cause.clone());
        }
        if owner.active.is_some() || owner.initial_count.is_none() || owner.initializing {
            return Err(refused("resident CUDA owner is not idle/admitted"));
        }
        if self.active.is_some() || self.device_role.is_some() {
            return Err(refused(
                "resident entry requires no preceding physical inputs/private outputs",
            ));
        }
        input.validate(&self.model_config).map_err(model_input)?;
        startup_enter(trace, PalsStartupBackendStage::InputPreparation);
        // prepare_tensors only produces owned Rust backing. Its actual Vec
        // capacities are admitted before the first native Tensor/bind/copy.
        let prepared = input
            .prepare_tensors(&self.model_config)
            .map_err(model_input)
            .inspect_err(|_| {
                startup_return(trace, PalsStartupBackendStage::InputPreparation, false)
            })?;
        let active_host = original_active_host_bytes(&prepared)?;
        owner
            .prepare(input, &self.model_config, active_host)
            .inspect_err(|_| {
                startup_return(trace, PalsStartupBackendStage::InputPreparation, false)
            })?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.active = Some(
                ActiveInputs::new(prepared, input.role == PalsRole::Critic, None).inspect_err(
                    |_| startup_return(trace, PalsStartupBackendStage::InputPreparation, false),
                )?,
            );
            if self
                .active
                .as_ref()
                .expect("original inputs installed")
                .host_bytes
                != active_host
            {
                startup_return(trace, PalsStartupBackendStage::InputPreparation, false);
                return Err(arithmetic(
                    "original input backing differs from admitted capacities",
                ));
            }
            startup_return(trace, PalsStartupBackendStage::InputPreparation, true);
            self.stats.admitted_role_requests = self.stats.admitted_role_requests.saturating_add(1);
            self.run_cuda_record_pages_inner(input, trace)
        }));
        match result {
            Ok(result) => {
                let owner = self
                    .cuda_record_pages
                    .as_mut()
                    .expect("resident owner installed");
                if let Err(error) = &result {
                    if owner.physical_completion_unknown() {
                        owner.quarantine = Some(error.clone());
                        self.quarantine = Some(error.clone());
                    }
                }
                if owner.quarantine.is_none() {
                    self.device_role = None;
                    self.active = None;
                    owner.release_completed()?;
                }
                result
            }
            Err(payload) => {
                let cause = unwind_cause(payload.as_ref());
                let owner = self
                    .cuda_record_pages
                    .as_mut()
                    .expect("resident owner installed");
                owner.quarantine = Some(cause.clone());
                self.quarantine = Some(cause);
                std::panic::resume_unwind(payload)
            }
        }
    }
    fn run_cuda_record_pages_inner(
        &mut self,
        input: &PalsModelInput,
        trace: Option<&StartupRoleTrace>,
    ) -> NativeResult<PalsRawOutput> {
        let owner = self
            .cuda_record_pages
            .as_mut()
            .expect("resident owner installed");
        owner.run_public_subset(
            self.public.as_mut().expect("validated public session"),
            self.active
                .as_ref()
                .expect("original physical inputs installed"),
            &mut self.stats,
            trace,
        )?;
        owner.run_packing()?;
        owner.publish_selected_slices()?;
        {
            let owner = self
                .cuda_record_pages
                .as_mut()
                .expect("resident owner installed");
            let active = owner
                .active
                .as_mut()
                .expect("resident invocation installed");
            if !active.completion.as_ref().is_some_and(|proof| {
                proof.publication_finished
                    && proof.full_input_key == active.plan.full_input_key
                    && proof.namespace == owner.registry.namespace
                    && proof.instance_seal == owner.registry.instance_seal
            }) {
                return Err(refused(
                    "resident private read lacks actual completed packing capability",
                ));
            }
            if active.plan.full_input_key
                != input
                    .canonical_input_key(&self.model_config)
                    .map_err(model_input)?
            {
                return Err(refused("resident private original input identity changed"));
            }
            // Even a partially constructed output/binding must not release the
            // joined/pinned CUDA owners after an unknown native operation.
            active.phase = Phase::PrivateStarted;
        }
        // Existing DeviceRole retains original candidate/query/D semantics and
        // fresh outputs. No helper score replaces the model's private inputs.
        self.device_role = Some(
            device::DeviceRole::new(
                self.shared_pc
                    .as_ref()
                    .expect("validated shared P/C session"),
                input.candidates.len().max(1),
                input.divergence_features.len().max(1),
                input.role == PalsRole::Critic,
                true,
            )
            .map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "resident private outputs allocation failed",
                    error,
                )
            })?,
        );
        let owner = self
            .cuda_record_pages
            .as_mut()
            .expect("resident owner installed");
        let active = owner
            .active
            .as_mut()
            .expect("resident invocation installed");
        bump(&mut owner.stats.private_joined_runs_attempted)?;
        startup_enter(trace, PalsStartupBackendStage::PrivateRun);
        let packing = &active.packing;
        let ran = self
            .device_role
            .as_mut()
            .expect("private outputs installed")
            .run_joined(
                self.shared_pc
                    .as_mut()
                    .expect("validated shared P/C session"),
                self.active
                    .as_ref()
                    .expect("original physical inputs installed"),
                packing.key.as_ref().expect("completed joined key"),
                packing.value.as_ref().expect("completed joined value"),
                packing.mask.as_ref().expect("current mask"),
                &mut self.stats,
            );
        startup_return(trace, PalsStartupBackendStage::PrivateRun, ran.is_ok());
        ran.map_err(|error| {
            native(
                CauseCode::OrtRun,
                "resident private binding/copy/Run/fence failed",
                error,
            )
        })?;
        active.phase = Phase::PrivateFenced;
        bump(&mut owner.stats.private_joined_runs_completed)?;
        startup_enter(trace, PalsStartupBackendStage::PrivateOutputValidation);
        let validated = (|| -> NativeResult<PalsRawOutput> {
            let output = self
                .device_role
                .as_ref()
                .expect("private output fenced")
                .extract(input)?;
            output
                .decode(input, &self.model_config)
                .map_err(model_input)?;
            self.stats.validated_role_outputs = self.stats.validated_role_outputs.saturating_add(1);
            Ok(output)
        })();
        startup_return(
            trace,
            PalsStartupBackendStage::PrivateOutputValidation,
            validated.is_ok(),
        );
        validated
    }
}

fn original_active_host_bytes(value: &PreparedPalsTensors) -> NativeResult<u64> {
    // Same actual capacities as ActiveInputs::new, without invoking ORT. Warm
    // is explicitly excluded. The created owner is compared before any Run.
    let mut total = size_of::<ActiveInputs>() as u64;
    for capacity in [value.board.capacity(), value.candidates.capacity()] {
        total = add(total, mul(capacity, 8)?)?;
    }
    for capacity in [
        value.metadata.capacity(),
        value.records.capacity(),
        value.query.capacity(),
        value.divergences.capacity(),
    ] {
        total = add(total, mul(capacity, 4)?)?;
    }
    for capacity in [
        value.record_mask.capacity(),
        value.candidate_mask.capacity(),
        value.divergence_mask.capacity(),
    ] {
        total = add(
            total,
            u64::try_from(capacity)
                .map_err(|_| arithmetic("original mask capacity conversion overflow"))?,
        )?;
    }
    add(total, 1)
}

fn unwind_cause(payload: &(dyn std::any::Any + Send)) -> BackendError {
    let text = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string native panic");
    let bounded: String = text.chars().take(256).collect();
    fail(
        K::BackendFailure,
        S::Backend,
        "resident native unwind; physical completion not released",
    )
    .with_external_cause(CauseCode::OrtRun, &std::io::Error::other(bounded))
}

impl CudaRecordPages {
    fn prepare(
        &mut self,
        input: &PalsModelInput,
        config: &PalsModelConfig,
        active_host: u64,
    ) -> NativeResult<()> {
        let plan = self.registry.plan(input, config).map_err(page_error)?;
        let original = self
            .declaration
            .original_input_and_transfer_payload
            .ok_or_else(|| refused("resident original input/transfer declaration missing"))?;
        if original.host < add(plan.known_input_host_payload, active_host)?
            || original.device < active_host
        {
            return Err(arithmetic(
                "resident original input/transfer declaration is below actual backing",
            ));
        }
        let private = self
            .declaration
            .private_output_payload
            .ok_or_else(|| refused("resident private output declaration missing"))?;
        // Shared P/C owns its D output even on a proposer request. The old
        // role-specific generic minimum omits that additional output backing.
        let additional_d = if input.role == PalsRole::Proposer {
            mul(input.divergence_features.len().max(1), 8)?
        } else {
            0
        };
        if private.host < add(plan.known_private_output_host_payload, additional_d)? {
            return Err(arithmetic(
                "resident shared private output declaration is below actual backing",
            ));
        }
        let reservation = self
            .registry
            .reserve_invocation(&plan, self.declaration)
            .map_err(page_error)?;
        let pending_needed =
            plan.base.is_none() || !plan.missing.is_empty() || plan.zero_base_required;
        let pending = if pending_needed {
            let generation = self
                .registry
                .highest_generation
                .checked_add(1)
                .ok_or_else(|| arithmetic("resident block generation overflow"))?;
            Some(PendingCudaPublicBlock {
                descriptor: DevicePublicBlockDescriptor::from_input(
                    self.registry.namespace,
                    generation,
                    input,
                    config,
                    &plan.missing,
                )
                .map_err(page_error)?,
                inputs: None,
                binding: None,
                key: None,
                value: None,
                mask: None,
                allocator: None,
            })
        } else {
            None
        };
        let (base, ports, certify_board, certify_records) = selection(
            &plan,
            pending.as_ref().map(|block| &block.descriptor),
            input,
            config,
        )?;
        bump(&mut self.stats.admitted_views)?;
        self.stats.last_unique_whole_pins = plan.pins.len();
        self.stats.last_reserved_host_bytes = reservation.total.host;
        self.stats.last_reserved_device_bytes = reservation.total.device;
        // Empty staged fields are installed before Tensor allocation or binding.
        self.active = Some(ActiveCudaPageInvocation {
            phase: Phase::Prepared,
            plan,
            reservation,
            base,
            ports,
            certify_board,
            certify_records,
            pending,
            packing: PackingAllocation {
                binding: None,
                key: None,
                value: None,
                finite: None,
                mask: None,
                offsets: Vec::new(),
                stop: None,
                allocator: None,
            },
            completion: None,
            published_pin: None,
        });
        Ok(())
    }
}

fn feature_equal(a: &[f32; 16], b: &[f32; 16]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn selection(
    plan: &DevicePagePlan<CudaPublicBacking>,
    pending: Option<&DevicePublicBlockDescriptor>,
    input: &PalsModelInput,
    config: &PalsModelConfig,
) -> NativeResult<(
    Port,
    [Port; DEVICE_PAGE_RECORD_CAPACITY],
    bool,
    [bool; DEVICE_PAGE_RECORD_CAPACITY],
)> {
    let actual = input.independent_public_plan(config).map_err(model_input)?;
    let certify_board = plan.base.is_none();
    let base = Port {
        source: if let Some(location) = plan.base {
            Source::Hit(location)
        } else {
            let _ = pending.ok_or_else(|| refused("resident board miss lacks pending producer"))?;
            Source::Pending(0)
        },
        certification_required: !certify_board,
    };
    let inactive = Port {
        source: match base.source {
            Source::Hit(location) => Source::Hit(DeviceProjectionOffset {
                token_offset: BOARD as u16,
                ..location
            }),
            Source::Pending(_) => Source::Pending(BOARD as u16),
        },
        certification_required: false,
    };
    let mut ports = [inactive; DEVICE_PAGE_RECORD_CAPACITY];
    let mut certified = [false; DEVICE_PAGE_RECORD_CAPACITY];
    for (index, location) in plan.records.iter().enumerate() {
        ports[index] = if let Some(location) = location {
            Port {
                source: Source::Hit(*location),
                certification_required: true,
            }
        } else {
            let descriptor =
                pending.ok_or_else(|| refused("resident record miss lacks pending producer"))?;
            let subset = descriptor
                .features
                .iter()
                .position(|features| feature_equal(features, &actual.features[index]))
                .ok_or_else(|| {
                    refused("resident pending feature mapping differs from original input")
                })?;
            if !descriptor.record_mask[subset] {
                return Err(refused("active resident record maps to zero padding"));
            }
            certified[subset] = true;
            Port {
                source: Source::Pending(
                    u16::try_from(BOARD + subset)
                        .map_err(|_| arithmetic("resident offset overflow"))?,
                ),
                certification_required: false,
            }
        };
    }
    if plan.records.is_empty() {
        if let Some(descriptor) = pending {
            if !descriptor.zero_padding() || !certify_board {
                return Err(refused(
                    "empty resident view requires actual zero-padded base",
                ));
            }
            certified[0] = true;
        }
    }
    Ok((base, ports, certify_board, certified))
}

fn device_number(domain: DevicePageDomain) -> NativeResult<i32> {
    match domain {
        DevicePageDomain::Cuda { device_id, .. } => {
            i32::try_from(device_id).map_err(|_| refused("resident device ID is out of range"))
        }
        _ => Err(refused("resident native backing requires CUDA domain")),
    }
}
fn cuda_tensor(tensor: &Tensor<f32>, tokens: usize, device: i32) -> bool {
    matches!(tensor.dtype(), ValueType::Tensor { ty, shape, dimension_symbols }
        if *ty == TensorElementType::Float32 && shape.as_ref() == [1, 2, tokens as i64, 64]
        && dimension_symbols.iter().all(String::is_empty))
        && tensor.memory_info().allocation_device() == AllocationDevice::CUDA
        && tensor.memory_info().device_id() == device
}
fn cpu_tensor<T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug>(
    tensor: &Tensor<T>,
) -> bool {
    tensor.memory_info().is_cpu_accessible()
}

impl CudaRecordPages {
    fn run_public_subset(
        &mut self,
        session: &mut Session,
        original: &ActiveInputs,
        stats: &mut PalsBackendStats,
        trace: Option<&StartupRoleTrace>,
    ) -> NativeResult<()> {
        let device = device_number(self.registry.namespace.domain)?;
        let active = self.active.as_mut().expect("resident invocation installed");
        if active.reservation.total.host != self.stats.last_reserved_host_bytes
            || active.reservation.total.device != self.stats.last_reserved_device_bytes
        {
            return Err(refused(
                "resident active aggregate reservation changed before bind",
            ));
        }
        let Some(pending) = &mut active.pending else {
            return Ok(());
        };
        let tokens = pending.descriptor.tokens();
        let records = pending.descriptor.features.len();
        // No callback-local CUDA owner is created. Even partial allocation is
        // captured, and the phase prevents unwinding/failed bind from releasing it.
        active.phase = Phase::PublicStarted;
        let mut flattened = Vec::new();
        flattened
            .try_reserve_exact(records * 16)
            .map_err(|_| arithmetic("public subset host allocation failed"))?;
        if flattened.capacity() > records * 16 {
            return Err(arithmetic("public subset backing exceeds exact bound"));
        }
        flattened.extend(pending.descriptor.features.iter().flatten().copied());
        pending.inputs = Some(PublicSubsetInputs {
            board: alias(&original.board).map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "cannot retain original board",
                    error,
                )
            })?,
            metadata: alias(&original.metadata).map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "cannot retain original metadata",
                    error,
                )
            })?,
            records: Tensor::from_array(([1, records, 16], flattened)).map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "cannot own public subset features",
                    error,
                )
            })?,
            mask: Tensor::from_array(([1, records], pending.descriptor.record_mask.clone()))
                .map_err(|error| {
                    native(
                        CauseCode::TensorCreate,
                        "cannot own public subset mask",
                        error,
                    )
                })?,
        });
        pending.allocator = Some(
            Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    device,
                    AllocatorType::Device,
                    MemoryType::Default,
                )
                .map_err(|error| {
                    native(
                        CauseCode::TensorCreate,
                        "public CUDA memory information failed",
                        error,
                    )
                })?,
            )
            .map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "public CUDA allocator failed",
                    error,
                )
            })?,
        );
        pending.key = Some(
            Tensor::new(
                pending
                    .allocator
                    .as_ref()
                    .expect("pending allocator installed"),
                [1, 2, tokens, 64],
            )
            .map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "public CUDA key allocation failed",
                    error,
                )
            })?,
        );
        pending.value = Some(
            Tensor::new(
                pending
                    .allocator
                    .as_ref()
                    .expect("pending allocator installed"),
                [1, 2, tokens, 64],
            )
            .map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "public CUDA value allocation failed",
                    error,
                )
            })?,
        );
        pending.mask = Some(
            Tensor::new(&Allocator::default(), [1, tokens]).map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "public CPU mask allocation failed",
                    error,
                )
            })?,
        );
        pending.binding = Some(session.create_binding().map_err(|error| {
            native(
                CauseCode::TensorCreate,
                "public binding allocation failed",
                error,
            )
        })?);
        let binding = pending.binding.as_mut().expect("pending binding installed");
        let bind = (|| -> ort::Result<()> {
            binding.bind_output(
                "memory_key",
                alias(pending.key.as_ref().expect("pending key installed"))?,
            )?;
            binding.bind_output(
                "memory_value",
                alias(pending.value.as_ref().expect("pending value installed"))?,
            )?;
            binding.bind_output(
                "memory_mask",
                alias(pending.mask.as_ref().expect("pending mask installed"))?,
            )?;
            let inputs = pending.inputs.as_ref().expect("subset inputs installed");
            binding.bind_input("board", &inputs.board)?;
            binding.bind_input("metadata", &inputs.metadata)?;
            binding.bind_input("records", &inputs.records)?;
            binding.bind_input("record_mask", &inputs.mask)?;
            binding.synchronize_inputs()
        })();
        bind.map_err(|error| {
            native(
                CauseCode::OrtRun,
                "public subset binding/copy/input sync failed",
                error,
            )
        })?;
        bump(&mut self.stats.public_subset_runs_attempted)?;
        stats.public_nn_runs_attempted = stats.public_nn_runs_attempted.saturating_add(1);
        startup_enter(trace, PalsStartupBackendStage::PublicRun);
        let ran = run_fixed_binding(session, binding).and_then(|()| binding.synchronize_outputs());
        startup_return(trace, PalsStartupBackendStage::PublicRun, ran.is_ok());
        ran.map_err(|error| {
            native(
                CauseCode::OrtRun,
                "public subset Run/output sync failed",
                error,
            )
        })?;
        active.phase = Phase::PublicFenced;
        bump(&mut self.stats.public_subset_runs_completed)?;
        stats.public_nn_runs_completed = stats.public_nn_runs_completed.saturating_add(1);
        stats.completed_nn_inputs = stats.completed_nn_inputs.saturating_add(1);
        if !cuda_tensor(
            pending.key.as_ref().expect("pending key installed"),
            tokens,
            device,
        ) || !cuda_tensor(
            pending.value.as_ref().expect("pending value installed"),
            tokens,
            device,
        ) || !cpu_tensor(pending.mask.as_ref().expect("pending mask installed"))
        {
            return Err(fail(
                K::NumericalFailure,
                S::Output,
                "completed public subset output type/shape/device differs",
            ));
        }
        let (shape, mask) = pending
            .mask
            .as_ref()
            .expect("pending mask installed")
            .try_extract_tensor::<bool>()
            .map_err(|error| {
                native(
                    CauseCode::PolicyExtract,
                    "cannot read completed public subset mask",
                    error,
                )
            })?;
        let expected =
            std::iter::repeat_n(true, BOARD).chain(pending.descriptor.record_mask.iter().copied());
        if shape.as_ref() != [1, tokens as i64]
            || mask.len() != tokens
            || !mask.iter().copied().eq(expected)
        {
            return Err(fail(
                K::NumericalFailure,
                S::Output,
                "completed public subset exact mask differs",
            ));
        }
        // Public completion proves mask/output ownership, not finite K/V.
        binding.clear_inputs();
        pending.inputs = None;
        Ok(())
    }

    fn run_packing(&mut self) -> NativeResult<()> {
        let device = device_number(self.registry.namespace.domain)?;
        let active = self.active.as_mut().expect("resident invocation installed");
        let session = self
            .packing_session
            .as_mut()
            .ok_or_else(|| refused("packing session absent"))?;
        let count = self
            .initial_count
            .as_ref()
            .ok_or_else(|| refused("packing initialized count absent"))?;
        if count.graph_sha256 != self.graph.registration().graph.sha256
            || count.count != PACKING_NODES
            || active.plan.namespace != self.registry.namespace
            || active.plan.instance_seal != self.registry.instance_seal
        {
            return Err(refused("packing Session/plan/body origin differs"));
        }
        let tokens = active.plan.memory_tokens();
        active.phase = Phase::PackingStarted;
        let packing = &mut active.packing;
        packing
            .offsets
            .try_reserve_exact(DEVICE_PAGE_RECORD_CAPACITY)
            .map_err(|_| arithmetic("packing control backing allocation failed"))?;
        if packing.offsets.capacity() > DEVICE_PAGE_RECORD_CAPACITY {
            return Err(arithmetic("packing control container exceeds bound"));
        }
        for port in active.ports {
            let offset = match port.source {
                Source::Hit(location) => location.token_offset,
                Source::Pending(offset) => offset,
            };
            packing
                .offsets
                .push(
                    Tensor::from_array(([1], vec![i64::from(offset)])).map_err(|error| {
                        native(
                            CauseCode::TensorCreate,
                            "packing offset Tensor failed",
                            error,
                        )
                    })?,
                );
        }
        packing.stop = Some(
            Tensor::from_array(([1], vec![tokens as i64])).map_err(|error| {
                native(CauseCode::TensorCreate, "packing stop Tensor failed", error)
            })?,
        );
        packing.mask = Some(
            Tensor::from_array(([1, tokens], active.plan.current_mask.clone())).map_err(
                |error| {
                    native(
                        CauseCode::TensorCreate,
                        "packing current mask Tensor failed",
                        error,
                    )
                },
            )?,
        );
        packing.finite = Some(
            Tensor::new(&Allocator::default(), Vec::<usize>::new()).map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "packing finite scalar allocation failed",
                    error,
                )
            })?,
        );
        packing.allocator = Some(
            Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    device,
                    AllocatorType::Device,
                    MemoryType::Default,
                )
                .map_err(|error| {
                    native(
                        CauseCode::TensorCreate,
                        "packing CUDA memory information failed",
                        error,
                    )
                })?,
            )
            .map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "packing CUDA allocator failed",
                    error,
                )
            })?,
        );
        packing.key = Some(
            Tensor::new(
                packing
                    .allocator
                    .as_ref()
                    .expect("packing allocator installed"),
                [1, 2, tokens, 64],
            )
            .map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "packing CUDA key allocation failed",
                    error,
                )
            })?,
        );
        packing.value = Some(
            Tensor::new(
                packing
                    .allocator
                    .as_ref()
                    .expect("packing allocator installed"),
                [1, 2, tokens, 64],
            )
            .map_err(|error| {
                native(
                    CauseCode::TensorCreate,
                    "packing CUDA value allocation failed",
                    error,
                )
            })?,
        );
        packing.binding = Some(session.create_binding().map_err(|error| {
            native(
                CauseCode::TensorCreate,
                "packing binding allocation failed",
                error,
            )
        })?);
        let binding = packing.binding.as_mut().expect("packing binding installed");
        let bound = (|| -> NativeResult<()> {
            binding
                .bind_output(
                    "memory_key",
                    alias(packing.key.as_ref().expect("joined key installed")).map_err(
                        |error| native(CauseCode::TensorCreate, "joined key alias failed", error),
                    )?,
                )
                .map_err(|error| {
                    native(
                        CauseCode::OrtRun,
                        "packing key output binding failed",
                        error,
                    )
                })?;
            binding
                .bind_output(
                    "memory_value",
                    alias(packing.value.as_ref().expect("joined value installed")).map_err(
                        |error| native(CauseCode::TensorCreate, "joined value alias failed", error),
                    )?,
                )
                .map_err(|error| {
                    native(
                        CauseCode::OrtRun,
                        "packing value output binding failed",
                        error,
                    )
                })?;
            binding
                .bind_output(
                    "finite_output",
                    alias(packing.finite.as_ref().expect("finite scalar installed")).map_err(
                        |error| {
                            native(CauseCode::TensorCreate, "finite scalar alias failed", error)
                        },
                    )?,
                )
                .map_err(|error| {
                    native(
                        CauseCode::OrtRun,
                        "packing finite output binding failed",
                        error,
                    )
                })?;
            let (key, value) = resolve_pair(
                &active.plan,
                active.pending.as_ref(),
                active.base,
                self.registry.namespace.domain,
            )?;
            binding.bind_input("board_key", key).map_err(|error| {
                native(CauseCode::OrtRun, "packing board key input failed", error)
            })?;
            binding.bind_input("board_value", value).map_err(|error| {
                native(CauseCode::OrtRun, "packing board value input failed", error)
            })?;
            for (index, port) in active.ports.iter().copied().enumerate() {
                let (key, value) = resolve_pair(
                    &active.plan,
                    active.pending.as_ref(),
                    port,
                    self.registry.namespace.domain,
                )?;
                for (offset, tensor) in [(0, key), (1, value)] {
                    let expected = self
                        .graph
                        .input_descriptor(2 + index * 3 + offset)
                        .map_err(body_error)?;
                    binding
                        .bind_input(expected.name.as_str().map_err(body_error)?, tensor)
                        .map_err(|error| {
                            native(CauseCode::OrtRun, "packing record data input failed", error)
                        })?;
                }
                let expected = self
                    .graph
                    .input_descriptor(2 + index * 3 + 2)
                    .map_err(body_error)?;
                binding
                    .bind_input(
                        expected.name.as_str().map_err(body_error)?,
                        &packing.offsets[index],
                    )
                    .map_err(|error| {
                        native(
                            CauseCode::OrtRun,
                            "packing record offset input failed",
                            error,
                        )
                    })?;
            }
            binding
                .bind_input(
                    "record_stop",
                    packing.stop.as_ref().expect("stop control installed"),
                )
                .map_err(|error| native(CauseCode::OrtRun, "packing stop input failed", error))?;
            binding.synchronize_inputs().map_err(|error| {
                native(
                    CauseCode::OrtRun,
                    "packing input synchronization failed",
                    error,
                )
            })
        })();
        bound?;
        bump(&mut self.stats.packing_runs_attempted)?;
        // Weight-free routing is not a learned NN input or a visit/CPU task.
        run_fixed_binding(session, binding)
            .and_then(|()| binding.synchronize_outputs())
            .map_err(|error| {
                native(
                    CauseCode::OrtRun,
                    "packing Run/output synchronization failed",
                    error,
                )
            })?;
        active.phase = Phase::PackingFenced;
        bump(&mut self.stats.packing_runs_completed)?;
        if !cuda_tensor(
            packing.key.as_ref().expect("joined key installed"),
            tokens,
            device,
        ) || !cuda_tensor(
            packing.value.as_ref().expect("joined value installed"),
            tokens,
            device,
        ) || !cpu_tensor(packing.finite.as_ref().expect("finite scalar installed"))
            || !cpu_tensor(packing.mask.as_ref().expect("current mask installed"))
        {
            return Err(fail(
                K::NumericalFailure,
                S::Output,
                "completed packing output shape/type/device differs",
            ));
        }
        let (shape, finite) = packing
            .finite
            .as_ref()
            .expect("finite scalar installed")
            .try_extract_tensor::<bool>()
            .map_err(|error| {
                native(
                    CauseCode::PolicyExtract,
                    "cannot read completed packing finite scalar",
                    error,
                )
            })?;
        if !shape.is_empty() || finite != [true] {
            return Err(fail(
                K::NumericalFailure,
                S::Output,
                "completed packing K/V is non-finite or finite scalar differs",
            ));
        }
        let (shape, mask) = packing
            .mask
            .as_ref()
            .expect("current mask installed")
            .try_extract_tensor::<bool>()
            .map_err(|error| {
                native(
                    CauseCode::PolicyExtract,
                    "cannot read current original mask",
                    error,
                )
            })?;
        if shape.as_ref() != [1, tokens as i64] || mask != active.plan.current_mask.as_slice() {
            return Err(fail(
                K::NumericalFailure,
                S::Output,
                "completed packing current mask differs",
            ));
        }
        // This is the sole constructor of native completion/slice authority.
        active.completion = Some(PackingCompletedSlices {
            instance_seal: active.plan.instance_seal,
            namespace: active.plan.namespace,
            full_input_key: active.plan.full_input_key,
            packing_graph: self.graph.registration().graph.sha256,
            pending_identity: active.pending.as_ref().map(|block| {
                (
                    block.descriptor.block_key,
                    block.descriptor.block_generation,
                )
            }),
            board: active.certify_board,
            records: active.certify_records,
            publication_finished: false,
        });
        binding.clear_inputs();
        Ok(())
    }

    fn publish_selected_slices(&mut self) -> NativeResult<()> {
        let active = self.active.as_mut().expect("resident invocation installed");
        let proof = active
            .completion
            .as_mut()
            .ok_or_else(|| refused("native slice publication lacks actual packing completion"))?;
        if active.phase != Phase::PackingFenced
            || proof.publication_finished
            || proof.instance_seal != self.registry.instance_seal
            || proof.namespace != self.registry.namespace
            || proof.full_input_key != active.plan.full_input_key
            || proof.packing_graph != self.graph.registration().graph.sha256
        {
            return Err(refused("native slice completion origin/stage differs"));
        }
        if let Some(mut pending) = active.pending.take() {
            if proof.pending_identity
                != Some((
                    pending.descriptor.block_key,
                    pending.descriptor.block_generation,
                ))
                || (!proof.board && !proof.records.iter().any(|value| *value))
                || proof
                    .records
                    .iter()
                    .enumerate()
                    .any(|(index, value)| *value && index >= pending.descriptor.features.len())
            {
                return Err(refused(
                    "native slice certification differs from selected pending producer",
                ));
            }
            // Both public and packing are fenced. Remove the Send-only binding
            // before placing immutable Tensor handles in the Arc-backed bank.
            pending.binding = None;
            pending.inputs = None;
            pending.mask = None;
            let backing = CudaPublicBacking {
                domain: self.registry.namespace.domain,
                tokens: pending.descriptor.tokens(),
                key: pending.key.take().expect("fenced public key"),
                value: pending.value.take().expect("fenced public value"),
                _allocator: Mutex::new(pending.allocator.take().expect("fenced public allocator")),
            };
            let block = DevicePublicBlock {
                descriptor: pending.descriptor,
                backing,
                certified_board: proof.board,
                certified_records: proof.records,
            };
            active.published_pin = Some(self.registry.publish(block).map_err(page_error)?);
            bump(&mut self.stats.published_blocks)?;
            if proof.board {
                bump(&mut self.stats.certified_board_slices)?;
            }
            for record in proof.records {
                if record {
                    bump(&mut self.stats.certified_record_slices)?;
                }
            }
        } else if proof.pending_identity.is_some() {
            return Err(refused("native completion lost its pending producer"));
        }
        proof.publication_finished = true;
        Ok(())
    }
}

fn resolve_pair<'a>(
    plan: &'a DevicePagePlan<CudaPublicBacking>,
    pending: Option<&'a PendingCudaPublicBlock>,
    port: Port,
    domain: DevicePageDomain,
) -> NativeResult<(&'a Tensor<f32>, &'a Tensor<f32>)> {
    let device = device_number(domain)?;
    let (key, value, tokens, offset) = match port.source {
        Source::Hit(location) => {
            if location.instance_seal != plan.instance_seal {
                return Err(refused("resident hit registry seal differs"));
            }
            let owner = plan
                .pins
                .iter()
                .find(|owner| {
                    owner.descriptor.block_key == location.block_key
                        && owner.descriptor.block_generation == location.block_generation
                })
                .ok_or_else(|| refused("resident hit has no actual whole-owner pin"))?;
            if owner.descriptor.namespace != plan.namespace
                || owner.backing.domain != domain
                || (port.certification_required
                    && !owner.accepts_offset(usize::from(location.token_offset)))
            {
                return Err(refused(
                    "resident hit namespace/slice certification differs",
                ));
            }
            (
                &owner.backing.key,
                &owner.backing.value,
                owner.backing.tokens,
                usize::from(location.token_offset),
            )
        }
        Source::Pending(offset) => {
            if port.certification_required {
                return Err(refused("pending slice cannot claim existing certification"));
            }
            let owner = pending.ok_or_else(|| refused("resident pending port lost its owner"))?;
            if owner.descriptor.namespace != plan.namespace {
                return Err(refused("resident pending namespace differs"));
            }
            (
                owner
                    .key
                    .as_ref()
                    .ok_or_else(|| refused("pending key not installed"))?,
                owner
                    .value
                    .as_ref()
                    .ok_or_else(|| refused("pending value not installed"))?,
                owner.descriptor.tokens(),
                usize::from(offset),
            )
        }
    };
    if offset >= tokens
        || (offset == 0 && BOARD > tokens)
        || !cuda_tensor(key, tokens, device)
        || !cuda_tensor(value, tokens, device)
    {
        return Err(refused(
            "resident port lacks in-bounds exact CUDA whole backing",
        ));
    }
    Ok((key, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    // These are pure observation/parser/bit-identity fixtures. They construct no
    // CUDA Tensor/Session, initialized capability, finite proof or certified bank.
    #[test]
    fn all_cuda_summary_is_count_only_and_rejects_ambiguity() {
        let valid = vec![
            "Node placements".into(),
            "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 275".into(),
        ];
        assert_eq!(count_summary_lines(&valid).expect("count observed"), 275);
        for invalid in [
            vec![
                "unknown placement text".into(),
                "Node placements".into(),
                "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 275".into(),
            ],
            vec!["All nodes placed on [CUDAExecutionProvider]. Number of nodes: 275".into()],
            vec!["Node placements".into()],
            vec![
                "Node placements".into(),
                "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 274".into(),
            ],
            vec![
                "Node placements".into(),
                "All nodes placed on [CPUExecutionProvider]. Number of nodes: 275".into(),
            ],
            vec![
                "Node placements".into(),
                "Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 275".into(),
            ],
            vec![
                "Node placements".into(),
                "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 275".into(),
                "MemcpyFromHost (Memcpy)".into(),
            ],
            vec![
                "Node placements".into(),
                "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 275".into(),
                "Node placements".into(),
            ],
        ] {
            assert!(count_summary_lines(&invalid).is_err());
        }
        let mut repeated = valid.clone();
        repeated.push(valid[1].clone());
        assert!(count_summary_lines(&repeated).is_err());
    }

    #[test]
    fn logical_phase_never_substitutes_for_physical_sync() {
        for phase in [
            Phase::PublicStarted,
            Phase::PackingStarted,
            Phase::PrivateStarted,
        ] {
            assert!(phase.unknown());
        }
        for phase in [
            Phase::Prepared,
            Phase::PublicFenced,
            Phase::PackingFenced,
            Phase::PrivateFenced,
        ] {
            assert!(!phase.unknown());
        }
        // Phase::PackingFenced alone cannot be a PackingCompletedSlices token.
        // The token constructor remains only in actual run_packing after finite.
    }

    #[test]
    fn projection_input_identity_preserves_float_bits() {
        let zero = [0.0; 16];
        let mut negative_zero = zero;
        negative_zero[3] = -0.0;
        assert!(!feature_equal(&zero, &negative_zero));
        let mut changed = zero;
        changed[7] = 1.0;
        assert!(!feature_equal(&zero, &changed));
        assert!(feature_equal(&changed, &changed));
    }

    #[test]
    fn original_input_reservation_counts_retained_capacity_not_length() {
        let input = PalsModelInput {
            role: PalsRole::Proposer,
            board: vec![0; 64],
            metadata: [0.; 16],
            records: vec![],
            required_critical_records: vec![],
            candidates: vec![],
            divergence_features: vec![],
            query: [0.; 16],
            situation_revision: 1,
            history_digest: [5; 32],
            model_epoch: [2; 32],
        };
        let mut prepared = input
            .prepare_tensors(&PalsModelConfig::default())
            .expect("owned Rust inputs");
        let before = original_active_host_bytes(&prepared).expect("known capacities");
        let board_capacity = prepared.board.capacity();
        let board_length = prepared.board.len();
        prepared
            .board
            .try_reserve_exact(512)
            .expect("bounded fixture reserve");
        assert_eq!(prepared.board.len(), board_length);
        assert!(prepared.board.capacity() > board_capacity);
        let after = original_active_host_bytes(&prepared).expect("larger backing counted");
        assert_eq!(
            after - before,
            ((prepared.board.capacity() - board_capacity) * 8) as u64
        );
        assert!(mul(usize::MAX, 8).is_err());
        // No Tensor/Session is created and no native completion is certified.
    }
}
