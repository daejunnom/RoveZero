use rz_contracts::ProcessEpoch;
use rz_uci::search_driver::SearchSessionDriver;
#[cfg(feature = "onnx-cpu")]
use rz_uci::{
    EngineIdentity,
    engine::EvaluatorFactory,
    native_attestation::{ReceiptWriter, StartupReceiptV1, TerminationReceiptV1},
    native_bootstrap::{NativeConfig, NativeCpuFactory, NativeProvider},
    native_profile::{ProfileWriter, ProfiledWrite},
};
use rz_uci::{
    bootstrap::CpuMockFactory,
    engine::{self, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock},
    forward_lines,
};
#[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
use rz_uci::{
    native_bootstrap::NativeCudaFactory,
    native_cuda_attestation::{CudaReceiptWriter, CudaStartupReceiptV1, CudaTerminationReceiptV1},
};
use std::{io, sync::Arc, thread, time::Duration};

fn main() {
    if let Err(error) = run() {
        eprintln!("RoveZero: {error}");
        std::process::exit(2);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments: Vec<_> = std::env::args().skip(1).collect();
    let work_receipts = SearchWorkOptions::take(&mut arguments)?;
    let mut search = None;
    arguments.retain(|argument| {
        if let Some(value) = argument.strip_prefix("--search=") {
            if search.is_some() {
                search = Some("duplicate".to_owned());
            } else {
                search = Some(value.to_owned());
            }
            false
        } else {
            true
        }
    });
    match search.as_deref().unwrap_or("puct") {
        "puct" => {}
        "cpu" => return run_own_cpu(arguments, work_receipts),
        "pals" => return run_pals(arguments, work_receipts),
        _ => return Err("unsupported or duplicate --search; expected puct, cpu, or pals".into()),
    }
    if work_receipts.is_some() {
        return Err("search work receipts require the CPU or PALS driver; legacy PUCT retains its existing receipt path".into());
    }
    if arguments
        .iter()
        .any(|argument| argument == "--onnx-cpu" || argument == "--onnx-cuda")
    {
        return run_native(arguments);
    }
    run_mock(arguments)
}

fn run_pals(
    arguments: Vec<String>,
    work_receipts: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = None;
    let mut rounds = None;
    let mut cpu_nodes = None;
    let mut cpu_depth = None;
    let mut situations = None;
    let mut native = PalsNativeOptions::default();
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--pals-model=") {
            if model.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS model".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-rounds=") {
            if rounds.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate PALS round limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-cpu-nodes=") {
            if cpu_nodes.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate PALS CPU node limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-cpu-depth=") {
            if cpu_depth.replace(value.parse::<u16>()?).is_some() {
                return Err("duplicate PALS CPU depth".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-max-situations=") {
            if situations.replace(value.parse::<usize>()?).is_some() {
                return Err("duplicate PALS situation capacity".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-export-manifest=") {
            if native.manifest.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS export manifest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-export-sha256=") {
            if native.manifest_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS export hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-path=") {
            if native.runtime.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS runtime path".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-sha256=") {
            if native.runtime_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS runtime hash".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-runtime-cache-root=") {
            if native
                .runtime_cache_root
                .replace(value.to_owned())
                .is_some()
            {
                return Err("duplicate PALS runtime cache root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-output-root=") {
            if native.output_root.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS output root".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-launch-sha256=") {
            if native.launch_hash.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS launch digest".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-endpoint-id=") {
            if native.endpoint_id.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS endpoint ID".into());
            }
        } else if let Some(value) = argument.strip_prefix("--pals-provider=") {
            if native.provider.replace(value.to_owned()).is_some() {
                return Err("duplicate PALS provider".into());
            }
        } else {
            return Err("PALS requires an explicit model and finite PALS flags; LC0/ORT arguments are not implicitly reused".into());
        }
    }
    let max_cpu_nodes = cpu_nodes.unwrap_or(100_000);
    let rounds = rounds.unwrap_or(16);
    let cpu_depth = cpu_depth.unwrap_or(2);
    let mut config = rz_search::pals::engine::PalsConfig::default();
    config.max_nodes = situations.unwrap_or(4096);
    config.validate()?;
    if rounds == 0
        || rounds > 1_000_000
        || max_cpu_nodes == 0
        || cpu_depth == 0
        || cpu_depth > rz_search::cpu::CpuConfig::default().max_depth
    {
        return Err(
            "PALS round/CPU work limits must be finite and within the registered profile".into(),
        );
    }
    if model.as_deref() == Some("onnx") {
        if work_receipts.is_some() {
            return Err("native PALS uses pals-output-root/launch-sha256/endpoint-id for one combined native/search receipt".into());
        }
        return run_native_pals(native, config, rounds, max_cpu_nodes, cpu_depth);
    }
    if model.as_deref() != Some("legal-order-mock") {
        return Err("PALS requires explicit --pals-model=legal-order-mock or onnx; no model fallback was started".into());
    }
    if native.any() {
        return Err(
            "PALS mock selection cannot accept neural assets or a provider declaration".into(),
        );
    }
    let driver = Arc::new(rz_uci::search_driver::PalsSessionDriver::new(
        config,
        rz_search::pals::engine::LegalOrderRoleMock,
        rz_search::cpu::CpuConfig::default(),
        rounds,
        max_cpu_nodes,
        cpu_depth,
        rz_uci::EngineIdentity {
            name: "RoveZero PALS explicit legal-order CPU mock + own CPU_R".into(),
            author: "RoveZero contributors".into(),
        },
    )?);
    let mut settings = EngineSettings::default();
    settings.search.max_simulations = max_cpu_nodes;
    serve_search_process(driver, settings, work_receipts)?;
    Ok(())
}

#[derive(Default)]
struct PalsNativeOptions {
    manifest: Option<String>,
    manifest_hash: Option<String>,
    runtime: Option<String>,
    runtime_hash: Option<String>,
    runtime_cache_root: Option<String>,
    provider: Option<String>,
    output_root: Option<String>,
    launch_hash: Option<String>,
    endpoint_id: Option<String>,
}
impl PalsNativeOptions {
    fn any(&self) -> bool {
        self.manifest.is_some()
            || self.manifest_hash.is_some()
            || self.runtime.is_some()
            || self.runtime_hash.is_some()
            || self.runtime_cache_root.is_some()
            || self.provider.is_some()
            || self.output_root.is_some()
            || self.launch_hash.is_some()
            || self.endpoint_id.is_some()
    }
}

#[cfg(not(feature = "onnx-cpu"))]
fn run_native_pals(
    _native: PalsNativeOptions,
    _config: rz_search::pals::engine::PalsConfig,
    _rounds: u64,
    _cpu_nodes: u64,
    _cpu_depth: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "PALS ONNX startup requires feature onnx-cpu; no provider or model fallback was started"
            .into(),
    )
}

#[cfg(feature = "onnx-cpu")]
fn run_native_pals(
    native: PalsNativeOptions,
    config: rz_search::pals::engine::PalsConfig,
    rounds: u64,
    cpu_nodes: u64,
    cpu_depth: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    // PALS owns a distinct manifest and tensors. LC0 NativeConfig and its
    // attestation cannot authorize or describe this model.
    if native.provider.as_deref() != Some("cpu") {
        return Err("PALS requires explicit --pals-provider=cpu; CUDA bindings are not enabled by this path".into());
    }
    let receipt_config =
        match (native.output_root, native.launch_hash, native.endpoint_id) {
            (None, None, None) => None,
            (Some(root), Some(launch), Some(endpoint)) => {
                let root = std::path::PathBuf::from(root);
                if !root.is_absolute() {
                    return Err("PALS output root must be absolute".into());
                }
                rz_eval::asset::parse_sha256(&launch)?;
                Some((root, launch, endpoint))
            }
            _ => return Err(
                "PALS receipt output requires output root, launch digest and endpoint ID together"
                    .into(),
            ),
        };
    let manifest =
        std::path::PathBuf::from(native.manifest.ok_or("PALS export manifest is required")?);
    let runtime_path =
        std::path::PathBuf::from(native.runtime.ok_or("PALS runtime path is required")?);
    if !manifest.is_absolute() || !runtime_path.is_absolute() {
        return Err("PALS model/runtime paths must be absolute".into());
    }
    let manifest_hash = native
        .manifest_hash
        .ok_or("PALS export SHA-256 is required")?;
    let runtime_hash = native
        .runtime_hash
        .ok_or("PALS runtime SHA-256 is required")?;
    rz_eval::asset::parse_sha256(&manifest_hash)?;
    rz_eval::asset::parse_sha256(&runtime_hash)?;
    let cache = match native.runtime_cache_root {
        Some(path) => {
            let path = std::path::PathBuf::from(path);
            if !path.is_absolute() {
                return Err("PALS runtime cache root must be absolute".into());
            }
            rz_eval::runtime_pin::RuntimeCache::open(&path)?
        }
        None => rz_eval::runtime_pin::RuntimeCache::for_user()?,
    };
    let pin = cache.library(&runtime_path, &runtime_hash)?;
    let runtime = rz_eval::onnx::OrtRuntime::load(&pin)?;
    let backend = rz_eval::pals_onnx::PalsOnnxBackend::load(
        &manifest,
        &manifest_hash,
        runtime,
        rz_eval::pals_onnx::PalsOnnxConfig::cpu(),
    )?;
    let mut settings = EngineSettings::default();
    settings.search.max_simulations = cpu_nodes;
    let model = rz_uci::pals_native::NativeRoleModel::new_with_drain_limit(
        backend,
        settings.shutdown_limit,
    )?;
    let finish = model.finish_handle();
    let driver = match rz_uci::search_driver::PalsSessionDriver::new(
        config,
        model,
        rz_search::cpu::CpuConfig::default(),
        rounds,
        cpu_nodes,
        cpu_depth,
        rz_uci::EngineIdentity {
            name: "RoveZero PALS P/C ONNX CPU + own CPU_R".into(),
            author: "RoveZero contributors".into(),
        },
    ) {
        Ok(driver) => Arc::new(driver),
        Err(primary) => {
            let cleanup = finish
                .finish(std::time::Instant::now() + settings.shutdown_limit)
                .err();
            return Err(Box::new(PalsFinishError {
                primary: Some(Box::new(primary)),
                cleanup,
                publication: None,
                work_observation: None,
            }));
        }
    };
    let startup_work = match driver.work_receipt() {
        Ok(work) => work,
        Err(work_observation) => {
            let cleanup = finish
                .finish(std::time::Instant::now() + settings.shutdown_limit)
                .err();
            return Err(Box::new(PalsFinishError {
                primary: None,
                cleanup,
                publication: None,
                work_observation: Some(work_observation),
            }));
        }
    };
    let mut receipts = match receipt_config {
        Some((root, launch, endpoint)) => {
            let opened = rz_uci::pals_attestation::PalsReceiptWriter::open(
                &root,
                &endpoint,
                &launch,
                &runtime_hash,
            );
            match opened {
                Ok(mut writer) => match writer.startup(finish.receipt(), startup_work.clone()) {
                    Ok(()) => Some(writer),
                    Err(publication) => {
                        let cleanup = finish
                            .finish(std::time::Instant::now() + settings.shutdown_limit)
                            .err();
                        return Err(Box::new(PalsFinishError {
                            primary: None,
                            cleanup,
                            publication: Some(publication),
                            work_observation: None,
                        }));
                    }
                },
                Err(publication) => {
                    let cleanup = finish
                        .finish(std::time::Instant::now() + settings.shutdown_limit)
                        .err();
                    return Err(Box::new(PalsFinishError {
                        primary: None,
                        cleanup,
                        publication: Some(publication),
                        work_observation: None,
                    }));
                }
            }
        }
        None => None,
    };
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let process = EngineProcess::with_search_driver(driver.clone(), owners, clock);
    let served = serve_process(process, settings);
    let finished = finish.finish(std::time::Instant::now() + settings.shutdown_limit);
    let observed_work = driver.work_receipt();
    let publication = receipts.as_mut().and_then(|writer| {
        writer
            .termination(
                finish.receipt(),
                served.is_ok(),
                observed_work.as_ref().ok().cloned().flatten(),
            )
            .err()
    });
    if publication.is_some() || observed_work.is_err() {
        return Err(Box::new(PalsFinishError {
            primary: served
                .err()
                .map(|error| Box::new(error) as Box<dyn std::error::Error>),
            cleanup: finished.err(),
            publication,
            work_observation: observed_work.err(),
        }));
    }
    match (served, finished) {
        (Ok(()), Ok(receipt)) => {
            eprintln!(
                "PALS native role shutdown: completed={} delivered={} consumed={} canceled={} expired={} failed={} invalid={} in_flight={} joined={} buffers_released={} quarantined={}",
                receipt.completed_role_inputs,
                receipt.delivered_role_inputs,
                receipt.search_consumed_role_inputs,
                receipt.canceled_requests,
                receipt.expired_requests,
                receipt.failed_physical_role_calls,
                receipt.invalid_role_outputs,
                receipt.physical_runs_in_flight,
                receipt.physical_shutdown_confirmed,
                receipt.native_buffers_released,
                receipt.quarantined,
            );
            Ok(())
        }
        (Err(primary), Ok(_)) => Err(Box::new(primary)),
        (primary, Err(cleanup)) => Err(Box::new(PalsFinishError {
            primary: primary
                .err()
                .map(|error| Box::new(error) as Box<dyn std::error::Error>),
            cleanup: Some(cleanup),
            publication: None,
            work_observation: None,
        })),
    }
}

#[cfg(feature = "onnx-cpu")]
#[derive(Debug)]
struct PalsFinishError {
    primary: Option<Box<dyn std::error::Error>>,
    cleanup: Option<rz_search::pals::engine::RoleError>,
    publication: Option<rz_uci::process_receipts::ProcessReceiptError>,
    work_observation: Option<rz_contracts::ContractError>,
}
#[cfg(feature = "onnx-cpu")]
impl std::fmt::Display for PalsFinishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(primary) = &self.primary {
            write!(formatter, "{primary}; ")?;
        }
        if let Some(cleanup) = &self.cleanup {
            write!(formatter, "PALS physical shutdown failed: {cleanup}; ")?;
        }
        if let Some(publication) = &self.publication {
            write!(formatter, "PALS receipt publication failed: {publication}")?;
        }
        if let Some(observation) = &self.work_observation {
            write!(formatter, "PALS work observation failed: {observation}")?;
        }
        Ok(())
    }
}
#[cfg(feature = "onnx-cpu")]
impl std::error::Error for PalsFinishError {}

fn run_own_cpu(
    arguments: Vec<String>,
    work_receipts: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut max_depth = None;
    let mut max_nodes = None;
    let mut tt_entries = None;
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--cpu-max-depth=") {
            if max_depth.replace(value.parse::<u16>()?).is_some() {
                return Err("duplicate CPU depth limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--cpu-max-nodes=") {
            if max_nodes.replace(value.parse::<u64>()?).is_some() {
                return Err("duplicate CPU node limit".into());
            }
        } else if let Some(value) = argument.strip_prefix("--cpu-tt-entries=") {
            if tt_entries.replace(value.parse::<usize>()?).is_some() {
                return Err("duplicate CPU TT limit".into());
            }
        } else {
            return Err("CPU search only accepts --cpu-max-depth, --cpu-max-nodes and --cpu-tt-entries; neural provider/model flags are inapplicable".into());
        }
    }
    let max_nodes = max_nodes.unwrap_or(100_000);
    // Explicit finite limits: default profile is own bootstrap CPU_R, not NNUE.
    let mut config = rz_search::cpu::CpuConfig::default();
    config.max_depth = max_depth.unwrap_or(8);
    config.tt_entries = tt_entries.unwrap_or(65_536);
    let driver = Arc::new(rz_uci::search_driver::CpuSessionDriver::new(
        config, max_nodes,
    )?);
    let mut settings = EngineSettings::default();
    // Legacy field is the UCI work admission cap. CPU interprets go nodes as
    // actual CPU nodes; no completed simulation or NN receipt is fabricated.
    settings.search.max_simulations = max_nodes;
    serve_search_process(driver, settings, work_receipts)?;
    Ok(())
}

struct SearchWorkOptions {
    root: std::path::PathBuf,
    launch_sha256: String,
    endpoint_id: String,
}
impl SearchWorkOptions {
    fn take(arguments: &mut Vec<String>) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let mut root = None;
        let mut launch = None;
        let mut endpoint = None;
        let mut duplicate = false;
        arguments.retain(|argument| {
            let replaced = if let Some(value) = argument.strip_prefix("--search-work-output-root=")
            {
                root.replace(value.to_owned())
            } else if let Some(value) = argument.strip_prefix("--search-work-launch-sha256=") {
                launch.replace(value.to_owned())
            } else if let Some(value) = argument.strip_prefix("--search-work-endpoint-id=") {
                endpoint.replace(value.to_owned())
            } else {
                return true;
            };
            duplicate |= replaced.is_some();
            false
        });
        if duplicate {
            return Err("duplicate search work receipt argument".into());
        }
        match (root, launch, endpoint) {
            (None, None, None) => Ok(None),
            (Some(root), Some(launch_sha256), Some(endpoint_id)) => {
                let root = std::path::PathBuf::from(root);
                if !root.is_absolute() {
                    return Err("search work output root must be absolute".into());
                }
                rz_eval::asset::parse_sha256(&launch_sha256)?;
                #[cfg(not(feature = "search-work-receipts"))]
                {
                    let _ = (root, launch_sha256, endpoint_id);
                    Err("unsupported feature: search work receipts require feature search-work-receipts; no driver was started".into())
                }
                #[cfg(feature = "search-work-receipts")]
                {
                    Ok(Some(Self {
                        root,
                        launch_sha256,
                        endpoint_id,
                    }))
                }
            }
            _ => Err(
                "search work output requires root, launch digest and endpoint ID together".into(),
            ),
        }
    }
}

fn serve_search_process(
    driver: Arc<dyn SearchSessionDriver>,
    settings: EngineSettings,
    receipt: Option<SearchWorkOptions>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "search-work-receipts")]
    let mut receipts = match receipt {
        Some(receipt) => {
            let mut writer = rz_uci::pals_attestation::SearchWorkReceiptWriter::open(
                &receipt.root,
                &receipt.endpoint_id,
                &receipt.launch_sha256,
            )?;
            writer.startup(driver.work_receipt()?)?;
            Some(writer)
        }
        None => None,
    };
    #[cfg(not(feature = "search-work-receipts"))]
    if receipt.is_some() {
        return Err("unsupported feature: search-work-receipts".into());
    }
    let process = EngineProcess::with_search_driver(
        Arc::clone(&driver),
        Arc::new(OwnerRegistry::default()),
        ProcessClock::new(ProcessEpoch(1)),
    );
    let served = serve_process(process, settings);
    #[cfg(feature = "search-work-receipts")]
    {
        let observed = driver.work_receipt();
        let publication = receipts.as_mut().and_then(|writer| {
            writer
                .termination(observed.as_ref().ok().cloned().flatten(), served.is_ok())
                .err()
        });
        if observed.is_err() || publication.is_some() {
            return Err(Box::new(SearchWorkFinishError {
                primary: served.err(),
                observation: observed.err(),
                publication,
            }));
        }
    }
    served?;
    Ok(())
}
#[cfg(feature = "search-work-receipts")]
#[derive(Debug)]
struct SearchWorkFinishError {
    primary: Option<engine::EngineError>,
    observation: Option<rz_contracts::ContractError>,
    publication: Option<rz_uci::process_receipts::ProcessReceiptError>,
}
#[cfg(feature = "search-work-receipts")]
impl std::fmt::Display for SearchWorkFinishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
#[cfg(feature = "search-work-receipts")]
impl std::error::Error for SearchWorkFinishError {}

fn run_mock(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut explicit_mock = false;
    let mut model_adapter = None;
    let mut delay = Duration::ZERO;
    for argument in arguments {
        if argument == "--cpu-mock" {
            explicit_mock = true;
        } else if let Some(value) = argument.strip_prefix("--model-adapter=") {
            if model_adapter.replace(value.to_owned()).is_some() {
                return Err("duplicate model adapter".into());
            }
        } else if let Some(value) = argument.strip_prefix("--mock-delay-ms=") {
            delay = Duration::from_millis(value.parse()?);
        } else {
            return Err(
                "unsupported argument; supported flags: --cpu-mock [--mock-delay-ms=0..1000]"
                    .into(),
            );
        }
    }
    if !explicit_mock {
        return Err("this executable requires --cpu-mock; no neural backend is enabled".into());
    }
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1)); // Issued once for this executable process.
    match model_adapter.as_deref().unwrap_or("lc0") {
        "lc0" => {
            let factory = Arc::new(CpuMockFactory::new(&owners, delay)?);
            serve_process(
                EngineProcess::new(factory, owners, clock),
                EngineSettings::default(),
            )?;
        }
        "entity-candidate-mock" => {
            if !delay.is_zero() {
                return Err("entity mock does not support scripted delay".into());
            }
            let factory = Arc::new(rz_uci::bootstrap::EntityMockFactory::new(&owners)?);
            let process = EngineProcess::new(factory.clone(), owners, clock).with_identity(
                rz_uci::EngineIdentity {
                    name: "RoveZero entity candidate CPU mock".into(),
                    author: "RoveZero contributors".into(),
                },
            );
            let served = serve_process(process, EngineSettings::default());
            let finished = factory.finish(std::time::Instant::now() + Duration::from_secs(2));
            served?;
            finished?;
        }
        _ => return Err("unsupported model adapter".into()),
    }
    Ok(())
}

#[cfg(not(feature = "onnx-cpu"))]
fn run_native(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Do not inspect or echo native asset arguments when this provider is absent.
    if arguments.iter().any(|argument| argument == "--onnx-cuda") {
        return Err("unsupported feature: --onnx-cuda requires build feature onnx-cuda on Linux; no provider was started".into());
    }
    Err(
        "unsupported feature: --onnx-cpu requires build feature onnx-cpu; no provider was started"
            .into(),
    )
}

#[cfg(feature = "onnx-cpu")]
fn run_native(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // NativeConfig owns exact flag/asset/hash validation and bounded public errors.
    let config = NativeConfig::parse(arguments)?;
    match config.provider() {
        NativeProvider::Cpu => run_cpu(config),
        NativeProvider::Cuda => run_cuda(config),
    }
}
#[cfg(feature = "onnx-cpu")]
fn run_cpu(config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let factory = Arc::new(NativeCpuFactory::load(&owners, &config)?);
    let trace = factory.source_trace();
    let mut profile_writer = if config.profiling_requested() {
        Some(ProfileWriter::open(&config)?)
    } else {
        None
    };
    let mut receipts = if config.attestation_requested() {
        Some(ReceiptWriter::open(&config)?)
    } else {
        None
    };
    let startup = if let Some(writer) = receipts.as_mut() {
        let startup = StartupReceiptV1::capture(&factory, &clock)?;
        writer.startup(&startup)?;
        Some(startup)
    } else {
        None
    };
    let process =
        EngineProcess::new(factory.clone(), owners, clock).with_identity(EngineIdentity {
            name: "RoveZero LC0 weights ONNX CPU integration".into(),
            author: "RoveZero contributors".into(),
        });
    let settings = config.engine_settings();
    // Retain the native owner outside EngineProcess. Report/drain acceptance runs
    // for both successful protocol service and EngineError; neither implies GPU support.
    let served = serve_native_process(process, settings, trace.clone());
    let finished = factory.finish(served);
    if let (Some(writer), Some(startup)) = (receipts.as_mut(), startup.as_ref()) {
        let receipt = match TerminationReceiptV1::from_result(startup, &finished) {
            Ok(receipt) => receipt,
            Err(error) => return Err(Box::new(error.retaining(finished))),
        };
        if let Err(error) = writer.termination(&receipt) {
            // Publication must not replace an original service/drain failure.
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(trace)) = (profile_writer.as_mut(), trace.as_ref())
        && let Err(error) = writer.publish(trace, &finished)
    {
        return Err(Box::new(error.retaining(finished)));
    }
    let report = finished?;
    eprintln!("{report}");
    Ok(())
}

#[cfg(all(
    feature = "onnx-cpu",
    not(all(feature = "onnx-cuda", target_os = "linux"))
))]
fn run_cuda(_config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    Err("explicit CUDA native startup requires the onnx-cuda feature on Linux; no provider was started".into())
}
#[cfg(all(feature = "onnx-cuda", target_os = "linux"))]
fn run_cuda(config: NativeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let factory = Arc::new(NativeCudaFactory::load(&owners, &config)?);
    let trace = factory.source_trace();
    #[cfg(feature = "experimental-batch")]
    let mut batch_writer = if config.batch_attestation_requested() {
        Some(rz_uci::native_batch_attestation::BatchReceiptWriter::open(
            &config,
        )?)
    } else {
        None
    };
    #[cfg(feature = "experimental-batch")]
    let batch_startup = if let Some(writer) = batch_writer.as_mut() {
        Some(writer.startup(
            CudaStartupReceiptV1::capture(&factory, &clock)?,
            &config.engine_settings(),
        )?)
    } else {
        None
    };
    let mut profile_writer = if config.profiling_requested() {
        Some(ProfileWriter::open(&config)?)
    } else {
        None
    };
    let mut receipts = if config.attestation_requested() {
        Some(CudaReceiptWriter::open(&config)?)
    } else {
        None
    };
    let startup = if let Some(writer) = receipts.as_mut() {
        let startup = CudaStartupReceiptV1::capture(&factory, &clock)?;
        writer.startup(&startup)?;
        // Capture the exact settings subsequently served. Experimental paths
        // do not issue this fixed-profile attestation (factory guard above).
        let search = rz_uci::native_cuda_attestation::CudaSearchReceiptV1::capture(
            &startup,
            &config.engine_settings(),
        )?;
        writer.search_config(&search)?;
        Some(startup)
    } else {
        None
    };
    let process =
        EngineProcess::new(factory.clone(), owners, clock).with_identity(EngineIdentity {
            name: "RoveZero LC0 weights ONNX CUDA integration".into(),
            author: "RoveZero contributors".into(),
        });
    let served = serve_native_process(process, config.engine_settings(), trace.clone());
    let finished = factory.finish(served);
    #[cfg(feature = "experimental-batch")]
    if let (Some(writer), Some(startup)) = (batch_writer.as_mut(), batch_startup.as_ref()) {
        if let Err(error) = writer.termination(startup, &factory, &finished) {
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(startup)) = (receipts.as_mut(), startup.as_ref()) {
        let receipt = match CudaTerminationReceiptV1::from_result(startup, &finished) {
            Ok(receipt) => receipt,
            Err(error) => return Err(Box::new(error.retaining(finished))),
        };
        if let Err(error) = writer.termination(&receipt) {
            return Err(Box::new(error.retaining(finished)));
        }
    }
    if let (Some(writer), Some(trace)) = (profile_writer.as_mut(), trace.as_ref())
        && let Err(error) = writer.publish(trace, &finished)
    {
        return Err(Box::new(error.retaining(finished)));
    }
    let report = finished?;
    eprintln!("{report}");
    Ok(())
}

fn serve_process(
    process: EngineProcess,
    settings: EngineSettings,
) -> Result<(), engine::EngineError> {
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    // A blocked OS stdin read is owned by this independent producer. It cannot
    // delay stop/search/drain, and process exit terminates it after quit.
    thread::spawn(move || {
        let _ = forward_lines(
            &mut io::stdin().lock(),
            &input,
            settings.parser.max_line_bytes,
        );
    });
    engine::serve(
        events,
        sender,
        &mut io::stdout().lock(),
        &mut io::stderr().lock(),
        process,
        settings,
    )
}

#[cfg(feature = "onnx-cpu")]
fn serve_native_process(
    process: EngineProcess,
    settings: EngineSettings,
    trace: Option<rz_telemetry::source::SourceJournal<rz_contracts::CompletionContext>>,
) -> Result<(), engine::EngineError> {
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    thread::spawn(move || {
        let _ = forward_lines(
            &mut io::stdin().lock(),
            &input,
            settings.parser.max_line_bytes,
        );
    });
    engine::serve(
        events,
        sender,
        &mut ProfiledWrite::new(io::stdout().lock(), trace),
        &mut io::stderr().lock(),
        process,
        settings,
    )
}
