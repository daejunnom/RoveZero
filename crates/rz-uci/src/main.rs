use rz_contracts::ProcessEpoch;
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
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|argument| argument == "--onnx-cpu" || argument == "--onnx-cuda")
    {
        return run_native(arguments);
    }
    run_mock(arguments)
}

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
