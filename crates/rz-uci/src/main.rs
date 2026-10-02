use rz_contracts::ProcessEpoch;
#[cfg(feature = "onnx-cpu")]
use rz_uci::{
    EngineIdentity,
    native_bootstrap::{NativeConfig, NativeCpuFactory},
};
use rz_uci::{
    bootstrap::CpuMockFactory,
    engine::{self, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock},
    forward_lines,
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
    if arguments.iter().any(|argument| argument == "--onnx-cpu") {
        return run_native(arguments);
    }
    run_mock(arguments)
}

fn run_mock(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut explicit_mock = false;
    let mut delay = Duration::ZERO;
    for argument in arguments {
        if argument == "--cpu-mock" {
            explicit_mock = true;
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
    let factory = Arc::new(CpuMockFactory::new(&owners, delay)?);
    let process = EngineProcess::new(factory, owners, clock);
    serve_process(process, EngineSettings::default())?;
    Ok(())
}

#[cfg(not(feature = "onnx-cpu"))]
fn run_native(_arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Do not inspect or echo native asset arguments when this provider is absent.
    Err(
        "unsupported feature: --onnx-cpu requires build feature onnx-cpu; no provider was started"
            .into(),
    )
}

#[cfg(feature = "onnx-cpu")]
fn run_native(arguments: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    // NativeConfig owns exact flag/asset/hash validation and bounded public errors.
    let config = NativeConfig::parse(arguments)?;
    let owners = Arc::new(OwnerRegistry::default());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let factory = Arc::new(NativeCpuFactory::load(&owners, &config)?);
    let process =
        EngineProcess::new(factory.clone(), owners, clock).with_identity(EngineIdentity {
            name: "RoveZero Maia ONNX CPU integration".into(),
            author: "RoveZero contributors".into(),
        });
    let settings = EngineSettings {
        max_workers: 1,
        ..EngineSettings::default()
    };
    // Retain the native owner outside EngineProcess. Report/drain acceptance runs
    // for both successful protocol service and EngineError; neither implies GPU support.
    let served = serve_process(process, settings);
    let report = factory.finish(served)?;
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
