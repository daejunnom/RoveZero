use rz_contracts::ProcessEpoch;
use rz_uci::{
    bootstrap::CpuMockFactory,
    engine::{self, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock},
    forward_lines,
};
use std::{io, sync::Arc, thread, time::Duration};

fn main() {
    if let Err(error) = run() {
        eprintln!("RoveZero CPU mock: {error}");
        std::process::exit(2);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut explicit_mock = false;
    let mut delay = Duration::ZERO;
    for argument in std::env::args().skip(1) {
        if argument == "--cpu-mock" {
            explicit_mock = true;
        } else if let Some(value) = argument.strip_prefix("--mock-delay-ms=") {
            delay = Duration::from_millis(value.parse()?);
        } else {
            return Err(format!(
                "unsupported argument: {argument}; use --cpu-mock [--mock-delay-ms=0..1000]"
            )
            .into());
        }
    }
    if !explicit_mock {
        return Err("this executable requires --cpu-mock; no neural backend is enabled".into());
    }
    let owners = Arc::new(OwnerRegistry::default());
    let factory = Arc::new(CpuMockFactory::new(&owners, delay)?);
    let clock = ProcessClock::new(ProcessEpoch(1)); // Issued once for this executable process.
    let process = EngineProcess::new(factory, owners, clock);
    let settings = EngineSettings::default();
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
    )?;
    Ok(())
}
