//! Explicit actual collection only; no implicit training or cloud work.
#[cfg(target_os = "linux")]
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(status) => {
            println!("{status}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("selfplay-collect halted: {error}");
            ExitCode::FAILURE
        }
    }
}
fn run() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 4
        || arguments[0] != "--declaration"
        || arguments[2] != "--declaration-sha256"
    {
        return Err("usage: rz-pals-selfplay-collect --declaration /absolute/prepared-declaration.json --declaration-sha256 <sha256>".into());
    }
    #[cfg(not(target_os = "linux"))]
    return Err("selfplay-collect original process transport requires Linux".into());
    #[cfg(target_os = "linux")]
    {
        use std::sync::{Arc, atomic::AtomicBool};
        let cancel = Arc::new(AtomicBool::new(false));
        let mut hooks = vec![];
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            match signal_hook::flag::register(signal, Arc::clone(&cancel)) {
                Ok(id) => hooks.push(id),
                Err(error) => {
                    for id in hooks {
                        signal_hook::low_level::unregister(id);
                    }
                    return Err(error.into());
                }
            }
        }
        let result = rz_arena::pals_collect_v3::run_selfplay_collect_cli_v3(
            Path::new(&arguments[1]),
            &arguments[3],
            &cancel,
        )
        .map_err(Into::into);
        for id in hooks {
            signal_hook::low_level::unregister(id);
        }
        result
    }
}
