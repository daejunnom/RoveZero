//! Explicit standalone real-action collection. The normal generation owner
//! should pass retained original games across P/C freeze through the typed API.
//! This command runs additional source games only through four explicit flags.
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(status) => {
            println!("{status}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("collect-action-outcomes halted: {error}");
            ExitCode::FAILURE
        }
    }
}
fn run() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 8
        || arguments[0] != "--declaration"
        || arguments[2] != "--declaration-sha256"
        || arguments[4] != "--fresh-source-generation-declaration"
        || arguments[6] != "--fresh-source-generation-sha256"
    {
        return Err("usage: rz-pals-collect-action-outcomes --declaration /absolute/action-declaration.json --declaration-sha256 <sha256> --fresh-source-generation-declaration /absolute/source-declaration.json --fresh-source-generation-sha256 <sha256>; explicit fresh source runs 128 additional real games. Retained-source generation orchestration uses run_action_outcomes_owned_cli_v3 without extra games".into());
    }
    #[cfg(not(target_os = "linux"))]
    return Err("actual action collection requires Linux original process transport".into());
    #[cfg(target_os = "linux")]
    {
        use std::path::Path;
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
        let result = (|| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
            let source = rz_arena::pals_collect_v3::run_selfplay_collect_owned_cli_v3(
                Path::new(&arguments[5]),
                &arguments[7],
                &cancel,
            )?;
            let source_status = source.status().clone();
            let archives = source.into_archive().into_archives();
            let action = rz_arena::pals_collect_v3::run_action_outcomes_owned_cli_v3(
                Path::new(&arguments[1]),
                &arguments[3],
                archives,
                Arc::clone(&cancel),
            )?;
            Ok(serde_json::json!({
                "schema":"rz-pals-explicit-fresh-source-action-command/1",
                "additional_source_games_executed":128,
                "source_generation":source_status,"action_collection":action.status(),
                "optimizer_updates_executed":0,
            }))
        })();
        for id in hooks {
            signal_hook::low_level::unregister(id);
        }
        result
    }
}
