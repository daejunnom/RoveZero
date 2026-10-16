//! One explicit original-window game. No implicit cloud or training operation.
use std::process::ExitCode;

#[cfg(target_os = "linux")]
struct RegisteredCancellationSignals {
    ids: Vec<signal_hook::SigId>,
}

#[cfg(target_os = "linux")]
impl RegisteredCancellationSignals {
    fn register(
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self, std::io::Error> {
        let mut registered = Self { ids: Vec::new() };
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            // A registration failure drops the already registered IDs too.
            registered.ids.push(signal_hook::flag::register(
                signal,
                std::sync::Arc::clone(&cancel),
            )?);
        }
        Ok(registered)
    }
}

#[cfg(target_os = "linux")]
impl Drop for RegisteredCancellationSignals {
    fn drop(&mut self) {
        for id in self.ids.drain(..) {
            signal_hook::low_level::unregister(id);
        }
    }
}

#[cfg(target_os = "linux")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use rz_arena::pals_collect_v3::{
        begin_generation_game_child_v3, generation_game_child_window_v3,
        run_generation_game_child_with_cancellation_v3,
    };
    use std::io::{Read, Write};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let _signals = RegisteredCancellationSignals::register(Arc::clone(&cancel))?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 || args[0] != "--generation-single-game-child-v3" {
        return Err(
            "expected explicit single-game mode and two original monotonic deadlines".into(),
        );
    }
    let execution_tick = args[1].parse::<u64>()?;
    let whole_tick = args[2].parse::<u64>()?;
    let (execution, whole) = generation_game_child_window_v3(execution_tick, whole_tick)?;
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .take(rz_arena::MAX_ORIGINAL_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    let authority = begin_generation_game_child_v3(bytes, execution, whole)?;
    let output = run_generation_game_child_with_cancellation_v3(
        authority,
        execution,
        whole,
        Arc::clone(&cancel),
    )?;
    let mut stdout = std::io::stdout().lock();
    if cancel.load(Ordering::Acquire) {
        return Err(rz_arena::ArenaError::Invalid(
            "original game cancellation latched before completed stdout publication".into(),
        )
        .into());
    }
    stdout.write_all(&output)?;
    stdout.flush()?;
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    Err("actual original-process generation worker requires Linux".into())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pals_generation_game_worker_failed: {error}");
            ExitCode::FAILURE
        }
    }
}
