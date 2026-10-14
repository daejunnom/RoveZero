//! Explicit, non-training first-initialization G0 numerical-only bootstrap.
use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 5
        || arguments[0] != "--execute"
        || arguments[1] != "--registration"
        || arguments[3] != "--registration-sha256"
    {
        eprintln!(
            "usage: rz-pals-g0-bootstrap-numerical --execute --registration /absolute/g0-launch.json --registration-sha256 <sha256>; the default invocation executes no initialization, NN or training"
        );
        return ExitCode::FAILURE;
    }
    let path = std::path::Path::new(&arguments[2]);
    if !path.is_absolute()
        || arguments[4].len() != 64
        || !arguments[4].bytes().all(|b| b.is_ascii_hexdigit())
    {
        eprintln!("G0 numerical requires an absolute registration path and hexadecimal SHA-256");
        return ExitCode::FAILURE;
    }
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(target_os = "linux")]
    let mut hooks = Vec::new();
    #[cfg(target_os = "linux")]
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        match signal_hook::flag::register(signal, std::sync::Arc::clone(&cancel)) {
            Ok(id) => hooks.push(id),
            Err(error) => {
                for id in hooks {
                    signal_hook::low_level::unregister(id);
                }
                eprintln!("G0 cancellation registration failed: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    // On Windows and other unsupported platforms this returns the library's
    // typed refusal before reading inputs or starting any child. Unix signal
    // hooks are compiled only for the actual Linux execution path.
    let result = rz_arena::pals_collect_v3::run_g0_bootstrap_numerical_original_v5(
        path,
        &arguments[4],
        cancel,
    );
    #[cfg(target_os = "linux")]
    for id in hooks {
        signal_hook::low_level::unregister(id);
    }
    match result {
        Ok(owner) => {
            // Keep the original process and descriptor owner alive through
            // serialization. This JSON cannot reconstruct it for another call.
            println!("{}", owner.parent_receipt());
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("G0 original numerical halted: {}", failure.error);
            println!(
                "{}",
                serde_json::json!({"schema":"rz-pals-original-g0-bootstrap-numerical-failure/2",
                "error":failure.error.to_string(),"registration":failure.registration_pin,
                "charged_read_bytes":failure.charged_read_bytes,"read_usage_uncertain":failure.read_usage_uncertain,
                "preservation_reports":failure.preservation_reports,"automatic_retry":false,
                "numerical_accepted":false,"numerical_only":true,"startup_admission_not_issued":true,"training_authority_issued":false,"arena_accepted":false})
            );
            // The opaque failed owner is retained until this process exits;
            // its JSON is never accepted as an original-process continuation.
            ExitCode::FAILURE
        }
    }
}
