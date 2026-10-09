//! V4 opening/lock/snapshot preparation and an explicit finite execute command.
//! All arguments naming files/roots are absolute. No automatic retry or cloud use.
use rz_arena::pals_launch::v4::PalsArenaLaunchV4;
use std::{
    error::Error,
    fs,
    io::{Read, Write},
    path::Path,
};

fn read(path: &Path) -> Result<String, Box<dyn Error>> {
    for part in path.components() {
        let name = part.as_os_str().to_string_lossy().to_ascii_lowercase();
        if name.starts_with(".env")
            || name.ends_with(".key")
            || name.ends_with(".pem")
            || name.starts_with("id_rsa")
            || name.starts_with("id_ed25519")
            || name.contains("credential")
            || name.contains("service-account")
            || name.contains("service_account")
            || name.contains("api_key")
        {
            return Err("secret paths cannot be launch JSON inputs".into());
        }
    }
    if !path.is_absolute()
        || fs::symlink_metadata(path)?.file_type().is_symlink()
        || !fs::metadata(path)?.is_file()
    {
        return Err("absolute regular non-symlink JSON required".into());
    }
    let mut file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(256 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 256 * 1024 {
        return Err("launch JSON exceeds 256KiB".into());
    }
    Ok(String::from_utf8(bytes)?)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("absolute output path required".into());
    }
    let parent = path
        .parent()
        .ok_or("output parent required")?
        .canonicalize()?;
    for ancestor in parent.ancestors() {
        if fs::symlink_metadata(ancestor.join(".git")).is_ok() {
            return Err("artifacts must stay outside Git".into());
        }
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("lock") if args.len()==3=>{
            let input=PalsArenaLaunchV4::from_json(&read(Path::new(&args[1]))?)?;
            let lock=input.lock()?;write_new(Path::new(&args[2]),lock.to_json()?.as_bytes())?;
            println!("pals_arena_sha256={} metadata_locked=true engines_started=false nn_ready=false training_executed=false",lock.sha256());Ok(())
        },
        Some("opening") if args.len()==3=>{
            use rz_arena::NativeLaunchDeclaration;
            let lock=PalsArenaLaunchV4::from_json(&read(Path::new(&args[1]))?)?.lock()?;
            let view=lock.view();let pgn=rz_arena::opening_pgn_for_spec(view.opening,view.max_plies)?;
            write_new(Path::new(&args[2]),pgn.as_bytes())?;
            println!("opening_written=true engines_started=false");Ok(())
        },
        Some("prepare"|"execute") if args.len()==5=>prepare_or_execute(&args[1..],args[0]=="execute"),
        _=>Err("usage: pals_pair_prepare_v4 opening LAUNCH_JSON NEW_PGN | lock LAUNCH_JSON NEW_LOCK | prepare LOCK_JSON ASSET_ROOT OUTPUT_ROOT UNIQUE_LABEL | execute LOCK_JSON ASSET_ROOT OUTPUT_ROOT UNIQUE_LABEL".into()),
    }
}
#[cfg(not(target_os = "linux"))]
fn prepare_or_execute(_args: &[String], _execute: bool) -> Result<(), Box<dyn Error>> {
    Err("PALS native arena preparation/execution requires Linux; no engine started".into())
}
#[cfg(target_os = "linux")]
fn prepare_or_execute(args: &[String], execute: bool) -> Result<(), Box<dyn Error>> {
    use rz_arena::pals_launch::v4::*;
    use std::sync::{Arc, atomic::AtomicBool};
    let lock = LockedPalsArenaLaunchV4::from_json(&read(Path::new(&args[0]))?)?;
    for root in [&args[1], &args[2]] {
        if !Path::new(root).is_absolute() {
            return Err("absolute asset/output roots required".into());
        }
    }
    if execute {
        let affinity = verify_pals_inherited_resources_v4(&lock)?;
        eprintln!(
            "pals_inherited_affinity={affinity:?} cgroup_memory_limits=verified gpu_peak=unknown"
        );
    }
    let owner =
        prepare_pals_pair_launch_v4(&lock, Path::new(&args[1]), Path::new(&args[2]), &args[3])?;
    if !execute {
        println!(
            "snapshot_count={} input_sha256={} engines_started=false nn_ready=false training_executed=false",
            owner.snapshots().len(),
            lock.sha256()
        );
        return Ok(());
    }
    let cancel = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, Arc::clone(&cancel))?;
    }
    match run_pals_pair_observed_v4(owner, Some(&cancel)) {
        Ok(output) => {
            let cleanup_ms = output
                .native
                .receipt
                .process
                .cleanup_elapsed_ns
                .map(|ns| ns / 1_000_000 + u64::from(ns % 1_000_000 != 0));
            let assembly = assemble_pals_core_receipt_v4(&output, cleanup_ms);
            let artifact = save_pals_core_assembly_v4(&output, &assembly)?;
            println!(
                "receipt={} core_assembly={} integration_checks_passed={} scored_games={} observed_native_sessions={} core_available={} pilot_pair_eligible={} strength_eligible=false training_executed=false",
                output.native.receipt_artifact.path,
                artifact.path,
                output.native.receipt.integration_checks_passed,
                output.native.receipt.scored_games,
                output.native.receipt.provider_sessions.len(),
                assembly.core.is_some(),
                assembly
                    .core
                    .as_ref()
                    .is_some_and(|core| core.pair_eligible),
            );
            if let Some(error) = assembly.assembly_error {
                return Err(format!(
                    "Core assembly unavailable: {error}; raw arena receipt/PGN/work retained"
                )
                .into());
            }
            Ok(())
        }
        Err(failure) => {
            eprintln!(
                "PALS pair failed: {failure}; original evidence and unresolved ownership are retained"
            );
            Err(failure)
        }
    }
}
