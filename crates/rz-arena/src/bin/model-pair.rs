//! Explicit V2 lock/execute entry point; inherited Linux limits are checked.
use rz_experiments::{LockedManifestV2, RunManifestV2};
use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
};
fn read(path: &Path) -> Result<String, Box<dyn Error>> {
    if !path.is_absolute()
        || fs::symlink_metadata(path)?.file_type().is_symlink()
        || fs::metadata(path)?.len() > rz_experiments::MAX_MANIFEST_BYTES as u64
    {
        return Err("bounded absolute regular manifest required".into());
    }
    Ok(fs::read_to_string(path)?)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("lock") if args.len()==3=>{
            let lock=RunManifestV2::from_json(&read(Path::new(&args[1]))?)?.lock()?;
            let output=PathBuf::from(&args[2]);
            if !output.is_absolute(){return Err("absolute lock output required".into());}
            for parent in output.ancestors().skip(1) {if parent.join(".git").exists(){return Err("run locks stay outside Git".into());}}
            let mut file=fs::OpenOptions::new().write(true).create_new(true).open(output)?;
            file.write_all(lock.to_json()?.as_bytes())?;file.sync_all()?;
            println!("input_sha256={} execution_ready=false",lock.sha256());Ok(())
        },
        Some("execute") if args.len()==5=>execute(&args[1..]),
        _=>Err("usage: model-pair lock INPUT_JSON NEW_LOCK_JSON | execute LOCK_JSON ASSET_ROOT OUTPUT_ROOT UNIQUE_LABEL".into()),
    }
}
#[cfg(not(target_os = "linux"))]
fn execute(_args: &[String]) -> Result<(), Box<dyn Error>> {
    Err("V2 native execution requires Linux; no engine was started".into())
}
#[cfg(target_os = "linux")]
fn execute(args: &[String]) -> Result<(), Box<dyn Error>> {
    use std::sync::{Arc, atomic::AtomicBool};
    let lock = LockedManifestV2::from_json(&read(Path::new(&args[0]))?)?;
    let resource = &lock.input().resources;
    let status = fs::read_to_string("/proc/self/status")?;
    let cpus = status
        .lines()
        .find_map(|s| s.strip_prefix("Cpus_allowed_list:"))
        .ok_or("missing affinity evidence")?
        .trim();
    let mut actual = Vec::new();
    for segment in cpus.split(',') {
        if let Some((lo, hi)) = segment.split_once('-') {
            let (lo, hi) = (lo.parse::<u32>()?, hi.parse::<u32>()?);
            if hi < lo || hi - lo > 4096 {
                return Err("invalid CPU affinity evidence".into());
            }
            actual.extend(lo..=hi);
        } else {
            actual.push(segment.parse()?);
        }
    }
    let mut requested = resource.affinity.clone();
    requested.sort_unstable();
    actual.sort_unstable();
    if actual != requested {
        return Err("inherited CPU affinity differs from V2 resource declaration".into());
    }
    let cgroup = fs::read_to_string("/proc/self/cgroup")?;
    let relative = cgroup
        .lines()
        .find_map(|s| s.strip_prefix("0::"))
        .ok_or("cgroup v2 required")?;
    if relative.split('/').any(|s| s == "..") {
        return Err("invalid cgroup path".into());
    }
    let cgroup = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    for (name, value) in [
        ("memory.high", resource.memory_high_bytes),
        ("memory.max", resource.memory_max_bytes),
        ("memory.swap.max", resource.swap_max_bytes),
    ] {
        if fs::read_to_string(cgroup.join(name))?.trim() != value.to_string() {
            return Err(format!("inherited {name} differs from declaration").into());
        }
    }
    eprintln!("model_pair_resource_policy=affinity_and_memory_cgroup_verified gpu_peak=unknown");
    let cancel = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, Arc::clone(&cancel))?;
    }
    let owner = rz_arena::prepare_model_endpoint_launch(
        &lock,
        Path::new(&args[1]),
        Path::new(&args[2]),
        &args[3],
    )?;
    match rz_arena::run_model_endpoint_pair(owner, Some(&cancel)) {
        Ok(output) => {
            println!(
                "receipt={} integration_checks_passed={} scored_games={} strength_eligible=false",
                output.receipt_artifact.path,
                output.receipt.integration_checks_passed,
                output.receipt.scored_games
            );
            Ok(())
        }
        Err(failure) => {
            eprintln!(
                "V2 pair failed: {failure}; evidence and ownership retained by the native lifecycle"
            );
            Err(failure)
        }
    }
}
