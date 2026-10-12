//! Explicit V2 lock/execute entry point; inherited Linux limits are checked.
#[cfg(target_os = "linux")]
use rz_experiments::LockedManifestV2;
use rz_experiments::RunManifestV2;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Write, path::Path};
fn read(path: &Path) -> Result<String, Box<dyn Error>> {
    if !path.is_absolute()
        || fs::symlink_metadata(path)?.file_type().is_symlink()
        || fs::metadata(path)?.len() > rz_experiments::MAX_MANIFEST_BYTES as u64
    {
        return Err("bounded absolute regular manifest required".into());
    }
    Ok(fs::read_to_string(path)?)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("absolute output outside Git required".into());
    }
    for parent in path.ancestors().skip(1) {
        if parent.join(".git").exists() {
            return Err("run artifacts stay outside Git".into());
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
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hardware") if args.len() == 1 => { println!("{}",serde_json::to_string_pretty(&rz_arena::probe_match_hardware()?)?); Ok(()) },
        Some("engine-exec") if args.len() >= 4 => engine_exec(&args[1..]),
        Some("resource-plan") if args.len() == 2 => {
            let input = RunManifestV2::from_json(&read(Path::new(&args[1]))?)?;
            let execution = input.match_execution.as_ref().ok_or("match_execution is required")?;
            println!("{}", serde_json::to_string_pretty(&execution.plan()?)?); Ok(())
        },
        Some("lock") if args.len()==3=>{
            let input=RunManifestV2::from_json(&read(Path::new(&args[1]))?)?;
            rz_arena::validate_opening_artifact_for_spec(&input.opening,input.max_plies,&input.opening_artifact)?;
            let lock=input.lock()?;
            write_new(Path::new(&args[2]),lock.to_json()?.as_bytes())?;
            println!("input_sha256={} execution_ready=false",lock.sha256());Ok(())
        },
        Some("opening") if args.len()==3=>{
            let input=RunManifestV2::from_json(&read(Path::new(&args[1]))?)?;
            let text=rz_arena::opening_pgn_for_spec(&input.opening,input.max_plies)?;
            write_new(Path::new(&args[2]),text.as_bytes())?;
            println!("opening_bytes={} opening_sha256={:x} engines_started=false",text.len(),Sha256::digest(text.as_bytes()));
            Ok(())
        },
        Some("execute") if args.len()==5=>execute(&args[1..]),
        _=>Err("usage: model-pair hardware | resource-plan INPUT_JSON | opening INPUT_JSON NEW_PGN | lock INPUT_JSON NEW_LOCK_JSON | execute LOCK_JSON ASSET_ROOT OUTPUT_ROOT UNIQUE_LABEL | engine-exec CPU_IDS GPU_IDS_OR_DASH PROGRAM [ARGS...]".into()),
    }
}
#[cfg(not(target_os = "linux"))]
fn engine_exec(_args: &[String]) -> Result<(), Box<dyn Error>> {
    Err("engine placement requires Linux".into())
}
#[cfg(target_os = "linux")]
fn engine_exec(args: &[String]) -> Result<(), Box<dyn Error>> {
    use nix::{
        sched::{CpuSet, sched_getaffinity, sched_setaffinity},
        unistd::Pid,
    };
    use std::{os::unix::process::CommandExt, process::Command};
    if args[0].len() > 8192 || args[1].len() > 1296 || !Path::new(&args[2]).is_absolute() {
        return Err("invalid placement argv".into());
    }
    let mut cpus = CpuSet::new();
    let mut ids = std::collections::BTreeSet::new();
    for text in args[0].split(',') {
        let cpu = text.parse::<usize>()?;
        if !ids.insert(cpu) {
            return Err("placement CPU is duplicated".into());
        }
        cpus.set(cpu)?;
    }
    let gpu_mask = if args[1] == "-" { "" } else { &args[1] };
    if !gpu_mask.is_empty()
        && !gpu_mask.split(',').all(|id| {
            !id.is_empty()
                && id.len() <= 80
                && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
    {
        return Err("invalid GPU mask".into());
    }
    sched_setaffinity(Pid::from_raw(0), &cpus)?;
    if sched_getaffinity(Pid::from_raw(0))? != cpus {
        return Err("CPU placement readback differs".into());
    }
    eprintln!(
        "RZ_PLACEMENT_V1 cpus={} gpus={} gpu_memory_enforcement=false",
        args[0], args[1]
    );
    let error = Command::new(&args[2])
        .args(&args[3..])
        .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        .env("CUDA_VISIBLE_DEVICES", gpu_mask)
        .exec();
    Err(error.into())
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
    if let Some(execution) = &lock.input().match_execution {
        let plan = execution.plan()?;
        eprintln!("match_resource_plan={}", serde_json::to_string(&plan)?);
    }
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
