//! CPU-only pinned Stockfish probe, separate from neural validation and games.
#[cfg(not(target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("Stockfish local probe requires Linux".into())
}
#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rz_arena::{CleanupStatus, OwnedArtifactTreeWatch, ProcessLimits, ProcessStop};
    use rz_experiments::{ArtifactRef, NativeEngineRole};
    use sha2::{Digest, Sha256};
    use std::{
        fs::{self, File, OpenOptions},
        io::{Read, Seek, SeekFrom, Write},
        os::unix::fs::OpenOptionsExt,
        path::Path,
    };
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: stockfish_check ABS_BINARY SHA256 NEW_ABS_OUTPUT".into());
    }
    let binary = Path::new(&args[0]);
    let output = Path::new(&args[2]);
    if !binary.is_absolute() || !output.is_absolute() {
        return Err("absolute binary and output required".into());
    }
    for path in [binary, output] {
        for parent in path.ancestors() {
            if parent.join(".git").exists()
                || fs::symlink_metadata(parent).is_ok_and(|m| m.file_type().is_symlink())
            {
                return Err("probe inputs and outputs must be outside Git and unlinked".into());
            }
        }
    }
    fs::create_dir(output)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(binary)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > 256 * 1024 * 1024 {
        return Err("bounded regular external binary required".into());
    }
    let hash = |file: &mut File| -> Result<String, std::io::Error> {
        file.seek(SeekFrom::Start(0))?;
        let mut h = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            h.update(&buffer[..n]);
        }
        file.seek(SeekFrom::Start(0))?;
        Ok(format!("{:x}", h.finalize()))
    };
    if hash(&mut file)? != args[1] {
        return Err("installed Stockfish binary hash differs".into());
    }
    let artifact = ArtifactRef {
        path: binary.strip_prefix("/")?.to_string_lossy().into(),
        bytes: metadata.len(),
        sha256: args[1].clone(),
        source: "https://github.com/official-stockfish/Stockfish/releases/tag/sf_19".into(),
        license: "GPL-3.0".into(),
    };
    let endpoint = rz_arena::stockfish19_endpoint(artifact, NativeEngineRole::Candidate);
    let cwd = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(output)?;
    let limits = ProcessLimits {
        wall_ms: 30000,
        shutdown_grace_ms: 15000,
        max_output_bytes: 256 * 1024,
        max_child_processes: 16,
    };
    let watch = OwnedArtifactTreeWatch {
        max_total_bytes: 2 * 1024 * 1024,
        max_file_bytes: 512 * 1024,
        max_files: 32,
        max_depth: 4,
    };
    let mut observations = Vec::new();
    for (name, commands) in [
        ("identification", b"uci\nquit\n".to_vec()),
        (
            "readiness",
            rz_arena::uci_preflight_commands(&endpoint.requested_options, true)?,
        ),
    ] {
        let result = rz_arena::supervise_protocol_in_directory(
            &file,
            &[],
            &cwd,
            limits,
            None,
            &watch,
            &commands,
        )?;
        for (suffix, bytes) in [
            ("stdout.log", result.stdout.as_slice()),
            ("stderr.log", result.stderr.as_slice()),
        ] {
            let mut out = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output.join(format!("{name}.{suffix}")))?;
            out.write_all(bytes)?;
            out.sync_all()?;
        }
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join(format!("{name}.process.json")))?;
        out.write_all(&serde_json::to_vec_pretty(&result.receipt)?)?;
        out.sync_all()?;
        if result.receipt.stop != ProcessStop::Exited
            || result.receipt.exit_code != Some(0)
            || result.receipt.group_cleanup != CleanupStatus::Gone
            || result.pending_child.is_some()
            || !result.receipt.errors.is_empty()
        {
            return Err("Stockfish process/quit/drain gate failed; exact evidence retained".into());
        }
        let ad = rz_arena::parse_uci_advertisement(&result.stdout)?;
        rz_arena::validate_uci_options(
            &ad,
            &endpoint.expected_uci_name,
            &endpoint.requested_options,
        )?;
        observations.push(ad);
        if name == "readiness" {
            let text = std::str::from_utf8(&result.stdout)?;
            let legal = rz_position::Position::startpos().legal_moves();
            if text.lines().filter(|s| *s == "readyok").count() != 2
                || !text
                    .lines()
                    .filter_map(|s| s.strip_prefix("bestmove "))
                    .filter_map(|s| s.split_whitespace().next())
                    .any(|s| legal.iter().any(|m| m.to_string() == s))
            {
                return Err("readiness/search/stop witness missing".into());
            }
        }
    }
    if observations[0] != observations[1] || hash(&mut file)? != args[1] {
        return Err("external identity changed between sessions".into());
    }
    let record = serde_json::json!({"status":"passed","schema_version":2,"endpoint":endpoint,"advertisement":observations[0],"supported_requested_options":true,"two_fresh_processes":true,"stop_legal_bestmove":true,"quit_and_owned_group_cleanup":true,"actual_option_values":"unknown_UCI_has_no_general_readback","selected_isa":"unknown_see_compiler_output","scope":"CPU_protocol_probe_only_not_game_or_neural_acceptance"});
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("stockfish-preflight.json"))?;
    out.write_all(&serde_json::to_vec_pretty(&record)?)?;
    out.sync_all()?;
    println!("Stockfish 19 CPU pin/options/readiness/stop/quit passed");
    Ok(())
}
