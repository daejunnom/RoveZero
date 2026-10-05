//! Bounded synthetic pre-spawn I/O probe. Never loads NN libraries or runs engines.
//! Usage: native_prepare_profile NEW_OUTPUT_ROOT NEW_SOURCE_ROOT INPUT_BYTES ITERATIONS LABEL

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("native preparation profiling requires Linux; no probe was run");
    std::process::exit(2);
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rz_arena::{
        FASTCHESS_SOURCE_COMMIT, FASTCHESS_SOURCE_URL, FASTCHESS_VERSION, opening_pgn_for_spec,
        prepare_native_launch,
    };
    use rz_experiments::*;
    use sha2::{Digest, Sha256};
    use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, time::*};

    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 5 {
        return Err("requires NEW_OUTPUT_ROOT NEW_SOURCE_ROOT INPUT_BYTES ITERATIONS LABEL".into());
    }
    let root = PathBuf::from(&args[0]);
    let source = PathBuf::from(&args[1]);
    let bytes: usize = args[2].to_str().ok_or("invalid input bytes")?.parse()?;
    let iterations: usize = args[3].to_str().ok_or("invalid iterations")?.parse()?;
    let label = args[4].to_str().ok_or("invalid label")?;
    if !(64 * 1024..=8 * 1024 * 1024).contains(&bytes)
        || !(1..=15).contains(&iterations)
        || label.is_empty()
        || label.len() > 64
        || !label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        || !root.is_absolute()
        || !source.is_absolute()
        || root.exists()
        || source.exists()
    {
        return Err(
            "requires new absolute root, input 64KiB..8MiB, iterations 1..15 and ASCII label"
                .into(),
        );
    }
    for selected in [&root, &source] {
        let parent = selected
            .parent()
            .ok_or("output has no parent")?
            .canonicalize()?;
        for ancestor in parent.ancestors() {
            match fs::symlink_metadata(ancestor.join(".git")) {
                Ok(_) => return Err("diagnostic output must be outside Git".into()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    fs::create_dir(&root)?;
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
    let output = root.join("attempts");
    fs::create_dir(&source)?;
    fs::create_dir(&output)?;
    let reference = |name: &str, data: &[u8]| -> Result<ArtifactRef, std::io::Error> {
        fs::write(source.join(name), data)?;
        Ok(ArtifactRef {
            path: name.into(),
            bytes: data.len() as u64,
            sha256: format!("{:x}", Sha256::digest(data)),
            source: "https://github.com/daejunnom/RoveZero".into(),
            license: "Synthetic I/O bytes only; not model or executable assets".into(),
        })
    };
    let opening = OpeningSpec {
        id: "synthetic-preparation".into(),
        initial: InitialPosition::Startpos,
        fen: None,
        moves: vec!["e2e4".into(), "e7e5".into()],
        history: HistoryCompleteness::Complete,
        history_origin: "startpos-complete-trace".into(),
    };
    let opening_artifact = reference("opening.pgn", opening_pgn_for_spec(&opening, 8)?.as_bytes())?;
    let artifacts: Vec<_> = NativeArtifactRole::ALL
        .into_iter()
        .enumerate()
        .map(|(i, role)| {
            let size = match role {
                NativeArtifactRole::SourceWeights
                | NativeArtifactRole::Onnx
                | NativeArtifactRole::OrtLibrary => bytes.min(role.byte_limit() as usize),
                _ => 4096,
            };
            reference(&format!("input-{i}"), &vec![i as u8 + 1; size])
                .map(|artifact| NativeArtifactBinding { role, artifact })
        })
        .collect::<Result<_, _>>()?;
    let baseline = CpuNativeLaunchSpecV1 {
        role: NativeEngineRole::Baseline,
        engine_id: "synthetic-baseline".into(),
        source_commit: "a".repeat(40),
        target: "x86_64-unknown-linux-gnu".into(),
        artifacts,
        profile: NativeCpuProfileV1::fixed("b".repeat(64), "c".repeat(64)),
    };
    let mut candidate = baseline.clone();
    candidate.role = NativeEngineRole::Candidate;
    candidate.engine_id = "synthetic-candidate".into();
    let mut runner = reference("fastchess", b"synthetic-not-an-executable")?;
    runner.source = FASTCHESS_SOURCE_URL.into();
    let spec = IntegrationPairSpecV1 {
        schema_version: 1,
        purpose: NativeIntegrationPurpose::CpuNnIntegration,
        strength_eligible: false,
        contract_revision: "0.1".into(),
        run_id: "synthetic-preparation".into(),
        pair_id: "synthetic-preparation-pair".into(),
        engines: [baseline, candidate],
        white_order: [NativeEngineRole::Baseline, NativeEngineRole::Candidate],
        opening,
        opening_artifact,
        runner: ToolIdentity {
            version: FASTCHESS_VERSION.into(),
            source_url: FASTCHESS_SOURCE_URL.into(),
            source_commit: FASTCHESS_SOURCE_COMMIT.into(),
            binary: runner,
            dirty: false,
            dirty_patch: None,
            build_mode: "synthetic-not-built".into(),
            compiler: "synthetic-not-built".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            isa: "x86-64".into(),
        },
        clock: NativeMovetimeV1 { movetime_ms: 100 },
        max_plies: 8,
        timeouts: NativeTimeoutsV1 {
            startup_ms: 1000,
            handshake_ms: 1000,
            runtime_ms: 10000,
            drain_ms: 1000,
            shutdown_ms: 1000,
        },
        budget: NativeResourceBudgetV1 {
            max_input_bytes: 32 * 1024 * 1024,
            max_output_bytes: 256 * 1024,
            max_runtime_bytes: 512 * 1024,
            max_artifact_bytes: 34 * 1024 * 1024,
            max_child_processes: 3,
            max_runtime_files: 32,
            max_runtime_depth: 4,
            address_space_per_process_bytes: 2 * 1024 * 1024 * 1024,
        },
    }
    .lock()?;
    let mut samples = Vec::new();
    for iteration in 0..iterations {
        let attempt = format!("sample-{iteration:02}");
        let utc_start_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let start = Instant::now();
        let owner = prepare_native_launch(&spec, &source, &output, &attempt)
            .map_err(|failure| failure.to_string())?;
        let elapsed_ns = start.elapsed().as_nanos();
        let utc_end_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        if !owner
            .snapshots()
            .iter()
            .all(|s| s.distinct_source_inode && s.closed_writer_read_only)
        {
            return Err("snapshot ownership verification failed".into());
        }
        let input_bytes: u64 = owner.snapshots().iter().map(|s| s.bytes).sum();
        let snapshots = owner.snapshots().len();
        drop(owner);
        samples.push(
            serde_json::json!({"iteration":iteration,"prepare_monotonic_ns":elapsed_ns,
            "utc_start_unix_ns":utc_start_ns,"utc_end_unix_ns":utc_end_ns,
            "input_bytes":input_bytes,"snapshots":snapshots,"ownership_verified":true}),
        );
        // Only this exclusively created synthetic attempt is inactive; no child
        // was spawned and no real model/library/executable was loaded.
        let path = output.join(attempt);
        if fs::symlink_metadata(&path)?.file_type().is_symlink() {
            return Err("synthetic attempt pathname replaced".into());
        }
        fs::set_permissions(path.join("inputs"), fs::Permissions::from_mode(0o700))?;
        fs::remove_dir_all(path)?;
    }
    let report = serde_json::json!({"schema_version":1,"scope":"synthetic pre-spawn preparation only",
        "label":label,"requested_asset_bytes":bytes,"iterations":iterations,"samples":samples,
        "source_recently_written":true,"cold_cache_claim":false,"engines_spawned":false,
        "model_loaded":false,"gpu_used":false,"source_input_sha256":spec.sha256()});
    let serialized = serde_json::to_string_pretty(&report)?;
    fs::write(root.join("profile.json"), &serialized)?;
    println!("{serialized}");
    Ok(())
}
