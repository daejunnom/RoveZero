//! One fresh, bounded synthetic pair. This is not a formal arena permit.
//! The external runner's clocks, draw profile, CPU/RAM isolation and per-game
//! RNG seeds do not satisfy the formal protocol; these remain explicit limits.
use crate::{ArenaError, ArenaPlan, LedgerSummary, PairPgnAudit, ProcessReceipt};
#[cfg(target_os = "linux")]
use crate::{
    ArtifactWatch, CleanupStatus, Event, GameOutcome, Ledger, LedgerLimits, PgnLimits,
    ProcessLimits, ProcessStop, audit_pair_pgn, build_fastchess_invocation, opening_pgn,
    supervise_in_directory, validate_engine_contract_revision,
};
use rz_experiments::ArtifactRef;
use serde::Serialize;
#[cfg(target_os = "linux")]
use sha2::{Digest, Sha256};
#[cfg(target_os = "linux")]
use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{Read, Write};
use std::path::Path;
use std::process::Child;
use std::sync::atomic::AtomicBool;

#[cfg(target_os = "linux")]
const METADATA_CAP: u64 = 64 * 1024;
#[cfg(target_os = "linux")]
const OWN_SOURCE: &str = "https://github.com/daejunnom/RoveZero";

#[derive(Debug, Serialize)]
pub struct FixturePairReceipt {
    pub receipt_version: u32,
    pub execution_ready: bool,
    pub validation_scope: String,
    pub input_sha256: String,
    pub plan_sha256: String,
    pub pair_id: String,
    pub attempt: u32,
    pub engine_contract_revision: String,
    pub contract_source_commit: String,
    pub rules_source_commit: String,
    pub runner_source_commit: String,
    pub runner_binary_sha256: String,
    pub process: ProcessReceipt,
    pub artifacts: Vec<ArtifactRef>,
    pub pgn_audit: Option<PairPgnAudit>,
    pub audit_error: Option<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug)]
pub struct FixturePairOutput {
    pub receipt: FixturePairReceipt,
    pub receipt_artifact: ArtifactRef,
    pub ledger_tip_sha256: String,
    pub summary: LedgerSummary,
    /// A bounded cleanup may leave a child in uninterruptible kernel IO.
    /// Library callers must retain this ownership and subsequently reap it.
    pub pending_child: Option<Child>,
}

/// Post-spawn failures retain the receipt, captured bytes and pending-child
/// ownership even if the output filesystem cannot accept further evidence.
#[derive(Debug)]
pub struct FixtureExecutionFailure {
    pub cause: ArenaError,
    pub process: crate::ProcessOutput,
}

/// Consume one locked single-pair fixture plan. Creates a new basename directory
/// below artifact_root, preserves every failed attempt's outputs, and never
/// resumes or overwrites an earlier attempt. Retries require an external owner.
/// Input files must not be modified in place while this function uses them.
pub fn run_fixture_pair(
    plan: &ArenaPlan,
    artifact_root: &Path,
    output_directory: &str,
    cancel: Option<&AtomicBool>,
) -> Result<FixturePairOutput, ArenaError> {
    #[cfg(target_os = "linux")]
    {
        linux_run(plan, artifact_root, output_directory, cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (plan, artifact_root, output_directory, cancel);
        Err(ArenaError::Invalid(
            "fixture execution requires Linux".into(),
        ))
    }
}

#[cfg(target_os = "linux")]
fn linux_run(
    plan: &ArenaPlan,
    artifact_root: &Path,
    output_directory: &str,
    cancel: Option<&AtomicBool>,
) -> Result<FixturePairOutput, ArenaError> {
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use std::os::fd::AsRawFd;

    validate_engine_contract_revision(plan)?;
    if plan.pairs().len() != 1 {
        return Err(ArenaError::Invalid(
            "fixture execution requires exactly one planned pair".into(),
        ));
    }
    if output_directory.is_empty()
        || output_directory.len() > 128
        || !output_directory
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err(ArenaError::Invalid(
            "output directory must be a new ASCII basename".into(),
        ));
    }
    let input = plan.manifest().input();
    let pair = &plan.pairs()[0];
    let mut engine_files = Vec::new();
    let mut commands = BTreeMap::new();
    // Preflight the entire immutable input lock, then pin each actual launch file.
    plan.manifest()
        .verify_artifacts(artifact_root, input.budget.max_artifact_bytes)?;
    let runner = input
        .protocol
        .runner
        .binary
        .open_verified(artifact_root, input.budget.max_artifact_bytes)?;
    for engine in &input.engines {
        let file = engine
            .tool
            .binary
            .open_verified(artifact_root, input.budget.max_artifact_bytes)?;
        commands.insert(
            engine.id.clone(),
            std::path::PathBuf::from(format!(
                "/proc/{}/fd/{}",
                std::process::id(),
                file.as_raw_fd()
            )),
        );
        engine_files.push(file);
    }
    build_fastchess_invocation(
        plan,
        &pair.id,
        &commands,
        Path::new("/fixture/opening.pgn"),
        Path::new("/fixture/match.pgn"),
    )?;
    let opening = opening_pgn(plan, &pair.id)?;
    let unique: BTreeMap<_, _> = plan
        .manifest()
        .declared_artifacts()
        .into_iter()
        .map(|a| (&a.path, a.bytes))
        .collect();
    let input_bytes = unique
        .values()
        .try_fold(0u64, |sum, bytes| sum.checked_add(*bytes))
        .ok_or_else(|| ArenaError::Budget("input artifact size overflow".into()))?;
    // Reserve space for retained streams and two independently bounded metadata
    // files. The external PGN guard is a snapshot, not a kernel disk quota.
    let stream_cap = input
        .budget
        .max_output_bytes
        .checked_sub(METADATA_CAP * 2)
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            ArenaError::Budget("fixture output budget needs stream and metadata reserves".into())
        })?;
    let watch_cap = input
        .budget
        .max_artifact_bytes
        .checked_sub(input_bytes)
        .and_then(|n| n.checked_sub(input.budget.max_output_bytes))
        .filter(|n| *n > opening.len() as u64)
        .ok_or_else(|| {
            ArenaError::Budget(
                "fixture artifact budget needs opening, PGN and output reserves".into(),
            )
        })?;
    let total_grace = input
        .lifecycle
        .shutdown_timeout_ms
        .checked_mul(2)
        .ok_or_else(|| ArenaError::Budget("shutdown grace overflow".into()))?;
    let wall_ms = input
        .budget
        .max_wall_ms
        .checked_sub(total_grace)
        .filter(|n| *n > 0)
        .ok_or_else(|| ArenaError::Budget("wall budget must include two shutdown graces".into()))?;
    // The runner is the leader, plus both engines. A declaration of two children
    // cannot cover a three-process match.
    if input.budget.max_child_processes < 3 {
        return Err(ArenaError::Budget(
            "runner plus two engines require at least three child processes".into(),
        ));
    }
    let root = Dir::open_ambient_dir(artifact_root, cap_std::ambient_authority())
        .map_err(|_| ArenaError::Io("cannot pin artifact output root".into()))?;
    root.create_dir(output_directory).map_err(|_| {
        ArenaError::Io("cannot create exclusive attempt directory; prior outputs preserved".into())
    })?;
    let directory = root
        .open_dir_nofollow(output_directory)
        .map_err(|_| ArenaError::Io("cannot pin attempt directory".into()))?;
    // Pin all subsequent outputs and runner cwd to one directory handle.
    let cwd_handle = directory
        .try_clone()
        .map_err(|_| ArenaError::Io("cannot clone attempt directory".into()))?
        .into_std_file();
    let cwd = std::path::PathBuf::from(format!(
        "/proc/{}/fd/{}",
        std::process::id(),
        cwd_handle.as_raw_fd()
    ));
    let invocation = build_fastchess_invocation(
        plan,
        &pair.id,
        &commands,
        &cwd.join("opening.pgn"),
        &cwd.join("match.pgn"),
    )?;
    let put = |name: &str, bytes: &[u8], cap: u64| -> Result<ArtifactRef, ArenaError> {
        if bytes.len() as u64 > cap {
            return Err(ArenaError::Budget(format!(
                "{name} exceeds reserved byte budget"
            )));
        }
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let mut file = directory.open_with(name, &options).map_err(|_| {
            ArenaError::Io(format!("cannot create {name}; existing output preserved"))
        })?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| {
                ArenaError::Io(format!("cannot finish {name}; partial output preserved"))
            })?;
        Ok(artifact(output_directory, name, bytes))
    };
    let opening_artifact = put("opening.pgn", opening.as_bytes(), watch_cap)?;
    let mut process = supervise_in_directory(
        &runner,
        &invocation.args,
        &cwd_handle,
        ProcessLimits {
            wall_ms,
            shutdown_grace_ms: input.lifecycle.shutdown_timeout_ms,
            max_output_bytes: stream_cap,
            max_child_processes: input.budget.max_child_processes,
        },
        cancel,
        Some(&ArtifactWatch {
            relative_files: vec!["opening.pgn".into(), "match.pgn".into()],
            max_total_bytes: watch_cap,
        }),
    )?;
    let persisted = (|| -> Result<FixturePairOutput, ArenaError> {
        let mut artifacts = vec![
            opening_artifact,
            put("stdout.log", &process.stdout, stream_cap)?,
            put("stderr.log", &process.stderr, stream_cap)?,
        ];
        let pgn = match directory.symlink_metadata("match.pgn") {
            Ok(metadata) => {
                if !metadata.is_file() {
                    Err(ArenaError::Integrity(
                        "match PGN is not a regular file".into(),
                    ))
                } else {
                    let mut options = OpenOptions::new();
                    options
                        .read(true)
                        .follow(FollowSymlinks::No)
                        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
                    let file = directory
                        .open_with("match.pgn", &options)
                        .map_err(|_| ArenaError::Io("cannot open match PGN".into()))?;
                    read_pgn(file.into_std(), watch_cap.min(crate::MAX_JSON_BYTES as u64))
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(ArenaError::Integrity("runner produced no match PGN".into()))
            }
            Err(_) => Err(ArenaError::Io("cannot inspect match PGN".into())),
        };
        if let Ok(text) = &pgn {
            artifacts.push(artifact(output_directory, "match.pgn", text.as_bytes()));
        }
        let completed = process.receipt.stop == ProcessStop::Exited
            && process.receipt.exit_code == Some(0)
            && process.receipt.group_cleanup == CleanupStatus::Gone;
        let audit = if completed {
            pgn.as_ref()
                .map_err(|e| ArenaError::Integrity(e.to_string()))
                .and_then(|text| {
                    audit_pair_pgn(
                        plan,
                        &pair.id,
                        text,
                        PgnLimits {
                            max_bytes: usize::try_from(watch_cap.min(crate::MAX_JSON_BYTES as u64))
                                .map_err(|_| {
                                    ArenaError::Budget("PGN limit exceeds platform".into())
                                })?,
                            max_plies: input.protocol.max_plies,
                        },
                    )
                })
        } else {
            Err(ArenaError::Io(
                "runner did not finish with verified process-group cleanup".into(),
            ))
        };
        let mut limitations = invocation.limitations;
        limitations.extend([
            "CPU affinity and RAM quota are unverified".into(),
            "artifact and child limits use process snapshots, not kernel quotas".into(),
            "process exit does not attest runtime/GPU drain".into(),
        ]);
        let receipt = FixturePairReceipt {
            receipt_version: 1,
            execution_ready: false,
            validation_scope: "synthetic_pair_process_and_native_rules".into(),
            input_sha256: plan.manifest().sha256().into(),
            plan_sha256: plan.sha256().into(),
            pair_id: pair.id.clone(),
            attempt: 1,
            engine_contract_revision: "0.1".into(),
            contract_source_commit: crate::CONTRACT_SOURCE_COMMIT.into(),
            rules_source_commit: crate::RULES_SOURCE_COMMIT.into(),
            runner_source_commit: input.protocol.runner.source_commit.clone(),
            runner_binary_sha256: input.protocol.runner.binary.sha256.clone(),
            process: process.receipt.clone(),
            artifacts,
            audit_error: audit.as_ref().err().map(ToString::to_string),
            pgn_audit: audit.ok(),
            limitations,
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|e| ArenaError::Integrity(e.to_string()))?;
        let receipt_artifact = put("process-receipt.json", &receipt_bytes, METADATA_CAP)?;
        let mut ledger = Ledger::new(
            plan,
            LedgerLimits {
                max_events: 4,
                max_bytes: METADATA_CAP,
            },
        )?;
        ledger.append(Event::PairStarted {
            pair_id: pair.id.clone(),
            attempt: 1,
            process_run_id: format!(
                "pid-{}-{}",
                receipt.process.pid,
                &receipt_artifact.sha256[..16]
            ),
        })?;
        for (ordinal, index) in pair.execution_order.iter().enumerate() {
            let outcome = if let Some(audit) = &receipt.pgn_audit {
                let game = &audit.games[ordinal];
                if game.classification == "rules_terminal" {
                    GameOutcome::RulesTerminal {
                        result: game.result,
                        reason: game.terminal_reason.clone().ok_or_else(|| {
                            ArenaError::Integrity("audited terminal lacks native reason".into())
                        })?,
                        evidence: receipt_artifact.clone(),
                    }
                } else if game.classification == "incomplete" {
                    GameOutcome::Incomplete {
                        reason: "locked maximum plies reached; runner draw is excluded".into(),
                        evidence: receipt_artifact.clone(),
                    }
                } else {
                    GameOutcome::EngineLoss {
                        loser_engine: game.loser_engine.clone().ok_or_else(|| {
                            ArenaError::Integrity("audited loss lacks engine".into())
                        })?,
                        reason: game.engine_failure.ok_or_else(|| {
                            ArenaError::Integrity("audited loss lacks failure kind".into())
                        })?,
                        evidence: receipt_artifact.clone(),
                    }
                }
            } else if receipt.process.stop == ProcessStop::Cancelled {
                GameOutcome::Incomplete {
                    reason: "fixture cancelled; no partial pair scoring".into(),
                    evidence: receipt_artifact.clone(),
                }
            } else if completed {
                GameOutcome::ContractInvalid {
                    reason: "native PGN audit rejected runner output".into(),
                    evidence: receipt_artifact.clone(),
                }
            } else {
                GameOutcome::InfrastructureInvalid {
                    cause: "fixture runner or process cleanup failed".into(),
                    evidence: receipt_artifact.clone(),
                }
            };
            ledger.append(Event::GameRecorded {
                pair_id: pair.id.clone(),
                attempt: 1,
                game_id: pair.games[*index].id.clone(),
                outcome,
            })?;
        }
        ledger.append(Event::PairClosed {
            pair_id: pair.id.clone(),
            attempt: 1,
        })?;
        put("ledger.jsonl", ledger.to_jsonl()?.as_bytes(), METADATA_CAP)?;
        Ok(FixturePairOutput {
            receipt,
            receipt_artifact,
            ledger_tip_sha256: ledger.tip_sha256().into(),
            summary: ledger.summary()?,
            pending_child: None,
        })
    })();
    match persisted {
        Ok(mut output) => {
            output.pending_child = process.pending_child.take();
            Ok(output)
        }
        Err(cause) => Err(ArenaError::Execution(Box::new(FixtureExecutionFailure {
            cause,
            process,
        }))),
    }
}

#[cfg(target_os = "linux")]
fn artifact(directory: &str, name: &str, bytes: &[u8]) -> ArtifactRef {
    ArtifactRef {
        path: format!("{directory}/{name}"),
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: bytes.len() as u64,
        source: OWN_SOURCE.into(),
        license: "MIT - synthetic fixture execution evidence".into(),
    }
}

#[cfg(target_os = "linux")]
fn read_pgn(file: File, cap: u64) -> Result<String, ArenaError> {
    let metadata = file
        .metadata()
        .map_err(|_| ArenaError::Io("cannot inspect opened PGN".into()))?;
    if !metadata.is_file() || metadata.len() > cap {
        return Err(ArenaError::Budget(
            "PGN exceeds bounded regular-file profile".into(),
        ));
    }
    let mut bytes = Vec::new();
    file.take(cap + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| ArenaError::Io("cannot read bounded PGN".into()))?;
    if bytes.len() as u64 > cap {
        return Err(ArenaError::Budget("PGN grew past byte budget".into()));
    }
    String::from_utf8(bytes).map_err(|_| ArenaError::Invalid("PGN is not UTF-8".into()))
}
