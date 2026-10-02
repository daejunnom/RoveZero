//! Bounded actual CPU NN integration execution and independent evidence audit.
//! Process cleanup, B's NN evidence and A's PGN audit remain separate gates.
//! Neither this API nor a successful integration receipt permits strength runs.

use crate::{
    ArenaError, CleanupStatus, NativeLaunchOwner, NativeSnapshotReceipt, PairPgnAudit,
    ProcessOutput, ProcessReceipt,
};
use serde::Serialize;
use std::{
    fmt,
    sync::{
        Mutex, TryLockError,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Debug, Serialize)]
pub struct NativeProviderSessionAudit {
    pub role: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub completed_by_runtime: u64,
    pub actual_cpu_inference_observed: bool,
    pub physical_drain: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativePairReceipt {
    pub receipt_version: u32,
    pub execution_ready: bool,
    pub strength_eligible: bool,
    pub validation_scope: String,
    pub input_sha256: String,
    pub pair_id: String,
    pub contract_revision: String,
    pub runner_source_commit: String,
    pub runner_binary_sha256: String,
    pub process: ProcessReceipt,
    pub snapshots: Vec<NativeSnapshotReceipt>,
    pub artifacts: Vec<rz_experiments::ArtifactRef>,
    pub provider_sessions: Vec<NativeProviderSessionAudit>,
    pub provider_audit_error: Option<String>,
    pub pgn_audit: Option<PairPgnAudit>,
    pub pgn_audit_error: Option<String>,
    pub integration_checks_passed: bool,
    /// Integration-only runs grant no scoring authority, including valid draws.
    pub scored_games: u32,
    pub incomplete_games: u32,
    pub primary_error: Option<String>,
    pub cleanup_verified: bool,
    pub unresolved_owner_retained: bool,
    pub limitations: Vec<String>,
}

struct NativeRunBundle {
    owner: Option<NativeLaunchOwner>,
    process: Option<ProcessOutput>,
    receipt: Option<NativePairReceipt>,
    receipt_artifact: Option<rz_experiments::ArtifactRef>,
    original_error: Option<String>,
}
impl NativeRunBundle {
    fn unresolved(&self) -> bool {
        self.process.as_ref().is_some_and(|p| {
            p.receipt.group_cleanup != CleanupStatus::Gone || p.pending_child.is_some()
        })
    }
}
impl Drop for NativeRunBundle {
    fn drop(&mut self) {
        if !self.unresolved() {
            return;
        }
        crate::native_launch::close_native_admission();
        let retained = Box::new(RetainedNativeScope {
            _owner: self.owner.take().expect("native bundle owns its lease"),
            process: self
                .process
                .take()
                .expect("unresolved bundle owns process evidence"),
            receipt: self.receipt.take(),
            original_error: self.original_error.take(),
        });
        // The active lease forbids a second unresolved bundle. Do not abort the
        // parent: abort would not ensure its runner/process group is terminated.
        // A busy/poisoned synchronization boundary must not free child inputs.
        match QUARANTINE.try_lock() {
            Ok(mut slot) => retain_in_slot(&mut slot, retained),
            Err(TryLockError::Poisoned(error)) => retain_in_slot(&mut error.into_inner(), retained),
            Err(TryLockError::WouldBlock) => {
                QUARANTINE_FALLBACK.store(true, Ordering::Release);
                std::mem::forget(retained);
            }
        }
    }
}
struct RetainedNativeScope {
    _owner: NativeLaunchOwner,
    process: ProcessOutput,
    receipt: Option<NativePairReceipt>,
    original_error: Option<String>,
}
static QUARANTINE: Mutex<Option<Box<RetainedNativeScope>>> = Mutex::new(None);
static QUARANTINE_FALLBACK: AtomicBool = AtomicBool::new(false);
fn retain_in_slot(slot: &mut Option<Box<RetainedNativeScope>>, retained: Box<RetainedNativeScope>) {
    if slot.is_none() {
        *slot = Some(retained);
    } else {
        // Never overwrite an owned scope. This can only be an invariant failure
        // because the unreleased first lease prevents a second admission.
        QUARANTINE_FALLBACK.store(true, Ordering::Release);
        std::mem::forget(retained);
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeQuarantineStatus {
    pub admission_closed: bool,
    pub slot_observed: bool,
    pub slot_busy: bool,
    pub bounded_fallback_retained: bool,
    pub process_id: Option<u32>,
    pub pending_leader_handle: Option<bool>,
    pub receipt_retained: bool,
    pub original_error_retained: bool,
}
/// Passive, nonblocking observation only. No numeric-PID signal, wait/reap,
/// release, deletion, retry or reopening of native admission is provided.
pub fn native_quarantine_status() -> NativeQuarantineStatus {
    let mut status = NativeQuarantineStatus {
        admission_closed: crate::native_admission_closed(),
        slot_observed: false,
        slot_busy: false,
        bounded_fallback_retained: QUARANTINE_FALLBACK.load(Ordering::Acquire),
        process_id: None,
        pending_leader_handle: None,
        receipt_retained: false,
        original_error_retained: false,
    };
    let read = |slot: &Option<Box<RetainedNativeScope>>, status: &mut NativeQuarantineStatus| {
        status.slot_observed = true;
        if let Some(scope) = slot {
            status.process_id = Some(scope.process.receipt.pid);
            status.pending_leader_handle = Some(scope.process.pending_child.is_some());
            status.receipt_retained = scope.receipt.is_some();
            status.original_error_retained = scope.original_error.is_some();
        }
    };
    match QUARANTINE.try_lock() {
        Ok(slot) => read(&slot, &mut status),
        Err(TryLockError::Poisoned(e)) => read(&e.into_inner(), &mut status),
        Err(TryLockError::WouldBlock) => status.slot_busy = true,
    }
    status
}

pub struct NativePairOutput {
    pub receipt: NativePairReceipt,
    pub receipt_artifact: rz_experiments::ArtifactRef,
    bundle: NativeRunBundle,
}
impl NativePairOutput {
    pub fn process(&self) -> &ProcessOutput {
        self.bundle
            .process
            .as_ref()
            .expect("completed wrapper retains process evidence")
    }
    pub fn launch_owner(&self) -> &NativeLaunchOwner {
        self.bundle
            .owner
            .as_ref()
            .expect("completed wrapper retains input owner")
    }
}
impl fmt::Debug for NativePairOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativePairOutput")
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}
pub struct NativePairFailure {
    pub cause: ArenaError,
    pub receipt: Option<NativePairReceipt>,
    pub receipt_artifact: Option<rz_experiments::ArtifactRef>,
    bundle: NativeRunBundle,
}
impl NativePairFailure {
    pub fn process(&self) -> Option<&ProcessOutput> {
        self.bundle.process.as_ref()
    }
    pub fn launch_owner(&self) -> Option<&NativeLaunchOwner> {
        self.bundle.owner.as_ref()
    }
}
impl fmt::Debug for NativePairFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativePairFailure")
            .field("cause", &self.cause)
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}
impl fmt::Display for NativePairFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "native pair failed: {}; original process/input ownership retained={}",
            self.cause,
            self.bundle.owner.is_some()
        )
    }
}
impl std::error::Error for NativePairFailure {}

/// Consume exactly one prepared lease. Post-spawn failures retain the original
/// process output and every input/cwd pin, including when receipt saving fails.
pub fn run_native_pair(
    owner: NativeLaunchOwner,
    cancel: Option<&AtomicBool>,
) -> Result<NativePairOutput, Box<NativePairFailure>> {
    let mut bundle = NativeRunBundle {
        owner: Some(owner),
        process: None,
        receipt: None,
        receipt_artifact: None,
        original_error: None,
    };
    #[cfg(target_os = "linux")]
    let result = linux::run(&mut bundle, cancel);
    #[cfg(not(target_os = "linux"))]
    let result: Result<(NativePairReceipt, rz_experiments::ArtifactRef), ArenaError> = {
        let _ = cancel;
        Err(ArenaError::Invalid(
            "CPU NN integration runner requires Linux".into(),
        ))
    };
    match result {
        Ok((receipt, receipt_artifact)) => {
            bundle.receipt = Some(receipt.clone());
            Ok(NativePairOutput {
                receipt,
                receipt_artifact,
                bundle,
            })
        }
        Err(cause) => {
            bundle.original_error = Some(cause.to_string());
            Err(Box::new(NativePairFailure {
                cause,
                receipt: bundle.receipt.clone(),
                receipt_artifact: bundle.receipt_artifact.clone(),
                bundle,
            }))
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::validate_native_provider_record_fields;

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use crate::native_launch::{
        NATIVE_PAIR_METADATA_CAP,
        linux::{pin, verify_copy},
    };
    use crate::{
        GameSpec, PairSpec, PgnLimits, PgnOutcomePolicy, ProcessStop, audit_pair_pgn_for_spec,
        canonical_sha256, supervise_in_directory_with_tree,
    };
    use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
    use rz_experiments::{
        ArtifactRef, CpuNativeLaunchSpecV1, NativeArtifactRole, NativeEngineRole, OutcomePolicy,
    };
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs::File,
        io::Read,
        os::unix::fs::MetadataExt,
    };
    const PROVIDER_JSON_CAP: u64 = 256 * 1024;
    const STARTUP: &str = "native-cpu-startup.v1.json";
    const TERMINATION: &str = "native-cpu-termination.v1.json";
    fn invalid(detail: &str) -> ArenaError {
        ArenaError::Integrity(detail.into())
    }
    fn artifact(owner: &NativeLaunchOwner, name: &str, bytes: &[u8]) -> ArtifactRef {
        ArtifactRef {
            path: format!("{}/{name}", owner.snapshot.output_directory),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.len() as u64,
            source: "RoveZero CPU NN integration execution evidence".into(),
            license: "MIT execution evidence; external input rights remain separate".into(),
        }
    }
    fn put(
        owner: &NativeLaunchOwner,
        name: &str,
        bytes: &[u8],
        cap: u64,
    ) -> Result<ArtifactRef, ArenaError> {
        if bytes.len() as u64 > cap {
            return Err(ArenaError::Budget(
                "native evidence exceeds reserved bytes".into(),
            ));
        }
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .mode(0o600);
        let mut file = owner
            .snapshot
            .directory
            .open_with(name, &options)
            .map_err(|_| {
                ArenaError::Io(
                    "cannot create exclusive native evidence; prior/partial bytes retained".into(),
                )
            })?;
        use std::io::Write;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| {
                ArenaError::Io("cannot finish native evidence; partial bytes retained".into())
            })?;
        Ok(artifact(owner, name, bytes))
    }
    fn read_file(directory: &Dir, name: &str, cap: u64) -> Result<Vec<u8>, ArenaError> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = directory
            .open_with(name, &options)
            .map_err(|_| {
                ArenaError::Io("native evidence file is missing or cannot be pinned".into())
            })?
            .into_std();
        read_bytes(file, cap)
    }
    fn read_bytes(mut file: File, cap: u64) -> Result<Vec<u8>, ArenaError> {
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(0))
            .map_err(|_| invalid("cannot rewind pinned native evidence"))?;
        let metadata = file
            .metadata()
            .map_err(|_| invalid("native evidence metadata unavailable"))?;
        if !metadata.is_file() || metadata.len() > cap {
            return Err(ArenaError::Budget(
                "native evidence is not a bounded regular file".into(),
            ));
        }
        let limit = cap
            .checked_add(1)
            .ok_or_else(|| ArenaError::Budget("native read limit overflow".into()))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(
                usize::try_from(metadata.len())
                    .map_err(|_| ArenaError::Budget("native evidence cannot fit host".into()))?,
            )
            .map_err(|_| ArenaError::Budget("native evidence allocation failed".into()))?;
        file.take(limit)
            .read_to_end(&mut bytes)
            .map_err(|_| ArenaError::Io("cannot read pinned native evidence".into()))?;
        if bytes.len() as u64 > cap {
            return Err(ArenaError::Budget(
                "native evidence grew beyond limit".into(),
            ));
        }
        Ok(bytes)
    }
    fn verify_inputs(owner: &mut NativeLaunchOwner) -> Result<(), ArenaError> {
        for item in &mut owner.snapshot.pins {
            verify_copy(&mut item.file, &item.artifact)?;
            let relative = item
                .path
                .strip_prefix(&owner.snapshot.path)
                .map_err(|_| invalid("native pin escaped owned directory"))?;
            let name = relative
                .file_name()
                .ok_or_else(|| invalid("native input has no direct copy name"))?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .follow(FollowSymlinks::No)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
            let path_file = owner
                .snapshot
                .input_directory
                .open_with(name, &options)
                .map_err(|_| invalid("native input pathname changed"))?
                .into_std();
            let held = item
                .file
                .metadata()
                .map_err(|_| invalid("native pin metadata missing"))?;
            let named = path_file
                .metadata()
                .map_err(|_| invalid("native named input metadata missing"))?;
            if (held.dev(), held.ino()) != (named.dev(), named.ino()) {
                return Err(invalid(
                    "native input pathname no longer names its owned inode",
                ));
            }
        }
        Ok(())
    }
    fn pair(owner: &NativeLaunchOwner) -> Result<PairSpec, ArenaError> {
        let input = owner.spec.input();
        let game = |ordinal: usize| -> Result<GameSpec, ArenaError> {
            let white = input.engine(input.white_order[ordinal])?;
            let black = input.engine(input.white_order[ordinal].other())?;
            Ok(GameSpec {
                id: format!("{}-game-{}", input.pair_id, ordinal + 1),
                white_engine: white.engine_id.clone(),
                black_engine: black.engine_id.clone(),
                engine_seeds: BTreeMap::new(),
                engine_slots: BTreeMap::new(),
            })
        };
        Ok(PairSpec {
            id: input.pair_id.clone(),
            ordinal: 0,
            opening: input.opening.clone(),
            opening_input_sha256: canonical_sha256(&input.opening)?,
            games: [game(0)?, game(1)?],
            execution_order: [0, 1],
        })
    }
    pub(super) fn run(
        bundle: &mut NativeRunBundle,
        cancel: Option<&AtomicBool>,
    ) -> Result<(NativePairReceipt, ArtifactRef), ArenaError> {
        let owner = bundle.owner.as_mut().expect("prepared native owner");
        verify_inputs(owner)?;
        let s = &owner.snapshot;
        let process = supervise_in_directory_with_tree(
            &s.pins[s.runner_index].file,
            &s.invocation.args,
            &s.cwd,
            s.limits,
            cancel,
            &s.watch,
        )?;
        bundle.process = Some(process);
        let owner = bundle
            .owner
            .as_ref()
            .expect("native owner retained after spawn");
        let process = bundle
            .process
            .as_ref()
            .expect("native process output retained after spawn");
        let completed = process.receipt.stop == ProcessStop::Exited
            && process.receipt.exit_code == Some(0)
            && process.receipt.group_cleanup == CleanupStatus::Gone
            && process.pending_child.is_none()
            && process.receipt.errors.is_empty();
        let mut receipt = NativePairReceipt {
            receipt_version: 1,
            execution_ready: false,
            strength_eligible: false,
            validation_scope: "cpu_nn_pair_integration_process_provider_and_native_rules".into(),
            input_sha256: owner.spec.sha256().into(),
            pair_id: owner.spec.input().pair_id.clone(),
            contract_revision: "0.1".into(),
            runner_source_commit: owner.spec.input().runner.source_commit.clone(),
            runner_binary_sha256: owner.spec.input().runner.binary.sha256.clone(),
            process: process.receipt.clone(),
            snapshots: owner.snapshot.receipts.clone(),
            artifacts: Vec::new(),
            provider_sessions: Vec::new(),
            provider_audit_error: None,
            pgn_audit: None,
            pgn_audit_error: None,
            integration_checks_passed: false,
            scored_games: 0,
            incomplete_games: if process.receipt.stop == ProcessStop::Cancelled {
                2
            } else {
                0
            },
            primary_error: None,
            cleanup_verified: process.receipt.group_cleanup == CleanupStatus::Gone
                && process.pending_child.is_none(),
            unresolved_owner_retained: bundle.unresolved(),
            limitations: owner.snapshot.invocation.limitations.clone(),
        };
        // Capture the receipt before each fallible save so original process and
        // cleanup evidence still survive a disk/persistence failure.
        bundle.receipt = Some(receipt.clone());
        receipt.artifacts.push(put(
            owner,
            "stdout.log",
            &process.stdout,
            owner.snapshot.limits.max_output_bytes,
        )?);
        receipt.artifacts.push(put(
            owner,
            "stderr.log",
            &process.stderr,
            owner.snapshot.limits.max_output_bytes,
        )?);
        let pgn_cap = owner
            .spec
            .input()
            .budget
            .max_output_bytes
            .checked_sub(NATIVE_PAIR_METADATA_CAP)
            .and_then(|n| n.checked_sub(owner.snapshot.limits.max_output_bytes))
            .ok_or_else(|| ArenaError::Budget("native PGN reservation underflow".into()))?;
        let pgn_bytes = read_file(&owner.snapshot.directory, "match.pgn", pgn_cap);
        if let Ok(bytes) = &pgn_bytes {
            receipt.artifacts.push(artifact(owner, "match.pgn", bytes));
        }
        // Fastchess may persist config.json. Bound it together with the PGN;
        // absent config is permitted, but a non-regular/invalid existing file is not.
        match owner.snapshot.directory.symlink_metadata("config.json") {
            Ok(meta) => {
                if !meta.is_file() {
                    return Err(invalid("native runner config is not regular"));
                }
                let remaining =
                    pgn_cap.saturating_sub(pgn_bytes.as_ref().map_or(0, |b| b.len() as u64));
                let bytes = read_file(
                    &owner.snapshot.directory,
                    "config.json",
                    remaining.min(NATIVE_PAIR_METADATA_CAP),
                )?;
                receipt
                    .artifacts
                    .push(artifact(owner, "config.json", &bytes));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(ArenaError::Io("cannot inspect native runner config".into())),
        }
        if completed {
            match pgn_bytes
                .and_then(|bytes| {
                    String::from_utf8(bytes).map_err(|_| invalid("native match PGN is not UTF-8"))
                })
                .and_then(|pgn| {
                    audit_pair_pgn_for_spec(
                        &pair(owner)?,
                        &pgn,
                        PgnLimits {
                            max_bytes: usize::try_from(pgn_cap.min(crate::MAX_JSON_BYTES as u64))
                                .map_err(|_| {
                                ArenaError::Budget("native PGN cap cannot fit host".into())
                            })?,
                            max_plies: owner.spec.input().max_plies,
                        },
                        PgnOutcomePolicy {
                            engine_failure: OutcomePolicy::Loss,
                            max_plies_outcome: OutcomePolicy::Incomplete,
                            max_game_plies: owner.spec.input().max_plies,
                        },
                    )
                }) {
                Ok(audit) => {
                    receipt.incomplete_games = audit
                        .games
                        .iter()
                        .filter(|g| g.classification == "incomplete")
                        .count() as u32;
                    receipt.pgn_audit = Some(audit)
                }
                Err(error) => receipt.pgn_audit_error = Some(error.to_string()),
            }
            match audit_providers(owner, &mut receipt.artifacts) {
                Ok(sessions) => receipt.provider_sessions = sessions,
                Err(error) => receipt.provider_audit_error = Some(error.to_string()),
            }
        } else {
            receipt.primary_error = Some(
                "native runner did not exit successfully with verified owned-group cleanup".into(),
            );
            receipt.pgn_audit_error =
                Some("not audited: process completion/cleanup gate failed".into());
            receipt.provider_audit_error =
                Some("not audited: partial provider evidence is not accepted".into());
        }
        bundle.receipt = Some(receipt.clone());
        verify_inputs(
            bundle
                .owner
                .as_mut()
                .expect("native owner retained through postcheck"),
        )?;
        receipt.integration_checks_passed = completed
            && receipt.pgn_audit.is_some()
            && receipt.provider_audit_error.is_none()
            && receipt.provider_sessions.len() == 4;
        if completed && !receipt.integration_checks_passed {
            receipt.primary_error =
                Some("native pair evidence audit rejected an integration gate".into());
        }
        bundle.receipt = Some(receipt.clone());
        let bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|_| invalid("cannot serialize bounded native receipt"))?;
        let artifact = put(
            bundle.owner.as_ref().expect("receipt retains owner"),
            "native-pair-receipt.v1.json",
            &bytes,
            NATIVE_PAIR_METADATA_CAP,
        )?;
        bundle.receipt_artifact = Some(artifact.clone());
        if !receipt.integration_checks_passed {
            return Err(invalid(
                "native pair failed process, provider or Rules integration acceptance",
            ));
        }
        Ok((receipt, artifact))
    }

    fn json(bytes: &[u8]) -> Result<Value, ArenaError> {
        let text =
            std::str::from_utf8(bytes).map_err(|_| invalid("provider receipt is not UTF-8"))?;
        Ok(rz_experiments::decode_json::<Value>(text)?)
    }
    fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, ArenaError> {
        v.get(key)
            .ok_or_else(|| invalid("provider receipt is missing a required field"))
    }
    fn number(v: &Value, key: &str) -> Result<u64, ArenaError> {
        field(v, key)?
            .as_u64()
            .ok_or_else(|| invalid("provider receipt integer type differs"))
    }
    fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, ArenaError> {
        field(v, key)?
            .as_str()
            .ok_or_else(|| invalid("provider receipt string type differs"))
    }
    fn equals(v: &Value, key: &str, expected: &str) -> Result<(), ArenaError> {
        if text(v, key)? != expected {
            Err(invalid("provider receipt locked string identity differs"))
        } else {
            Ok(())
        }
    }
    fn flag(v: &Value, key: &str, expected: bool) -> Result<(), ArenaError> {
        if field(v, key)?.as_bool() != Some(expected) {
            Err(invalid("provider receipt boolean profile differs"))
        } else {
            Ok(())
        }
    }
    fn n(v: &Value, key: &str, expected: u64) -> Result<(), ArenaError> {
        if number(v, key)? != expected {
            Err(invalid("provider receipt numeric profile differs"))
        } else {
            Ok(())
        }
    }
    fn none(v: &Value, key: &str) -> Result<(), ArenaError> {
        if !field(v, key)?.is_null() {
            Err(invalid(
                "provider receipt preserves a failure or unexpected retained state",
            ))
        } else {
            Ok(())
        }
    }
    fn keys(v: &Value, expected: &[&str]) -> Result<(), ArenaError> {
        let map = v
            .as_object()
            .ok_or_else(|| invalid("provider receipt object type differs"))?;
        if map.len() != expected.len() || expected.iter().any(|k| !map.contains_key(*k)) {
            Err(invalid(
                "provider receipt schema has unknown or missing fields",
            ))
        } else {
            Ok(())
        }
    }
    fn registry(v: &Value) -> Result<(), ArenaError> {
        keys(v, &["owner", "slot", "generation", "manifest_sha256"])?;
        if number(v, "owner")? == 0 {
            return Err(invalid("provider registry owner is zero"));
        }
        number(v, "slot")?;
        number(v, "generation")?;
        hash(text(v, "manifest_sha256")?)
    }
    fn hash(value: &str) -> Result<(), ArenaError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            Err(invalid("provider digest is not canonical sha256"))
        } else {
            Ok(())
        }
    }
    fn epoch(v: &Value, expected: u64) -> Result<(), ArenaError> {
        keys(v, &["epoch", "sequence"])?;
        n(v, "epoch", expected)?;
        if number(v, "sequence")? == 0 {
            Err(invalid("provider execution/selection sequence is zero"))
        } else {
            Ok(())
        }
    }

    fn check_startup(
        startup: &Value,
        engine: &CpuNativeLaunchSpecV1,
        session: &str,
    ) -> Result<(), ArenaError> {
        keys(
            startup,
            &[
                "schema_version",
                "kind",
                "loaded",
                "process_id",
                "process_run_id",
                "process_epoch",
                "executable",
                "profile",
            ],
        )?;
        n(startup, "schema_version", 1)?;
        equals(startup, "kind", "startup")?;
        flag(startup, "loaded", true)?;
        let pid = number(startup, "process_id")?;
        if pid == 0 || pid > u32::MAX as u64 || session != format!("native-process-{pid}") {
            return Err(invalid(
                "provider session directory and process identity differ",
            ));
        }
        equals(startup, "process_run_id", session)?;
        if number(startup, "process_epoch")? == 0 {
            return Err(invalid("provider process epoch is zero"));
        }
        let executable = field(startup, "executable")?;
        keys(executable, &["sha256", "bytes", "identity_source"])?;
        equals(
            executable,
            "sha256",
            &engine.artifact(NativeArtifactRole::Binary)?.sha256,
        )?;
        n(
            executable,
            "bytes",
            engine.artifact(NativeArtifactRole::Binary)?.bytes,
        )?;
        equals(executable, "identity_source", "linux_proc_self_exe")?;
        let p = field(startup, "profile")?;
        keys(
            p,
            &[
                "source_weights_gzip_sha256",
                "source_weights_protobuf_sha256",
                "onnx_sha256",
                "onnx_bytes",
                "export_manifest_sha256",
                "converter_commit",
                "converter_binary_sha256",
                "ort_library_sha256",
                "ort_build_info_sha256",
                "ort_release",
                "backend_sha256",
                "model",
                "encoding",
                "action_map_sha256",
                "history_policy_sha256",
                "history_length",
                "contract_major",
                "contract_minor",
                "provider",
                "precision",
                "max_batch_items",
                "intra_threads",
                "max_workers",
                "full_steps",
                "min_steps",
                "max_steps",
                "require_full",
                "evaluation_mode",
                "history_fill",
            ],
        )?;
        for (key, role) in [
            (
                "source_weights_gzip_sha256",
                NativeArtifactRole::SourceWeights,
            ),
            ("onnx_sha256", NativeArtifactRole::Onnx),
            ("export_manifest_sha256", NativeArtifactRole::ExportManifest),
            ("ort_library_sha256", NativeArtifactRole::OrtLibrary),
        ] {
            equals(p, key, &engine.artifact(role)?.sha256)?;
        }
        n(
            p,
            "onnx_bytes",
            engine.artifact(NativeArtifactRole::Onnx)?.bytes,
        )?;
        equals(p, "backend_sha256", &engine.profile.expected_backend_sha256)?;
        for key in [
            "source_weights_protobuf_sha256",
            "converter_binary_sha256",
            "ort_build_info_sha256",
            "action_map_sha256",
            "history_policy_sha256",
        ] {
            hash(text(p, key)?)?;
        }
        equals(p, "ort_release", "1.22.0")?;
        equals(p, "provider", "cpu")?;
        equals(p, "precision", "fp32")?;
        equals(p, "evaluation_mode", "fresh")?;
        equals(p, "history_fill", "no")?;
        n(p, "contract_major", 0)?;
        n(p, "contract_minor", 1)?;
        n(p, "history_length", 8)?;
        for key in [
            "max_batch_items",
            "intra_threads",
            "max_workers",
            "full_steps",
            "min_steps",
            "max_steps",
        ] {
            n(p, key, 1)?;
        }
        flag(p, "require_full", true)?;
        let model = field(p, "model")?;
        let encoding = field(p, "encoding")?;
        registry(model)?;
        registry(encoding)?;
        equals(
            model,
            "manifest_sha256",
            &engine.artifact(NativeArtifactRole::ExportManifest)?.sha256,
        )?;
        equals(
            encoding,
            "manifest_sha256",
            &engine.profile.expected_encoding_sha256,
        )?;
        Ok(())
    }
    fn check_completion(c: &Value, startup: &Value) -> Result<(), ArenaError> {
        keys(
            c,
            &[
                "contract_major",
                "contract_minor",
                "request",
                "selection",
                "game",
                "root",
                "state_owner",
                "state_revision",
                "state_sha256",
                "legal_order_sha256",
                "input_sha256",
                "model",
                "encoding",
                "backend_sha256",
                "precision",
                "min_steps",
                "max_steps",
                "require_full",
                "execution",
                "actual",
            ],
        )?;
        let p = field(startup, "profile")?;
        let process_epoch = number(startup, "process_epoch")?;
        n(c, "contract_major", 0)?;
        n(c, "contract_minor", 1)?;
        equals(c, "backend_sha256", text(p, "backend_sha256")?)?;
        equals(c, "precision", "fp32")?;
        n(c, "min_steps", 1)?;
        n(c, "max_steps", 1)?;
        flag(c, "require_full", true)?;
        for key in ["request", "selection", "execution"] {
            epoch(field(c, key)?, process_epoch)?;
        }
        for key in ["game", "root", "state_owner", "state_revision"] {
            number(c, key)?;
        }
        for key in ["state_sha256", "legal_order_sha256", "input_sha256"] {
            hash(text(c, key)?)?;
        }
        if field(c, "model")? != field(p, "model")?
            || field(c, "encoding")? != field(p, "encoding")?
        {
            return Err(invalid(
                "completion registry differs from loaded startup registry",
            ));
        }
        let actual = field(c, "actual")?;
        keys(
            actual,
            &[
                "precision",
                "steps",
                "full",
                "backend_sha256",
                "execution",
                "provenance",
                "prior_execution",
            ],
        )?;
        equals(actual, "precision", "fp32")?;
        n(actual, "steps", 1)?;
        flag(actual, "full", true)?;
        equals(actual, "backend_sha256", text(p, "backend_sha256")?)?;
        equals(actual, "provenance", "computed")?;
        none(actual, "prior_execution")?;
        if field(actual, "execution")? != field(c, "execution")? {
            return Err(invalid(
                "actual computed execution differs from completed context",
            ));
        }
        Ok(())
    }
    fn check_termination(term: &Value, startup: &Value) -> Result<u64, ArenaError> {
        keys(
            term,
            &[
                "schema_version",
                "kind",
                "process_run_id",
                "startup",
                "run_succeeded",
                "actual_cpu_inference_observed",
                "physical_drain",
                "report",
                "original_service_failure",
                "collection_failure",
                "retained_owner_and_evidence",
            ],
        )?;
        n(term, "schema_version", 1)?;
        equals(term, "kind", "termination")?;
        if field(term, "startup")? != startup {
            return Err(invalid(
                "termination startup copy differs from actual startup record",
            ));
        }
        equals(term, "process_run_id", text(startup, "process_run_id")?)?;
        flag(term, "run_succeeded", true)?;
        flag(term, "actual_cpu_inference_observed", true)?;
        equals(term, "physical_drain", "confirmed")?;
        none(term, "original_service_failure")?;
        none(term, "collection_failure")?;
        flag(term, "retained_owner_and_evidence", false)?;
        let report = field(term, "report")?;
        keys(
            report,
            &[
                "model_manifest_sha256",
                "backend_sha256",
                "origin",
                "completed_by_runtime",
                "first_completed",
                "last_completed",
                "completed_scope",
                "canceled",
                "expired",
                "failures",
                "overflow",
                "boundary_error",
                "poison_error",
            ],
        )?;
        let p = field(startup, "profile")?;
        equals(
            report,
            "model_manifest_sha256",
            text(p, "export_manifest_sha256")?,
        )?;
        equals(report, "backend_sha256", text(p, "backend_sha256")?)?;
        equals(report, "origin", "cpu_onnx")?;
        equals(
            report,
            "completed_scope",
            "process_aggregate_normal_poll_first_last_not_full_root_or_physical_journal",
        )?;
        let count = number(report, "completed_by_runtime")?;
        if count == 0 {
            return Err(invalid(
                "loaded process has no D-accepted actual CPU completion",
            ));
        }
        check_completion(field(report, "first_completed")?, startup)?;
        check_completion(field(report, "last_completed")?, startup)?;
        if field(report, "failures")?
            .as_array()
            .is_none_or(|a| !a.is_empty())
        {
            return Err(invalid("native process retains failed inference receipts"));
        }
        for key in ["overflow", "boundary_error", "poison_error"] {
            none(report, key)?;
        }
        for key in ["canceled", "expired"] {
            let a = field(report, key)?;
            keys(a, &["count", "first", "last"])?;
            let count = number(a, "count")?;
            if (count == 0 && (!field(a, "first")?.is_null() || !field(a, "last")?.is_null()))
                || (count > 0 && (field(a, "first")?.is_null() || field(a, "last")?.is_null()))
            {
                return Err(invalid("native audit aggregate presence/count differs"));
            }
            if count > 0 {
                for end in ["first", "last"] {
                    let cause = field(a, end)?;
                    keys(
                        cause,
                        &[
                            "projection",
                            "projection_truncated",
                            "max_depth",
                            "max_nodes",
                            "projected_nodes",
                        ],
                    )?;
                    flag(cause, "projection_truncated", false)?;
                    n(cause, "max_depth", 8)?;
                    n(cause, "max_nodes", 128)?;
                    let nodes = number(cause, "projected_nodes")?;
                    if nodes == 0 || nodes > 128 {
                        return Err(invalid("native audit cause node bound differs"));
                    }
                    let projection = field(cause, "projection")?;
                    equals(projection, "type", "native_diagnostic")?;
                    equals(projection, "kind", "dispatch_refused")?;
                    let physical = field(field(projection, "failure")?, "cause")?;
                    none(physical, "backend")?;
                    let contract = field(field(physical, "contract")?, "cause")?;
                    equals(
                        contract,
                        "code",
                        if key == "canceled" {
                            "canceled"
                        } else {
                            "expired"
                        },
                    )?;
                }
            }
        }
        Ok(count)
    }
    /// Decode and compare bounded public wire fields only. Calling this with
    /// synthetic JSON is not evidence that a process or a neural backend ran.
    pub fn validate_native_provider_record_fields(
        startup_bytes: &[u8],
        termination_bytes: &[u8],
        engine: &CpuNativeLaunchSpecV1,
        session: &str,
    ) -> Result<NativeProviderSessionAudit, ArenaError> {
        if startup_bytes.len() as u64 > PROVIDER_JSON_CAP
            || termination_bytes.len() as u64 > PROVIDER_JSON_CAP
        {
            return Err(ArenaError::Budget(
                "provider wire record exceeds 256KiB".into(),
            ));
        }
        engine.validate()?;
        let startup = json(startup_bytes)?;
        let term = json(termination_bytes)?;
        check_startup(&startup, engine, session)?;
        let completed = check_termination(&term, &startup)?;
        Ok(NativeProviderSessionAudit {
            role: match engine.role {
                NativeEngineRole::Baseline => "baseline",
                NativeEngineRole::Candidate => "candidate",
            }
            .into(),
            process_id: u32::try_from(number(&startup, "process_id")?)
                .map_err(|_| invalid("provider PID exceeds native range"))?,
            process_run_id: session.into(),
            startup_sha256: format!("{:x}", Sha256::digest(startup_bytes)),
            termination_sha256: format!("{:x}", Sha256::digest(termination_bytes)),
            completed_by_runtime: completed,
            actual_cpu_inference_observed: true,
            physical_drain: "confirmed".into(),
        })
    }
    fn runtime_tree(
        directory: &Dir,
        depth: usize,
        files: &mut u32,
        dirs: &mut u32,
        bytes: &mut u64,
        owner: &NativeLaunchOwner,
    ) -> Result<(), ArenaError> {
        let budget = owner.spec.input().budget;
        if depth > budget.max_runtime_depth as usize {
            return Err(ArenaError::Budget("native runtime depth exceeded".into()));
        }
        for entry in directory
            .entries()
            .map_err(|_| invalid("cannot inspect native runtime tree"))?
        {
            let entry = entry.map_err(|_| invalid("native runtime tree entry unavailable"))?;
            let name = entry.file_name();
            let metadata = directory
                .symlink_metadata(&name)
                .map_err(|_| invalid("native runtime tree metadata unavailable"))?;
            if metadata.file_type().is_symlink() {
                return Err(invalid("native runtime tree contains symlink"));
            }
            if metadata.is_dir() {
                *dirs = dirs.checked_add(1).ok_or_else(|| {
                    ArenaError::Budget("native runtime directory count overflow".into())
                })?;
                if files
                    .checked_add(*dirs)
                    .is_none_or(|n| n > budget.max_runtime_files)
                {
                    return Err(ArenaError::Budget(
                        "native runtime file+directory count exceeded".into(),
                    ));
                }
                runtime_tree(
                    &directory
                        .open_dir_nofollow(&name)
                        .map_err(|_| invalid("native runtime directory could not be pinned"))?,
                    depth + 1,
                    files,
                    dirs,
                    bytes,
                    owner,
                )?;
            } else if metadata.is_file() {
                *files = files.checked_add(1).ok_or_else(|| {
                    ArenaError::Budget("native runtime file count overflow".into())
                })?;
                if files
                    .checked_add(*dirs)
                    .is_none_or(|n| n > budget.max_runtime_files)
                {
                    return Err(ArenaError::Budget(
                        "native runtime file+directory count exceeded".into(),
                    ));
                }
                let mut options = OpenOptions::new();
                options
                    .read(true)
                    .follow(FollowSymlinks::No)
                    .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
                let file = directory
                    .open_with(&name, &options)
                    .map_err(|_| invalid("native runtime file could not be pinned"))?
                    .into_std();
                let metadata = file
                    .metadata()
                    .map_err(|_| invalid("native runtime pinned metadata unavailable"))?;
                if !metadata.is_file() || metadata.nlink() != 1 {
                    return Err(invalid("native runtime input is nonregular or hardlinked"));
                }
                *bytes = bytes
                    .checked_add(metadata.len())
                    .ok_or_else(|| ArenaError::Budget("native runtime byte sum overflow".into()))?;
                if *bytes > budget.max_runtime_bytes {
                    return Err(ArenaError::Budget(
                        "native runtime aggregate byte budget exceeded".into(),
                    ));
                }
            } else {
                return Err(invalid("native runtime tree contains nonregular entry"));
            }
        }
        Ok(())
    }
    fn audit_providers(
        owner: &NativeLaunchOwner,
        artifacts: &mut Vec<ArtifactRef>,
    ) -> Result<Vec<NativeProviderSessionAudit>, ArenaError> {
        let mut sessions = Vec::new();
        let mut ids = BTreeSet::new();
        let mut scanned = 0u32;
        let (mut tree_files, mut tree_dirs, mut tree_bytes) = (0, 0, 0);
        for (role, name) in [
            (NativeEngineRole::Baseline, "baseline-runtime"),
            (NativeEngineRole::Candidate, "candidate-runtime"),
        ] {
            let engine = owner.spec.input().engine(role)?;
            let directory = owner
                .snapshot
                .directory
                .open_dir_nofollow(name)
                .map_err(|_| invalid("role runtime directory is unavailable"))?;
            runtime_tree(
                &directory,
                0,
                &mut tree_files,
                &mut tree_dirs,
                &mut tree_bytes,
                owner,
            )?;
            let mut session_names = Vec::new();
            for entry in directory
                .entries()
                .map_err(|_| invalid("cannot inspect native role runtime entries"))?
            {
                scanned = scanned
                    .checked_add(1)
                    .ok_or_else(|| ArenaError::Budget("native session scan overflow".into()))?;
                if scanned > owner.spec.input().budget.max_runtime_files {
                    return Err(ArenaError::Budget(
                        "native session scan entry bound exceeded".into(),
                    ));
                }
                let entry =
                    entry.map_err(|_| invalid("native role runtime entry is unavailable"))?;
                let os = entry.file_name();
                let text = os
                    .to_str()
                    .ok_or_else(|| invalid("native role runtime name is not UTF-8"))?;
                if text.starts_with("native-process-") {
                    session_names.push(text.to_owned());
                }
            }
            session_names.sort();
            if session_names.len() != 2 {
                return Err(invalid(
                    "each role requires exactly two fresh native process sessions",
                ));
            }
            for session in session_names {
                let dir = directory
                    .open_dir_nofollow(&session)
                    .map_err(|_| invalid("native session is not a real owned directory"))?;
                let startup_bytes = read_file(&dir, STARTUP, PROVIDER_JSON_CAP)?;
                let termination_bytes = read_file(&dir, TERMINATION, PROVIDER_JSON_CAP)?;
                artifacts.push(artifact(
                    owner,
                    &format!("{name}/{session}/{STARTUP}"),
                    &startup_bytes,
                ));
                artifacts.push(artifact(
                    owner,
                    &format!("{name}/{session}/{TERMINATION}"),
                    &termination_bytes,
                ));
                let startup = json(&startup_bytes)?;
                let term = json(&termination_bytes)?;
                check_startup(&startup, engine, &session)?;
                let completed = check_termination(&term, &startup)?;
                let pid = u32::try_from(number(&startup, "process_id")?)
                    .map_err(|_| invalid("provider PID exceeds native range"))?;
                if !ids.insert(pid) {
                    return Err(invalid("native pair reused a process identity"));
                }
                // Compare conversion provenance with the exact copied export
                // manifest bytes rather than trusting additional receipt strings.
                let export = pin(
                    &owner.snapshot.pins,
                    engine.artifact(NativeArtifactRole::ExportManifest)?,
                )?;
                let manifest = json(&read_bytes(
                    export
                        .file
                        .try_clone()
                        .map_err(|_| invalid("cannot clone export pin"))?,
                    64 * 1024,
                )?)?;
                let p = field(&startup, "profile")?;
                for (observed, declared) in [
                    ("source_weights_protobuf_sha256", "source_protobuf_sha256"),
                    ("converter_commit", "converter_commit"),
                    ("converter_binary_sha256", "converter_binary_sha256"),
                ] {
                    equals(p, observed, text(&manifest, declared)?)?;
                }
                sessions.push(NativeProviderSessionAudit {
                    role: match role {
                        NativeEngineRole::Baseline => "baseline",
                        NativeEngineRole::Candidate => "candidate",
                    }
                    .into(),
                    process_id: pid,
                    process_run_id: session,
                    startup_sha256: format!("{:x}", Sha256::digest(&startup_bytes)),
                    termination_sha256: format!("{:x}", Sha256::digest(&termination_bytes)),
                    completed_by_runtime: completed,
                    actual_cpu_inference_observed: true,
                    physical_drain: "confirmed".into(),
                });
            }
        }
        Ok(sessions)
    }
}
