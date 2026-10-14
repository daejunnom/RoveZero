//! One original G0 owner across Collection and a separate P/C numerical stage.
//! Neither successful numerical stage grants startup, arena or training access.
use rz_arena::pals_collect_v3::{
    G0NumericalFailureV5, preflight_g0_deployment_intent_original_v5,
    run_g0_bootstrap_numerical_original_v5, run_g0_deployment_numerical_sealed_original_v5,
    seal_g0_deployment_preflighted_intent_original_v5,
};
use serde::Serialize;
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Debug, PartialEq, Eq)]
struct Arguments {
    collection: PathBuf,
    collection_sha256: String,
    intent: PathBuf,
    intent_sha256: String,
}
impl Arguments {
    fn parse(args: &[String]) -> Option<Self> {
        if args.len() != 9
            || args[0] != "--execute"
            || args[1] != "--collection-registration"
            || args[3] != "--collection-registration-sha256"
            || args[5] != "--deployment-intent"
            || args[7] != "--deployment-intent-sha256"
            || !Path::new(&args[2]).is_absolute()
            || !Path::new(&args[6]).is_absolute()
            || args[2] == args[6]
            || [&args[4], &args[8]]
                .iter()
                .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return None;
        }
        Some(Self {
            collection: args[2].clone().into(),
            collection_sha256: args[4].clone(),
            intent: args[6].clone().into(),
            intent_sha256: args[8].clone(),
        })
    }
}

#[derive(Serialize)]
struct FailedProcess<'a> {
    schema: &'static str,
    stage: &'static str,
    index: usize,
    receipt: &'a rz_arena::process::ProcessReceipt,
    transport: &'a rz_arena::process::OriginalProcessObservation,
    stdout: &'a [u8],
    stderr: &'a [u8],
    pending_child_retained: bool,
}

/// Match the existing game owner's unresolved-cleanup custody rule. Retain
/// the whole opaque owner, including its original descriptors, until process
/// termination; never turn incomplete cleanup into a normal dropping path.
struct FailureCustody<T> {
    original: Option<Box<T>>,
    quarantine_required: bool,
}
impl<T> FailureCustody<T> {
    fn new(original: T, quarantine_required: bool) -> Self {
        Self {
            original: Some(Box::new(original)),
            quarantine_required,
        }
    }
    fn get(&self) -> &T {
        self.original.as_ref().expect("original failure custody")
    }
}
impl<T> Drop for FailureCustody<T> {
    fn drop(&mut self) {
        if self.quarantine_required
            && let Some(original) = self.original.take()
        {
            let _quarantined_until_process_termination = Box::leak(original);
        }
    }
}

fn needs_quarantine(failure: &G0NumericalFailureV5) -> bool {
    failure.processes.iter().any(|process| {
        process.process().pending_child.is_some()
            || process.process().receipt.group_cleanup == rz_arena::CleanupStatus::Unverified
    })
}

/// Preserve exact bounded raw bytes, including non-UTF8, without cloning them
/// into a second Value tree or substituting a shortened Debug representation.
fn preserve_failure(stage: &'static str, failure: &G0NumericalFailureV5) -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    for (index, process) in failure.processes.iter().enumerate() {
        let raw = process.process();
        serde_json::to_writer(
            &mut stderr,
            &FailedProcess {
                schema: "rz-pals-g0-original-failed-process/1",
                stage,
                index,
                receipt: &raw.receipt,
                transport: process.observation(),
                stdout: &raw.stdout,
                stderr: &raw.stderr,
                pending_child_retained: raw.pending_child.is_some(),
            },
        )?;
        stderr.write_all(b"\n")?;
    }
    serde_json::to_writer(
        &mut stderr,
        &serde_json::json!({
            "schema":"rz-pals-g0-original-failure-evidence/1", "stage":stage,
            "child_receipts":failure.child_receipts,
            "preservation_reports":failure.preservation_reports,
            "registration":failure.registration_pin,
            "charged_read_bytes":failure.charged_read_bytes,
            "read_usage_uncertain":failure.read_usage_uncertain,
            "automatic_retry":false
        }),
    )?;
    stderr.write_all(b"\n")?;
    stderr.flush()
}

fn halted(stage: &str, error: &dyn std::fmt::Display, extra: serde_json::Value) -> ExitCode {
    println!(
        "{}",
        serde_json::json!({
            "schema":"rz-pals-g0-collection-deployment-result/1", "status":"halted",
            "stage":stage, "error":error.to_string(), "evidence":extra,
            "automatic_retry":false, "numerical_only":true,
            "startup_admission_not_issued":true, "training_authority_issued":false,
            "arena_accepted":false
        })
    );
    ExitCode::FAILURE
}

fn execute(args: Arguments, cancel: Arc<AtomicBool>) -> ExitCode {
    let intent = match preflight_g0_deployment_intent_original_v5(
        &args.intent,
        &args.intent_sha256,
        Arc::clone(&cancel),
    ) {
        Ok(intent) => intent,
        Err(failure) => {
            return halted(
                "deployment_intent_preflight",
                &failure.error,
                serde_json::json!({"intent":failure.intent_pin,
                    "charged_read_bytes":failure.charged_read_bytes,
                    "read_usage_uncertain":failure.read_usage_uncertain,
                    "collection_started":false}),
            );
        }
    };
    let collection = match run_g0_bootstrap_numerical_original_v5(
        &args.collection,
        &args.collection_sha256,
        Arc::clone(&cancel),
    ) {
        Ok(owner) => Arc::new(owner),
        Err(failure) => {
            let quarantine = needs_quarantine(&failure);
            let custody = FailureCustody::new((failure, intent), quarantine);
            let (failure, intent) = custody.get();
            let preservation = preserve_failure("collection", failure);
            return halted(
                "collection",
                &failure.error,
                serde_json::json!({"quarantine_required":quarantine,
                    "deployment_intent_preflight":intent.observation(),
                    "opaque_failure_owner_retained":quarantine,
                    "failure_evidence_written":preservation.is_ok(),
                    "preservation_error":preservation.err().map(|e| e.to_string())}),
            );
        }
    };
    // The old numerical window is finished. This only seals the actual, now
    // known acceptance pin; it neither reconstructs nor renews the old owner.
    let seal = match seal_g0_deployment_preflighted_intent_original_v5(
        &intent,
        &collection,
        Arc::clone(&cancel),
    ) {
        Ok(seal) => seal,
        Err(failure) => {
            return halted(
                "deployment_intent_seal",
                &failure.error,
                serde_json::json!({"collection":collection.parent_receipt(),
                    "intent":failure.intent_pin, "charged_read_bytes":failure.charged_read_bytes,
                    "read_usage_uncertain":failure.read_usage_uncertain}),
            );
        }
    };
    // Both the original registration descriptor and Collection owner remain
    // alive. The new stage joins its opened registration to this exact seal FD.
    match run_g0_deployment_numerical_sealed_original_v5(&seal, Arc::clone(&collection), cancel) {
        Ok(deployment) => {
            println!(
                "{}",
                serde_json::json!({
                    "schema":"rz-pals-g0-collection-deployment-result/1", "status":"numerical_stages_returned",
                    "collection":collection.parent_receipt(), "deployment_seal":seal.observation(),
                    "deployment_intent_preflight":intent.observation(),
                    "deployment":deployment.parent_receipt(), "automatic_retry":false,
                    "numerical_only":true, "startup_admission_not_issued":true,
                    "training_authority_issued":false, "arena_accepted":false
                })
            );
            ExitCode::SUCCESS
        }
        Err(failure) => {
            let quarantine = needs_quarantine(&failure.error);
            let custody = FailureCustody::new((failure, seal, collection, intent), quarantine);
            let (failure, seal, collection, intent) = custody.get();
            let preservation = preserve_failure("deployment", &failure.error);
            halted(
                "deployment",
                &failure.error.error,
                serde_json::json!({"collection":collection.parent_receipt(),
                    "deployment_intent_preflight":intent.observation(),
                    "deployment_seal":seal.observation(),
                    "quarantine_required":quarantine,
                    "opaque_failure_owner_retained":quarantine,
                    "failure_evidence_written":preservation.is_ok(),
                    "preservation_error":preservation.err().map(|e| e.to_string())}),
            )
        }
    }
}

fn main() -> ExitCode {
    let raw: Vec<_> = std::env::args().skip(1).collect();
    let Some(args) = Arguments::parse(&raw) else {
        eprintln!(
            "usage: rz-pals-g0-collection-deployment-numerical --execute --collection-registration /absolute/collection.json --collection-registration-sha256 <sha256> --deployment-intent /absolute/deployment-intent.json --deployment-intent-sha256 <sha256>; default invocation runs no NN or training"
        );
        return ExitCode::FAILURE;
    };
    let cancel = Arc::new(AtomicBool::new(false));
    #[cfg(target_os = "linux")]
    let mut hooks = Vec::new();
    #[cfg(target_os = "linux")]
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        match signal_hook::flag::register(signal, Arc::clone(&cancel)) {
            Ok(id) => hooks.push(id),
            Err(error) => {
                for id in hooks {
                    signal_hook::low_level::unregister(id);
                }
                return halted("signal_registration", &error, serde_json::Value::Null);
            }
        }
    }
    // Unsupported platforms refuse inside the original library before opening
    // registrations. There is no fallback to a decoded or mock owner here.
    let result = execute(args, cancel);
    #[cfg(target_os = "linux")]
    for id in hooks {
        signal_hook::low_level::unregister(id);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> Vec<String> {
        let root = std::env::temp_dir();
        vec![
            "--execute".into(),
            "--collection-registration".into(),
            root.join("g0-collection.json")
                .to_string_lossy()
                .into_owned(),
            "--collection-registration-sha256".into(),
            "a".repeat(64),
            "--deployment-intent".into(),
            root.join("g0-deployment-intent.json")
                .to_string_lossy()
                .into_owned(),
            "--deployment-intent-sha256".into(),
            "b".repeat(64),
        ]
    }

    #[test]
    fn explicit_two_stage_declarations_are_required_before_execution() {
        assert!(Arguments::parse(&[]).is_none());
        let mut args = valid();
        assert!(Arguments::parse(&args).is_some());
        args[0] = "--resume-owner-from-json".into();
        assert!(Arguments::parse(&args).is_none());
    }

    #[test]
    fn relative_paths_bad_pins_aliases_and_extra_authority_are_refused() {
        for index in [2, 6] {
            let mut args = valid();
            args[index] = "relative.json".into();
            assert!(Arguments::parse(&args).is_none());
        }
        for index in [4, 8] {
            let mut args = valid();
            args[index] = "g".repeat(64);
            assert!(Arguments::parse(&args).is_none());
        }
        let mut args = valid();
        args[6] = args[2].clone();
        assert!(Arguments::parse(&args).is_none());
        let mut args = valid();
        args.push("--train".into());
        assert!(Arguments::parse(&args).is_none());
    }

    #[test]
    fn only_verified_cleanup_releases_the_failure_custody() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Dropped(Arc<AtomicUsize>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let count = Arc::new(AtomicUsize::new(0));
        drop(FailureCustody::new(Dropped(Arc::clone(&count)), false));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        drop(FailureCustody::new(Dropped(Arc::clone(&count)), true));
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
