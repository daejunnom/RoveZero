//! One bounded JSON V task, --candidate-only fresh single-candidate check, or
//! --prepare-semantic Rules-only preparation, or --line-continuation factual
//! whole-line/endpoint check. Modes are exclusive.
//! Private V tasks alone can explicitly select --cpu-ordering=legal-see-v1.
//! Native PALS/CPU_R and other private modes keep their historical selection.
//! No child UCI engine/GPU/model/teacher is launched.
use rz_search::cpu::CpuOrderingPolicy;
use rz_uci::pals_cpu_task::candidate::{
    self, CandidateRawReport, CandidateReceipt, CandidateTaskError,
};
use rz_uci::pals_cpu_task::continuation::{self, ContinuationError, ContinuationReceipt};
use rz_uci::pals_cpu_task::semantic::{self, SemanticError, SemanticReceipt};
use rz_uci::pals_cpu_task::{
    CpuTaskError, MAX_REQUEST_BYTES, MAX_SELF_BINARY_BYTES, MAX_WALL_TIME_MS, capabilities,
    capabilities_with_ordering, dispatch_started, dispatch_started_with_ordering,
    request_admission, request_admission_with_ordering,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn failure(stage: &'static str, message: impl std::fmt::Display) -> CpuTaskError {
    CpuTaskError {
        code: "cpu_task_cli_failed",
        stage,
        message: message
            .to_string()
            .chars()
            .take(512)
            .collect::<String>()
            .into_boxed_str(),
        known_nodes: None,
        failed_check_work: None,
        baseline: None,
        after: None,
        elapsed_ms: None,
        deadline_exceeded: false,
        output_limit: 1024,
    }
}

fn read_request(started: Instant) -> Result<Vec<u8>, CpuTaskError> {
    // A blocked stdin cannot defeat the CLI's finite outer bound. This worker
    // owns only stdin, creates no child, and dies with a nonzero process exit.
    let (send, receive) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("cpu-task-stdin".into())
        .spawn(move || {
            let mut bytes = Vec::new();
            let result = io::stdin()
                .lock()
                .take(MAX_REQUEST_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map(|_| bytes);
            let _ = send.send(result);
        })
        .map_err(|e| failure("stdin", e))?;
    let remaining = Duration::from_millis(MAX_WALL_TIME_MS).saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(failure("stdin", "finite outer stdin deadline expired"));
    }
    let bytes = receive
        .recv_timeout(remaining)
        .map_err(|e| failure("stdin", e))?
        .map_err(|e| failure("stdin", e))?;
    if bytes.is_empty() || bytes.len() > MAX_REQUEST_BYTES {
        return Err(failure("stdin", "request exceeds 1..=512KiB admission"));
    }
    Ok(bytes)
}

fn own_binary_sha(deadline: Instant) -> Result<String, CpuTaskError> {
    // On Linux the proc handle names the loaded executable inode even if a
    // path is replaced after launch. Other hosts hash current_exe's path: this
    // does not provide Linux's loaded-inode guarantee, and callers pin the scope.
    #[cfg(target_os = "linux")]
    let path = std::path::PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe().map_err(|e| failure("self_binary", e))?;
    let mut file = File::open(path).map_err(|e| failure("self_binary", e))?;
    let metadata = file.metadata().map_err(|e| failure("self_binary", e))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_SELF_BINARY_BYTES {
        return Err(failure(
            "self_binary",
            "loaded executable exceeds bounded regular-file admission",
        ));
    }
    let mut sha = Sha256::new();
    let mut block = [0_u8; 32 * 1024];
    let mut total = 0_u64;
    loop {
        if Instant::now() >= deadline {
            return Err(failure(
                "self_binary",
                "absolute request deadline expired while hashing own image",
            ));
        }
        let read = file
            .read(&mut block)
            .map_err(|e| failure("self_binary", e))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| failure("self_binary", "executable byte counter overflow"))?;
        if total > MAX_SELF_BINARY_BYTES || total > metadata.len() {
            return Err(failure(
                "self_binary",
                "executable changed or exceeded finite byte bound",
            ));
        }
        sha.update(&block[..read]);
    }
    let after = file.metadata().map_err(|e| failure("self_binary", e))?;
    if total != metadata.len()
        || after.len() != metadata.len()
        || after.modified().ok() != metadata.modified().ok()
    {
        return Err(failure(
            "self_binary",
            "executable changed during identity verification",
        ));
    }
    Ok(format!("{:x}", sha.finalize()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CliMode {
    Verifier,
    Candidate,
    Semantic,
    Continuation,
}

struct CliContext {
    started: Instant,
    deadline: Instant,
    output_limit: usize,
    mode: CliMode,
    // Retain actual work if final stdout delivery fails. Deadline exhaustion
    // still grants no stderr grace; supervisors preserve partial pipe evidence.
    candidate_report: Option<Box<CandidateRawReport>>,
    // Only an actual completed preparation is retained. Its presence in a
    // transport error is diagnostic, never successful delivery or CPU work.
    semantic_receipt: Option<Box<SemanticReceipt>>,
    continuation_receipt: Option<Box<ContinuationReceipt>>,
}

enum CliError {
    V(CpuTaskError),
    Candidate(CandidateTaskError),
    Semantic(SemanticError),
    Continuation(ContinuationError),
}

impl CliContext {
    fn transport_error(&self, error: CpuTaskError) -> CliError {
        match self.mode {
            CliMode::Candidate => {
                let mut error = CandidateTaskError::new(error.stage, error.message);
                if let Some(report) = &self.candidate_report {
                    error = error.with_report(report);
                }
                CliError::Candidate(error)
            }
            CliMode::Semantic => {
                let mut error = SemanticError::new(error.stage, error.message);
                error.receipt = self.semantic_receipt.clone();
                CliError::Semantic(error)
            }
            CliMode::Continuation => {
                let mut error = ContinuationError::new(error.stage, error.message);
                if let Some(receipt) = &self.continuation_receipt {
                    error = error.with_receipt(receipt);
                }
                CliError::Continuation(error)
            }
            CliMode::Verifier => CliError::V(error),
        }
    }
}

fn cli_binary_scope() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux_loaded_executable_inode"
    } else {
        "current_exe_path_hash"
    }
}

fn arguments(args: &[String]) -> Result<(CliMode, bool), CpuTaskError> {
    match args {
        [] => Ok((CliMode::Verifier, false)),
        [a] if a == "--capabilities" => Ok((CliMode::Verifier, true)),
        [a] if a == "--candidate-only" => Ok((CliMode::Candidate, false)),
        [a, b] if a == "--candidate-only" && b == "--capabilities" => {
            Ok((CliMode::Candidate, true))
        }
        [a] if a == "--prepare-semantic" => Ok((CliMode::Semantic, false)),
        [a, b] if a == "--prepare-semantic" && b == "--capabilities" => {
            Ok((CliMode::Semantic, true))
        }
        [a] if a == "--line-continuation" => Ok((CliMode::Continuation, false)),
        [a, b] if a == "--line-continuation" && b == "--capabilities" => {
            Ok((CliMode::Continuation, true))
        }
        _ => Err(failure(
            "arguments",
            "supported modes: no arguments, --capabilities, --candidate-only [--capabilities], --prepare-semantic [--capabilities], --line-continuation [--capabilities]; modes are exclusive and ordered",
        )),
    }
}

fn arguments_with_ordering(
    args: &[String],
) -> Result<(CliMode, bool, CpuOrderingPolicy), CpuTaskError> {
    match args {
        [flag] if flag == "--cpu-ordering=legal-see-v1" => {
            Ok((CliMode::Verifier, false, CpuOrderingPolicy::LegalSeeV1))
        }
        [flag, capability]
            if flag == "--cpu-ordering=legal-see-v1" && capability == "--capabilities" =>
        {
            Ok((CliMode::Verifier, true, CpuOrderingPolicy::LegalSeeV1))
        }
        _ => arguments(args)
            .map(|(mode, capabilities)| (mode, capabilities, CpuOrderingPolicy::LegacyMvvLvaV1)),
    }
}

fn annotate(mut error: CpuTaskError, context: &CliContext) -> CpuTaskError {
    error.elapsed_ms = Some(
        context
            .started
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64,
    );
    error.deadline_exceeded = Instant::now() >= context.deadline;
    error.output_limit = context.output_limit;
    error
}

fn run(context: &mut CliContext) -> Result<Vec<u8>, CliError> {
    let started = context.started;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    context.mode = match args.first().map(String::as_str) {
        Some("--candidate-only") => CliMode::Candidate,
        Some("--prepare-semantic") => CliMode::Semantic,
        Some("--line-continuation") => CliMode::Continuation,
        _ => CliMode::Verifier,
    };
    let (mode, show_capabilities, ordering) =
        arguments_with_ordering(&args).map_err(|e| context.transport_error(e))?;
    context.mode = mode;
    if show_capabilities {
        let deadline = started
            .checked_add(Duration::from_millis(MAX_WALL_TIME_MS))
            .ok_or_else(|| {
                context.transport_error(failure("admission", "capability deadline overflow"))
            })?;
        let binary = own_binary_sha(deadline).map_err(|e| context.transport_error(e))?;
        let capabilities = match mode {
            CliMode::Verifier => {
                if ordering == CpuOrderingPolicy::LegacyMvvLvaV1 {
                    capabilities().map_err(CliError::V)?
                } else {
                    capabilities_with_ordering(ordering).map_err(CliError::V)?
                }
            }
            CliMode::Candidate => candidate::capabilities().map_err(CliError::Candidate)?,
            CliMode::Semantic => semantic::capabilities().map_err(CliError::Semantic)?,
            CliMode::Continuation => {
                continuation::capabilities().map_err(CliError::Continuation)?
            }
        };
        if matches!(mode, CliMode::Semantic | CliMode::Continuation) {
            context.output_limit = 8192;
        }
        let mut value: serde_json::Value = serde_json::from_slice(&capabilities)
            .map_err(|e| context.transport_error(failure("capabilities", e)))?;
        if mode == CliMode::Semantic {
            value["current_binary_sha256"] = json!(binary);
        } else {
            value["cpu_binary_sha256"] = json!(binary);
        }
        if mode != CliMode::Verifier {
            value["binary_pin_scope"] = json!(cli_binary_scope());
        }
        let mut result = serde_json::to_vec(&value)
            .map_err(|e| context.transport_error(failure("capabilities", e)))?;
        result.push(b'\n');
        if matches!(mode, CliMode::Semantic | CliMode::Continuation)
            && result.len() > context.output_limit
        {
            return Err(context.transport_error(failure(
                "capabilities",
                "CLI private-check capabilities exceed the finite 8KiB output bound",
            )));
        }
        return Ok(result);
    }
    let bytes = read_request(started).map_err(|e| context.transport_error(e))?;
    let (deadline, output_limit) = match mode {
        CliMode::Verifier => {
            let admission = if ordering == CpuOrderingPolicy::LegacyMvvLvaV1 {
                request_admission(&bytes, started).map_err(CliError::V)?
            } else {
                request_admission_with_ordering(&bytes, started, ordering).map_err(CliError::V)?
            };
            (admission.deadline, admission.output_limit)
        }
        CliMode::Candidate => {
            let admission =
                candidate::request_admission(&bytes, started).map_err(CliError::Candidate)?;
            (admission.deadline, admission.output_limit)
        }
        CliMode::Semantic => {
            let admission = admit_semantic(&bytes, context)?;
            (admission.deadline, admission.output_limit)
        }
        CliMode::Continuation => {
            let admission = admit_continuation(&bytes, context)?;
            (admission.deadline, admission.output_limit)
        }
    };
    context.deadline = deadline;
    context.output_limit = output_limit;
    let binary = own_binary_sha(context.deadline).map_err(|e| context.transport_error(e))?;
    if mode == CliMode::Verifier {
        return if ordering == CpuOrderingPolicy::LegacyMvvLvaV1 {
            dispatch_started(&bytes, &binary, started).map_err(CliError::V)
        } else {
            dispatch_started_with_ordering(&bytes, &binary, started, ordering).map_err(CliError::V)
        };
    }
    if mode == CliMode::Semantic {
        return prepare_semantic(&bytes, &binary, context);
    }
    if mode == CliMode::Continuation {
        return check_continuation(&bytes, &binary, context);
    }
    let bytes =
        candidate::dispatch_started(&bytes, &binary, started).map_err(CliError::Candidate)?;
    let mut receipt: CandidateReceipt = serde_json::from_slice(&bytes)
        .map_err(|e| context.transport_error(failure("candidate_receipt", e)))?;
    context.candidate_report = Some(Box::new(receipt.report.clone()));
    // This is an actual CLI observation, separate from a request's declared
    // source registration. Linux loaded-inode and other-host path hashes differ.
    receipt.binary_pin_scope = cli_binary_scope().into();
    receipt.elapsed_ms = context
        .started
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let mut bytes = serde_json::to_vec(&receipt)
        .map_err(|e| context.transport_error(failure("candidate_receipt", e)))?;
    bytes.push(b'\n');
    if bytes.len() > context.output_limit {
        return Err(context.transport_error(failure(
            "output_bound",
            "CLI candidate receipt exceeds admitted output bound",
        )));
    }
    if Instant::now() >= context.deadline {
        return Err(context.transport_error(failure(
            "receipt_deadline",
            "original absolute wall allowance expired after CLI image observation",
        )));
    }
    Ok(bytes)
}

fn admit_semantic(
    bytes: &[u8],
    context: &mut CliContext,
) -> Result<semantic::SemanticAdmission, CliError> {
    semantic::request_admission(bytes, context.started).map_err(|error| {
        if error.deadline_exceeded {
            // The module knows the valid request's original clock expired but
            // did not return an admission. Do not retain the larger outer clock
            // as stderr grace. No exact unknown deadline is fabricated here.
            context.deadline = context.deadline.min(context.started);
        }
        CliError::Semantic(error)
    })
}

fn prepare_semantic(
    bytes: &[u8],
    verified_binary: &str,
    context: &mut CliContext,
) -> Result<Vec<u8>, CliError> {
    let output = match semantic::prepare_started(bytes, verified_binary, context.started) {
        Ok(output) => output,
        Err(mut error) => {
            // A module error can contain real completed Rules preparation.
            // Retain it as diagnostic with the actual CLI image observation;
            // never turn the failed run into an admitted preparation.
            if let Some(receipt) = &mut error.receipt {
                receipt.binary_pin_scope = cli_binary_scope().into();
                context.semantic_receipt = Some(receipt.clone());
            }
            return Err(CliError::Semantic(error));
        }
    };
    let mut receipt: SemanticReceipt = serde_json::from_slice(&output)
        .map_err(|e| context.transport_error(failure("semantic_receipt", e)))?;
    receipt.binary_pin_scope = cli_binary_scope().into();
    receipt.elapsed_ms = context
        .started
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    context.semantic_receipt = Some(Box::new(receipt.clone()));
    if receipt.schema != semantic::SEMANTIC_SCHEMA
        || receipt.current_binary_sha256 != verified_binary
        || receipt.cpu_checks != 0
        || receipt.resource_policy.cpu_checks != 0
        || receipt.search_executed
        || receipt.model_executed
        || receipt.training_target_created
        || receipt.product_verifier_enabled
        || receipt.deadline_exceeded
        || receipt.resource_policy.max_output_bytes != context.output_limit
        || context.started.checked_add(Duration::from_millis(
            receipt.resource_policy.max_wall_time_ms,
        )) != Some(context.deadline)
    {
        return Err(context.transport_error(failure(
            "semantic_receipt",
            "preparation identity, no-dispatch scope, or original resource admission differs",
        )));
    }
    let mut output = serde_json::to_vec(&receipt)
        .map_err(|e| context.transport_error(failure("semantic_receipt", e)))?;
    output.push(b'\n');
    if output.len() > context.output_limit {
        return Err(context.transport_error(failure(
            "output_bound",
            "CLI semantic receipt exceeds admitted output bound",
        )));
    }
    if Instant::now() >= context.deadline {
        return Err(context.transport_error(failure(
            "receipt_deadline",
            "original absolute wall allowance expired after CLI image observation",
        )));
    }
    Ok(output)
}

fn admit_continuation(
    bytes: &[u8],
    context: &mut CliContext,
) -> Result<rz_uci::pals_cpu_task::CpuTaskAdmission, CliError> {
    continuation::request_admission(bytes, context.started).map_err(|error| {
        if error.deadline_exceeded {
            // Expired admitted wall time never inherits the larger stdin clock
            // as diagnostic output grace, even when admission returned no value.
            context.deadline = context.deadline.min(context.started);
        }
        CliError::Continuation(error)
    })
}

fn check_continuation(
    bytes: &[u8],
    verified_binary: &str,
    context: &mut CliContext,
) -> Result<Vec<u8>, CliError> {
    let output = match continuation::dispatch_started(bytes, verified_binary, context.started) {
        Ok(output) => output,
        Err(mut error) => {
            if let Some(receipt) = &mut error.receipt {
                receipt.binary_pin_scope = cli_binary_scope().into();
                context.continuation_receipt = Some(receipt.clone());
            }
            return Err(CliError::Continuation(error));
        }
    };
    let mut receipt: ContinuationReceipt = serde_json::from_slice(&output)
        .map_err(|error| context.transport_error(failure("continuation_receipt", error)))?;
    receipt.binary_pin_scope = cli_binary_scope().into();
    receipt.elapsed_ms = context
        .started
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    context.continuation_receipt = Some(Box::new(receipt.clone()));
    if receipt.schema != continuation::CONTINUATION_SCHEMA
        || receipt.cpu_binary_sha256 != verified_binary
        || receipt.cpu_calls > 1
        || receipt.cpu_calls != u8::from(receipt.report.is_some())
        || receipt.fresh_engine != receipt.report.is_some()
        || receipt.conditions.resource_policy.max_checks != 1
        || receipt.conditions.resource_policy.max_output_bytes != context.output_limit
        || !receipt.line_rules_validated
        || receipt.line.is_empty()
        || receipt.line.len() > continuation::MAX_LINE_PLIES
        || receipt.training_target_created
        || receipt.product_verifier_enabled
        || receipt.deadline_exceeded
        || context.started.checked_add(Duration::from_millis(
            receipt.conditions.resource_policy.max_wall_time_ms,
        )) != Some(context.deadline)
    {
        return Err(context.transport_error(failure(
            "continuation_receipt",
            "actual continuation identity, one-check scope or original clock differs",
        )));
    }
    let mut output = serde_json::to_vec(&receipt)
        .map_err(|error| context.transport_error(failure("continuation_receipt", error)))?;
    output.push(b'\n');
    if output.len() > context.output_limit {
        return Err(context.transport_error(failure(
            "output_bound",
            "CLI continuation receipt exceeds admitted output bound",
        )));
    }
    if Instant::now() >= context.deadline {
        return Err(context.transport_error(failure(
            "receipt_deadline",
            "original absolute wall allowance expired after CLI image observation",
        )));
    }
    Ok(output)
}

fn bounded_write<W: Write + Send + 'static>(
    mut writer: W,
    bytes: Vec<u8>,
    deadline: Instant,
) -> Result<(), CpuTaskError> {
    if Instant::now() >= deadline {
        return Err(failure(
            "output_deadline",
            "original absolute wall allowance expired before output",
        ));
    }
    let (send, receive) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("cpu-task-output".into())
        .spawn(move || {
            let result = writer.write_all(&bytes).and_then(|_| writer.flush());
            let _ = send.send(result);
        })
        .map_err(|e| failure("output", e))?;
    receive
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|e| failure("output_deadline", e))?
        .map_err(|e| failure("output", e))?;
    if Instant::now() >= deadline {
        return Err(failure(
            "output_deadline",
            "late output cannot be accepted as success",
        ));
    }
    Ok(())
}

fn output_file(stdout: bool) -> Result<File, CpuTaskError> {
    // Duplicate the native handle safely and write through File. A blocked
    // worker must never hold stdio's global lock during Rust runtime cleanup.
    #[cfg(unix)]
    {
        use std::os::fd::AsFd;
        let handle = if stdout {
            io::stdout().as_fd().try_clone_to_owned()
        } else {
            io::stderr().as_fd().try_clone_to_owned()
        };
        handle.map(File::from).map_err(|e| failure("output", e))
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsHandle;
        let handle = if stdout {
            io::stdout().as_handle().try_clone_to_owned()
        } else {
            io::stderr().as_handle().try_clone_to_owned()
        };
        handle.map(File::from).map_err(|e| failure("output", e))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = stdout;
        Err(failure(
            "output",
            "deadline-bounded native output is unsupported on this target",
        ))
    }
}

fn main() -> ExitCode {
    let started = Instant::now();
    let Some(deadline) = started.checked_add(Duration::from_millis(MAX_WALL_TIME_MS)) else {
        return ExitCode::from(1);
    };
    let mut context = CliContext {
        started,
        deadline,
        output_limit: 1024,
        mode: CliMode::Verifier,
        candidate_report: None,
        semantic_receipt: None,
        continuation_receipt: None,
    };
    match run(&mut context) {
        Ok(bytes) => {
            match output_file(true).and_then(|file| bounded_write(file, bytes, context.deadline)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) if context.mode != CliMode::Verifier => {
                    write_error(context.transport_error(error), &context)
                }
                Err(_) => ExitCode::from(1),
            }
        }
        Err(error) => write_error(error, &context),
    }
}

fn write_error(error: CliError, context: &CliContext) -> ExitCode {
    let mut bytes = match error {
        CliError::V(error) => {
            let error = annotate(error, context);
            // One bounded diagnostic on stderr; success emits no diagnostics.
            // If the richer receipt does not fit, retain honest known/unknown
            // work counters rather than replacing the error with a fake result.
            let mut bytes = serde_json::to_vec(&error)
                .unwrap_or_else(|_| b"{\"code\":\"cpu_task_error_serialization_failed\"}".to_vec());
            if bytes.len() + 1 > error.output_limit {
                bytes=serde_json::to_vec(&json!({"code":error.code,"stage":error.stage,
                    "known_nodes":error.known_nodes,"failed_check_work":error.failed_check_work,
                    "baseline_present":error.baseline.is_some(),"after_present":error.after.is_some(),
                    "elapsed_ms":error.elapsed_ms,"deadline_exceeded":error.deadline_exceeded,
                    "full_error_omitted_for_output_bound":true}))
                    .unwrap_or_else(|_|b"{\"code\":\"cpu_task_failed\"}".to_vec());
            }
            bytes
        }
        CliError::Candidate(mut error) => {
            error.elapsed_ms = Some(
                context
                    .started
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64,
            );
            error.deadline_exceeded = Instant::now() >= context.deadline;
            error.output_limit = context.output_limit;
            let mut bytes = serde_json::to_vec(&error).unwrap_or_else(|_| {
                b"{\"code\":\"cpu_candidate_error_serialization_failed\"}".to_vec()
            });
            if bytes.len() + 1 > error.output_limit {
                bytes = serde_json::to_vec(
                    &json!({"schema":error.schema,"code":error.code,"stage":error.stage,
                    "known_nodes":error.known_nodes,"failed_check_work":error.failed_check_work,
                    "report_present":error.report.is_some(),"cpu_calls":error.cpu_calls,
                    "elapsed_ms":error.elapsed_ms,"deadline_exceeded":error.deadline_exceeded,
                    "full_error_omitted_for_output_bound":true}),
                )
                .unwrap_or_else(|_| b"{\"code\":\"cpu_candidate_task_failed\"}".to_vec());
            }
            bytes
        }
        CliError::Semantic(mut error) => {
            error.elapsed_ms = Some(
                context
                    .started
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64,
            );
            error.deadline_exceeded |= Instant::now() >= context.deadline;
            error.output_limit = context.output_limit;
            // Prefer the original module diagnostic; a final delivery failure
            // receives the actual preparation captured before output dispatch.
            if error.receipt.is_none() {
                error.receipt = context.semantic_receipt.clone();
            }
            let mut bytes = serde_json::to_vec(&error).unwrap_or_else(|_| {
                b"{\"code\":\"semantic_branch_error_serialization_failed\"}".to_vec()
            });
            if bytes.len() + 1 > error.output_limit {
                bytes = serde_json::to_vec(&json!({
                    "schema": error.schema, "code": error.code, "stage": error.stage,
                    "cpu_checks": error.cpu_checks, "receipt_present": error.receipt.is_some(),
                    "context_sha256": error.receipt.as_ref().map(|r| &r.context_sha256),
                    "branch_meaning_sha256": error.receipt.as_ref().map(|r| &r.branch_meaning_sha256),
                    "current_binary_sha256": error.receipt.as_ref().map(|r| &r.current_binary_sha256),
                    "binary_pin_scope": error.receipt.as_ref().map(|r| &r.binary_pin_scope),
                    "elapsed_ms": error.elapsed_ms, "deadline_exceeded": error.deadline_exceeded,
                    "full_error_omitted_for_output_bound": true
                })).unwrap_or_else(|_| b"{\"code\":\"semantic_branch_failed\"}".to_vec());
            }
            bytes
        }
        CliError::Continuation(mut error) => {
            error.elapsed_ms = Some(
                context
                    .started
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64,
            );
            error.deadline_exceeded |= Instant::now() >= context.deadline;
            error.output_limit = context.output_limit;
            if error.receipt.is_none() {
                if let Some(receipt) = &context.continuation_receipt {
                    error = error.with_receipt(receipt);
                }
            }
            let mut bytes = serde_json::to_vec(&error).unwrap_or_else(|_| {
                b"{\"code\":\"cpu_line_continuation_error_serialization_failed\"}".to_vec()
            });
            if bytes.len() + 1 > error.output_limit {
                bytes = serde_json::to_vec(&json!({"schema":error.schema,"code":error.code,
                    "stage":error.stage,"known_nodes":error.known_nodes,
                    "failed_check_work":error.failed_check_work,"report_present":error.report.is_some(),
                    "receipt_present":error.receipt.is_some(),"cpu_calls":error.cpu_calls,
                    "context_sha256":error.receipt.as_ref().map(|receipt|&receipt.context_sha256),
                    "line_sha256":error.receipt.as_ref().map(|receipt|&receipt.line_sha256),
                    "cpu_binary_sha256":error.receipt.as_ref().map(|receipt|&receipt.cpu_binary_sha256),
                    "binary_pin_scope":error.receipt.as_ref().map(|receipt|&receipt.binary_pin_scope),
                    "elapsed_ms":error.elapsed_ms,"deadline_exceeded":error.deadline_exceeded,
                    "full_error_omitted_for_output_bound":true}))
                    .unwrap_or_else(|_|b"{\"code\":\"cpu_line_continuation_failed\"}".to_vec());
            }
            if bytes.len() + 1 > error.output_limit {
                // Identity-rich diagnostics may need a smaller work-only
                // diagnostic under the same original admitted output bound.
                bytes = serde_json::to_vec(&json!({"schema":error.schema,"code":error.code,
                    "stage":error.stage,"known_nodes":error.known_nodes,
                    "failed_check_work":error.failed_check_work,"cpu_calls":error.cpu_calls,
                    "report_present":error.report.is_some(),"receipt_present":error.receipt.is_some(),
                    "elapsed_ms":error.elapsed_ms,"deadline_exceeded":error.deadline_exceeded,
                    "full_error_omitted_for_output_bound":true}))
                    .unwrap_or_else(|_|b"{\"code\":\"cpu_line_continuation_failed\"}".to_vec());
            }
            bytes
        }
    };
    bytes.push(b'\n');
    // No extra output grace after the original absolute wall allowance.
    let _ = output_file(false).and_then(|file| bounded_write(file, bytes, context.deadline));
    ExitCode::from(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BlockedWriter(mpsc::Receiver<()>);
    impl Write for BlockedWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            self.0
                .recv()
                .map_err(|_| io::Error::other("test writer release failed"))?;
            Err(io::Error::other("released test writer"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn unread_output_cannot_block_beyond_the_absolute_deadline() {
        let (send, receive) = mpsc::sync_channel(1);
        let deadline = Instant::now() + Duration::from_millis(20);
        let error = bounded_write(BlockedWriter(receive), vec![b'x'; 65536], deadline).unwrap_err();
        assert_eq!(error.stage, "output_deadline");
        let _ = send.send(());
    }

    #[test]
    fn cli_failure_uses_admitted_deadline_and_actual_elapsed() {
        let started = Instant::now() - Duration::from_millis(25);
        let context = CliContext {
            started,
            deadline: started + Duration::from_millis(1),
            output_limit: 4096,
            mode: CliMode::Verifier,
            candidate_report: None,
            semantic_receipt: None,
            continuation_receipt: None,
        };
        let error = annotate(failure("self_binary", "deadline expired"), &context);
        assert!(error.deadline_exceeded);
        assert!(error.elapsed_ms.unwrap() >= 25);
        assert_eq!(error.output_limit, 4096);
        assert_eq!(error.known_nodes, None);
    }

    #[test]
    fn candidate_mode_is_explicit_and_never_replaces_v_capabilities() {
        assert_eq!(arguments(&[]).unwrap(), (CliMode::Verifier, false));
        assert_eq!(
            arguments(&["--capabilities".into()]).unwrap(),
            (CliMode::Verifier, true)
        );
        assert_eq!(
            arguments(&["--candidate-only".into()]).unwrap(),
            (CliMode::Candidate, false)
        );
        assert_eq!(
            arguments(&["--candidate-only".into(), "--capabilities".into()]).unwrap(),
            (CliMode::Candidate, true)
        );
        assert!(arguments(&["--capabilities".into(), "--candidate-only".into()]).is_err());
        let v: serde_json::Value = serde_json::from_slice(&capabilities().unwrap()).unwrap();
        let candidate: serde_json::Value =
            serde_json::from_slice(&candidate::capabilities().unwrap()).unwrap();
        assert_eq!(v["schema"], "rz-pals-private-cpu-task/1");
        assert_eq!(v["max_checks"], 2);
        assert_eq!(candidate["schema"], candidate::CANDIDATE_SCHEMA);
        assert_eq!(candidate["max_checks"], 1);
    }

    #[test]
    fn private_see_flag_is_exclusive_and_does_not_change_legacy_modes() {
        assert_eq!(
            arguments_with_ordering(&[]).unwrap(),
            (CliMode::Verifier, false, CpuOrderingPolicy::LegacyMvvLvaV1)
        );
        assert_eq!(
            arguments_with_ordering(&["--capabilities".into()]).unwrap(),
            (CliMode::Verifier, true, CpuOrderingPolicy::LegacyMvvLvaV1)
        );
        assert_eq!(
            arguments_with_ordering(&["--cpu-ordering=legal-see-v1".into()]).unwrap(),
            (CliMode::Verifier, false, CpuOrderingPolicy::LegalSeeV1)
        );
        assert_eq!(
            arguments_with_ordering(&[
                "--cpu-ordering=legal-see-v1".into(),
                "--capabilities".into()
            ])
            .unwrap(),
            (CliMode::Verifier, true, CpuOrderingPolicy::LegalSeeV1)
        );
        for arguments in [
            vec!["--cpu-ordering=unknown"],
            vec!["--cpu-ordering=legacy-mvv-lva-v1"],
            vec!["--capabilities", "--cpu-ordering=legal-see-v1"],
            vec!["--candidate-only", "--cpu-ordering=legal-see-v1"],
            vec!["--cpu-ordering=legal-see-v1", "--candidate-only"],
            vec!["--prepare-semantic", "--cpu-ordering=legal-see-v1"],
            vec!["--line-continuation", "--cpu-ordering=legal-see-v1"],
            vec!["--cpu-ordering=legal-see-v1", "--cpu-ordering=legal-see-v1"],
        ] {
            assert!(
                arguments_with_ordering(
                    &arguments.into_iter().map(String::from).collect::<Vec<_>>()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn semantic_mode_is_exclusive_and_keeps_no_dispatch_capabilities() {
        assert_eq!(
            arguments(&["--prepare-semantic".into()]).unwrap(),
            (CliMode::Semantic, false)
        );
        assert_eq!(
            arguments(&["--prepare-semantic".into(), "--capabilities".into()]).unwrap(),
            (CliMode::Semantic, true)
        );
        for flags in [
            vec!["--candidate-only", "--prepare-semantic"],
            vec!["--prepare-semantic", "--candidate-only"],
            vec!["--capabilities", "--prepare-semantic"],
            vec!["--prepare-semantic", "--prepare-semantic"],
            vec!["--prepare-semantic", "--capabilities", "--candidate-only"],
            vec!["--prepare-semantic", "--unknown"],
        ] {
            assert!(arguments(&flags.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
        }
        let semantic: serde_json::Value =
            serde_json::from_slice(&semantic::capabilities().unwrap()).unwrap();
        assert_eq!(semantic["schema"], semantic::SEMANTIC_SCHEMA);
        assert_eq!(semantic["cpu_checks"], 0);
        assert_eq!(semantic["model_calls"], 0);
        assert_eq!(semantic["training_target_created"], false);
        assert_eq!(semantic["product_verifier_enabled"], false);
        assert_eq!(semantic["native_v_supported"], false);
    }

    fn semantic_rules_request(binary: &str) -> semantic::SemanticRequest {
        use rz_position::Position;
        use rz_position::contracts::ContractPosition;
        use rz_uci::engine::OwnerRegistry;
        use rz_uci::pals_native::pals_history_digest;

        let position = Position::startpos();
        let owner = OwnerRegistry::default().allocate().unwrap();
        let state = ContractPosition::new(owner, position.clone())
            .export()
            .unwrap();
        let state_sha = state
            .snapshot()
            .identity()
            .semantic
            .0
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let history_sha = pals_history_digest(&position)
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let mut request = semantic::SemanticRequest {
            schema: semantic::SEMANTIC_SCHEMA.into(),
            question: semantic::SemanticQuestion::UnrestrictedRecheck,
            parent_input_sha256: "b".repeat(64),
            before_result_anchor_sha256: "c".repeat(64),
            position_command: "position startpos".into(),
            expected_board_fen: position.to_fen(),
            rules_state_sha256: state_sha,
            rules_history_sha256: history_sha,
            prefix: Vec::new(),
            root_moves: Vec::new(),
            claimed_line: Vec::new(),
            current_binary_sha256: binary.into(),
            max_wall_time_ms: 5000,
            max_output_bytes: 1024 * 1024,
            context_sha256: String::new(),
        };
        request.context_sha256 = semantic::request_context_sha256(&request).unwrap();
        request
    }

    #[test]
    fn semantic_delivery_failure_retains_actual_rules_preparation_as_diagnostic() {
        // This synthetic verified-argument pin exercises transport only. The
        // run path separately hashes its actual executable on the same clock.
        let binary = "a".repeat(64);
        let bytes = serde_json::to_vec(&semantic_rules_request(&binary)).unwrap();
        let started = Instant::now();
        let admission = semantic::request_admission(&bytes, started).unwrap();
        let mut context = CliContext {
            started,
            deadline: admission.deadline,
            output_limit: admission.output_limit,
            mode: CliMode::Semantic,
            candidate_report: None,
            semantic_receipt: None,
            continuation_receipt: None,
        };
        let output = match prepare_semantic(&bytes, &binary, &mut context) {
            Ok(output) => output,
            Err(_) => panic!("Rules preparation fixture failed"),
        };
        assert_eq!(output.last(), Some(&b'\n'));
        let emitted: SemanticReceipt = serde_json::from_slice(&output).unwrap();
        assert_eq!(emitted.binary_pin_scope, cli_binary_scope());
        assert_eq!(emitted.cpu_checks, 0);
        let CliError::Semantic(error) =
            context.transport_error(failure("output", "delivery failed"))
        else {
            panic!("semantic delivery error lost its schema");
        };
        assert_eq!(error.schema, semantic::SEMANTIC_SCHEMA);
        assert_eq!(error.stage, "output");
        assert_eq!(error.cpu_checks, 0);
        let retained = error.receipt.as_ref().unwrap();
        assert_eq!(
            retained.branch_meaning_sha256,
            emitted.branch_meaning_sha256
        );
        assert_eq!(retained.context_sha256, emitted.context_sha256);
        assert_eq!(retained.current_binary_sha256, binary);
        assert_eq!(retained.target.legal_moves.len(), 20);
        assert!(!retained.search_executed && !retained.model_executed);
        assert!(!retained.training_target_created && !retained.product_verifier_enabled);
        let diagnostic: serde_json::Value = serde_json::to_value(&error).unwrap();
        assert_eq!(diagnostic["code"], "semantic_branch_failed");
        assert_eq!(
            diagnostic["receipt"]["branch_meaning_sha256"],
            emitted.branch_meaning_sha256
        );
    }

    #[test]
    fn semantic_expired_delivery_is_typed_and_retains_preparation() {
        let binary = "a".repeat(64);
        let request = semantic_rules_request(&binary);
        let bytes = serde_json::to_vec(&request).unwrap();
        let started = Instant::now();
        let admission = semantic::request_admission(&bytes, started).unwrap();
        let mut context = CliContext {
            started,
            deadline: admission.deadline,
            output_limit: admission.output_limit,
            mode: CliMode::Semantic,
            candidate_report: None,
            semantic_receipt: None,
            continuation_receipt: None,
        };
        assert!(prepare_semantic(&bytes, &binary, &mut context).is_ok());
        // The direct writer expiry fixture checks typed final-delivery routing.
        // Production supplies context.deadline, never a fresh writer allowance.
        let output_error = bounded_write(io::sink(), vec![b'\n'], Instant::now()).unwrap_err();
        let CliError::Semantic(error) = context.transport_error(output_error) else {
            panic!("expired semantic output was not a typed failure");
        };
        assert_eq!(error.stage, "output_deadline");
        assert!(error.receipt.is_some());
        assert_eq!(error.cpu_checks, 0);
    }

    #[test]
    fn expired_semantic_admission_does_not_grant_global_stderr_grace() {
        let mut request = semantic_rules_request(&"a".repeat(64));
        request.max_wall_time_ms = 1;
        request.context_sha256 = semantic::request_context_sha256(&request).unwrap();
        let bytes = serde_json::to_vec(&request).unwrap();
        let started = Instant::now() - Duration::from_millis(25);
        let global_deadline = started + Duration::from_millis(MAX_WALL_TIME_MS);
        let mut context = CliContext {
            started,
            deadline: global_deadline,
            output_limit: 1024,
            mode: CliMode::Semantic,
            candidate_report: None,
            semantic_receipt: None,
            continuation_receipt: None,
        };
        let error = match admit_semantic(&bytes, &mut context) {
            Err(CliError::Semantic(error)) => error,
            _ => panic!("expired original semantic request was not rejected"),
        };
        assert_eq!(error.stage, "admission_deadline");
        assert!(error.deadline_exceeded);
        assert_eq!(error.cpu_checks, 0);
        assert!(error.receipt.is_none());
        assert_eq!(context.started, started);
        assert_eq!(context.deadline, started);
        assert!(context.deadline < global_deadline);
        let (send, receive) = mpsc::sync_channel(1);
        let output = bounded_write(BlockedWriter(receive), vec![b'\n'], context.deadline);
        assert_eq!(output.unwrap_err().stage, "output_deadline");
        // The expired check dropped the writer without dispatching a worker.
        assert!(send.send(()).is_err());
    }

    #[test]
    fn continuation_mode_is_exclusive_and_keeps_legacy_schemas_unchanged() {
        assert_eq!(
            arguments(&["--line-continuation".into()]).unwrap(),
            (CliMode::Continuation, false)
        );
        assert_eq!(
            arguments(&["--line-continuation".into(), "--capabilities".into()]).unwrap(),
            (CliMode::Continuation, true)
        );
        for flags in [
            vec!["--line-continuation", "--candidate-only"],
            vec!["--candidate-only", "--line-continuation"],
            vec!["--line-continuation", "--prepare-semantic"],
            vec!["--prepare-semantic", "--line-continuation"],
            vec!["--capabilities", "--line-continuation"],
            vec!["--line-continuation", "--line-continuation"],
        ] {
            assert!(arguments(&flags.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
        }
        let value: serde_json::Value =
            serde_json::from_slice(&continuation::capabilities().unwrap()).unwrap();
        assert_eq!(value["schema"], continuation::CONTINUATION_SCHEMA);
        assert_eq!(value["max_checks"], 1);
        assert_eq!(value["terminal_checks"], 0);
        assert_eq!(value["max_line_plies"], 64);
        assert_eq!(value["training_target_created"], false);
        assert_eq!(value["ranking_created"], false);
        let legacy: serde_json::Value = serde_json::from_slice(&capabilities().unwrap()).unwrap();
        assert_eq!(legacy["schema"], "rz-pals-private-cpu-task/1");
        assert_eq!(legacy["max_checks"], 2);
    }

    fn continuation_rules_request(binary: &str) -> continuation::LineContinuationRequest {
        use rz_contracts::pals::Move16;
        use rz_position::Position;
        use rz_position::contracts::ContractPosition;
        use rz_uci::engine::OwnerRegistry;
        use rz_uci::pals_native::pals_history_digest;

        let owners = OwnerRegistry::default();
        let root_position = Position::startpos();
        let root = ContractPosition::new(owners.allocate().unwrap(), root_position.clone());
        let root_view = root.export().unwrap();
        let movement = root_view
            .legal_moves()
            .moves()
            .iter()
            .copied()
            .find(|movement| {
                rz_position::BoardMove::try_from(*movement)
                    .unwrap()
                    .to_string()
                    == "e2e4"
            })
            .unwrap();
        let endpoint = root
            .fork_from_view(owners.allocate().unwrap(), &root_view, movement)
            .unwrap();
        // No second application: export the child made by the actual Rules fork.
        let endpoint_view = endpoint.export().unwrap();
        let hex = |digest: &[u8; 32]| {
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let mut request = continuation::LineContinuationRequest {
            schema: continuation::CONTINUATION_SCHEMA.into(),
            task_id: "cli-line-fixture".into(),
            captured_input_sha256: "1".repeat(64),
            checker_namespace_sha256: "2".repeat(64),
            before_result_anchor_sha256: "3".repeat(64),
            frozen_epoch: 4,
            input_revision: 5,
            position_command: "position startpos".into(),
            expected_root_board_fen: root_position.to_fen(),
            root_rules_state_sha256: hex(&root_view.snapshot().identity().semantic.0),
            root_rules_history_sha256: hex(&pals_history_digest(&root_position).unwrap()),
            expected_root_legal_moves: root_view
                .legal_moves()
                .moves()
                .iter()
                .map(|movement| Move16::encode(*movement).bits())
                .collect(),
            root_side_to_move: semantic::SideToMove::White,
            line: vec![Move16::encode(movement).bits()],
            line_sha256: String::new(),
            expected_endpoint_board_fen: endpoint.position().to_fen(),
            endpoint_rules_state_sha256: hex(&endpoint_view.snapshot().identity().semantic.0),
            endpoint_rules_history_sha256: hex(&pals_history_digest(endpoint.position()).unwrap()),
            expected_endpoint_legal_moves: endpoint_view
                .legal_moves()
                .moves()
                .iter()
                .map(|movement| Move16::encode(*movement).bits())
                .collect(),
            endpoint_side_to_move: semantic::SideToMove::Black,
            cpu_binary_sha256: binary.into(),
            cpu_profile_sha256: String::new(),
            horizon: 1,
            node_budget: 100_000,
            tt_entries: 64,
            quiescence_ply: 8,
            max_wall_time_ms: 10_000,
            max_output_bytes: 1024 * 1024,
            context_sha256: String::new(),
        };
        request.cpu_profile_sha256 = continuation::profile_sha256(&request).unwrap();
        request.line_sha256 = continuation::line_sha256(&request).unwrap();
        request.context_sha256 = continuation::request_context_sha256(&request).unwrap();
        request
    }

    #[test]
    fn continuation_delivery_failure_retains_actual_endpoint_work_and_image_scope() {
        let binary = "a".repeat(64);
        let bytes = serde_json::to_vec(&continuation_rules_request(&binary)).unwrap();
        let started = Instant::now();
        let admission = continuation::request_admission(&bytes, started).unwrap();
        let mut context = CliContext {
            started,
            deadline: admission.deadline,
            output_limit: admission.output_limit,
            mode: CliMode::Continuation,
            candidate_report: None,
            semantic_receipt: None,
            continuation_receipt: None,
        };
        let output = match check_continuation(&bytes, &binary, &mut context) {
            Ok(output) => output,
            Err(_) => panic!("whole-line CLI fixture failed"),
        };
        let receipt: ContinuationReceipt = serde_json::from_slice(&output).unwrap();
        assert_eq!(receipt.binary_pin_scope, cli_binary_scope());
        assert_eq!(receipt.cpu_calls, 1);
        let CliError::Continuation(error) =
            context.transport_error(failure("output", "delivery failed"))
        else {
            panic!("continuation delivery lost its typed diagnostic");
        };
        assert_eq!(error.cpu_calls, 1);
        let retained = error.receipt.as_ref().unwrap();
        assert_eq!(retained.context_sha256, receipt.context_sha256);
        assert_eq!(retained.line_sha256, receipt.line_sha256);
        assert_eq!(retained.binary_pin_scope, cli_binary_scope());
        assert_eq!(
            error.known_nodes.as_deref(),
            Some(&receipt.report.unwrap().nodes)
        );
    }

    #[test]
    fn expired_continuation_admission_never_grants_outer_clock_diagnostic_grace() {
        let mut request = continuation_rules_request(&"a".repeat(64));
        request.max_wall_time_ms = 1;
        request.context_sha256 = continuation::request_context_sha256(&request).unwrap();
        let started = Instant::now() - Duration::from_millis(25);
        let mut context = CliContext {
            started,
            deadline: started + Duration::from_millis(MAX_WALL_TIME_MS),
            output_limit: 1024,
            mode: CliMode::Continuation,
            candidate_report: None,
            semantic_receipt: None,
            continuation_receipt: None,
        };
        let error = match admit_continuation(&serde_json::to_vec(&request).unwrap(), &mut context) {
            Err(CliError::Continuation(error)) => error,
            _ => panic!("expired continuation was admitted"),
        };
        assert_eq!(error.stage, "admission_deadline");
        assert!(error.deadline_exceeded);
        assert_eq!(error.cpu_calls, 0);
        assert_eq!(context.deadline, started);
        assert!(context.continuation_receipt.is_none());
    }
}
