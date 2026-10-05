//! Explicit CUDA integration consumer. CPU V1 wire and profile are unchanged.
//! Public wire checks can run on synthetic bytes and confer no NN/GPU evidence.
//! Process, private input, Rules replay and native drain gates are independent.

use crate::native_launch::{NativeEngineView, NativeLaunchDeclaration, NativePairView};
use crate::native_runner::NativeProviderDeclaration;
use crate::{
    ArenaError, NativeLaunchOwner, NativePairFailure, NativePairOutput, NativePairReceipt,
    NativePreparationFailure,
};
#[cfg(target_os = "linux")]
use rz_experiments::CudaNativeLaunchSpec;
use rz_experiments::{
    ArtifactRef, CudaBundleFileRoleV1, LockedCudaIntegrationPairSpecV1, NativeEngineRole,
    NativeResourceBudgetV1,
};
use rz_experiments::{CudaLaunchProfile, LockedCudaIntegrationPairSpec};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::atomic::AtomicBool};

pub type NativeCudaLaunchOwner = NativeLaunchOwner<LockedCudaIntegrationPairSpecV1>;
pub type NativeCudaPairOutput = NativePairOutput<LockedCudaIntegrationPairSpecV1>;
pub type NativeCudaPairFailure = NativePairFailure<LockedCudaIntegrationPairSpecV1>;
pub type NativeCudaPairReceipt = NativePairReceipt<NativeCudaProviderSessionAudit>;

#[derive(Clone, Debug, Serialize)]
pub struct NativeCudaProviderSessionAudit {
    pub role: String,
    pub process_id: u32,
    pub process_run_id: String,
    pub startup_sha256: String,
    pub termination_sha256: String,
    pub completed_by_runtime: u64,
    pub search_root_initializations: u64,
    pub search_non_root_backups: u64,
    pub actual_cuda_inference_observed: bool,
    pub physical_drain: String,
    pub runtime_bundle_sha256: String,
    pub cuda_profile_sha256: String,
}

/// Verify the fixed Linux Fastchess renderer, not an engine's self-report.
/// The caller supplies four identities from validated startup records and the
/// supervisor-owned, bounded stdout whose ArtifactRef is already retained.
/// This pinned source logs raw wait status: only decimal `0` proves exit zero.
/// Missing records (including earlier reaping), malformed records or renderer
/// changes are unsupported evidence and fail closed, never inferred as success.
pub fn validate_native_cuda_process_exit_trace(
    stdout: &[u8],
    expected_pids: &[u32],
) -> Result<(), ArenaError> {
    const MAX_BYTES: usize = 64 * 1024 * 1024;
    const MAX_LINES: usize = 131_072;
    const MAX_LINE_BYTES: usize = 4096;
    let fail = |reason: &str| ArenaError::Integrity(reason.into());
    if expected_pids.len() != 4
        || expected_pids
            .iter()
            .any(|pid| *pid == 0 || *pid > i32::MAX as u32)
        || expected_pids
            .iter()
            .enumerate()
            .any(|(index, pid)| expected_pids[..index].contains(pid))
    {
        return Err(fail(
            "CUDA exit trace requires exactly four distinct native PIDs",
        ));
    }
    if stdout.is_empty() || stdout.len() > MAX_BYTES || !stdout.ends_with(b"\n") {
        return Err(fail(
            "CUDA exit trace is empty, over budget or unterminated",
        ));
    }
    let text = std::str::from_utf8(stdout).map_err(|_| fail("CUDA runner stdout is not UTF-8"))?;
    let mut seen = [false; 4];
    for (index, line) in text.split_terminator('\n').enumerate() {
        if index >= MAX_LINES || line.len() > MAX_LINE_BYTES {
            return Err(fail("CUDA runner trace line budget exceeded"));
        }
        // Logger::readFromEngine adds its own anchored [Engine] prefix to each
        // protocol line; a quoted TRACE-looking payload cannot supply evidence.
        if !line.starts_with("[TRACE")
            || !(line.contains("Process with pid")
                || line.contains("Force terminating process with pid"))
        {
            continue;
        }
        if !line.is_ascii() {
            return Err(fail("CUDA process exit renderer is not ASCII"));
        }
        let message = cuda_exit_trace_message(line)
            .ok_or_else(|| fail("CUDA process exit renderer is malformed"))?;
        if message.starts_with("Force terminating process with pid:") {
            return Err(fail("CUDA native process required force termination"));
        }
        let (pid_text, status) = message
            .strip_prefix("Process with pid: ")
            .and_then(|value| value.split_once(" terminated with status: "))
            .ok_or_else(|| fail("CUDA native process exit record is malformed"))?;
        let pid = pid_text
            .parse::<u32>()
            .map_err(|_| fail("CUDA native process exit PID is malformed"))?;
        if pid_text != pid.to_string() {
            return Err(fail(
                "CUDA native process exit PID is not canonical decimal",
            ));
        }
        let slot = expected_pids
            .iter()
            .position(|expected| *expected == pid)
            .ok_or_else(|| fail("CUDA runner trace contains a foreign native PID"))?;
        if seen[slot] {
            return Err(fail("CUDA native process exit evidence is duplicated"));
        }
        if status != "0" {
            return Err(fail(
                "CUDA native process exit status is nonzero or malformed",
            ));
        }
        seen[slot] = true;
    }
    if seen.iter().any(|present| !present) {
        return Err(fail("CUDA native process exit evidence is incomplete"));
    }
    Ok(())
}

fn cuda_exit_trace_message(line: &str) -> Option<&str> {
    // Pinned logger.hpp:157: [label left-width6] [time width15]
    // <thread right-width20> fastchess --- message. TRACE_THREAD is nonempty.
    let (time, tail) = line.strip_prefix("[TRACE ] [")?.split_once("] <")?;
    let time = time.as_bytes();
    if time.len() != 15
        || time[2] != b':'
        || time[5] != b':'
        || time[8] != b'.'
        || time
            .iter()
            .enumerate()
            .any(|(index, byte)| !matches!(index, 2 | 5 | 8) && !byte.is_ascii_digit())
        || (time[0] - b'0') * 10 + time[1] - b'0' > 23
        || (time[3] - b'0') * 10 + time[4] - b'0' > 59
        || (time[6] - b'0') * 10 + time[7] - b'0' > 59
    {
        return None;
    }
    let (thread, message) = tail.split_once("> fastchess --- ")?;
    let digits = thread.trim_start_matches(' ');
    if thread.len() != 20
        || digits.is_empty()
        || digits.starts_with('0')
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || digits.parse::<u64>().is_err()
    {
        return None;
    }
    Some(message)
}

impl<P: CudaLaunchProfile> crate::native_launch::sealed::Sealed
    for LockedCudaIntegrationPairSpec<P>
{
}
impl<P: CudaLaunchProfile> NativeLaunchDeclaration for LockedCudaIntegrationPairSpec<P> {
    fn input_sha256(&self) -> &str {
        self.sha256()
    }
    fn view(&self) -> NativePairView<'_> {
        let p = self.input();
        let b = p.budget;
        // A shared supervisor view carries declared observation limits only.
        // The original GPU spec remains owned; no CPU launch/profile is built.
        NativePairView {
            pair_id: &p.pair_id,
            white_order: p.white_order,
            opening: &p.opening,
            opening_artifact: &p.opening_artifact,
            runner: &p.runner,
            clock: p.clock,
            max_plies: p.max_plies,
            timeouts: p.timeouts,
            budget: NativeResourceBudgetV1 {
                max_input_bytes: b.max_input_bytes,
                max_output_bytes: b.max_output_bytes,
                max_runtime_bytes: b.max_runtime_bytes,
                max_artifact_bytes: b.max_artifact_bytes,
                max_child_processes: b.max_child_processes,
                max_runtime_files: b.max_runtime_files,
                max_runtime_depth: b.max_runtime_depth,
                address_space_per_process_bytes: b.address_space_per_process_bytes,
            },
        }
    }
    fn declared_inputs(&self) -> Vec<&ArtifactRef> {
        self.declared_artifacts()
    }
    fn unique_bytes(&self) -> Result<u64, rz_experiments::ManifestError> {
        self.unique_input_bytes()
    }
    fn engine_view(
        &self,
        role: NativeEngineRole,
    ) -> Result<NativeEngineView<'_>, rz_experiments::ManifestError> {
        let engine = self.input().engine(role)?;
        Ok(NativeEngineView {
            engine_id: &engine.engine_id,
            artifacts: &engine.artifacts,
            cuda_bundle: Some(&engine.cuda_bundle),
            search: engine.profile.search_options(),
        })
    }
    fn provider_name(&self) -> &'static str {
        "CUDA"
    }
    fn advise_drop_input_cache(&self) -> bool {
        P::VERSION == 2
    }
    fn additional_manifest(&self) -> Option<&ArtifactRef> {
        // A sealed, already-validated lock contains both roles and identical inputs.
        Some(&self.input().engines[0].cuda_bundle.manifest)
    }
    fn validate_additional_manifest(&self, bytes: &[u8]) -> Result<(), ArenaError> {
        validate_cuda_bundle_manifest_fields(bytes, &self.input().engines[0].cuda_bundle)
    }
}
impl<P: CudaLaunchProfile> NativeProviderDeclaration for LockedCudaIntegrationPairSpec<P> {
    type Audit = NativeCudaProviderSessionAudit;
    fn scope(&self) -> &'static str {
        "cuda_nn_pair_integration_process_provider_search_and_native_rules"
    }
    fn receipt_filename(&self) -> &'static str {
        if P::VERSION == 1 {
            "native-cuda-pair-receipt.v1.json"
        } else {
            "native-cuda-pair-receipt.v2.json"
        }
    }
    fn startup_filename(&self) -> &'static str {
        "native-cuda-startup.v1.json"
    }
    fn termination_filename(&self) -> &'static str {
        "native-cuda-termination.v1.json"
    }
    #[cfg(target_os = "linux")]
    fn validate_process_exit_trace(
        &self,
        stdout: &[u8],
        expected_pids: &[u32],
    ) -> Result<(), ArenaError> {
        validate_native_cuda_process_exit_trace(stdout, expected_pids)
    }
    #[cfg(target_os = "linux")]
    fn validate_records(
        &self,
        role: NativeEngineRole,
        startup: &[u8],
        termination: &[u8],
        session: &str,
    ) -> Result<(Self::Audit, u32), ArenaError> {
        let audit = validate_native_cuda_provider_record_fields(
            startup,
            termination,
            self.input().engine(role)?,
            session,
        )?;
        let pid = audit.process_id;
        Ok((audit, pid))
    }
    #[cfg(target_os = "linux")]
    fn verify_session_evidence(
        &self,
        role: NativeEngineRole,
        pid: u32,
        startup: &[u8],
        runtime_directory: &cap_std::fs::Dir,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        let mut evidence = linux::placement_evidence(pid, startup, runtime_directory)?;
        if let Some(search) = self.input().engine(role)?.profile.search_options() {
            let relative = format!(
                "native-process-{pid}/{}",
                rz_uci::native_cuda_attestation::SEARCH_FILE
            );
            let bytes =
                crate::native_runner::linux::read_file(runtime_directory, &relative, 256 * 1024)?;
            validate_native_cuda_search_record_fields(&bytes, startup, search)?;
            evidence.push((relative, bytes));
        }
        Ok(evidence)
    }
}

/// Source bytes are verified and copied before any runner/native child starts.
/// Manifest/library filenames remain siblings for C's closed bundle loader.
pub fn prepare_native_cuda_launch<P: CudaLaunchProfile>(
    spec: &LockedCudaIntegrationPairSpec<P>,
    source_root: &Path,
    output_root: &Path,
    output_directory: &str,
) -> Result<NativeLaunchOwner<LockedCudaIntegrationPairSpec<P>>, Box<NativePreparationFailure>> {
    crate::native_launch::prepare_native_launch_for(
        spec,
        source_root,
        output_root,
        output_directory,
    )
}
/// One two-game integration pair. A cutoff remains Incomplete with zero scoring
/// authority; unresolved cleanup retains GPU copies and the original Child.
pub fn run_native_cuda_pair<P: CudaLaunchProfile>(
    owner: NativeLaunchOwner<LockedCudaIntegrationPairSpec<P>>,
    cancel: Option<&AtomicBool>,
) -> Result<
    NativePairOutput<LockedCudaIntegrationPairSpec<P>>,
    Box<NativePairFailure<LockedCudaIntegrationPairSpec<P>>>,
> {
    crate::native_runner::run_native_pair_for(owner, cancel)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleWireV1 {
    schema_version: u32,
    files: Vec<BundleFileWireV1>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleFileWireV1 {
    role: CudaBundleFileRoleV1,
    filename: String,
    bytes: u64,
    sha256: String,
}

/// Compare bounded manifest *bytes* with E's independently locked closed file
/// declarations. This check executes no native code and is not a GPU proof.
pub fn validate_cuda_bundle_manifest_fields(
    bytes: &[u8],
    binding: &rz_experiments::CudaBundleBindingV1,
) -> Result<(), ArenaError> {
    use sha2::{Digest, Sha256};
    binding.validate()?;
    if bytes.len() > 64 * 1024
        || bytes.len() as u64 != binding.manifest.bytes
        || format!("{:x}", Sha256::digest(bytes)) != binding.manifest.sha256
    {
        return Err(ArenaError::Integrity(
            "CUDA bundle raw manifest identity differs".into(),
        ));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| ArenaError::Integrity("CUDA bundle manifest is not UTF-8".into()))?;
    let wire: BundleWireV1 = rz_experiments::decode_json(text)?;
    if wire.schema_version != 1 || wire.files.len() != 19 {
        return Err(ArenaError::Integrity(
            "CUDA bundle manifest has unsupported schema or file count".into(),
        ));
    }
    let mut observed = std::collections::BTreeSet::new();
    for file in &wire.files {
        if !observed.insert(file.filename.as_str()) {
            return Err(ArenaError::Integrity(
                "CUDA bundle manifest repeats a native filename".into(),
            ));
        }
        let declared = binding.file(file.role, &file.filename)?;
        if file.bytes != declared.bytes || file.sha256 != declared.sha256 {
            return Err(ArenaError::Integrity(
                "CUDA bundle manifest file identity differs".into(),
            ));
        }
    }
    // Binding.validate replays C's binary-codec hash and fixed closed filenames.
    Ok(())
}

#[cfg(target_os = "linux")]
pub use linux::{
    validate_cuda_placement_trace_fields, validate_native_cuda_provider_record_fields,
    validate_native_cuda_search_record_fields,
};

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use crate::native_runner::linux::{
        check_completion, equals, field, flag, hash, json, keys, n, none, number, registry, text,
    };
    use rz_experiments::NativeArtifactRole;
    use rz_uci::native_cuda_attestation::{CudaStartupReceiptV1, CudaTerminationReceiptV1};
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    fn invalid(detail: &str) -> ArenaError {
        ArenaError::Integrity(detail.into())
    }

    /// Strict B CUDA V1 DTO consumption plus the fixed E declaration and final
    /// search guards. Synthetic DTOs attest only this parser's rejection rules.
    pub fn validate_native_cuda_provider_record_fields<P: CudaLaunchProfile>(
        startup_bytes: &[u8],
        termination_bytes: &[u8],
        engine: &CudaNativeLaunchSpec<P>,
        session: &str,
    ) -> Result<NativeCudaProviderSessionAudit, ArenaError> {
        if startup_bytes.len() > 256 * 1024 || termination_bytes.len() > 256 * 1024 {
            return Err(ArenaError::Budget(
                "CUDA provider record exceeds 256KiB".into(),
            ));
        }
        engine.validate()?;
        let raw_startup = json(startup_bytes)?;
        let raw_term = json(termination_bytes)?;
        keys(
            &raw_startup,
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
        keys(
            &raw_term,
            &[
                "schema_version",
                "kind",
                "process_run_id",
                "startup",
                "run_succeeded",
                "actual_cuda_inference_observed",
                "actual_rules_search_backup_observed",
                "physical_drain",
                "report",
                "original_service_failure",
                "collection_failure",
                "runtime_mapping_failure",
                "retained_owner_and_evidence",
            ],
        )?;
        let _startup_dto: CudaStartupReceiptV1 = serde_json::from_value(raw_startup.clone())
            .map_err(|_| invalid("CUDA startup DTO differs from B V1"))?;
        let _term_dto: CudaTerminationReceiptV1 = serde_json::from_value(raw_term.clone())
            .map_err(|_| invalid("CUDA termination DTO differs from B V1"))?;
        // Preserve raw presence: deserializing missing Option fields as None
        // must not turn an incomplete record into an explicit null receipt.
        let startup = raw_startup;
        let term = raw_term;
        check_startup(&startup, engine, session)?;
        let (count, root, nonroot) = check_termination(&term, &startup)?;
        Ok(NativeCudaProviderSessionAudit {
            role: match engine.role {
                NativeEngineRole::Baseline => "baseline",
                NativeEngineRole::Candidate => "candidate",
            }
            .into(),
            process_id: u32::try_from(number(&startup, "process_id")?)
                .map_err(|_| invalid("CUDA process PID exceeds native range"))?,
            process_run_id: session.into(),
            startup_sha256: format!("{:x}", Sha256::digest(startup_bytes)),
            termination_sha256: format!("{:x}", Sha256::digest(termination_bytes)),
            completed_by_runtime: count,
            search_root_initializations: root,
            search_non_root_backups: nonroot,
            actual_cuda_inference_observed: true,
            physical_drain: "confirmed".into(),
            runtime_bundle_sha256: engine.cuda_bundle.canonical_sha256.clone(),
            cuda_profile_sha256: text(field(&startup, "profile")?, "cuda_profile_sha256")?.into(),
        })
    }

    fn check_startup<P: CudaLaunchProfile>(
        startup: &Value,
        engine: &CudaNativeLaunchSpec<P>,
        session: &str,
    ) -> Result<(), ArenaError> {
        n(startup, "schema_version", 1)?;
        equals(startup, "kind", "startup")?;
        flag(startup, "loaded", true)?;
        let pid = number(startup, "process_id")?;
        if pid == 0 || pid > u32::MAX as u64 || session != format!("native-process-{pid}") {
            return Err(invalid("CUDA session path and process identity differ"));
        }
        equals(startup, "process_run_id", session)?;
        if number(startup, "process_epoch")? == 0 {
            return Err(invalid("CUDA process epoch is zero"));
        }
        let executable = field(startup, "executable")?;
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
        equals(
            p,
            "backend_sha256",
            &engine.profile.runtime_profile().expected_backend_sha256,
        )?;
        for key in [
            "source_weights_protobuf_sha256",
            "converter_binary_sha256",
            "ort_build_info_sha256",
            "action_map_sha256",
            "history_policy_sha256",
        ] {
            hash(text(p, key)?)?;
        }
        let converter = text(p, "converter_commit")?;
        if converter.len() != 40
            || !converter
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(invalid("CUDA converter source identity is malformed"));
        }
        equals(p, "ort_release", "1.22.0")?;
        equals(p, "provider", "cuda")?;
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
        n(p, "device_id", 0)?;
        n(
            p,
            "arena_bytes",
            engine.profile.runtime_profile().arena_bytes,
        )?;
        flag(p, "tf32", false)?;
        equals(
            p,
            "runtime_bundle_manifest_sha256",
            &engine.cuda_bundle.manifest.sha256,
        )?;
        equals(
            p,
            "runtime_bundle_sha256",
            &engine.cuda_bundle.canonical_sha256,
        )?;
        hash(text(p, "cuda_profile_sha256")?)?;
        flag(p, "runtime_mapping_verified", true)?;
        if number(p, "executed_cuda_nodes")? == 0 {
            return Err(invalid("CUDA session placement has no executed CUDA nodes"));
        }
        let files = field(p, "runtime_bundle_files")?
            .as_array()
            .ok_or_else(|| invalid("CUDA bundle receipt list type differs"))?;
        if files.len() != 19 {
            return Err(invalid("CUDA startup lacks closed 19 runtime files"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for file in files {
            let filename = text(file, "filename")?;
            if !seen.insert(filename) {
                return Err(invalid("CUDA startup repeats bundle filenames"));
            }
            let role: CudaBundleFileRoleV1 =
                serde_json::from_value(field(file, "role")?.clone())
                    .map_err(|_| invalid("CUDA bundle receipt role differs"))?;
            let declared = engine.cuda_bundle.file(role, filename)?;
            n(file, "bytes", declared.bytes)?;
            equals(file, "sha256", &declared.sha256)?;
        }
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
            &engine.profile.runtime_profile().expected_encoding_sha256,
        )?;
        let resident = field(p, "session_resident_admission")?;
        n(resident, "host_bytes", 0)?;
        n(
            resident,
            "device_bytes",
            engine.profile.runtime_profile().arena_bytes,
        )?;
        n(resident, "pinned_bytes", 0)?;
        equals(
            resident,
            "scope",
            "declaration_not_measured_vram_or_hardcap",
        )?;
        Ok(())
    }

    /// Exact search metadata emitted from the same EngineSettings passed to
    /// serve. This is initialization evidence, not a per-move timing proof.
    pub fn validate_native_cuda_search_record_fields(
        bytes: &[u8],
        startup: &[u8],
        expected: rz_experiments::NativeCudaSearchV2,
    ) -> Result<(), ArenaError> {
        if bytes.len() > 256 * 1024 || startup.len() > 256 * 1024 {
            return Err(ArenaError::Budget(
                "CUDA search record exceeds budget".into(),
            ));
        }
        let receipt = json(bytes)?;
        let _: rz_uci::native_cuda_attestation::CudaSearchReceiptV1 =
            serde_json::from_value(receipt.clone())
                .map_err(|_| invalid("CUDA search record differs from B schema"))?;
        let startup_json = json(startup)?;
        n(&receipt, "schema_version", 1)?;
        equals(&receipt, "kind", "search_config")?;
        equals(
            &receipt,
            "process_run_id",
            text(&startup_json, "process_run_id")?,
        )?;
        equals(
            &receipt,
            "startup_sha256",
            &format!("{:x}", Sha256::digest(startup)),
        )?;
        n(&receipt, "simulations", expected.simulations)?;
        equals(&receipt, "final_selection", expected.final_selection.cli())?;
        n(
            &receipt,
            "policy_temperature_milli",
            u64::from(expected.policy_temperature_milli),
        )?;
        flag(&receipt, "raw_cache", expected.raw_cache)?;
        for (key, value) in [
            ("batch_size", 1),
            ("search_workers", 1),
            ("output_margin_ms", 10),
            ("drain_margin_ms", 10),
            ("shutdown_ms", 2000),
            ("max_nodes", 20000),
            ("max_edges", 100000),
            ("max_depth", 128),
        ] {
            n(&receipt, key, value)?;
        }
        equals(
            &receipt,
            "scope",
            "startup_config_bound_to_served_settings_not_per_move_timing_or_strength",
        )
    }

    fn check_termination(term: &Value, startup: &Value) -> Result<(u64, u64, u64), ArenaError> {
        n(term, "schema_version", 1)?;
        equals(term, "kind", "termination")?;
        if field(term, "startup")? != startup {
            return Err(invalid(
                "CUDA termination clone differs from actual startup",
            ));
        }
        equals(term, "process_run_id", text(startup, "process_run_id")?)?;
        flag(term, "run_succeeded", true)?;
        flag(term, "actual_cuda_inference_observed", true)?;
        flag(term, "actual_rules_search_backup_observed", true)?;
        equals(term, "physical_drain", "confirmed")?;
        none(term, "original_service_failure")?;
        none(term, "collection_failure")?;
        none(term, "runtime_mapping_failure")?;
        flag(term, "retained_owner_and_evidence", false)?;
        let r = field(term, "report")?;
        let p = field(startup, "profile")?;
        keys(
            r,
            &[
                "model_manifest_sha256",
                "backend_sha256",
                "origin",
                "completed_by_runtime",
                "first_completed",
                "last_completed",
                "completed_scope",
                "search_root_initializations",
                "search_non_root_backups",
                "search_scope",
                "observations",
                "canceled",
                "expired",
                "failures",
                "overflow",
                "boundary_error",
                "poison_error",
            ],
        )?;
        equals(
            r,
            "model_manifest_sha256",
            text(p, "export_manifest_sha256")?,
        )?;
        equals(r, "backend_sha256", text(p, "backend_sha256")?)?;
        equals(r, "origin", "cuda_onnx")?;
        equals(
            r,
            "completed_scope",
            "process_aggregate_normal_poll_first_last_not_full_root_or_physical_journal",
        )?;
        let count = number(r, "completed_by_runtime")?;
        if count == 0 {
            return Err(invalid("CUDA process has no D-accepted computed result"));
        }
        for end in ["first_completed", "last_completed"] {
            check_completion(field(r, end)?, startup)?;
        }
        if count == 1 && field(r, "first_completed")? != field(r, "last_completed")? {
            return Err(invalid(
                "single CUDA completion has inconsistent first/last receipts",
            ));
        }
        for key in ["overflow", "boundary_error", "poison_error"] {
            none(r, key)?;
        }
        if field(r, "failures")?
            .as_array()
            .is_none_or(|a| !a.is_empty())
        {
            return Err(invalid("CUDA process preserves failed inference receipts"));
        }
        equals(
            r,
            "search_scope",
            "unchanged_final_tree_guard_accepted_count_first_last_not_every_root_or_bestmove_proof",
        )?;
        let root = search_aggregate(field(r, "search_root_initializations")?, startup, false)?;
        let nonroot = search_aggregate(field(r, "search_non_root_backups")?, startup, true)?;
        if root.checked_add(nonroot).is_none_or(|used| used > count) {
            return Err(invalid(
                "CUDA final search consumption exceeds D completion count",
            ));
        }
        let observations = field(r, "observations")?;
        equals(
            observations,
            "scope",
            "drained_event_counts_and_loss_not_full_journal_or_physical_inference_count",
        )?;
        for key in ["scheduler_dropped", "delivery_dropped"] {
            n(observations, key, 0)?;
        }
        for key in ["scheduler_counter_overflow", "delivery_counter_overflow"] {
            flag(observations, key, false)?;
        }
        for key in ["scheduler_events", "delivery_events"] {
            if number(observations, key)? == 0 {
                return Err(invalid(
                    "CUDA native observations lack scheduler or delivery events",
                ));
            }
        }
        // A drain-discarded count is retained as an observation, never physical
        // inference totals. Cancellation/expiry remains separate typed causes.
        number(observations, "drain_discarded_results")?;
        for key in ["canceled", "expired"] {
            check_audit_aggregate(field(r, key)?, key)?;
        }
        Ok((count, root, nonroot))
    }
    fn search_aggregate(
        aggregate: &Value,
        startup: &Value,
        nonroot: bool,
    ) -> Result<u64, ArenaError> {
        keys(aggregate, &["count", "first", "last"])?;
        let count = number(aggregate, "count")?;
        if count == 0 {
            return Err(invalid(
                "CUDA pair requires actual guarded root and non-root search consumption",
            ));
        }
        for end in ["first", "last"] {
            let receipt = field(aggregate, end)?;
            keys(receipt, &["completed", "traversed_edges"])?;
            check_completion(field(receipt, "completed")?, startup)?;
            let edges = number(receipt, "traversed_edges")?;
            if (!nonroot && edges != 0) || (nonroot && (edges == 0 || edges > 4095)) {
                return Err(invalid(
                    "CUDA guarded search traversal classification differs",
                ));
            }
        }
        if count == 1 && field(aggregate, "first")? != field(aggregate, "last")? {
            return Err(invalid(
                "single guarded search event has inconsistent first/last receipts",
            ));
        }
        Ok(count)
    }
    fn check_audit_aggregate(aggregate: &Value, kind: &str) -> Result<(), ArenaError> {
        keys(aggregate, &["count", "first", "last"])?;
        let count = number(aggregate, "count")?;
        if count == 0 {
            none(aggregate, "first")?;
            none(aggregate, "last")?;
            return Ok(());
        }
        for end in ["first", "last"] {
            let cause = field(aggregate, end)?;
            flag(cause, "projection_truncated", false)?;
            n(cause, "max_depth", 8)?;
            n(cause, "max_nodes", 128)?;
            let nodes = number(cause, "projected_nodes")?;
            if nodes == 0 || nodes > 128 {
                return Err(invalid("CUDA canceled/expired cause node bound differs"));
            }
            let projection = field(cause, "projection")?;
            equals(projection, "type", "native_diagnostic")?;
            equals(projection, "kind", "dispatch_refused")?;
            let physical = field(field(projection, "failure")?, "cause")?;
            none(physical, "backend")?;
            equals(
                field(field(physical, "contract")?, "cause")?,
                "code",
                if kind == "canceled" {
                    "canceled"
                } else {
                    "expired"
                },
            )?;
        }
        Ok(())
    }

    /// Independent bounded check of retained ORT placement trace bytes. Probe
    /// nodes prove only the declared placement probe, not every later root/run.
    pub fn validate_cuda_placement_trace_fields(
        bytes: &[u8],
        expected_sha256: &str,
        expected_nodes: u64,
    ) -> Result<(), ArenaError> {
        hash(expected_sha256)?;
        if bytes.len() > 4 * 1024 * 1024
            || format!("{:x}", Sha256::digest(bytes)) != expected_sha256
        {
            return Err(invalid(
                "CUDA retained placement trace identity or size differs",
            ));
        }
        let events = json(bytes)?;
        let events = events
            .as_array()
            .ok_or_else(|| invalid("CUDA placement trace must be a JSON event array"))?;
        let mut count = 0u64;
        for event in events {
            if event.get("cat").and_then(Value::as_str) != Some("Node") {
                continue;
            }
            let name = text(event, "name")?;
            let provider = event.pointer("/args/provider");
            if name.ends_with("_fence_before") || name.ends_with("_fence_after") {
                if provider.is_some_and(|p| p.as_str() != Some("CUDAExecutionProvider")) {
                    return Err(invalid(
                        "CUDA placement trace contains a foreign fence provider",
                    ));
                }
                continue;
            }
            if !name.ends_with("_kernel_time")
                || provider.and_then(Value::as_str) != Some("CUDAExecutionProvider")
            {
                return Err(invalid(
                    "CUDA placement trace contains a missing, unknown or CPU node provider",
                ));
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| ArenaError::Budget("CUDA placement node count overflow".into()))?;
        }
        if count == 0 || count != expected_nodes {
            return Err(invalid("CUDA placement trace executed node count differs"));
        }
        Ok(())
    }
    pub(super) fn placement_evidence(
        pid: u32,
        startup: &[u8],
        runtime_directory: &cap_std::fs::Dir,
    ) -> Result<Vec<(String, Vec<u8>)>, ArenaError> {
        use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
        use cap_std::fs::{OpenOptions, OpenOptionsExt};
        use std::{io::Read, os::unix::fs::MetadataExt};
        let startup = json(startup)?;
        let profile = field(&startup, "profile")?;
        let directory_name = format!("native-cuda-placement-{pid}");
        let directory = runtime_directory
            .open_dir_nofollow(&directory_name)
            .map_err(|_| invalid("CUDA process lacks its owned placement directory"))?;
        let mut entries = directory
            .entries()
            .map_err(|_| invalid("cannot inspect CUDA placement directory"))?;
        let entry = entries
            .next()
            .ok_or_else(|| invalid("CUDA placement directory is empty"))?
            .map_err(|_| invalid("CUDA placement trace entry is unavailable"))?;
        if entries.next().is_some() {
            return Err(invalid(
                "CUDA placement directory contains additional or ambiguous traces",
            ));
        }
        let os_name = entry.file_name();
        let name = os_name
            .to_str()
            .ok_or_else(|| invalid("CUDA placement trace filename is not UTF-8"))?;
        if name.len() > 128
            || !name.starts_with("placement_")
            || !name.ends_with(".json")
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        {
            return Err(invalid(
                "CUDA placement trace filename is outside its known format",
            ));
        }
        let mut options = OpenOptions::new();
        options
            .read(true)
            .follow(FollowSymlinks::No)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = directory
            .open_with(&os_name, &options)
            .map_err(|_| invalid("cannot pin CUDA placement trace"))?
            .into_std();
        let metadata = file
            .metadata()
            .map_err(|_| invalid("CUDA placement trace metadata is unavailable"))?;
        if !metadata.is_file() || metadata.nlink() != 1 || metadata.len() > 4 * 1024 * 1024 {
            return Err(invalid(
                "CUDA placement trace is not a bounded exclusive regular file",
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(
                usize::try_from(metadata.len())
                    .map_err(|_| invalid("CUDA placement trace cannot fit host"))?,
            )
            .map_err(|_| ArenaError::Budget("CUDA placement trace allocation failed".into()))?;
        file.take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| invalid("cannot read CUDA placement trace"))?;
        validate_cuda_placement_trace_fields(
            &bytes,
            text(profile, "cuda_profile_sha256")?,
            number(profile, "executed_cuda_nodes")?,
        )?;
        Ok(vec![(format!("{directory_name}/{name}"), bytes)])
    }
}
