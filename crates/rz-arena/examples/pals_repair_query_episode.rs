//! One bounded, registered CPU replay and independent native witness.
//! The subprocess report remains reported material; no JSON imports a capability.
//! Run through the external resource/cleanup supervisor. No training or GPU work.

#[cfg(not(all(target_os = "linux", feature = "pals-collection-onnx")))]
fn main() -> std::process::ExitCode {
    eprintln!(
        "pals_repair_query_episode requires Linux and pals-collection-onnx; no execution started"
    );
    std::process::ExitCode::FAILURE
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    match enabled::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pals_repair_query_episode refused: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
mod enabled {
    use rz_arena::{ArenaError, ProcessLimits};
    use rz_arena::pals_replay::{
        RepairReplayDeliveryRegistration, ReplayCallerPolicy,
        ReplayDeliveryRegistration, supervise_prepared_observed_replay,
        CapturedRepairWitnessFailure, witness_captured_repair_inputs,
    };
    use rz_arena::pals_replay::utility_observation::{
        begin_registered_defend_coverage_episode, observe_repair_coverage_cost,
    };
    use rz_uci::pals_native::{NativeInvocationBudget, NativeOwnerOptions, NativeRoleModel};
    use rz_eval::pals_onnx::PalsOnnxConfig;
    use rz_eval::runtime_pin::RuntimeCache;
    use rz_uci::pals_cpu_task::strategic_action::ArtifactPin;
    use rz_uci::pals_cpu_task::strategic_action::native_replay::{
        CpuFreshAssetProfile, NativeReplayPrimary,
    };
    use rz_uci::pals_cpu_task::strategic_action::native_replay::query_prior::captured_cpu_witness::{
        IndependentRepairCpuFailure, observe_independent_repair_cpu_witness,
    };
    use rz_uci::pals_cpu_task::strategic_action::replay_inputs::{
        ReplayConfigInput, ReplayConstructionBudget, ReplayInputMode, ReplayOriginals,
        ReplayPreparationDeclaration, ReplayResourceDeclaration, SemanticReceiptProducerScope,
        check_replay_inputs_with_semantic_scope,
    };
    use rz_uci::pals_cpu_task::strategic_action::replay_launch_preparation::{
        ExpectedPinsWire, ReplayLaunchAssetPaths, ReplayLaunchBudget, ReplayLaunchDeclaration,
        ReplayLaunchExpected, ReplayLaunchOriginals, prepare_repair_observed_replay_launch_bundle,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Write};
    use std::mem::ManuallyDrop;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    const PLAN_SCHEMA: &str = "rz-pals-repair-query-episode-acceptance-plan/1";
    const PLAN_BYTES: u64 = 512 * 1024;
    const ORIGINAL_BYTES: u64 = 4 * 1024 * 1024;
    const REPORT_BYTES: usize = 16 * 1024 * 1024;
    const NORMAL_REPORT_CREDIT: usize = 24 * 1024 * 1024;
    const FAILURE_REPORT_CREDIT: usize = 16 * 1024 * 1024;
    // Preserve envelope/receipts/two original 4MiB streams within the separate
    // 16MiB failure lane. A larger typed witness failure remains owned; an audit
    // exceeding this explicit diagnostic budget is partial, never a complete fact.
    const CAPTURED_FAILURE_REPORT_BYTES: usize = 7 * 1024 * 1024;
    const REPORT_FILES: usize = 24;
    const SELECTION_POLICY: &str = "registered_defend_response_coverage/2";

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PinnedFile {
        path: PathBuf,
        artifact: ArtifactPin,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Plan {
        schema: String,
        selection_policy: String,
        registered_action_index: usize,
        supervisor_unit: String,
        episode_cpu_nodes_limit: u64,
        episode_nn_inputs_limit: u64,
        replay_expected: ExpectedPinsWire,
        registration: PinnedFile,
        prepared_action: PinnedFile,
        semantic_receipt: PinnedFile,
        catalogue: PinnedFile,
        cpu_fresh_profile: PinnedFile,
        query_prior_source: ArtifactPin,
        repair_evidence_source: ArtifactPin,
        config: ReplayConfigInput,
        resources: ReplayResourceDeclaration,
        expected_semantic_receipt_producer_scope: SemanticReceiptProducerScope,
        replay_binary_pin_scope: String,
        replay_binary: PathBuf,
        export_manifest: PathBuf,
        runtime_library: PathBuf,
        runtime_cache_root: PathBuf,
        output_root: PathBuf,
        output_directory_name: String,
    }

    fn invalid(message: &str) -> ArenaError {
        ArenaError::Invalid(message.into())
    }

    fn io_error(error: std::io::Error) -> ArenaError {
        ArenaError::Io(error.to_string())
    }

    fn json_error(error: serde_json::Error) -> ArenaError {
        invalid(&format!("closed acceptance JSON: {error}"))
    }

    fn actual_pin(bytes: &[u8]) -> ArtifactPin {
        ArtifactPin {
            bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    }

    fn checked_path(path: &Path) -> Result<(), ArenaError> {
        if !path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(invalid("absolute non-parent path required"));
        }
        let mut prefix = PathBuf::new();
        for part in path.components() {
            prefix.push(part);
            if std::fs::symlink_metadata(&prefix)
                .map_err(io_error)?
                .file_type()
                .is_symlink()
            {
                return Err(invalid("symlink in supplied path"));
            }
        }
        Ok(())
    }

    fn read_pinned(input: &PinnedFile, maximum: u64) -> Result<Vec<u8>, ArenaError> {
        if input.artifact.bytes == 0 || input.artifact.bytes > maximum {
            return Err(invalid("registered file byte budget"));
        }
        checked_path(&input.path)?;
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&input.path)
            .map_err(io_error)?;
        let before = file.metadata().map_err(io_error)?;
        if !before.is_file() || before.nlink() != 1 || before.len() != input.artifact.bytes {
            return Err(invalid("registered regular single-link file required"));
        }
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(maximum + 1)
            .read_to_end(&mut bytes)
            .map_err(io_error)?;
        let after = file.metadata().map_err(io_error)?;
        if before.dev() != after.dev()
            || before.ino() != after.ino()
            || before.len() != after.len()
            || actual_pin(&bytes) != input.artifact
        {
            return Err(invalid("registered file content/identity differs"));
        }
        Ok(bytes)
    }

    #[derive(Serialize)]
    struct PublicationAttempt {
        name: String,
        reserved_bytes: usize,
        written_bytes: usize,
        sync_completed: bool,
        postmortem_only: bool,
        serialization_error: Option<String>,
        publication_error: Option<String>,
    }
    struct ReportStore {
        root: PathBuf,
        original: Instant,
        postmortem: Instant,
        normal_credit: usize,
        failure_credit: usize,
        normal_files: usize,
        failure_files: usize,
        attempts: Vec<PublicationAttempt>,
    }
    impl ReportStore {
        fn new(root: PathBuf, original: Instant) -> Result<Self, ArenaError> {
            Ok(Self {
                root,
                original,
                postmortem: original
                    .checked_add(Duration::from_secs(2))
                    .ok_or_else(|| invalid("fixed postmortem clock overflow"))?,
                normal_credit: NORMAL_REPORT_CREDIT,
                failure_credit: FAILURE_REPORT_CREDIT,
                normal_files: REPORT_FILES,
                failure_files: REPORT_FILES,
                attempts: Vec::with_capacity(2 * REPORT_FILES),
            })
        }
        fn reserve(
            &mut self,
            name: &str,
            bytes: usize,
            postmortem_only: bool,
        ) -> Result<(usize, Instant), ArenaError> {
            let until = if postmortem_only {
                self.postmortem
            } else {
                self.original
            };
            if bytes > REPORT_BYTES
                || Instant::now() >= until
                || name.is_empty()
                || !name
                    .bytes()
                    .all(|v| v.is_ascii_alphanumeric() || matches!(v, b'-' | b'.'))
            {
                return Err(invalid("fixed report name/byte/clock bound"));
            }
            let (credit, files) = if postmortem_only {
                (&mut self.failure_credit, &mut self.failure_files)
            } else {
                (&mut self.normal_credit, &mut self.normal_files)
            };
            if bytes > *credit || *files == 0 {
                return Err(invalid("cumulative report byte/file credit exhausted"));
            }
            // Reservation is charged before serialization/create, and never
            // rolled back after a partial write, sync failure or late return.
            *credit -= bytes;
            *files -= 1;
            self.attempts.push(PublicationAttempt {
                name: name.into(),
                reserved_bytes: bytes,
                written_bytes: 0,
                sync_completed: false,
                postmortem_only,
                serialization_error: None,
                publication_error: None,
            });
            Ok((self.attempts.len() - 1, until))
        }
        fn write_reserved(
            &mut self,
            index: usize,
            bytes: &[u8],
            until: Instant,
        ) -> Result<(), ArenaError> {
            let result = (|| {
                if Instant::now() >= until {
                    return Err(invalid("publication clock exhausted before create"));
                }
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                    .open(self.root.join(&self.attempts[index].name))
                    .map_err(io_error)?;
                while self.attempts[index].written_bytes < bytes.len() {
                    if Instant::now() >= until {
                        return Err(invalid("publication clock exhausted during write"));
                    }
                    let n = file
                        .write(&bytes[self.attempts[index].written_bytes..])
                        .map_err(io_error)?;
                    if n == 0 {
                        return Err(invalid("publication zero write"));
                    }
                    self.attempts[index].written_bytes += n;
                }
                file.sync_all().map_err(io_error)?;
                self.attempts[index].sync_completed = true;
                drop(file);
                if Instant::now() >= until {
                    return Err(invalid("publication returned after fixed clock"));
                }
                Ok(())
            })();
            if let Err(error) = &result {
                self.attempts[index].publication_error = Some(error.to_string());
            }
            result
        }
        fn raw(&mut self, name: &str, bytes: &[u8], failure_only: bool) -> Result<(), ArenaError> {
            let (index, until) = self.reserve(name, bytes.len(), failure_only)?;
            self.write_reserved(index, bytes, until)
        }
        fn json<T: Serialize>(
            &mut self,
            name: &str,
            value: &T,
            cap: usize,
            failure_only: bool,
        ) -> Result<(), ArenaError> {
            let (index, until) = self.reserve(name, cap, failure_only)?;
            let mut writer = self.json_writer(index, cap, until)?;
            if let Err(error) = serde_json::to_writer(&mut writer, value) {
                self.attempts[index].serialization_error =
                    Some(format!("bounded serialization: {error}"));
                // Partial JSON is diagnostic bytes, never a complete artifact.
                let _ = self.write_reserved(index, &writer.bytes, until);
                return Err(invalid(
                    "bounded report serialization failed; partial diagnostic retained",
                ));
            }
            self.write_reserved(index, &writer.bytes, until)
        }
        fn json_writer(
            &mut self,
            index: usize,
            cap: usize,
            until: Instant,
        ) -> Result<ReportWriter, ArenaError> {
            let mut writer = ReportWriter {
                bytes: Vec::new(),
                cap,
                until,
            };
            if writer.bytes.try_reserve_exact(cap).is_err() {
                self.attempts[index].serialization_error =
                    Some("bounded report allocation failed".into());
                return Err(invalid("bounded report allocation failed"));
            }
            Ok(writer)
        }
        fn publication_audit(&mut self, name: &str) -> Result<(), ArenaError> {
            let cap = 64 * 1024;
            let (index, until) = self.reserve(name, cap, true)?;
            let mut writer = self.json_writer(index, cap, until)?;
            // Serialize the bounded ledger by borrow after reserving credit;
            // do not build a second Value tree before its output bound.
            if let Err(error) = serde_json::to_writer(&mut writer, &self.attempts) {
                self.attempts[index].serialization_error =
                    Some(format!("bounded audit serialization: {error}"));
                let _ = self.write_reserved(index, &writer.bytes, until);
                return Err(invalid("bounded publication audit serialization failed"));
            }
            self.write_reserved(index, &writer.bytes, until)
        }
        fn captured_failure(
            &mut self,
            error: &CapturedRepairWitnessFailure,
        ) -> Result<(), ArenaError> {
            let (index, until) = self.reserve(
                "captured-witness-primary-failure.json",
                CAPTURED_FAILURE_REPORT_BYTES,
                true,
            )?;
            let mut writer = self.json_writer(index, CAPTURED_FAILURE_REPORT_BYTES, until)?;
            if let Err(primary) = error.write_audit(&mut writer, until) {
                self.attempts[index].serialization_error = Some(primary.to_string());
                let _ = self.write_reserved(index, &writer.bytes, until);
                return Err(invalid(
                    "captured witness failure audit incomplete; typed primary retained",
                ));
            }
            self.write_reserved(index, &writer.bytes, until)
        }
    }
    struct ReportWriter {
        bytes: Vec<u8>,
        cap: usize,
        until: Instant,
    }
    impl Write for ReportWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if Instant::now() >= self.until
                || self
                    .bytes
                    .len()
                    .checked_add(bytes.len())
                    .is_none_or(|n| n > self.cap)
            {
                return Err(std::io::Error::other(
                    "fixed diagnostic serialization byte/clock bound",
                ));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    enum WorkFailure {
        Arena(ArenaError),
        Native(NativeReplayPrimary),
        Cpu(Box<IndependentRepairCpuFailure>),
        Captured(Box<CapturedRepairWitnessFailure>),
    }
    impl From<ArenaError> for WorkFailure {
        fn from(error: ArenaError) -> Self {
            Self::Arena(error)
        }
    }
    fn await_external_quarantine<T, U>(
        owned: &ManuallyDrop<T>,
        primary: &ManuallyDrop<U>,
        deadline: Instant,
    ) -> ! {
        // Neither object is dropped, signalled, retried or reaped here. The
        // independently configured unit supervisor must close the whole cgroup.
        let _ = (owned, primary);
        let fallback = deadline
            .checked_add(Duration::from_secs(32))
            .unwrap_or(deadline);
        while Instant::now() < fallback {
            std::thread::park_timeout(Duration::from_millis(20));
        }
        // SAFETY: failure-only Linux termination bypasses Rust and C destructors
        // so unverified pending custody is not relabelled as local cleanup.
        unsafe { libc::_exit(75) }
    }
    fn supervisor_identity(unit: &str) -> Result<Value, ArenaError> {
        if unit.len() > 128
            || !unit.ends_with(".service")
            || !unit
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || matches!(v, b'-' | b'.'))
        {
            return Err(invalid("registered external supervisor unit required"));
        }
        let mut bytes = Vec::new();
        File::open("/proc/self/cgroup")
            .map_err(io_error)?
            .take(64 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(io_error)?;
        if bytes.len() > 64 * 1024 {
            return Err(invalid("bounded own cgroup observation"));
        }
        let body = std::str::from_utf8(&bytes).map_err(|_| invalid("own cgroup UTF-8"))?;
        let suffix = format!("/{unit}");
        let group = body
            .lines()
            .find_map(|line| {
                line.strip_prefix("0::")
                    .filter(|path| path.ends_with(&suffix))
            })
            .ok_or_else(|| invalid("process is outside exact registered supervisor cgroup"))?;
        let invocation = std::env::var("INVOCATION_ID")
            .map_err(|_| invalid("actual systemd invocation identity missing"))?;
        if invocation.len() != 32 || !invocation.bytes().all(|v| v.is_ascii_hexdigit()) {
            return Err(invalid("actual systemd invocation identity malformed"));
        }
        Ok(
            json!({"unit":unit,"invocation_id":invocation,"own_cgroup":group,
            "membership_change_protection_requires_external_receipt":true,
            "physical_cleanup_requires_external_receipt":true}),
        )
    }

    pub(super) fn run() -> Result<(), ArenaError> {
        let mut args = std::env::args_os().skip(1);
        if args.next().as_deref() != Some(std::ffi::OsStr::new("--plan")) {
            return Err(invalid("usage: --plan ABSOLUTE --plan-sha256 SHA256"));
        }
        let plan_path = PathBuf::from(args.next().ok_or_else(|| invalid("plan path missing"))?);
        if args.next().as_deref() != Some(std::ffi::OsStr::new("--plan-sha256")) {
            return Err(invalid("independent plan SHA256 required"));
        }
        let plan_sha = args
            .next()
            .and_then(|value| value.into_string().ok())
            .ok_or_else(|| invalid("plan SHA256 missing"))?;
        if args.next().is_some()
            || plan_sha.len() != 64
            || !plan_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid(
                "closed arguments and lowercase independent SHA256 required",
            ));
        }
        checked_path(&plan_path)?;
        let length = std::fs::metadata(&plan_path).map_err(io_error)?.len();
        let raw_plan = read_pinned(
            &PinnedFile {
                path: plan_path,
                artifact: ArtifactPin {
                    bytes: length,
                    sha256: plan_sha,
                },
            },
            PLAN_BYTES,
        )?;
        let plan: Plan = serde_json::from_slice(&raw_plan).map_err(json_error)?;
        if plan.schema != PLAN_SCHEMA
            || plan.selection_policy != SELECTION_POLICY
            || plan.registered_action_index > 7
            || plan.episode_cpu_nodes_limit == 0
            || plan.episode_cpu_nodes_limit > 80_000_000
            || plan.episode_nn_inputs_limit == 0
            || plan.episode_nn_inputs_limit > 4096
            || plan
                .resources
                .cpu_nodes
                .checked_mul(2)
                .is_none_or(|needed| needed > plan.episode_cpu_nodes_limit)
            || plan.output_directory_name.is_empty()
            || plan.output_directory_name.len() > 64
            || !plan
                .output_directory_name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(invalid("closed plan schema/selection/output child"));
        }
        let expected = plan.replay_expected.clone().into_expected();
        if plan.registration.artifact != expected.registration_artifact
            || plan.prepared_action.artifact != expected.prepared_action_artifact
            || plan.semantic_receipt.artifact != expected.semantic_receipt_artifact
            || plan.catalogue.artifact != expected.binding.catalogue_artifact
        {
            return Err(invalid("independent original/catalogue pins differ"));
        }
        let registration = read_pinned(&plan.registration, ORIGINAL_BYTES)?;
        let prepared = read_pinned(&plan.prepared_action, ORIGINAL_BYTES)?;
        let semantic = read_pinned(&plan.semantic_receipt, ORIGINAL_BYTES)?;
        let catalogue = read_pinned(&plan.catalogue, ORIGINAL_BYTES)?;
        let profile_raw = read_pinned(&plan.cpu_fresh_profile, PLAN_BYTES)?;
        let profile: CpuFreshAssetProfile =
            serde_json::from_slice(&profile_raw).map_err(json_error)?;
        checked_path(&plan.output_root)?;
        if !std::fs::metadata(&plan.output_root)
            .map_err(io_error)?
            .is_dir()
            || plan
                .output_root
                .starts_with(std::env::current_dir().map_err(io_error)?)
        {
            return Err(invalid(
                "caller-owned output directory outside current checkout required",
            ));
        }
        for path in [
            &plan.replay_binary,
            &plan.export_manifest,
            &plan.runtime_library,
            &plan.runtime_cache_root,
        ] {
            checked_path(path)?;
        }
        let output = plan.output_root.join(&plan.output_directory_name);
        std::fs::create_dir(&output).map_err(io_error)?;
        let supervisor = supervisor_identity(&plan.supervisor_unit)?;
        let cancel = AtomicBool::new(false);
        let (selection, mut episode) = begin_registered_defend_coverage_episode(
            &expected.parent,
            &expected.binding,
            &catalogue,
            &prepared,
            &plan.prepared_action.artifact,
            plan.registered_action_index,
            1,
            Duration::from_millis(plan.resources.whole_wall_ms),
            &cancel,
        )?;
        let started = episode.original_started();
        let deadline = episode.original_deadline();
        let mut store = ReportStore::new(output.clone(), deadline)?;
        let declaration = ReplayLaunchDeclaration {
            replay: ReplayPreparationDeclaration {
                mode: ReplayInputMode::RepairOpponent4n,
                config: &plan.config,
                resources: &plan.resources,
            },
            expected_semantic_receipt_producer_scope: plan.expected_semantic_receipt_producer_scope,
            assets: ReplayLaunchAssetPaths {
                export_manifest: &plan.export_manifest,
                runtime_library: &plan.runtime_library,
                runtime_cache_root: &plan.runtime_cache_root,
            },
            output_root: &output,
            bundle_directory_name: "bundle",
            replay_binary_pin_scope: &plan.replay_binary_pin_scope,
        };
        let bundle = match prepare_repair_observed_replay_launch_bundle(
            ReplayLaunchOriginals {
                replay: ReplayOriginals {
                    registration: &registration,
                    prepared_action: &prepared,
                    semantic_receipt: &semantic,
                },
                cpu_fresh_profile: &profile_raw,
            },
            declaration,
            ReplayLaunchExpected {
                replay: &expected,
                cpu_fresh_profile_artifact: &plan.cpu_fresh_profile.artifact,
                cpu_fresh_profile: &profile,
            },
            ReplayLaunchBudget {
                construction: ReplayConstructionBudget {
                    max_outer_bytes: 2 * 1024 * 1024,
                    max_construction_credit_bytes: 16 * 1024 * 1024,
                },
                max_publication_bytes: 16 * 1024 * 1024,
                max_publication_backing_bytes: 16 * 1024 * 1024,
            },
            started,
            &cancel,
        ) {
            Ok(bundle) => bundle,
            Err(primary) => {
                let retained = ManuallyDrop::new(primary);
                let view = json!({"schema":"rz-pals-repair-episode-preparation-failure/1",
                    "stage":retained.stage,"cause":format!("{:?}",retained.cause),
                    "quarantine_error":retained.quarantine_error.as_ref().map(|e|format!("{e:?}")),
                    "diagnostics":retained.diagnostics,"publication":retained.evidence.publication(),
                    "retained_request_present":retained.evidence.retained_request().is_some(),
                    "partial_serialization_bytes":retained.evidence.partial_serialization().len(),
                    "postmortem_only":true,"original_deadline_exceeded":Instant::now()>=deadline,
                    "new_work_authority":false,"full_typed_primary_retained":true});
                let _ = store.json("preparation-failure.json", &view, 128 * 1024, true);
                if let Some(request) = retained.evidence.retained_request() {
                    let _ = store.raw("failed-prepared-request.raw", request.as_bytes(), true);
                }
                let _ = store.raw(
                    "partial-publication.raw",
                    retained.evidence.partial_serialization(),
                    true,
                );
                for (at, payload) in retained
                    .evidence
                    .retained_payloads()
                    .iter()
                    .take(12)
                    .enumerate()
                {
                    let _ = store.raw(
                        &format!("retained-payload-{at}.raw"),
                        payload.as_bytes(),
                        true,
                    );
                }
                let _ = store.publication_audit("failure-publication-attempts.json");
                await_external_quarantine(&retained, &ManuallyDrop::new(()), deadline);
            }
        };
        let nn_preflight = (|| -> Result<(u64, u64), WorkFailure> {
            let mut admitted = bundle
                .payloads()
                .iter()
                .filter_map(|payload| payload.prepared_request());
            let request = admitted
                .next()
                .ok_or_else(|| invalid("prepared request owner missing before spawn"))?;
            if admitted.next().is_some() {
                return Err(invalid("multiple prepared request owners before spawn").into());
            }
            let checked = check_replay_inputs_with_semantic_scope(
                request.as_bytes(),
                &expected,
                plan.expected_semantic_receipt_producer_scope,
                started,
            )
            .map_err(|error| WorkFailure::Native(NativeReplayPrimary::Input(Box::new(error))))?;
            if checked.mode() != ReplayInputMode::RepairOpponent4n
                || checked.deadline() != deadline
                || checked.execution_deadline() != bundle.execution_deadline()
            {
                return Err(invalid("source NN preflight original mode/clock differs").into());
            }
            let roles = checked
                .mode_requirements()
                .map_err(|error| WorkFailure::Native(NativeReplayPrimary::Run(error)))?
                .actual_role_calls();
            // Frozen CPU Fresh loads perform zero NN graph rows. Each role has
            // one private graph plus at most one public cache-miss graph. The
            // independent captured owner additionally checks C <= this same R.
            let upper = roles
                .checked_mul(4)
                .ok_or_else(|| invalid("whole NN preflight overflow"))?;
            if upper > plan.episode_nn_inputs_limit {
                return Err(invalid(
                    "declared episode NN limit is below source-owned 4R bound; no spawn",
                )
                .into());
            }
            Ok((roles, upper))
        })();
        let (source_role_calls, nn_inputs_upper) = match nn_preflight {
            Ok(bounds) => bounds,
            Err(primary) => {
                let view = json!({"schema":"rz-pals-repair-episode-nn-preflight-failure/1",
                    "source_child_spawned":false,"model_loaded":false,"nn_executed":false,
                    "new_work_authority":false,"postmortem_only":true,
                    "episode_nn_inputs_limit":plan.episode_nn_inputs_limit});
                let _ = store.json("nn-preflight-failure.json", &view, 64 * 1024, true);
                match &primary {
                    WorkFailure::Native(error) => {
                        let _ = store.json("nn-preflight-primary.json", error, 128 * 1024, true);
                    }
                    WorkFailure::Arena(error) => {
                        let _ = store.json(
                            "nn-preflight-primary.json",
                            &json!({"cause":error.to_string()}),
                            64 * 1024,
                            true,
                        );
                    }
                    WorkFailure::Cpu(_) => {}
                    WorkFailure::Captured(_) => {}
                }
                return Err(invalid(
                    "source-owned NN preflight refused before any child/model work; evidence retained",
                ));
            }
        };
        let directory = File::open(bundle.directory()).map_err(io_error)?;
        let program = File::open(&plan.replay_binary).map_err(io_error)?;
        let capture = match supervise_prepared_observed_replay(
            bundle,
            &program,
            &directory,
            ReplayCallerPolicy {
                process: ProcessLimits {
                    wall_ms: plan
                        .resources
                        .whole_wall_ms
                        .checked_sub(plan.resources.cleanup_reserve_ms)
                        .ok_or_else(|| invalid("original execution partition"))?,
                    shutdown_grace_ms: plan.resources.cleanup_reserve_ms.div_ceil(2),
                    max_output_bytes: plan.resources.output_bytes as u64,
                    max_child_processes: 1,
                },
                maximum_binary_bytes: 512 * 1024 * 1024,
                verification_read_bytes: 512 * 1024 * 1024 + 32 * 1024 * 1024,
            },
            &cancel,
        ) {
            Ok(capture) => capture,
            Err(primary) => {
                let retained = ManuallyDrop::new(primary);
                let view = json!({"schema":"rz-pals-repair-episode-caller-failure/1",
                    "cause":retained.cause.to_string(),"bundle_artifact":retained.bundle().payloads()
                        .iter().map(|payload|payload.artifact()).collect::<Vec<_>>(),
                    "postmortem_only":true,"original_deadline_exceeded":Instant::now()>=deadline,
                    "full_typed_primary_retained":true,"new_work_authority":false});
                let _ = store.json("caller-failure.json", &view, 128 * 1024, true);
                await_external_quarantine(&retained, &ManuallyDrop::new(()), deadline);
            }
        };
        // Protect custody before the first fallible publication/serialization.
        // Only confirmed success explicitly drops it; every failure holds it.
        let mut capture = ManuallyDrop::new(capture);
        let mut model_owner: Option<ManuallyDrop<NativeRoleModel>> = None;
        let work = (|| -> Result<(), WorkFailure> {
            let process = capture.process().process();
            store.json(
                "child-process-receipt.json",
                &process.receipt,
                64 * 1024,
                false,
            )?;
            store.raw("child-stdout.raw", &process.stdout, false)?;
            store.raw("child-stderr.raw", &process.stderr, false)?;
            if !capture.transport_complete() || capture.loaded_binary_error().is_some() {
                return Err(invalid(
                    "actual child closure/publication/loaded binary not confirmed; raw retained",
                )
                .into());
            }
            let material = capture.prepare_reported_repair_prior_material(
                &RepairReplayDeliveryRegistration {
                    prior: ReplayDeliveryRegistration {
                        query_prior_source: plan.query_prior_source.clone(),
                    },
                    repair_evidence_source: plan.repair_evidence_source.clone(),
                },
                &cancel,
            )?;
            let cpu = observe_independent_repair_cpu_witness(
                material.original_input(),
                material.checked_report().reported_rules(),
                &expected,
                plan.expected_semantic_receipt_producer_scope,
                &cancel,
            )
            .map_err(|error| WorkFailure::Cpu(Box::new(error)))?;
            let cpu_audit = cpu
                .audit_json()
                .map_err(|error| WorkFailure::Cpu(Box::new(error)))?;
            store.raw("independent-cpu-audit.json", &cpu_audit, false)?;
            let cache = RuntimeCache::open(&plan.runtime_cache_root)
                .map_err(|error| invalid(&format!("witness runtime cache: {error}")))?;
            let runtime = cache
                .library(&plan.runtime_library, &profile.runtime_library.sha256)
                .map_err(|error| invalid(&format!("witness runtime pin: {error}")))?;
            let runtime_sha: String = runtime
                .binary_digest()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            if runtime.bundle_digest().is_some()
                || runtime.storage().bytes != profile.runtime_library.bytes
                || runtime_sha != profile.runtime_library.sha256
            {
                return Err(invalid("witness runtime single-library byte/domain differs").into());
            }
            let mut config = PalsOnnxConfig::cpu();
            config.intra_threads = profile.intra_threads;
            config.cache_public_memory = profile.cache_public_memory;
            config.device_public_memory = false;
            let reserve = Duration::from_millis(plan.resources.cleanup_reserve_ms);
            let budget = NativeInvocationBudget::new(
                started,
                material.original_input().execution_deadline(),
                deadline,
                reserve,
            )
            .map_err(|error| invalid(&format!("witness original budget: {error:?}")))?;
            let model = NativeRoleModel::load_pinned_cpu_fresh_for_invocation(
                &plan.export_manifest,
                &profile.export_manifest.sha256,
                &runtime,
                config,
                NativeOwnerOptions {
                    drain_limit: reserve,
                    host_record_pages: None,
                },
                budget,
                &cancel,
            )
            .map_err(|error| WorkFailure::Native(NativeReplayPrimary::Load(Box::new(error))))?;
            model_owner = Some(ManuallyDrop::new(model));
            let witness = witness_captured_repair_inputs(
                &material,
                model_owner.as_mut().expect("actual loaded model"),
                &cpu,
                budget,
                &cancel,
            )
            .map_err(|error| WorkFailure::Captured(Box::new(error)))?;
            let witness_audit = witness.audit_json()?;
            store.raw("independent-captured-witness.json", &witness_audit, false)?;
            let query = episode.admit_captured_repair_prior(
                &material,
                &witness,
                &catalogue,
                &plan.catalogue.artifact,
                &cancel,
            )?;
            let cost = observe_repair_coverage_cost(&query, &selection, &cancel)?;
            if let Some(costs) = cost.costs() {
                if costs.cpu_nodes > plan.episode_cpu_nodes_limit
                    || costs.physical_nn_inputs > plan.episode_nn_inputs_limit
                {
                    return Err(
                        invalid("declared whole episode CPU/NN count bound exceeded").into(),
                    );
                }
            } else {
                return Err(invalid(
                    "whole episode CPU/NN counters unknown; count limits cannot be verified",
                )
                .into());
            }
            let cost_audit = cost.audit_json()?;
            store.raw("whole-action-cost.json", &cost_audit, false)?;
            let query_audit_raw = query.audit_json()?;
            store.raw("next-query-audit.json", &query_audit_raw, false)?;
            let query_audit: Value =
                serde_json::from_slice(&query_audit_raw).map_err(json_error)?;
            let summary = json!({
                "schema": "rz-pals-repair-query-episode-acceptance/1",
                "selection_policy": SELECTION_POLICY,
                "registered_action_index": selection.action_index(),
                "plan_artifact": actual_pin(&raw_plan),
                "source_scope": "captured_nn_input_reinference_and_independent_cpu_condition_reexecution",
                "reported_child_body_artifact": actual_pin(material.raw_native_bytes()),
                "independent_witness_artifact": actual_pin(&witness_audit),
                "different_run_ids_and_timestamps_are_not_raw_equality": true,
                "child_transport_complete": capture.transport_complete(),
                "witness_final_cpu_completion_checked": cpu.final_cpu_completion_checked(),
                "source_child_score_equivalence":"unknown_not_structurally_reported",
                "next_query": query_audit,
                "elapsed_before_final_publication_ms": started.elapsed().as_millis(),
                "whole_wall_ms": plan.resources.whole_wall_ms,
                "original_clock_retained": true,
                "publication_blocking_hard_timeout_supported": false,
                "gpu_executed": false,
                "training_executed": false,
                "v_model_selection_executed": false,
                "utility_target_admitted": false,
                "storage_cleanup_requires_external_owned_supervisor_receipt": true,
                "supervisor":supervisor,
                "exclusive_immutable_ancestor_paths_required":true,
                "hostile_ancestor_race_protection_proven":false,
                "normal_report_credit_bytes":NORMAL_REPORT_CREDIT,
                "failure_report_credit_bytes":FAILURE_REPORT_CREDIT,
                "captured_failure_report_bytes":CAPTURED_FAILURE_REPORT_BYTES,
                "report_files_per_lane":REPORT_FILES,
                "episode_cpu_nodes_limit":plan.episode_cpu_nodes_limit,
                "episode_nn_inputs_limit":plan.episode_nn_inputs_limit,
                "source_owned_role_bound":source_role_calls,
                "preflight_nn_inputs_upper":nn_inputs_upper,
                "nn_preflight_scope":"source_child_and_recorded_witness_private_1_public_miss_at_most_1_per_role;cpu_fresh_startup_0;captured_C_at_most_R",
            });
            store.json("acceptance.json", &summary, 128 * 1024, false)?;
            Ok(())
        })();
        if let Err(primary) = work {
            let retained = ManuallyDrop::new(primary);
            // Publish the primary envelope/process receipt before large raw bytes.
            let view = json!({"schema":"rz-pals-repair-episode-failure/1",
                "postmortem_only":true,"original_deadline_exceeded":Instant::now()>=deadline,
                "full_typed_primary_retained":true,"new_work_authority":false,
                "pending_child_retained":capture.process().process().pending_child.is_some(),
                "actual_postflight_error":capture.postflight_error().map(|error|error.to_string()),
                "actual_loaded_binary_error":capture.loaded_binary_error().map(|error|error.to_string()),
                "external_cgroup_cleanup_unconfirmed":true,"supervisor":supervisor});
            let _ = store.json("episode-failure.json", &view, 64 * 1024, true);
            match &*retained {
                WorkFailure::Native(error) => {
                    let _ = store.json("native-primary-failure.json", error, 4 * 1024 * 1024, true);
                }
                WorkFailure::Cpu(error) => {
                    if let Ok(raw) = error.audit_json() {
                        let _ = store.raw("cpu-primary-failure.json", &raw, true);
                    }
                }
                WorkFailure::Arena(error) => {
                    let _ = store.json(
                        "arena-primary-failure.json",
                        &json!({"cause":error.to_string()}),
                        64 * 1024,
                        true,
                    );
                }
                WorkFailure::Captured(error) => {
                    let _ = store.captured_failure(error);
                }
            }
            if let Some(owner) = model_owner.as_ref() {
                // Read the last actual native-owner snapshot only. This never
                // performs a new finish, kill, reap or CPU/NN operation.
                let snapshot = owner.finish_handle().receipt();
                let _ = store.json(
                    "failure-native-owner-receipt.json",
                    &snapshot,
                    128 * 1024,
                    true,
                );
            }
            let process = capture.process().process();
            let _ = store.json(
                "failure-child-process-receipt.json",
                &process.receipt,
                64 * 1024,
                true,
            );
            let _ = store.raw("failure-child-stdout.raw", &process.stdout, true);
            let _ = store.raw("failure-child-stderr.raw", &process.stderr, true);
            let _ = store.publication_audit("failure-publication-attempts.json");
            // model_owner also remains in this scope while the external unit
            // supervisor closes the group; never run an extra native cleanup.
            let _ = &model_owner;
            await_external_quarantine(&capture, &retained, deadline);
        }
        if capture.process().process().pending_child.is_some() || !capture.transport_complete() {
            await_external_quarantine(&capture, &ManuallyDrop::new(()), deadline);
        }
        // SAFETY: the complete original transport and the typed recorded-input
        // wrapper's physical worker join were checked before successful Query.
        unsafe {
            if let Some(owner) = model_owner.as_mut() {
                ManuallyDrop::drop(owner);
            }
            ManuallyDrop::drop(&mut capture);
        }
        println!(
            "registered actual CPU repair/Query episode completed; external final cleanup receipt remains required"
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn serialization_refusal_irrevocably_consumes_reserved_credit() {
            let until = Instant::now().checked_add(Duration::from_secs(60)).unwrap();
            let mut store = ReportStore::new(PathBuf::new(), until).unwrap();
            let (index, deadline) = store.reserve("bounded.json", 8, false).unwrap();
            let mut writer = store.json_writer(index, 8, deadline).unwrap();
            assert!(serde_json::to_writer(&mut writer, &"x".repeat(32)).is_err());
            assert_eq!(store.normal_credit, NORMAL_REPORT_CREDIT - 8);
            assert_eq!(store.normal_files, REPORT_FILES - 1);
            assert_eq!(store.failure_credit, FAILURE_REPORT_CREDIT);
            assert_eq!(store.attempts.len(), 1);
            assert!(writer.bytes.len() <= 8);
        }

        #[test]
        fn failure_lane_uses_original_deadline_plus_two_without_reopening_work() {
            let now = Instant::now();
            let original = now.checked_sub(Duration::from_secs(1)).unwrap();
            let mut store = ReportStore::new(PathBuf::new(), original).unwrap();
            assert_eq!(store.postmortem, original + Duration::from_secs(2));
            assert!(store.reserve("normal.json", 16, false).is_err());
            let (_, until) = store.reserve("failure.json", 16, true).unwrap();
            assert_eq!(until, store.postmortem);
            assert_eq!(store.normal_credit, NORMAL_REPORT_CREDIT);
            assert_eq!(store.failure_credit, FAILURE_REPORT_CREDIT - 16);
            store.postmortem = now.checked_sub(Duration::from_millis(1)).unwrap();
            assert!(store.reserve("late.json", 16, true).is_err());
            assert_eq!(store.failure_credit, FAILURE_REPORT_CREDIT - 16);
        }

        #[test]
        fn cumulative_credit_and_namespace_are_checked_before_publication() {
            let until = Instant::now().checked_add(Duration::from_secs(60)).unwrap();
            let mut store = ReportStore::new(PathBuf::new(), until).unwrap();
            assert!(store.reserve("../escape.json", 1, false).is_err());
            assert_eq!(store.normal_credit, NORMAL_REPORT_CREDIT);
            store.reserve("first.json", REPORT_BYTES, false).unwrap();
            assert!(store.reserve("second.json", REPORT_BYTES, false).is_err());
            assert_eq!(store.normal_credit, NORMAL_REPORT_CREDIT - REPORT_BYTES);
            assert_eq!(store.normal_files, REPORT_FILES - 1);
            store.reserve("failure.json", REPORT_BYTES, true).unwrap();
            assert_eq!(store.failure_credit, 0);
            assert!(store.reserve("another-failure.json", 1, true).is_err());
        }
    }
}
