//! One registered actual V-free process/root, one fixed diagnostic recipe.
//! Register this example's real mapped image; no baseline image may be relabelled.
//! Stdout is a bounded raw observation, never Y1/T1/R1 acceptance. The parent
//! supervisor must retain original process-group/pipe/deadline/artifact guards.
#[cfg(not(feature = "pals-adaptive-diagnostics"))]
fn main() -> std::process::ExitCode {
    eprintln!("pals_continuation_diagnostic requires pals-adaptive-diagnostics");
    std::process::ExitCode::FAILURE
}

#[cfg(feature = "pals-adaptive-diagnostics")]
fn main() -> std::process::ExitCode {
    actual::main()
}

#[cfg(feature = "pals-adaptive-diagnostics")]
mod actual {
    use rz_contracts::ProcessEpoch;
    use rz_contracts::pals::RoleOutput;
    use rz_contracts::pals_adaptive::RoleTicketEnvelope;
    use rz_experiments::{PalsCoordinatorPolicyV5, PalsCpuSelectionV5};
    use rz_position::Position;
    use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
    use rz_search::pals::engine::PostRepairRecheckPolicy;
    use rz_search::pals::review_context::ReviewView;
    use rz_uci::pals_adaptive::{
        AdaptiveCpuSliceDiagnostic, AdaptiveCpuSliceDiagnosticError,
        AdaptiveCpuSliceDiagnosticOptions, ContinuationDiagnostic, ContinuationDiagnosticOptions,
    };
    use rz_uci::pals_adaptive_bootstrap::diagnostic::{
        DeploymentDiagnosticObserversV5, prepare_deployment_diagnostic_actual_with_cancellation,
    };
    use rz_uci::pals_adaptive_startup::{
        PalsStartupOptionsV5, PinnedPalsStartupV5, parse_pals_search_settings_v5,
    };
    use rz_uci::pals_native::ticketed::{
        NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
        VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
        VirtualReviewPreparedTicket,
    };
    use rz_uci::search_driver::SearchSessionDriver;
    use serde_json::json;
    use std::fmt::Debug;
    use std::io::{self, Write};
    use std::process::ExitCode;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, TryLockError};
    use std::time::{Duration, Instant};

    const RAW_BYTES_MAX: usize = 4 * 1024 * 1024;
    const HOT_RECORDS_MAX: usize = 64;
    const ROOT: Duration = Duration::from_secs(5);
    const Y1_FEN: &str = "3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1";

    #[derive(Clone, Copy, Debug)]
    enum Case {
        Y1Natural,
        T1Pending,
        R1Repair,
    }
    #[derive(Clone, Copy, Debug)]
    enum HotEvent {
        NativeSubmitted {
            ticket: RoleTicketEnvelope,
        },
        NativeCompleted {
            ticket: RoleTicketEnvelope,
            physical: PhysicalRoleCompletion,
            output_present: bool,
            evidence: Option<TicketExecutionEvidence>,
            invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
        },
        NativeRetired {
            ticket: RoleTicketEnvelope,
            accepted_by_native: bool,
        },
        Dispatch {
            ticket: RoleTicketEnvelope,
            source_question: rz_contracts::Digest,
            source_view: rz_contracts::pals_adaptive::ReviewViewSeal,
            stage: rz_search::pals::coordinator::RefinementStage,
        },
        CpuSliceInstall(Result<(), AdaptiveCpuSliceDiagnosticError>),
    }
    impl HotEvent {
        fn wire(&self) -> serde_json::Value {
            match self {
                Self::NativeSubmitted { ticket } => json!({
                    "source":"NativeSubmitted","ticket":opaque(ticket)
                }),
                Self::NativeCompleted {
                    ticket,
                    physical,
                    output_present,
                    evidence,
                    invocation,
                } => json!({
                    "source":"NativeCompleted","ticket":opaque(ticket),"physical":opaque(physical),
                    "output_present":output_present,
                    "completed_nn_inputs":evidence.and_then(|e|e.completed_nn_inputs),
                    "measurement_complete":evidence.map(|e|e.measurement_complete),
                    "execution_evidence_debug":opaque(evidence),"invocation_debug":opaque(invocation)
                }),
                Self::NativeRetired {
                    ticket,
                    accepted_by_native,
                } => json!({
                    "source":"NativeRetired","ticket":opaque(ticket),"accepted_by_native":accepted_by_native
                }),
                Self::Dispatch {
                    ticket,
                    source_question,
                    source_view,
                    stage,
                } => json!({
                    "source":"SearchDispatch","ticket":opaque(ticket),"source_question":opaque(source_question),
                    "source_view":opaque(source_view),"stage":opaque(stage)
                }),
                Self::CpuSliceInstall(result) => {
                    json!({"source":"CpuSliceInstall","result_debug":opaque(result)})
                }
            }
        }
    }
    struct HotJournal {
        records: Mutex<Vec<HotEvent>>,
        unavailable: AtomicU64,
    }
    impl HotJournal {
        fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let mut records = Vec::new();
            records.try_reserve_exact(HOT_RECORDS_MAX)?;
            Ok(Self {
                records: Mutex::new(records),
                unavailable: AtomicU64::new(0),
            })
        }
        fn record(&self, event: HotEvent) {
            match self.records.try_lock() {
                Ok(mut records) if records.len() < HOT_RECORDS_MAX => records.push(event),
                Ok(_) => {
                    self.unavailable.fetch_or(1, Ordering::AcqRel);
                }
                Err(TryLockError::WouldBlock) => {
                    self.unavailable.fetch_or(2, Ordering::AcqRel);
                }
                Err(TryLockError::Poisoned(_)) => {
                    self.unavailable.fetch_or(4, Ordering::AcqRel);
                }
            }
        }
        fn readback(&self) -> Result<Vec<HotEvent>, &'static str> {
            match self.records.try_lock() {
                Ok(records) => Ok(records.clone()),
                Err(TryLockError::WouldBlock) => Err("HotJournalBusy"),
                Err(TryLockError::Poisoned(_)) => Err("HotJournalPoisoned"),
            }
        }
    }
    struct NativeObserver(Arc<HotJournal>);
    impl VirtualReviewCollectionObserver for NativeObserver {
        fn submitted(
            &mut self,
            ticket: &RoleTicketEnvelope,
            _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
            _: &VirtualReviewPreparedTicket,
        ) -> Result<(), RoleFailure> {
            self.0.record(HotEvent::NativeSubmitted { ticket: *ticket });
            Ok(())
        }
        fn completed(
            &mut self,
            ticket: &RoleTicketEnvelope,
            physical: PhysicalRoleCompletion,
            output: Option<&RoleOutput>,
            evidence: Option<TicketExecutionEvidence>,
            invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
        ) -> Result<(), RoleFailure> {
            self.0.record(HotEvent::NativeCompleted {
                ticket: *ticket,
                physical,
                output_present: output.is_some(),
                evidence,
                invocation,
            });
            Ok(())
        }
        fn retired(
            &mut self,
            ticket: &RoleTicketEnvelope,
            accepted: bool,
        ) -> Result<(), RoleFailure> {
            self.0.record(HotEvent::NativeRetired {
                ticket: *ticket,
                accepted_by_native: accepted,
            });
            Ok(())
        }
    }
    fn fail(message: &'static str) -> Box<dyn std::error::Error> {
        io::Error::other(message).into()
    }
    fn opaque(value: &impl Debug) -> String {
        format!("{value:?}")
    }
    fn emit(value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() >= RAW_BYTES_MAX {
            return Err(fail("DiagnosticOutputCapacity"));
        }
        let mut output = io::stdout().lock();
        output.write_all(&bytes)?;
        output.write_all(b"\n")?;
        output.flush()?;
        Ok(())
    }
    fn arguments() -> Result<(Case, u64, ProcessEpoch, Vec<String>), Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.len() != 6 || args.iter().any(|s| s.len() > 8192) {
            return Err(fail("SixBoundedArgumentsRequired"));
        }
        let mut case = None;
        let mut wall = None;
        let mut epoch = None;
        let mut startup = Vec::with_capacity(3);
        for arg in args {
            if let Some(value) = arg.strip_prefix("--diagnostic-case=") {
                if case.is_some() {
                    return Err(fail("DuplicateCase"));
                }
                case = Some(match value {
                    "y1-natural" => Case::Y1Natural,
                    "t1-pending" => Case::T1Pending,
                    "r1-repair" => Case::R1Repair,
                    _ => return Err(fail("UnknownCase")),
                });
            } else if let Some(value) = arg.strip_prefix("--diagnostic-wall-ms=") {
                if wall.replace(value.parse::<u64>()?).is_some() {
                    return Err(fail("DuplicateWall"));
                }
            } else if let Some(value) = arg.strip_prefix("--diagnostic-epoch=") {
                if epoch.replace(value.parse::<u64>()?).is_some() {
                    return Err(fail("DuplicateEpoch"));
                }
            } else {
                startup.push(arg);
            }
        }
        let wall = wall.ok_or_else(|| fail("WallRequired"))?;
        let epoch = epoch.ok_or_else(|| fail("EpochRequired"))?;
        if wall == 0 || wall > 60_000 || epoch == 0 || startup.len() != 3 {
            return Err(fail("FiniteOriginalWindowRequired"));
        }
        Ok((
            case.ok_or_else(|| fail("CaseRequired"))?,
            wall,
            ProcessEpoch(epoch),
            startup,
        ))
    }
    fn run() -> Result<bool, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let (case, wall_ms, epoch, startup_args) = arguments()?;
        let until = started
            .checked_add(Duration::from_millis(wall_ms))
            .ok_or_else(|| fail("WindowOverflow"))?;
        let options = PalsStartupOptionsV5::parse(&startup_args)?;
        if options.metadata_preflight {
            return Err(fail("ActualOnly"));
        }
        let cwd = std::env::current_dir()?;
        let pinned = Arc::new(PinnedPalsStartupV5::open(options, &cwd)?);
        if wall_ms > pinned.lock.manifest.window.wall_time_max_ms {
            return Err(fail("OriginalWindowExceeded"));
        }
        let endpoint = &pinned.config.endpoint;
        if endpoint.cpu.cpu_r != PalsCpuSelectionV5::Own
            || endpoint.cpu.cpu_t != PalsCpuSelectionV5::Disabled
            || pinned
                .config
                .native_options
                .native_loading_profile
                .is_some()
            || pinned.config.native_options.cuda_device_id.is_some()
            || pinned.config.native_options.cuda_arena_bytes.is_some()
            || pinned.config.native_options.device_public_memory
        {
            return Err(fail("ActualCpuOwnNoTeacherProfileRequired"));
        }
        let selected = parse_pals_search_settings_v5(&pinned.config, until)?;
        let (line, beam, role_max, cpu_max) = match case {
            Case::Y1Natural => (2, 1, 8, 32),
            Case::T1Pending => (2, 2, 8, 128),
            Case::R1Repair => (4, 1, 16, 1024),
        };
        let expected_policy = if matches!(case, Case::T1Pending) {
            PalsCoordinatorPolicyV5::CoordinatorAdaptive
        } else {
            PalsCoordinatorPolicyV5::CoordinatorFixed
        };
        if selected.config.line_plies != line
            || selected.config.beam_width != beam
            || selected.config.max_role_calls != role_max
            || selected.limits.max_cpu_nodes != cpu_max
            || selected.limits.cpu_depth != 2
            || endpoint.coordinator.policy != expected_policy
            || (matches!(case, Case::Y1Natural)
                && endpoint.coordinator.slices.slice_native_nodes_max != 1)
            || (matches!(case, Case::T1Pending) && endpoint.coordinator.slices.role_tickets_max < 2)
            || (matches!(case, Case::R1Repair)
                && selected.post_repair_recheck == PostRepairRecheckPolicy::Disabled)
        {
            return Err(fail("PinnedCaseRecipeDiffersNoAutomaticRewrite"));
        }
        let drain = Duration::from_millis(pinned.config.native_options.drain_ms);
        let config_pin = pinned.config_sha256.clone();
        let registered_image = pinned.config.endpoint.binary.clone();
        let hot = Arc::new(HotJournal::new()?);
        let cpu = if matches!(case, Case::Y1Natural) {
            Some(AdaptiveCpuSliceDiagnostic::new(
                AdaptiveCpuSliceDiagnosticOptions { records_max: 64 },
            )?)
        } else {
            None
        };
        let continuation = if matches!(case, Case::Y1Natural) {
            None
        } else {
            Some(ContinuationDiagnostic::new(
                ContinuationDiagnosticOptions {
                    pending_snapshots_max: if matches!(case, Case::T1Pending) {
                        16
                    } else {
                        0
                    },
                    resolution_records_max: if matches!(case, Case::R1Repair) {
                        64
                    } else {
                        0
                    },
                    role_consumption: matches!(case, Case::R1Repair),
                },
            )?)
        };
        let dispatch_hot = Arc::clone(&hot);
        let admission_hot = Arc::clone(&hot);
        let admission_cpu = cpu.clone();
        let observers = DeploymentDiagnosticObserversV5 {
            native_observer: Box::new(NativeObserver(Arc::clone(&hot))),
            dispatch_observer: Arc::new(move |ticket, input, _| {
                dispatch_hot.record(HotEvent::Dispatch {
                    ticket: *ticket,
                    source_question: input.question_digest,
                    source_view: input.view_seal,
                    stage: input.stage,
                });
                Ok(())
            }),
            root_admission: Arc::new(move |manager| {
                if let Some(cpu) = &admission_cpu {
                    admission_hot.record(HotEvent::CpuSliceInstall(
                        manager.install_cpu_slice_diagnostic(cpu.clone()),
                    ));
                }
                Ok(())
            }),
            root_completion: Arc::new(|_, _| Ok(())),
        };
        // One signal created before actual preparation and retained through all
        // original startup/cleanup stages. No callback resets cancellation.
        let startup_cancel = Arc::new(AtomicBool::new(false));
        let prepared = prepare_deployment_diagnostic_actual_with_cancellation(
            pinned,
            epoch,
            observers,
            until,
            Arc::clone(&startup_cancel),
        )?;
        let driver = prepared.driver();
        // All subsequent errors still call finish_process on this same owner.
        let mut authority = None;
        let mut run_elapsed = None;
        let execution = (|| -> Result<_, Box<dyn std::error::Error>> {
            if let Some(diagnostic) = &continuation {
                driver.install_continuation_diagnostic(diagnostic.clone())?;
            }
            let hard = Instant::now()
                .checked_add(ROOT)
                .ok_or_else(|| fail("RootOverflow"))?;
            let shutdown = hard
                .checked_add(drain)
                .ok_or_else(|| fail("RootDrainOverflow"))?;
            if shutdown > until || startup_cancel.load(Ordering::Acquire) {
                return Err(fail("OriginalWindowCannotFitOneRoot"));
            }
            let context = prepared.context(hard, shutdown)?;
            authority = Some(context.authority);
            let position = if matches!(case, Case::Y1Natural) {
                Position::from_fen(Y1_FEN)?
            } else {
                Position::startpos()
            };
            let run_started = Instant::now();
            let returned = driver.run(&position, &context, &mut |_| {});
            run_elapsed = Some(u64::try_from(run_started.elapsed().as_nanos())?);
            Ok(returned)
        })();
        // This performs original current-context retirement and original drain.
        // Physical unknown/quarantine remains on the retained owner on error.
        let process = prepared.finish_process(until);
        let completion = driver.completion_diagnostic();
        let cpu_snapshot = cpu.as_ref().map(|d| d.snapshot());
        let continuation_snapshot = continuation.as_ref().map(|d| d.snapshot());
        let hot_snapshot = hot.readback();
        let scalar_observations = scalar_wire::project(
            &prepared.clock(),
            authority,
            &hot_snapshot,
            &cpu_snapshot,
            &continuation_snapshot,
            continuation.as_ref().map(ContinuationDiagnostic::options),
        );
        let runtime_closure = scalar_wire::runtime_closure(&execution, &process, &completion);
        let available = hot.unavailable.load(Ordering::Acquire) == 0
            && hot_snapshot.is_ok()
            && authority.is_some()
            && completion.as_ref().is_ok_and(|c| {
                c.unavailable_reason.is_none()
                    && c.last
                        .as_ref()
                        .is_some_and(|last| Some(last.authority) == authority)
            })
            && cpu_snapshot.as_ref().is_none_or(|s| {
                s.as_ref()
                    .is_ok_and(|s| s.unavailable.is_none() && s.authority == authority)
            })
            && continuation_snapshot.as_ref().is_none_or(|s| {
                s.as_ref().is_ok_and(|s| {
                    s.unavailable_bits == 0
                        && s.authority == authority
                        && (!matches!(case, Case::R1Repair)
                            || (s.role_install_observed == Some(true)
                                && s.role_readback_root_fence == Some(true)
                                && s.role_unavailable.is_none()))
                })
            });
        let successful_return = execution.as_ref().is_ok_and(|run| run.is_ok())
            && process
                .as_ref()
                .is_ok_and(|p| p.permits_archive_close() && p.store_archive_closed());
        emit(&json!({
            "domain":"rovezero.actual-continuation-diagnostic-raw/3",
            "case":opaque(&case), "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER",
            "source_commit":env!("RZ_PALS_COMPILED_SOURCE_COMMIT"),
            "startup_config_sha256":config_pin,"registered_image":registered_image,
            "authority":authority.as_ref().map(opaque),"original_wall_ms":wall_ms,
            "root_hard_ms":ROOT.as_millis(),"run_elapsed_ns":run_elapsed,
            "scalar_observations":scalar_observations,
            "runtime_closure":runtime_closure,
            "driver_execution_debug":opaque(&execution),"completion_debug":opaque(&completion),
            "original_process_finish_debug":opaque(&process),
            "diagnostic_capture_available":available,"hot_unavailable_bits":hot.unavailable.load(Ordering::Acquire),
            "hot_records":hot_snapshot.as_ref().ok().map(|events|events.iter().map(HotEvent::wire).collect::<Vec<_>>()),
            "hot_readback_error":hot_snapshot.as_ref().err(),"cpu_slice_debug":opaque(&cpu_snapshot),
            "continuation_debug":opaque(&continuation_snapshot),
            "event_presence_verdict":"ROOT_INDEPENDENT_AUDIT_REQUIRED",
            "requested_yield_acceptance":"NOT_ISSUED",
            "external_signal_cancellation":"NOT_OBSERVED_BY_THIS_CONSUMER",
            "training_executed":false,"optimizer_steps":0
        }))?;
        Ok(successful_return && available)
    }

    // Auxiliary raw projection only. Borrow original DTOs after original drain;
    // never reconstruct a view/payload, parse Debug, or issue acceptance.
    mod scalar_wire {
        use super::{HotEvent, opaque};
        use rz_contracts::pals::{RepresentationKey, Role, SearchAuthority};
        use rz_contracts::pals_adaptive::{
            CpuJobEnvelope, CpuJobId, ReviewViewSeal, RoleTicketEnvelope,
        };
        use rz_contracts::{Deadline, Digest, ExecutionId, RequestId, StateIdentity};
        use rz_search::pals::coordinator::{FrameId, StepExecution};
        use rz_search::pals::engine::adaptive_port::{
            AdaptiveRoleConsumptionEvent, AdaptiveRoleConsumptionOutcome,
        };
        use rz_uci::engine::ProcessClock;
        use rz_uci::pals_adaptive::{
            AdaptiveCpuSliceDiagnosticError, AdaptiveCpuSliceDiagnosticSnapshot,
            AdaptiveCpuSliceObservation, AdaptiveCpuSliceResultObservation,
            AdaptivePendingQuestionRow, ContinuationDiagnosticError, ContinuationDiagnosticOptions,
            ContinuationDiagnosticSnapshot, ContinuationPendingObservation,
            ContinuationResolutionObservation, ContinuationResolutionSource,
            ContinuationTransitionKind,
        };
        use serde_json::{Value, json};
        use std::time::Instant;
        const DOMAIN: &str = "rovezero.continuation-diagnostic-scalar/1";
        fn hex(value: Digest) -> String {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut text = String::with_capacity(64);
            for byte in value.0 {
                text.push(HEX[(byte >> 4) as usize] as char);
                text.push(HEX[(byte & 15) as usize] as char);
            }
            text
        }
        fn authority(a: &SearchAuthority) -> Value {
            json!({"domain":"search_authority","process_epoch":a.epoch.0.to_string(),
                "game_generation":a.game.0.to_string(),"root_generation":a.root.0.to_string(),
                "implementation_digest":hex(a.implementation)})
        }
        fn request(id: RequestId) -> Value {
            json!({"domain":"role_request","process_epoch":id.epoch.0.to_string(),"request_sequence":id.sequence.to_string()})
        }
        fn execution(id: ExecutionId) -> Value {
            json!({"domain":"execution","process_epoch":id.epoch.0.to_string(),"execution_sequence":id.sequence.to_string()})
        }
        fn frame(id: FrameId) -> Value {
            json!({"domain":"refinement_frame","authority":authority(&id.authority),"frame_sequence":id.sequence.to_string()})
        }
        fn job(id: CpuJobId) -> Value {
            json!({"domain":"cpu_job","authority":authority(&id.authority),"job_sequence":id.sequence.to_string()})
        }
        fn state(id: StateIdentity) -> Value {
            json!({"domain":"rules_state","rules_state_owner":id.owner.0.to_string(),
                "state_revision":id.revision.0.to_string(),"semantic_digest":hex(id.semantic)})
        }
        fn deadline(value: Deadline) -> Value {
            json!({"domain":"monotonic_deadline","process_clock_epoch":value.clock.0.0.to_string(),
                "deadline_at_ns":value.at.0.to_string()})
        }
        fn original_instant(clock: &ProcessClock, instant: Instant) -> Value {
            match clock.tick_at(instant) {
                Ok(tick) => {
                    json!({"status":"COPIED_ORIGINAL_CLOCK","process_clock_epoch":clock.epoch().0.to_string(),
                    "monotonic_tick_ns":tick.0.to_string()})
                }
                Err(error) => {
                    json!({"status":"UNAVAILABLE","process_clock_epoch":clock.epoch().0.to_string(),
                    "monotonic_tick_ns":null,"original_clock_error_debug":opaque(&error)})
                }
            }
        }
        fn role(value: Role) -> &'static str {
            match value {
                Role::Proposer => "Proposer",
                Role::Critic => "Critic",
                Role::Validator => "Validator",
            }
        }
        fn view(value: &ReviewViewSeal) -> Value {
            json!({"domain":"review_view_seal","authority":authority(&value.authority),
                "view_sequence":value.view_sequence.to_string(),"context_policy_digest":hex(value.context_policy),
                "view_digest":hex(value.view_digest),"root_state":state(value.root_state),
                "actual_tokens":value.actual_tokens,"selected_records":value.selected_records,
                "anchors":value.anchors,"length_bucket":value.length_bucket})
        }
        fn representation(value: &RepresentationKey) -> Value {
            json!({"input_digest":hex(value.input),"history_digest":hex(value.history),
                "candidates_digest":hex(value.candidates),"mask_positions_digest":hex(value.mask_positions),
                "model_digest":hex(value.model),"encoding_digest":hex(value.encoding),
                "precision":opaque(&value.precision),"frozen_model_epoch":value.frozen_epoch.to_string(),
                "record_revision":value.record_revision.to_string(),"role":role(value.role)})
        }
        fn ticket(value: &RoleTicketEnvelope) -> Value {
            json!({"request":request(value.id),"authority":authority(&value.authority),
                "ticket_question_digest":hex(value.question),"view":view(&value.view),
                "representation":representation(&value.representation),"role":role(value.role),
                "original_deadline":deadline(value.original_deadline)})
        }
        fn step(value: StepExecution) -> Value {
            match value {
                StepExecution::Role(id) => json!({"kind":"Role","request":request(id)}),
                StepExecution::Cpu {
                    job: id,
                    execution: run,
                } => json!({"kind":"Cpu","job":job(id),"execution":execution(run)}),
                StepExecution::ExternalCpu {
                    job: id,
                    execution: run,
                } => json!({"kind":"ExternalCpu","job":job(id),"execution":execution(run)}),
            }
        }
        fn hot(value: &HotEvent) -> Value {
            match value {
                HotEvent::NativeSubmitted { ticket: value } => {
                    json!({"source":"NativeSubmitted","ticket":ticket(value),"native_os_owner":null})
                }
                HotEvent::NativeCompleted {
                    ticket: value,
                    physical,
                    output_present,
                    evidence,
                    invocation,
                } => json!({
                    "source":"NativeCompleted","ticket":ticket(value),"physical_completion":opaque(physical),
                    "output_present":output_present,"native_os_owner":null,
                    "execution":evidence.and_then(|e|e.execution).map(execution),
                    "completed_nn_inputs":evidence.and_then(|e|e.completed_nn_inputs),
                    "measurement_complete":evidence.map(|e|e.measurement_complete),
                    "physical_fresh":evidence.and_then(|e|e.physical_fresh),
                    "physical_invocation":invocation.map(|i|json!({"invocation_key":hex(i.invocation_key),
                        "accepted_seed_digest":i.accepted_seed_digest.map(hex),"physical_fresh":i.physical_fresh}))
                }),
                HotEvent::NativeRetired {
                    ticket: value,
                    accepted_by_native,
                } => json!({
                    "source":"NativeRetired","ticket":ticket(value),"accepted_by_native":accepted_by_native,
                    "semantic_consumption":null,"native_os_owner":null}),
                HotEvent::Dispatch {
                    ticket: value,
                    source_question,
                    source_view,
                    stage,
                } => json!({
                    "source":"SearchDispatch","ticket":ticket(value),"source_question_digest":hex(*source_question),
                    "submitted_view":view(source_view),"stage":opaque(stage)}),
                HotEvent::CpuSliceInstall(result) => match result {
                    Ok(()) => json!({"source":"CpuSliceInstall","install_return":"Ok"}),
                    Err(error) => {
                        json!({"source":"CpuSliceInstall","install_return":"Err","error_kind":opaque(error)})
                    }
                },
            }
        }
        fn pending_row(row: &AdaptivePendingQuestionRow) -> Value {
            json!({"preparation":{"domain":"native_preparation","preparation_sequence":row.preparation.0.to_string()},
                "frame":row.frame.map(frame),"stage":opaque(&row.stage),"source_question_digest":hex(row.question),
                "view":view(&row.view),"retained_view_digest":hex(row.retained_view_digest),
                "retained_view_seal_digest":hex(row.retained_view_seal_digest),
                "selected_record_revisions":row.selected_record_revisions.iter().map(u64::to_string).collect::<Vec<_>>(),
                "ticket":row.ticket.as_ref().map(ticket),"registry_observed_phase":row.registry_observed_phase.map(|p|opaque(&p)),
                "dispatch_pending":row.dispatch_pending,"completion_received":row.completion_received,
                "consumption_recorded":row.consumption_recorded})
        }
        fn pending(event: &ContinuationPendingObservation, clock: &ProcessClock) -> Value {
            let snapshot = match &event.snapshot {
                Ok(snapshot) => {
                    json!({"status":"COPIED_SOURCE_DTO","authority":authority(&snapshot.authority),
                    "phase":opaque(&snapshot.phase),"sampled_tick_ns":snapshot.sampled_at.0.to_string(),
                    "original_deadline":deadline(snapshot.original_deadline),
                    "questions":snapshot.questions.iter().map(pending_row).collect::<Vec<_>>()})
                }
                Err(error) => {
                    json!({"status":"UNAVAILABLE","source_error_kind":opaque(error),"questions":null})
                }
            };
            json!({"source":opaque(&event.source),"trigger_request":request(event.trigger),
                "queue_admitted_at":event.queue_admitted_at.map(|instant|original_instant(clock,instant)),
                "snapshot":snapshot})
        }
        fn consumed(event: &AdaptiveRoleConsumptionEvent) -> Value {
            let outcome = match event.outcome {
                AdaptiveRoleConsumptionOutcome::SemanticResult => json!({"kind":"SemanticResult"}),
                AdaptiveRoleConsumptionOutcome::ConsumedView => json!({"kind":"ConsumedView"}),
                AdaptiveRoleConsumptionOutcome::DuplicateNoNewConsumption => {
                    json!({"kind":"DuplicateNoNewConsumption"})
                }
                AdaptiveRoleConsumptionOutcome::Rejected(reason) => {
                    json!({"kind":"Rejected","reason":opaque(&reason)})
                }
            };
            json!({"authority":authority(&event.authority),"request":request(event.request),"role":role(event.role),
                "ticket_question_digest":hex(event.question),"submitted_source_question_digest":event.submitted_question.map(hex),
                "ticket_view_digest":hex(event.ticket_view),"submitted_view_digest":hex(event.submitted_view),
                "frame":event.frame.map(frame),"source_stage":opaque(&event.stage),"outcome":outcome,
                "completed_evidence_digest":event.completed_evidence.map(hex),"prior_semantic_attempt":event.prior_semantic_attempt})
        }
        fn resolved(event: &ContinuationResolutionObservation) -> Value {
            let transition = match event.transition {
                ContinuationTransitionKind::Continue { next_stage } => {
                    json!({"kind":"Continue","next_stage":opaque(&next_stage)})
                }
                ContinuationTransitionKind::Complete { scope } => {
                    json!({"kind":"Complete","scope":opaque(&scope)})
                }
                ContinuationTransitionKind::Unknown { reason } => {
                    json!({"kind":"Unknown","reason":reason})
                }
            };
            let source = match event.source {
                ContinuationResolutionSource::MachineTransition {
                    checked_rules_pair,
                    accepted,
                } => {
                    json!({"kind":"MachineTransition","checked_rules_pair":checked_rules_pair,"accepted":accepted})
                }
                ContinuationResolutionSource::ClosedFramePublication { published } => {
                    json!({"kind":"ClosedFramePublication","published":published})
                }
            };
            json!({"authority":authority(&event.authority),"evidence":{"frame":frame(event.evidence.frame),
                "stage":opaque(&event.evidence.stage),"execution":step(event.evidence.execution),
                "submitted_question_digest":hex(event.evidence.submitted_question),"submitted_view":view(&event.evidence.submitted_view),
                "completed_evidence_digest":hex(event.evidence.completed_evidence)},"transition":transition,"source":source,
                "checked_at":{"process_clock_epoch":event.clock.0.0.to_string(),"checked_tick_ns":event.checked_at.0.to_string()},
                "closed_frame":event.closed_frame.map(frame),"closed_scope":event.closed_scope.map(|s|opaque(&s)),
                "closed_unknown":event.closed_unknown})
        }
        fn job_envelope(value: &CpuJobEnvelope) -> Value {
            json!({"job":job(value.id),"state":state(value.state_identity),"question_digest":hex(value.question),
                "root_restriction_digest":hex(value.root_restriction),"evaluation_namespace_digest":hex(value.evaluation_namespace),
                "search_conditions_digest":hex(value.search_conditions),"total_native_node_credit":value.total_native_node_credit,
                "hard_deadline":deadline(value.hard_deadline)})
        }
        fn cpu_slice(value: &AdaptiveCpuSliceObservation) -> Value {
            let s = &value.scope;
            let g = &value.grant;
            let result = match &value.result {
                AdaptiveCpuSliceResultObservation::Report {
                    completion,
                    completed_depth,
                    reported_work,
                    score_scope,
                    reused_completed_depth,
                    yield_cause,
                    yield_ack_sequence,
                    publish_ack_sequence,
                    elapsed,
                } => json!({"kind":"Report", "completion":opaque(completion),
                        "completed_depth":completed_depth,"reported_work":{"nodes":reported_work.nodes,
                            "quiescence_nodes":reported_work.quiescence_nodes,"tt_hits":reported_work.tt_hits},
                        "score_scope":opaque(score_scope),"reused_completed_depth":reused_completed_depth,
                        "yield_cause":yield_cause.map(|c|opaque(&c)),"yield_ack_sequence":yield_ack_sequence.map(|s|s.to_string()),
                        "publish_ack_sequence":publish_ack_sequence.to_string(),"elapsed_duration_ns":elapsed.as_nanos().to_string()}),
                AdaptiveCpuSliceResultObservation::Error(error) => {
                    json!({"kind":"Error","source_error_kind":opaque(error)})
                }
            };
            json!({"source_sequence":value.sequence.to_string(),"scope":{"authority":authority(&s.authority),
                "frame":frame(s.frame),"stage":opaque(&s.stage),"question_digest":hex(s.question),"view":view(&s.view),
                "public_context_digest":hex(s.public_context),"job":job_envelope(&s.job),
                "requested_yield_sequence":s.requested_yield_sequence.map(|s|s.to_string()),"prior_yield_ack":s.prior_yield_ack.to_string(),
                "binding_progress_sequence":s.binding_progress_sequence.to_string(),"progress_since_slice":s.progress_since_slice,
                "feedback_already_forwarded":s.feedback_already_forwarded},
                "grant":{"job":job(g.job),"execution":execution(g.execution),
                    "cpu_owner":{"domain":"cpu_worker","worker_id":g.owner.0.to_string(),"os_owner":null},
                    "node_credit":g.node_credit,"soft_lease_duration_ns":g.soft_lease.as_nanos().to_string(),
                    "control_sequence":g.control_sequence.to_string()},
                "actual_work":value.actual_work.map(|w|json!({"nodes":w.nodes,"quiescence_nodes":w.quiescence_nodes,"tt_hits":w.tt_hits})),
                "result":result,"cumulative_nodes":value.cumulative_nodes,"report_progress_sequence":value.report_progress_sequence.to_string(),
                "continuation_retained":value.continuation_retained,"terminal":value.terminal,"admission_rejected":value.admission_rejected,
                "accounting_after_settlement":value.accounting_after_settlement.map(|a|json!({"global_spent":a.global_spent,
                    "global_reserved":a.global_reserved,"job_spent":a.job_spent,"job_reserved":a.job_reserved,"job_quarantined":a.job_quarantined}))})
        }
        fn presence(enabled: bool, empty: bool) -> &'static str {
            if !enabled {
                "DISABLED"
            } else if empty {
                "NOT_OBSERVED"
            } else {
                "SOURCE_RECORDS_PRESENT_ONLY"
            }
        }
        fn cpu_channel(
            value: &Option<
                Result<AdaptiveCpuSliceDiagnosticSnapshot, AdaptiveCpuSliceDiagnosticError>,
            >,
        ) -> Value {
            match value {
                None => json!({"enabled":false,"event_presence":"DISABLED","records":null}),
                Some(Err(error)) => {
                    json!({"enabled":true,"status":"UNAVAILABLE","source_error_kind":opaque(error),"records":null})
                }
                Some(Ok(s)) => {
                    json!({"enabled":true,"authority":s.authority.as_ref().map(authority),
                    "event_presence":presence(true,s.records.is_empty()),"records_max":s.records_max,
                    "source_unavailable":s.unavailable.map(|u|opaque(&u)),"records":s.records.iter().map(cpu_slice).collect::<Vec<_>>()})
                }
            }
        }
        fn continuation_channel(
            value: &Option<Result<ContinuationDiagnosticSnapshot, ContinuationDiagnosticError>>,
            options: Option<ContinuationDiagnosticOptions>,
            clock: &ProcessClock,
        ) -> Value {
            match value {
                None => json!({"enabled":false,"event_presence":"DISABLED","snapshot":null}),
                Some(Err(error)) => {
                    json!({"enabled":true,"status":"UNAVAILABLE","source_error_kind":opaque(error),"snapshot":null})
                }
                Some(Ok(s)) => {
                    let pending_enabled = options.is_some_and(|o| o.pending_snapshots_max != 0);
                    let resolution_enabled = options.is_some_and(|o| o.resolution_records_max != 0);
                    let role_enabled = options.is_some_and(|o| o.role_consumption);
                    json!({"enabled":true,"authority":s.authority.as_ref().map(authority),"source_unavailable_bits":s.unavailable_bits,
                        "pending":{"enabled":pending_enabled,"event_presence":presence(pending_enabled,s.pending.is_empty()),
                            "records":pending_enabled.then(||s.pending.iter().map(|e|pending(e,clock)).collect::<Vec<_>>())},
                        "resolutions":{"enabled":resolution_enabled,"event_presence":presence(resolution_enabled,s.resolutions.is_empty()),
                            "records":resolution_enabled.then(||s.resolutions.iter().map(resolved).collect::<Vec<_>>())},
                        "role_consumption":{"enabled":role_enabled,"event_presence":presence(role_enabled,s.role_events.is_empty()),
                            "install_observed":s.role_install_observed,"readback_original_root_fence":s.role_readback_root_fence,
                            "source_unavailable":s.role_unavailable.map(|u|opaque(&u)),
                            "records":role_enabled.then(||s.role_events.iter().map(consumed).collect::<Vec<_>>())}})
                }
            }
        }

        // Direct final DTO projection after the original finish_process call.
        // A logical return, root reuse, process shutdown and parent supervision
        // are separate facts. None of these observations replaces a guard.
        fn physical(value: rz_uci::search_driver::DriverPhysicalCompletion) -> &'static str {
            use rz_uci::search_driver::DriverPhysicalCompletion;
            match value {
                DriverPhysicalCompletion::Confirmed => "Confirmed",
                DriverPhysicalCompletion::Unknown => "Unknown",
            }
        }
        fn failure(value: &rz_uci::search_driver::SearchSessionFailure) -> Value {
            json!({"code":value.code,"detail":value.detail,
                "physical_completion_classification":physical(value.physical_completion),
                "classification_scope":"ORIGINAL_DRIVER_FAILURE_ONLY_NOT_A_PHYSICAL_FENCE"})
        }
        fn root_closure(value: &rz_uci::pals_adaptive::AdaptiveShutdownReceipt) -> Value {
            json!({"domain":"adaptive_root_shutdown_receipt","authority":authority(&value.authority),
                "native_owner_joined":value.native_owner_joined,
                "native_owner_quiesced":value.native_owner_quiesced,
                "native_physical_shutdown_confirmed":value.native_physical_shutdown_confirmed,
                "own_cpu_joined":value.own_cpu_joined,"expected_own_cpu_owners":value.expected_own_cpu_owners,
                "own_cpu_panicked":value.own_cpu_panicked,
                "external_shutdown_confirmed":value.external_shutdown_confirmed,
                "external_owner_quiesced":value.external_owner_quiesced,
                "auxiliary_shutdown_confirmed":value.auxiliary_shutdown_confirmed,
                "native_cpu_spent":value.native_cpu_spent,"native_cpu_reserved":value.native_cpu_reserved,
                "pending_tickets":value.pending_tickets,"quarantined":value.quarantined,
                "permits_semantic_closure":value.permits_semantic_closure(),
                "permits_root_reuse":value.permits_root_reuse(),
                "permits_archive_close":value.permits_archive_close(),
                "native_os_owner":null,"native_os_owner_observation":"NOT_OBSERVED_BY_THIS_CONSUMER"})
        }
        fn checker_shutdown(value: rz_search::cpu_checker::CheckerShutdown) -> Value {
            json!({"domain":"external_checker_shutdown_receipt",
                "process_identity":value.process_identity.map(|p|json!({"domain":"external_os_process",
                    "pid":p.pid.to_string(),"process_group":p.process_group.to_string(),"proc_start_ticks":p.proc_start_ticks.to_string()})),
                "stop_sent":value.stop_sent,"quit_sent":value.quit_sent,"exit_observed":value.exit_observed,
                "stdout_drained":value.stdout_drained,"stderr_drained":value.stderr_drained,
                "exit_code":value.exit_code,"exit_signal":value.exit_signal,"cleanup_complete":value.cleanup_complete,
                "quarantined":value.quarantined,"ownership_lost":value.ownership_lost,
                "stdout_bytes":value.stdout_bytes,"stderr_bytes":value.stderr_bytes})
        }
        fn semantic_completion(value: rz_search::pals::engine::PalsCompletion) -> &'static str {
            use rz_search::pals::engine::PalsCompletion;
            match value {
                PalsCompletion::RoundLimit => "RoundLimit",
                PalsCompletion::CpuNodeLimit => "CpuNodeLimit",
                PalsCompletion::RoleCallLimit => "RoleCallLimit",
                PalsCompletion::Deadline => "Deadline",
                PalsCompletion::Canceled => "Canceled",
                PalsCompletion::Capacity => "Capacity",
                PalsCompletion::Terminal => "Terminal",
            }
        }
        pub(super) fn runtime_closure(
            execution: &Result<
                Result<
                    rz_uci::search_driver::SearchSessionReport,
                    rz_uci::search_driver::SearchSessionFailure,
                >,
                Box<dyn std::error::Error>,
            >,
            process: &Result<
                rz_uci::pals_adaptive::AdaptiveSessionProcessReceipt,
                rz_uci::search_driver::SearchSessionFailure,
            >,
            completion: &Result<
                rz_uci::pals_adaptive::AdaptiveCompletionDiagnostic,
                rz_contracts::ContractError,
            >,
        ) -> Value {
            let driver = match execution {
                Ok(Ok(_)) => json!({"status":"DRIVER_RETURNED_OK","failure":null,
                    "consumer_error":null,"physical_completion_classification":null}),
                Ok(Err(error)) => json!({"status":"DRIVER_RETURNED_ERROR","failure":failure(error),
                    "consumer_error":null,"physical_completion_classification":physical(error.physical_completion)}),
                Err(error) => {
                    json!({"status":"CONSUMER_FAILED_DRIVER_RETURN_NOT_OBSERVED","failure":null,
                    "consumer_error":error.to_string().chars().take(512).collect::<String>(),"physical_completion_classification":null})
                }
            };
            let process = match process {
                Ok(receipt) => json!({"status":"ORIGINAL_PROCESS_RECEIPT_RETURNED","failure":null,
                    "receipt":{"domain":"adaptive_session_process_receipt","process_epoch":receipt.epoch().0.to_string(),
                        "last_authority":receipt.last_authority().as_ref().map(authority),
                        "native_owner_joined":receipt.native_owner_joined(),
                        "native_physical_shutdown_confirmed":receipt.native_physical_shutdown_confirmed(),
                        "store_archive_closed":receipt.store_archive_closed(),
                        "permits_archive_close":receipt.permits_archive_close(),"permits_process_close":receipt.permits_process_close(),
                        "external_shutdown":receipt.external_shutdown().map(checker_shutdown),
                        "latest_root":receipt.latest_root().map(root_closure),
                        "native_os_owner":null,"native_os_owner_observation":"NOT_OBSERVED_BY_THIS_CONSUMER"}}),
                Err(error) => {
                    json!({"status":"ORIGINAL_PROCESS_FINISH_ERROR","failure":failure(error),
                    "receipt":null,"receipt_unavailable_reason":"ORIGINAL_FINISH_DID_NOT_RETURN_A_RECEIPT"})
                }
            };
            let completion = match completion {
                Ok(value) => {
                    json!({"status":if value.unavailable_reason.is_some(){"UNAVAILABLE"}else if value.last.is_some(){"OBSERVED"}else{"NOT_OBSERVED"},
                    "process_epoch":value.process_epoch.0.to_string(),"implementation_digest":hex(value.implementation),
                    "unavailable_reason":value.unavailable_reason,"last":value.last.as_ref().map(|last|json!({
                        "authority":authority(&last.authority),"semantic_completion":last.semantic_completion.map(semantic_completion),
                        "driver_return":match last.driver_return{rz_uci::pals_adaptive::AdaptiveCompletionReturn::Ok=>"Ok",rz_uci::pals_adaptive::AdaptiveCompletionReturn::Err=>"Err"},
                        "failure":last.failure.as_ref().map(|f|json!({"code":f.code,"detail":f.detail,"detail_truncated":f.detail_truncated,
                            "physical_completion_classification":physical(f.physical_completion)}))}))})
                }
                Err(error) => {
                    json!({"status":"UNAVAILABLE","last":null,"source_error_debug":opaque(error)})
                }
            };
            json!({"domain":"rovezero.continuation-diagnostic-runtime-closure/1",
                "driver_execution":driver,"final_semantic_observation":completion,"process_finish":process,
                "parent_supervision":"NOT_OBSERVED_BY_THIS_CONSUMER",
                "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER"})
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use rz_contracts::{
                ClockDomain, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch, RootGeneration,
                StateRevision,
            };
            use rz_search::pals::coordinator::{RefinementStage, StepEvidence};
            use rz_search::pals::engine::adaptive_port::AdaptiveRoleConsumptionStage;
            use rz_uci::pals_adaptive::{
                AdaptivePendingSnapshotUnavailable, ContinuationPendingSource,
            };
            use std::time::Duration;
            fn scope() -> SearchAuthority {
                SearchAuthority {
                    epoch: ProcessEpoch(42),
                    game: GameGeneration(5),
                    root: RootGeneration(7),
                    implementation: Digest([0xa5; 32]),
                }
            }
            fn seal() -> ReviewViewSeal {
                ReviewViewSeal {
                    authority: scope(),
                    view_sequence: 11,
                    context_policy: Digest([1; 32]),
                    view_digest: Digest([2; 32]),
                    root_state: StateIdentity {
                        owner: OwnerId(19),
                        revision: StateRevision(23),
                        semantic: Digest([3; 32]),
                    },
                    actual_tokens: 128,
                    selected_records: 0,
                    anchors: 1,
                    length_bucket: 128,
                }
            }

            #[test]
            fn closure_wire_keeps_failure_classification_separate_from_missing_receipt() {
                use rz_uci::search_driver::{DriverPhysicalCompletion, SearchSessionFailure};
                let execution = Ok(Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "OriginalDriverFailure",
                    detail: "authored DTO".into(),
                }));
                let process = Err(SearchSessionFailure::physical_completion_unknown());
                let completion = Ok(rz_uci::pals_adaptive::AdaptiveCompletionDiagnostic {
                    process_epoch: rz_contracts::ProcessEpoch(9),
                    implementation: Digest([4; 32]),
                    last: None,
                    observed_driver_returns: None,
                    successful_driver_return_reason_counts: None,
                    semantic_observed_but_driver_failed: None,
                    unobserved_completion_returns: None,
                    unavailable_reason: Some("authored unavailable"),
                });
                let value = runtime_closure(&execution, &process, &completion);
                assert_eq!(
                    value["driver_execution"]["physical_completion_classification"],
                    "Confirmed"
                );
                assert_eq!(
                    value["process_finish"]["failure"]["physical_completion_classification"],
                    "Unknown"
                );
                assert!(value["process_finish"]["receipt"].is_null());
                assert_eq!(value["final_semantic_observation"]["status"], "UNAVAILABLE");
                assert!(value["final_semantic_observation"]["last"].is_null());
                assert_eq!(value["parent_supervision"], "NOT_OBSERVED_BY_THIS_CONSUMER");
            }
            #[test]
            fn closure_wire_semantic_result_does_not_mint_driver_or_physical_success() {
                use rz_uci::pals_adaptive::{
                    AdaptiveCompletionDiagnostic, AdaptiveCompletionObservation,
                    AdaptiveCompletionReturn,
                };
                use rz_uci::search_driver::SearchSessionFailure;
                let execution = Err(Box::<dyn std::error::Error>::from(
                    "authored consumer failure",
                ));
                let process = Err(SearchSessionFailure::physical_completion_unknown());
                let completion = Ok(AdaptiveCompletionDiagnostic {
                    process_epoch: rz_contracts::ProcessEpoch(9),
                    implementation: Digest([4; 32]),
                    last: Some(AdaptiveCompletionObservation {
                        authority: scope(),
                        semantic_completion: Some(
                            rz_search::pals::engine::PalsCompletion::RoleCallLimit,
                        ),
                        driver_return: AdaptiveCompletionReturn::Err,
                        failure: None,
                    }),
                    observed_driver_returns: None,
                    successful_driver_return_reason_counts: None,
                    semantic_observed_but_driver_failed: None,
                    unobserved_completion_returns: None,
                    unavailable_reason: Some("authored prefix"),
                });
                let value = runtime_closure(&execution, &process, &completion);
                assert_eq!(
                    value["driver_execution"]["status"],
                    "CONSUMER_FAILED_DRIVER_RETURN_NOT_OBSERVED"
                );
                assert!(value["driver_execution"]["physical_completion_classification"].is_null());
                assert_eq!(
                    value["final_semantic_observation"]["last"]["semantic_completion"],
                    "RoleCallLimit"
                );
                assert_eq!(
                    value["final_semantic_observation"]["last"]["driver_return"],
                    "Err"
                );
                assert!(value["process_finish"]["receipt"].is_null());
            }

            #[test]
            fn scalar_wire_keeps_large_ids_and_typed_domains_distinct() {
                let a = scope();
                let r = request(RequestId::new(a.epoch, u64::MAX));
                let e = execution(ExecutionId {
                    epoch: a.epoch,
                    sequence: u64::MAX,
                });
                let f = frame(FrameId {
                    authority: a,
                    sequence: u64::MAX,
                });
                assert_eq!(r["request_sequence"], u64::MAX.to_string());
                assert_eq!(e["execution_sequence"], u64::MAX.to_string());
                assert_eq!(f["frame_sequence"], u64::MAX.to_string());
                assert_eq!(r["domain"], "role_request");
                assert_eq!(e["domain"], "execution");
                assert_eq!(f["domain"], "refinement_frame");
                assert_ne!(r, e);
                assert_eq!(f["authority"]["root_generation"], "7");
                assert_eq!(hex(Digest([0xa5; 32])), "a5".repeat(32));
                assert_eq!(view(&seal())["root_state"]["rules_state_owner"], "19");
            }
            #[test]
            fn scalar_wire_does_not_merge_semantic_consume_machine_and_publication() {
                let a = scope();
                let f = FrameId {
                    authority: a,
                    sequence: 29,
                };
                let request_id = RequestId::new(a.epoch, 31);
                let evidence = StepEvidence {
                    frame: f,
                    stage: RefinementStage::Repair,
                    execution: StepExecution::Role(request_id),
                    submitted_question: Digest([4; 32]),
                    submitted_view: seal(),
                    completed_evidence: Digest([5; 32]),
                };
                let accepted = ContinuationResolutionObservation {
                    authority: a,
                    evidence,
                    transition: ContinuationTransitionKind::Unknown {
                        reason: "projection-only source case",
                    },
                    checked_at: MonotonicTick(100),
                    clock: ClockDomain(a.epoch),
                    source: ContinuationResolutionSource::MachineTransition {
                        checked_rules_pair: false,
                        accepted: true,
                    },
                    closed_frame: Some(f),
                    closed_scope: None,
                    closed_unknown: Some("projection-only source case"),
                };
                let mut failed_publication = accepted;
                failed_publication.source =
                    ContinuationResolutionSource::ClosedFramePublication { published: false };
                let first = resolved(&accepted);
                let second = resolved(&failed_publication);
                assert_eq!(first["source"]["kind"], "MachineTransition");
                assert_eq!(first["source"]["accepted"], true);
                assert!(first["source"].get("published").is_none());
                assert_eq!(second["source"]["kind"], "ClosedFramePublication");
                assert_eq!(second["source"]["published"], false);
                assert!(second["source"].get("accepted").is_none());
                assert_eq!(first["evidence"], second["evidence"]);
                let semantic = AdaptiveRoleConsumptionEvent {
                    authority: a,
                    request: request_id,
                    role: Role::Proposer,
                    question: Digest([4; 32]),
                    submitted_question: Some(Digest([4; 32])),
                    ticket_view: Digest([2; 32]),
                    submitted_view: Digest([2; 32]),
                    frame: Some(f),
                    stage: AdaptiveRoleConsumptionStage::SemanticCommit,
                    outcome: AdaptiveRoleConsumptionOutcome::SemanticResult,
                    completed_evidence: Some(Digest([5; 32])),
                    prior_semantic_attempt: Some(false),
                };
                let mut first_consume = semantic;
                first_consume.stage = AdaptiveRoleConsumptionStage::ViewConsumption;
                first_consume.outcome = AdaptiveRoleConsumptionOutcome::ConsumedView;
                first_consume.completed_evidence = None;
                first_consume.prior_semantic_attempt = None;
                first_consume.frame = None;
                let semantic = consumed(&semantic);
                let consumed = consumed(&first_consume);
                assert_eq!(semantic["outcome"]["kind"], "SemanticResult");
                assert_eq!(consumed["outcome"]["kind"], "ConsumedView");
                assert!(consumed["completed_evidence_digest"].is_null());
                assert!(consumed["prior_semantic_attempt"].is_null());
                assert_eq!(semantic["request"], consumed["request"]);
            }
            #[test]
            fn scalar_wire_preserves_disabled_empty_failed_and_unknown_owner() {
                let clock = ProcessClock::new(scope().epoch);
                let empty = project(&clock, None, &Ok(Vec::new()), &None, &None, None);
                assert!(empty["native_os_owner"].is_null());
                assert!(empty["authority"].is_null());
                assert_eq!(empty["hot"]["event_presence"], "NOT_OBSERVED");
                assert_eq!(empty["cpu_slices"]["event_presence"], "DISABLED");
                assert!(empty["cpu_slices"]["records"].is_null());
                assert_eq!(empty["case_acceptance"], "NOT_ISSUED_BY_DIAGNOSTIC_RUNNER");
                let failed = project(
                    &clock,
                    None,
                    &Err("HotJournalPoisoned"),
                    &Some(Err(AdaptiveCpuSliceDiagnosticError::Poisoned)),
                    &Some(Err(ContinuationDiagnosticError::Busy)),
                    None,
                );
                assert_eq!(failed["hot"]["event_presence"], "UNAVAILABLE");
                assert!(failed["hot"]["records"].is_null());
                assert_eq!(failed["cpu_slices"]["status"], "UNAVAILABLE");
                assert!(failed["continuation"]["snapshot"].is_null());
            }
            #[test]
            fn scalar_wire_does_not_zero_fill_unavailable_snapshot_or_original_clock() {
                let before = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
                let clock = ProcessClock::new(scope().epoch);
                let event = ContinuationPendingObservation {
                    source: ContinuationPendingSource::QueueAdmitted,
                    trigger: RequestId::new(scope().epoch, 3),
                    queue_admitted_at: Some(before),
                    snapshot: Err(AdaptivePendingSnapshotUnavailable::RegistryBindingMismatch),
                };
                let wire = pending(&event, &clock);
                assert_eq!(wire["queue_admitted_at"]["status"], "UNAVAILABLE");
                assert!(wire["queue_admitted_at"]["monotonic_tick_ns"].is_null());
                assert_eq!(wire["snapshot"]["status"], "UNAVAILABLE");
                assert!(wire["snapshot"]["questions"].is_null());
                assert_eq!(wire["trigger_request"]["request_sequence"], "3");
            }
        }

        pub(super) fn project(
            clock: &ProcessClock,
            source_authority: Option<SearchAuthority>,
            hot_value: &Result<Vec<HotEvent>, &'static str>,
            cpu: &Option<
                Result<AdaptiveCpuSliceDiagnosticSnapshot, AdaptiveCpuSliceDiagnosticError>,
            >,
            continuation: &Option<
                Result<ContinuationDiagnosticSnapshot, ContinuationDiagnosticError>,
            >,
            options: Option<ContinuationDiagnosticOptions>,
        ) -> Value {
            json!({"domain":DOMAIN,"projection_scope":"TYPED_SOURCE_IDENTITIES_AND_STATUS_ONLY",
                "authority":source_authority.as_ref().map(authority),"original_process_clock_epoch":clock.epoch().0.to_string(),
                "native_os_owner":null,"native_os_owner_observation":"NOT_OBSERVED_BY_THIS_CONSUMER",
                "hot":{"event_presence":match hot_value {Ok(events)=>presence(true,events.is_empty()),Err(_)=>"UNAVAILABLE"},
                    "source_error":hot_value.as_ref().err(),"records":hot_value.as_ref().ok().map(|events|events.iter().map(hot).collect::<Vec<_>>())},
                "cpu_slices":cpu_channel(cpu),"continuation":continuation_channel(continuation,options,clock),
                "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER"})
        }
    }

    pub fn main() -> ExitCode {
        match run() {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(error) => {
                let detail: String = error.to_string().chars().take(512).collect();
                let _ = emit(
                    &json!({"domain":"rovezero.actual-continuation-diagnostic-raw/3",
                    "status":"PREPARATION_OR_PUBLICATION_FAILED","detail":detail,
                    "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER",
                    "actual_nn_scope":"UNKNOWN_UNLESS_ORIGINAL_RAW_PROVES_IT","scalar_observations":null,"runtime_closure":null}),
                );
                ExitCode::FAILURE
            }
        }
    }
}
