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
            "domain":"rovezero.actual-continuation-diagnostic-raw/1",
            "case":opaque(&case), "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER",
            "source_commit":env!("RZ_PALS_COMPILED_SOURCE_COMMIT"),
            "startup_config_sha256":config_pin,"registered_image":registered_image,
            "authority":authority.as_ref().map(opaque),"original_wall_ms":wall_ms,
            "root_hard_ms":ROOT.as_millis(),"run_elapsed_ns":run_elapsed,
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
    pub fn main() -> ExitCode {
        match run() {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(error) => {
                let detail: String = error.to_string().chars().take(512).collect();
                let _ = emit(
                    &json!({"domain":"rovezero.actual-continuation-diagnostic-raw/1",
                    "status":"PREPARATION_OR_PUBLICATION_FAILED","detail":detail,
                    "case_acceptance":"NOT_ISSUED_BY_DIAGNOSTIC_RUNNER",
                    "actual_nn_scope":"UNKNOWN_UNLESS_ORIGINAL_RAW_PROVES_IT"}),
                );
                ExitCode::FAILURE
            }
        }
    }
}
