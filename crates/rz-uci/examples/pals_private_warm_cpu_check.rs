//! Finite, opt-in actual NativeWarm CPU role/acceptance check.
//!
//! Required inputs (all paths absolute and outside this checkout):
//! --export-manifest PATH --export-manifest-sha256 HEX64
//! --runtime PATH --runtime-sha256 HEX64 --runtime-output-root DIRECTORY
//!
//! No model export, training, child engine, CUDA or synthetic latent is used.
//! This is a checked role-consumer harness, not a full PalsEngine search or a
//! strength/performance experiment. The caller owns the runtime copy directory.
//! A 60-second outer deadline includes pinning/loading/startup/calls; an additional
//! five seconds is reserved for known worker shutdown. An external supervisor
//! should also bound the process/output pipe to 65 seconds.

#[cfg(not(feature = "onnx-cpu"))]
fn main() -> std::process::ExitCode {
    eprintln!("pals_private_warm_cpu_check requires the explicit onnx-cpu feature");
    std::process::ExitCode::FAILURE
}

#[cfg(feature = "onnx-cpu")]
fn main() -> std::process::ExitCode {
    check::main()
}

#[cfg(feature = "onnx-cpu")]
mod check {
    use rz_eval::pals_model::PalsModelConfig;
    use rz_eval::pals_onnx::PalsOnnxConfig;
    use rz_eval::runtime_pin::RuntimeLibraryPin;
    use rz_position::{BoardMove, Position};
    use rz_search::pals::engine::{
        RecordKind, RoleAcceptance, RoleError, RoleEvaluation, RoleLogicalContext, RoleModel,
        RoleQuery, RoleQueryPurpose, RoleRecord, RoleSearchClosure,
    };
    use rz_search::pals::store::{LineId, Move16, SituationId, StateId};
    use rz_uci::pals_native::{
        NativeOwnerOptions, NativePrivateWarmObservation, NativeQueryKind, NativeRoleFinishHandle,
        NativeRoleModel, prepare_role_input,
    };
    use serde_json::{Value, json};
    use sha2::{Digest as _, Sha256};
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::{Duration, Instant};

    const WALL: Duration = Duration::from_secs(60);
    const CLEANUP: Duration = Duration::from_secs(5);
    const OUTPUT_LIMIT: usize = 1024 * 1024;
    const SCHEMA: &str = "rz-pals-native-private-warm-cpu-check/1";

    struct Args {
        manifest: PathBuf,
        manifest_sha256: String,
        runtime: PathBuf,
        runtime_sha256: String,
        runtime_output_root: PathBuf,
    }

    #[derive(Clone, Copy)]
    struct Failure {
        stage: &'static str,
        code: &'static str,
    }

    impl Failure {
        fn wire(self) -> Value {
            json!({"stage": self.stage, "code": self.code})
        }
    }

    fn role_failure(stage: &'static str, error: RoleError) -> Failure {
        Failure {
            stage,
            code: match error {
                RoleError::Unavailable => "unavailable",
                RoleError::InvalidOutput => "invalid_output",
                RoleError::Canceled => "canceled",
                RoleError::Deadline => "deadline",
                RoleError::PhysicalCompletionUnknown => "physical_completion_unknown",
                RoleError::Backend(_) => "backend_error",
            },
        }
    }

    fn require(condition: bool, stage: &'static str) -> Result<(), Failure> {
        if condition {
            Ok(())
        } else {
            Err(Failure {
                stage,
                code: "assertion_failed",
            })
        }
    }

    fn args() -> Result<Args, Failure> {
        let mut values: [Option<String>; 5] = std::array::from_fn(|_| None);
        let mut arguments = std::env::args().skip(1);
        for _ in 0..5 {
            let flag = arguments.next().ok_or(Failure {
                stage: "arguments",
                code: "five_explicit_pinned_inputs_required",
            })?;
            let index = match flag.as_str() {
                "--export-manifest" => 0,
                "--export-manifest-sha256" => 1,
                "--runtime" => 2,
                "--runtime-sha256" => 3,
                "--runtime-output-root" => 4,
                _ => {
                    return Err(Failure {
                        stage: "arguments",
                        code: "unknown_option",
                    });
                }
            };
            let value = arguments.next().ok_or(Failure {
                stage: "arguments",
                code: "missing_value",
            })?;
            require(value.len() <= 4096 && values[index].is_none(), "arguments")?;
            values[index] = Some(value);
        }
        require(arguments.next().is_none(), "arguments")?;
        let [manifest, manifest_sha256, runtime, runtime_sha256, root] = values;
        let args = Args {
            manifest: manifest
                .ok_or(Failure {
                    stage: "arguments",
                    code: "export_manifest_required",
                })?
                .into(),
            manifest_sha256: manifest_sha256.ok_or(Failure {
                stage: "arguments",
                code: "export_digest_required",
            })?,
            runtime: runtime
                .ok_or(Failure {
                    stage: "arguments",
                    code: "runtime_required",
                })?
                .into(),
            runtime_sha256: runtime_sha256.ok_or(Failure {
                stage: "arguments",
                code: "runtime_digest_required",
            })?,
            runtime_output_root: root
                .ok_or(Failure {
                    stage: "arguments",
                    code: "runtime_output_root_required",
                })?
                .into(),
        };
        for digest in [&args.manifest_sha256, &args.runtime_sha256] {
            require(
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "digest_arguments",
            )?;
        }
        // RuntimeLibraryPin requires caller-owned storage outside Git. Do not
        // create an output root or erase runtime copies on the caller's behalf.
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or(Failure {
                stage: "arguments",
                code: "checkout_boundary_unavailable",
            })?
            .canonicalize()
            .map_err(|_| Failure {
                stage: "arguments",
                code: "checkout_boundary_unavailable",
            })?;
        for path in [&args.manifest, &args.runtime, &args.runtime_output_root] {
            require(path.is_absolute(), "absolute_external_inputs")?;
            let resolved = path.canonicalize().map_err(|_| Failure {
                stage: "absolute_external_inputs",
                code: "input_unavailable",
            })?;
            require(!resolved.starts_with(&checkout), "absolute_external_inputs")?;
        }
        require(args.runtime_output_root.is_dir(), "runtime_output_root")?;
        Ok(args)
    }

    pub fn main() -> ExitCode {
        let started = Instant::now();
        let args = match args() {
            Ok(args) => args,
            Err(error) => return publish(failure_report(error, started)),
        };
        let deadline = started + WALL;
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let (send, receive) = mpsc::sync_channel(1);
        if std::thread::Builder::new()
            .name("pals-actual-native-warm-cpu-check".into())
            .spawn(move || {
                let report = execute(args, started, deadline, &worker_cancel);
                let _ = send.send(report);
            })
            .is_err()
        {
            return publish(failure_report(
                Failure {
                    stage: "worker",
                    code: "spawn_failed",
                },
                started,
            ));
        }
        match receive.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(report) => publish(report),
            Err(_) => {
                cancel.store(true, Ordering::Release);
                let report = receive.recv_timeout((deadline + CLEANUP).saturating_duration_since(Instant::now())).unwrap_or_else(|_| {
                    json!({
                        "schema": SCHEMA,
                        "passed": false,
                        "provider": "cpu",
                        "gpu_execution": false,
                        "actual_training": false,
                        "failure": {"stage": "outer_deadline", "code": "worker_completion_unconfirmed"},
                        "elapsed_ms": started.elapsed().as_millis(),
                        "native_shutdown_confirmed": false,
                        "physical_completion": "unknown",
                        "logical_cancel_does_not_certify_native_fence": true
                    })
                });
                // The extra interval is cleanup only. A report received here
                // can pass only if work completed within the original deadline.
                let mut report = report;
                if report
                    .get("work_completed_within_deadline")
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    report["passed"] = json!(false);
                }
                report["supervisor_cancel_issued_at_work_deadline"] = json!(true);
                // Return from main without joining or fabricating recovery of
                // a stuck native owner. Output failure still exits nonzero.
                publish(report)
            }
        }
    }

    fn failure_report(error: Failure, started: Instant) -> Value {
        json!({
            "schema": SCHEMA, "passed": false, "provider": "cpu",
            "gpu_execution": false, "actual_training": false,
            "failure": error.wire(), "elapsed_ms": started.elapsed().as_millis()
        })
    }

    fn publish(report: Value) -> ExitCode {
        let passed = report.get("passed").and_then(Value::as_bool) == Some(true);
        let bytes = match serde_json::to_vec(&report) {
            Ok(bytes) if bytes.len() < OUTPUT_LIMIT => bytes,
            _ => {
                let _ = io::stdout().write_all(b"{\"schema\":\"rz-pals-native-private-warm-cpu-check/1\",\"passed\":false,\"failure\":\"output_limit_or_serialization\"}\n");
                return ExitCode::FAILURE;
            }
        };
        let mut output = io::stdout().lock();
        if output
            .write_all(&bytes)
            .and_then(|_| output.write_all(b"\n"))
            .is_err()
        {
            return ExitCode::FAILURE;
        }
        if passed {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }

    fn execute(args: Args, started: Instant, deadline: Instant, cancel: &AtomicBool) -> Value {
        let pin = match RuntimeLibraryPin::copy_verified(
            &args.runtime,
            &args.runtime_output_root,
            &args.runtime_sha256,
        ) {
            Ok(pin) => pin,
            Err(_) => {
                return failure_report(
                    Failure {
                        stage: "runtime_pin",
                        code: "pin_failed",
                    },
                    started,
                );
            }
        };
        if Instant::now() >= deadline {
            return failure_report(
                Failure {
                    stage: "runtime_pin",
                    code: "deadline",
                },
                started,
            );
        }
        let mut model = match NativeRoleModel::load_pinned_cpu_private_warm_with_options(
            &args.manifest,
            &args.manifest_sha256,
            &pin,
            PalsOnnxConfig::cpu(),
            NativeOwnerOptions {
                drain_limit: CLEANUP,
                host_record_pages: None,
            },
        ) {
            Ok(model) => model,
            Err(error) => {
                return failure_report(
                    role_failure("load_native_cpu_private_warm", error),
                    started,
                );
            }
        };
        let handle = model.finish_handle();
        let mut steps = Vec::new();
        let result = checks(&mut model, &handle, deadline, cancel, &mut steps);
        let identity = model.source_identity();
        let work_completed_within_deadline = result.is_ok() && Instant::now() <= deadline;
        model.finish_search(if result.is_ok() {
            RoleSearchClosure::Completed
        } else {
            RoleSearchClosure::Failed
        });
        // Cleanup is bounded independently and uses the real SingleWorker fence.
        let cleanup_started = Instant::now();
        let cleanup = handle.finish(cleanup_started + CLEANUP);
        let receipt = handle.receipt();
        let cleanup_error = cleanup
            .err()
            .map(|error| role_failure("native_shutdown", error).wire());
        // This explicitly selected CPU loader is ready without a startup NN
        // probe. Require that receipt fact and actual cumulative worker Stats;
        // never invent a zero startup snapshot or subtract a missing snapshot.
        let nn_role_cumulative = receipt.backend_stats.as_ref().map(|stats| {
            [
                stats.role_nn_runs_attempted,
                stats.role_nn_runs_completed,
                stats.role_nn_runs_failed_known,
            ]
        });
        let actual_nn_and_delivery_verified = receipt.execution.provider == "cpu"
            && receipt.execution.private_warm.is_some()
            && receipt.startup_probe.is_none()
            && nn_role_cumulative == Some([8, 8, 0])
            && receipt.physically_completed_role_calls == 8
            && receipt.completed_role_inputs == 8
            && receipt.delivered_role_inputs == 8
            && receipt.failed_physical_role_calls == 0
            && receipt.invalid_role_outputs == 0
            && receipt.search_consumed_role_inputs == 6;
        let passed = result.is_ok()
            && cleanup_error.is_none()
            && receipt.physical_shutdown_confirmed
            && receipt.native_buffers_released
            && !receipt.quarantined
            && receipt.physical_runs_in_flight == 0
            && actual_nn_and_delivery_verified
            && work_completed_within_deadline
            && Instant::now() <= deadline + CLEANUP;
        json!({
            "schema": SCHEMA,
            "passed": passed,
            "provider": "cpu",
            "gpu_execution": false,
            "gpu_validation": "not_run",
            "actual_training": false,
            "full_pals_search": false,
            "search_consumer": "public_checked_role_acceptance_harness",
            "strength_or_speedup_claim": false,
            "wall_limit_ms": WALL.as_millis(),
            "cleanup_limit_ms": CLEANUP.as_millis(),
            "elapsed_ms": started.elapsed().as_millis(),
            "cleanup_elapsed_ms": cleanup_started.elapsed().as_millis(),
            "work_completed_within_deadline": work_completed_within_deadline,
            "actual_nn_and_delivery_verified": actual_nn_and_delivery_verified,
            "cpu_startup_probe": "not_run_by_selected_cpu_loader",
            "actual_role_nn_cumulative": nn_role_cumulative.map(|counts| json!({
                "attempted": counts[0], "completed": counts[1], "failed_known": counts[2]
            })),
            "source_identity": identity,
            "failure": result.err().map(Failure::wire),
            "cleanup_failure": cleanup_error,
            "steps": steps,
            "final_receipt": receipt,
            "query_freeze_evidence_scope": "native_public_observation_of_first_actual_input_digest_and_query_hit;_private_16_value_tensor_not_exported_by_public_API",
            "full_seed_evidence_scope": "loaded_baseline_16x384_model_contract_and_native_known_completion_checked_seed_acceptance;_no_harness_latent"
        })
    }

    fn observed(
        handle: &NativeRoleFinishHandle,
        stage: &'static str,
    ) -> Result<NativePrivateWarmObservation, Failure> {
        let receipt = handle.receipt();
        require(
            !receipt.quarantined && receipt.physical_runs_in_flight == 0,
            stage,
        )?;
        let observation = receipt.private_warm_observation.ok_or(Failure {
            stage,
            code: "warm_observation_missing",
        })?;
        require(
            !observation.admission_closed
                && !observation.quarantined
                && !observation.retained_prepared_input,
            stage,
        )?;
        Ok(observation)
    }

    // These are hashes of this harness's actual checked legal continuations,
    // using the production domain/order/Move16 encoding, not random focus tags.
    fn line_digest(domain: &[u8], line: &[BoardMove]) -> Result<[u8; 32], Failure> {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-role-ordered-move16/1\0");
        hash.update((domain.len() as u64).to_le_bytes());
        hash.update(domain);
        hash.update((line.len() as u64).to_le_bytes());
        for &movement in line {
            hash.update(
                Move16::pack(movement)
                    .map_err(|_| Failure {
                        stage: "logical_context",
                        code: "invalid_move",
                    })?
                    .bits()
                    .to_le_bytes(),
            );
        }
        Ok(hash.finalize().into())
    }

    fn context(
        game: u64,
        child: bool,
        purpose: RoleQueryPurpose,
        prefix: &[BoardMove],
        proposal: &[BoardMove],
        revision: u64,
    ) -> Result<RoleLogicalContext, Failure> {
        let mut divergence = Sha256::new();
        divergence.update(b"rz-pals-role-ordered-divergence/1\0");
        divergence.update(0_u64.to_le_bytes());
        Ok(RoleLogicalContext {
            game_generation: game,
            search_generation: game + 1,
            situation: SituationId {
                slot: usize::from(child),
                generation: game + 1,
            },
            state: StateId(usize::from(child)),
            focus: LineId(usize::from(child)),
            purpose,
            prefix: prefix.to_vec(),
            focus_sha256: line_digest(b"focus", proposal)?,
            prefix_sha256: line_digest(b"prefix", prefix)?,
            proposal_sha256: line_digest(b"proposal", proposal)?,
            refutation_sha256: None,
            divergence_sha256: divergence.finalize().into(),
            public_revision: revision,
            situation_revision: revision,
        })
    }

    #[derive(Clone, Copy)]
    struct Call<'a> {
        position: &'a Position,
        legal: &'a [BoardMove],
        proposal: &'a [BoardMove],
        records: &'a [RoleRecord],
        context: &'a RoleLogicalContext,
        deadline: Instant,
        cancel: &'a AtomicBool,
    }

    impl Call<'_> {
        fn query(&self) -> RoleQuery<'_> {
            RoleQuery {
                position: self.position,
                legal: self.legal,
                prefix: &self.context.prefix,
                proposal: self.proposal,
                counterexample: None,
                records: self.records,
                revision: self.context.public_revision,
                deadline: self.deadline,
                cancel: self.cancel,
            }
        }
        fn accept(&self, model: &mut NativeRoleModel) -> Result<(), RoleError> {
            model.accepted_output_checked(RoleAcceptance {
                snapshot: &self.position.snapshot(),
                context: self.context,
                deadline: self.deadline,
                cancel: self.cancel,
            })
        }
    }

    fn evaluation(
        model: &mut NativeRoleModel,
        call: &Call<'_>,
        stage: &'static str,
    ) -> Result<RoleEvaluation, Failure> {
        let evaluation = match call.context.purpose {
            RoleQueryPurpose::ProposePolicy => {
                model.propose_with_context(call.query(), call.context)
            }
            RoleQueryPurpose::ReplyPolicy => model.reply_with_context(call.query(), call.context),
            _ => {
                return Err(Failure {
                    stage,
                    code: "unsupported_harness_role",
                });
            }
        }
        .map_err(|error| role_failure(stage, error))?;
        require(
            evaluation.logits.len() == call.legal.len()
                && evaluation.logits.iter().all(|v| v.is_finite())
                && evaluation
                    .wdl
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                && (evaluation.wdl.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
            stage,
        )?;
        Ok(evaluation)
    }

    fn accepted(
        model: &mut NativeRoleModel,
        handle: &NativeRoleFinishHandle,
        call: &Call<'_>,
        stage: &'static str,
        warm: bool,
        steps: &mut Vec<Value>,
    ) -> Result<BoardMove, Failure> {
        let before = handle.receipt();
        let bank_before = observed(handle, stage)?;
        let output = evaluation(model, call, stage)?;
        let staged = observed(handle, stage)?;
        require(
            staged.pending_acceptance
                && staged.accepted_seeds == bank_before.accepted_seeds
                && staged.known_seed_completions == bank_before.known_seed_completions + 1,
            stage,
        )?;
        call.accept(model)
            .map_err(|error| role_failure(stage, error))?;
        let after = handle.receipt();
        let bank_after = observed(handle, stage)?;
        require(
            after.search_consumed_role_inputs == before.search_consumed_role_inputs + 1
                && bank_after.accepted_seeds == bank_before.accepted_seeds + 1
                && !bank_after.pending_acceptance
                && bank_after.active_lease.is_none()
                && bank_after.pinned_entries == 0
                && bank_after.warm_seed_admissions
                    == bank_before.warm_seed_admissions + u64::from(warm)
                && bank_after.fresh_seed_admissions
                    == bank_before.fresh_seed_admissions + u64::from(!warm),
            stage,
        )?;
        if warm {
            require(
                bank_after.query_hits == bank_before.query_hits + 1
                    && bank_after.query_freezes == bank_before.query_freezes
                    && bank_after.first_query_input_sha256_per_role
                        == bank_before.first_query_input_sha256_per_role,
                stage,
            )?;
        }
        let selected = output
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| call.legal[index])
            .ok_or(Failure {
                stage,
                code: "no_legal_candidate",
            })?;
        steps.push(
            json!({"stage": stage, "expected_mode": if warm { "approx_warm_v1" } else { "fresh" },
            "checked_accepted": true, "legal_logits": output.logits.len(), "wdl": output.wdl,
            "known_native_role_calls": after.physically_completed_role_calls,
            "checked_consumer_calls": after.search_consumed_role_inputs,
            "warm_observation": bank_after}),
        );
        Ok(selected)
    }

    fn public_record(
        position: &Position,
        movement: BoardMove,
        kind: RecordKind,
        revision: u64,
        state: StateId,
    ) -> Result<RoleRecord, Failure> {
        require(
            position.legal_moves().contains(&movement),
            "public_record_rules_validation",
        )?;
        Ok(RoleRecord {
            revision,
            parent_revision: None,
            supersedes_revision: None,
            origin_state: state,
            kind,
            line: vec![movement],
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: position.side_to_move(),
            critical: false,
        })
    }

    fn checks(
        model: &mut NativeRoleModel,
        handle: &NativeRoleFinishHandle,
        deadline: Instant,
        cancel: &AtomicBool,
        steps: &mut Vec<Value>,
    ) -> Result<(), Failure> {
        let loaded = handle.receipt();
        let bank = observed(handle, "cpu_loader_ready_before_role_calls")?;
        require(
            loaded.execution.provider == "cpu"
                && loaded.execution.private_warm.is_some()
                && loaded.startup_probe.is_none()
                && loaded.backend_stats.is_none()
                && PalsModelConfig::baseline().latent_elements() == 6144
                && model
                    .source_identity()
                    .model_configuration
                    .latent_elements()
                    == 6144
                && loaded.physically_completed_role_calls == 0
                && loaded.completed_role_inputs == 0
                && loaded.delivered_role_inputs == 0
                && loaded.search_consumed_role_inputs == 0
                && loaded.completed_new_game_resets == 0
                && bank.accepted_seeds == 0
                && bank.known_seed_completions == 0
                && bank.fresh_seed_admissions == 0
                && bank.warm_seed_admissions == 0
                && bank.query_freezes == 0
                && bank.query_hits == 0
                && bank.entries_per_role == [0; 3],
            "cpu_loader_ready_before_role_calls",
        )?;
        steps.push(json!({
            "stage": "cpu_loader_ready_before_role_calls",
            "receipt": loaded,
            "pre_call_role_callback_and_bank_counts_observed_zero": true,
            "pre_call_nn_stats": "unobserved;_final_worker_stats_are_actual_cumulative_counts",
            "cpu_startup_probe": "not_run_by_selected_cpu_loader"
        }));
        let game = handle.receipt().game_generation;
        let position = Position::startpos();
        let legal = position.legal_moves();
        let p0 = context(game, false, RoleQueryPurpose::ProposePolicy, &[], &[], 0)?;
        let first = Call {
            position: &position,
            legal: &legal,
            proposal: &[],
            records: &[],
            context: &p0,
            deadline,
            cancel,
        };
        let proposal = accepted(model, handle, &first, "proposer_fresh", false, steps)?;
        let records = vec![public_record(
            &position,
            proposal,
            RecordKind::Proposal,
            1,
            p0.state,
        )?];
        let mut p1 = p0.clone();
        p1.public_revision = 1;
        p1.situation_revision = 1;
        let second = Call {
            records: &records,
            context: &p1,
            ..first
        };
        let first_prepared = prepare_role_input(
            &first.query(),
            NativeQueryKind::Propose,
            model.model_epoch(),
        )
        .map_err(|error| role_failure("query_change_preparation", error))?;
        let second_prepared = prepare_role_input(
            &second.query(),
            NativeQueryKind::Propose,
            model.model_epoch(),
        )
        .map_err(|error| role_failure("query_change_preparation", error))?;
        let config = PalsModelConfig::baseline();
        let first_tensors = first_prepared
            .prepare_tensors(&config)
            .map_err(|_| Failure {
                stage: "query_change_preparation",
                code: "invalid_first_public_tensors",
            })?;
        let second_tensors = second_prepared
            .prepare_tensors(&config)
            .map_err(|_| Failure {
                stage: "query_change_preparation",
                code: "invalid_second_public_tensors",
            })?;
        // Empty and one-record views both have one physical record slot. Test
        // actual padded FP32 payload bits and validity masks, not slot lengths.
        let record_payload_changed = first_tensors
            .records
            .iter()
            .map(|v| v.to_bits())
            .ne(second_tensors.records.iter().map(|v| v.to_bits()));
        require(
            first_prepared.query.map(f32::to_bits) != second_prepared.query.map(f32::to_bits)
                && record_payload_changed
                && first_tensors.record_mask == [false]
                && second_tensors.record_mask == [true]
                && first_tensors.records.iter().all(|v| v.to_bits() == 0),
            "public_record_revision_changes_pre_freeze_input",
        )?;
        steps.push(json!({
            "stage": "public_record_revision_changes_pre_freeze_input",
            "public_record_payload_bits_changed": record_payload_changed,
            "first_record_mask": first_tensors.record_mask,
            "second_record_mask": second_tensors.record_mask,
            "first_public_memory_key": first_tensors.public_memory_key,
            "second_public_memory_key": second_tensors.public_memory_key
        }));
        accepted(
            model,
            handle,
            &second,
            "proposer_warm_changed_public_revision",
            true,
            steps,
        )?;

        // C has its own real Rules child state/prefix and distinct seed role.
        let child = position.preview_move(proposal).map_err(|_| Failure {
            stage: "critic_rules_child",
            code: "illegal_proposal",
        })?;
        let child_legal = child.legal_moves();
        let proposal_line = [proposal];
        let c0 = context(
            game,
            true,
            RoleQueryPurpose::ReplyPolicy,
            &proposal_line,
            &proposal_line,
            1,
        )?;
        let critic_first = Call {
            position: &child,
            legal: &child_legal,
            proposal: &proposal_line,
            records: &records,
            context: &c0,
            deadline,
            cancel,
        };
        let reply = accepted(
            model,
            handle,
            &critic_first,
            "critic_independent_fresh",
            false,
            steps,
        )?;
        let mut critic_records = records.clone();
        critic_records.push(public_record(
            &child,
            reply,
            RecordKind::Counterexample,
            2,
            c0.state,
        )?);
        let mut c1 = c0.clone();
        c1.public_revision = 2;
        c1.situation_revision = 2;
        let critic_second = Call {
            records: &critic_records,
            context: &c1,
            ..critic_first
        };
        accepted(
            model,
            handle,
            &critic_second,
            "critic_warm_changed_public_revision",
            true,
            steps,
        )?;
        require(
            observed(handle, "independent_role_banks")?.entries_per_role == [1, 1, 0],
            "independent_role_banks",
        )?;

        let value_context = context(game, false, RoleQueryPurpose::ValueFresh, &[], &[], 2)?;
        let value_call = Call {
            position: &position,
            legal: &legal,
            proposal: &[],
            records: &critic_records,
            context: &value_context,
            deadline,
            cancel,
        };
        let value_before = observed(handle, "value_fresh")?;
        let consumed_before = handle.receipt().search_consumed_role_inputs;
        let value = model
            .evaluate_value_with_context(value_call.query(), &value_context)
            .map_err(|error| role_failure("value_fresh", error))?;
        value
            .validate(
                &position,
                model.value_identity().ok_or(Failure {
                    stage: "value_fresh",
                    code: "value_identity_missing",
                })?,
            )
            .map_err(|error| role_failure("value_validation", error))?;
        value_call
            .accept(model)
            .map_err(|error| role_failure("value_acceptance", error))?;
        let value_after = observed(handle, "value_fresh")?;
        require(
            value_after.entries_per_role == value_before.entries_per_role
                && value_after.accepted_seeds == value_before.accepted_seeds
                && value_after.fresh_seed_admissions == value_before.fresh_seed_admissions
                && value_after.warm_seed_admissions == value_before.warm_seed_admissions
                && value_after.query_hits == value_before.query_hits
                && value_after.query_freezes == value_before.query_freezes
                && handle.receipt().search_consumed_role_inputs == consumed_before + 1,
            "value_bank_free_fresh",
        )?;
        steps.push(json!({"stage": "value_bank_free_fresh", "wdl": value.wdl, "input_sha256": value.input_sha256, "warm_observation": value_after}));

        let resets_before = handle.receipt().completed_new_game_resets;
        model.new_game_with_generation(Some(game + 1));
        let pending_reset = handle.receipt();
        require(
            pending_reset.completed_new_game_resets == resets_before
                && observed(handle, "logical_new_game_before_actual_fence")?.game_generation
                    == game,
            "logical_reset_not_a_fence",
        )?;
        steps.push(
            json!({"stage": "logical_new_game_before_actual_fence", "receipt": pending_reset}),
        );
        let reset_context = context(
            game + 1,
            false,
            RoleQueryPurpose::ProposePolicy,
            &[],
            &[],
            0,
        )?;
        let after_reset = Call {
            position: &position,
            legal: &legal,
            proposal: &[],
            records: &[],
            context: &reset_context,
            deadline,
            cancel,
        };
        accepted(
            model,
            handle,
            &after_reset,
            "actual_new_game_fence_then_proposer_fresh",
            false,
            steps,
        )?;
        let reset_bank = observed(handle, "actual_reset_observation")?;
        require(
            handle.receipt().completed_new_game_resets == resets_before + 1
                && reset_bank.game_generation == game + 1
                && reset_bank.entries_per_role == [1, 0, 0]
                && reset_bank.frozen_contexts_per_role == [1, 0],
            "actual_reset_observation",
        )?;

        // Deterministic rejection after an actual delivered NN result. This
        // changes only the test's control flag; it never manufactures a fence.
        let rejected_cancel = AtomicBool::new(false);
        let canceled_call = Call {
            cancel: &rejected_cancel,
            ..after_reset
        };
        let before_cancel = observed(handle, "cancel_after_delivery")?;
        let consumed = handle.receipt().search_consumed_role_inputs;
        evaluation(model, &canceled_call, "cancel_candidate_actual_nn")?;
        rejected_cancel.store(true, Ordering::Release);
        require(
            matches!(canceled_call.accept(model), Err(RoleError::Canceled)),
            "cancel_acceptance_rejected",
        )?;
        model.finish_search(RoleSearchClosure::Canceled);
        let closed_cancel = observed(handle, "cancel_known_closure")?;
        require(
            !closed_cancel.pending_acceptance
                && closed_cancel.active_lease.is_none()
                && closed_cancel.pinned_entries == 0
                && closed_cancel.accepted_seeds == before_cancel.accepted_seeds
                && handle.receipt().search_consumed_role_inputs == consumed,
            "cancel_no_seed_or_consumption",
        )?;
        steps.push(json!({"stage": "cancel_after_actual_delivery_known_closure", "accepted": false, "warm_observation": closed_cancel}));

        require(
            deadline.saturating_duration_since(Instant::now()) > Duration::from_secs(3),
            "late_case_budget",
        )?;
        let late_cancel = AtomicBool::new(false);
        let late_deadline = Instant::now() + Duration::from_secs(2);
        let late_call = Call {
            deadline: late_deadline,
            cancel: &late_cancel,
            ..after_reset
        };
        let before_late = observed(handle, "late_acceptance")?;
        let consumed = handle.receipt().search_consumed_role_inputs;
        evaluation(model, &late_call, "late_candidate_actual_nn")?;
        std::thread::sleep(
            late_deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
        );
        require(
            matches!(late_call.accept(model), Err(RoleError::Deadline)),
            "late_acceptance_rejected",
        )?;
        model.finish_search(RoleSearchClosure::Deadline);
        let closed_late = observed(handle, "late_known_closure")?;
        require(
            !closed_late.pending_acceptance
                && closed_late.active_lease.is_none()
                && closed_late.pinned_entries == 0
                && closed_late.accepted_seeds == before_late.accepted_seeds
                && handle.receipt().search_consumed_role_inputs == consumed,
            "late_no_seed_or_consumption",
        )?;
        steps.push(json!({"stage": "late_actual_delivery_known_closure", "accepted": false, "warm_observation": closed_late}));
        require(
            handle.receipt().search_consumed_role_inputs == 6
                && closed_late.accepted_seeds == 5
                && closed_late.fresh_seed_admissions == 3
                && closed_late.warm_seed_admissions == 4
                && closed_late.known_seed_completions == 7,
            "final_checked_counts",
        )?;
        Ok(())
    }
}
