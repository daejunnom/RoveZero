//! Explicit CPU/mock composition of A Rules, B search, C encoding and D runtime.
//! No model file or native inference library is loaded by this entry point.

use crate::engine::{
    EvaluatorFactory, EvaluatorProfile, ManagedEvaluator, OwnerRegistry, ProcessClock,
    SearchAuthority,
};
use rz_contracts::*;
use rz_encoding::classical::HistoryFill;
use rz_eval::{
    RawOutput,
    contracts::{
        ClassicalProjection, HOST_BYTES_PER_ITEM, MaiaBinding, ScriptedRuntimeBackend,
        encoding_manifest,
    },
    mock::{self, Callback, ScriptIdentity, Step},
};
use rz_position::contracts::RulesState;
use rz_runtime::{
    Clock, DrainState, Limits, Resources,
    contracts::{ContractClock, ContractEvaluator, ContractsAdapter, SharedScope},
};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub const MAX_EVALUATIONS: u64 = 129;

fn error(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}

#[derive(Clone)]
struct RuntimeClock(ProcessClock);
impl Clock for RuntimeClock {
    type Tick = MonotonicTick;
    fn now(&self) -> MonotonicTick {
        self.0.now().unwrap_or(MonotonicTick(u64::MAX))
    }
    fn elapsed(&self, earlier: MonotonicTick, later: MonotonicTick) -> Duration {
        Duration::from_nanos(later.0.saturating_sub(earlier.0))
    }
}
impl ContractClock for RuntimeClock {
    fn domain(&self) -> ClockDomain {
        self.0.domain()
    }
}

pub struct CpuMockFactory {
    projection: ClassicalProjection,
    profile: EvaluatorProfile,
    delay: Duration,
}
impl CpuMockFactory {
    pub fn new(owners: &OwnerRegistry, delay: Duration) -> Result<Self, ContractError> {
        if delay > Duration::from_secs(1) {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "mock delay exceeds the explicit one second limit",
            ));
        }
        let script = script(delay)?;
        let backend = Digest(script.identity_digest());
        let fill = HistoryFill::No;
        let encoding = EncodingHandle {
            owner: owners.allocate()?,
            slot: 0,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(fill),
        };
        let model = ModelHandle {
            owner: owners.allocate()?,
            slot: 0,
            generation: SlotGeneration(1),
            manifest: Digest(rz_eval::asset::sha256(
                b"rz-cpu-mock-model/1;policy=zero1858;wdl=.4,.3,.3;fullsteps=1;fp32",
            )),
        };
        let binding = MaiaBinding::new(model, encoding, fill, backend, 1)?;
        let profile = EvaluatorProfile {
            model: Arc::clone(binding.model()),
            precision: PrecisionProfile::Fp32,
            backend,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            bytes: ByteBudget {
                host: HOST_BYTES_PER_ITEM,
                device: 0,
                pinned: 0,
            },
        };
        Ok(Self {
            projection: ClassicalProjection::new(binding),
            profile,
            delay,
        })
    }
}
fn script(
    delay: Duration,
) -> Result<mock::ScriptedBackend<(ExecutionId, RequestId)>, ContractError> {
    let step = Step {
        callbacks: vec![Callback {
            after: delay,
            reply: Ok(RawOutput {
                policy_logits: vec![0.0; rz_encoding::POLICY_SIZE],
                wdl: vec![0.4, 0.3, 0.3],
            }),
        }],
        device_complete_after: delay,
        cancel_ack_after: Some(Duration::ZERO),
    };
    mock::ScriptedBackend::new(
        ScriptIdentity {
            name: "RoveZero explicit CPU mock classic112/1".into(),
            seed: 0,
        },
        vec![step; MAX_EVALUATIONS as usize],
        mock::Limits {
            max_steps: MAX_EVALUATIONS as usize,
            max_in_flight: 1,
            max_pending_events: 4,
            max_script_values: MAX_EVALUATIONS as usize * (rz_encoding::POLICY_SIZE + 3),
        },
    )
    .map_err(|_| {
        error(
            ErrorCode::InvalidInput,
            Stage::Contract,
            "invalid bounded CPU mock script",
        )
    })
}
impl EvaluatorFactory for CpuMockFactory {
    fn profile(&self) -> EvaluatorProfile {
        self.profile.clone()
    }
    fn input_key(&self, state: &RulesState, legal: &[Move]) -> Result<EvalInputKey, ContractError> {
        self.projection.input_key(state, legal)
    }
    fn create(
        &self,
        clock: ProcessClock,
        authority: SearchAuthority,
    ) -> Result<Box<dyn ManagedEvaluator>, ContractError> {
        // Reserve a disjoint finite physical ID range across every worker/root.
        // Refused construction burns its range; no same-epoch reuse is possible.
        let high_water = clock.reserve_executions(MAX_EVALUATIONS)?;
        let runtime_clock = RuntimeClock(clock.clone());
        let scope = SharedScope::new(authority.current_scope()?);
        let adapter = ContractsAdapter::with_execution_high_water(
            scope.clone(),
            runtime_clock.clone(),
            1,
            high_water,
        )?;
        let script = script(self.delay)?;
        if Digest(script.identity_digest()) != self.profile.backend {
            return Err(error(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "mock script identity changed",
            ));
        }
        let backend = ScriptedRuntimeBackend::new(
            script,
            runtime_clock,
            self.projection.clone(),
            Arc::clone(&self.profile.model),
            self.profile.backend,
        )?;
        let limits = Limits {
            max_requests: 1,
            max_batch_items: 1,
            max_executions: 1,
            max_batch_wait: Duration::ZERO,
            max_queue_age: Duration::from_secs(30),
            deadline_reserve: Duration::ZERO,
            memory: Resources {
                host_bytes: HOST_BYTES_PER_ITEM * 2,
                device_bytes: 0,
                pinned_bytes: 0,
            },
        };
        let evaluator = ContractEvaluator::new(adapter, backend, limits, 256)?;
        Ok(Box::new(CpuMockRuntime {
            evaluator,
            scope,
            authority,
            clock,
            submissions: 0,
            pending: None,
        }))
    }
}
struct CpuMockRuntime {
    evaluator: ContractEvaluator<RulesState, ScriptedRuntimeBackend<RuntimeClock>, RuntimeClock>,
    scope: SharedScope,
    authority: SearchAuthority,
    clock: ProcessClock,
    submissions: u64,
    pending: Option<EvalContext>,
}
impl CpuMockRuntime {
    fn refresh(&self) -> Result<(), ContractError> {
        self.scope.update(self.authority.current_scope()?);
        Ok(())
    }
}
impl Evaluator<RulesState> for CpuMockRuntime {
    fn submit(&mut self, request: Arc<EvalRequest<RulesState>>) -> Result<(), ContractError> {
        self.refresh()?;
        if self.submissions >= MAX_EVALUATIONS {
            return Err(error(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "per-worker physical ID range limit reached",
            ));
        }
        self.submissions += 1;
        self.pending = Some(request.context());
        self.evaluator.submit(request)
    }
    fn poll(&mut self) -> Option<EvalResult> {
        if let Err(error) = self.refresh() {
            self.authority.cancel();
            return self.pending.take().map(|request| {
                EvalResult::Failed(EvalFailure {
                    context: CompletionContext {
                        request,
                        execution: None,
                    },
                    error,
                    recovery: RecoveryOutcome::NotAttempted,
                })
            });
        }
        let result = self.evaluator.poll();
        if result.is_some() {
            self.pending = None;
        }
        result
    }
    fn cancel(&mut self, request: RequestId) -> Result<(), ContractError> {
        self.evaluator.cancel(request)
    }
}
impl ManagedEvaluator for CpuMockRuntime {
    fn shutdown(&mut self, until: Instant) -> Result<(), ContractError> {
        self.authority.cancel();
        self.evaluator.begin_shutdown(self.clock.deadline(until)?)?;
        loop {
            self.refresh()?;
            while self.evaluator.poll().is_some() {} // Consume canceled receipts and release reservations.
            let snapshot = self.evaluator.shutdown_snapshot();
            match snapshot.drain {
                DrainState::Drained if snapshot.state.reserved_requests == 0 => return Ok(()),
                DrainState::TimedOut { .. } => {
                    return Err(error(
                        ErrorCode::BackendFailure,
                        Stage::Backend,
                        "CPU mock physical drain timed out with resources pinned",
                    ));
                }
                _ if Instant::now() >= until => {
                    return Err(error(
                        ErrorCode::Expired,
                        Stage::Backend,
                        "CPU mock shutdown deadline reached",
                    ));
                }
                _ => thread::sleep(Duration::from_millis(1)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PositionPort, PositionSpec, engine::RulesUciPort};

    #[test]
    fn real_cpu_mock_runtime_uses_disjoint_execution_ranges_and_enforces_the_cap() {
        let owners = Arc::new(OwnerRegistry::default());
        let factory = CpuMockFactory::new(&owners, Duration::ZERO).unwrap();
        let position =
            RulesUciPort::new(Arc::clone(&owners), rz_position::PositionLimits::default())
                .prepare(&PositionSpec::default())
                .unwrap()
                .snapshot;
        let profile = factory.profile();
        let clock = ProcessClock::new(ProcessEpoch(99));
        let scope = AcceptanceScope {
            game: GameGeneration(1),
            root: RootGeneration(1),
            model: profile.model.handle(),
            encoding: profile.model.encoding().handle,
            backend: profile.backend,
        };
        let authority = SearchAuthority::isolated(scope);
        let mut first = factory.create(clock.clone(), authority.clone()).unwrap();
        let mut second = factory.create(clock.clone(), authority).unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        let mut sequence = 0;
        let mut evaluate = |runtime: &mut dyn ManagedEvaluator| {
            sequence += 1;
            let context = EvalContext {
                revision: CONTRACT_REVISION,
                request: RequestId::new(clock.epoch(), sequence),
                selection: SelectionId::new(clock.epoch(), sequence),
                game: scope.game,
                root: scope.root,
                state: position.state().snapshot().identity(),
                legal_order: position.state().legal_moves().order(),
                input: factory
                    .input_key(
                        position.state().rules(),
                        position.state().legal_moves().moves(),
                    )
                    .unwrap(),
                model: scope.model,
                encoding: scope.encoding,
                precision: profile.precision,
                compute: profile.compute,
                backend: scope.backend,
            };
            let request = Arc::new(
                EvalRequest::try_new(
                    context,
                    position.state().snapshot().clone(),
                    position.state().legal_moves().clone(),
                    Arc::clone(&profile.model),
                    clock.deadline(until).unwrap(),
                    CancelToken::new(),
                    profile.bytes,
                )
                .unwrap(),
            );
            if let Err(error) = runtime.submit(request) {
                return Err(error);
            }
            loop {
                if let Some(result) = runtime.poll() {
                    let EvalResult::Completed(output) = result else {
                        panic!("fresh CPU/mock evaluation failed: {result:?}")
                    };
                    assert_eq!(output.context, context);
                    assert_eq!(output.actual.provenance, CacheProvenance::Computed);
                    assert_eq!(output.policy.probabilities().len(), 20);
                    return Ok(output.actual.execution.unwrap());
                }
                assert!(Instant::now() < until, "bounded CPU/mock completion");
                thread::sleep(Duration::from_millis(1));
            }
        };
        let first_execution = evaluate(first.as_mut()).unwrap();
        let second_execution = evaluate(second.as_mut()).unwrap();
        assert_eq!(first_execution.epoch, second_execution.epoch);
        assert_ne!(first_execution, second_execution);
        assert!(second_execution.sequence > first_execution.sequence);
        for _ in 1..MAX_EVALUATIONS {
            evaluate(first.as_mut()).unwrap();
        }
        assert_eq!(
            evaluate(first.as_mut()).unwrap_err().code,
            ErrorCode::ResourceExhausted
        );
        first
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
        second
            .shutdown(Instant::now() + Duration::from_secs(2))
            .unwrap();
    }
}
