use crate::*;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodingDescriptor {
    pub handle: EncodingHandle,
    pub history_length: u16,
    pub action_map: Digest,
    pub history_policy: Digest,
}

/// Immutable prevalidated model manifest. Device availability is a separate fact.
#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    handle: ModelHandle,
    encoding: EncodingDescriptor,
    precisions: Arc<[PrecisionProfile]>,
    full_steps: u32,
    max_batch_items: usize,
}

impl ModelDescriptor {
    pub fn try_new(
        handle: ModelHandle,
        encoding: EncodingDescriptor,
        precisions: Vec<PrecisionProfile>,
        full_steps: u32,
        max_batch_items: usize,
    ) -> Result<Self, ContractError> {
        if precisions.is_empty() || full_steps == 0 || max_batch_items == 0 {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Contract,
                "model requires precision, full steps and finite batch limit",
            ));
        }
        Ok(Self {
            handle,
            encoding,
            precisions: precisions.into(),
            full_steps,
            max_batch_items,
        })
    }
    pub fn handle(&self) -> ModelHandle {
        self.handle
    }
    pub fn encoding(&self) -> &EncodingDescriptor {
        &self.encoding
    }
    pub fn full_steps(&self) -> u32 {
        self.full_steps
    }
    pub fn max_batch_items(&self) -> usize {
        self.max_batch_items
    }
    pub fn supports(&self, precision: PrecisionProfile) -> bool {
        self.precisions.contains(&precision)
    }
}

/// Echoed identity only; request cancellation and physical execution are separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvalContext {
    pub revision: SchemaVersion,
    pub request: RequestId,
    pub selection: SelectionId,
    pub game: GameGeneration,
    pub root: RootGeneration,
    pub state: StateIdentity,
    pub legal_order: LegalOrderIdentity,
    pub input: EvalInputKey,
    pub model: ModelHandle,
    pub encoding: EncodingHandle,
    pub precision: PrecisionProfile,
    pub compute: ComputeBudget,
    pub backend: Digest,
}

/// Acceptance authority must be supplied again at finalization and backup.
#[derive(Clone, Copy, Debug)]
pub struct AcceptanceScope {
    pub game: GameGeneration,
    pub root: RootGeneration,
    pub model: ModelHandle,
    pub encoding: EncodingHandle,
    pub backend: Digest,
}

#[derive(Debug)]
pub struct EvalRequest<P> {
    context: EvalContext,
    position: PositionSnapshot<P>,
    legal: LegalMoveView,
    model: Arc<ModelDescriptor>,
    deadline: Deadline,
    cancel: CancelToken,
    bytes: ByteBudget,
}

impl<P> EvalRequest<P> {
    pub fn try_new(
        context: EvalContext,
        position: PositionSnapshot<P>,
        legal: LegalMoveView,
        model: Arc<ModelDescriptor>,
        deadline: Deadline,
        cancel: CancelToken,
        bytes: ByteBudget,
    ) -> Result<Self, ContractError> {
        context.revision.validate()?;
        context.compute.validate()?;
        if context.request.epoch != context.selection.epoch
            || context.request.epoch != deadline.clock.0
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "request, selection and deadline belong to different process epochs",
            ));
        }
        if position.classification().play_status != PlayStatus::Ongoing || legal.moves().is_empty()
        {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "exact terminal must bypass neural evaluation",
            ));
        }
        if context.state != position.identity()
            || context.state != legal.state()
            || context.legal_order != legal.order()
            || context.model != model.handle()
            || context.encoding != model.encoding().handle
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "request state, legal view or model/encoding identity differs",
            ));
        }
        if !model.supports(context.precision)
            || context.compute.max_steps > model.full_steps()
            || (context.compute.require_full && context.compute.max_steps != model.full_steps())
        {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "unsupported precision or compute budget",
            ));
        }
        Ok(Self {
            context,
            position,
            legal,
            model,
            deadline,
            cancel,
            bytes,
        })
    }
    pub fn context(&self) -> EvalContext {
        self.context
    }
    pub fn position(&self) -> &PositionSnapshot<P> {
        &self.position
    }
    pub fn legal(&self) -> &LegalMoveView {
        &self.legal
    }
    pub fn model(&self) -> &Arc<ModelDescriptor> {
        &self.model
    }
    pub fn deadline(&self) -> Deadline {
        self.deadline
    }
    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }
    pub fn byte_budget(&self) -> ByteBudget {
        self.bytes
    }

    pub fn validate_acceptance(
        &self,
        scope: AcceptanceScope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), ContractError> {
        if self.context.game != scope.game
            || self.context.root != scope.root
            || self.context.model != scope.model
            || self.context.encoding != scope.encoding
            || self.context.backend != scope.backend
        {
            return Err(ContractError::new(
                ErrorCode::Stale,
                Stage::Admission,
                "game, root, registry or backend generation no longer current",
            ));
        }
        if self.cancel.is_canceled() {
            return Err(ContractError::new(
                ErrorCode::Canceled,
                Stage::Admission,
                "logical owner canceled",
            ));
        }
        self.deadline.accepts(clock, now)
    }
}

/// Legal probabilities in exactly the requested order. No silent clipping/fill.
#[derive(Clone, Debug, PartialEq)]
pub struct LegalPolicy {
    probabilities: Arc<[f64]>,
}

impl LegalPolicy {
    pub fn try_new(probabilities: Vec<f64>, sum_tolerance: f64) -> Result<Self, ContractError> {
        if !sum_tolerance.is_finite() || !(0.0..=0.01).contains(&sum_tolerance) {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Output,
                "invalid probability tolerance",
            ));
        }
        let sum: f64 = probabilities.iter().sum();
        if probabilities.is_empty()
            || probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || !sum.is_finite()
            || sum <= 0.0
            || (sum - 1.0).abs() > sum_tolerance
        {
            return Err(ContractError::new(
                ErrorCode::NumericalFailure,
                Stage::Output,
                "invalid legal policy",
            ));
        }
        Ok(Self {
            probabilities: probabilities.into(),
        })
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    /// Explicit adapter operation after admissibility validation; record this policy
    /// in the model manifest. In particular it handles valid f32 -> f64 roundoff.
    pub fn normalized(&self) -> Vec<f64> {
        let sum: f64 = self.probabilities.iter().sum();
        self.probabilities.iter().map(|p| p / sum).collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheProvenance {
    Computed,
    ExactFeatureReuse,
    RawEvalHit {
        source_execution: Option<ExecutionId>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActualCompute {
    pub precision: PrecisionProfile,
    pub steps: u32,
    pub full: bool,
    pub backend: Digest,
    /// None for a raw cache hit: prior provenance is not a new execution.
    pub execution: Option<ExecutionId>,
    pub provenance: CacheProvenance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Viewpoint {
    SideToMove,
}

#[derive(Clone, Debug)]
pub struct EvalOutput {
    pub context: EvalContext,
    pub legal: LegalMoveView,
    pub policy: LegalPolicy,
    pub wdl: Wdl,
    pub viewpoint: Viewpoint,
    pub actual: ActualCompute,
}

impl EvalOutput {
    /// Call again immediately before Search's exactly-once consume/backup commit.
    /// This validates data/authority; it does not consume a SelectionTicket.
    pub fn validate_for<P>(
        &self,
        request: &EvalRequest<P>,
        scope: AcceptanceScope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), ContractError> {
        request.validate_acceptance(scope, clock, now)?;
        if self.context != request.context
            || self.legal != request.legal
            || self.policy.probabilities().len() != request.legal.moves().len()
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "response context or ordered legal payload differs",
            ));
        }
        let budget = request.context.compute;
        if self.actual.precision != request.context.precision
            || self.actual.backend != request.context.backend
            || self.actual.steps < budget.min_steps
            || self.actual.steps > budget.max_steps
            || self.actual.full != (self.actual.steps == request.model.full_steps())
            || (budget.require_full && !self.actual.full)
        {
            return Err(ContractError::new(
                ErrorCode::UnsupportedContract,
                Stage::Output,
                "actual precision/backend/steps do not satisfy request",
            ));
        }
        match self.actual.provenance {
            CacheProvenance::RawEvalHit { .. } if self.actual.execution.is_none() => {}
            CacheProvenance::Computed | CacheProvenance::ExactFeatureReuse
                if self.actual.execution.is_some() => {}
            _ => {
                return Err(ContractError::new(
                    ErrorCode::IdentityMismatch,
                    Stage::Output,
                    "cache provenance and new physical execution disagree",
                ))
            }
        }
        if self
            .actual
            .execution
            .is_some_and(|execution| execution.epoch != request.context.request.epoch)
        {
            return Err(ContractError::new(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "new physical execution belongs to a different process epoch",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionContext {
    pub request: EvalContext,
    pub execution: Option<ExecutionId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecoveryOutcome {
    NotAttempted,
    Failed,
    Completed,
}

#[derive(Clone, Debug)]
pub struct EvalFailure {
    pub context: CompletionContext,
    pub error: ContractError,
    pub recovery: RecoveryOutcome,
}

#[derive(Clone, Debug)]
pub enum EvalResult {
    Completed(EvalOutput),
    Canceled(CompletionContext),
    Expired(CompletionContext),
    Stale(CompletionContext),
    Failed(EvalFailure),
}

/// Logical evaluator boundary implemented by the Runtime adapter. poll returns
/// finalized outcomes, not raw backend callbacks. Physical leases stay Runtime-owned.
/// Each admitted RequestId publishes exactly once; cancel is idempotent, does not
/// release physical buffers and never authorizes Search backup.
pub trait Evaluator<P: Send + Sync + 'static> {
    fn submit(&mut self, request: Arc<EvalRequest<P>>) -> Result<(), ContractError>;
    fn poll(&mut self) -> Option<EvalResult>;
    fn cancel(&mut self, request: RequestId) -> Result<(), ContractError>;
}
