//! CPU-only proof of entity input and candidate-sized policy, not a trained model.
use crate::model_adapter::{ModelAdapter, ModelCapabilities, PhysicalFailure};
use crate::native_runtime_bridge::NativePhysicalWorker;
use crate::worker::SingleWorker;
use rz_contracts::*;
use rz_position::{contracts::RulesState, Square};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

pub const HOST_BYTES_PER_ITEM: u64 = 64 * 1024;
const MAX_CANDIDATES: usize = 256;
const SUM_TOLERANCE: f64 = 1e-5;
fn fail(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}
pub fn encoding_manifest() -> Digest {
    Digest(Sha256::digest(b"rz-entity-candidate-mock-input-v1;absolute-squares;known-history8;ordered-candidates;mask=all1").into())
}
pub fn model_manifest() -> Digest {
    Digest(
        Sha256::digest(b"rz-entity-candidate-mock-v1;deterministic;candidate-policy;wdl=.4,.3,.3")
            .into(),
    )
}
pub fn backend_identity() -> Digest {
    Digest(Sha256::digest(b"rz-entity-candidate-cpu-mock-v1").into())
}

#[derive(Clone, Debug)]
pub struct EntityCandidateMockAdapter {
    model: Arc<ModelDescriptor>,
}
#[derive(Debug)]
pub struct EntityInput {
    pub entities: Vec<[u8; 3]>,
    pub history: Vec<[u8; 68]>,
    pub known_history: usize,
    pub clocks: [u32; 2],
    pub candidates: Vec<Move>,
}
pub struct PreparedEntity {
    pub request: Arc<EvalRequest<RulesState>>,
    pub input: EntityInput,
}
pub struct EntityBatch {
    pub execution: ExecutionId,
    pub items: Vec<PreparedEntity>,
}
#[derive(Clone, Debug)]
pub struct CandidateRaw {
    pub logits: Vec<f32>,
    pub wdl: [f32; 3],
}

impl EntityCandidateMockAdapter {
    pub fn new(model: ModelHandle, encoding: EncodingHandle) -> Result<Self, ContractError> {
        if model.manifest != model_manifest() || encoding.manifest != encoding_manifest() {
            return Err(fail(
                ErrorCode::IdentityMismatch,
                Stage::Contract,
                "entity mock semantic identity mismatch",
            ));
        }
        Ok(Self {
            model: Arc::new(ModelDescriptor::try_new(
                model,
                EncodingDescriptor {
                    handle: encoding,
                    history_length: 8,
                    action_map: Digest(Sha256::digest(b"ordered-legal-candidates-v1").into()),
                    history_policy: Digest(Sha256::digest(b"known-only-no-padding-v1").into()),
                },
                vec![PrecisionProfile::Fp32],
                1,
                1,
            )?),
        })
    }
    fn encode(&self, state: &RulesState, legal: &[Move]) -> Result<EntityInput, ContractError> {
        if legal.is_empty() || legal.len() > MAX_CANDIDATES {
            return Err(fail(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "entity candidate count outside 1..256",
            ));
        }
        let snapshot = state.snapshot();
        let mut entities = Vec::new();
        entities.try_reserve_exact(64).map_err(|_| {
            fail(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "entity allocation failed",
            )
        })?;
        for index in 0..64 {
            if let Some(piece) = snapshot.piece_at(Square::new(index).expect("bounded square")) {
                entities.push([index, piece.kind as u8, piece.color as u8]);
            }
        }
        let mut history = Vec::new();
        history.try_reserve_exact(8).map_err(|_| {
            fail(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "entity history allocation failed",
            )
        })?;
        for previous in snapshot.known_history().take(8) {
            let mut frame = [0; 68];
            for index in 0..64 {
                frame[index as usize] = previous
                    .piece_at(Square::new(index).expect("bounded square"))
                    .map_or(0, |p| 1 + p.kind as u8 + 6 * p.color as u8);
            }
            frame[64] = previous.side_to_move() as u8;
            frame[65] = previous.castling_rights();
            frame[66] = previous.en_passant_target().map_or(64, Square::index);
            frame[67] = u8::from(
                previous
                    .known_history()
                    .skip(1)
                    .any(|p| p.repetition_identity() == previous.repetition_identity()),
            );
            history.push(frame);
        }
        let mut candidates = Vec::new();
        candidates.try_reserve_exact(legal.len()).map_err(|_| {
            fail(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "candidate allocation failed",
            )
        })?;
        candidates.extend_from_slice(legal);
        Ok(EntityInput {
            entities,
            history,
            known_history: snapshot.known_history_len(),
            clocks: [snapshot.halfmove_clock(), snapshot.fullmove_number()],
            candidates,
        })
    }
    fn key(&self, input: &EntityInput) -> EvalInputKey {
        let mut h = Sha256::new();
        h.update(b"rz-entity-candidate-input-v1\0");
        h.update(self.model.encoding().handle.manifest.0);
        h.update((input.entities.len() as u64).to_le_bytes());
        for entity in &input.entities {
            h.update(entity);
        }
        h.update((input.history.len() as u64).to_le_bytes());
        for frame in &input.history {
            h.update(frame);
        }
        h.update((input.known_history as u64).to_le_bytes());
        for clock in input.clocks {
            h.update(clock.to_le_bytes());
        }
        h.update((input.candidates.len() as u64).to_le_bytes());
        for m in &input.candidates {
            h.update([
                m.from.index(),
                m.to.index(),
                m.promotion.map_or(0, |p| 1 + p as u8),
                1,
            ]);
        }
        EvalInputKey(Digest(h.finalize().into()))
    }
    pub fn infer(input: &EntityInput) -> CandidateRaw {
        CandidateRaw {
            logits: input
                .candidates
                .iter()
                .map(|m| f32::from(m.to.index() % 8) / 32.0)
                .collect(),
            wdl: [0.4, 0.3, 0.3],
        }
    }
    pub fn worker(&self) -> Result<NativePhysicalWorker<Self>, PhysicalFailure> {
        let adapter = self.clone();
        Ok(SingleWorker::spawn(move |batch: &EntityBatch| {
            batch
                .items
                .iter()
                .map(|item| adapter.decode(item, &Self::infer(&item.input), batch.execution))
                .collect()
        })?)
    }
}
impl ModelAdapter for EntityCandidateMockAdapter {
    type PreparedInput = PreparedEntity;
    type PreparedBatch = EntityBatch;
    type RawOutput = CandidateRaw;
    fn model(&self) -> &Arc<ModelDescriptor> {
        &self.model
    }
    fn backend(&self) -> Digest {
        backend_identity()
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            input: "entity-history-candidates-v1",
            policy_head: "ordered-candidate-logits",
            host_bytes_per_item: HOST_BYTES_PER_ITEM,
            mixed_legal_batching: false,
        }
    }
    fn input_key(&self, state: &RulesState, legal: &[Move]) -> Result<EvalInputKey, ContractError> {
        Ok(self.key(&self.encode(state, legal)?))
    }
    fn prepare(
        &self,
        request: Arc<EvalRequest<RulesState>>,
    ) -> Result<PreparedEntity, ContractError> {
        let c = request.context();
        if c.model != self.model.handle()
            || c.encoding != self.model.encoding().handle
            || c.backend != self.backend()
            || request.model().encoding() != self.model.encoding()
            || request.model().full_steps() != 1
            || request.model().max_batch_items() != 1
        {
            return Err(fail(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "entity request model/encoding/backend differs",
            ));
        }
        if c.precision != PrecisionProfile::Fp32
            || c.compute.min_steps != 1
            || c.compute.max_steps != 1
            || !c.compute.require_full
        {
            return Err(fail(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "entity mock requires full CPU mock evaluation",
            ));
        }
        if request.byte_budget().host < HOST_BYTES_PER_ITEM
            || request.byte_budget().device != 0
            || request.byte_budget().pinned != 0
        {
            return Err(fail(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "entity staging budget differs",
            ));
        }
        let input = self.encode(request.position().state(), request.legal().moves())?;
        if self.key(&input) != c.input {
            return Err(fail(
                ErrorCode::IdentityMismatch,
                Stage::Admission,
                "actual entity input differs",
            ));
        }
        Ok(PreparedEntity { request, input })
    }
    fn batch(
        &self,
        execution: ExecutionId,
        items: Vec<PreparedEntity>,
    ) -> Result<EntityBatch, ContractError> {
        if items.len() != 1 || items[0].request.context().request.epoch != execution.epoch {
            return Err(fail(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "entity mock accepts same-epoch B1 only",
            ));
        }
        Ok(EntityBatch { execution, items })
    }
    fn decode(
        &self,
        item: &PreparedEntity,
        raw: &CandidateRaw,
        execution: ExecutionId,
    ) -> Result<EvalOutput, PhysicalFailure> {
        let request = &item.request;
        if execution.epoch != request.context().request.epoch {
            return Err(fail(
                ErrorCode::IdentityMismatch,
                Stage::Output,
                "entity completion epoch differs",
            )
            .into());
        }
        if raw.logits.len() != request.legal().moves().len()
            || raw.logits.iter().any(|v| !v.is_finite())
        {
            return Err(fail(
                ErrorCode::NumericalFailure,
                Stage::Output,
                "invalid candidate policy head",
            )
            .into());
        }
        let max = raw.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut policy = Vec::new();
        policy.try_reserve_exact(raw.logits.len()).map_err(|_| {
            fail(
                ErrorCode::ResourceExhausted,
                Stage::Output,
                "candidate output allocation failed",
            )
        })?;
        let sum: f32 = raw.logits.iter().map(|x| (*x - max).exp()).sum();
        policy.extend(raw.logits.iter().map(|x| f64::from((*x - max).exp() / sum)));
        let [w, d, l] = raw.wdl;
        Ok(EvalOutput {
            context: request.context(),
            legal: request.legal().clone(),
            policy: LegalPolicy::try_new(policy, SUM_TOLERANCE)?,
            wdl: Wdl::try_new(w, d, l, SUM_TOLERANCE as f32)?,
            viewpoint: Viewpoint::SideToMove,
            actual: ActualCompute {
                precision: PrecisionProfile::Fp32,
                steps: 1,
                full: true,
                backend: self.backend(),
                execution: Some(execution),
                provenance: CacheProvenance::Computed,
            },
        })
    }
    fn validate_runtime(&self, max_batch: usize) -> Result<(), ContractError> {
        if max_batch != 1 {
            return Err(fail(
                ErrorCode::UnsupportedContract,
                Stage::Contract,
                "entity mock runtime is B1",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::{contracts::ContractPosition, Position};
    fn adapter() -> EntityCandidateMockAdapter {
        EntityCandidateMockAdapter::new(
            ModelHandle {
                owner: OwnerId(91),
                slot: 0,
                generation: SlotGeneration(1),
                manifest: model_manifest(),
            },
            EncodingHandle {
                owner: OwnerId(92),
                slot: 0,
                generation: SlotGeneration(1),
                manifest: encoding_manifest(),
            },
        )
        .unwrap()
    }
    fn request(
        adapter: &EntityCandidateMockAdapter,
        position: Position,
    ) -> Arc<EvalRequest<RulesState>> {
        let frozen = ContractPosition::new(OwnerId(93), position)
            .export()
            .unwrap();
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(ProcessEpoch(1), 1),
            selection: SelectionId::new(ProcessEpoch(1), 1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            state: frozen.snapshot().identity(),
            legal_order: frozen.legal_moves().order(),
            input: adapter
                .input_key(frozen.rules(), frozen.legal_moves().moves())
                .unwrap(),
            model: adapter.model().handle(),
            encoding: adapter.model().encoding().handle,
            precision: PrecisionProfile::Fp32,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            backend: adapter.backend(),
        };
        Arc::new(
            EvalRequest::try_new(
                context,
                frozen.snapshot().clone(),
                frozen.legal_moves().clone(),
                adapter.model().clone(),
                Deadline {
                    clock: ClockDomain(ProcessEpoch(1)),
                    at: MonotonicTick(u64::MAX),
                },
                CancelToken::new(),
                ByteBudget {
                    host: HOST_BYTES_PER_ITEM,
                    device: 0,
                    pinned: 0,
                },
            )
            .unwrap(),
        )
    }
    #[test]
    fn candidate_head_is_variable_and_preserves_both_sides_and_promotions() {
        let a = adapter();
        for fen in [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            // Include a rook so the request is ongoing under Rules' dead-position policy.
            let fen = fen.replace("8/4K3", "7R/4K3");
            let r = request(&a, Position::from_fen(&fen).unwrap());
            let p = a.prepare(r.clone()).unwrap();
            let raw = EntityCandidateMockAdapter::infer(&p.input);
            assert_eq!(raw.logits.len(), r.legal().moves().len());
            let o = a
                .decode(&p, &raw, ExecutionId::new(ProcessEpoch(1), 1))
                .unwrap();
            assert_eq!(o.legal.moves(), r.legal().moves());
            assert_eq!(o.viewpoint, Viewpoint::SideToMove);
            assert_eq!(o.wdl, Wdl::try_new(0.4, 0.3, 0.3, 1e-5).unwrap());
            if fen.contains("P7") {
                assert_eq!(
                    o.legal
                        .moves()
                        .iter()
                        .filter(|m| m.promotion.is_some())
                        .count(),
                    4
                );
            }
        }
        assert!(!a.capabilities().mixed_legal_batching);
    }
    #[test]
    fn input_key_covers_candidate_order_history_and_output_failures() {
        let a = adapter();
        let r = request(&a, Position::startpos());
        let p = a.prepare(r.clone()).unwrap();
        let mut input = a.encode(r.position().state(), r.legal().moves()).unwrap();
        let key = a.key(&input);
        input.candidates.reverse();
        assert_ne!(key, a.key(&input));
        input.candidates.reverse();
        input.known_history += 1;
        assert_ne!(key, a.key(&input));
        for raw in [
            CandidateRaw {
                logits: vec![0.0; 1],
                wdl: [0.4, 0.3, 0.3],
            },
            CandidateRaw {
                logits: vec![f32::NAN; p.input.candidates.len()],
                wdl: [0.4, 0.3, 0.3],
            },
            CandidateRaw {
                logits: vec![0.0; p.input.candidates.len()],
                wdl: [f32::NAN, 0.0, 1.0],
            },
        ] {
            assert!(a
                .decode(&p, &raw, ExecutionId::new(ProcessEpoch(1), 1))
                .is_err());
        }
        assert!(a
            .decode(
                &p,
                &EntityCandidateMockAdapter::infer(&p.input),
                ExecutionId::new(ProcessEpoch(2), 1)
            )
            .is_err());
        assert!(a
            .batch(ExecutionId::new(ProcessEpoch(1), 1), Vec::new())
            .is_err());
    }
}
