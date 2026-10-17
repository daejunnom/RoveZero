//! 원 manager hook와 native 입력 owner 사이의 유한 수집 어댑터다.
//! 파일 I/O·신경망 인코딩·forward는 수행하지 않으며, 실제 퇴역 row만
//! caller 소유 outbox로 이동한다. 오류가 원 물리 owner의 처리를 대체하지 않는다.
use super::*;
use crate::pals_collect_v3::{GenerationActionSupportV3, GenerationCaptureWorkScopeV3};
use rz_search::pals::coordinator::FrameInput;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionObserver, AdaptiveCollectionStage,
};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

const CAPACITY: usize = 16;

/// 실제 수집 실행에서 사전 등록한 식별자다. 이 값만으로 native·교사·학습
/// 권한을 발급하지 않으며, 원 root admission과 실제 ticket을 추가로 요구한다.
#[derive(Clone)]
pub struct RoventCollectionMetadata {
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub lineage: RoventLineage,
}

struct CollectionState {
    producer: RoventCorpusProducer,
    registration: RoventNativeCaptureRegistration,
    receiver: RoventCapturedInputReceiver,
    metadata: RoventCollectionMetadata,
    admission: Option<RoventCollectionRootAdmission>,
    actual_position: Option<Position>,
    actual_history: Vec<BoardMove>,
    slots: BTreeMap<RequestId, u64>,
    retired: VecDeque<RoventRetiredRecord>,
    failure: Option<AdaptiveCollectionError>,
}

/// caller는 root 사이에도 이 원 owner를 보존한다. 공개 record를 읽어 이
/// owner의 private native witness나 원 퇴역 proof를 재구성할 수 없다.
#[derive(Clone)]
pub struct RoventCollectionOwner {
    state: Arc<Mutex<CollectionState>>,
}

impl RoventCollectionOwner {
    pub fn new(
        manifest: RoventCorpusManifest,
        metadata: RoventCollectionMetadata,
    ) -> Result<(Self, RoventNativeInputObserver), ArenaError> {
        check(
            generation_identifier(&metadata.game_id)
                && generation_identifier(&metadata.opening_id)
                && generation_identifier(&metadata.line_genealogy_id)
                && manifest.opening_splits.contains_key(&metadata.opening_id),
            "collection metadata/opening is not registered",
        )?;
        let producer = RoventCorpusProducer::new(manifest)?;
        let (registration, observer, receiver) = rovent_native_capture_channel();
        Ok((
            Self {
                state: Arc::new(Mutex::new(CollectionState {
                    producer,
                    registration,
                    receiver,
                    metadata,
                    admission: None,
                    actual_position: None,
                    actual_history: Vec::new(),
                    slots: BTreeMap::new(),
                    retired: VecDeque::new(),
                    failure: None,
                })),
            },
            observer,
        ))
    }

    pub fn observer<Q: 'static, O: ProposerRankingOutput + 'static>(
        &self,
    ) -> Box<dyn AdaptiveCollectionObserver<Q, O>> {
        Box::new(RoventManagerCollectionObserver {
            owner: self.clone(),
            marker: PhantomData,
        })
    }

    /// teacher/outcome 부착·publication은 manager 밖의 원 caller가 수행한다.
    /// 이 borrow는 원 native·교사·Rules proof를 대신하지 않는다.
    pub fn with_producer<T>(
        &self,
        action: impl FnOnce(&mut RoventCorpusProducer) -> Result<T, ArenaError>,
    ) -> Result<T, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("collection owner poisoned"))?;
        action(&mut state.producer)
    }

    pub fn take_retired(&self) -> Result<Option<RoventRetiredRecord>, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("collection owner poisoned"))?;
        Ok(state.retired.pop_front())
    }

    pub fn failure(&self) -> Result<Option<AdaptiveCollectionError>, ArenaError> {
        Ok(self
            .state
            .lock()
            .map_err(|_| invalid("collection owner poisoned"))?
            .failure
            .clone())
    }
}

struct RoventManagerCollectionObserver<Q, O> {
    owner: RoventCollectionOwner,
    marker: PhantomData<fn(Q, O)>,
}

impl CollectionState {
    fn drain_captures(&mut self) -> Result<(), ArenaError> {
        self.receiver.admit_available(&mut self.producer)?;
        Ok(())
    }
    fn cancel_collector_copy(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), ArenaError> {
        if let Some(reservation) = self.registration.cancel_unstarted(ticket.id)? {
            self.producer.discard_reserved_input(reservation.sequence());
        }
        if let Some(sequence) = self.slots.remove(&ticket.id) {
            self.producer.discard_reserved_input(sequence);
        }
        self.producer.discard_unpublished(ticket.id);
        Ok(())
    }
    fn current(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<&RoventCollectionRootAdmission, ArenaError> {
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("original root was not admitted"))?;
        admission
            .validate_ticket(ticket)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(admission)
    }
}

impl<Q, O> RoventManagerCollectionObserver<Q, O> {
    fn apply<T>(
        &self,
        stage: AdaptiveCollectionStage,
        action: impl FnOnce(&mut CollectionState) -> Result<T, ArenaError>,
    ) -> Result<T, AdaptiveCollectionError> {
        let mut state = self.owner.state.lock().map_err(|_| {
            AdaptiveCollectionError::new(
                stage,
                "RoventCollectionPoisoned",
                "original collection owner poisoned",
            )
        })?;
        match action(&mut state) {
            Ok(value) => Ok(value),
            Err(error) => {
                let error = AdaptiveCollectionError::new(
                    stage,
                    "RoventCollectionEvidence",
                    error.to_string(),
                );
                state.failure.get_or_insert_with(|| error.clone());
                Err(error)
            }
        }
    }
}

impl<Q, O: ProposerRankingOutput> AdaptiveCollectionObserver<Q, O>
    for RoventManagerCollectionObserver<Q, O>
{
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            check(
                state.admission.is_none()
                    && state.slots.is_empty()
                    && state.producer.incomplete_capture_count() == 0
                    && state.failure.is_none(),
                "previous collection root is not settled",
            )?;
            state.producer.validate_admission(admission)?;
            let snapshot = admission.rules().rules().snapshot();
            let trace = snapshot
                .uci_replay(16_384)
                .map_err(|error| invalid(error.to_string()))?;
            let position = Position::from_snapshot(
                snapshot,
                rz_position::PositionLimits {
                    max_history_positions: 16_385,
                    ..rz_position::PositionLimits::default()
                },
            )
            .map_err(|error| invalid(error.to_string()))?;
            state.actual_position = Some(position);
            state.actual_history = trace.moves;
            state.admission = Some(admission.clone());
            Ok(())
        })
    }
    fn dispatch(
        &mut self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            state.current(ticket)?;
            check(
                state.failure.is_none()
                    && state.retired.len() + state.slots.len() < CAPACITY
                    && !state.slots.contains_key(&ticket.id)
                    && input.view_seal == ticket.view
                    && input.question_digest == ticket.question,
                "collection dispatch scope/queue capacity",
            )?;
            let manifest = state.producer.manifest();
            let context = GenerationCaptureContextV3 {
                game_id: state.metadata.game_id.clone(),
                opening_id: state.metadata.opening_id.clone(),
                line_genealogy_id: state.metadata.line_genealogy_id.clone(),
                model_epoch: manifest.model_epoch,
                context_epoch: manifest.context_epoch,
                capture_sequence: 0,
                work_scope: GenerationCaptureWorkScopeV3::from_native_ticket(ticket),
                actual_position: state
                    .actual_position
                    .as_ref()
                    .ok_or_else(|| invalid("original root Rules context missing"))?
                    .clone(),
                actual_history: state.actual_history.clone(),
                action_support: GenerationActionSupportV3::non_validator(),
            };
            let reservation = state
                .producer
                .reserve_input(context, state.metadata.lineage.clone())?;
            let sequence = reservation.sequence();
            if let Err(error) = state.registration.register(ticket, reservation) {
                state.producer.discard_reserved_input(sequence);
                return Err(error);
            }
            state.slots.insert(ticket.id, sequence);
            Ok(())
        })
    }
    fn native_event(
        &mut self,
        event: &TicketNativeEvent<O>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::NativeEvent, |state| {
            state.drain_captures()?;
            state.producer.observe_native_event(event)?;
            if let TicketNativeEvent::Retired {
                accepted: false,
                result: Err(error),
                ..
            } = event
            {
                return Err(invalid(format!(
                    "original native retirement failed: {error:?}"
                )));
            }
            Ok(())
        })
    }
    fn accepted(
        &mut self,
        accepted: &AcceptedRoleStep,
        _: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            let admission = state.current(accepted.ticket())?.clone();
            check(
                state.retired.len() < CAPACITY,
                "retired collection outbox full",
            )?;
            let row = state.producer.retire_accepted(
                &admission,
                accepted,
                output.proposer_ranking_output(),
            )?;
            state.slots.remove(&accepted.ticket().id);
            state.retired.push_back(row);
            Ok(())
        })
    }
    fn root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptance,
        _: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            let ticket = &accepted.ranking().source().ticket;
            check(
                state.current(ticket)?.raw_sha256() == accepted.admission().raw_sha256()
                    && state.retired.len() < CAPACITY,
                "initial P admission/outbox mismatch",
            )?;
            let row = state
                .producer
                .retire_root_ranking(accepted, output.proposer_ranking_output())?;
            state.slots.remove(&ticket.id);
            state.retired.push_back(row);
            Ok(())
        })
    }
    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Abandoned, |state| {
            state.current(ticket)?;
            state.drain_captures()?;
            state.cancel_collector_copy(ticket)
        })
    }
    fn settled(&mut self, tickets: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Settled, |state| {
            state.drain_captures()?;
            for ticket in tickets {
                state.current(ticket)?;
                state.cancel_collector_copy(ticket)?;
            }
            check(
                state.slots.is_empty() && state.producer.incomplete_capture_count() == 0,
                "fenced root retains unresolved collector inputs",
            )?;
            state.admission = None;
            state.actual_position = None;
            state.actual_history.clear();
            // 원 완료·퇴역 row와 최초 오류는 caller가 회수할 때까지 보존한다.
            Ok(())
        })
    }
}
