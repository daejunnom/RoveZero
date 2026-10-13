//! Bounded actual collection over the production native manager. The startup
//! owner provides registered native sessions; there is no mock-success or
//! metadata-only path to a playable turn or a process-close proof.
use super::*;
use rz_search::pals::engine::adaptive_port::AdaptivePortQuestion;
use rz_search::pals::engine::{PalsCompletion, RoleModel};
use rz_uci::pals_adaptive::{
    AdaptiveCollectionDispatchObserver, AdaptiveExecutionManager, AdaptiveManagerError,
    AdaptivePortDriver, CompletedValidatorAction,
};
use rz_uci::pals_adaptive_bootstrap::ActualNativeBackendV5;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_ROOT_RECORDS: usize = 128;
const MAX_ROOT_EVIDENCE: usize = 4096;
const MAX_GAME_LABEL_INDEX: usize = 32_768;
type ActualManager<M> = AdaptiveExecutionManager<
    AdaptivePortQuestion,
    rz_uci::pals_adaptive::ProjectedNativeBackend<AdaptivePortQuestion, ActualNativeBackendV5>,
    AdaptivePortDriver<M>,
>;

impl GenerationIsolatedActionV3 {
    /// The projected production backend, actual collection events and manager
    /// exclusive closure capability are all required. A generic/mock manager
    /// or arbitrary completed/serialized record cannot mint an action member.
    pub fn from_actual_manager<M: RoleModel + 'static>(
        manager: &mut ActualManager<M>,
        capture: &mut GenerationGameCaptureV3,
        registration: &FrozenActionRegistrationV3,
    ) -> Result<Self, ArenaError> {
        let receipt = manager
            .shutdown_receipt()
            .map_err(|e| invalid(e.to_string()))?;
        check(
            receipt.permits_root_reuse()
                && receipt.counters.nn_input_count_unknown == 0
                && receipt.counters.completed_nn_inputs > 0
                && manager.failures().next().is_none(),
            "isolated action requires a known production native root fence",
        )?;
        capture.poll()?;
        check(
            capture
                .owners
                .iter()
                .all(|owner| owner.join.pending_count() == 0),
            "isolated action capture not retired",
        )?;
        let semantic = manager
            .driver_mut()
            .take_action_outcome()
            .ok_or_else(|| invalid("isolated root has no actual action outcome"))?;
        check(
            manager.driver_mut().take_action_outcome().is_none(),
            "isolated root executed more than one action",
        )?;
        let completed = manager
            .complete_exclusive_action(semantic)
            .map_err(|e| invalid(e.to_string()))?;
        let ticket = super::capture::generation_role_ticket_json_v3(completed.semantic().source());
        let mut originals = capture.evidence.values().filter(|raw| {
            raw.get("schema").and_then(|v| v.as_str())
                == Some("rz-pals-generation-submission-evidence/1")
                && raw.get("ticket") == Some(&ticket)
        });
        let key = originals
            .next()
            .and_then(|raw| raw.get("input_sha256"))
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| invalid("isolated action lacks original native input publication"))?;
        check(
            originals.next().is_none(),
            "isolated action native publication is ambiguous",
        )?;
        let record = capture
            .records
            .get(&key)
            .ok_or_else(|| invalid("isolated action original accepted row unavailable"))?;
        let outcome = capture_isolated_validator_action_v3(
            record,
            completed,
            &mut *capture
                .chronology
                .lock()
                .map_err(|_| invalid("collection clock poisoned"))?,
            &mut capture.evidence,
            registration,
        )?;
        let record = capture
            .records
            .remove(&key)
            .expect("original row validated before removal");
        let additional_records = std::mem::take(&mut capture.records).into_values().collect();
        Ok(Self {
            record,
            outcome,
            evidence: std::mem::take(&mut capture.evidence),
            root_authority: receipt.authority,
            additional_records,
        })
    }
}

/// A real root from an actual PALS turn, preserved before that chosen move is
/// applied. Collection cannot fabricate this value from a FEN or PGN flag.
#[derive(Clone)]
pub struct GenerationSavedActionRootV3 {
    game_id: String,
    opening_id: String,
    position: Position,
    source_epoch: u64,
    // A reissued saved root retains its original verified archive owner. The
    // field is never filled by a JSON/FEN constructor or a passing receipt.
    _source_archive: Option<Arc<VerifiedGenerationGameArchiveV3>>,
}
impl GenerationSavedActionRootV3 {
    fn from_actual_turn(
        assignment: &GenerationGameAssignmentV3,
        position: &Position,
        turn: &GenerationActualTurnV3,
    ) -> Result<Self, ArenaError> {
        check(
            turn.root == exact_rules_anchor(position)?
                && generation_identifier(&assignment.game_id)
                && generation_identifier(&assignment.opening_id),
            "saved action root differs from actual accepted engine turn",
        )?;
        Ok(Self {
            game_id: assignment.game_id.clone(),
            opening_id: assignment.opening_id.clone(),
            position: position.clone(),
            source_epoch: turn.epoch,
            _source_archive: None,
        })
    }
    pub(super) fn from_verified_archive(
        source: Arc<VerifiedGenerationGameArchiveV3>,
        position: Position,
        source_epoch: u64,
    ) -> Result<Self, ArenaError> {
        let assignment = source.request().assignment()?;
        check(source_epoch > 0, "verified saved root source epoch")?;
        Ok(Self {
            game_id: assignment.game_id,
            opening_id: assignment.opening_id,
            position,
            source_epoch,
            _source_archive: Some(source),
        })
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    pub fn game_id(&self) -> &str {
        &self.game_id
    }
    pub fn opening_id(&self) -> &str {
        &self.opening_id
    }
    pub fn source_epoch(&self) -> u64 {
        self.source_epoch
    }
}

pub trait GenerationActionOutcomeFactoryV3 {
    /// Preparation must issue its actual immutable Search question before any
    /// member action. It also owns the content-verified native frozen PC pins.
    fn prepare_context(
        &mut self,
        root: &GenerationSavedActionRootV3,
        plan: &ActionOutcomeCollectionPlanV3,
        until: Instant,
    ) -> Result<FrozenActionRegistrationV3, ArenaError>;
    /// Exactly one registered action runs in a fresh isolated Search root.
    /// Its production manager and observer chain mint the private member.
    fn execute_isolated(
        &mut self,
        root: &GenerationSavedActionRootV3,
        registration: &FrozenActionRegistrationV3,
        action: usize,
        chronology: Arc<Mutex<GenerationChronologyV3>>,
        until: Instant,
    ) -> Result<GenerationIsolatedActionV3, ArenaError>;
    fn close_context(
        &mut self,
        until: Instant,
    ) -> Result<GenerationActualActionContextCloseV3, ArenaError>;
}

pub struct GenerationActualActionContextCloseV3 {
    epochs: BTreeSet<u64>,
}
impl GenerationActualActionContextCloseV3 {
    pub fn from_actual_session_receipt(
        receipt: &rz_uci::pals_adaptive::AdaptiveSessionProcessReceipt,
    ) -> Result<Self, ArenaError> {
        check(
            receipt.permits_process_close(),
            "action context requires actual native/CPU_T/archive process closure",
        )?;
        Ok(Self {
            epochs: BTreeSet::from([receipt.epoch().0]),
        })
    }
    /// Every isolated action has a fresh process owner. Closing a context joins
    /// their real process receipts; a root fence cannot stand in for any one
    /// of these process closures, and duplicate epochs are rejected.
    pub fn from_actual_session_receipts(
        receipts: &[rz_uci::pals_adaptive::AdaptiveSessionProcessReceipt],
    ) -> Result<Self, ArenaError> {
        check(
            !receipts.is_empty() && receipts.len() <= 8,
            "action context process closure capacity",
        )?;
        let mut epochs = BTreeSet::new();
        for receipt in receipts {
            check(
                receipt.permits_process_close() && epochs.insert(receipt.epoch().0),
                "action context requires distinct actual process closures",
            )?;
        }
        Ok(Self { epochs })
    }
}
pub type GenerationSavedRootObserverV3 =
    Arc<dyn Fn(GenerationSavedActionRootV3) -> Result<(), ArenaError> + Send + Sync>;

/// Issued by reading an actual concrete native manager after its own physical
/// root fence. Callers cannot wrap an arbitrary PalsResult or `passed` flag.
pub struct GenerationActualTurnV3 {
    root: GenerationRulesAnchorV3,
    best_move: Option<BoardMove>,
    epoch: u64,
    completion: PalsCompletion,
    actions: Vec<CompletedValidatorAction>,
}
impl GenerationActualTurnV3 {
    pub fn from_actual_manager<M: RoleModel + 'static>(
        manager: &mut ActualManager<M>,
        completion: PalsCompletion,
    ) -> Result<Self, ArenaError> {
        let receipt = manager
            .shutdown_receipt()
            .map_err(|e| invalid(e.to_string()))?;
        check(
            receipt.permits_semantic_closure()
                && receipt.counters.nn_input_count_unknown == 0
                && receipt.counters.completed_nn_inputs > 0
                && manager.failures().next().is_none(),
            "actual turn requires known native inference and physical root fence",
        )?;
        let port = manager
            .driver()
            .port()
            .ok_or_else(|| invalid("actual Search owner unavailable after root fence"))?;
        let actual_root = Position::from_snapshot(
            port.root_rules().rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let root = exact_rules_anchor(&actual_root)?;
        let result = manager
            .driver()
            .result(completion)
            .map_err(|e| invalid(e.to_string()))?;
        let mut actions = vec![];
        while let Some(semantic) = manager.driver_mut().take_action_outcome() {
            check(
                actions.len() < 128,
                "actual root action completion capacity",
            )?;
            actions.push(
                manager
                    .driver()
                    .complete_action(semantic, &receipt)
                    .map_err(|e| invalid(e.to_string()))?,
            );
        }
        Ok(Self {
            root,
            best_move: result.best_move,
            epoch: receipt.authority.epoch.0,
            completion: result.completion,
            actions,
        })
    }
    pub fn best_move(&self) -> Option<BoardMove> {
        self.best_move
    }
    pub fn completion(&self) -> PalsCompletion {
        self.completion
    }
}

pub struct GenerationActualGameCloseV3 {
    epochs: BTreeSet<u64>,
}
impl GenerationActualGameCloseV3 {
    pub fn process_epochs(&self) -> impl Iterator<Item = rz_contracts::ProcessEpoch> + '_ {
        self.epochs.iter().copied().map(rz_contracts::ProcessEpoch)
    }
    pub fn from_actual_session_receipts(
        a: &rz_uci::pals_adaptive::AdaptiveSessionProcessReceipt,
        b: &rz_uci::pals_adaptive::AdaptiveSessionProcessReceipt,
    ) -> Result<Self, ArenaError> {
        check(
            a.permits_process_close() && b.permits_process_close() && a.epoch() != b.epoch(),
            "both actual native session processes and archives must be closed",
        )?;
        Ok(Self {
            epochs: BTreeSet::from([a.epoch().0, b.epoch().0]),
        })
    }
    /// Whole-game closure joins both actual model/session owners. A preceding
    /// reusable root fence cannot be relabeled as a process shutdown.
    pub fn from_actual_managers<A: RoleModel + 'static, B: RoleModel + 'static>(
        a: &ActualManager<A>,
        b: &ActualManager<B>,
    ) -> Result<Self, ArenaError> {
        let a = a.shutdown_receipt().map_err(|e| invalid(e.to_string()))?;
        let b = b.shutdown_receipt().map_err(|e| invalid(e.to_string()))?;
        check(
            a.permits_archive_close()
                && b.permits_archive_close()
                && a.authority.epoch != b.authority.epoch,
            "both actual game processes must be physically closed",
        )?;
        Ok(Self {
            epochs: BTreeSet::from([a.authority.epoch.0, b.authority.epoch.0]),
        })
    }
}

/// Startup implementation owns real native P/C/V+CPU_T sessions. Its only
/// successful turn and close values can be issued by concrete native managers.
pub trait GenerationSelfplayGameV3 {
    fn engine_names(&self) -> [String; 2];
    /// Actual process clocks are registered before any turn. Array order is
    /// engine A/B and remains independent of the paired color assignment.
    fn process_epochs(&self) -> [rz_contracts::ProcessEpoch; 2];
    fn decide(
        &mut self,
        position: &Position,
        remaining: Duration,
        increment: Duration,
        hard_deadline: Instant,
        collector: &mut GenerationGameCaptureV3,
    ) -> Result<GenerationActualTurnV3, ArenaError>;
    fn close(
        &mut self,
        until: Instant,
        collector: &mut GenerationGameCaptureV3,
    ) -> Result<GenerationActualGameCloseV3, ArenaError>;
}
pub trait GenerationSelfplayFactoryV3 {
    type Game: GenerationSelfplayGameV3;
    fn start_game(
        &mut self,
        assignment: &GenerationGameAssignmentV3,
        initial: &Position,
        native_observers: [NativeGenerationObserverV3; 2],
        dispatch_observers: [AdaptiveCollectionDispatchObserver; 2],
        teacher_observers: [GenerationTeacherObserverV3; 2],
        until: Instant,
    ) -> Result<Self::Game, ArenaError>;
}

#[derive(Clone)]
struct CaptureMetadata {
    game_id: String,
    opening_id: String,
    model_epoch: u64,
    context_epoch: u64,
    root: Option<Position>,
}
struct CaptureOwner {
    _registration: Arc<GenerationCaptureRegistrationV3>,
    receiver: GenerationCaptureReceiverV3,
    join: GenerationCaptureJoinV3,
    metadata: Arc<Mutex<CaptureMetadata>>,
    teacher_receiver: super::teacher::GenerationTeacherReceiverV3,
    teacher_observer: GenerationTeacherObserverV3,
}

/// Private input/future-label ownership for one actual game. Native events are
/// drained outside the manager's tick; file writes remain on the archive owner.
pub struct GenerationGameCaptureV3 {
    owners: [CaptureOwner; 2],
    chronology: Arc<Mutex<GenerationChronologyV3>>,
    records: BTreeMap<String, GenerationRecordV3>,
    evidence: BTreeMap<String, serde_json::Value>,
    process_epochs: BTreeSet<u64>,
}
impl GenerationGameCaptureV3 {
    /// Original sealed observations retained by this capture owner. This does
    /// not issue target or completion authority and never reconstructs a row.
    pub fn retained_evidence(&self) -> &BTreeMap<String, serde_json::Value> {
        &self.evidence
    }
    pub fn new(
        game_id: String,
        opening_id: String,
        model_epoch: u64,
        context_epoch: u64,
        chronology: Arc<Mutex<GenerationChronologyV3>>,
    ) -> Result<
        (
            Self,
            [NativeGenerationObserverV3; 2],
            [AdaptiveCollectionDispatchObserver; 2],
        ),
        ArenaError,
    > {
        check(
            generation_identifier(&game_id)
                && generation_identifier(&opening_id)
                && model_epoch > 0
                && context_epoch > 0,
            "registered collection game/epochs",
        )?;
        let make = || {
            let (registration, observer, receiver) = generation_capture_channel_v3();
            let (teacher_observer, teacher_receiver) =
                super::teacher::teacher_channel_v3(chronology.clone());
            let registration = Arc::new(registration);
            let metadata = Arc::new(Mutex::new(CaptureMetadata {
                game_id: game_id.clone(),
                opening_id: opening_id.clone(),
                model_epoch,
                context_epoch,
                root: None,
            }));
            let hook_registration = registration.clone();
            let hook_metadata = metadata.clone();
            let hook_clock = chronology.clone();
            let hook: AdaptiveCollectionDispatchObserver =
                Arc::new(move |ticket, _input, validator| {
                    let metadata = hook_metadata
                        .lock()
                        .map_err(|_| AdaptiveManagerError::Invalid("collection metadata poisoned"))?
                        .clone();
                    let actual_position = metadata.root.ok_or(AdaptiveManagerError::Invalid(
                        "collection root not activated",
                    ))?;
                    let history = actual_position
                        .snapshot()
                        .uci_replay(16_384)
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?
                        .moves;
                    let capture_sequence = hook_clock
                        .lock()
                        .map_err(|_| AdaptiveManagerError::Invalid("collection clock poisoned"))?
                        .next_event_sequence()
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
                    let action_support = match validator {
                        Some(question) => {
                            GenerationActionSupportV3::from_validator_question(question)
                        }
                        None => GenerationActionSupportV3::non_validator(),
                    };
                    hook_registration
                        .register(
                            *ticket,
                            GenerationCaptureContextV3 {
                                game_id: metadata.game_id.clone(),
                                opening_id: metadata.opening_id,
                                line_genealogy_id: metadata.game_id.clone(),
                                model_epoch: metadata.model_epoch,
                                context_epoch: metadata.context_epoch,
                                capture_sequence,
                                work_scope: GenerationCaptureWorkScopeV3::from_native_ticket(
                                    ticket,
                                ),
                                actual_position,
                                actual_history: history,
                                action_support,
                            },
                        )
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))
                });
            (
                CaptureOwner {
                    _registration: registration,
                    receiver,
                    join: GenerationCaptureJoinV3::new(),
                    metadata,
                    teacher_receiver,
                    teacher_observer,
                },
                observer,
                hook,
            )
        };
        let (a, oa, ha) = make();
        let (b, ob, hb) = make();
        Ok((
            Self {
                owners: [a, b],
                chronology,
                records: BTreeMap::new(),
                evidence: BTreeMap::new(),
                process_epochs: BTreeSet::new(),
            },
            [oa, ob],
            [ha, hb],
        ))
    }
    pub fn activate_root(&mut self, actual: &Position) -> Result<(), ArenaError> {
        self.poll()?;
        check(
            self.records.is_empty()
                && self.evidence.is_empty()
                && self.owners.iter().all(|o| {
                    o.join.pending_count() == 0 && o.teacher_receiver.pending_count() == 0
                }),
            "previous root capture must be persisted before next admission",
        )?;
        for owner in &self.owners {
            owner
                .metadata
                .lock()
                .map_err(|_| invalid("collection metadata poisoned"))?
                .root = Some(actual.clone());
        }
        for owner in &mut self.owners {
            owner.teacher_receiver.reset_root()?;
        }
        Ok(())
    }
    pub fn teacher_observers(&self) -> [GenerationTeacherObserverV3; 2] {
        [
            self.owners[0].teacher_observer.clone(),
            self.owners[1].teacher_observer.clone(),
        ]
    }
    pub fn register_teacher_producer(
        &self,
        producer: &GenerationProducerAuthorityV3,
    ) -> Result<(), ArenaError> {
        for observer in self.teacher_observers() {
            observer.expect_source(
                &producer.teacher_binary_sha256,
                &producer.teacher_profile_sha256,
            )?;
        }
        Ok(())
    }
    pub fn poll(&mut self) -> Result<bool, ArenaError> {
        let mut changed = false;
        for owner in &mut self.owners {
            while let Some(event) = owner.receiver.try_event()? {
                let retirement = match &event {
                    GenerationNativeCaptureEventV3::Retired { ticket, .. } => Some(*ticket),
                    _ => None,
                };
                if let Some(record) = owner.join.apply(event)? {
                    check(
                        self.records.len() < MAX_ROOT_RECORDS
                            && !self.records.contains_key(&record.input_sha256),
                        "actual root collection row capacity/duplicate",
                    )?;
                    self.records.insert(record.input_sha256.clone(), record);
                }
                for receipt in owner.join.take_receipts() {
                    check(
                        self.evidence.len() < MAX_ROOT_EVIDENCE,
                        "actual root raw receipt capacity",
                    )?;
                    let schema = receipt
                        .get("schema")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| invalid("actual receipt schema"))?;
                    let sha = generation_digest(schema, &receipt)?;
                    self.evidence.insert(sha, receipt);
                }
                // This acknowledgement means received/owned raw evidence, not
                // durable publication. Admission below stays bounded by the
                // private root records/evidence caps until archive flush.
                if let Some(ticket) = retirement {
                    owner.receiver.acknowledge_persisted_retirement(&ticket)?;
                }
                changed = true;
            }
        }
        for owner in &mut self.owners {
            changed |= owner
                .teacher_receiver
                .poll(&mut self.records, &mut self.evidence)?;
        }
        Ok(changed)
    }
    pub fn accepted_records(&self) -> impl Iterator<Item = &GenerationRecordV3> {
        self.records.values()
    }
    pub fn attach_teacher_policy(
        &mut self,
        input_sha: &str,
        binding: &TeacherInputBindingV3,
        original: &rz_search::external_cpu::worker::ExternalJob,
        actual: &rz_search::external_cpu::worker::ExternalWorkerCompletion,
        binary_sha: &str,
        profile_sha: &str,
    ) -> Result<bool, ArenaError> {
        let record = self
            .records
            .get_mut(input_sha)
            .ok_or_else(|| invalid("teacher target before original accepted capture"))?;
        let sequence = self
            .chronology
            .lock()
            .map_err(|_| invalid("collection clock poisoned"))?
            .next_event_sequence()?;
        attach_completed_teacher_policy_v3(
            record,
            binding,
            original,
            actual,
            sequence,
            binary_sha,
            profile_sha,
            &mut self.evidence,
        )
    }
    fn complete_turn(&mut self, turn: &mut GenerationActualTurnV3) -> Result<(), ArenaError> {
        self.poll()?;
        check(
            self.owners
                .iter()
                .all(|o| o.join.pending_count() == 0 && o.teacher_receiver.pending_count() == 0)
                && self.evidence.values().any(|raw| {
                    raw.get("schema").and_then(|v| v.as_str())
                        == Some("rz-pals-generation-submission-evidence/1")
                        && raw
                            .get("ticket")
                            .and_then(|t| t.get("epoch"))
                            .and_then(|v| v.as_u64())
                            == Some(turn.epoch)
                }),
            "actual root fence left incomplete collection tickets",
        )?;
        self.process_epochs.insert(turn.epoch);
        for action in turn.actions.drain(..) {
            let ticket = super::capture::generation_role_ticket_json_v3(action.semantic().source());
            let mut original = self.evidence.values().filter(|raw| {
                raw.get("schema").and_then(|v| v.as_str())
                    == Some("rz-pals-generation-submission-evidence/1")
                    && raw.get("ticket") == Some(&ticket)
            });
            let key = original
                .next()
                .and_then(|raw| raw.get("input_sha256"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .ok_or_else(|| invalid("actual V action lacks its accepted original input"))?;
            check(
                original.next().is_none(),
                "V action original ticket publication is ambiguous",
            )?;
            let row = self
                .records
                .get_mut(&key)
                .ok_or_else(|| invalid("V action original row unavailable"))?;
            let mut clock = self
                .chronology
                .lock()
                .map_err(|_| invalid("collection clock poisoned"))?;
            let outcome =
                capture_completed_validator_action_v3(row, action, &mut clock, &mut self.evidence)?;
            let sequence = clock.next_event_sequence()?;
            attach_actual_task_outcomes_v3(row, vec![outcome], sequence, &mut self.evidence)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct DeferredWdlInput {
    input_sha256: String,
    game_id: String,
    role: GenerationRoleV3,
    question_id: String,
    view_sha256: String,
    anchor: GenerationRulesAnchorV3,
}
pub struct GenerationCollectionGameReportV3 {
    pub game_id: String,
    pub opening_id: String,
    pub engine_a_white: bool,
    pub rules_terminated: bool,
    pub termination: CollectionPgnTerminationV3,
    pub recorded_inputs: usize,
    pub pgn_receipt_sha256: String,
}

/// A single actual game plus its joined, readback-sealed archive. This is
/// distinct from a completed 128-game generation and cannot be deserialized.
pub struct GenerationStandaloneGameArchiveV3 {
    directory: PathBuf,
    manifest: GenerationChunkManifestV3,
    executed: ExecutedGenerationGameV3,
}
struct ExecutedGenerationGameV3 {
    report: GenerationCollectionGameReportV3,
    close: GenerationActualGameCloseV3,
    trajectory: ActualGameTrajectoryV3,
    names: [String; 2],
    decisions: Vec<GenerationGameDecisionTraceV3>,
}
/// Raw actual decision boundaries, retained for independent Rules/clock/PGN
/// replay by the source-pinned whole-game parent. This is not turn authority.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationGameDecisionTraceV3 {
    pub movement: u16,
    pub elapsed_ns: u64,
}
impl GenerationStandaloneGameArchiveV3 {
    pub fn directory(&self) -> &std::path::Path {
        &self.directory
    }
    pub fn manifest(&self) -> &GenerationChunkManifestV3 {
        &self.manifest
    }
    pub fn report(&self) -> &GenerationCollectionGameReportV3 {
        &self.executed.report
    }
    pub fn actual_close(&self) -> &GenerationActualGameCloseV3 {
        &self.executed.close
    }
    pub fn actual_trajectory(&self) -> &ActualGameTrajectoryV3 {
        &self.executed.trajectory
    }
    pub fn engine_names(&self) -> &[String; 2] {
        &self.executed.names
    }
    pub fn decision_trace(&self) -> &[GenerationGameDecisionTraceV3] {
        &self.executed.decisions
    }
}

/// Owns one fresh, bounded archive and the pre-outcome opening assignment.
/// Each invocation executes exactly the next color-exchanged pair. The caller
/// renews finite cloud windows between pairs; a partial run is never finished
/// or admitted to the trainer as the completed 128-game generation.
pub struct GenerationSelfplayCollectorV3 {
    plan: SelfplayCollectionPlanV3,
    producer: GenerationProducerAuthorityV3,
    writer: GenerationWriterWorkerV3,
    chronology: Arc<Mutex<GenerationChronologyV3>>,
    model_epoch: u64,
    context_epoch: u64,
    next_game: usize,
    input_seals: BTreeSet<String>,
    actual_game_ids: Vec<String>,
    reported_pairs: usize,
    unsubmitted: Vec<GenerationWriteCommandV3>,
    partial: Option<GenerationPartialArchiveV3>,
    halted: bool,
    saved_root_observer: Option<GenerationSavedRootObserverV3>,
}
impl GenerationSelfplayCollectorV3 {
    pub fn new(
        plan: SelfplayCollectionPlanV3,
        producer: GenerationProducerAuthorityV3,
        model_epoch: u64,
        context_epoch: u64,
        directory: PathBuf,
        archive_deadline: Instant,
    ) -> Result<Self, ArenaError> {
        plan.validate()?;
        producer.validate(&plan.opening_splits())?;
        check(
            producer.input_seals.is_empty()
                && producer.actual_game_ids.is_empty()
                && producer.opening_split_sha256 == plan.split_sha256()?
                && model_epoch > 0
                && context_epoch > 0,
            "collection source pins/splits must be registered before outcomes",
        )?;
        let writer = GenerationWriterWorkerV3::spawn(
            directory,
            plan.run_id.clone(),
            plan.maximum_output_bytes,
            archive_deadline,
        )?;
        Ok(Self {
            plan,
            producer,
            writer,
            chronology: Arc::new(Mutex::new(GenerationChronologyV3::default())),
            model_epoch,
            context_epoch,
            next_game: 0,
            input_seals: BTreeSet::new(),
            actual_game_ids: vec![],
            reported_pairs: 0,
            unsubmitted: vec![],
            partial: None,
            halted: false,
            saved_root_observer: None,
        })
    }
    pub fn attempted_games(&self) -> usize {
        self.next_game
    }
    pub fn reported_pairs(&self) -> usize {
        self.reported_pairs
    }
    pub fn set_saved_root_observer(
        &mut self,
        observer: GenerationSavedRootObserverV3,
    ) -> Result<(), ArenaError> {
        check(
            self.next_game == 0 && !self.halted && self.saved_root_observer.is_none(),
            "actual saved-root source observer must precede collection",
        )?;
        self.saved_root_observer = Some(observer);
        Ok(())
    }
    pub fn cancel(&self) {
        self.writer.cancel();
    }
    /// On a window timeout the owner remains in this collector. A caller can
    /// renew only this bounded cleanup wait; cancel does not imply a join.
    pub fn shutdown_archive(&mut self, until: Instant) -> Result<(), ArenaError> {
        self.halted = true;
        self.writer.cancel();
        self.await_archive_join(until)
    }
    fn await_archive_join(&mut self, until: Instant) -> Result<(), ArenaError> {
        loop {
            if self.writer.try_join()? {
                return Ok(());
            }
            check(
                Instant::now() < until,
                "archive completion unknown; owner retained for bounded cleanup",
            )?;
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    pub fn partial_archive(&self) -> Option<&GenerationPartialArchiveV3> {
        self.partial.as_ref()
    }
    /// Commands that the archive owner never accepted remain with the source
    /// owner. They are not included in a durable-input or completed-run count.
    pub fn take_unsubmitted(&mut self) -> Vec<GenerationWriteCommandV3> {
        std::mem::take(&mut self.unsubmitted)
    }
    fn submit(
        &mut self,
        mut command: GenerationWriteCommandV3,
        until: Instant,
    ) -> Result<(), ArenaError> {
        loop {
            match self.writer.try_submit(command) {
                Ok(()) => return Ok(()),
                Err((error, owned)) => {
                    command = owned;
                    if matches!(error, ArenaError::Budget(_)) && Instant::now() < until {
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    self.halted = true;
                    self.unsubmitted.push(command);
                    return Err(error);
                }
            }
        }
    }
    fn wait_ready(&self, until: Instant) -> Result<(), ArenaError> {
        loop {
            match self.writer.try_event() {
                Some(GenerationWriterEventV3::Ready) => return Ok(()),
                Some(GenerationWriterEventV3::Failed(error)) => return Err(invalid(error)),
                Some(GenerationWriterEventV3::Finished { .. }) => {
                    return Err(invalid("writer finished before collection"));
                }
                Some(GenerationWriterEventV3::PartialPreserved(_)) => {
                    return Err(invalid("partial archive is not open collection"));
                }
                None => {
                    check(Instant::now() < until, "archive preparation deadline")?;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
    pub fn run_next_pair<F: GenerationSelfplayFactoryV3>(
        &mut self,
        factory: &mut F,
        until: Instant,
    ) -> Result<Vec<GenerationCollectionGameReportV3>, ArenaError> {
        // Register the whole pair's cleanup ceiling before any startup.
        let window_cleanup_until = until
            .checked_add(Duration::from_secs(30))
            .ok_or_else(|| invalid("collection original cleanup deadline overflow"))?;
        check(
            !self.halted
                && Instant::now() < until
                && self.next_game < 128
                && self.next_game % 2 == 0,
            "collection requires next finite complete pair",
        )?;
        if self.next_game == 0 {
            self.wait_ready(until)?;
        }
        let mut reports = vec![];
        for _ in 0..2 {
            reports.push(
                self.run_assigned_game(factory, until, window_cleanup_until, None)?
                    .report,
            );
        }
        self.reported_pairs += 1;
        Ok(reports)
    }
    /// Executes one preassigned game under the same actual body as paired
    /// collection. This sibling-only entry cannot finish the 128-game owner.
    #[allow(dead_code)] // Preserve the no-cancellation sibling compatibility entry.
    pub(super) fn run_standalone_game_child<F: GenerationSelfplayFactoryV3>(
        &mut self,
        game_ordinal: usize,
        factory: &mut F,
        until: Instant,
        archive_until: Instant,
    ) -> Result<GenerationStandaloneGameArchiveV3, ArenaError> {
        self.run_standalone_game_child_with_cancellation(
            game_ordinal,
            factory,
            until,
            archive_until,
            None,
        )
    }
    /// The whole-game worker supplies its original latched signal. A cancel
    /// observed during a game follows the same close/partial archive failure
    /// path as other infrastructure errors and never grants completed output.
    pub(super) fn run_standalone_game_child_with_cancellation<F: GenerationSelfplayFactoryV3>(
        &mut self,
        game_ordinal: usize,
        factory: &mut F,
        until: Instant,
        archive_until: Instant,
        cancel: Option<&AtomicBool>,
    ) -> Result<GenerationStandaloneGameArchiveV3, ArenaError> {
        check(
            !self.halted
                && self.next_game == 0
                && self.reported_pairs == 0
                && self.input_seals.is_empty()
                && self.actual_game_ids.is_empty()
                && game_ordinal < 128
                && Instant::now() < until
                && until <= archive_until
                && !cancel.is_some_and(|signal| signal.load(Ordering::Acquire)),
            "standalone child requires one fresh preassigned game archive",
        )?;
        let cleanup_until = until
            .checked_add(Duration::from_secs(30))
            .ok_or_else(|| invalid("standalone original cleanup deadline overflow"))?
            .min(archive_until);
        self.wait_ready(until)?;
        self.next_game = game_ordinal;
        let executed = self.run_assigned_game(factory, until, cleanup_until, cancel)?;
        check(
            self.actual_game_ids.len() == 1 && self.reported_pairs == 0,
            "standalone archive cannot contain a second game or paired completion",
        )?;
        let (directory, manifest) = self.finish_registered_archive(archive_until)?;
        Ok(GenerationStandaloneGameArchiveV3 {
            directory,
            manifest,
            executed,
        })
    }
    fn run_assigned_game<F: GenerationSelfplayFactoryV3>(
        &mut self,
        factory: &mut F,
        until: Instant,
        window_cleanup_until: Instant,
        cancel: Option<&AtomicBool>,
    ) -> Result<ExecutedGenerationGameV3, ArenaError> {
        check(
            !cancel.is_some_and(|signal| signal.load(Ordering::Acquire)),
            "original whole-game canceled before actual startup",
        )?;
        let assignments = self.plan.games()?;
        let assignment = &assignments[self.next_game];
        let opening = self
            .plan
            .openings
            .iter()
            .find(|o| o.id == assignment.opening_id)
            .ok_or_else(|| invalid("preassigned opening absent"))?;
        let mut initial = match &opening.initial_fen {
            Some(fen) => Position::from_fen(fen).map_err(|e| invalid(e.to_string()))?,
            None => Position::startpos(),
        };
        for movement in &opening.moves {
            initial
                .make_move(BoardMove::from_uci(movement).map_err(|e| invalid(e.to_string()))?)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let (capture, observers, hooks) = GenerationGameCaptureV3::new(
            assignment.game_id.clone(),
            assignment.opening_id.clone(),
            self.model_epoch,
            self.context_epoch,
            self.chronology.clone(),
        )?;
        let mut capture = capture;
        let teacher_hooks = capture.teacher_observers();
        for hook in &teacher_hooks {
            hook.expect_source(
                &self.producer.teacher_binary_sha256,
                &self.producer.teacher_profile_sha256,
            )?;
        }
        let mut game =
            factory.start_game(assignment, &initial, observers, hooks, teacher_hooks, until)?;
        let names = game.engine_names();
        let epochs = game.process_epochs();
        let (white, black) = if assignment.engine_a_white {
            (names[0].clone(), names[1].clone())
        } else {
            (names[1].clone(), names[0].clone())
        };
        let initialized = (|| -> Result<_, ArenaError> {
            check(
                epochs[0] != epochs[1],
                "selfplay engines require distinct actual process epochs",
            )?;
            Ok((
                CollectionGamePgnV3::begin(assignment.game_id.clone(), white, black, &initial)?,
                ActualGameTrajectoryV3::begin(assignment.game_id.clone(), initial)?,
            ))
        })();
        let (mut pgn, mut trajectory) = match initialized {
            Ok(value) => value,
            Err(error) => {
                let cleanup = game.close(
                    (Instant::now() + Duration::from_secs(30)).min(window_cleanup_until),
                    &mut capture,
                );
                return Err(invalid(format!(
                    "game initialization: {error}; actual game close: {}",
                    match cleanup {
                        Ok(_) => "confirmed by actual owners".to_owned(),
                        Err(error) => error.to_string(),
                    }
                )));
            }
        };
        let mut deferred = vec![];
        let persisted_before_game = self.input_seals.len();
        let (mut rules_authority, mut rules_receipt) = (None, None);
        let (mut termination, mut result) = (
            CollectionPgnTerminationV3::PlyLimit,
            CollectionPgnResultV3::Incomplete,
        );
        let mut decision_failure = None;
        let mut decisions = vec![];
        let played = (|| -> Result<(), ArenaError> {
            for _ in 0..self.plan.max_plies {
                check(
                    !cancel.is_some_and(|signal| signal.load(Ordering::Acquire)),
                    "original whole-game canceled between actual decisions",
                )?;
                if matches!(
                    trajectory
                        .position()
                        .classify_position()
                        .map_err(|e| invalid(e.to_string()))?
                        .play_status,
                    rz_position::PlayStatus::Terminal { .. }
                ) {
                    let (authority, receipt) = trajectory.terminate_rules()?;
                    result = match authority.outcome {
                        GenerationOutcomeV3::WhiteWin => CollectionPgnResultV3::WhiteWin,
                        GenerationOutcomeV3::BlackWin => CollectionPgnResultV3::BlackWin,
                        GenerationOutcomeV3::Draw => CollectionPgnResultV3::Draw,
                    };
                    termination = CollectionPgnTerminationV3::RulesAutomatic;
                    rules_authority = Some(authority);
                    rules_receipt = Some(receipt);
                    break;
                }
                check(
                    Instant::now() < until,
                    "pair window ended before next actual decision",
                )?;
                let before = trajectory.position().clone();
                let side = before.side_to_move();
                capture.activate_root(&before)?;
                let started = Instant::now();
                let deadline = started
                    .checked_add(pgn.remaining(side))
                    .ok_or_else(|| invalid("game clock deadline overflow"))?
                    .min(until);
                let decision = game.decide(
                    &before,
                    pgn.remaining(side),
                    Duration::from_millis(self.plan.increment_ms),
                    deadline,
                    &mut capture,
                );
                let decision_elapsed = started.elapsed(); // exact returned decision boundary
                if cancel.is_some_and(|signal| signal.load(Ordering::Acquire)) {
                    pgn.record_failed_decision(
                        &before,
                        decision_elapsed,
                        CollectionDecisionFailureV3::Infrastructure,
                    )?;
                    return Err(invalid(
                        "original whole-game canceled during actual decision",
                    ));
                }
                match decision {
                    Ok(mut turn) => {
                        let expected = if (side == Color::White) == assignment.engine_a_white {
                            epochs[0]
                        } else {
                            epochs[1]
                        };
                        let captured = (|| -> Result<(), ArenaError> {
                            check(
                                turn.root == exact_rules_anchor(&before)?
                                    && turn.epoch == expected.0,
                                "actual engine turn belongs to another exact Rules root",
                            )?;
                            capture.complete_turn(&mut turn)?;
                            if let Some(observer) = &self.saved_root_observer {
                                observer(GenerationSavedActionRootV3::from_actual_turn(
                                    assignment, &before, &turn,
                                )?)?;
                            }
                            self.flush_root(&mut capture, &mut deferred, until)
                        })();
                        if let Err(error) = captured {
                            pgn.record_failed_decision(
                                &before,
                                decision_elapsed,
                                CollectionDecisionFailureV3::Infrastructure,
                            )?;
                            return Err(error);
                        }
                        let Some(movement) = turn.best_move else {
                            decision_failure =
                                Some("ongoing actual engine returned no move".into());
                            pgn.record_failed_decision(
                                &before,
                                decision_elapsed,
                                CollectionDecisionFailureV3::EngineCrash,
                            )?;
                            termination = pgn
                                .observed_failure()
                                .ok_or_else(|| invalid("missing failed decision ledger"))?;
                            result = winner_for_loss(side);
                            break;
                        };
                        decisions.push(GenerationGameDecisionTraceV3 {
                            movement: Move16::pack(movement)
                                .map_err(|e| invalid(e.to_string()))?
                                .bits(),
                            elapsed_ns: u64::try_from(decision_elapsed.as_nanos())
                                .map_err(|_| invalid("actual decision trace duration overflow"))?,
                        });
                        match pgn.push_actual_move(&before, movement, decision_elapsed) {
                            Ok(()) => trajectory.play_pals_move(movement)?,
                            Err(error) => {
                                if let Some(failure) = pgn.observed_failure() {
                                    termination = failure;
                                    result = winner_for_loss(side);
                                    break;
                                }
                                return Err(error);
                            }
                        }
                    }
                    Err(error) => {
                        decision_failure = Some(error.to_string());
                        pgn.record_failed_decision(
                            &before,
                            decision_elapsed,
                            CollectionDecisionFailureV3::Infrastructure,
                        )?;
                        termination = pgn
                            .observed_failure()
                            .ok_or_else(|| invalid("failed actual decision ledger"))?;
                        let failure = serde_json::json!({"schema":"rz-pals-generation-game-failure-evidence/1","game_id":assignment.game_id,
                        "opening_id":assignment.opening_id,"stage":"actual_decision","error":error.to_string()});
                        self.submit(GenerationWriteCommandV3::Evidence(failure), until)?;
                        break;
                    }
                }
            }
            // A final legal move can be terminal at the exact ply bound.
            if rules_authority.is_none() && termination == CollectionPgnTerminationV3::PlyLimit {
                if matches!(
                    trajectory
                        .position()
                        .classify_position()
                        .map_err(|e| invalid(e.to_string()))?
                        .play_status,
                    rz_position::PlayStatus::Terminal { .. }
                ) {
                    let (authority, receipt) = trajectory.terminate_rules()?;
                    result = match authority.outcome {
                        GenerationOutcomeV3::WhiteWin => CollectionPgnResultV3::WhiteWin,
                        GenerationOutcomeV3::BlackWin => CollectionPgnResultV3::BlackWin,
                        GenerationOutcomeV3::Draw => CollectionPgnResultV3::Draw,
                    };
                    termination = CollectionPgnTerminationV3::RulesAutomatic;
                    rules_authority = Some(authority);
                    rules_receipt = Some(receipt);
                }
            }
            Ok(())
        })();
        let close_until = Instant::now()
            .checked_add(Duration::from_secs(30))
            .ok_or_else(|| invalid("collection cleanup deadline overflow"))?
            .min(window_cleanup_until);
        let close_result = game.close(close_until, &mut capture);
        // Any encoding/archive/rules/window failure still closes both real
        // session owners before it propagates. Unknown close errors above
        // stop the run and retain the owner's existing quarantine state.
        let close = match (played, close_result) {
            (Ok(()), Ok(close)) => close,
            (played, closed) => {
                let reason = match (played, closed) {
                    (Err(error), Ok(_)) => error.to_string(),
                    (Ok(()), Err(error)) => format!("physical game close: {error}"),
                    (Err(error), Err(close)) => {
                        format!("{error}; physical game close: {close}")
                    }
                    (Ok(()), Ok(_)) => unreachable!(),
                };
                // Raw moves/clocks are preserved even when closure is
                // unknown. This does not grant Rules WDL or normal finish.
                let _ = capture.poll();
                let _ = self.flush_root(&mut capture, &mut deferred, close_until);
                self.retain_unsubmitted_root(&mut capture);
                let clock_audit = pgn.clock_audit()?;
                let text = pgn.abort_infrastructure()?;
                let raw = serde_json::json!({"schema":"rz-pals-generation-game-pgn-evidence/1","game_id":assignment.game_id,
                    "opening_id":assignment.opening_id,"engine_a_white":assignment.engine_a_white,"pgn":text,"clock_audit":clock_audit});
                let _ = self.submit(GenerationWriteCommandV3::Evidence(raw), close_until);
                let failure = serde_json::json!({"schema":"rz-pals-generation-game-failure-evidence/1","game_id":assignment.game_id,
                    "opening_id":assignment.opening_id,"stage":"actual_game_or_close","error":reason});
                let _ = self.submit(GenerationWriteCommandV3::Evidence(failure), close_until);
                self.halted = true;
                let preserved = self.preserve_partial(&reason, close_until);
                return Err(invalid(match preserved {
                    Ok(()) => {
                        format!("actual collection halted; partial archive sealed: {reason}")
                    }
                    Err(error) => {
                        format!("actual collection halted: {reason}; partial preservation: {error}")
                    }
                }));
            }
        };
        let clock_audit = pgn.clock_audit()?;
        let text = pgn.finish(result, termination)?;
        let raw = serde_json::json!({"schema":"rz-pals-generation-game-pgn-evidence/1","game_id":assignment.game_id,
            "opening_id":assignment.opening_id,"engine_a_white":assignment.engine_a_white,"pgn":text,"clock_audit":clock_audit});
        let sha = generation_digest("rz-pals-generation-game-pgn-evidence/1", &raw)?;
        let finalized = (|| -> Result<(), ArenaError> {
            self.submit(GenerationWriteCommandV3::Evidence(raw), close_until)?;
            capture.poll()?;
            check(
                capture.process_epochs.is_subset(&close.epochs)
                    && close.epochs.len() == 2
                    && capture.owners.iter().all(|o| o.join.pending_count() == 0),
                "actual game process closure scope",
            )?;
            self.flush_root(&mut capture, &mut deferred, close_until)?;
            let sequence = self
                .chronology
                .lock()
                .map_err(|_| invalid("collection clock poisoned"))?
                .next_event_sequence()?;
            if let (Some(authority), Some(receipt)) = (rules_authority.as_ref(), rules_receipt) {
                self.submit(GenerationWriteCommandV3::Evidence(receipt), close_until)?;
                self.submit(
                    GenerationWriteCommandV3::Trajectory(authority.clone()),
                    close_until,
                )?;
                for input in &deferred {
                    if let Some(receipt) = late_wdl_receipt(input, authority, sequence)? {
                        self.submit(GenerationWriteCommandV3::Evidence(receipt), close_until)?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = finalized {
            self.halted = true;
            self.retain_unsubmitted_root(&mut capture);
            self.preserve_partial(&error.to_string(), close_until)?;
            return Err(error);
        }
        let report = GenerationCollectionGameReportV3 {
            game_id: assignment.game_id.clone(),
            opening_id: assignment.opening_id.clone(),
            engine_a_white: assignment.engine_a_white,
            rules_terminated: rules_authority.is_some(),
            termination,
            recorded_inputs: self.input_seals.len() - persisted_before_game,
            pgn_receipt_sha256: sha,
        };
        self.actual_game_ids.push(assignment.game_id.clone());
        self.next_game += 1;
        if let Some(error) = decision_failure {
            // A failed real inference/admission is never followed by a
            // second GPU game under the old execution identity. The raw
            // failure and PGN remain in this partial owned archive.
            self.halted = true;
            self.preserve_partial(&error, close_until)?;
            return Err(invalid(format!(
                "actual collection stopped after preserved failed game: {error}"
            )));
        }
        // Unknown physics/closure is an error above, so it never silently
        // advances this source owner to the next game/model process.
        Ok(ExecutedGenerationGameV3 {
            report,
            close,
            trajectory,
            names,
            decisions,
        })
    }
    fn flush_root(
        &mut self,
        capture: &mut GenerationGameCaptureV3,
        deferred: &mut Vec<DeferredWdlInput>,
        until: Instant,
    ) -> Result<(), ArenaError> {
        while let Some((_, receipt)) = capture.evidence.pop_first() {
            self.submit(GenerationWriteCommandV3::Evidence(receipt), until)?;
        }
        while let Some((_, record)) = capture.records.first_key_value() {
            check(
                self.input_seals.len() < GENERATION_MAX_RECORDS
                    && !self.input_seals.contains(&record.input_sha256),
                "actual collection duplicate/input cap",
            )?;
            if record.input.role != GenerationRoleV3::Validator {
                check(
                    deferred.len() < MAX_GAME_LABEL_INDEX,
                    "bounded game future-WDL index",
                )?;
            }
            let (_, record) = capture.records.pop_first().expect("record peeked above");
            if record.input.role != GenerationRoleV3::Validator {
                deferred.push(DeferredWdlInput {
                    input_sha256: record.input_sha256.clone(),
                    game_id: record.input.game_id.clone(),
                    role: record.input.role,
                    question_id: record.input.question_id.clone(),
                    view_sha256: record.input.view_sha256.clone(),
                    anchor: record.input.rules_anchor.clone(),
                });
            }
            let seal = record.input_sha256.clone();
            self.submit(GenerationWriteCommandV3::Record(Box::new(record)), until)?;
            self.input_seals.insert(seal);
        }
        Ok(())
    }
    fn retain_unsubmitted_root(&mut self, capture: &mut GenerationGameCaptureV3) {
        self.unsubmitted.extend(
            std::mem::take(&mut capture.evidence)
                .into_values()
                .map(GenerationWriteCommandV3::Evidence),
        );
        self.unsubmitted.extend(
            std::mem::take(&mut capture.records)
                .into_values()
                .map(|record| GenerationWriteCommandV3::Record(Box::new(record))),
        );
    }
    /// Flush/readback and join are required before reporting raw preservation.
    /// Its domain differs from a complete generation and is trainer-ineligible.
    pub fn preserve_partial(&mut self, reason: &str, until: Instant) -> Result<(), ArenaError> {
        self.halted = true;
        if self.partial.is_some() {
            return Ok(());
        }
        let mut authority = self.producer.clone();
        authority.input_seals = self.input_seals.iter().cloned().collect();
        authority.actual_game_ids = self.actual_game_ids.clone();
        let submitted = self.submit(
            GenerationWriteCommandV3::PreservePartial {
                authority,
                splits: self.plan.opening_splits(),
                reason: reason.to_owned(),
            },
            until,
        );
        if let Err(error) = submitted {
            self.writer.cancel();
            self.await_archive_join(until)?;
            return Err(error);
        }
        let result = loop {
            match self.writer.try_event() {
                Some(GenerationWriterEventV3::PartialPreserved(partial)) => {
                    self.partial = Some(partial);
                    break Ok(());
                }
                Some(GenerationWriterEventV3::Failed(error)) => break Err(invalid(error)),
                Some(GenerationWriterEventV3::Ready) => {}
                Some(GenerationWriterEventV3::Finished { .. }) => {
                    break Err(invalid(
                        "normal generation publication cannot replace partial preservation",
                    ));
                }
                None => {
                    if Instant::now() >= until {
                        break Err(invalid("partial archive deadline; owner retained"));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        };
        if result.is_err() {
            self.writer.cancel();
        }
        self.await_archive_join(until)?;
        result
    }
    pub fn finish(
        &mut self,
        until: Instant,
    ) -> Result<(PathBuf, GenerationChunkManifestV3), ArenaError> {
        check(
            !self.halted
                && self.next_game == 128
                && self.reported_pairs == 64
                && self.actual_game_ids.len() == 128,
            "partial selfplay run cannot claim completed generation",
        )?;
        self.finish_registered_archive(until)
    }
    fn finish_registered_archive(
        &mut self,
        until: Instant,
    ) -> Result<(PathBuf, GenerationChunkManifestV3), ArenaError> {
        self.producer.input_seals = self.input_seals.iter().cloned().collect();
        self.producer.actual_game_ids = self.actual_game_ids.clone();
        if let Err(error) = self.submit(
            GenerationWriteCommandV3::Finish {
                authority: self.producer.clone(),
                splits: self.plan.opening_splits(),
            },
            until,
        ) {
            self.writer.cancel();
            self.await_archive_join(until)?;
            return Err(error);
        }
        let result = loop {
            match self.writer.try_event() {
                Some(GenerationWriterEventV3::Finished {
                    directory,
                    manifest,
                }) => break Ok((directory, manifest)),
                Some(GenerationWriterEventV3::Failed(error)) => break Err(invalid(error)),
                Some(GenerationWriterEventV3::Ready) => {
                    break Err(invalid("duplicate archive readiness"));
                }
                Some(GenerationWriterEventV3::PartialPreserved(_)) => {
                    break Err(invalid(
                        "partial archive cannot be reported as complete generation",
                    ));
                }
                None => {
                    if Instant::now() >= until {
                        break Err(invalid(
                            "collection final manifest deadline; owner retained",
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        };
        if result.is_err() {
            self.halted = true;
            self.writer.cancel();
        }
        self.await_archive_join(until)?;
        result
    }
}

pub struct GenerationActionContextReportV3 {
    pub group_sha256: String,
    pub unique_context_count: usize,
    pub duplicate_context: bool,
    pub action_members: usize,
    pub known_pairs: usize,
}
/// The generation quota counts semantic frozen groups, not raw rows, IDs or
/// action members. All raw duplicate outcomes remain in the pinned archive.
pub struct GenerationActionOutcomeCollectorV3 {
    plan: ActionOutcomeCollectionPlanV3,
    archive: GenerationSelfplayCollectorV3,
    attempted_contexts: u32,
    groups: BTreeSet<String>,
    active_group: Option<FrozenActionGroupV3>,
    rejected_member: Option<GenerationIsolatedActionV3>,
    ready: bool,
}
impl GenerationActionOutcomeCollectorV3 {
    pub fn new(
        plan: ActionOutcomeCollectionPlanV3,
        mut openings: SelfplayCollectionPlanV3,
        producer: GenerationProducerAuthorityV3,
        context_epoch: u64,
        directory: PathBuf,
        archive_deadline: Instant,
    ) -> Result<Self, ArenaError> {
        plan.validate()?;
        check(
            producer.model_sha256 == plan.frozen_pc_weights_sha256,
            "action corpus producer must pin actual frozen shared/P/C parameters",
        )?;
        openings.run_id = plan.run_id.clone();
        openings.maximum_output_bytes = plan.maximum_output_bytes;
        let archive = GenerationSelfplayCollectorV3::new(
            openings,
            producer,
            plan.frozen_pc_epoch,
            context_epoch,
            directory,
            archive_deadline,
        )?;
        Ok(Self {
            plan,
            archive,
            attempted_contexts: 0,
            groups: BTreeSet::new(),
            active_group: None,
            rejected_member: None,
            ready: false,
        })
    }
    pub fn attempted_contexts(&self) -> u32 {
        self.attempted_contexts
    }
    pub fn unique_contexts(&self) -> usize {
        self.groups.len()
    }
    pub fn partial_archive(&self) -> Option<&GenerationPartialArchiveV3> {
        self.archive.partial_archive()
    }
    pub fn retained_group(&self) -> Option<&FrozenActionGroupV3> {
        self.active_group.as_ref()
    }
    pub fn take_unsubmitted(&mut self) -> Vec<GenerationWriteCommandV3> {
        self.archive.take_unsubmitted()
    }
    pub fn shutdown_archive(&mut self, until: Instant) -> Result<(), ArenaError> {
        self.archive.shutdown_archive(until)
    }
    /// Preserve diagnostic raw rows after a source-probe/admission failure as
    /// well as after member failures. This cannot satisfy the context quota.
    pub fn preserve_partial(&mut self, reason: &str, until: Instant) -> Result<(), ArenaError> {
        if self.archive.partial_archive().is_some() {
            return Ok(());
        }
        self.archive.halted = true;
        self.archive.preserve_partial(reason, until)
    }
    pub fn run_context<F: GenerationActionOutcomeFactoryV3>(
        &mut self,
        factory: &mut F,
        root: &GenerationSavedActionRootV3,
        until: Instant,
    ) -> Result<GenerationActionContextReportV3, ArenaError> {
        let window_cleanup_until = until
            .checked_add(Duration::from_secs(30))
            .ok_or_else(|| invalid("action original cleanup deadline overflow"))?;
        check(
            !self.archive.halted
                && self.active_group.is_none()
                && self.attempted_contexts < self.plan.maximum_contexts
                && self.groups.len() < self.plan.desired_contexts as usize
                && Instant::now() < until,
            "action collection context/window quota",
        )?;
        check(
            self.archive
                .plan
                .openings
                .iter()
                .any(|opening| opening.id == root.opening_id),
            "saved actual root opening is not preregistered",
        )?;
        if !self.ready {
            self.archive.wait_ready(until)?;
            self.ready = true;
        }
        self.attempted_contexts += 1;
        let registration = factory.prepare_context(root, &self.plan, until)?;
        if let Err(error) = registration.verify_collection_source(
            &self.plan,
            &self.archive.producer,
            root.position(),
        ) {
            let cleanup = factory.close_context(
                (Instant::now() + Duration::from_secs(30)).min(window_cleanup_until),
            );
            return Err(invalid(format!(
                "action source admission: {error}; actual context close: {}",
                match cleanup {
                    Ok(_) => "confirmed by actual owners".to_owned(),
                    Err(error) => error.to_string(),
                }
            )));
        }
        self.active_group = Some(FrozenActionGroupV3::begin(registration));
        let mut epochs = BTreeSet::new();
        let played = (|| -> Result<(), ArenaError> {
            while let Some(action) = self
                .active_group
                .as_ref()
                .and_then(FrozenActionGroupV3::next_action)
            {
                let member = factory.execute_isolated(
                    root,
                    self.active_group.as_ref().unwrap().registration(),
                    action,
                    self.archive.chronology.clone(),
                    until,
                )?;
                if member.record.input.game_id != root.game_id
                    || member.record.input.opening_id != root.opening_id
                {
                    self.rejected_member = Some(member);
                    return Err(invalid(
                        "actual isolated input changed its original selfplay root attribution",
                    ));
                }
                epochs.insert(member.root_authority.epoch.0);
                if let Err((error, owned)) = self.active_group.as_mut().unwrap().push_actual(member)
                {
                    self.rejected_member = Some(owned);
                    return Err(error);
                }
            }
            Ok(())
        })();
        let close_until = (Instant::now() + Duration::from_secs(30)).min(window_cleanup_until);
        let closed = factory.close_context(close_until);
        let checked = (|| -> Result<GenerationActualActionContextCloseV3, ArenaError> {
            played?;
            let closed = closed?;
            check(
                epochs.is_subset(&closed.epochs),
                "actual isolated action owners were not physically closed",
            )?;
            Ok(closed)
        })();
        if let Err(error) = checked {
            self.archive.halted = true;
            self.archive
                .preserve_partial(&error.to_string(), close_until)?;
            return Err(error);
        }
        let publication = self.active_group.as_mut().unwrap().finish(
            &mut *self
                .archive
                .chronology
                .lock()
                .map_err(|_| invalid("collection clock poisoned"))?,
        )?;
        let emitted = (|| -> Result<(), ArenaError> {
            for raw in publication.evidence.into_values() {
                self.archive
                    .submit(GenerationWriteCommandV3::Evidence(raw), close_until)?;
            }
            for record in publication.records {
                check(
                    self.archive.input_seals.len() < GENERATION_MAX_RECORDS
                        && !self.archive.input_seals.contains(&record.input_sha256),
                    "action corpus raw input cap/duplicate",
                )?;
                let seal = record.input_sha256.clone();
                self.archive.submit(
                    GenerationWriteCommandV3::Record(Box::new(record)),
                    close_until,
                )?;
                self.archive.input_seals.insert(seal);
            }
            Ok(())
        })();
        if let Err(error) = emitted {
            self.archive.halted = true;
            self.archive
                .preserve_partial(&error.to_string(), close_until)?;
            return Err(error);
        }
        self.active_group = None;
        if !self
            .archive
            .actual_game_ids
            .iter()
            .any(|game| game == &root.game_id)
        {
            self.archive.actual_game_ids.push(root.game_id.clone());
        }
        let duplicate_context = !self.groups.insert(publication.group_sha256.clone());
        Ok(GenerationActionContextReportV3 {
            group_sha256: publication.group_sha256,
            unique_context_count: self.groups.len(),
            duplicate_context,
            action_members: publication.member_count,
            known_pairs: publication.known_pair_count,
        })
    }
    pub fn finish(
        &mut self,
        until: Instant,
    ) -> Result<(PathBuf, GenerationChunkManifestV3), ArenaError> {
        check(
            !self.archive.halted
                && self.active_group.is_none()
                && self.groups.len() == self.plan.desired_contexts as usize
                && self.attempted_contexts <= self.plan.maximum_contexts,
            "InsufficientValidTargets: incomplete actual frozen action contexts",
        )?;
        self.archive.finish_registered_archive(until)
    }
}
fn winner_for_loss(loser: Color) -> CollectionPgnResultV3 {
    match loser {
        Color::White => CollectionPgnResultV3::BlackWin,
        Color::Black => CollectionPgnResultV3::WhiteWin,
    }
}
fn late_wdl_receipt(
    input: &DeferredWdlInput,
    actual: &GenerationTrajectoryAuthorityV3,
    observed: u64,
) -> Result<Option<serde_json::Value>, ArenaError> {
    actual.validate()?;
    check(
        input.game_id == actual.game_id,
        "late WDL actual game mismatch",
    )?;
    if input.role == GenerationRoleV3::Validator || !actual.anchors.contains(&input.anchor) {
        return Ok(None);
    }
    let target = GenerationWdlTargetV3 {
        observed_sequence: observed,
        question_id: input.question_id.clone(),
        view_sha256: input.view_sha256.clone(),
        raw_evidence_sha256: "0".repeat(64),
        source: "rules_selfplay_trajectory".into(),
        game_id: input.game_id.clone(),
        anchor: input.anchor.clone(),
        trajectory_membership_sha256: actual.sha256()?,
        rules_result_sha256: actual.result_evidence_sha256.clone(),
        ending: actual.ending,
        outcome: actual.outcome,
    };
    let mut target = serde_json::to_value(target).map_err(|e| invalid(e.to_string()))?;
    target
        .as_object_mut()
        .ok_or_else(|| invalid("late WDL target shape"))?
        .remove("raw_evidence_sha256");
    Ok(Some(
        serde_json::json!({"schema":"rz-pals-generation-head-evidence/1","input_sha256":input.input_sha256,"head":"wdl","target":target}),
    ))
}
