//! Bounded actual-completion authority for asynchronous V3 acknowledgements.
//! A later submission never replaces another completed ticket. Only the
//! manager's matching acceptance callback promotes a physically complete
//! final latent; no public constructor accepts an arbitrary caller seed.
use super::*;
use rz_contracts::pals::{Role, SearchAuthority};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::RequestId;
use std::collections::VecDeque;

const CAPACITY: usize = 16;
#[derive(Clone, Debug)]
pub struct VirtualReviewCompletion {
    pub raw: PalsRawOutput,
    pub completed_nn_inputs: u64,
    pub warm_started: bool,
    /// Actual physical invocation domain. The ticket representation seals the
    /// immutable source input; accepted ApproxWarm additionally seals its seed.
    pub invocation_key: [u8; 32],
    pub accepted_seed_digest: Option<[u8; 32]>,
    owner: u64,
    ticket: RoleTicketEnvelope,
    raw_digest: [u8; 32],
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SeedKey {
    authority: SearchAuthority,
    policy: Digest,
    context: model::ReviewWarmContextKey,
}
struct Completed {
    ticket: RoleTicketEnvelope,
    input: [u8; 32],
    raw: [u8; 32],
    seed_key: Option<SeedKey>,
    invocation_key: [u8; 32],
    accepted_seed_digest: Option<[u8; 32]>,
}
struct AcceptedSeed {
    key: SeedKey,
    latent: Vec<f32>,
}
pub(super) struct TicketSeeds {
    owner: u64,
    authority: Option<SearchAuthority>,
    // Closing a physically idle root revokes its authority even when a caller
    // presents a fresh request sequence. Game/root teardown never reopens it.
    revoked_authority: Option<SearchAuthority>,
    /// This physical owner accepts one monotonically increasing ticket stream.
    /// Retiring a completion, replacing a root or resetting a game cannot make
    /// a spent RequestId executable again. A new process epoch needs a new owner.
    last_admitted: Option<RequestId>,
    completed: BTreeMap<RequestId, Completed>,
    seeds: VecDeque<AcceptedSeed>,
}
impl TicketSeeds {
    pub(super) fn new() -> Result<Self, BackendError> {
        static OWNERS: AtomicU64 = AtomicU64::new(1);
        let owner = OWNERS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| {
                fail(
                    K::ResourceExhausted,
                    S::Admission,
                    "V3 seed owner namespace exhausted",
                )
            })?;
        Ok(Self {
            owner,
            authority: None,
            revoked_authority: None,
            last_admitted: None,
            completed: BTreeMap::new(),
            seeds: VecDeque::new(),
        })
    }
    pub(super) fn clear(&mut self) {
        if let Some(authority) = self.authority {
            self.revoked_authority = Some(authority);
        }
        self.authority = None;
        self.completed.clear();
        self.seeds.clear();
    }
    pub(super) fn admit(
        &mut self,
        ticket: &RoleTicketEnvelope,
        invocation: &VirtualReviewInvocation,
    ) -> Result<(), BackendError> {
        let input = &invocation.input;
        let admitted = input.validate().map_err(model_input)?;
        let role = match input.role {
            PalsRole::Proposer => Role::Proposer,
            PalsRole::Critic => Role::Critic,
            PalsRole::Validator => Role::Validator,
        };
        if ticket.view.actual_tokens as usize != admitted.token_count
            || ticket.view.selected_records as usize != input.records.len()
            || ticket.view.anchors as usize != input.anchors.len()
            || ticket.view.length_bucket as usize != admitted.length_bucket
            || ticket.role != role
            || ticket.representation.role != role
            || ticket.id.epoch != ticket.authority.epoch
            || ticket.view.authority != ticket.authority
            || ticket.representation.history.0 != invocation.request_history
            || ticket.representation.model.0 != invocation.identity.model_epoch
            || ticket.representation.precision != PrecisionProfile::Fp32
            || ticket.representation.input.0
                != input
                    .fresh_key(&invocation.identity)
                    .map_err(model_input)?
                    .0
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 ticket and immutable submitted tensors differ",
            ));
        }
        ticket
            .view
            .validate(ticket.authority, input.token_budget)
            .map_err(model_input)?;
        if ticket.id.sequence == 0
            || self.last_admitted.is_some_and(|last| {
                ticket.id.epoch != last.epoch || ticket.id.sequence <= last.sequence
            })
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 physical owner rejects reused or decreasing ticket sequence",
            ));
        }
        if self.revoked_authority.is_some_and(|closed| {
            ticket.authority.epoch != closed.epoch
                || ticket.authority.game.0 < closed.game.0
                || (ticket.authority.game == closed.game
                    && ticket.authority.root.0 <= closed.root.0)
        }) {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 quiesced root authority cannot be reopened by a new ticket sequence",
            ));
        }
        if self.authority != Some(ticket.authority) {
            // Manager root replacement revokes old pending results/seeds. The
            // caller must not replay an older authority as a new live root.
            if let Some(old) = self.authority {
                if old.epoch != ticket.authority.epoch
                    || ticket.authority.game.0 < old.game.0
                    || (ticket.authority.game == old.game && ticket.authority.root.0 < old.root.0)
                    || (ticket.authority.game == old.game && ticket.authority.root == old.root)
                {
                    return Err(fail(
                        K::IdentityMismatch,
                        S::Admission,
                        "V3 stale root cannot restore accepted seed authority",
                    ));
                }
            }
            self.clear();
            self.authority = Some(ticket.authority);
        }
        if self.completed.contains_key(&ticket.id) {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "duplicate V3 ticket execution",
            ));
        }
        if self.completed.len() >= CAPACITY {
            return Err(fail(
                K::ResourceExhausted,
                S::Admission,
                "V3 completed ticket ledger full",
            ));
        }
        self.last_admitted = Some(ticket.id);
        Ok(())
    }
    fn seed_key(
        ticket: &RoleTicketEnvelope,
        invocation: &VirtualReviewInvocation,
    ) -> Result<Option<SeedKey>, BackendError> {
        if invocation.input.role == PalsRole::Validator
            || invocation.input.query.purpose == model::ReviewQueryPurpose::ValueFresh
        {
            return Ok(None);
        }
        let context = invocation
            .input
            .warm_context_key(&invocation.identity)
            .map_err(model_input)?;
        Ok(Some(SeedKey {
            authority: ticket.authority,
            policy: ticket.view.context_policy,
            // A submitted question digest seals sequence and immutable view,
            // so it changes when an allowed public record revision changes.
            // The typed model context already seals exact role/query/candidates,
            // model/backend/precision and every ordered Rules/history anchor.
            // Completion/acceptance still use the entire original ticket.
            context,
        }))
    }
    pub(super) fn seed(
        &self,
        ticket: &RoleTicketEnvelope,
        invocation: &VirtualReviewInvocation,
    ) -> Result<Option<Vec<f32>>, BackendError> {
        let key = Self::seed_key(ticket, invocation)?;
        Ok(key.and_then(|key| {
            self.seeds
                .iter()
                .find(|seed| seed.key == key)
                .map(|seed| seed.latent.clone())
        }))
    }
    pub(super) fn completed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        invocation: &VirtualReviewInvocation,
        raw: PalsRawOutput,
        completed_nn_inputs: u64,
        accepted_seed_digest: Option<[u8; 32]>,
    ) -> Result<VirtualReviewCompletion, BackendError> {
        let input = invocation
            .input
            .fresh_key(&invocation.identity)
            .map_err(model_input)?
            .0;
        let raw_digest = raw_digest(&raw);
        let seed_key = Self::seed_key(ticket, invocation)?;
        if self.authority != Some(ticket.authority)
            || self.completed.contains_key(&ticket.id)
            || self.completed.len() >= CAPACITY
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Output,
                "V3 duplicate, revoked or unreserved completed execution",
            ));
        }
        let invocation_key = match accepted_seed_digest {
            Some(seed) => {
                invocation
                    .input
                    .approximate_warm_key(&invocation.identity, seed)
                    .map_err(model_input)?
                    .0
            }
            None => input,
        };
        self.completed.insert(
            ticket.id,
            Completed {
                ticket: *ticket,
                input,
                raw: raw_digest,
                seed_key,
                invocation_key,
                accepted_seed_digest,
            },
        );
        Ok(VirtualReviewCompletion {
            raw,
            completed_nn_inputs,
            warm_started: accepted_seed_digest.is_some(),
            invocation_key,
            accepted_seed_digest,
            owner: self.owner,
            ticket: *ticket,
            raw_digest,
        })
    }
    pub(super) fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        invocation: &VirtualReviewInvocation,
        completion: &VirtualReviewCompletion,
        enable_seed: bool,
    ) -> Result<(), BackendError> {
        let stored = self.completed.get(&ticket.id).ok_or_else(|| {
            fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 acceptance has no original completed ticket",
            )
        })?;
        let input = invocation
            .input
            .fresh_key(&invocation.identity)
            .map_err(model_input)?
            .0;
        if completion.owner != self.owner
            || completion.ticket != *ticket
            || stored.ticket != *ticket
            || self.authority != Some(ticket.authority)
            || stored.input != input
            || ticket.representation.history.0 != invocation.request_history
            || completion.raw_digest != stored.raw
            || raw_digest(&completion.raw) != stored.raw
            || completion.invocation_key != stored.invocation_key
            || completion.accepted_seed_digest != stored.accepted_seed_digest
            || completion.warm_started != stored.accepted_seed_digest.is_some()
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 acceptance differs from physical completion owner/ticket/input/output",
            ));
        }
        let key = stored.seed_key;
        if enable_seed {
            if let Some(key) = key {
                if completion.raw.private_latent.len() != 16 * 384
                    || completion.raw.private_latent.iter().any(|v| !v.is_finite())
                {
                    return Err(fail(
                        K::NumericalFailure,
                        S::Output,
                        "V3 accepted seed must be finite final latent",
                    ));
                }
                if let Some(index) = self.seeds.iter().position(|seed| seed.key == key) {
                    self.seeds.remove(index);
                }
                while self.seeds.len() >= CAPACITY {
                    self.seeds.pop_front();
                }
                self.seeds.push_back(AcceptedSeed {
                    key,
                    latent: completion.raw.private_latent.clone(),
                });
            }
        }
        self.completed.remove(&ticket.id);
        Ok(())
    }
    pub(super) fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), BackendError> {
        match self.completed.get(&ticket.id) {
            Some(stored) if stored.ticket == *ticket => {
                self.completed.remove(&ticket.id);
                Ok(())
            }
            _ => Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "V3 discard has no exact completed ticket",
            )),
        }
    }
    pub(super) fn owned_bytes(&self) -> u64 {
        // Exact Rust-owned accepted latent and VecDeque backing capacities.
        // BTreeMap node allocations/native metadata are not inferred from
        // element counts and are intentionally outside this known measure.
        self.seeds
            .iter()
            .map(|seed| seed.latent.capacity() as u64 * 4)
            .sum::<u64>()
            + (self.seeds.capacity() * std::mem::size_of::<AcceptedSeed>()) as u64
    }
}
pub(super) fn latent_digest(values: &[f32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-review-accepted-final-latent/1");
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_le_bytes());
    }
    hash.finalize().into()
}
fn raw_digest(output: &PalsRawOutput) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-review-native-completion-output/1");
    for values in [
        Some(output.candidate_logits.as_slice()),
        Some(output.wdl_logits.as_slice()),
        Some(output.private_latent.as_slice()),
        output.divergence_logits.as_deref(),
        output.task_logits.as_ref().map(|v| v.as_slice()),
    ] {
        if let Some(values) = values {
            hash.update([1]);
            hash.update((values.len() as u64).to_le_bytes());
            for v in values {
                hash.update(v.to_le_bytes());
            }
        } else {
            hash.update([0]);
        }
    }
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals_model::PalsCandidateToken;
    use model::{
        ReviewAnchorInput, ReviewQueryInput, ReviewQueryPurpose, ReviewRecordInput,
        ReviewRecordKind,
    };
    use rz_contracts::pals::RepresentationKey;
    use rz_contracts::pals_adaptive::ReviewViewSeal;
    use rz_contracts::{
        ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch,
        RootGeneration, StateIdentity, StateRevision,
    };

    fn invocation() -> VirtualReviewInvocation {
        let movement = PalsCandidateToken {
            from: 12,
            to: 28,
            promotion: 0,
        };
        VirtualReviewInvocation {
            input: PalsVirtualReviewInput {
                schema: model::INPUT_SCHEMA.into(),
                role: PalsRole::Proposer,
                anchors: vec![ReviewAnchorInput {
                    board: vec![0; 64],
                    metadata: [0.; 16],
                }],
                records: vec![],
                query: ReviewQueryInput {
                    purpose: ReviewQueryPurpose::ProposePolicy,
                    perspective: 1,
                    prefix: vec![movement],
                    proposal: vec![],
                    counter: vec![],
                    counter_present: false,
                    anchors: [0; 3],
                },
                candidates: vec![movement],
                divergences: vec![],
                token_budget: 512,
            },
            identity: ReviewExecutionIdentity {
                model_epoch: [1; 32],
                rules_history: [2; 32],
                backend: [3; 32],
                anchor_history_seals: vec![[2; 32]],
                fp32: true,
            },
            request_history: [9; 32],
        }
    }
    fn ticket(invocation: &VirtualReviewInvocation, sequence: u64) -> RoleTicketEnvelope {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([4; 32]),
        };
        let admitted = invocation.input.validate().unwrap();
        let role = match invocation.input.role {
            PalsRole::Proposer => Role::Proposer,
            PalsRole::Critic => Role::Critic,
            PalsRole::Validator => Role::Validator,
        };
        RoleTicketEnvelope {
            id: RequestId::new(authority.epoch, sequence),
            authority,
            question: Digest([5; 32]),
            role,
            original_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(5000),
            },
            view: ReviewViewSeal {
                authority,
                view_sequence: sequence,
                context_policy: Digest([6; 32]),
                view_digest: Digest([7; 32]),
                root_state: StateIdentity {
                    owner: OwnerId(1),
                    revision: StateRevision(1),
                    semantic: Digest([8; 32]),
                },
                actual_tokens: admitted.token_count as u16,
                selected_records: invocation.input.records.len() as u16,
                anchors: invocation.input.anchors.len() as u8,
                length_bucket: admitted.length_bucket as u16,
            },
            representation: RepresentationKey {
                input: Digest(invocation.input.fresh_key(&invocation.identity).unwrap().0),
                history: Digest(invocation.request_history),
                candidates: Digest([10; 32]),
                mask_positions: Digest([11; 32]),
                model: Digest(invocation.identity.model_epoch),
                encoding: Digest([12; 32]),
                precision: PrecisionProfile::Fp32,
                frozen_epoch: 0,
                record_revision: sequence,
                role,
            },
        }
    }
    fn raw(value: f32) -> PalsRawOutput {
        PalsRawOutput {
            candidate_logits: vec![value],
            wdl_logits: [value; 3],
            private_latent: vec![value; 6144],
            divergence_logits: None,
            task_logits: None,
        }
    }
    #[test]
    fn prefix_endpoint_history_is_distinct_and_pending_completions_are_independent() {
        let invocation = invocation();
        let first = ticket(&invocation, 1);
        let second = ticket(&invocation, 2);
        assert_ne!(
            first.representation.history.0,
            invocation.identity.rules_history
        );
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &invocation).unwrap();
        let completion = seeds
            .completed(&first, &invocation, raw(1.), 2, None)
            .unwrap();
        assert!(seeds.seed(&second, &invocation).unwrap().is_none());
        seeds.admit(&second, &invocation).unwrap();
        let second_completion = seeds
            .completed(&second, &invocation, raw(2.), 1, None)
            .unwrap();
        seeds
            .accept(&first, &invocation, &completion, true)
            .unwrap();
        assert_eq!(
            seeds.seed(&second, &invocation).unwrap().unwrap(),
            vec![1.; 6144]
        );
        assert_eq!(seeds.completed.len(), 1);
        seeds
            .accept(&second, &invocation, &second_completion, false)
            .unwrap();
        assert!(seeds.completed.is_empty());
        assert!(seeds
            .accept(&first, &invocation, &completion, true)
            .is_err());
    }
    #[test]
    fn duplicate_output_does_not_replace_original_and_modified_completion_cannot_seed() {
        let invocation = invocation();
        let ticket = ticket(&invocation, 1);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&ticket, &invocation).unwrap();
        let completion = seeds
            .completed(&ticket, &invocation, raw(1.), 2, None)
            .unwrap();
        assert!(seeds
            .completed(&ticket, &invocation, raw(2.), 2, None)
            .is_err());
        let mut forged = completion.clone();
        forged.raw.private_latent[0] = 3.;
        assert!(seeds.accept(&ticket, &invocation, &forged, true).is_err());
        assert_eq!(seeds.completed.len(), 1);
        seeds
            .accept(&ticket, &invocation, &completion, true)
            .unwrap();
    }
    #[test]
    fn actual_warm_invocation_seals_the_accepted_seed_separately() {
        let invocation = invocation();
        let first = ticket(&invocation, 1);
        let second = ticket(&invocation, 2);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &invocation).unwrap();
        let first_completion = seeds
            .completed(&first, &invocation, raw(1.), 2, None)
            .unwrap();
        seeds
            .accept(&first, &invocation, &first_completion, true)
            .unwrap();
        let seed = seeds.seed(&second, &invocation).unwrap().unwrap();
        let digest = latent_digest(&seed);
        seeds.admit(&second, &invocation).unwrap();
        let warm = seeds
            .completed(&second, &invocation, raw(2.), 1, Some(digest))
            .unwrap();
        assert!(warm.warm_started);
        assert_eq!(warm.accepted_seed_digest, Some(digest));
        assert_ne!(warm.invocation_key, second.representation.input.0);
        assert_eq!(
            warm.invocation_key,
            invocation
                .input
                .approximate_warm_key(&invocation.identity, digest)
                .unwrap()
                .0
        );
        let mut forged = warm.clone();
        forged.accepted_seed_digest = Some([99; 32]);
        assert!(seeds.accept(&second, &invocation, &forged, true).is_err());
        seeds.accept(&second, &invocation, &warm, true).unwrap();
    }
    #[test]
    fn root_replacement_revokes_old_completion_and_same_root_authority_cannot_change() {
        let invocation = invocation();
        let first = ticket(&invocation, 1);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &invocation).unwrap();
        let completion = seeds
            .completed(&first, &invocation, raw(1.), 2, None)
            .unwrap();
        let mut mismatched = ticket(&invocation, 2);
        mismatched.authority.implementation = Digest([99; 32]);
        mismatched.view.authority = mismatched.authority;
        assert!(seeds.admit(&mismatched, &invocation).is_err());
        let mut next = ticket(&invocation, 3);
        next.authority.root = RootGeneration(2);
        next.view.authority = next.authority;
        seeds.admit(&next, &invocation).unwrap();
        assert!(seeds
            .accept(&first, &invocation, &completion, true)
            .is_err());
        assert!(seeds.admit(&first, &invocation).is_err());
        assert!(seeds.completed.is_empty());
    }
    #[test]
    fn full_completed_ledger_rejects_admission_until_exact_discard() {
        let invocation = invocation();
        let mut seeds = TicketSeeds::new().unwrap();
        for sequence in 1..=16 {
            let ticket = ticket(&invocation, sequence);
            seeds.admit(&ticket, &invocation).unwrap();
            seeds
                .completed(&ticket, &invocation, raw(1.), 2, None)
                .unwrap();
        }
        let next = ticket(&invocation, 17);
        assert!(seeds.admit(&next, &invocation).is_err());
        seeds.discard(&ticket(&invocation, 1)).unwrap();
        seeds.admit(&next, &invocation).unwrap();
        assert!(seeds.discard(&ticket(&invocation, 1)).is_err());
    }
    #[test]
    fn retiring_or_resetting_a_ticket_does_not_grant_reexecution() {
        let invocation = invocation();
        let first = ticket(&invocation, 1);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &invocation).unwrap();
        let completion = seeds
            .completed(&first, &invocation, raw(1.), 2, None)
            .unwrap();
        seeds
            .accept(&first, &invocation, &completion, false)
            .unwrap();
        assert!(seeds.admit(&first, &invocation).is_err());
        seeds.clear();
        assert!(seeds.admit(&first, &invocation).is_err());
        let mut next = ticket(&invocation, 2);
        next.authority.game = GameGeneration(2);
        next.view.authority = next.authority;
        seeds.admit(&next, &invocation).unwrap();
    }
    #[test]
    fn idle_root_close_revokes_completions_and_seed_without_resetting_request_floor() {
        let input = invocation();
        let first = ticket(&input, 1);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &input).unwrap();
        let done = seeds.completed(&first, &input, raw(1.), 2, None).unwrap();
        seeds.accept(&first, &input, &done, true).unwrap();
        seeds.clear();
        assert!(seeds.completed.is_empty());
        assert!(seeds.seeds.is_empty());
        assert!(seeds.seed(&first, &input).unwrap().is_none());
        let mut same = ticket(&input, 2);
        assert!(seeds.admit(&same, &input).is_err());
        same.authority.root = rz_contracts::RootGeneration(same.authority.root.0 + 1);
        same.view.authority = same.authority;
        seeds.admit(&same, &input).unwrap();
        assert!(seeds.admit(&first, &input).is_err());
    }
    #[test]
    fn present_empty_counter_cannot_reuse_absent_counter_seed_context() {
        let source = invocation();
        let mut present = source.clone();
        present.input.query.counter_present = true;
        assert!(present.input.query.counter.is_empty());
        assert_ne!(
            source.input.fresh_key(&source.identity).unwrap(),
            present.input.fresh_key(&present.identity).unwrap()
        );
        assert_ne!(
            source.input.warm_context_key(&source.identity).unwrap(),
            present.input.warm_context_key(&present.identity).unwrap()
        );
        let first = ticket(&source, 1);
        let second = ticket(&present, 2);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &source).unwrap();
        let completion = seeds.completed(&first, &source, raw(1.), 2, None).unwrap();
        seeds.accept(&first, &source, &completion, true).unwrap();
        assert!(seeds.seed(&second, &present).unwrap().is_none());
    }
    #[test]
    fn revised_public_view_and_submission_question_reuse_only_accepted_exact_context() {
        let source = invocation();
        let first = ticket(&source, 1);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &source).unwrap();
        let completion = seeds.completed(&first, &source, raw(1.), 2, None).unwrap();
        seeds.accept(&first, &source, &completion, true).unwrap();
        let mut revised = source.clone();
        revised.input.records.push(ReviewRecordInput {
            kind: ReviewRecordKind::Counterexample,
            perspective: 1,
            completed_iteration: true,
            rules_terminal: false,
            line: vec![PalsCandidateToken {
                from: 52,
                to: 36,
                promotion: 0,
            }],
            anchor: 0,
            parent: None,
            supersedes: None,
        });
        let mut second = ticket(&revised, 2);
        second.question = Digest([88; 32]);
        second.view.view_digest = Digest([89; 32]);
        assert_ne!(first.question, second.question);
        assert_ne!(first.view, second.view);
        assert_ne!(first.representation.input, second.representation.input);
        seeds.admit(&second, &revised).unwrap();
        let seed = seeds.seed(&second, &revised).unwrap().unwrap();
        assert_eq!(seed, vec![1.; 6144]);
        assert!(seeds.accept(&second, &revised, &completion, true).is_err());
        let next = seeds
            .completed(&second, &revised, raw(2.), 1, Some(latent_digest(&seed)))
            .unwrap();
        assert!(next.warm_started);
        seeds.accept(&second, &revised, &next, true).unwrap();
        assert!(seeds.accept(&second, &revised, &next, true).is_err());
    }
    #[test]
    fn accepted_context_cannot_cross_query_role_history_epoch_authority_or_policy() {
        let source = invocation();
        let first = ticket(&source, 1);
        let mut seeds = TicketSeeds::new().unwrap();
        seeds.admit(&first, &source).unwrap();
        let completion = seeds.completed(&first, &source, raw(1.), 2, None).unwrap();
        seeds.accept(&first, &source, &completion, true).unwrap();
        let mut changed = source.clone();
        changed.input.query.prefix.push(PalsCandidateToken {
            from: 52,
            to: 36,
            promotion: 0,
        });
        assert!(seeds
            .seed(&ticket(&changed, 2), &changed)
            .unwrap()
            .is_none());
        changed = source.clone();
        changed.input.role = PalsRole::Critic;
        assert!(seeds
            .seed(&ticket(&changed, 2), &changed)
            .unwrap()
            .is_none());
        changed = source.clone();
        changed.identity.rules_history = [22; 32];
        changed.identity.anchor_history_seals[0] = [22; 32];
        assert!(seeds
            .seed(&ticket(&changed, 2), &changed)
            .unwrap()
            .is_none());
        changed = source.clone();
        changed.identity.model_epoch = [23; 32];
        assert!(seeds
            .seed(&ticket(&changed, 2), &changed)
            .unwrap()
            .is_none());
        changed = source.clone();
        changed.identity.backend = [24; 32];
        assert!(seeds
            .seed(&ticket(&changed, 2), &changed)
            .unwrap()
            .is_none());
        for axis in 0..3 {
            let mut other = ticket(&source, 2);
            match axis {
                0 => other.authority.root = RootGeneration(2),
                1 => other.authority.game = GameGeneration(2),
                _ => {
                    other.authority.epoch = ProcessEpoch(2);
                    other.id.epoch = ProcessEpoch(2);
                }
            }
            other.view.authority = other.authority;
            assert!(seeds.seed(&other, &source).unwrap().is_none());
        }
        let mut other = ticket(&source, 2);
        other.view.context_policy = Digest([25; 32]);
        assert!(seeds.seed(&other, &source).unwrap().is_none());
        changed = source.clone();
        changed.identity.fp32 = false;
        assert!(seeds.seed(&ticket(&source, 2), &changed).is_err());
    }
}
