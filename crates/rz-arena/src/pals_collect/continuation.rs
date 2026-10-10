//! Evidence for the actual C continuation lane. It shares the existing native
//! call owner and reservations, but never emits the legacy single-Reply witness.
use super::*;

const REGISTRATION_VERSION: &str = "rz-pals-native-continuation-registration/1";
const DOMAIN: &str = "rz-pals-native-post-repair-continuation/2";
const ARTIFACT: &str = "native-continuation-traces.jsonl";
const OBSERVER_VERSION: &str = "pals-post-repair-continuation-observer/2";

/// An independently pinned, immutable registration for the separate C lane.
/// Registration permits observation; it never certifies execution or utility.
#[derive(Clone, Debug)]
pub struct PalsNativeContinuationRegistration {
    pub(super) inner: PalsNativeRefinementRegistration,
}
impl Serialize for PalsNativeContinuationRegistration {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(serializer)
    }
}
impl PalsNativeContinuationRegistration {
    pub fn read_pinned(path: &Path, expected_sha256: &str) -> Result<Self, ArenaError> {
        let bytes =
            PalsNativeRefinementRegistration::read_registration_file(path, expected_sha256)?;
        Self::from_registration_bytes(&bytes, expected_sha256)
    }
    pub fn from_registration_bytes(
        bytes: &[u8],
        expected_sha256: &str,
    ) -> Result<Self, ArenaError> {
        if bytes.is_empty()
            || bytes.len() > MAX_REFINEMENT_REGISTRATION_BYTES
            || !valid_sha(expected_sha256)
            || format!("{:x}", Sha256::digest(bytes)) != expected_sha256
        {
            return Err(invalid(
                "continuation registration differs from its independent raw pin",
            ));
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|_| invalid("continuation registration is not UTF-8"))?;
        let wire: RefinementRegistrationWire = crate::decode_json(text)?;
        if wire.version != REGISTRATION_VERSION
            || wire.search_policy
                != PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1()
            || !valid_sha(&wire.base_registry_canonical_sha256)
            || !valid_sha(&wire.collector_binary_sha256)
        {
            return Err(invalid("invalid closed native C continuation registration"));
        }
        Ok(Self {
            inner: PalsNativeRefinementRegistration {
                wire,
                raw_sha256: expected_sha256.into(),
            },
        })
    }
    pub fn validate_against(&self, base: &PalsNativeCollectionRegistry) -> Result<(), ArenaError> {
        self.inner.validate_against(base)
    }
    pub fn raw_sha256(&self) -> &str {
        self.inner.raw_sha256()
    }
}

pub(super) struct PendingContinuation {
    identity: RecheckIdentity,
    descriptor: String,
    prepared_payload: String,
    initial_context: RoleLogicalContext,
    examined_responses: Vec<BoardMove>,
    // Capacity is reserved before submission. Filled only by the first actual
    // physical raw result, never by the search's selected-response callback.
    initial_ranking: Vec<BoardMove>,
    repaired: Vec<BoardMove>,
    refutation: Vec<BoardMove>,
    limits: PalsLimits,
    deadline_tick: u64,
    parent_epoch: u64,
    parent_sequence: u64,
    calls: Vec<(RequestId, String)>,
}

impl PendingContinuation {
    pub(super) fn record_initial_ranking(
        &mut self,
        id: RequestId,
        legal: &[BoardMove],
        indices: &[usize],
    ) -> Result<(), RoleError> {
        if self.calls.first().map(|(first, _)| *first) != Some(id) {
            return Ok(());
        }
        if !self.initial_ranking.is_empty()
            || indices.len() != legal.len()
            || self.initial_ranking.capacity() < legal.len()
        {
            return Err(role_error(
                "initial continuation ranking lacks unique reserved raw completion",
            ));
        }
        self.initial_ranking
            .extend(indices.iter().map(|&i| legal[i]));
        Ok(())
    }

    fn expected_response(&self) -> Option<BoardMove> {
        let original = self.repaired[self.initial_context.prefix.len()];
        self.initial_ranking
            .iter()
            .copied()
            .find(|movement| *movement != original && !self.examined_responses.contains(movement))
            .or_else(|| {
                self.initial_ranking
                    .iter()
                    .copied()
                    .find(|movement| *movement != original)
            })
    }
}

impl Sink {
    fn continuation_selected(&self) -> bool {
        self.source
            .native
            .as_ref()
            .and_then(|n| n.get("pals_search_policy"))
            .is_some_and(|actual| {
                serde_json::to_value(
                    PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1(),
                )
                .is_ok_and(|expected| &expected == actual)
            })
    }
    fn continuation_row(
        &mut self,
        stage: &str,
        identity: RecheckIdentity,
        descriptor: &str,
        data: serde_json::Value,
    ) -> Result<String, RoleError> {
        if self.prepaid_recheck_rows == 0 {
            return Err(role_error("continuation has no reserved trace row"));
        }
        let identity = recheck_identity_json(identity);
        let payload =
            canonical_sha256(&(DOMAIN, stage, &identity, descriptor, &data)).map_err(role_error)?;
        let row = serde_json::json!({"domain":DOMAIN,"stage":stage,"game_id":self.context()?.game,
            "identity":identity,"descriptor_sha256":descriptor,"payload_sha256":payload,
            "observer_elapsed_us":self.started.elapsed().as_micros().min(u64::MAX as u128) as u64,"data":data});
        let json = bounded_json(&row, MAX_RECHECK_ROW_BYTES, false).map_err(role_error)?;
        self.prepaid_recheck_rows -= 1;
        self.trace.rows.push(PalsNativeTraceRow {
            artifact: ARTIFACT,
            json,
        });
        Ok(payload)
    }
    pub(super) fn prepare_continuation(
        &mut self,
        event: RecheckPrepared<'_>,
    ) -> Result<(), RoleError> {
        if self.pending_continuation.is_some()
            || self.pending_recheck.is_some()
            || self.rejected_recheck.is_some()
        {
            return Err(role_error("continuation overlaps an unclosed observation"));
        }
        self.rejected_recheck = Some(event.identity);
        if event.policy != PostRepairRecheckPolicy::ActualOpponentContinuationV1
            || !self.continuation_selected()
            || self.closed_rechecks.contains(&event.identity)
            || self.closed_rechecks.len() >= MAX_RECHECK_ATTEMPTS
        {
            return Err(role_error(
                "unselected, duplicate or unbounded continuation",
            ));
        }
        self.producer
            .as_ref()
            .ok_or_else(|| role_error("continuation requires a strict producer"))?
            .verify_source(&self.source)
            .map_err(role_error)?;
        event.config.validate().map_err(role_error)?;
        if event.cancel.load(Ordering::Acquire)
            || Instant::now() >= event.limits.deadline
            || event.repaired.len() > event.config.line_plies
            || event.refutation.len() > event.config.line_plies
            || event.anchor_ply == 0
            || event.anchor_ply >= event.repaired.len()
            || event.reply_context.purpose != RoleQueryPurpose::ReplyPolicy
            || event.reply_context.prefix.as_slice() != &event.repaired[..event.anchor_ply]
            || event.reply_context.game_generation != event.identity.game_generation
            || event.reply_context.search_generation != event.identity.search_generation
            || event.reply_context.state != event.anchor_state
            || event.reply_context.situation != event.anchor_situation
            || event.repair_record.kind != RecordKind::Repair
            || event.repair_record.revision != event.identity.repair_record_revision
            // Repair is historical. This reviewed engine's verification can
            // append one public record before Reply; it cannot advance further.
            || !(event.reply_context.public_revision == event.repair_record.revision
                || event.repair_record.revision.checked_add(1)
                    == Some(event.reply_context.public_revision))
            || event.repair_record.origin_state != event.root_state
            || event.repair_record.line.as_slice() != event.repaired
            || !matches!(
                event.checker_identity,
                rz_search::cpu_checker::CheckerIdentity::Owned(_)
            )
        {
            return Err(role_error(
                "continuation prepared controls, Repair or OwnCPU namespace differ",
            ));
        }
        let root = &self.context()?.position;
        if !event.root_snapshot.same_state(&root.snapshot()) {
            return Err(role_error("continuation root/history differs"));
        }
        let root_sha = state_sha(root).map_err(role_error)?;
        let (anchor, _) =
            replay_line(root, &event.repaired[..event.anchor_ply]).map_err(role_error)?;
        if !event.anchor_snapshot.same_state(&anchor.snapshot()) {
            return Err(role_error(
                "continuation anchor differs from exact Rules replay",
            ));
        }
        let legal = anchor.legal_moves();
        if event.examined_responses.len() > legal.len()
            || event
                .examined_responses
                .iter()
                .enumerate()
                .any(|(i, movement)| {
                    !legal.contains(movement) || event.examined_responses[..i].contains(movement)
                })
        {
            return Err(role_error(
                "continuation examined responses are not unique legal anchor edges",
            ));
        }
        let mut initial_ranking = Vec::new();
        initial_ranking
            .try_reserve_exact(legal.len())
            .map_err(role_error)?;
        let (repaired, _) = replay_line(root, event.repaired).map_err(role_error)?;
        let (parents, epoch, sequence) = self.parent_repair_chain(&event, &root_sha)?;
        let context = logical_json(event.reply_context)?;
        let data = serde_json::json!({"engine_observer_version":OBSERVER_VERSION,
            "checked_source_sha256":canonical_sha256(&("rz-pals-collector-checked-source/1",&self.source)).map_err(role_error)?,
            "refinement_registration_sha256":self.source.native.as_ref().and_then(|n|n.get("refinement_registration_sha256")),
            "policy":PalsSearchPolicyIdentityV3::expected_actual_opponent_continuation_v1(),
            "root_rules_state_sha256":root_sha,"anchor_rules_state_sha256":state_sha(&anchor).map_err(role_error)?,
            "anchor_ply":event.anchor_ply,"initial_reply_context":context,"parent_repair_chain":parents,
            "examined_responses":pack(event.examined_responses).map_err(role_error)?,
            "initial_selection_rule":"ranked_unexamined_different_else_ranked_different/1",
            "repaired":pack(event.repaired).map_err(role_error)?,"refutation":pack(event.refutation).map_err(role_error)?,
            "repaired_endpoint":recheck_endpoint_json(&event.repaired_endpoint,&repaired)?,
            "checker_identity":event.checker_identity,"cpu_condition":event.cpu_condition,
            "limits":{"max_rounds":event.limits.max_rounds,"max_cpu_nodes":event.limits.max_cpu_nodes,"cpu_depth":event.limits.cpu_depth},
            "engine_deadline_tick":event.deadline_tick,"engine_deadline_tick_origin":"engine_monotonic_clock_origin",
            "prepared_before_reply_submit":true,"maximum_reply_calls":event.repaired.len()-event.anchor_ply,
            "assurance":"multi-Reply observation; not a legacy witness, all-defenses proof or training target"});
        let descriptor = canonical_sha256(&(
            DOMAIN,
            "prepared-descriptor",
            recheck_identity_json(event.identity),
            &data,
        ))
        .map_err(role_error)?;
        let _ = bounded_json(&serde_json::json!({"domain":DOMAIN,"stage":"prepared","game_id":self.context()?.game,
            "identity":recheck_identity_json(event.identity),"descriptor_sha256":descriptor,"payload_sha256":"00".repeat(32),
            "observer_elapsed_us":u64::MAX,"data":data}),MAX_RECHECK_ROW_BYTES,false).map_err(role_error)?;
        let rows = event.repaired.len() - event.anchor_ply + 2; // prepared, each bound call, finished
        if self
            .trace
            .rows
            .len()
            .checked_add(self.prepaid_recheck_rows)
            .and_then(|n| n.checked_add(self.prepaid_native_rows))
            .and_then(|n| n.checked_add(rows))
            .is_none_or(|n| n > MAX_ROWS * 8)
        {
            return Err(role_error(
                "continuation trace row credit exhausted before dispatch",
            ));
        }
        self.reserve(
            (MAX_RECHECK_ROW_BYTES as u64 + 1)
                .checked_mul(rows as u64)
                .ok_or_else(|| role_error("continuation trace reservation overflow"))?,
        )?;
        self.prepaid_recheck_rows += rows;
        let prepared_payload =
            self.continuation_row("prepared", event.identity, &descriptor, data)?;
        self.pending_continuation = Some(PendingContinuation {
            identity: event.identity,
            descriptor,
            prepared_payload,
            initial_context: event.reply_context.clone(),
            examined_responses: event.examined_responses.to_vec(),
            initial_ranking,
            repaired: event.repaired.to_vec(),
            refutation: event.refutation.to_vec(),
            limits: event.limits,
            deadline_tick: event.deadline_tick,
            parent_epoch: epoch,
            parent_sequence: sequence,
            calls: Vec::new(),
        });
        self.rejected_recheck = None;
        Ok(())
    }
    fn accepted_continuation_call(&self, id: RequestId) -> Result<&CallLogical, RoleError> {
        let call = self
            .calls
            .get(&id)
            .ok_or_else(|| role_error("continuation call disappeared"))?;
        let meta = call
            .logical
            .as_ref()
            .ok_or_else(|| role_error("continuation call has no actual logical input"))?;
        if !call.physical
            || !call.delivered
            || !call.accepted
            || call.rejected
            || call.physical_unknown
            || !meta.accepted_context_checked
            || !meta.producer_metadata_admitted
            || meta.chosen_first.is_none()
        {
            return Err(role_error(
                "continuation lacks physical/delivery/consumption/context/producer evidence",
            ));
        }
        Ok(meta)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check_continuation_reply(
        &self,
        id: RequestId,
        kind: NativeQueryKind,
        position: &Position,
        prefix: &[BoardMove],
        proposal: &[BoardMove],
        counterexample: Option<&[BoardMove]>,
        logical: Option<&RoleLogicalContext>,
        until: Instant,
    ) -> Result<(), RoleError> {
        let pending = self
            .pending_continuation
            .as_ref()
            .ok_or_else(|| role_error("continuation pending missing"))?;
        let context =
            logical.ok_or_else(|| role_error("continuation actual logical context missing"))?;
        let start = pending.initial_context.prefix.len();
        if kind != NativeQueryKind::Reply
            || pending.calls.len() >= pending.repaired.len() - start
            || prefix.len() != start + pending.calls.len()
            || prefix[..start] != pending.repaired[..start]
            || proposal != pending.repaired.as_slice()
            || counterexample != Some(pending.refutation.as_slice())
            || id.epoch.0 != pending.parent_epoch
            || until != pending.limits.deadline
            || context.purpose != RoleQueryPurpose::ReplyPolicy
            || context.prefix.as_slice() != prefix
            || context.game_generation != pending.identity.game_generation
            || context.search_generation != pending.identity.search_generation
            || context.public_revision != pending.initial_context.public_revision
        {
            return Err(role_error(
                "actual continuation input differs from registered lane or current prefix",
            ));
        }
        let previous_sequence = if let Some((previous, _)) = pending.calls.last() {
            let previous_meta = self.accepted_continuation_call(*previous)?;
            if &prefix[..prefix.len() - 1] != previous_meta.prefix.as_slice()
                || (pending.calls.len() > 1 && prefix.last().copied() != previous_meta.chosen_first)
            {
                return Err(role_error(
                    "continuation prefix does not extend the preceding accepted C output",
                ));
            }
            if pending.calls.len() == 1
                && prefix.last().copied() != self.expected_initial_continuation_response()?
            {
                return Err(role_error(
                    "initial continuation alternative differs from actual raw ranking and examined anchor edges",
                ));
            }
            previous.sequence
        } else {
            if context != &pending.initial_context {
                return Err(role_error("initial continuation context differs"));
            }
            pending.parent_sequence
        };
        if id.sequence <= previous_sequence {
            return Err(role_error(
                "continuation RequestId is duplicate or reordered",
            ));
        }
        let (exact, _) = replay_line(&self.context()?.position, prefix).map_err(role_error)?;
        if !position.snapshot().same_state(&exact.snapshot()) {
            return Err(role_error(
                "continuation state/history differs from actual prefix",
            ));
        }
        Ok(())
    }
    fn expected_initial_continuation_response(&self) -> Result<Option<BoardMove>, RoleError> {
        let pending = self
            .pending_continuation
            .as_ref()
            .ok_or_else(|| role_error("initial continuation pending missing"))?;
        let id = pending
            .calls
            .first()
            .ok_or_else(|| role_error("initial continuation has no bound raw call"))?
            .0;
        let meta = self.accepted_continuation_call(id)?;
        if pending.initial_ranking.len() != meta.legal.len()
            || pending
                .initial_ranking
                .iter()
                .enumerate()
                .any(|(i, movement)| {
                    !meta.legal.contains(movement)
                        || pending.initial_ranking[..i].contains(movement)
                })
        {
            return Err(role_error(
                "initial continuation lacks a complete unique actual raw ranking",
            ));
        }
        Ok(pending.expected_response())
    }
    pub(super) fn bind_continuation_reply(&mut self, id: RequestId) -> Result<(), RoleError> {
        let pending = self
            .pending_continuation
            .as_ref()
            .ok_or_else(|| role_error("continuation pending missing before bind"))?;
        let call = self
            .calls
            .get(&id)
            .ok_or_else(|| role_error("continuation bind has no sealed input"))?;
        let meta = call
            .logical
            .as_ref()
            .ok_or_else(|| role_error("continuation bind has no actual context"))?;
        let identity = pending.identity;
        let descriptor = pending.descriptor.clone();
        let data = serde_json::json!({"prepared_payload_sha256":pending.prepared_payload,"ordinal":pending.calls.len(),
            "previous_bound_payload_sha256":pending.calls.last().map(|(_,p)|p),
            "process_epoch":id.epoch.0,"request_sequence":id.sequence,"input_sha256":call.input_sha256,
            "input_row_sha256":meta.input_row_sha256,"sidecar_row_sha256":meta.sidecar_row_sha256,"sidecar_sha256":meta.sidecar_sha256,
            "lineage_row_sha256":meta.lineage_row_sha256,"canonical_tensor_sha256":meta.canonical_tensor_sha256,
            "logical_context":logical_json(&meta.context)?,"position_rules_state_sha256":meta.position_sha256,
            "producer_metadata_admitted":meta.producer_metadata_admitted,"bound_before_submit":true,
            "physical_completion_observed":false,"delivery_observed":false,"search_consumption_observed":false});
        let payload = self.continuation_row("reply_bound", identity, &descriptor, data)?;
        self.pending_continuation
            .as_mut()
            .ok_or_else(|| role_error("continuation disappeared during bind"))?
            .calls
            .push((id, payload));
        Ok(())
    }
    pub(super) fn finish_continuation(
        &mut self,
        event: RecheckFinished<'_>,
    ) -> Result<(), RoleError> {
        if event.policy != PostRepairRecheckPolicy::ActualOpponentContinuationV1
            || !self.continuation_selected()
            || self.closed_rechecks.contains(&event.identity)
        {
            return Err(role_error("continuation finish policy or identity differs"));
        }
        let Some(pending) = self.pending_continuation.as_ref() else {
            if self.rejected_recheck.is_some_and(|id| id != event.identity)
                || event.prepared_accepted
                || event.reply_call_attempted
                || event.reply_accepted
                || event.publication.is_some()
                || event.counterline_completed
            {
                return Err(role_error(
                    "unreserved continuation claims positive execution evidence",
                ));
            }
            self.trace.failure.get_or_insert_with(|| {
                "continuation preparation rejected without native dispatch".into()
            });
            self.rejected_recheck = None;
            if self.closed_rechecks.len() < MAX_RECHECK_ATTEMPTS {
                self.closed_rechecks.push(event.identity);
            }
            return Ok(());
        };
        if pending.identity != event.identity
            || !event.prepared_accepted
            || event.reply_context != &pending.initial_context
            || event.limits.deadline != pending.limits.deadline
            || event.limits.max_cpu_nodes != pending.limits.max_cpu_nodes
            || event.limits.cpu_depth != pending.limits.cpu_depth
            || event.limits.max_rounds != pending.limits.max_rounds
            || event.deadline_tick != pending.deadline_tick
            || event.full_suffix_replayed
            || event.counterline.len() > pending.repaired.len()
            || (!pending.calls.is_empty() && !event.reply_call_attempted)
        {
            return Err(role_error(
                "continuation finish changed its original scope or claims suffix replay",
            ));
        }
        let start = pending.initial_context.prefix.len();
        let expected_response = if event.reply_accepted {
            self.expected_initial_continuation_response()?
        } else {
            None
        };
        if let Some(movement) = event.selected_response {
            if event.counterline.len() <= start
                || event.counterline[..start] != pending.repaired[..start]
                || event.counterline[start] != movement
                || movement == pending.repaired[start]
                || !event.reply_accepted
                || expected_response != Some(movement)
            {
                return Err(role_error(
                    "continuation returned another initial response or prefix",
                ));
            }
            replay_line(&self.context()?.position, event.counterline).map_err(role_error)?;
        } else if !event.counterline.is_empty()
            || event.counterline_completed
            || event.publication.is_some()
        {
            return Err(role_error(
                "continuation claims a line without a selected alternative",
            ));
        }
        if event.disposition == rz_search::pals::engine::RecheckDisposition::NoAlternativeResponse
            && (!event.reply_accepted || expected_response.is_some())
        {
            return Err(role_error(
                "continuation claims no alternative despite the accepted raw ranking",
            ));
        }
        let mut calls = Vec::with_capacity(pending.calls.len());
        for (ordinal, (id, bound)) in pending.calls.iter().enumerate() {
            let call = self
                .calls
                .get(id)
                .ok_or_else(|| role_error("bound continuation call disappeared"))?;
            let meta = call
                .logical
                .as_ref()
                .ok_or_else(|| role_error("bound continuation context disappeared"))?;
            let ply = start + ordinal;
            if event.counterline.len() > ply {
                self.accepted_continuation_call(*id)?;
                if meta.prefix.as_slice() != &event.counterline[..ply]
                    || (ordinal > 0 && meta.chosen_first != Some(event.counterline[ply]))
                    || (ordinal == 0 && !meta.legal.contains(&event.counterline[ply]))
                {
                    return Err(role_error(
                        "continuation selected tail is not the preceding actual accepted C policy",
                    ));
                }
            } else if event.counterline_completed || event.publication.is_some() {
                return Err(role_error(
                    "continuation complete/publication lacks its actual selected output",
                ));
            }
            calls.push(serde_json::json!({"ordinal":ordinal,"process_epoch":id.epoch.0,"request_sequence":id.sequence,
                "bound_payload_sha256":bound,"input_sha256":call.input_sha256,"physical":call.physical,
                "delivered":call.delivered,"search_consumed":call.accepted,"accepted_context_checked":meta.accepted_context_checked,
                "producer_metadata_admitted":meta.producer_metadata_admitted,"rejected":call.rejected,"physical_unknown":call.physical_unknown}));
        }
        if event.reply_accepted {
            self.accepted_continuation_call(
                pending
                    .calls
                    .first()
                    .ok_or_else(|| role_error("accepted continuation has no initial RequestId"))?
                    .0,
            )?;
        }
        if event.counterline.len() > start + pending.calls.len()
            || event.counterline_completed
                != (event.selected_response.is_some()
                    && event.counterline.len() == pending.repaired.len())
            || (event.publication.is_some() && (!event.counterline_completed || !event.comparable))
        {
            return Err(role_error(
                "continuation completion exceeds actual call coverage or comparable evidence",
            ));
        }
        let root = &self.context()?.position;
        let (repaired, _) = replay_line(root, &pending.repaired).map_err(role_error)?;
        let counter = event
            .counter_endpoint
            .as_ref()
            .map(|end| {
                let (position, _) = replay_line(root, event.counterline).map_err(role_error)?;
                recheck_endpoint_json(end, &position)
            })
            .transpose()?;
        let publication=event.publication.as_ref().map(|p|serde_json::json!({"observation_id":p.observation_id.0,
            "observation":recheck_observation_json(p.observation),"conclusion":{"status":format!("{:?}",p.conclusion.status),
            "evidence":p.conclusion.evidence.map(|id|id.0),"revision":p.conclusion.revision}}));
        let identity = pending.identity;
        let descriptor = pending.descriptor.clone();
        let data = serde_json::json!({"prepared_payload_sha256":pending.prepared_payload,"calls":calls,
            "prepared_accepted":event.prepared_accepted,"initial_reply_attempted":event.reply_call_attempted,"initial_reply_accepted":event.reply_accepted,
            "selected_response":event.selected_response.map(|m|Move16::pack(m).map(|m|m.bits())).transpose().map_err(role_error)?,
            "expected_initial_response":expected_response.map(|m|Move16::pack(m).map(|m|m.bits())).transpose().map_err(role_error)?,
            "initial_selection_checked":event.reply_accepted && (event.selected_response.is_some()
                || event.disposition == rz_search::pals::engine::RecheckDisposition::NoAlternativeResponse),
            "counterline":pack(event.counterline).map_err(role_error)?,"counterline_completed":event.counterline_completed,
            "full_suffix_replayed":false,"repaired_endpoint":recheck_endpoint_json(&event.repaired_endpoint,&repaired)?,"counter_endpoint":counter,
            "comparable":event.comparable,"publication":publication,"disposition":format!("{:?}",event.disposition),
            "original_error":event.original_error.map(failure_text),"engine_deadline_tick":event.deadline_tick,
            "cancelled_at_observer":event.cancel.load(Ordering::Acquire),"deadline_expired_at_observer":Instant::now()>=event.limits.deadline,
            "assurance":"actual call lifecycle observation; conditional line result; no all-defenses or training-target authority"});
        self.continuation_row("finished", identity, &descriptor, data)?;
        self.pending_continuation = None;
        self.closed_rechecks.push(identity);
        Ok(())
    }
}
