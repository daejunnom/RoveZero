//! Pure checked role consumption and continuation construction. The physical
//! owner must acknowledge the ORIGINAL ticket before calling commit_role.
use super::*;

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub fn commit_role(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<AdaptivePortStep, PalsError> {
        self.validate_role(work, ticket, output)?;
        let observations_before = self.engine.stores.observations.next_observation_sequence();
        if self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        let query = &work.input.question;
        if matches!(
            query.stage,
            RefinementStage::ProposerRanking
                | RefinementStage::ValidatorTaskRanking
                | RefinementStage::CpuValidation
                | RefinementStage::CpuDiscovery
                | RefinementStage::CounterValidation
                | RefinementStage::RepairValidation
                | RefinementStage::RecheckCounterValidation
        ) {
            return Err(RoleError::Unsupported("stage is not a role completion").into());
        }
        if query.context.purpose == RoleQueryPurpose::ValueFresh {
            // Missing physical completion or a mismatched value identity is a
            // pure admission failure. It must not consume the one semantic
            // attempt reserved for a fully checked response.
            self.checked_role_value(work, ticket, output, physical)?;
        }
        let prepared_fresh_pair = if query.stage == RefinementStage::FreshRepairedEndpoint {
            let next = self
                .fresh_pending
                .get(&query.question_digest)
                .ok_or(RoleError::InvalidOutput)?;
            // The legacy full-context baseline freezes the original raw rows
            // once for both endpoints. Equal selected views do not authorize
            // substituting a later raw snapshot for the second Fresh value.
            let same_raw_context = match (
                &query.legacy_full_records,
                &next.question.legacy_full_records,
            ) {
                (None, None) => true,
                (Some(first), Some(second)) => {
                    Arc::ptr_eq(first, second)
                        && first.authority() == self.authority
                        && second.authority() == self.authority
                        && first.digest() == second.digest()
                        && first.validates(&self.engine.stores)
                        && second.validates(&self.engine.stores)
                }
                _ => false,
            };
            if !same_raw_context {
                return Err(StoreError::InvalidEvidence(
                    "Fresh pair legacy raw context differs from frozen submission",
                )
                .into());
            }
            let seal = FreshPairSeal::from_ticket(
                ticket,
                work.input.public_context,
                self.engine.nodes[self.root].position.side_to_move().into(),
            );
            if !seal.validates_public_view(&next.view_seal, next.public_context) {
                return Err(RoleError::InvalidOutput.into());
            }
            Some(seal)
        } else {
            None
        };
        if !self.attempted_role_commits.insert(ticket.id) {
            return Err(
                StoreError::InvalidEvidence("role semantic commit was already attempted").into(),
            );
        }
        let evidence = role_output_digest(ticket, output);
        let mut fresh_pair = None;
        let rules_pair = None;
        let transition = match query.stage {
            RefinementStage::ProposalContinuation
            | RefinementStage::CounterContinuation
            | RefinementStage::Repair => {
                let ranked = rank_moves(work.input.view.query().legal.as_slice(), output)?;
                let movement = *ranked.first().ok_or(RoleError::InvalidOutput)?;
                let node = self.connect_hot(query.node, movement)?;
                let mut prefix = query.context.prefix.clone();
                prefix.push(movement);
                let done = prefix.len() >= self.engine.config.line_plies
                    || self.engine.nodes[node].terminal.is_some();
                let next = if done {
                    let (kind, stage) = match query.stage {
                        RefinementStage::ProposalContinuation => {
                            (RecordKind::Proposal, RefinementStage::CpuValidation)
                        }
                        RefinementStage::CounterContinuation => (
                            RecordKind::Counterexample,
                            RefinementStage::CounterValidation,
                        ),
                        _ => (RecordKind::Repair, RefinementStage::RepairValidation),
                    };
                    self.record_hot(
                        kind,
                        &prefix,
                        if kind == RecordKind::Counterexample {
                            query.counter_source
                        } else {
                            None
                        },
                    )?;
                    match kind {
                        RecordKind::Proposal => self.counters.proposals += 1,
                        RecordKind::Counterexample => self.counters.refutations += 1,
                        RecordKind::Repair => self.counters.repairs += 1,
                        _ => {}
                    }
                    stage
                } else {
                    query.stage
                };
                let proposal = if query.stage == RefinementStage::ProposalContinuation && done {
                    prefix.clone()
                } else {
                    query.proposal.clone()
                };
                let counter = if query.stage == RefinementStage::CounterContinuation && done {
                    Some(prefix.clone())
                } else {
                    query.counter.clone()
                };
                let repaired = if query.stage == RefinementStage::Repair && done {
                    Some(prefix.clone())
                } else {
                    query.repaired.clone()
                };
                let mut next = self.blueprint(
                    node,
                    next,
                    query.first,
                    prefix,
                    proposal,
                    counter,
                    repaired,
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                next = next.with_counter_source(query.counter_source);
                AdaptivePortTransition::Continue(next)
            }
            RefinementStage::CriticDivergence => {
                let RolePayload::Counterexample {
                    divergence_logits, ..
                } = &output.payload
                else {
                    return Err(RoleError::InvalidOutput.into());
                };
                let sites = &work.input.view.query().divergences;
                let mut ranked: Vec<_> = (0..sites.len()).collect();
                ranked.sort_by(|a, b| {
                    divergence_logits[*b]
                        .total_cmp(&divergence_logits[*a])
                        .then(a.cmp(b))
                });
                let best = if ranked.is_empty() {
                    None
                } else {
                    Some(
                        ranked[usize::try_from(self.root_round)
                            .map_err(|_| PalsError::InvalidLimits)?
                            % ranked.len()],
                    )
                };
                if let Some(best) = best {
                    let ply = sites[best];
                    let prefix = query.proposal[..ply].to_vec();
                    let node = self.replay_node(&prefix)?;
                    AdaptivePortTransition::Continue(self.blueprint(
                        node,
                        RefinementStage::CpuDiscovery,
                        query.first,
                        prefix,
                        query.proposal.clone(),
                        None,
                        None,
                        Some(ply),
                        false,
                        None,
                        None,
                    )?)
                } else {
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoDivergence,
                    }
                }
            }
            RefinementStage::CriticReply => {
                let ply = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
                let original = *query.proposal.get(ply).ok_or(RoleError::InvalidOutput)?;
                let cpu_response = query
                    .counter
                    .as_ref()
                    .and_then(|line| line.get(ply))
                    .copied();
                let ranked = rank_moves(&work.input.view.query().legal, output)?;
                let examined = &self.engine.nodes[query.node].edges;
                let movement = ranked
                    .iter()
                    .copied()
                    .find(|mv| {
                        *mv != original
                            && Some(*mv) != cpu_response
                            && !examined.iter().any(|edge| edge.movement == *mv)
                    })
                    .or_else(|| {
                        ranked
                            .iter()
                            .copied()
                            .find(|mv| *mv != original && Some(*mv) != cpu_response)
                    });
                let critic = if let Some(movement) = movement {
                    let mut prefix = query.context.prefix.clone();
                    prefix.push(movement);
                    Some(Self::pending_counter(query, prefix, None)?)
                } else {
                    None
                };
                if let Some(cpu) = query.counter.as_ref().filter(|_| cpu_response.is_some()) {
                    if let Some(critic) = critic {
                        self.enqueue_counter(critic)?;
                    }
                    self.begin_counter(Self::pending_counter(
                        query,
                        cpu.clone(),
                        query.counter_source,
                    )?)?
                } else if let Some(critic) = critic {
                    self.begin_counter(critic)?
                } else {
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoAlternative,
                    }
                }
            }
            RefinementStage::ProposalValueFresh
            | RefinementStage::CounterValueFresh
            | RefinementStage::RepairValueFresh => {
                self.commit_value(work, ticket, output, physical)?;
                self.after_validation(query, Some(work))?
            }
            RefinementStage::RecheckRepairedValueFresh => {
                self.commit_value(work, ticket, output, physical)?;
                self.critic_recheck_after_repaired_fresh(query)?
            }
            RefinementStage::CriticRecheck => {
                let repaired = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
                let ranked = rank_moves(&work.input.view.query().legal, output)?;
                let prefix_len = query.context.prefix.len();
                let movement = if query.recheck {
                    ranked.first().copied()
                } else {
                    let old = *repaired.get(prefix_len).ok_or(RoleError::InvalidOutput)?;
                    let examined = &self.engine.nodes[query.node].edges;
                    ranked
                        .iter()
                        .copied()
                        .find(|mv| *mv != old && !examined.iter().any(|edge| edge.movement == *mv))
                        .or_else(|| ranked.iter().copied().find(|mv| *mv != old))
                };
                if let Some(movement) = movement {
                    let node = self.connect_hot(query.node, movement)?;
                    let mut counterline = query.context.prefix.clone();
                    counterline.push(movement);
                    if self.engine.nodes[node].terminal.is_some()
                        || counterline.len() >= repaired.len()
                    {
                        if counterline.len() != repaired.len() {
                            AdaptivePortTransition::Unknown {
                                reason: "incomplete conditional recheck line",
                            }
                        } else {
                            self.record_hot(RecordKind::Counterexample, &counterline, None)?;
                            AdaptivePortTransition::Continue(self.blueprint(
                                node,
                                RefinementStage::RecheckCounterValidation,
                                query.first,
                                counterline.clone(),
                                query.proposal.clone(),
                                Some(counterline),
                                Some(repaired.clone()),
                                query.attack_ply,
                                true,
                                None,
                                None,
                            )?)
                        }
                    } else {
                        AdaptivePortTransition::Continue(self.blueprint(
                            node,
                            RefinementStage::CriticRecheck,
                            query.first,
                            counterline,
                            repaired.clone(),
                            query.counter.clone(),
                            Some(repaired.clone()),
                            query.attack_ply,
                            true,
                            None,
                            None,
                        )?)
                    }
                } else {
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoAlternative,
                    }
                }
            }
            RefinementStage::FreshRepairedEndpoint => {
                let seal = prepared_fresh_pair.ok_or(RoleError::InvalidOutput)?;
                let observation = self.commit_value(work, ticket, output, physical)?;
                let mut next = self
                    .fresh_pending
                    .remove(&query.question_digest)
                    .ok_or(RoleError::InvalidOutput)?;
                let mut next_query = next.question.as_ref().clone();
                next_query.paired_repaired = Some(observation);
                next_query.paired_fresh_seal = Some(seal);
                next.question = Arc::new(next_query);
                fresh_pair = Some(seal);
                AdaptivePortTransition::Continue(self.blueprint_ready(next)?)
            }
            RefinementStage::FreshCounterEndpoint => {
                let counter = self.commit_value(work, ticket, output, physical)?;
                let repaired = query.paired_repaired.ok_or(RoleError::InvalidOutput)?;
                if self.publish_fresh_comparison(query, repaired, counter)? {
                    self.queue_strict_repair(work, query.context.prefix.clone())?;
                }
                fresh_pair = Some(FreshPairSeal::from_ticket(
                    ticket,
                    work.input.public_context,
                    self.engine.nodes[self.root].position.side_to_move().into(),
                ));
                AdaptivePortTransition::Complete {
                    scope: FrameCompletionScope::CompletedFreshPair,
                }
            }
            _ => return Err(RoleError::Unsupported("stage is not a role completion").into()),
        };
        let mut transition = self.finish_branch(work, transition)?;
        transition.bind_role_cause(work, ticket)?;
        self.bind_deferred_recheck_role_cause(work, ticket)?;
        self.accepted.insert(ticket.id);
        self.committed_role_questions
            .insert(ticket.id, ticket.question);
        self.last_role_commit = Some((
            work.input.question_digest,
            ticket.id,
            observations_before,
            self.engine.stores.observations.next_observation_sequence(),
        ));
        self.counters.role_calls += 1;
        self.counters.consumed_role_outputs += 1;
        match query.context.purpose {
            RoleQueryPurpose::ValueFresh => {
                self.counters.value_calls += 1;
                self.counters.completed_value_calls += 1;
                self.counters.accepted_value_outputs += 1;
            }
            RoleQueryPurpose::RepairPolicy => {
                self.counters.repair_calls += 1;
                self.counters.completed_repair_calls += 1;
                self.counters.accepted_repair_outputs += 1;
            }
            RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::DivergencePolicy => {
                self.counters.critic_calls += 1;
                self.counters.completed_critic_calls += 1;
                self.counters.accepted_critic_outputs += 1;
            }
            _ => {
                self.counters.proposer_calls += 1;
                self.counters.completed_proposer_calls += 1;
                self.counters.accepted_proposer_outputs += 1;
            }
        }
        Ok(AdaptivePortStep {
            evidence,
            transition,
            fresh_pair,
            rules_pair,
            consumption_gate: Some((ticket.id, ticket.question, ticket.view.view_digest)),
            feedback_gate: None,
            queue_question: Some((work.frame, work.input.question_digest)),
        })
    }

    /// CPU report itself was already checked/committed by cpu_commit. This
    /// helper creates only the next exact question, never another CPU call.
    pub fn after_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        observation: ObservationId,
        _complete: bool,
    ) -> Result<AdaptivePortStep, PalsError> {
        self.check_work(work)?;
        let query = &work.input.question;
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-accepted-cpu-evidence/1\0");
        hash.update(work.input.question_digest.0);
        hash.update((observation.0 as u64).to_le_bytes());
        let evidence = ContractDigest(hash.finalize().into());
        if !self.cpu_observation_consumed(work, observation)
            && !self.external_observation_consumed(work, observation)
        {
            return Ok(AdaptivePortStep {
                evidence,
                transition: self.finish_branch(
                    work,
                    AdaptivePortTransition::Unknown {
                        reason: "raw CPU observation was not consumed by this exact question",
                    },
                )?,
                fresh_pair: None,
                rules_pair: None,
                consumption_gate: None,
                feedback_gate: None,
                queue_question: None,
            });
        }
        let transition = if query.stage == RefinementStage::CpuDiscovery {
            let raw = self.engine.stores.observations.get(observation)?;
            let pv = raw
                .cpu_pv
                .map(|line| self.engine.stores.lines.moves(line))
                .transpose()?
                .unwrap_or_default();
            // A discovered restricted score is never assigned to its leaf. Only
            // exact legal line and immutable source observation enter memory.
            let alternative = !pv.is_empty()
                && query
                    .attack_ply
                    .and_then(|ply| query.proposal.get(ply))
                    .copied()
                    != pv.first().copied();
            if !alternative
                && !query.discovery_restricted
                && query.validator_root_moves.is_none()
                && query.position.legal_moves().iter().any(|movement| {
                    Some(*movement)
                        != query
                            .attack_ply
                            .and_then(|ply| query.proposal.get(ply))
                            .copied()
                })
            {
                let mut next = self.blueprint(
                    query.node,
                    RefinementStage::CpuDiscovery,
                    query.first,
                    query.context.prefix.clone(),
                    query.proposal.clone(),
                    None,
                    None,
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                next = next.with_restricted_discovery();
                AdaptivePortTransition::Continue(next)
            } else {
                let cpu_line = if alternative {
                    let mut line = query.context.prefix.clone();
                    let available = self.engine.config.line_plies.saturating_sub(line.len());
                    line.extend(pv.iter().take(available));
                    self.replay_node(&line)?;
                    self.record_hot(RecordKind::Counterexample, &line, Some(observation))?;
                    Some(line)
                } else {
                    None
                };
                let mut next = self.blueprint(
                    query.node,
                    RefinementStage::CriticReply,
                    query.first,
                    query.context.prefix.clone(),
                    query.proposal.clone(),
                    cpu_line,
                    query.repaired.clone(),
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                next = next.with_counter_source(if alternative { Some(observation) } else { None });
                AdaptivePortTransition::Continue(next)
            }
        } else if query.stage == RefinementStage::RecheckCounterValidation {
            let pair = self.prepare_final_pair(
                work,
                query.repaired.clone().ok_or(RoleError::InvalidOutput)?,
                query.counter.clone().ok_or(RoleError::InvalidOutput)?,
            )?;
            let transition = self.finish_branch(work, pair.0)?;
            return Ok(AdaptivePortStep {
                evidence,
                transition,
                fresh_pair: None,
                rules_pair: pair.1,
                consumption_gate: None,
                feedback_gate: Some((work.input.question_digest, observation)),
                queue_question: Some((work.frame, work.input.question_digest)),
            });
        } else if self.engine.uses_model_resolver() {
            let stage = match query.stage {
                RefinementStage::CpuValidation => RefinementStage::ProposalValueFresh,
                RefinementStage::CounterValidation => RefinementStage::CounterValueFresh,
                RefinementStage::RepairValidation => RefinementStage::RepairValueFresh,
                _ => return Err(RoleError::InvalidOutput.into()),
            };
            AdaptivePortTransition::Continue(self.blueprint(
                query.node,
                stage,
                query.first,
                query.context.prefix.clone(),
                query.proposal.clone(),
                query.counter.clone(),
                query.repaired.clone(),
                query.attack_ply,
                false,
                None,
                None,
            )?)
        } else {
            self.after_validation(query, Some(work))?
        };
        if matches!(
            transition,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck
            }
        ) {
            self.bind_deferred_recheck_feedback_cause(work, observation)?;
        }
        let transition = self.finish_branch(work, transition)?;
        Ok(AdaptivePortStep {
            evidence,
            transition,
            fresh_pair: None,
            rules_pair: None,
            consumption_gate: None,
            feedback_gate: Some((work.input.question_digest, observation)),
            queue_question: Some((work.frame, work.input.question_digest)),
        })
    }

    pub(super) fn after_validation(
        &mut self,
        query: &AdaptivePortQuestion,
        parent: Option<&CoordinatorWork<ReviewView, AdaptivePortQuestion>>,
    ) -> Result<AdaptivePortTransition, PalsError> {
        match query.stage {
            RefinementStage::CpuValidation | RefinementStage::ProposalValueFresh => {
                let proposal = if query.proposal.is_empty() {
                    query.context.prefix.clone()
                } else {
                    query.proposal.clone()
                };
                if self.divergence_indices(&proposal)?.is_empty() {
                    Ok(AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoDivergence,
                    })
                } else {
                    Ok(AdaptivePortTransition::Continue(self.blueprint(
                        self.root,
                        RefinementStage::CriticDivergence,
                        query.first,
                        Vec::new(),
                        proposal,
                        None,
                        None,
                        None,
                        false,
                        None,
                        None,
                    )?))
                }
            }
            RefinementStage::CounterValidation | RefinementStage::CounterValueFresh => {
                self.publish_supported_counter(query)?;
                let first = query.first.ok_or(RoleError::InvalidOutput)?;
                if !self.engine.admit_iterative_repair(first) {
                    return Ok(AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::RepairNotAdmitted,
                    });
                }
                let ply = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
                let counter = query.counter.as_ref().ok_or(RoleError::InvalidOutput)?;
                let prefix = counter
                    .get(..=ply)
                    .ok_or(RoleError::InvalidOutput)?
                    .to_vec();
                let node = self.replay_node(&prefix)?;
                self.begin_repair_continuation(
                    node,
                    query.first,
                    prefix,
                    query.proposal.clone(),
                    counter.clone(),
                    ply,
                    query.counter_source,
                )
            }
            RefinementStage::RepairValidation | RefinementStage::RepairValueFresh => {
                self.publish_supported_repair(query)?;
                if self.engine.post_repair_recheck == PostRepairRecheckPolicy::Disabled {
                    return Ok(AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::NoAlternative,
                    });
                }
                let checked = self.critic_recheck_after_repaired_fresh(query)?;
                if !matches!(checked, AdaptivePortTransition::Continue(_)) {
                    return Ok(checked);
                }
                let repaired = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
                let repaired_node = self.replay_node(repaired)?;
                let fixed_iterative = self.root_fanout == AdaptiveRootFanoutPolicy::SequentialFixed
                    && self.engine.post_repair_recheck
                        == PostRepairRecheckPolicy::IterativeFrozenModelWdlV2;
                if self.engine.nodes[repaired_node].terminal.is_some() && !fixed_iterative {
                    return Ok(checked);
                }
                let blueprint = self.blueprint(
                    repaired_node,
                    RefinementStage::RecheckRepairedValueFresh,
                    query.first,
                    repaired.clone(),
                    repaired.clone(),
                    query.counter.clone(),
                    Some(repaired.clone()),
                    query.attack_ply,
                    false,
                    None,
                    None,
                )?;
                if fixed_iterative {
                    let work = parent.ok_or(RoleError::InvalidOutput)?;
                    self.defer_fixed_recheck(work, blueprint)
                } else {
                    Ok(AdaptivePortTransition::Continue(blueprint))
                }
            }
            _ => Err(RoleError::InvalidOutput.into()),
        }
    }

    /// Mirrors synchronous `follow(Call::Repair)`: a full or terminal prefix
    /// is a zero-step Repair, not permission to append one more model move.
    /// The caller already admitted this Repair under the per-first-move cap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_repair_continuation(
        &mut self,
        node: usize,
        first: Option<BoardMove>,
        prefix: Vec<BoardMove>,
        proposal: Vec<BoardMove>,
        counter: Vec<BoardMove>,
        attack: usize,
        source: Option<ObservationId>,
    ) -> Result<AdaptivePortTransition, PalsError> {
        self.check_live()?;
        let first = first.ok_or(RoleError::InvalidOutput)?;
        if prefix.len() > self.engine.config.line_plies
            || proposal.len() > self.engine.config.line_plies
            || counter.len() > self.engine.config.line_plies
            || prefix.first() != Some(&first)
            || proposal.first() != Some(&first)
            || counter.first() != Some(&first)
            || counter.get(..=attack) != Some(prefix.as_slice())
            || proposal.get(..attack) != prefix.get(..attack)
            || proposal.get(attack) == counter.get(attack)
            || self.replay_node(&prefix)? != node
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let zero_step = prefix.len() == self.engine.config.line_plies
            || self.engine.nodes[node].terminal.is_some();
        if !zero_step {
            return Ok(AdaptivePortTransition::Continue(
                self.blueprint(
                    node,
                    RefinementStage::Repair,
                    Some(first),
                    prefix,
                    proposal,
                    Some(counter),
                    None,
                    Some(attack),
                    false,
                    None,
                    None,
                )?
                .with_counter_source(source),
            ));
        }

        // Retain the unchanged line's actual Repair record and genealogy. No
        // accepted Repair output, revised own move or new witness is invented.
        self.record_hot(RecordKind::Repair, &prefix, None)?;
        self.counters.repairs = self
            .counters
            .repairs
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        if self.engine.nodes[node].terminal.is_some() {
            return Ok(AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoRepairContinuation,
            });
        }
        let cached_cpu = self.has_reusable_completed_cpu(node, prefix.len())?;
        if cached_cpu {
            // One new semantic verification lookup consumes the already
            // accepted exact estimate. It adds no physical execution or record.
            self.counters.evidence_cache_hits = self
                .counters
                .evidence_cache_hits
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            self.counters.consumed_cached_cpu_values = self
                .counters
                .consumed_cached_cpu_values
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            if !self.engine.uses_model_resolver() {
                return Ok(AdaptivePortTransition::Complete {
                    scope: FrameCompletionScope::NoRepairContinuation,
                });
            }
        }
        let next = if cached_cpu {
            // A model-resolver value must still be Fresh for the new record
            // revision, even though own CPU verification used exact evidence.
            RefinementStage::RepairValueFresh
        } else {
            RefinementStage::RepairValidation
        };
        Ok(AdaptivePortTransition::Continue(
            self.blueprint(
                node,
                next,
                Some(first),
                prefix.clone(),
                proposal,
                Some(counter),
                Some(prefix),
                Some(attack),
                false,
                None,
                None,
            )?
            .with_counter_source(source),
        ))
    }

    /// The extra Fresh endpoint was already accepted. Continue to the actual
    /// opponent anchor without publishing Repair or enqueueing it a second time.
    pub(super) fn critic_recheck_after_repaired_fresh(
        &mut self,
        query: &AdaptivePortQuestion,
    ) -> Result<AdaptivePortTransition, PalsError> {
        let repaired = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
        let counter = query.counter.as_ref().ok_or(RoleError::InvalidOutput)?;
        let attack = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
        let root_color = self.engine.nodes[self.root].position.side_to_move();
        let mut position = self.engine.nodes[self.root].position.clone();
        let mut changed = None;
        let mut anchor = None;
        for (ply, movement) in repaired.iter().enumerate() {
            if ply > attack
                && position.side_to_move() == root_color
                && counter.get(ply) != Some(movement)
                && changed.is_none()
            {
                changed = Some(ply);
            }
            if changed.is_some_and(|change| ply > change)
                && position.side_to_move() != root_color
                && matches!(
                    position.classify_position()?.play_status,
                    PlayStatus::Ongoing
                )
            {
                anchor = Some(ply);
                break;
            }
            position.make_move(*movement)?;
        }
        if let Some(anchor) = anchor {
            let prefix = repaired[..anchor].to_vec();
            let node = self.replay_node(&prefix)?;
            Ok(AdaptivePortTransition::Continue(self.blueprint(
                node,
                RefinementStage::CriticRecheck,
                query.first,
                prefix,
                repaired.clone(),
                Some(counter.clone()),
                Some(repaired.clone()),
                Some(attack),
                false,
                None,
                None,
            )?))
        } else {
            Ok(AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoAlternative,
            })
        }
    }

    pub(super) fn record_hot(
        &mut self,
        kind: RecordKind,
        line: &[BoardMove],
        cpu: Option<ObservationId>,
    ) -> Result<Option<(LineId, ObservationId)>, PalsError> {
        self.record_hot_with_cpu_value(kind, line, cpu, None)
    }

    /// Publish the same model-facing verification projection as the legacy
    /// own checker, after its independently retained raw task was consumed.
    pub(super) fn record_cpu_verification_hot(
        &mut self,
        line: &[BoardMove],
        score: i32,
        depth: u16,
        scope: CpuScoreScope,
        perspective: Color,
    ) -> Result<(), PalsError> {
        self.record_hot_with_cpu_value(
            RecordKind::CpuVerification,
            line,
            None,
            Some((score, depth, scope, perspective)),
        )?;
        Ok(())
    }

    fn record_hot_with_cpu_value(
        &mut self,
        kind: RecordKind,
        line: &[BoardMove],
        cpu: Option<ObservationId>,
        value: Option<(i32, u16, CpuScoreScope, Color)>,
    ) -> Result<Option<(LineId, ObservationId)>, PalsError> {
        if value.is_some() && kind != RecordKind::CpuVerification {
            return Err(StoreError::InvalidEvidence("CPU verification projection kind").into());
        }
        if self.engine.records.len() >= self.engine.config.max_records {
            // Raw/failed evidence is not deleted on the manager to keep running.
            return Err(
                StoreError::PinSaturated("adaptive public records need archive worker").into(),
            );
        }
        let root_state = self.engine.nodes[self.root].state;
        let revision = self
            .engine
            .revision
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let supersedes_revision = self
            .engine
            .records
            .iter()
            .rev()
            .find(|record| {
                record.origin_state == root_state
                    && record.kind == kind
                    && record.line.first() == line.first()
            })
            .map(|record| record.revision);
        let parent_revision = self
            .engine
            .records
            .iter()
            .rev()
            .find(|record| {
                record.origin_state == root_state
                    && record.line.first() == line.first()
                    && match kind {
                        RecordKind::Repair => record.kind == RecordKind::Counterexample,
                        RecordKind::Counterexample => {
                            matches!(record.kind, RecordKind::Proposal | RecordKind::Repair)
                        }
                        RecordKind::CpuVerification => record.line == line,
                        RecordKind::Proposal => false,
                    }
            })
            .map(|record| record.revision);
        let mut replay = self.engine.nodes[self.root].position.clone();
        for movement in line {
            replay.make_move(*movement)?;
        }
        let record = RoleRecord {
            revision,
            parent_revision,
            supersedes_revision,
            origin_state: root_state,
            kind,
            line: line.to_vec(),
            value: value.map(|value| value.0),
            completed_depth: value.map_or(0, |value| value.1),
            score_scope: value.map(|value| value.2),
            cpu_observation: cpu,
            perspective: value.map_or(
                self.engine.nodes[self.root].position.side_to_move(),
                |value| value.3,
            ),
            critical: true,
        };
        if cpu.is_some() || kind == RecordKind::CpuVerification {
            self.engine.publish_record(record, None, None)?;
            return Ok(None);
        }
        let focus = self
            .engine
            .stores
            .situations
            .get(self.engine.nodes[self.root].situation)?
            .focus;
        let line = self.engine.stores.hot_append_line(focus, line)?;
        let observation = self.engine.stores.hot_append_observation(Observation {
            state: root_state,
            line: Some(line),
            source: stable_id(self.engine.model.identity()),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(self.engine.model.identity()),
                encoding: stable_id("pals-selected-review-v3"),
                input: revision,
            },
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            score: RawScore::Unknown,
            budget: 0,
            kind: match kind {
                RecordKind::Proposal => ObservationKind::Proposal,
                RecordKind::Counterexample => ObservationKind::Refutation,
                RecordKind::Repair => ObservationKind::Repair,
                RecordKind::CpuVerification => ObservationKind::Proposal,
            },
            supersedes: None,
            execution: None,
        })?;
        self.engine.pin_archive_active_record(line, observation)?;
        self.engine
            .publish_record(record, None, Some(observation))?;
        Ok(Some((line, observation)))
    }

    fn checked_role_value(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<ModelValueOutput, PalsError> {
        self.verify_registered_model_namespace(ticket, output)?;
        let physical = physical.ok_or(RoleError::InvalidOutput)?;
        if ticket.role != Role::Proposer
            || ticket.question != work.input.question_digest
            || ticket.representation != output.key
            || work.input.question.context.purpose != RoleQueryPurpose::ValueFresh
            || physical.physical_fresh != Some(true)
            || !physical.measurement_complete
            || physical.completed_nn_inputs != Some(1)
            || physical.execution.is_none()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let query = &work.input.question;
        let RolePayload::Proposal { wdl, .. } = output.payload else {
            return Err(RoleError::InvalidOutput.into());
        };
        let identity = self
            .engine
            .model_registered_value
            .clone()
            .ok_or(RoleError::Unavailable)?;
        if self
            .engine
            .model
            .registered_representation_namespace()
            .is_some_and(|namespace| namespace.value_identity() != &identity)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let value = ModelValueOutput {
            identity: identity.clone(),
            input_sha256: output.key.input.0,
            state: query.position.position_identity(),
            perspective: query.position.side_to_move(),
            wdl: wdl.probabilities(),
        };
        value.validate(&query.position, &identity)?;
        Ok(value)
    }

    fn commit_value(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<ObservationId, PalsError> {
        // Recheck the immutable producer namespace at the actual Store commit
        // boundary, even though admission already validated it before attempt.
        let value = self.checked_role_value(work, ticket, output, physical)?;
        let identity = value.identity.clone();
        let query = &work.input.question;
        let observation = self.engine.stores.hot_append_observation(Observation {
            state: query.context.state,
            line: None,
            source: stable_id(&identity.semantics),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(&identity.model),
                encoding: stable_id(&identity.encoding),
                input: u64::from_le_bytes(
                    output.key.input.0[..8]
                        .try_into()
                        .map_err(|_| RoleError::InvalidOutput)?,
                ),
            },
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: Some(identity),
            model_value_input: Some(output.key.input.0),
            cpu_condition: None,
            cpu_pv: None,
            score: RawScore::ContextWdl {
                win: value.wdl[0],
                draw: value.wdl[1],
                loss: value.wdl[2],
                perspective: value.perspective,
                context_revision: query.context.public_revision,
            },
            budget: 1,
            kind: ObservationKind::Proposal,
            supersedes: None,
            execution: None,
        })?;
        self.engine
            .stores
            .hot_add_dependency(observation, query.context.situation)?;
        self.engine
            .stores
            .hot_add_dependency(observation, self.engine.nodes[self.root].situation)?;
        self.engine.nodes[query.node].model_value = Some(value);
        self.engine.nodes[query.node].model_observation = Some(observation);
        self.engine.nodes[query.node].model_value_revision = Some(query.context.public_revision);
        Ok(observation)
    }

    pub(super) fn prepare_final_pair(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        repaired: Vec<BoardMove>,
        counter: Vec<BoardMove>,
    ) -> Result<(AdaptivePortTransition, Option<AdaptiveRulesPairProof>), PalsError> {
        let query = &work.input.question;
        let repaired_node = self.replay_node(&repaired)?;
        let counter_node = self.replay_node(&counter)?;
        match (
            self.engine.nodes[repaired_node].terminal,
            self.engine.nodes[counter_node].terminal,
        ) {
            (Some(_), Some(_)) => {
                // Exact endpoints can be compared through Rules, but this alone
                // does not certify every defense or fabricate NN executions.
                let perspective = self.engine.nodes[self.root].position.side_to_move();
                let old = self.engine.endpoint_expectation(repaired_node, perspective);
                let new = self.engine.endpoint_expectation(counter_node, perspective);
                if matches!((old,new),(Some(old),Some(new)) if new<old) {
                    self.record_hot(RecordKind::Counterexample, &counter, None)?;
                    self.queue_strict_repair(work, counter.clone())?;
                }
                let proof = AdaptiveRulesPairProof {
                    authority: self.authority,
                    root_state: work.input.view_seal.root_state,
                    public_context: work.input.public_context,
                    submitted_question: work.input.question_digest,
                    repaired_line: repaired,
                    counter_line: counter,
                    repaired: Arc::new(
                        self.issue_rules_position(
                            self.engine.nodes[repaired_node].position.clone(),
                        )?
                        .export()
                        .map_err(|_| RoleError::InvalidOutput)?,
                    ),
                    counter: Arc::new(
                        self.issue_rules_position(
                            self.engine.nodes[counter_node].position.clone(),
                        )?
                        .export()
                        .map_err(|_| RoleError::InvalidOutput)?,
                    ),
                    perspective: perspective.into(),
                };
                proof.validate(work)?;
                Ok((
                    AdaptivePortTransition::Complete {
                        scope: FrameCompletionScope::CompletedRulesPair,
                    },
                    Some(proof),
                ))
            }
            (Some(_), None) | (None, Some(_)) => Ok((
                AdaptivePortTransition::Unknown {
                    reason: "mixed terminal and model Fresh pair",
                },
                None,
            )),
            (None, None) => {
                let blueprint = self.blueprint_fresh_pair(
                    query,
                    repaired_node,
                    repaired,
                    counter_node,
                    counter,
                )?;
                Ok((AdaptivePortTransition::Continue(blueprint), None))
            }
        }
    }
    fn publish_fresh_comparison(
        &mut self,
        query: &AdaptivePortQuestion,
        repaired: ObservationId,
        counter: ObservationId,
    ) -> Result<bool, PalsError> {
        let before = self.engine.stores.observations.get(repaired)?;
        let after = self.engine.stores.observations.get(counter)?;
        if before.model_value_identity != after.model_value_identity
            || before.model_value_identity != self.engine.model_registered_value
            || before.model_value_input.is_none()
            || after.model_value_input.is_none()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let perspective = self.engine.nodes[self.root].position.side_to_move();
        let expected = |observation: &Observation| -> Result<(f32, u64), PalsError> {
            if let RawScore::ContextWdl {
                win,
                loss,
                perspective: side,
                context_revision,
                ..
            } = observation.score
            {
                Ok((
                    if side == perspective {
                        win - loss
                    } else {
                        loss - win
                    },
                    context_revision,
                ))
            } else {
                Err(RoleError::InvalidOutput.into())
            }
        };
        let (before_value, before_revision) = expected(before)?;
        let (after_value, after_revision) = expected(after)?;
        if before_revision != after_revision {
            return Err(RoleError::InvalidOutput.into());
        }
        if after_value >= before_value {
            self.counters.frozen_wdl_unresolved += 1;
            return Ok(false);
        }
        if self.conditional_witnesses.len() >= 64 {
            return Err(PalsError::Capacity);
        }
        let repaired_line = query.repaired.as_ref().ok_or(RoleError::InvalidOutput)?;
        let root = self.engine.nodes[self.root].situation;
        let focus = self.engine.stores.situations.get(root)?.focus;
        let line = self.engine.stores.hot_append_line(focus, repaired_line)?;
        let evidence = self.engine.stores.hot_append_observation(Observation {
            state: self.engine.nodes[self.root].state,
            line: Some(line),
            source: stable_id(self.engine.search_identity()),
            epoch: 0,
            scope: EvidenceScope::Model {
                model: stable_id(self.engine.model.identity()),
                encoding: stable_id("pals-frozen-pair-v3"),
                input: before_revision,
            },
            value_identity: None,
            checker_identity: None,
            checker_work: None,
            external_report: None,
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: None,
            cpu_pv: None,
            score: RawScore::ConditionalWdl {
                expectation: after_value,
                perspective,
                context_revision: before_revision,
                repaired,
                counter,
            },
            budget: 0,
            kind: ObservationKind::Refutation,
            supersedes: None,
            execution: None,
        })?;
        self.engine
            .stores
            .hot_refute_continuation(root, line, evidence)?;
        self.conditional_witnesses.insert(
            evidence,
            (ObservationKind::Refutation, query.context.prefix.clone()),
        );
        self.counters.supported_refutations += 1;
        Ok(true)
    }

    pub fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), PalsError> {
        if closed.id.authority != self.authority || self.closed.contains(&closed.id) {
            return Err(StoreError::StaleConsumer.into());
        }
        if let Some((movement, _)) = self
            .round_candidates
            .values()
            .find(|(_, frame)| *frame == Some(closed.id))
        {
            let first = BoardMove::try_from(
                closed
                    .original_first_move
                    .decode()
                    .map_err(|_| RoleError::InvalidOutput)?,
            )
            .map_err(|_| RoleError::InvalidOutput)?;
            if first != *movement {
                return Err(RoleError::InvalidOutput.into());
            }
            self.last_main_closed = Some(closed.id);
        }
        self.closed.insert(closed.id);
        Ok(())
    }
    pub fn result(&self, completion: PalsCompletion) -> Result<PalsResult, PalsError> {
        let position = &self.engine.nodes[self.root].position;
        if self.engine.nodes[self.root].terminal.is_some() {
            return self
                .engine
                .unsearched_result(position, PalsCompletion::Terminal, self.started);
        }
        let legal = position.legal_moves();
        let mut counters = self.counters;
        if self.engine.uses_model_resolver() {
            return self.engine.model_result(
                self.root,
                &legal,
                completion,
                self.started,
                &mut counters,
            );
        }
        let mut best = legal.first().copied();
        let mut score = None;
        let mut scope = PalsValueScope::Unknown;
        let mut root_values = Vec::with_capacity(legal.len());
        for movement in legal {
            let edge = self.engine.nodes[self.root]
                .edges
                .iter()
                .find(|edge| edge.movement == movement);
            let value = edge
                .and_then(|edge| {
                    self.engine
                        .value(edge.child, self.engine.config.line_plies.saturating_sub(1))
                })
                .map(|value| -value);
            let child_scope = edge.map_or(PalsValueScope::Unknown, |edge| {
                if self.engine.nodes[edge.child].terminal.is_some() {
                    PalsValueScope::RulesTerminal
                } else if value.is_some() {
                    PalsValueScope::RestrictedEstimate
                } else {
                    PalsValueScope::Unknown
                }
            });
            if let Some(value) = value
                && better_root_choice(value, child_scope, score, scope)
            {
                best = Some(movement);
                score = Some(value);
                scope = child_scope;
            }
            let resolved_value = value.map_or(PalsResolvedValue::Unknown, |value| {
                PalsResolvedValue::OwnedRaw {
                    value,
                    perspective: position.side_to_move(),
                }
            });
            root_values.push(PalsRootValue {
                movement,
                score: value,
                resolved_value,
                scope: child_scope,
                examined_replies: edge.map_or(0, |edge| self.engine.nodes[edge.child].edges.len()),
                unexplored_replies: edge.map(|edge| {
                    self.engine.nodes[edge.child]
                        .position
                        .legal_moves()
                        .len()
                        .saturating_sub(self.engine.nodes[edge.child].edges.len())
                }),
            });
        }
        counters.retained_situations = self.engine.nodes.len();
        counters.root_scope_observation_complete = true;
        Ok(PalsResult {
            best_move: best,
            score,
            resolved_value: score.map_or(PalsResolvedValue::Unknown, |value| {
                PalsResolvedValue::OwnedRaw {
                    value,
                    perspective: position.side_to_move(),
                }
            }),
            model_value_identity: self.engine.model_registered_value.clone(),
            checker_identity: self.engine.checker_registered_identity.clone(),
            value_scope: scope,
            terminal: None,
            completion,
            counters,
            root_values,
            elapsed: self.started.elapsed(),
            model_identity: self.engine.model.identity().to_owned(),
            resolver_version: self.engine.resolver_version(),
        })
    }
}
fn rank_moves(legal: &[BoardMove], output: &RoleOutput) -> Result<Vec<BoardMove>, PalsError> {
    let logits = match &output.payload {
        RolePayload::Proposal {
            candidate_logits, ..
        }
        | RolePayload::Counterexample {
            candidate_logits, ..
        } => candidate_logits,
        _ => return Err(RoleError::InvalidOutput.into()),
    };
    if logits.len() != legal.len() {
        return Err(RoleError::InvalidOutput.into());
    }
    let mut indices = (0..legal.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    Ok(indices.into_iter().map(|index| legal[index]).collect())
}
fn role_output_digest(ticket: &RoleTicketEnvelope, output: &RoleOutput) -> ContractDigest {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-consumed-role-proof/1\0");
    hash.update(ticket.question.0);
    hash.update(ticket.view.view_digest.0);
    hash.update(output.key.input.0);
    hash.update(output.id.sequence.to_le_bytes());
    let mut values = Vec::new();
    match &output.payload {
        RolePayload::Proposal {
            candidate_logits,
            wdl,
        } => {
            values.extend_from_slice(candidate_logits);
            values.extend_from_slice(&wdl.probabilities());
        }
        RolePayload::Counterexample {
            candidate_logits,
            divergence_logits,
            wdl,
        } => {
            values.extend_from_slice(candidate_logits);
            values.extend_from_slice(divergence_logits);
            values.extend_from_slice(&wdl.probabilities());
        }
        RolePayload::TaskRanking { task_logits } => values.extend_from_slice(task_logits),
    }
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    ContractDigest(hash.finalize().into())
}
