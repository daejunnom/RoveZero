use rz_arena::{
    ArenaPlan, EngineFailureKind, Event, GameOutcome, GameResult, Ledger, LedgerLimits, PlanLimits,
};
use rz_experiments::{ArtifactRef, RunManifest};
use serde_json::Value;

fn manifest(pairs: u64) -> RunManifest {
    let mut input = RunManifest::from_json(include_str!(
        "../../../experiments/baselines/fixtures/e01-input.json"
    ))
    .unwrap();
    input.plan.pairs = pairs;
    input.budget.max_pairs = pairs;
    input.budget.max_games = pairs * 4;
    input
}

fn plan(pairs: u64) -> ArenaPlan {
    ArenaPlan::build(
        &manifest(pairs).lock().unwrap(),
        PlanLimits {
            max_pairs: pairs,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
}

fn limits() -> LedgerLimits {
    LedgerLimits {
        max_events: 100,
        max_bytes: 1_048_576,
    }
}

fn evidence() -> ArtifactRef {
    ArtifactRef {
        path: "evidence/synthetic-declarations.json".into(),
        sha256: "a".repeat(64),
        bytes: 1,
        source: "https://github.com/daejunnom/RoveZero".into(),
        license: "MIT".into(),
    }
}

fn input_artifact_bytes(plan: &ArenaPlan) -> u64 {
    let mut unique = std::collections::BTreeMap::new();
    for artifact in plan.manifest().declared_artifacts() {
        unique.insert(&artifact.path, artifact.bytes);
    }
    unique.values().copied().sum()
}

fn start(plan: &ArenaPlan, pair: usize, attempt: u32) -> Event {
    Event::PairStarted {
        pair_id: plan.pairs()[pair].id.clone(),
        attempt,
        process_run_id: format!("synthetic-pair-{pair}-attempt-{attempt}"),
    }
}

fn record(plan: &ArenaPlan, pair: usize, game: usize, attempt: u32, outcome: GameOutcome) -> Event {
    Event::GameRecorded {
        pair_id: plan.pairs()[pair].id.clone(),
        attempt,
        game_id: plan.pairs()[pair].games[game].id.clone(),
        outcome,
    }
}

fn close(plan: &ArenaPlan, pair: usize, attempt: u32) -> Event {
    Event::PairClosed {
        pair_id: plan.pairs()[pair].id.clone(),
        attempt,
    }
}

fn terminal(plan: &ArenaPlan, pair: usize, game: usize, candidate_points: u8) -> GameOutcome {
    let candidate = &plan.manifest().input().engines[0].id;
    let candidate_white = &plan.pairs()[pair].games[game].white_engine == candidate;
    let result = match candidate_points {
        1 => GameResult::Draw,
        2 if candidate_white => GameResult::WhiteWin,
        2 => GameResult::BlackWin,
        0 if candidate_white => GameResult::BlackWin,
        0 => GameResult::WhiteWin,
        _ => panic!("test scores are 0..=2"),
    };
    GameOutcome::RulesTerminal {
        result,
        reason: if candidate_points == 1 {
            "stalemate"
        } else {
            "checkmate"
        }
        .into(),
        evidence: evidence(),
    }
}

fn record_pair(ledger: &mut Ledger, plan: &ArenaPlan, pair: usize, attempt: u32, scores: [u8; 2]) {
    for game in plan.pairs()[pair].execution_order {
        ledger
            .append(record(
                plan,
                pair,
                game,
                attempt,
                terminal(plan, pair, game, scores[game]),
            ))
            .unwrap();
    }
    ledger.append(close(plan, pair, attempt)).unwrap();
}

fn reject_unchanged(ledger: &mut Ledger, event: Event) {
    let before = ledger.to_jsonl().unwrap();
    assert!(ledger.append(event).is_err());
    assert_eq!(before, ledger.to_jsonl().unwrap());
}

fn lines(input: &str) -> Vec<Value> {
    input
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn jsonl(lines: &[Value]) -> String {
    let mut output = String::new();
    for value in lines {
        output.push_str(&serde_json::to_string(value).unwrap());
        output.push('\n');
    }
    output
}

#[test]
fn hand_counted_ll_dl_dd_wl_wd_ww_preserve_distinct_raw_outcomes() {
    let plan = plan(6);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    for (pair, scores) in [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2]]
        .into_iter()
        .enumerate()
    {
        ledger.append(start(&plan, pair, 1)).unwrap();
        record_pair(&mut ledger, &plan, pair, 1, scores);
    }
    let summary = ledger.summary().unwrap();
    assert_eq!(summary.completed_pairs, 6);
    assert_eq!((summary.wins, summary.draws, summary.losses), (4, 4, 4));
    assert_eq!(summary.n, [1, 1, 2, 1, 1]);
    assert_eq!((summary.excluded_pairs, summary.pending_pairs), (0, 0));
    assert!(!summary.execution_ready);
    assert!(!summary.interpretation_blocked);
    let input = ledger.to_jsonl().unwrap();
    let replay = Ledger::from_jsonl(&plan, &input, limits()).unwrap();
    assert_eq!(replay.to_jsonl().unwrap(), input);
    assert_eq!(replay.summary().unwrap(), summary);
    assert_eq!(replay.events().count(), 24);
}

#[test]
fn partial_pair_has_no_score_and_cannot_close() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let game = plan.pairs()[0].execution_order[0];
    ledger
        .append(record(&plan, 0, game, 1, terminal(&plan, 0, game, 2)))
        .unwrap();
    reject_unchanged(&mut ledger, close(&plan, 0, 1));
    let summary = ledger.summary().unwrap();
    assert_eq!((summary.wins, summary.draws, summary.losses), (0, 0, 0));
    assert_eq!((summary.completed_pairs, summary.pending_pairs), (0, 1));
    let replay = Ledger::from_jsonl(&plan, &ledger.to_jsonl().unwrap(), limits()).unwrap();
    assert_eq!(replay.summary().unwrap(), summary);
}

#[test]
fn infrastructure_retry_replaces_entire_pair_without_salvaging_success() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let [first, second] = plan.pairs()[0].execution_order;
    ledger
        .append(record(&plan, 0, first, 1, terminal(&plan, 0, first, 2)))
        .unwrap();
    ledger
        .append(record(
            &plan,
            0,
            second,
            1,
            GameOutcome::InfrastructureInvalid {
                cause: "synthetic host interruption".into(),
                evidence: evidence(),
            },
        ))
        .unwrap();
    ledger.append(close(&plan, 0, 1)).unwrap();
    let excluded = ledger.summary().unwrap();
    assert_eq!((excluded.completed_pairs, excluded.excluded_pairs), (0, 1));
    assert_eq!((excluded.wins, excluded.draws, excluded.losses), (0, 0, 0));
    reject_unchanged(
        &mut ledger,
        Event::PairStarted {
            pair_id: plan.pairs()[0].id.clone(),
            attempt: 2,
            process_run_id: "synthetic-pair-0-attempt-1".into(),
        },
    );
    ledger.append(start(&plan, 0, 2)).unwrap();
    record_pair(&mut ledger, &plan, 0, 2, [1, 1]);
    let summary = ledger.summary().unwrap();
    assert_eq!((summary.completed_pairs, summary.excluded_pairs), (1, 0));
    assert_eq!(
        (
            summary.attempts,
            summary.closed_attempts,
            summary.recorded_games
        ),
        (2, 2, 4)
    );
    assert_eq!((summary.wins, summary.draws, summary.losses), (0, 2, 0));
    assert_eq!(summary.failure_counts.infrastructure_invalid, 1);
    reject_unchanged(&mut ledger, start(&plan, 0, 3));
}

#[test]
fn incomplete_pair_is_excluded_and_never_retried_or_scored_as_draw() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let [first, second] = plan.pairs()[0].execution_order;
    ledger
        .append(record(&plan, 0, first, 1, terminal(&plan, 0, first, 2)))
        .unwrap();
    ledger
        .append(record(
            &plan,
            0,
            second,
            1,
            GameOutcome::Incomplete {
                reason: "max-plies reached".into(),
                evidence: evidence(),
            },
        ))
        .unwrap();
    ledger.append(close(&plan, 0, 1)).unwrap();
    let summary = ledger.summary().unwrap();
    assert_eq!(summary.excluded_pairs, 1);
    assert_eq!(summary.failure_counts.incomplete, 1);
    assert_eq!((summary.wins, summary.draws, summary.losses), (0, 0, 0));
    reject_unchanged(&mut ledger, start(&plan, 0, 2));
}

#[test]
fn each_engine_failure_is_a_loss_not_an_infrastructure_retry() {
    for reason in [
        EngineFailureKind::IllegalMove,
        EngineFailureKind::Crash,
        EngineFailureKind::Timeout,
    ] {
        let plan = plan(1);
        let mut ledger = Ledger::new(&plan, limits()).unwrap();
        ledger.append(start(&plan, 0, 1)).unwrap();
        let [first, second] = plan.pairs()[0].execution_order;
        ledger
            .append(record(
                &plan,
                0,
                first,
                1,
                GameOutcome::EngineLoss {
                    loser_engine: plan.manifest().input().engines[0].id.clone(),
                    reason,
                    evidence: evidence(),
                },
            ))
            .unwrap();
        ledger
            .append(record(&plan, 0, second, 1, terminal(&plan, 0, second, 1)))
            .unwrap();
        ledger.append(close(&plan, 0, 1)).unwrap();
        let summary = ledger.summary().unwrap();
        assert_eq!((summary.wins, summary.draws, summary.losses), (0, 1, 1));
        assert_eq!(summary.n, [0, 1, 0, 0, 0]);
        assert_eq!(
            summary.failure_counts.illegal_move
                + summary.failure_counts.crash
                + summary.failure_counts.timeout,
            1
        );
        reject_unchanged(&mut ledger, start(&plan, 0, 2));
    }
}

#[test]
fn contract_fault_blocks_new_starts_but_preserves_pending_records() {
    for simultaneous in [false, true] {
        let plan = plan(2);
        let mut ledger = Ledger::new(&plan, limits()).unwrap();
        ledger.append(start(&plan, 0, 1)).unwrap();
        let [first, second] = plan.pairs()[0].execution_order;
        let outcome = if simultaneous {
            GameOutcome::SimultaneousFailure {
                reason: "both engines failed".into(),
                evidence: evidence(),
            }
        } else {
            GameOutcome::ContractInvalid {
                reason: "rules contract disagreement".into(),
                evidence: evidence(),
            }
        };
        ledger.append(record(&plan, 0, first, 1, outcome)).unwrap();
        reject_unchanged(&mut ledger, start(&plan, 1, 1));
        ledger
            .append(record(&plan, 0, second, 1, terminal(&plan, 0, second, 1)))
            .unwrap();
        ledger.append(close(&plan, 0, 1)).unwrap();
        let summary = ledger.summary().unwrap();
        assert!(summary.interpretation_blocked);
        assert_eq!(
            (
                summary.completed_pairs,
                summary.excluded_pairs,
                summary.pending_pairs
            ),
            (0, 1, 1)
        );
        assert_eq!(summary.draws, 0);
        assert_eq!(
            summary.failure_counts.contract_invalid + summary.failure_counts.simultaneous_failure,
            1
        );
    }
}

#[test]
fn attempt_game_order_membership_and_process_ids_are_strict_and_atomic() {
    let plan = plan(2);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    let [first, second] = plan.pairs()[0].execution_order;
    reject_unchanged(
        &mut ledger,
        record(&plan, 0, first, 1, terminal(&plan, 0, first, 1)),
    );
    reject_unchanged(&mut ledger, start(&plan, 0, 0));
    reject_unchanged(&mut ledger, start(&plan, 0, 2));
    ledger.append(start(&plan, 0, 1)).unwrap();
    reject_unchanged(&mut ledger, start(&plan, 0, 1));
    reject_unchanged(&mut ledger, start(&plan, 0, 2));
    reject_unchanged(
        &mut ledger,
        record(&plan, 0, second, 1, terminal(&plan, 0, second, 1)),
    );
    let wrong_pair_game = Event::GameRecorded {
        pair_id: plan.pairs()[0].id.clone(),
        attempt: 1,
        game_id: plan.pairs()[1].games[first].id.clone(),
        outcome: terminal(&plan, 1, first, 1),
    };
    reject_unchanged(&mut ledger, wrong_pair_game);
    ledger
        .append(record(&plan, 0, first, 1, terminal(&plan, 0, first, 1)))
        .unwrap();
    reject_unchanged(
        &mut ledger,
        record(&plan, 0, first, 1, terminal(&plan, 0, first, 2)),
    );
    ledger
        .append(record(&plan, 0, second, 1, terminal(&plan, 0, second, 1)))
        .unwrap();
    ledger.append(close(&plan, 0, 1)).unwrap();
    reject_unchanged(&mut ledger, close(&plan, 0, 1));
    reject_unchanged(
        &mut ledger,
        record(&plan, 0, second, 1, terminal(&plan, 0, second, 1)),
    );
    let reused = Event::PairStarted {
        pair_id: plan.pairs()[1].id.clone(),
        attempt: 1,
        process_run_id: "synthetic-pair-0-attempt-1".into(),
    };
    reject_unchanged(&mut ledger, reused);
}

#[test]
fn terminal_reason_result_consistency_and_adjudication_policy_are_checked() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let game = plan.pairs()[0].execution_order[0];
    for (result, reason) in [
        (GameResult::Draw, "checkmate"),
        (GameResult::WhiteWin, "stalemate"),
        (GameResult::Draw, "cancelled"),
    ] {
        reject_unchanged(
            &mut ledger,
            record(
                &plan,
                0,
                game,
                1,
                GameOutcome::RulesTerminal {
                    result,
                    reason: reason.into(),
                    evidence: evidence(),
                },
            ),
        );
    }
    reject_unchanged(
        &mut ledger,
        record(
            &plan,
            0,
            game,
            1,
            GameOutcome::ProtocolAdjudicated {
                result: GameResult::Draw,
                policy_id: "unlocked-policy".into(),
                evidence: evidence(),
            },
        ),
    );
    reject_unchanged(
        &mut ledger,
        record(
            &plan,
            0,
            game,
            1,
            GameOutcome::EngineLoss {
                loser_engine: "absent-engine".into(),
                reason: EngineFailureKind::Crash,
                evidence: evidence(),
            },
        ),
    );
}

#[test]
fn incomplete_or_engine_loss_cannot_be_hidden_by_an_infrastructure_retry() {
    for bad in [
        GameOutcome::Incomplete {
            reason: "cancelled".into(),
            evidence: evidence(),
        },
        GameOutcome::EngineLoss {
            loser_engine: "candidate-fixture".into(),
            reason: EngineFailureKind::Crash,
            evidence: evidence(),
        },
    ] {
        let plan = plan(1);
        let mut ledger = Ledger::new(&plan, limits()).unwrap();
        ledger.append(start(&plan, 0, 1)).unwrap();
        let [first, second] = plan.pairs()[0].execution_order;
        ledger.append(record(&plan, 0, first, 1, bad)).unwrap();
        ledger
            .append(record(
                &plan,
                0,
                second,
                1,
                GameOutcome::InfrastructureInvalid {
                    cause: "separate host failure".into(),
                    evidence: evidence(),
                },
            ))
            .unwrap();
        ledger.append(close(&plan, 0, 1)).unwrap();
        reject_unchanged(&mut ledger, start(&plan, 0, 2));
    }
}

#[test]
fn evidence_paths_identity_and_byte_budget_are_validated_atomically() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let [first, second] = plan.pairs()[0].execution_order;
    for invalid in [
        ArtifactRef {
            path: "../escape".into(),
            ..evidence()
        },
        ArtifactRef {
            sha256: "0".repeat(64),
            ..evidence()
        },
        ArtifactRef {
            bytes: plan.manifest().input().budget.max_artifact_bytes + 1,
            ..evidence()
        },
    ] {
        reject_unchanged(
            &mut ledger,
            record(
                &plan,
                0,
                first,
                1,
                GameOutcome::Incomplete {
                    reason: "synthetic reason".into(),
                    evidence: invalid,
                },
            ),
        );
    }
    ledger
        .append(record(&plan, 0, first, 1, terminal(&plan, 0, first, 1)))
        .unwrap();
    reject_unchanged(
        &mut ledger,
        record(
            &plan,
            0,
            second,
            1,
            GameOutcome::Incomplete {
                reason: "synthetic reason".into(),
                evidence: ArtifactRef {
                    sha256: "b".repeat(64),
                    ..evidence()
                },
            },
        ),
    );
    ledger
        .append(record(&plan, 0, second, 1, terminal(&plan, 0, second, 1)))
        .unwrap();
}

#[test]
fn hash_sequence_reordering_and_content_tampering_are_rejected() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    record_pair(&mut ledger, &plan, 0, 1, [1, 1]);
    let original = lines(&ledger.to_jsonl().unwrap());
    for mutation in 0..5 {
        let mut values = original.clone();
        match mutation {
            0 => values[1]["seq"] = 2.into(),
            1 => values[2]["prev_sha256"] = "0".repeat(64).into(),
            2 => values[2]["event_sha256"] = "0".repeat(64).into(),
            3 => values[1]["event"]["process_run_id"] = "tampered-process".into(),
            4 => values.swap(1, 2),
            _ => unreachable!(),
        }
        assert!(Ledger::from_jsonl(&plan, &jsonl(&values), limits()).is_err());
    }
}

#[test]
fn ledger_cannot_be_replayed_under_another_input_or_pair_plan() {
    let plan = plan(1);
    let input = Ledger::new(&plan, limits()).unwrap().to_jsonl().unwrap();
    let mut other_input = manifest(1);
    other_input.plan.seed += 1;
    let other = ArenaPlan::build(
        &other_input.lock().unwrap(),
        PlanLimits {
            max_pairs: 1,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap();
    assert!(Ledger::from_jsonl(&other, &input, limits()).is_err());
    for field in [
        "ledger_version",
        "canonicalization",
        "execution_ready",
        "input_sha256",
        "plan_sha256",
    ] {
        let mut values = lines(&input);
        values[0][field] = match field {
            "ledger_version" => 2.into(),
            "execution_ready" => true.into(),
            _ => "different".into(),
        };
        assert!(Ledger::from_jsonl(&plan, &jsonl(&values), limits()).is_err());
    }
}

#[test]
fn truncated_missing_newline_duplicate_and_unknown_json_fields_are_rejected() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let input = ledger.to_jsonl().unwrap();
    assert!(Ledger::from_jsonl(&plan, input.trim_end(), limits()).is_err());
    assert!(Ledger::from_jsonl(&plan, &input[..input.len() - 5], limits()).is_err());
    assert!(Ledger::from_jsonl(&plan, "", limits()).is_err());
    assert!(Ledger::from_jsonl(&plan, &(input.clone() + "\n"), limits()).is_err());
    let duplicate = input.replacen(
        "\"ledger_version\":1",
        "\"ledger_version\":1,\"ledger_version\":1",
        1,
    );
    assert!(Ledger::from_jsonl(&plan, &duplicate, limits()).is_err());
    let mut values = lines(&input);
    values[1]["event"]["unknown"] = true.into();
    assert!(Ledger::from_jsonl(&plan, &jsonl(&values), limits()).is_err());
}

#[test]
fn event_and_output_budgets_reject_without_mutating_the_previous_ledger() {
    let plan = plan(1);
    assert!(
        Ledger::new(
            &plan,
            LedgerLimits {
                max_events: 0,
                max_bytes: 1000
            }
        )
        .is_err()
    );
    assert!(
        Ledger::new(
            &plan,
            LedgerLimits {
                max_events: 1,
                max_bytes: 0
            }
        )
        .is_err()
    );
    assert!(
        Ledger::new(
            &plan,
            LedgerLimits {
                max_events: 1,
                max_bytes: 1
            }
        )
        .is_err()
    );
    let mut ledger = Ledger::new(
        &plan,
        LedgerLimits {
            max_events: 1,
            max_bytes: 10_000,
        },
    )
    .unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let game = plan.pairs()[0].execution_order[0];
    reject_unchanged(
        &mut ledger,
        record(&plan, 0, game, 1, terminal(&plan, 0, game, 1)),
    );
    let bytes = Ledger::new(&plan, limits())
        .unwrap()
        .to_jsonl()
        .unwrap()
        .len() as u64;
    let mut header_only = Ledger::new(
        &plan,
        LedgerLimits {
            max_events: 100,
            max_bytes: bytes,
        },
    )
    .unwrap();
    reject_unchanged(&mut header_only, start(&plan, 0, 1));
    let input = ledger.to_jsonl().unwrap();
    assert!(
        Ledger::from_jsonl(
            &plan,
            &input,
            LedgerLimits {
                max_events: 100,
                max_bytes: input.len() as u64 - 1
            }
        )
        .is_err()
    );
}

#[test]
fn input_budget_is_enforced_even_if_caller_requests_more_output_bytes() {
    let plan = plan(1);
    let input = Ledger::new(&plan, limits()).unwrap().to_jsonl().unwrap();
    let padded = format!("{}{}", " ".repeat(1_048_576), input);
    assert!(
        Ledger::from_jsonl(
            &plan,
            &padded,
            LedgerLimits {
                max_events: 100,
                max_bytes: u64::MAX
            }
        )
        .is_err()
    );
}

#[test]
fn trusted_tip_detects_deletion_of_a_complete_suffix() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    let header_tip = ledger.tip_sha256().to_owned();
    ledger.verify_tip(&header_tip).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    record_pair(&mut ledger, &plan, 0, 1, [1, 1]);
    let tip = ledger.tip_sha256().to_owned();
    assert_ne!(tip, header_tip);
    ledger.verify_tip(&tip).unwrap();
    let input = ledger.to_jsonl().unwrap();
    let mut values = lines(&input);
    values.pop(); // Remove the entire PairClosed record, retaining complete JSONL.
    let prefix = Ledger::from_jsonl(&plan, &jsonl(&values), limits()).unwrap();
    assert_eq!(prefix.summary().unwrap().completed_pairs, 0);
    assert_eq!(prefix.summary().unwrap().pending_pairs, 1);
    assert!(prefix.verify_tip(&tip).is_err());
    assert!(prefix.verify_tip(&"0".repeat(64)).is_err());
    Ledger::from_jsonl(&plan, &input, limits())
        .unwrap()
        .verify_tip(&tip)
        .unwrap();
}

#[test]
fn an_adjudication_boolean_never_unlocks_an_unspecified_policy() {
    let mut input = manifest(1);
    input.protocol.adjudication_enabled = true;
    let plan = ArenaPlan::build(
        &input.lock().unwrap(),
        PlanLimits {
            max_pairs: 1,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap();
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let game = plan.pairs()[0].execution_order[0];
    for policy_id in [
        "tablebase-win",
        "evaluation-adjudication",
        "automatic-draw-claim",
    ] {
        reject_unchanged(
            &mut ledger,
            record(
                &plan,
                0,
                game,
                1,
                GameOutcome::ProtocolAdjudicated {
                    result: GameResult::WhiteWin,
                    policy_id: policy_id.into(),
                    evidence: evidence(),
                },
            ),
        );
    }
}

#[test]
fn large_direct_fields_are_rejected_before_hashing_or_serialization() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let game = plan.pairs()[0].execution_order[0];
    let event = record(
        &plan,
        0,
        game,
        1,
        GameOutcome::Incomplete {
            reason: "x".repeat(4 * 1024 * 1024 + 1),
            evidence: evidence(),
        },
    );
    let before = ledger.to_jsonl().unwrap();
    assert!(matches!(
        ledger.append(event),
        Err(rz_arena::ArenaError::Invalid(_))
    ));
    assert_eq!(before, ledger.to_jsonl().unwrap());
}

#[test]
fn unique_evidence_refs_share_the_aggregate_artifact_budget() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let [first, second] = plan.pairs()[0].execution_order;
    let cap = plan.manifest().input().budget.max_artifact_bytes;
    ledger
        .append(record(
            &plan,
            0,
            first,
            1,
            GameOutcome::Incomplete {
                reason: "synthetic first log".into(),
                evidence: ArtifactRef {
                    bytes: cap - input_artifact_bytes(&plan) - 1,
                    ..evidence()
                },
            },
        ))
        .unwrap();
    reject_unchanged(
        &mut ledger,
        record(
            &plan,
            0,
            second,
            1,
            GameOutcome::Incomplete {
                reason: "synthetic second log".into(),
                evidence: ArtifactRef {
                    path: "evidence/second.json".into(),
                    bytes: 2,
                    ..evidence()
                },
            },
        ),
    );
}

#[test]
fn input_artifact_identity_cannot_be_replaced_by_an_evidence_declaration() {
    let plan = plan(1);
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let game = plan.pairs()[0].execution_order[0];
    let original = plan.manifest().declared_artifacts()[0].clone();
    let mut conflicting = original.clone();
    conflicting.sha256 = "b".repeat(64);
    assert_ne!(conflicting.sha256, original.sha256);
    reject_unchanged(
        &mut ledger,
        record(
            &plan,
            0,
            game,
            1,
            GameOutcome::Incomplete {
                reason: "synthetic outcome".into(),
                evidence: conflicting,
            },
        ),
    );
}

#[test]
fn exact_input_identity_reuse_does_not_charge_the_artifact_budget_twice() {
    let original_plan = plan(1);
    let mut input = manifest(1);
    input.budget.max_artifact_bytes = input_artifact_bytes(&original_plan);
    let plan = ArenaPlan::build(
        &input.lock().unwrap(),
        PlanLimits {
            max_pairs: 1,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap();
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let original = plan.manifest().declared_artifacts()[0].clone();
    for game in plan.pairs()[0].execution_order {
        ledger
            .append(record(
                &plan,
                0,
                game,
                1,
                GameOutcome::Incomplete {
                    reason: "synthetic declaration referencing exact input identity".into(),
                    evidence: original.clone(),
                },
            ))
            .unwrap();
    }
    ledger.append(close(&plan, 0, 1)).unwrap();
    assert_eq!(ledger.summary().unwrap().failure_counts.incomplete, 2);
    Ledger::from_jsonl(&plan, &ledger.to_jsonl().unwrap(), limits()).unwrap();
}

#[test]
fn input_and_new_evidence_share_one_artifact_budget() {
    let original_plan = plan(1);
    let mut input = manifest(1);
    input.budget.max_artifact_bytes = input_artifact_bytes(&original_plan) + 1;
    let plan = ArenaPlan::build(
        &input.lock().unwrap(),
        PlanLimits {
            max_pairs: 1,
            max_json_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap();
    let mut ledger = Ledger::new(&plan, limits()).unwrap();
    ledger.append(start(&plan, 0, 1)).unwrap();
    let game = plan.pairs()[0].execution_order[0];
    reject_unchanged(
        &mut ledger,
        record(
            &plan,
            0,
            game,
            1,
            GameOutcome::Incomplete {
                reason: "synthetic second artifact".into(),
                evidence: ArtifactRef {
                    bytes: 2,
                    ..evidence()
                },
            },
        ),
    );
    ledger
        .append(record(
            &plan,
            0,
            game,
            1,
            GameOutcome::Incomplete {
                reason: "synthetic artifact within total budget".into(),
                evidence: evidence(),
            },
        ))
        .unwrap();
}
