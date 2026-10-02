use rz_arena::{ArenaError, ArenaPlan, PlanLimits};
use rz_experiments::{HistoryCompleteness, InitialPosition, RunManifest};
use serde_json::{Value, json};

const FIXTURE: &str = include_str!("../../../experiments/baselines/fixtures/e01-input.json");
const INPUT_SHA256: &str = "3763c90ba72e8bf412a84bd187fa2d6e7973994e0a31df3c2c3795075493e567";
const OPENING_SHA256: &str = "2bfb56da02967f952312886e257018974843fb0ea9a81fa5b0265e94f4240e46";
const PLAN_SHA256: &str = "980d61aadc3da58fe75720b125f7017a08ad40c5ae58c490f6e411e89429511f";

fn fixture() -> RunManifest {
    RunManifest::from_json(FIXTURE).expect("parse E01 synthetic fixture")
}

fn limits() -> PlanLimits {
    PlanLimits {
        max_pairs: 32,
        max_json_bytes: 4 * 1024 * 1024,
    }
}

fn pairs(mut input: RunManifest, count: u64) -> RunManifest {
    input.plan.pairs = count;
    input.budget.max_pairs = count;
    input.budget.max_games = count.checked_mul(4).expect("bounded test pair count");
    input
}

fn plan(input: RunManifest) -> ArenaPlan {
    ArenaPlan::build(&input.lock().expect("lock test input"), limits()).expect("build test plan")
}

#[test]
fn pair_changes_engine_colors_and_preserves_the_opening() {
    let input = fixture();
    let original = input.input.openings[0].clone();
    let planned = plan(input);
    assert_eq!(planned.manifest().sha256(), INPUT_SHA256);
    // Independently generated with the complete one-pair Python payload.
    assert_eq!(planned.sha256(), PLAN_SHA256);
    assert_eq!(planned.pairs().len(), 1);
    let pair = &planned.pairs()[0];
    assert_eq!(pair.id, "fixture-001/pair-0");
    assert_eq!(pair.ordinal, 0);
    assert_eq!(pair.opening, original);
    assert_eq!(pair.opening_input_sha256, OPENING_SHA256);
    assert_eq!(pair.games[0].id, "fixture-001/pair-0/game-0");
    assert_eq!(pair.games[0].white_engine, "candidate-fixture");
    assert_eq!(pair.games[0].black_engine, "baseline-fixture");
    assert_eq!(pair.games[1].white_engine, "baseline-fixture");
    assert_eq!(pair.games[1].black_engine, "candidate-fixture");
    assert_eq!(planned.pair(&pair.id), Some(pair));
    assert!(planned.pair("unknown-pair").is_none());
    assert_eq!(pair.execution_order, [1, 0]);
    for game in &pair.games {
        assert_eq!(game.engine_slots["candidate-fixture"], 1);
        assert_eq!(game.engine_slots["baseline-fixture"], 0);
    }
}

#[test]
fn seed_derivation_matches_independent_python_big_endian_sha256_vectors() {
    // Python reference hashes sorted compact UTF-8 SeedIdentity JSON, then
    // int.from_bytes(hashlib.sha256(...).digest()[:8], 'big').
    let planned = plan(pairs(fixture(), 2));
    let expected = [
        [
            [6680294958970736159_u64, 256420853447138087_u64],
            [962006781126052703_u64, 2558959215239338756_u64],
        ],
        [
            [6943108029034343453_u64, 17434321812138447844_u64],
            [17295385335100807678_u64, 13895428471914316497_u64],
        ],
    ];
    for (pair, expected_games) in planned.pairs().iter().zip(expected) {
        for (game, expected_seeds) in pair.games.iter().zip(expected_games) {
            assert_eq!(game.engine_seeds["candidate-fixture"], expected_seeds[0]);
            assert_eq!(game.engine_seeds["baseline-fixture"], expected_seeds[1]);
        }
    }
}

#[test]
fn declared_openings_repeat_round_robin_and_alias_ids_share_one_input_cluster() {
    let mut input = pairs(fixture(), 5);
    let mut alias = input.input.openings[0].clone();
    alias.id = "alias-opening".into();
    input.input.openings.push(alias);
    let planned = plan(input);
    let expected = [
        "startpos-fixture",
        "alias-opening",
        "startpos-fixture",
        "alias-opening",
        "startpos-fixture",
    ];
    for (pair, expected_id) in planned.pairs().iter().zip(expected) {
        assert_eq!(pair.opening.id, expected_id);
        assert_eq!(pair.opening_input_sha256, OPENING_SHA256);
    }
}

#[test]
fn black_to_move_fen_rights_ep_counters_unknown_history_and_promotion_tokens_survive() {
    let mut input = fixture();
    let opening = &mut input.input.openings[0];
    opening.initial = InitialPosition::Fen;
    opening.fen = Some("r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 83".into());
    opening.history = HistoryCompleteness::UnknownPrefix;
    opening.history_origin = "known-midgame-prefix".into();
    // These are syntax vectors, not a claim that this trace is legal from FEN.
    opening.moves = ["a2a1q", "a7a8r", "b2b1b", "b7b8n"]
        .map(String::from)
        .to_vec();
    let expected = opening.clone();
    let planned = plan(input);
    assert_eq!(planned.pairs()[0].opening, expected);
    assert_eq!(planned.pairs()[0].games[0].black_engine, "baseline-fixture");
    assert_eq!(
        planned.pairs()[0].games[1].black_engine,
        "candidate-fixture"
    );
    let restored = ArenaPlan::from_json(&planned.to_json().unwrap(), limits()).unwrap();
    assert_eq!(restored.pairs()[0].opening, expected);
}

#[test]
fn history_fill_repetition_and_full_trace_changes_separate_opening_input_clusters() {
    let original = plan(fixture()).pairs()[0].opening_input_sha256.clone();
    let mut variants = [fixture(), fixture(), fixture(), fixture()];
    variants[0].input.history_fill_policy = "different-history-fill".into();
    variants[1].input.repetition_policy = "different-repetition-policy".into();
    variants[2].input.openings[0].history_origin = "different-known-origin".into();
    variants[3].input.openings[0].moves.push("g1f3".into());
    for variant in variants {
        assert_ne!(plan(variant).pairs()[0].opening_input_sha256, original);
    }
}

#[test]
fn alternating_execution_and_logical_slots_are_balanced_without_seed_overflow() {
    let mut input = pairs(fixture(), 2);
    input.plan.seed = u64::MAX;
    let planned = plan(input);
    assert_eq!(planned.pairs()[0].execution_order, [1, 0]);
    assert_eq!(planned.pairs()[1].execution_order, [0, 1]);
    for (index, pair) in planned.pairs().iter().enumerate() {
        let candidate_slot = if index == 0 { 1 } else { 0 };
        assert_eq!(pair.games[0].engine_slots, pair.games[1].engine_slots);
        assert_eq!(
            pair.games[0].engine_slots["candidate-fixture"],
            candidate_slot
        );
        assert_eq!(
            pair.games[0].engine_slots["baseline-fixture"],
            1 - candidate_slot
        );
    }
}

#[test]
fn repeated_builds_and_whole_pair_retry_policy_keep_the_same_game_seeds() {
    let original = fixture();
    let planned = plan(original.clone());
    assert_eq!(
        plan(original.clone()).to_json().unwrap(),
        planned.to_json().unwrap()
    );
    let mut fewer_retries = original;
    fewer_retries.plan.max_retries_per_pair = 0;
    let different_lock = plan(fewer_retries);
    assert_eq!(different_lock.pairs(), planned.pairs());
    assert_ne!(different_lock.sha256(), planned.sha256());
}

#[test]
fn changing_each_locked_seed_component_changes_the_derived_engine_seed() {
    let original_seed = plan(fixture()).pairs()[0].games[0].engine_seeds["candidate-fixture"];
    let mut variants = [fixture(), fixture(), fixture()];
    variants[0].plan.seed += 1;
    variants[1].input.seed += 1;
    *variants[2]
        .plan
        .engine_seeds
        .get_mut("candidate-fixture")
        .unwrap() += 1;
    for variant in variants {
        assert_ne!(
            plan(variant).pairs()[0].games[0].engine_seeds["candidate-fixture"],
            original_seed
        );
    }
}

#[test]
fn unsupported_selection_order_resume_and_stop_policies_are_explicitly_rejected() {
    let mut variants = [fixture(), fixture(), fixture(), fixture()];
    variants[0].input.selection_policy = "random-after-observing-results".into();
    variants[1].plan.hardware_order = "unbalanced".into();
    variants[2].plan.incomplete_pair_policy = "keep-only-winning-game".into();
    variants[3].plan.stop_policy = "unbounded".into();
    for variant in variants {
        let locked = variant.lock().unwrap();
        assert!(matches!(
            ArenaPlan::build(&locked, limits()),
            Err(ArenaError::Invalid(_))
        ));
    }
}

#[test]
fn explicit_pair_json_limits_and_cumulative_repeated_opening_bytes_are_enforced() {
    let one = plan(fixture());
    let exact_bytes = one.to_json().unwrap().len();
    let exact = PlanLimits {
        max_pairs: 1,
        max_json_bytes: exact_bytes,
    };
    ArenaPlan::build(one.manifest(), exact).unwrap();
    ArenaPlan::from_json(&one.to_json().unwrap(), exact).unwrap();
    let too_small = PlanLimits {
        max_json_bytes: exact_bytes - 1,
        ..exact
    };
    assert!(matches!(
        ArenaPlan::build(one.manifest(), too_small),
        Err(ArenaError::Budget(_))
    ));
    assert!(matches!(
        ArenaPlan::from_json(&one.to_json().unwrap(), too_small),
        Err(ArenaError::Budget(_))
    ));
    for bad in [
        PlanLimits {
            max_pairs: 0,
            ..limits()
        },
        PlanLimits {
            max_json_bytes: 0,
            ..limits()
        },
        PlanLimits {
            max_json_bytes: 4 * 1024 * 1024 + 1,
            ..limits()
        },
    ] {
        assert!(matches!(
            ArenaPlan::build(one.manifest(), bad),
            Err(ArenaError::Budget(_))
        ));
    }
    let two = pairs(fixture(), 2).lock().unwrap();
    assert!(matches!(
        ArenaPlan::build(&two, exact),
        Err(ArenaError::Budget(_))
    ));

    let mut huge_repetition = fixture();
    huge_repetition.input.openings[0].moves = vec!["e2e4".into(); 800];
    let single_large = plan(huge_repetition.clone());
    let large_limit = PlanLimits {
        max_pairs: u64::MAX,
        max_json_bytes: single_large.to_json().unwrap().len(),
    };
    huge_repetition.plan.pairs = u64::MAX / 4;
    huge_repetition.plan.max_retries_per_pair = 0;
    huge_repetition.budget.max_pairs = u64::MAX;
    huge_repetition.budget.max_games = u64::MAX;
    let locked = huge_repetition.lock().unwrap();
    // This fails on the second retained pair instead of reserving u64::MAX/4
    // pairs or repeatedly cloning the large opening without a byte ceiling.
    assert!(matches!(
        ArenaPlan::build(&locked, large_limit),
        Err(ArenaError::Budget(_))
    ));
}

#[test]
fn compact_plan_budget_accepts_a_large_input_despite_pretty_formatting_expansion() {
    let mut input = fixture();
    let opening = input.input.openings[0].clone();
    input.input.openings = (0..64)
        .map(|ordinal| {
            let mut repeated = opening.clone();
            repeated.id = format!("large-opening-{ordinal}");
            repeated.moves = vec!["e2e4".into(); 4096];
            repeated
        })
        .collect();
    let locked = input.lock().unwrap();
    assert!(locked.to_json().unwrap().len() > 4 * 1024 * 1024);
    let budget = PlanLimits {
        max_pairs: 1,
        max_json_bytes: 3 * 1024 * 1024,
    };
    let planned = ArenaPlan::build(&locked, budget)
        .expect("compact serialization stays within the explicit budget");
    let encoded = planned.to_json().unwrap();
    assert!(encoded.len() <= budget.max_json_bytes);
    assert_eq!(planned.pairs().len(), 1);
    let restored = ArenaPlan::from_json(&encoded, budget).unwrap();
    assert_eq!(restored.sha256(), planned.sha256());
}

#[test]
fn plan_roundtrip_preserves_the_locked_input_and_remains_unready() {
    let planned = plan(pairs(fixture(), 3));
    let encoded = planned.to_json().unwrap();
    let value: Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(value["execution_ready"], false);
    assert_eq!(value["input_lock"]["execution_ready"], false);
    let restored = ArenaPlan::from_json(&encoded, limits()).unwrap();
    assert_eq!(restored.sha256(), planned.sha256());
    assert_eq!(restored.manifest().sha256(), planned.manifest().sha256());
    assert_eq!(restored.pairs(), planned.pairs());
}

#[test]
fn tampered_pair_assignments_seeds_openings_order_and_digests_are_rejected() {
    let original: Value = serde_json::from_str(&plan(fixture()).to_json().unwrap()).unwrap();
    for (pointer, replacement) in [
        ("/pairs/0/games/0/white_engine", json!("baseline-fixture")),
        ("/pairs/0/games/0/engine_seeds/candidate-fixture", json!(0)),
        ("/pairs/0/games/0/engine_slots/candidate-fixture", json!(0)),
        ("/pairs/0/opening/moves/0", json!("d2d4")),
        ("/pairs/0/opening_input_sha256", json!("1".repeat(64))),
        ("/pairs/0/execution_order", json!([0, 1])),
        ("/input_sha256", json!("1".repeat(64))),
        ("/plan_sha256", json!("1".repeat(64))),
        ("/execution_ready", json!(true)),
        ("/plan_version", json!(2)),
        ("/algorithm", json!("different-algorithm")),
    ] {
        let mut altered = original.clone();
        *altered.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            matches!(
                ArenaPlan::from_json(&altered.to_string(), limits()),
                Err(ArenaError::Integrity(_))
            ),
            "accepted tampering at {pointer}"
        );
    }
    let mut altered_lock = original;
    altered_lock["input_lock"]["input"]["plan"]["seed"] = json!(99);
    assert!(matches!(
        ArenaPlan::from_json(&altered_lock.to_string(), limits()),
        Err(ArenaError::Manifest(_))
    ));
}

#[test]
fn a_recomputed_checksum_cannot_authorize_an_inconsistent_derived_pair() {
    use sha2::{Digest, Sha256};

    let original: Value = serde_json::from_str(&plan(fixture()).to_json().unwrap()).unwrap();
    for (pointer, replacement) in [
        ("/pairs/0/games/0/engine_seeds/candidate-fixture", json!(0)),
        ("/pairs/0/opening/moves/0", json!("d2d4")),
        ("/pairs/0/execution_order", json!([0, 1])),
    ] {
        let mut attack = original.clone();
        *attack.pointer_mut(pointer).unwrap() = replacement;
        attack.as_object_mut().unwrap().remove("plan_sha256");
        attack.sort_all_objects();
        let checksum = format!("{:x}", Sha256::digest(serde_json::to_vec(&attack).unwrap()));
        attack["plan_sha256"] = json!(checksum);
        assert!(matches!(
            ArenaPlan::from_json(&attack.to_string(), limits()),
            Err(ArenaError::Integrity(_))
        ));
    }
}

#[test]
fn duplicate_keys_and_unknown_fields_cannot_hide_plan_or_game_changes() {
    let planned = plan(fixture()).to_json().unwrap();
    let duplicate = planned.replacen('{', "{\"plan_version\":1,", 1);
    assert!(ArenaPlan::from_json(&duplicate, limits()).is_err());
    for location in ["", "/pairs/0", "/pairs/0/games/0"] {
        let mut extra: Value = serde_json::from_str(&planned).unwrap();
        extra.pointer_mut(location).unwrap()["hidden_change"] = json!(true);
        assert!(ArenaPlan::from_json(&extra.to_string(), limits()).is_err());
    }
}
