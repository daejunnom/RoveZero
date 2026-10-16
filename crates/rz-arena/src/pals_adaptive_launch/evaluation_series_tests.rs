//! Pure authored result/statistics fixtures only. No fixture grants native-run
//! custody, Rules replay, clock acceptance, or actual evaluation authority.

use super::{pair_half_points, summarize_scores};
use rz_experiments::{
    ArtifactRef, PalsGameReceiptV5, PalsObservedV3, PalsResultV3, PalsTerminationV3,
};

fn evidence(label: &str) -> ArtifactRef {
    ArtifactRef {
        path: format!("authored-evaluation-fixture/{label}.json"),
        sha256: "a".repeat(64),
        bytes: 1,
        source: "https://example.org/authored-pure-statistics-fixture".into(),
        license: "MIT".into(),
    }
}

fn game(index: u32, white: &str, black: &str, result: PalsResultV3) -> PalsGameReceiptV5 {
    let termination = match result {
        PalsResultV3::Draw => PalsTerminationV3::RulesDraw,
        PalsResultV3::Incomplete => PalsTerminationV3::PlyLimit,
        _ => PalsTerminationV3::Checkmate,
    };
    PalsGameReceiptV5 {
        game_index: index,
        white_endpoint: white.into(),
        black_endpoint: black.into(),
        base_ms: 120_000,
        increment_ms: 1_000,
        plies: if result == PalsResultV3::Incomplete {
            256
        } else {
            17
        },
        result,
        termination,
        failed_endpoint: None,
        pgn: evidence("raw-pgn"),
        rules_clock_pgn_check: PalsObservedV3::Observed {
            value: evidence("parent-rules-clock"),
            method: "authored pure conversion fixture; not actual run proof".into(),
        },
    }
}

fn pair(first: PalsResultV3, second: PalsResultV3) -> [PalsGameReceiptV5; 2] {
    [game(0, "g0", "g1", first), game(1, "g1", "g0", second)]
}

fn failed_game(index: u32, termination: PalsTerminationV3) -> PalsGameReceiptV5 {
    let mut receipt = if index == 0 {
        game(0, "g0", "g1", PalsResultV3::WhiteWin)
    } else {
        game(1, "g1", "g0", PalsResultV3::BlackWin)
    };
    receipt.termination = termination;
    receipt.failed_endpoint = Some("g1".into());
    receipt
}

#[test]
fn candidate_g1_scores_use_actual_colors_for_all_five_pair_bins() {
    use PalsResultV3::{BlackWin as B, Draw as D, WhiteWin as W};
    for (first, second, expected) in [
        (W, B, 0),
        (D, B, 1),
        (D, D, 2),
        (B, B, 2),
        (B, D, 3),
        (B, W, 4),
    ] {
        assert_eq!(
            pair_half_points(&pair(first, second), "g0", "g1").unwrap(),
            Some(expected)
        );
    }
}

#[test]
fn swapped_candidate_argument_changes_viewpoint_without_reversing_the_board() {
    let games = pair(PalsResultV3::BlackWin, PalsResultV3::WhiteWin);
    assert_eq!(pair_half_points(&games, "g0", "g1").unwrap(), Some(4));
    assert_eq!(pair_half_points(&games, "g1", "g0").unwrap(), Some(0));
}

#[test]
fn actual_paired_color_exchange_allows_candidate_white_in_the_first_game() {
    let games = [
        game(0, "g1", "g0", PalsResultV3::WhiteWin),
        game(1, "g0", "g1", PalsResultV3::BlackWin),
    ];
    assert_eq!(pair_half_points(&games, "g0", "g1").unwrap(), Some(4));
}

#[test]
fn engine_losses_remain_scored_and_are_not_removed_from_the_pair() {
    for termination in [
        PalsTerminationV3::IllegalMove,
        PalsTerminationV3::EngineCrash,
        PalsTerminationV3::TimeForfeit,
    ] {
        let games = [failed_game(0, termination), failed_game(1, termination)];
        assert_eq!(pair_half_points(&games, "g0", "g1").unwrap(), Some(0));
        assert_eq!(pair_half_points(&games, "g1", "g0").unwrap(), Some(4));
    }
}

#[test]
fn failure_offender_must_match_an_endpoint_and_the_declared_losing_side() {
    for offender in [None, Some("outsider"), Some("g0")] {
        let mut games = [
            failed_game(0, PalsTerminationV3::EngineCrash),
            game(1, "g1", "g0", PalsResultV3::Draw),
        ];
        games[0].failed_endpoint = offender.map(str::to_owned);
        assert!(pair_half_points(&games, "g0", "g1").is_err());
    }
    let mut games = [
        failed_game(0, PalsTerminationV3::EngineCrash),
        game(1, "g1", "g0", PalsResultV3::Draw),
    ];
    games[0].result = PalsResultV3::Draw;
    assert!(pair_half_points(&games, "g0", "g1").is_err());
}

#[test]
fn incomplete_pair_retains_attempt_but_never_becomes_a_draw_or_half_pair_score() {
    for games in [
        pair(PalsResultV3::Incomplete, PalsResultV3::WhiteWin),
        pair(PalsResultV3::BlackWin, PalsResultV3::Incomplete),
        pair(PalsResultV3::Incomplete, PalsResultV3::Incomplete),
    ] {
        assert_eq!(pair_half_points(&games, "g0", "g1").unwrap(), None);
    }
}

#[test]
fn incomplete_result_does_not_bypass_clock_color_or_audit_validation() {
    let mut games = pair(PalsResultV3::Incomplete, PalsResultV3::Draw);
    games[1].base_ms = 119_999;
    assert!(pair_half_points(&games, "g0", "g1").is_err());
    games[1].base_ms = 120_000;
    games[1].rules_clock_pgn_check = PalsObservedV3::Unknown;
    assert!(pair_half_points(&games, "g0", "g1").is_err());
    games[1].rules_clock_pgn_check = PalsObservedV3::Observed {
        value: evidence("parent-rules-clock"),
        method: "authored fixture".into(),
    };
    games[1].white_endpoint = "g0".into();
    games[1].black_endpoint = "g1".into();
    assert!(pair_half_points(&games, "g0", "g1").is_err());
}

#[test]
fn exact_two_games_distinct_endpoints_and_color_exchange_are_required() {
    let games = pair(PalsResultV3::Draw, PalsResultV3::Draw);
    assert!(pair_half_points(&[], "g0", "g1").is_err());
    assert!(pair_half_points(&games[..1], "g0", "g1").is_err());
    let extra = [games[0].clone(), games[1].clone(), games[0].clone()];
    assert!(pair_half_points(&extra, "g0", "g1").is_err());
    assert!(pair_half_points(&games, "g0", "g0").is_err());
    assert!(pair_half_points(&games, "", "g1").is_err());
    let mut wrong = games.clone();
    wrong[1].white_endpoint = "outsider".into();
    assert!(pair_half_points(&wrong, "g0", "g1").is_err());
    wrong = games.clone();
    wrong[1].white_endpoint = "g0".into();
    wrong[1].black_endpoint = "g1".into();
    assert!(pair_half_points(&wrong, "g0", "g1").is_err());
    wrong = games;
    wrong[1].game_index = 0;
    assert!(pair_half_points(&wrong, "g0", "g1").is_err());
}

#[test]
fn fixed_fischer_clock_ply_ceiling_and_observed_audit_are_mandatory() {
    let games = pair(PalsResultV3::Draw, PalsResultV3::Draw);
    for index in 0..2 {
        let mut wrong = games.clone();
        wrong[index].base_ms = 120_001;
        assert!(pair_half_points(&wrong, "g0", "g1").is_err());
        wrong = games.clone();
        wrong[index].increment_ms = 999;
        assert!(pair_half_points(&wrong, "g0", "g1").is_err());
        wrong = games.clone();
        wrong[index].plies = 257;
        assert!(pair_half_points(&wrong, "g0", "g1").is_err());
        wrong = games.clone();
        wrong[index].rules_clock_pgn_check = PalsObservedV3::Unknown;
        assert!(pair_half_points(&wrong, "g0", "g1").is_err());
    }
    let mut boundary = games;
    boundary[0].plies = 256;
    boundary[1].plies = 256;
    assert_eq!(pair_half_points(&boundary, "g0", "g1").unwrap(), Some(2));
}

#[test]
fn empty_or_partial_series_has_no_final_interval_and_preserves_attempted_count() {
    let empty = summarize_scores(&[], 32).unwrap();
    assert_eq!(empty.attempted_pairs, 0);
    assert_eq!(empty.completed_pairs, 0);
    assert_eq!(empty.counts, [0; 5]);
    assert_eq!(empty.score_rate, None);
    assert_eq!(empty.final_interval_95, None);

    let partial = summarize_scores(&[Some(4), None, Some(0)], 32).unwrap();
    assert_eq!(partial.attempted_pairs, 3);
    assert_eq!(partial.completed_pairs, 2);
    assert_eq!(partial.counts, [1, 0, 0, 0, 1]);
    assert_eq!(partial.score_rate, Some(0.5));
    assert_eq!(partial.final_interval_95, None);
}

#[test]
fn thirty_two_attempts_with_an_unscored_pair_cannot_claim_final_coverage() {
    let mut scores = vec![Some(2); 32];
    scores[7] = None;
    let summary = summarize_scores(&scores, 32).unwrap();
    assert_eq!(summary.attempted_pairs, 32);
    assert_eq!(summary.completed_pairs, 31);
    assert_eq!(summary.counts, [0, 0, 31, 0, 0]);
    assert_eq!(summary.score_rate, Some(0.5));
    assert_eq!(summary.final_interval_95, None);
}

#[test]
fn fixed_thirty_two_pairs_and_bounded_pair_scores_cannot_be_changed_after_results() {
    for expected in [0, 1, 31, 33, usize::MAX] {
        assert!(summarize_scores(&[], expected).is_err());
    }
    assert!(summarize_scores(&[Some(2); 33], 32).is_err());
    for score in [5, u8::MAX] {
        assert!(summarize_scores(&[Some(score)], 32).is_err());
    }
}

#[test]
fn complete_half_score_series_uses_predeclared_pair_cluster_hoeffding_radius() {
    let scores = vec![Some(2); 32];
    let summary = summarize_scores(&scores, 32).unwrap();
    assert_eq!(summary.attempted_pairs, 32);
    assert_eq!(summary.completed_pairs, 32);
    assert_eq!(summary.counts, [0, 0, 32, 0, 0]);
    assert_eq!(summary.score_rate, Some(0.5));
    let interval = summary.final_interval_95.unwrap();
    let epsilon = (40.0_f64.ln() / (2.0 * 32.0)).sqrt();
    assert!((interval[0] - (0.5 - epsilon)).abs() < 1e-12);
    assert!((interval[1] - (0.5 + epsilon)).abs() < 1e-12);
    // The independence unit is the opening pair, not its two games.
    let wrong_game_radius = (40.0_f64.ln() / (2.0 * 64.0)).sqrt();
    assert!((interval[1] - (0.5 + wrong_game_radius)).abs() > 0.01);
}

#[test]
fn zero_and_full_score_hoeffding_intervals_clamp_without_claiming_certainty() {
    let epsilon = (40.0_f64.ln() / 64.0).sqrt();
    let zero = summarize_scores(&[Some(0); 32], 32).unwrap();
    assert_eq!(zero.score_rate, Some(0.0));
    assert_eq!(zero.counts, [32, 0, 0, 0, 0]);
    let zero_ci = zero.final_interval_95.unwrap();
    assert_eq!(zero_ci[0], 0.0);
    assert!((zero_ci[1] - epsilon).abs() < 1e-12);
    assert!(zero_ci[1] > 0.0);

    let full = summarize_scores(&[Some(4); 32], 32).unwrap();
    assert_eq!(full.score_rate, Some(1.0));
    assert_eq!(full.counts, [0, 0, 0, 0, 32]);
    let full_ci = full.final_interval_95.unwrap();
    assert!((full_ci[0] - (1.0 - epsilon)).abs() < 1e-12);
    assert_eq!(full_ci[1], 1.0);
    assert!(full_ci[0] < 1.0);
}

#[test]
fn pair_frequency_weighted_score_matches_original_per_pair_half_points() {
    let scores = (0..32).map(|i| Some((i % 5) as u8)).collect::<Vec<_>>();
    let summary = summarize_scores(&scores, 32).unwrap();
    assert_eq!(summary.counts, [7, 7, 6, 6, 6]);
    assert_eq!(summary.completed_pairs, 32);
    let raw_sum: u64 = scores.iter().map(|v| u64::from(v.unwrap())).sum();
    let bin_sum: u64 = summary
        .counts
        .iter()
        .enumerate()
        .map(|(half_points, count)| half_points as u64 * count)
        .sum();
    assert_eq!(raw_sum, bin_sum);
    assert_eq!(summary.score_rate, Some(raw_sum as f64 / (4.0 * 32.0)));
    let interval = summary.final_interval_95.unwrap();
    assert!(interval.iter().all(|value| value.is_finite()));
    assert!(0.0 <= interval[0] && interval[0] <= interval[1] && interval[1] <= 1.0);
}
