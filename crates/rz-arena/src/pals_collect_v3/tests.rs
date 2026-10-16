use super::*;
use std::fs;
#[path = "partial_archive_tests.rs"]
mod partial_archive_tests;
#[test]
fn actual_rules_trajectory_refuses_ongoing_and_preserves_opening_history() {
    let mut position = Position::startpos();
    position
        .make_move(BoardMove::from_uci("e2e4").unwrap())
        .unwrap();
    let mut game = ActualGameTrajectoryV3::begin("g1".into(), position).unwrap();
    assert_eq!(game.actual_history().unwrap().len(), 1);
    assert!(game.terminate_rules().is_err());
    assert!(
        game.claim_rules_draw(rz_position::ClaimReason::ThreefoldRepetition)
            .is_err()
    );
}
#[test]
fn rules_mate_and_repetition_are_actual_game_outcomes() {
    let mut game = ActualGameTrajectoryV3::begin("mate".into(), Position::startpos()).unwrap();
    for m in ["f2f3", "e7e5", "g2g4", "d8h4"] {
        game.play_pals_move(BoardMove::from_uci(m).unwrap())
            .unwrap();
    }
    let (authority, receipt) = game.terminate_rules().unwrap();
    assert_eq!(authority.ending, GenerationRulesEndingV3::Checkmate);
    assert_eq!(authority.outcome, GenerationOutcomeV3::BlackWin);
    assert_eq!(authority.anchors.len(), 5);
    assert_eq!(
        generation_digest("rz-pals-rules-outcome-evidence/1", &receipt).unwrap(),
        authority.result_evidence_sha256
    );
    assert!(game.terminate_rules().is_err());
    let mut game = ActualGameTrajectoryV3::begin("repeat".into(), Position::startpos()).unwrap();
    for m in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        game.play_pals_move(BoardMove::from_uci(m).unwrap())
            .unwrap();
    }
    let (authority, _) = game
        .claim_rules_draw(rz_position::ClaimReason::ThreefoldRepetition)
        .unwrap();
    assert_eq!(authority.outcome, GenerationOutcomeV3::Draw);
}
#[test]
fn exact_anchor_distinguishes_same_fen_with_different_history() {
    let mut played = Position::startpos();
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        played.make_move(BoardMove::from_uci(m).unwrap()).unwrap();
    }
    let imported = Position::from_fen(&played.to_fen()).unwrap();
    let a = exact_rules_anchor(&played).unwrap();
    let b = exact_rules_anchor(&imported).unwrap();
    assert_eq!(a.board_fen, b.board_fen);
    assert_ne!(a.rules_history_sha256, b.rules_history_sha256);
    assert_ne!(a.rules_state_sha256, b.rules_state_sha256);
}
#[test]
fn finite_first_generation_pair_assignments_do_not_reverse_the_board() {
    let openings = (0..64)
        .map(|i| GenerationOpeningV3 {
            id: format!("o-{i}"),
            initial_fen: None,
            moves: vec![],
            split: if i < 48 {
                GenerationSplitV3::Train
            } else if i < 56 {
                GenerationSplitV3::Validation
            } else {
                GenerationSplitV3::Holdout
            },
        })
        .collect();
    let mut plan = SelfplayCollectionPlanV3 {
        run_id: "g0".into(),
        openings,
        games: 128,
        base_time_ms: 120_000,
        increment_ms: 1000,
        max_plies: 512,
        concurrent_games: 1,
        seed: 29,
        sampling_seed: 29,
        process_restart_each_game: true,
        ponder: false,
        maximum_output_bytes: GENERATION_OUTPUT_CAP,
    };
    let games = plan.games().unwrap();
    assert_eq!(games.len(), 128);
    assert_eq!(games[0].opening_id, games[1].opening_id);
    assert!(games[0].engine_a_white && !games[1].engine_a_white);
    assert_eq!(games[0].split, games[1].split);
    plan.openings[0].split = GenerationSplitV3::Holdout;
    assert!(plan.validate().is_err());
    plan.openings[0].split = GenerationSplitV3::Train;
    plan.games = 127;
    assert!(plan.validate().is_err());
}
#[test]
fn writer_readback_is_pinned_and_never_promotes_recovery_flags() {
    let root = fs::canonicalize(std::env::temp_dir()).unwrap();
    let path = root.join(format!(
        "rz-generation-v3-test-{}-{}",
        std::process::id(),
        NEXT_OWNER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut writer =
        GenerationChunkWriterV3::create(&path, "fixture-schema-only".into(), 32 * 1024 * 1024)
            .unwrap();
    let splits = BTreeMap::from([("o1".into(), GenerationSplitV3::Train)]);
    let authority = GenerationProducerAuthorityV3 {
        schema: "rz-pals-actual-collection-authority/1".into(),
        source_kind: "actual_selfplay".into(),
        source_sha256: "1".repeat(64),
        model_sha256: "2".repeat(64),
        runtime_sha256: "3".repeat(64),
        teacher_binary_sha256: "4".repeat(64),
        teacher_profile_sha256: "5".repeat(64),
        opening_split_sha256: generation_digest("rz-pals-opening-split/1", &splits).unwrap(),
        registered_before_outcomes: true,
        input_seals: vec![],
        actual_game_ids: vec![],
    };
    let receipt = serde_json::json!({"schema":"rz-pals-rules-outcome-evidence/1","game_id":"fixture","ending":"checkmate","outcome":"white_win","rules_terminal":true});
    writer.append_evidence(&receipt).unwrap();
    writer.publish_authority(&authority, &splits).unwrap();
    let manifest = writer.finish().unwrap();
    assert!(!writer.all_chunks_recovered());
    assert_eq!(manifest.chunks.len(), 1);
    let pin = manifest.chunks[0].clone();
    assert!(
        writer
            .observe_recovered_chunk(&pin, &writer.directory().join(&pin.path))
            .is_err()
    );
    let recovered = root.join(format!(
        "rz-generation-v3-recovery-test-{}-{}",
        std::process::id(),
        NEXT_OWNER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&recovered).unwrap();
    let copied = recovered.join(&pin.path);
    fs::copy(writer.directory().join(&pin.path), &copied).unwrap();
    writer.observe_recovered_chunk(&pin, &copied).unwrap();
    assert!(writer.all_chunks_recovered());
    fs::write(writer.directory().join(&pin.path), b"corrupt").unwrap();
    assert!(verify_chunk(writer.directory(), &pin).is_err());
    assert!(writer.append_evidence(&receipt).is_err());
    // Only the exact fresh test-owned directory is removed after all handles
    // close. No recursive scan, model cache or other task path is touched.
    let actual = writer.directory().to_path_buf();
    drop(writer);
    assert_eq!(actual, fs::canonicalize(&path).unwrap());
    assert!(actual.starts_with(&root));
    fs::remove_dir_all(actual).unwrap();
    let recovered = fs::canonicalize(recovered).unwrap();
    assert!(recovered.starts_with(&root));
    fs::remove_file(copied).unwrap();
    fs::remove_dir(recovered).unwrap();
}

#[test]
fn archive_worker_preserves_terminal_and_rejected_owned_command() {
    use std::time::{Duration, Instant};
    let root = fs::canonicalize(std::env::temp_dir()).unwrap();
    let path = root.join(format!(
        "rz-generation-v3-worker-test-{}-{}",
        std::process::id(),
        NEXT_OWNER.fetch_add(1, Ordering::Relaxed)
    ));
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut worker = GenerationWriterWorkerV3::spawn(
        path.clone(),
        "fixture-archive-only".into(),
        32 * 1024 * 1024,
        deadline,
    )
    .unwrap();
    let ready = loop {
        match worker.try_event() {
            Some(GenerationWriterEventV3::Ready) => break true,
            Some(GenerationWriterEventV3::Failed(error)) => {
                panic!("archive source fixture: {error}")
            }
            Some(GenerationWriterEventV3::Finished { .. }) => panic!("publication before Finish"),
            Some(GenerationWriterEventV3::PartialPreserved(_)) => {
                panic!("partial publication before request")
            }
            None => {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    };
    assert!(ready);
    let receipt = serde_json::json!({"schema":"rz-pals-rules-outcome-evidence/1","game_id":"fixture","ending":"checkmate","outcome":"white_win","rules_terminal":true});
    assert!(
        worker
            .try_submit(GenerationWriteCommandV3::Evidence(receipt.clone()))
            .is_ok()
    );
    let splits = BTreeMap::from([("fixture-opening".into(), GenerationSplitV3::Train)]);
    let authority = GenerationProducerAuthorityV3 {
        schema: "rz-pals-actual-collection-authority/1".into(),
        source_kind: "actual_selfplay".into(),
        source_sha256: "1".repeat(64),
        model_sha256: "2".repeat(64),
        runtime_sha256: "3".repeat(64),
        teacher_binary_sha256: "4".repeat(64),
        teacher_profile_sha256: "5".repeat(64),
        opening_split_sha256: generation_digest("rz-pals-opening-split/1", &splits).unwrap(),
        registered_before_outcomes: true,
        input_seals: vec![],
        actual_game_ids: vec![],
    };
    assert!(
        worker
            .try_submit(GenerationWriteCommandV3::Finish { authority, splits })
            .is_ok()
    );
    // A full or closed owner returns the same owned row. Manager may persist
    // failure/raw evidence elsewhere; it cannot silently count this as written.
    let rejected = worker.try_submit(GenerationWriteCommandV3::Evidence(receipt.clone()));
    match rejected {
        Err((_, GenerationWriteCommandV3::Evidence(actual))) => assert_eq!(actual, receipt),
        _ => panic!("late archive admission accepted or command lost"),
    };
    let (directory, manifest) = loop {
        match worker.try_event() {
            Some(GenerationWriterEventV3::Finished {
                directory,
                manifest,
            }) => break (directory, manifest),
            Some(GenerationWriterEventV3::Failed(error)) => {
                panic!("archive source fixture: {error}")
            }
            Some(GenerationWriterEventV3::Ready) => panic!("duplicate readiness"),
            Some(GenerationWriterEventV3::PartialPreserved(_)) => {
                panic!("partial publication cannot substitute complete manifest")
            }
            None => {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    };
    assert_eq!(manifest.total_records, 0); // no real neural input or training row
    assert_eq!(manifest.chunks.len(), 1);
    verify_chunk(&directory, &manifest.chunks[0]).unwrap();
    loop {
        if worker.try_join().unwrap() {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    drop(worker);
    assert_eq!(directory, fs::canonicalize(&path).unwrap());
    assert!(directory.starts_with(&root));
    fs::remove_dir_all(directory).unwrap();
}
