use rz_encoding::classical::{encode, ClassicalError, Frame, HistoryFill, Input, INPUT_VALUES};

fn frame() -> Frame {
    Frame {
        pieces: [
            [1 << 12, 1 << 18, 0, 0, 0, 1 << 4],
            [1 << 51, 0, 0, 0, 0, 1 << 60],
        ],
        repeated: false,
        en_passant_target: None,
    }
}

fn input(history: &[Frame]) -> Input<'_> {
    Input {
        history,
        black_to_move: false,
        castling: [true, false, false, true],
        halfmove_clock: 23,
        history_fill: HistoryFill::No,
    }
}

fn set_plane(values: &mut [f32], plane: usize, value: f32) {
    values[plane * 64..(plane + 1) * 64].fill(value);
}

#[test]
fn white_input_matches_a_hand_constructed_full_112_plane_tensor() {
    let history = [frame()];
    let encoded = encode(input(&history)).unwrap();
    let mut expected = vec![0.0; INPUT_VALUES];
    for (plane, square) in [(0, 12), (1, 18), (5, 4), (6, 51), (11, 60)] {
        expected[plane * 64 + square] = 1.0;
    }
    for (plane, value) in [(104, 1.0), (107, 1.0), (109, 23.0), (111, 1.0)] {
        set_plane(&mut expected, plane, value);
    }
    assert_eq!(encoded.values(), expected);
    assert_eq!(encoded.known_frames, 1);
    assert_eq!(encoded.padded_frames, 0);
}

#[test]
fn every_history_frame_uses_current_black_viewpoint_and_keeps_file_order() {
    let mut prior = frame();
    prior.pieces[0][0] = 1 << 28;
    prior.repeated = true;
    let history = [frame(), prior];
    let mut request = input(&history);
    request.black_to_move = true;
    let encoded = encode(request).unwrap();
    let mut expected = vec![0.0; INPUT_VALUES];
    // Current black pawn d7 becomes own d2, king e8 becomes own e1.
    // White pieces become enemy e7/c6/e8. Prior white pawn e4 becomes e5.
    for (plane, square) in [
        (0, 11),
        (5, 4),
        (6, 52),
        (7, 42),
        (11, 60),
        (13, 11),
        (18, 4),
        (19, 36),
        (20, 42),
        (24, 60),
    ] {
        expected[plane * 64 + square] = 1.0;
    }
    for (plane, value) in [
        (25, 1.0),
        (105, 1.0),
        (106, 1.0),
        (108, 1.0),
        (109, 23.0),
        (111, 1.0),
    ] {
        set_plane(&mut expected, plane, value);
    }
    assert_eq!(encoded.values(), expected);
}

#[test]
fn model_fill_is_explicit_and_does_not_change_the_known_history() {
    let history = [frame()];
    let original = history;
    let zeros = encode(input(&history)).unwrap();
    let mut request = input(&history);
    request.history_fill = HistoryFill::RepeatOldest;
    let filled = encode(request).unwrap();
    assert_eq!(history, original);
    assert_eq!(filled.known_frames, 1);
    assert_eq!(filled.padded_frames, 7);
    assert_eq!(zeros.padded_frames, 0);
    assert_eq!(filled.history_fill.reference_option(), "always");
    assert_ne!(zeros.values(), filled.values());
    for i in 1..8 {
        assert_eq!(
            &filled.values()[i * 13 * 64..(i + 1) * 13 * 64],
            &filled.values()[..13 * 64]
        );
    }
}

#[test]
fn no_fill_infers_only_one_ep_predecessor_in_both_colors() {
    for (black_to_move, color, current, target, previous) in [
        (false, 1, 35, 43, 51), // black d7-d5, EP d6
        (true, 0, 28, 20, 12),  // white e2-e4, EP e3
    ] {
        let mut board = frame();
        board.pieces[color][0] = 1 << current;
        board.en_passant_target = Some(target);
        let history = [board];
        let mut request = input(&history);
        request.black_to_move = black_to_move;
        let result = encode(request).unwrap();
        let square = |s| if black_to_move { s ^ 56 } else { s };
        // The double-pushed pawn is the opponent in both cases.
        assert_eq!(result.values()[6 * 64 + square(current)], 1.0);
        assert_eq!(result.values()[19 * 64 + square(previous)], 1.0);
        assert_eq!(result.values()[19 * 64 + square(current)], 0.0);
        assert!(result.values()[26 * 64..104 * 64].iter().all(|&v| v == 0.0));
        assert_eq!(result.padded_frames, 1);
        assert!(result.inferred_ep_predecessor);
        assert_eq!(history[0], board);
        request.history_fill = HistoryFill::RepeatOldest;
        let filled = encode(request).unwrap();
        for history_index in 1..8 {
            assert_eq!(
                filled.values()[(history_index * 13 + 6) * 64 + square(previous)],
                1.0
            );
            assert_eq!(
                filled.values()[(history_index * 13 + 6) * 64 + square(current)],
                0.0
            );
        }
    }
}

#[test]
fn same_board_with_different_history_repetition_or_counter_changes_model_input() {
    let history = [frame(), frame()];
    let baseline = encode(input(&history)).unwrap();
    let mut different = history;
    different[1].pieces[0][0] = 1 << 28;
    assert_ne!(
        baseline.values(),
        encode(input(&different)).unwrap().values()
    );
    different = history;
    different[0].repeated = true;
    assert_ne!(
        baseline.values(),
        encode(input(&different)).unwrap().values()
    );
    let mut request = input(&history);
    request.halfmove_clock = 100;
    let encoded = encode(request).unwrap();
    assert_eq!(encoded.values()[109 * 64], 100.0);
    assert_ne!(baseline.values(), encoded.values());
}

#[test]
fn malformed_projections_are_rejected_before_tensor_use() {
    assert_eq!(encode(input(&[])), Err(ClassicalError::EmptyHistory));
    assert_eq!(
        encode(input(&[frame(); 9])),
        Err(ClassicalError::TooManyFrames(9))
    );
    let mut bad = frame();
    bad.pieces[1][4] = 1 << 12;
    assert_eq!(
        encode(input(&[bad])),
        Err(ClassicalError::OverlappingPieces { frame: 0 })
    );
    for target in [0, 255, 20, 43] {
        let mut bad = frame();
        bad.en_passant_target = Some(target);
        assert_eq!(
            encode(input(&[bad])),
            Err(ClassicalError::InvalidEnPassant { frame: 0 })
        );
    }
    let mut bad = frame();
    bad.pieces[0][0] = (1 << 28) | (1 << 12);
    bad.en_passant_target = Some(20);
    assert_eq!(
        encode(input(&[bad])),
        Err(ClassicalError::InvalidEnPassant { frame: 0 })
    );
}
