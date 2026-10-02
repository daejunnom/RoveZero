use rz_uci::{Command, GoLimits, ParseError, ParserLimits, PositionBase, PositionSpec, parse};

fn parse_default(line: &str) -> Result<Command, ParseError> {
    parse(line, ParserLimits::default())
}

#[test]
fn protocol_commands_accept_whitespace_and_one_physical_line_ending() {
    for (line, expected) in [
        (" uci\r\n", Command::Uci),
        ("\tisready\t", Command::IsReady),
        ("stop\n", Command::Stop),
        ("quit", Command::Quit),
        ("ucinewgame", Command::NewGame),
    ] {
        assert_eq!(parse_default(line), Ok(expected), "{line:?}");
    }
}

#[test]
fn unexpected_protocol_arguments_and_empty_input_are_rejected() {
    for line in ["uci x", "isready x", "stop x", "quit x", "ucinewgame x"] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
    assert_eq!(parse_default(" \t\r\n"), Err(ParseError::Empty));
}

#[test]
fn multiline_non_ascii_and_control_bytes_are_rejected() {
    for line in [
        "uci\nquit",
        "uci\rquit",
        "uci\0",
        "uci\u{000b}",
        "uci\u{000c}",
        "uci\u{007f}",
        "한글",
    ] {
        assert_eq!(
            parse_default(line),
            Err(ParseError::InvalidText),
            "{line:?}"
        );
    }
}

#[test]
fn unsupported_modes_and_unknown_commands_remain_distinct() {
    assert_eq!(
        parse_default("ponderhit"),
        Err(ParseError::Unsupported("ponderhit".into()))
    );
    assert_eq!(
        parse_default("go ponder"),
        Err(ParseError::Unsupported("ponder".into()))
    );
    assert_eq!(
        parse_default("go depth 3"),
        Err(ParseError::Unsupported("depth".into()))
    );
    assert_eq!(
        parse_default("go searchmoves e2e4"),
        Err(ParseError::Unsupported("searchmoves".into()))
    );
    assert_eq!(
        parse_default("debug on"),
        Ok(Command::Unknown("debug".into()))
    );
}

#[test]
fn go_parses_each_supported_budget_mode_without_losing_units() {
    assert_eq!(
        parse_default("go movetime 150 nodes 23"),
        Ok(Command::Go(GoLimits {
            movetime_ms: Some(150),
            nodes: Some(23),
            ..GoLimits::default()
        }))
    );
    assert_eq!(
        parse_default("go wtime 900 btime 800 winc 5 binc 7 movestogo 12 nodes 99"),
        Ok(Command::Go(GoLimits {
            white_time_ms: Some(900),
            black_time_ms: Some(800),
            white_increment_ms: Some(5),
            black_increment_ms: Some(7),
            moves_to_go: Some(12),
            nodes: Some(99),
            ..GoLimits::default()
        }))
    );
    assert_eq!(
        parse_default("go infinite"),
        Ok(Command::Go(GoLimits {
            infinite: true,
            ..GoLimits::default()
        }))
    );
    assert_eq!(
        parse_default("go nodes 1"),
        Ok(Command::Go(GoLimits {
            nodes: Some(1),
            ..GoLimits::default()
        }))
    );
    assert!(parse_default("go movetime 0").is_ok());
    assert!(parse_default("go wtime 0 btime 0").is_ok());
}

#[test]
fn go_rejects_conflicts_missing_fields_duplicates_and_overflow() {
    for line in [
        "go",
        "go movetime",
        "go nodes 0",
        "go movestogo 0 wtime 1 btime 1",
        "go wtime 1",
        "go btime 1",
        "go winc 1",
        "go binc 1",
        "go movestogo 1",
        "go movetime 1 wtime 1 btime 1",
        "go infinite movetime 1",
        "go infinite nodes 1",
        "go infinite wtime 1 btime 1",
        "go infinite infinite",
        "go nodes 1 nodes 2",
        "go movetime 1 movetime 2",
        "go wtime 1 btime 1 winc 1 winc 2",
        "go wtime 1 btime 1 movestogo 4294967296",
        "go nodes 18446744073709551616",
        "go movetime -1",
        "go movetime +1",
        "go movetime 1.5",
        "go movetime NaN",
    ] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
}

#[test]
fn position_preserves_fen_and_complete_ordered_move_trace() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    assert_eq!(
        parse_default(&format!("position fen {fen} moves e2e4 e7e5 g1f3")),
        Ok(Command::Position(PositionSpec {
            base: PositionBase::Fen(fen.into()),
            moves: vec!["e2e4".into(), "e7e5".into(), "g1f3".into()],
        }))
    );
    for suffix in ["q", "r", "b", "n"] {
        let movement = format!("a7a8{suffix}");
        assert_eq!(
            parse_default(&format!("position startpos moves {movement}")),
            Ok(Command::Position(PositionSpec {
                base: PositionBase::StartPos,
                moves: vec![movement]
            }))
        );
    }
}

#[test]
fn malformed_position_text_is_rejected_before_rules() {
    for line in [
        "position",
        "position startpos moves",
        "position startpos e2e4",
        "position startpos moves e9e4",
        "position startpos moves e2e2",
        "position startpos moves e2e4qz",
        "position startpos moves a7a8k",
        "position startpos moves a7a8Q",
        "position fen 8/8/8/8/8/8/8/8 w - - 0",
        "position fen 8/8/8/8/8/8/8 w - - 0 1",
        "position fen 9/8/8/8/8/8/8/8 w - - 0 1",
        "position fen 7/8/8/8/8/8/8/8 w - - 0 1",
        "position fen 8/8/8/8/8/8/8/8 x - - 0 1",
        "position fen 8/8/8/8/8/8/8/8 w KK - 0 1",
        "position fen 8/8/8/8/8/8/8/8 w - a4 0 1",
        "position fen 8/8/8/8/8/8/8/8 w - - -1 1",
        "position fen 8/8/8/8/8/8/8/8 w - - 0 0",
        "position fen 8/8/8/8/8/8/8/8 w - - 4294967296 1",
        "position fen 8/8/8/8/8/8/8/8 w - - 0 1 extra",
    ] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
    // Text validity is insufficient to establish chess semantics; Rules owns that check.
    assert!(parse_default("position fen 8/8/8/8/8/8/8/8 w - - 0 1").is_ok());
}

#[test]
fn option_parser_preserves_multiword_names_values_buttons_and_empty_strings() {
    assert_eq!(
        parse_default("setoption name Clear Hash"),
        Ok(Command::SetOption {
            name: "Clear Hash".into(),
            value: None
        })
    );
    assert_eq!(
        parse_default("setoption name Weights File value one two"),
        Ok(Command::SetOption {
            name: "Weights File".into(),
            value: Some("one two".into())
        })
    );
    assert_eq!(
        parse_default("setoption name Weights File value"),
        Ok(Command::SetOption {
            name: "Weights File".into(),
            value: Some(String::new())
        })
    );
    for line in [
        "setoption",
        "setoption value true",
        "setoption name",
        "setoption name value true",
    ] {
        assert!(
            matches!(parse_default(line), Err(ParseError::Malformed(_))),
            "{line:?}"
        );
    }
}

#[test]
fn configured_external_input_limits_are_enforced_at_the_boundary() {
    let short_line = ParserLimits {
        max_line_bytes: 3,
        ..ParserLimits::default()
    };
    assert_eq!(parse("uci", short_line), Ok(Command::Uci));
    assert_eq!(parse("uci\n", short_line), Err(ParseError::LineTooLong));
    let bounded = ParserLimits {
        max_moves: 2,
        max_option_name_bytes: 4,
        max_option_value_bytes: 3,
        ..ParserLimits::default()
    };
    assert!(parse("position startpos moves e2e4 e7e5", bounded).is_ok());
    assert_eq!(
        parse("position startpos moves e2e4 e7e5 g1f3", bounded),
        Err(ParseError::Limit("move trace"))
    );
    assert!(parse("setoption name Hash value 123", bounded).is_ok());
    assert_eq!(
        parse("setoption name HashX value 123", bounded),
        Err(ParseError::Limit("option name"))
    );
    assert_eq!(
        parse("setoption name Hash value 1234", bounded),
        Err(ParseError::Limit("option value"))
    );
}
