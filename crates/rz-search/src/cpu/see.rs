//! Rules-backed static material exchange, used only by opt-in move ordering.
//!
//! Exact minimax of the restricted exchange game: after the first legal
//! capture/promotion, either side may decline for zero or make a Rules-legal
//! capture on the same target. Declining is an abstract ordering choice even
//! in check, not a legal chess evasion. Other moves, mate/draw outcomes, PST,
//! evaluator values and search TT are excluded. This is neither a full
//! tactical proof nor a pruning bound. Recaptures remove one piece; standard
//! Rules admit at most 32 board pieces. All four promotions, EP removals,
//! discovered attacks, pins and king safety come from Rules views/deltas.

use super::{is_tactical, material, Abort, CpuError};
use rz_position::{BoardMove, Color, Position, RuleMoveDelta, Square};

const MAX_EXCHANGE_PLIES: u16 = 32;

pub(super) fn score(
    position: &Position,
    mv: BoardMove,
    visit: &mut impl FnMut() -> Result<(), Abort>,
) -> Result<i32, Abort> {
    visit()?;
    if !is_tactical(position, mv) {
        return Err(Abort::Error(CpuError::Unsupported(
            "legal SEE requires a capture or promotion",
        )));
    }
    // The clone has its own Rules owner; do not apply the source's legal view.
    let mut exchange = position.clone();
    let view = exchange.ordered_legal_moves();
    let mover = exchange.side_to_move();
    let undo = exchange.make_from_view(&view, mv)?;
    let gain = material_delta(undo.delta(), mover);
    let reply = recaptures(&mut exchange, mv.to, 1, visit);
    // Restore before propagating aborts/errors; the live input never changes.
    exchange.unmake(undo)?;
    Ok(gain - reply?)
}

fn recaptures(
    position: &mut Position,
    target: Square,
    plies: u16,
    visit: &mut impl FnMut() -> Result<(), Abort>,
) -> Result<i32, Abort> {
    visit()?;
    let view = position.ordered_legal_moves();
    let mover = position.side_to_move();
    let mut best = 0; // Optional decline, not a stand-pat evaluation.
    for &mv in view.moves() {
        if mv.to != target {
            continue;
        }
        if plies >= MAX_EXCHANGE_PLIES {
            return Err(Abort::Error(CpuError::Unsupported(
                "legal SEE exchange ply bound exhausted",
            )));
        }
        // Unmake advances revision; rebind the view for each sibling move.
        let current = position.ordered_legal_moves();
        let undo = position.make_from_view(&current, mv)?;
        if !undo.delta().kind.capture {
            position.unmake(undo)?;
            return Err(Abort::Error(CpuError::Unsupported(
                "legal SEE target recapture was not a capture",
            )));
        }
        let gain = material_delta(undo.delta(), mover);
        let reply = recaptures(position, target, plies + 1, visit);
        position.unmake(undo)?;
        best = best.max(gain - reply?);
    }
    Ok(best)
}

fn material_delta(delta: &RuleMoveDelta, mover: Color) -> i32 {
    let relative = |color: Color, value: i32| if color == mover { value } else { -value };
    let added: i32 = delta
        .additions
        .iter()
        .map(|change| relative(change.piece.color, material(change.piece.kind)))
        .sum();
    let removed: i32 = delta
        .removals
        .iter()
        .map(|change| relative(change.piece.color, material(change.piece.kind)))
        .sum();
    added - removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::{PieceKind, PositionError};

    fn scored(fen: &str, text: &str) -> i32 {
        let position = Position::from_fen(fen).unwrap();
        let before = position.snapshot();
        let identity = position.position_identity();
        let mut work = 0;
        let actual = score(&position, text.parse().unwrap(), &mut || {
            work += 1;
            assert!(work <= 256, "tiny exchange fixture exceeded bounded work");
            Ok(())
        })
        .unwrap();
        assert!(position.matches_snapshot(&before));
        assert_eq!(position.position_identity(), identity);
        actual
    }

    #[test]
    fn simple_exchange_and_decline_are_material_only() {
        assert_eq!(scored("7k/8/8/3p4/4P3/8/8/K7 w - - 0 1", "e4d5"), 100);
        assert_eq!(scored("7k/8/2p5/3p4/4P3/8/8/K7 w - - 0 1", "e4d5"), 0);
        assert_eq!(scored("7k/8/2p5/3p4/8/8/8/K2Q4 w - - 0 1", "d1d5"), -800);
    }

    #[test]
    fn xray_recapture_uses_updated_rules_occupancy() {
        assert_eq!(scored("7k/3r4/8/3p4/8/8/3R4/K2R4 w - - 0 1", "d2d5"), 100);
    }

    #[test]
    fn pinned_defender_is_not_an_available_recapture() {
        assert_eq!(scored("4k3/8/4p3/3p4/2P5/8/8/K3R3 w - - 0 1", "c4d5"), 100);
        assert_eq!(scored("7k/8/4p3/3p4/2P5/8/8/K3R3 w - - 0 1", "c4d5"), 0);
    }

    #[test]
    fn king_recapture_must_be_rules_safe() {
        assert_eq!(scored("8/8/4k3/3p4/2P1P3/8/8/K7 w - - 0 1", "c4d5"), 100);
        assert_eq!(scored("8/8/4k3/3p4/2P5/8/8/K7 w - - 0 1", "c4d5"), 0);
    }

    #[test]
    fn en_passant_uses_removed_pawn_and_opened_file() {
        assert_eq!(scored("7k/8/8/3pP3/8/8/8/K7 w - d6 0 2", "e5d6"), 100);
        assert_eq!(scored("3r3k/8/8/3pP3/8/8/8/K7 w - d6 0 2", "e5d6"), 0);
    }

    #[test]
    fn first_quiet_and_capture_promotions_keep_all_four_values() {
        for (suffix, gain) in [("q", 800), ("r", 400), ("b", 230), ("n", 220)] {
            assert_eq!(
                scored("7k/P7/8/8/8/8/8/K7 w - - 0 1", &format!("a7a8{suffix}")),
                gain
            );
            assert_eq!(
                scored("1r5k/P7/8/8/8/8/8/K7 w - - 0 1", &format!("a7b8{suffix}")),
                gain + 500
            );
        }
    }

    #[test]
    fn recapture_promotion_selects_best_legal_material_exchange() {
        let fen = "7k/8/8/8/8/8/pR6/1n5K w - - 0 1";
        let mut position = Position::from_fen(fen).unwrap();
        position.make_uci("b2b1").unwrap();
        let mut kinds: Vec<_> = position
            .legal_moves()
            .into_iter()
            .filter(|mv| mv.to == "b1".parse::<Square>().unwrap())
            .map(|mv| mv.promotion.unwrap())
            .collect();
        kinds.sort();
        let mut expected = vec![
            PieceKind::Queen,
            PieceKind::Rook,
            PieceKind::Bishop,
            PieceKind::Knight,
        ];
        expected.sort();
        assert_eq!(kinds, expected);
        assert_eq!(scored(fen, "b2b1"), -980);
    }

    #[test]
    fn black_mover_has_same_relative_material_sign() {
        assert_eq!(scored("k7/8/8/4p3/3P4/8/8/7K b - - 0 1", "e5d4"), 100);
        assert_eq!(scored("7k/8/8/8/8/8/p7/7K b - - 0 1", "a2a1q"), 800);
    }

    #[test]
    fn illegal_pin_and_ep_discovered_check_are_rejected() {
        for (fen, mv) in [
            ("4r2k/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
            ("7k/8/8/r4pPK/8/8/8/8 w - f6 0 2", "g5f6"),
        ] {
            let position = Position::from_fen(fen).unwrap();
            let before = position.snapshot();
            assert!(matches!(
                score(&position, mv.parse().unwrap(), &mut || Ok(())),
                Err(Abort::Error(CpuError::Rules(PositionError::IllegalMove)))
            ));
            assert!(position.matches_snapshot(&before));
        }
    }

    #[test]
    fn aborted_recap_restores_rules_state_before_error_propagation() {
        let mut position = Position::from_fen("7k/8/2p5/3P4/8/8/8/K7 b - - 0 1").unwrap();
        let before = position.snapshot();
        let identity = position.position_identity();
        let mut work = 0;
        let result = recaptures(&mut position, "d5".parse().unwrap(), 1, &mut || {
            work += 1;
            if work == 2 {
                Err(Abort::Stop(super::super::CpuCompletion::Canceled))
            } else {
                Ok(())
            }
        });
        assert!(matches!(
            result,
            Err(Abort::Stop(super::super::CpuCompletion::Canceled))
        ));
        assert!(before.same_state(&position.snapshot()));
        assert_eq!(position.position_identity(), identity);
        assert_eq!(work, 2);
    }
}
