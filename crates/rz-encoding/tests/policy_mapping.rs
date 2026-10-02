use rz_encoding::policy::{canonical_square, castling_index, index, slot, slots, PolicyError};
use rz_encoding::POLICY_SIZE;
use std::collections::HashSet;

#[test]
fn selected_external_reference_indices_cover_boundaries_castling_and_promotions() {
    // Format reference: LC0 fd71a2d921b689c5f479d3227c3806c8e272d9c5.
    // Small selected outputs, not a vendored/generated external mapping table.
    let cases = [
        (0, 1, None, 0),      // a1b1
        (0, 10, None, 9),     // a1c2
        (12, 28, None, 322),  // e2e4
        (63, 62, None, 1791), // h8g8: final base slot
        (48, 56, Some('n'), 1401),
        (48, 56, Some('q'), 1792),
        (48, 56, Some('r'), 1793),
        (48, 56, Some('b'), 1794),
        (48, 57, Some('q'), 1795),
        (55, 63, Some('b'), 1857),
    ];
    for (from, to, promotion, expected) in cases {
        assert_eq!(index(from, to, promotion), Ok(expected));
    }
    assert_eq!(castling_index(4, 7), Ok(103));
    assert_eq!(castling_index(4, 0), Ok(97));
    assert_ne!(index(4, 6, None), castling_index(4, 7));
}

#[test]
fn all_model_slots_are_unique_and_have_a_bounded_inverse() {
    let mut seen = HashSet::new();
    for (i, action) in slots().iter().enumerate() {
        assert!(seen.insert((action.from, action.to, action.promotion)));
        assert_eq!(index(action.from, action.to, action.promotion), Ok(i));
        assert_eq!(slot(i), Ok(*action));
    }
    assert_eq!(seen.len(), POLICY_SIZE);
    assert_eq!(
        slot(POLICY_SIZE),
        Err(PolicyError::InvalidIndex(POLICY_SIZE))
    );
    assert_eq!(
        slots().iter().filter(|s| s.promotion.is_none()).count(),
        1792
    );
}

#[test]
fn all_four_promotion_kinds_stay_distinct_for_capture_and_non_capture() {
    for from in 48u8..56 {
        for to in 56u8..64 {
            if (from % 8).abs_diff(to % 8) <= 1 {
                let ids: HashSet<_> = ['q', 'r', 'b', 'n']
                    .map(|piece| index(from, to, Some(piece)).unwrap())
                    .into_iter()
                    .collect();
                assert_eq!(ids.len(), 4);
            }
        }
    }
}

#[test]
fn black_promotions_and_castling_flip_ranks_without_flipping_files() {
    assert_eq!(canonical_square(12, true), Ok(52)); // e2 -> e7
    assert_eq!(canonical_square(4, true), Ok(60)); // e1 -> e8
    for promotion in ['q', 'r', 'b', 'n'] {
        assert_eq!(
            index(
                canonical_square(12, true).unwrap(),
                canonical_square(5, true).unwrap(),
                Some(promotion),
            ),
            index(52, 61, Some(promotion))
        );
    }
    assert_eq!(
        castling_index(
            canonical_square(60, true).unwrap(),
            canonical_square(63, true).unwrap(),
        ),
        Ok(103)
    );
}

#[test]
fn unsupported_geometry_is_an_error_not_a_zero_policy_slot() {
    assert_eq!(index(64, 1, None), Err(PolicyError::InvalidSquare(64)));
    assert_eq!(index(0, 0, None), Err(PolicyError::UnrepresentableMove));
    assert_eq!(index(0, 19, None), Err(PolicyError::UnrepresentableMove));
    assert_eq!(
        index(48, 58, Some('n')),
        Err(PolicyError::InvalidPromotionGeometry)
    );
    assert_eq!(
        index(8, 0, Some('q')),
        Err(PolicyError::InvalidPromotionGeometry)
    );
    assert_eq!(
        index(48, 56, Some('k')),
        Err(PolicyError::UnsupportedPromotion('k'))
    );
    assert_eq!(
        castling_index(4, 6),
        Err(PolicyError::InvalidCastlingGeometry)
    );
}
