use rz_encoding::POLICY_SIZE;
use rz_eval::output::{validate_maia, Head, OutputError};
use rz_eval::RawOutput;

fn raw() -> RawOutput {
    RawOutput {
        policy_logits: vec![0.0; POLICY_SIZE],
        wdl: vec![0.2, 0.3, 0.5],
    }
}

#[test]
fn gather_then_softmax_matches_hand_computed_weights_and_preserves_wdl() {
    let mut output = raw();
    output.policy_logits[7] = 2.0f32.ln();
    output.policy_logits[12] = 3.0f32.ln();
    output.policy_logits[1401] = 5.0f32.ln();
    output.policy_logits[9] = f32::MAX; // Not a legal move; cannot affect normalization.
    let checked = validate_maia(&output, &[12, 7, 1401]).unwrap();
    for (&actual, expected) in checked.policy().iter().zip([0.3, 0.2, 0.5]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(checked.wdl(), [0.2, 0.3, 0.5]); // No second softmax.
    let reordered = validate_maia(&output, &[1401, 12, 7]).unwrap();
    for (&actual, expected) in reordered.policy().iter().zip([0.5, 0.3, 0.2]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(checked.wdl(), reordered.wdl());
}

#[test]
fn stable_normalization_handles_extreme_finite_logits_and_one_legal_move() {
    let mut output = raw();
    output.policy_logits[0] = f32::MAX;
    output.policy_logits[1] = -f32::MAX;
    assert_eq!(
        validate_maia(&output, &[0, 1]).unwrap().policy(),
        &[1.0, 0.0]
    );
    assert_eq!(validate_maia(&output, &[1]).unwrap().policy(), &[1.0]);
    output.policy_logits[0] = -f32::MAX;
    assert_eq!(
        validate_maia(&output, &[0, 1]).unwrap().policy(),
        &[0.5, 0.5]
    );
}

#[test]
fn malformed_heads_and_non_finite_outputs_are_not_repaired() {
    let mut output = raw();
    output.wdl.pop();
    assert_eq!(
        validate_maia(&output, &[0]),
        Err(OutputError::WrongShape {
            head: Head::Wdl,
            actual: 2,
            expected: 3,
        })
    );
    let mut output = raw();
    output.policy_logits.pop();
    assert_eq!(
        validate_maia(&output, &[0]),
        Err(OutputError::WrongShape {
            head: Head::Policy,
            actual: 1857,
            expected: POLICY_SIZE,
        })
    );
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut output = raw();
        output.policy_logits[99] = value; // Invalid even when masked out.
        assert_eq!(
            validate_maia(&output, &[0]),
            Err(OutputError::NonFinite {
                head: Head::Policy,
                index: 99,
            })
        );
        let mut output = raw();
        output.wdl[1] = value;
        assert_eq!(
            validate_maia(&output, &[0]),
            Err(OutputError::NonFinite {
                head: Head::Wdl,
                index: 1,
            })
        );
    }
}

#[test]
fn invalid_wdl_probabilities_are_not_clipped_or_renormalized() {
    for wdl in [vec![-0.1, 0.6, 0.5], vec![1.1, 0.0, -0.1]] {
        let mut output = raw();
        output.wdl = wdl;
        assert_eq!(
            validate_maia(&output, &[0]),
            Err(OutputError::WdlOutOfRange { index: 0 })
        );
    }
    let mut output = raw();
    output.wdl = vec![0.0, 0.0, 0.0];
    assert_eq!(
        validate_maia(&output, &[0]),
        Err(OutputError::InvalidWdlSum { actual: 0.0 })
    );
    output.wdl = vec![0.25, 0.5, 0.2501];
    assert!(matches!(
        validate_maia(&output, &[0]),
        Err(OutputError::InvalidWdlSum { .. })
    ));
}

#[test]
fn empty_duplicate_out_of_range_and_oversized_legal_orders_fail() {
    let output = raw();
    assert_eq!(
        validate_maia(&output, &[]),
        Err(OutputError::EmptyLegalPolicy)
    );
    assert_eq!(
        validate_maia(&output, &[12, 12]),
        Err(OutputError::DuplicatePolicyIndex(12))
    );
    assert_eq!(
        validate_maia(&output, &[POLICY_SIZE]),
        Err(OutputError::InvalidPolicyIndex(POLICY_SIZE))
    );
    assert_eq!(
        validate_maia(&output, &[usize::MAX]),
        Err(OutputError::InvalidPolicyIndex(usize::MAX))
    );
    assert_eq!(
        validate_maia(&output, &vec![0; POLICY_SIZE + 1]),
        Err(OutputError::TooManyLegalIndices)
    );
}

#[test]
fn all_model_indices_can_be_normalized_without_an_artificial_small_move_cap() {
    let indices: Vec<_> = (0..POLICY_SIZE).collect();
    let checked = validate_maia(&raw(), &indices).unwrap();
    let sum: f64 = checked.policy().iter().map(|&p| f64::from(p)).sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(checked
        .policy()
        .iter()
        .all(|&p| p == 1.0 / POLICY_SIZE as f32));
}
