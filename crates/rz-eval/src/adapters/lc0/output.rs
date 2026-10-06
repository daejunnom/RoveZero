//! Validate the selected FP32 Maia export's heads before runtime finalization.
//!
//! Policy is 1858 raw logits; WDL is three already-softmaxed probabilities in
//! side-to-move W/D/L order. The first baseline uses policy temperature 1.0.
//! This does not validate request identity, legality, generation, or deadline.

use crate::RawOutput;
use rz_encoding::POLICY_SIZE;
use std::fmt;

/// Output admissibility, distinct from reference-parity tolerances. Changing
/// this fixed baseline profile must change the eventual model/compute identity.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    Policy,
    Wdl,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutputError {
    WrongShape {
        head: Head,
        actual: usize,
        expected: usize,
    },
    NonFinite {
        head: Head,
        index: usize,
    },
    WdlOutOfRange {
        index: usize,
    },
    InvalidWdlSum {
        actual: f64,
    },
    EmptyLegalPolicy,
    TooManyLegalIndices,
    InvalidPolicyIndex(usize),
    DuplicatePolicyIndex(usize),
    AllocationFailed,
    InvalidPolicyNormalization,
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Maia output validation: {self:?}")
    }
}

impl std::error::Error for OutputError {}

/// Model heads only, not the shared EvalResult. The adapter must attach and
/// validate the original request's identity and ordered legal move identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedHeads {
    policy: Vec<f32>,
    wdl: [f32; 3],
}

impl ValidatedHeads {
    pub fn policy(&self) -> &[f32] {
        &self.policy
    }

    pub fn wdl(&self) -> [f32; 3] {
        self.wdl
    }

    pub fn into_parts(self) -> (Vec<f32>, [f32; 3]) {
        (self.policy, self.wdl)
    }
}

/// Validate all required heads, gather exactly the supplied legal order, and
/// softmax only those logits. The caller obtains indices from checked Rules
/// moves through the encoding adapter; a representable index is not legal proof.
pub fn validate_maia(
    raw: &RawOutput,
    ordered_legal_indices: &[usize],
) -> Result<ValidatedHeads, OutputError> {
    let (policy, wdl) = validate_policy(raw, ordered_legal_indices, |p| p, f64::from)?;
    Ok(ValidatedHeads { policy, wdl })
}

/// Write contract probabilities once, preserving the baseline's f32 rounding
/// before widening. This changes storage only, never softmax/reduction order.
#[cfg(feature = "experimental-policy-buffer")]
pub(crate) fn validate_maia_contract(
    raw: &RawOutput,
    ordered_legal_indices: &[usize],
) -> Result<(Vec<f64>, [f32; 3]), OutputError> {
    validate_policy(raw, ordered_legal_indices, f64::from, |p| p)
}

fn validate_policy<T: Copy>(
    raw: &RawOutput,
    ordered_legal_indices: &[usize],
    store: impl Fn(f32) -> T,
    widen: impl Fn(T) -> f64,
) -> Result<(Vec<T>, [f32; 3]), OutputError> {
    for (head, values, expected) in [
        (Head::Policy, raw.policy_logits.as_slice(), POLICY_SIZE),
        (Head::Wdl, raw.wdl.as_slice(), 3),
    ] {
        if values.len() != expected {
            return Err(OutputError::WrongShape {
                head,
                actual: values.len(),
                expected,
            });
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(OutputError::NonFinite { head, index });
            }
        }
    }
    for (index, &probability) in raw.wdl.iter().enumerate() {
        if !(0.0..=1.0).contains(&probability) {
            return Err(OutputError::WdlOutOfRange { index });
        }
    }
    let wdl_sum: f64 = raw.wdl.iter().map(|&p| f64::from(p)).sum();
    if (wdl_sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(OutputError::InvalidWdlSum { actual: wdl_sum });
    }
    if ordered_legal_indices.is_empty() {
        return Err(OutputError::EmptyLegalPolicy);
    }
    if ordered_legal_indices.len() > POLICY_SIZE {
        return Err(OutputError::TooManyLegalIndices);
    }
    let mut seen = [false; POLICY_SIZE];
    for &index in ordered_legal_indices {
        let present = seen
            .get_mut(index)
            .ok_or(OutputError::InvalidPolicyIndex(index))?;
        if *present {
            return Err(OutputError::DuplicatePolicyIndex(index));
        }
        *present = true;
    }

    let maximum = ordered_legal_indices
        .iter()
        .map(|&i| f64::from(raw.policy_logits[i]))
        .fold(f64::NEG_INFINITY, f64::max);
    let weight = |index: usize| (f64::from(raw.policy_logits[index]) - maximum).exp();
    let denominator: f64 = ordered_legal_indices.iter().map(|&i| weight(i)).sum();
    let mut policy = Vec::new();
    policy
        .try_reserve_exact(ordered_legal_indices.len())
        .map_err(|_| OutputError::AllocationFailed)?;
    for &index in ordered_legal_indices {
        policy.push(store((weight(index) / denominator) as f32));
    }
    let sum: f64 = policy.iter().map(|&p| widen(p)).sum();
    if policy
        .iter()
        .map(|&p| widen(p))
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
        || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err(OutputError::InvalidPolicyNormalization);
    }
    Ok((policy, [raw.wdl[0], raw.wdl[1], raw.wdl[2]]))
}

#[cfg(all(test, feature = "experimental-policy-buffer"))]
mod contract_policy_tests {
    use super::*;

    fn raw() -> RawOutput {
        RawOutput {
            policy_logits: (0..POLICY_SIZE).map(|i| (i % 17) as f32 * 0.125).collect(),
            wdl: vec![0.5, 0.3, 0.2],
        }
    }

    #[test]
    fn direct_contract_storage_preserves_rounding_order_and_bits() {
        let mut raw = raw();
        for indices in [vec![3], vec![12, 7, 1401], (0..POLICY_SIZE).rev().collect()] {
            let baseline = validate_maia(&raw, &indices).unwrap();
            let (policy, wdl) = validate_maia_contract(&raw, &indices).unwrap();
            assert_eq!(wdl.map(f32::to_bits), baseline.wdl().map(f32::to_bits));
            assert_eq!(
                policy.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
                baseline
                    .policy()
                    .iter()
                    .map(|&p| f64::from(p).to_bits())
                    .collect::<Vec<_>>()
            );
        }
        raw.policy_logits.fill(0.0);
        let (policy, _) = validate_maia_contract(&raw, &[12, 7, 1401]).unwrap();
        assert_eq!(policy, vec![f64::from(1.0_f32 / 3.0); 3]);
        assert_ne!(policy[0].to_bits(), (1.0_f64 / 3.0).to_bits());
        raw.policy_logits[12] = f32::MAX;
        raw.policy_logits[7] = -f32::MAX;
        assert_eq!(
            validate_maia_contract(&raw, &[7, 12]).unwrap().0,
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn direct_contract_storage_keeps_validation_and_error_precedence() {
        let raw = raw();
        for indices in [
            vec![],
            vec![12, 12],
            vec![usize::MAX],
            vec![0; POLICY_SIZE + 1],
        ] {
            assert_eq!(
                validate_maia_contract(&raw, &indices).unwrap_err(),
                validate_maia(&raw, &indices).unwrap_err()
            );
        }
        let mut cases = Vec::new();
        let mut malformed = raw.clone();
        malformed.policy_logits.pop();
        malformed.wdl.pop();
        cases.push(malformed);
        let mut masked_nan = raw.clone();
        masked_nan.policy_logits[99] = f32::NAN;
        masked_nan.wdl.pop();
        cases.push(masked_nan);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            let mut invalid = raw.clone();
            invalid.wdl[0] = value;
            cases.push(invalid);
        }
        let mut invalid_sum = raw;
        invalid_sum.wdl.fill(0.0);
        cases.push(invalid_sum);
        for raw in cases {
            assert_eq!(
                validate_maia_contract(&raw, &[]).unwrap_err(),
                validate_maia(&raw, &[]).unwrap_err()
            );
        }
    }
}
