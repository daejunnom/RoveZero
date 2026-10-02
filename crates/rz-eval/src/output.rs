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
        policy.push((weight(index) / denominator) as f32);
    }
    let sum: f64 = policy.iter().map(|&p| f64::from(p)).sum();
    if policy
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err(OutputError::InvalidPolicyNormalization);
    }
    Ok(ValidatedHeads {
        policy,
        wdl: [raw.wdl[0], raw.wdl[1], raw.wdl[2]],
    })
}
