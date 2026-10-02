use crate::tree::SearchError;

/// Statistics of one parent edge, always expressed from that parent's viewpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeStats {
    pub prior: f64,
    pub visits: u64,
    pub value_sum: f64,
}

impl EdgeStats {
    pub fn q(self) -> f64 {
        if self.visits == 0 { 0.0 } else { self.value_sum / self.visits as f64 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyIdentity {
    pub algorithm: &'static str,
    pub revision: u32,
    pub configuration: String,
}

/// Replaceable selection only. The tree retains ownership of visits and backup.
pub trait SelectionPolicy {
    fn identity(&self) -> PolicyIdentity;
    /// Return an index in the supplied ordered legal edges.
    fn select(&self, edges: &[EdgeStats]) -> Result<usize, SearchError>;
}

#[derive(Clone, Copy, Debug)]
pub struct Puct {
    c_puct: f64,
}

impl Default for Puct {
    fn default() -> Self { Self { c_puct: 1.5 } }
}

impl Puct {
    pub fn new(c_puct: f64) -> Result<Self, SearchError> {
        if !c_puct.is_finite() || !(0.0..=1_000_000.0).contains(&c_puct) {
            return Err(SearchError::InvalidConfiguration("c_puct"));
        }
        Ok(Self { c_puct })
    }

    pub fn scores(&self, edges: &[EdgeStats]) -> Result<Vec<f64>, SearchError> {
        if edges.is_empty() { return Err(SearchError::NoLegalEdges); }
        let parent_visits = edges.iter().try_fold(0_u64, |sum, e| {
            sum.checked_add(e.visits).ok_or(SearchError::CounterOverflow)
        })?;
        let scale = (parent_visits.max(1) as f64).sqrt();
        edges.iter().map(|e| {
            if !e.prior.is_finite() || !(0.0..=1.0).contains(&e.prior)
                || !e.value_sum.is_finite() || e.value_sum.abs() > e.visits as f64
            {
                return Err(SearchError::InvalidStatistics);
            }
            let score = e.q() + self.c_puct * e.prior * scale / (1.0 + e.visits as f64);
            if !score.is_finite() { return Err(SearchError::InvalidStatistics); }
            Ok(score)
        }).collect()
    }
}

impl SelectionPolicy for Puct {
    fn identity(&self) -> PolicyIdentity {
        PolicyIdentity {
            algorithm: "rz-puct",
            revision: 1,
            configuration: format!("c_puct={};fpu=0;root_noise=off;ties=legal-order;scalar=f64", self.c_puct),
        }
    }

    fn select(&self, edges: &[EdgeStats]) -> Result<usize, SearchError> {
        let scores = self.scores(edges)?;
        let mut best = 0;
        for i in 1..scores.len() {
            // Strict comparison preserves the supplied legal order for exact ties.
            if scores[i] > scores[best] { best = i; }
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_score_vectors() {
        let p = Puct::default();
        let cases = [
            ([EdgeStats { prior: 0.5, visits: 0, value_sum: 0.0 }; 2], [0.75, 0.75], 0),
            ([EdgeStats { prior: 0.5, visits: 1, value_sum: -1.0 }, EdgeStats { prior: 0.5, visits: 0, value_sum: 0.0 }], [-0.625, 0.75], 1),
            ([EdgeStats { prior: 0.1, visits: 3, value_sum: 3.0 }, EdgeStats { prior: 0.9, visits: 1, value_sum: -1.0 }], [1.075, 0.35], 0),
        ];
        for (edges, expected, selected) in cases {
            for (actual, reference) in p.scores(&edges).unwrap().iter().zip(expected) {
                assert!((actual-reference).abs() < 1e-12);
            }
            assert_eq!(p.select(&edges).unwrap(), selected);
        }
    }

    #[test]
    fn malformed_statistics_and_configuration_are_errors() {
        assert!(Puct::new(f64::NAN).is_err());
        assert!(Puct::new(-1.0).is_err());
        assert_eq!(Puct::default().select(&[]), Err(SearchError::NoLegalEdges));
        let bad = [EdgeStats { prior: 0.5, visits: 0, value_sum: 1.0 }];
        assert_eq!(Puct::default().select(&bad), Err(SearchError::InvalidStatistics));
        let overflow = [EdgeStats { prior: 0.5, visits: u64::MAX, value_sum: 0.0 }, EdgeStats { prior: 0.5, visits: 1, value_sum: 0.0 }];
        assert_eq!(Puct::default().select(&overflow), Err(SearchError::CounterOverflow));
    }
}
