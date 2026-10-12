use crate::RuntimeFault;
use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

pub const COST_WINDOW: usize = 128;
pub const CALIBRATION_SAMPLES: usize = 8;
pub const MAX_ROLE_BACKLOG: usize = 16;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InvocationClass {
    Fresh,
    Warm,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CacheClass {
    Cold,
    Hit,
    Miss,
}

/// Full semantic digest bytes, never neural-network numeric features.
/// The embedding contract owns canonicalization of model/role/device meaning.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CostClass {
    pub model: [u8; 32],
    pub role_or_checker: [u8; 32],
    pub input_bucket: u16,
    pub invocation: InvocationClass,
    pub cache: CacheClass,
    pub device: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CostStage {
    QueueWait,
    CpuSearch,
    YieldLatency,
    Preparation,
    GpuService,
    PhysicalCompletionWait,
    Acceptance,
}
impl CostStage {
    fn index(self) -> usize {
        self as usize
    }
}
const STAGES: usize = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CensorReason {
    Interrupted,
    Throttled,
    PhysicalUnknown,
    Failed,
    ResourceContention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostDisposition {
    Completed,
    Censored(CensorReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservedCost {
    Known(Duration),
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostObservation {
    pub elapsed: ObservedCost,
    pub disposition: CostDisposition,
}
impl CostObservation {
    fn completed_elapsed(self) -> Option<Duration> {
        match (self.disposition, self.elapsed) {
            (CostDisposition::Completed, ObservedCost::Known(value)) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostEstimate {
    pub ewma: Duration,
    pub p95: Duration,
    /// Completed observations with known elapsed time, eligible for calibration.
    pub completed_in_window: usize,
    pub censored_in_window: usize,
    pub unknown_in_window: usize,
    pub total_completed: u64,
    pub total_censored: u64,
    pub total_unknown: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostCounts {
    pub completed_in_window: usize,
    pub censored_in_window: usize,
    pub unknown_in_window: usize,
    pub total_completed: u64,
    pub total_censored: u64,
    pub total_unknown: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemainingCost {
    Known(Duration),
    /// Includes unfinished jobs with no calibrated conditional tail sample.
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BacklogEstimate {
    Known(Duration),
    /// Known portions are a lower bound, not an estimate of the unknown jobs.
    Unknown {
        known_lower_bound: Duration,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct RunningCost {
    pub class: CostClass,
    pub elapsed: ObservedCost,
}

#[derive(Debug)]
struct Series {
    window: VecDeque<CostObservation>,
    /// Kept sorted at observation time; manager queries do not clone or sort.
    completed: Vec<Duration>,
    ewma_ns: Option<u128>,
    total_completed: u64,
    total_censored: u64,
    total_unknown: u64,
    window_censored: usize,
    window_unknown: usize,
}
impl Series {
    fn try_new() -> Result<Self, RuntimeFault> {
        let mut window = VecDeque::new();
        let mut completed = Vec::new();
        window
            .try_reserve_exact(COST_WINDOW)
            .map_err(|_| RuntimeFault::ResourceLimit)?;
        completed
            .try_reserve_exact(COST_WINDOW)
            .map_err(|_| RuntimeFault::ResourceLimit)?;
        Ok(Self {
            window,
            completed,
            ewma_ns: None,
            total_completed: 0,
            total_censored: 0,
            total_unknown: 0,
            window_censored: 0,
            window_unknown: 0,
        })
    }
    fn observe(&mut self, observation: CostObservation) -> Result<(), RuntimeFault> {
        // Refuse counter overflow before mutating the retained observations.
        let next_completed = self.total_completed.checked_add(u64::from(
            observation.disposition == CostDisposition::Completed,
        ));
        let next_censored = self.total_censored.checked_add(u64::from(matches!(
            observation.disposition,
            CostDisposition::Censored(_)
        )));
        let next_unknown = self
            .total_unknown
            .checked_add(u64::from(observation.elapsed == ObservedCost::Unknown));
        let (Some(next_completed), Some(next_censored), Some(next_unknown)) =
            (next_completed, next_censored, next_unknown)
        else {
            return Err(RuntimeFault::ResourceOverflow);
        };
        if self.window.len() == COST_WINDOW {
            let old = self.window.pop_front().expect("bounded window is nonempty");
            self.window_censored -=
                usize::from(matches!(old.disposition, CostDisposition::Censored(_)));
            self.window_unknown -= usize::from(old.elapsed == ObservedCost::Unknown);
            if let Some(elapsed) = old.completed_elapsed() {
                let position = self
                    .completed
                    .binary_search(&elapsed)
                    .expect("completed observation has one sorted entry");
                self.completed.remove(position);
            }
        }
        self.window.push_back(observation);
        self.window_censored += usize::from(matches!(
            observation.disposition,
            CostDisposition::Censored(_)
        ));
        self.window_unknown += usize::from(observation.elapsed == ObservedCost::Unknown);
        self.total_completed = next_completed;
        self.total_censored = next_censored;
        self.total_unknown = next_unknown;
        if let Some(elapsed) = observation.completed_elapsed() {
            let position = self
                .completed
                .binary_search(&elapsed)
                .unwrap_or_else(|position| position);
            self.completed.insert(position, elapsed);
            self.ewma_ns = Some(match self.ewma_ns {
                None => elapsed.as_nanos(),
                Some(previous) => (previous * 7 + elapsed.as_nanos()) / 8,
            });
        }
        Ok(())
    }
    fn counts(&self) -> CostCounts {
        CostCounts {
            completed_in_window: self.completed.len(),
            censored_in_window: self.window_censored,
            unknown_in_window: self.window_unknown,
            total_completed: self.total_completed,
            total_censored: self.total_censored,
            total_unknown: self.total_unknown,
        }
    }
    fn estimate(&self) -> Option<CostEstimate> {
        if self.completed.len() < CALIBRATION_SAMPLES {
            return None;
        }
        let counts = self.counts();
        Some(CostEstimate {
            ewma: from_ns(self.ewma_ns.expect("completed sample initializes EWMA")),
            p95: percentile95(&self.completed),
            completed_in_window: counts.completed_in_window,
            censored_in_window: counts.censored_in_window,
            unknown_in_window: counts.unknown_in_window,
            total_completed: counts.total_completed,
            total_censored: counts.total_censored,
            total_unknown: counts.total_unknown,
        })
    }
    fn remaining(&self, elapsed: Duration) -> RemainingCost {
        let Some(estimate) = self.estimate() else {
            return RemainingCost::Unknown;
        };
        if elapsed < estimate.p95 {
            return RemainingCost::Known(estimate.p95 - elapsed);
        }
        // An expected completion time is not a physical completion fence.
        // An overdue job needs observed tail evidence; zero is never invented.
        let first_tail = self.completed.partition_point(|sample| *sample <= elapsed);
        let tail = &self.completed[first_tail..];
        if tail.is_empty() {
            RemainingCost::Unknown
        } else {
            RemainingCost::Known(percentile95(tail) - elapsed)
        }
    }
}

fn from_ns(nanos: u128) -> Duration {
    // Every input is Duration and this is its convex average.
    Duration::new(
        (nanos / 1_000_000_000) as u64,
        (nanos % 1_000_000_000) as u32,
    )
}
fn percentile95(sorted: &[Duration]) -> Duration {
    let rank = (sorted.len() * 95).div_ceil(100);
    sorted[rank - 1]
}

/// Bounded and deterministic: no allocation or sorting on an estimate query.
#[derive(Debug)]
pub struct CostEstimator {
    classes: BTreeMap<CostClass, [Series; STAGES]>,
    max_classes: usize,
}
impl CostEstimator {
    pub fn new(max_classes: usize) -> Result<Self, RuntimeFault> {
        if max_classes == 0 || max_classes > 4096 {
            return Err(RuntimeFault::InvalidLimits);
        }
        Ok(Self {
            classes: BTreeMap::new(),
            max_classes,
        })
    }
    pub fn observe(
        &mut self,
        class: CostClass,
        stage: CostStage,
        observation: CostObservation,
    ) -> Result<(), RuntimeFault> {
        if !self.classes.contains_key(&class) {
            if self.classes.len() == self.max_classes {
                return Err(RuntimeFault::ResourceLimit);
            }
            // Allocate the complete bounded class before registering it.
            let series = [
                Series::try_new()?,
                Series::try_new()?,
                Series::try_new()?,
                Series::try_new()?,
                Series::try_new()?,
                Series::try_new()?,
                Series::try_new()?,
            ];
            self.classes.insert(class, series);
        }
        self.classes.get_mut(&class).expect("class is registered")[stage.index()]
            .observe(observation)
    }
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }
    pub fn counts(&self, class: CostClass, stage: CostStage) -> Option<CostCounts> {
        self.classes
            .get(&class)
            .map(|series| series[stage.index()].counts())
    }
    pub fn estimate(&self, class: CostClass, stage: CostStage) -> Option<CostEstimate> {
        self.classes
            .get(&class)
            .and_then(|series| series[stage.index()].estimate())
    }
    pub fn remaining(
        &self,
        class: CostClass,
        stage: CostStage,
        elapsed: Duration,
    ) -> RemainingCost {
        self.classes
            .get(&class)
            .map_or(RemainingCost::Unknown, |series| {
                series[stage.index()].remaining(elapsed)
            })
    }
    /// One physical GPU execution plus at most sixteen logically queued roles.
    /// Cache hits must not be submitted here as physical GPU service jobs.
    pub fn gpu_backlog(
        &self,
        running: Option<RunningCost>,
        queued: &[CostClass],
    ) -> Result<BacklogEstimate, RuntimeFault> {
        if queued.len() > MAX_ROLE_BACKLOG {
            return Err(RuntimeFault::ResourceLimit);
        }
        let mut total = Duration::ZERO;
        let mut unknown = false;
        if let Some(running) = running {
            let remaining = match running.elapsed {
                ObservedCost::Known(elapsed) => {
                    self.remaining(running.class, CostStage::GpuService, elapsed)
                }
                ObservedCost::Unknown => RemainingCost::Unknown,
            };
            match remaining {
                RemainingCost::Known(value) => total = value,
                RemainingCost::Unknown => unknown = true,
            }
        }
        for class in queued {
            match self.estimate(*class, CostStage::GpuService) {
                Some(estimate) => {
                    total = total
                        .checked_add(estimate.p95)
                        .ok_or(RuntimeFault::ResourceOverflow)?
                }
                None => unknown = true,
            }
        }
        Ok(if unknown {
            BacklogEstimate::Unknown {
                known_lower_bound: total,
            }
        } else {
            BacklogEstimate::Known(total)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn class() -> CostClass {
        CostClass {
            model: [1; 32],
            role_or_checker: [2; 32],
            input_bucket: 8,
            invocation: InvocationClass::Fresh,
            cache: CacheClass::Miss,
            device: [3; 32],
        }
    }
    fn complete(estimator: &mut CostEstimator, stage: CostStage, nanos: u64) {
        estimator
            .observe(
                class(),
                stage,
                CostObservation {
                    elapsed: ObservedCost::Known(Duration::from_nanos(nanos)),
                    disposition: CostDisposition::Completed,
                },
            )
            .unwrap();
    }
    #[test]
    fn stage_and_class_calibration_never_use_censored_samples() {
        let mut estimator = CostEstimator::new(2).unwrap();
        for _ in 0..7 {
            complete(&mut estimator, CostStage::GpuService, 80);
        }
        for _ in 0..9 {
            estimator
                .observe(
                    class(),
                    CostStage::GpuService,
                    CostObservation {
                        elapsed: ObservedCost::Known(Duration::from_secs(99)),
                        disposition: CostDisposition::Censored(CensorReason::Interrupted),
                    },
                )
                .unwrap();
        }
        assert!(estimator.estimate(class(), CostStage::GpuService).is_none());
        complete(&mut estimator, CostStage::GpuService, 160);
        let estimate = estimator.estimate(class(), CostStage::GpuService).unwrap();
        assert_eq!(estimate.ewma, Duration::from_nanos(90));
        assert_eq!(estimate.p95, Duration::from_nanos(160));
        assert_eq!(estimate.censored_in_window, 9);
        assert!(estimator.estimate(class(), CostStage::Acceptance).is_none());
        let mut warm = class();
        warm.invocation = InvocationClass::Warm;
        assert!(estimator.estimate(warm, CostStage::GpuService).is_none());
        for _ in 0..COST_WINDOW {
            estimator
                .observe(
                    class(),
                    CostStage::GpuService,
                    CostObservation {
                        elapsed: ObservedCost::Known(Duration::ZERO),
                        disposition: CostDisposition::Censored(CensorReason::Throttled),
                    },
                )
                .unwrap();
        }
        assert!(estimator.estimate(class(), CostStage::GpuService).is_none());
        assert_eq!(
            estimator
                .counts(class(), CostStage::GpuService)
                .unwrap()
                .total_completed,
            8
        );
    }
    #[test]
    fn overdue_physical_work_has_unknown_or_positive_observed_tail() {
        let mut estimator = CostEstimator::new(1).unwrap();
        for value in 1..=20 {
            complete(&mut estimator, CostStage::GpuService, value);
        }
        assert_eq!(
            estimator.remaining(class(), CostStage::GpuService, Duration::from_nanos(19)),
            RemainingCost::Known(Duration::from_nanos(1))
        );
        assert_eq!(
            estimator.remaining(class(), CostStage::GpuService, Duration::from_nanos(20)),
            RemainingCost::Unknown
        );
        assert_eq!(
            estimator
                .gpu_backlog(
                    Some(RunningCost {
                        class: class(),
                        elapsed: ObservedCost::Known(Duration::from_nanos(21))
                    }),
                    &[class()]
                )
                .unwrap(),
            BacklogEstimate::Unknown {
                known_lower_bound: Duration::from_nanos(19)
            }
        );
    }
    #[test]
    fn class_limit_and_queue_limit_are_explicit() {
        let mut estimator = CostEstimator::new(1).unwrap();
        complete(&mut estimator, CostStage::GpuService, 1);
        let mut other = class();
        other.device = [4; 32];
        assert_eq!(
            estimator.observe(
                other,
                CostStage::GpuService,
                CostObservation {
                    elapsed: ObservedCost::Known(Duration::ZERO),
                    disposition: CostDisposition::Completed
                }
            ),
            Err(RuntimeFault::ResourceLimit)
        );
        assert_eq!(estimator.class_count(), 1);
        assert_eq!(
            estimator.gpu_backlog(None, &[class(); 17]),
            Err(RuntimeFault::ResourceLimit)
        );
    }
    #[test]
    fn unknown_completed_and_censored_costs_remain_independent_facts() {
        let mut estimator = CostEstimator::new(1).unwrap();
        for _ in 0..7 {
            complete(&mut estimator, CostStage::GpuService, 8);
        }
        estimator
            .observe(
                class(),
                CostStage::GpuService,
                CostObservation {
                    elapsed: ObservedCost::Unknown,
                    disposition: CostDisposition::Completed,
                },
            )
            .unwrap();
        estimator
            .observe(
                class(),
                CostStage::GpuService,
                CostObservation {
                    elapsed: ObservedCost::Unknown,
                    disposition: CostDisposition::Censored(CensorReason::ResourceContention),
                },
            )
            .unwrap();
        assert!(estimator.estimate(class(), CostStage::GpuService).is_none());
        let counts = estimator.counts(class(), CostStage::GpuService).unwrap();
        assert_eq!(counts.completed_in_window, 7);
        assert_eq!(counts.censored_in_window, 1);
        assert_eq!(counts.unknown_in_window, 2);
        assert_eq!(counts.total_completed, 8);
        assert_eq!(counts.total_censored, 1);
        assert_eq!(counts.total_unknown, 2);
        complete(&mut estimator, CostStage::GpuService, 8);
        let estimate = estimator.estimate(class(), CostStage::GpuService).unwrap();
        assert_eq!(estimate.p95, Duration::from_nanos(8));
        assert_eq!(estimate.unknown_in_window, 2);
    }
}
