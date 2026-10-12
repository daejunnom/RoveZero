use super::{BacklogEstimate, CostEstimate, CALIBRATION_SAMPLES};
use crate::RuntimeFault;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinatorPolicy {
    Synchronous,
    Fixed,
    Adaptive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControllerConfig {
    pub policy: CoordinatorPolicy,
    pub fixed_node_credit: u64,
    pub fixed_soft_lease: Duration,
    pub initial_workers: u8,
    /// Caller must separately admit this resource/topology profile.
    pub max_workers: u8,
    pub consecutive_observations: u8,
    pub worker_cooldown: Duration,
}
impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            policy: CoordinatorPolicy::Fixed,
            fixed_node_credit: 4096,
            fixed_soft_lease: Duration::from_millis(8),
            initial_workers: 1,
            max_workers: 1,
            consecutive_observations: 3,
            worker_cooldown: Duration::from_secs(2),
        }
    }
}
impl ControllerConfig {
    fn validate(self) -> Result<Self, RuntimeFault> {
        if self.fixed_node_credit == 0
            || self.fixed_node_credit > 4096
            || self.fixed_soft_lease.is_zero()
            || self.fixed_soft_lease > Duration::from_millis(8)
            || ![1, 2, 4].contains(&self.initial_workers)
            || ![1, 2, 4].contains(&self.max_workers)
            || self.initial_workers > self.max_workers
            || self.consecutive_observations < 2
            || self.worker_cooldown < Duration::from_secs(2)
        {
            return Err(RuntimeFault::InvalidLimits);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerProgress {
    pub has_new_consistent_progress: bool,
    /// Unknown gain never authorizes an adaptive Yield.
    pub expected_feedback_gain: Option<Duration>,
}

#[derive(Clone, Copy, Debug)]
pub struct ControllerInput {
    /// Must advance only for a new relevant observation, not every manager tick.
    pub observation_sequence: u64,
    /// Monotonic time in the owner-selected clock domain.
    pub now: Duration,
    pub deadline_remaining: Duration,
    pub role_service: Option<CostEstimate>,
    pub yield_overhead: Option<CostEstimate>,
    pub backlog: BacklogEstimate,
    pub waiting_cpu_jobs: usize,
    /// Advice addresses only this selected worker, never all workers at once.
    pub selected_progress: Option<WorkerProgress>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BacklogBand {
    Low,
    Middle,
    High,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SliceAdvice {
    pub node_credit: u64,
    pub soft_lease: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControllerAdvice {
    pub calibrated: bool,
    pub band: BacklogBand,
    pub workers: u8,
    pub workers_changed: bool,
    pub slice: Option<SliceAdvice>,
    pub yield_selected_worker: bool,
    /// GPU policy is always one physical execution, immediate B1 dispatch.
    pub physical_gpu_limit: u8,
    pub gpu_batch_wait: Duration,
}

/// Deterministic advice only. No work, thread, NN call or archive I/O runs here.
#[derive(Debug)]
pub struct AdaptiveController {
    config: ControllerConfig,
    workers: u8,
    last_sequence: u64,
    last_now: Duration,
    last_worker_change: Duration,
    last_band: BacklogBand,
    consecutive: u8,
}
impl AdaptiveController {
    pub fn new(config: ControllerConfig) -> Result<Self, RuntimeFault> {
        let config = config.validate()?;
        Ok(Self {
            workers: config.initial_workers,
            config,
            last_sequence: 0,
            last_now: Duration::ZERO,
            last_worker_change: Duration::ZERO,
            last_band: BacklogBand::Unknown,
            consecutive: 0,
        })
    }
    pub fn workers(&self) -> u8 {
        self.workers
    }
    pub fn advise(&mut self, input: ControllerInput) -> Result<ControllerAdvice, RuntimeFault> {
        if input.now < self.last_now || input.observation_sequence < self.last_sequence {
            return Err(RuntimeFault::Stale);
        }
        self.last_now = input.now;
        let calibrated = self.config.policy == CoordinatorPolicy::Adaptive
            && input
                .role_service
                .is_some_and(|sample| sample.completed_in_window >= CALIBRATION_SAMPLES)
            && input
                .yield_overhead
                .is_some_and(|sample| sample.completed_in_window >= CALIBRATION_SAMPLES);
        let band = if calibrated {
            match (input.backlog, input.role_service) {
                (BacklogEstimate::Known(backlog), Some(service)) if !service.p95.is_zero() => {
                    if backlog <= service.p95 {
                        BacklogBand::Low
                    } else if service
                        .p95
                        .checked_mul(3)
                        .is_some_and(|high| backlog >= high)
                    {
                        BacklogBand::High
                    } else {
                        BacklogBand::Middle
                    }
                }
                _ => BacklogBand::Unknown,
            }
        } else {
            BacklogBand::Unknown
        };
        let minimum_slice = if calibrated {
            input
                .yield_overhead
                .and_then(|sample| sample.p95.checked_mul(4))
        } else {
            Some(self.config.fixed_soft_lease)
        };
        let slice = minimum_slice.and_then(|minimum| {
            let lease = self.config.fixed_soft_lease.max(minimum);
            let reserve = if calibrated {
                input
                    .yield_overhead
                    .map_or(Duration::ZERO, |sample| sample.p95)
            } else {
                Duration::ZERO
            };
            let available = input.deadline_remaining.checked_sub(reserve)?;
            if available < minimum || available.is_zero() {
                return None;
            }
            Some(SliceAdvice {
                node_credit: self.config.fixed_node_credit,
                soft_lease: lease.min(available),
            })
        });
        let mut workers_changed = false;
        if input.observation_sequence > self.last_sequence {
            self.last_sequence = input.observation_sequence;
            if calibrated && matches!(band, BacklogBand::Low | BacklogBand::High) {
                self.consecutive = if band == self.last_band {
                    self.consecutive.saturating_add(1)
                } else {
                    1
                };
            } else {
                self.consecutive = 0;
            }
            self.last_band = band;
            if calibrated
                && slice.is_some()
                && self.consecutive >= self.config.consecutive_observations
                && input.now.saturating_sub(self.last_worker_change) >= self.config.worker_cooldown
            {
                let target = match band {
                    BacklogBand::High => (self.workers / 2).max(1),
                    BacklogBand::Low if input.waiting_cpu_jobs > usize::from(self.workers) => {
                        (self.workers * 2).min(self.config.max_workers)
                    }
                    _ => self.workers,
                };
                if target != self.workers {
                    self.workers = target;
                    self.last_worker_change = input.now;
                    self.consecutive = 0;
                    workers_changed = true;
                }
            }
        }
        let yield_selected_worker = calibrated
            && slice.is_some()
            && matches!(band, BacklogBand::Low | BacklogBand::Middle)
            && input.selected_progress.is_some_and(|progress| {
                progress.has_new_consistent_progress
                    && progress
                        .expected_feedback_gain
                        .zip(input.yield_overhead)
                        .is_some_and(|(gain, overhead)| gain > overhead.p95)
            });
        Ok(ControllerAdvice {
            calibrated,
            band,
            workers: self.workers,
            workers_changed,
            slice,
            yield_selected_worker,
            physical_gpu_limit: 1,
            gpu_batch_wait: Duration::ZERO,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn estimate(ms: u64) -> CostEstimate {
        CostEstimate {
            ewma: Duration::from_millis(ms),
            p95: Duration::from_millis(ms),
            completed_in_window: 8,
            censored_in_window: 0,
            unknown_in_window: 0,
            total_completed: 8,
            total_censored: 0,
            total_unknown: 0,
        }
    }
    fn input(sequence: u64, now_ms: u64) -> ControllerInput {
        ControllerInput {
            observation_sequence: sequence,
            now: Duration::from_millis(now_ms),
            deadline_remaining: Duration::from_secs(1),
            role_service: Some(estimate(10)),
            yield_overhead: Some(estimate(3)),
            backlog: BacklogEstimate::Known(Duration::ZERO),
            waiting_cpu_jobs: 64,
            selected_progress: Some(WorkerProgress {
                has_new_consistent_progress: true,
                expected_feedback_gain: Some(Duration::from_millis(9)),
            }),
        }
    }
    fn controller() -> AdaptiveController {
        AdaptiveController::new(ControllerConfig {
            policy: CoordinatorPolicy::Adaptive,
            max_workers: 4,
            ..ControllerConfig::default()
        })
        .unwrap()
    }
    #[test]
    fn calibration_and_deadline_bound_admission_and_yield() {
        let mut controller = controller();
        let mut first = input(1, 0);
        first.role_service.as_mut().unwrap().completed_in_window = 7;
        let advice = controller.advise(first).unwrap();
        assert!(!advice.calibrated);
        assert!(!advice.yield_selected_worker);
        assert_eq!(advice.slice.unwrap().soft_lease, Duration::from_millis(8));
        let advice = controller.advise(input(2, 1)).unwrap();
        assert_eq!(advice.slice.unwrap().soft_lease, Duration::from_millis(12));
        assert!(advice.yield_selected_worker);
        let mut late = input(3, 2);
        late.deadline_remaining = Duration::from_millis(14);
        let advice = controller.advise(late).unwrap();
        assert!(advice.slice.is_none());
        assert!(!advice.yield_selected_worker);
        assert_eq!(advice.physical_gpu_limit, 1);
        assert_eq!(advice.gpu_batch_wait, Duration::ZERO);
    }
    #[test]
    fn worker_hysteresis_counts_new_observations_and_enforces_two_second_cooldown() {
        let mut controller = controller();
        assert!(!controller.advise(input(1, 2000)).unwrap().workers_changed);
        assert!(!controller.advise(input(1, 2001)).unwrap().workers_changed);
        assert!(!controller.advise(input(2, 2002)).unwrap().workers_changed);
        assert_eq!(controller.advise(input(3, 2003)).unwrap().workers, 2);
        for sequence in 4..=6 {
            assert!(
                !controller
                    .advise(input(sequence, 2003 + sequence))
                    .unwrap()
                    .workers_changed
            );
        }
        assert_eq!(controller.advise(input(7, 4003)).unwrap().workers, 4);
        let mut high = input(8, 6004);
        high.backlog = BacklogEstimate::Known(Duration::from_millis(30));
        controller.advise(high).unwrap();
        high.observation_sequence = 9;
        controller.advise(high).unwrap();
        high.observation_sequence = 10;
        assert_eq!(controller.advise(high).unwrap().workers, 2);
    }
    #[test]
    fn unknown_backlog_and_unknown_gain_never_authorize_yield() {
        let mut controller = controller();
        let mut unknown = input(1, 2000);
        unknown.backlog = BacklogEstimate::Unknown {
            known_lower_bound: Duration::from_secs(3),
        };
        let advice = controller.advise(unknown).unwrap();
        assert_eq!(advice.band, BacklogBand::Unknown);
        assert!(!advice.yield_selected_worker);
        unknown.observation_sequence = 2;
        unknown.backlog = BacklogEstimate::Known(Duration::ZERO);
        unknown
            .selected_progress
            .as_mut()
            .unwrap()
            .expected_feedback_gain = None;
        assert!(!controller.advise(unknown).unwrap().yield_selected_worker);
        unknown.observation_sequence = 1;
        assert_eq!(controller.advise(unknown), Err(RuntimeFault::Stale));
    }
}
