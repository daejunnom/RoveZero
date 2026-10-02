//! A manually clocked backend simulator, without threads, sleeps, or a runtime.
//!
//! Tickets are supplied by the caller, so actual contract IDs/generations can be
//! preserved without inventing parallel shared types here. A callback is only a
//! candidate result: duplicate, canceled, stale, partial, and invalid callbacks
//! must still be rejected at the real runtime/adapter boundary. Cancellation
//! never removes a scheduled physical completion.

use crate::RawOutput;
use sha2::{Digest as _, Sha256};
use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InjectedFailure {
    pub code: String,
    pub stage: String,
}

/// These are backend observations, not logical terminal outcomes.
#[derive(Clone, Debug, PartialEq)]
pub enum EventKind {
    Callback(Result<RawOutput, InjectedFailure>),
    CancelAcknowledged,
    DeviceCompleted,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Callback {
    /// Delay relative to submission, in the mock clock domain.
    pub after: Duration,
    pub reply: Result<RawOutput, InjectedFailure>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// Multiple callbacks intentionally permit duplicate delivery tests.
    pub callbacks: Vec<Callback>,
    pub device_complete_after: Duration,
    /// None simulates a backend that cannot acknowledge cancellation.
    pub cancel_ack_after: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptIdentity {
    pub name: String,
    /// A replay annotation. Script contents, not a hidden RNG, define events.
    pub seed: u64,
}

/// Finite simulator limits; these do not claim to measure process memory.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_steps: usize,
    pub max_in_flight: usize,
    pub max_pending_events: usize,
    pub max_script_values: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_steps: 128,
            max_in_flight: 16,
            max_pending_events: 128,
            max_script_values: 1_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event<K> {
    pub ticket: K,
    pub script_step: usize,
    /// Scheduled time, even when the consumer polls later.
    pub at: Duration,
    pub kind: EventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockError {
    InvalidLimits,
    InvalidScriptIdentity,
    ScriptLimit,
    ScriptExhausted,
    InFlightLimit,
    EventLimit,
    DuplicateTicket,
    UnknownTicket,
    ClockMovedBackwards,
    TimeOverflow,
    SequenceOverflow,
    AllocationFailed,
}

impl fmt::Display for MockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mock backend: {self:?}")
    }
}

impl std::error::Error for MockError {}

#[derive(Debug)]
struct Scheduled<K> {
    order: u64,
    event: Event<K>,
}

#[derive(Debug)]
struct Active<K> {
    ticket: K,
    step: usize,
    remaining: usize,
    cancel_requested: bool,
    cancel_ack_after: Option<Duration>,
}

#[derive(Debug)]
pub struct ScriptedBackend<K> {
    identity: ScriptIdentity,
    digest: [u8; 32],
    limits: Limits,
    steps: VecDeque<Step>,
    active: Vec<Active<K>>,
    events: Vec<Scheduled<K>>,
    now: Duration,
    next_step: usize,
    next_order: u64,
}

impl<K: Clone + Eq> ScriptedBackend<K> {
    pub fn new(
        identity: ScriptIdentity,
        steps: Vec<Step>,
        limits: Limits,
    ) -> Result<Self, MockError> {
        if limits.max_steps == 0
            || limits.max_in_flight == 0
            || limits.max_pending_events == 0
            || limits.max_script_values == 0
        {
            return Err(MockError::InvalidLimits);
        }
        if identity.name.is_empty() || identity.name.len() > 256 {
            return Err(MockError::InvalidScriptIdentity);
        }
        if steps.len() > limits.max_steps {
            return Err(MockError::ScriptLimit);
        }
        let mut values = 0usize;
        for step in &steps {
            let events = step
                .callbacks
                .len()
                .checked_add(1)
                .ok_or(MockError::EventLimit)?;
            if events > limits.max_pending_events {
                return Err(MockError::EventLimit);
            }
            for callback in &step.callbacks {
                let size = match &callback.reply {
                    Ok(raw) => raw.policy_logits.len().checked_add(raw.wdl.len()),
                    Err(error) => error.code.len().checked_add(error.stage.len()),
                }
                .ok_or(MockError::ScriptLimit)?;
                values = values.checked_add(size).ok_or(MockError::ScriptLimit)?;
                if values > limits.max_script_values {
                    return Err(MockError::ScriptLimit);
                }
            }
        }
        let digest = script_digest(&identity, &steps);
        Ok(Self {
            identity,
            digest,
            limits,
            steps: steps.into(),
            active: Vec::new(),
            events: Vec::new(),
            now: Duration::ZERO,
            next_step: 0,
            next_order: 0,
        })
    }

    pub fn identity(&self) -> &ScriptIdentity {
        &self.identity
    }

    /// Actual ordered script receipt, including delays, injected errors and raw
    /// float bits. Name/seed annotations alone never identify an execution plan.
    pub fn identity_digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn limits(&self) -> Limits {
        self.limits
    }
    pub fn pending_events(&self) -> usize {
        self.events.len()
    }

    pub fn now(&self) -> Duration {
        self.now
    }

    pub fn in_flight(&self) -> usize {
        self.active.len()
    }

    /// Advances the clock without delivering events, allowing the consumer to
    /// interleave cancellation/root changes with individual completion events.
    pub fn advance_to(&mut self, now: Duration) -> Result<(), MockError> {
        if now < self.now {
            return Err(MockError::ClockMovedBackwards);
        }
        self.now = now;
        Ok(())
    }

    /// Admission errors neither consume a script step nor schedule partial work.
    pub fn submit(&mut self, ticket: K) -> Result<usize, MockError> {
        if self.active.iter().any(|active| active.ticket == ticket) {
            return Err(MockError::DuplicateTicket);
        }
        if self.active.len() >= self.limits.max_in_flight {
            return Err(MockError::InFlightLimit);
        }
        let step = self.steps.front().ok_or(MockError::ScriptExhausted)?;
        let count = step.callbacks.len() + 1; // Checked by new().
        self.check_event_capacity(count)?;
        for delay in step
            .callbacks
            .iter()
            .map(|callback| callback.after)
            .chain(std::iter::once(step.device_complete_after))
        {
            self.now.checked_add(delay).ok_or(MockError::TimeOverflow)?;
        }
        self.events
            .try_reserve(count)
            .map_err(|_| MockError::AllocationFailed)?;
        self.active
            .try_reserve(1)
            .map_err(|_| MockError::AllocationFailed)?;

        let step = self.steps.pop_front().expect("front checked above");
        let index = self.next_step;
        self.next_step += 1;
        self.active.push(Active {
            ticket: ticket.clone(),
            step: index,
            remaining: count,
            cancel_requested: false,
            cancel_ack_after: step.cancel_ack_after,
        });
        for callback in step.callbacks {
            self.schedule(
                ticket.clone(),
                index,
                self.now + callback.after,
                EventKind::Callback(callback.reply),
            );
        }
        self.schedule(
            ticket,
            index,
            self.now + step.device_complete_after,
            EventKind::DeviceCompleted,
        );
        Ok(index)
    }

    /// True on the first cancellation request. Repeated cancellation is inert.
    /// Outputs and device completion remain scheduled, including late callbacks.
    pub fn cancel(&mut self, ticket: &K) -> Result<bool, MockError> {
        let index = self
            .active
            .iter()
            .position(|active| &active.ticket == ticket)
            .ok_or(MockError::UnknownTicket)?;
        let active = &self.active[index];
        if active.cancel_requested {
            return Ok(false);
        }
        if let Some(delay) = active.cancel_ack_after {
            let due = self.now.checked_add(delay).ok_or(MockError::TimeOverflow)?;
            self.check_event_capacity(1)?;
            self.events
                .try_reserve(1)
                .map_err(|_| MockError::AllocationFailed)?;
            let step = self.active[index].step;
            self.schedule(ticket.clone(), step, due, EventKind::CancelAcknowledged);
            self.active[index].remaining += 1;
        }
        self.active[index].cancel_requested = true;
        Ok(true)
    }

    /// Delivers one due event, ordered by scheduled time then insertion order.
    /// Consumer lag never rewrites the physical event's original timestamp.
    pub fn next_event(&mut self) -> Option<Event<K>> {
        let index = self
            .events
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.event.at <= self.now)
            .min_by_key(|(_, entry)| (entry.event.at, entry.order))
            .map(|(index, _)| index)?;
        let event = self.events.swap_remove(index).event;
        let active_index = self
            .active
            .iter()
            .position(|active| active.ticket == event.ticket)
            .expect("scheduled events keep their owner alive");
        self.active[active_index].remaining -= 1;
        if self.active[active_index].remaining == 0 {
            self.active.swap_remove(active_index);
        }
        Some(event)
    }

    fn check_event_capacity(&self, count: usize) -> Result<(), MockError> {
        if self
            .events
            .len()
            .checked_add(count)
            .filter(|&total| total <= self.limits.max_pending_events)
            .is_none()
        {
            return Err(MockError::EventLimit);
        }
        let count = u64::try_from(count).map_err(|_| MockError::SequenceOverflow)?;
        self.next_order
            .checked_add(count)
            .ok_or(MockError::SequenceOverflow)?;
        Ok(())
    }

    fn schedule(&mut self, ticket: K, step: usize, at: Duration, kind: EventKind) {
        // Callers preflight sequence/time/capacity before mutating the script.
        let order = self.next_order;
        self.next_order += 1;
        self.events.push(Scheduled {
            order,
            event: Event {
                ticket,
                script_step: step,
                at,
                kind,
            },
        });
    }
}

fn script_digest(identity: &ScriptIdentity, steps: &[Step]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-scripted-backend/1\0");
    hash.update((identity.name.len() as u64).to_le_bytes());
    hash.update(identity.name.as_bytes());
    hash.update(identity.seed.to_le_bytes());
    hash.update((steps.len() as u64).to_le_bytes());
    let duration = |hash: &mut Sha256, value: Duration| {
        hash.update(value.as_secs().to_le_bytes());
        hash.update(value.subsec_nanos().to_le_bytes());
    };
    for step in steps {
        hash.update((step.callbacks.len() as u64).to_le_bytes());
        for callback in &step.callbacks {
            duration(&mut hash, callback.after);
            match &callback.reply {
                Ok(raw) => {
                    hash.update([0]);
                    for values in [&raw.policy_logits, &raw.wdl] {
                        hash.update((values.len() as u64).to_le_bytes());
                        for value in values {
                            hash.update(value.to_bits().to_le_bytes());
                        }
                    }
                }
                Err(error) => {
                    hash.update([1]);
                    for text in [&error.code, &error.stage] {
                        hash.update((text.len() as u64).to_le_bytes());
                        hash.update(text.as_bytes());
                    }
                }
            }
        }
        duration(&mut hash, step.device_complete_after);
        hash.update([u8::from(step.cancel_ack_after.is_some())]);
        if let Some(delay) = step.cancel_ack_after {
            duration(&mut hash, delay);
        }
    }
    hash.finalize().into()
}
