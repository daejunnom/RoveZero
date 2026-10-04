//! Sequence-based wakeups. This channel never carries or consumes a result.
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct CompletionSignal(Arc<(Mutex<u64>, Condvar)>);
impl CompletionSignal {
    pub fn version(&self) -> u64 {
        *self.0 .0.lock().unwrap_or_else(|p| p.into_inner())
    }
    pub fn notify(&self) {
        let mut version = self.0 .0.lock().unwrap_or_else(|p| p.into_inner());
        *version = version.saturating_add(1);
        self.0 .1.notify_all();
    }
    /// Call version BEFORE rechecking completion/cancellation/deadline. A wake
    /// in that gap changes the predicate, so entering wait cannot lose it.
    /// Saturation disables blocking; signal poison also permits another poll.
    pub fn wait_changed(&self, observed: u64, timeout: Duration) -> bool {
        let Ok(version) = self.0 .0.lock() else {
            return true;
        };
        if *version != observed || *version == u64::MAX {
            return true;
        }
        match self
            .0
             .1
            .wait_timeout_while(version, timeout, |v| *v == observed && *v != u64::MAX)
        {
            Ok((version, _)) => *version != observed || *version == u64::MAX,
            Err(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wake_before_wait_during_wait_and_deadline_are_bounded() {
        let signal = CompletionSignal::default();
        let version = signal.version();
        signal.notify();
        assert!(signal.wait_changed(version, Duration::from_secs(1)));
        let observed = signal.version();
        let cloned = signal.clone();
        let waiter =
            std::thread::spawn(move || cloned.wait_changed(observed, Duration::from_secs(1)));
        signal.notify();
        assert!(waiter.join().unwrap());
        assert!(!signal.wait_changed(signal.version(), Duration::ZERO));
    }
}
