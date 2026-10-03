//! Opt-in, bounded host-source D02 journal, separate from CPU/CUDA V1 receipts.
//! Native Run is a synchronous host interval including provider internals. No
//! kernel-only timing, memcpy timing or process VRAM is inferred from it.

use crate::native_attestation::AttestationError;
use crate::native_bootstrap::{NativeConfig, NativeRunError, NativeRunReport};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use rz_contracts::{CompletionContext, EvalContext, ExecutionId, RequestId};
use rz_telemetry::{
    profile::SampleSummary,
    source::{SourceJournal, SourceSnapshot, SourceSpan, SourceStage},
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, Write},
    time::{Duration, Instant},
};

pub const PROFILE_FILE: &str = "native-d02.v1.json";
const MAX_PROFILE_BYTES: usize = 8 * 1024 * 1024;

/// Pass-through writes and flushes. It copies no protocol text and grants no
/// publication authority. Spans describe the process output stream, not an eval.
pub struct ProfiledWrite<W> {
    inner: W,
    trace: Option<SourceJournal<CompletionContext>>,
}
impl<W> ProfiledWrite<W> {
    pub fn new(inner: W, trace: Option<SourceJournal<CompletionContext>>) -> Self {
        Self { inner, trace }
    }
}
impl<W: Write> Write for ProfiledWrite<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let start = self.trace.as_ref().map(|_| Instant::now());
        let result = self.inner.write(bytes);
        if let (Some(trace), Some(start)) = (&self.trace, start) {
            trace.record(
                None,
                SourceStage::ProtocolWrite,
                start,
                Instant::now(),
                result.is_ok(),
            );
        }
        result
    }
    fn flush(&mut self) -> io::Result<()> {
        let start = self.trace.as_ref().map(|_| Instant::now());
        let result = self.inner.flush();
        if let (Some(trace), Some(start)) = (&self.trace, start) {
            trace.record(
                None,
                SourceStage::ProtocolWrite,
                start,
                Instant::now(),
                result.is_ok(),
            );
        }
        result
    }
}

pub struct ProfileWriter {
    _directory: Dir,
    file: cap_std::fs::File,
    attempted: bool,
}
impl ProfileWriter {
    /// Reserve a fresh private file before timed protocol service. Failure is
    /// explicit; an existing report is never overwritten, even after PID reuse.
    pub fn open(config: &NativeConfig) -> Result<Self, AttestationError> {
        let root = config
            .attestation_output_root()
            .map_err(|_| AttestationError::boundary("invalid D02 output root"))?;
        let parent = root
            .parent()
            .ok_or_else(|| AttestationError::boundary("D02 output root needs a parent"))?;
        let name = root
            .file_name()
            .ok_or_else(|| AttestationError::boundary("D02 output root needs a name"))?;
        let parent = Dir::open_ambient_dir(parent, ambient_authority())
            .map_err(|error| AttestationError::io("pin D02 output parent", error))?;
        let root = parent
            .open_dir_nofollow(name)
            .map_err(|error| AttestationError::io("pin D02 output directory", error))?;
        let slot = format!("native-d02-process-{}", std::process::id());
        root.create_dir(&slot)
            .map_err(|error| AttestationError::io("reserve fresh D02 process slot", error))?;
        let directory = root
            .open_dir_nofollow(&slot)
            .map_err(|error| AttestationError::io("pin D02 process slot", error))?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let file = directory
            .open_with(PROFILE_FILE, &options)
            .map_err(|error| AttestationError::io("reserve new D02 profile", error))?;
        Ok(Self {
            _directory: directory,
            file,
            attempted: false,
        })
    }
    /// Called after factory finish. A failed/unconfirmed drain yields an
    /// explicitly incomplete snapshot; it cannot attest producer shutdown.
    pub fn publish(
        &mut self,
        trace: &SourceJournal<CompletionContext>,
        finished: &Result<NativeRunReport, NativeRunError>,
    ) -> Result<(), AttestationError> {
        if self.attempted {
            return Err(AttestationError::boundary(
                "D02 publication already attempted",
            ));
        }
        self.attempted = true;
        let snapshot = trace.snapshot();
        let report = finished.as_ref().ok().or_else(|| {
            finished
                .as_ref()
                .err()
                .and_then(|error| error.report.as_deref())
        });
        let document = ProfileDocument::capture(&snapshot, finished.is_ok(), report);
        let mut bounded = BoundedWrite {
            inner: &mut self.file,
            written: 0,
        };
        serde_json::to_writer(&mut bounded, &document)
            .map_err(|_| AttestationError::boundary("cannot encode bounded D02 profile"))?;
        bounded
            .write_all(b"\n")
            .map_err(|error| AttestationError::io("finish bounded D02 profile", error))?;
        self.file
            .sync_all()
            .map_err(|error| AttestationError::io("sync D02 profile", error))
    }
}

struct BoundedWrite<W> {
    inner: W,
    written: usize,
}
impl<W: Write> Write for BoundedWrite<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_PROFILE_BYTES.saturating_sub(self.written) {
            return Err(io::Error::other("D02 profile exceeds its byte limit"));
        }
        let written = self.inner.write(bytes)?;
        self.written += written;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Serialize)]
struct Distribution {
    samples: usize,
    p50_ns: Option<u64>,
    p95_ns: Option<u64>,
    p99_ns: Option<u64>,
}
impl Distribution {
    fn from_samples(samples: Vec<Duration>) -> Self {
        let summary = SampleSummary::from_complete_samples(samples);
        Self {
            samples: summary.retained,
            p50_ns: summary.p50.and_then(nanos),
            p95_ns: summary.p95.and_then(nanos),
            p99_ns: summary.p99.and_then(nanos),
        }
    }
}

#[derive(Serialize)]
struct KeyWire {
    process_epoch: u64,
    request: u64,
    selection: u64,
    game: u64,
    root: u64,
    execution: Option<(u64, u64)>,
    state_owner: u64,
    state_revision: u64,
    state_digest: String,
    input_digest: String,
    legal_order_digest: String,
    model_manifest: String,
    encoding_manifest: String,
    backend_digest: String,
}
impl From<CompletionContext> for KeyWire {
    fn from(key: CompletionContext) -> Self {
        let context = key.request;
        Self {
            process_epoch: context.request.epoch.0,
            request: context.request.sequence,
            selection: context.selection.sequence,
            game: context.game.0,
            root: context.root.0,
            execution: key.execution.map(|id| (id.epoch.0, id.sequence)),
            state_owner: context.state.owner.0,
            state_revision: context.state.revision.0,
            state_digest: hex(context.state.semantic.0),
            input_digest: hex(context.input.0.0),
            legal_order_digest: hex(context.legal_order.0.0),
            model_manifest: hex(context.model.manifest.0),
            encoding_manifest: hex(context.encoding.manifest.0),
            backend_digest: hex(context.backend.0),
        }
    }
}
#[derive(Serialize)]
struct SpanWire {
    arrival_index: usize,
    stage: String,
    key: Option<KeyWire>,
    start_ns: Option<u64>,
    end_ns: Option<u64>,
    succeeded: bool,
}
#[derive(Default)]
struct Frame<'a> {
    context: Option<EvalContext>,
    execution: Option<ExecutionId>,
    stages: BTreeMap<SourceStage, &'a SourceSpan<CompletionContext>>,
}
#[derive(Serialize)]
struct ProfileDocument {
    schema: &'static str,
    source_clock: &'static str,
    journal_capacity: usize,
    producers_joined: bool,
    journal_complete: bool,
    accepted_request_timeline_complete: bool,
    dropped: u64,
    external_events_dropped: u64,
    invalid_intervals: u64,
    poisoned: bool,
    counter_overflow: bool,
    unbound_scheduler_events: usize,
    duplicate_request_stages: usize,
    identity_mismatches: usize,
    accepted_timeline_errors: usize,
    counts: BTreeMap<&'static str, usize>,
    distributions: BTreeMap<String, Distribution>,
    device_kernel_timing: &'static str,
    device_transfer_timing: &'static str,
    process_vram: &'static str,
    protocol_output_scope: &'static str,
    records: Vec<SpanWire>,
}
impl ProfileDocument {
    fn capture(
        snapshot: &SourceSnapshot<CompletionContext>,
        joined: bool,
        report: Option<&NativeRunReport>,
    ) -> Self {
        let mut frames = BTreeMap::<RequestId, Frame<'_>>::new();
        let mut durations = BTreeMap::<String, Vec<Duration>>::new();
        let mut physical = BTreeSet::new();
        let mut consumed = BTreeSet::new();
        let mut input_executions = BTreeMap::<_, BTreeSet<ExecutionId>>::new();
        let mut duplicates = 0;
        let mut identity_errors = 0;
        let mut unbound = 0;
        let records = snapshot
            .records
            .iter()
            .enumerate()
            .map(|(index, span)| {
                if span.start < span.end {
                    durations
                        .entry(format!("{:?}", span.stage))
                        .or_default()
                        .push(span.end.duration_since(span.start));
                }
                if let Some(key) = span.key {
                    let frame = frames.entry(key.request.request).or_default();
                    if frame.context.is_some_and(|context| context != key.request) {
                        identity_errors += 1;
                    }
                    frame.context = Some(key.request);
                    if let Some(execution) = key.execution {
                        if frame
                            .execution
                            .is_some_and(|previous| previous != execution)
                        {
                            identity_errors += 1;
                        }
                        frame.execution = Some(execution);
                    }
                    if frame.stages.insert(span.stage, span).is_some() {
                        duplicates += 1;
                    }
                    if span.stage == SourceStage::PhysicalWorker
                        && span.succeeded
                        && let Some(execution) = key.execution
                    {
                        physical.insert(execution);
                        input_executions
                            .entry(key.request.input)
                            .or_default()
                            .insert(execution);
                    }
                    if span.stage == SourceStage::SearchBackup && span.succeeded {
                        consumed.insert(key.request.request);
                    }
                } else if span.stage != SourceStage::ProtocolWrite {
                    unbound += 1;
                }
                SpanWire {
                    arrival_index: index,
                    stage: format!("{:?}", span.stage),
                    key: span.key.map(Into::into),
                    start_ns: span
                        .start
                        .checked_duration_since(snapshot.origin)
                        .and_then(nanos),
                    end_ns: span
                        .end
                        .checked_duration_since(snapshot.origin)
                        .and_then(nanos),
                    succeeded: span.succeeded,
                }
            })
            .collect();
        let mut timeline_errors = 0;
        for request in &consumed {
            let frame = &frames[request];
            let required = [
                SourceStage::SearchPreparation,
                SourceStage::Admitted,
                SourceStage::DispatchStarted,
                SourceStage::EncodingPreparation,
                SourceStage::NativePreparation,
                SourceStage::NativeInvocation,
                SourceStage::NativeOutput,
                SourceStage::PhysicalWorker,
                SourceStage::PhysicalReadyObserved,
                SourceStage::ValidationStarted,
                SourceStage::ValidationFinished,
                SourceStage::DeliveryAccepted,
                SourceStage::SearchBackup,
            ];
            if frame.execution.is_none()
                || required
                    .iter()
                    .any(|stage| !frame.stages.get(stage).is_some_and(|span| span.succeeded))
            {
                timeline_errors += 1;
                continue;
            }
            for (name, first, last, first_start, last_start) in [
                (
                    "Queue",
                    SourceStage::Admitted,
                    SourceStage::DispatchStarted,
                    true,
                    true,
                ),
                (
                    "DispatchStartToWorker",
                    SourceStage::DispatchStarted,
                    SourceStage::PhysicalWorker,
                    true,
                    true,
                ),
                (
                    "ReadyObservationLag",
                    SourceStage::PhysicalWorker,
                    SourceStage::PhysicalReadyObserved,
                    false,
                    false,
                ),
                (
                    "RuntimeValidation",
                    SourceStage::ValidationStarted,
                    SourceStage::ValidationFinished,
                    true,
                    false,
                ),
                (
                    "DeliveryToBackup",
                    SourceStage::DeliveryAccepted,
                    SourceStage::SearchBackup,
                    false,
                    true,
                ),
                (
                    "PreparationToBackup",
                    SourceStage::SearchPreparation,
                    SourceStage::SearchBackup,
                    true,
                    false,
                ),
            ] {
                let first = frame.stages[&first];
                let last = frame.stages[&last];
                let start = if first_start { first.start } else { first.end };
                let end = if last_start { last.start } else { last.end };
                match end.checked_duration_since(start) {
                    Some(duration) => durations.entry(name.into()).or_default().push(duration),
                    None => timeline_errors += 1,
                }
            }
        }
        let journal_complete = joined
            && snapshot.dropped == 0
            && snapshot.external_events_dropped == 0
            && snapshot.invalid_intervals == 0
            && !snapshot.poisoned
            && !snapshot.counter_overflow;
        let expected_consumed = report.map(|report| {
            report.search_root_initializations.count + report.search_non_root_backups.count
        });
        let consumed_matches = expected_consumed
            .is_some_and(|expected| usize::try_from(expected).ok() == Some(consumed.len()));
        let counts = BTreeMap::from([
            ("requests", frames.len()),
            ("physical_completions", physical.len()),
            ("search_consumed", consumed.len()),
            (
                "same_input_repeat_executions",
                input_executions
                    .values()
                    .map(|executions| executions.len().saturating_sub(1))
                    .sum(),
            ),
            (
                "completed_not_consumed",
                physical.len().saturating_sub(consumed.len()),
            ),
        ]);
        Self {
            schema: "rovezero.native-d02.host-source.v1",
            source_clock: "owned process Instant; owner ticks projected through their original origin",
            journal_capacity: snapshot.capacity,
            producers_joined: joined,
            journal_complete,
            accepted_request_timeline_complete: journal_complete
                && consumed_matches
                && !consumed.is_empty()
                && timeline_errors == 0
                && duplicates == 0
                && identity_errors == 0
                && unbound == 0,
            dropped: snapshot.dropped,
            external_events_dropped: snapshot.external_events_dropped,
            invalid_intervals: snapshot.invalid_intervals,
            poisoned: snapshot.poisoned,
            counter_overflow: snapshot.counter_overflow,
            unbound_scheduler_events: unbound,
            duplicate_request_stages: duplicates,
            identity_mismatches: identity_errors,
            accepted_timeline_errors: timeline_errors,
            counts,
            distributions: durations
                .into_iter()
                .map(|(stage, samples)| (stage, Distribution::from_samples(samples)))
                .collect(),
            device_kernel_timing: "not measured; NativeInvocation includes synchronous ORT/provider work",
            device_transfer_timing: "not measured; no transfer hook",
            process_vram: "not measured by this journal",
            protocol_output_scope: "actual write/flush spans for the process stream; not per-evaluation output",
            records,
        }
    }
}
fn nanos(duration: Duration) -> Option<u64> {
    u64::try_from(duration.as_nanos()).ok()
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_output_and_error_semantics_survive_a_full_passive_journal() {
        struct Partial(Vec<u8>);
        impl Write for Partial {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                let count = bytes.len().min(2);
                self.0.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }
        }
        let trace = SourceJournal::try_new(1).unwrap();
        let mut output = ProfiledWrite::new(Partial(Vec::new()), Some(trace.clone()));
        output.write_all(b"bestmove e2e4\n").unwrap();
        assert_eq!(output.inner.0, b"bestmove e2e4\n");
        assert_eq!(
            output.flush().unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
        assert!(trace.snapshot().dropped > 0);
        assert!(
            !ProfileDocument::capture(&trace.snapshot(), true, None)
                .accepted_request_timeline_complete
        );
    }
    #[test]
    fn upstream_loss_and_unconfirmed_shutdown_cannot_be_complete() {
        let trace = SourceJournal::<CompletionContext>::try_new(2).unwrap();
        trace.note_external_loss(3, false);
        let report = ProfileDocument::capture(&trace.snapshot(), true, None);
        assert!(!report.journal_complete);
        assert_eq!(report.external_events_dropped, 3);
        let clean = SourceJournal::<CompletionContext>::try_new(2).unwrap();
        assert!(!ProfileDocument::capture(&clean.snapshot(), false, None).journal_complete);
    }
}
