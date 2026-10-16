//! A bounded cold scan issues one exact, move-only hot append permission.
//!
//! The archive worker owns the store while scanning. It never publishes an
//! observation. The semantic owner later consumes the permission without I/O.
use super::*;
use crate::external_cpu::ExternalGoIssue;
use crate::external_cpu::worker::{ExternalWorkerReport, ExternalWorkerScope};
use rz_contracts::pals_adaptive::ExternalCpuJobEnvelope;
use std::sync::atomic::{AtomicU64, Ordering};

/// An actual worker report's issued-go attestation, after the search owner has
/// validated admission, its original question and its monotonic deadline.
/// No public constructor or Clone implementation permits an external caller to
/// turn NotIssued/Unknown into a physical request or replace its raw report.
#[derive(Debug)]
pub struct ForeignGoIssuedProof {
    scope: ExternalWorkerScope,
    job: ExternalCpuJobEnvelope,
    request_id: u64,
    observation_digest: [u8; 32],
    retain_until: Instant,
}

impl ForeignGoIssuedProof {
    pub fn retain_until(&self) -> Instant {
        self.retain_until
    }
    /// The caller establishes the envelope's actual admission and converts its
    /// original hard deadline to this process's Instant domain. This method
    /// checks the retained worker payload and never claims fresh admission.
    pub(crate) fn checked_worker_report(
        worker: &ExternalWorkerReport,
        observation: &Observation,
        retain_until: Instant,
    ) -> Result<Self, StoreError> {
        let raw = observation
            .external_report
            .as_deref()
            .ok_or(StoreError::InvalidEvidence(
                "foreign issued proof lacks raw report",
            ))?;
        if worker.go_issue != ExternalGoIssue::Issued
            || worker.scope.job != worker.job.id
            || worker.grant.job != worker.job.id
            || worker.scope.execution != worker.grant.execution
            || worker.scope.owner != worker.grant.owner
            || worker.scope.execution.epoch != worker.job.id.authority.epoch
            || worker.scope.execution.sequence == 0
            || !worker.controls.terminal
            || worker.exact_resume_supported
        {
            return Err(StoreError::InvalidEvidence(
                "foreign go was not exactly issued",
            ));
        }
        let matches_raw = match &worker.result {
            Ok(completed) => {
                completed.go_issue == ExternalGoIssue::Issued
                    && completed.strict_native_nodes.is_none()
                    && &completed.report == raw
            }
            Err(_) => worker.attempt.as_ref().is_some_and(|attempt| {
                attempt.external.as_ref().is_some_and(|external| {
                    external.request_id == raw.request_id
                        && external.partial_report.as_ref() == Some(raw)
                })
            }),
        };
        if !matches_raw {
            return Err(StoreError::InvalidEvidence(
                "foreign issued proof has another raw report",
            ));
        }
        if Instant::now() >= retain_until {
            return Err(StoreError::ArchiveDeadline);
        }
        observation.validate()?;
        Ok(Self {
            scope: worker.scope,
            job: worker.job,
            request_id: raw.request_id,
            observation_digest: observation_digest(observation)?,
            retain_until,
        })
    }
}

/// Exact observation and live fact-owner seal. This is not a general permission
/// to append another report, reuse a scan or bypass the original deadline.
/// By-value consumption burns it even if the one hot allocation fails.
#[derive(Debug)]
pub struct ForeignAppendPermit {
    observation: Observation,
    issued: ForeignGoIssuedProof,
    owner: StoreReadOwner,
    observations_owner: Weak<()>,
    store_generation: u64,
    observation_high_water: usize,
    archive_owner: Option<Weak<()>>,
    archive_high_water: Option<u64>,
    nonce: u64,
    deadline: Instant,
}

impl ForeignAppendPermit {
    pub(crate) fn observation(&self) -> &Observation {
        &self.observation
    }
    pub fn observation_digest(&self) -> [u8; 32] {
        self.issued.observation_digest
    }
    pub fn request_id(&self) -> u64 {
        self.issued.request_id
    }
    pub fn native_execution(&self) -> rz_contracts::ExecutionId {
        self.issued.scope.execution
    }
    pub fn native_job(&self) -> rz_contracts::pals_adaptive::CpuJobId {
        self.issued.job.id
    }
    pub fn nonce(&self) -> u64 {
        self.nonce
    }
}

impl PalsStores {
    /// Worker-only preparation: validate hot handles and scan every committed
    /// cold generation under the original finite I/O/deadline allowance. The
    /// observation remains unpublished inside the returned move-only permit.
    pub fn prepare_foreign_append_permit(
        &mut self,
        observation: Observation,
        issued: ForeignGoIssuedProof,
        budget: &mut ArchiveIoBudget,
    ) -> Result<ForeignAppendPermit, StoreError> {
        self.reject_archive_allocation_scope()?;
        budget.check()?;
        if budget.deadline > issued.retain_until || Instant::now() >= issued.retain_until {
            return Err(StoreError::ArchiveDeadline);
        }
        let raw = observation
            .external_report
            .as_deref()
            .ok_or(StoreError::InvalidEvidence(
                "foreign append permit lacks raw report",
            ))?;
        if raw.request_id != issued.request_id
            || observation_digest(&observation)? != issued.observation_digest
        {
            return Err(StoreError::InvalidEvidence(
                "foreign append permit payload differs",
            ));
        }
        self.validate_observation_handles(&observation)?;
        let owner = self.read_owner();
        let observations_owner = Arc::downgrade(&self.observations.identity);
        let store_generation = self.generation;
        let observation_high_water = self.observations.next_observation_sequence();
        let (archive_owner, archive_high_water) = self.foreign_archive_seal();
        if self.archive_enabled() {
            self.with_archive_read_budget(budget, |bounded| {
                self.check_foreign_cold_request(&observation, bounded)
            })?;
        }
        budget.check()?;
        // Even an exclusive worker may have failed/canceled an archive
        // transaction. Never issue proof over a changed generation namespace.
        if !owner.matches(self)
            || !observations_owner.ptr_eq(&Arc::downgrade(&self.observations.identity))
            || self.generation != store_generation
            || self.observations.next_observation_sequence() != observation_high_water
            || !archive_seal_matches(self, &archive_owner, archive_high_water)
        {
            return Err(StoreError::ForeignPermitStale);
        }
        static NEXT_NONCE: AtomicU64 = AtomicU64::new(1);
        let nonce = NEXT_NONCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |previous| {
                previous.checked_add(1)
            })
            .map_err(|_| StoreError::RevisionExhausted)?;
        Ok(ForeignAppendPermit {
            observation,
            issued,
            owner,
            observations_owner,
            store_generation,
            observation_high_water,
            archive_owner,
            archive_high_water,
            nonce,
            deadline: budget.deadline,
        })
    }

    /// Semantic-owner commit: one hot append, no archive scan, reclaim, retry,
    /// deadline renewal or task consumption. Capacity/stale/error consumes the
    /// permit; the caller needs a new bounded worker scan for another attempt.
    pub fn hot_append_foreign_permitted(
        &mut self,
        permit: ForeignAppendPermit,
    ) -> Result<ObservationId, StoreError> {
        self.reject_archive_allocation_scope()?;
        if !permit.owner.matches(self)
            || !permit
                .observations_owner
                .ptr_eq(&Arc::downgrade(&self.observations.identity))
            || permit.store_generation != self.generation
            || permit.observation_high_water != self.observations.next_observation_sequence()
            || !archive_seal_matches(self, &permit.archive_owner, permit.archive_high_water)
        {
            return Err(StoreError::ForeignPermitStale);
        }
        if Instant::now() >= permit.deadline || Instant::now() >= permit.issued.retain_until {
            return Err(StoreError::ArchiveDeadline);
        }
        // Exact bytes cannot change: the opaque permit owns the Observation.
        // Revalidate current hot task/line facts without loading cold records.
        self.append_observation_once(permit.observation)
    }

    fn foreign_archive_seal(&self) -> (Option<Weak<()>>, Option<u64>) {
        self.archive.as_ref().map_or((None, None), |archive| {
            (
                Some(Arc::downgrade(&archive.identity)),
                Some(archive.next_generation),
            )
        })
    }
}

fn archive_seal_matches(
    stores: &PalsStores,
    expected_owner: &Option<Weak<()>>,
    expected_high_water: Option<u64>,
) -> bool {
    match (&stores.archive, expected_owner, expected_high_water) {
        (None, None, None) => true,
        (Some(actual), Some(owner), Some(high_water)) => {
            owner.ptr_eq(&Arc::downgrade(&actual.identity)) && actual.next_generation == high_water
        }
        _ => false,
    }
}

/// Canonical existing archive encoding, hashed by a streaming writer rather
/// than a second serialized raw-report buffer. This only runs on preparation.
fn observation_digest(observation: &Observation) -> Result<[u8; 32], StoreError> {
    struct HashWriter(Sha256);
    impl Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let wire = ObservationWire::encode(observation)?;
    let mut writer = HashWriter(Sha256::new());
    writer.0.update(b"rz-pals-foreign-exact-append-permit/1\0");
    serde_json::to_writer(&mut writer, &wire)
        .map_err(|_| StoreError::InvalidEvidence("foreign permit observation encoding"))?;
    Ok(writer.0.finalize().into())
}
