//! Distinct batch-experiment wire. V1 files/producer and validation stay B1.
use crate::{
    engine::EngineSettings,
    native_attestation::{AttestationError, ReceiptWriter, bounded_json, hex},
    native_bootstrap::{NativeConfig, NativeCudaFactory, NativeRunError, NativeRunReport},
    native_cuda_attestation::{
        CudaSearchReceiptV1, CudaStartupReceiptV1, CudaTerminationReceiptV1,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub const STARTUP_FILE: &str = "native-cuda-batch-startup.v1.json";
pub const TERMINATION_FILE: &str = "native-cuda-batch-termination.v1.json";
pub const JOURNAL_FILE: &str = "native-cuda-batch-journal.v1.json";
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchStartupReceipt {
    pub schema_version: u32,
    pub kind: String,
    pub provider: CudaStartupReceiptV1,
    pub search: CudaSearchReceiptV1,
    pub max_pending: usize,
    pub max_batch_wait_us: u64,
    pub physical_workers: usize,
    pub max_physical_executions: usize,
    pub classification: String,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchTerminationReceipt {
    pub schema_version: u32,
    pub kind: String,
    pub provider: CudaTerminationReceiptV1,
    pub startup_sha256: String,
    pub journal_sha256: String,
    pub journal_bytes: u64,
    pub journal_ownership_retained: bool,
    pub journal_summary: Option<rz_eval::batch_journal::BatchJournalAudit>,
    pub journal_error: Option<String>,
}
pub struct BatchReceiptWriter {
    writer: ReceiptWriter,
    startup_hash: String,
}
impl BatchReceiptWriter {
    pub fn open(config: &NativeConfig) -> Result<Self, AttestationError> {
        if !config.batch_attestation_requested() {
            return Err(AttestationError::boundary(
                "batch writer requires explicit batch attestation",
            ));
        }
        Ok(Self {
            writer: ReceiptWriter::open_named(config, STARTUP_FILE, TERMINATION_FILE)?,
            startup_hash: String::new(),
        })
    }
    pub fn startup(
        &mut self,
        provider: CudaStartupReceiptV1,
        settings: &EngineSettings,
    ) -> Result<BatchStartupReceipt, AttestationError> {
        let width = provider.profile.max_batch_items;
        let mut search = CudaSearchReceiptV1::capture(&provider, settings)?;
        search.batch_size = width
            .try_into()
            .map_err(|_| AttestationError::boundary("batch width overflow"))?;
        let startup = BatchStartupReceipt {
            schema_version: 1,
            kind: "batch_experiment_startup".into(),
            provider,
            search,
            max_pending: width,
            max_batch_wait_us: if width > 1 { 200 } else { 0 },
            physical_workers: 1,
            max_physical_executions: 1,
            classification: "S".into(),
        };
        self.startup_hash = hex(&Sha256::digest(bounded_json(&startup)?).into());
        self.writer.publish_startup(&startup)?;
        Ok(startup)
    }
    pub fn termination(
        &mut self,
        startup: &BatchStartupReceipt,
        factory: &NativeCudaFactory,
        result: &Result<NativeRunReport, NativeRunError>,
    ) -> Result<(), AttestationError> {
        let provider = CudaTerminationReceiptV1::from_result(&startup.provider, result)?;
        let journal = factory
            .batch_journal(result.is_ok())
            .ok_or_else(|| AttestationError::boundary("batch journal missing"))?;
        let (journal_sha256, journal_bytes) = self.writer.publish_large_auxiliary(
            JOURNAL_FILE,
            &journal,
            rz_eval::batch_journal::MAX_BATCH_RECEIPT_BYTES,
        )?;
        let (journal_summary, journal_error) =
            match rz_eval::batch_journal::audit_batch_journal(&journal, result.is_ok()) {
                Ok(summary) => (Some(summary), None),
                Err(error) => (None, Some(error.to_string())),
            };
        let invalid = journal_error.is_some();
        self.writer.publish_termination(&BatchTerminationReceipt {
            schema_version: 1,
            kind: "batch_experiment_termination".into(),
            provider,
            startup_sha256: self.startup_hash.clone(),
            journal_sha256,
            journal_bytes,
            journal_ownership_retained: result.is_err(),
            journal_summary,
            journal_error,
        })?;
        if invalid && result.is_ok() {
            return Err(AttestationError::boundary(
                "batch journal correctness failed; raw journal retained",
            ));
        }
        Ok(())
    }
}
