//! Actual frozen P/C collection. The native adapter owns encoding, inference,
//! physical fences and delivery; this observer only seals bounded evidence.
use super::*;
use rz_contracts::RequestId;
use rz_eval::error::BackendError;
use rz_eval::pals_model::PalsRawOutput;
use rz_eval::pals_onnx::PalsOnnxConfig;
use rz_eval::runtime_pin::RuntimeLibraryPin;
use rz_search::pals::engine::{PALS_SEARCH_VERSION, RoleError};
use rz_uci::pals_native::{
    NativePreparedContext, NativeQueryKind, NativeRoleFinishHandle, NativeRoleModel,
    NativeRoleObserver, NativeRoleRejection,
};
use std::sync::{Arc, Mutex};

const REGISTRY_VERSION: &str = "rz-pals-native-collection-registry/1";
const RAW_RESERVE: u64 = 128 * 1024;
const STAGE_RESERVE: u64 = 32 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsCollectionGraphPin {
    pub role: String,
    pub sha256: String,
    pub serialized_bytes: u64,
}

/// Supplied independently of the loaded model. No collector output can enroll
/// its own hashes. Checkpoint bytes are separately verified: an export epoch
/// declaration alone is not evidence that the actual checkpoint was checked.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PalsNativeCollectionRegistry {
    pub version: String,
    pub collector_binary_sha256: String,
    pub model_configuration: PalsModelConfig,
    pub model_configuration_sha256: String,
    pub checkpoint_sha256: String,
    pub export_manifest_sha256: String,
    pub graphs: Vec<PalsCollectionGraphPin>,
    pub runtime_sha256: String,
    pub encoding_sha256: String,
    pub encoder_source_sha256: String,
    pub cpu_configuration_sha256: String,
    pub frozen_epoch: u64,
    pub training_state: String,
    pub provider: String,
    pub intra_threads: usize,
    pub cache_public_memory: bool,
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl PalsNativeCollectionRegistry {
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.model_configuration
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        if self.version != REGISTRY_VERSION
            || self.provider != "cpu"
            || self.training_state != "untrained"
            || self.frozen_epoch == 0
            || !(1..=2).contains(&self.intra_threads)
            || self.graphs.is_empty()
            || self.graphs.len() > 4
            || self.graphs.iter().any(|g| {
                !ident(&g.role)
                    || !valid_sha(&g.sha256)
                    || g.serialized_bytes == 0
                    || g.serialized_bytes > 256 * 1024 * 1024
            })
            || self
                .graphs
                .iter()
                .map(|g| &g.role)
                .collect::<BTreeSet<_>>()
                .len()
                != self.graphs.len()
            || [
                &self.collector_binary_sha256,
                &self.model_configuration_sha256,
                &self.checkpoint_sha256,
                &self.export_manifest_sha256,
                &self.runtime_sha256,
                &self.encoding_sha256,
                &self.encoder_source_sha256,
                &self.cpu_configuration_sha256,
            ]
            .iter()
            .any(|s| !valid_sha(s))
            || canonical_sha256(&self.model_configuration)? != self.model_configuration_sha256
        {
            return Err(invalid("invalid independent CPU PALS collection registry"));
        }
        Ok(())
    }
    pub fn read_pinned(path: &Path, expected_sha256: &str) -> Result<Self, ArenaError> {
        if !path.is_absolute()
            || !valid_sha(expected_sha256)
            || path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == ".env" || n.starts_with(".env."))
        {
            return Err(invalid(
                "native collection registry requires an absolute public data file and hash",
            ));
        }
        if file_sha(path, 256 * 1024)? != expected_sha256 {
            return Err(invalid(
                "native collection registry bytes differ from independent pin",
            ));
        }
        let file = File::open(path).map_err(io)?;
        let mut bytes = Vec::new();
        file.take(256 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        if bytes.len() > 256 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != expected_sha256 {
            return Err(invalid(
                "native collection registry changed during admission",
            ));
        }
        let value: Self = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        value.validate()?;
        Ok(value)
    }
    pub fn owned_sources(&self) -> Result<PalsOwnedSources, ArenaError> {
        self.validate()?;
        let sources = PalsOwnedSources {
            cpu_binary_sha256: BTreeSet::from([self.collector_binary_sha256.clone()]),
            input_sources: BTreeSet::from([PalsInputSource::OwnPals {
                model_configuration_sha256: self.model_configuration_sha256.clone(),
                model_weights_sha256: self.checkpoint_sha256.clone(),
            }]),
        };
        sources.validate()?;
        Ok(sources)
    }
}

struct Call {
    input_sha256: String,
    physical: bool,
    delivered: bool,
    accepted: bool,
    rejected: bool,
    physical_unknown: bool,
}
struct Sink {
    context: Option<PalsNativeCaptureContext>,
    source: PalsCollectionSourceDescription,
    trace: PalsNativeTrace,
    started: Instant,
    calls: BTreeMap<RequestId, Call>,
    raw_sources: BTreeSet<String>,
    total_rows: usize,
}
fn role_error(error: impl std::fmt::Display) -> RoleError {
    RoleError::Backend(failure_text(error))
}
impl Sink {
    fn context(&self) -> Result<&PalsNativeCaptureContext, RoleError> {
        self.context
            .as_ref()
            .ok_or_else(|| role_error("native capture context not admitted"))
    }
    fn row<T: Serialize>(&mut self, artifact: &'static str, value: &T) -> Result<(), RoleError> {
        let json = bounded_json(value, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?;
        if self.trace.rows.len() >= MAX_ROWS * 8 {
            return Err(role_error("native trace row limit"));
        }
        self.trace.rows.push(PalsNativeTraceRow { artifact, json });
        Ok(())
    }
    fn event(
        &mut self,
        id: RequestId,
        stage: &str,
        detail: serde_json::Value,
    ) -> Result<(), RoleError> {
        let input = self
            .calls
            .get(&id)
            .ok_or_else(|| role_error("native event has no sealed request"))?;
        self.row("native-events.jsonl", &serde_json::json!({
            "domain":"rz-pals-native-call-event/1", "game_id":self.context()?.game,
            "process_epoch":id.epoch.0,"request_sequence":id.sequence,
            "input_sha256":input.input_sha256, "stage":stage,
            "observer_elapsed_us":self.started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            "detail":detail
        }))
    }
    fn reserve(&mut self, bytes: u64) -> Result<(), RoleError> {
        let next = self
            .trace
            .reserved_bytes
            .checked_add(bytes)
            .filter(|n| *n <= self.context.as_ref().map_or(0, |c| c.max_bytes))
            .ok_or_else(|| {
                role_error("native collection output credit exhausted before dispatch")
            })?;
        self.trace.reserved_bytes = next;
        Ok(())
    }
    fn capture(
        &mut self,
        id: RequestId,
        prepared: &PalsModelInput,
        context: NativePreparedContext<'_>,
    ) -> Result<(), RoleError> {
        if self.calls.contains_key(&id) || self.total_rows >= MAX_ROWS {
            return Err(role_error("duplicate native request or input row limit"));
        }
        prepared
            .validate(&PalsModelConfig::baseline())
            .map_err(role_error)?;
        if prepared.model_epoch != self.source.model_epoch {
            return Err(role_error(
                "actual prepared model epoch differs from independent registry",
            ));
        }
        let (position, prefix, proposal, counterexample, records, revision, kind, divergences) =
            match context {
                NativePreparedContext::Role { query, kind } => (
                    query.position,
                    query.prefix,
                    query.proposal,
                    query.counterexample,
                    query.records,
                    query.revision,
                    kind,
                    Vec::new(),
                ),
                NativePreparedContext::Divergence { query } => (
                    query.root,
                    &[][..],
                    query.proposal,
                    None,
                    query.records,
                    query.revision,
                    NativeQueryKind::Divergence,
                    query.candidates.to_vec(),
                ),
            };
        let capture = self.context()?;
        let (replayed, _) = replay_line(&capture.position, prefix).map_err(role_error)?;
        if !replayed.snapshot().same_state(&position.snapshot())
            || prepared.situation_revision != revision
        {
            return Err(role_error(
                "native query prefix does not replay to its actual Rules state/revision",
            ));
        }
        let eligible = kind == NativeQueryKind::Propose
            && prefix.is_empty()
            && capture.position.snapshot().same_state(&position.snapshot());
        let game = capture.game.clone();
        let opening = capture.opening.clone();
        let actual_moves = capture.actual_moves.clone();
        let mut complete_line = actual_moves.clone();
        complete_line.extend_from_slice(prefix);
        let sequence = self
            .trace
            .sequence
            .checked_add(1)
            .ok_or_else(|| role_error("native capture sequence exhausted"))?;
        let role = if matches!(kind, NativeQueryKind::Reply | NativeQueryKind::Divergence) {
            PalsDataRole::Critic
        } else {
            PalsDataRole::Proposer
        };
        let mut raw_rows = Vec::new();
        let mut selected = Vec::new();
        for token in &prepared.records {
            let index = usize::try_from(
                token
                    .record_id
                    .checked_sub(1)
                    .ok_or_else(|| role_error("native record ID zero"))?,
            )
            .map_err(role_error)?;
            let record = records
                .get(index)
                .ok_or_else(|| role_error("native public record is not in actual query"))?;
            if record.revision != token.revision || record.revision > revision {
                return Err(role_error("native public raw-source revision mismatch"));
            }
            let raw = serde_json::json!({"domain":"rz-pals-native-public-source/1","game_id":game,
                "record_index":token.record_id,"revision":record.revision,
                "origin_state_id":record.origin_state.0,"origin_state_id_is_advisory":true,
                "origin_rules_state_sha256":null,"origin_rules_identity_observation":"unknown",
                "kind":format!("{:?}",record.kind),"line":pack(&record.line).map_err(role_error)?,
                "value":record.value,"completed_depth":record.completed_depth,
                "scope":format!("{:?}",record.score_scope),"white_score_perspective":record.perspective==Color::White,
                "critical":token.critical,"source_cpu_profile_sha256":self.source.cpu_profile_sha256});
            let json = bounded_json(&raw, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?;
            let digest = format!("{:x}", Sha256::digest(&json));
            selected.push(PalsPublicRecord {
                observation_sha256: digest.clone(),
                situation_revision: record.revision,
            });
            if !self.raw_sources.contains(&digest) {
                raw_rows.push((
                    digest,
                    PalsNativeTraceRow {
                        artifact: "public-record-sources.jsonl",
                        json,
                    },
                ));
            }
        }
        let input = PalsInputSnapshot {
            game_id: game.clone(),
            opening_id: opening.id.clone(),
            line_genealogy_id: format!("{game}-actual-and-conditional"),
            position_command: position_command(&opening, &complete_line),
            board_fen: position.to_fen(),
            actual_history: pack(&actual_moves).map_err(role_error)?,
            rules_state_sha256: state_sha(position).map_err(role_error)?,
            rules_history_sha256: hex(prepared.history_digest),
            transposition_sha256: transposition_sha(position),
            encoding_sha256: self.source.encoding_sha256.clone(),
            source: self.source.source.clone(),
            frozen_epoch: self.source.frozen_epoch,
            input_revision: revision,
            capture_sequence: sequence,
            white_to_move: position.side_to_move() == Color::White,
            role,
            legal_moves: pack(position.ordered_legal_moves().moves()).map_err(role_error)?,
            public_records: selected.clone(),
        }
        .seal()
        .map_err(role_error)?;
        if kind != NativeQueryKind::Divergence {
            let candidates = prepared
                .candidates
                .iter()
                .map(|c| c.packed())
                .collect::<Result<Vec<_>, _>>()
                .map_err(role_error)?;
            if candidates != input.snapshot().legal_moves {
                return Err(role_error(
                    "native prepared candidates differ from Rules legal order",
                ));
            }
        }
        let tensor_json = serde_json::to_string(prepared).map_err(role_error)?;
        let mut sidecar = PalsNativeInputSidecar {
            version: "rz-pals-native-input-sidecar/1".into(),
            input_sha256: input.sha256().into(),
            encoding_sha256: self.source.encoding_sha256.clone(),
            encoder_source_sha256: self.source.encoder_source_sha256.clone(),
            model_epoch_kind: "frozen_model_epoch".into(),
            canonical_tensor_sha256: hex(prepared
                .canonical_input_key(&PalsModelConfig::baseline())
                .map_err(role_error)?),
            tensor_sha256: format!("{:x}", Sha256::digest(tensor_json.as_bytes())),
            tensor_json,
            record_sources: selected,
            sha256: String::new(),
        };
        sidecar.sha256 = canonical_sha256(&(
            sidecar.version.as_str(),
            sidecar.input_sha256.as_str(),
            sidecar.encoding_sha256.as_str(),
            sidecar.encoder_source_sha256.as_str(),
            sidecar.model_epoch_kind.as_str(),
            sidecar.canonical_tensor_sha256.as_str(),
            sidecar.tensor_json.as_str(),
            sidecar.tensor_sha256.as_str(),
            &sidecar.record_sources,
        ))
        .map_err(role_error)?;
        let divergence = kind == NativeQueryKind::Divergence;
        let lineage = serde_json::json!({"input_sha256":input.sha256(),"game_id":game,
            "process_epoch":id.epoch.0,"request_sequence":id.sequence,"native_query_kind":format!("{:?}",kind),
            "actual_played_history":pack(&actual_moves).map_err(role_error)?,"virtual_prefix":pack(prefix).map_err(role_error)?,
            "proposal":pack(proposal).map_err(role_error)?,"counterexample":counterexample.map(pack).transpose().map_err(role_error)?,
            "divergence_plies":divergences,"actual_outcome_eligible":eligible,
            "counterfactual_wdl":"masked","training_admission":if divergence{"deferred_divergence_head"}else{"ordinary_role"}});
        let rows = [
            PalsNativeTraceRow {
                artifact: if divergence {
                    "native-divergence-inputs.jsonl"
                } else {
                    "inputs.jsonl"
                },
                json: bounded_json(&input, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?,
            },
            PalsNativeTraceRow {
                artifact: if divergence {
                    "native-divergence-sidecars.jsonl"
                } else {
                    "native-inputs.jsonl"
                },
                json: bounded_json(&sidecar, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?,
            },
            PalsNativeTraceRow {
                artifact: "input-lineage.jsonl",
                json: bounded_json(&lineage, MAX_JSON_RECORD_BYTES, false).map_err(role_error)?,
            },
        ];
        let exact_bytes = rows
            .iter()
            .chain(raw_rows.iter().map(|(_, r)| r))
            .try_fold(0_u64, |n, r| {
                n.checked_add(r.json.len() as u64 + 1)
                    .ok_or_else(|| role_error("prepared native bytes overflow"))
            })?;
        self.reserve(
            exact_bytes
                .checked_add(RAW_RESERVE + STAGE_RESERVE)
                .ok_or_else(|| role_error("native dispatch reservation overflow"))?,
        )?;
        // The immutable snapshot and actual tensor are sealed before returning
        // to native submit. No later observation is admitted into these bytes.
        self.trace.sequence = sequence;
        self.total_rows += 1;
        for (digest, row) in raw_rows {
            self.raw_sources.insert(digest);
            self.trace.rows.push(row);
        }
        self.trace.rows.extend(rows);
        if !divergence {
            self.trace.inputs.push((input.clone(), eligible));
        }
        self.calls.insert(
            id,
            Call {
                input_sha256: input.sha256().into(),
                physical: false,
                delivered: false,
                accepted: false,
                rejected: false,
                physical_unknown: false,
            },
        );
        self.event(id,"prepared",serde_json::json!({"prepared_before_submit":true,"native_query_kind":format!("{:?}",kind)}))
    }
}
struct Observer(Arc<Mutex<Sink>>);

/// Factory/CLI failures can precede the main output receipt. Keep the native
/// owner behind the same finite cleanup and emit a small lifecycle receipt on
/// that failure stream; an explicit collection receipt marks it reported.
struct NativeFinishGuard {
    handle: NativeRoleFinishHandle,
    attempted: bool,
    reported: bool,
    error: Option<String>,
}
fn lifecycle_receipt(
    receipt: &rz_uci::pals_native::NativeRoleReceipt,
) -> Result<serde_json::Value, ArenaError> {
    let mut value = serde_json::to_value(receipt).map_err(|e| invalid(e.to_string()))?;
    if let Some(residency) = value.get_mut("residency").and_then(|v| v.as_object_mut()) {
        let bank = residency
            .remove("reader_initializer_bank")
            .unwrap_or(serde_json::Value::Null);
        let summary = if bank.is_null() {
            serde_json::Value::Null
        } else {
            let bytes = bounded_json(&bank, 256 * 1024, false)?;
            serde_json::json!({"canonical_json_sha256":canonical_sha256(&bank)?,"serialized_json_bytes":bytes.len(),
                "shared_parameters":bank.get("shared_parameters").and_then(|v|v.as_array()).map(Vec::len),
                "observation_scope":"export-manifest-serialized-ledger; native residency unknown"})
        };
        residency.insert("reader_initializer_bank_summary".into(), summary);
    }
    // The large serialized ledger is already bound by the export manifest SHA;
    // receipt reserve holds lifecycle facts and a small ledger identity only.
    let _ = bounded_json(&value, 48 * 1024, false)?;
    Ok(value)
}
impl NativeFinishGuard {
    fn new(handle: NativeRoleFinishHandle) -> Self {
        Self {
            handle,
            attempted: false,
            reported: false,
            error: None,
        }
    }
    fn receipt(&self) -> rz_uci::pals_native::NativeRoleReceipt {
        self.handle.receipt()
    }
    fn finish(&mut self, until: Instant) -> Result<(), RoleError> {
        self.attempted = true;
        let result = self.handle.finish(until);
        self.error = result.as_ref().err().map(failure_text);
        result.map(|_| ())
    }
}
impl Drop for NativeFinishGuard {
    fn drop(&mut self) {
        if self.reported {
            return;
        }
        if !self.attempted {
            if let Some(until) = Instant::now().checked_add(Duration::from_secs(30)) {
                let _ = self.finish(until);
            } else {
                self.error = Some("native cleanup deadline overflow".into());
            }
        }
        let native = self.receipt();
        let receipt = serde_json::json!({"domain":"rz-pals-collection-pre-receipt-shutdown/1",
            "receipt":lifecycle_receipt(&native).unwrap_or_else(|e|serde_json::json!({
                "physical_shutdown_confirmed":native.physical_shutdown_confirmed,"native_buffers_released":native.native_buffers_released,
                "quarantined":native.quarantined,"physical_runs_in_flight":native.physical_runs_in_flight,
                "compact_receipt_failure":failure_text(e)})),"finish_error":self.error,"actual_training_executed":false});
        match bounded_json(&receipt, 64 * 1024, false) {
            Ok(bytes) => eprintln!("{}", String::from_utf8_lossy(&bytes)),
            Err(_) => {
                eprintln!("pals_collect: native pre-receipt shutdown evidence serialization failed")
            }
        }
    }
}
impl Observer {
    fn with(
        &self,
        action: impl FnOnce(&mut Sink) -> Result<(), RoleError>,
    ) -> Result<(), RoleError> {
        let mut sink = self
            .0
            .lock()
            .map_err(|_| role_error("native collector sink poisoned"))?;
        let result = action(&mut sink);
        if let Err(error) = &result {
            sink.trace.failure = Some(failure_text(error));
        }
        result
    }
}
fn raw_bits(raw: &PalsRawOutput) -> serde_json::Value {
    serde_json::json!({"representation":"f32_ieee754_bits",
        "candidate_logits_bits":raw.candidate_logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "wdl_logits_bits":raw.wdl_logits.map(f32::to_bits),
        "divergence_logits_bits":raw.divergence_logits.as_ref().map(|v|v.iter().map(|x|x.to_bits()).collect::<Vec<_>>()),
        "task_logits_bits":raw.task_logits.map(|v|v.map(f32::to_bits)),
        "private_latent_bits":raw.private_latent.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        "prediction_is_future_label":false})
}
impl NativeRoleObserver for Observer {
    fn prepared(
        &mut self,
        id: RequestId,
        input: &PalsModelInput,
        context: NativePreparedContext<'_>,
    ) -> Result<(), RoleError> {
        self.with(|s| s.capture(id, input, context))
    }
    fn physically_completed(
        &mut self,
        id: RequestId,
        result: Result<&PalsRawOutput, &BackendError>,
    ) -> Result<(), RoleError> {
        self.with(|s| {
            let c=s.calls.get_mut(&id).ok_or_else(||role_error("physical result lacks prepared call"))?;
            if c.physical || c.physical_unknown {return Err(role_error("duplicate or unknown physical completion"));}
            c.physical=true;
            let input_sha256=c.input_sha256.clone();
            let raw=match result {Ok(raw)=>raw_bits(raw),Err(error)=>serde_json::json!({"failure":failure_text(error),"kind":format!("{:?}",error)})};
            let json=bounded_json(&serde_json::json!({"domain":"rz-pals-native-physical-raw/1",
                "process_epoch":id.epoch.0,"request_sequence":id.sequence,"input_sha256":input_sha256,
                "physical_completion_confirmed":true,"success":result.is_ok(),"raw":raw}),RAW_RESERVE as usize-1,false).map_err(role_error)?;
            s.trace.rows.push(PalsNativeTraceRow {artifact:"native-raw-outputs.jsonl",json});
            s.event(id,"physically_completed",serde_json::json!({"success":result.is_ok(),"logical_acceptance_inferred":false}))
        })
    }
    fn delivered(&mut self, id: RequestId) -> Result<(), RoleError> {
        self.with(|s| {
            let c = s
                .calls
                .get_mut(&id)
                .ok_or_else(|| role_error("delivery lacks prepared call"))?;
            if !c.physical || c.delivered || c.rejected || c.physical_unknown {
                return Err(role_error(
                    "delivery without known un-rejected physical output",
                ));
            }
            c.delivered = true;
            s.event(
                id,
                "delivered",
                serde_json::json!({"search_consumed":false}),
            )
        })
    }
    fn rejected(&mut self, id: RequestId, reason: NativeRoleRejection) -> Result<(), RoleError> {
        self.with(|s| {
            let c = s
                .calls
                .get_mut(&id)
                .ok_or_else(|| role_error("rejection lacks prepared call"))?;
            if c.accepted || c.rejected {
                return Err(role_error(
                    "duplicate rejection or rejected consumed output",
                ));
            }
            c.rejected = true;
            if c.physical && reason == NativeRoleRejection::PhysicalCompletionUnknown {
                return Err(role_error("known physical completion cannot become unknown due to observer/logical rejection"));
            }
            c.physical_unknown = reason == NativeRoleRejection::PhysicalCompletionUnknown;
            s.event(
                id,
                "logically_rejected",
                serde_json::json!({"reason":format!("{:?}",reason),"search_consumed":false}),
            )
        })
    }
    fn accepted(&mut self, id: RequestId) -> Result<(), RoleError> {
        self.with(|s| {
            let c = s
                .calls
                .get_mut(&id)
                .ok_or_else(|| role_error("consumption lacks prepared call"))?;
            if !c.physical || !c.delivered || c.accepted || c.rejected || c.physical_unknown {
                return Err(role_error("invalid/duplicate native search consumption"));
            }
            c.accepted = true;
            s.event(
                id,
                "search_consumed",
                serde_json::json!({"search_consumed":true}),
            )
        })
    }
}

pub struct OwnPalsOnnxCollectionDriver {
    engine: PalsEngine<NativeRoleModel>,
    finish: NativeFinishGuard,
    sink: Arc<Mutex<Sink>>,
    description: PalsCollectionSourceDescription,
    max_rounds: u64,
}
impl OwnPalsOnnxCollectionDriver {
    pub fn load_cpu(
        export: &Path,
        checkpoint: &Path,
        pin: &RuntimeLibraryPin,
        registry: &PalsNativeCollectionRegistry,
        cpu: CpuConfig,
        pals: PalsConfig,
        max_rounds: u64,
    ) -> Result<Self, ArenaError> {
        registry.validate()?;
        pals.validate().map_err(|e| invalid(e.to_string()))?;
        if !(1..=16).contains(&max_rounds) {
            return Err(invalid("native collection max rounds must be 1..=16"));
        }
        let actual_binary = file_sha(&std::env::current_exe().map_err(io)?, 256 * 1024 * 1024)?;
        if !checkpoint.is_absolute()
            || file_sha(checkpoint, 256 * 1024 * 1024)? != registry.checkpoint_sha256
            || actual_binary != registry.collector_binary_sha256
            || hex(pin.binary_digest()) != registry.runtime_sha256
            || native_encoding_sha()? != registry.encoding_sha256
            || native_source_sha()? != registry.encoder_source_sha256
        {
            return Err(invalid(
                "actual native collection assets differ from independent registry",
            ));
        }
        let cpu = CpuEngine::new(cpu).map_err(|e| invalid(e.to_string()))?;
        let cpu_source =
            source_description(&cpu, "own-pals-native-cpu-tasks", serde_json::Value::Null)?;
        if cpu_source.cpu_profile_sha256 != registry.cpu_configuration_sha256 {
            return Err(invalid(
                "actual own CPU task configuration differs from independent registry",
            ));
        }
        let native_config = PalsOnnxConfig {
            intra_threads: registry.intra_threads,
            cache_public_memory: registry.cache_public_memory,
            ..PalsOnnxConfig::cpu()
        };
        let mut model = NativeRoleModel::load_pinned(
            export,
            &registry.export_manifest_sha256,
            pin,
            native_config,
            Duration::from_secs(30),
        )
        .map_err(|e| invalid(e.to_string()))?;
        let mut finish = NativeFinishGuard::new(model.finish_handle());
        let identity = model.source_identity();
        let actual = finish.receipt();
        let graphs: Vec<_> = actual
            .residency
            .graphs
            .iter()
            .map(|g| PalsCollectionGraphPin {
                role: g.role.clone(),
                sha256: hex(g.sha256),
                serialized_bytes: g.serialized_bytes,
            })
            .collect();
        let mut expected_graphs = registry.graphs.clone();
        let mut actual_graphs = graphs.clone();
        expected_graphs.sort_by(|a, b| a.role.cmp(&b.role));
        actual_graphs.sort_by(|a, b| a.role.cmp(&b.role));
        if identity.trained
            || identity.frozen_epoch != registry.frozen_epoch
            || identity.model_configuration != registry.model_configuration
            || hex(identity.checkpoint_sha256) != registry.checkpoint_sha256
            || hex(identity.export_manifest_sha256) != registry.export_manifest_sha256
            || hex(identity.execution.runtime_sha256) != registry.runtime_sha256
            || identity.execution.provider != "cpu"
            || identity.execution.runtime_bundle_sha256.is_some()
            || actual_graphs != expected_graphs
        {
            let _ = finish.finish(Instant::now() + Duration::from_secs(30));
            return Err(invalid(
                "loaded native source/epoch/graphs/provider differs from registered untrained CPU source",
            ));
        }
        let configuration = serde_json::to_value(&identity.model_configuration)
            .map_err(|e| invalid(e.to_string()))?;
        let description = PalsCollectionSourceDescription {
            mode: "own-pals-onnx-untrained-cpu".into(),
            source: PalsInputSource::OwnPals {
                model_configuration_sha256: registry.model_configuration_sha256.clone(),
                model_weights_sha256: registry.checkpoint_sha256.clone(),
            },
            cpu_profile_sha256: cpu_source.cpu_profile_sha256.clone(),
            implementation_sha256: actual_binary,
            encoding_sha256: registry.encoding_sha256.clone(),
            encoder_source_sha256: registry.encoder_source_sha256.clone(),
            configuration,
            model_epoch: identity.checkpoint_sha256,
            model_epoch_kind: "frozen_model_epoch".into(),
            frozen_epoch: identity.frozen_epoch,
            native: Some(
                serde_json::json!({"independent_registry":registry,"loaded_source":identity,
                "cpu_task_source":cpu_source,"graphs":graphs,"search_version":PALS_SEARCH_VERSION,
                "search_configuration":{"max_rounds":max_rounds,"beam_width":pals.beam_width,"line_plies":pals.line_plies,"max_nodes":pals.max_nodes,
                    "max_records":pals.max_records,"max_role_calls":pals.max_role_calls,"cpu_nodes_per_task":pals.cpu_nodes_per_task},
                "provider":"cpu","precision":"fp32","training_state":"Untrained","actual_training_executed":false,
                "startup_probe_in_collection":false,"trace_persistence":"seal-before-submit; prepaid-buffer-drain-after-search"}),
            ),
        };
        checked_source(&description)?;
        let sink = Arc::new(Mutex::new(Sink {
            context: None,
            source: description.clone(),
            trace: PalsNativeTrace::default(),
            started: Instant::now(),
            calls: BTreeMap::new(),
            raw_sources: BTreeSet::new(),
            total_rows: 0,
        }));
        model
            .set_observer(Box::new(Observer(Arc::clone(&sink))))
            .map_err(|e| invalid(e.to_string()))?;
        let engine = PalsEngine::new(pals, model, cpu).map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            engine,
            finish,
            sink,
            description,
            max_rounds,
        })
    }
}
impl PalsCollectionDriver for OwnPalsOnnxCollectionDriver {
    fn description(&self) -> &PalsCollectionSourceDescription {
        &self.description
    }
    fn new_game(&mut self) {
        self.engine.new_game();
    }
    fn records_actual_native_calls(&self) -> bool {
        true
    }
    fn analyze(
        &mut self,
        _: &Position,
        _: &PalsFrozenInput,
        _: &PalsModelInput,
        _: CpuLimits,
        _: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError> {
        Err(invalid(
            "real PALS collection requires actual native capture context",
        ))
    }
    fn analyze_native(
        &mut self,
        context: PalsNativeCaptureContext,
        limits: CpuLimits,
        cancel: &AtomicBool,
    ) -> Result<PalsCollectionDecision, ArenaError> {
        let position = context.position.clone();
        {
            let mut s = self
                .sink
                .lock()
                .map_err(|_| invalid("native collector sink poisoned"))?;
            if s.context.is_some() || !s.trace.rows.is_empty() {
                return Err(invalid("prior native trace was not drained"));
            }
            s.trace.sequence = context.sequence;
            s.context = Some(context);
            s.started = Instant::now();
            s.calls.clear();
            s.raw_sources.clear();
        }
        let before = self
            .engine
            .records()
            .iter()
            .map(|r| r.revision)
            .max()
            .unwrap_or(0);
        let result = self.engine.search(
            &position,
            PalsLimits {
                deadline: limits
                    .deadline
                    .ok_or_else(|| invalid("native collection deadline missing"))?,
                max_rounds: self.max_rounds,
                max_cpu_nodes: limits.max_nodes,
                cpu_depth: limits.max_depth,
            },
            cancel,
        );
        let counters = self.engine.last_search_counters();
        {
            let mut s = self
                .sink
                .lock()
                .map_err(|_| invalid("native collector sink poisoned"))?;
            let c = counters.unwrap_or_default();
            s.trace.cpu_nodes = c.cpu_nodes;
            s.trace.cpu_jobs = c.cpu_tasks;
            s.trace.cpu_work_observation_incomplete =
                c.cpu_work_observation_incomplete || (result.is_err() && counters.is_none());
            s.reserve(8192).map_err(|e| invalid(e.to_string()))?;
            let summary = serde_json::json!({"domain":"rz-pals-native-search-work/1",
                "cpu_nodes":c.cpu_nodes,"cpu_tasks_requested":c.cpu_tasks_requested,"cpu_reports_returned":c.cpu_tasks,
                "cpu_work_observation_incomplete":s.trace.cpu_work_observation_incomplete,
                "cpu_task_configuration_sha256":self.description.cpu_profile_sha256,
                "search_result":result.as_ref().err().map(failure_text),"role_calls":c.role_calls,
                "search_consumed_role_outputs":c.consumed_role_outputs});
            s.row("native-work-summary.jsonl", &summary)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let r = result.map_err(|e| invalid(e.to_string()))?;
        let mut genealogy = Vec::new();
        for record in self.engine.records().iter().filter(|r| r.revision > before) {
            genealogy.push(PalsCollectedLine {
                kind: format!("{:?}", record.kind),
                moves: pack(&record.line)?,
                raw_value: record.value,
                completed_depth: record.completed_depth,
                raw_scope: format!("{:?}", record.score_scope),
                white_perspective: record.perspective == Color::White,
            });
        }
        Ok(PalsCollectionDecision {
            best_move: r.best_move,
            pv: r.best_move.into_iter().collect(),
            nodes: r.counters.cpu_nodes,
            completed_depth: 0,
            completed_estimate: false,
            genealogy,
            raw: serde_json::json!({"source":"actual_frozen_pals_onnx_cpu","resolver_version":r.resolver_version,"training_state":"Untrained",
                "completion":format!("{:?}",r.completion),"value_scope":format!("{:?}",r.value_scope),"raw_score":r.score,
                "cpu_nodes":r.counters.cpu_nodes,"cpu_tasks":r.counters.cpu_tasks,"cpu_task_configuration_sha256":self.description.cpu_profile_sha256,
                "role_calls":r.counters.role_calls,"search_consumed_role_outputs":r.counters.consumed_role_outputs,
                "model":r.model_identity,"raw_wdl_is_target":false}),
        })
    }
    fn take_native_trace(&mut self) -> Result<PalsNativeTrace, ArenaError> {
        let mut s = self
            .sink
            .lock()
            .map_err(|_| invalid("native collector sink poisoned"))?;
        let missing = s
            .calls
            .iter()
            .find(|(_, c)| !c.physical && !c.rejected)
            .map(|(id, _)| {
                format!(
                    "native request {}:{} returned without physical/rejection evidence",
                    id.epoch.0, id.sequence
                )
            });
        if let Some(missing) = missing {
            s.trace.failure = Some(missing);
        }
        let consumed: BTreeSet<_> = s
            .calls
            .values()
            .filter(|c| c.accepted)
            .map(|c| c.input_sha256.clone())
            .collect();
        for (input, eligible) in &mut s.trace.inputs {
            *eligible &= consumed.contains(input.sha256());
        }
        s.context = None;
        Ok(std::mem::take(&mut s.trace))
    }
    fn finish_collection(
        &mut self,
        until: Instant,
    ) -> Result<Option<serde_json::Value>, ArenaError> {
        let finish_error = self.finish.finish(until).err().map(failure_text);
        let receipt = self.finish.receipt();
        let failed = finish_error.is_some()
            || receipt.quarantined
            || !receipt.physical_shutdown_confirmed
            || !receipt.native_buffers_released
            || receipt.physical_runs_in_flight != 0
            || receipt.observer_failures != 0;
        let value = serde_json::json!({"receipt":lifecycle_receipt(&receipt)?,"finish_error":finish_error,
            "_collection_failure":if failed{Some("native collector did not confirm physical shutdown/buffer lifetime/observer evidence")}else{None},
            "collection_accepted":!failed});
        Ok(Some(value))
    }
    fn receipt_preserved(&mut self) {
        self.finish.reported = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::ProcessEpoch;
    use rz_search::pals::engine::{DivergenceQuery, RoleQuery};
    use rz_uci::pals_native::{prepare_divergence_input, prepare_role_input};

    // Declared fake weight identities are confined to this observer fixture:
    // these tests never load, register, execute or claim an actual NN asset.
    fn sink(position: &Position, credit: u64) -> Arc<Mutex<Sink>> {
        let configuration = serde_json::to_value(PalsModelConfig::baseline()).unwrap();
        Arc::new(Mutex::new(Sink {
            context: Some(PalsNativeCaptureContext {
                position: position.clone(),
                opening: PalsCollectionOpening::default(),
                game: "fixture-g1".into(),
                actual_moves: Vec::new(),
                sequence: 4,
                max_bytes: credit,
            }),
            source: PalsCollectionSourceDescription {
                mode: "observer-fixture".into(),
                source: PalsInputSource::OwnPals {
                    model_configuration_sha256: canonical_sha256(&configuration).unwrap(),
                    model_weights_sha256: "07".repeat(32),
                },
                cpu_profile_sha256: "03".repeat(32),
                implementation_sha256: "04".repeat(32),
                encoding_sha256: native_encoding_sha().unwrap(),
                encoder_source_sha256: native_source_sha().unwrap(),
                configuration,
                model_epoch: [7; 32],
                model_epoch_kind: "frozen_model_epoch".into(),
                frozen_epoch: 1,
                native: None,
            },
            trace: PalsNativeTrace {
                sequence: 4,
                ..PalsNativeTrace::default()
            },
            started: Instant::now(),
            calls: BTreeMap::new(),
            raw_sources: BTreeSet::new(),
            total_rows: 0,
        }))
    }
    fn raw(count: usize) -> PalsRawOutput {
        PalsRawOutput {
            candidate_logits: vec![0.; count],
            wdl_logits: [0.; 3],
            divergence_logits: None,
            task_logits: None,
            private_latent: vec![0.; PalsModelConfig::baseline().latent_elements()],
        }
    }
    fn id(sequence: u64) -> RequestId {
        RequestId::new(ProcessEpoch(17), sequence)
    }
    #[test]
    fn wrong_epoch_rejects_before_reservation_or_dispatch_evidence() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [8; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        assert!(
            observer
                .prepared(
                    id(1),
                    &prepared,
                    NativePreparedContext::Role {
                        query: &query,
                        kind: NativeQueryKind::Propose
                    }
                )
                .is_err()
        );
        let s = shared.lock().unwrap();
        assert_eq!(s.trace.reserved_bytes, 0);
        assert!(s.calls.is_empty());
        assert!(s.trace.rows.is_empty());
        assert!(s.trace.failure.is_some());
    }
    #[test]
    fn exact_tensor_seal_precedes_raw_delivery_and_exactly_once_consumption() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
            )
            .unwrap();
        let frozen = shared.lock().unwrap().trace.inputs[0].0.clone();
        assert_eq!(frozen.snapshot().input_revision, 9);
        assert_eq!(frozen.snapshot().capture_sequence, 5);
        assert_eq!(frozen.snapshot().frozen_epoch, 1);
        let sidecar: PalsNativeInputSidecar = serde_json::from_slice(
            &shared
                .lock()
                .unwrap()
                .trace
                .rows
                .iter()
                .find(|r| r.artifact == "native-inputs.jsonl")
                .unwrap()
                .json,
        )
        .unwrap();
        assert_eq!(
            sidecar.tensor_json,
            serde_json::to_string(&prepared).unwrap()
        );
        let output = raw(legal.len());
        observer.physically_completed(id(1), Ok(&output)).unwrap();
        assert!(!shared.lock().unwrap().calls[&id(1)].accepted);
        observer.delivered(id(1)).unwrap();
        observer.accepted(id(1)).unwrap();
        assert!(observer.accepted(id(1)).is_err());
        let s = shared.lock().unwrap();
        let events: Vec<_> = s
            .trace
            .rows
            .iter()
            .filter(|r| r.artifact == "native-events.jsonl")
            .map(|r| {
                serde_json::from_slice::<serde_json::Value>(&r.json).unwrap()["stage"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(
            events,
            vec![
                "prepared",
                "physically_completed",
                "delivered",
                "search_consumed"
            ]
        );
        assert_eq!(s.trace.inputs[0].0, frozen);
        assert!(
            s.trace.reserved_bytes >= s.trace.rows.iter().map(|r| r.json.len() as u64 + 1).sum()
        );
    }
    #[test]
    fn cancellation_after_delivery_never_consumes_or_rewrites_physical_evidence() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
            )
            .unwrap();
        let mut output = raw(legal.len());
        output.candidate_logits[0] = f32::from_bits(0x7fc01234);
        observer.physically_completed(id(1), Ok(&output)).unwrap();
        observer.delivered(id(1)).unwrap();
        cancel.store(true, Ordering::Release);
        observer
            .rejected(id(1), NativeRoleRejection::Canceled)
            .unwrap();
        assert!(observer.accepted(id(1)).is_err());
        let s = shared.lock().unwrap();
        assert!(s.calls[&id(1)].physical);
        assert!(!s.calls[&id(1)].accepted);
        let raw: serde_json::Value = serde_json::from_slice(
            &s.trace
                .rows
                .iter()
                .find(|r| r.artifact == "native-raw-outputs.jsonl")
                .unwrap()
                .json,
        )
        .unwrap();
        assert_eq!(
            raw["raw"]["candidate_logits_bits"][0].as_u64(),
            Some(0x7fc01234)
        );
    }
    #[test]
    fn unknown_physical_completion_cannot_deliver_or_reuse_the_request() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
            )
            .unwrap();
        observer
            .rejected(id(1), NativeRoleRejection::PhysicalCompletionUnknown)
            .unwrap();
        assert!(observer.delivered(id(1)).is_err());
        assert!(observer.accepted(id(1)).is_err());
        assert!(
            observer
                .prepared(
                    id(1),
                    &prepared,
                    NativePreparedContext::Role {
                        query: &query,
                        kind: NativeQueryKind::Propose
                    }
                )
                .is_err()
        );
        assert!(shared.lock().unwrap().calls[&id(1)].physical_unknown);
    }
    #[test]
    fn same_root_repair_and_divergence_never_receive_actual_game_wdl() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let response = [legal[0]];
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &response,
            counterexample: Some(&response),
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Repair, [7; 32]).unwrap();
        let shared = sink(&position, 2 * 1024 * 1024);
        let mut observer = Observer(Arc::clone(&shared));
        observer
            .prepared(
                id(1),
                &prepared,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Repair,
                },
            )
            .unwrap();
        assert!(!shared.lock().unwrap().trace.inputs[0].1);
        let divergence_proposal = [
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        let dq = DivergenceQuery {
            root: &position,
            proposal: &divergence_proposal,
            candidates: &[1],
            records: &[],
            revision: 10,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let divergence = prepare_divergence_input(&dq, [7; 32]).unwrap();
        observer
            .prepared(
                id(2),
                &divergence,
                NativePreparedContext::Divergence { query: &dq },
            )
            .unwrap();
        let s = shared.lock().unwrap();
        assert_eq!(s.trace.inputs.len(), 1);
        assert!(
            s.trace
                .rows
                .iter()
                .any(|r| r.artifact == "native-divergence-inputs.jsonl")
        );
        assert!(
            s.trace
                .rows
                .iter()
                .any(|r| r.artifact == "native-divergence-sidecars.jsonl")
        );
    }
    #[test]
    fn delegated_credit_reserves_raw_and_close_events_before_any_dispatch() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 9,
            deadline: Instant::now() + Duration::from_secs(2),
            cancel: &cancel,
        };
        let prepared = prepare_role_input(&query, NativeQueryKind::Propose, [7; 32]).unwrap();
        let shared = sink(&position, RAW_RESERVE + STAGE_RESERVE - 1);
        let mut observer = Observer(Arc::clone(&shared));
        assert!(
            observer
                .prepared(
                    id(1),
                    &prepared,
                    NativePreparedContext::Role {
                        query: &query,
                        kind: NativeQueryKind::Propose
                    }
                )
                .is_err()
        );
        let s = shared.lock().unwrap();
        assert!(s.calls.is_empty());
        assert_eq!(s.trace.reserved_bytes, 0);
        assert!(s.trace.rows.is_empty());
    }
}
