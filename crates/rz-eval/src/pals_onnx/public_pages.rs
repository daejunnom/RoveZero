//! Opt-in host pages for the registered independent record encoder. This is
//! exact feature reuse, not a summary, CPU evidence reuse, or a device cache.
use super::*;
use crate::pals_model::{IndependentPublicPlan, INDEPENDENT_RECORD_PROJECTION_SEMANTICS};

const BOARD_TOKENS: usize = 66;
const HEADS: usize = 2;
const DIMENSION: usize = 64;
const MAX_HOST_BYTES: u64 = 16 * 1024 * 1024;

/// Explicit, graph-pinned admission. The declaration does not replace the
/// independent full-versus-page numerical acceptance required by the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostRecordPagePolicy {
    pub public_graph_sha256: [u8; 32],
    pub projection_semantics: &'static str,
    pub max_page_entries: usize,
    /// Complete page backing owners (including capacity), excluding ORT's
    /// unobserved allocator/workspace and the separately bounded transient set.
    pub max_page_bytes: u64,
    /// Simultaneously live, shape-bounded input/join/page-copy/output storage.
    /// This is a host reservation, not a measured native or VRAM peak.
    pub max_transient_bytes: u64,
}
impl HostRecordPagePolicy {
    /// Finite starting policy, never silently enabled. The caller still pins
    /// the graph and independently accepts full-versus-page numerical results.
    pub const fn for_registered_graph(public_graph_sha256: [u8; 32]) -> Self {
        Self {
            public_graph_sha256,
            projection_semantics: INDEPENDENT_RECORD_PROJECTION_SEMANTICS,
            max_page_entries: 257,
            max_page_bytes: 2 * 1024 * 1024,
            max_transient_bytes: 2 * 1024 * 1024,
        }
    }
    pub const fn for_profile_registered_graph(
        profile: PalsModelProfile,
        public_graph_sha256: [u8; 32],
    ) -> Self {
        Self {
            projection_semantics: profile.independent_projection_semantics(),
            // Full-line input and its missing-record subset overlap until the
            // native fence. Charge both, rather than shrinking the sequence.
            max_transient_bytes: if profile.uses_full_line() {
                8 * 1024 * 1024
            } else {
                2 * 1024 * 1024
            },
            ..Self::for_registered_graph(public_graph_sha256)
        }
    }
    fn entry_budget(self) -> u64 {
        2 * self.max_page_entries as u64 * MemoryBank::<HostPage>::entry_slot_bytes()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HostRecordPageStats {
    pub view_hits: u64,
    pub view_misses: u64,
    pub board_hits: u64,
    pub board_misses: u64,
    /// Unique feature projections, not observations, visits or NN inputs.
    pub record_hits: u64,
    pub record_misses: u64,
    pub public_calls_attempted: u64,
    pub public_calls_completed: u64,
    /// Actual slots supplied to the public graph, including masked padding.
    /// These slots are not the physical B1 NN-input count.
    pub encoded_record_tokens: u64,
    pub submitted_record_tokens: u64,
    /// The existing graph still recomputes the contextual board on every
    /// subset call, even if its board page was already cached.
    pub contextual_board_encodes_completed: u64,
    pub joins_completed: u64,
    pub joined_bytes: u64,
    pub evicted_pages: u64,
    pub max_reserved_host_transient_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostRecordPageSnapshot {
    pub policy: HostRecordPagePolicy,
    pub bank: MemoryBankSnapshot,
    pub stats: HostRecordPageStats,
    pub active_pin_count: usize,
    pub retained_join_bytes: u64,
    pub active_subset_bytes: u64,
    pub active_join_backing_bytes: u64,
    pub active_full_input_bytes: u64,
    pub transient_reservation_bytes: u64,
    pub quarantined: bool,
}

struct HostPage {
    tokens: usize,
    // Box slices own exactly these elements; never a slice retaining a larger
    // public output owner. The bank charges the full header and both arrays.
    key: Box<[f32]>,
    value: Box<[f32]>,
}
impl HostPage {
    fn bytes(tokens: usize) -> u64 {
        // Include the known two atomic counters of the Arc allocation, without
        // pretending to observe allocator rounding/native heap overhead.
        (std::mem::size_of::<Self>()
            + 2 * std::mem::size_of::<usize>()
            + 2 * HEADS * tokens * DIMENSION * 4) as u64
    }
}

struct PublicSubset {
    records: Tensor<f32>,
    mask: Tensor<bool>,
    bytes: u64,
    slots: usize,
    lines: Option<PublicSubsetLines>,
}
struct PublicSubsetLines {
    tokens: Tensor<i64>,
    mask: Tensor<bool>,
}
struct JoinBuffers {
    key: Vec<f32>,
    value: Vec<f32>,
    mask: Vec<bool>,
}
impl JoinBuffers {
    fn bytes(&self) -> u64 {
        public_owner_bytes(
            self.key.capacity(),
            self.value.capacity(),
            self.mask.capacity(),
        )
    }
}

pub(super) struct HostRecordPages {
    policy: HostRecordPagePolicy,
    bank: MemoryBank<HostPage>,
    stats: HostRecordPageStats,
    pins: Vec<(MemoryKey, MemoryPin<HostPage>)>,
    subset: Option<PublicSubset>,
    join_buffers: Option<JoinBuffers>,
    joined: Option<PublicMemory>,
    joined_bytes: u64,
    transient_reservation: u64,
}

impl HostRecordPages {
    pub(super) fn release_completed(&mut self) {
        // Called only after the existing physical completion/quarantine guard.
        self.pins.clear();
        self.subset = None;
        self.join_buffers = None;
        self.transient_reservation = 0;
    }
    pub(super) fn clear(&mut self) -> Result<(), BackendError> {
        self.bank.clear().map_err(public_bank_error)?;
        self.joined = None;
        self.joined_bytes = 0;
        Ok(())
    }
    pub(super) fn joined(&self) -> Option<&PublicMemory> {
        self.joined.as_ref()
    }
    pub(super) fn bank_snapshot(&self) -> MemoryBankSnapshot {
        self.bank.snapshot()
    }
    fn snapshot(&self, quarantined: bool) -> HostRecordPageSnapshot {
        HostRecordPageSnapshot {
            policy: self.policy,
            bank: self.bank.snapshot(),
            stats: self.stats.clone(),
            active_pin_count: self.pins.len(),
            retained_join_bytes: self.joined_bytes,
            active_subset_bytes: self.subset.as_ref().map_or(0, |subset| subset.bytes),
            active_join_backing_bytes: self.join_buffers.as_ref().map_or(0, JoinBuffers::bytes),
            active_full_input_bytes: 0,
            transient_reservation_bytes: self.transient_reservation,
            quarantined,
        }
    }
}

impl PalsOnnxBackend {
    /// Enable only before the first admission. Default whole-input and existing
    /// device K/V remain unchanged. Never silently substitute a host page path.
    pub fn enable_host_record_pages(
        &mut self,
        policy: HostRecordPagePolicy,
    ) -> Result<(), BackendError> {
        if self.config.device_public_memory || !self.config.cache_public_memory {
            return Err(fail(
                K::BackendUnavailable,
                S::Admission,
                "host record pages require explicit host caching without device public memory",
            ));
        }
        if self.quarantine.is_some()
            || self.active.is_some()
            || self.stats.admitted_role_requests != 0
            || self.record_pages.is_some()
        {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "host record page policy must be selected once before the first admission",
            ));
        }
        let graph = self
            .residency
            .graphs
            .iter()
            .find(|graph| graph.role == "public")
            .ok_or_else(|| {
                fail(
                    K::IdentityMismatch,
                    S::Admission,
                    "public graph identity is absent",
                )
            })?;
        if graph.sha256 != policy.public_graph_sha256
            || policy.projection_semantics
                != self.model_config.profile.independent_projection_semantics()
        {
            return Err(fail(
                K::IdentityMismatch,
                S::Admission,
                "host record page graph or independent projection declaration differs",
            ));
        }
        if !(2..=1024).contains(&policy.max_page_entries)
            || policy.max_page_bytes == 0
            || policy.max_page_bytes > MAX_HOST_BYTES
            || policy.max_transient_bytes == 0
            || policy.max_transient_bytes > MAX_HOST_BYTES
        {
            return Err(fail(
                K::InvalidInput,
                S::Admission,
                "host record page limits are invalid",
            ));
        }
        let value_budget = policy
            .max_page_bytes
            .checked_sub(policy.entry_budget())
            .filter(|budget| *budget > 0)
            .ok_or_else(|| {
                fail(
                    K::InvalidInput,
                    S::Admission,
                    "host page budget cannot reserve its finite entry backing",
                )
            })?;
        self.record_pages = Some(HostRecordPages {
            policy,
            bank: MemoryBank::new(policy.max_page_entries, value_budget)
                .map_err(public_bank_error)?,
            stats: HostRecordPageStats::default(),
            pins: Vec::new(),
            subset: None,
            join_buffers: None,
            joined: None,
            joined_bytes: 0,
            transient_reservation: 0,
        });
        Ok(())
    }

    /// Available even during quarantine: actual retained ownership evidence.
    pub fn host_record_page_snapshot(&self) -> Option<HostRecordPageSnapshot> {
        self.record_pages.as_ref().map(|pages| {
            let mut snapshot = pages.snapshot(self.quarantine.is_some());
            snapshot.active_full_input_bytes =
                self.active.as_ref().map_or(0, |input| input.host_bytes);
            snapshot
        })
    }

    fn projection_key(
        &self,
        kind: PublicPageKind,
        content: [u8; 32],
        policy: HostRecordPagePolicy,
    ) -> MemoryKey {
        projected_page_key_for(
            kind,
            content,
            policy,
            self.epoch,
            self.manifest_digest,
            self.game_generation,
            self.model_config.profile,
        )
    }

    pub(super) fn prepare_record_pages(
        &mut self,
        input: &PalsModelInput,
        trace: Option<&StartupRoleTrace>,
    ) -> Result<(), BackendError> {
        // Reinstall on every outcome, including an unknown CUDA completion.
        // Pins, subset tensors and join buffers then remain with the owner.
        let mut pages = self.record_pages.take().expect("record policy installed");
        let result = self.prepare_page_view(input, trace, &mut pages);
        self.record_pages = Some(pages);
        result
    }

    fn prepare_page_view(
        &mut self,
        input: &PalsModelInput,
        trace: Option<&StartupRoleTrace>,
        pages: &mut HostRecordPages,
    ) -> Result<(), BackendError> {
        let plan = input
            .independent_public_plan(&self.model_config)
            .map_err(model_input)?;
        let board_key = self.projection_key(
            PublicPageKind::ContextualBoard,
            plan.board_content,
            pages.policy,
        );
        let record_keys: Vec<_> = plan
            .record_contents
            .iter()
            .map(|content| {
                self.projection_key(
                    PublicPageKind::IndependentRecordProjection,
                    *content,
                    pages.policy,
                )
            })
            .collect();
        // A previous joined view is released only at the next known-complete
        // invocation; it is not a bank page and never licenses evidence reuse.
        pages.joined = None;
        pages.joined_bytes = 0;
        let mut missing = Vec::new();
        if let Some(pin) = pages.bank.acquire(board_key) {
            pages.stats.board_hits = pages.stats.board_hits.saturating_add(1);
            pages.pins.push((board_key, pin));
        } else {
            pages.stats.board_misses = pages.stats.board_misses.saturating_add(1);
            missing.push((board_key, BOARD_TOKENS));
        }
        let mut miss_records = Vec::new();
        for (index, key) in record_keys.iter().enumerate() {
            if record_keys[..index].contains(key) {
                continue;
            }
            if let Some(pin) = pages.bank.acquire(*key) {
                pages.stats.record_hits = pages.stats.record_hits.saturating_add(1);
                pages.pins.push((*key, pin));
            } else {
                pages.stats.record_misses = pages.stats.record_misses.saturating_add(1);
                missing.push((*key, 1));
                miss_records.push(index);
            }
        }
        let sizes: Vec<_> = missing
            .iter()
            .map(|(key, tokens)| (*key, HostPage::bytes(*tokens)))
            .collect();
        pages
            .bank
            .preflight_batch(&sizes)
            .map_err(public_bank_error)?;
        // Container slots are an actual additional host owner. Reserve them
        // without evicting any page, and charge their complete capacity against
        // the explicit page budget as well as the later transactional values.
        // The bank's value limit already excludes this fixed, finite entry
        // backing budget. Copied new pages overlap old pages before atomic
        // eviction and are additionally charged to the transient reservation.
        let entry_budget = pages.policy.entry_budget();
        pages
            .bank
            .reserve_batch_backing(&sizes, entry_budget)
            .map_err(public_bank_error)?;
        pages.pins.reserve(missing.len());
        let tokens = BOARD_TOKENS + record_keys.len();
        let native_tokens = BOARD_TOKENS + miss_records.len().max(1);
        let copied_pages: u64 = sizes.iter().map(|(_, bytes)| *bytes).sum();
        let native_output_bytes =
            (2 * HEADS * native_tokens * DIMENSION * 4 + native_tokens) as u64;
        let planner_bytes = plan_capacity_bytes(&plan)
            + (record_keys.capacity() * std::mem::size_of::<MemoryKey>()
                + missing.capacity() * std::mem::size_of::<(MemoryKey, usize)>()
                + miss_records.capacity() * std::mem::size_of::<usize>()
                + sizes.capacity() * std::mem::size_of::<(MemoryKey, u64)>()
                + pages.pins.capacity() * std::mem::size_of::<(MemoryKey, MemoryPin<HostPage>)>()
                + 2 * missing.len()
                    * (std::mem::size_of::<(MemoryKey, HostPage, u64)>()
                        + std::mem::size_of::<(MemoryKey, u64)>()
                        + std::mem::size_of::<(MemoryKey, Arc<HostPage>, u64)>()
                        + std::mem::size_of::<MemoryPin<HostPage>>())) as u64;
        let conversion_bytes = missing
            .iter()
            .map(|(_, count)| HostPage::bytes(*count))
            .max()
            .unwrap_or(0);
        let fixed_reservation = self.active.as_ref().expect("prepared input installed").host_bytes
            + std::mem::size_of::<HostRecordPages>() as u64
            + planner_bytes + copied_pages + conversion_bytes
            // Native role outputs plus the separately owned decoded raw arrays
            // overlap the input, cache pins and joined view until role fence.
            + 2 * role_output_bytes(input)
            + if missing.is_empty() { 0 } else { native_output_bytes };
        let subset_slots = miss_records.len().max(1);
        let predicted = fixed_reservation
            + public_owner_bytes(
                HEADS * tokens * DIMENSION,
                HEADS * tokens * DIMENSION,
                tokens,
            )
            + if missing.is_empty() {
                0
            } else {
                (subset_slots * 16 * 4
                    + subset_slots
                    + std::mem::size_of::<PublicSubset>()
                    + if plan.record_lines.is_some() {
                        subset_slots * MAX_LINE_PLY * (3 * 8 + 1)
                    } else {
                        0
                    }) as u64
            };
        if predicted > pages.policy.max_transient_bytes {
            return Err(fail(
                K::ResourceExhausted,
                S::Admission,
                "host page transient reservation is refused before join/subset allocation",
            ));
        }
        pages.join_buffers = Some(JoinBuffers {
            key: vec![0.; HEADS * tokens * DIMENSION],
            value: vec![0.; HEADS * tokens * DIMENSION],
            mask: std::iter::repeat_n(true, BOARD_TOKENS)
                .chain(plan.record_mask.iter().copied())
                .collect(),
        });
        if !missing.is_empty() {
            let slots = miss_records.len().max(1);
            let mut features = vec![0.; slots * 16];
            let mut mask = vec![false; slots];
            let mut line_arrays = plan.record_lines.as_ref().map(|_| {
                (
                    vec![0; slots * MAX_LINE_PLY * 3],
                    vec![false; slots * MAX_LINE_PLY],
                )
            });
            for (slot, index) in miss_records.iter().enumerate() {
                features[slot * 16..(slot + 1) * 16].copy_from_slice(&plan.features[*index]);
                mask[slot] = plan.record_mask[*index];
                if let (Some(source), Some((tokens, line_mask))) =
                    (&plan.record_lines, &mut line_arrays)
                {
                    for (ply, movement) in source[*index].iter().enumerate() {
                        let offset = (slot * MAX_LINE_PLY + ply) * 3;
                        tokens[offset..offset + 3].copy_from_slice(&[
                            movement.from as i64,
                            movement.to as i64,
                            movement.promotion as i64,
                        ]);
                        line_mask[slot * MAX_LINE_PLY + ply] = true;
                    }
                }
            }
            let bytes = (features.capacity() * 4
                + mask.capacity()
                + std::mem::size_of::<PublicSubset>()
                + line_arrays.as_ref().map_or(0, |(tokens, line_mask)| {
                    tokens.capacity() * 8 + line_mask.capacity()
                })) as u64;
            let lines = line_arrays
                .map(|(tokens, mask)| {
                    Ok::<_, BackendError>(PublicSubsetLines {
                        tokens: Tensor::from_array(([1, slots, MAX_LINE_PLY, 3], tokens))
                            .map_err(tensor_create)?,
                        mask: Tensor::from_array(([1, slots, MAX_LINE_PLY], mask))
                            .map_err(tensor_create)?,
                    })
                })
                .transpose()?;
            pages.subset = Some(PublicSubset {
                records: Tensor::from_array(([1, slots, 16], features)).map_err(tensor_create)?,
                mask: Tensor::from_array(([1, slots], mask)).map_err(tensor_create)?,
                bytes,
                slots,
                lines,
            });
        }
        // Reserve every known overlapping owner before Run: original prepared
        // inputs, subset input, plan/index/pin storage, joined backing, complete
        // native public outputs, copied missing pages and bank entry backing.
        // Unknown native session/workspace/allocator overhead is NOT a measured
        // or guaranteed process/VRAM peak and is not invented here.
        let reservation = fixed_reservation
            + pages
                .join_buffers
                .as_ref()
                .expect("join backing installed")
                .bytes()
            + pages.subset.as_ref().map_or(0, |subset| subset.bytes);
        if reservation > pages.policy.max_transient_bytes {
            return Err(fail(
                K::ResourceExhausted,
                S::Admission,
                "host record page transient backing reservation exceeds its explicit limit",
            ));
        }
        pages.transient_reservation = reservation;
        pages.stats.max_reserved_host_transient_bytes = pages
            .stats
            .max_reserved_host_transient_bytes
            .max(reservation);
        if missing.is_empty() {
            pages.stats.view_hits = pages.stats.view_hits.saturating_add(1);
            if let Some(trace) = trace {
                trace.record(
                    PalsStartupBackendStage::PublicCacheHit,
                    PalsStartupStageBoundary::Observed,
                );
            }
        } else {
            pages.stats.view_misses = pages.stats.view_misses.saturating_add(1);
            self.fill_missing_pages(
                pages,
                board_key,
                &record_keys,
                &miss_records,
                &missing,
                trace,
            )?;
        }
        let mut buffers = pages.join_buffers.take().expect("joined backing installed");
        let board = pages
            .pins
            .iter()
            .find(|(key, _)| *key == board_key)
            .expect("board page installed");
        copy_page(&mut buffers, tokens, 0, &board.1)?;
        for (slot, key) in record_keys.iter().enumerate() {
            let page = pages
                .pins
                .iter()
                .find(|(candidate, _)| candidate == key)
                .expect("record page installed");
            copy_page(&mut buffers, tokens, BOARD_TOKENS + slot, &page.1)?;
        }
        let joined_bytes = buffers.bytes();
        pages.joined = Some(PublicMemory {
            tokens,
            memory_key: Tensor::from_array(([1, HEADS, tokens, DIMENSION], buffers.key))
                .map_err(tensor_create)?,
            memory_value: Tensor::from_array(([1, HEADS, tokens, DIMENSION], buffers.value))
                .map_err(tensor_create)?,
            mask: Tensor::from_array(([1, tokens], buffers.mask)).map_err(tensor_create)?,
        });
        pages.joined_bytes = joined_bytes;
        pages.stats.joins_completed = pages.stats.joins_completed.saturating_add(1);
        pages.stats.joined_bytes = pages.stats.joined_bytes.saturating_add(joined_bytes);
        Ok(())
    }

    fn fill_missing_pages(
        &mut self,
        pages: &mut HostRecordPages,
        board_key: MemoryKey,
        record_keys: &[MemoryKey],
        miss_records: &[usize],
        missing: &[(MemoryKey, usize)],
        trace: Option<&StartupRoleTrace>,
    ) -> Result<(), BackendError> {
        let active = self.active.as_ref().expect("prepared inputs installed");
        let subset = pages.subset.as_ref().expect("subset installed before Run");
        let slots = subset.slots;
        let tokens = BOARD_TOKENS + slots;
        self.stats.public_nn_runs_attempted = self.stats.public_nn_runs_attempted.saturating_add(1);
        pages.stats.public_calls_attempted = pages.stats.public_calls_attempted.saturating_add(1);
        pages.stats.submitted_record_tokens = pages
            .stats
            .submitted_record_tokens
            .saturating_add(slots as u64);
        startup_enter(trace, PalsStartupBackendStage::PublicRun);
        let mut values = ort::inputs!["board" => &active.board, "metadata" => &active.metadata,
            "records" => &subset.records, "record_mask" => &subset.mask];
        if let Some(lines) = &subset.lines {
            values.extend(ort::inputs!["record_line_tokens" => &lines.tokens, "record_line_mask" => &lines.mask]);
        }
        let result = self
            .public
            .as_mut()
            .expect("public session installed")
            .run(values);
        startup_return(trace, PalsStartupBackendStage::PublicRun, result.is_ok());
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                let failure = native(
                    CauseCode::OrtRun,
                    "PALS record subset native Run failed",
                    error,
                );
                if matches!(self.config.provider, Provider::Cuda { .. }) {
                    self.quarantine = Some(failure.clone());
                } else {
                    self.stats.public_nn_runs_failed_known =
                        self.stats.public_nn_runs_failed_known.saturating_add(1);
                }
                return Err(failure);
            }
        };
        self.stats.public_nn_runs_completed = self.stats.public_nn_runs_completed.saturating_add(1);
        self.stats.completed_nn_inputs = self.stats.completed_nn_inputs.saturating_add(1);
        pages.stats.public_calls_completed = pages.stats.public_calls_completed.saturating_add(1);
        pages.stats.encoded_record_tokens = pages
            .stats
            .encoded_record_tokens
            .saturating_add(slots as u64);
        pages.stats.contextual_board_encodes_completed = pages
            .stats
            .contextual_board_encodes_completed
            .saturating_add(1);
        startup_enter(trace, PalsStartupBackendStage::PublicOutputPreparation);
        let extract = |name: &str| -> Result<&[f32], BackendError> {
            let (shape, values) = outputs[name].try_extract_tensor::<f32>().map_err(|error| {
                native(
                    CauseCode::PolicyExtract,
                    "record subset K/V extraction failed",
                    error,
                )
            })?;
            if shape.as_ref() != [1, HEADS as i64, tokens as i64, DIMENSION as i64]
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(fail(
                    K::NumericalFailure,
                    S::Output,
                    "record subset K/V shape or finite failure",
                ));
            }
            Ok(values)
        };
        let key_values = extract("memory_key")?;
        let value_values = extract("memory_value")?;
        let (shape, mask) = outputs["memory_mask"]
            .try_extract_tensor::<bool>()
            .map_err(|error| {
                native(
                    CauseCode::PolicyExtract,
                    "record subset mask extraction failed",
                    error,
                )
            })?;
        let (_, expected_mask) = subset.mask.try_extract_tensor::<bool>().map_err(|error| {
            native(
                CauseCode::PolicyExtract,
                "record subset input mask extraction failed",
                error,
            )
        })?;
        if shape.as_ref() != [1, tokens as i64]
            || mask[..BOARD_TOKENS].iter().any(|value| !*value)
            || mask[BOARD_TOKENS..] != *expected_mask
        {
            return Err(fail(
                K::NumericalFailure,
                S::Output,
                "record subset mask differs from actual supplied slots",
            ));
        }
        let mut additions = Vec::with_capacity(missing.len());
        for (key, page_tokens) in missing {
            let offset = if *key == board_key {
                0
            } else {
                let slot = miss_records
                    .iter()
                    .position(|index| record_keys[*index] == *key)
                    .expect("missing record slot exists");
                BOARD_TOKENS + slot
            };
            let page = extract_page(key_values, value_values, tokens, offset, *page_tokens)?;
            additions.push((*key, page, HostPage::bytes(*page_tokens)));
        }
        drop(outputs);
        let before = pages.bank.len();
        let added = additions.len();
        // The transaction repeats its complete ownership/budget validation
        // before eviction. Native/extraction failures never destroy old pages.
        let pins = pages
            .bank
            .insert_batch(additions)
            .map_err(public_bank_error)?;
        pages.stats.evicted_pages = pages
            .stats
            .evicted_pages
            .saturating_add((before + added - pages.bank.len()) as u64);
        for ((key, _), pin) in missing.iter().zip(pins) {
            pages.pins.push((*key, pin));
        }
        self.public_encodes = self.public_encodes.saturating_add(1);
        self.stats.validated_public_outputs = self.stats.validated_public_outputs.saturating_add(1);
        startup_return(
            trace,
            PalsStartupBackendStage::PublicOutputPreparation,
            true,
        );
        Ok(())
    }
}

fn tensor_create(error: ort::Error) -> BackendError {
    native(
        CauseCode::TensorCreate,
        "cannot own bounded host record page tensor",
        error,
    )
}
#[cfg(test)]
fn projected_page_key(
    kind: PublicPageKind,
    content: [u8; 32],
    policy: HostRecordPagePolicy,
    epoch: [u8; 32],
    manifest: [u8; 32],
    game_generation: u64,
) -> MemoryKey {
    projected_page_key_for(
        kind,
        content,
        policy,
        epoch,
        manifest,
        game_generation,
        PalsModelProfile::LegacySummaryV1,
    )
}
fn projected_page_key_for(
    kind: PublicPageKind,
    content: [u8; 32],
    policy: HostRecordPagePolicy,
    epoch: [u8; 32],
    manifest: [u8; 32],
    game_generation: u64,
    profile: PalsModelProfile,
) -> MemoryKey {
    // Bind the content to both the full checkpoint epoch and actual public
    // graph. The old full canonical input/public key is never rewritten.
    let mut hash = sha2::Sha256::new();
    use sha2::Digest as _;
    hash.update(content);
    hash.update(epoch);
    hash.update(policy.public_graph_sha256);
    hash.update(profile.independent_projection_semantics().as_bytes());
    if !profile.is_legacy() {
        hash.update(profile.as_str().as_bytes());
    }
    MemoryKey::PublicPage(PublicPageKey {
        kind,
        content: Digest(hash.finalize().into()),
        model: Digest(manifest),
        encoding: Digest(asset::sha256(profile.encoding_schema().as_bytes())),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 0,
        game_generation,
    })
}
fn plan_capacity_bytes(plan: &IndependentPublicPlan) -> u64 {
    (std::mem::size_of::<IndependentPublicPlan>()
        + plan.record_contents.capacity() * 32
        + plan.features.capacity() * 16 * 4
        + plan.record_mask.capacity()) as u64
        + plan.record_lines.as_ref().map_or(0, |lines| {
            (lines.capacity() * std::mem::size_of::<Vec<crate::pals_model::PalsCandidateToken>>()
                + lines
                    .iter()
                    .map(|line| {
                        line.capacity()
                            * std::mem::size_of::<crate::pals_model::PalsCandidateToken>()
                    })
                    .sum::<usize>()) as u64
        })
}
fn role_output_bytes(input: &PalsModelInput) -> u64 {
    let floats = input.candidates.len().max(1)
        + 3
        + 16 * 384
        + if input.role == PalsRole::Critic {
            input.divergence_features.len().max(1)
        } else {
            1
        };
    (floats * 4 + std::mem::size_of::<PalsRawOutput>() + 1) as u64
}
fn extract_page(
    key: &[f32],
    value: &[f32],
    tokens: usize,
    offset: usize,
    length: usize,
) -> Result<HostPage, BackendError> {
    if offset.checked_add(length).is_none_or(|end| end > tokens)
        || key.len() != HEADS * tokens * DIMENSION
        || value.len() != key.len()
    {
        return Err(fail(
            K::NumericalFailure,
            S::Output,
            "public page slice is outside actual output",
        ));
    }
    let copy = |values: &[f32]| -> Box<[f32]> {
        (0..HEADS)
            .flat_map(|head| {
                let start = (head * tokens + offset) * DIMENSION;
                values[start..start + length * DIMENSION].iter().copied()
            })
            .collect()
    };
    Ok(HostPage {
        tokens: length,
        key: copy(key),
        value: copy(value),
    })
}
fn copy_page(
    buffers: &mut JoinBuffers,
    tokens: usize,
    offset: usize,
    page: &HostPage,
) -> Result<(), BackendError> {
    if offset
        .checked_add(page.tokens)
        .is_none_or(|end| end > tokens)
        || page.key.len() != HEADS * page.tokens * DIMENSION
        || page.value.len() != page.key.len()
    {
        return Err(fail(
            K::NumericalFailure,
            S::Output,
            "joined public page shape differs",
        ));
    }
    for head in 0..HEADS {
        let source = head * page.tokens * DIMENSION;
        let target = (head * tokens + offset) * DIMENSION;
        let count = page.tokens * DIMENSION;
        buffers.key[target..target + count].copy_from_slice(&page.key[source..source + count]);
        buffers.value[target..target + count].copy_from_slice(&page.value[source..source + count]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn v2_page_policy_and_keys_bind_complete_encoder_profile() {
        let profile = PalsModelProfile::FullLineInteractionV2;
        let policy = HostRecordPagePolicy::for_profile_registered_graph(profile, [1; 32]);
        assert_eq!(
            policy.projection_semantics,
            profile.independent_projection_semantics()
        );
        assert!(
            policy.max_transient_bytes
                > HostRecordPagePolicy::for_registered_graph([1; 32]).max_transient_bytes
        );
        let key = projected_page_key_for(
            PublicPageKind::IndependentRecordProjection,
            [2; 32],
            policy,
            [3; 32],
            [4; 32],
            1,
            profile,
        );
        let other = projected_page_key_for(
            PublicPageKind::IndependentRecordProjection,
            [2; 32],
            policy,
            [3; 32],
            [4; 32],
            1,
            PalsModelProfile::FullLineV2,
        );
        assert_ne!(key, other);
        let MemoryKey::PublicPage(page) = key else {
            panic!("public page expected")
        };
        assert_eq!(
            page.encoding,
            Digest(asset::sha256(profile.encoding_schema().as_bytes()))
        );
    }
    #[test]
    fn projection_pages_fence_graph_checkpoint_encoding_precision_and_game() {
        let policy = HostRecordPagePolicy::for_registered_graph([1; 32]);
        let key = projected_page_key(
            PublicPageKind::IndependentRecordProjection,
            [2; 32],
            policy,
            [3; 32],
            [4; 32],
            1,
        );
        for changed in [
            projected_page_key(
                PublicPageKind::ContextualBoard,
                [2; 32],
                policy,
                [3; 32],
                [4; 32],
                1,
            ),
            projected_page_key(
                PublicPageKind::IndependentRecordProjection,
                [9; 32],
                policy,
                [3; 32],
                [4; 32],
                1,
            ),
            projected_page_key(
                PublicPageKind::IndependentRecordProjection,
                [2; 32],
                HostRecordPagePolicy::for_registered_graph([9; 32]),
                [3; 32],
                [4; 32],
                1,
            ),
            projected_page_key(
                PublicPageKind::IndependentRecordProjection,
                [2; 32],
                policy,
                [9; 32],
                [4; 32],
                1,
            ),
            projected_page_key(
                PublicPageKind::IndependentRecordProjection,
                [2; 32],
                policy,
                [3; 32],
                [9; 32],
                1,
            ),
            projected_page_key(
                PublicPageKind::IndependentRecordProjection,
                [2; 32],
                policy,
                [3; 32],
                [4; 32],
                2,
            ),
        ] {
            assert_ne!(key, changed);
        }
        let MemoryKey::PublicPage(page) = key else {
            panic!("public page expected")
        };
        assert_eq!(
            page.encoding,
            Digest(asset::sha256(PALS_ENCODING_SCHEMA.as_bytes()))
        );
        assert_eq!(page.precision, PrecisionProfile::Fp32);
    }
    #[test]
    fn page_join_preserves_original_order_head_stride_and_false_padding() {
        let tokens = 69;
        let raw: Vec<_> = (0..HEADS * tokens * DIMENSION)
            .map(|index| index as f32)
            .collect();
        let board = extract_page(&raw, &raw, tokens, 0, BOARD_TOKENS).unwrap();
        let record_a = extract_page(&raw, &raw, tokens, 66, 1).unwrap();
        let record_b = extract_page(&raw, &raw, tokens, 67, 1).unwrap();
        let mut joined = JoinBuffers {
            key: vec![0.; raw.len()],
            value: vec![0.; raw.len()],
            mask: vec![true; tokens],
        };
        joined.mask[68] = false;
        copy_page(&mut joined, tokens, 0, &board).unwrap();
        copy_page(&mut joined, tokens, 66, &record_b).unwrap();
        copy_page(&mut joined, tokens, 67, &record_a).unwrap();
        copy_page(&mut joined, tokens, 68, &record_a).unwrap();
        for head in 0..HEADS {
            assert_eq!(
                &joined.key[head * tokens * 64..(head * tokens + 66) * 64],
                &raw[head * tokens * 64..(head * tokens + 66) * 64]
            );
            assert_eq!(
                &joined.key[(head * tokens + 66) * 64..(head * tokens + 67) * 64],
                &raw[(head * tokens + 67) * 64..(head * tokens + 68) * 64]
            );
            assert_eq!(
                &joined.value[(head * tokens + 68) * 64..(head * tokens + 69) * 64],
                &raw[(head * tokens + 66) * 64..(head * tokens + 67) * 64]
            );
        }
        assert!(!joined.mask[68]);
        assert!(extract_page(&raw, &raw, tokens, 68, 2).is_err());
    }
    #[test]
    fn logical_cancel_does_not_release_page_or_unfinished_join_owners() {
        let policy = HostRecordPagePolicy {
            public_graph_sha256: [7; 32],
            projection_semantics: INDEPENDENT_RECORD_PROJECTION_SEMANTICS,
            max_page_entries: 2,
            max_page_bytes: 1024 * 1024,
            max_transient_bytes: 1024 * 1024,
        };
        let page_key = MemoryKey::PublicPage(PublicPageKey {
            kind: PublicPageKind::IndependentRecordProjection,
            content: Digest([1; 32]),
            model: Digest([2; 32]),
            encoding: Digest([3; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 0,
            game_generation: 0,
        });
        let mut bank = MemoryBank::new(2, policy.max_page_bytes).unwrap();
        let pin = bank
            .insert(
                page_key,
                HostPage {
                    tokens: 1,
                    key: vec![1.; HEADS * DIMENSION].into_boxed_slice(),
                    value: vec![2.; HEADS * DIMENSION].into_boxed_slice(),
                },
                HostPage::bytes(1),
            )
            .unwrap();
        let mut pages = HostRecordPages {
            policy,
            bank,
            stats: HostRecordPageStats::default(),
            pins: vec![(page_key, pin)],
            subset: None,
            joined: None,
            join_buffers: Some(JoinBuffers {
                key: vec![0.; HEADS * 67 * DIMENSION],
                value: vec![0.; HEADS * 67 * DIMENSION],
                mask: vec![false; 67],
            }),
            joined_bytes: 0,
            transient_reservation: 100000,
        };
        // The logical cancel flag has no ownership-release operation. Actual
        // CUDA-unknown Run and backend Drop retain this same complete state.
        let canceled = AtomicBool::new(true);
        assert!(canceled.load(Ordering::Acquire));
        let unknown = pages.snapshot(true);
        assert_eq!(unknown.active_pin_count, 1);
        assert_eq!(unknown.bank.pinned_entries, 1);
        assert!(unknown.active_join_backing_bytes > 0);
        assert_eq!(unknown.transient_reservation_bytes, 100000);
        assert!(pages.clear().is_err());
        assert_eq!(pages.snapshot(true), unknown);
        // Only the known physical-completion path calls this release method.
        pages.release_completed();
        pages.clear().unwrap();
        assert_eq!(pages.snapshot(false).bank.entries, 0);
        assert_eq!(pages.snapshot(false).active_join_backing_bytes, 0);
    }
}
