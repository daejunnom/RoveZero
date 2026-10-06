//! Exact raw-head reuse, bounded independently of physical request reservations.
//! A staged result becomes reusable only after D's final acceptance. No Rules
//! history, normalized legal policy, edge statistics or device buffers are kept.
use crate::{rules_projection::ClassicalProjection, RawOutput};
use rz_contracts::*;
use rz_encoding::classical::EncodedInput;
use rz_position::contracts::RulesState;
use rz_runtime::contracts::ExactRawReuse;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct RawCacheLimits {
    pub max_entries: usize,
    pub max_bytes: usize,
}
impl Default for RawCacheLimits {
    fn default() -> Self {
        Self {
            max_entries: 64,
            max_bytes: 4 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RawCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub stage_failures: u64,
    pub entries: usize,
    pub staged: usize,
    pub retained_bytes: usize,
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct Profile {
    epoch: ProcessEpoch,
    game: GameGeneration,
    model: ModelHandle,
    encoding: EncodingHandle,
    backend: Digest,
    precision: PrecisionProfile,
    compute: ComputeBudget,
}
impl From<EvalContext> for Profile {
    fn from(c: EvalContext) -> Self {
        Self {
            epoch: c.request.epoch,
            game: c.game,
            model: c.model,
            encoding: c.encoding,
            backend: c.backend,
            precision: c.precision,
            compute: c.compute,
        }
    }
}
struct Entry {
    request: RequestId,
    profile: Profile,
    key: EvalInputKey,
    encoded: EncodedInput,
    // Immutable host heads. A hit pins the same allocation while normalization
    // runs outside the cache lock, even if clear/eviction removes the entry.
    raw: Arc<RawOutput>,
    source: ExecutionId,
    charge: usize,
}
struct Store {
    limits: RawCacheLimits,
    profile: Option<Profile>,
    entries: VecDeque<Entry>,
    staged: Vec<Entry>,
    // Only registered requests may stage results. Final rejection removes the
    // ID before a late physical completion, without an unbounded tombstone log.
    live: Vec<RequestId>,
    stats: RawCacheStats,
}
#[derive(Clone)]
pub(crate) struct RawCache(Arc<Mutex<Store>>, Arc<AtomicBool>);
impl std::fmt::Debug for RawCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawCache").finish_non_exhaustive()
    }
}
fn failed(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::BackendFailure, Stage::Output, detail)
}
impl RawCache {
    pub fn disabled() -> Self {
        Self(
            Arc::new(Mutex::new(Store {
                limits: RawCacheLimits {
                    max_entries: 0,
                    max_bytes: 0,
                },
                profile: None,
                entries: VecDeque::new(),
                staged: Vec::new(),
                live: Vec::new(),
                stats: RawCacheStats::default(),
            })),
            Arc::new(AtomicBool::new(false)),
        )
    }
    pub fn configure(&self, limits: RawCacheLimits) -> Result<(), ContractError> {
        if limits.max_entries > 1024 || limits.max_bytes > 64 * 1024 * 1024 {
            return Err(ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Contract,
                "raw cache limits exceed hard ceiling",
            ));
        }
        let mut store = self
            .0
            .lock()
            .map_err(|_| failed("raw cache owner poisoned"))?;
        store.limits = limits;
        store.profile = None;
        store.entries = VecDeque::new();
        store.staged = Vec::new();
        store.live = Vec::new();
        store.stats = RawCacheStats::default();
        self.1.store(
            limits.max_entries > 0 && limits.max_bytes > 0,
            Ordering::Release,
        );
        Ok(())
    }
    pub(crate) fn enabled(&self) -> bool {
        self.1.load(Ordering::Acquire)
    }
    pub fn clear(&self) -> Result<(), ContractError> {
        let mut store = self
            .0
            .lock()
            .map_err(|_| failed("raw cache owner poisoned"))?;
        store.profile = None;
        store.entries = VecDeque::new();
        store.staged = Vec::new();
        store.live = Vec::new();
        store.stats = RawCacheStats::default();
        Ok(())
    }
    pub fn stats(&self) -> Result<RawCacheStats, ContractError> {
        let store = self
            .0
            .lock()
            .map_err(|_| failed("raw cache owner poisoned"))?;
        Ok(RawCacheStats {
            entries: store.entries.len(),
            staged: store.staged.len(),
            ..store.stats
        })
    }
    fn lookup(
        &self,
        context: EvalContext,
        encoded: &EncodedInput,
    ) -> Result<Option<(Arc<RawOutput>, ExecutionId)>, ContractError> {
        let mut store = self
            .0
            .lock()
            .map_err(|_| failed("raw cache owner poisoned"))?;
        if store.limits.max_entries == 0 || store.limits.max_bytes == 0 {
            return Ok(None);
        }
        let profile = Profile::from(context);
        if store.profile != Some(profile) {
            store.entries.clear();
            store.staged.clear();
            store.live.clear();
            store.stats.retained_bytes = 0;
            store.profile = Some(profile);
        }
        let index = store.entries.iter().position(|entry| {
            entry.profile == profile
                && entry.key == context.input
                && exact_input(&entry.encoded, encoded)
        });
        if let Some(index) = index {
            let entry = store.entries.remove(index).expect("matched entry");
            let result = (Arc::clone(&entry.raw), entry.source);
            store.entries.push_back(entry);
            store.stats.hits = store.stats.hits.saturating_add(1);
            Ok(Some(result))
        } else {
            store.stats.misses = store.stats.misses.saturating_add(1);
            if store.live.len() < store.limits.max_entries && store.live.try_reserve(1).is_ok() {
                store.live.push(context.request);
            } else {
                store.stats.stage_failures = store.stats.stage_failures.saturating_add(1);
            }
            Ok(None)
        }
    }
    pub(crate) fn stage(
        &self,
        context: EvalContext,
        encoded: &EncodedInput,
        raw: &RawOutput,
        source: ExecutionId,
    ) {
        let Ok(mut store) = self.0.lock() else {
            return;
        };
        let profile = Profile::from(context);
        if store.profile != Some(profile)
            || store.limits.max_entries == 0
            || !store.live.contains(&context.request)
        {
            return;
        }
        if source.epoch != context.request.epoch
            || raw.policy_logits.len() != rz_encoding::POLICY_SIZE
            || raw.wdl.len() != 3
            || raw
                .policy_logits
                .iter()
                .chain(&raw.wdl)
                .any(|value| !value.is_finite())
        {
            return;
        }
        let charge = (encoded.values().len() + raw.policy_logits.len() + raw.wdl.len()) * 4
            + std::mem::size_of::<Entry>()
            // Arc allocates the RawOutput descriptor and its two reference
            // counters separately from the entry and the two Vec payloads.
            + std::mem::size_of::<RawOutput>()
            + 2 * std::mem::size_of::<usize>()
            + 1024;
        if charge > store.limits.max_bytes
            || store
                .staged
                .iter()
                .any(|entry| entry.request == context.request)
        {
            return;
        }
        while !store.entries.is_empty()
            && (store.entries.len() + store.staged.len() >= store.limits.max_entries
                || store.stats.retained_bytes > store.limits.max_bytes - charge)
        {
            let old = store.entries.pop_front().expect("nonempty entries");
            store.stats.retained_bytes -= old.charge;
            store.stats.evictions = store.stats.evictions.saturating_add(1);
        }
        if store.staged.len() >= store.limits.max_entries
            || store.stats.retained_bytes > store.limits.max_bytes - charge
        {
            store.stats.stage_failures = store.stats.stage_failures.saturating_add(1);
            return;
        }
        store.staged.push(Entry {
            request: context.request,
            profile,
            key: context.input,
            encoded: encoded.clone(),
            raw: Arc::new(raw.clone()),
            source,
            charge,
        });
        store.stats.retained_bytes += charge;
    }
    fn observe(&self, request: &EvalRequest<RulesState>, output: Option<&EvalOutput>) {
        let Ok(mut store) = self.0.lock() else {
            return;
        };
        if let Some(index) = store
            .live
            .iter()
            .position(|id| *id == request.context().request)
        {
            store.live.swap_remove(index);
        }
        let Some(index) = store
            .staged
            .iter()
            .position(|entry| entry.request == request.context().request)
        else {
            return;
        };
        let entry = store.staged.swap_remove(index);
        if store.profile == Some(entry.profile)
            && output.is_some_and(|output| {
                output.context == request.context()
                    && output.actual.provenance == CacheProvenance::Computed
                    && output.actual.execution == Some(entry.source)
            })
        {
            store.entries.push_back(entry);
        } else {
            store.stats.retained_bytes -= entry.charge;
        }
    }
}
fn exact_input(a: &EncodedInput, b: &EncodedInput) -> bool {
    a.known_frames == b.known_frames
        && a.padded_frames == b.padded_frames
        && a.inferred_ep_predecessor == b.inferred_ep_predecessor
        && a.history_fill == b.history_fill
        && a.values().len() == b.values().len()
        && a.values()
            .iter()
            .zip(b.values())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

pub struct RawCacheProvider {
    pub(crate) projection: ClassicalProjection,
    pub(crate) cache: RawCache,
}
impl ExactRawReuse<RulesState> for RawCacheProvider {
    fn lookup(
        &mut self,
        request: Arc<EvalRequest<RulesState>>,
    ) -> Result<Option<EvalOutput>, ContractError> {
        if !self.cache.enabled() {
            return Ok(None);
        }
        let prepared = self.projection.prepare(request)?;
        let Some((raw, source)) = self
            .cache
            .lookup(prepared.request().context(), prepared.encoded())?
        else {
            return Ok(None);
        };
        let mut output = prepared.reused_output(&raw, source)?;
        output.actual.execution = None;
        output.actual.provenance = CacheProvenance::RawEvalHit {
            source_execution: Some(source),
        };
        Ok(Some(output))
    }
    fn observe(&mut self, request: &EvalRequest<RulesState>, output: Option<&EvalOutput>) {
        self.cache.observe(request, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{encoding_manifest, MaiaBinding, HOST_BYTES_PER_ITEM};
    use rz_encoding::classical::HistoryFill;
    use rz_position::contracts::ContractPosition;
    use rz_position::Position;

    #[test]
    fn in_progress_hits_survive_concurrent_clear_and_release_the_last_raw_owner() {
        let epoch = ProcessEpoch(91);
        let encoding = EncodingHandle {
            owner: OwnerId(92),
            slot: 1,
            generation: SlotGeneration(1),
            manifest: encoding_manifest(HistoryFill::No),
        };
        let model = ModelHandle {
            owner: OwnerId(92),
            slot: 2,
            generation: SlotGeneration(1),
            manifest: Digest([93; 32]),
        };
        let binding =
            MaiaBinding::new(model, encoding, HistoryFill::No, Digest([94; 32]), 1).unwrap();
        let projection = ClassicalProjection::new(binding);
        projection
            .configure_raw_cache(RawCacheLimits {
                max_entries: 1,
                max_bytes: 128 * 1024,
            })
            .unwrap();
        let frozen = ContractPosition::new(OwnerId(95), Position::startpos())
            .export()
            .unwrap();
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: RequestId::new(epoch, 1),
            selection: SelectionId::new(epoch, 1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            state: frozen.snapshot().identity(),
            legal_order: frozen.legal_moves().order(),
            input: projection
                .input_key(frozen.rules(), frozen.legal_moves().moves())
                .unwrap(),
            model,
            encoding,
            precision: PrecisionProfile::Fp32,
            compute: ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            backend: projection.backend(),
        };
        let request = Arc::new(
            EvalRequest::try_new(
                context,
                frozen.snapshot().clone(),
                frozen.legal_moves().clone(),
                Arc::clone(projection.model()),
                Deadline {
                    clock: ClockDomain(epoch),
                    at: MonotonicTick(100),
                },
                CancelToken::new(),
                ByteBudget {
                    host: HOST_BYTES_PER_ITEM + 4096,
                    device: 0,
                    pinned: 0,
                },
            )
            .unwrap(),
        );
        let cache = projection.raw_cache_provider().cache;
        let prepared = projection.prepare(Arc::clone(&request)).unwrap();
        assert!(cache.lookup(context, prepared.encoded()).unwrap().is_none());
        let raw = RawOutput {
            policy_logits: vec![0.0; rz_encoding::POLICY_SIZE],
            wdl: vec![0.5, 0.3, 0.2],
        };
        let execution = ExecutionId::new(epoch, 1);
        let original = prepared.output(&raw, execution).unwrap();
        cache.observe(&request, Some(&original));
        let (first, source) = cache.lookup(context, prepared.encoded()).unwrap().unwrap();
        let (second, _) = cache.lookup(context, prepared.encoded()).unwrap().unwrap();
        let lifetime = Arc::downgrade(&first);
        assert!(Arc::ptr_eq(&first, &second));

        let clearing = cache.clone();
        std::thread::spawn(move || clearing.clear().unwrap())
            .join()
            .unwrap();
        assert_eq!(cache.stats().unwrap().retained_bytes, 0);
        assert!(cache.lookup(context, prepared.encoded()).unwrap().is_none());
        let result = prepared.reused_output(&second, source).unwrap();
        assert_eq!(result.policy, original.policy);
        assert_eq!(result.wdl, original.wdl);
        drop(first);
        assert!(lifetime.upgrade().is_some());
        drop(second);
        assert!(lifetime.upgrade().is_none());
    }
}
