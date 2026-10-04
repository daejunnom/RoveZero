//! Actual A/C/D/B with injected raw heads. No ORT, GPU or performance claims.
#![cfg(feature = "experimental-batch")]
use rz_contracts::*;
use rz_encoding::classical::HistoryFill;
use rz_eval::RawOutput;
use rz_eval::contracts::{HOST_BYTES_PER_ITEM, MaiaBinding, encoding_manifest};
use rz_eval::native_runtime_bridge::{
    NativeAdmissionPolicy, NativeRuntimeBackend, NativeWorkerOwner,
};
use rz_eval::rules_projection::ClassicalProjection;
use rz_eval::worker::SingleWorker;
use rz_runtime::contracts::{ContractEvaluator, ContractsAdapter, SharedScope};
use rz_runtime::{Clock, DrainState, Limits, Resources};
use rz_search::TreeLimits;
use rz_search::contract_time::ContractDeadlines;
use rz_search::contracts::{
    ContractSearch, ContractSearchConfig, ContractSearchStatus, IdAllocator,
};
use rz_uci::engine::{OwnerRegistry, RulesUciPort};
use rz_uci::{PositionBase, PositionPort, PositionSpec};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;
const EPOCH: ProcessEpoch = ProcessEpoch(510);
#[derive(Clone, Default)]
struct Manual(Arc<AtomicU64>);
impl Clock for Manual {
    type Tick = MonotonicTick;
    fn now(&self) -> Self::Tick {
        MonotonicTick(self.0.load(Ordering::Acquire))
    }
    fn elapsed(&self, earlier: Self::Tick, later: Self::Tick) -> Duration {
        Duration::from_nanos(later.0.saturating_sub(earlier.0))
    }
}
impl rz_runtime::contracts::ContractClock for Manual {
    fn domain(&self) -> ClockDomain {
        ClockDomain(EPOCH)
    }
}
impl rz_search::contract_time::ContractClock for Manual {
    fn domain(&self) -> ClockDomain {
        ClockDomain(EPOCH)
    }
    fn now(&self) -> Result<MonotonicTick, ContractError> {
        Ok(Clock::now(self))
    }
}

#[test]
fn actual_rules_inputs_complete_fixed_visits_with_real_multi_item_physical_batches() {
    #[cfg(feature = "experimental-raw-cache")]
    let modes = [(1, true), (4, false), (4, true)];
    #[cfg(not(feature = "experimental-raw-cache"))]
    let modes = [(1, false), (4, false)];
    for (width, cache_enabled) in modes {
        let owners = Arc::new(OwnerRegistry::default());
        let port = RulesUciPort::new(owners, rz_position::PositionLimits::default());
        let prepared = port
            .prepare(&PositionSpec {
                base: PositionBase::StartPos,
                moves: Vec::new(),
            })
            .unwrap();
        let projection = ClassicalProjection::new(
            MaiaBinding::new(
                ModelHandle {
                    owner: OwnerId(511),
                    slot: 1,
                    generation: SlotGeneration(1),
                    manifest: Digest([51; 32]),
                },
                EncodingHandle {
                    owner: OwnerId(511),
                    slot: 2,
                    generation: SlotGeneration(1),
                    manifest: encoding_manifest(HistoryFill::No),
                },
                HistoryFill::No,
                Digest([52; 32]),
                4,
            )
            .unwrap(),
        );
        #[cfg(feature = "experimental-raw-cache")]
        if cache_enabled {
            projection
                .configure_raw_cache(rz_eval::raw_cache::RawCacheLimits::default())
                .unwrap();
        }
        let invocations = Arc::new(AtomicU64::new(0));
        let largest = Arc::new(AtomicU64::new(0));
        let counted = invocations.clone();
        let largest_batch = largest.clone();
        let worker = SingleWorker::spawn(
            move |batch: &rz_eval::contracts::PreparedBatch<rz_position::contracts::RulesState>| {
                counted.fetch_add(1, Ordering::AcqRel);
                largest_batch.fetch_max(batch.requests().len() as u64, Ordering::AcqRel);
                let raw = RawOutput {
                    policy_logits: vec![0.0; rz_encoding::POLICY_SIZE],
                    wdl: vec![0.5, 0.3, 0.2],
                };
                batch
                    .requests()
                    .iter()
                    .map(|item| item.physical_output(&raw, batch.execution()))
                    .collect()
            },
        )
        .unwrap();
        let owner = NativeWorkerOwner::from_worker_batched(
            worker,
            projection.clone(),
            64,
            NativeAdmissionPolicy::Cpu,
            4,
        )
        .unwrap();
        let scope = AcceptanceScope {
            game: GameGeneration(1),
            root: RootGeneration(1),
            model: projection.model().handle(),
            encoding: projection.model().encoding().handle,
            backend: projection.backend(),
        };
        let shared = SharedScope::new(scope);
        let clock = Manual::default();
        let adapter = ContractsAdapter::new(shared.clone(), clock.clone(), width)
            .unwrap()
            .with_mixed_legal_batching();
        let backend = NativeRuntimeBackend::new(owner.clone(), clock.clone()).unwrap();
        let mut runtime = ContractEvaluator::new(
            adapter,
            backend,
            Limits {
                max_requests: width,
                max_batch_items: width,
                max_executions: 1,
                max_batch_wait: Duration::from_nanos(20),
                max_queue_age: Duration::from_nanos(10000),
                deadline_reserve: Duration::ZERO,
                memory: Resources {
                    host_bytes: 4 * (HOST_BYTES_PER_ITEM + 8192),
                    device_bytes: 0,
                    pinned_bytes: 0,
                },
            },
            128,
        )
        .unwrap();
        #[cfg(feature = "experimental-raw-cache")]
        runtime.set_raw_reuse(Box::new(projection.raw_cache_provider()));
        let deadline = |at| Deadline {
            clock: ClockDomain(EPOCH),
            at: MonotonicTick(at),
        };
        let ids = Arc::new(IdAllocator::new(EPOCH));
        let mut peaks = 0;
        let mut previous_stats = None;
        for root_number in 1..=2 {
            let scope = AcceptanceScope {
                root: RootGeneration(root_number),
                ..scope
            };
            shared.update(scope);
            let mut search = ContractSearch::new(
                prepared.snapshot.clone(),
                ContractSearchConfig {
                    scope,
                    model: projection.model().clone(),
                    precision: PrecisionProfile::Fp32,
                    compute: ComputeBudget {
                        min_steps: 1,
                        max_steps: 1,
                        require_full: true,
                    },
                    bytes: ByteBudget {
                        host: HOST_BYTES_PER_ITEM,
                        device: 0,
                        pinned: 0,
                    },
                    policy_tolerance: 1e-5,
                    wdl_tolerance: 1e-5,
                    cancellation: CancelToken::new(),
                    deadlines: ContractDeadlines {
                        soft: deadline(9000),
                        admission: deadline(9000),
                        hard: deadline(10000),
                        output: deadline(11000),
                    },
                    ids: ids.clone(),
                    max_simulations: 32,
                    tree_limits: TreeLimits {
                        max_nodes: 1024,
                        max_edges: 8192,
                        max_depth: 32,
                        ..TreeLimits::default()
                    },
                },
            )
            .unwrap();
            search.set_parallelism(width).unwrap();
            let before = invocations.load(Ordering::Acquire);
            for iteration in 0..100_000 {
                clock
                    .0
                    .fetch_max(root_number * 1000 + iteration / 100, Ordering::AcqRel);
                search.pump(
                    &mut runtime,
                    &clock,
                    || scope,
                    |position, _| {
                        projection.input_key(
                            position.state().rules(),
                            position.state().legal_moves().moves(),
                        )
                    },
                );
                peaks = peaks.max(search.pending_requests());
                if search.is_finished() {
                    break;
                }
                std::thread::yield_now();
            }
            let result = search.outcome();
            assert!(
                matches!(result.status, ContractSearchStatus::Completed),
                "{result:?}"
            );
            assert_eq!(result.counters.completed_visits, 32);
            assert_eq!(result.counters.root_initializations, 1);
            assert_eq!(result.counters.reservations_released, 33);
            assert_eq!(result.metrics.accepted_outputs, 33);
            assert_eq!(search.pending_requests(), 0);
            if width == 1 {
                if let Some(previous) = previous_stats {
                    assert_eq!(result.root_stats, previous);
                }
                previous_stats = Some(result.root_stats.clone());
            }

            assert!(
                prepared
                    .snapshot
                    .state()
                    .legal_moves()
                    .moves()
                    .contains(&result.best_move.unwrap())
            );
            if root_number == 2 && cache_enabled {
                // Require non-root cache backup as well as root initialization.
                assert!(result.metrics.accepted_raw_cache_hits > 1);
                if width == 1 {
                    assert_eq!(result.metrics.accepted_raw_cache_hits, 33);
                    assert_eq!(invocations.load(Ordering::Acquire), before);
                }
            } else {
                assert_eq!(result.metrics.accepted_raw_cache_hits, 0);
            }
            #[cfg(not(feature = "experimental-raw-cache"))]
            let _ = before;
        }
        if width == 4 {
            assert!(peaks > 1);
            assert!(largest.load(Ordering::Acquire) > 1);
            assert!(invocations.load(Ordering::Acquire) < 66);
        }
        assert!(peaks <= width);
        runtime.begin_shutdown(deadline(10000)).unwrap();
        runtime.pump();
        assert_eq!(runtime.shutdown_snapshot().drain, DrainState::Drained);
        assert_eq!(runtime.state().reserved, Resources::default());
    }
}
