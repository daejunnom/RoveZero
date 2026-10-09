//! Explicit correctness lane for actual resident CUDA records. Independent
//! PyTorch anchor outputs and freshly run whole-input ORT have different scopes.
//! Snapshots describe this owner; they do not create a fence or measured peak.
use super::*;
use rz_eval::error::BackendError;
use rz_eval::pals_device_resources::NativeCudaRecordPagesResourceInput;
use rz_eval::pals_onnx::{
    load_checked_fixed_packing_graph, CheckedFixedPackingGraph, CudaRecordPageSnapshot,
    PalsCudaRecordPageRegistration,
};
use rz_eval::worker::SingleWorker;

const FLAG: &str = "--check-cuda-record-pages";
const WINDOW_SECONDS: u64 = 240;
// The fixed sequence asserts retained hits, rather than testing a small-cache
// eviction policy. Six anchors are reset before the fifteen derived views.
const MIN_RETAINED_BLOCKS: usize = 16;
const MIN_RETAINED_BANK_BYTES: u64 = 16 * 194 * 2 * 64 * 4 * 2;
const MIN_REGISTRY_ENTRIES: usize = 512;
type Worker = SingleWorker<PalsNativeCommand, Result<PalsNativeResult, BackendError>>;

pub(super) struct Selection {
    checked_graph: CheckedFixedPackingGraph,
    resources: NativeCudaRecordPagesResourceInput,
    manifest_sha: [u8; 32],
    graph_sha: [u8; 32],
    resource_sha: [u8; 32],
}

pub(super) struct Context<'a> {
    pub fixtures: &'a Fixtures,
    pub runtime: OrtRuntime,
    pub export: &'a Path,
    pub export_sha: &'a str,
    pub policy: &'a PalsCudaControlPolicy,
    pub config: PalsOnnxConfig,
    pub report: &'a Path,
}

impl Selection {
    /// Selection is closed and validated before runtime/cache mutation. All six
    /// trailing fields are mandatory; the original eleven CUDA arguments retain
    /// their meanings. Artifact validation grants no native execution authority.
    pub(super) fn take(args: &mut Vec<String>) -> Result<Option<Self>, Box<dyn Error>> {
        let positions: Vec<_> = args
            .iter()
            .enumerate()
            .filter_map(|(index, arg)| (arg == FLAG).then_some(index))
            .collect();
        if positions.is_empty() {
            return Ok(None);
        }
        if !cfg!(target_os = "linux") {
            return Err("resident CUDA correctness is registered for Linux only".into());
        }
        if positions != [11] || args.len() != 17 || args[6] != "cuda-control-shim" {
            return Err("resident correctness requires exactly the eleven cuda-control-shim arguments followed by --check-cuda-record-pages MANIFEST SHA256 GRAPH SHA256 RESOURCES.json".into());
        }
        let manifest_sha = asset::parse_sha256(&args[13])?;
        let graph_sha = asset::parse_sha256(&args[15])?;
        let resource_path = Path::new(&args[16]);
        if !resource_path.is_absolute() {
            return Err("resident resource declaration path must be absolute".into());
        }
        let resource_bytes = asset::read_bounded(resource_path, 64 * 1024)?;
        let resources: NativeCudaRecordPagesResourceInput =
            serde_json::from_slice(&resource_bytes)?;
        if resources.limits.max_blocks < MIN_RETAINED_BLOCKS
            || resources.limits.max_bank_whole_payload_bytes < MIN_RETAINED_BANK_BYTES
            || resources.limits.max_registry_entries < MIN_REGISTRY_ENTRIES
        {
            return Err("resident retained-hit correctness sequence requires at least sixteen blocks, their fixed maximum K/V payload bank, and 512 registry entries; a smaller cache needs a separate eviction comparison".into());
        }
        let (_, _, declaration, budget) = resources.validated_parts()?;
        let checked_graph = load_checked_fixed_packing_graph(
            Path::new(&args[12]),
            Path::new(&args[14]),
            manifest_sha,
            graph_sha,
            declaration,
            budget,
        )?;
        args.truncate(11);
        Ok(Some(Self {
            checked_graph,
            resources,
            manifest_sha,
            graph_sha,
            resource_sha: asset::sha256(&resource_bytes),
        }))
    }

    pub(super) fn check(
        self,
        mut context: Context<'_>,
    ) -> Result<serde_json::Value, Box<dyn Error>> {
        let start = Instant::now();
        context.config.cache_public_memory = true;
        context.config.device_public_memory = false;
        if !matches!(context.config.provider, Provider::Cuda { device_id: 0, .. })
            || context.runtime.native_loading_profile()
                != Some(NativeLoadingProfile::CuDnnShimLazyV1)
        {
            return Err(
                "resident correctness cannot change its declared CUDA/shim conditions".into(),
            );
        }
        let load = |suffix: &str| {
            PalsOnnxBackend::load_with_cuda_control_policy(
                context.export,
                context.export_sha,
                context.runtime.clone(),
                context.config,
                context.policy.clone(),
                &context
                    .report
                    .with_extension(format!("cuda-record-{suffix}")),
            )
        };
        let mut whole = load("whole")?;
        let mut pages = load("resident")?;
        let (limits, invocation, _, _) = self.resources.validated_parts()?;
        pages.install_registered_cuda_record_pages(PalsCudaRecordPageRegistration {
            checked_graph: self.checked_graph,
            process_epoch: rz_contracts::ProcessEpoch(u64::from(std::process::id())),
            frozen_epoch: 1,
            limits,
            invocation,
            profile_root: context.report.with_extension("cuda-record-packing"),
        })?;
        let initialized = idle_snapshot(&pages)?;
        let checkpoint = asset::parse_sha256(&context.fixtures.checkpoint_sha256)?;
        if pages.model_epoch() != checkpoint
            || whole.model_epoch() != checkpoint
            || pages.is_trained() != context.fixtures.trained
            || whole.is_trained() != context.fixtures.trained
            || initialized.live_blocks != 0
            || initialized.stats.admitted_views != 0
        {
            return Err(
                "resident initialization changed fixture identity or executed a role".into(),
            );
        }
        let mut anchors = Vec::new();
        for case in &context.fixtures.cases {
            window(start)?;
            let before = idle_snapshot(&pages)?;
            let raw = pages.run(&case.input)?;
            let after = idle_snapshot(&pages)?;
            let mut row = verify(&raw, case, &context.fixtures.config)?;
            row["resident_accounting"] = accounting(&before, &after)?;
            anchors.push(row);
        }
        pages.clear_public_memory()?;
        let anchor_reset = idle_snapshot(&pages)?;
        empty_bank(&anchor_reset)?;
        let mut derived = Vec::new();
        for (name, input) in derived_record_cases(context.fixtures)? {
            window(start)?;
            whole.clear_public_memory()?;
            let expected = whole.run(&input)?;
            let case = Case {
                name: name.into(),
                input: input.clone(),
                expected,
                public_memory: whole
                    .public_memory_witness()?
                    .ok_or("whole comparator witness missing")?,
            };
            let before = idle_snapshot(&pages)?;
            window(start)?;
            let raw = pages.run(&input)?;
            let after = idle_snapshot(&pages)?;
            let mut row = verify(&raw, &case, &context.fixtures.config)?;
            row["input_key"] = json!(hex_digest(
                &input.canonical_input_key(&PalsModelConfig::baseline())?
            ));
            row["records"] = json!(input.records.len());
            row["first_accounting"] = accounting(&before, &after)?;
            let public = delta(
                after.stats.public_subset_runs_completed,
                before.stats.public_subset_runs_completed,
            )?;
            match name {
                "derived_back_to_empty_padding"
                | "derived_record_reorder"
                | "derived_critical_change"
                | "derived_id_shift"
                | "derived_eviction_and_remaining_id_shift"
                | "derived_role_order_p_to_c"
                | "derived_role_order_c_to_p"
                | "derived_role_order_p_to_c_again"
                    if public != 0 =>
                {
                    return Err(
                        "feature-identical view recomputed resident public CUDA records".into(),
                    );
                }
                "derived_actual_zero_feature_record"
                | "derived_one_record"
                | "derived_two_records_append"
                | "derived_record_correction"
                | "derived_append_after_eviction_128"
                    if public != 1 =>
                {
                    return Err(
                        "new resident feature slot did not execute one public subset graph".into(),
                    );
                }
                _ => {}
            }
            window(start)?;
            verify(&pages.run(&input)?, &case, &context.fixtures.config)?;
            let repeated = idle_snapshot(&pages)?;
            row["repeat_accounting"] = accounting(&after, &repeated)?;
            if repeated.stats.public_subset_runs_completed
                != after.stats.public_subset_runs_completed
                || repeated.stats.published_blocks != after.stats.published_blocks
            {
                return Err("repeat recomputed or republished resident representations".into());
            }
            derived.push(row);
        }
        whole.verify_runtime()?;
        let whole_stats = whole.snapshot_stats()?;
        drop(whole);
        pages.verify_runtime()?;
        let before_worker = idle_snapshot(&pages)?;
        let model_residency = pages.residency().clone();
        let mut worker = pages.controlled_worker()?;
        let reset = command(&mut worker, PalsNativeCommand::NewGame, start)?;
        if !matches!(reset, PalsNativeResult::NewGame) {
            return Err("resident NewGame returned another command".into());
        }
        let after_worker_reset = observed(&mut worker, start)?;
        empty_bank(&after_worker_reset)?;
        if after_worker_reset.game_generation != before_worker.game_generation + 1 {
            return Err(
                "resident worker reset did not advance its original game generation".into(),
            );
        }
        match command(
            &mut worker,
            PalsNativeCommand::Evaluate(context.fixtures.cases[0].input.clone()),
            start,
        )? {
            PalsNativeResult::Evaluation(raw) => {
                verify(&raw, &context.fixtures.cases[0], &context.fixtures.config)?;
            }
            _ => return Err("resident Evaluate returned another command".into()),
        }
        let final_snapshot = observed(&mut worker, start)?;
        accounting(&after_worker_reset, &final_snapshot)?;
        let final_stats = match command(&mut worker, PalsNativeCommand::SnapshotStats, start)? {
            PalsNativeResult::Stats(stats) => stats,
            _ => return Err("resident Stats returned another command".into()),
        };
        final_stats.validate()?;
        loop {
            window(start)?;
            match worker.try_shutdown() {
                Poll::Ready(result) => {
                    result?;
                    break;
                }
                Poll::Pending => std::thread::sleep(Duration::from_millis(1)),
            }
        }
        Ok(json!({
            "schema":"rz-pals-resident-cuda-record-correctness/1", "status":"passed",
            "scope":"independent_numeric_and_exclusive_physical_worker_only",
            "provider":"CUDAExecutionProvider", "precision":"fp32", "tf32":false,
            "packing_manifest_sha256":hex_digest(&self.manifest_sha),
            "packing_graph_sha256":hex_digest(&self.graph_sha),
            "resource_declaration_sha256":hex_digest(&self.resource_sha),
            "resources_are_declarations_not_measured_peaks":true,
            "retained_hit_sequence_minimum":{"blocks":MIN_RETAINED_BLOCKS,
                "bank_bytes":MIN_RETAINED_BANK_BYTES,"registry_entries":MIN_REGISTRY_ENTRIES},
            "process_epoch_scope":"numeric_process_local_pid_not_product_registry",
            "anchors_reference":"independent_pytorch_fp32_tf32_off",
            "derived_reference":"fresh_whole_input_ort_same_cuda_model",
            "anchors":anchors, "derived_cases":derived, "initialized":initialized,
            "anchor_reset":anchor_reset, "before_worker":before_worker,
            "after_worker_reset":after_worker_reset, "final_snapshot_before_shutdown":final_snapshot,
            "model_residency":model_residency, "whole_comparator_stats":stats_json(&whole_stats),
            "resident_backend_stats":stats_json(&final_stats),
            "worker_metadata_ack":"confirmed", "physical_shutdown":"confirmed",
            "logical_cancel_late_result_acceptance":"requires_product_runtime_execution",
            "public_kv_host_tensor_comparison":"not_run_device_backing_not_copied",
            "native_parameter_storage_sharing":"unknown", "vram_peak":"unknown",
            "normal_process_exit":"requires_external_exit_zero_and_cleanup",
            "training_executed":false, "window_seconds":WINDOW_SECONDS,
            "wall_seconds":start.elapsed().as_secs_f64()
        }))
    }
}

fn window(start: Instant) -> Result<(), Box<dyn Error>> {
    if start.elapsed() >= Duration::from_secs(WINDOW_SECONDS) {
        return Err("resident CUDA correctness exhausted its original finite window".into());
    }
    Ok(())
}

fn validate_idle(snapshot: &CudaRecordPageSnapshot) -> Result<(), Box<dyn Error>> {
    if snapshot.initialized_provider_count != 275
        || snapshot.packing_native_sessions != 1
        || snapshot.active_invocation
        || snapshot.physical_completion_unknown
        || snapshot.quarantined
        || snapshot.device_id != 0
        || snapshot.runtime_identity_scope != "closed_cuda_library_bundle"
    {
        return Err("resident owner has no idle initialized CUDA completion evidence".into());
    }
    Ok(())
}

fn idle_snapshot(backend: &PalsOnnxBackend) -> Result<CudaRecordPageSnapshot, Box<dyn Error>> {
    let snapshot = backend
        .cuda_record_page_snapshot()?
        .ok_or("selected resident owner missing")?;
    validate_idle(&snapshot)?;
    Ok(snapshot)
}

fn empty_bank(snapshot: &CudaRecordPageSnapshot) -> Result<(), Box<dyn Error>> {
    if snapshot.live_blocks != 0
        || snapshot.certified_projections != 0
        || snapshot.whole_owner_host_bytes != 0
        || snapshot.whole_owner_device_bytes != 0
    {
        return Err("NewGame retained resident public blocks or projections".into());
    }
    Ok(())
}

fn accounting(
    before: &CudaRecordPageSnapshot,
    after: &CudaRecordPageSnapshot,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let public = delta(
        after.stats.public_subset_runs_completed,
        before.stats.public_subset_runs_completed,
    )?;
    if public > 1
        || delta(
            after.stats.public_subset_runs_attempted,
            before.stats.public_subset_runs_attempted,
        )? != public
        || delta(
            after.stats.packing_runs_attempted,
            before.stats.packing_runs_attempted,
        )? != 1
        || delta(
            after.stats.packing_runs_completed,
            before.stats.packing_runs_completed,
        )? != 1
        || delta(
            after.stats.private_joined_runs_attempted,
            before.stats.private_joined_runs_attempted,
        )? != 1
        || delta(
            after.stats.private_joined_runs_completed,
            before.stats.private_joined_runs_completed,
        )? != 1
        || delta(after.stats.admitted_views, before.stats.admitted_views)? != 1
    {
        return Err("resident public/packing/private physical accounting differs".into());
    }
    Ok(
        json!({"public_nn_inputs_completed":public, "private_nn_inputs_completed":1,
        "auxiliary_packing_runs_completed":1, "physical_model_nn_inputs_completed":1+public,
        "snapshot_after":after, "search_consumed_evaluations":"not_applicable_numeric_only"}),
    )
}

fn command(
    worker: &mut Worker,
    input: PalsNativeCommand,
    start: Instant,
) -> Result<PalsNativeResult, Box<dyn Error>> {
    window(start)?;
    let mut lease = worker.submit(input)?;
    loop {
        window(start)?;
        match lease.poll() {
            PhysicalPoll::Ready(result) => {
                let result = result?;
                if !matches!(lease.poll(), PhysicalPoll::Consumed) {
                    return Err("resident physical result can be consumed more than once".into());
                }
                return Ok(result);
            }
            PhysicalPoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            PhysicalPoll::Quarantined => {
                return Err("resident physical completion unknown; owner retained".into())
            }
            PhysicalPoll::Consumed => {
                return Err("resident physical command completed twice".into())
            }
        }
    }
}

fn observed(worker: &mut Worker, start: Instant) -> Result<CudaRecordPageSnapshot, Box<dyn Error>> {
    match command(worker, PalsNativeCommand::SnapshotCudaRecordPages, start)? {
        PalsNativeResult::CudaRecordPagesObserved(snapshot) => {
            validate_idle(&snapshot)?;
            Ok(*snapshot)
        }
        _ => Err("resident metadata observation returned another response or no owner".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_resident_selection_preserves_legacy_arguments() {
        let mut args = vec!["cpu".to_owned(), "legacy".to_owned()];
        let original = args.clone();
        assert!(Selection::take(&mut args).unwrap().is_none());
        assert_eq!(args, original);
    }

    #[test]
    fn partial_duplicate_and_other_provider_selections_refuse_before_file_access() {
        let mut partial = vec![FLAG.to_owned()];
        assert!(Selection::take(&mut partial).is_err());
        let mut duplicate = vec![FLAG.to_owned(), FLAG.to_owned()];
        assert!(Selection::take(&mut duplicate).is_err());
        for provider in ["cpu", "cuda", "cuda-device", "cuda-control"] {
            let mut args = vec!["unopened".to_owned(); 17];
            args[6] = provider.into();
            args[11] = FLAG.into();
            assert!(Selection::take(&mut args).is_err());
            assert_eq!(args.len(), 17);
        }
    }
}
