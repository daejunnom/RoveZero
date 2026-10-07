//! Independent PyTorch reference -> frozen Rust/ORT P/C acceptance.
//! Generated fixtures and reports belong to caller-owned external roots.
use rz_eval::asset;
use rz_eval::onnx::{OrtRuntime, Provider};
use rz_eval::pals_model::{
    PalsModelConfig, PalsModelInput, PalsRawOutput, PalsRole, PALS_ENCODING_SCHEMA,
    PALS_MODEL_SCHEMA,
};
use rz_eval::pals_onnx::{
    HostRecordPagePolicy, HostRecordPageSnapshot, PalsBackendStats, PalsCudaControlPolicy,
    PalsNativeCommand, PalsNativeResult, PalsOnnxBackend, PalsOnnxConfig, PalsPublicMemoryWitness,
};
use rz_eval::runtime_pin::{CudaRuntimeBundleSpec, RuntimeBundleFileRole, RuntimeCache};
use rz_eval::worker::PhysicalPoll;
use rz_native_loader::{NativeLoadingProfile, NativeMappingObservation};
use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::task::Poll;
use std::time::{Duration, Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixtures {
    schema: String,
    checkpoint_sha256: String,
    rules_certified: bool,
    trained: bool,
    reference: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    input: PalsModelInput,
    expected: PalsRawOutput,
    public_memory: PalsPublicMemoryWitness,
}

fn validate_fixture_coverage(fixtures: &Fixtures) -> Result<(), Box<dyn Error>> {
    let required = [
        ("empty_context", PalsRole::Proposer, 0, 0, 0),
        ("critic_divergence", PalsRole::Critic, 3, 7, 4),
        ("proposer_order", PalsRole::Proposer, 3, 7, 0),
        ("proposer_order_reversed", PalsRole::Proposer, 3, 7, 0),
        ("same_board_other_history", PalsRole::Proposer, 3, 7, 0),
        ("all_promotions", PalsRole::Critic, 1, 4, 1),
    ];
    if fixtures.schema != PALS_MODEL_SCHEMA
        || fixtures.rules_certified
        || fixtures.reference != "pytorch_fp32_tf32_off"
        || fixtures.cases.len() != required.len()
    {
        return Err("expected registered six-case P/C numeric-only reference fixtures".into());
    }
    for (name, role, records, candidates, divergences) in required {
        let matching: Vec<_> = fixtures
            .cases
            .iter()
            .filter(|case| case.name == name)
            .collect();
        if matching.len() != 1 {
            return Err("P/C numeric reference case is missing or duplicated".into());
        }
        let input = &matching[0].input;
        input.validate(&PalsModelConfig::baseline())?;
        if input.role != role
            || input.records.len() != records
            || input.candidates.len() != candidates
            || input.divergence_features.len() != divergences
        {
            return Err("P/C numeric reference case does not cover its declared role/shape".into());
        }
        let witness = &matching[0].public_memory;
        let tokens = PalsModelConfig::baseline().public_memory_tokens(records)?;
        if witness.tokens != tokens
            || witness.memory_key.len() != 2 * tokens * 64
            || witness.memory_value.len() != 2 * tokens * 64
            || witness.memory_mask.len() != tokens
            || witness
                .memory_key
                .iter()
                .chain(&witness.memory_value)
                .any(|value| !value.is_finite())
            || witness.memory_mask[..66].iter().any(|value| !*value)
            || witness.memory_mask[66..]
                .iter()
                .filter(|value| **value)
                .count()
                != records
        {
            return Err("public-memory reference witness shape/finite/mask differs".into());
        }
    }
    let case = |name: &str| {
        fixtures
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("coverage verified")
    };
    let ordered = &case("proposer_order").input;
    let mut reversed = ordered.clone();
    reversed.candidates.reverse();
    if reversed != case("proposer_order_reversed").input {
        return Err("candidate order comparison changed another input condition".into());
    }
    let history = &case("same_board_other_history").input;
    let mut other_history = ordered.clone();
    other_history.history_digest = history.history_digest;
    if ordered.history_digest == history.history_digest || other_history != *history {
        return Err("history comparison did not preserve the same board/other inputs".into());
    }
    let promotions = &case("all_promotions").input.candidates;
    if !promotions.iter().enumerate().all(|(index, candidate)| {
        candidate.from == 48 && candidate.to == 56 && candidate.promotion == index as u8 + 1
    }) {
        return Err("promotion reference does not cover queen/rook/bishop/knight order".into());
    }
    Ok(())
}

fn compare(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> Result<f64, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err("reference tensor shape differs".into());
    }
    let mut maximum: f64 = 0.;
    for (a, e) in actual.iter().zip(expected) {
        let delta = (f64::from(*a) - f64::from(*e)).abs();
        if !a.is_finite() || !e.is_finite() || delta > atol + rtol * f64::from(*e).abs() {
            return Err(
                format!("PALS numeric mismatch: actual={a}, expected={e}, abs={delta}").into(),
            );
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}
fn verify(raw: &PalsRawOutput, case: &Case) -> Result<serde_json::Value, Box<dyn Error>> {
    let config = PalsModelConfig::baseline();
    let actual = raw.decode(&case.input, &config)?;
    let expected = case.expected.decode(&case.input, &config)?;
    let candidate = compare(
        &raw.candidate_logits,
        &case.expected.candidate_logits,
        1e-4,
        1e-3,
    )?;
    let wdl_logits = compare(&raw.wdl_logits, &case.expected.wdl_logits, 1e-4, 1e-3)?;
    let latent = compare(
        &raw.private_latent,
        &case.expected.private_latent,
        1e-4,
        1e-3,
    )?;
    let policy = compare(
        &actual.candidate_policy,
        &expected.candidate_policy,
        1e-4,
        0.,
    )?;
    let wdl = compare(&actual.wdl, &expected.wdl, 1e-4, 0.)?;
    let divergence = match (&raw.divergence_logits, &case.expected.divergence_logits) {
        (Some(a), Some(e)) => Some(compare(a, e, 1e-4, 1e-3)?),
        (None, None) => None,
        _ => return Err("private critic head differs".into()),
    };
    Ok(
        json!({"case":case.name,"candidate_logit_max_abs":candidate,"wdl_logit_max_abs":wdl_logits,
        "latent_max_abs":latent,"policy_max_abs":policy,"wdl_max_abs":wdl,"divergence_logit_max_abs":divergence}),
    )
}
fn verify_public(
    backend: &PalsOnnxBackend,
    case: &Case,
) -> Result<serde_json::Value, Box<dyn Error>> {
    match backend.public_memory_witness()? {
        Some(actual) => {
            if actual.tokens != case.public_memory.tokens
                || actual.memory_mask != case.public_memory.memory_mask
            {
                return Err("independent public-memory shape/mask mismatch".into());
            }
            let key = compare(
                &actual.memory_key,
                &case.public_memory.memory_key,
                1e-4,
                1e-3,
            )?;
            let value = compare(
                &actual.memory_value,
                &case.public_memory.memory_value,
                1e-4,
                1e-3,
            )?;
            Ok(
                json!({"status":"passed","key_max_abs":key,"value_max_abs":value,"mask":"exact","source":"completed_host_cache"}),
            )
        }
        None if backend.config().device_public_memory => Ok(
            json!({"status":"not_run","cause":"device_KV_host_witness_unavailable","key_max_abs":null,"value_max_abs":null}),
        ),
        None => Err("completed public-memory cache witness missing".into()),
    }
}
fn outside_git(path: &Path) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("external artifact path must be absolute".into());
    }
    for ancestor in path.canonicalize()?.ancestors() {
        if ancestor.join(".git").try_exists()? {
            return Err("generated artifacts must be outside Git".into());
        }
    }
    Ok(())
}
fn hex_digest(value: &[u8; 32]) -> String {
    value.iter().map(|v| format!("{v:02x}")).collect()
}
fn stats_json(stats: &PalsBackendStats) -> serde_json::Value {
    json!({"admitted_role_requests":stats.admitted_role_requests,
        "public_cache_hits":stats.public_cache_hits,"public_cache_misses":stats.public_cache_misses,
        "public_nn_runs_attempted":stats.public_nn_runs_attempted,"public_nn_runs_completed":stats.public_nn_runs_completed,
        "public_nn_runs_failed_known":stats.public_nn_runs_failed_known,
        "role_nn_runs_attempted":stats.role_nn_runs_attempted,"role_nn_runs_completed":stats.role_nn_runs_completed,
        "role_nn_runs_failed_known":stats.role_nn_runs_failed_known,"completed_nn_inputs":stats.completed_nn_inputs,
        "validated_public_outputs":stats.validated_public_outputs,"validated_role_outputs":stats.validated_role_outputs,
        "new_game_resets":stats.new_game_resets,"live_public_cache_entries":stats.live_public_cache_entries})
}

const HOST_PAGE_CHECK_FLAG: &str = "--check-host-record-pages";
const HOST_PAGE_CHECK_SECONDS: u64 = 300;

fn page_snapshot_json(snapshot: &HostRecordPageSnapshot) -> serde_json::Value {
    let stats = &snapshot.stats;
    json!({"policy":{"public_graph_sha256":hex_digest(&snapshot.policy.public_graph_sha256),
        "projection_semantics":snapshot.policy.projection_semantics,
        "max_page_entries":snapshot.policy.max_page_entries,
        "max_page_bytes":snapshot.policy.max_page_bytes,
        "max_transient_bytes":snapshot.policy.max_transient_bytes},
        "bank":{"entries":snapshot.bank.entries,"reserved_bytes":snapshot.bank.reserved_bytes,
            "entry_backing_bytes":snapshot.bank.entry_backing_bytes,
            "max_entries":snapshot.bank.max_entries,"max_value_bytes":snapshot.bank.max_bytes,
            "pinned_entries":snapshot.bank.pinned_entries,"pinned_bytes":snapshot.bank.pinned_bytes},
        "actual_page_counters":{"view_hits":stats.view_hits,"view_misses":stats.view_misses,
            "board_hits":stats.board_hits,"board_misses":stats.board_misses,
            "record_hits":stats.record_hits,"record_misses":stats.record_misses,
            "public_calls_attempted":stats.public_calls_attempted,
            "public_calls_completed":stats.public_calls_completed,
            "submitted_record_tokens":stats.submitted_record_tokens,
            "encoded_record_tokens":stats.encoded_record_tokens,
            "contextual_board_encodes_completed":stats.contextual_board_encodes_completed,
            "joins_completed":stats.joins_completed,"joined_bytes":stats.joined_bytes,
            "evicted_pages":stats.evicted_pages,
            "max_reserved_host_transient_bytes":stats.max_reserved_host_transient_bytes},
        "active_pin_count":snapshot.active_pin_count,"retained_join_bytes":snapshot.retained_join_bytes,
        "active_subset_bytes":snapshot.active_subset_bytes,
        "active_join_backing_bytes":snapshot.active_join_backing_bytes,
        "active_full_input_bytes":snapshot.active_full_input_bytes,
        "transient_reservation_bytes":snapshot.transient_reservation_bytes,
        "quarantined":snapshot.quarantined,
        "scope":"declared_and_actual_host_backing_not_native_allocator_or_vram_peak"})
}

fn completed_page_snapshot(
    backend: &PalsOnnxBackend,
    joined: bool,
) -> Result<HostRecordPageSnapshot, Box<dyn Error>> {
    let snapshot = backend
        .host_record_page_snapshot()
        .ok_or("host page mode is absent")?;
    if snapshot.quarantined
        || snapshot.active_pin_count != 0
        || snapshot.active_subset_bytes != 0
        || snapshot.active_join_backing_bytes != 0
        || snapshot.active_full_input_bytes != 0
        || snapshot.transient_reservation_bytes != 0
        || snapshot.bank.pinned_entries != 0
        || snapshot.bank.pinned_bytes != 0
        || snapshot.bank.entries > snapshot.bank.max_entries
        || snapshot.bank.reserved_bytes > snapshot.bank.max_bytes
        || snapshot
            .bank
            .reserved_bytes
            .checked_add(snapshot.bank.entry_backing_bytes)
            .is_none_or(|bytes| bytes > snapshot.policy.max_page_bytes)
        || snapshot.stats.max_reserved_host_transient_bytes > snapshot.policy.max_transient_bytes
        || (snapshot.retained_join_bytes != 0) != joined
    {
        return Err("host record page completion, pin release or backing budget differs".into());
    }
    let native = backend.snapshot_stats()?;
    if native.public_cache_hits != 0
        || native.public_cache_misses != 0
        || native.live_public_cache_entries != 0
    {
        return Err(
            "host record page counters leaked into the legacy whole-input cache counters".into(),
        );
    }
    Ok(snapshot)
}

fn numeric_window(start: Instant) -> Result<(), Box<dyn Error>> {
    if start.elapsed() >= Duration::from_secs(HOST_PAGE_CHECK_SECONDS) {
        return Err("finite CPU host-page correctness window exhausted before next Run".into());
    }
    Ok(())
}

fn reseal_numeric_input(input: &mut PalsModelInput) -> Result<(), Box<dyn Error>> {
    // Derived inputs are model-tensor cases, not fabricated Rules positions,
    // CPU observations or training targets. Retain the reference history/epoch.
    input.required_critical_records = input
        .records
        .iter()
        .filter(|record| record.critical)
        .map(|record| record.record_id)
        .collect();
    let config = PalsModelConfig::baseline();
    input.validate(&config)?;
    let prepared = input.prepare_tensors(&config)?;
    if prepared.input_key != input.canonical_input_key(&config)?
        || prepared.public_memory_key != input.public_memory_key(&config)?
    {
        return Err("derived full canonical/encoding identity did not reseal".into());
    }
    Ok(())
}

fn derived_record_cases(
    fixtures: &Fixtures,
) -> Result<Vec<(&'static str, PalsModelInput)>, Box<dyn Error>> {
    let case = |name: &str| {
        fixtures
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("six-case coverage verified")
            .input
            .clone()
    };
    let base = case("proposer_order");
    let critic = case("critic_divergence");
    let seed = base.records[0].clone();
    let mut empty = base.clone();
    empty.records.clear();
    let mut zero = empty.clone();
    let mut zero_record = seed.clone();
    zero_record.record_id = 1;
    zero_record.revision = 1;
    zero_record.critical = false;
    zero_record.features = [0.; 16];
    zero.records.push(zero_record);
    let mut one = base.clone();
    one.records.truncate(1);
    one.records[0].critical = false;
    one.records[0].features[15] = 0.03125;
    let mut two = base.clone();
    two.records.truncate(2);
    for record in &mut two.records {
        record.critical = false;
    }
    two.records[0] = one.records[0].clone();
    two.records[1].features[15] = 0.09375;
    let mut corrected = two.clone();
    corrected.records[0].features[3] += 0.125;
    corrected.records[0].features[15] = 0.1875;
    corrected.records[0].revision = corrected.records[0]
        .revision
        .checked_add(1)
        .ok_or("derived record revision overflow")?;
    let mut reordered = corrected.clone();
    reordered.records.reverse();
    let mut critical = reordered.clone();
    critical.records[0].critical = true;
    let mut shifted = critical.clone();
    for (index, record) in shifted.records.iter_mut().enumerate() {
        record.record_id = index as u64 + 40;
    }
    let mut many = base.clone();
    many.records = (0..128)
        .map(|index| {
            let mut record = seed.clone();
            record.record_id = index + 1;
            record.revision = index + 1;
            record.critical = false;
            record.features[15] = index as f32 / 128.;
            record
        })
        .collect();
    let mut evicted = many.clone();
    evicted.records.remove(0);
    for (index, record) in evicted.records.iter_mut().enumerate() {
        record.record_id = index as u64 + 1;
    }
    let mut appended = evicted.clone();
    let mut added = seed;
    added.record_id = 128;
    added.revision = 200;
    added.critical = false;
    added.features[15] = 2.;
    appended.records.push(added);
    let mut critic_view = appended.clone();
    critic_view.role = PalsRole::Critic;
    critic_view.divergence_features = critic.divergence_features;
    critic_view.query = critic.query;
    let mut cases = vec![
        ("derived_empty_false_padding", empty.clone()),
        ("derived_actual_zero_feature_record", zero.clone()),
        ("derived_back_to_empty_padding", empty),
        ("derived_one_record", one),
        ("derived_two_records_append", two),
        ("derived_record_correction", corrected),
        ("derived_record_reorder", reordered),
        ("derived_critical_change", critical),
        ("derived_id_shift", shifted),
        ("derived_128_records", many),
        ("derived_eviction_and_remaining_id_shift", evicted),
        ("derived_append_after_eviction_128", appended.clone()),
        ("derived_role_order_p_to_c", critic_view.clone()),
        ("derived_role_order_c_to_p", appended),
        ("derived_role_order_p_to_c_again", critic_view),
    ];
    for (_, input) in &mut cases {
        reseal_numeric_input(input)?;
    }
    if cases.len() > 20 {
        return Err("derived correctness case cap exceeded".into());
    }
    Ok(cases)
}

fn delta(after: u64, before: u64) -> Result<u64, Box<dyn Error>> {
    after
        .checked_sub(before)
        .ok_or_else(|| "native/page counter moved backwards".into())
}

fn verify_page_run_accounting(
    before: &PalsBackendStats,
    after: &PalsBackendStats,
    page_before: &HostRecordPageSnapshot,
    page_after: &HostRecordPageSnapshot,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let public = delta(
        after.public_nn_runs_completed,
        before.public_nn_runs_completed,
    )?;
    let misses = delta(
        page_after.stats.record_misses,
        page_before.stats.record_misses,
    )?;
    let encoded = delta(
        page_after.stats.encoded_record_tokens,
        page_before.stats.encoded_record_tokens,
    )?;
    if public > 1
        || delta(after.admitted_role_requests, before.admitted_role_requests)? != 1
        || delta(after.role_nn_runs_attempted, before.role_nn_runs_attempted)? != 1
        || delta(after.role_nn_runs_completed, before.role_nn_runs_completed)? != 1
        || delta(
            after.public_nn_runs_attempted,
            before.public_nn_runs_attempted,
        )? != public
        || delta(after.completed_nn_inputs, before.completed_nn_inputs)? != 1 + public
        || delta(
            page_after.stats.public_calls_attempted,
            page_before.stats.public_calls_attempted,
        )? != public
        || delta(
            page_after.stats.public_calls_completed,
            page_before.stats.public_calls_completed,
        )? != public
        || delta(
            page_after.stats.contextual_board_encodes_completed,
            page_before.stats.contextual_board_encodes_completed,
        )? != public
        || delta(
            page_after.stats.submitted_record_tokens,
            page_before.stats.submitted_record_tokens,
        )? != encoded
        || encoded != if public == 0 { 0 } else { misses.max(1) }
        || delta(
            page_after.stats.joins_completed,
            page_before.stats.joins_completed,
        )? != 1
    {
        return Err("derived host page run changed physical NN/token/join accounting".into());
    }
    Ok(
        json!({"public_calls_completed":public,"physical_nn_inputs_completed":1+public,
        "unique_record_misses":misses,"encoded_record_slots":encoded,
        "board_recomputed_by_existing_combined_graph":public!=0}),
    )
}

fn check_host_record_pages(
    fixtures: &Fixtures,
    runtime: OrtRuntime,
    export: &Path,
    export_sha: &str,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let start = Instant::now();
    let config = PalsOnnxConfig::cpu();
    let mut whole = PalsOnnxBackend::load(export, export_sha, runtime.clone(), config)?;
    let mut pages = PalsOnnxBackend::load(export, export_sha, runtime, config)?;
    let graph_sha = pages
        .residency()
        .graphs
        .iter()
        .find(|graph| graph.role == "public")
        .ok_or("registered public graph identity missing")?
        .sha256;
    let policy = HostRecordPagePolicy::for_registered_graph(graph_sha);
    pages.enable_host_record_pages(policy)?;
    let checkpoint = asset::parse_sha256(&fixtures.checkpoint_sha256)?;
    if !matches!(whole.config().provider, Provider::Cpu)
        || !matches!(pages.config().provider, Provider::Cpu)
        || whole.model_epoch() != checkpoint
        || pages.model_epoch() != checkpoint
        || whole.is_trained() != fixtures.trained
        || pages.is_trained() != fixtures.trained
    {
        return Err(
            "host record page correctness mode changed CPU/checkpoint/provenance conditions".into(),
        );
    }
    let mut anchors = Vec::new();
    for case in &fixtures.cases {
        numeric_window(start)?;
        let raw = pages.run(&case.input)?;
        let mut row = verify(&raw, case)?;
        row["public_memory"] = verify_public(&pages, case)?;
        row["page_ownership"] = page_snapshot_json(&completed_page_snapshot(&pages, true)?);
        anchors.push(row);
    }
    // These six anchor references remain independent PyTorch evidence. Derived
    // cases below use freshly run whole-input ORT as the explicit comparator.
    pages.clear_public_memory()?;
    let anchor_clear = completed_page_snapshot(&pages, false)?;
    if anchor_clear.bank.entries != 0 || anchor_clear.bank.reserved_bytes != 0 {
        return Err("page anchor reset did not retire its representations".into());
    }
    let derived = derived_record_cases(fixtures)?;
    let mut rows = Vec::new();
    let mut first_case = None;
    for (name, input) in &derived {
        numeric_window(start)?;
        whole.clear_public_memory()?;
        let whole_before = whole.snapshot_stats()?;
        let expected = whole.run(input)?;
        let witness = whole
            .public_memory_witness()?
            .ok_or("whole comparator witness missing")?;
        let case = Case {
            name: (*name).into(),
            input: input.clone(),
            expected,
            public_memory: witness,
        };
        let page_before = completed_page_snapshot(&pages, !rows.is_empty())?;
        let native_before = pages.snapshot_stats()?;
        numeric_window(start)?;
        let actual = pages.run(input)?;
        let native_after = pages.snapshot_stats()?;
        let page_after = completed_page_snapshot(&pages, true)?;
        let mut row = verify(&actual, &case)?;
        row["public_memory"] = verify_public(&pages, &case)?;
        row["input_key"] = json!(hex_digest(
            &input.canonical_input_key(&PalsModelConfig::baseline())?
        ));
        row["full_public_key"] = json!(hex_digest(
            &input.public_memory_key(&PalsModelConfig::baseline())?
        ));
        row["records"] = json!(input.records.len());
        row["role"] = json!(input.role);
        row["first_run_accounting"] =
            verify_page_run_accounting(&native_before, &native_after, &page_before, &page_after)?;
        let public_delta = delta(
            native_after.public_nn_runs_completed,
            native_before.public_nn_runs_completed,
        )?;
        let record_misses = delta(
            page_after.stats.record_misses,
            page_before.stats.record_misses,
        )?;
        // These controlled mutations must exercise actual page reuse/subset
        // execution, rather than merely compare another full re-encode.
        match *name {
            "derived_actual_zero_feature_record"
            | "derived_back_to_empty_padding"
            | "derived_record_reorder"
            | "derived_critical_change"
            | "derived_id_shift"
            | "derived_eviction_and_remaining_id_shift"
            | "derived_role_order_p_to_c"
            | "derived_role_order_c_to_p"
            | "derived_role_order_p_to_c_again"
                if public_delta != 0 || record_misses != 0 =>
            {
                return Err(
                    "feature-identical derived view failed to reuse existing public pages".into(),
                );
            }
            "derived_one_record"
            | "derived_two_records_append"
            | "derived_record_correction"
            | "derived_append_after_eviction_128"
                if public_delta != 1 || record_misses != 1 =>
            {
                return Err(
                    "single-record derived delta did not execute exactly one missing feature slot"
                        .into(),
                );
            }
            _ => {}
        }
        row["page_after_first"] = page_snapshot_json(&page_after);
        // Each repeat is a fresh private role invocation over exact reused K/V.
        numeric_window(start)?;
        let repeated = pages.run(input)?;
        let repeat_native = pages.snapshot_stats()?;
        let repeat_pages = completed_page_snapshot(&pages, true)?;
        verify(&repeated, &case)?;
        let repeat_counts =
            verify_page_run_accounting(&native_after, &repeat_native, &page_after, &repeat_pages)?;
        if repeat_native.public_nn_runs_completed != native_after.public_nn_runs_completed
            || repeat_pages.stats.record_misses != page_after.stats.record_misses
            || repeat_pages.stats.board_misses != page_after.stats.board_misses
            || repeat_pages.stats.view_hits != page_after.stats.view_hits + 1
        {
            return Err("repeated host page view unexpectedly recomputed a representation".into());
        }
        row["repeated_page_accounting"] = repeat_counts;
        let whole_after = whole.snapshot_stats()?;
        if whole_after.public_nn_runs_completed != whole_before.public_nn_runs_completed + 1
            || whole_after.completed_nn_inputs != whole_before.completed_nn_inputs + 2
        {
            return Err("fresh whole comparator did not physically execute both graphs".into());
        }
        numeric_window(start)?;
        verify(&whole.run(input)?, &case)?;
        let whole_repeat = whole.snapshot_stats()?;
        let whole_ownership = whole.host_public_page_snapshot();
        if whole_repeat.public_nn_runs_completed != whole_after.public_nn_runs_completed
            || whole_repeat.completed_nn_inputs != whole_after.completed_nn_inputs + 1
            || whole_repeat.public_cache_hits != whole_after.public_cache_hits + 1
            || whole_ownership.pinned_entries != 0
            || whole_ownership.pinned_bytes != 0
        {
            return Err("repeated whole comparator NN/cache/pin accounting differs".into());
        }
        row["whole_after_repeat"] = stats_json(&whole_repeat);
        if first_case.is_none() {
            first_case = Some(case);
        }
        rows.push(row);
    }
    let page_before_clear = completed_page_snapshot(&pages, true)?;
    let native_before_clear = pages.snapshot_stats()?;
    pages.clear_public_memory()?;
    let page_after_clear = completed_page_snapshot(&pages, false)?;
    let native_after_clear = pages.snapshot_stats()?;
    if page_after_clear.bank.entries != 0
        || page_after_clear.bank.reserved_bytes != 0
        || pages.public_memory_witness()?.is_some()
        || native_after_clear.completed_nn_inputs != native_before_clear.completed_nn_inputs
        || native_after_clear.new_game_resets != native_before_clear.new_game_resets + 1
    {
        return Err("host page clear did not retire representations without NN work".into());
    }
    numeric_window(start)?;
    let first = first_case.ok_or("derived correctness cases are empty")?;
    let fresh_after_clear = pages.run(&first.input)?;
    let after_fresh_native = pages.snapshot_stats()?;
    let after_fresh_pages = completed_page_snapshot(&pages, true)?;
    verify(&fresh_after_clear, &first)?;
    verify_public(&pages, &first)?;
    let fresh_counts = verify_page_run_accounting(
        &native_after_clear,
        &after_fresh_native,
        &page_after_clear,
        &after_fresh_pages,
    )?;
    if after_fresh_native.public_nn_runs_completed
        != native_after_clear.public_nn_runs_completed + 1
    {
        return Err("game clear reused an old generation's host page".into());
    }
    whole.verify_runtime()?;
    pages.verify_runtime()?;
    let whole_final = whole.snapshot_stats()?;
    drop(whole);
    drop(pages);
    Ok(
        json!({"schema":"rovezero.pals-host-record-pages-correctness.v1","status":"passed",
        "scope":"model_tensor_numeric_only_no_rules_or_training_certification_no_performance_claim",
        "provider":"CPUExecutionProvider","precision":"fp32","tf32":false,
        "max_window_seconds":HOST_PAGE_CHECK_SECONDS,"native_deadline_scope":"before_each_synchronous_Run_not_forced_cancellation",
        "encoding_schema":PALS_ENCODING_SCHEMA,"manifest_sha256":export_sha,"checkpoint_sha256":fixtures.checkpoint_sha256,
        "independent_reference_anchor":{"source":"original_registered_six_case_pytorch_fp32_tf32_off",
            "whole_results":"top_level_cases_unchanged","page_cases":anchors},
        "derived_whole_page_equivalence":{"source":"fresh_whole_input_ORT_same_pinned_export_and_runtime",
            "input_source":"deterministic_derivatives_of_registered_fixtures","cases":rows},
        "tolerance":{"raw_atol":1e-4,"raw_rtol":1e-3,"policy_wdl_max_abs":1e-4,"mask":"exact"},
        "anchor_clear":page_snapshot_json(&anchor_clear),
        "final_clear":{"before":page_snapshot_json(&page_before_clear),"after":page_snapshot_json(&page_after_clear),
            "neural_runs":"zero","fresh_after_clear":fresh_counts},
        "final_page_stats":page_snapshot_json(&after_fresh_pages),"final_native_stats":stats_json(&after_fresh_native),
        "final_whole_stats":stats_json(&whole_final),
        "native_owner_drop":"returned_after_completed_CPU_runs","process_exit":"requires_external_process_exit_and_cleanup",
        "native_allocator_peak":"unknown","vram_peak":"unknown"}),
    )
}

fn mapping_json(phase: &str, observed: &NativeMappingObservation) -> serde_json::Value {
    json!({"phase":phase,"scope":"proc_self_maps_at_audit_boundary",
        "loading_profile":observed.profile.identifier(),
        "declared_nvidia_files":observed.declared_nvidia_files,
        "required_nvidia_files":observed.required_nvidia_files,
        "mapped_nvidia_files":observed.mapped_nvidia_files,
        "mapped_nvidia_count":observed.mapped_nvidia_files.len(),
        "deferred_nvidia_not_mapped":observed.deferred_nvidia_not_mapped,
        "deferred_absence_meaning":"not_mapped_at_observation_not_unused",
        "mapped_ort_files":observed.mapped_ort_files,
        "kernel_placement":"separate_cuda_placement_witness",
        "vram_residency":"unknown","normal_process_exit":"requires_external_supervisor"})
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let check_record_pages = args.last().is_some_and(|arg| arg == HOST_PAGE_CHECK_FLAG);
    if check_record_pages {
        args.pop();
        if args.len() != 8 || args[6] != "cpu" {
            return Err("--check-host-record-pages is an explicit CPU-only optional flag after the existing eight CPU arguments".into());
        }
    }
    if args.len() == 3 && args[0] == "--check-cuda-control-policy" {
        // Pure bounded source-policy parsing: no runtime load, session, Run or
        // provider claim. The external owner preserves stdout as evidence.
        PalsCudaControlPolicy::from_inventory(Path::new(&args[1]), &args[2])?;
        println!("Pinned static PALS CUDA control v2 accepted; explicit graph optimization=disable; native placement and GPU execution not observed");
        return Ok(());
    }
    if !matches!(args.len(), 8 | 9 | 11) {
        return Err("usage: pals_model_check EXPORT.json EXPORT_SHA256 ORT_LIBRARY ORT_SHA256 FIXTURES.json REPORT.json cpu|cuda|cuda-device|cuda-control|cuda-control-shim CACHE_ROOT [CUDA_BUNDLE.json] [INVENTORY_V2.json INVENTORY_SHA256]; CPU-only optional trailing --check-host-record-pages".into());
    }
    let is_cuda = match (args[6].as_str(), args.len()) {
        ("cpu", 8) => false,
        ("cuda" | "cuda-device", 9) => true,
        ("cuda-control" | "cuda-control-shim", 11) => true,
        _ => return Err("CUDA requires its explicit pinned bundle; CPU excludes it".into()),
    };
    let report = Path::new(&args[5]);
    outside_git(report.parent().ok_or("report has no existing parent")?)?;
    if report.exists() {
        return Err("refusing to overwrite registered PALS acceptance report".into());
    }
    let fixture_path = Path::new(&args[4]);
    if !fixture_path.is_absolute() {
        return Err("fixture must be absolute".into());
    }
    let fixture_bytes = asset::read_bounded(fixture_path, 16 * 1024 * 1024)?;
    let fixture_digest = asset::sha256(&fixture_bytes);
    let fixtures: Fixtures = serde_json::from_slice(&fixture_bytes)?;
    validate_fixture_coverage(&fixtures)?;
    let cache = RuntimeCache::open(Path::new(&args[7]))?;
    let pin = if is_cuda {
        let spec_bytes = asset::read_bounded(Path::new(&args[8]), 64 * 1024)?;
        let spec = CudaRuntimeBundleSpec::from_json(std::str::from_utf8(&spec_bytes)?)?;
        let core = spec
            .files
            .iter()
            .find(|f| f.role == RuntimeBundleFileRole::Core)
            .ok_or("bundle core absent")?;
        let path = Path::new(&args[2]);
        if path.file_name().and_then(|n| n.to_str()) != Some(core.filename.as_str())
            || args[3] != core.sha256
        {
            return Err("ORT declaration differs from explicit CUDA bundle".into());
        }
        cache.cuda_bundle(path.parent().ok_or("runtime core has no parent")?, &spec)?
    } else {
        cache.library(Path::new(&args[2]), &args[3])?
    };
    // Shim lazy is an explicitly registered diagnostic variable. Other modes,
    // including the existing cuda-control mode, retain eager16 bootstrap.
    let runtime = (if args[6] == "cuda-control-shim" {
        OrtRuntime::load_with_cuda_loading_profile(&pin, NativeLoadingProfile::CuDnnShimLazyV1)
    } else {
        OrtRuntime::load(&pin)
    })
    .map_err(|mut failure| {
        if is_cuda {
            failure.detail = "PALS CUDA runtime bootstrap failed before session creation";
        }
        failure
    })?;
    let runtime_digest = runtime.binary_digest();
    let runtime_bundle_digest = runtime.bundle_digest();
    let loading_profile = runtime.native_loading_profile();
    let loading_profile_digest = runtime.native_loading_profile_digest();
    let mut loading_observations = Vec::new();
    if is_cuda {
        loading_observations.push(mapping_json(
            "dependencies_before_sessions",
            &runtime.cuda_mapping_observation(false)?,
        ));
    }
    let control_policy = if matches!(args[6].as_str(), "cuda-control" | "cuda-control-shim") {
        Some(PalsCudaControlPolicy::from_inventory(
            Path::new(&args[9]),
            &args[10],
        )?)
    } else {
        None
    };
    let mut config = PalsOnnxConfig::cpu();
    if is_cuda {
        config.provider = Provider::Cuda {
            device_id: 0,
            arena_bytes: 2 * 1024 * 1024 * 1024,
        };
        config.device_public_memory = args[6] == "cuda-device";
    }
    let load = |runtime, config, suffix: &str| {
        if let Some(policy) = &control_policy {
            PalsOnnxBackend::load_with_cuda_control_policy(
                Path::new(&args[0]),
                &args[1],
                runtime,
                config,
                policy.clone(),
                &report.with_extension(format!("cuda-control-{suffix}")),
            )
        } else {
            PalsOnnxBackend::load(Path::new(&args[0]), &args[1], runtime, config)
        }
    };
    let mut backend = load(runtime.clone(), config, "primary")?;
    let graph_optimization = backend.graph_optimization();
    let epoch = backend.model_epoch();
    if epoch != asset::parse_sha256(&fixtures.checkpoint_sha256)?
        || backend.is_trained() != fixtures.trained
    {
        return Err("reference weight epoch differs".into());
    }
    for case in &fixtures.cases {
        if case.name.is_empty() || case.name.len() > 128 || case.input.model_epoch != epoch {
            return Err("fixture identity differs".into());
        }
        case.input.validate(&PalsModelConfig::baseline())?;
    }
    // CUDA tests are bounded by the shell/process owner in addition to these
    // logical deadlines. This check never claims a deadline stopped native Run.
    let began = Instant::now();
    let mut reports = Vec::new();
    let mut raw_outputs = Vec::new();
    let mut proposer_seen = false;
    let mut critic_seen = false;
    let mut placement_witness = None;
    for case in &fixtures.cases {
        if began.elapsed() >= Duration::from_secs(120) {
            return Err("finite numeric window exhausted before next physical Run".into());
        }
        let raw = backend.run(&case.input)?;
        let pages = backend.host_public_page_snapshot();
        if pages.max_entries != 1
            || pages.pinned_entries != 0
            || pages.pinned_bytes != 0
            || pages.reserved_bytes > pages.max_bytes
            || pages.entries != usize::from(!config.device_public_memory)
        {
            return Err(
                "completed role retained an active host public-page pin or exceeded its bank"
                    .into(),
            );
        }
        let mut report = verify(&raw, case)?;
        report["public_memory"] = verify_public(&backend, case)?;
        reports.push(report);
        let repeated = backend.run(&case.input)?;
        if repeated != raw {
            return Err("fresh private-role repeat changed exact PALS result".into());
        }
        proposer_seen |= case.input.role == PalsRole::Proposer;
        critic_seen |= case.input.role == PalsRole::Critic;
        if control_policy.is_some() && proposer_seen && critic_seen && placement_witness.is_none() {
            placement_witness = Some(backend.verify_cuda_placement()?);
        }
        raw_outputs.push(raw);
    }
    let mut validator = fixtures.cases[0].input.clone();
    validator.role = PalsRole::Validator;
    validator.divergence_features.clear();
    if backend.run(&validator).is_ok() {
        return Err("P/C product admitted private V role".into());
    }
    let public_encodes = backend.public_encodes;
    let public_cache_hits = backend.public_cache_hits;
    backend.verify_runtime()?;
    if is_cuda {
        loading_observations.push(mapping_json(
            "primary_numeric_physical_completion",
            &runtime.cuda_mapping_observation(true)?,
        ));
    }
    let trained = backend.is_trained();
    let residency = backend.residency().clone();
    let pages_before_reset = backend.host_public_page_snapshot();
    let encodes_before_new_game = backend.public_encodes;
    backend.clear_public_memory()?;
    let pages_after_reset = backend.host_public_page_snapshot();
    if pages_after_reset.entries != 0
        || pages_after_reset.reserved_bytes != 0
        || pages_after_reset.pinned_entries != 0
        || pages_after_reset.pinned_bytes != 0
    {
        return Err("NewGame did not retire the completed host public-memory bank".into());
    }
    if backend.public_encodes != encodes_before_new_game {
        return Err("new game cache reset executed a neural graph".into());
    }
    if backend.run(&fixtures.cases[0].input)? != raw_outputs[0]
        || backend.public_encodes != encodes_before_new_game + 1
    {
        return Err("new game did not force a fresh equivalent public-memory encode".into());
    }
    // Separate sessions verify cache eviction/fresh encode equivalence. No
    // measured performance improvement is inferred from these correctness calls.
    config.cache_public_memory = false;
    let mut fresh = load(runtime.clone(), config, "fresh")?;
    let mut fresh_proposer_seen = false;
    let mut fresh_critic_seen = false;
    let mut fresh_placement_witness = None;
    for (case, expected) in fixtures.cases.iter().zip(&raw_outputs) {
        if fresh.run(&case.input)? != *expected {
            return Err("public memory cache changed exact PALS output".into());
        }
        let pages = fresh.host_public_page_snapshot();
        if pages.entries != 0 || pages.reserved_bytes != 0 || pages.pinned_entries != 0 {
            return Err("disabled cache retained a completed host public page".into());
        }
        fresh_proposer_seen |= case.input.role == PalsRole::Proposer;
        fresh_critic_seen |= case.input.role == PalsRole::Critic;
        if control_policy.is_some()
            && fresh_proposer_seen
            && fresh_critic_seen
            && fresh_placement_witness.is_none()
        {
            fresh_placement_witness = Some(fresh.verify_cuda_placement()?);
        }
    }
    fresh.verify_runtime()?;
    drop(fresh);
    let before_worker_stats = backend.snapshot_stats()?;
    let mut worker = backend.controlled_worker()?;
    let mut reset = worker.submit(PalsNativeCommand::NewGame)?;
    let reset_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match reset.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::NewGame)) => break,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(PalsNativeResult::Evaluation(_))) => {
                return Err("cache reset unexpectedly returned a neural evaluation".into());
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::Stats(_))) => {
                return Err("cache reset unexpectedly returned native statistics".into())
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeVerified)) => {
                return Err("cache reset unexpectedly returned runtime verification".into())
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::CudaPlacementVerified(_))) => {
                return Err("cache reset unexpectedly returned CUDA placement verification".into())
            }
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeMappingsObserved(_))) => {
                return Err("cache reset unexpectedly returned runtime mapping observation".into())
            }
            PhysicalPoll::Quarantined => {
                return Err("new game cache reset remains quarantined".into())
            }
            PhysicalPoll::Consumed => return Err("new game cache reset completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < reset_deadline => {
                std::thread::sleep(Duration::from_millis(1));
            }
            PhysicalPoll::Pending => {
                return Err("new game cache reset has not physically completed".into())
            }
        }
    }
    let mut lease = worker.submit(PalsNativeCommand::Evaluate(fixtures.cases[0].input.clone()))?;
    let physical_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match lease.poll() {
            PhysicalPoll::Ready(result) => {
                match result? {
                    PalsNativeResult::Evaluation(raw) => {
                        verify(&raw, &fixtures.cases[0])?;
                    }
                    PalsNativeResult::NewGame => {
                        return Err("evaluation unexpectedly returned a cache reset".into())
                    }
                    PalsNativeResult::Stats(_) => {
                        return Err("evaluation unexpectedly returned native statistics".into())
                    }
                    PalsNativeResult::RuntimeVerified => {
                        return Err("evaluation unexpectedly returned runtime verification".into())
                    }
                    PalsNativeResult::CudaPlacementVerified(_) => {
                        return Err(
                            "evaluation unexpectedly returned CUDA placement verification".into(),
                        )
                    }
                    PalsNativeResult::RuntimeMappingsObserved(_) => {
                        return Err(
                            "evaluation unexpectedly returned runtime mapping observation".into(),
                        )
                    }
                }
                break;
            }
            PhysicalPoll::Quarantined => {
                return Err("physical completion is unknown; quarantined job retained".into())
            }
            PhysicalPoll::Consumed => return Err("physical lease completed twice".into()),
            PhysicalPoll::Pending => {}
        }
        if Instant::now() >= physical_deadline {
            return Err("physical lease still live at finite check deadline".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut verification = worker.submit(PalsNativeCommand::VerifyRuntime)?;
    let verification_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match verification.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::RuntimeVerified)) => break,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(_)) => {
                return Err("runtime verification returned another command response".into())
            }
            PhysicalPoll::Quarantined => {
                return Err(
                    "runtime verification cannot confirm unknown physical completion".into(),
                )
            }
            PhysicalPoll::Consumed => return Err("runtime verification completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < verification_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            PhysicalPoll::Pending => {
                return Err("runtime verification has no physical ACK before deadline".into())
            }
        }
    }
    if let Some(expected) = &placement_witness {
        let mut verification = worker.submit(PalsNativeCommand::VerifyCudaPlacement)?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match verification.poll() {
                PhysicalPoll::Ready(Ok(PalsNativeResult::CudaPlacementVerified(witness)))
                    if *witness == *expected =>
                {
                    break
                }
                PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
                PhysicalPoll::Ready(Ok(_)) => {
                    return Err(
                        "CUDA placement control returned a different immutable witness or response"
                            .into(),
                    )
                }
                PhysicalPoll::Quarantined => {
                    return Err("CUDA placement control has unknown physical completion".into())
                }
                PhysicalPoll::Consumed => {
                    return Err("CUDA placement control completed twice".into())
                }
                PhysicalPoll::Pending if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                PhysicalPoll::Pending => {
                    return Err("CUDA placement control has no finite physical ACK".into())
                }
            }
        }
    }
    let mut snapshot = worker.submit(PalsNativeCommand::SnapshotStats)?;
    let stats_deadline = Instant::now() + Duration::from_secs(30);
    let stats = loop {
        match snapshot.poll() {
            PhysicalPoll::Ready(Ok(PalsNativeResult::Stats(stats))) => break stats,
            PhysicalPoll::Ready(Err(error)) => return Err(error.into()),
            PhysicalPoll::Ready(Ok(_)) => {
                return Err("native statistics returned another command response".into())
            }
            PhysicalPoll::Quarantined => {
                return Err("native statistics cannot confirm unknown physical completion".into())
            }
            PhysicalPoll::Consumed => return Err("native statistics completed twice".into()),
            PhysicalPoll::Pending if Instant::now() < stats_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            PhysicalPoll::Pending => {
                return Err("native statistics have no physical ACK before deadline".into())
            }
        }
    };
    stats.validate()?;
    if stats.admitted_role_requests != before_worker_stats.admitted_role_requests + 1
        || stats.role_nn_runs_attempted != before_worker_stats.role_nn_runs_attempted + 1
        || stats.role_nn_runs_completed != before_worker_stats.role_nn_runs_completed + 1
        || stats.public_nn_runs_attempted != before_worker_stats.public_nn_runs_attempted + 1
        || stats.public_nn_runs_completed != before_worker_stats.public_nn_runs_completed + 1
        || stats.completed_nn_inputs != before_worker_stats.completed_nn_inputs + 2
        || stats.new_game_resets != before_worker_stats.new_game_resets + 1
        || stats.live_public_cache_entries != 1
    {
        return Err("NewGame/SnapshotStats control changed NN accounting or failed to force fresh public encode".into());
    }
    let shutdown_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match worker.try_shutdown() {
            Poll::Ready(result) => {
                result?;
                break;
            }
            Poll::Pending if Instant::now() < shutdown_deadline => {
                std::thread::sleep(Duration::from_millis(1))
            }
            Poll::Pending => return Err("physical owner has not completed shutdown".into()),
        }
    }
    if is_cuda {
        loading_observations.push(mapping_json(
            "final_worker_joined",
            &runtime.cuda_mapping_observation(true)?,
        ));
    }
    let host_page_evidence = json!({"scope":"direct_owner_after_numeric_cases_before_worker",
        "page_kind":"whole_input","frozen_numeric_epoch":0,
        "weight_epoch_identity":"full_checkpoint_digest_in_exact_content_key",
        "bank_max_entries":pages_before_reset.max_entries,"bank_max_bytes":pages_before_reset.max_bytes,
        "before_new_game":{"entries":pages_before_reset.entries,"reserved_bytes":pages_before_reset.reserved_bytes,
            "pinned_entries":pages_before_reset.pinned_entries,"pinned_bytes":pages_before_reset.pinned_bytes},
        "after_new_game":{"entries":pages_after_reset.entries,"reserved_bytes":pages_after_reset.reserved_bytes,
            "pinned_entries":pages_after_reset.pinned_entries,"pinned_bytes":pages_after_reset.pinned_bytes},
        "cache_off_completed_page_retirement":"passed","runtime_allocator_peak":"unknown","vram_peak":"unknown"});
    let cuda_evidence = json!({"cuda_mode":args[6],"graph_optimization":graph_optimization,
        "cuda_control_inventory_sha256":control_policy.as_ref().map(|_| args[10].as_str()),
        "cuda_placement_witness":placement_witness,"fresh_cuda_placement_witness":fresh_placement_witness});
    let mut receipt = json!({"schema": PALS_MODEL_SCHEMA,"status":"passed","trained":trained,"rules_certified":false,
        "model_epoch":hex_digest(&epoch),"manifest_sha256":args[1],"fixture_sha256":hex_digest(&fixture_digest),
        "runtime_sha256":hex_digest(&runtime_digest),"runtime":"1.22.0","rust_ort":"2.0.0-rc.10",
        "provider":if is_cuda {"CUDAExecutionProvider"} else {"CPUExecutionProvider"},"precision":"fp32","tf32":false,
        "physical_completion":"confirmed","physical_shutdown":"confirmed","public_encodes":public_encodes,
        "public_cache_hits":public_cache_hits,"fresh_cache_equivalence":"passed","private_role_repeat":"passed",
        "public_counter_scope":"six_reference_cases_before_new_game_checks",
        "v_rejection":"passed","new_game_cache_reset":"passed","new_game_physical_ack":"confirmed",
        "stats_physical_ack":"confirmed","control_neural_runs":"zero","backend_stats":stats_json(&stats),
        "runtime_verify_physical_ack":"confirmed",
        "backend_stats_scope":"backend_lifetime_through_final_snapshot_ack",
        "device_public_memory":config.device_public_memory,"session_residency":residency,"vram_peak":"unknown","cases":reports});
    let fields = receipt
        .as_object_mut()
        .ok_or("acceptance receipt is not an object")?;
    fields.insert("host_public_page_ownership".into(), host_page_evidence);
    fields.insert(
        "native_loading".into(),
        json!({
        "schema":"rovezero.native-loading-observation.v1",
        "profile":loading_profile.map(NativeLoadingProfile::identifier),
        "profile_sha256":loading_profile_digest.as_ref().map(hex_digest),
        "experimental_candidate":loading_profile == Some(NativeLoadingProfile::CuDnnShimLazyV1),
        "runtime_bundle_sha256":runtime_bundle_digest.as_ref().map(hex_digest),
        "declared_bundle_file_count":if is_cuda {19} else {1},
        "mapping_observations":loading_observations,
        "native_termination_acceptance":"requires_external_process_exit_zero_and_cleanup",
        "preload_cause":"not_proven"}),
    );
    fields.extend(
        cuda_evidence
            .as_object()
            .ok_or("CUDA evidence is not an object")?
            .clone(),
    );
    if check_record_pages {
        fields.insert(
            "host_record_page_correctness".into(),
            check_host_record_pages(&fixtures, runtime.clone(), Path::new(&args[0]), &args[1])?,
        );
    }
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report)?;
    serde_json::to_writer_pretty(&mut output, &receipt)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    println!("PALS P/C numeric/physical checks passed; trained={trained}; GPU peak unknown");
    Ok(())
}
