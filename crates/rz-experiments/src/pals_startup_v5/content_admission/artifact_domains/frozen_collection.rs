//! Exact 4000-update producer metadata. This validator does not execute Torch,
//! issue an original trainer capability or certify loaded native tensors.
use super::*;

pub const PALS_FROZEN_PC_COLLECTION_EXPORT_V5_DOMAIN: &str =
    "rz-pals-frozen-pc-collection-export/2";

#[derive(Clone, Debug, Serialize)]
pub struct PalsFrozenPcCollectionMetadataV5 {
    checkpoint_sha256: String,
    named_shared_pc_sha256: String,
    model_epoch: u64,
    context_epoch: u64,
    checkpoint_model_state_sha256: String,
    checkpoint_optimizer_state_sha256: String,
}
impl PalsFrozenPcCollectionMetadataV5 {
    pub fn checkpoint_sha256(&self) -> &str {
        &self.checkpoint_sha256
    }
    pub fn named_shared_pc_sha256(&self) -> &str {
        &self.named_shared_pc_sha256
    }
    pub fn model_epoch(&self) -> u64 {
        self.model_epoch
    }
    pub fn context_epoch(&self) -> u64 {
        self.context_epoch
    }
}

// The completed startup metadata wrapper preserves its native-options Fresh
// guard before projection. A numerical-only scope carries no Warm capability.
pub(super) fn validate_frozen_collection_scope(
    p: &Value,
    registered_spec: &Value,
    config: &PalsNumericalScopeV5,
    export: &Value,
) -> Result<PalsFrozenPcCollectionMetadataV5, ManifestError> {
    need(
        config.mode == PalsEngineModeV5::Collection
            && config.model.profile == "virtual_review_v1"
            && config.model.checkpoint_domain.as_deref() == Some(CHECKPOINT_DOMAIN),
        "frozen P/C requires its own Fresh Collection domain",
    )?;
    fields(
        p,
        &[
            "schema",
            "eligibility",
            "generation_id",
            "spec",
            "spec_sha256",
            "source_sha256",
            "runtime_sha256",
            "checkpoint",
            "pc_admission",
            "collection",
            "validator_present",
            "validator_training_updates",
            "deployment_exported",
            "native_loaded_assets_verified",
            "numeric_status",
            "cuda_status",
            "arena_eligible",
        ],
    )?;
    for (key, value) in [
        ("schema", json!(PALS_FROZEN_PC_COLLECTION_EXPORT_V5_DOMAIN)),
        ("eligibility", json!("frozen_pc_collection_only")),
        ("validator_present", json!(true)),
        ("validator_training_updates", json!(0)),
        ("deployment_exported", json!(false)),
        ("native_loaded_assets_verified", json!(false)),
        ("numeric_status", json!("not_run")),
        ("cuda_status", json!("not_run")),
        ("arena_eligible", json!(false)),
    ] {
        exact(
            &p[key],
            value,
            "frozen P/C export cannot claim V training, native loading, arena or G1",
        )?;
    }
    uint(&p["validator_training_updates"])?;
    validate_generation_spec(registered_spec)?;
    exact(
        &p["spec"],
        registered_spec.clone(),
        "frozen P/C recipe differs from independent registration",
    )?;
    let spec = &p["spec"];
    exact(
        &p["spec_sha256"],
        json!(semantic_digest(GENERATION_SPEC_DOMAIN, spec)?),
        "frozen P/C recipe semantic identity differs",
    )?;
    for key in ["generation_id", "source_sha256", "runtime_sha256"] {
        exact(
            &p[key],
            spec[key].clone(),
            "frozen P/C source/runtime/generation join differs",
        )?;
    }
    exact(
        &export["trained"],
        json!(true),
        "frozen P/C must preserve nonzero training",
    )?;
    need(
        uint(&export["training_steps"])? == 4_000,
        "frozen P/C cannot be relabelled as untrained or complete G1",
    )?;
    let checkpoint = &p["checkpoint"];
    validate_frozen_checkpoint(checkpoint, spec, &config.model.weights, config.model.epoch)?;
    validate_admission(&p["pc_admission"], spec, "pc", checkpoint)?;
    let descriptors = &p["collection"];
    let bytes = validate_role_artifacts(descriptors, "collection")?
        .checked_add(uint(&checkpoint["bytes"])?)
        .ok_or_else(|| fail("frozen output size overflow"))?;
    need(bytes <= MANAGED_BYTES, "frozen output exceeds managed cap")?;
    identity_join(
        &descriptors["manifest"],
        artifact_for(config, Purpose::ExportManifest)?,
    )?;
    let actual_graphs = array(&export["graphs"], 3)?;
    let pins = array(&descriptors["graphs"], 3)?;
    need(
        actual_graphs.len() == pins.len(),
        "frozen native graph roster differs",
    )?;
    for (actual, pin) in actual_graphs.iter().zip(pins) {
        exact(
            &actual["role"],
            pin["role"].clone(),
            "frozen native graph order differs",
        )?;
        exact(
            &actual["file"],
            json!(text(&pin["file"])?.rsplit('/').next().unwrap()),
            "frozen native graph filename differs",
        )?;
        exact(
            &actual["sha256"],
            pin["sha256"].clone(),
            "frozen native graph hash differs",
        )?;
        let purpose = match text(&pin["role"])? {
            "public" => Purpose::PublicGraph,
            "shared_pc" => Purpose::SharedPcGraph,
            "validator" => Purpose::ValidatorGraph,
            _ => return Err(fail("frozen graph role unsupported")),
        };
        identity_join(pin, artifact_for(config, purpose)?)?;
    }
    Ok(PalsFrozenPcCollectionMetadataV5 {
        checkpoint_sha256: config.model.weights.sha256.clone(),
        named_shared_pc_sha256: sha(&checkpoint["frozen_pc_sha256"])?.into(),
        model_epoch: config.model.epoch,
        context_epoch: uint(&checkpoint["context_epoch"])?,
        checkpoint_model_state_sha256: sha(&checkpoint["state_sha256"]["model"])?.into(),
        checkpoint_optimizer_state_sha256: sha(&checkpoint["state_sha256"]["optimizers"])?.into(),
    })
}

fn validate_frozen_checkpoint(
    c: &Value,
    spec: &Value,
    weights: &ArtifactRef,
    epoch: u64,
) -> Result<(), ManifestError> {
    fields(
        c,
        &[
            "schema",
            "checkpoint_domain",
            "eligibility",
            "file",
            "sha256",
            "bytes",
            "phase",
            "pending_microbatches",
            "completed_updates",
            "dispatched_updates",
            "ledger_sha256",
            "cost",
            "state_sha256",
            "config_sha256",
            "frozen_pc_sha256",
            "frozen_pc_model_epoch",
            "context_epoch",
            "nonzero_parameter_owners",
            "cuda_devices",
        ],
    )?;
    for (key, value) in [
        ("schema", json!("rz-pals-verified-frozen-pc-checkpoint/1")),
        ("checkpoint_domain", json!(CHECKPOINT_DOMAIN)),
        ("eligibility", json!("actual_training_pending_acceptance")),
        ("phase", json!("v")),
        ("pending_microbatches", json!(0)),
        (
            "nonzero_parameter_owners",
            json!(["shared", "proposer", "critic"]),
        ),
        ("cuda_devices", json!([0])),
    ] {
        exact(
            &c[key],
            value,
            "frozen P/C checkpoint phase, owners or domain differs",
        )?;
    }
    uint(&c["pending_microbatches"])?;
    for device in array(&c["cuda_devices"], 1)? {
        uint(device)?;
    }
    // The producer records the actual durable trainer basename. The separately
    // owned startup snapshot may use checkpoint.pt, but it must not relabel the
    // original checkpoint path in training provenance.
    let filename = text(&c["file"])?;
    need(
        filename.len() <= 128
            && !filename.starts_with('.')
            && filename.ends_with(".pt")
            && filename
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')),
        "frozen checkpoint metadata requires the bounded original trainer basename",
    )?;
    identity_join(c, weights)?;
    need(
        (1..=MANAGED_BYTES).contains(&uint(&c["bytes"])?),
        "frozen checkpoint size exceeds bound",
    )?;
    for key in ["completed_updates", "dispatched_updates"] {
        fields(&c[key], &ROLES)?;
    }
    for (role, count) in [("proposer", 2000), ("critic", 2000), ("validator", 0)] {
        let completed = uint(&c["completed_updates"][role])?;
        let dispatched = uint(&c["dispatched_updates"][role])?;
        need(
            completed == count && dispatched >= count && (role != "validator" || dispatched == 0),
            "frozen checkpoint requires actual P=2000/C=2000/V=0; failed dispatch cost is not refunded",
        )?;
    }
    fields(&c["cost"], &COST_FIELDS)?;
    for key in COST_FIELDS {
        uint(&c["cost"][key])?;
    }
    let cost = &c["cost"];
    need(
        ["forward_calls", "backward_calls", "samples"]
            .iter()
            .all(|k| cost[*k].as_u64().unwrap_or(0) >= 32_000)
            && uint(&cost["backward_calls"])? <= uint(&cost["samples"])?
            && uint(&cost["samples"])? <= uint(&cost["forward_calls"])?
            && uint(&cost["wall_time_ns"])? > 0
            && uint(&cost["output_bytes"])? <= MANAGED_BYTES,
        "frozen accumulated work is missing, all-masked or inconsistent",
    )?;
    fields(
        &c["state_sha256"],
        &[
            "model",
            "optimizers",
            "schedulers",
            "rng",
            "sampler",
            "parameter_updates",
            "membership",
            "bucket",
        ],
    )?;
    for value in c["state_sha256"].as_object().unwrap().values() {
        sha(value)?;
    }
    for key in ["ledger_sha256", "config_sha256", "frozen_pc_sha256"] {
        sha(&c[key])?;
    }
    need(
        uint(&c["frozen_pc_model_epoch"])? == epoch
            && epoch > uint(&spec["model_epoch"])?
            && uint(&c["context_epoch"])? == uint(&spec["context_epoch"])?,
        "frozen P/C new model epoch or original context epoch differs",
    )?;
    exact(
        &c["config_sha256"],
        json!(semantic_digest(
            "rz-pals-generation-model-config/1",
            &compact_configuration()
        )?),
        "frozen model configuration differs",
    )?;
    let ledger = json!({"spec":spec,"completed":c["completed_updates"],"dispatched":c["dispatched_updates"],
        "phase":"v","next_role":"validator","pending_microbatches":0,"cost":cost});
    exact(
        &c["ledger_sha256"],
        json!(semantic_digest(
            "rz-pals-generation-ledger-state/1",
            &ledger
        )?),
        "frozen completed update/cost ledger changed",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{checkpoint_fixture, spec_fixture, weights_fixture};
    use super::*;
    fn fixture() -> (Value, Value) {
        let spec = spec_fixture();
        let mut c = checkpoint_fixture(&spec);
        c["schema"] = json!("rz-pals-verified-frozen-pc-checkpoint/1");
        c["phase"] = json!("v");
        c["completed_updates"]["validator"] = json!(0);
        c["dispatched_updates"]["validator"] = json!(0);
        c["nonzero_parameter_owners"] = json!(["shared", "proposer", "critic"]);
        let ledger = json!({"spec":spec,"completed":c["completed_updates"],"dispatched":c["dispatched_updates"],
            "phase":"v","next_role":"validator","pending_microbatches":0,"cost":c["cost"]});
        c["ledger_sha256"] =
            json!(semantic_digest("rz-pals-generation-ledger-state/1", &ledger).unwrap());
        (spec, c)
    }
    #[test]
    fn exact_frozen_boundary_does_not_become_g0_g1_or_partial_v() {
        let (spec, c) = fixture();
        validate_frozen_checkpoint(&c, &spec, &weights_fixture(), 1).unwrap();
        assert!(validate_checkpoint_summary(&c, &spec, &weights_fixture(), 1).is_err());
        for (key, value) in [
            ("phase", json!("complete")),
            ("pending_microbatches", json!(1)),
            ("schema", json!("rz-pals-verified-generation-checkpoint/1")),
            ("frozen_pc_model_epoch", json!(0)),
            ("checkpoint_domain", json!("diagnostic")),
            ("file", json!("../checkpoint.pt")),
        ] {
            let mut changed = c.clone();
            changed[key] = value;
            assert!(validate_frozen_checkpoint(&changed, &spec, &weights_fixture(), 1).is_err());
        }
        let mut changed = c.clone();
        changed["completed_updates"]["validator"] = json!(1);
        assert!(validate_frozen_checkpoint(&changed, &spec, &weights_fixture(), 1).is_err());
        let mut changed = c.clone();
        changed["completed_updates"]["critic"] = json!(1999);
        assert!(validate_frozen_checkpoint(&changed, &spec, &weights_fixture(), 1).is_err());
        let mut original_name = c.clone();
        original_name["file"] = json!("checkpoint-004000.pt");
        validate_frozen_checkpoint(&original_name, &spec, &weights_fixture(), 1).unwrap();
        let mut changed = c;
        changed["state_sha256"]
            .as_object_mut()
            .unwrap()
            .remove("optimizers");
        assert!(validate_frozen_checkpoint(&changed, &spec, &weights_fixture(), 1).is_err());
    }
}
