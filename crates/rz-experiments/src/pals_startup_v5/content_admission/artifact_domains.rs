//! Metadata admission for independently registered G0 and actual-generation assets.
//!
//! The launch owner must independently stream-verify every registered artifact
//! before using this result. This reader never loads a tensor/optimizer, grants
//! initializer capability, runs a neural model, or grants numerical/arena acceptance.
//! This public metadata summary is lower scope than the private content-admission
//! token; constructing or receiving its booleans never grants actual-input admission.
use crate::ManifestError;
use crate::{
    ArtifactRef, PalsAdaptiveStartupConfigV5, PalsArtifactDomainV5, PalsEngineModeV5,
    PalsStartupAssetPurposeV5 as Purpose, decode_json,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const METADATA_LIMIT: usize = 4 << 20;
const MANAGED_BYTES: u64 = 18 << 30;
const INITIALIZATION_DOMAIN: &str = "rz-pals-g0-initialization-evidence/1";
const GENERATION_DOMAIN: &str = "rz-pals-actual-generation-export/1";
const GENERATION_SPEC_DOMAIN: &str = "rz-pals-train-generation/1";
const CHECKPOINT_DOMAIN: &str = "rz-pals-actual-generation-checkpoint/1";
const INITIALIZER_FILES: [&str; 8] = [
    "__init__.py",
    "config.py",
    "model.py",
    "artifacts.py",
    "controlled_initialization.py",
    "virtual_review.py",
    "onnx_shared.py",
    "initialization_evidence.py",
];
const ROLES: [&str; 3] = ["proposer", "critic", "validator"];
const UPDATES: [(&str, u64); 3] = [("proposer", 2_000), ("critic", 2_000), ("validator", 1_000)];
const COST_FIELDS: [&str; 8] = [
    "forward_calls",
    "backward_calls",
    "samples",
    "wall_time_ns",
    "partial_accumulation_ns",
    "replay_ns",
    "masked_groups",
    "output_bytes",
];

#[derive(Clone, Debug, Serialize)]
pub struct PalsArtifactDomainAdmissionV5 {
    pub artifact_domain: PalsArtifactDomainV5,
    pub checkpoint_sha256: String,
    pub initialization_evidence_sha256: Option<String>,
    /// The registered byte hash; it is not Python's semantic JSON digest.
    pub provenance_file_sha256: Option<String>,
    pub generation_specification_file_sha256: Option<String>,
    pub generation_id: Option<String>,
    pub spec_semantic_sha256: Option<String>,
    pub training_source_sha256: Option<String>,
    pub training_runtime_sha256: Option<String>,
    /// The parent joins each named initializer source against its actual source inventory.
    pub initializer_source_commit: Option<String>,
    pub initializer_source_pins: BTreeMap<String, String>,
    pub producer_metadata_verified: bool,
    pub registered_artifact_bytes_verified_by_launch_owner: bool,
    pub tensor_optimizer_content_verified: bool,
    pub numerical_acceptance_verified: bool,
    pub arena_accepted: bool,
}

fn fail(message: &str) -> ManifestError {
    ManifestError::Artifact(format!("V5 artifact domain: {message}"))
}
fn need(ok: bool, message: &str) -> Result<(), ManifestError> {
    if ok { Ok(()) } else { Err(fail(message)) }
}
fn fields(v: &Value, names: &[&str]) -> Result<(), ManifestError> {
    let map = v.as_object().ok_or_else(|| fail("object required"))?;
    need(
        map.len() == names.len() && names.iter().all(|n| map.contains_key(*n)),
        "exact metadata fields differ",
    )
}
fn text(v: &Value) -> Result<&str, ManifestError> {
    let s = v.as_str().ok_or_else(|| fail("text required"))?;
    need(
        !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control),
        "text is empty or exceeds its bound",
    )?;
    Ok(s)
}
fn sha(v: &Value) -> Result<&str, ManifestError> {
    let s = text(v)?;
    need(
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "lowercase SHA-256 required",
    )?;
    Ok(s)
}
fn uint(v: &Value) -> Result<u64, ManifestError> {
    v.as_u64()
        .ok_or_else(|| fail("unsigned integer required; bool/float is not a count"))
}
fn exact(v: &Value, expected: Value, message: &str) -> Result<(), ManifestError> {
    need(v == &expected, message)
}
fn array(v: &Value, maximum: usize) -> Result<&Vec<Value>, ManifestError> {
    let a = v.as_array().ok_or_else(|| fail("bounded array required"))?;
    need(a.len() <= maximum, "array exceeds its registered bound")?;
    Ok(a)
}
fn parse_bytes(bytes: &[u8]) -> Result<Value, ManifestError> {
    need(
        !bytes.is_empty() && bytes.len() <= METADATA_LIMIT,
        "metadata byte limit exceeded",
    )?;
    let input = std::str::from_utf8(bytes).map_err(|_| fail("UTF-8 metadata required"))?;
    Ok(decode_json(input)?)
}
fn read_metadata(
    artifact: &ArtifactRef,
    reader: &mut impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ManifestError>,
) -> Result<Value, ManifestError> {
    artifact.validate()?;
    need(
        artifact.bytes > 0 && artifact.bytes <= METADATA_LIMIT as u64,
        "metadata artifact exceeds admission bound",
    )?;
    let bytes = reader(artifact, METADATA_LIMIT)?;
    need(
        bytes.len() as u64 == artifact.bytes
            && format!("{:x}", Sha256::digest(&bytes)) == artifact.sha256,
        "actual metadata bytes differ from the registered artifact",
    )?;
    parse_bytes(&bytes)
}
fn artifact_for(
    config: &PalsAdaptiveStartupConfigV5,
    purpose: Purpose,
) -> Result<&ArtifactRef, ManifestError> {
    Ok(&config.asset(purpose)?.artifact)
}
fn profile_configuration(profile: &str) -> Value {
    json!({"width":384,"query_heads":6,"kv_heads":2,"head_dimension":64,"board_blocks":2,
        "record_blocks":1,"record_fields":4,"latent_slots":16,"recurrent_blocks":2,"iterations":2,
        "ffn_width":1024,"max_records":if profile == "virtual_review_v1" {64} else {128},"max_candidates":256,"max_divergences":128,"profile":profile})
}
fn compact_configuration() -> Value {
    profile_configuration("virtual_review_v1")
}
fn profile_configuration_check(v: &Value, profile: &str) -> Result<(), ManifestError> {
    need(
        matches!(profile, "virtual_review_v1" | "full_line_interaction_v2"),
        "unsupported initializer model profile",
    )?;
    exact(
        v,
        profile_configuration(profile),
        "unregistered model configuration",
    )?;
    for (name, value) in v.as_object().unwrap() {
        if name != "profile" {
            uint(value)?;
        }
    }
    Ok(())
}
fn file_pin(v: &Value, filename: &str, maximum: u64) -> Result<(), ManifestError> {
    fields(v, &["file", "sha256", "bytes"])?;
    exact(
        &v["file"],
        json!(filename),
        "file pin uses another registered relative name",
    )?;
    sha(&v["sha256"])?;
    need(
        (1..=maximum).contains(&uint(&v["bytes"])?),
        "file pin size exceeds its bound",
    )
}
fn identity_join(v: &Value, artifact: &ArtifactRef) -> Result<(), ManifestError> {
    need(
        sha(&v["sha256"])? == artifact.sha256 && uint(&v["bytes"])? == artifact.bytes,
        "registered artifact and producer byte identity differ",
    )
}

/// Validate producer metadata against independently registered startup inputs.
/// The callback may read small JSON only; it is never asked to materialize model
/// weights or graphs. The result explicitly leaves physical/file/source and GPU
/// acceptance to the launch owner and independent numerical gate. It does not mint
/// the private actual-content admission token or grant execution authority.
pub fn validate_pals_artifact_domain_v5(
    config: &PalsAdaptiveStartupConfigV5,
    export_bytes: &[u8],
    mut read_artifact: impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ManifestError>,
) -> Result<PalsArtifactDomainAdmissionV5, ManifestError> {
    config.endpoint.model.validate(config.endpoint.mode)?;
    need(
        !config.training_executed,
        "arena startup cannot execute training",
    )?;
    let export_artifact = artifact_for(config, Purpose::ExportManifest)?;
    need(
        export_bytes.len() as u64 == export_artifact.bytes
            && format!("{:x}", Sha256::digest(export_bytes)) == export_artifact.sha256,
        "export manifest bytes differ from registration",
    )?;
    let export = parse_bytes(export_bytes)?;
    profile_configuration_check(&export["config"], &config.endpoint.model.profile)?;
    let legacy = config.endpoint.model.profile == "full_line_interaction_v2";
    if legacy {
        // Historical separate-PC serialization kept model.v1 even though the
        // full-line model/encoding semantics were V2. The parent's strict
        // legacy export parser validates that original descriptor/layout.
        need(
            matches!(
                export["schema"].as_str(),
                Some("rovezero.pals-model.v1" | "rovezero.pals-model.v2")
            ),
            "legacy serialized export schema is unsupported",
        )?;
    } else {
        exact(
            &export["schema"],
            json!(config.endpoint.model.model_schema),
            "export schema differs from registered compact model",
        )?;
    }
    for (key, value) in [
        ("model_semantics", &config.endpoint.model.model_semantics),
        ("encoding_schema", &config.endpoint.model.encoding_schema),
        ("model_profile", &config.endpoint.model.profile),
    ] {
        exact(
            &export[key],
            json!(value),
            "export and registered model meanings differ",
        )?;
    }
    exact(
        &export["checkpoint_sha256"],
        json!(config.endpoint.model.weights.sha256),
        "export references another checkpoint",
    )?;
    exact(
        &export["precision"],
        json!("fp32"),
        "export precision differs",
    )?;
    exact(&export["tf32"], json!(false), "TF32 must remain disabled")?;
    let collection = config.endpoint.mode == PalsEngineModeV5::Collection;
    exact(
        &export["roles"],
        if collection {
            json!(ROLES)
        } else {
            json!(["proposer", "critic"])
        },
        "export role bank differs from registered Collection/Deployment mode",
    )?;
    exact(
        &export["validator_present"],
        json!(collection),
        "Deployment cannot load V and Collection cannot omit V",
    )?;
    for key in ["numeric_status", "cuda_status"] {
        exact(
            &export[key],
            json!("not_run"),
            "producer export cannot substitute for independent numerical/CUDA acceptance",
        )?;
    }
    let mut result = PalsArtifactDomainAdmissionV5 {
        artifact_domain: config.endpoint.model.artifact_domain,
        checkpoint_sha256: config.endpoint.model.weights.sha256.clone(),
        initialization_evidence_sha256: None,
        provenance_file_sha256: None,
        generation_specification_file_sha256: None,
        generation_id: None,
        spec_semantic_sha256: None,
        training_source_sha256: None,
        training_runtime_sha256: None,
        initializer_source_commit: None,
        initializer_source_pins: BTreeMap::new(),
        producer_metadata_verified: false,
        registered_artifact_bytes_verified_by_launch_owner: false,
        tensor_optimizer_content_verified: false,
        numerical_acceptance_verified: false,
        arena_accepted: false,
    };
    match config.endpoint.model.artifact_domain {
        PalsArtifactDomainV5::Untrained => {
            need(
                config.endpoint.model.provenance.is_none()
                    && config.endpoint.model.checkpoint_domain.is_none(),
                "G0 cannot carry a training/diagnostic provenance domain",
            )?;
            exact(
                &export["trained"],
                json!(false),
                "G0 cannot be a trained export",
            )?;
            need(
                uint(&export["training_steps"])? == 0,
                "G0 has optimizer updates",
            )?;
            let evidence_pin = artifact_for(config, Purpose::InitializationEvidence)?;
            let evidence = read_metadata(evidence_pin, &mut read_artifact)?;
            validate_g0(&evidence, config, &export, &mut read_artifact, &mut result)?;
            result.initialization_evidence_sha256 = Some(evidence_pin.sha256.clone());
        }
        PalsArtifactDomainV5::RealGeneration => {
            need(
                config.endpoint.model.profile == "virtual_review_v1",
                "actual generation cannot relabel a legacy model profile",
            )?;
            need(
                config.endpoint.model.checkpoint_domain.as_deref() == Some(CHECKPOINT_DOMAIN),
                "G1 checkpoint domain differs",
            )?;
            exact(
                &export["trained"],
                json!(true),
                "G1 requires a nonzero trained export",
            )?;
            need(
                uint(&export["training_steps"])? == 5_000,
                "G1 first generation is incomplete",
            )?;
            let provenance_pin = artifact_for(config, Purpose::Provenance)?;
            need(
                config.endpoint.model.provenance.as_ref() == Some(provenance_pin),
                "G1 provenance is not independently registered",
            )?;
            let provenance = read_metadata(provenance_pin, &mut read_artifact)?;
            let spec_pin = artifact_for(config, Purpose::GenerationSpecification)?;
            let registered_spec = read_metadata(spec_pin, &mut read_artifact)?;
            validate_g1(&provenance, &registered_spec, config, &export)?;
            result.provenance_file_sha256 = Some(provenance_pin.sha256.clone());
            result.generation_specification_file_sha256 = Some(spec_pin.sha256.clone());
            result.generation_id = Some(text(&provenance["generation_id"])?.into());
            result.spec_semantic_sha256 = Some(sha(&provenance["spec_sha256"])?.into());
            result.training_source_sha256 = Some(sha(&provenance["source_sha256"])?.into());
            result.training_runtime_sha256 = Some(sha(&provenance["runtime_sha256"])?.into());
        }
        PalsArtifactDomainV5::Diagnostic => {
            return Err(fail(
                "diagnostic checkpoints cannot become actual V5 paired assets",
            ));
        }
    }
    result.producer_metadata_verified = true;
    Ok(result)
}

fn validate_g0(
    e: &Value,
    config: &PalsAdaptiveStartupConfigV5,
    export: &Value,
    reader: &mut impl FnMut(&ArtifactRef, usize) -> Result<Vec<u8>, ManifestError>,
    result: &mut PalsArtifactDomainAdmissionV5,
) -> Result<(), ManifestError> {
    fields(
        e,
        &[
            "schema",
            "generation_id",
            "initialization_id",
            "model",
            "source",
            "checkpoint",
            "metadata",
            "metadata_fields",
            "checkpoint_inventory",
            "initialization",
            "software",
            "claims",
            "acceptance",
        ],
    )?;
    let legacy = config.endpoint.model.profile == "full_line_interaction_v2";
    let expected_configuration = profile_configuration(&config.endpoint.model.profile);
    exact(
        &e["schema"],
        json!(if legacy {
            "rz-pals-legacy-initialization-observation/1"
        } else {
            INITIALIZATION_DOMAIN
        }),
        "wrong initialization evidence domain; legacy and G0 cannot be relabelled",
    )?;
    exact(
        &e["generation_id"],
        json!(if legacy { "legacy_baseline" } else { "g0" }),
        "initializer generation differs",
    )?;
    need(
        text(&e["initialization_id"])?.len() <= 128,
        "initialization ID exceeds its bound",
    )?;
    let model = &e["model"];
    fields(
        model,
        &[
            "schema",
            "semantics",
            "encoding",
            "profile",
            "config",
            "roles",
            "seed",
        ],
    )?;
    exact(
        model,
        json!({"schema":config.endpoint.model.model_schema,"semantics":config.endpoint.model.model_semantics,
        "encoding":config.endpoint.model.encoding_schema,"profile":config.endpoint.model.profile,"config":expected_configuration,"roles":ROLES,"seed":23}),
        "registered initializer model, seed or role bank differs",
    )?;
    profile_configuration_check(&model["config"], &config.endpoint.model.profile)?;
    need(uint(&model["seed"])? == 23, "G0 seed is not 23")?;
    let metadata = &e["metadata_fields"];
    fields(
        metadata,
        &[
            "schema",
            "checkpoint",
            "checkpoint_sha256",
            "config",
            "seed",
            "trained",
            "training_steps",
            "external_weights",
            "external_teacher",
            "precision",
            "tf32",
            "torch_version",
            "parameter_count",
            "roles",
            "status",
        ],
    )?;
    for (key, value) in [
        ("schema", json!(config.endpoint.model.model_semantics)),
        ("checkpoint", json!("untrained.pt")),
        (
            "config",
            profile_configuration(&config.endpoint.model.profile),
        ),
        ("seed", json!(23)),
        ("training_steps", json!(0)),
        ("roles", json!(ROLES)),
        ("status", json!("random_initialization_only")),
        ("precision", json!("fp32")),
    ] {
        exact(
            &metadata[key],
            value,
            "initializer checkpoint metadata differs",
        )?;
    }
    uint(&metadata["seed"])?;
    uint(&metadata["training_steps"])?;
    profile_configuration_check(&metadata["config"], &config.endpoint.model.profile)?;
    for key in ["trained", "external_weights", "external_teacher", "tf32"] {
        exact(
            &metadata[key],
            json!(false),
            "G0 metadata asserts training/external import",
        )?;
    }
    sha(&metadata["checkpoint_sha256"])?;
    text(&metadata["torch_version"])?;
    let parameter_count = uint(&metadata["parameter_count"])?;
    need(
        (1..=1 << 28).contains(&parameter_count),
        "G0 parameter count is empty or exceeds its bound",
    )?;
    for (key, filename, maximum) in [
        ("checkpoint", "untrained.pt", 1u64 << 30),
        ("metadata", "checkpoint.json", 65536),
    ] {
        fields(&e[key], &["name", "bytes", "sha256"])?;
        exact(
            &e[key]["name"],
            json!(filename),
            "initializer named artifact differs",
        )?;
        sha(&e[key]["sha256"])?;
        need(
            (1..=maximum).contains(&uint(&e[key]["bytes"])?),
            "initializer named artifact exceeds its bound",
        )?;
    }
    identity_join(&e["checkpoint"], &config.endpoint.model.weights)?;
    exact(
        &e["checkpoint"]["sha256"],
        metadata["checkpoint_sha256"].clone(),
        "metadata and checkpoint bytes differ",
    )?;
    let metadata_pin = artifact_for(config, Purpose::InitializationCheckpointMetadata)?;
    identity_join(&e["metadata"], metadata_pin)?;
    let actual_metadata = read_metadata(metadata_pin, reader)?;
    exact(
        &actual_metadata,
        metadata.clone(),
        "embedded initializer metadata differs from its actual registered file",
    )?;
    exact(
        &export["config"],
        metadata["config"].clone(),
        "export and initializer model configurations differ",
    )?;
    let names = inventory(&e["checkpoint_inventory"], parameter_count)?;
    let init = &e["initialization"];
    fields(
        init,
        &[
            "mode",
            "producer",
            "invocation_status",
            "inventory_producer",
            "allowlist",
            "copied",
            "ignored_source",
            "ignored_target",
            "partial_initialization",
        ],
    )?;
    exact(
        init,
        json!({"mode":"fresh_random","producer":"rz_pals_model.artifacts.initialize_checkpoint",
        "invocation_status":"captured_registered_initializer_return","inventory_producer":"rz_pals_model.controlled_initialization._parameter_inventory",
        "allowlist":{},"copied":[],"ignored_source":[],"ignored_target":names,
        "partial_initialization":"requires_separate_actual_producer_registration"}),
        "G0 initializer capture cannot be replaced by a prepared/partial copy claim",
    )?;
    fields(&e["source"], &["schema", "source_commit", "pins"])?;
    exact(
        &e["source"]["schema"],
        json!(if legacy {
            "rz-pals-legacy-initialization-sources/1"
        } else {
            "rz-pals-g0-initialization-sources/1"
        }),
        "initializer source registration differs",
    )?;
    exact(
        &e["source"]["source_commit"],
        json!(config.endpoint.source_commit),
        "initializer and registered source commit differ",
    )?;
    fields(&e["source"]["pins"], &INITIALIZER_FILES)?;
    for name in INITIALIZER_FILES {
        result
            .initializer_source_pins
            .insert(name.into(), sha(&e["source"]["pins"][name])?.into());
    }
    result.initializer_source_commit = Some(config.endpoint.source_commit.clone());
    validate_software(&e["software"], metadata)?;
    exact(
        &e["claims"],
        json!({"training_steps":0,"nn_forwards":0,"optimizer_steps":0,"trained":false,"diagnostic":false,"external_weights":false,"external_teacher":false,"learned_transfer_claim":false,"automatic_asset_conversion":false}),
        "initializer observation cannot grant learned transfer or execution acceptance",
    )?;
    for n in ["training_steps", "nn_forwards", "optimizer_steps"] {
        uint(&e["claims"][n])?;
    }
    exact(
        &e["acceptance"],
        json!({"checkpoint_tensor_validation":"finite_cpu_fp32_weights_only_inventory","independent_numeric_validation":"pending","export_validation":"pending","arena_eligibility":"not_granted","capability_scope":if legacy {"legacy_initializer_observation_only"} else {"initializer_return_and_file_identity_only"}}),
        "initializer evidence cannot grant independent GPU or arena acceptance",
    )
}

fn inventory(v: &Value, expected_values: u64) -> Result<Vec<String>, ManifestError> {
    let entries = array(v, 4096)?;
    need(!entries.is_empty(), "complete parameter inventory missing")?;
    let mut previous = "";
    let mut names = Vec::with_capacity(entries.len());
    let mut total = 0u64;
    for entry in entries {
        fields(entry, &["name", "shape", "dtype", "bytes", "sha256"])?;
        let name = text(&entry["name"])?;
        need(name > previous, "parameter names must be sorted and unique")?;
        previous = name;
        let mut count = 1u64;
        for dimension in array(&entry["shape"], 8)? {
            let size = uint(dimension)?;
            need(
                (1..=1048576).contains(&size),
                "parameter shape dimension is invalid",
            )?;
            count = count
                .checked_mul(size)
                .ok_or_else(|| fail("parameter shape overflow"))?;
        }
        need(
            count <= 1 << 28 && uint(&entry["bytes"])? == count * 4,
            "parameter shape and FP32 bytes differ",
        )?;
        exact(&entry["dtype"], json!("fp32_le"), "parameter dtype differs")?;
        sha(&entry["sha256"])?;
        total = total
            .checked_add(count)
            .ok_or_else(|| fail("parameter count overflow"))?;
        names.push(name.to_owned());
    }
    need(
        total == expected_values,
        "weights-only inventory does not cover the declared parameter count",
    )?;
    Ok(names)
}
fn validate_software(v: &Value, metadata: &Value) -> Result<(), ManifestError> {
    fields(
        v,
        &[
            "pytorch_version",
            "python_version",
            "python_implementation",
            "torch_git_version",
            "torch_cuda_build",
            "torch_thread_settings",
            "deterministic_algorithms",
            "tf32_matmul",
            "tf32_cudnn",
            "observation",
            "device_capability",
        ],
    )?;
    for k in ["pytorch_version", "python_version", "python_implementation"] {
        text(&v[k])?;
    }
    for k in ["torch_git_version", "torch_cuda_build"] {
        if !v[k].is_null() {
            text(&v[k])?;
        }
    }
    fields(&v["torch_thread_settings"], &["intra", "inter"])?;
    for k in ["intra", "inter"] {
        need(
            (1..=65536).contains(&uint(&v["torch_thread_settings"][k])?),
            "actual Torch thread count is invalid",
        )?;
    }
    for k in ["deterministic_algorithms", "tf32_matmul", "tf32_cudnn"] {
        need(
            v[k].is_boolean(),
            "software execution settings require boolean readback",
        )?;
    }
    exact(
        &v["pytorch_version"],
        metadata["torch_version"].clone(),
        "producer and actual Torch readback differ",
    )?;
    exact(
        &v["observation"],
        json!("already_loaded_torch_after_registered_initializer"),
        "software readback was not captured after initializer",
    )?;
    exact(
        &v["device_capability"],
        json!("not_probed"),
        "G0 initializer cannot claim actual GPU capability",
    )
}

fn validate_generation_spec(spec: &Value) -> Result<(), ManifestError> {
    fields(
        spec,
        &[
            "schema",
            "generation_id",
            "source_sha256",
            "dataset_sha256",
            "split_sha256",
            "model_sha256",
            "runtime_sha256",
            "profile",
            "model_epoch",
            "context_epoch",
            "device",
            "precision",
            "tf32",
            "cpu_threads",
            "microbatch",
            "accumulation",
            "optimizer",
            "role_updates",
            "checkpoint_interval",
            "seeds",
            "loss_weights",
            "eligibility",
            "collection_inference",
        ],
    )?;
    text(&spec["generation_id"])?;
    for k in [
        "source_sha256",
        "dataset_sha256",
        "split_sha256",
        "model_sha256",
        "runtime_sha256",
    ] {
        sha(&spec[k])?;
    }
    for k in [
        "model_epoch",
        "context_epoch",
        "cpu_threads",
        "microbatch",
        "accumulation",
        "checkpoint_interval",
    ] {
        uint(&spec[k])?;
    }
    for (k, expected) in [
        ("schema", json!(GENERATION_SPEC_DOMAIN)),
        ("profile", json!("virtual_review_v1")),
        ("device", json!("cuda:0")),
        ("precision", json!("fp32")),
        ("tf32", json!(false)),
        ("cpu_threads", json!(2)),
        ("microbatch", json!(1)),
        ("accumulation", json!(8)),
        ("checkpoint_interval", json!(250)),
        ("eligibility", json!("actual_training_pending_acceptance")),
        (
            "role_updates",
            json!({"proposer":2000,"critic":2000,"validator":1000}),
        ),
        (
            "seeds",
            json!({"initialization":23,"sampling":29,"training":31,"arena":1}),
        ),
        (
            "loss_weights",
            json!({"policy":1.0,"wdl":1.0,"divergence":1.0,"task":1.0}),
        ),
        (
            "optimizer",
            json!({"name":"adamw","lr":1e-4,"betas":[0.9,0.999],"epsilon":1e-8,"weight_decay":0.01,"clip_norm":1.0,"schedule":"constant"}),
        ),
    ] {
        exact(&spec[k], expected, "unregistered first-generation recipe")?;
    }
    for map in ["role_updates", "seeds"] {
        for v in spec[map].as_object().unwrap().values() {
            uint(v)?;
        }
    }
    exact(
        &spec["collection_inference"],
        json!({"mode":"fresh","batch_size":1}),
        "first generation cannot train on Warm or non-B1 collection inputs",
    )?;
    uint(&spec["collection_inference"]["batch_size"])?;
    // Python's recipe uses float-valued optimizer/loss slots. Do not accept an
    // integer-typed replacement merely because a numerical comparison is equal.
    for key in ["lr", "epsilon", "weight_decay", "clip_norm"] {
        need(
            spec["optimizer"][key].is_f64(),
            "optimizer scalar must retain its registered float type",
        )?;
    }
    for v in spec["optimizer"]["betas"]
        .as_array()
        .unwrap()
        .iter()
        .chain(spec["loss_weights"].as_object().unwrap().values())
    {
        need(
            v.is_f64(),
            "recipe scalar must retain its registered float type",
        )?;
    }
    Ok(())
}

/// Only registered recipe floats occur in the metadata digests below. Encode
/// those finite values with Python's exponent spelling; never treat a raw file
/// hash and this semantic digest as the same authority.
fn python_canonical(v: &Value, out: &mut String) -> Result<(), ManifestError> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_u64() || n.is_i64() {
                out.push_str(&n.to_string());
            } else {
                let f = n
                    .as_f64()
                    .ok_or_else(|| fail("finite canonical float required"))?;
                let spelling = [
                    (1e-4, "0.0001"),
                    (1e-8, "1e-08"),
                    (0.9, "0.9"),
                    (0.999, "0.999"),
                    (0.01, "0.01"),
                    (1.0, "1.0"),
                ]
                .into_iter()
                .find(|(x, _)| *x == f)
                .map(|(_, s)| s)
                .ok_or_else(|| {
                    fail("unregistered floating metadata cannot be canonically joined")
                })?;
                out.push_str(spelling);
            }
        }
        Value::String(s) => out.push_str(
            &serde_json::to_string(s).map_err(|_| fail("canonical text encoding failed"))?,
        ),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                python_canonical(x, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let sorted: BTreeMap<_, _> = map.iter().collect();
            for (i, (k, x)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(
                    &serde_json::to_string(k)
                        .map_err(|_| fail("canonical field encoding failed"))?,
                );
                out.push(':');
                python_canonical(x, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}
fn semantic_digest(domain: &str, v: &Value) -> Result<String, ManifestError> {
    let mut encoded = String::new();
    python_canonical(&json!([domain, v]), &mut encoded)?;
    Ok(format!("{:x}", Sha256::digest(encoded.as_bytes())))
}

fn validate_g1(
    p: &Value,
    registered_spec: &Value,
    config: &PalsAdaptiveStartupConfigV5,
    export: &Value,
) -> Result<(), ManifestError> {
    fields(
        p,
        &[
            "schema",
            "source_kind",
            "eligibility",
            "spec",
            "spec_sha256",
            "generation_id",
            "source_sha256",
            "runtime_sha256",
            "checkpoint",
            "pc_admission",
            "v_admission",
            "collection",
            "deployment",
            "numeric_status",
            "cuda_status",
            "arena_accepted",
        ],
    )?;
    exact(
        &p["schema"],
        json!(GENERATION_DOMAIN),
        "wrong actual-generation provenance domain",
    )?;
    exact(
        &p["source_kind"],
        json!("actual_selfplay"),
        "diagnostic corpus cannot become actual-generation provenance",
    )?;
    exact(
        &p["eligibility"],
        json!("evaluation_candidate_pending_gpu_acceptance"),
        "actual export cannot grant arena eligibility",
    )?;
    for k in ["numeric_status", "cuda_status"] {
        exact(
            &p[k],
            json!("not_run"),
            "producer provenance cannot substitute for independent numerical/CUDA acceptance",
        )?;
    }
    exact(
        &p["arena_accepted"],
        json!(false),
        "producer cannot self-accept arena results",
    )?;
    validate_generation_spec(registered_spec)?;
    exact(
        &p["spec"],
        registered_spec.clone(),
        "provenance specification differs from independently registered generation input",
    )?;
    let spec = &p["spec"];
    exact(
        &p["spec_sha256"],
        json!(semantic_digest(GENERATION_SPEC_DOMAIN, spec)?),
        "generation semantic spec digest differs",
    )?;
    for k in ["generation_id", "source_sha256", "runtime_sha256"] {
        exact(
            &p[k],
            spec[k].clone(),
            "generation source/runtime identity differs from preregistered spec",
        )?;
    }
    let checkpoint = &p["checkpoint"];
    validate_checkpoint_summary(
        checkpoint,
        spec,
        &config.endpoint.model.weights,
        config.endpoint.model.epoch,
    )?;
    for phase in ["pc", "v"] {
        validate_admission(&p[format!("{phase}_admission")], spec, phase, checkpoint)?;
    }
    need(
        p["pc_admission"]["opening_splits"] == p["v_admission"]["opening_splits"],
        "P/C and V corpus opening assignments differ",
    )?;
    for key in ["teacher_binary_sha256", "teacher_profile_sha256"] {
        need(
            p["pc_admission"]["producer"][key] == p["v_admission"]["producer"][key],
            "frozen generation changed teacher configuration",
        )?;
    }
    let mut total = uint(&checkpoint["bytes"])?;
    for kind in ["collection", "deployment"] {
        total = total
            .checked_add(validate_role_artifacts(&p[kind], kind)?)
            .ok_or_else(|| fail("generation output size overflow"))?;
    }
    need(
        total <= MANAGED_BYTES,
        "actual independent exports exceed managed output cap",
    )?;
    let kind = if config.endpoint.mode == PalsEngineModeV5::Collection {
        "collection"
    } else {
        "deployment"
    };
    let descriptors = &p[kind];
    identity_join(
        &descriptors["manifest"],
        artifact_for(config, Purpose::ExportManifest)?,
    )?;
    exact(
        &export["roles"],
        descriptors["roles"].clone(),
        "native manifest role bank differs from actual generation",
    )?;
    exact(
        &export["validator_present"],
        descriptors["validator_present"].clone(),
        "native manifest V ownership differs from actual generation",
    )?;
    let declarations = array(&export["graphs"], 3)?;
    let graph_pins = descriptors["graphs"].as_array().unwrap();
    need(
        declarations.len() == graph_pins.len(),
        "native graph count differs from actual producer",
    )?;
    for (declaration, pin) in declarations.iter().zip(graph_pins) {
        exact(
            &declaration["role"],
            pin["role"].clone(),
            "native graph role order differs",
        )?;
        let relative = text(&pin["file"])?;
        exact(
            &declaration["file"],
            json!(relative.rsplit('/').next().unwrap()),
            "native graph filename differs",
        )?;
        exact(
            &declaration["sha256"],
            pin["sha256"].clone(),
            "native graph byte hash differs",
        )?;
        let purpose = match text(&pin["role"])? {
            "public" => Purpose::PublicGraph,
            "shared_pc" => Purpose::SharedPcGraph,
            "validator" => Purpose::ValidatorGraph,
            _ => return Err(fail("unsupported graph role")),
        };
        identity_join(pin, artifact_for(config, purpose)?)?;
    }
    Ok(())
}

fn validate_checkpoint_summary(
    checkpoint: &Value,
    spec: &Value,
    weights: &ArtifactRef,
    registered_epoch: u64,
) -> Result<(), ManifestError> {
    fields(
        checkpoint,
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
    for (k, v) in [
        ("schema", json!("rz-pals-verified-generation-checkpoint/1")),
        ("checkpoint_domain", json!(CHECKPOINT_DOMAIN)),
        ("eligibility", json!("actual_training_pending_acceptance")),
        ("file", json!("checkpoint.pt")),
        ("phase", json!("complete")),
        ("pending_microbatches", json!(0)),
        (
            "nonzero_parameter_owners",
            json!(["shared", "proposer", "critic", "validator"]),
        ),
        ("cuda_devices", json!([0])),
    ] {
        exact(
            &checkpoint[k],
            v,
            "diagnostic, partial or empty checkpoint cannot become G1",
        )?;
    }
    uint(&checkpoint["pending_microbatches"])?;
    for dev in array(&checkpoint["cuda_devices"], 1)? {
        uint(dev)?;
    }
    need(
        (1..=MANAGED_BYTES).contains(&uint(&checkpoint["bytes"])?),
        "checkpoint exceeds managed output cap",
    )?;
    identity_join(checkpoint, weights)?;
    for name in ["completed_updates", "dispatched_updates"] {
        fields(&checkpoint[name], &ROLES)?;
    }
    for (role, expected) in UPDATES {
        need(
            uint(&checkpoint["completed_updates"][role])? == expected
                && uint(&checkpoint["dispatched_updates"][role])? >= expected,
            "actual update/dispatch count is incomplete or refunded",
        )?;
    }
    fields(&checkpoint["cost"], &COST_FIELDS)?;
    for key in COST_FIELDS {
        uint(&checkpoint["cost"][key])?;
    }
    let cost = &checkpoint["cost"];
    need(
        ["forward_calls", "backward_calls", "samples"]
            .iter()
            .all(|n| cost[*n].as_u64().unwrap_or(0) >= 40_000),
        "all-masked/unexecuted work cannot earn actual updates",
    )?;
    need(
        uint(&cost["backward_calls"])? <= uint(&cost["samples"])?
            && uint(&cost["samples"])? <= uint(&cost["forward_calls"])?
            && uint(&cost["wall_time_ns"])? > 0
            && uint(&cost["output_bytes"])? <= MANAGED_BYTES,
        "actual full accumulated work ledger is inconsistent",
    )?;
    let state = &checkpoint["state_sha256"];
    fields(
        state,
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
    for digest in state.as_object().unwrap().values() {
        sha(digest)?;
    }
    sha(&checkpoint["ledger_sha256"])?;
    sha(&checkpoint["config_sha256"])?;
    sha(&checkpoint["frozen_pc_sha256"])?;
    let frozen_epoch = uint(&checkpoint["frozen_pc_model_epoch"])?;
    need(
        frozen_epoch > uint(&spec["model_epoch"])?
            && frozen_epoch == registered_epoch
            && uint(&checkpoint["context_epoch"])? == uint(&spec["context_epoch"])?,
        "frozen P/C or model/context epoch differs",
    )?;
    exact(
        &checkpoint["config_sha256"],
        json!(semantic_digest(
            "rz-pals-generation-model-config/1",
            &compact_configuration()
        )?),
        "trained checkpoint belongs to another model configuration",
    )?;
    let ledger = json!({"spec":spec,"completed":checkpoint["completed_updates"],"dispatched":checkpoint["dispatched_updates"],"phase":"complete","next_role":"validator","pending_microbatches":0,"cost":cost});
    exact(
        &checkpoint["ledger_sha256"],
        json!(semantic_digest(
            "rz-pals-generation-ledger-state/1",
            &ledger
        )?),
        "earned update/cost ledger changed after verification",
    )?;
    Ok(())
}

fn validate_admission(
    a: &Value,
    spec: &Value,
    phase: &str,
    checkpoint: &Value,
) -> Result<(), ManifestError> {
    fields(
        a,
        &[
            "schema",
            "spec_sha256",
            "dataset_sha256",
            "frozen_pc_sha256",
            "phase",
            "coverage",
            "data_format",
            "producer",
            "opening_splits",
            "model_epoch",
            "context_epoch",
            "native_chain_verified",
            "inference_profile",
            "profile_exclusions",
            "actual_training_executed",
            "optimizer_updates",
            "arena_eligible",
        ],
    )?;
    for (k, v) in [
        ("schema", json!(GENERATION_SPEC_DOMAIN)),
        (
            "spec_sha256",
            json!(semantic_digest(GENERATION_SPEC_DOMAIN, spec)?),
        ),
        ("phase", json!(phase)),
        ("data_format", json!("rz-pals-generation-chunk-manifest/1")),
        ("native_chain_verified", json!(true)),
        ("actual_training_executed", json!(false)),
        ("optimizer_updates", json!(0)),
        ("arena_eligible", json!(false)),
    ] {
        exact(
            &a[k],
            v,
            "actual data admission differs from native-chain source",
        )?;
    }
    uint(&a["optimizer_updates"])?;
    sha(&a["dataset_sha256"])?;
    exact(
        &a["inference_profile"],
        json!({"mode":"fresh","batch_size":1}),
        "actual first-generation data must retain Fresh/B1 profile",
    )?;
    uint(&a["inference_profile"]["batch_size"])?;
    fields(
        &a["profile_exclusions"],
        &[
            "cross_split_records_excluded",
            "cross_split_identities",
            "duplicate_inputs_excluded",
            "duplicate_semantic_inputs_excluded",
            "warm_seed_inputs_excluded",
        ],
    )?;
    for count in a["profile_exclusions"].as_object().unwrap().values() {
        uint(count)?;
    }
    let expected_model = if phase == "pc" {
        &spec["model_sha256"]
    } else {
        &checkpoint["frozen_pc_sha256"]
    };
    let expected_epoch = if phase == "pc" {
        &spec["model_epoch"]
    } else {
        &checkpoint["frozen_pc_model_epoch"]
    };
    if phase == "pc" {
        exact(
            &a["dataset_sha256"],
            spec["dataset_sha256"].clone(),
            "P/C corpus is not registered G0 corpus",
        )?;
        need(
            a["frozen_pc_sha256"].is_null(),
            "P/C collection cannot claim a later frozen model",
        )?;
    } else {
        exact(
            &a["frozen_pc_sha256"],
            checkpoint["frozen_pc_sha256"].clone(),
            "V corpus belongs to another frozen P/C",
        )?;
    }
    need(
        uint(&a["model_epoch"])? == uint(expected_epoch)?
            && uint(&a["context_epoch"])? == uint(&spec["context_epoch"])?,
        "actual corpus model/context epoch differs",
    )?;
    let producer = &a["producer"];
    fields(
        producer,
        &[
            "schema",
            "source_kind",
            "source_sha256",
            "model_sha256",
            "runtime_sha256",
            "teacher_binary_sha256",
            "teacher_profile_sha256",
            "opening_split_sha256",
            "registered_before_outcomes",
            "actual_game_ids",
        ],
    )?;
    exact(
        &producer["schema"],
        json!("rz-pals-actual-collection-authority/2"),
        "renamed diagnostic collection authority is rejected",
    )?;
    exact(
        &producer["source_kind"],
        json!("actual_selfplay"),
        "collection was not actual selfplay",
    )?;
    exact(
        &producer["registered_before_outcomes"],
        json!(true),
        "opening split was not registered before outcomes",
    )?;
    for k in [
        "source_sha256",
        "model_sha256",
        "runtime_sha256",
        "teacher_binary_sha256",
        "teacher_profile_sha256",
        "opening_split_sha256",
    ] {
        sha(&producer[k])?;
    }
    for (k, v) in [
        ("source_sha256", &spec["source_sha256"]),
        ("runtime_sha256", &spec["runtime_sha256"]),
        ("opening_split_sha256", &spec["split_sha256"]),
        ("model_sha256", expected_model),
    ] {
        exact(
            &producer[k],
            v.clone(),
            "actual data source/runtime/model/split differs",
        )?;
    }
    let games = array(&producer["actual_game_ids"], 8192)?;
    need(
        !games.is_empty() && (phase != "pc" || games.len() == 128),
        "first actual generation must preserve 128 P/C games",
    )?;
    let mut unique = BTreeSet::new();
    for game in games {
        need(unique.insert(text(game)?), "actual game identity repeated")?;
    }
    let splits = a["opening_splits"]
        .as_object()
        .ok_or_else(|| fail("opening split object required"))?;
    need(
        splits.len() == 64,
        "actual generation requires 64 registered collection openings",
    )?;
    let mut counts = BTreeMap::new();
    for (k, v) in splits {
        text(&Value::String(k.clone()))?;
        let name = text(v)?;
        need(
            matches!(name, "train" | "validation" | "holdout"),
            "unsupported opening split",
        )?;
        *counts.entry(name).or_insert(0usize) += 1;
    }
    need(
        counts.get("train") == Some(&48)
            && counts.get("validation") == Some(&8)
            && counts.get("holdout") == Some(&8),
        "actual corpus split is not 48/8/8",
    )?;
    exact(
        &spec["split_sha256"],
        json!(semantic_digest(
            "rz-pals-opening-split/1",
            &a["opening_splits"]
        )?),
        "opening mapping differs from registered split pin",
    )?;
    let coverage = a["coverage"]
        .as_object()
        .ok_or_else(|| fail("bounded target coverage evidence required"))?;
    need(coverage.len() <= 128, "target coverage exceeded bound")?;
    for (k, v) in coverage {
        text(&Value::String(k.clone()))?;
        uint(v)?;
    }
    let minima: &[(&str, u64)] = if phase == "pc" {
        &[
            ("proposer_valid_samples", 1024),
            ("critic_valid_samples", 1024),
        ]
    } else {
        &[
            ("validator_known_pairs", 128),
            ("validator_new_witness_contexts", 32),
        ]
    };
    for (k, minimum) in minima {
        need(
            a["coverage"][*k].as_u64().unwrap_or(0) >= *minimum,
            "InsufficientValidTargets: cannot replace actual targets with fixture/all-masked rows",
        )?;
    }
    if phase == "v" {
        let unique_inputs = uint(&a["coverage"]["validator_unique_inputs"])?;
        let witness_contexts = uint(&a["coverage"]["validator_new_witness_contexts"])?;
        need(
            (32..=8192).contains(&unique_inputs) && witness_contexts <= unique_inputs,
            "actual V collection context cap exceeded",
        )?;
    }
    Ok(())
}
fn validate_role_artifacts(a: &Value, kind: &str) -> Result<u64, ManifestError> {
    fields(a, &["roles", "validator_present", "manifest", "graphs"])?;
    let include_v = kind == "collection";
    exact(
        &a["roles"],
        if include_v {
            json!(ROLES)
        } else {
            json!(["proposer", "critic"])
        },
        "actual export bank differs from Collection/Deployment mode",
    )?;
    exact(
        &a["validator_present"],
        json!(include_v),
        "Deployment cannot contain a private V bank",
    )?;
    file_pin(
        &a["manifest"],
        &format!("{kind}/export.json"),
        METADATA_LIMIT as u64,
    )?;
    let expected = if include_v {
        vec![
            ("public", "public_memory.onnx"),
            ("shared_pc", "shared_pc_if.onnx"),
            ("validator", "role_validator.onnx"),
        ]
    } else {
        vec![
            ("public", "public_memory.onnx"),
            ("shared_pc", "shared_pc_if.onnx"),
        ]
    };
    let graphs = array(&a["graphs"], 3)?;
    need(
        graphs.len() == expected.len(),
        "actual graph ownership count differs",
    )?;
    let mut total = uint(&a["manifest"]["bytes"])?;
    for (pin, (role, file)) in graphs.iter().zip(expected) {
        fields(pin, &["role", "file", "sha256", "bytes"])?;
        exact(&pin["role"], json!(role), "actual graph role order differs")?;
        let identity = json!({"file":pin["file"],"sha256":pin["sha256"],"bytes":pin["bytes"]});
        file_pin(&identity, &format!("{kind}/{file}"), MANAGED_BYTES)?;
        total = total
            .checked_add(uint(&pin["bytes"])?)
            .ok_or_else(|| fail("graph output size overflow"))?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec_fixture() -> Value {
        json!({"schema":GENERATION_SPEC_DOMAIN,"generation_id":"g1","source_sha256":"a".repeat(64),"dataset_sha256":"b".repeat(64),"split_sha256":"c".repeat(64),"model_sha256":"d".repeat(64),"runtime_sha256":"e".repeat(64),"profile":"virtual_review_v1","model_epoch":0,"context_epoch":0,"device":"cuda:0","precision":"fp32","tf32":false,"cpu_threads":2,"microbatch":1,"accumulation":8,"optimizer":{"name":"adamw","lr":1e-4,"betas":[0.9,0.999],"epsilon":1e-8,"weight_decay":0.01,"clip_norm":1.0,"schedule":"constant"},"role_updates":{"proposer":2000,"critic":2000,"validator":1000},"checkpoint_interval":250,"seeds":{"initialization":23,"sampling":29,"training":31,"arena":1},"loss_weights":{"policy":1.0,"wdl":1.0,"divergence":1.0,"task":1.0},"eligibility":"actual_training_pending_acceptance","collection_inference":{"mode":"fresh","batch_size":1}})
    }
    fn weights_fixture() -> ArtifactRef {
        ArtifactRef {
            path: "fixture/checkpoint.pt".into(),
            sha256: "f".repeat(64),
            bytes: 64,
            source: "https://example.org/authored-metadata-fixture".into(),
            license: "MIT".into(),
        }
    }
    fn checkpoint_fixture(spec: &Value) -> Value {
        let weights = weights_fixture();
        let cost = json!({"forward_calls":40_000,"backward_calls":40_000,"samples":40_000,"wall_time_ns":1,"partial_accumulation_ns":0,"replay_ns":0,"masked_groups":0,"output_bytes":64});
        let updates = json!({"proposer":2000,"critic":2000,"validator":1000});
        let state_names = [
            "model",
            "optimizers",
            "schedulers",
            "rng",
            "sampler",
            "parameter_updates",
            "membership",
            "bucket",
        ];
        let state: BTreeMap<_, _> = state_names
            .into_iter()
            .map(|n| (n, "b".repeat(64)))
            .collect();
        let ledger = json!({"spec":spec,"completed":updates,"dispatched":updates,"phase":"complete","next_role":"validator","pending_microbatches":0,"cost":cost});
        json!({"schema":"rz-pals-verified-generation-checkpoint/1","checkpoint_domain":CHECKPOINT_DOMAIN,"eligibility":"actual_training_pending_acceptance","file":"checkpoint.pt","sha256":weights.sha256,"bytes":weights.bytes,"phase":"complete","pending_microbatches":0,"completed_updates":updates,"dispatched_updates":updates,"ledger_sha256":semantic_digest("rz-pals-generation-ledger-state/1",&ledger).unwrap(),"cost":cost,"state_sha256":state,"config_sha256":semantic_digest("rz-pals-generation-model-config/1",&compact_configuration()).unwrap(),"frozen_pc_sha256":"a".repeat(64),"frozen_pc_model_epoch":1,"context_epoch":0,"nonzero_parameter_owners":["shared","proposer","critic","validator"],"cuda_devices":[0]})
    }
    #[test]
    fn actual_recipe_keeps_unsigned_counts_and_registered_float_slots() {
        let spec = spec_fixture();
        validate_generation_spec(&spec).unwrap();
        for (path, value) in [
            ("cpu_threads", json!(true)),
            ("accumulation", json!(1)),
            ("tf32", json!(true)),
        ] {
            let mut bad = spec.clone();
            bad[path] = value;
            assert!(validate_generation_spec(&bad).is_err());
        }
        let mut bad = spec.clone();
        bad["optimizer"]["clip_norm"] = json!(1);
        assert!(validate_generation_spec(&bad).is_err());
        let mut bad = spec;
        bad["unexpected"] = json!(false);
        assert!(validate_generation_spec(&bad).is_err());
    }
    #[test]
    fn all_masked_partial_diagnostic_or_changed_checkpoint_summary_is_rejected() {
        let spec = spec_fixture();
        let good = checkpoint_fixture(&spec);
        validate_checkpoint_summary(&good, &spec, &weights_fixture(), 1).unwrap();
        for (name, value) in [
            (
                "checkpoint_domain",
                json!("rz-pals-diagnostic-checkpoint/1"),
            ),
            ("pending_microbatches", json!(1)),
            ("phase", json!("v")),
            ("ledger_sha256", json!("c".repeat(64))),
            ("frozen_pc_model_epoch", json!(0)),
        ] {
            let mut bad = good.clone();
            bad[name] = value;
            assert!(validate_checkpoint_summary(&bad, &spec, &weights_fixture(), 1).is_err());
        }
        let mut bad = good.clone();
        bad["completed_updates"]["proposer"] = json!(0);
        assert!(validate_checkpoint_summary(&bad, &spec, &weights_fixture(), 1).is_err());
        let mut bad = good.clone();
        bad["cost"]["samples"] = json!(0);
        assert!(validate_checkpoint_summary(&bad, &spec, &weights_fixture(), 1).is_err());
        let mut bad = good;
        bad["state_sha256"]
            .as_object_mut()
            .unwrap()
            .remove("optimizers");
        assert!(validate_checkpoint_summary(&bad, &spec, &weights_fixture(), 1).is_err());
    }
    fn admission_fixture(phase: &str) -> (Value, Value, Value) {
        let mut spec = spec_fixture();
        let split: BTreeMap<_, _> = (0..64)
            .map(|i| {
                (
                    format!("opening-{i:02}"),
                    if i < 48 {
                        "train"
                    } else if i < 56 {
                        "validation"
                    } else {
                        "holdout"
                    },
                )
            })
            .collect();
        let split = serde_json::to_value(split).unwrap();
        spec["split_sha256"] = json!(semantic_digest("rz-pals-opening-split/1", &split).unwrap());
        let checkpoint = checkpoint_fixture(&spec);
        let coverage = if phase == "pc" {
            json!({"proposer_valid_samples":1024,"critic_valid_samples":1024})
        } else {
            json!({"validator_known_pairs":128,"validator_new_witness_contexts":32,"validator_unique_inputs":32})
        };
        let model = if phase == "pc" {
            spec["model_sha256"].clone()
        } else {
            checkpoint["frozen_pc_sha256"].clone()
        };
        let games: Vec<_> = (0..if phase == "pc" { 128 } else { 1 })
            .map(|i| format!("game-{i:03}"))
            .collect();
        let admission = json!({"schema":GENERATION_SPEC_DOMAIN,"spec_sha256":semantic_digest(GENERATION_SPEC_DOMAIN,&spec).unwrap(),"dataset_sha256":if phase=="pc"{spec["dataset_sha256"].clone()}else{json!("a".repeat(64))},"frozen_pc_sha256":if phase=="pc"{Value::Null}else{checkpoint["frozen_pc_sha256"].clone()},"phase":phase,"coverage":coverage,"data_format":"rz-pals-generation-chunk-manifest/1","producer":{"schema":"rz-pals-actual-collection-authority/2","source_kind":"actual_selfplay","source_sha256":spec["source_sha256"],"model_sha256":model,"runtime_sha256":spec["runtime_sha256"],"teacher_binary_sha256":"c".repeat(64),"teacher_profile_sha256":"d".repeat(64),"opening_split_sha256":spec["split_sha256"],"registered_before_outcomes":true,"actual_game_ids":games},"opening_splits":split,"model_epoch":if phase=="pc"{0}else{1},"context_epoch":0,"native_chain_verified":true,"inference_profile":{"mode":"fresh","batch_size":1},"profile_exclusions":{"cross_split_records_excluded":0,"cross_split_identities":0,"duplicate_inputs_excluded":0,"duplicate_semantic_inputs_excluded":0,"warm_seed_inputs_excluded":0},"actual_training_executed":false,"optimizer_updates":0,"arena_eligible":false});
        (admission, spec, checkpoint)
    }
    #[test]
    fn actual_targets_keep_fresh_profile_unique_context_and_preoutcome_split() {
        for phase in ["pc", "v"] {
            let (good, spec, checkpoint) = admission_fixture(phase);
            validate_admission(&good, &spec, phase, &checkpoint).unwrap();
            let mut bad = good.clone();
            bad["inference_profile"]["mode"] = json!("approx_warm");
            assert!(validate_admission(&bad, &spec, phase, &checkpoint).is_err());
            let mut bad = good.clone();
            bad["producer"]["registered_before_outcomes"] = json!(false);
            assert!(validate_admission(&bad, &spec, phase, &checkpoint).is_err());
            let mut bad = good.clone();
            bad["coverage"] = json!({});
            assert!(validate_admission(&bad, &spec, phase, &checkpoint).is_err());
            if phase == "v" {
                for value in [json!(0), json!(null), json!(8193)] {
                    let mut bad = good.clone();
                    bad["coverage"]["validator_unique_inputs"] = value;
                    assert!(validate_admission(&bad, &spec, phase, &checkpoint).is_err());
                }
            }
        }
    }
    #[test]
    fn canonical_recipe_float_digest_does_not_silently_use_rust_exponent_spelling() {
        let mut out = String::new();
        python_canonical(&json!({"epsilon":1e-8,"clip":1.0,"integer":1}), &mut out).unwrap();
        assert_eq!(out, "{\"clip\":1.0,\"epsilon\":1e-08,\"integer\":1}");
        assert!(python_canonical(&json!(0.00123), &mut String::new()).is_err());
    }
    #[test]
    fn producer_inventory_is_complete_ordered_and_does_not_validate_tensor_content() {
        let v = json!([{"name":"a","shape":[2,3],"dtype":"fp32_le","bytes":24,"sha256":"a".repeat(64)},{"name":"b","shape":[],"dtype":"fp32_le","bytes":4,"sha256":"b".repeat(64)}]);
        assert_eq!(inventory(&v, 7).unwrap(), vec!["a", "b"]);
        assert!(inventory(&v, 6).is_err());
        let mut bad = v.clone();
        bad[0]["shape"][0] = json!(true);
        assert!(inventory(&bad, 7).is_err());
        let mut bad = v;
        bad[1]["name"] = json!("a");
        assert!(inventory(&bad, 7).is_err());
    }
    #[test]
    fn duplicate_metadata_keys_are_not_hidden_by_value_projection() {
        assert!(parse_bytes(br#"{"schema":"g0","schema":"g1"}"#).is_err());
    }
    #[test]
    fn deployment_descriptor_cannot_carry_validator_or_unpinned_graph() {
        let v = json!({"roles":["proposer","critic"],"validator_present":false,"manifest":{"file":"deployment/export.json","bytes":100,"sha256":"a".repeat(64)},"graphs":[{"role":"public","file":"deployment/public_memory.onnx","sha256":"b".repeat(64),"bytes":20},{"role":"shared_pc","file":"deployment/shared_pc_if.onnx","sha256":"c".repeat(64),"bytes":30}]});
        assert_eq!(validate_role_artifacts(&v, "deployment").unwrap(), 150);
        let mut bad = v.clone();
        bad["validator_present"] = json!(true);
        assert!(validate_role_artifacts(&bad, "deployment").is_err());
        let mut bad = v;
        bad["graphs"][0]["bytes"] = json!(0);
        assert!(validate_role_artifacts(&bad, "deployment").is_err());
    }
}
