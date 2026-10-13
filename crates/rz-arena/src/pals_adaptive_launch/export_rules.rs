//! Pure model-content and exact Rules-declaration joins. No runtime is loaded.
use super::*;
use rz_experiments::PalsNumericalScopeV5;

#[cfg(any(feature = "native-cuda", feature = "pals-collection-onnx"))]
struct ExportContent {
    config: rz_eval::pals_model::PalsModelConfig,
    rules_profile: String,
    rules_semantic: [u8; 32],
    rules_source: [u8; 32],
}

/// Historical FullLineInteractionV2 artifacts keep their original export
/// schema/layout. Compact V3 has a separate strict parser and capability.
#[cfg(any(feature = "native-cuda", feature = "pals-collection-onnx"))]
pub fn validate_export_content_v5(
    config: &PalsAdaptiveStartupConfigV5,
    bytes: &[u8],
) -> Result<(), ArenaError> {
    checked_export(&PalsNumericalScopeV5::from_config(config), bytes).map(|_| ())
}

#[cfg(any(feature = "native-cuda", feature = "pals-collection-onnx"))]
fn checked_export(
    config: &PalsNumericalScopeV5,
    bytes: &[u8],
) -> Result<ExportContent, ArenaError> {
    use rz_eval::pals_onnx::virtual_review::{
        VirtualReviewExecutionMode, validate_pals_legacy_export_manifest,
        validate_virtual_review_export_manifest,
    };
    let m = &config.model;
    if m.profile == "virtual_review_v1" {
        require(
            m.semantic_sha256
                == hex(rz_eval::pals_model::PalsModelProfile::VirtualReviewV1
                    .registered_semantics_sha256())
                && m.encoding_sha256
                    == hex(rz_eval::pals_model::virtual_review::registered_encoding_sha256()),
            "compact model/encoding registry identity differs from the compiled semantic and tensor ABI registries",
        )?;
    }
    let collection = config.mode == PalsEngineModeV5::Collection;
    let mode = if collection {
        VirtualReviewExecutionMode::Collection
    } else {
        VirtualReviewExecutionMode::Deployment
    };
    // Normalize only this reader's local observation, never the asset bytes,
    // schema, profile, or encoding identity.
    let (
        actual_sha,
        actual_len,
        model_config,
        semantics,
        encoding,
        profile,
        checkpoint,
        roles,
        validator_present,
        trained,
        training_steps,
        rules_profile,
        rules_semantic,
        rules_source,
        graphs,
    ) = if m.profile == "virtual_review_v1" {
        let s = validate_virtual_review_export_manifest(bytes, mode)
            .map_err(|e| invalid(&format!("strict compact export: {e}")))?;
        (
            s.manifest_sha256(),
            s.manifest_bytes(),
            s.config,
            s.model_semantics,
            s.encoding_schema,
            s.model_profile,
            s.checkpoint_sha256,
            s.roles,
            s.validator_present,
            s.trained,
            s.training_steps,
            s.rules_input_profile,
            s.rules_input_semantic_sha256,
            s.rules_encoder_source_sha256,
            s.graphs
                .into_iter()
                .map(|g| (g.role, g.file, g.sha256))
                .collect::<Vec<_>>(),
        )
    } else if m.profile == "full_line_interaction_v2" {
        let s = validate_pals_legacy_export_manifest(bytes, mode)
            .map_err(|e| invalid(&format!("strict historical V2 export: {e}")))?;
        (
            s.manifest_sha256(),
            s.manifest_bytes(),
            s.config,
            s.model_semantics,
            s.encoding_schema,
            s.model_profile,
            s.checkpoint_sha256,
            s.roles,
            s.validator_present,
            s.trained,
            s.training_steps,
            s.rules_input_profile
                .ok_or_else(|| invalid("legacy export lacks exact Rules profile"))?,
            s.rules_input_semantic_sha256
                .ok_or_else(|| invalid("legacy export lacks Rules semantics"))?,
            s.rules_encoder_source_sha256
                .ok_or_else(|| invalid("legacy export lacks Rules source"))?,
            s.graphs
                .into_iter()
                .map(|g| (g.role, g.file, g.sha256))
                .collect::<Vec<_>>(),
        )
    } else {
        return Err(invalid("unsupported actual model profile"));
    };
    let export = config.asset(PalsStartupAssetPurposeV5::ExportManifest)?;
    require(
        hex(actual_sha) == export.artifact.sha256 && actual_len as u64 == export.artifact.bytes,
        "actual export differs from registered bounded artifact",
    )?;
    require(
        semantics == m.model_semantics
            && encoding == m.encoding_schema
            && profile.as_str() == m.profile
            && hex(checkpoint) == m.weights.sha256
            && model_config.width == m.width as usize
            && model_config.latent_slots == m.latent_count as usize
            && model_config.max_records == m.max_records as usize,
        "actual export model/input/configuration/weights differs",
    )?;
    let expected = if collection {
        vec!["proposer", "critic", "validator"]
    } else {
        vec!["proposer", "critic"]
    };
    require(
        roles.iter().map(String::as_str).collect::<Vec<_>>() == expected
            && validator_present == collection,
        "export violates PCV/V-free capability",
    )?;
    let mut seen = BTreeSet::new();
    for (role, file, sha) in graphs {
        let purpose = match role.as_str() {
            "public" => PalsStartupAssetPurposeV5::PublicGraph,
            "shared_pc" => PalsStartupAssetPurposeV5::SharedPcGraph,
            "proposer" if !collection => PalsStartupAssetPurposeV5::ProposerGraph,
            "critic" if !collection => PalsStartupAssetPurposeV5::CriticGraph,
            "validator" if collection => PalsStartupAssetPurposeV5::ValidatorGraph,
            _ => return Err(invalid("forbidden/undeclared graph capability")),
        };
        require(seen.insert(role), "graph role duplicated")?;
        let a = config.asset(purpose)?;
        require(
            hex(sha) == a.artifact.sha256
                && Some(file.as_str())
                    == Path::new(&a.snapshot_path)
                        .file_name()
                        .and_then(|s| s.to_str()),
            "export graph name/hash differs from separately pinned actual asset",
        )?;
    }
    let expected_graphs = if collection {
        3
    } else if seen.contains("shared_pc") {
        2
    } else {
        3
    };
    require(
        seen.len() == expected_graphs,
        "export graph capability set incomplete",
    )?;
    match m.artifact_domain {
        PalsArtifactDomainV5::Untrained => require(
            !trained && training_steps == 0,
            "untrained export contains optimizer claims",
        )?,
        PalsArtifactDomainV5::RealGeneration => require(
            trained && training_steps == 5000,
            "real-generation export lacks registered complete update count",
        )?,
        PalsArtifactDomainV5::FrozenPcCollection => require(
            collection && trained && training_steps == 4000,
            "frozen P/C export must retain its 4000-update Collection-only domain",
        )?,
        PalsArtifactDomainV5::Diagnostic => {
            return Err(invalid(
                "diagnostic export is not a real paired-game candidate",
            ));
        }
    }
    // Trained flags/counts are structural checks only. Independent registered
    // generation provenance and actual numeric evidence remain mandatory.
    Ok(ExportContent {
        config: model_config,
        rules_profile,
        rules_semantic,
        rules_source,
    })
}

/// Actual post-prepare numerical scope; no endpoint or startup lock is invented.
#[cfg(any(feature = "native-cuda", feature = "pals-collection-onnx"))]
pub fn validate_scope_export_content_v5(
    scope: &PalsNumericalScopeV5,
    bytes: &[u8],
) -> Result<(), ArenaError> {
    scope.validate_pre_acceptance()?;
    checked_export(scope, bytes).map(|_| ())
}

#[cfg(any(feature = "native-cuda", feature = "pals-collection-onnx"))]
pub fn validate_export_rules_content_v5(
    config: &PalsAdaptiveStartupConfigV5,
    export: &[u8],
    rules: &[u8],
) -> Result<(), ArenaError> {
    checked_export_rules(&PalsNumericalScopeV5::from_config(config), export, rules)
}

#[cfg(any(feature = "native-cuda", feature = "pals-collection-onnx"))]
pub fn validate_scope_export_rules_content_v5(
    scope: &PalsNumericalScopeV5,
    export: &[u8],
    rules: &[u8],
) -> Result<(), ArenaError> {
    scope.validate_pre_acceptance()?;
    checked_export_rules(scope, export, rules)
}
#[cfg(any(feature = "native-cuda", feature = "pals-collection-onnx"))]
fn checked_export_rules(
    config: &PalsNumericalScopeV5,
    export: &[u8],
    rules: &[u8],
) -> Result<(), ArenaError> {
    let e = checked_export(config, export)?;
    let a = config.asset(PalsStartupAssetPurposeV5::RulesDescriptor)?;
    require(
        rules.len() as u64 == a.artifact.bytes
            && format!("{:x}", Sha256::digest(rules)) == a.artifact.sha256,
        "actual Rules descriptor differs from registration",
    )?;
    if config.model.profile == "virtual_review_v1" {
        let r =
            rz_uci::pals_native::virtual_encoding::validate_pals_virtual_rules_descriptor_bytes(
                rules,
            )
            .map_err(|e| invalid(&format!("compiled compact Rules join: {e}")))?;
        require(
            r.config == e.config
                && r.semantic_sha256 == e.rules_semantic
                && r.encoder_source_sha256 == e.rules_source
                && e.rules_profile
                    == rz_uci::pals_native::virtual_encoding::PALS_VIRTUAL_RULES_ENCODING,
            "compiled Rules descriptor and model export differ",
        )
    } else {
        let r = rz_uci::pals_native::validate_pals_legacy_rules_descriptor_bytes(rules, &e.config)
            .map_err(|error| invalid(&format!("compiled historical Rules join: {error}")))?;
        require(
            r.config == e.config
                && r.semantic_sha256 == e.rules_semantic
                && r.encoder_source_sha256 == e.rules_source
                && e.rules_profile == rz_uci::pals_native::pals_rules_profile_for_config(&e.config),
            "compiled historical Rules descriptor and unchanged model export differ",
        )
    }
}

#[cfg(not(any(feature = "native-cuda", feature = "pals-collection-onnx")))]
pub fn validate_export_content_v5(
    _: &PalsAdaptiveStartupConfigV5,
    _: &[u8],
) -> Result<(), ArenaError> {
    Err(invalid(
        "native export admission requires explicit ONNX capability",
    ))
}
#[cfg(not(any(feature = "native-cuda", feature = "pals-collection-onnx")))]
pub fn validate_export_rules_content_v5(
    _: &PalsAdaptiveStartupConfigV5,
    _: &[u8],
    _: &[u8],
) -> Result<(), ArenaError> {
    Err(invalid(
        "native Rules admission requires explicit ONNX capability",
    ))
}

#[cfg(not(any(feature = "native-cuda", feature = "pals-collection-onnx")))]
pub fn validate_scope_export_content_v5(
    _: &PalsNumericalScopeV5,
    _: &[u8],
) -> Result<(), ArenaError> {
    Err(invalid(
        "native export admission requires explicit ONNX capability",
    ))
}

#[cfg(not(any(feature = "native-cuda", feature = "pals-collection-onnx")))]
pub fn validate_scope_export_rules_content_v5(
    _: &PalsNumericalScopeV5,
    _: &[u8],
    _: &[u8],
) -> Result<(), ArenaError> {
    Err(invalid(
        "native Rules admission requires explicit ONNX capability",
    ))
}
