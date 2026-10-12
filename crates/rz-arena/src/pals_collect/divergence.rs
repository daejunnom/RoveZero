//! Before-submit causal binding for the actual native divergence input.
//! Legality and features remain owned by the existing Rules/Native encoder.
use super::*;
use rz_eval::pals_model::PalsRole;
use rz_search::pals::engine::DivergenceQuery;
use rz_uci::pals_native::{pals_history_digest, prepare_divergence_input};

pub(super) const CONTEXT_VERSION: &str = "rz-pals-native-divergence-context/1";
pub(super) const CONTEXT_ARTIFACT: &str = "native-divergence-contexts.jsonl";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeDivergenceSite {
    pub slot: u32,
    /// Zero-based ply in the root-anchored proposal, before its original move.
    pub divergence_ply: u32,
    pub prefix_rules_state_sha256: String,
    pub prefix_rules_history_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeDivergenceContext {
    pub version: String,
    pub input_sha256: String,
    pub native_request: [u64; 2],
    /// Existing canonical sidecar seal; exact row bytes stay in the journal.
    pub tensor_sidecar_sha256: String,
    pub captured_input_revision: u64,
    pub proposal_move16: Vec<u16>,
    pub challenged_line_sha256: String,
    pub divergence_sites: Vec<NativeDivergenceSite>,
    pub sha256: String,
}

pub(super) fn prepare_context(
    id: RequestId,
    input: &PalsFrozenInput,
    sidecar: &PalsNativeInputSidecar,
    prepared: &PalsModelInput,
    query: &DivergenceQuery<'_>,
) -> Result<NativeDivergenceContext, RoleError> {
    // This encoder verifies bounded, unique opponent-turn plies and the legal
    // move order. Only immutable divergence features are compared: its newly
    // sampled remaining-time query must never replace the captured query.
    let expected = prepare_divergence_input(query, prepared.model_epoch)?;
    if prepared.role != PalsRole::Critic
        || !prepared.candidates.is_empty()
        || prepared.situation_revision != query.revision
        || input.snapshot().input_revision != query.revision
        || sidecar.input_sha256 != input.sha256()
        || prepared.history_digest != expected.history_digest
        || prepared.divergence_features.len() != expected.divergence_features.len()
        || !prepared
            .divergence_features
            .iter()
            .zip(&expected.divergence_features)
            .all(|(actual, expected)| {
                actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
            })
    {
        return Err(role_error(
            "native divergence context differs from its actual prepared input",
        ));
    }
    let root_state = state_sha(query.root).map_err(role_error)?;
    if input.snapshot().rules_state_sha256 != root_state
        || input.snapshot().rules_history_sha256 != hex(expected.history_digest)
    {
        return Err(role_error("native divergence root Rules identity mismatch"));
    }
    // Reuse the Rules-owner terminal/legality check as well as make/unmake.
    replay_line(query.root, query.proposal).map_err(role_error)?;
    let proposal_move16 = pack(query.proposal).map_err(role_error)?;
    let challenged_line_sha256 =
        canonical_sha256(&("rz-pals-challenged-line/1", &root_state, &proposal_move16))
            .map_err(role_error)?;
    let mut divergence_sites = Vec::with_capacity(query.candidates.len());
    for (slot, &ply) in query.candidates.iter().enumerate() {
        // The successful encoder above checked range/uniqueness before this slice.
        let (prefix, _) = replay_line(query.root, &query.proposal[..ply]).map_err(role_error)?;
        if prefix.side_to_move() == query.root.side_to_move() {
            return Err(role_error("native divergence site is not an opponent turn"));
        }
        divergence_sites.push(NativeDivergenceSite {
            slot: u32::try_from(slot).map_err(role_error)?,
            divergence_ply: u32::try_from(ply).map_err(role_error)?,
            prefix_rules_state_sha256: state_sha(&prefix).map_err(role_error)?,
            prefix_rules_history_sha256: hex(pals_history_digest(&prefix)?),
        });
    }
    let mut context = NativeDivergenceContext {
        version: CONTEXT_VERSION.into(),
        input_sha256: input.sha256().into(),
        native_request: [id.epoch.0, id.sequence],
        tensor_sidecar_sha256: sidecar.sha256.clone(),
        captured_input_revision: query.revision,
        proposal_move16,
        challenged_line_sha256,
        divergence_sites,
        sha256: String::new(),
    };
    let mut body = serde_json::to_value(&context).map_err(role_error)?;
    let _ = body
        .as_object_mut()
        .ok_or_else(|| role_error("native divergence context is not an object"))?
        .remove("sha256");
    context.sha256 = canonical_sha256(&(CONTEXT_VERSION, body)).map_err(role_error)?;
    Ok(context)
}
