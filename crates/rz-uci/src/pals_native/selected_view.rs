//! Encode already selected immutable rows, with no second latest-kind scan.
use super::*;
use rz_position::PositionLimits;
use rz_search::pals::review_context::ReviewView;

pub(super) fn validate_selected_query(
    view: &ReviewView,
    query: &RoleQuery<'_>,
) -> Result<(), RoleError> {
    if view.anchor_mode() != rz_search::pals::review_context::ReviewAnchorMode::CurrentRootOnly {
        return Err(RoleError::Unsupported(
            "V2 sixteen-feature input does not consume compact eight-feature view semantics",
        ));
    }
    let selected = view.query();
    if query.prefix != selected.prefix
        || query.proposal != selected.proposal
        || query.counterexample != selected.counterexample.as_deref()
        || query.legal != selected.legal
    {
        return Err(RoleError::InvalidOutput);
    }
    let anchor = view.anchors().first().ok_or(RoleError::InvalidOutput)?;
    let mut exact = Position::from_snapshot(&anchor.snapshot, PositionLimits::default())
        .map_err(model_error)?;
    for movement in &selected.prefix {
        exact.make_move(*movement).map_err(model_error)?;
    }
    if !query.position.snapshot().same_state(&exact.snapshot()) {
        return Err(RoleError::InvalidOutput);
    }
    if !view.records().is_empty() && !query.position.snapshot().same_state(&anchor.snapshot) {
        return Err(RoleError::Unsupported(
            "V2 has no anchor tensor: root-relative selected records cannot be relabeled at a virtual prefix endpoint",
        ));
    }
    // Legal-looking geometry is insufficient to transfer an older anchor's
    // meaning. Records admitted above now share the exact input board/history.
    for record in view.records() {
        let mut endpoint = query.position.clone();
        for movement in &record.line {
            endpoint.make_move(*movement).map_err(model_error)?;
        }
    }
    Ok(())
}

pub(super) fn selected_records(
    view: &ReviewView,
    revision: u64,
    config: &PalsModelConfig,
) -> Result<EncodedPalsRecords, RoleError> {
    if view.anchor_mode() != rz_search::pals::review_context::ReviewAnchorMode::CurrentRootOnly {
        return Err(RoleError::Unsupported(
            "V2 input does not consume compact view feature semantics",
        ));
    }
    if view.records().len() > 64
        || view.records().len() > config.max_records
        || view.records().len() != view.records_metadata().len()
        || view
            .records_metadata()
            .iter()
            .any(|metadata| metadata.anchor != 0)
    {
        return Err(RoleError::Backend("selected V2 input cannot interpret old anchor boards; compact V3 requires its explicit adapter".into()));
    }
    let mut tokens = Vec::with_capacity(view.records().len());
    let mut required = Vec::new();
    let mut lines = config
        .profile
        .uses_full_line()
        .then(|| Vec::with_capacity(view.records().len()));
    for (index, (record, metadata)) in view
        .records()
        .iter()
        .zip(view.records_metadata())
        .enumerate()
    {
        if record.revision > revision
            || record.line.len() > 256
            || [metadata.parent, metadata.supersedes]
                .into_iter()
                .flatten()
                .any(|reference| {
                    usize::from(reference) >= view.records().len()
                        || usize::from(reference) == index
                })
        {
            return Err(RoleError::InvalidOutput);
        }
        let id = index as u64 + 1;
        let mut features = [0.0; 16];
        features[0] = match record.kind {
            RecordKind::Proposal => 0.0,
            RecordKind::Counterexample => 1.0,
            RecordKind::Repair => 2.0,
            RecordKind::CpuVerification => 3.0,
        };
        features[1] = flag(record.value.is_some());
        features[2] = record.value.unwrap_or(0) as f32 / 30_000.0;
        features[3] = f32::from(record.completed_depth) / 64.0;
        features[4] = match record.score_scope {
            None => -1.0,
            Some(CpuScoreScope::FrontierOnly) => 0.0,
            Some(CpuScoreScope::CompletedIteration) => 1.0,
            Some(CpuScoreScope::RulesTerminal) => 2.0,
        };
        features[5] = flag(record.perspective == Color::White);
        features[6] = if config.profile.uses_full_line() {
            0.0
        } else {
            record.origin_state.0 as f32
        };
        features[7] = record.line.len() as f32 / 256.0;
        if let Some(movement) = record.line.first() {
            let token = candidate(*movement);
            token.validate().map_err(model_error)?;
            features[8] = f32::from(token.from) / 63.0;
            features[9] = f32::from(token.to) / 63.0;
            features[10] = f32::from(token.promotion) / 4.0;
            features[11] = 1.0;
        }
        if let Some(movement) = record.line.last() {
            let token = candidate(*movement);
            token.validate().map_err(model_error)?;
            features[12] = f32::from(token.from) / 63.0;
            features[13] = f32::from(token.to) / 63.0;
            features[14] = f32::from(token.promotion) / 4.0;
        }
        features[15] = flag(record.value.is_none());
        if record.critical {
            required.push(id);
        }
        tokens.push(PalsRecordToken {
            record_id: id,
            revision: record.revision,
            critical: record.critical,
            features,
        });
        if let Some(lines) = &mut lines {
            lines.push(PalsRecordLine {
                moves: record.line.iter().copied().map(candidate).collect(),
                parent: metadata.parent.map(usize::from),
                supersedes: metadata.supersedes.map(usize::from),
                parent_required: metadata.parent.is_some(),
                supersedes_required: metadata.supersedes.is_some(),
            });
        }
    }
    Ok((tokens, required, lines))
}
