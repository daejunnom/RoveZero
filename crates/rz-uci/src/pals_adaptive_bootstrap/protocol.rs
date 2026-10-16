//! The ordinary UCI owner is shared by V5 native and labelled metadata probes.
//! A probe obtains legal stop moves from Rules and constructs no CPU engine.
use crate::engine::{self, EngineProcess, EngineSettings, OwnerRegistry, ProcessClock};
use crate::search_driver::{
    DriverWork, SearchKind, SearchSessionContext, SearchSessionDriver, SearchSessionFailure,
    SearchSessionReport,
};
use crate::{EngineIdentity, forward_lines};
use rz_contracts::Digest;
use rz_position::{BoardMove, Position};
use std::{io, sync::Arc, thread};

pub fn serve_driver(
    driver: Arc<dyn SearchSessionDriver>,
    clock: ProcessClock,
    owners: Arc<OwnerRegistry>,
    settings: EngineSettings,
) -> Result<(), engine::EngineError> {
    let process = EngineProcess::with_search_driver(driver, owners, clock);
    let (sender, events) = engine::event_channel();
    let input = sender.clone();
    // Only stdin may block here. Search, stop and physical drain have independent
    // owners; EOF/quit is processed by the existing bounded protocol engine.
    thread::Builder::new()
        .name("pals-adaptive-stdin".into())
        .spawn(move || {
            let _ = forward_lines(
                &mut io::stdin().lock(),
                &input,
                settings.parser.max_line_bytes,
            );
        })
        .map_err(|_| {
            engine::EngineError::Contract(rz_contracts::ContractError::new(
                rz_contracts::ErrorCode::BackendFailure,
                rz_contracts::Stage::Admission,
                "adaptive stdin owner thread creation failed",
            ))
        })?;
    engine::serve(
        events,
        sender,
        &mut io::stdout().lock(),
        &mut io::stderr().lock(),
        process,
        settings,
    )
}

pub struct RulesOnlyPreflightDriver {
    pub identity: EngineIdentity,
    pub implementation: Digest,
}
impl SearchSessionDriver for RulesOnlyPreflightDriver {
    fn kind(&self) -> SearchKind {
        SearchKind::Pals
    }
    fn identity(&self) -> EngineIdentity {
        self.identity.clone()
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        _: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if !context
            .accepts()
            .map_err(|e| SearchSessionFailure::debug("PreflightAuthority", &e))?
        {
            return Err(SearchSessionFailure::debug(
                "PreflightExpired",
                &"Rules-only probe revoked",
            ));
        }
        let best_move = position.legal_moves().into_iter().next();
        Ok(SearchSessionReport {
            best_move,
            work: DriverWork::Pals {
                rounds: 0,
                cpu_nodes: 0,
                completed_tasks: 0,
                consumed_role_outputs: 0,
                retained_situations: 0,
            },
        })
    }
}
