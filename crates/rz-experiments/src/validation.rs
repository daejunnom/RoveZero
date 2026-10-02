use crate::manifest::*;
use crate::{ManifestError, Violation};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
struct Checks(Vec<Violation>);

impl Checks {
    fn require(&mut self, ok: bool, path: impl Into<String>, code: &'static str, message: &str) {
        if !ok {
            self.0.push(Violation {
                path: path.into(),
                code,
                message: message.into(),
            });
        }
    }
    fn text(&mut self, path: &str, value: &str) {
        self.require(
            !value.is_empty()
                && value.len() <= 4096
                && value.trim() == value
                && !value.chars().any(char::is_control)
                && !["tbd", "todo", "unknown", "latest", "placeholder", "unset"]
                    .contains(&value.to_ascii_lowercase().as_str()),
            path,
            "UnresolvedValue",
            "must be bounded, explicit text without controls or placeholders",
        );
    }
    fn id(&mut self, path: &str, value: &str) {
        self.text(path, value);
        self.require(
            value.len() <= 128
                && value
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c)),
            path,
            "InvalidId",
            "must contain only ASCII letters, digits, hyphen, underscore or dot",
        );
    }
    fn hash(&mut self, path: &str, value: &str, length: usize) {
        self.require(
            value.len() == length
                && value
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                && value.bytes().any(|c| c != b'0'),
            path,
            "InvalidDigest",
            "must be a non-placeholder lowercase hexadecimal digest/commit",
        );
    }
    fn url(&mut self, path: &str, value: &str) {
        self.text(path, value);
        let public = value.strip_prefix("https://").is_some_and(|rest| {
            let host = rest.split('/').next().unwrap_or_default();
            !host.is_empty()
                && !host.contains('@')
                && !host.contains(':')
                && !rest.contains('?')
                && !rest.contains('#')
                && !rest.contains('\\')
                && !rest.chars().any(char::is_whitespace)
        });
        self.require(
            public,
            path,
            "InvalidSource",
            "must be a public HTTPS source without credentials, query or fragment",
        );
    }
    fn positive(&mut self, path: &str, value: u64) {
        self.require(
            value > 0,
            path,
            "InvalidBudget",
            "must be positive and finite",
        );
    }
    fn artifact(&mut self, path: &str, value: &ArtifactRef) {
        self.require(
            safe_artifact_path(&value.path),
            format!("{path}.path"),
            "UnsafeArtifactPath",
            "must be a logical relative path without secrets, drive, backslash or traversal",
        );
        self.hash(&format!("{path}.sha256"), &value.sha256, 64);
        self.positive(&format!("{path}.bytes"), value.bytes);
        self.url(&format!("{path}.source"), &value.source);
        self.text(&format!("{path}.license"), &value.license);
    }
    fn tool(&mut self, path: &str, value: &ToolIdentity) {
        self.text(&format!("{path}.version"), &value.version);
        self.url(&format!("{path}.source_url"), &value.source_url);
        self.hash(&format!("{path}.source_commit"), &value.source_commit, 40);
        self.artifact(&format!("{path}.binary"), &value.binary);
        self.require(
            value.dirty == value.dirty_patch.is_some(),
            format!("{path}.dirty_patch"),
            "DirtyPatchMismatch",
            "dirty source requires a patch artifact; clean source forbids one",
        );
        if let Some(patch) = &value.dirty_patch {
            self.artifact(&format!("{path}.dirty_patch"), patch);
        }
        for (field, text) in [
            ("build_mode", &value.build_mode),
            ("compiler", &value.compiler),
            ("target", &value.target),
            ("isa", &value.isa),
        ] {
            self.text(&format!("{path}.{field}"), text);
        }
    }
    fn options(&mut self, path: &str, values: &BTreeMap<String, String>) {
        self.require(
            values.len() <= 256,
            path,
            "InputLimit",
            "at most 256 options",
        );
        let mut names = BTreeSet::new();
        for (name, value) in values {
            self.text(path, name);
            self.require(
                value.len() <= 4096 && !value.chars().any(char::is_control),
                path,
                "InvalidOption",
                "option values must be bounded and contain no controls",
            );
            self.require(
                names.insert(name.to_ascii_lowercase()),
                path,
                "DuplicateOption",
                "option names must be unique ignoring ASCII case",
            );
            self.require(
                !(name.eq_ignore_ascii_case("ponder") && !value.eq_ignore_ascii_case("false")),
                path,
                "ForbiddenPonder",
                "ponder must be disabled",
            );
        }
    }
}

pub(crate) fn safe_artifact_path(value: &str) -> bool {
    if value.is_empty()
        || value.len() > 1024
        || value.contains(['\\', ':', '%'])
        || value.starts_with('/')
        || value.chars().any(char::is_control)
    {
        return false;
    }
    value.split('/').all(|part| {
        let p = part.to_ascii_lowercase();
        let stem = p.split('.').next().unwrap_or_default();
        let windows_device = [
            "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7",
            "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
        ]
        .contains(&stem);
        !p.is_empty()
            && p != "."
            && p != ".."
            && p.trim() == p
            && !windows_device
            && !p.ends_with('.')
            && !p.starts_with('.')
            && !p.contains("credential")
            && !p.contains("service-account")
            && !p.contains("service_account")
            && !p.starts_with("id_rsa")
            && !p.starts_with("id_ed25519")
            && !p.ends_with(".pem")
            && !p.ends_with(".key")
            && !p.contains("api-key")
            && !p.contains("api_key")
    })
}

fn utc_date(value: &str) -> bool {
    let b = value.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return false;
    }
    let number = |start: usize, end: usize| -> Option<u32> {
        if !b[start..end].iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(&b[start..end]).ok()?.parse().ok()
    };
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        number(0, 4),
        number(5, 7),
        number(8, 10),
        number(11, 13),
        number(14, 16),
        number(17, 19),
    ) else {
        return false;
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    year > 0 && day > 0 && day <= days && hour < 24 && minute < 60 && second < 60
}

fn uci_move(value: &str) -> bool {
    let b = value.as_bytes();
    (b.len() == 4 || b.len() == 5)
        && (b'a'..=b'h').contains(&b[0])
        && (b'1'..=b'8').contains(&b[1])
        && (b'a'..=b'h').contains(&b[2])
        && (b'1'..=b'8').contains(&b[3])
        && b[0..2] != b[2..4]
        && (b.len() == 4 || b"qrbn".contains(&b[4]))
}

pub(crate) fn validate(m: &RunManifest) -> Result<(), ManifestError> {
    let mut c = Checks::default();
    c.require(
        m.schema_version == SCHEMA_VERSION,
        "schema_version",
        "UnsupportedSchema",
        "only E01 schema 1 is supported",
    );
    c.id("experiment_id", &m.experiment_id);
    c.id("run_id", &m.run_id);
    c.text("question", &m.question);
    c.require(
        utc_date(&m.created_utc),
        "created_utc",
        "InvalidTime",
        "expected a real UTC timestamp YYYY-MM-DDTHH:MM:SSZ",
    );
    if let Some(parent) = &m.parent_run {
        c.id("parent_run", parent);
        c.require(
            parent != &m.run_id,
            "parent_run",
            "InvalidRelation",
            "run cannot be its own parent",
        );
    }
    c.require(
        m.candidate_ids.len() <= 76,
        "candidate_ids",
        "InputLimit",
        "at most 76 research cards",
    );
    let mut cards = BTreeSet::new();
    for id in &m.candidate_ids {
        let b = id.as_bytes();
        let valid = b.len() == 8
            && &b[..5] == b"CARD-"
            && (b'A'..=b'F').contains(&b[5])
            && b[6..].iter().all(u8::is_ascii_digit)
            && (1..=[7, 9, 8, 10, 12, 30][usize::from(b[5] - b'A')])
                .contains(&((b[6] - b'0') * 10 + b[7] - b'0'));
        c.require(
            valid && cards.insert(id),
            "candidate_ids",
            "InvalidCard",
            "expected unique CARD-A01..CARD-F30 identifiers",
        );
    }
    c.require(
        m.engines.len() == 2,
        "engines",
        "InvalidPair",
        "exactly two engine configurations are required",
    );
    c.require(
        (m.purpose == RunPurpose::Fixture) == (m.comparison == Comparison::Fixture),
        "comparison",
        "FixtureMismatch",
        "fixture purpose and fixture comparison must agree",
    );
    if let Some(revision) = &m.contract_revision {
        c.text("contract_revision", revision);
    }
    if m.purpose == RunPurpose::Formal {
        c.require(
            m.contract_revision.is_some(),
            "contract_revision",
            "MissingContract",
            "formal comparison requires the published integration contract revision",
        );
        c.require(
            m.input.split == InputSplit::Holdout,
            "input.split",
            "HoldoutRequired",
            "formal comparison requires a separate holdout",
        );
        c.require(
            !matches!(m.clock, ClockSpec::T3 { .. }),
            "clock",
            "DiagnosticTrack",
            "T3 is diagnostic and cannot support formal strength comparison",
        );
        c.require(
            m.statistics.ci_method != CiMethod::None,
            "statistics.ci_method",
            "MissingStatistics",
            "formal comparison requires a predeclared interval method",
        );
    }
    let mut engine_ids = BTreeSet::new();
    for (i, engine) in m.engines.iter().enumerate() {
        let p = format!("engines[{i}]");
        c.id(&format!("{p}.id"), &engine.id);
        c.require(
            engine_ids.insert(engine.id.clone()),
            &p,
            "DuplicateEngine",
            "engine configuration IDs must differ",
        );
        c.tool(&format!("{p}.tool"), &engine.tool);
        for (field, text) in [
            ("model_id", &engine.model_id),
            ("encoding_id", &engine.encoding_id),
            ("evaluator_id", &engine.evaluator_id),
            ("search_id", &engine.search_id),
            ("runtime_id", &engine.runtime_id),
            ("backend", &engine.backend),
            ("precision", &engine.precision),
        ] {
            c.text(&format!("{p}.{field}"), text);
        }
        if let Some(weight) = &engine.weight {
            c.artifact(&format!("{p}.weight"), weight);
        }
        if m.purpose == RunPurpose::Formal {
            c.require(
                engine.kind != EngineKind::Fixture && engine.weight.is_some(),
                &p,
                "FixtureEngine",
                "formal comparison requires real engine and weight identities",
            );
            c.require(
                engine.observation.is_some(),
                &p,
                "UnverifiedOptions",
                "formal inputs require separate applied options/backend/device evidence",
            );
        }
        if m.purpose != RunPurpose::Fixture {
            c.require(
                engine.kind != EngineKind::Fixture,
                &p,
                "FixtureEngine",
                "fixture engine must be explicitly labelled as a fixture run",
            );
        }
        c.options(&format!("{p}.requested_options"), &engine.requested_options);
        if let Some(o) = &engine.observation {
            c.options(
                &format!("{p}.observation.applied_options"),
                &o.applied_options,
            );
            c.text(&format!("{p}.observation.device_id"), &o.device_id);
            c.artifact(&format!("{p}.observation.evidence"), &o.evidence);
            c.require(
                o.backend == engine.backend
                    && o.precision == engine.precision
                    && o.device == engine.device,
                &p,
                "AppliedIdentityMismatch",
                "observed backend, precision and device must match requested identity",
            );
            for (name, value) in &engine.requested_options {
                let applied = o
                    .applied_options
                    .iter()
                    .find(|(n, _)| n.eq_ignore_ascii_case(name))
                    .map(|(_, v)| v);
                c.require(
                    applied == Some(value),
                    &p,
                    "OptionNotApplied",
                    "every requested option needs a matching observed value",
                );
            }
        }
        c.require(
            engine.device != Device::Gpu || m.hardware.gpu.is_some(),
            &p,
            "MissingGpu",
            "GPU engine requires explicit GPU hardware",
        );
    }
    c.require(
        m.plan.engine_seeds.keys().cloned().collect::<BTreeSet<_>>() == engine_ids,
        "plan.engine_seeds",
        "SeedMismatch",
        "seed map must name exactly both engine configurations",
    );
    if m.comparison == Comparison::ExternalLc0 {
        c.require(
            m.engines
                .iter()
                .filter(|e| e.kind == EngineKind::Lc0)
                .count()
                == 1
                && m.engines
                    .iter()
                    .filter(|e| e.kind == EngineKind::RoveZero)
                    .count()
                    == 1,
            "engines",
            "ComparisonMismatch",
            "external LC0 comparison requires one LC0 and one RoveZero",
        );
    }
    if matches!(
        m.comparison,
        Comparison::InternalSearch | Comparison::Runtime | Comparison::InternalWeights
    ) && m.engines.len() == 2
    {
        let a = &m.engines[0];
        let b = &m.engines[1];
        c.require(
            a.kind == EngineKind::RoveZero && b.kind == EngineKind::RoveZero,
            "engines",
            "ComparisonMismatch",
            "internal controls require RoveZero configurations",
        );
        c.require(
            a.evaluator_id == b.evaluator_id
                && a.encoding_id == b.encoding_id
                && a.backend == b.backend
                && a.precision == b.precision
                && a.device == b.device,
            "engines",
            "ConfoundedControl",
            "internal control must keep evaluator/encoding/backend/precision/device fixed",
        );
        let allowed = m.allowed_option_change.as_deref();
        if let Some(name) = allowed {
            c.text("allowed_option_change", name);
        }
        let normalize = |map: &BTreeMap<String, String>| -> BTreeMap<String, String> {
            map.iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                .collect()
        };
        let mut requested_a = normalize(&a.requested_options);
        let mut requested_b = normalize(&b.requested_options);
        if let Some(name) = allowed {
            let key = name.to_ascii_lowercase();
            let va = requested_a.remove(&key);
            let vb = requested_b.remove(&key);
            c.require(
                va.is_some() && vb.is_some() && va != vb,
                "allowed_option_change",
                "InvalidControlVariable",
                "declared option must exist in both engines and actually differ",
            );
        }
        c.require(
            requested_a == requested_b,
            "engines.requested_options",
            "ConfoundedControl",
            "only the single declared control option may differ",
        );
        if let (Some(oa), Some(ob)) = (&a.observation, &b.observation) {
            let mut applied_a = normalize(&oa.applied_options);
            let mut applied_b = normalize(&ob.applied_options);
            if let Some(name) = allowed {
                applied_a.remove(&name.to_ascii_lowercase());
                applied_b.remove(&name.to_ascii_lowercase());
            }
            c.require(
                applied_a == applied_b,
                "engines.observation",
                "ConfoundedControl",
                "observed default/applied options must also preserve the control",
            );
        }
        let expected_change = match m.comparison {
            Comparison::InternalSearch => Change::Search,
            Comparison::InternalWeights => Change::Model,
            _ => Change::MeaningPreserving,
        };
        c.require(
            m.change == expected_change,
            "change",
            "ControlChangeMismatch",
            "comparison and E/A/S change classification must agree",
        );
        if m.comparison != Comparison::InternalWeights {
            c.require(
                a.model_id == b.model_id
                    && a.weight.as_ref().map(|w| &w.sha256) == b.weight.as_ref().map(|w| &w.sha256),
                "engines",
                "ConfoundedControl",
                "search/runtime control must keep model and weight fixed",
            );
        }
        if m.comparison != Comparison::InternalSearch {
            c.require(
                a.search_id == b.search_id,
                "engines",
                "ConfoundedControl",
                "weight/runtime control must keep search fixed",
            );
        } else {
            c.require(
                a.search_id != b.search_id,
                "engines.search_id",
                "InvalidControlVariable",
                "search comparison requires distinct search identities",
            );
        }
        if m.comparison != Comparison::Runtime {
            c.require(
                a.runtime_id == b.runtime_id,
                "engines.runtime_id",
                "ConfoundedControl",
                "search/weight control must keep runtime fixed",
            );
        }
        if m.comparison == Comparison::Runtime {
            c.require(
                a.runtime_id != b.runtime_id || allowed.is_some(),
                "engines.runtime_id",
                "InvalidControlVariable",
                "runtime comparison requires a declared runtime or option change",
            );
        }
        if m.comparison == Comparison::InternalWeights {
            c.require(
                a.weight.as_ref().map(|w| &w.sha256) != b.weight.as_ref().map(|w| &w.sha256),
                "engines.weight",
                "InvalidControlVariable",
                "weight comparison requires different weight digests",
            );
        }
    } else {
        c.require(
            m.allowed_option_change.is_none(),
            "allowed_option_change",
            "InvalidControlVariable",
            "control option is only used by internal comparisons",
        );
    }
    let h = &m.hardware;
    for (field, text) in [
        ("cpu_model", &h.cpu_model),
        ("operating_system", &h.operating_system),
        ("runtime_version", &h.runtime_version),
        ("assignment_policy", &h.assignment_policy),
    ] {
        c.text(&format!("hardware.{field}"), text);
    }
    c.positive("hardware.physical_cores", u64::from(h.physical_cores));
    c.positive("hardware.threads", u64::from(h.threads));
    c.positive("hardware.ram_limit_bytes", h.ram_limit_bytes);
    c.require(
        h.affinity.len() == h.threads as usize
            && h.affinity.iter().collect::<BTreeSet<_>>().len() == h.affinity.len(),
        "hardware.affinity",
        "AffinityMismatch",
        "must list one distinct logical CPU per declared thread",
    );
    c.require(
        h.helper_core_limit <= h.physical_cores && h.helper_ram_limit_bytes <= h.ram_limit_bytes,
        "hardware",
        "HelperBudget",
        "helper allocation must fit the total core and RAM ceilings",
    );
    c.require(
        !h.tablebase_enabled || h.tablebase_io_limit_bytes > 0,
        "hardware.tablebase_io_limit_bytes",
        "MissingBudget",
        "enabled tablebase needs a finite positive I/O ceiling",
    );
    if let Some(g) = &h.gpu {
        for (field, text) in [
            ("model", &g.model),
            ("driver", &g.driver),
            ("runtime", &g.runtime),
            ("power_policy", &g.power_policy),
        ] {
            c.text(&format!("hardware.gpu.{field}"), text);
        }
        c.positive("hardware.gpu.count", u64::from(g.count));
        c.positive("hardware.gpu.vram_limit_bytes", g.vram_limit_bytes);
        c.require(
            h.residency != Residency::CpuOnly,
            "hardware.residency",
            "ResidencyMismatch",
            "GPU allocation requires an explicit residency policy",
        );
        c.require(
            g.count != 1 || m.budget.workers == 1,
            "budget.workers",
            "GpuConcurrency",
            "single GPU comparisons are limited to one concurrent game",
        );
    } else {
        c.require(
            h.residency == Residency::CpuOnly,
            "hardware.residency",
            "ResidencyMismatch",
            "CPU-only hardware must use CPU-only residency",
        );
    }
    if matches!(m.research_path, ResearchPath::Gpu | ResearchPath::Hybrid) {
        c.require(
            h.gpu.is_some() && m.engines.iter().any(|e| e.device == Device::Gpu),
            "research_path",
            "MissingGpu",
            "GPU/hybrid runs require actual GPU identities",
        );
    }
    c.artifact("input.opening_artifact", &m.input.opening_artifact);
    for (field, text) in [
        ("selection_policy", &m.input.selection_policy),
        ("history_fill_policy", &m.input.history_fill_policy),
        ("repetition_policy", &m.input.repetition_policy),
    ] {
        c.text(&format!("input.{field}"), text);
    }
    c.require(
        !m.input.openings.is_empty() && m.input.openings.len() <= 4096,
        "input.openings",
        "InputLimit",
        "require 1..4096 declared openings",
    );
    let mut opening_ids = BTreeSet::new();
    for opening in &m.input.openings {
        c.id("input.openings.id", &opening.id);
        c.require(
            opening_ids.insert(&opening.id),
            "input.openings.id",
            "DuplicateOpening",
            "opening IDs must be unique",
        );
        c.text("input.openings.history_origin", &opening.history_origin);
        c.require(
            opening.moves.len() <= 4096 && opening.moves.iter().all(|s| uci_move(s)),
            "input.openings.moves",
            "MalformedMove",
            "at most 4096 syntactically valid UCI moves; legality is A/E02's separate gate",
        );
        match opening.initial {
            InitialPosition::Startpos => c.require(
                opening.fen.is_none() && opening.history == HistoryCompleteness::Complete,
                "input.openings",
                "HistoryMismatch",
                "startpos trace has complete history and no alternate FEN",
            ),
            InitialPosition::Fen => {
                c.require(
                    opening.fen.is_some() && opening.history == HistoryCompleteness::UnknownPrefix,
                    "input.openings",
                    "HistoryMismatch",
                    "FEN-only prefix must remain unknown",
                );
                if let Some(fen) = &opening.fen {
                    c.text("input.openings.fen", fen);
                    c.require(
                        fen.split_whitespace().count() == 6,
                        "input.openings.fen",
                        "MalformedFen",
                        "FEN must have six fields; semantic validation requires Position",
                    );
                }
            }
        }
    }
    let p = &m.protocol;
    c.tool("protocol.runner", &p.runner);
    for (field, text) in [
        ("uci_adapter_version", &p.uci_adapter_version),
        ("rules_reference_id", &p.rules_reference_id),
        ("rules_version", &p.rules_version),
        ("draw_profile", &p.draw_profile),
        ("terminal_priority", &p.terminal_priority),
        ("dead_position_scope", &p.dead_position_scope),
        ("tablebase_policy", &p.tablebase_policy),
    ] {
        c.text(&format!("protocol.{field}"), text);
    }
    c.positive("protocol.max_plies", u64::from(p.max_plies));
    c.require(p.max_plies_outcome == OutcomePolicy::Incomplete && p.cancellation_outcome == OutcomePolicy::Incomplete
        && p.engine_failure == OutcomePolicy::Loss && p.simultaneous_failure == OutcomePolicy::ContractInvalid,
        "protocol", "InvalidFailurePolicy", "engine faults count as loss; move limit/cancel are incomplete; dual faults are contract-invalid");
    match m.clock {
        ClockSpec::T1 { base_ms, .. } => c.positive("clock.base_ms", base_ms),
        ClockSpec::T2 { movetime_ms } => c.positive("clock.movetime_ms", movetime_ms),
        ClockSpec::T3 {
            unique_evaluation_budget,
        } => c.positive("clock.unique_evaluation_budget", unique_evaluation_budget),
    }
    let l = &m.lifecycle;
    c.require(
        l.clock_source == "runner_monotonic"
            && l.clock_boundary == "go_write_to_valid_bestmove_read",
        "lifecycle",
        "ClockBoundary",
        "supported clock is runner monotonic go-write to valid-bestmove-read",
    );
    for (field, text) in [
        ("request_deadline_policy", &l.request_deadline_policy),
        ("loading_policy", &l.loading_policy),
        ("compile_policy", &l.compile_policy),
        ("warmup_policy", &l.warmup_policy),
        ("newgame_reset_policy", &l.newgame_reset_policy),
        ("correction_policy", &l.correction_policy),
    ] {
        c.text(&format!("lifecycle.{field}"), text);
    }
    c.positive("lifecycle.handshake_timeout_ms", l.handshake_timeout_ms);
    c.positive("lifecycle.drain_timeout_ms", l.drain_timeout_ms);
    c.positive("lifecycle.shutdown_timeout_ms", l.shutdown_timeout_ms);
    c.require(
        !l.ponder && !l.opponent_turn_compute && !l.online_weights && !l.cross_game_results,
        "lifecycle",
        "UnfairComputation",
        "ponder, opponent-turn compute, online weights and cross-game results are prohibited",
    );
    if let Some(warmup) = &l.warmup_input {
        c.artifact("lifecycle.warmup_input", warmup);
        c.positive("lifecycle.warmup_max_ms", l.warmup_max_ms);
        c.require(
            l.warmup_cache_reset && warmup.sha256 != m.input.opening_artifact.sha256,
            "lifecycle.warmup_input",
            "WarmupLeak",
            "warmup input must differ from evaluation artifact and reset its cache",
        );
    } else {
        c.require(
            l.warmup_max_ms == 0,
            "lifecycle.warmup_max_ms",
            "WarmupMismatch",
            "no warmup input means no warmup time budget",
        );
    }
    c.positive("plan.pairs", m.plan.pairs);
    for (field, text) in [
        ("hardware_order", &m.plan.hardware_order),
        ("incomplete_pair_policy", &m.plan.incomplete_pair_policy),
        ("stop_policy", &m.plan.stop_policy),
    ] {
        c.text(&format!("plan.{field}"), text);
    }
    c.require(
        m.plan.pairs <= m.budget.max_pairs,
        "plan.pairs",
        "BudgetExceeded",
        "planned pairs exceed pair ceiling",
    );
    let attempts = m
        .plan
        .pairs
        .checked_mul(u64::from(m.plan.max_retries_per_pair) + 1)
        .and_then(|n| n.checked_mul(2));
    c.require(
        attempts.is_some_and(|n| n <= m.budget.max_games),
        "budget.max_games",
        "BudgetExceeded",
        "game ceiling must cover planned pairs and bounded whole-pair retries without overflow",
    );
    c.text("statistics.implementation", &m.statistics.implementation);
    c.text(
        "statistics.promotion_policy",
        &m.statistics.promotion_policy,
    );
    c.require(
        m.engines
            .first()
            .is_some_and(|e| e.id == m.statistics.score_viewpoint),
        "statistics.score_viewpoint",
        "ViewpointMismatch",
        "score viewpoint is the first candidate engine configuration",
    );
    c.require(
        m.statistics.budget_end_policy == "inconclusive",
        "statistics.budget_end_policy",
        "InvalidStopPolicy",
        "budget termination is inconclusive",
    );
    match m.statistics.ci_method {
        CiMethod::None => c.require(
            m.statistics.bootstrap_replicates == 0 && m.statistics.confidence_percent == 0,
            "statistics",
            "InvalidStatistics",
            "disabled CI has zero replicates and confidence",
        ),
        CiMethod::OpeningClusterBootstrap => c.require(
            m.statistics.bootstrap_replicates > 0
                && (1..100).contains(&m.statistics.confidence_percent)
                && m.statistics.minimum_clusters >= 2,
            "statistics",
            "InvalidStatistics",
            "bootstrap requires finite replicates, 1..99 confidence and at least two clusters",
        ),
    }
    for (field, value) in [
        ("max_pairs", m.budget.max_pairs),
        ("max_games", m.budget.max_games),
        ("max_wall_ms", m.budget.max_wall_ms),
        ("workers", u64::from(m.budget.workers)),
        (
            "max_child_processes",
            u64::from(m.budget.max_child_processes),
        ),
        ("max_output_bytes", m.budget.max_output_bytes),
        ("max_artifact_bytes", m.budget.max_artifact_bytes),
    ] {
        c.positive(&format!("budget.{field}"), value);
    }
    c.require(
        m.budget
            .workers
            .checked_mul(2)
            .is_some_and(|n| n <= m.budget.max_child_processes),
        "budget.max_child_processes",
        "ChildBudget",
        "must accommodate both engine children for every worker without overflow",
    );
    let mut artifacts = BTreeMap::new();
    for artifact in m.artifacts() {
        if let Some(previous) = artifacts.insert(&artifact.path, artifact) {
            c.require(
                previous == artifact,
                "artifacts",
                "ConflictingArtifact",
                "one logical path cannot carry conflicting identities or provenance",
            );
        }
    }
    let total = artifacts
        .values()
        .try_fold(0_u64, |sum, a| sum.checked_add(a.bytes));
    c.require(
        total.is_some_and(|n| n <= m.budget.max_artifact_bytes),
        "budget.max_artifact_bytes",
        "ArtifactBudget",
        "unique declared artifact bytes must fit the artifact ceiling without overflow",
    );
    if c.0.is_empty() {
        Ok(())
    } else {
        Err(ManifestError::Validation(c.0))
    }
}
