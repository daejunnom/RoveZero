use crate::{ArenaError, canonical_sha256, decode_json};
use rz_experiments::{LockedManifest, OpeningSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{self, Write};

pub const PLAN_VERSION: u32 = 1;
pub const PLAN_ALGORITHM: &str = "rz-e02-paired-v1";
pub const MAX_PLAN_JSON_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct PlanLimits {
    pub max_pairs: u64,
    pub max_json_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GameSpec {
    pub id: String,
    pub white_engine: String,
    pub black_engine: String,
    pub engine_seeds: BTreeMap<String, u64>,
    /// Logical slots are planned assignments, not evidence of actual hardware.
    pub engine_slots: BTreeMap<String, u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairSpec {
    pub id: String,
    pub ordinal: u64,
    pub opening: OpeningSpec,
    /// Identity of the declared opening input, not a reconstructed Rules state.
    pub opening_input_sha256: String,
    pub games: [GameSpec; 2],
    pub execution_order: [usize; 2],
}

#[derive(Clone, Debug)]
pub struct ArenaPlan {
    manifest: LockedManifest,
    input_lock: Value,
    pairs: Vec<PairSpec>,
    sha256: String,
    max_json_bytes: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanEnvelope {
    plan_version: u32,
    algorithm: String,
    execution_ready: bool,
    input_lock: Value,
    input_sha256: String,
    pairs: Vec<PairSpec>,
    plan_sha256: String,
}

#[derive(Serialize)]
struct PlanPayload<'a> {
    plan_version: u32,
    algorithm: &'static str,
    execution_ready: bool,
    input_lock: &'a Value,
    input_sha256: &'a str,
    pairs: &'a [PairSpec],
}

#[derive(Serialize)]
struct OutputEnvelope<'a> {
    #[serde(flatten)]
    input: PlanPayload<'a>,
    plan_sha256: &'a str,
}

#[derive(Serialize)]
struct OpeningIdentity<'a> {
    domain: &'static str,
    schema_version: u32,
    initial: rz_experiments::InitialPosition,
    fen: &'a Option<String>,
    moves: &'a [String],
    history: rz_experiments::HistoryCompleteness,
    history_origin: &'a str,
    history_fill_policy: &'a str,
    repetition_policy: &'a str,
}

#[derive(Serialize)]
struct SeedIdentity<'a> {
    domain: &'static str,
    schema_version: u32,
    plan_seed: u64,
    input_seed: u64,
    base_engine_seed: u64,
    pair_ordinal: u64,
    game_ordinal: usize,
    engine_id: &'a str,
}

// Count compact JSON without allocating a full copy or accepting a partial
// serialization. Build charges each repeated opening before retaining its pair.
struct ByteCounter {
    count: usize,
    maximum: usize,
    exceeded: bool,
}

impl Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(next) = self.count.checked_add(bytes.len()) else {
            self.exceeded = true;
            return Err(io::Error::other("JSON byte count overflow"));
        };
        if next > self.maximum {
            self.exceeded = true;
            return Err(io::Error::other("JSON byte ceiling exceeded"));
        }
        self.count = next;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn json_length<T: Serialize>(value: &T, maximum: usize) -> Result<usize, ArenaError> {
    let mut counter = ByteCounter {
        count: 0,
        maximum,
        exceeded: false,
    };
    serde_json::to_writer(&mut counter, value).map_err(|error| {
        if counter.exceeded {
            ArenaError::Budget("serialized plan exceeds its explicit JSON byte ceiling".into())
        } else {
            ArenaError::Invalid(format!("cannot serialize plan: {error}"))
        }
    })?;
    Ok(counter.count)
}

impl PlanLimits {
    fn validate(self) -> Result<(), ArenaError> {
        if self.max_pairs == 0 || !(1..=MAX_PLAN_JSON_BYTES).contains(&self.max_json_bytes) {
            return Err(ArenaError::Budget(
                "positive max_pairs and a JSON byte ceiling in 1..=4 MiB are required".into(),
            ));
        }
        Ok(())
    }
}

impl ArenaPlan {
    pub fn build(manifest: &LockedManifest, limits: PlanLimits) -> Result<Self, ArenaError> {
        limits.validate()?;
        let input = manifest.input();
        input.validate().map_err(ArenaError::Manifest)?;
        if input.plan.pairs > limits.max_pairs {
            return Err(ArenaError::Budget(
                "planned pair count exceeds the explicit pair ceiling".into(),
            ));
        }
        for (actual, supported) in [
            (input.input.selection_policy.as_str(), "declared-order"),
            (input.plan.hardware_order.as_str(), "balanced-alternating"),
            (
                input.plan.incomplete_pair_policy.as_str(),
                "preserve-incomplete-pair-no-score",
            ),
            (
                input.plan.stop_policy.as_str(),
                "bounded-budget-or-contract-failure",
            ),
        ] {
            if actual != supported {
                return Err(ArenaError::Invalid(format!(
                    "unsupported planning policy: expected {supported}"
                )));
            }
        }
        // Bound the existing input before materializing its nested lock JSON.
        json_length(input, limits.max_json_bytes)?;
        let input_lock = decode_json(&manifest.to_compact_json().map_err(ArenaError::Manifest)?)?;
        let mut plan = Self {
            manifest: manifest.clone(),
            input_lock,
            pairs: Vec::new(),
            sha256: String::new(),
            max_json_bytes: limits.max_json_bytes,
        };
        let mut used_bytes = json_length(
            &OutputEnvelope {
                input: plan.payload(),
                plan_sha256: &"0".repeat(64),
            },
            limits.max_json_bytes,
        )?;
        // Never reserve from an untrusted pair count. Accumulated serialized
        // pair bytes bound repetition before the next pair enters this vector.
        for ordinal in 0..input.plan.pairs {
            let opening =
                &input.input.openings[(ordinal % input.input.openings.len() as u64) as usize];
            let identity = OpeningIdentity {
                domain: "rz-e02-opening-input-v1",
                schema_version: PLAN_VERSION,
                initial: opening.initial,
                fen: &opening.fen,
                moves: &opening.moves,
                history: opening.history,
                history_origin: &opening.history_origin,
                history_fill_policy: &input.input.history_fill_policy,
                repetition_policy: &input.input.repetition_policy,
            };
            let parity = ((ordinal & 1) ^ (input.plan.seed & 1)) as u8;
            let pair_id = format!("{}/pair-{ordinal}", input.run_id);
            let candidate = &input.engines[0].id;
            let baseline = &input.engines[1].id;
            let slots =
                BTreeMap::from([(candidate.clone(), parity), (baseline.clone(), 1 - parity)]);
            let games = [0, 1].map(|game_ordinal| {
                let mut seeds = BTreeMap::new();
                for engine in &input.engines {
                    let identity = SeedIdentity {
                        domain: "rz-e02-engine-seed-v1",
                        schema_version: PLAN_VERSION,
                        plan_seed: input.plan.seed,
                        input_seed: input.input.seed,
                        base_engine_seed: input.plan.engine_seeds[&engine.id],
                        pair_ordinal: ordinal,
                        game_ordinal,
                        engine_id: &engine.id,
                    };
                    let hash = canonical_sha256(&identity)?;
                    let seed = u64::from_str_radix(&hash[..16], 16)
                        .map_err(|_| ArenaError::Integrity("invalid seed SHA-256".into()))?;
                    seeds.insert(engine.id.clone(), seed);
                }
                Ok::<_, ArenaError>(GameSpec {
                    id: format!("{pair_id}/game-{game_ordinal}"),
                    white_engine: if game_ordinal == 0 {
                        candidate.clone()
                    } else {
                        baseline.clone()
                    },
                    black_engine: if game_ordinal == 0 {
                        baseline.clone()
                    } else {
                        candidate.clone()
                    },
                    engine_seeds: seeds,
                    engine_slots: slots.clone(),
                })
            });
            let [game_a, game_b] = games;
            let pair = PairSpec {
                id: pair_id,
                ordinal,
                opening: opening.clone(),
                opening_input_sha256: canonical_sha256(&identity)?,
                games: [game_a?, game_b?],
                execution_order: if parity == 0 { [0, 1] } else { [1, 0] },
            };
            let separator = usize::from(ordinal != 0);
            let remaining = limits
                .max_json_bytes
                .checked_sub(used_bytes)
                .and_then(|remaining| remaining.checked_sub(separator))
                .ok_or_else(|| ArenaError::Budget("serialized pair budget exhausted".into()))?;
            let bytes = json_length(&pair, remaining)?;
            used_bytes = used_bytes
                .checked_add(separator)
                .and_then(|n| n.checked_add(bytes))
                .ok_or_else(|| ArenaError::Budget("serialized pair byte count overflow".into()))?;
            plan.pairs.push(pair);
        }
        plan.sha256 = canonical_sha256(&plan.payload())?;
        Ok(plan)
    }

    pub fn manifest(&self) -> &LockedManifest {
        &self.manifest
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn pairs(&self) -> &[PairSpec] {
        &self.pairs
    }
    pub fn pair(&self, id: &str) -> Option<&PairSpec> {
        self.pairs.iter().find(|pair| pair.id == id)
    }

    fn payload(&self) -> PlanPayload<'_> {
        PlanPayload {
            plan_version: PLAN_VERSION,
            algorithm: PLAN_ALGORITHM,
            execution_ready: false,
            input_lock: &self.input_lock,
            input_sha256: self.manifest.sha256(),
            pairs: &self.pairs,
        }
    }

    pub fn to_json(&self) -> Result<String, ArenaError> {
        let envelope = OutputEnvelope {
            input: self.payload(),
            plan_sha256: &self.sha256,
        };
        json_length(&envelope, self.max_json_bytes)?;
        serde_json::to_string(&envelope)
            .map_err(|error| ArenaError::Invalid(format!("cannot serialize plan: {error}")))
    }

    pub fn from_json(input: &str, limits: PlanLimits) -> Result<Self, ArenaError> {
        limits.validate()?;
        if input.len() > limits.max_json_bytes {
            return Err(ArenaError::Budget(
                "plan JSON exceeds its explicit byte ceiling".into(),
            ));
        }
        let envelope: PlanEnvelope = decode_json(input)?;
        if envelope.plan_version != PLAN_VERSION
            || envelope.algorithm != PLAN_ALGORITHM
            || envelope.execution_ready
        {
            return Err(ArenaError::Integrity(
                "unsupported plan version/algorithm or execution readiness".into(),
            ));
        }
        if envelope.pairs.len() as u64 > limits.max_pairs {
            return Err(ArenaError::Budget(
                "serialized pair count exceeds the explicit pair ceiling".into(),
            ));
        }
        let lock_json = serde_json::to_string(&envelope.input_lock).map_err(|error| {
            ArenaError::Integrity(format!("cannot serialize input lock: {error}"))
        })?;
        let manifest = LockedManifest::from_json(&lock_json).map_err(ArenaError::Manifest)?;
        if manifest.sha256() != envelope.input_sha256 {
            return Err(ArenaError::Integrity("plan input SHA-256 mismatch".into()));
        }
        let rebuilt = Self::build(&manifest, limits)?;
        if rebuilt.pairs != envelope.pairs || rebuilt.sha256 != envelope.plan_sha256 {
            return Err(ArenaError::Integrity(
                "derived pair plan or SHA-256 mismatch".into(),
            ));
        }
        Ok(rebuilt)
    }
}
