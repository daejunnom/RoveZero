//! Explicit metadata-only CPU control admission for a pinned PALS graph.
//! Strict CUDA remains the default. Initialization placement is checked before
//! any Run; later kernel profiles attest execution and cannot replace that gate.
use super::*;
use ort::logging::LogLevel;
use ort::session::builder::SessionBuilder;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const POLICY_SCHEMA: &str = "rovezero.pals-cuda-metadata-control.v1";
const MAX_INVENTORY_BYTES: usize = 1024 * 1024;
const MAX_PLACEMENT_BYTES: usize = 1024 * 1024;
const MAX_PROFILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_NODES: usize = 5000;
const CUDA: &str = "CUDAExecutionProvider";
const CPU: &str = "CPUExecutionProvider";

fn policy_error(detail: &'static str) -> BackendError {
    fail(K::BackendUnavailable, S::Backend, detail)
}

#[derive(Clone)]
pub struct PalsCudaControlPolicy {
    pub(super) manifest: [u8; 32],
    pub(super) inventory: [u8; 32],
    graphs: BTreeMap<String, GraphPolicy>,
}
#[derive(Clone)]
struct GraphPolicy {
    digest: [u8; 32],
    nodes: BTreeMap<String, String>,
    controls: BTreeMap<String, String>,
    major: BTreeMap<String, String>,
    proposer_private: BTreeMap<String, String>,
    critic_private: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct Inventory {
    schema: String,
    scope: String,
    manifest_sha256: String,
    actual_provider_placement: String,
    cpu_allowlist: String,
    total_nodes: usize,
    graphs: Vec<InventoryGraph>,
}
#[derive(Deserialize)]
struct InventoryGraph {
    role: String,
    sha256: String,
    inventory: InventoryScope,
}
#[derive(Deserialize)]
struct InventoryScope {
    path: String,
    inputs: Vec<InventoryInput>,
    integral_initializers: Vec<IntegralSeed>,
    nodes: Vec<InventoryNode>,
    subgraphs: Vec<InventoryScope>,
}
#[derive(Deserialize)]
struct InventoryInput {
    name: String,
    dtype: String,
    shape: serde_json::Value,
}
#[derive(Deserialize)]
struct IntegralSeed {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    source: String,
}
impl IntegralSeed {
    fn validate(&self, source: &str) -> Result<(), BackendError> {
        if self.name.is_empty()
            || self.name.len() > 256
            || self.source != source
            || !matches!(self.dtype.as_str(), "INT64" | "INT32" | "BOOL")
            || self.shape.len() > 1
            || self.shape.iter().any(|dimension| *dimension > 64)
        {
            return Err(policy_error(
                "PALS metadata seed dtype/shape/provenance is invalid",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Origin {
    Metadata,
    Integral,
    Role,
    Data,
}
impl Origin {
    fn label(self) -> &'static str {
        match self {
            Self::Metadata => "shape_metadata",
            Self::Integral => "small_integral_constant",
            Self::Role => "role_control",
            Self::Data => "model_data_or_unknown",
        }
    }
}
#[derive(Deserialize)]
struct InventoryNode {
    index: usize,
    name: String,
    domain: String,
    op: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    input_origins: Vec<String>,
    constant_integral_seeds: Vec<IntegralSeed>,
    static_annotation: String,
    actual_provider: String,
}
impl PalsCudaControlPolicy {
    /// Explicit reviewed source inventory, not runtime-discovered fallback.
    /// Hash pinning grants no GPU success: every session still needs its own
    /// complete pre-Run placement audit and actual selected P/C kernel witness.
    pub fn from_inventory(path: &Path, expected_sha256: &str) -> Result<Self, BackendError> {
        if !path.is_absolute() {
            return Err(policy_error("PALS control inventory must be absolute"));
        }
        let bytes = asset::read_bounded(path, MAX_INVENTORY_BYTES)?;
        let digest = asset::sha256(&bytes);
        if digest != parse_sha256(expected_sha256)? {
            return Err(policy_error("PALS control inventory hash differs"));
        }
        let inventory: Inventory = serde_json::from_slice(&bytes).map_err(|error| {
            policy_error("cannot decode pinned PALS control inventory")
                .with_external_cause(CauseCode::ProfileParse, &error)
        })?;
        if inventory.schema != "rovezero.pals-static-control-inventory.v2"
            || inventory.scope != "read_only_serialized_graph_metadata_no_runtime"
            || inventory.actual_provider_placement != "not_observed"
            || inventory.cpu_allowlist != "not_created"
            || !(1..=MAX_NODES).contains(&inventory.total_nodes)
            || inventory.graphs.len() != 2
        {
            return Err(policy_error(
                "PALS inventory scope is not the reviewed static graph schema",
            ));
        }
        let mut graphs = BTreeMap::new();
        let mut count = 0;
        for graph in inventory.graphs {
            if !matches!(graph.role.as_str(), "public" | "shared_pc")
                || graphs.contains_key(&graph.role)
            {
                return Err(policy_error(
                    "PALS control graph role is unsupported or duplicated",
                ));
            }
            let mut policy = GraphPolicy {
                digest: parse_sha256(&graph.sha256)?,
                nodes: BTreeMap::new(),
                controls: BTreeMap::new(),
                major: BTreeMap::new(),
                proposer_private: BTreeMap::new(),
                critic_private: BTreeMap::new(),
            };
            let mut names = BTreeSet::new();
            inspect_scope(
                &graph.inventory,
                &BTreeMap::new(),
                &mut policy,
                &mut names,
                &mut count,
            )?;
            if policy.major.is_empty() {
                return Err(policy_error(
                    "PALS control inventory has no neural computation",
                ));
            }
            graphs.insert(graph.role, policy);
        }
        if count != inventory.total_nodes
            || count > MAX_NODES
            || graphs["shared_pc"].proposer_private.is_empty()
            || graphs["shared_pc"].critic_private.is_empty()
        {
            return Err(policy_error(
                "PALS inventory has incomplete node or P/C branch coverage",
            ));
        }
        Ok(Self {
            manifest: parse_sha256(&inventory.manifest_sha256)?,
            inventory: digest,
            graphs,
        })
    }
    pub(super) fn validate_pins(
        &self,
        manifest: [u8; 32],
        graph: &GraphManifest,
    ) -> Result<(), BackendError> {
        let expected = self.graphs.get(&graph.role).ok_or_else(|| {
            policy_error("PALS explicit control policy is P/C shared-layout only")
        })?;
        if self.manifest != manifest || expected.digest != parse_sha256(&graph.sha256)? {
            return Err(policy_error(
                "PALS explicit control policy graph/manifest pin differs",
            ));
        }
        Ok(())
    }
    pub(super) fn configure(
        &self,
        builder: SessionBuilder,
        role: &str,
        profile_prefix: &Path,
    ) -> Result<(SessionBuilder, Arc<Mutex<PlacementLog>>), BackendError> {
        if !self.graphs.contains_key(role) {
            return Err(policy_error(
                "PALS graph is absent from explicit CUDA control policy",
            ));
        }
        let log = Arc::new(Mutex::new(PlacementLog::default()));
        let captured = Arc::clone(&log);
        let id = format!("pals-placement-{role}");
        let expected_id = id.clone();
        let builder = builder
            .with_log_id(id)
            .and_then(|builder| builder.with_log_level(LogLevel::Verbose))
            .and_then(|builder| builder.with_log_verbosity(1))
            .and_then(|builder| {
                builder.with_logger(Box::new(move |level, _, id, location, message| {
                    // rc.10 mistakenly reads category from code_location. Only id,
                    // location and message are usable. Never panic across this FFI.
                    // The same ORT function emits a warning after mixed node
                    // lists. Only its Verbose placement stream is parseable.
                    if level == LogLevel::Verbose
                        && id == expected_id
                        && location.contains("VerifyEachNodeIsAssignedToAnEp")
                    {
                        if let Ok(mut log) = captured.lock() {
                            log.capture(message);
                        }
                    }
                }))
            })
            .and_then(|builder| builder.with_profiling(profile_prefix))
            .map_err(|error| {
                native(
                    CauseCode::SessionConfiguration,
                    "cannot enable explicit PALS placement audit",
                    error,
                )
            })?;
        Ok((builder, log))
    }
    pub(super) fn verify_initialization(
        &self,
        role: &str,
        log: &Arc<Mutex<PlacementLog>>,
    ) -> Result<PalsGraphPlacement, BackendError> {
        let log = log
            .lock()
            .map_err(|_| policy_error("PALS placement collector was poisoned"))?;
        let graph = self
            .graphs
            .get(role)
            .ok_or_else(|| policy_error("unknown placement graph"))?;
        verify_placement(role, graph, &log)
    }
    pub(super) fn verify_profiles(
        &self,
        public: &[u8],
        shared: &[u8],
        placements: Vec<PalsGraphPlacement>,
        runtime: [u8; 32],
        bundle: [u8; 32],
    ) -> Result<PalsCudaPlacementWitness, BackendError> {
        let mut roles = BTreeSet::new();
        if placements.len() != 2
            || placements.iter().any(|placement| {
                let Some(graph) = self.graphs.get(&placement.role) else {
                    return true;
                };
                !roles.insert(placement.role.clone())
                || placement.graph_sha256 != graph.digest
                || placement.assigned_nodes == 0
                || placement.assigned_nodes > graph.nodes.len()
                || placement.cuda_nodes == 0
                || placement
                    .cuda_nodes
                    .checked_add(placement.approved_cpu_control_nodes)
                    != Some(placement.assigned_nodes)
                || placement.recursive_coverage
                    != "ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run"
            })
        {
            return Err(policy_error(
                "PALS kernel witness lacks both pinned pre-Run graph placements",
            ));
        }
        let public = verify_profile(public, &self.graphs["public"])?;
        let shared = verify_profile(shared, &self.graphs["shared_pc"])?;
        if public.neural_kernels == 0
            || shared.proposer_private_kernels == 0
            || shared.critic_private_kernels == 0
        {
            return Err(policy_error(
                "PALS profiles do not prove public and both selected P/C neural branches",
            ));
        }
        Ok(PalsCudaPlacementWitness {
            schema: POLICY_SCHEMA.into(),
            inventory_sha256: self.inventory,
            manifest_sha256: self.manifest,
            runtime_sha256: runtime,
            runtime_bundle_sha256: bundle,
            initialization: placements,
            public,
            shared_pc: shared,
            category_provenance: "rc10-category-unavailable-id-location-message-used".into(),
        })
    }
}

fn inspect_scope(
    scope: &InventoryScope,
    inherited: &BTreeMap<String, Origin>,
    policy: &mut GraphPolicy,
    names: &mut BTreeSet<String>,
    count: &mut usize,
) -> Result<(), BackendError> {
    let mut origins = inherited.clone();
    for input in &scope.inputs {
        let origin = if input.name == "role_is_critic" {
            if input.dtype != "BOOL" || input.shape != serde_json::json!([]) {
                return Err(policy_error("PALS role control seed is not scalar BOOL"));
            }
            Origin::Role
        } else {
            Origin::Data
        };
        origins.insert(input.name.clone(), origin);
    }
    for seed in &scope.integral_initializers {
        seed.validate("graph_initializer_metadata")?;
        if origins
            .insert(seed.name.clone(), Origin::Integral)
            .is_some()
        {
            return Err(policy_error(
                "PALS metadata initializer shadows another input/seed",
            ));
        }
    }
    let mut child_count = 0;
    for (index, node) in scope.nodes.iter().enumerate() {
        *count += 1;
        if *count > MAX_NODES
            || node.index != index
            || node.name.is_empty()
            || node.name.len() > 256
            || !names.insert(node.name.clone())
            || node.actual_provider != "unknown"
            || !matches!(node.domain.as_str(), "" | "ai.onnx")
        {
            return Err(policy_error(
                "PALS inventory node identity is ambiguous or unsupported",
            ));
        }
        if node.inputs.len() > 64
            || node.outputs.is_empty()
            || node.outputs.len() > 16
            || node.input_origins.len() != node.inputs.len()
        {
            return Err(policy_error("PALS metadata node has invalid IO provenance"));
        }
        policy.nodes.insert(node.name.clone(), node.op.clone());
        let actual_origins: Vec<_> = node
            .inputs
            .iter()
            .map(|name| origins.get(name).copied().unwrap_or(Origin::Data))
            .collect();
        if node
            .inputs
            .iter()
            .zip(&actual_origins)
            .zip(&node.input_origins)
            .any(|((name, origin), recorded)| {
                recorded
                    != if name.is_empty() {
                        "absent_optional_input"
                    } else {
                        origin.label()
                    }
            })
        {
            return Err(policy_error(
                "PALS metadata input-flow provenance cannot be independently reproduced",
            ));
        }
        let mut output_origin = Origin::Data;
        if !node.constant_integral_seeds.is_empty() {
            if node.op != "Constant" || node.constant_integral_seeds.len() != node.outputs.len() {
                return Err(policy_error("PALS integral Constant source is malformed"));
            }
            for seed in &node.constant_integral_seeds {
                seed.validate("constant_node_tensor_metadata")?;
                if !node.outputs.contains(&seed.name) {
                    return Err(policy_error(
                        "PALS Constant seed does not identify its exact output",
                    ));
                }
            }
            output_origin = Origin::Integral;
        }
        let allowed = match node.static_annotation.as_str() {
            "shape_metadata_candidate" => {
                let valid = matches!(node.op.as_str(), "Shape" | "Size")
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1;
                if valid {
                    output_origin = Origin::Metadata;
                }
                valid
            }
            "shape_metadata_chain_candidate" => {
                // Exact named flow, not a general opcode whitelist. The pinned
                // producer includes bounded integral initializer/Constant
                // seeds. Captured scope provenance is independently recomputed.
                matches!(
                    node.op.as_str(),
                    "Gather" | "Concat" | "Unsqueeze" | "Slice" | "Mul"
                ) && !node.inputs.is_empty()
                    && actual_origins.contains(&Origin::Metadata)
                    && actual_origins
                        .iter()
                        .all(|origin| matches!(origin, Origin::Metadata | Origin::Integral))
            }
            "role_scalar_identity_candidate" => {
                let valid = node.op == "Identity"
                    && node.inputs == ["role_is_critic"]
                    && actual_origins == [Origin::Role];
                if valid {
                    output_origin = Origin::Role;
                }
                valid
            }
            _ => false,
        };
        if allowed {
            policy.controls.insert(node.name.clone(), node.op.clone());
            if output_origin == Origin::Data {
                output_origin = Origin::Metadata;
            }
        } else if node.static_annotation == "shape_metadata_chain_candidate"
            || node.static_annotation == "shape_metadata_candidate"
            || node.static_annotation == "role_scalar_identity_candidate"
        {
            return Err(policy_error(
                "PALS metadata annotation conflicts with exact node flow",
            ));
        }
        if node.static_annotation == "major_nn_requires_cuda" {
            if !matches!(
                node.op.as_str(),
                "MatMul"
                    | "Gemm"
                    | "Conv"
                    | "Attention"
                    | "MultiHeadAttention"
                    | "Softmax"
                    | "LayerNormalization"
                    | "SimplifiedLayerNormalization"
                    | "SkipLayerNormalization"
                    | "Gelu"
                    | "FastGelu"
                    | "BiasGelu"
                    | "Relu"
                    | "Sigmoid"
                    | "Tanh"
            ) {
                return Err(policy_error(
                    "PALS neural annotation is not a reviewed neural opcode",
                ));
            }
            policy.major.insert(node.name.clone(), node.op.clone());
            if scope.path.contains("/then_branch") {
                policy
                    .critic_private
                    .insert(node.name.clone(), node.op.clone());
            }
            if scope.path.contains("/else_branch") {
                policy
                    .proposer_private
                    .insert(node.name.clone(), node.op.clone());
            }
        }
        // If dispatch stays on CUDA EP. Its CPU condition does not grant a CPU
        // fallback permission to this node or to its private branch kernels.
        if node.op == "If"
            && (node.static_annotation != "role_if_dispatch_candidate"
                || node.inputs != ["role_is_critic"]
                || actual_origins != [Origin::Role])
        {
            return Err(policy_error(
                "PALS If control condition is not the exact role scalar",
            ));
        }
        // Capture the lexical outer scope at this exact node. Future outputs,
        // especially the If's own outputs, cannot seed its private branch.
        let prefix = format!("{}/node{}:{}/", scope.path, node.index, node.name);
        for child in scope
            .subgraphs
            .iter()
            .filter(|child| child.path.starts_with(&prefix))
        {
            if node.op != "If" {
                return Err(policy_error("PALS has an undeclared nested graph"));
            }
            inspect_scope(child, &origins, policy, names, count)?;
            child_count += 1;
        }
        for name in &node.outputs {
            if name.is_empty()
                || name.len() > 256
                || origins.insert(name.clone(), output_origin).is_some()
            {
                return Err(policy_error(
                    "PALS metadata output aliases another source value",
                ));
            }
        }
    }
    if child_count != scope.subgraphs.len() {
        return Err(policy_error(
            "PALS source subgraph is missing its exact parent node",
        ));
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct PlacementLog {
    lines: Vec<String>,
    bytes: usize,
    overflow: bool,
}
impl PlacementLog {
    fn capture(&mut self, message: &str) {
        if self.overflow {
            return;
        }
        let Some(bytes) = self.bytes.checked_add(message.len()) else {
            self.overflow = true;
            return;
        };
        if bytes > MAX_PLACEMENT_BYTES || self.lines.len() > MAX_NODES + 8 {
            self.overflow = true;
            return;
        }
        self.bytes = bytes;
        self.lines.push(message.to_owned());
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsGraphPlacement {
    pub role: String,
    pub graph_sha256: [u8; 32],
    pub log_sha256: [u8; 32],
    pub assigned_nodes: usize,
    pub cuda_nodes: usize,
    pub approved_cpu_control_nodes: usize,
    pub recursive_coverage: String,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsKernelWitness {
    pub profile_sha256: [u8; 32],
    pub cuda_kernels: usize,
    pub approved_cpu_control_kernels: usize,
    pub neural_kernels: usize,
    pub proposer_private_kernels: usize,
    pub critic_private_kernels: usize,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PalsCudaPlacementWitness {
    pub schema: String,
    pub inventory_sha256: [u8; 32],
    pub manifest_sha256: [u8; 32],
    pub runtime_sha256: [u8; 32],
    pub runtime_bundle_sha256: [u8; 32],
    pub initialization: Vec<PalsGraphPlacement>,
    pub public: PalsKernelWitness,
    pub shared_pc: PalsKernelWitness,
    pub category_provenance: String,
}
fn placement_header(line: &str, prefix: &str) -> Option<(String, usize)> {
    let rest = line.strip_prefix(prefix)?;
    let (provider, count) = rest.split_once("]. Number of nodes: ")?;
    let count = count.parse::<usize>().ok()?;
    Some((provider.into(), count))
}
fn verify_placement(
    role: &str,
    policy: &GraphPolicy,
    log: &PlacementLog,
) -> Result<PalsGraphPlacement, BackendError> {
    if log.overflow {
        return Err(policy_error("PALS placement collector overflowed"));
    }
    let mut began = false;
    let mut headers: BTreeMap<String, usize> = BTreeMap::new();
    let mut observed: BTreeMap<String, usize> = BTreeMap::new();
    let mut selected = None;
    let mut names = BTreeSet::new();
    for line in &log.lines {
        let line = line.trim();
        if line == "Node placements" {
            if began {
                return Err(policy_error(
                    "PALS placement has ambiguous multiple initialization sections",
                ));
            }
            began = true;
            continue;
        }
        if !began {
            continue;
        }
        if let Some((provider, count)) = placement_header(line, "All nodes placed on [") {
            let _ = (provider, count);
            // ORT's single-provider summary omits node identities. This exact
            // policy cannot prove that no unknown/fused node was admitted.
            return Err(policy_error(
                "PALS placement summary lacks exact compiled-node identities",
            ));
        }
        if let Some((provider, count)) = placement_header(line, "Node(s) placed on [") {
            if !matches!(provider.as_str(), CUDA | CPU)
                || count == 0
                || count > MAX_NODES
                || headers.insert(provider.clone(), count).is_some()
            {
                return Err(policy_error(
                    "PALS placement provider header is invalid or duplicated",
                ));
            }
            selected = Some(provider);
            continue;
        }
        if let Some(provider) = &selected {
            let (op, name) = line
                .split_once(" (")
                .and_then(|(op, rest)| rest.strip_suffix(')').map(|name| (op, name)))
                .ok_or_else(|| policy_error("PALS placement node record is incomplete"))?;
            if !names.insert(name.to_owned())
                || name.len() > 256
                || op.is_empty()
                || policy.nodes.get(name).map(String::as_str) != Some(op)
                || (provider == CPU && policy.controls.get(name).map(String::as_str) != Some(op))
            {
                return Err(policy_error(
                    "PALS unknown/fused or unapproved exact node placement",
                ));
            }
            *observed.entry(provider.clone()).or_default() += 1;
        } else {
            return Err(policy_error(
                "PALS placement record has no provider section",
            ));
        }
    }
    if headers != observed || headers.get(CUDA).copied().unwrap_or(0) == 0 || headers.len() != 2 {
        return Err(policy_error(
            "PALS placement lacks complete recursive CUDA/CPU node coverage",
        ));
    }
    let (cuda_nodes, cpu_nodes) = (headers[CUDA], headers[CPU]);
    if !began || cuda_nodes + cpu_nodes > policy.nodes.len() {
        return Err(policy_error("PALS placement evidence missing"));
    }
    Ok(PalsGraphPlacement {
        role: role.into(),
        graph_sha256: policy.digest,
        log_sha256: asset::sha256(log.lines.join("\n").as_bytes()),
        assigned_nodes: cuda_nodes + cpu_nodes,
        cuda_nodes,
        approved_cpu_control_nodes: cpu_nodes,
        recursive_coverage:
            "ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run".into(),
    })
}
fn bounded_metadata_output(value: &serde_json::Value) -> bool {
    let Some(outputs) = value.as_array() else {
        return false;
    };
    !outputs.is_empty()
        && outputs.iter().all(|output| {
            let Some(types) = output.as_object() else {
                return false;
            };
            types.len() == 1
                && types.iter().all(|(kind, dims)| {
                    if !matches!(kind.as_str(), "int64" | "int32" | "bool") {
                        return false;
                    }
                    let Some(dims) = dims.as_array() else {
                        return false;
                    };
                    dims.len() <= 4
                        && dims
                            .iter()
                            .try_fold(1_u64, |n, dim| n.checked_mul(dim.as_u64()?))
                            .is_some_and(|elements| elements <= 64)
                })
        })
}
fn verify_profile(bytes: &[u8], graph: &GraphPolicy) -> Result<PalsKernelWitness, BackendError> {
    if bytes.len() > MAX_PROFILE_BYTES {
        return Err(policy_error("PALS kernel profile exceeds bounded output"));
    }
    let events: Vec<serde_json::Value> = serde_json::from_slice(bytes).map_err(|error| {
        policy_error("cannot decode actual PALS kernel profile")
            .with_external_cause(CauseCode::ProfileParse, &error)
    })?;
    let mut witness = PalsKernelWitness {
        profile_sha256: asset::sha256(bytes),
        cuda_kernels: 0,
        approved_cpu_control_kernels: 0,
        neural_kernels: 0,
        proposer_private_kernels: 0,
        critic_private_kernels: 0,
    };
    for event in events {
        if event.get("cat").and_then(|v| v.as_str()) != Some("Node") {
            continue;
        }
        let name = event
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| policy_error("PALS node profile lacks name"))?;
        let provider = event.pointer("/args/provider").and_then(|v| v.as_str());
        if name.ends_with("_fence_before") || name.ends_with("_fence_after") {
            if provider.is_some_and(|provider| provider != CUDA && provider != CPU) {
                return Err(policy_error("PALS fence record has an unknown provider"));
            }
            continue;
        }
        let node = name
            .strip_suffix("_kernel_time")
            .ok_or_else(|| policy_error("unknown PALS Node profile record"))?;
        let op = event
            .pointer("/args/op_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| policy_error("PALS actual kernel record lacks op type"))?;
        match provider {
            Some(CUDA) => {
                if graph.nodes.get(node).map(String::as_str) != Some(op) {
                    return Err(policy_error(
                        "PALS actual CUDA kernel has unknown/fused identity or opcode",
                    ));
                }
                witness.cuda_kernels += 1;
                if graph.major.get(node).map(String::as_str) == Some(op) {
                    witness.neural_kernels += 1;
                }
                if graph.proposer_private.get(node).map(String::as_str) == Some(op) {
                    witness.proposer_private_kernels += 1;
                }
                if graph.critic_private.get(node).map(String::as_str) == Some(op) {
                    witness.critic_private_kernels += 1;
                }
            }
            Some(CPU)
                if graph.controls.get(node).map(String::as_str) == Some(op)
                    && event
                        .pointer("/args/output_type_shape")
                        .is_some_and(bounded_metadata_output) =>
            {
                witness.approved_cpu_control_kernels += 1;
            }
            _ => {
                return Err(policy_error(
                    "PALS kernel is unapproved CPU inference or unknown placement",
                ))
            }
        }
    }
    if witness.cuda_kernels == 0 {
        return Err(policy_error("PALS profile contains no actual CUDA kernels"));
    }
    Ok(witness)
}

pub(super) fn owned_profile_root(path: &Path) -> Result<PathBuf, BackendError> {
    if !path.is_absolute() {
        return Err(policy_error(
            "PALS explicit audit output root must be absolute",
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| policy_error("PALS audit output parent is absent"))?;
    let parent = parent.canonicalize().map_err(|error| {
        policy_error("PALS audit output parent is unavailable")
            .with_external_cause(CauseCode::ProfilingStart, &error)
    })?;
    for ancestor in parent.ancestors() {
        if ancestor.join(".git").exists() {
            return Err(policy_error("PALS native profiles must be outside Git"));
        }
    }
    let name = path
        .file_name()
        .ok_or_else(|| policy_error("PALS audit output root has no name"))?;
    let output = parent.join(name);
    std::fs::create_dir(&output).map_err(|error| {
        policy_error("PALS audit output root must be new and exclusive")
            .with_external_cause(CauseCode::ProfilingStart, &error)
    })?;
    Ok(output)
}

pub(super) fn record_initial_log(
    log: &Arc<Mutex<PlacementLog>>,
    path: &Path,
    role: &str,
    session_created: bool,
    gate: Option<&Result<PalsGraphPlacement, BackendError>>,
) -> Result<(), BackendError> {
    use std::io::Write;
    let log = log
        .lock()
        .map_err(|_| policy_error("PALS placement collector was poisoned"))?;
    if !matches!(role, "public" | "shared_pc") || !path.is_absolute() {
        return Err(policy_error("PALS placement metadata owner is invalid"));
    }
    let mut selected = None;
    let mut headers = Vec::new();
    let mut nodes = Vec::new();
    let mut unparsed = Vec::new();
    let mut began = false;
    for (sequence, line) in log.lines.iter().enumerate() {
        let line = line.trim();
        if line == "Node placements" {
            began = true;
            continue;
        }
        let header = placement_header(line, "Node(s) placed on [")
            .map(|(provider, count)| (provider, count, false))
            .or_else(|| {
                placement_header(line, "All nodes placed on [")
                    .map(|(provider, count)| (provider, count, true))
            });
        if let Some((provider, count, all_nodes_summary)) = header {
            if provider.len() <= 256 {
                selected = Some(provider.clone());
                headers.push(PlacementHeader {
                    sequence,
                    provider,
                    declared_nodes: count,
                    all_nodes_summary,
                });
                continue;
            }
        }
        if began {
            if let Some((op, name)) = line
                .split_once(" (")
                .and_then(|(op, rest)| rest.strip_suffix(')').map(|name| (op, name)))
            {
                if op.len() <= 256 && name.len() <= 256 {
                    nodes.push(PlacementNode {
                        sequence,
                        provider: selected.clone(),
                        name,
                        op,
                    });
                    continue;
                }
            }
        }
        // Preserve incomplete-stream identity without copying arbitrary logger
        // text or any values into the metadata-only file.
        unparsed.push(PlacementUnparsed {
            sequence,
            line_sha256: asset::sha256(line.as_bytes()),
        });
    }
    let evidence = PlacementFile {
        schema: "rovezero.pals-initial-placement.v1",
        source: "ort-1.22-finalize-recursive-verbose-node-metadata",
        role,
        session_created,
        before_first_neural_run: true,
        gate: match gate {
            Some(Ok(_)) => "passed",
            Some(Err(_)) => "rejected",
            None => "not_reached",
        },
        gate_detail: gate
            .and_then(|result| result.as_ref().err())
            .map(|error| error.detail),
        collected_bytes: log.bytes,
        collector_overflow: log.overflow,
        provider_headers: headers,
        nodes,
        unparsed,
    };
    let mut json = BoundedPlacementJson { bytes: Vec::new() };
    serde_json::to_writer_pretty(&mut json, &evidence).map_err(|error| {
        policy_error("cannot serialize bounded PALS placement metadata")
            .with_external_cause(CauseCode::ProfilingFinish, &error)
    })?;
    json.write_all(b"\n").map_err(|error| {
        policy_error("PALS placement metadata exceeds bounded output")
            .with_external_cause(CauseCode::ProfilingFinish, &error)
    })?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            policy_error("cannot create exclusive PALS placement metadata")
                .with_external_cause(CauseCode::ProfilingStart, &error)
        })?;
    file.write_all(&json.bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| {
            policy_error("cannot preserve bounded PALS placement metadata")
                .with_external_cause(CauseCode::ProfilingFinish, &error)
        })
}

#[derive(Serialize)]
struct PlacementHeader {
    sequence: usize,
    provider: String,
    declared_nodes: usize,
    all_nodes_summary: bool,
}
#[derive(Serialize)]
struct PlacementNode<'a> {
    sequence: usize,
    provider: Option<String>,
    name: &'a str,
    op: &'a str,
}
#[derive(Serialize)]
struct PlacementUnparsed {
    sequence: usize,
    line_sha256: [u8; 32],
}
#[derive(Serialize)]
struct PlacementFile<'a> {
    schema: &'static str,
    source: &'static str,
    role: &'a str,
    session_created: bool,
    before_first_neural_run: bool,
    gate: &'static str,
    gate_detail: Option<&'static str>,
    collected_bytes: usize,
    collector_overflow: bool,
    provider_headers: Vec<PlacementHeader>,
    nodes: Vec<PlacementNode<'a>>,
    unparsed: Vec<PlacementUnparsed>,
}
struct BoundedPlacementJson {
    bytes: Vec<u8>,
}
impl std::io::Write for BoundedPlacementJson {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let required = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|required| *required <= MAX_PROFILE_BYTES)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "PALS placement JSON exceeds 4MiB",
                )
            })?;
        if required > self.bytes.capacity() {
            let target = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .max(required)
                .clamp(1024, MAX_PROFILE_BYTES);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| std::io::Error::other("PALS placement JSON allocation failed"))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn finish_profile(session: &mut Session, root: &Path) -> Result<Vec<u8>, BackendError> {
    let name = session.end_profiling().map_err(|error| {
        native(
            CauseCode::ProfilingFinish,
            "cannot complete actual PALS kernel profile",
            error,
        )
    })?;
    let path = Path::new(&name);
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        policy_error("actual PALS kernel profile is unavailable")
            .with_external_cause(CauseCode::ProfilingFinish, &error)
    })?;
    let actual = path.canonicalize().map_err(|error| {
        policy_error("cannot resolve PALS kernel profile")
            .with_external_cause(CauseCode::ProfilingFinish, &error)
    })?;
    if !path.is_absolute()
        || !metadata.is_file()
        || metadata.file_type().is_symlink()
        || actual.parent() != Some(root)
        || !actual
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                (name.starts_with("pals-public-kernels_")
                    || name.starts_with("pals-shared_pc-kernels_"))
                    && name.ends_with(".json")
            })
    {
        return Err(policy_error(
            "PALS native profile escapes its exclusive owner root",
        ));
    }
    asset::read_bounded(&actual, MAX_PROFILE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct IoRoot(PathBuf);
    impl Drop for IoRoot {
        fn drop(&mut self) {
            // Only the two exact files owned by this test; no recursive cleanup.
            for name in [
                "public-initial-placement.json",
                "shared_pc-initial-placement.json",
            ] {
                let _ = std::fs::remove_file(self.0.join(name));
            }
            let _ = std::fs::remove_dir(&self.0);
        }
    }
    fn io_root() -> IoRoot {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rz-pals-placement-io-{}-{nonce}",
            std::process::id()
        ));
        IoRoot(owned_profile_root(&path).unwrap())
    }
    fn graph() -> GraphPolicy {
        GraphPolicy {
            digest: [1; 32],
            nodes: [
                ("shape_node".into(), "Shape".into()),
                ("nn".into(), "MatMul".into()),
                ("p_nn".into(), "Gemm".into()),
                ("c_nn".into(), "Gemm".into()),
            ]
            .into(),
            controls: [("shape_node".into(), "Shape".into())].into(),
            major: [
                ("nn".into(), "MatMul".into()),
                ("p_nn".into(), "Gemm".into()),
                ("c_nn".into(), "Gemm".into()),
            ]
            .into(),
            proposer_private: [("p_nn".into(), "Gemm".into())].into(),
            critic_private: [("c_nn".into(), "Gemm".into())].into(),
        }
    }
    fn log(lines: &[&str]) -> PlacementLog {
        let mut result = PlacementLog::default();
        for line in lines {
            result.capture(line);
        }
        result
    }
    #[test]
    fn rejected_pre_run_metadata_is_saved_exclusively_without_nn_values() {
        let root = io_root();
        let captured = Arc::new(Mutex::new(log(&[
            "Node placements",
            "Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1",
            "MatMul (p_nn)",
            "Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 1",
            "MatMul (nn)",
            "unrecognized callback record",
        ])));
        let gate = verify_placement("public", &graph(), &captured.lock().unwrap());
        let detail = gate.as_ref().unwrap_err().detail;
        let path = root.0.join("public-initial-placement.json");
        record_initial_log(&captured, &path, "public", true, Some(&gate)).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let file: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(file["gate"], "rejected");
        assert_eq!(file["gate_detail"], detail);
        assert_eq!(file["before_first_neural_run"], true);
        assert_eq!(file["nodes"][0]["provider"], CPU);
        assert_eq!(file["nodes"][0]["name"], "p_nn");
        assert_eq!(file["nodes"][0]["op"], "MatMul");
        assert!(file.get("lines").is_none());
        assert!(!std::str::from_utf8(&bytes)
            .unwrap()
            .contains("unrecognized callback record"));
        assert!(record_initial_log(&captured, &path, "public", true, Some(&gate)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(record_initial_log(
            &captured,
            &root.0.join("absent").join("record.json"),
            "public",
            true,
            Some(&gate)
        )
        .is_err());
        assert_eq!(gate.as_ref().unwrap_err().detail, detail);
        let failure_path = root.0.join("shared_pc-initial-placement.json");
        record_initial_log(&captured, &failure_path, "shared_pc", false, None).unwrap();
        let failure: serde_json::Value =
            serde_json::from_slice(&std::fs::read(failure_path).unwrap()).unwrap();
        assert_eq!(failure["session_created"], false);
        assert_eq!(failure["gate"], "not_reached");
    }
    #[test]
    fn placement_json_byte_ceiling_rejects_before_extending_output() {
        use std::io::Write;
        let mut output = BoundedPlacementJson { bytes: Vec::new() };
        output.write_all(&vec![0; MAX_PROFILE_BYTES]).unwrap();
        assert!(output.write_all(b"x").is_err());
        assert_eq!(output.bytes.len(), MAX_PROFILE_BYTES);
    }
    #[test]
    fn metadata_candidate_cannot_hide_model_data_or_unbounded_seed() {
        let node = |index,
                    name: &str,
                    op: &str,
                    inputs: &[&str],
                    outputs: &[&str],
                    annotation: &str,
                    input_origins: &[&str]| InventoryNode {
            index,
            name: name.into(),
            op: op.into(),
            domain: String::new(),
            inputs: inputs.iter().map(|name| (*name).into()).collect(),
            outputs: outputs.iter().map(|name| (*name).into()).collect(),
            static_annotation: annotation.into(),
            actual_provider: "unknown".into(),
            input_origins: input_origins.iter().map(|name| (*name).into()).collect(),
            constant_integral_seeds: Vec::new(),
        };
        let mut scope = InventoryScope {
            path: "public".into(),
            inputs: vec![InventoryInput {
                name: "data".into(),
                dtype: "FLOAT".into(),
                shape: serde_json::json!([1, 384]),
            }],
            integral_initializers: vec![IntegralSeed {
                name: "index".into(),
                dtype: "INT64".into(),
                shape: vec![],
                source: "graph_initializer_metadata".into(),
            }],
            nodes: vec![
                node(
                    0,
                    "shape",
                    "Shape",
                    &["data"],
                    &["dimensions"],
                    "shape_metadata_candidate",
                    &["model_data_or_unknown"],
                ),
                node(
                    1,
                    "leaky",
                    "Gather",
                    &["data", "index"],
                    &["output"],
                    "shape_metadata_chain_candidate",
                    &["model_data_or_unknown", "small_integral_constant"],
                ),
            ],
            subgraphs: vec![],
        };
        let inspect = |scope: &InventoryScope| {
            inspect_scope(
                scope,
                &BTreeMap::new(),
                &mut graph(),
                &mut BTreeSet::new(),
                &mut 0,
            )
        };
        assert!(inspect(&scope).is_err());
        scope.nodes[1].inputs[0] = "dimensions".into();
        scope.nodes[1].input_origins[0] = "shape_metadata".into();
        assert!(inspect(&scope).is_ok());
        scope.integral_initializers[0].shape = vec![65];
        assert!(inspect(&scope).is_err());
        scope.integral_initializers[0].shape = vec![];
        scope.integral_initializers[0].dtype = "FLOAT".into();
        assert!(inspect(&scope).is_err());
    }
    #[test]
    fn initialization_rejects_nn_cpu_missing_coverage_and_unknown_provider_before_run() {
        let valid = [
            "Node placements",
            " Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1",
            "  Shape (shape_node)",
            " Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 1",
            "  MatMul (nn)",
        ];
        assert_eq!(
            verify_placement("public", &graph(), &log(&valid))
                .unwrap()
                .assigned_nodes,
            2
        );
        let mut changed = valid;
        changed[2] = "  MatMul (shape_node)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        changed[2] = "  Shape (nn)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        changed = valid;
        changed[4] = "  Shape (nn)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        changed[4] = "  FusedMatMul (unregistered)";
        assert!(verify_placement("public", &graph(), &log(&changed)).is_err());
        assert!(verify_placement("public", &graph(), &log(&valid[..4])).is_err());
        assert!(verify_placement(
            "public",
            &graph(),
            &log(&[
                "Node placements",
                "All nodes placed on [CPUExecutionProvider]. Number of nodes: 2"
            ])
        )
        .is_err());
        assert!(verify_placement(
            "public",
            &graph(),
            &log(&[
                "Node placements",
                "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 0"
            ])
        )
        .is_err());
        assert!(verify_placement(
            "public",
            &graph(),
            &PlacementLog {
                overflow: true,
                ..Default::default()
            }
        )
        .is_err());
    }
    #[test]
    fn kernel_controls_require_exact_identity_integral_bounded_output_and_real_cuda() {
        let cuda = serde_json::json!({"cat":"Node","name":"nn_kernel_time","args":{"provider":CUDA,"op_name":"MatMul"}});
        let control = serde_json::json!({"cat":"Node","name":"shape_node_kernel_time","args":{"provider":CPU,"op_name":"Shape","output_type_shape":[{"int64":[4]}]}});
        let profile = |events| serde_json::to_vec(&events).unwrap();
        assert_eq!(
            verify_profile(&profile(vec![cuda.clone(), control.clone()]), &graph())
                .unwrap()
                .approved_cpu_control_kernels,
            1
        );
        let mut bad = control.clone();
        bad["args"]["output_type_shape"] = serde_json::json!([{"float":[4]}]);
        assert!(verify_profile(&profile(vec![cuda.clone(), bad]), &graph()).is_err());
        let mut bad = control;
        bad["args"]["output_type_shape"] = serde_json::json!([{"int64":[65]}]);
        assert!(verify_profile(&profile(vec![cuda, bad]), &graph()).is_err());
        let fake_nn = serde_json::json!({"cat":"Node","name":"nn_kernel_time","args":{"provider":CUDA,"op_name":"Shape"}});
        assert!(verify_profile(&profile(vec![fake_nn]), &graph()).is_err());
        assert!(verify_profile(b"[]", &graph()).is_err());
    }

    #[test]
    fn kernel_witness_requires_both_exact_pre_run_placements() {
        let policy = PalsCudaControlPolicy {
            manifest: [2; 32],
            inventory: [3; 32],
            graphs: [("public".into(), graph()), ("shared_pc".into(), graph())].into(),
        };
        let profile = |node: &str, op: &str| serde_json::json!({"cat":"Node","name":format!("{node}_kernel_time"),"args":{"provider":CUDA,"op_name":op}});
        let public = serde_json::to_vec(&vec![profile("nn", "MatMul")]).unwrap();
        let shared =
            serde_json::to_vec(&vec![profile("p_nn", "Gemm"), profile("c_nn", "Gemm")]).unwrap();
        assert!(policy
            .verify_profiles(&public, &shared, vec![], [4; 32], [5; 32])
            .is_err());
        let make = |role: &str| PalsGraphPlacement {
            role: role.into(),
            graph_sha256: [1; 32],
            log_sha256: [6; 32],
            assigned_nodes: 2,
            cuda_nodes: 1,
            approved_cpu_control_nodes: 1,
            recursive_coverage:
                "ort-1.22-finalize-recursive-exact-named-provider-coverage-before-first-run".into(),
        };
        assert!(policy
            .verify_profiles(
                &public,
                &shared,
                vec![make("public"), make("shared_pc")],
                [4; 32],
                [5; 32]
            )
            .is_ok());
        assert!(policy
            .verify_profiles(
                &public,
                &shared,
                vec![make("public"), make("public")],
                [4; 32],
                [5; 32]
            )
            .is_err());
        let mut wrong = make("shared_pc");
        wrong.graph_sha256 = [7; 32];
        assert!(policy
            .verify_profiles(
                &public,
                &shared,
                vec![make("public"), wrong],
                [4; 32],
                [5; 32]
            )
            .is_err());
    }
}
