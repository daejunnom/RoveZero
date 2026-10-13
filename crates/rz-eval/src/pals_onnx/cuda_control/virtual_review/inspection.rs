//! Source-pinned V3 inventory inspection. Metadata permission is reproduced
//! from lexical input flow; it never grants a CPU neural or generic If fallback.
use super::*;

const WARM_GRAPH: &str = "virtual_review_shared_pc_if_approx_warm_v1";
const WARM_MODE: &str = "private_virtual_warm_mode";
const WARM_ANNOTATION: &str = "virtual_warm_if_dispatch_candidate";
const WARM_SEED: &str = "consume_complete_virtual_private_seed";

type PrivateKernelInventory = BTreeMap<String, String>;
type InspectedGraphPolicy = (GraphPolicy, PrivateKernelInventory, PrivateKernelInventory);
type InspectedPrivateBindings = (
    BTreeMap<String, Origin>,
    PrivateKernelInventory,
    PrivateKernelInventory,
);

struct Route {
    root: String,
    mode: Option<String>,
    role_transfer: RoleTransferPolicy,
    required: BTreeSet<String>,
}

/// Returns the complete policy plus independently named V-private and task
/// kernels. A P/C placement or profile cannot stand in for either V witness.
pub(super) fn inspect(
    graph: &InventoryGraphV3,
    count: &mut usize,
) -> Result<InspectedGraphPolicy, BackendError> {
    if !matches!(
        graph.role.as_str(),
        "public" | "shared_pc" | "shared_warm" | "validator"
    ) || graph.inventory.path != graph.role
    {
        return Err(policy_error("V3 inventory root/role is not registered"));
    }
    let route = if matches!(graph.role.as_str(), "shared_pc" | "shared_warm") {
        Some(control_route(graph)?)
    } else {
        None
    };
    let mut policy = GraphPolicy {
        digest: parse_sha256(&graph.sha256)?,
        nodes: BTreeMap::new(),
        controls: BTreeMap::new(),
        major: BTreeMap::new(),
        proposer_private: BTreeMap::new(),
        critic_private: BTreeMap::new(),
        role_transfer: None,
        required_cuda_dispatch: route
            .as_ref()
            .map(|value| value.required.clone())
            .unwrap_or_default(),
    };
    let mut local_count = 0;
    let (weight_origins, validator, task) = inspect_private_bindings(graph)?;
    inspect_scope_v3(
        &graph.inventory,
        &weight_origins,
        &mut policy,
        &mut BTreeSet::new(),
        &mut local_count,
        route.as_ref(),
        None,
        &graph.role,
        0,
    )?;
    policy.role_transfer = route.map(|value| value.role_transfer);
    if policy.major.is_empty()
        || (matches!(graph.role.as_str(), "shared_pc" | "shared_warm")
            && (policy.proposer_private.is_empty() || policy.critic_private.is_empty()))
        || (graph.role == "validator" && (validator.is_empty() || task.is_empty()))
        || (graph.role != "validator" && (!validator.is_empty() || !task.is_empty()))
    {
        return Err(policy_error(
            "V3 inventory lacks its own complete neural/private head coverage",
        ));
    }
    let combined = count
        .checked_add(local_count)
        .filter(|value| *value <= MAX_VIRTUAL_NODES)
        .ok_or_else(|| policy_error("V3 total inventory exceeds its separate node budget"))?;
    *count = combined;
    Ok((policy, validator, task))
}

fn major(op: &str) -> bool {
    matches!(
        op,
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
    )
}

fn origin_label(origin: Origin) -> &'static str {
    // This is a separate semantic domain. The legacy Origin::label and legacy
    // inspector remain unchanged, even though the storage enum is reused.
    if origin == Origin::CudaWarmMode {
        "virtual_warm_mode_control"
    } else {
        origin.label()
    }
}

fn valid_name(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

fn descriptors(values: &[InventoryInput]) -> bool {
    let mut names = BTreeSet::new();
    values.len() <= 64
        && values.iter().all(|value| {
            let Some(shape) = value.shape.as_array() else {
                return false;
            };
            valid_name(&value.name)
                && names.insert(&value.name)
                && matches!(value.dtype.as_str(), "FLOAT" | "INT64" | "INT32" | "BOOL")
                && shape.len() <= 8
                && shape.iter().all(|axis| {
                    axis.as_u64().is_some_and(|value| value > 0)
                        || axis.as_str().is_some_and(valid_name)
                })
        })
}

type DependencyKey = (String, usize, String, String, String, String);

/// Export-ledger hashes are declarations covered by source pins. Independently
/// reproduce direct serialized node dependencies here; provider permission is
/// still determined later by initialization placement and actual kernel events.
fn inspect_private_bindings(
    graph: &InventoryGraphV3,
) -> Result<InspectedPrivateBindings, BackendError> {
    let mut weights = BTreeMap::new();
    let mut origins = BTreeMap::new();
    if graph.weight_bindings.is_empty()
        || graph.weight_bindings.len() > MAX_NODES
        || graph.weight_bindings.len() > graph.inventory.initializer_count.unwrap_or(0)
        || graph.private_neural_dependencies.len() > MAX_NODES * 64
    {
        return Err(policy_error(
            "V3 bounded FLOAT parameter/dependency ledger is missing",
        ));
    }
    for weight in &graph.weight_bindings {
        if !valid_name(&weight.initializer)
            || !valid_name(&weight.parameter)
            || weight.dtype != "FLOAT"
            || weight.shape.is_empty()
            || weight.shape.len() > 8
            || weight.shape.contains(&0)
            || (weight.initializer != weight.parameter
                && weight.initializer != format!("{}.linear_transpose", weight.parameter))
            || weights
                .insert(weight.initializer.as_str(), weight)
                .is_some()
        {
            return Err(policy_error(
                "V3 weight dependency lacks its exact FLOAT parameter descriptor",
            ));
        }
        origins.insert(weight.initializer.clone(), Origin::Data);
    }
    let mut declared = BTreeSet::new();
    for dependency in &graph.private_neural_dependencies {
        let key = (
            dependency.scope_path.clone(),
            dependency.node_index,
            dependency.node_name.clone(),
            dependency.op.clone(),
            dependency.initializer.clone(),
            dependency.parameter.clone(),
        );
        if !declared.insert(key) {
            return Err(policy_error(
                "V3 private dependency declaration is duplicated",
            ));
        }
    }
    let mut actual = BTreeSet::new();
    let mut validator = BTreeMap::new();
    let mut task = BTreeMap::new();
    inspect_weight_dependency_scope(
        &graph.inventory,
        &weights,
        &mut actual,
        &mut validator,
        &mut task,
        0,
        &mut 0,
    )?;
    if actual != declared {
        return Err(policy_error(
            "V3 private ledger differs from exact scope/index/name/op/initializer node inputs",
        ));
    }
    Ok((origins, validator, task))
}

#[allow(clippy::too_many_arguments)]
fn inspect_weight_dependency_scope(
    scope: &InventoryScope,
    weights: &BTreeMap<&str, &WeightBindingV3>,
    actual: &mut BTreeSet<DependencyKey>,
    validator: &mut BTreeMap<String, String>,
    task: &mut BTreeMap<String, String>,
    depth: usize,
    count: &mut usize,
) -> Result<(), BackendError> {
    if depth > 32
        || scope.path.len() > 4096
        || scope.nodes.len() > MAX_NODES
        || scope
            .integral_initializers
            .iter()
            .any(|seed| weights.contains_key(seed.name.as_str()))
    {
        return Err(policy_error(
            "V3 private descriptor shadows integral metadata or exceeds scope bounds",
        ));
    }
    for node in &scope.nodes {
        *count = count
            .checked_add(1)
            .filter(|value| *value <= MAX_NODES)
            .ok_or_else(|| policy_error("V3 private dependency scan exceeds graph node budget"))?;
        for input in &node.inputs {
            let Some(weight) = weights.get(input.as_str()) else {
                if input.starts_with("experts.") {
                    return Err(policy_error(
                        "V3 serialized private input is missing its exact parameter binding",
                    ));
                }
                continue;
            };
            if !weight.parameter.starts_with("experts.") || !major(&node.op) {
                continue;
            }
            let key = (
                scope.path.clone(),
                node.index,
                node.name.clone(),
                node.op.clone(),
                input.clone(),
                weight.parameter.clone(),
            );
            if !actual.insert(key) {
                return Err(policy_error(
                    "V3 direct private dependency is ambiguous or repeated",
                ));
            }
            if weight.parameter.starts_with("experts.validator.") {
                if weight.parameter == "experts.validator.task_head.weight" {
                    task.insert(node.name.clone(), node.op.clone());
                } else {
                    validator.insert(node.name.clone(), node.op.clone());
                }
            }
        }
        // A single physical kernel cannot prove both the task head and its
        // separate non-task private computation. This is independent of input
        // ordering when a node directly consumes both kinds of parameter.
        if task.contains_key(&node.name) {
            validator.remove(&node.name);
        }
    }
    for child in &scope.subgraphs {
        inspect_weight_dependency_scope(child, weights, actual, validator, task, depth + 1, count)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn inspect_scope_v3(
    scope: &InventoryScope,
    inherited: &BTreeMap<String, Origin>,
    policy: &mut GraphPolicy,
    names: &mut BTreeSet<String>,
    count: &mut usize,
    route: Option<&Route>,
    private_role: Option<PrivateRole>,
    graph_role: &str,
    depth: usize,
) -> Result<(), BackendError> {
    if depth > 32
        || scope.path.len() > 4096
        || scope
            .initializer_count
            .is_none_or(|value| value > MAX_NODES)
        || !descriptors(&scope.inputs)
        || !descriptors(&scope.outputs)
        || scope.integral_initializers.len() > scope.initializer_count.unwrap_or(0)
        || (depth > 0
            && (!scope.inputs.is_empty()
                || scope.initializer_count != Some(0)
                || !scope.integral_initializers.is_empty()))
    {
        return Err(policy_error(
            "V3 scope metadata/initializer/depth bound is invalid",
        ));
    }
    let mut origins = inherited.clone();
    for input in &scope.inputs {
        let origin = if input.name == "role_is_critic" {
            if input.dtype != "BOOL"
                || input.shape != serde_json::json!([])
                || route.is_none()
                || depth != 0
            {
                return Err(policy_error("V3 role seed is not its scalar root BOOL"));
            }
            Origin::Role
        } else if input.name == "warm_start" {
            if input.dtype != "BOOL"
                || input.shape != serde_json::json!([])
                || !route.is_some_and(|value| value.mode.is_some() && value.root == scope.path)
            {
                return Err(policy_error(
                    "V3 mode seed is not its scalar Warm root BOOL",
                ));
            }
            Origin::CudaWarmMode
        } else {
            Origin::Data
        };
        if origins.insert(input.name.clone(), origin).is_some() {
            return Err(policy_error("V3 nested input shadows an inherited value"));
        }
    }
    for seed in &scope.integral_initializers {
        seed.validate("graph_initializer_metadata")?;
        if !valid_name(&seed.name)
            || origins
                .insert(seed.name.clone(), Origin::Integral)
                .is_some()
        {
            return Err(policy_error(
                "V3 integral initializer aliases an input or seed",
            ));
        }
    }
    let mut visited_children = BTreeSet::new();
    for (index, node) in scope.nodes.iter().enumerate() {
        *count = count
            .checked_add(1)
            .ok_or_else(|| policy_error("V3 node count overflows"))?;
        let Some(attributes) = &node.attribute_names else {
            return Err(policy_error("V3 node attributes are not inventoried"));
        };
        let mut attribute_names = BTreeSet::new();
        if *count > MAX_NODES
            || node.index != index
            || !valid_name(&node.name)
            || !valid_name(&node.op)
            || !names.insert(node.name.clone())
            || node.actual_provider != "unknown"
            || !matches!(node.domain.as_str(), "" | "ai.onnx")
            || node.inputs.len() > 64
            || node.outputs.is_empty()
            || node.outputs.len() > 16
            || node.input_origins.len() != node.inputs.len()
            || attributes.len() > 32
            || attributes
                .iter()
                .any(|name| !valid_name(name) || !attribute_names.insert(name))
            || node
                .inputs
                .iter()
                .any(|name| !name.is_empty() && !valid_name(name))
        {
            return Err(policy_error(
                "V3 node identity/IO/attribute metadata is invalid",
            ));
        }
        policy.nodes.insert(node.name.clone(), node.op.clone());
        let actual: Vec<_> = node
            .inputs
            .iter()
            .map(|name| origins.get(name).copied().unwrap_or(Origin::Data))
            .collect();
        if node
            .inputs
            .iter()
            .zip(&actual)
            .zip(&node.input_origins)
            .any(|((name, origin), label)| {
                label
                    != if name.is_empty() {
                        "absent_optional_input"
                    } else {
                        origin_label(*origin)
                    }
            })
        {
            return Err(policy_error(
                "V3 exact lexical provenance cannot be reproduced",
            ));
        }
        let mut output_origin = Origin::Data;
        if !node.constant_integral_seeds.is_empty() {
            if node.op != "Constant"
                || !node.inputs.is_empty()
                || node.constant_integral_seeds.len() != node.outputs.len()
            {
                return Err(policy_error("V3 integral Constant metadata is malformed"));
            }
            let mut constants = BTreeSet::new();
            for seed in &node.constant_integral_seeds {
                seed.validate("constant_node_tensor_metadata")?;
                if !node.outputs.contains(&seed.name) || !constants.insert(&seed.name) {
                    return Err(policy_error(
                        "V3 Constant does not name its exact unique output",
                    ));
                }
            }
            output_origin = Origin::Integral;
        }
        let metadata = match node.static_annotation.as_str() {
            "shape_metadata_candidate" => {
                if !matches!(node.op.as_str(), "Shape" | "Size")
                    || node.inputs.len() != 1
                    || node.outputs.len() != 1
                {
                    return Err(policy_error(
                        "V3 shape annotation conflicts with actual flow",
                    ));
                }
                output_origin = Origin::Metadata;
                true
            }
            "shape_metadata_chain_candidate" => {
                if !matches!(
                    node.op.as_str(),
                    "Gather" | "Concat" | "Unsqueeze" | "Slice" | "Mul"
                ) || actual.is_empty()
                    || !actual.contains(&Origin::Metadata)
                    || !actual
                        .iter()
                        .all(|value| matches!(value, Origin::Metadata | Origin::Integral))
                {
                    return Err(policy_error(
                        "V3 shape chain consumes data or an unsupported operator",
                    ));
                }
                output_origin = Origin::Metadata;
                true
            }
            "role_scalar_identity_candidate" => {
                if node.op != "Identity"
                    || node.inputs != ["role_is_critic"]
                    || actual != [Origin::Role]
                    || node.outputs != ["is_critic"]
                    || !route.is_some_and(|value| {
                        value.root == scope.path
                            && value.role_transfer.destination_node == node.name
                    })
                {
                    return Err(policy_error(
                        "V3 role Identity differs from its exact root destination",
                    ));
                }
                output_origin = Origin::Role;
                true
            }
            "none" | "major_nn_requires_cuda" | "role_if_dispatch_candidate" | WARM_ANNOTATION => {
                false
            }
            _ => return Err(policy_error("V3 static node annotation is unregistered")),
        };
        if metadata {
            policy.controls.insert(node.name.clone(), node.op.clone());
        }
        if (node.static_annotation == "major_nn_requires_cuda") != major(&node.op) {
            return Err(policy_error(
                "V3 neural classification differs from its actual operator",
            ));
        }
        if major(&node.op) {
            policy.major.insert(node.name.clone(), node.op.clone());
            match private_role {
                Some(PrivateRole::Proposer)
                    if node
                        .inputs
                        .iter()
                        .any(|name| name.starts_with("experts.proposer.")) =>
                {
                    policy
                        .proposer_private
                        .insert(node.name.clone(), node.op.clone());
                }
                Some(PrivateRole::Critic)
                    if node
                        .inputs
                        .iter()
                        .any(|name| name.starts_with("experts.critic.")) =>
                {
                    policy
                        .critic_private
                        .insert(node.name.clone(), node.op.clone());
                }
                _ => {}
            }
        }
        if graph_role == "validator"
            && node.inputs.iter().any(|name| {
                name.starts_with("experts.proposer.") || name.starts_with("experts.critic.")
            })
        {
            return Err(policy_error(
                "V3 validator graph consumes P/C private weights",
            ));
        }
        if node.inputs.iter().any(|name| name.starts_with("experts."))
            && (graph_role == "public"
                || (matches!(graph_role, "shared_pc" | "shared_warm") && private_role.is_none()))
        {
            return Err(policy_error("V3 public/shared execution consumes private weights outside its selected role branch"));
        }
        if private_role.is_some_and(|role| {
            node.inputs.iter().any(|name| match role {
                PrivateRole::Proposer => {
                    name.starts_with("experts.critic.") || name.starts_with("experts.validator.")
                }
                PrivateRole::Critic => {
                    name.starts_with("experts.proposer.") || name.starts_with("experts.validator.")
                }
            })
        }) {
            return Err(policy_error(
                "V3 selected private branch consumes another role's weights",
            ));
        }
        let role_if = node.op == "If"
            && node.static_annotation == "role_if_dispatch_candidate"
            && node.inputs == ["role_is_critic"]
            && actual == [Origin::Role]
            && route.is_some_and(|value| {
                value
                    .role_transfer
                    .host_condition_nodes
                    .contains(&node.name)
            });
        let mode_if = node.op == "If"
            && node.static_annotation == WARM_ANNOTATION
            && node.inputs == ["warm_start"]
            && actual == [Origin::CudaWarmMode]
            && route.is_some_and(|value| {
                value.root == scope.path && value.mode.as_ref() == Some(&node.name)
            });
        if (node.op == "If" && !role_if && !mode_if)
            || (node.static_annotation == "role_if_dispatch_candidate" && !role_if)
            || (node.static_annotation == WARM_ANNOTATION && !mode_if)
        {
            return Err(policy_error(
                "V3 If is not its exact registered role/mode control",
            ));
        }
        if node.op == "If" {
            let (then, otherwise) = direct_if_branches(scope, node)?;
            for (child, selected) in [
                (then, PrivateRole::Critic),
                (otherwise, PrivateRole::Proposer),
            ] {
                if !visited_children.insert(child.path.clone()) {
                    return Err(policy_error("V3 branch scope is duplicated"));
                }
                inspect_scope_v3(
                    child,
                    &origins,
                    policy,
                    names,
                    count,
                    route,
                    if role_if {
                        Some(selected)
                    } else {
                        private_role
                    },
                    graph_role,
                    depth + 1,
                )?;
            }
        }
        // Capture branches before registering If outputs. Future/outer outputs
        // cannot masquerade as completed data in a selected private branch.
        for output in &node.outputs {
            if !valid_name(output) || origins.insert(output.clone(), output_origin).is_some() {
                return Err(policy_error(
                    "V3 output aliases an input, ancestor, or another output",
                ));
            }
        }
    }
    if visited_children.len() != scope.subgraphs.len()
        || scope
            .outputs
            .iter()
            .any(|value| !origins.contains_key(&value.name))
    {
        return Err(policy_error(
            "V3 scope has an orphan branch or undefined output",
        ));
    }
    Ok(())
}

fn empty_branch(branch: &InventoryScope) -> bool {
    branch.inputs.is_empty()
        && branch.initializer_count == Some(0)
        && branch.integral_initializers.is_empty()
}

fn control_route(graph: &InventoryGraphV3) -> Result<Route, BackendError> {
    let scope = &graph.inventory;
    let warm = graph.role == "shared_warm";
    let scalar = serde_json::json!([]);
    let latent = serde_json::json!(["batch", 16, 384]);
    if !descriptor(&scope.inputs, "role_is_critic", "BOOL", &scalar)
        || !descriptor(&scope.outputs, "is_critic", "BOOL", &scalar)
    {
        return Err(policy_error("V3 root role ABI is not scalar BOOL"));
    }
    let (mode, initial_owner, initial, seed) = if warm {
        if graph.serialized_graph_name != WARM_GRAPH
            || !descriptor(&scope.inputs, "warm_start", "BOOL", &scalar)
            || !descriptor(&scope.inputs, "initial_latent", "FLOAT", &latent)
        {
            return Err(policy_error("V3 Warm root name or seed/mode ABI differs"));
        }
        let modes: Vec<_> = scope
            .nodes
            .iter()
            .filter(|node| node.name == WARM_MODE)
            .collect();
        if modes.len() != 1 {
            return Err(policy_error("V3 Warm mode is absent or duplicated"));
        }
        let mode = modes[0];
        if mode.op != "If"
            || !mode.domain.is_empty()
            || mode.inputs != ["warm_start"]
            || mode.outputs.len() != 1
            || mode.static_annotation != WARM_ANNOTATION
            || !mode.constant_integral_seeds.is_empty()
            || !exported_serial_name(&mode.outputs[0], "selected_private_initial_")
        {
            return Err(policy_error(
                "V3 Warm mode is not the registered initial selector",
            ));
        }
        let (warm_branch, fresh_branch) = direct_if_branches(scope, mode)?;
        let warm_output = format!("{}__warm_branch", mode.outputs[0]);
        let fresh_output = format!("{}__fresh_branch", mode.outputs[0]);
        if !empty_branch(warm_branch)
            || !empty_branch(fresh_branch)
            || warm_branch.outputs.len() != 1
            || fresh_branch.outputs.len() != 1
            || !descriptor(&warm_branch.outputs, &warm_output, "FLOAT", &latent)
            || !descriptor(&fresh_branch.outputs, &fresh_output, "FLOAT", &latent)
            || warm_branch.nodes.len() != 1
            || !warm_branch.subgraphs.is_empty()
            || fresh_branch.nodes.len() < 2
        {
            return Err(policy_error(
                "V3 Warm branches copy inputs/weights or omit full Fresh initialization",
            ));
        }
        let seed = &warm_branch.nodes[0];
        if seed.index != 0
            || seed.name != WARM_SEED
            || seed.op != "Identity"
            || !seed.domain.is_empty()
            || seed.inputs != ["initial_latent"]
            || seed.outputs != [warm_output]
            || seed.static_annotation != "none"
            || !seed.constant_integral_seeds.is_empty()
            || seed
                .attribute_names
                .as_ref()
                .is_none_or(|names| !names.is_empty())
        {
            return Err(policy_error(
                "V3 Warm branch is not one complete accepted-seed Identity",
            ));
        }
        let initial = fresh_branch
            .nodes
            .last()
            .ok_or_else(|| policy_error("V3 Fresh prefix is empty"))?;
        if initial.outputs != [fresh_output] {
            return Err(policy_error(
                "V3 Fresh prefix does not end in its original initial If",
            ));
        }
        (Some(mode), fresh_branch, initial, Some((warm_branch, seed)))
    } else {
        if scope
            .inputs
            .iter()
            .any(|value| matches!(value.name.as_str(), "warm_start" | "initial_latent"))
        {
            return Err(policy_error("V3 Fresh graph includes Warm ABI"));
        }
        let initials: Vec<_> = scope
            .nodes
            .iter()
            .filter(|node| exported_serial_name(&node.name, "route_private_initial_"))
            .collect();
        if initials.len() != 1 {
            return Err(policy_error(
                "V3 Fresh initial route is absent or duplicated",
            ));
        }
        (None, scope, initials[0], None)
    };
    if initial.op != "If"
        || !initial.domain.is_empty()
        || !exported_serial_name(&initial.name, "route_private_initial_")
        || initial.inputs != ["role_is_critic"]
        || initial.outputs.len() != 1
        || initial.static_annotation != "role_if_dispatch_candidate"
        || !initial.constant_integral_seeds.is_empty()
    {
        return Err(policy_error(
            "V3 initial private route is not the original role If",
        ));
    }
    let mut role_consumers = Vec::new();
    consumers(scope, "role_is_critic", &mut role_consumers);
    let mut conditions = Vec::new();
    let mut destination = None;
    let mut initial_count = 0;
    let mut ffn_count = 0;
    let mut heads_count = 0;
    for (owner, node) in role_consumers {
        if node.inputs != ["role_is_critic"] {
            return Err(policy_error("V3 role scalar has a data consumer"));
        }
        if owner.path == scope.path && node.op == "Identity" {
            if node.name != "output_is_critic"
                || node.outputs != ["is_critic"]
                || node.static_annotation != "role_scalar_identity_candidate"
                || destination.replace(node.name.clone()).is_some()
                || node
                    .attribute_names
                    .as_ref()
                    .is_none_or(|names| !names.is_empty())
            {
                return Err(policy_error(
                    "V3 role scalar destination is not exact and unique",
                ));
            }
        } else if node.op == "If"
            && node.static_annotation == "role_if_dispatch_candidate"
            && ((owner.path == initial_owner.path && node.name == initial.name)
                || (owner.path == scope.path
                    && (exported_serial_name(&node.name, "route_private_ffn_")
                        || exported_serial_name(&node.name, "route_private_heads_"))))
        {
            let (then, otherwise) = direct_if_branches(owner, node)?;
            if [then, otherwise].iter().any(|branch| {
                !empty_branch(branch)
                    || branch.outputs.len() != node.outputs.len()
                    || branch.outputs.iter().any(|value| value.dtype != "FLOAT")
            }) {
                return Err(policy_error(
                    "V3 role branch copies IO/weights or differs from exact floating outputs",
                ));
            }
            if node.name == initial.name {
                initial_count += 1;
            } else if exported_serial_name(&node.name, "route_private_ffn_") {
                ffn_count += 1;
            } else if exported_serial_name(&node.name, "route_private_heads_") {
                heads_count += 1;
            }
            conditions.push(node.name.clone());
        } else {
            return Err(policy_error("V3 role scalar has an unregistered consumer"));
        }
    }
    if conditions.len() != 6 || initial_count != 1 || ffn_count != 4 || heads_count != 1 {
        return Err(policy_error(
            "V3 P/C route is not exactly one initial, four FFN, one head If",
        ));
    }
    let destination = destination.ok_or_else(|| policy_error("V3 role destination is absent"))?;
    let mut required: BTreeSet<_> = conditions.iter().cloned().collect();
    required.insert(destination.clone());
    if let (Some(mode), Some((warm_branch, seed))) = (mode, seed) {
        let mut mode_consumers = Vec::new();
        consumers(scope, "warm_start", &mut mode_consumers);
        let mut seed_consumers = Vec::new();
        consumers(scope, "initial_latent", &mut seed_consumers);
        if mode_consumers.len() != 1
            || mode_consumers[0].0.path != scope.path
            || mode_consumers[0].1.name != mode.name
            || seed_consumers.len() != 1
            || seed_consumers[0].0.path != warm_branch.path
            || seed_consumers[0].1.name != seed.name
        {
            return Err(policy_error(
                "V3 Warm seed/mode has an additional or wrong consumer",
            ));
        }
        required.insert(mode.name.clone());
        required.insert(seed.name.clone());
    }
    Ok(Route {
        root: scope.path.clone(),
        mode: mode.map(|value| value.name.clone()),
        role_transfer: RoleTransferPolicy {
            scope: scope.path.clone(),
            destination_node: destination,
            host_condition_nodes: conditions,
        },
        required,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn value(name: &str, dtype: &str, shape: Value) -> Value {
        json!({"name":name,"dtype":dtype,"shape":shape})
    }
    fn node(
        index: usize,
        name: &str,
        op: &str,
        inputs: &[&str],
        outputs: &[&str],
        annotation: &str,
        origins: &[&str],
    ) -> Value {
        json!({"index":index,"name":name,"op":op,"domain":"","inputs":inputs,"outputs":outputs,
            "input_origins":origins,"static_annotation":annotation,"actual_provider":"unknown",
            "constant_integral_seeds":[],"attribute_names":if op=="If" {json!(["else_branch","then_branch"])}else{json!([])}})
    }
    fn scope(
        path: &str,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        nodes: Vec<Value>,
        children: Vec<Value>,
    ) -> Value {
        json!({"path":path,"inputs":inputs,"outputs":outputs,"nodes":nodes,"subgraphs":children,
            "integral_initializers":[],"initializer_count":0})
    }
    fn graph(role: &str, layout: &str, mut inventory: Value) -> InventoryGraphV3 {
        fn ledger(
            scope: &Value,
            weights: &mut BTreeMap<String, Value>,
            dependencies: &mut Vec<Value>,
        ) {
            for node in scope["nodes"].as_array().unwrap() {
                for input in node["inputs"].as_array().unwrap() {
                    let initializer = input.as_str().unwrap();
                    if !initializer.starts_with("experts.") && !initializer.starts_with("shared.") {
                        continue;
                    }
                    let parameter = initializer
                        .strip_suffix(".linear_transpose")
                        .unwrap_or(initializer);
                    weights.entry(initializer.into()).or_insert_with(|| json!({
                        "initializer":initializer,"parameter":parameter,"dtype":"FLOAT","shape":[384,384],
                        "bytes":384*384*4,"sha256":"22".repeat(32),"source_parameter_sha256":"33".repeat(32)}));
                    if initializer.starts_with("experts.") && major(node["op"].as_str().unwrap()) {
                        dependencies.push(json!({"scope_path":scope["path"],"node_index":node["index"],
                            "node_name":node["name"],"op":node["op"],"initializer":initializer,"parameter":parameter}));
                    }
                }
            }
            for child in scope["subgraphs"].as_array().unwrap() {
                ledger(child, weights, dependencies);
            }
        }
        let mut weights = BTreeMap::new();
        let mut dependencies = Vec::new();
        ledger(&inventory, &mut weights, &mut dependencies);
        inventory["initializer_count"] =
            json!(weights.len() + inventory["integral_initializers"].as_array().unwrap().len());
        InventoryGraphV3 {
            role: role.into(),
            file: "unused_source_fixture.onnx".into(),
            sha256: "11".repeat(32),
            serialized_graph_name: layout.into(),
            inventory: serde_json::from_value(inventory).unwrap(),
            weight_bindings: weights
                .into_values()
                .map(|value| serde_json::from_value(value).unwrap())
                .collect(),
            private_neural_dependencies: dependencies
                .into_iter()
                .map(|value| serde_json::from_value(value).unwrap())
                .collect(),
        }
    }
    fn metadata_graph() -> Value {
        let mut result = scope(
            "public",
            vec![value("data", "FLOAT", json!([1, 384]))],
            vec![value("nn_out", "FLOAT", json!([1, 384]))],
            vec![
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
                    "index_metadata",
                    "Gather",
                    &["dimensions", "index"],
                    &["dimension"],
                    "shape_metadata_chain_candidate",
                    &["shape_metadata", "small_integral_constant"],
                ),
                node(
                    2,
                    "nn",
                    "MatMul",
                    &["data", "shared.weight"],
                    &["nn_out"],
                    "major_nn_requires_cuda",
                    &["model_data_or_unknown", "model_data_or_unknown"],
                ),
            ],
            vec![],
        );
        result["initializer_count"] = json!(1);
        result["integral_initializers"] = json!([{"name":"index","dtype":"INT64","shape":[],"source":"graph_initializer_metadata"}]);
        result
    }
    fn pc_graph(root: &str) -> Value {
        let latent = json!(["batch", 16, 384]);
        let mut nodes = Vec::new();
        let mut children = Vec::new();
        for index in 0..6 {
            let name = if index == 0 {
                "route_private_initial_1".into()
            } else if index == 5 {
                "route_private_heads_6".into()
            } else {
                format!("route_private_ffn_{}", index + 1)
            };
            let output = if index == 0 {
                "selected_private_initial_7".into()
            } else {
                format!("selected_{index}")
            };
            nodes.push(node(
                index,
                &name,
                "If",
                &["role_is_critic"],
                &[&output],
                "role_if_dispatch_candidate",
                &["role_control"],
            ));
            for (suffix, role) in [("then_branch", "critic"), ("else_branch", "proposer")] {
                let path = format!("{root}/node{index}:{name}/{suffix}");
                let private_output = format!("{role}_{index}_out");
                let weight = format!("experts.{role}.projection.weight");
                children.push(scope(
                    &path,
                    vec![],
                    vec![value(&private_output, "FLOAT", latent.clone())],
                    vec![node(
                        0,
                        &format!("{role}_{index}_nn"),
                        "MatMul",
                        &["data", &weight],
                        &[&private_output],
                        "major_nn_requires_cuda",
                        &["model_data_or_unknown", "model_data_or_unknown"],
                    )],
                    vec![],
                ));
            }
        }
        nodes.push(node(
            6,
            "output_is_critic",
            "Identity",
            &["role_is_critic"],
            &["is_critic"],
            "role_scalar_identity_candidate",
            &["role_control"],
        ));
        scope(
            root,
            vec![
                value("role_is_critic", "BOOL", json!([])),
                value("data", "FLOAT", latent),
            ],
            vec![value("is_critic", "BOOL", json!([]))],
            nodes,
            children,
        )
    }

    fn warm_graph() -> Value {
        let root = "shared_warm";
        let mut result = pc_graph(root);
        let selected = "selected_private_initial_7";
        let fresh_path = format!("{root}/node0:{WARM_MODE}/else_branch");
        let seed_path = format!("{root}/node0:{WARM_MODE}/then_branch");
        let initial_name = "route_private_initial_1";
        let fresh_output = format!("{selected}__fresh_branch");
        let seed_output = format!("{selected}__warm_branch");
        let mut initial = result["nodes"][0].clone();
        initial["index"] = json!(1);
        initial["outputs"] = json!([fresh_output]);
        let mut branches = Vec::new();
        for suffix in ["then_branch", "else_branch"] {
            let old = format!("{root}/node0:{initial_name}/{suffix}");
            let index = result["subgraphs"]
                .as_array()
                .unwrap()
                .iter()
                .position(|branch| branch["path"] == json!(old))
                .unwrap();
            let mut branch = result["subgraphs"].as_array_mut().unwrap().remove(index);
            branch["path"] = json!(format!("{fresh_path}/node1:{initial_name}/{suffix}"));
            branches.push(branch);
        }
        let latent = json!(["batch", 16, 384]);
        result["subgraphs"].as_array_mut().unwrap().push(scope(
            &fresh_path,
            vec![],
            vec![value(&fresh_output, "FLOAT", latent.clone())],
            vec![
                node(
                    0,
                    "fresh_prefix_nn",
                    "MatMul",
                    &["data", "shared.query.weight"],
                    &["prepared_query"],
                    "major_nn_requires_cuda",
                    &["model_data_or_unknown", "model_data_or_unknown"],
                ),
                initial,
            ],
            branches,
        ));
        result["subgraphs"].as_array_mut().unwrap().push(scope(
            &seed_path,
            vec![],
            vec![value(&seed_output, "FLOAT", latent.clone())],
            vec![node(
                0,
                WARM_SEED,
                "Identity",
                &["initial_latent"],
                &[&seed_output],
                "none",
                &["model_data_or_unknown"],
            )],
            vec![],
        ));
        result["nodes"][0] = node(
            0,
            WARM_MODE,
            "If",
            &["warm_start"],
            &[selected],
            WARM_ANNOTATION,
            &["virtual_warm_mode_control"],
        );
        result["inputs"].as_array_mut().unwrap().extend([
            value("initial_latent", "FLOAT", latent),
            value("warm_start", "BOOL", json!([])),
        ]);
        result
    }

    #[test]
    fn lexical_metadata_permission_cannot_include_data_or_unbounded_integral_seed() {
        let source = metadata_graph();
        let mut count = 2;
        let (policy, v, task) =
            inspect(&graph("public", PUBLIC_GRAPH, source.clone()), &mut count).unwrap();
        assert_eq!(count, 5);
        assert_eq!(policy.controls.len(), 2);
        assert_eq!(policy.major.len(), 1);
        assert!(v.is_empty() && task.is_empty());
        let mut data = source.clone();
        data["nodes"][1]["inputs"][0] = json!("data");
        data["nodes"][1]["input_origins"][0] = json!("model_data_or_unknown");
        assert!(inspect(&graph("public", PUBLIC_GRAPH, data), &mut 0).is_err());
        let mut unbounded = source;
        unbounded["integral_initializers"][0]["shape"] = json!([65]);
        assert!(inspect(&graph("public", PUBLIC_GRAPH, unbounded), &mut 0).is_err());
    }

    #[test]
    fn fresh_role_route_requires_six_named_branches_and_unique_scalar_destination() {
        let source = pc_graph("shared_pc");
        let (policy, _, _) =
            inspect(&graph("shared_pc", FRESH_LAYOUT, source.clone()), &mut 0).unwrap();
        assert_eq!(policy.required_cuda_dispatch.len(), 7);
        assert_eq!(policy.proposer_private.len(), 6);
        assert_eq!(policy.critic_private.len(), 6);
        let mut shadow = source.clone();
        shadow["subgraphs"][0]["inputs"] = json!([value("role_is_critic", "BOOL", json!([]))]);
        assert!(inspect(&graph("shared_pc", FRESH_LAYOUT, shadow), &mut 0).is_err());
        let mut missing = source.clone();
        missing["subgraphs"].as_array_mut().unwrap().pop();
        assert!(inspect(&graph("shared_pc", FRESH_LAYOUT, missing), &mut 0).is_err());
        let mut vector = source;
        vector["inputs"][0]["shape"] = json!([1]);
        assert!(inspect(&graph("shared_pc", FRESH_LAYOUT, vector), &mut 0).is_err());
    }

    #[test]
    fn validator_needs_its_own_named_private_and_task_head_kernels() {
        let source = scope(
            "validator",
            vec![value("data", "FLOAT", json!([1, 384]))],
            vec![value("task_logits", "FLOAT", json!([1, 7]))],
            vec![
                node(
                    0,
                    "v_private",
                    "MatMul",
                    &[
                        "data",
                        "experts.validator.query_projection.weight.linear_transpose",
                    ],
                    &["v_context"],
                    "major_nn_requires_cuda",
                    &["model_data_or_unknown", "model_data_or_unknown"],
                ),
                node(
                    1,
                    "v_task",
                    "MatMul",
                    &[
                        "v_context",
                        "experts.validator.task_head.weight.linear_transpose",
                    ],
                    &["task_logits"],
                    "major_nn_requires_cuda",
                    &["model_data_or_unknown", "model_data_or_unknown"],
                ),
            ],
            vec![],
        );
        let (_, private, task) =
            inspect(&graph("validator", VALIDATOR_GRAPH, source.clone()), &mut 0).unwrap();
        assert_eq!(private.len(), 1);
        assert_eq!(task.len(), 1);
        assert!(!private.contains_key("v_task"));
        assert!(!task.contains_key("v_private"));
        let task_only_profile = serde_json::to_vec(&json!([{
            "cat":"Node","name":"v_task_kernel_time",
            "args":{"provider":CUDA,"op_name":"MatMul"}
        }]))
        .unwrap();
        assert_eq!(count_cuda_named(&task_only_profile, &private).unwrap(), 0);
        assert_eq!(count_cuda_named(&task_only_profile, &task).unwrap(), 1);
        let mut mislabeled = source;
        mislabeled["nodes"][1]["inputs"][1] =
            json!("experts.validator.wdl_head.weight.linear_transpose");
        assert!(inspect(&graph("validator", VALIDATOR_GRAPH, mislabeled), &mut 0).is_err());
    }

    #[test]
    fn single_task_kernel_with_both_weights_cannot_replace_non_task_private_evidence() {
        let source = scope(
            "validator",
            vec![value("data", "FLOAT", json!([1, 384]))],
            vec![value("task_logits", "FLOAT", json!([1, 7]))],
            vec![node(
                0,
                "v_task",
                "Gemm",
                &[
                    "data",
                    "experts.validator.query_projection.weight.linear_transpose",
                    "experts.validator.task_head.weight.linear_transpose",
                ],
                &["task_logits"],
                "major_nn_requires_cuda",
                &[
                    "model_data_or_unknown",
                    "model_data_or_unknown",
                    "model_data_or_unknown",
                ],
            )],
            vec![],
        );
        let registered = graph("validator", VALIDATOR_GRAPH, source);
        let (_, private, task) = inspect_private_bindings(&registered).unwrap();
        assert!(private.is_empty());
        assert_eq!(task.len(), 1);
        assert!(inspect(&registered, &mut 0).is_err());
    }

    #[test]
    fn warm_route_covers_full_fresh_prefix_complete_seed_and_all_six_role_dispatches() {
        let source = warm_graph();
        let (policy, _, _) =
            inspect(&graph("shared_warm", WARM_GRAPH, source.clone()), &mut 0).unwrap();
        assert_eq!(policy.required_cuda_dispatch.len(), 9);
        assert!(policy.required_cuda_dispatch.contains(WARM_SEED));
        assert!(policy.required_cuda_dispatch.contains(WARM_MODE));
        assert_eq!(
            policy
                .role_transfer
                .as_ref()
                .unwrap()
                .host_condition_nodes
                .len(),
            6
        );
        let mut legacy = source.clone();
        legacy["nodes"][0]["input_origins"][0] = json!("cuda_warm_mode_control");
        assert!(inspect(&graph("shared_warm", WARM_GRAPH, legacy), &mut 0).is_err());
        let mut partial = source;
        let seed = partial["subgraphs"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|branch| {
                branch["path"].as_str().unwrap().ends_with("/then_branch")
                    && branch["nodes"][0]["name"] == json!(WARM_SEED)
            })
            .unwrap();
        seed["nodes"][0]["op"] = json!("Slice");
        assert!(inspect(&graph("shared_warm", WARM_GRAPH, partial), &mut 0).is_err());
    }

    #[test]
    fn graph_budget_is_separate_from_total_inventory_and_errors_do_not_change_total() {
        let source = metadata_graph();
        let mut count = MAX_VIRTUAL_NODES - 3;
        inspect(&graph("public", PUBLIC_GRAPH, source.clone()), &mut count).unwrap();
        assert_eq!(count, MAX_VIRTUAL_NODES);
        assert!(inspect(&graph("public", PUBLIC_GRAPH, source), &mut count).is_err());
        assert_eq!(count, MAX_VIRTUAL_NODES);
    }

    #[test]
    fn actual_private_dependencies_cannot_be_omitted_added_or_retyped_as_metadata() {
        let source = pc_graph("shared_pc");
        let mut missing = graph("shared_pc", FRESH_LAYOUT, source.clone());
        missing.private_neural_dependencies.pop();
        assert!(inspect(&missing, &mut 0).is_err());
        let mut wrong = graph("shared_pc", FRESH_LAYOUT, source.clone());
        wrong.private_neural_dependencies[0].node_name = "other_named_kernel".into();
        assert!(inspect(&wrong, &mut 0).is_err());
        let mut dtype = graph("shared_pc", FRESH_LAYOUT, source.clone());
        dtype.weight_bindings[0].dtype = "INT64".into();
        assert!(inspect(&dtype, &mut 0).is_err());
        let mut shadow = graph("shared_pc", FRESH_LAYOUT, source);
        shadow.inventory.initializer_count = Some(shadow.weight_bindings.len() + 1);
        shadow.inventory.integral_initializers.push(IntegralSeed {
            name: shadow.weight_bindings[0].initializer.clone(),
            dtype: "INT64".into(),
            shape: vec![],
            source: "graph_initializer_metadata".into(),
        });
        assert!(inspect(&shadow, &mut 0).is_err());
    }
}
