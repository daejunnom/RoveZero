use super::*;
use serde_json::json;
fn schema_row(role: GenerationRoleV3) -> GenerationRecordV3 {
    // This deliberately tiny schema fixture cannot meet actual generation
    // admission counts. No engine, teacher, model or optimizer executes here.
    let anchor = GenerationRulesAnchorV3 {
        rules_state_sha256: "a".repeat(64),
        rules_history_sha256: "b".repeat(64),
        board_fen: "8/8/8/8/8/8/8/K6k w - - 0 1".into(),
        white_to_move: true,
    };
    let view = GenerationSelectedViewV3 {
        context_policy_sha256: "c".repeat(64),
        anchors: vec![anchor.clone()],
        records: vec![],
        query: GenerationSelectedQueryV3 {
            purpose: "propose_policy".into(),
            anchors: [0; 3],
            prefix: vec![],
            proposal: vec![],
            counter: vec![],
            counter_present: false,
            white_to_move: true,
        },
        token_budget: 512,
        length_bucket: 8,
    };
    let input = GenerationInputV3 {
        game_id: "g1".into(),
        opening_id: "o1".into(),
        line_genealogy_id: "l1".into(),
        transposition_sha256: "d".repeat(64),
        role,
        model_epoch: 0,
        context_epoch: 0,
        capture_sequence: 10,
        question_id: "q1".into(),
        worker_id: "w1".into(),
        job_id: "j1".into(),
        slice_id: "s1".into(),
        rules_anchor: anchor,
        actual_history: vec![64],
        view_sha256: view.sha256().unwrap(),
        selected_view: view,
        legal_moves: vec![64, 128],
        action_support_mask: [true; 7],
        divergence_record_ids: vec![],
        encoded_model_input: json!({"schema":"rovezero.pals-virtual-review-input.v1","role":role,"anchors":[{"board":vec![0;64],"metadata":vec![0.0;16]}],"records":[],"query":{"purpose":"propose_policy","perspective":1,"anchors":[0,0,0],"prefix":[],"proposal":[],"counter":[],"counter_present":false},"candidates":[[0,1,0],[0,2,0]],"divergences":[],"token_budget":512}),
    };
    GenerationRecordV3 {
        schema: GENERATION_DATA_DOMAIN.into(),
        source_kind: "actual_selfplay".into(),
        eligibility: "actual_generation_candidate".into(),
        input_sha256: input.sha256().unwrap(),
        input,
        future_targets: GenerationTargetsV3::default(),
    }
}
#[test]
fn frozen_action_alias_codec_preserves_originals_and_rejects_reordered_or_unknown_fields() {
    // Codec-only fixture: no native chain, asset proof, teacher or generation
    // admission exists, and no actual execution is claimed by this test.
    let actions = vec![0u8, 1];
    let schedule = generation_digest("rz-pals-generation-action-enumeration/1", &actions).unwrap();
    let mut group = GenerationFrozenActionGroupReceiptV3 {
        schema: GENERATION_ACTION_GROUP_DOMAIN.into(),
        input_sha256: "1".repeat(64),
        group_sha256: String::new(),
        semantic_input_sha256: "2".repeat(64),
        frozen_context_sha256: "3".repeat(64),
        frozen_pc_epoch: 1,
        frozen_pc_weights_sha256: "4".repeat(64),
        registered_actions: actions,
        action_schedule_sha256: schedule,
        selection_provenance: "registered-supported-enumeration-v1".into(),
        members: vec![
            GenerationFrozenActionMemberV3 {
                action: GenerationTaskV3::DefendResponse,
                index: 0,
                actual_input_sha256: "1".repeat(64),
                causal_receipt_sha256: "5".repeat(64),
                action_authority_sha256: "6".repeat(64),
            },
            GenerationFrozenActionMemberV3 {
                action: GenerationTaskV3::AttackRepair,
                index: 1,
                actual_input_sha256: "7".repeat(64),
                causal_receipt_sha256: "8".repeat(64),
                action_authority_sha256: "9".repeat(64),
            },
        ],
    };
    group.group_sha256 = generation_digest(
        "rz-pals-generation-frozen-action-group/1",
        &json!([
            group.semantic_input_sha256,
            group.frozen_context_sha256,
            group.frozen_pc_epoch,
            group.frozen_pc_weights_sha256,
            group.registered_actions,
            group.action_schedule_sha256
        ]),
    )
    .unwrap();
    group.validate().unwrap();
    let original = json!(group);
    let mut copied: GenerationFrozenActionGroupReceiptV3 =
        serde_json::from_value(original.clone()).unwrap();
    assert_eq!(copied.members[1].actual_input_sha256, "7".repeat(64));
    copied.members.swap(0, 1);
    assert!(copied.validate().is_err());
    copied = serde_json::from_value(original.clone()).unwrap();
    copied.members[1].actual_input_sha256 = copied.members[0].actual_input_sha256.clone();
    assert!(copied.validate().is_err());
    copied = serde_json::from_value(original.clone()).unwrap();
    copied.frozen_pc_weights_sha256 = "a".repeat(64);
    assert!(copied.validate().is_err());
    let mut unknown = original;
    unknown["completed"] = json!(true);
    assert!(serde_json::from_value::<GenerationFrozenActionGroupReceiptV3>(unknown).is_err());
}
#[test]
fn python_independent_v3_canonical_vectors_match() {
    let record = schema_row(GenerationRoleV3::Proposer);
    // Calculated by generation_data.view_digest/seal_input on the equivalent
    // pure Python schema fixture, with neural imports forbidden.
    assert_eq!(
        record.input.view_sha256,
        "6e14cdf813f72d8c213b84d1d25c2250cfb63d5be264e428be85081fb031f0c0"
    );
    assert_eq!(
        record.input_sha256,
        "4d7c90754c1ff149dc1289d90401346a40a2ec18fef5936ed2888c730ac75205"
    );
}
fn outcome(
    action: GenerationTaskV3,
    witnesses: usize,
    cost: Option<u64>,
) -> GenerationActionOutcomeV3 {
    GenerationActionOutcomeV3 {
        action,
        execution_id: format!("e-{}", action.index()),
        completed: true,
        wall_cost_ns: cost,
        new_witnesses: (0..witnesses)
            .map(|i| GenerationWitnessV3 {
                witness_id: format!("new-{}-{i}", action.index()),
                kind: GenerationRecordKindV3::Counter,
                observed_sequence: 15,
                raw_evidence_sha256: "9".repeat(64),
            })
            .collect(),
        causal_receipt_sha256: "8".repeat(64),
    }
}
#[test]
fn input_seal_survives_targets_but_rejects_future_and_history_changes() {
    let mut row = schema_row(GenerationRoleV3::Proposer);
    row.validate().unwrap();
    let before = row.input_sha256.clone();
    row.input.actual_history.push(192);
    assert!(row.validate().is_err());
    row.input_sha256 = row.input.sha256().unwrap();
    assert_ne!(before, row.input_sha256);
    row.input.encoded_model_input["actual_history"] = json!([]);
    assert!(row.input.validate().is_err());
}
#[test]
fn f32_seal_projection_ignores_decimal_json_rendering_but_preserves_actual_bits() {
    let mut a = schema_row(GenerationRoleV3::Proposer).input;
    let mut b = a.clone();
    a.encoded_model_input["anchors"][0]["metadata"][9] = json!(0.00006103515625f64);
    b.encoded_model_input["anchors"][0]["metadata"][9] = json!(6.103515625e-5f64);
    assert_eq!(a.sha256().unwrap(), b.sha256().unwrap());
    b.encoded_model_input["anchors"][0]["metadata"][9] = json!(0.00006104f64);
    assert_ne!(a.sha256().unwrap(), b.sha256().unwrap());
}
#[test]
fn teacher_policy_order_and_wdl_perspective_are_independent() {
    let row = schema_row(GenerationRoleV3::Proposer);
    let mut policy = GenerationPolicyTargetV3 {
        observed_sequence: 20,
        question_id: "q1".into(),
        view_sha256: row.input.view_sha256.clone(),
        raw_evidence_sha256: "e".repeat(64),
        source: "stockfish19_completed_multipv".into(),
        completed: true,
        stale: false,
        anchor: row.input.rules_anchor.clone(),
        ranked_moves: vec![128, 64],
        teacher_binary_sha256: "6".repeat(64),
        teacher_profile_sha256: "7".repeat(64),
        go_sequence: 1,
        completed_depth: 1,
    };
    assert!(
        (policy.probabilities(&row.input).unwrap()[0] - 1.0 / (1.0 + 1.0f64.exp())).abs() < 1e-15
    );
    policy.stale = true;
    assert!(policy.probabilities(&row.input).is_err());
    policy.stale = false;
    policy.ranked_moves.push(192);
    assert!(policy.probabilities(&row.input).is_err());
    let mut wdl = GenerationWdlTargetV3 {
        observed_sequence: 20,
        question_id: "q1".into(),
        view_sha256: row.input.view_sha256.clone(),
        raw_evidence_sha256: "e".repeat(64),
        source: "rules_selfplay_trajectory".into(),
        game_id: "g1".into(),
        anchor: row.input.rules_anchor.clone(),
        trajectory_membership_sha256: "f".repeat(64),
        rules_result_sha256: "a".repeat(64),
        ending: GenerationRulesEndingV3::Checkmate,
        outcome: GenerationOutcomeV3::WhiteWin,
    };
    assert_eq!(wdl.probabilities(&row.input).unwrap(), [1., 0., 0.]);
    wdl.outcome = GenerationOutcomeV3::BlackWin;
    assert_eq!(wdl.probabilities(&row.input).unwrap(), [0., 0., 1.]);
    wdl.source = "stockfish_mate".into();
    assert!(wdl.probabilities(&row.input).is_err());
}
#[test]
fn pareto_crossing_unknown_and_empty_defer_do_not_supply_utility() {
    let input = schema_row(GenerationRoleV3::Validator).input;
    let result = generation_derive_tasks(
        &input,
        &[
            outcome(GenerationTaskV3::AttackRepair, 2, Some(20)),
            outcome(GenerationTaskV3::DefendResponse, 1, Some(10)),
            outcome(GenerationTaskV3::Defer, 0, Some(0)),
        ],
    )
    .unwrap();
    assert!(result.pairs.is_empty());
    let result = generation_derive_tasks(
        &input,
        &[
            outcome(GenerationTaskV3::AttackRepair, 2, Some(10)),
            outcome(GenerationTaskV3::DefendResponse, 0, Some(20)),
            outcome(GenerationTaskV3::ResumeTask, 5, None),
        ],
    )
    .unwrap();
    assert_eq!(
        result.pairs,
        vec![[
            GenerationTaskV3::AttackRepair.index() as u8,
            GenerationTaskV3::DefendResponse.index() as u8
        ]]
    );
    assert!(result.new_witness_context);
    let result = generation_derive_tasks(
        &input,
        &[
            outcome(GenerationTaskV3::AttackRepair, 0, Some(10)),
            outcome(GenerationTaskV3::Defer, 0, Some(0)),
        ],
    )
    .unwrap();
    assert!(result.pairs.is_empty());
}
#[test]
fn support_and_known_masks_are_separate_and_old_witness_rejected() {
    let mut input = schema_row(GenerationRoleV3::Validator).input;
    let mut actual = outcome(GenerationTaskV3::AttackRepair, 1, Some(10));
    actual.new_witnesses[0].observed_sequence = 10;
    assert!(generation_derive_tasks(&input, &[actual.clone()]).is_err());
    actual.new_witnesses[0].observed_sequence = 11;
    input.action_support_mask[1] = false;
    assert!(generation_derive_tasks(&input, &[actual]).is_err());
}
#[test]
fn raw_receipt_strings_cannot_replace_full_head_evidence() {
    let mut row = schema_row(GenerationRoleV3::Proposer);
    row.future_targets.policy = Some(GenerationPolicyTargetV3 {
        observed_sequence: 20,
        question_id: "q1".into(),
        view_sha256: row.input.view_sha256.clone(),
        raw_evidence_sha256: "e".repeat(64),
        source: "stockfish19_completed_multipv".into(),
        completed: true,
        stale: false,
        anchor: row.input.rules_anchor.clone(),
        ranked_moves: vec![64],
        teacher_binary_sha256: "6".repeat(64),
        teacher_profile_sha256: "7".repeat(64),
        go_sequence: 1,
        completed_depth: 1,
    });
    assert!(
        validate_generation_head_evidence(&row, &BTreeMap::from([("e".repeat(64), json!("pass"))]))
            .is_err()
    );
}

fn teacher_evidence_codec_fixture() -> (GenerationRecordV3, BTreeMap<String, serde_json::Value>) {
    // Codec-only synthetic receipts exercise rejection/joins. They do not
    // mint the private native/worker capabilities or admit a real generation.
    let mut row = schema_row(GenerationRoleV3::Proposer);
    row.input.worker_id = "native-owner-55".into();
    row.input_sha256 = row.input.sha256().unwrap();
    let mut target = GenerationPolicyTargetV3 {
        observed_sequence: 20,
        question_id: row.input.question_id.clone(),
        view_sha256: row.input.view_sha256.clone(),
        raw_evidence_sha256: "0".repeat(64),
        source: "stockfish19_completed_multipv".into(),
        completed: true,
        stale: false,
        anchor: row.input.rules_anchor.clone(),
        ranked_moves: vec![64],
        teacher_binary_sha256: "6".repeat(64),
        teacher_profile_sha256: "7".repeat(64),
        go_sequence: 1,
        completed_depth: 1,
    };
    let mut value = serde_json::to_value(&target).unwrap();
    value.as_object_mut().unwrap().remove("raw_evidence_sha256");
    let head = json!({"schema":"rz-pals-generation-head-evidence/1","input_sha256":row.input_sha256,"head":"policy","target":value});
    let head_sha = generation_digest(head["schema"].as_str().unwrap(), &head).unwrap();
    target.raw_evidence_sha256 = head_sha.clone();
    row.future_targets.policy = Some(target.clone());
    let submitted = json!({"schema":"rz-pals-generation-teacher-submission-evidence/1","observed_sequence":11,
        "execution_epoch":55,"execution_sequence":3,"job_sequence":2,"root_generation":0,
        "teacher_binary_sha256":target.teacher_binary_sha256,"teacher_profile_sha256":target.teacher_profile_sha256,
        "rules_anchor":row.input.rules_anchor,"process":{"pid":50,"process_group":50,"proc_start_ticks":100},
        "evaluation_namespace":"a".repeat(64),"uci_name":"fixture","source_cpu_question_sha256":"b".repeat(64),
        "source_cpu_view_sha256":"c".repeat(64)});
    let completed = json!({"schema":"rz-pals-generation-teacher-completion-evidence/1","observed_sequence":20,
        "submission_observed_sequence":11,"execution_epoch":55,"execution_sequence":3,"job_sequence":2,"root_generation":0,
        "teacher_binary_sha256":target.teacher_binary_sha256,"teacher_profile_sha256":target.teacher_profile_sha256,
        "physical_process_identity":submitted["process"],"evaluation_namespace":submitted["evaluation_namespace"],
        "successful_owner_join":true,"source_scope_joined":true,"original_question_target_eligible":true,"cancel_sequence":0,
        "exact_resume_supported":false,"go_issue":"Issued","result":{"identity":{"binary_sha256":target.teacher_binary_sha256},
            "uci_name":"fixture","completion":"BestMove","go_issue":"Issued","go_sequence":1,
            "best_move_move16":64,"readiness_barrier_completed":true,"policy_target_known":true,"requested_multipv":1,
            "opinions":[{"rank":1,"depth":1,"bound":"ExactReported","pv_move16":[64]}]}});
    let submitted_sha =
        generation_digest(submitted["schema"].as_str().unwrap(), &submitted).unwrap();
    let completed_sha =
        generation_digest(completed["schema"].as_str().unwrap(), &completed).unwrap();
    let binding = json!({"schema":"rz-pals-generation-teacher-input-binding-evidence/1","input_sha256":row.input_sha256,
        "policy_head_evidence_sha256":head_sha,"teacher_submission_sha256":submitted_sha,"teacher_completion_sha256":completed_sha,
        "rules_anchor":row.input.rules_anchor,"observed_sequence":20,"interpretation":"conditional_exact_endpoint_opinion"});
    let binding_sha = generation_digest(binding["schema"].as_str().unwrap(), &binding).unwrap();
    (
        row,
        BTreeMap::from([
            (head_sha, head),
            (submitted_sha, submitted),
            (completed_sha, completed),
            (binding_sha, binding),
        ]),
    )
}
fn mutate_teacher_completion(
    evidence: &mut BTreeMap<String, serde_json::Value>,
    change: impl FnOnce(&mut serde_json::Value),
) {
    let old = evidence
        .iter()
        .find(|(_, v)| v["schema"] == "rz-pals-generation-teacher-completion-evidence/1")
        .unwrap()
        .0
        .clone();
    let mut completed = evidence.remove(&old).unwrap();
    change(&mut completed);
    let fresh = generation_digest(completed["schema"].as_str().unwrap(), &completed).unwrap();
    evidence.insert(fresh.clone(), completed);
    let old_binding = evidence
        .iter()
        .find(|(_, v)| v["schema"] == "rz-pals-generation-teacher-input-binding-evidence/1")
        .unwrap()
        .0
        .clone();
    let mut binding = evidence.remove(&old_binding).unwrap();
    binding["teacher_completion_sha256"] = json!(fresh);
    let binding_sha = generation_digest(binding["schema"].as_str().unwrap(), &binding).unwrap();
    evidence.insert(binding_sha, binding);
}
#[test]
fn teacher_policy_requires_independent_owner_receipts_beside_head_self_copy() {
    let (row, mut raw) = teacher_evidence_codec_fixture();
    assert!(validate_generation_head_evidence(&row, &raw).is_ok());
    raw.retain(|_, v| v["schema"] == "rz-pals-generation-head-evidence/1");
    assert!(validate_generation_head_evidence(&row, &raw).is_err());
}
#[test]
fn teacher_policy_rejects_late_or_canceled_source_even_with_resealed_hashes() {
    for change in [0, 1, 2] {
        let (row, mut raw) = teacher_evidence_codec_fixture();
        mutate_teacher_completion(&mut raw, |receipt| match change {
            0 => receipt["original_question_target_eligible"] = json!(false),
            1 => receipt["cancel_sequence"] = json!(1),
            _ => receipt["observed_sequence"] = json!(10),
        });
        assert!(validate_generation_head_evidence(&row, &raw).is_err());
    }
}
#[test]
fn teacher_policy_rejects_old_go_owner_and_namespace_even_with_resealed_hashes() {
    for change in [0, 1, 2] {
        let (row, mut raw) = teacher_evidence_codec_fixture();
        mutate_teacher_completion(&mut raw, |receipt| match change {
            0 => receipt["result"]["go_sequence"] = json!(2),
            1 => receipt["physical_process_identity"]["proc_start_ticks"] = json!(101),
            _ => receipt["evaluation_namespace"] = json!("f".repeat(64)),
        });
        assert!(validate_generation_head_evidence(&row, &raw).is_err());
    }
}
#[test]
fn teacher_policy_rejects_partial_bound_or_changed_ranked_move() {
    for change in [0, 1, 2] {
        let (row, mut raw) = teacher_evidence_codec_fixture();
        mutate_teacher_completion(&mut raw, |receipt| match change {
            0 => receipt["result"]["opinions"][0]["bound"] = json!("Upper"),
            1 => receipt["result"]["opinions"][0]["pv_move16"] = json!([128]),
            _ => receipt["result"]["requested_multipv"] = json!(2),
        });
        assert!(validate_generation_head_evidence(&row, &raw).is_err());
    }
}
#[test]
fn all_cross_split_leaked_roots_are_excluded_symmetrically() {
    let a = schema_row(GenerationRoleV3::Proposer);
    let mut b = a.clone();
    b.input.game_id = "g2".into();
    b.input.opening_id = "o2".into();
    b.input.line_genealogy_id = "l2".into();
    b.input_sha256 = b.input.sha256().unwrap();
    let (outputs, report) = generation_split_records(
        &[a, b],
        &BTreeMap::from([
            ("o1".into(), GenerationSplitV3::Train),
            ("o2".into(), GenerationSplitV3::Holdout),
        ]),
    )
    .unwrap();
    assert_eq!(report.cross_split_records_excluded, 2);
    assert!(outputs.values().all(Vec::is_empty));
}
#[test]
fn strict_codec_rejects_old_domain_unknown_fields_duplicate_keys() {
    let row = schema_row(GenerationRoleV3::Proposer);
    let mut value = serde_json::to_value(&row).unwrap();
    value["schema"] = json!("rz-pals-data/2");
    let parsed: GenerationRecordV3 = serde_json::from_value(value).unwrap();
    assert!(parsed.validate().is_err());
    let mut value = serde_json::to_value(row).unwrap();
    value["extra"] = json!(true);
    assert!(serde_json::from_value::<GenerationRecordV3>(value).is_err());
    assert!(
        crate::decode_json::<GenerationRecordV3>("{\"schema\":\"a\",\"schema\":\"b\"}").is_err()
    );
}
