use super::*;

fn arguments() -> Vec<String> {
    [
        "--input-lock=inputs/rovent-input-lock.json",
        "--native-assets=inputs/rovent-native-assets.json",
        "--process-epoch=23",
        "--startup-ms=60000",
        "--root-policy=dynamic-go",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

#[test]
fn product_arguments_require_one_explicit_closed_selection_and_policy() {
    let args = Arguments::parse(&arguments()).unwrap();
    assert_eq!(args.root_policy, RoventRootAdmissionPolicy::DynamicGo);
    assert_eq!(args.process_epoch, 23);
    assert_eq!(args.external_r_sources, None);
    assert!(!args.startup_diagnostics);
    assert!(!args.lifecycle_diagnostics);
    let mut lifecycle = arguments();
    lifecycle.push("--lifecycle-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&lifecycle).unwrap().lifecycle_diagnostics);
    lifecycle.push("--lifecycle-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&lifecycle).is_err());
    let mut diagnostics = arguments();
    diagnostics.push("--startup-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&diagnostics).unwrap().startup_diagnostics);
    diagnostics.push("--startup-diagnostics=stderr-v1".into());
    assert!(Arguments::parse(&diagnostics).is_err());
    let mut exact = arguments();
    exact[4] = "--root-policy=exact-registered".into();
    assert_eq!(
        Arguments::parse(&exact).unwrap().root_policy,
        RoventRootAdmissionPolicy::ExactRegistered
    );
    for index in 0..arguments().len() {
        let mut missing = arguments();
        missing.remove(index);
        assert!(Arguments::parse(&missing).is_err());
    }
    for duplicate in arguments() {
        let mut args = arguments();
        args.push(duplicate);
        assert!(Arguments::parse(&args).is_err());
    }
}

#[test]
fn product_arguments_reject_unregistered_paths_and_unbounded_controls() {
    for bad in [
        "--input-lock=../lock.json",
        "--native-assets=/tmp/assets.json",
        "--input-lock=inputs/../lock.json",
        "--native-assets=inputs\\assets.json",
        "--process-epoch=0",
        "--process-epoch=18446744073709551616",
        "--startup-ms=0",
        "--startup-ms=3600001",
        "--startup-ms=+1",
        "--root-policy=legacy-v5",
        "--external-r-sources=../external.json",
        "--retirement-sink=/tmp/registration.json",
        "--evidence-root=relative/evidence",
        "--unknown=true",
        "--startup-diagnostics=true",
        "--startup-diagnostics=stdout-v1",
        "--lifecycle-diagnostics=true",
        "--lifecycle-diagnostics=stdout-v1",
    ] {
        let mut args = arguments();
        let key = bad.split_once('=').unwrap().0;
        args.retain(|argument| argument.split_once('=').unwrap().0 != key);
        args.push(bad.into());
        assert!(Arguments::parse(&args).is_err(), "{bad}");
    }
}

#[test]
fn external_source_mapping_is_explicit_closed_and_independent_of_native_assets() {
    let mut args = arguments();
    args.push("--external-r-sources=inputs/external-sources.json".into());
    assert_eq!(
        Arguments::parse(&args)
            .unwrap()
            .external_r_sources
            .as_deref(),
        Some("inputs/external-sources.json")
    );
    args.push("--external-r-sources=inputs/other.json".into());
    assert!(Arguments::parse(&args).is_err());

    let wire = br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"inputs/external","assets":[]}"#;
    let selection = ExternalSelectionWire::parse(wire).unwrap();
    assert_eq!(selection.working_directory_snapshot_path, "inputs/external");
    assert!(selection.assets.is_empty());
    for bad in [
        br#"{"domain":"rz-rovent-native-assets/1","working_directory_snapshot_path":"inputs/external","assets":[]}"#.as_slice(),
        br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"../external","assets":[]}"#.as_slice(),
        br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"inputs/external","assets":[],"external_question_limit":{"kind":"depth","depth":1}}"#.as_slice(),
        br#"{"domain":"rz-rovent-external-r-sources/1","working_directory_snapshot_path":"inputs/external"}"#.as_slice(),
    ] {
        assert!(ExternalSelectionWire::parse(bad).is_err());
    }
}

#[test]
fn retirement_arguments_keep_the_original_registration_and_explicit_absolute_root() {
    let root = std::env::current_dir().unwrap().join("managed-evidence");
    let mut args = arguments();
    args.push("--retirement-sink=inputs/retirement-registration.json".into());
    args.push(format!("--evidence-root={}", root.display()));
    let parsed = Arguments::parse(&args).unwrap();
    assert_eq!(
        parsed.retirement_sink.as_deref(),
        Some("inputs/retirement-registration.json")
    );
    assert_eq!(parsed.evidence_root.as_deref(), Some(root.as_path()));
    for repeated in [
        "--retirement-sink=inputs/other-registration.json".into(),
        format!("--evidence-root={}", root.display()),
    ] {
        let mut repeated_args = args.clone();
        repeated_args.push(repeated);
        assert!(Arguments::parse(&repeated_args).is_err());
    }
    assert!(managed_path(root.join("..").join("elsewhere").to_str().unwrap()).is_err());
}

#[test]
fn native_asset_wire_cannot_admit_a_legacy_or_incomplete_selection() {
    assert!(SelectionWire::parse(br#"{"domain":"rz-pals-startup/5"}"#).is_err());
    assert!(
        SelectionWire::parse(br#"{"domain":"rz-rovent-native-assets/1","graphs":[]}"#).is_err()
    );
    assert!(SelectionWire::parse(br#"{} {}"#).is_err());
}
