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
        "--unknown=true",
    ] {
        let mut args = arguments();
        let key = bad.split_once('=').unwrap().0;
        args.retain(|argument| argument.split_once('=').unwrap().0 != key);
        args.push(bad.into());
        assert!(Arguments::parse(&args).is_err(), "{bad}");
    }
}

#[test]
fn native_asset_wire_cannot_admit_a_legacy_or_incomplete_selection() {
    assert!(SelectionWire::parse(br#"{"domain":"rz-pals-startup/5"}"#).is_err());
    assert!(
        SelectionWire::parse(br#"{"domain":"rz-rovent-native-assets/1","graphs":[]}"#).is_err()
    );
    assert!(SelectionWire::parse(br#"{} {}"#).is_err());
}
