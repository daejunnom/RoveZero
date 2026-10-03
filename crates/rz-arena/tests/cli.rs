use rz_experiments::RunManifest;
use serde_json::{Value, json};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
const PLAN_BOUNDS: [&str; 4] = ["--max-pairs", "16", "--max-plan-bytes", "1048576"];
const LEDGER_BOUNDS: [&str; 8] = [
    "--max-pairs",
    "16",
    "--max-plan-bytes",
    "1048576",
    "--max-events",
    "16",
    "--max-ledger-bytes",
    "65536",
];

#[cfg(feature = "native-cuda")]
#[test]
fn cuda_cli_requires_exact_paths_before_assets_and_exposes_integration_only_help() {
    for (args, expected) in [
        (
            vec!["native-cuda-lock"],
            "native-cuda-lock requires exactly INPUT OUTPUT",
        ),
        (
            vec!["native-cuda-lock", "missing", "out", "extra"],
            "native-cuda-lock requires exactly INPUT OUTPUT",
        ),
        (
            vec!["native-cuda-pair", "missing", "source", "output"],
            "native-cuda-pair requires exactly LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME",
        ),
    ] {
        assert!(assert_failure(&run(args)).contains(expected));
    }
    let help = run(["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();
    assert!(text.contains("native-cuda-lock INPUT OUTPUT"));
    assert!(text.contains("native-cuda-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME"));
    assert!(text.contains("strength_eligible=false"));
}

#[cfg(not(feature = "native-cuda"))]
#[test]
fn cuda_cli_is_not_enabled_by_default() {
    assert!(
        assert_failure(&run(["native-cuda-lock", "missing", "out"]))
            .contains("invalid or missing command")
    );
    let help = run(["--help"]);
    assert!(help.status.success());
    assert!(
        !String::from_utf8(help.stdout)
            .unwrap()
            .contains("native-cuda-lock")
    );
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let root = std::env::temp_dir().join("rovezero-e02-cli-tests");
        fs::create_dir_all(&root).expect("create temporary test root");
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        let path = root.join(format!("{}-{timestamp}-{sequence}", std::process::id()));
        fs::create_dir(&path).expect("create temporary test directory");
        Self(path)
    }

    fn file(&self, name: &str, contents: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, contents).expect("write test input");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut child = Command::new(env!("CARGO_BIN_EXE_rz-arena"))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run arena CLI");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().expect("inspect CLI status").is_some() {
            return child.wait_with_output().expect("collect CLI output");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap timed out CLI");
            panic!("CLI exceeded five-second test deadline: {output:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn with_bounds(command: &str, paths: &[&Path], ledger: bool) -> Output {
    let mut args: Vec<OsString> = vec![command.into()];
    args.extend(paths.iter().map(|path| path.as_os_str().to_owned()));
    args.extend(
        if ledger {
            LEDGER_BOUNDS.as_slice()
        } else {
            PLAN_BOUNDS.as_slice()
        }
        .iter()
        .map(OsString::from),
    );
    run(args)
}

fn assert_failure(output: &Output) -> String {
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty(), "failure must not print success");
    let error = String::from_utf8(output.stderr.clone()).expect("UTF-8 diagnostic");
    assert!(!error.trim().is_empty());
    error
}

fn assert_structural_success(output: &Output) {
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    let message = String::from_utf8_lossy(&output.stdout);
    assert!(message.contains("execution_ready=false"));
    assert!(message.contains("validation_scope=structural_only"));
}

struct PlanFixture {
    locked: PathBuf,
    plan: PathBuf,
    json: Value,
}

fn make_plan(directory: &TestDirectory, suffix: &str) -> PlanFixture {
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/baselines/fixtures/e01-input.json");
    let input = fs::read_to_string(input).expect("read shared synthetic E01 fixture");
    let mut manifest = RunManifest::from_json(&input).expect("parse fixture");
    manifest.run_id = format!("cli-fixture-{suffix}");
    let locked = directory.file(
        &format!("{suffix}-locked.json"),
        manifest
            .lock()
            .expect("lock synthetic fixture")
            .to_json()
            .expect("serialize input lock"),
    );
    let plan = directory.0.join(format!("{suffix}-plan.json"));
    assert_structural_success(&with_bounds("plan", &[&locked, &plan], false));
    let json =
        serde_json::from_slice(&fs::read(&plan).expect("read plan")).expect("parse plan JSON");
    PlanFixture { locked, plan, json }
}

fn init_ledger(directory: &TestDirectory, plan: &Path) -> PathBuf {
    let ledger = directory.0.join("initial-ledger.jsonl");
    assert_structural_success(&with_bounds("ledger-init", &[plan, &ledger], true));
    ledger
}

fn append(
    directory: &TestDirectory,
    plan: &Path,
    ledger: &Path,
    event: Value,
    stage: usize,
) -> PathBuf {
    let event_path = directory.file(
        &format!("event-{stage}.json"),
        serde_json::to_vec(&event).expect("serialize synthetic event"),
    );
    let next = directory.0.join(format!("ledger-{stage}.jsonl"));
    assert_structural_success(&with_bounds(
        "ledger-append",
        &[plan, ledger, &event_path, &next],
        true,
    ));
    next
}

fn audit(plan: &Path, ledger: &Path) -> Value {
    let output = with_bounds("audit", &[plan, ledger], true);
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty());
    let value: Value = serde_json::from_slice(&output.stdout).expect("parse audit summary");
    assert_eq!(value["execution_ready"], false);
    assert_eq!(value["validation_scope"], "structural_only");
    assert_eq!(value["tip_sha256"].as_str().expect("ledger tip").len(), 64);
    value
}

fn audit_with_tip(plan: &Path, ledger: &Path, tip: &str) -> Output {
    let mut args = vec![
        OsString::from("audit"),
        plan.as_os_str().to_owned(),
        ledger.as_os_str().to_owned(),
    ];
    args.extend(LEDGER_BOUNDS.iter().map(OsString::from));
    args.extend([OsString::from("--expected-tip"), OsString::from(tip)]);
    run(args)
}

fn evidence() -> Value {
    json!({
        "path": "e01-identity.txt",
        "sha256": "72e6620415dcbbba3ed738dc398848595a64f0e0dfd75c286cd1da51e53ae877",
        "bytes": 63,
        "source": "https://github.com/daejunnom/RoveZero",
        "license": "MIT - synthetic test fixture",
    })
}

#[test]
fn synthetic_complete_pair_can_be_planned_recorded_and_audited() {
    let directory = TestDirectory::new();
    let fixture = make_plan(&directory, "complete");
    let pair = &fixture.json["pairs"][0];
    let mut ledger = init_ledger(&directory, &fixture.plan);
    assert_eq!(
        audit(&fixture.plan, &ledger)["summary"]["completed_pairs"],
        0
    );
    ledger = append(
        &directory,
        &fixture.plan,
        &ledger,
        json!({"event":"pair_started", "pair_id":pair["id"], "attempt":1,
            "process_run_id":"cli-synthetic-process-1"}),
        1,
    );
    for position in 0..2 {
        let index = pair["execution_order"][position]
            .as_u64()
            .expect("game order index") as usize;
        ledger = append(
            &directory,
            &fixture.plan,
            &ledger,
            json!({"event":"game_recorded", "pair_id":pair["id"], "attempt":1,
                "game_id":pair["games"][index]["id"],
                "outcome":{"kind":"rules_terminal", "result":"white_win",
                    "reason":"checkmate", "evidence":evidence()}}),
            position + 2,
        );
    }
    ledger = append(
        &directory,
        &fixture.plan,
        &ledger,
        json!({"event":"pair_closed", "pair_id":pair["id"], "attempt":1}),
        4,
    );
    let value = audit(&fixture.plan, &ledger);
    assert_eq!(value["summary"]["completed_pairs"], 1);
    assert_eq!(value["summary"]["wins"], 1);
    assert_eq!(value["summary"]["losses"], 1);
    assert_eq!(value["summary"]["draws"], 0);
    assert_eq!(value["summary"]["n"], json!([0, 0, 1, 0, 0]));
}

#[test]
fn incomplete_pair_does_not_receive_a_draw_score() {
    let directory = TestDirectory::new();
    let fixture = make_plan(&directory, "incomplete");
    let pair = &fixture.json["pairs"][0];
    let mut ledger = init_ledger(&directory, &fixture.plan);
    ledger = append(
        &directory,
        &fixture.plan,
        &ledger,
        json!({"event":"pair_started", "pair_id":pair["id"], "attempt":1,
            "process_run_id":"cli-synthetic-incomplete-process"}),
        1,
    );
    assert_eq!(audit(&fixture.plan, &ledger)["summary"]["pending_pairs"], 1);
    for position in 0..2 {
        let index = pair["execution_order"][position]
            .as_u64()
            .expect("game order index") as usize;
        ledger = append(
            &directory,
            &fixture.plan,
            &ledger,
            json!({"event":"game_recorded", "pair_id":pair["id"], "attempt":1,
                "game_id":pair["games"][index]["id"],
                "outcome":{"kind":"incomplete", "reason":"synthetic-budget-end",
                    "evidence":evidence()}}),
            position + 2,
        );
    }
    ledger = append(
        &directory,
        &fixture.plan,
        &ledger,
        json!({"event":"pair_closed", "pair_id":pair["id"], "attempt":1}),
        4,
    );
    let value = audit(&fixture.plan, &ledger);
    assert_eq!(value["summary"]["completed_pairs"], 0);
    assert_eq!(value["summary"]["excluded_pairs"], 1);
    assert_eq!(value["summary"]["draws"], 0);
    assert_eq!(value["summary"]["n"], json!([0, 0, 0, 0, 0]));
}

#[test]
fn missing_unknown_and_invalid_bounds_fail_before_reading_files() {
    for args in [
        vec![],
        vec!["run", "input"],
        vec!["plan", "input", "output"],
        vec!["ledger-init", "plan", "output", "--max-pairs", "1"],
        vec!["plan", "input", "output", "--root", "artifacts"],
        vec![
            "plan",
            "input",
            "output",
            "--max-pairs",
            "0",
            "--max-plan-bytes",
            "10",
        ],
        vec![
            "plan",
            "input",
            "output",
            "--max-pairs",
            "1",
            "--max-plan-bytes",
            "4194305",
        ],
        vec![
            "plan",
            "input",
            "output",
            "--max-pairs",
            "1",
            "--max-pairs",
            "2",
        ],
        vec![
            "plan",
            "input",
            "output",
            "--max-pairs",
            "18446744073709551616",
        ],
        vec![
            "plan",
            "input",
            "output",
            "--max-pairs",
            "1",
            "--max-events",
            "1",
        ],
    ] {
        let error = assert_failure(&run(args));
        assert!(!error.contains("cannot inspect input"));
    }
}

#[test]
fn ledger_bound_cannot_exceed_manifest_output_budget() {
    let directory = TestDirectory::new();
    let fixture = make_plan(&directory, "output-budget");
    let output = directory.0.join("new-ledger.jsonl");
    let mut args = vec![
        OsString::from("ledger-init"),
        fixture.plan.as_os_str().to_owned(),
        output.as_os_str().to_owned(),
    ];
    args.extend(LEDGER_BOUNDS.iter().map(OsString::from));
    *args.last_mut().expect("ledger byte bound") = "1048577".into();
    let error = assert_failure(&run(args));
    assert!(error.contains("manifest output budget"));
    assert!(!output.exists());
}

#[test]
fn existing_outputs_are_preserved() {
    let directory = TestDirectory::new();
    let fixture = make_plan(&directory, "preservation");
    let original = fs::read(&fixture.plan).expect("read original plan");
    let error = assert_failure(&with_bounds(
        "plan",
        &[&fixture.locked, &fixture.plan],
        false,
    ));
    assert!(error.contains("cannot create output"));
    assert_eq!(
        fs::read(&fixture.plan).expect("read preserved plan"),
        original
    );
    let output = directory.file("existing-ledger.jsonl", b"existing evidence\n");
    assert_failure(&with_bounds("ledger-init", &[&fixture.plan, &output], true));
    assert_eq!(
        fs::read(output).expect("read existing ledger"),
        b"existing evidence\n"
    );
}

#[test]
fn wrong_plan_and_partial_ledger_are_rejected() {
    let directory = TestDirectory::new();
    let original = make_plan(&directory, "original");
    let replacement = make_plan(&directory, "replacement");
    let ledger = init_ledger(&directory, &original.plan);
    assert_failure(&with_bounds("audit", &[&replacement.plan, &ledger], true));
    let mut contents = fs::read(&ledger).expect("read ledger");
    contents.extend_from_slice(b"{\"event\":");
    let partial = directory.file("partial-ledger.jsonl", contents);
    assert_failure(&with_bounds("audit", &[&original.plan, &partial], true));
}

#[test]
fn audit_checkpoint_detects_wrong_tip_and_complete_record_suffix_removal() {
    let directory = TestDirectory::new();
    let fixture = make_plan(&directory, "checkpoint");
    let initial = init_ledger(&directory, &fixture.plan);
    let ledger = append(
        &directory,
        &fixture.plan,
        &initial,
        json!({"event":"pair_started", "pair_id":fixture.json["pairs"][0]["id"],
            "attempt":1, "process_run_id":"cli-checkpoint-process"}),
        1,
    );
    let original = audit(&fixture.plan, &ledger);
    let tip = original["tip_sha256"]
        .as_str()
        .expect("original ledger tip");
    let output = audit_with_tip(&fixture.plan, &ledger, tip);
    assert!(output.status.success(), "{output:?}");
    let output: Value = serde_json::from_slice(&output.stdout).expect("checkpoint audit JSON");
    assert_eq!(output["tip_sha256"], tip);
    assert_eq!(output["execution_ready"], false);
    assert_eq!(output["validation_scope"], "structural_only");
    let mut wrong_tip = tip.to_string();
    wrong_tip.replace_range(0..1, if tip.starts_with('0') { "1" } else { "0" });
    assert_failure(&audit_with_tip(&fixture.plan, &ledger, &wrong_tip));

    let bytes = fs::read(&ledger).expect("read complete JSONL records");
    assert_eq!(bytes.last(), Some(&b'\n'));
    let boundary = bytes[..bytes.len() - 1]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .expect("header newline before final event");
    let shortened = directory.file("shortened-ledger.jsonl", &bytes[..=boundary]);
    // A hash chain alone cannot distinguish an earlier complete prefix from
    // deletion. A separately retained tip checkpoint makes this change visible.
    let prefix = audit(&fixture.plan, &shortened);
    assert_ne!(prefix["tip_sha256"], tip);
    assert_eq!(prefix["summary"]["completed_pairs"], 0);
    assert_failure(&audit_with_tip(&fixture.plan, &shortened, tip));
}

#[test]
fn expected_tip_is_audit_only_and_requires_lowercase_sha256() {
    let valid = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    for args in [
        vec!["plan", "input", "output", "--expected-tip", valid],
        vec!["ledger-init", "plan", "output", "--expected-tip", valid],
        vec![
            "ledger-append",
            "plan",
            "ledger",
            "event",
            "output",
            "--expected-tip",
            valid,
        ],
        vec!["audit", "plan", "ledger", "--expected-tip", "abc"],
        vec![
            "audit",
            "plan",
            "ledger",
            "--expected-tip",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ],
        vec![
            "audit",
            "plan",
            "ledger",
            "--expected-tip",
            "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
        ],
        vec![
            "audit",
            "plan",
            "ledger",
            "--expected-tip",
            valid,
            "--expected-tip",
            valid,
        ],
    ] {
        let error = assert_failure(&run(args));
        assert!(!error.contains("cannot inspect input"));
    }
}

#[test]
fn malformed_event_does_not_publish_a_new_ledger() {
    let directory = TestDirectory::new();
    let fixture = make_plan(&directory, "bad-event");
    let ledger = init_ledger(&directory, &fixture.plan);
    let original = fs::read(&ledger).expect("read initial ledger");
    let event = directory.file(
        "bad-event.json",
        b"{\"event\":\"pair_started\",\"unexpected\":true}",
    );
    let output = directory.0.join("new-ledger.jsonl");
    assert_failure(&with_bounds(
        "ledger-append",
        &[&fixture.plan, &ledger, &event, &output],
        true,
    ));
    assert!(!output.exists());
    assert_eq!(
        fs::read(ledger).expect("read preserved initial ledger"),
        original
    );
}

#[test]
fn plan_input_is_bounded_and_strict_utf8() {
    let directory = TestDirectory::new();
    let too_large = directory.file("large.json", vec![b' '; 4 * 1024 * 1024 + 1]);
    let not_utf8 = directory.file("not-utf8.json", [b'{', 0xff, b'}']);
    let output = directory.0.join("plan.json");
    let error = assert_failure(&with_bounds("plan", &[&too_large, &output], false));
    assert!(error.contains("byte bound"));
    let error = assert_failure(&with_bounds("plan", &[&not_utf8, &output], false));
    assert!(error.contains("UTF-8"));
    assert!(!output.exists());
}
