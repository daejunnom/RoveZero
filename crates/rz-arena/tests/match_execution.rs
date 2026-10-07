use rz_arena::audit_ponder_protocol;
fn policies() -> String {
    "RZ_PONDER_POLICY_V1 engine=a enabled=true\nRZ_PONDER_POLICY_V1 engine=b enabled=true\n"
        .repeat(2)
}
#[test]
fn ponder_trace_accounts_for_hits_misses_and_end_of_game_stop() {
    let trace = policies()
        + "RZ_PONDER_V1 engine=a event=start valid=true\nRZ_PONDER_V1 engine=b event=start valid=true\nRZ_PONDER_V1 engine=a event=hit valid=true\nRZ_PONDER_V1 engine=b event=stop valid=true\n";
    let a = audit_ponder_protocol(trace.as_bytes(), ["a", "b"]).unwrap();
    assert_eq!((a.started, a.hits, a.stopped), (2, 1, 1));
}
#[test]
fn pending_duplicate_foreign_and_failed_ponder_results_are_not_accepted() {
    for rows in [
        "RZ_PONDER_V1 engine=a event=start valid=true\n",
        "RZ_PONDER_V1 engine=a event=hit valid=true\n",
        "RZ_PONDER_V1 engine=c event=start valid=true\n",
        "RZ_PONDER_FAILURE_V1 cleanup=false\n",
        "RZ_PONDER_V1 engine=a event=stop valid=false\n",
    ] {
        assert!(audit_ponder_protocol((policies() + rows).as_bytes(), ["a", "b"]).is_err());
    }
}
#[cfg(target_os = "linux")]
#[test]
fn engine_exec_applies_and_reads_back_cpu_affinity_before_literal_exec() {
    use nix::{sched::sched_getaffinity, unistd::Pid};
    use std::process::Command;
    let available = sched_getaffinity(Pid::from_raw(0)).unwrap();
    let cpu = (0..1024)
        .find(|i| available.is_set(*i).unwrap_or(false))
        .unwrap();
    let output=Command::new(env!("CARGO_BIN_EXE_model-pair")).args(["engine-exec",&cpu.to_string(),"-","/usr/bin/python3","-c","import os; print(sorted(os.sched_getaffinity(0))); print(repr(os.environ['CUDA_VISIBLE_DEVICES']))"]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("[{cpu}]\n''\n")
    );
    let invalid = Command::new(env!("CARGO_BIN_EXE_model-pair"))
        .args([
            "engine-exec",
            "4095",
            "-",
            "/usr/bin/python3",
            "-c",
            "print('started')",
        ])
        .output()
        .unwrap();
    assert!(!invalid.status.success() && invalid.stdout.is_empty());
}
