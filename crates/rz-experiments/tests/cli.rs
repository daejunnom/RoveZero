use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let parent = std::env::temp_dir().join("rovezero-e01-cli-tests");
        fs::create_dir_all(&parent).expect("create temporary test root");
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        let path = parent.join(format!("{}-{timestamp}-{sequence}", std::process::id()));
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
    let mut command = Command::new(env!("CARGO_BIN_EXE_rz-experiments"));
    command.args(args);
    run_command(command)
}

fn run_command(mut command: Command) -> Output {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run CLI");
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

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/baselines/fixtures/e01-input.json")
}

fn assert_failure(output: &Output) -> String {
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty(), "failures must not print success");
    let diagnostic = String::from_utf8(output.stderr.clone()).expect("UTF-8 diagnostic");
    assert!(!diagnostic.trim().is_empty());
    diagnostic
}

#[test]
fn incomplete_or_unknown_commands_fail() {
    for args in [
        vec![],
        vec!["validate"],
        vec!["lock", "input.json"],
        vec!["verify"],
        vec!["run", "input.json"],
        vec!["validate", "input.json", "extra"],
    ] {
        assert_failure(&run(args));
    }
}

#[test]
fn malformed_json_does_not_create_a_lock() {
    let directory = TestDirectory::new();
    let input = directory.file("bad.json", b"{broken");
    let output = directory.0.join("locked.json");
    assert_failure(&run([OsStr::new("validate"), input.as_os_str()]));
    assert_failure(&run([
        OsStr::new("lock"),
        input.as_os_str(),
        output.as_os_str(),
    ]));
    assert!(!output.exists());
    assert_failure(&run([OsStr::new("verify"), input.as_os_str()]));
}

#[test]
fn json_input_is_bounded_and_strict_utf8() {
    let directory = TestDirectory::new();
    let oversized = directory.file("large.json", vec![b' '; 4 * 1024 * 1024 + 1]);
    let invalid_utf8 = directory.file("non-utf8.json", [b'{', 0xff, b'}']);
    for command in ["validate", "verify"] {
        let error = assert_failure(&run([OsStr::new(command), oversized.as_os_str()]));
        assert!(error.contains("4 MiB"));
        let error = assert_failure(&run([OsStr::new(command), invalid_utf8.as_os_str()]));
        assert!(error.contains("UTF-8"));
    }
    let output = directory.0.join("locked.json");
    let error = assert_failure(&run([
        OsStr::new("lock"),
        oversized.as_os_str(),
        output.as_os_str(),
    ]));
    assert!(error.contains("4 MiB"));
    assert!(!output.exists());
}

#[test]
fn existing_output_is_preserved() {
    let directory = TestDirectory::new();
    let output = directory.file("locked.json", b"existing evidence\n");
    let error = assert_failure(&run([
        OsStr::new("lock"),
        fixture().as_os_str(),
        output.as_os_str(),
    ]));
    assert!(error.contains("cannot create output"));
    assert_eq!(
        fs::read(&output).expect("read existing output"),
        b"existing evidence\n"
    );
}

#[test]
fn input_lock_and_integrity_verification_retain_readiness_false() {
    let directory = TestDirectory::new();
    let locked = directory.0.join("locked.json");
    let input = fixture();
    let output = run([OsStr::new("validate"), input.as_os_str()]);
    assert!(output.status.success(), "{:?}", output);
    assert!(String::from_utf8_lossy(&output.stdout).contains("execution_ready=false"));
    let output = run([OsStr::new("lock"), input.as_os_str(), locked.as_os_str()]);
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(output.stdout, b"input locked; execution_ready=false\n");
    let output = run([OsStr::new("verify"), locked.as_os_str()]);
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(
        output.stdout,
        b"locked input integrity verified; execution_ready=false\n"
    );
    let artifacts = input.parent().expect("fixture directory");
    let output = run([
        OsStr::new("verify"),
        locked.as_os_str(),
        OsStr::new("--artifact-root"),
        artifacts.as_os_str(),
        OsStr::new("--max-artifact-bytes"),
        OsStr::new("1048576"),
    ]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        output.stdout,
        b"locked input integrity and artifacts verified; execution_ready=false\n"
    );
}

#[test]
fn artifact_verification_requires_an_explicit_positive_budget() {
    for options in [
        vec!["--artifact-root", "artifacts"],
        vec!["--max-artifact-bytes", "100"],
        vec!["--artifact-root", "artifacts", "--max-artifact-bytes", "0"],
        vec!["--artifact-root", "artifacts", "--max-artifact-bytes", "-1"],
        vec![
            "--artifact-root",
            "artifacts",
            "--max-artifact-bytes",
            "18446744073709551616",
        ],
        vec!["--artifact-root", "artifacts", "--artifact-root", "other"],
        vec!["--max-artifact-bytes"],
    ] {
        let mut args = vec!["verify", "input-is-not-read.json"];
        args.extend(options);
        let error = assert_failure(&run(args));
        assert!(!error.contains("cannot open input"));
    }
}

#[test]
fn requested_artifact_verification_rejects_missing_files() {
    let directory = TestDirectory::new();
    let locked = directory.0.join("locked.json");
    let artifacts = directory.0.join("artifacts");
    fs::create_dir(&artifacts).expect("create empty artifact root");
    let output = run([
        OsStr::new("lock"),
        fixture().as_os_str(),
        locked.as_os_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    // Supplying both options must actually inspect files, rather than only verify
    // the envelope as the default verify command does.
    let error = assert_failure(&run([
        OsStr::new("verify"),
        locked.as_os_str(),
        OsStr::new("--max-artifact-bytes"),
        OsStr::new("1048576"),
        OsStr::new("--artifact-root"),
        artifacts.as_os_str(),
    ]));
    assert!(error.contains("artifact verification"));
}

#[cfg(unix)]
#[test]
fn fifo_input_is_rejected_without_waiting_for_a_writer() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let directory = TestDirectory::new();
    let fifo = directory.0.join("input.fifo");
    let path = CString::new(fifo.as_os_str().as_bytes()).expect("FIFO path without NUL");
    // The C string remains live throughout this call; mkfifo does not retain it.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    for command in ["validate", "verify"] {
        let error = assert_failure(&run([OsStr::new(command), fifo.as_os_str()]));
        assert!(error.contains("regular file"));
    }
    let output = directory.0.join("locked.json");
    assert_failure(&run([
        OsStr::new("lock"),
        fifo.as_os_str(),
        output.as_os_str(),
    ]));
    assert!(!output.exists());
}

#[cfg(unix)]
#[test]
fn partial_write_failure_preserves_output_and_other_files() {
    use std::os::unix::process::CommandExt;

    let directory = TestDirectory::new();
    let output = directory.0.join("locked.json");
    let sibling = directory.file("preserved.txt", b"existing evidence\n");
    let mut command = Command::new(env!("CARGO_BIN_EXE_rz-experiments"));
    command.args([
        OsStr::new("lock"),
        fixture().as_os_str(),
        output.as_os_str(),
    ]);
    // Restrict only this child. Ignoring SIGXFSZ makes write_all return EFBIG so
    // the CLI's write-error handling runs instead of terminating by signal.
    // These calls use initialized values and only async-signal-safe libc APIs.
    unsafe {
        command.pre_exec(|| {
            if libc::signal(libc::SIGXFSZ, libc::SIG_IGN) == libc::SIG_ERR {
                return Err(std::io::Error::last_os_error());
            }
            let limit = libc::rlimit {
                rlim_cur: 1024,
                rlim_max: 1024,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let error = assert_failure(&run_command(command));
    assert!(error.contains("partial output preserved"));
    let partial = fs::read(&output).expect("partial output retained for inspection");
    assert!(!partial.is_empty());
    assert!(partial.len() <= 1024);
    assert_eq!(
        fs::read(sibling).expect("read preserved sibling"),
        b"existing evidence\n"
    );
}
