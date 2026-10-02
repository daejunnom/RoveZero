use rz_experiments::{LockedManifest, MAX_MANIFEST_BYTES, RunManifest};
use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const MAX_JSON_BYTES: u64 = MAX_MANIFEST_BYTES as u64;
const USAGE: &str = "Usage:\n  rz-experiments validate MANIFEST\n  rz-experiments lock MANIFEST OUTPUT\n  rz-experiments verify LOCKED [--artifact-root ROOT --max-artifact-bytes N]\n\nOnly input validation and locking are implemented. All successful results retain execution_ready=false.";

enum Command {
    Help,
    Validate(PathBuf),
    Lock {
        input: PathBuf,
        output: PathBuf,
    },
    Verify {
        input: PathBuf,
        artifact_check: Option<(PathBuf, u64)>,
    },
}

fn main() -> ExitCode {
    match parse_command(env::args_os().skip(1).collect()).and_then(execute) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rz-experiments: {error}");
            ExitCode::from(2)
        }
    }
}

fn parse_command(args: Vec<OsString>) -> Result<Command, String> {
    let command = args.first().and_then(|arg| arg.to_str());
    match command {
        Some("--help" | "-h") if args.len() == 1 => Ok(Command::Help),
        Some("validate") if args.len() == 2 => Ok(Command::Validate(PathBuf::from(&args[1]))),
        Some("lock") if args.len() == 3 => Ok(Command::Lock {
            input: PathBuf::from(&args[1]),
            output: PathBuf::from(&args[2]),
        }),
        Some("verify") if args.len() >= 2 => parse_verify(&args),
        _ => Err(format!("invalid or missing command arguments\n{USAGE}")),
    }
}

fn parse_verify(args: &[OsString]) -> Result<Command, String> {
    let mut root = None;
    let mut budget = None;
    let mut index = 2;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| "verify option requires a value".to_string())?;
        match args[index].to_str() {
            Some("--artifact-root") if root.is_none() => root = Some(PathBuf::from(value)),
            Some("--max-artifact-bytes") if budget.is_none() => {
                let parsed = value
                    .to_str()
                    .and_then(|text| text.parse::<u64>().ok())
                    .filter(|number| *number > 0)
                    .ok_or_else(|| "--max-artifact-bytes must be a positive u64".to_string())?;
                budget = Some(parsed);
            }
            _ => return Err("unknown or duplicate verify option".to_string()),
        }
        index += 2;
    }
    let artifact_check = match (root, budget) {
        (None, None) => None,
        (Some(root), Some(budget)) => Some((root, budget)),
        _ => {
            return Err(
                "--artifact-root and --max-artifact-bytes must be supplied together".to_string(),
            );
        }
    };
    Ok(Command::Verify {
        input: PathBuf::from(&args[1]),
        artifact_check,
    })
}

fn execute(command: Command) -> Result<String, String> {
    match command {
        Command::Help => Ok(USAGE.to_string()),
        Command::Validate(input) => {
            RunManifest::from_json(&read_json(&input)?)
                .and_then(|manifest| manifest.validate())
                .map_err(|error| error.to_string())?;
            Ok("manifest valid; execution_ready=false".to_string())
        }
        Command::Lock { input, output } => {
            let locked = RunManifest::from_json(&read_json(&input)?)
                .and_then(|manifest| manifest.lock())
                .map_err(|error| error.to_string())?;
            let json = locked.to_json().map_err(|error| error.to_string())?;
            if json.len() as u64 > MAX_JSON_BYTES {
                return Err("locked JSON exceeds the 4 MiB input limit".to_string());
            }
            write_new_file(&output, json.as_bytes())?;
            Ok("input locked; execution_ready=false".to_string())
        }
        Command::Verify {
            input,
            artifact_check,
        } => {
            let locked = LockedManifest::from_json(&read_json(&input)?)
                .map_err(|error| error.to_string())?;
            match artifact_check {
                Some((root, budget)) => {
                    locked
                        .verify_artifacts(&root, budget)
                        .map_err(|error| error.to_string())?;
                    Ok(
                        "locked input integrity and artifacts verified; execution_ready=false"
                            .to_string(),
                    )
                }
                None => Ok("locked input integrity verified; execution_ready=false".to_string()),
            }
        }
    }
}

fn read_json(path: &Path) -> Result<String, String> {
    // Reject final symlinks and ordinary FIFO/device paths before opening.
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("cannot inspect input: {error}"))?;
    if !metadata.is_file() {
        return Err("input must be a regular file".to_string());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A replaced path must not block on a FIFO or follow a final symlink.
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("cannot open input: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot inspect input: {error}"))?;
    if !metadata.is_file() {
        return Err("input must be a regular file".to_string());
    }
    if metadata.len() > MAX_JSON_BYTES {
        return Err("JSON input exceeds 4 MiB".to_string());
    }
    // Read one sentinel byte so a growing file cannot bypass the size limit.
    let mut bytes = Vec::new();
    file.take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read input: {error}"))?;
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err("JSON input exceeds 4 MiB".to_string());
    }
    String::from_utf8(bytes).map_err(|_| "JSON input is not valid UTF-8".to_string())
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    // Validation and serialization finish before creating any output. create_new
    // rejects existing files and symlinks without a separate existence check.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create output (existing files are preserved): {error}"))?;
    let result = file.write_all(bytes).and_then(|()| file.sync_all());
    if let Err(error) = result {
        // A pathname can be replaced between checking its identity and unlinking.
        // Keep partial evidence rather than risking deletion of another file.
        return Err(format!(
            "cannot write output; partial output preserved for inspection: {error}"
        ));
    }
    Ok(())
}
