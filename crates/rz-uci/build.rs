//! Public source identity only. Never inspect authentication/configuration files.
use std::{env, path::Path, process::Command};

fn valid_sha(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn git_text(directory: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .current_dir(directory)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_owned())
}
fn watch_git_path(directory: &Path, name: &str) {
    if let Some(text) = git_text(directory, &["rev-parse", "--git-path", name]) {
        if !text.is_empty() && text.len() <= 4096 && !text.chars().any(char::is_control) {
            let path = Path::new(&text);
            let absolute = if path.is_absolute() {
                path.to_owned()
            } else {
                directory.join(path)
            };
            println!("cargo:rerun-if-changed={}", absolute.display());
        }
    }
}
fn main() {
    println!("cargo:rerun-if-env-changed=RZ_PALS_SOURCE_COMMIT");
    let directory = env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest directory");
    let directory = Path::new(&directory);
    let git = || {
        Command::new("git")
            .current_dir(directory)
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|text| text.trim().to_owned())
            .filter(|text| valid_sha(text))
    };
    let source = env::var("RZ_PALS_SOURCE_COMMIT")
        .ok()
        .filter(|text| valid_sha(text))
        .or_else(git);
    // A source-export build may register its public commit explicitly. Missing
    // metadata keeps legacy paths buildable; actual V5 startup refuses it.
    println!(
        "cargo:rustc-env=RZ_PALS_COMPILED_SOURCE_COMMIT={}",
        source.as_deref().unwrap_or("unregistered")
    );
    watch_git_path(directory, "HEAD");
    watch_git_path(directory, "packed-refs");
    if let Some(reference) = git_text(directory, &["symbolic-ref", "--quiet", "HEAD"]) {
        if reference.starts_with("refs/heads/")
            && reference
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != "..")
            && !reference.chars().any(|ch| ch.is_control() || ch == '\\')
        {
            watch_git_path(directory, &reference);
        }
    }
}
