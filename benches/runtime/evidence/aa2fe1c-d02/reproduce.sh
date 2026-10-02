#!/usr/bin/env bash
# Re-run existing checks/fixtures at the frozen source; leave all outputs in place.
set -euo pipefail
: "${RZ_D_REPO:?Set the absolute RoveZero checkout path}"
: "${RZ_D_REPLAY_ROOT:?Set an absolute output directory outside the checkout}"
source_sha=aa2fe1c3ac38274912017412e080a9bbd14c9485
cargo_bin=${RZ_D_CARGO:-cargo}
case "$RZ_D_REPO:$RZ_D_REPLAY_ROOT" in /*:/*) ;; *) echo 'Use absolute paths' >&2; exit 2 ;; esac
repo_path=$(cd "$RZ_D_REPO" && pwd -P)
mkdir -p "$RZ_D_REPLAY_ROOT"
output_path=$(cd "$RZ_D_REPLAY_ROOT" && pwd -P)
case "$output_path/" in "$repo_path/"*) echo 'Output must be outside the checkout' >&2; exit 2 ;; esac
git -C "$repo_path" worktree add --detach "$output_path/source" "$source_sha"
cd "$output_path/source"
mkdir -p "$output_path/logs" "$output_path/reports" "$output_path/build"
export CARGO_TARGET_DIR="$output_path/build" CARGO_TERM_COLOR=never
printf 'id\tstarted_utc\tfinished_utc\texit_code\n' > "$output_path/receipts.tsv"
failed=0
run_check() {
    local id=$1 start finish code
    shift
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%q ' "$@" > "$output_path/logs/$id.command.txt"
    printf '\n' >> "$output_path/logs/$id.command.txt"
    if "$@" > "$output_path/logs/$id.log" 2>&1; then code=0; else code=$?; failed=1; fi
    finish=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\n' "$id" "$start" "$finish" "$code" >> "$output_path/receipts.tsv"
}
run_check runtime-tests "$cargo_bin" +1.96.0 test --manifest-path crates/rz-runtime/Cargo.toml --locked --offline --all-targets --all-features
run_check telemetry-tests "$cargo_bin" +1.96.0 test --manifest-path crates/rz-telemetry/Cargo.toml --locked --offline --all-targets
run_check runtime-fmt "$cargo_bin" +1.96.0 fmt --manifest-path crates/rz-runtime/Cargo.toml -- --check
run_check telemetry-fmt "$cargo_bin" +1.96.0 fmt --manifest-path crates/rz-telemetry/Cargo.toml -- --check
run_check runtime-clippy "$cargo_bin" +1.96.0 clippy --manifest-path crates/rz-runtime/Cargo.toml --locked --offline --all-targets --all-features -- -D warnings
run_check telemetry-clippy "$cargo_bin" +1.96.0 clippy --manifest-path crates/rz-telemetry/Cargo.toml --locked --offline --all-targets -- -D warnings
run_check runtime-default-check "$cargo_bin" +1.96.0 check --manifest-path crates/rz-runtime/Cargo.toml --locked --offline --all-targets --no-default-features
run_check runtime-msrv-check "$cargo_bin" +1.85.0 check --manifest-path crates/rz-runtime/Cargo.toml --locked --offline --all-targets --all-features

# These are new runs of the frozen source, not byte-identical historical outputs.
run_trace() {
    local name=$1 profile=$2 scenario=$3 requests=$4 capacity=$5 start finish code
    local -a release_flag=()
    if [[ $profile == release ]]; then release_flag=(--release); fi
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if "$cargo_bin" +1.96.0 run --manifest-path crates/rz-runtime/Cargo.toml --locked --offline "${release_flag[@]}" --example cpu_trace -- --scenario "$scenario" --requests "$requests" --sample-capacity "$capacity" > "$output_path/reports/$name.tsv" 2> "$output_path/logs/$name.log"; then code=0; else code=$?; failed=1; fi
    finish=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\n' "$name" "$start" "$finish" "$code" >> "$output_path/receipts.tsv"
}
run_trace all-release release all 32 512
run_trace all-dev dev all 32 512
run_trace one-dev dev all 1 512
run_trace zero-dev dev all 16 0
run_trace loss-dev dev cold 9 1
run_trace long-loss-release release long 2048 2
run_trace long-zero-release release long 2048 0
printf 'Outputs preserved in %s; source %s\n' "$output_path" "$source_sha"
exit "$failed"
