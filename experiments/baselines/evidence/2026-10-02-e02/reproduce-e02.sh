#!/usr/bin/env bash
# Generated after preservation; not an original executed command transcript.
set -eu
: "${REPO_ROOT:?set separate checkout at source e48ff065a431bc6bdd835d75e46e2ed121661f53}"
: "${ARTIFACT_ROOT:?set external output root}"
CARGO="${CARGO:-cargo}"
EVIDENCE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "$ARTIFACT_ROOT"
RUN_ROOT="$(mktemp -d "$ARTIFACT_ROOT/e02-reproduction-XXXXXX")"
export CARGO_TARGET_DIR="$RUN_ROOT/build"
cd "$REPO_ROOT"
for crate in rz-experiments rz-arena; do
  "$CARGO" test --manifest-path "crates/$crate/Cargo.toml" --locked >"$RUN_ROOT/$crate-test.stdout" 2>"$RUN_ROOT/$crate-test.stderr"
  "$CARGO" fmt --manifest-path "crates/$crate/Cargo.toml" -- --check >"$RUN_ROOT/$crate-fmt.stdout" 2>"$RUN_ROOT/$crate-fmt.stderr"
  "$CARGO" clippy --manifest-path "crates/$crate/Cargo.toml" --all-targets --locked -- -D warnings >"$RUN_ROOT/$crate-clippy.stdout" 2>"$RUN_ROOT/$crate-clippy.stderr"
  "$CARGO" build --manifest-path "crates/$crate/Cargo.toml" --locked >"$RUN_ROOT/$crate-build.stdout" 2>"$RUN_ROOT/$crate-build.stderr"
done
EXPERIMENTS="$CARGO_TARGET_DIR/debug/rz-experiments"
ARENA="$CARGO_TARGET_DIR/debug/rz-arena"
SOURCE_RUN="$EVIDENCE_ROOT/runs/e02-synthetic-efu89kvr"
"$EXPERIMENTS" lock "$SOURCE_RUN/synthetic-input.json" "$RUN_ROOT/input-lock.json" >"$RUN_ROOT/lock.stdout" 2>"$RUN_ROOT/lock.stderr"
"$ARENA" plan "$RUN_ROOT/input-lock.json" "$RUN_ROOT/plan.json" --max-pairs 6 --max-plan-bytes 1048576 >"$RUN_ROOT/plan.stdout" 2>"$RUN_ROOT/plan.stderr"
"$ARENA" ledger-init "$RUN_ROOT/plan.json" "$RUN_ROOT/ledger-0.jsonl" --max-pairs 6 --max-plan-bytes 1048576 --max-events 64 --max-ledger-bytes 1048576 >"$RUN_ROOT/init.stdout" 2>"$RUN_ROOT/init.stderr"
for index in $(seq 1 24); do
  previous=$((index-1))
  "$ARENA" ledger-append "$RUN_ROOT/plan.json" "$RUN_ROOT/ledger-$previous.jsonl" "$SOURCE_RUN/event-$index.json" "$RUN_ROOT/ledger-$index.jsonl" --max-pairs 6 --max-plan-bytes 1048576 --max-events 64 --max-ledger-bytes 1048576 >"$RUN_ROOT/append-$index.stdout" 2>"$RUN_ROOT/append-$index.stderr"
done
"$ARENA" audit "$RUN_ROOT/plan.json" "$RUN_ROOT/ledger-24.jsonl" --max-pairs 6 --max-plan-bytes 1048576 --max-events 64 --max-ledger-bytes 1048576 --expected-tip 69e753347b4a84b7dc7a2473ad3f6f298901c873eda7b5f1edea3bc542b1aa35 >"$RUN_ROOT/audit.stdout" 2>"$RUN_ROOT/audit.stderr"
printf '%s\n' "$RUN_ROOT"
