# 원본 명령과 보존 이후 재현 절차

실제 당시 명령·exit·elapsed는 각 checkpoint의 `reproduction-receipt.json`과 `checks/*.command.json`이다. 실제 raw output은 `checks/*.stdout/.stderr` 및 `attempt-*/stdout.log/stderr.log`이며 redaction은 Inventory를 따른다. 이 문서의 build/run 준비 명령은 이후 작성했고 이번 보존 작업에서 실행하지 않았다.

## source와 준비

checkpoint 1은 `fcbe8b25d53ee96f48ca8dafd227e7c46f8d90ab`, checkpoint 3은 `fdd357f08fdca16666993ebdb8afc3c72c718821`, checkpoint 4/5는 `9e49b6679aeab4e96be0f6b90e51e26a0c7a9ead`인 별도 checkout에서 재현한다. `${REPO_ROOT}`를 checkout, `${CARGO}`를 Cargo 경로, `${ARTIFACT_ROOT}`를 저장소 외부 출력 root로 지정한다.

checkpoint 3/4의 원본 build argv와 stdout/stderr는 `checks/build*.command.json/.stdout/.stderr`에 있다. 다른 source에서 이를 실행한 결과를 과거 source의 결과로 표시하지 않는다. private Cargo.lock이 원본 checkpoint에 없다면 missing item이며 과거 E02 lock을 native build lock으로 전용하지 않는다.

```bash
CARGO_TARGET_DIR="${ARTIFACT_ROOT}/build-reproduction" "${CARGO}" build --manifest-path "${REPO_ROOT}/crates/rz-experiments/Cargo.toml" --locked
CARGO_TARGET_DIR="${ARTIFACT_ROOT}/build-reproduction" "${CARGO}" build --manifest-path "${REPO_ROOT}/crates/rz-arena/Cargo.toml" --bins --locked
```

외부 Fastchess source는 `f618e34540f94f4719ad3817950618dabe441318`이며 build command/flags/compiler는 형제 runner-research evidence를 따른다. 빌드/바이너리는 외부 root에만 둔다. 새 바이너리의 실제 SHA/bytes와 tool/source identity를 case input에 반영하고 input lock·plan을 다시 생성한다. source checkpoint별 `e01-openings.json`은 보존본에서 가져온다. 원본 lock을 새 binary identity로 가장하지 않는다.

## 각 phase 명령

`${RUN_ROOT}`는 새 실행 root, `${EXPERIMENTS}`·`${ARENA}`는 해당 source의 build binary이다. 다음은 원본 형태를 기반으로 작성한 재현 명령이며 실제 original argv는 각 command JSON을 확인한다.

```bash
"${EXPERIMENTS}" lock "${RUN_ROOT}/normal-input.json" "${RUN_ROOT}/normal-input-lock.json"
"${ARENA}" plan "${RUN_ROOT}/normal-input-lock.json" "${RUN_ROOT}/normal-plan.json" --max-pairs 1 --max-plan-bytes 1048576
"${ARENA}" fixture-pair "${RUN_ROOT}/normal-plan.json" "${RUN_ROOT}" attempt-normal --max-pairs 1 --max-plan-bytes 1048576
"${ARENA}" audit "${RUN_ROOT}/normal-plan.json" "${RUN_ROOT}/attempt-normal/ledger.jsonl" --max-pairs 1 --max-plan-bytes 1048576 --max-events 4 --max-ledger-bytes 65536
```

cutoff는 `cutoff-input/plan`·`attempt-cutoff`의 원래 vector를, cancel은 `cancel-input/plan`·`attempt-cancel`의 원래 vector를 사용한다. checkpoint 5는 cancel-only이다. cancel timer·cleanup 상태는 timing/OS에 따라 달라질 수 있으므로 실제 receipt의 `stop`, `exit_signal`, `group_cleanup`을 보존하고 Gone을 강요하지 않는다. Unverified는 성공/실행 준비로 바꾸지 않고 점수에서 제외하며 runtime/GPU drain을 증명하지 않는다. 재현 후 새 raw logs/receipts/ledger를 별도 checkpoint로 저장한다.

## 최종 8개 검사 재현

다음은 source `9e49b6679aeab4e96be0f6b90e51e26a0c7a9ead`의 원본 argv를 바탕으로 보존 이후 작성한 명령이며 이번 보존 단계에서 실행하지 않았다. 별도 source checkout에서 `${CARGO}`를 지정하고 새 raw stdout/stderr를 저장한다. `final-checks/inputs/end/`의 lock은 검사 이후 실제 저장한 입력 후보이며 원래 시작 snapshot이라고 표시하지 않는다.

```bash
"${CARGO}" fmt --manifest-path crates/rz-experiments/Cargo.toml --all -- --check
"${CARGO}" test --manifest-path crates/rz-experiments/Cargo.toml --all-targets --locked
"${CARGO}" clippy --manifest-path crates/rz-experiments/Cargo.toml --all-targets --locked -- -D warnings
"${CARGO}" fmt --manifest-path crates/rz-arena/Cargo.toml --all -- --check
"${CARGO}" test --manifest-path crates/rz-arena/Cargo.toml --all-features --all-targets --locked
"${CARGO}" clippy --manifest-path crates/rz-arena/Cargo.toml --all-features --all-targets --locked -- -D warnings
"${CARGO}" +1.85.0 check --manifest-path crates/rz-experiments/Cargo.toml --all-targets --locked
"${CARGO}" +1.90.0 check --manifest-path crates/rz-arena/Cargo.toml --all-features --all-targets --locked
```

actual 원본 commands/status/timeouts/source/cwd는 `final-checks/checks-result.json`과 각 command JSON에, 원본 stdout/stderr는 같은 stem의 파일에 남아 있다. rustc default/1.85/1.90 version은 after-check supplemental capture이며 원래 검사 시점의 선행 version capture로 가장하지 않는다.
