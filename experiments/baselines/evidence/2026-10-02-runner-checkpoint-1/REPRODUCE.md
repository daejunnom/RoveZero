# 보존 후 작성한 재현 절차

원본 실제 argv·exit·elapsed는 `run/reproduction-receipt.json` 및 `run/checks/*.command.json`이며 원본 stdout/stderr는 같은 stem의 별도 파일이다. 다음 준비 명령은 실행 당시 raw command transcript가 아닌, 나중에 작성한 재현 절차이다.

1. 별도 RoveZero checkout을 `fcbe8b25d53ee96f48ca8dafd227e7c46f8d90ab`에 맞추고 `${REPO_ROOT}`로 지정한다. `${CARGO}`는 사용할 Cargo 경로, `${ARTIFACT_ROOT}`는 새 저장소 외부 root이다.
2. `"${CARGO}" build --manifest-path "${REPO_ROOT}/crates/rz-experiments/Cargo.toml" --locked` 및 `"${CARGO}" build --manifest-path "${REPO_ROOT}/crates/rz-arena/Cargo.toml" --bins --locked`로 프로그램을 준비한다. 당시 native crate-local Cargo.lock/build raw stdout/stderr는 이 checkpoint에 없으며 이전 E02 lock을 native build의 lock이라고 사용하면 안 된다.
3. 외부 Fastchess `f618e34540f94f4719ad3817950618dabe441318`을 외부 root에서 빌드한다. 원본 tool build 설정은 형제 `2026-10-02-runner-research/reports/fastchess-build-and-smoke.json`에 보존되어 있다. runner와 fixture 바이너리는 repository에 복사하지 않는다.
4. 새 외부 실행 directory에서 `run/e01-openings.json`과 각 case input을 준비한다. 새 바이너리의 실제 SHA/bytes를 해당 input의 runner/engine ArtifactRef에 반영하고, 현재 실제 tool identity와 source revision이 일치하는지 확인한 뒤 lock과 plan을 다시 생성한다. 원본 lock이나 바이너리 identity를 새 build의 것으로 가장하지 않는다.
5. 각 case에 대해 아래 원본 형태의 명령을 실행하고 별도 raw stdout/stderr·receipt·ledger를 저장한다. `${RUN_ROOT}`와 `${EXPERIMENTS}`·`${ARENA}`는 새 외부 출력 directory와 build binary 경로이다.

```bash
"${EXPERIMENTS}" lock "${RUN_ROOT}/normal-input.json" "${RUN_ROOT}/normal-input-lock.json"
"${ARENA}" plan "${RUN_ROOT}/normal-input-lock.json" "${RUN_ROOT}/normal-plan.json" --max-pairs 1 --max-plan-bytes 1048576
"${ARENA}" fixture-pair "${RUN_ROOT}/normal-plan.json" "${RUN_ROOT}" attempt-normal --max-pairs 1 --max-plan-bytes 1048576
"${ARENA}" audit "${RUN_ROOT}/normal-plan.json" "${RUN_ROOT}/attempt-normal/ledger.jsonl" --max-pairs 1 --max-plan-bytes 1048576 --max-events 4 --max-ledger-bytes 65536
```

cutoff·cancel의 원래 exact argv도 각 `run/checks/<case>*.command.json`에 남아 있다. 위 normal file/stem을 각각 바꿔 같은 phase 순서로 실행한다. 당시 cancel은 preflight failure이므로 ProcessReceipt 생성이나 drain 성공을 기대했다는 원본 성공 결과는 없다. 수정된 source나 다른 timing에서 달라진 결과는 새 checkpoint로 기록한다.
