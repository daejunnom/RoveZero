# rz-arena — E02 pair 계획과 attempt 원장

같은 opening 입력에 흑백 엔진 배정만 교환하는 pair를 만들고, 모든 game attempt의
결과·실패를 보존하는 내부 구현이다. E01의 잠긴 입력을 소비하며 새 엔진·Rules·UCI를
구현하지 않는다. 계획·감사 출력은 모두 **`execution_ready=false`**다.

## PR #7과 계약 연결 경계

[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의 확인한 head는
`ae7bf5c20c3acdc12ef7e20aa1a88b5853ee99c8`이다. 총괄의 공통 Rust 계약 revision 0.1이
게시됐지만 draft/open이며 기준 `develop`에는 아직 병합되지 않았다. 반환된 리뷰 댓글,
commit status와 PR-triggered workflow run 목록은 비어 있었다. CI 통과로 기록하지 않는다.

이번 구현의 pair/game/attempt ID와 결과 enum은 E의 영속 기록 형식이다. 공통
RequestId·Move·신경망 WDL·StateIdentity를 복제하지 않는다. opening은 E01의
`OpeningSpec`을 재사용한다. 공통 계약의 `PositionSnapshot<P>`는 A가 복원한 상태를
연결하는 후속 접점이며, 이 코드의 입력 hash를 그 실제 semantic digest로 사용하지 않는다.
공통 오류·generation·runner/PGN/시계 사건을 연결할 때 계약 0.1의 정확한 revision과
실제 producer를 확인한다. 계약 연결 전에도 내부 구현을 진행하라는 사용자 지시를 따른다.

독립 package workspace로 빌드한다. root workspace·lockfile·CI·공통 타입은 I01 소유다.
총괄이 정식 member로 등록할 때 두 E package의 `[workspace]` 경계와 임시 lockfile을
정리하고 같은 통합 SHA로 인수해야 한다. 자체 코드는 [MIT LICENSE](LICENSE)를 따른다.
의존성은 E01 path dependency와 serde/serde_json/sha2, Unix CLI의 libc이며
권리와 버전은 [E01 README](../rz-experiments/README.md)의 목록과 같다.

## 불변 pair 계획

`ArenaPlan::build`, `from_json`, `to_json`을 제공한다. 계획은 잠긴 원본 입력·입력 SHA와
파생 pair·계획 SHA를 묶는다. 다시 읽을 때 원본 잠금을 검증하고 pair를 재생성해 비교한다.
알 수 없는 필드·중복 키·버전·ready=true·파생 값 변경을 거부한다.

현재 지원하는 정책은 `declared-order`, `balanced-alternating`,
`preserve-incomplete-pair-no-score`, `bounded-budget-or-contract-failure`다.
opening 목록을 round-robin으로 사용하고 결과를 보고 재선택하지 않는다. pair의 두 판은
하나의 동일한 `OpeningSpec`을 공유하며 실제 차례·FEN·기물 색·전체 이동열을 바꾸지 않는다.
`game-0`은 첫 엔진 백, `game-1`은 첫 엔진 흑이다. pair/game ID는
`run_id/pair-<0부터 시작한 ordinal>/game-0|1` 형식이다.

실행 순서 parity는 `(pair ordinal % 2) XOR (plan seed % 2)`다. logical resource slot도
pair마다 교차하지만 pair 안에서는 같은 엔진의 slot을 고정한다. 실제 장비 배정·동시
점유·GPU 공정성을 증명하는 값이 아니다. 실제 adapter가 시작 시 다시 확인해야 한다.

`plan_version=1`, `algorithm=rz-e02-paired-v1`이다. canonical digest는 재귀 정렬한
compact UTF-8 JSON의 SHA-256이며 plan 자체 SHA 필드는 제외한다.
opening identity domain은 `rz-e02-opening-input-v1`이다. initial/FEN/전체 moves/history
범위·origin 및 history-fill/repetition 정책을 포함하고 opening ID는 제외한다.
같은 입력에 다른 이름을 붙여도 같은 cluster로 묶는다. 합법 상태 복원 결과를 뜻하지 않는다.

seed domain은 `rz-e02-engine-seed-v1`이며 schema version, plan/input seed,
base engine seed, pair/game ordinal, engine ID를 canonical JSON으로 hash한다.
앞 8바이트를 big-endian u64로 해석한다. 재시도는 같은 계획·seed를 사용한다.
호출자는 `PlanLimits`의 양수 pair/JSON byte 상한을 명시해야 하며 JSON은 최대 4 MiB다.
반복 opening을 Vec에 넣기 전에 누적 직렬화 크기를 검사한다.

## 실패·pair 회계와 JSONL

`Ledger::new`, `append`, `from_jsonl`, `to_jsonl`, `summary`를 제공한다.
사건은 `PairStarted`, `GameRecorded`, `PairClosed`다. 첫 attempt는 1이고 재시도는
순서대로 증가한다. 첫 시작과 재시도는 새 `process_run_id`를 선언하며 원장 안에서
재사용할 수 없다. 이 토큰 자체는 실제 subprocess 시작 영수증이 아니다.

game 기록은 계획 순서로 한 번만 허용한다. 두 판의 기록 없이 pair를 닫을 수 없으며,
완료 pair만 후보 관점 W/D/L과 `n0..n4`에 한 번 포함한다. DD와 WL의 원시 결과는
지우지 않는다. 각 outcome은 안전한 논리 경로·SHA·byte 수·공개 출처·권리가 있는
evidence metadata를 요구한다. 같은 경로의 다른 identity와 고유 evidence byte 상한
초과를 거부한다. 실제 evidence 파일 내용·PGN의 진위는 아직 감사하지 않는다.

- rules terminal: checkmate는 승패, stalemate/dead-position/자동 5회/75수는 draw만 허용.
- engine loss: 불법 수·crash·timeout과 책임 엔진을 기록하고 패배로 계산.
- infrastructure invalid: 양판 attempt를 보존하고 닫힌 pair 전체만 제한 재시도.
- incomplete: 점수를 넣지 않고 제외·pending 범위를 보존. engine loss와 함께 발생한
  인프라 장애로 엔진 패배를 재시도할 수 없다.
- contract invalid 또는 simultaneous failure: 강도 해석을 중단하고 새 pair 시작을 거부.

`ProtocolAdjudicated`는 잠긴 구체 정책과 실제 adapter가 없으므로 v1에서 거부한다.
`adjudication_enabled=true`만으로 임의 판정을 득점에 넣지 않는다. claim/tablebase/평가점
판정은 정확한 정책 identity·근거와 연결하는 후속 범위다. CI/Elo/bootstrap 계산은 없다.

JSONL header는 version·canonicalization·manifest SHA·plan SHA를 묶고, 각 record는
seq·prev SHA·event SHA를 갖는다. 중복·순서 교체·내용 변경·다른 계획·잘린 JSON·
마지막 newline 누락을 거부한다. 모든 append는 검증·상한 검사 성공 후 반영되므로
실패 시 원장 상태가 유지된다. 호출자는 양수 events/bytes 상한을 명시하며 manifest의
output 상한도 적용한다. 한 줄 JSON은 최대 4 MiB다.

hash chain은 외부 producer의 인증 수단이 아니며 **완전한 record 경계의 suffix 삭제는
체인만으로 검출할 수 없다**. 별도 보존한 tip을 `verify_tip` 또는 audit의 `--expected-tip`으로
대조해야 한다. 신뢰할 checkpoint 없이 짧은 유효 prefix를 완료 기록으로 인증하지 않는다.
JSONL replay는 선언 기록의 구조 복원이며 실제 runner resume·새 process·원자적 파일 append·
다중 writer 수명 검증이 아니다. 실제 재개는 중단 pair를 분리하고 새 process 영수증과
사전 whole-pair 정책을 연결하는 후속 작업이다.

## CLI와 재현

저장소 루트에서 본인 소유의 외부 artifact root를 선택한다. 생성물은 환경 종료 전에
회수해야 한다. lockfile과 로그를 영구 원격 보존한 것으로 해석하지 않는다.

```sh
export ARTIFACT_ROOT=/path/outside/checkout/rovezero-e
export CARGO_TARGET_DIR="$ARTIFACT_ROOT/build"
mkdir -p "$ARTIFACT_ROOT/runs" "$ARTIFACT_ROOT/reports"
cargo check --manifest-path crates/rz-arena/Cargo.toml
cargo test --manifest-path crates/rz-arena/Cargo.toml --locked
cargo clippy --manifest-path crates/rz-arena/Cargo.toml --all-targets --locked -- -D warnings
cargo fmt --manifest-path crates/rz-arena/Cargo.toml -- --check

cargo run --manifest-path crates/rz-experiments/Cargo.toml -- lock experiments/baselines/fixtures/e01-input.json "$ARTIFACT_ROOT/runs/e02-input-lock.json"
cargo run --manifest-path crates/rz-arena/Cargo.toml -- plan "$ARTIFACT_ROOT/runs/e02-input-lock.json" "$ARTIFACT_ROOT/runs/e02-plan.json" --max-pairs 1 --max-plan-bytes 1048576
cargo run --manifest-path crates/rz-arena/Cargo.toml -- ledger-init "$ARTIFACT_ROOT/runs/e02-plan.json" "$ARTIFACT_ROOT/runs/e02-ledger.jsonl" --max-pairs 1 --max-plan-bytes 1048576 --max-events 64 --max-ledger-bytes 1048576
cargo run --manifest-path crates/rz-arena/Cargo.toml -- audit "$ARTIFACT_ROOT/runs/e02-plan.json" "$ARTIFACT_ROOT/runs/e02-ledger.jsonl" --max-pairs 1 --max-plan-bytes 1048576 --max-events 64 --max-ledger-bytes 1048576
```

`ledger-append PLAN LEDGER EVENT OUTPUT`는 한 사건 JSON을 검사하고 **새 파일**에 다음
원장을 쓴다. ledger-init/audit와 같은 네 상한 옵션이 필요하다. 기존 output은 덮어쓰지 않는다.
입력은 bounded UTF-8 regular file이며 Unix에서 final symlink·FIFO blocking을 방지한다.
쓰기 실패의 부분 파일은 보존하고 오류를 반환한다. audit의 `tip_sha256`을 외부 보존 위치에
기록한 경우 다음 감사에 `--expected-tip`으로 대조할 수 있다. audit의 성공은
`validation_scope=structural_only`이고 run 명령은 제공하지 않는다.

## 현재 인수 범위

내부 plan·사건 회계·JSONL 검증을 CPU와 합성 fixture로 확인한다. 실제 실행 receipt,
A의 checked opening 복원·독립 참조·전체 PGN, 외부 Fastchess/Cute Chess adapter,
clock/resource/drain/reset, 실제 runner 재개는 후속이다. 실제 NN·GPU·대국 강도·CI 성공은
이 내부 코드로 인수하지 않는다. 진행과 정확한 검사 근거는
[E 계획서](../../experiments/baselines/IMPLEMENTATION-PLAN-E.md)에 기록한다.
