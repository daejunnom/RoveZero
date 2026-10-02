# rz-arena — E02 pair·원장·fixture 실행

같은 opening 입력에 흑백 엔진 배정만 교환하는 pair를 만들고, 모든 game attempt의
결과·실패를 보존한다. E01의 잠긴 입력, A의 checked Rules, 고정 Fastchess를 연결한다.
테스트 전용 UCI script는 신경망·탐색이 없는 합성 엔진이다. 모든 출력은
**`execution_ready=false`**이며 실제 NN·GPU·강도 검증을 뜻하지 않는다.

## PR #7과 계약 연결 경계

[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의 확인한 head는
`18339c30754f4f38d957d96aeab481b018b98718`이다. open/ready-for-review이며
Contracts CPU workflow `37033676814`의 success를 관측했다. E CI의 성공과 구분한다.
공통 API source는 `67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`에 고정한다.
아직 `develop`에 병합되지 않은 [A PR #3](https://github.com/daejunnom/RoveZero/pull/3)의
Rules source `118dc0311a88e143be285703940321dc16261f6a`를 같은 계약 source와 연결한다.

이번 구현의 pair/game/attempt ID와 결과 enum은 E의 영속 기록 형식이다. 공통
RequestId·Move·신경망 WDL·StateIdentity를 복제하지 않는다. opening은 E01의
`OpeningSpec`을 재사용한다. 공통 계약의 `PositionSnapshot<P>`는 A가 복원한 상태를
연결하는 후속 접점이며, 이 코드의 입력 hash를 그 실제 semantic digest로 사용하지 않는다.
manifest schema 1과 engine contract 0.1은 별개다. 실행·PGN 경계에서는
`contract_revision="0.1"`만 native `SchemaVersion`으로 변환해 validate하며,
공통 `ContractError`의 code/stage를 보존한다. source SHA는 schema version과 따로 기록한다.

독립 package workspace로 빌드한다. root workspace·lockfile·CI·공통 타입은 I01 소유다.
총괄이 정식 member로 등록할 때 두 E package의 `[workspace]` 경계와 임시 lockfile을
정리하고 같은 통합 SHA로 인수해야 한다. 자체 코드는 [MIT LICENSE](LICENSE)를 따른다.
arena는 A의 요구에 맞춰 Rust 1.90 이상을 사용한다. E01은 1.85를 유지한다.
E01 path·serde/serde_json/sha2/libc 외에 cap-std/cap-fs-ext 4.0.3,
nix 0.30.1(MIT), signal-hook 0.3.18(MIT/Apache-2.0), 두 RoveZero Git pin(MIT)을 소비한다.
root 등록·MSRV·단일 lockfile 통합은 I 소유이며 이 PR에서 root를 변경하지 않는다.

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
evidence metadata를 요구한다. 입력·evidence 전체에서 같은 경로의 다른 identity와
고유 artifact byte 상한 초과를 거부한다. 같은 identity의 재사용은 중복으로 계산하지 않는다.
일반 원장 replay는 구조 검사다. 아래 fixture adapter에서 생산한 기록만 실제 process와
A의 PGN 감사 영수증을 연결한다. 외부 선언 원장을 실행 증명으로 승격하지 않는다.

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
`validation_scope=structural_only`다. 합성 실행은 아래 별도 명령을 사용한다.

## 현재 인수 범위

`fixture-pair PLAN ARTIFACT_ROOT NEW_OUTPUT_BASENAME --max-pairs 1 --max-plan-bytes N`은
고정 Fastchess `f618e34540f94f4719ad3817950618dabe441318`의 한 pair를 실제 실행한다.
fixture·CPU·계약 0.1·한 worker·runner 포함 child 최소 3·증분 없는 정수초 T1만 받는다.
명령 인수를 직접 전달하고 환경을 초기화하며, E01이 검증한 동일 ELF 핸들로 실행한다.
caller는 입력 파일 내용과 artifact root/attempt 이름을 실행 중 배타적으로 관리해야 한다.

stdout/stderr는 합산 hard byte cap, PGN/config는 regular-file snapshot cap을 적용한다.
유한 wall·취소·TERM→KILL·group 정리 상태와 PID·종료 code/signal을 영수증에 기록한다.
leader PID를 정리 관측 전 reap하지 않으며 외부 child reaper와 SIGCHLD 변경은 지원하지 않는다.
`pending_child`는 장기 library caller가 후속 reap해야 한다. 저장 실패는 captured bytes와
process ownership을 `ArenaError::Execution`으로 보존한다. CLI는 실패에 exit 2를 반환한다.
config.json 자동 저장과 전체 opening prefix도 artifact 예산에 포함한다.

PGN은 A의 실제 `ContractPosition.export()`와 checked transition으로 origin FEN·차례·
권리·EP·카운터·이력·합법 수·terminal·양색 엔진 배정·실행 순서를 검증한다.
입력 hash와 A semantic SHA는 별개다. SAN/UCI는 받지만 variation/임의 adjudication/
claim/불명확 engine-loss는 거부한다. 정확한 cutoff PGN의 runner Draw는 `incomplete`로
제외한다. 실제 engine-loss는 pinned 종료 이유로 책임이 확인될 때만 기록한다.

Fastchess clock boundary와 자동 draw profile, per-game RNG seed 적용, CPU/RAM/GPU 공정성,
kernel child/disk quota, runtime/GPU drain은 미검증이다. 탈출·순간 fork를 snapshot으로
보장하지 않는다. 취소 뒤 자식 좀비가 남은 실제 사례는 `Unverified`·pair 제외로 보존한다.
본 adapter는 deterministic script fixture만 지원하고 development/formal 강도 실행은 거부한다.

재현은 [run-e-fixture.py](../../experiments/baselines/scripts/run-e-fixture.py),
검사는 [check-e.py](../../experiments/baselines/scripts/check-e.py)를 사용한다.
Fastchess binary는 별도로 권리·source/build를 확인해 제공한다. 실행 원본과 재현 절차는
[원격 보존 목록](../../experiments/baselines/evidence/README.md)에 기록한다.

### E02 초기 내부 구현 당시 인수 범위

내부 plan·사건 회계·JSONL 검증을 CPU와 합성 fixture로 확인한다. 실제 실행 receipt,
A의 checked opening 복원·독립 참조·전체 PGN, 외부 Fastchess/Cute Chess adapter,
clock/resource/drain/reset, 실제 runner 재개는 후속이다. 실제 NN·GPU·대국 강도·CI 성공은
이 내부 코드로 인수하지 않는다. 진행과 정확한 검사 근거는
[E 계획서](../../experiments/baselines/IMPLEMENTATION-PLAN-E.md)에 기록한다.

## 실제 로컬 검증 — 2026-10-03, Asia/Seoul

Linux x86_64·Rust 1.99.0·기본 feature에서 arena CLI 10개, ledger 22개, planner 15개,
총 47개 테스트가 통과했다. 변경된 E01의 기존 31개도 통과해 전체 78개다.
두 package의 fmt/clippy `-D warnings`와 Rust 1.85.0
`cargo check --all-targets --locked`가 통과했다. README 링크와 Git whitespace도 검사했다.
다른 OS·실제 GPU·CI의 성공으로 재사용하지 않는다.

독립 Python SHA/seed vector와 대조한 E01 fixture 계획 SHA-256은
`980d61aadc3da58fe75720b125f7017a08ad40c5ae58c490f6e411e89429511f`,
opening 입력 SHA-256은 `2bfb56da02967f952312886e257018974843fb0ea9a81fa5b0265e94f4240e46`이다.
큰 입력의 compact 계획이 3 MiB에 들어가도 중간 pretty JSON이 4 MiB를 넘던 경계를
별도 회귀로 검사했다. compact nesting으로 정상 생성·roundtrip하며 기존 digest는 유지된다.

실제 CLI로 입력 잠금→계획→원장 생성→합성 사건 30개 추가→감사를 실행했다.
LL/DL/DD/WL/WD/WW 선언의 손계산과 W=D=L=4, `n=[1,1,2,1,1]`, 완료 pair 6이 일치했다.
이는 **합성 결과 선언의 회계**이며 실제 12판 체스를 실행한 기록이 아니다.
원장 끝의 완전한 record 하나를 지운 prefix는 구조적으로 유효하지만 원래 trusted tip과
대조하면 exit 2로 거부되는 것도 CLI에서 확인했다.

검사 로그·명령 영수증·입력/원장·독립 lockfile은 외부 artifact root의
`reports/e02/`, `runs/e02-synthetic-*/`에 보존한다. arena 임시 Cargo.lock의 SHA-256은
`d890a0f527543df34e5a47942a99ce3eeeefb2c687682b4e54c84e2531cdbcac`이다.
E01 lockfile은 기존 `3f0688d9…`와 같다. 환경 종료 전 회수와 I01 정식 lockfile 인계가 필요하다.
