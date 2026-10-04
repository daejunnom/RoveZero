# rz-arena — E02 pair·원장·fixture와 CPU 신경망 연결

## 실행 후 독립 pair PGN 감사

`cargo run --release -p rz-arena --example pair_pgn_audit -- PAIR_INPUT_JSON TWO_GAME_PGN`
은 기존 A Rules와 `audit_pair_pgn_for_spec`으로 두 판의 시작 상태·전체 이동열·흑백
배정·종료를 대조한다. 입력은 `schema_version=1`, 잠긴 `PairSpec`인 `pair`, 양수
`max_plies`를 담으며 알 수 없는 필드를 거부한다. JSON은 64KiB, PGN은 4MiB 이하로
읽고 ply 한도는 기존 감사기의 4095 이하 제한을 따른다. 엔진 실패는 Loss, 최대 ply
중단은 Incomplete로 분류한다. 이 예제는 실행 후 감사만 제공하며 native V1의 B1
제약을 확장하거나 GPU·시계 공정성·정식 강도 인수를 증명하지 않는다.

같은 opening 입력에 흑백 엔진 배정만 교환하는 pair를 만들고, 모든 game attempt의
결과·실패를 보존한다. E01의 잠긴 입력, A의 checked Rules와 bounded Fastchess
fixture adapter를 소비하며 새 엔진·Rules·UCI를 구현하지 않는다. 출력은 모두
**`execution_ready=false`**다. 합성 fixture와 실제 CPU 신경망 연결을 별도 실행 경로로 제공하며,
GPU·강도 검증은 별도 인수한다.

## 실제 CPU 신경망 pair

`IntegrationPairSpecV1`은 E의 별도 wire version 1이며 공통 엔진 계약 revision 0.1을
소비한다. 첫 연결 검사는 같은 binary·원본 가중치·ONNX·manifest·ORT·탐색을 두 역할에
사용하고 흑백만 교환한다. 내부 성능 변경을 허위로 선언해 강도 실험 형식에 끼워 넣지 않는다.
`strength_eligible=false`, `scored_games=0`을 유지한다.

```text
rz-arena native-lock INPUT OUTPUT
rz-arena native-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME
```

잠금은 64 KiB 이하의 선언 JSON·고유 입력 예산·CPU FP32 batch/thread/worker/step 1과
expected backend/encoding digest를 검사한다. `native-pair`는 Linux에서만 실행하며,
호출자가 독점 관리하는 저장소 밖의 입력·출력 root와 새 basename을 요구한다. 선정한
Maia 원본과 변환본, export manifest, 검증된 ORT 및 `onnx-cpu` feature로 빌드한 UCI
binary를 명시한다. 외부 가중치의 GPL 조건과 MIT 자체 코드의 권리는 분리한다.

준비 단계는 모든 source bytes를 검사한 뒤 별도 inode로 복사하고 writer를 닫는다.
실행 owner가 runner·engine·모델·ORT·cwd·역할별 runtime 디렉터리 핸들을 보존한다.
Unix 읽기 전용 권한은 동일 UID의 변경을 막는 OS seal이 아니며, 부모와 고정 프로그램의
독점 소유 가정을 요구한다. 프로세스·트리 한도는 관측 방식이다. 주소 공간 제한은
별도 실행 wrapper의 상속 설정·근거를 기록하며 이 라이브러리가 RAM 제한을 설치하지 않는다.

성공은 exit 0·프로세스 group `Gone`·pending child/감독 오류 없음·A 기반 PGN 감사·
정확히 네 개의 B startup/final 기록을 함께 요구한다. 역할마다 두 번 새 프로세스를
시작하고 CPU provider·입력 hash·registry·fresh computed 완료·확정 drain을 비교한다.
시작 시 `loaded=true`만으로 실제 추론을 인정하지 않는다. 완료 수는 D가 수락한
평가 응답 집계이며 물리 추론 호출 수나 전체 root별 journal을 대신하지 않는다.

준비 실패는 원래 cause·부분 복사·writer 종료·저장 실패를 별도 typed receipt에 보존한다.
실행 후 정리가 미확인되면 원래 Child와 전체 입력 owner를 같은 quarantine에 보존하고
새 입장을 닫는다. ownership loss 이후의 handle 보존은 숫자 PID의 wait/signal 권한을
복구하지 않는다. 보존은 해당 부모 프로세스 수명 범위이며 CLI 종료가 미확인 자식 정리를
보장하지 않는다. attempt 자동 삭제·재시도는 제공하지 않는다.

총괄은 `0f0d70d`의 실제 CPU NN 두 판을 인수했다. 6 ply 제한의 원시 PGN draw는
`Incomplete` 두 판으로 남기며 득점에 넣지 않았다. 정확한 source·binary·검사·보존 근거는
[CPU NN pair 인수 기록](../../docs/INTEGRATION-STATUS.md#후속-실제-cpu-신경망-pair-인수)을 따른다.

## PR #7과 계약 연결 경계

[PR #7](https://github.com/daejunnom/RoveZero/pull/7)의 공통 Rust 계약 revision 0.1은
`develop`의 `3e3cd80533a69ee2118f1c130e72a1cb0745ae8c`에 병합됐다. 현재 총괄
통합본은 같은 root path 계약과 A Rules를 소비한다. 당시 독립 branch의 PR 상태와
검사 보고를 현재 통합 SHA의 CI 성공으로 재사용하지 않는다.

이번 구현의 pair/game/attempt ID와 결과 enum은 E의 영속 기록 형식이다. 공통
RequestId·Move·신경망 WDL·StateIdentity를 복제하지 않는다. opening은 E01의
`OpeningSpec`을 재사용한다. 공통 계약의 `PositionSnapshot<P>`는 A가 복원한 상태를
연결하는 접점이며, 이 코드의 입력 hash를 그 실제 semantic digest로 사용하지 않는다.
공통 오류·generation·runner/PGN/시계 사건을 연결할 때 계약 0.1의 정확한 revision과
실제 producer를 확인한다. 계약 연결 전에도 내부 구현을 진행하라는 사용자 지시를 따른다.

현재 두 E package는 root workspace·lockfile을 공유한다. root Cargo·CI·공통 타입은
I01 소유다. 자체 코드는 [MIT LICENSE](LICENSE)를 따른다. 의존성 버전은 root Cargo와
[E01 README](../rz-experiments/README.md)를 따른다. Unix/Linux 전용 실행과 Windows의
명시적인 미지원 반환을 구분한다.

Linux fixture supervisor는 검증한 실행 파일, 유한 process group, 출력·artifact 한도와
A Rules 기반 PGN 감사를 제공한다. 전체 ply는 4095 이하이며 config·opening·PGN이
공유 artifact 예산을 사용한다. Fastchess 로그 인자는 기존 bounded stdout으로 연결하지만
실제 trace의 완전성은 별도 확인한다. 총괄은 고정 Fastchess의 normal/cutoff/artifact 거부/
live cancel 전체 fixture를 실제 실행해 인수했다. 정식 대국 시계·장비·통계 인수는
후속이며 `execution_ready=false`를 유지한다. 정확한 binary·소스·결과는
[총괄 인수 기록](../../docs/INTEGRATION-STATUS.md#후속-e-고정-fastchess-fixture-인수)을 따른다.

선택적 UCI combo `FixtureMode`는 `normal`, `crash`, `illegal`, `timeout` 네 값만
받는다. 기존 CLI `--mode`는 초기 기본값이며 적용 설정은 `ucinewgame` 뒤에도 남는다.
timeout은 `stop`/`quit`을 계속 처리하며 외부 runner가 유한 시간·취소를 소유한다.
builder는 Fixture·CPU·무가중치 제한 뒤에만 이 옵션을 전달한다. 실제 RoveZero/LC0의
argv·모델/provider 명세를 이 옵션으로 대체하지 않는다. crash/illegal binary 회귀와
실제 고정 runner의 네 gate를 구분하며, 강도 실험의 실패 판 득점 인수는 후속이다.

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
실제 evidence 파일 내용·PGN의 진위는 아직 감사하지 않는다.

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
`validation_scope=structural_only`이고 정식 run 명령은 제공하지 않는다.
후속 `fixture-pair`는 Linux의 합성 CPU smoke 전용이며 아래 범위를 따른다.

## 현재 인수 범위

현재 source에는 plan·사건 회계·JSONL 외에 A의 checked opening·PGN replay,
Fastchess 인자 adapter·verified file handle과 유한 Linux subprocess 수명 코드가 있다.
`fixture-pair`는 고정한 외부 runner와 실제 engine·입력·PGN의 identity를 연결한다.
총괄의 새 Linux 실행은 normal과 cutoff의 A/독립 PGN 대조, artifact 사전 거부,
실행 중 취소의 Cancelled·Gone과 득점 제외를 확인했다. 코드 반입·단위 fixture와
이 전체 실행을 구분한다. 정식 clock/draw/seed·CPU/RAM·GPU drain·통계·resume 인수는 후속이다.
`ProcessReceipt.elapsed_ns`는 subprocess와 cleanup 구간이며 앞뒤의 입력 검증·opening
생성·PGN 감사·기록 비용까지 포함한 전체 실행 시간이 아니다.
이 코드는 `execution_ready=false`를 유지한다. 실제 NN·GPU·대국 강도는 별도 인수한다.
진행과 정확한 검사 근거는
[E 계획서](../../experiments/baselines/IMPLEMENTATION-PLAN-E.md)에 기록한다.

## 담당의 단독 검증 보고 — 2026-10-03, Asia/Seoul

Linux x86_64·Rust 1.99.0·기본 feature에서 arena CLI 10개, ledger 22개, planner 15개,
총 47개 테스트가 통과했다. 변경된 E01의 기존 31개도 통과해 전체 78개다.
두 package의 fmt/clippy `-D warnings`와 Rust 1.85.0
`cargo check --all-targets --locked`가 통과했다. README 링크와 Git whitespace도 검사했다.
다른 OS·실제 GPU·CI의 성공으로 재사용하지 않는다.
아래 78개 보고는 후속 runner가 추가되기 전 담당 branch의 범위다. 현재 root 통합은
Rust 1.96.0·workspace MSRV 1.90을 사용하며 정확한 integration SHA와 실제 실행은
[총괄 인수 기록](../../docs/INTEGRATION-STATUS.md)에 따로 기록한다.
receipt의 초기 contract/rules source pin을 현재 binary의 integration SHA로 해석하지 않는다.

## Additive CUDA integration 소비자

`native-cuda` feature는 B의 별도 `CudaStartupReceiptV1`·`CudaTerminationReceiptV1`을
소비한다. CPU V1 profile·파일명·wire 검증은 그대로 유지하며, GPU 입력은
`CudaIntegrationPairSpecV1`의 별도 lock domain으로 잠근다. library API는
`prepare_native_cuda_launch`와 `run_native_cuda_pair`다. CUDA provider를 명시적으로
선택하지 않은 CPU 경로에 GPU 설정이나 proof를 넣지 않는다.

Feature를 포함해 빌드한 CLI에는 `native-cuda-lock INPUT OUTPUT`와
`native-cuda-pair LOCKED ARTIFACT_ROOT OUTPUT_ROOT NEW_OUTPUT_BASENAME`을 제공한다.
입력/lock은 64 KiB로 제한하고 새 output만 생성한다. Pair 명령의 SIGINT/SIGTERM은
기존 CPU 경로와 같은 유한 supervisor 취소 신호로 전달한다. typed preparation와
실행 실패 영수증을 stderr에도 보존하며 실패·미확인 cleanup을 성공 exit로 바꾸지 않는다.
Default feature의 CPU 명령·기본 help와 CPU wire는 유지한다.

GPU 명세는 device 0·FP32·TF32 off·batch 1·intra 1·worker 1·fresh/full 1·HistoryFill No와
1 GiB arena admission을 요구한다. 이 설정은 측정된 VRAM이나 aggregate allocation·
강제 hard cap의 증거가 아니다. raw manifest SHA, canonical binary-codec bundle SHA,
ORT Core identity는 각각 잠그며 19개 알려진 library filename·role·길이·SHA를 모두
검사한다. 두 엔진은 같은 source·asset·runtime·profile 입력을 사용한다.

GPU budget은 CPU 소형 input/tree 예약과 별도로 선언한다. 고유 입력 전체와
restart 두 판의 native process 네 개를 위한 **bundle 복사 네 회 + 18 MiB**를
runtime 예약에 포함한다. 18 MiB는 process당 4 MiB placement trace와 512 KiB
startup/final receipt 상한의 합이다. retained stream·PGN·config·E receipt는
output 예약에 포함하고, 전체 artifact 예약은 input/runtime/output 상한을 덮는다.
tree snapshot은 관측 한도이며 disk·RAM·VRAM의 kernel quota를 설치하지 않는다.

준비 단계는 manifest와 19개 library를 private `inputs/cuda-bundle/`에 서로 다른
inode·닫힌 writer·readonly 파일로 복사해 sibling basename을 보존한다. readonly
권한은 immutable seal이나 같은 UID에 대한 sandbox가 아니다. 총괄이 ancestor와
output을 독점하며 pinned program을 신뢰하는 실행 가정을 유지한다. 원본 오류와
준비 실패 attempt의 파일 소유자를 보존하고 기존 attempt를 자동 재사용하지 않는다.
spawn 후 cleanup이 미확인인 경우 입력·cwd·lease·원래 Child·receipt를 함께 보존하고
단일 shared native admission을 닫는다. CPU와 CUDA에 각각 새 unresolved slot을 만들지 않는다.

인수에는 Exited 0·Gone·pending child 부재·process 오류 부재, 네 unique native session,
matching binary/asset/bundle/profile, CudaOnnx의 D accepted computed completion,
최종 tree guard가 수락한 root/non-root consumption, confirmed physical drain과 A PGN
감사를 각각 요구한다. B의 observed placement trace SHA는 사전에 잠그지 않는다.
해당 PID의 알려진 runtime directory에 단일 bounded regular trace가 있는지 확인하고
실제 bytes·SHA·CUDA kernel provider·node count를 사후 대조한다. 추가·모호한 후보,
symlink·hardlink·CPU node·누락 기록·손실/overflow·원래 오류는 인수를 거부한다.

합성 metadata·wire·snapshot 테스트는 parser와 거부 경로의 검사다. 이 additive
소비자의 build/test·실제 GPU pair 실행은 source SHA별 총괄 인수로 기록한다.
`e45dd1050d8782ff4ae3e524341cdbac52209cf5`의 실제 CUDA integration pair는 네 native
session의 computed/search/drain과 각 PID의 Fastchess 종료 TRACE status 0을 확인했다.
그 전의 late SIGABRT·startup timeout 실패도 보존한다. 원본과 검사 범위는
[통합 인수 상태](../../docs/INTEGRATION-STATUS.md)에 둔다.
성공 receipt도 `execution_ready=false`, `strength_eligible=false`,
`scored_games=0`이며 cutoff draw는 Incomplete다. placement probe 및 process aggregate를
모든 root/bestmove의 실제 NN 실행 수나 모델 강도 증거로 확대하지 않는다.

## 후속 실행 자료 인수

E `91818e3d1309592cff561aa925688446c4a2e4e4`의 공개 보존 사본 454개 파일을
저장소 밖 `reports/coordinator-integration/recovered-pr-evidence/exact-blobs/E-91818e3`에
회수하고 전체 Git blob identity·길이를 확인했다. 독립 감사에서 inventory 448개와
process receipt 11개가 참조한 artifact 46개의 digest·크기가 맞았다. 출처는
[E 보존 커밋](https://github.com/daejunnom/RoveZero/tree/91818e3d1309592cff561aa925688446c4a2e4e4/experiments/baselines/evidence),
현재 인수는 [총괄 기록](../../docs/INTEGRATION-STATUS.md)을 따른다.

과거 7수 script fixture의 1승·1패, cutoff Incomplete, signal 15 취소와 cleanup
Unverified를 그대로 보존한다. 과거 Linux standalone의 140개 검사 결과를 현재
root 통합의 양 OS 검사로 재사용하지 않는다. 마스킹 전 클라우드 원본·제외 binary·
누락된 최초 build 로그까지 회수했다고 보고하지 않는다.

그 당시 `check-e.py`와 `run-e-fixture.py`는 별도 외부 원본 사본으로 보존한다.
개별 lockfile 가정·오래된 binary의 SHA 선언·일부 무제한 출력 및 자식 정리 경계가
현재 root와 맞지 않아 실행 도구로 등록하지 않았다. root fixture 도구를 제공하려면
단일 lock·실제 binary 빌드 영수증·유한 cleanup/output을 맞춘 뒤 별도 검사해야 한다.

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
