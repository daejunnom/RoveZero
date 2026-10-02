# D01 평가 런타임

담당은 D, 작업은 TASK-D01이다. 현재 기반은 문서 커밋 `9f0bc59`이며
총괄 TASK-I01의 `rz-contracts` Rust 선언과 root Cargo workspace는 아직 없다.
이 crate의 adapter는 총괄이 게시할 RequestId/ExecutionId, 요청·결과·오류 타입을
associated type으로 받는다. 공통 ID·WDL·EvalResult·계약 revision을 복제하지 않는다.

Scheduler의 책임은 유한 FIFO queue, 호환 batch, 논리 종결과 물리 실행의 수명이다.
B는 selection/reservation/backup을 소유하고 C는 실제 evaluator와 opaque Lease를 제공한다.
입력 metadata와 raw 평가의 identity는 adapter가 그대로 보존한다. 최초 기준은
cache/dedup/speculation/warm-start 없이 fresh 요청을 실행한다.

현재 standalone `[workspace]`는 root 파일을 수정하지 않고 D 소유 crate를 검사하기
위한 임시 구성이다. I01 workspace 통합 시 이 선언을 제거하고 공통 계약 adapter,
root dependency/feature/toolchain/CI는 총괄의 integration 변경으로 연결한다.

## 사용 경계

- `Adapter`가 공통 계약의 요청·실행 ID, clock tick, 요청·출력·오류·최종 결과 타입을
  제공한다. 요청의 fresh ID, state/legal order/model/encoding/precision/budget identity와
  수치 검증은 adapter의 책임이다. metadata는 요청 수명 동안 불변이어야 한다.
- `submit` 실패는 소유 request와 typed error를 `Rejected`로 반환하고 결과 채널을
  만들지 않는다. B는 이 경로에서도 자신의 selection 예약을 해제한다. 수락한 요청은
  `CompletionReceiver`로 한 번만 종결 결과를 받는다.
- 단일 owner가 `pump`를 호출한다. C의 backend dispatch/poll은 비블로킹이어야 한다.
  `Poll::Ready`는 launch 성공이 아니라 input/output/workspace의 **물리 사용 완료**다.
  입력 전체의 tagged ID 집합을 먼저 확인해 duplicate/unknown/missing 응답이 하나라도
  있으면 해당 batch에서 성공을 먼저 게시하지 않는다.
- enqueue validation, dispatch 직전, output validation 이후와 최종 성공 수락에서
  clock·취소·세대를 검사한다. 완료 시점은 같은 monotonic clock에서 `tick < deadline`이다.
  B는 별도의 backup commit에서 세대·deadline·selection consumed를 다시 확인한다.
- queue는 FIFO이며 인접한 정확한 batch key만 묶는다. max items·batch wait·deadline
  reserve에 따라 flush하고, incompatible key가 나오면 앞 묶음을 먼저 실행한다.
  queue age·동시 실행·요청 수와 host/device/pinned bytes는 유한 상한을 갖는다.

## 예산과 소유권

request별 `Resources`는 input/output/pending buffer의 검증된 upper bound다. backend는
그 외 batch workspace/staging의 추가 상한을 제공하며 dispatch 전에 예약한다.
byte 합산은 checked arithmetic을 쓴다. 모델 상주 메모리·caller/search 메모리는 이
scheduler의 예약 ledger 밖이다. telemetry 두 ring과 ID/queue/map/channel bookkeeping의
고정·항목별 비용도 manifest에서 별도로 계산한다. D의 memory limit은
그 비용을 제외한 승인된 잔여 예산으로 정해야 한다. `State.reserved`는 실제 VRAM 측정값이 아니다.

Entry, 물리 Execution과 완료 Delivery는 같은 RAII request reservation을 공유한다.
논리 취소·결과 수신만으로 물리 작업의 pin을 반환하지 않는다. 반대로 물리 완료만으로
아직 채널에서 읽지 않은 결과의 예약을 반환하지 않는다. `reserved_requests`에는
논리 종결 후의 미수신 결과와 취소된 잔여 실행도 포함되어 `max_requests` 우회를 막는다.
`recv/try_recv/recv_timeout` 성공은 payload 소유권을 caller에 넘기는 경계이며,
그 이후 보존·raw cache·search 메모리는 caller의 승인된 예산에서 관리한다.
receiver drop도 예약을 정상 반환한다. 이후 physical 완료가 남아 있으면 실행 pin은 유지된다.

## 종료와 복구

`begin_shutdown(deadline)`은 새 접수를 닫고 logical request를 취소한다. 반복 호출로
drain deadline을 연장하지 않는다. `pump`와 `drain_state`를 사용해 실제 완료를 기다리되
timeout은 `TimedOut`과 현재 예약량으로 보고한다. timeout 뒤에도 같은 인스턴스를
유지하여 pump/drain하거나 해당 엔진 프로세스를 종료한다. 자원이 남은 인스턴스를
버리고 새 scheduler를 반복 생성하는 복구는 금지한다.

완료되지 않은 scheduler를 drop하면 opaque provider/context·Lease·입력·예약을 함께
보존하는 최후의 quarantine을 수행한다. 이 경로는 의도적인 인스턴스별 유한 leak이며
drain 성공이 아니다. 정상 인계에는 **Drop 전에** drain 상태와 metrics를 기록해야 한다.
Drop 안에서만 기록한 quarantine metric은 객체와 함께 사라져 외부 진단 근거가 되지 않는다.
실제 GPU provider의 launch/poll은 panic/unwind 중에도 device pin/context를 보존해야 한다.

## 계측과 인수 상태

`rz-telemetry`는 logical outcome와 physical batch 완료를 따로 센다. 지연·batch 표본은
고정 용량 ring이며 P50/P95/P99는 유지된 표본의 snapshot에서 계산한다. 표본 손실과
counter overflow를 노출한다. 이 지연은 admission부터 logical finish까지이며 B의 backup,
UCI 출력, GPU 전송·완료 전체를 잰 D02 종단 프로파일은 아직 아니다.

이번 인계의 기반은 `9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`이다. Rust
1.99.0/Linux x86_64, CPU 5개·RAM 약 17 GiB의 클라우드에서 standalone으로
검사했다. 실제 GPU 장치와 GPU provider는 사용할 수 없었다. 최종 소스에서 다음이
통과했다. 실제 신경망을 사용한 검사는 아니다.

- `tests/lifecycle.rs`의 독립 CPU/mock 수명 검사 24개: queue/batch, 마감·세대·취소,
  malformed 결과, 미수신 delivery 예산, physical pin, timeout·quarantine과 실제
  CPU worker 실행 중 다른 스레드의 취소를 포함한다.
- `rz-telemetry`의 계측 검사 8개: logical/physical 분리, 고정 용량·표본 손실,
  percentile·counter overflow와 할당 전 capacity 검증을 포함한다.
- 두 crate의 `cargo fmt -- --check`, `cargo clippy --all-targets -- -D warnings`.

fixture의 ID·오류·출력은 검사 전용이며 공통 계약이 아니다. 기본 fixture는
요청 8개·batch 1개·동시 실행 1개·queue age 100 ms·deadline reserve 2 ms,
host/device/pinned 각각 1,000 byte의 **모의 예약 상한**과 ring 16개를 사용한다.
개별 검사는 같은 파일에서 상한을 바꿔 초과·overflow를 검증한다. GPU 실측 상한이나
성능 튜닝값으로 사용하지 않는다. 실제 thread 검사에는 2초 receive timeout을 둔다.

검증 target은 저장소 밖 `${RZ_D_ARTIFACT_ROOT}/target`에 둔다. 이 클라우드 작업의
build cache는 세션 동안만 보존하며 소스·fixture·검사 결과 요약은 draft PR에서
회수한다. 저장소 CI는 아직 구성되지 않아 원격 CI 결과는 없다. 재현 명령:

```sh
CARGO_TARGET_DIR="${RZ_D_ARTIFACT_ROOT}/target/runtime" cargo test --locked --offline --manifest-path crates/rz-runtime/Cargo.toml
CARGO_TARGET_DIR="${RZ_D_ARTIFACT_ROOT}/target/telemetry" cargo test --locked --offline --manifest-path crates/rz-telemetry/Cargo.toml
cargo fmt --manifest-path crates/rz-runtime/Cargo.toml -- --check
cargo fmt --manifest-path crates/rz-telemetry/Cargo.toml -- --check
cargo clippy --locked --offline --manifest-path crates/rz-runtime/Cargo.toml --all-targets -- -D warnings
cargo clippy --locked --offline --manifest-path crates/rz-telemetry/Cargo.toml --all-targets -- -D warnings
```

총괄 공통 계약 revision·C01 evaluator·B selection/backup의 실제 연결은 미인수다.
실제 신경망, 목표 GPU buffer 수명·수치·메모리, D02 GPU 종단 계측, D03 개선과
paired 대국은 미실행이다. GPU 없는 개발 환경의 mock 검사를 GPU 인수로 사용하지 않는다.
후속 I01/I02 인계에는 같은 SHA의 adapter·소비자 검사와 명령·설정·fixture·자원·
실행/미실행 결과를 포함한다. 같은 목표의 draft PR에 진행과 남은 연결을 기록한다.
