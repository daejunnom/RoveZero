# D01 평가 런타임

담당은 D, 작업은 TASK-D01과 D02 내부 계측이다. 브랜치 기반은 문서 커밋
`9f0bc59`이며 총괄 [PR #7](https://github.com/daejunnom/RoveZero/pull/7)의
`67284c4f66f7a7ae9f46fa63dfd50e7410eb6845`에서 게시한 공통 계약 0.1을 참조한다.
generic core의 associated type 경계와 `contracts` feature의 실제 타입 adapter를
함께 제공한다. 공통 ID·WDL·EvalResult·계약 revision을 복제하지 않는다.

Scheduler의 책임은 유한 FIFO queue, 호환 batch, 논리 종결과 물리 실행의 수명이다.
B는 selection/reservation/backup을 소유하고 C는 실제 evaluator와 opaque Lease를 제공한다.
입력 metadata와 raw 평가의 identity는 adapter가 그대로 보존한다. 최초 기준은
cache/dedup/speculation/warm-start 없이 fresh 요청을 실행한다.

현재 standalone `[workspace]`는 root 파일을 수정하지 않고 D 소유 crate를 검사하기
위한 임시 구성이다. 실제 계약은 위 SHA의 optional git dependency로 고정했다.
I01 workspace 통합 시 standalone 선언·crate lockfile·git pin을 정리하고 root
dependency/feature/toolchain/CI와 같은 integration SHA로 다시 검사한다.

## 공통 계약 0.1 연결

`contracts` feature의 `ContractsAdapter<P,C>`와 `ContractEvaluator<P,B,C>`는
실제 `rz-contracts::Evaluator<P>`를 구현한다. Rules의 불변 `Arc<EvalRequest<P>>`를
받고 opaque backend Lease를 유지한다. common poll까지 결과 receipt의 예약이 남는다.
poll은 buffered success를 현재 scope·취소·strict deadline으로 다시 검증한다.
성공 결과의 actual physical ExecutionId, fresh provenance, ordered legal payload와
compute budget을 확인한다. 결과를 받은 B는 backup commit에서도 다시 검증한다.

`SharedScope`는 game/root/model/encoding/backend 교체를 전달한다. ID 재사용 추적에
무한 집합을 만들지 않고 고정 epoch의 request sequence high-water를 사용하므로 제출은
엄격히 증가하는 순서여야 한다. 거절된 제출도 ID를 소비하며 재시도에는 새 ID가 필요하다.
한 epoch의 execution 발급자는 하나여야 한다. 런타임 재생성 시 기존 발급 high-water를
보존하거나 새 process epoch를 사용한다. 같은 epoch에 새로운 clock origin을 만들지
않고 `ContractSystemClock` clone을 공유한다. 모델별 batch 상한보다 큰 설정은 거절한다.

이 연결은 실제 공통 타입으로 검사한 D 경계다. C01 실제 evaluator와 B search를
같은 integration SHA에서 붙인 검사는 아직 아니다. 기본 feature는 계약 중립 core를
유지한다. 계약 조회·캐시 초기화 이후에는 두 구성을 `--locked --offline`으로 검사한다.

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
scheduler의 예약 ledger 밖이다. telemetry ring·관측 ring과 ID/queue/map/channel bookkeeping의
고정·항목별 비용도 manifest에서 별도로 계산한다. D의 memory limit은
그 비용을 제외한 승인된 잔여 예산으로 정해야 한다. heap을 소유한 generic ID의 clone
비용도 metadata 예산에 포함한다. `State.reserved`는 실제 VRAM 측정값이 아니다.
`State.peak_reserved`와 요청 수 high-water는 budget lock 안에서 갱신하므로 다른
스레드에서 receipt를 수신·drop해도 domain별 예약 peak를 잃지 않는다.

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

`Drained`는 physical 작업이 없다는 뜻이다. 미수신 terminal delivery가 소유한 예약은
남을 수 있다. `shutdown_snapshot`의 drain 상태와 ledger 상태를 함께 보고하고
result 수신·drop까지 끝내야 모든 요청 예약이 반환된다.

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

`Scheduler::take_observations`와 common wrapper의 같은 메서드는 유한 관측 ring을
시간 순서로 비운다. 용량은 constructor의 `metric_sample_capacity`와 같다. ring은
오래된 event를 버리고 drain별 `dropped`·`counter_overflow`를 보고한다. 용량 0에서도
요청 결과·예약 수명은 같으며 모든 event는 유실로 센다. generic ID의 크기·clone 비용과
관측 ring·drain 반환 Vec의 예산은 별도 metadata 예산이다.

관측 tick은 deadline과 같은 owner clock이다. 예약 snapshot은 다른 스레드에서
receipt가 해제된 정확한 시각을 뜻하지 않는다. `peak_reserved`는 lock에서 보존한
high-water이며 이를 profiler에 별도로 전달할 수 있다. `ValidationFinished.valid`는
payload 검증 사실이고 이후 stale/expired가 될 수 있다. `Finished.delivered`는 mailbox
enqueue 성공이며 실제 수신·selection consume·backup을 뜻하지 않는다.

D02의 bounded `rz-telemetry::profile::TraceCollector`는 CPU 준비·queue·dispatch·
backend physical·validation·backup·output·종단 span을 받아 class별 percentile,
batch 분포·예약 peak와 physical/consumed/unused 집계를 제공한다. 같은 monotonic
domain을 사용하며 wall span으로 처리량을 계산한다. 전체 phase 시간을 합산하지 않는다.
actual transfer가 없으면 Transfer는 absent다. engine ID·정확한 소비 권한은 caller가
검증하고 profiler의 중복 검사는 유한한 관측 window에만 적용된다. 자세한 사용·손실
경계는 [계측 설명](../rz-telemetry/PROFILE.md)에 있다.

[CPU trace 예제](examples/cpu_trace.rs)는 cold/warm·parent-child·sibling·transposition·
eviction·long 입력 관계 fixture를 독립 실행하고 runtime의 실제 queue·취소·receipt·
drain을 검사한다. fixture clock의 단계별 지연과 실제 `Instant` 실행 벽시계를 분리한다.
실제 B backup/UCI 출력·GPU provider hook 연결은 추가 인수가 필요하다. 입력 관계
label은 cache·dedup·warm-start 구현이나 D03 성능 개선을 뜻하지 않는다.

이번 인계의 기반은 `9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`이다. Rust
1.96.0/Linux x86_64, CPU 5개·RAM 약 17.6 GiB의 클라우드에서 standalone으로
검사했다. 실제 GPU 장치와 GPU provider는 사용할 수 없었다. 최종 소스에서 다음이
통과했다. 실제 신경망을 사용한 검사는 아니다.

- `tests/lifecycle.rs`의 독립 CPU/mock 수명 검사 30개: queue/batch, 마감·세대·취소,
  malformed 결과, 미수신 delivery 예산, physical pin, timeout·quarantine과 실제
  CPU worker 실행 중 다른 스레드의 취소를 포함한다.
- 관측 ring 내부 검사 3개와 실제 계약 adapter 검사 14개: 유한 이벤트 손실과
  capacity overflow, execution identity, epoch·clock domain·ID 재사용, 미수신
  common 결과 예산과 poll 시 재검증을 포함한다.
- `rz-telemetry`의 계측 검사 8개: logical/physical 분리, 고정 용량·표본 손실,
  percentile·counter overflow와 할당 전 capacity 검증을 포함한다.
- profiler 검사 10개와 CPU trace 예제 검사 4개: 겹친 span·exec-only wall 시작,
  실제 소비·미사용 작업, 시간·domain·유실·tracking 상한, 7종 trace와 작은 ring의
  2,048 요청 장기 재생을 포함한다. **합계 69개**가 통과했다.
- 두 crate의 Rust 1.96 `cargo fmt -- --check`, `cargo clippy --all-targets -- -D warnings`와
  runtime의 `--all-features` 구성. Rust 1.85에서 runtime `--all-targets --all-features`
  compile도 통과했다.

Rust 1.96 release 예제의 기본 7종 trace × 32 요청은 각 시나리오에서 physical 완료
32·성공 소비 31·취소 1·최종 미사용 1, host 예약 peak 4,608 byte·최종 예약 0,
관측 손실·관측 오류 0이었다. fixture 지연은 실제 호스트 성능 수치가 아니다.
상한·명령·TSV 해석은 [재현 기록](../../benches/runtime/README.md)을 따른다.

lifecycle와 CPU trace의 로컬 ID·오류·출력은 검사 전용이며 공통 계약이 아니다.
계약 adapter 검사는 실제 공통 타입을 사용하되 Rules·모델·head는 독립 fixture다.
기본 lifecycle fixture는
요청 8개·batch 1개·동시 실행 1개·queue age 100 ms·deadline reserve 2 ms,
host/device/pinned 각각 1,000 byte의 **모의 예약 상한**과 ring 16개를 사용한다.
개별 검사는 같은 파일에서 상한을 바꿔 초과·overflow를 검증한다. GPU 실측 상한이나
성능 튜닝값으로 사용하지 않는다. 실제 thread 검사에는 2초 receive timeout을 둔다.

검증 target은 저장소 밖 `${RZ_D_ARTIFACT_ROOT}/target`에 둔다. 이 클라우드 작업의
build cache와 원시 TSV는 세션 동안만 보존하며 소스·fixture·검사 결과 요약은 draft PR에서
회수한다. 이 D 브랜치의 runtime CI는 미실행이다. PR #7의 공통 계약 CI 성공은
별도 결과이며 D/B/C 소비자 통합 성공을 뜻하지 않는다. 재현 명령:

```sh
CARGO_TARGET_DIR="${RZ_D_ARTIFACT_ROOT}/target/runtime" cargo +1.96.0 test --locked --offline --manifest-path crates/rz-runtime/Cargo.toml --all-targets --all-features
CARGO_TARGET_DIR="${RZ_D_ARTIFACT_ROOT}/target/telemetry" cargo +1.96.0 test --locked --offline --manifest-path crates/rz-telemetry/Cargo.toml --all-targets
cargo +1.96.0 fmt --manifest-path crates/rz-runtime/Cargo.toml -- --check
cargo +1.96.0 fmt --manifest-path crates/rz-telemetry/Cargo.toml -- --check
cargo +1.96.0 clippy --locked --offline --manifest-path crates/rz-runtime/Cargo.toml --all-targets --all-features -- -D warnings
cargo +1.96.0 clippy --locked --offline --manifest-path crates/rz-telemetry/Cargo.toml --all-targets -- -D warnings
CARGO_TARGET_DIR="${RZ_D_ARTIFACT_ROOT}/target/msrv" cargo +1.85.0 check --locked --offline --manifest-path crates/rz-runtime/Cargo.toml --all-targets --all-features
```

공통 계약 0.1의 D adapter는 위 CPU 검사로 연결했다. C01 evaluator·B selection/backup과
root workspace의 실제 소비자 통합은 미인수다.
실제 신경망, 목표 GPU buffer 수명·수치·메모리, D02 GPU 종단 계측, D03 개선과
paired 대국은 미실행이다. GPU 없는 개발 환경의 mock 검사를 GPU 인수로 사용하지 않는다.
후속 I01/I02 인계에는 같은 SHA의 adapter·소비자 검사와 명령·설정·fixture·자원·
실행/미실행 결과를 포함한다. 같은 목표의 draft PR에 진행과 남은 연결을 기록한다.
