# rz-eval — C 평가 구현

담당: TASK-C01/C02/C03. 기준 규약은 저장소의
[구현 지시서](../../docs/IMPLEMENTATION-DIRECTIVES.md),
[계약](../../docs/CONTRACTS.md), [선정 기록](../../docs/WEIGHT-SELECTION.md)이다.

## 현재 제공 범위

`mock::ScriptedBackend<K>`는 수동 시계로 움직이는 **물리 backend 시험 도구**다.
`K`에는 호출자가 공통 계약의 요청/실행 문맥을 넣는다. 별도 RequestId, generation,
EvalRequest/Result, scheduler 또는 search 타입을 선언하지 않는다.

- `Step`이 callback 지연·원시 출력/실패·물리 완료·취소 응답을 정한다.
- `advance_to`와 `next_event`를 분리해 같은 tick에서 root 교체·취소·callback을
  원하는 순서로 소비할 수 있다. 지연 polling에도 원래 event 시각을 보존한다.
- 중복 callback, 다른 요청의 역순 완료, missing head, NaN/Inf를 주입할 수 있다.
  seed는 replay metadata이며 숨은 난수 대신 script가 실행을 결정한다.
- 취소는 callback이나 `DeviceCompleted`를 지우지 않는다. 논리 취소 성공은 D의
  finalization, buffer 해제는 실제 물리 완료, accepted backup은 B의 책임이다.
- script 크기·payload·in-flight·대기 event를 제한한다. admission과 cancel 거부는
  다음 script를 소비하거나 기존 physical completion을 잃지 않는다.

이 도구 자체의 성공을 D의 exactly-once finalization 또는 B의 backup 검증으로
보지 않는다. 실제 소비자와 연결한 검사는 해당 공통 계약이 게시된 뒤 수행한다.

## Maia 출력 검증

`output::validate_maia`는 선정 FP32 export의 1858개 raw policy logits와 이미 확률인
W/D/L head를 검증한다. 호출자가 checked legal move에서 구한 model index 목록을
요청 순서대로 넘기며, 합법 index를 먼저 gather한 뒤 temperature 1.0 softmax를 한다.
WDL에는 softmax를 다시 적용하지 않고 실제 차례 관점을 유지한다.

모든 필수 head의 shape·유한값을 검사한다. legal 목록의 중복·범위·빈 목록, WDL의
범위·합 오류를 명시적으로 거부하며 clipping·자동 renormalization·가짜 무승부로
수선하지 않는다. `ValidatedHeads`는 검증된 **모델 head**만 담고 공통 EvalResult의
요청·세대·관점 metadata를 대체하지 않는다. terminal 상태는 Rules/Search가 선행 처리한다.

첫 코드 profile의 확률 합 허용 오차는 `1e-5`다. 이는 probability admissibility이며
선정 기록의 독립 FP32 parity 기준(raw logit atol=1e-4/rtol=1e-3, 확률 max_abs=1e-4)과
구분한다. 공통 manifest 도입 때 이 수치 설정과 temperature를 compute identity에
반영한다. 값을 바꿔 실패를 숨기지 않는다.

## I01/A03 연결에 필요한 항목

원격 기준 `9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`에는 `rz-contracts`,
`rz-position`, 루트 Cargo와 CI가 없다. 아래 항목을 총괄의 단일 계약으로 받는다.

1. evaluator trait 및 논리 요청/물리 실행 분리, immutable state/legal view 접근.
2. request/game/root/model/encoding 식별과 deadline 단위, 취소·완료 전달 접점.
3. backend/numerical/resource 오류에 원시 실패 code·stage를 대응하는 방법.
4. model/encoding descriptor, output admissibility, CPU/mock 기본 feature와 의존성.

D의 초기 PR #4, `98a6b4f17555184a99ec65f634801a24fb1c8a22`의 `Backend` 접점도
읽기 전용 확인했다. 그 `poll(Lease)`의 Ready는 물리 완료와 입력별 tagged result를
뜻한다. 따라서 mock의 단순 Callback을 Ready로 취급할 수 없고 DeviceCompleted 및
필요 출력이 확인될 때까지 Lease가 pin을 보유해야 한다. 실제 D adapter·scheduler
연결 검사는 그 구현과 I01 계약이 게시된 뒤 수행한다.

현재 public 타입은 C 내부 backend 도구용이다. 총괄의 계약 revision을 발행하거나
제품 evaluator 접점을 대체한 상태가 아니다. 공통 계약이 생기면 얇은 adapter를
이 crate에 추가하고 B/D 소비자와 함께 검증한다.

## 개발 검사

루트 workspace 게시 전에는 crate별 manifest로 검사할 수 있다. `[workspace]`를
중첩 선언하지 않아 I01에서 그대로 workspace member로 등록할 수 있다. edition 2021은
이 crate의 현재 선언이며 최종 toolchain/MSRV·workspace 공통 설정은 총괄이 정한다.

```sh
cargo test --manifest-path crates/rz-eval/Cargo.toml
cargo clippy --manifest-path crates/rz-eval/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path crates/rz-eval/Cargo.toml -- --check
```

임시 standalone Cargo.lock은 추적하지 않는다. C 소유 `rz-encoding` path dependency만
추가했으며 외부 dependency는 없다. root
lockfile 도입은 총괄 소유다. build·toolchain·원시 결과는 저장소 밖 작업 전용 output
root에 둔다. 클라우드 작업 동안 보존하고 종료 전 재현 명령·검토된 결과는 PR에,
회수할 원시 근거는 별도 artifact로 인계한다. 자동 삭제는 하지 않는다.

이 단계에서는 실제 가중치 로드·CPU 신경망 parity·GPU 실행을 수행하지 않았다.
