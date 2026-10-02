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

## I01/A03 연결에 필요한 항목

원격 기준 `9f0bc598f6b2d8f863fd46af6a4fd73bfef1f0b8`에는 `rz-contracts`,
`rz-position`, 루트 Cargo와 CI가 없다. 아래 항목을 총괄의 단일 계약으로 받는다.

1. evaluator trait 및 논리 요청/물리 실행 분리, immutable state/legal view 접근.
2. request/game/root/model/encoding 식별과 deadline 단위, 취소·완료 전달 접점.
3. backend/numerical/resource 오류에 원시 실패 code·stage를 대응하는 방법.
4. model/encoding descriptor, output admissibility, CPU/mock 기본 feature와 의존성.

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

임시 standalone Cargo.lock은 추적하지 않는다. 아직 외부 dependency는 없으며 root
lockfile 도입은 총괄 소유다. build·toolchain·원시 결과는 저장소 밖 작업 전용 output
root에 둔다. 클라우드 작업 동안 보존하고 종료 전 재현 명령·검토된 결과는 PR에,
회수할 원시 근거는 별도 artifact로 인계한다. 자동 삭제는 하지 않는다.

이 단계에서는 실제 가중치 로드·CPU 신경망 parity·GPU 실행을 수행하지 않았다.
