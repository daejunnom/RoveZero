# rz-telemetry

D01 CPU/mock 런타임이 전달하는 이벤트를 수동적으로 집계합니다. 요청 상태를
제어하거나 ID·계약을 새로 정의하지 않으며 외부 의존성과 `unsafe`가 없습니다.

`Metrics::try_new(sample_capacity)`는 latency와 batch 크기 표본을 각각 최대
`sample_capacity`개 보존하는 ring을 미리 할당합니다. 두 ring의 원소 수×원소 크기를
할당 전에 모두 검사하고, `usize` overflow·`isize::MAX` byte 한계 초과는
`CapacityError::Overflow`, `try_reserve_exact`의 할당 실패는 `AllocationFailed`로
반환합니다. 용량 0은 정상입니다. `Metrics::new`는 신뢰된 제한 설정용 편의 API이며
같은 검사·할당 실패 시 panic하므로 외부 설정과 런타임 생성에는 `try_new`를 씁니다.

호출자는 representable 용량 검사와 별도로 운영 메모리 예산을 관리합니다. 두 ring
저장 공간, runtime의 요청·실행 메타데이터, snapshot의 표본 복사·분포 map 할당을
함께 고려하며 inference buffer 예산만으로 이 비용을 제한했다고 해석하지 않습니다.
`record(&mut self, Event)`는 추가 할당·정렬 없이 카운터와 표본을 갱신합니다.
thread 간 동기화는 호출자 책임입니다.

D02의 `profile::TraceCollector`는 별도 bounded tracking과 단계별 표본으로 종단
trace를 집계합니다. 요청·물리 실행·실제 소비·최종 미사용 항목을 분리하고, 겹친
단계의 합 대신 같은 domain의 wall span으로 처리량을 계산합니다. API와 관측
window·시간·메모리·GPU 경계는 [PROFILE.md](PROFILE.md), Scheduler에 연결한
재현 진입점은 [CPU trace 기록](../../benches/runtime/README.md)을 따릅니다.

`snapshot()`은 보존한 latency를 정렬해 nearest-rank P50/P95/P99를 계산하고,
보존한 batch 크기의 분포를 반환합니다. 결과에는 각 표본의 총수·보존수·유실수가
함께 들어갑니다. 유실에는 ring에서 덮어쓴 오래된 표본과 용량 0에서 생략한 표본이
포함됩니다. 이 분포·백분위는 최근 보존 표본의 값이며 전체 실행의 분포로 보고하지
않습니다. 빈 표본의 백분위는 `None`입니다. snapshot 계산은 요청 처리 밖에서 합니다.

`Finished`는 접수부터 논리 종료까지의 monotonic `Duration`을 받으며 성공·취소·
마감·세대 변경·실패를 각각 셉니다. 성공 종료는 탐색의 실제 결과 소비·backup을
입증하지 않습니다. `PhysicalCompleted`는 물리 batch 하나의 완료이며 논리 종료와
별도로 셉니다. `ReceiverDropped`는 결과 수신자 소실, `Quarantined`는 물리 완료를
확인할 수 없어 drain에서 격리한 execution 수입니다. 호출자는 각 전이를 한 번만
기록해야 합니다. 계측 자체가 중복 전이나 잘못된 상태를 숨기지 않습니다.

모든 누적 카운터는 `u64::MAX`에서 포화되고 `counter_overflow`가 영구적으로
`true`가 됩니다. 이 플래그가 켜진 결과의 카운터 관계는 정확한 집계로 해석하지 않습니다.

현재 root workspace와 공통 계약의 연결 전이라 독립 `[workspace]`로 빌드합니다.
검사 명령은 `cargo test --manifest-path crates/rz-telemetry/Cargo.toml`과
`cargo clippy --manifest-path crates/rz-telemetry/Cargo.toml --all-targets -- -D warnings`입니다.
이 crate의 검사는 CPU 계측 계약 검사이며 실제 GPU 종단 D02 또는 D03 개선 근거가
아닙니다.
