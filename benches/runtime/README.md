# D02 CPU/mock trace 재생

[cpu_trace 예제](../../crates/rz-runtime/examples/cpu_trace.rs)는 실제 `Scheduler`와
`rz-telemetry::profile::TraceCollector`를 연결하는 유한한 CPU/mock 재현 진입점이다.
Scheduler의 준비·대기·dispatch·물리 완료·출력 검증 사건과 소비자의 backup·출력
구간을 각각 기록하며, 요청 시작부터 출력까지의 종단 구간도 따로 기록한다.
단계 시간을 더해 전체 시간이나 처리량을 계산하지 않는다.

예제의 숫자 ID, 입력 값, backup checksum은 독립 fixture다. 체스 상태·신경망
가중치·탐색 방문 통계·공통 계약을 대신하지 않는다. `cpu-mock` provider는 실제
CPU 정수 계산 결과와 입력 `Arc`를 opaque lease에 보관하고, **fixture 시계의
3 tick이 지난 뒤** 물리 완료를 알린다. 한 tick은 1µs다. 단계 지연과
`fixture_*_rate`는 이 결정적 시계의 결과다. 실제 CPU 작업과 수명 경계를
검사하지만 실제 신경망의 CPU/GPU 지연·전송·VRAM·강도 근거는 아니다.
`actual_wall`은 시나리오 전체 재생의 `std::time::Instant` 경과 시간이며
`actual_wall_consumed_rate`는 이 벽시계로 나눈 실제 성공 소비 수다.
`actual_consumed_requests`는 observation transport와 독립적으로 성공 receipt를
받아 checksum backup을 한 횟수다. event 손실 때도 이 fixture 소비 수와 불완전한
collector의 `consumed_requests`를 구분한다.

## 재현 명령과 상한

저장소 루트에서 출력 루트를 저장소 밖의 작업 전용 디렉터리로 지정한다.
원시 TSV와 빌드 생성물은 그 루트에 두고 인수에 필요한 보고서는 작업 종료 전에
회수한다. 아래 명령은 해당 루트를 보존하며 자동 삭제하지 않는다. CI에서는
runner 임시 루트를 사용할 수 있다.

```sh
: "${RZ_D_OUTPUT_ROOT:?저장소 밖의 작업 전용 출력 루트를 지정하세요}"
mkdir -p "$RZ_D_OUTPUT_ROOT/reports" "$RZ_D_OUTPUT_ROOT/build"
CARGO_TARGET_DIR="$RZ_D_OUTPUT_ROOT/build" \
  cargo run --locked --offline --release \
  --manifest-path crates/rz-runtime/Cargo.toml --example cpu_trace -- \
  --scenario all --requests 32 --sample-capacity 512 \
  > "$RZ_D_OUTPUT_ROOT/reports/cpu-trace-all.tsv"
```

선택 인자는 `--scenario all|cold|warm|parent-child|sibling|transposition|eviction|long`,
`--requests 1..65536`, `--sample-capacity 0..65536`이다. 기본값은 각각 `all`,
32, 512다. 요청 수는 **시나리오마다** 적용되며 각 시나리오는 새 scheduler와
collector로 시작한다. 알 수 없는 인자·음수·범위를 넘는 값은 실행 전 오류다.
현재 Scheduler 생성자는 같은 값으로 계측 sample ring과 observation event ring을
제한하므로 `--sample-capacity`는 두 상한에 모두 적용된다.

| 항목 | 고정 상한 |
| --- | --- |
| 예약 요청 / batch item / 물리 실행 | 8 / 4 / 2 |
| 요청 host 예약 / batch 추가 host 예약 | 512 / 256 bytes |
| 전체 host 예약 | 4608 bytes |
| device / pinned 예약 | 0 / 0 bytes |
| batch wait / queue age / deadline reserve | 2 / 100 / 2 fixture µs |
| 요청 deadline | 준비 직후부터 1000 fixture µs |
| owner step 수 | `requests * 16 + 64` / 시나리오 |
| 벽시계 중단 | 10초 / 시나리오 |
| collector active 요청 / 실행 | 8 / 2 |

예약 수치는 입력·결과·batch overhead의 scheduler ledger이며 실제 process RSS가
아니다. provider·모델·메타데이터·계측 ring·보고서 정렬의 메모리는 별도다.
관측 transport와 collector의 저장량은 위 유한 설정으로 제한된다. 모든 정상
인수는 `try_recv`로 수행하므로 receipt 대기는 blocking하지 않는다. step·벽시계
상한 초과나 예상하지 않은 terminal 결과는 오류 exit로 남긴다.

## 일곱 입력 관계

| 시나리오 | 독립 입력 fixture 관계 |
| --- | --- |
| cold | 서로 다른 정수 입력 |
| warm | 두 입력을 반복하는 label; warm latent state 없음 |
| parent-child | 연속 입력을 부모→자식 관계로 표시 |
| sibling | 부모 label마다 네 입력, 인접 호환 batch key 적용 |
| transposition | 네 입력을 재방문; 모든 요청을 다시 실행 |
| eviction | 네 다른 label 뒤 원래 입력 재방문; cache 없음 |
| long | 유한한 연속 입력 trace; 누적 latent 갱신 없음 |

이 구분은 계측 도구의 trace class 분리를 검증한다. 실제 exact cache·dedup·
warm-state·eviction 정책의 동작이나 이득을 보고하지 않는다. 시나리오마다 첫
물리 실행에 속한 한 요청을 실행 중 취소하며, 그 요청의 늦은 물리 결과는 소비하지
않는다. 첫 요청이 살아 있을 때 같은 ID로 제출한 독립 시도는 중복 거부를
확인하고, 각 receipt 수신 직전 예약이 남아 있는지도 확인한다. 마지막에는 모든
receipt를 읽고 물리 실행을 drain하여 최종 세 예약 영역이 0인지 검사한다.

## TSV 해석

출력 첫 두 줄은 실행 종류·시계·유한 설정의 주석이고, 나머지는
`scenario`, `field`, `value`, `unit`의 네 열이다. 필드 이름과 시나리오 순서는
고정이다. 요청·실행·미소비 물리 항목은 서로 다른 카운터다. 각 단계는
`stage.<name>.total|retained|dropped|p50|p95|p99`를 제공한다. 백분위는 남아 있는
표본의 nearest rank이며 해당 시나리오와 `cpu-mock` backend의 표본이다.
전송은 GPU provider hook이 없어 `total=0`, 백분위 `na`로 남는다. missing 값을
0 지연으로 해석하지 않는다.

`physical_items_completed`는 완료된 물리 batch의 항목 수,
`consumed_requests`는 실제 successful receipt의 backup 소비 수다.
`unused_items_finalized`는 물리 완료 뒤 더 이상 소비하지 않을 항목 수이며 첫 실행
중 취소한 요청을 포함한다. `completed_unconsumed_items`는 아직 소비 가능성을
종료하지 않은 완료 항목이다. 정상 기본 실행의 최종값은 0이다. 한 항목짜리
시나리오는 실행 중 취소한 한 요청만 있으므로 successful consume이 0이어도 정상이다.

`reserved_peak.*`는 domain별 reservation ledger의 정확한 high-water mark다.
collector에 scheduler의 peak와 현재 값을 각각 전달하므로 receipt가 외부에서
해제되는 순간을 owner observation이 놓쳐도 peak를 유지한다. 서로 다른 domain의
peak가 동시에 발생했다는 의미나 실제 장치 사용량을 주장하지 않는다.

`completeness.*`는 event 손실·표본 eviction·잘못된 관측·active tracking·카운터
포화 등을 표시한다. `--sample-capacity 0`도 실행은 끝나지만 모든 runtime 사건이
생략되고 collector 관측이 불완전하므로 stage/처리량을 정상 전체 profile로
받아들이면 안 된다. 작은 capacity 역시 event transport 손실과 표본 eviction을
명시한다. 이 경우 runtime 결과 검사는 계속되지만 재구성할 수 없는 계측 사건은
추정해 보충하지 않는다.

collector는 최근 종료한 ID만 유한한 tombstone ring으로 기억한다.
`request_ids_evicted`나 `execution_ids_evicted`가 양수이면 전체 과거 ID에 대한
중복 탐지 근거는 없다. 예제는 새 숫자 ID만 사용하며 실행 도중의 의도적 중복
시도는 scheduler에서 거부된다. collector의 `caller_ids_must_be_unique`는 이
호출자 전제다. ledger 전체와 profile 완료 상태는 함께 보고하되 observation
손실을 실행 성공으로 감추지 않는다.

event 손실이 생기면 예제의 요청↔실행 관계 재구성 metadata를 비워 놓고 불완전
관측을 보고한다. 물리 완료 사건이 사라진 실행을 무기한 추적하지 않으며, 보조
execution map도 2개로 제한한다. 이 metadata 정리는 실제 scheduler의 lease나
reservation을 해제하지 않는다.

예제 검사 진입점은 `cargo test --locked --offline --manifest-path
crates/rz-runtime/Cargo.toml --examples`다. 네 검사는 일곱 시나리오의 전체 단계·
성공 소비·취소·늦은 미소비 완료, zero capacity의 불완전 계측, 단일 취소 요청,
2048 요청 trace의 capacity 1/2/4/16 유실과 보조 execution map의 peak 상한을
확인한다. TSV의 `auxiliary.peak_executions`도 같은 high-water mark를 제공한다.

이 진입점은 D02 내부 CPU/mock 도구의 baseline이다. 같은 weights·search 의미·
자원의 variant 대조나 D03 성능 개선 결과는 포함하지 않는다. 실제 B/C 소비자와
목표 GPU의 전송·완료·VRAM·전체 탐색 경로, 신경망 오차와 정식 대국은 별도 인수한다.
