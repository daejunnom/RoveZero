# 의미 보존 메모리·실행 최적화

기준일: 2026-10-06. 조사 기준은 PR #20의 `f442c41aa6f1885d4ae06aab874420a4cd8f7062`다.
이 문서는 사용자 요청으로 일반 기법과 RoveZero 적용 가설을 분리한 설계·코드 리뷰 기준이다.
이번 문서 작성은 엔진 코드·Cargo·CI·모델·실행 옵션의 변경이나 성능 실험을 포함하지 않는다.
후속으로 배정되는 소스 변경은 이 기준과 기존 역할·계약·검증 절차를 함께 따른다.

[AGENTS](../../AGENTS.md), [CONTRIBUTING](../../CONTRIBUTING.md),
[개발 기준](../ENGINEERING-STANDARDS.md), [공통 계약](../CONTRACTS.md)이 상위 운영 기준이다.
모델·정밀도·잠재 상태·탐색 정책 연구는 별도 [모델 개선 연구](MODEL-IMPROVEMENT-ROADMAP.md)에 둔다.
기존 [성능 구현·인수 기록](PERFORMANCE-OPTIMIZATION-PLAN.md)과
[76개 후보 등록부](../CANDIDATE-REGISTER.md)는 보존하며 덮어쓰거나 다시 번호를 매기지 않는다.

## 1. 범위와 적용 원칙

목표는 포인터·Vec·락의 개수를 무조건 줄이는 것이 아니라 **불필요한 할당·복사·간접 접근·
대기와 동시 생존량을 줄여 같은 계산을 더 효율적으로 수행하는 것**이다.
문서의 `MEM-*`는 기법 식별자이며 `TASK-*`, `CARD-*`, 기존 `OPT-*`와 다른 이름공간이다.
표의 메모리·속도 효과와 CPU/GPU 적합성은 모두 조건부 설계 추론이다. RoveZero의 배속·Elo
성과나 채택 확정이 아니며, GPU 직접 적합성이 낮은 후보와 부정적 결과도 삭제하지 않는다.

기존 E/A/S 분류를 유지한다. E는 의미 보존 실행 최적화, A는 근사·평가기/모델 변경,
S는 탐색·스케줄 변경이다. 수치 실행만 달라지는 fusion/정밀도 실험도 비트 일치를 확인하지
않은 채 엄격한 E로 승격하지 않는다. 필요하면 기존 분류에 수치 변경 주석을 붙이며 공통
계약 revision이나 새 전역 분류를 이 문서에서 만들지 않는다.

| 고정할 것 | 의미 보존 검증 조건 |
|---|---|
| 상태와 평가 | 같은 완전 상태·이력·합법 수 순서·모델·인코딩·정밀도·평가값 |
| 탐색 | 같은 완료 방문 예산과 통제한 결과 전달 순서에서 선택 경로·동률·N/Q·backup 일치 |
| 수치 | 합산 순서·반올림·FMA·SIMD reduction 차이를 임의로 섞지 않음 |
| 수명·실패 | 원래 오류·거절 원자성·취소·만료·중복·세대·quarantine·물리 완료 경계 유지 |
| 자원 | 동일 상한과 유한 종료. 큰 메모리 예약·새 queue가 오류를 숨기거나 한도를 우회하지 않음 |

더 빨라져 동일 벽시계 안에서 더 많은 유효 방문을 완료하고 다른 수를 고르는 것은 정상적인
개선이다. 반대로 이전 결과가 오기 전에 다음 leaf를 골라 배치를 채우면 S 실험이다.
정확 캐시 hit를 새 NN 실행으로 세지 않으며, 새 유효 traversal이 hit를 소비할 때만
기존 계약대로 한 번 backup한다. 벽시계 timestamp와 소유권 검사는 성능 때문에 제거하지 않는다.

## 2. 채택 기준

<a id="adoption-gate"></a>
**메모리와 속도가 상충하면, 동일 작업량의 전체 실행시간 증가가 5% 이하이고 사전에 지정한
peak 메모리가 20% 이상 감소할 때만 채택한다.** 사용자 표현의 속도 증가를 이 문서에서는
속도 저하에 대응하는 실행시간 증가 허용폭으로 명시적으로 해석한다.

`T1/T0 <= 1.05`와 `P1/P0 <= 0.80`을 모두 만족해야 한다. 실행시간 5% 증가와 처리량 5%
감소는 같지 않으며, 동일 작업량에서 전자는 처리량 비율 `1/1.05` 이상에 해당한다.
이 문단이 임계값의 단일 기준이며 다른 규약은 수치를 복제하지 않고 여기를 링크한다.

2026-10-06 총괄 인수에서 사용자는 이 기준을 **PR #20의 효과 인수가 남은 E 변경과
후속 E 변경 모두**에 적용하도록 선택했다. 구현·CPU 검사 성공은 효과 인수와 다르다.
기존 실행은 정확한 source·binary·모델·feature·환경 일치를 확인한 범위에서 재사용하고,
변경 전후의 시간·peak 대조가 없으면 효과 채택은 보류한다. 이 문단으로 과거 실행의
성공·실패를 다시 쓰거나 현재 구현을 자동으로 되돌리지는 않는다.

- 시간과 메모리를 모두 악화시키지 않으면서 한쪽이 실측 개선되는 경우는 별도 Pareto 개선으로
  검토한다. 속도가 빨라지는 대신 peak가 증가하는 경우를 위 예외로 자동 승인하지 않는다.
- 실행 전 primary peak 지표·측정 구간·시간 통계·반복 횟수·불확실성 판정 방식을 잠근다.
  측정 뒤 유리한 heap/RSS/PSS/VRAM만 골라 쓰지 않는다. 다른 메모리 영역으로 비용을 옮긴
  경우 함께 보고하고 전체 자원 상한과 OOM·tail latency·deadline의 안전 조건을 유지한다.
- 고정 작업량 A/B를 같은 환경에서 교차 반복한다. 경계에 걸리거나 peak가 unknown이면 보류한다.
  표본 평균 하나나 관측 가능한 시점만 골라 승인하지 않는다. 통계 방법은 실행 manifest로 확정한다.
- 성능 문턱을 통과해도 의미·수명·오류 계약에 실패하면 채택하지 않는다. 라이브러리 도입·기본값
  활성화·모델 승격·유료 실행 승인은 별도이며, 이 기준은 해당 권한을 새로 만들지 않는다.

설명용 예: 시간 +4%/peak -25%는 상충 문턱의 후보, +6%/-30%와 +3%/-15%는 탈락이다.
이는 측정 결과가 아니며, 문턱 통과가 자동 병합을 뜻하지 않는다.

## 3. 일반 기법: RoveZero와 독립적인 비교

`Vec<T>` 자체는 고정 크기의 제어 정보를 갖고 요소 저장소가 가변이다. `[T]`·`dyn Trait`
같은 동적 크기 타입과 문제를 구분한다. 메모리 감소는 별도 표시가 없으면 가능한 효과이지
보장이 아니다. 각 표의 역효과도 후속 실험에 그대로 남긴다.[^vec][^layout]

### 3.1 레이아웃·간접 접근

| ID / 기법 | 메모리 효율 | 속도 향상 가능성과 원인 | 하락·증가 가능성과 원인 |
|---|---|---|---|
| MEM-A01 연속 배열+인덱스 | 개별 할당·포인터·단편화 감소 가능 | 순회 지역성, allocator 호출·pointer chasing 감소 | 재할당·큰 여유 capacity, 주소 무효화, 드문 랜덤 접근 |
| MEM-A02 중첩 Vec 평탄화+start/len | 작은 할당·헤더·capacity 낭비 감소 | 인접 목록 순회와 prefetch 개선 | 개별 목록 성장·삭제 시 이동/빈 공간 관리; 청크형과 비교 |
| MEM-A03 hot/cold 분리 | 전체 bytes보다 hot working set 감소 | 자주 쓰는 필드의 캐시 효율 | cold도 늘 함께 읽으면 추가 접근·인덱스 비용 |
| MEM-A04 AoS/SoA/AoSoA | padding 감소 또는 여러 배열 여유 공간 증가 | 같은 필드의 연속 처리·SIMD | 객체 전체 랜덤 조회는 AoS가 유리할 수 있음 |
| MEM-A05 작은 정수·niche·정확 bit packing | 필드 폭 감소 가능 | 같은 cache line에 더 많은 항목 | 변환·마스킹·overflow, padding 때문에 총 크기 불변 가능 |
| MEM-A06 드문 enum payload 분리 | 큰 variant가 모든 객체를 키우는 비용 감소 | 흔한 객체의 이동·스캔 축소 | 드문 경로 할당·indirection; 무할당 오류 보존과 충돌 가능 |
| MEM-A07 필드 순서·padding 점검 | 실제 ABI에서 감소 여부 확인 | 캐시·전송량 감소 | repr(C)/align 추가로 더 커질 수 있고 repr(Rust)는 배치 비보장 |
| MEM-A08 선택적 cache-line padding | 대체로 메모리 증가 | 독립 writer의 false sharing 감소 | 모든 객체 정렬은 working set·TLB 압박; 읽기 공유에는 불필요 |
| MEM-A09 정적 dispatch/선택적 monomorphization | vtable 접근 감소, 코드 크기는 증가 가능 | inlining·상수 전파 | 타입별 코드 팽창·instruction cache miss·컴파일 비용 |
| MEM-A10 flat hash table | 노드 할당 감소, 빈 슬롯·control bytes 필요 | 연속 probe와 작은 metadata | 작은 집합은 선형 검색이 더 빠름; rehash·큰 value 이동 비용 |

근거: 연속 저장·간접 접근[^indirection], Rust 레이아웃·niche[^layout][^niche],
false sharing[^false-sharing], 배열 배치[^soa], flat hash 설계[^hash].
`repr(packed)`를 일반적인 속도 기법으로 사용하지 않는다. 정렬되지 않은 참조의 안전성과
실제 접근 비용을 해결한 별도 경계가 필요하다. 캐시 라인을 모든 플랫폼에서 64바이트로
고정하지 않고 목표 ISA와 실제 측정을 기록한다.

### 3.2 스택·고정/가변 배열

| ID / 기법 | 메모리 효율 | 속도 향상 가능성과 원인 | 하락·증가 가능성과 원인 |
|---|---|---|---|
| MEM-B01 작은 지역 고정 배열 | heap 대신 stack 공간 사용 | 할당·해제 제거 | 큰 frame·재귀·다중 thread의 stack 압박, 큰 값 이동 |
| MEM-B02 ArrayVec | heap 없이 최대 N 공간 상주 | 작은 엄격 상한에서 무할당 | 과대 N의 빈 공간; 초과를 잘라내면 의미 변경 |
| MEM-B03 SmallVec | 작은 입력은 inline, 큰 입력은 heap | 짧은 길이가 대부분일 때 allocator 감소 | 큰 inline N·spill 복사·상태 분기; Vec보다 큰 객체 가능 |
| MEM-B04 Vec 사전 할당 | 성장 중 임시 할당 감소, 여유 공간 상주 | 재할당·이동 감소 | 객체마다 최대 상한 예약; 조금씩 reserve_exact 반복 |
| MEM-B05 worker별 scratch Vec | 최대 동시량만 보존 가능 | 반복 alloc/free와 tail 감소 | 한 번의 큰 입력이 capacity를 계속 잡음; bounded 보존 필요 |
| MEM-B06 고정된 결과를 Box<[T]>로 | 여유 capacity·제어 정보 축소 가능 | 고정 길이·불변 결과 관리 | 변환 재할당 가능, 다시 키울 때 복사 |
| MEM-B07 최종 위치 직접 초기화 | 임시 배열·중복 buffer 감소 | 초기화 후 덮어쓰기 제거 | MaybeUninit의 초기화·Drop·panic·부분 실패 부담 |
| MEM-B08 임시 collect 제거/출력 buffer 재사용 | 중간 container 제거 | 단일 순회·복사 축소 | 비싼 계산 중복 또는 reduction 순서 변화 가능 |

근거: Vec의 capacity·clear·변환[^vec], SmallVec[^smallvec], ArrayVec[^arrayvec],
초기화 계약[^uninit]. Inline은 stack과 동의어가 아니다. heap 객체나 future에 넣으면 해당
객체가 커진다. SmallVec N은 길이 분포·동시 개수·요소 크기·spill 비율로 고른다.
설명용으로 요소 48 bytes×inline 32×10만 객체는 요소 공간만 약 146.5 MiB다.
전체 노드에 최대 합법 수 배열을 내장하는 결정은 이 비용을 먼저 계산한다.

### 3.3 소유권·Cow·arena·pool

| ID / 기법 | 메모리 효율 | 속도 향상 가능성과 원인 | 하락·증가 가능성과 원인 |
|---|---|---|---|
| MEM-C01 borrow/소유권 이동 | payload 복제 감소 | 큰 clone·할당 제거 | 작은 view가 큰 원본 수명을 연장하거나 생산자 재사용을 막음 |
| MEM-C02 Cow | 읽기만 하면 owned 복사 없음 | 수정·소유 변환이 드문 입력 | 거의 매번 수정하면 결국 복사+분기; 원본 retention 증가 |
| MEM-C03 Rc/Arc/borrow 구별 | 불변 큰 payload 공유 | 중복 데이터 제거 | Arc 원자 참조계수·간접 접근; Rc는 thread 경계 불가 |
| MEM-C04 Arc slice/Bytes view | 여러 view가 저장소 공유 | slice 복사 제거 | 작은 slice가 큰 원본 전체를 붙잡음; 작은 복사가 더 나을 수 있음 |
| MEM-C05 bump/region arena | 개별 metadata·단편화 감소 가능 | pointer bump 할당·일괄 회수 | 조기 사망 객체 retention·정렬 틈·Drop 미실행 위험 |
| MEM-C06 수명별 arena 분리 | 짧은 객체의 장기 보존 감소 | 함께 쓰는 데이터 지역성·회수 | arena별 청크 여유·관리 증가 |
| MEM-C07 slab/객체 pool | 동일 크기 재사용 | 할당·준비 비용 감소 | 유휴 slot·초기화 누락·stale handle·ABA |
| MEM-C08 thread/크기별 pool | 지역 재사용, pool별 여유 증가 | 공유 free-list 경쟁 감소 | 한 thread에 재고 집중·cross-thread free 비용 |
| MEM-C09 streaming/조기 drop | 동시 생존량·peak 감소 | 중복 복사·메모리 압력 감소 | 재읽기·재계산·너무 작은 chunk의 호출 비용 |
| MEM-C10 generation-tag 초기화 | tag 공간 추가 | 전체 배열 clear 회피 | 접근 검사·wrap·자원 Drop 누락; 논리 초기화와 실제 회수는 다름 |

근거: Cow[^cow], Arc[^arc], 공유 byte view[^bytes], arena[^bump], slab[^slab],
allocator의 지역성[^mimalloc]. 수명별 분리·streaming·generation 초기화는 일반 설계 가설이다.

Cow는 참조계수 포인터가 아니다. eager 복사비 C, 실제 복사 필요 비율 p, 관리비 D라면
단순 비용 모델은 `D+pC`다. `(1-p)C>D`일 때 후보가 되지만 원본 retention도 포함해야 한다.
`Arc::make_mut`는 별도 clone-on-write 수단이다. 거의 항상 변하는 상태 전체에 Cow를
붙이는 것을 기본 최적화로 정하지 않는다.

Bump reset이 모든 객체의 Drop을 실행한다고 가정하지 않는다. 일반 Vec 헤더를 arena에
넣어도 요소 heap이 arena로 옮겨지지 않는다. Arc·Vec·FD·native handle을 담을 경우
소멸자 실행과 부분 초기화 실패 정리를 설계한다. arena-aware container 또는 별도
소유 목록을 검토하며, 재사용 시 live/retained/reserved bytes를 따로 센다.

### 3.4 락·atomic·통신

| ID / 기법 | 메모리 효율 | 속도 향상 가능성과 원인 | 하락·증가 가능성과 원인 |
|---|---|---|---|
| MEM-D01 single-writer 소유권 | 공유 가변 구조 단순화 | owner 내부 락 제거 | 메시지 전달 비용·owner 병목 |
| MEM-D02 임계 구역 축소 | 보통 중립, immutable handle 추가 가능 | 큰 clone·포맷·검사를 락 밖으로 | lock 밖 데이터 수명·version을 보증해야 함 |
| MEM-D03 lock sharding | 락·분할 여유 공간 증가 가능 | 독립 접근 경쟁 감소 | hot shard·다중 lock 순서·불균형 |
| MEM-D04 bounded SPSC ring | 고정 공간·항목별 무할당 | 단일 생산자/소비자의 간단한 전달 | 실제 topology 불일치·가득 찬 queue의 backpressure |
| MEM-D05 bounded MPMC queue | 동적 증가 제한, 예약 공간 상주 | 여러 producer/consumer 전달 | CAS 재시도·cache bouncing; 짧은 mutex보다 느릴 수 있음 |
| MEM-D06 thread-local 통계+집계 | thread별 복사본 증가 | 공유 atomic·false sharing 감소 | 제어값을 지연 집계하면 동작 변경; 관측 통계에 우선 적용 |
| MEM-D07 최소 atomic ordering | 보통 중립 | 불필요 ordering 제약 감소 | Relaxed로 payload publication 불가; ISA별 이득 상이 |
| MEM-D08 알림/condvar/적응형 spin | 소량 관리 공간 | 주기 sleep 지연·busy-poll 감소 | wake 비용·코어 점유·lost wakeup·deadline race |
| MEM-D09 ArcSwap/RCU/epoch 회수 | 오래된 버전 retention 가능 | 읽기 위주 reader lock 감소 | 느린 reader가 회수 지연, writer·refcount 비용 |
| MEM-D10 seqlock 계열 | 작은 version metadata | 드문 writer·작은 snapshot | 쓰기 많으면 재시도; pointer 수명과 Rust data race 위험 |

근거: SPSC/MPMC 구현 계약[^queues], atomic memory ordering[^atomics], read-mostly 공유[^rcu],
seqlock[^seqlock], mutex 차이[^mutex]. Lock-free와 wait-free는 다르며 빠름의 증명이 아니다.
SeqCst를 일괄 Relaxed로 바꾸지 않는다. payload 게시/소비의 happens-before를 먼저 적는다.
Seqlock에서 나중에 version 불일치를 확인해도 이미 일어난 비원자적 data race는 복구되지 않는다.
Mutex 교체 시 poisoning·fairness·try_lock·panic 이후 상태가 같아야 한다. parking_lot의
non-poisoning을 기존 오류 검사 제거의 근거로 삼지 않는다.

### 3.5 비동기·제로 카피·GPU

| ID / 기법 | 메모리 효율 | 속도 향상 가능성과 원인 | 하락·증가 가능성과 원인 |
|---|---|---|---|
| MEM-E01 최종 buffer 직접 기록+I/O Binding | 중간 tensor·복사본 감소 | 인코딩→재포장 중복 제거 | 입력/출력 주소·shape·수명 제약; H2D 자체는 남을 수 있음 |
| MEM-E02 pinned host+async DMA | page-locked 공간 예약 증가 | 독립 작업과 transfer overlap | 등록비·host RAM 압력·겹칠 작업 부재 |
| MEM-E03 double/triple buffering | buffer 수만큼 증가 | 생산·복사·계산·소비 overlap | 의존성 강하면 낭비, queue latency 증가 |
| MEM-E04 mapped host memory | 별도 device 사본 감소 가능 | 통합 메모리·작고 한 번 읽는 입력 | 분리형 GPU의 반복 host 접근·PCIe 지연 |
| MEM-E05 stream-ordered GPU pool | 단편화 감소, reserved VRAM 유지 | 반복 malloc/free·전역 동기화 감소 | stream 의존성·retention·재사용 대기 |
| MEM-E06 tensor 수명 기반 workspace | 동시 미사용 buffer를 겹쳐 저장 | working set·할당 감소 | 잘못된 alias/liveness는 손상, 동적 shape 최대치 낭비 |
| MEM-E07 고정 buffer+CUDA Graph | graph·buffer retention 가능 | 반복 launch·CPU orchestration 감소 | capture 비용·shape/address 고정·graph 종류 증가 |

근거: ORT I/O Binding·CUDA 실행 계약[^ort-io][^ort-cuda], CUDA 메모리·전송·pool[^cuda][^cuda-pool],
tensor memory planning[^planning]. CPU 무복사, 비동기 DMA, GPU mapped host 접근을 구분한다.
Rust Pin은 CUDA page-locked memory가 아니다.[^pin] Host Acquire/Release만으로 device 완료를
증명할 수 없으며 event/stream/runtime fence가 필요하다.
물리 완료, payload 소비 종료, 모든 view 수명 종료를 확인한 뒤 buffer를 pool로 반환한다.
논리 취소·root 변경만으로 재사용하지 않는다. 완료 순서를 유지하더라도 미리 leaf를 골랐다면
기존 순차 탐색과 같지 않을 수 있다. E 실험은 새 탐색 선행 선택 없이 가능한 overlap부터 한다.

### 3.6 allocator 선택

| ID / 선택 | 메모리 효율 | 속도 향상 가능성과 원인 | 하락·증가 가능성과 원인 |
|---|---|---|---|
| MEM-F01 System 기준선 | 플랫폼에 따라 다름 | 기존 workload에 충분히 최적일 수 있음 | 작은 할당·다중 thread의 경쟁·단편화 |
| MEM-F02 mimalloc | page/free-list 분리·단편화 감소 후보 | 지역 할당·cross-thread free 경로 | purge 비용·cache retention; workload 의존 |
| MEM-F03 jemalloc | arena/size class/decay 조절 | 다중 thread 경쟁 감소 | arena·tcache 유휴 공간, OS 반환 지연 |
| MEM-F04 TCMalloc | per-CPU/thread cache 유지 | 작은 할당의 지역 fast path | CPU/thread 수·cache 크기에 따른 retention |

공식 설계·설정은 각 원문을 따른다.[^mimalloc][^jemalloc][^tcmalloc]
Rust global allocator를 바꿔도 ORT native heap·CUDA allocator까지 자동으로 바뀌지 않는다.
FFI에서는 원래 allocator의 대응 해제 함수를 유지한다. custom GlobalAlloc 자체 개발보다
검증된 구현의 한 변수 비교를 우선하며, 실패/OOM·alignment·thread·재진입 계약을 점검한다.[^global]
라이브러리 버전·feature·MSRV·플랫폼·라이선스와 source pin은 도입 PR에서 잠근다.

### 3.7 OS·컴파일러·기타

| ID / 기법 | 메모리 효율 | 속도 향상 가능성과 원인 | 하락·증가 가능성과 원인 |
|---|---|---|---|
| MEM-G01 readonly mmap/공유 | 파일 read buffer 중복·공유 페이지 절약 | 큰 복사·초기 로딩 축소 | page fault·random I/O·파일 변경/잘림·pin 검증 |
| MEM-G02 huge pages | page table 감소, 내부 낭비 가능 | TLB miss 감소 | compaction·큰 페이지 준비 지연·작은 객체 낭비 |
| MEM-G03 NUMA first-touch/affinity | 복제 전략에 따라 증가 | 원격 메모리 접근 감소 | imbalance·이동 제한·단일 소켓의 작은 이득 |
| MEM-G04 blocking/prefetch/접근 순서 | 유효 working set 감소 | cache reuse·메모리 지연 숨김 | 잘못된 prefetch의 대역폭 낭비; 의미 있는 순서 변경 금지 |
| MEM-G05 PGO/LTO/선택적 inline | 코드 크기 증감 모두 가능 | 실제 hot path 배치·최적화 | profile 과적합·instruction cache·compile 비용 |
| MEM-G06 정확 memoization/interning | 중복 제거 또는 cache overhead | 반복 계산·metadata 생성 축소 | miss·eviction·lookup·장기 retention |

근거: mmap[^mmap], huge pages/NUMA[^os], PGO·profile 설정[^pgo], cache/table 구조[^hash].
인덱스·배치·주소 순으로 재배열하더라도 합법 수 순서와 tie rule·FP reduction 순서는 유지한다.
`panic=abort`·unchecked 접근·검증 제거를 무관한 컴파일 속도 옵션처럼 섞지 않는다.

## 4. RoveZero 적용 가설

아래 소스는 조사 SHA에 고정한 근거다. 링크의 구현 존재를 최적화 활성화·GPU 인수·실측 이득으로
확대하지 않는다. 후속 코드 변경 때는 최신 head의 실제 feature와 실행 profile을 다시 확인한다.

| 항목 / 담당 | 현재 확인 및 적용 제안 | 보존해야 할 것 | CPU / GPU 직접 / GPU 시스템 적합성 추론 |
|---|---|---|---|
| tree / B | 이미 Vec<Node>+child index다. Expanded(Vec<Edge>)의 작은 할당을 평탄한 edge 저장소+범위와 비교.[^rz-tree] | legal 순서·tie·N/Q·local index overflow; 주소/borrow 안정성 | 높음 / 낮음 / 조건부 |
| pending·backup / B | owner별 scratch, pending별 독립 저장, 작은 경로의 SmallVec 비교 | 검증·산술·공간 확보 뒤 atomic commit; 부분 backup 금지 | 높음 / 낮음 / 조건부 |
| 상태·입력 / A/C/B | 기존 state-cache·prepared-input·history 최적화와 겹치는지 먼저 확인.[^rz-prepared] | 동일 이력·새 요청 identity·검증된 immutable 상태 | 높음 / 조건부 / 조건부 |
| raw cache / C/D | Arc<Mutex<Store>>의 큰 clone·정확 비교·LRU 관리 범위를 측정. immutable payload handle과 인덱스 분리.[^rz-cache] | staged/live·profile·exact input 확인·newgame reset·거절 결과 승격 금지 | 높음 / 낮음 / 조건부 |
| 출력 타입 / C | 미검증 RawOutput의 잘못된 길이 표현은 유지하고 검증 후 배열/slice/pool로 압축.[^rz-output] | missing/NaN/shape 오류를 0 padding으로 숨기지 않음 | 높음 / 조건부 / 조건부 |
| native owner / C/D | Mutex 안의 실행·shutdown·diagnostic 책임을 분해 계측한 뒤 임계 구역 축소.[^rz-worker] | poison·원래 오류·한 번만 회수·물리 완료 | 높음 / 낮음 / 조건부 |
| 진단 slot / C | 큰 오류 enum을 곧바로 Box로 바꾸지 말고 필요 시 preallocated slab+index 비교 | 실패 기록 순간 무할당 성질과 원래 오류 보존 | 높음 / 낮음 / 조건부 |
| I/O / C/D | 현재 fixed Binding도 동기 copy_into와 fence를 사용한다. 최종 host buffer 직접 기록부터 비교.[^rz-io] | GPU가 읽는 동안 변경·반환 금지; output 소비 수명 | 조건부 / 높음 / 높음 |
| arena·pool / B/C/D | root/scratch와 in-flight lease를 별도 수명으로 구성 | root reset만으로 GPU 입력 회수 금지; Drop와 ABA 검사 | 높음 / 조건부 / 조건부 |
| allocator·PGO / 총괄+D | 제품 소스의 할당 제거 뒤 System과 별도 feature/빌드 비교 | Cargo/CI 소유 경계·native allocator 구분·동일 실패 계약 | 높음 / 낮음 / 조건부 |

모든 등급은 **추론/미측정**이며 낮은 등급은 후보 삭제 사유가 아니다.
`rz-eval`의 기존 unsafe 제한을 성능을 이유로 일괄 해제하지 않는다. 불가피한 unsafe/FFI는
소유 담당과 총괄이 작은 경계·불변식·실패·검증 범위를 따로 검토한다.[^rz-output]

### 4.1 수명별 저장소 제안

| 저장소 | 대상 | 반환 조건 |
|---|---|---|
| 동기 scratch | 선택 점수·임시 path·backup 준비 | 동기 연산 및 모든 borrow 종료 |
| root arena | node·edge·root 전용 metadata | root consumer 종료; 참조된 요청 자료는 별도 수명 |
| in-flight pool | PreparedBatch·입출력·request context | 물리 완료+모든 소비/view 종료 |
| process 저장소 | session·model descriptor·runtime pin | 최종 drain·정상 종료 계약 |
| quarantine/진단 | 미완료 lease·원래 오류 | 일반 arena reset과 독립적인 보존·종료 정책 |

### 4.2 이미 구현한 작업을 다시 신규로 주장하지 않기

PR #20은 원본 streaming 검증, session commit 뒤 owned ONNX buffer 조기 해제,
검증된 runtime 공유 저장과 private snapshot cache hint를 이미 설명한다.[^pr20]
이를 이 문서에서 재구현 완료·신규 성능 이득으로 기록하지 않는다. 공유 library cache,
raw evaluation cache, state cache, Transformer KV cache를 구분한다.
PR #18의 OPT-01~12가 존재해도 compile feature와 runtime flag가 모두 실제로 활성화됐는지
확인한다. 기존 V1~V5 receipt의 한도를 조용히 풀거나 unsupported 조합을 성공으로 만들지 않는다.

### 4.3 A/A 관측과 변경 전후 A/B

PR #20의 V5 B1/B1 A/A는 같은 바이너리·visits·전체 시계에서 현재 준비 시간,
실행 편차와 자원 상태를 확인한다. 두 엔진의 흑백 교환은 변경 전후 비교가 아니므로
이 결과만으로 cache hint나 buffer 변경의 시간 개선률·peak 감소율을 판정하지 않는다.
효과 채택에는 같은 완전 입력·고정 작업량의 기준/변경 A/B가 별도로 필요하다.
startup과 warm 구간, primary peak와 반복·불확실성을 실행 전에 지정한다.
시계 대국에서 서로 다른 방문 수가 발생한 결과나 서로 다른 source의 과거 peak를
그 고정 작업량 대조로 대체하지 않는다. peak가 unknown이거나 비교 기준이 없으면 보류한다.

### 4.4 첫 고정 작업량 GPU 대조: OPT-09 버퍼 재사용 보류

2026-10-06, source `ef86138c13b2188f7ee1953eb8bed60b5fec504f`의 독립
`inference_bench`에서 기존 `experimental-io-buffers` 한 변수만 비교했다.
기본 B1/2/4/8/16 sweep은 유지하고, 명시적인 `--b1-buffers=baseline|reuse`
모드만 B1 고정 작업량 대조에 사용한다. 입력·BT4-it332·FP32·HistoryFill No·TF32 off,
단일 물리 worker, warm3/timed20은 같고 cache·dedup·I/O Binding·CUDA Graph는 끈다.
RTX 4050 Laptop 6GB·WSL Ubuntu에서 CPU 2개, `memory.high=6GiB`,
`memory.max=12GiB`를 양쪽에 동일 적용했다. 원시 자료·바이너리 해시·실행 명세는 Git 밖에 보존했다.

측정 전에 전체 프로세스 실행·물리 완료·결과 회수 시간 T와, backend 종료 후 한 번
읽은 프로세스 `VmHWM` peak RSS P를 primary로 지정했다. P는 모델 준비·warmup·측정·
종료를 포함하며 VRAM이나 Windows 전체 커밋이 아니다. 순서는 A1/B1/B2/A2/A3/B3다.

| 쌍 | 기준 전체 시간 (s) | 재사용 전체 시간 (s) | 기준 peak RSS (MiB) | 재사용 peak RSS (MiB) | T1/T0 | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 18.640 | 19.667 | 1904.117 | 1818.348 | 1.0551 | 0.9550 |
| 2 | 13.893 | 18.214 | 1822.012 | 1818.109 | 1.3111 | 0.9979 |
| 3 | 21.594 | 23.900 | 1820.590 | 1818.141 | 1.1068 | 0.9987 |

독립 참조 수치 검사와 32회 가변 B1의 보존 출력 불변 검사는 통과했다. 여섯 대조
실행의 23회 출력 해시는 동일하고 물리 완료·정상 종료를 확인했다. CUDA/OOM 오류,
강제 정리와 잔류 자식은 없었다. 최초 등록 오류로 이전 benchmark 바이너리가 새 인자를
거부한 시도는 모델/GPU 실행 전 실패로 별도 보존하고 측정 표에서 제외했다.
올바른 build target과 복사본의 digest 및 인자 경계를 확인한 뒤 전 대조를 시작했다.

**채택 보류:** 세 쌍에서 관측한 최대 시간 비율 1.3111과 peak 비율 0.9987은
시간 1.05·peak 0.80 문턱을 충족하지 않는다. 일관된 Pareto 개선도 관측하지 못했다.
모델·runtime 준비 시간과 RSS 편차가 있으므로 이 비율을 버퍼 재사용의 확정적인
인과 효과로 일반화하지 않는다. 기본 off를 유지하고, 실제 VRAM 최대치·Windows
전체 커밋은 unknown으로 둔다. 이 결과가 streaming 검증·모델 버퍼 조기 해제·runtime
공유·cache hint의 변경 전후 효과를 대신 검증하지 않는다.

### 4.5 private snapshot I/O 대조: 시간 개선 관측, 채택 보류

2026-10-06, source `a997f6cc1914b4acd0fbccebfa1750b5181bffc2`에서 MEM-C09의
작은 streaming chunk 호출 비용을 한 변수로 검사했다. arena의
`experimental-snapshot-io`는 [복사·재검증 경로](../../crates/rz-arena/src/native_launch.rs)의
블록만 기본 16KiB에서 64KiB로 바꾼다. 원본 전체 검증·복사 중 SHA·복사본 전체
재검증·바이트 예산·writer 종료·readonly pin·distinct inode 조건은 유지한다.
기본 feature는 바꾸지 않았으며 API·V1~V5 receipt·계약 revision도 그대로다.

저장소 밖의 얇은 probe가 제품 `prepare_native_cuda_launch`를 호출했다. 동일한 기존
V4 lock의 BT4·가중치·ORT/CUDA 파일·엔진 binary·opening 등 27개,
4,099,789,301bytes를 여섯 실행에서 모두 다시 검증하고 private copy로 준비했다.
lock 안의 엔진 source는 기존 `2e91cb9`이며 이 실험에서는 실행하지 않았다.
모델 로딩·추론·prelaunch/postgame recheck·대국은 측정 범위에 포함되지 않는다.
공통 dependency version/checksum과 helper source·Cargo.lock·두 binary digest를 고정했다.
WSL Ubuntu·CPU2·`memory.high=6GiB`·`memory.max=12GiB`·swap0,
실행당 240초·정리 30초·전체 1,650초·A1/B1/B2/A2/A3/B3 순서를 사전에 지정했다.

Primary T는 자식 시작부터 정상 exit·pipe drain·영수증 검증·최종 cgroup 수집까지다.
Primary P는 private file cache를 포함한 해당 cgroup의 `memory.peak`를 종료 후 한 번 읽은
값이다. 별도 `VmHWM` 관측으로 P를 대체하지 않았다. 원본은 기존 NTFS/ext4에 있고
새 사본은 ext4에 두었다. 전역 cache 비우기·원본 cache hint는 없으며 cold cache를 주장하지 않는다.

| 쌍 | 16KiB 전체 T (s) | 64KiB 전체 T (s) | 16KiB P (MiB) | 64KiB P (MiB) | T1/T0 | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 34.309 | 27.978 | 6144.250 | 4027.945 | 0.815469 | 0.655563 |
| 2 | 41.968 | 27.833 | 4028.004 | 4027.992 | 0.663209 | 0.999997 |
| 3 | 35.293 | 28.824 | 4027.922 | 4027.887 | 0.816701 | 0.999991 |

복사·복사본 재검증 구간은 각각 22.941/27.675/25.796초에서
19.291/18.177/18.370초로 짧아졌다. 변경하지 않은 원본 검증도 11.220/14.109/9.341초와
8.486/9.570/10.270초로 변동했다. 첫 쌍의 큰 P 감소는 나머지 쌍에서 재현되지 않았고,
file cache의 상태·cgroup 귀속·실행 순서 영향은 분리하지 못했다. 각 실행의 종료 시점
`memory.stat`과 RSS를 함께 보존했다. probe의 `VmHWM`은 모두 2.875MiB였지만 이는
NN을 로딩하지 않은 준비 단계의 관측이며 전체 대국의 Rust/ORT 메모리 증거가 아니다.

**채택 보류:** 시간 비율은 모두 1 미만이지만 두 쌍의 P 비율은 0.80 문턱을 충족하지
않았다. 별도 Pareto 검토는 모든 쌍의 T/P 비악화와 각 T 감소가 기준 T 범위보다 클 것을
사전에 요구했다. 기준 범위 7.659초에 비해 감소는 6.331/14.134/6.469초라 이 조건도
충족하지 못했다. 관측한 시간 개선을 보존하며 기본 16KiB·실험 off를 유지한다.
이 작은 준비 대조로 GPU 종단 지연·Windows 전체 commit·강도 개선을 판정하지 않는다.

여섯 실행의 모든 해시·소유권 검사가 통과했고 OOM·강제 종료·잔류 process는 없었다.
블록 경계를 넘는 끝부분 변조·증가·잘림, 기본/실험 검사와 strict Clippy도 통과했다.
초기 외부 하네스의 HOME 누락과 출력 pipe 연결 오류는 실패 영수증으로 별도 보존했다.
입력·binary·CPU·메모리 한도는 유지한 새 등록에서 위 여섯 실행을 완료했다. 성공한 실행의
재생성 private inputs만 pin 종료·자식 부재·정확한 경로를 확인한 뒤 회수했고 원본과 실패
사본·모든 결과는 보존했다. 원시 자료·명세·분석은 Git 밖 논리 경로
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-snapshot-io-20261006-v2/`에 둔다.

### 4.6 private snapshot 파일별 cache hint: peak 감소 관측, 채택 보류

2026-10-06, source `eaa41f36bc502a9e4b1af5b2c9d2781ca923a05e`에서 arena의
`experimental-snapshot-reclaim`을 기본 off로 추가했다. 원본 전체 검증·private copy의
전체 해시·writer 종료·readonly·distinct inode 조건이 끝난 파일에만 기존 Linux
`DONTNEED`를 적용한다. 8MiB 이상이 대상이며 원본·공유 runtime cache·GPU buffer는
대상이 아니다. 제품 prelaunch/postcheck에서는 파일의 전체 해시와 held/named inode
검증 뒤 적용하며 postcheck는 owned cleanup이 확인된 경우에만 적용한다. 기존 전체
파일의 before-launch·both-ready·after-postcheck 힌트는 유지한다. 실패는 경고·상태로
보존하며 대국 성공/실패를 변경하지 않는다. 힌트 성공은 실제 회수를 보장하지 않는다.[^fadvise]
API·V1~V5 receipt 필드·계약 revision·기본 feature를 변경하지 않았다.

앞선 chunk 실험과 분리하여 기본 **16KiB**를 고정했다. 같은 V4 lock의 27개,
4,099,789,301bytes를 제품 `prepare_native_cuda_launch`로 준비하고, 독립적인 전체
사본 재읽기/해시/readonly/inode oracle → 기존 B1 CUDA 추론 example → 물리 완료·자식
종료 뒤 독립 전체 사본 재검증을 수행했다. 제품 preparation의 파일별 힌트는 실제 코드로
실행했다. 이후 oracle의 파일별 힌트는 같은 정책의 독립 적용이며 실제 native runner의
prelaunch/postcheck를 직접 호출한 결과가 아니다. UCI·PUCT·readiness·대국은 실행하지 않았다.
따라서 실제 두 엔진 대국이나 전체 native runner의 효과 인수로 확대하지 않는다.

추론 source `ef86138c13b2188f7ee1953eb8bed60b5fec504f`와 binary digest를 양쪽에
동일하게 고정했고 현재 C backend·인코딩·계약·Cargo.lock이 동일함을 확인했다.
BT4·FP32·TF32 off·HistoryFill No·B1·warm3/timed20이며 buffer 재사용·notify·cache·dedup·
I/O Binding·CUDA Graph를 끈다. RTX 4050 Laptop 6GB·WSL Ubuntu·CPU2,
`memory.high=6GiB`·`memory.max=12GiB`·swap0, 실행당 240초·정리 30초·전체 2,160초를
사전 등록했다. A0/B0의 전체 준비 실행을 별도 보존한 뒤 A1/B1/B2/A2/A3/B3를 실행했다.
전역 cache 비우기·원본 hint는 없으며 warm 상태의 대조로 cold startup을 주장하지 않는다.

Primary T는 자식 시작부터 exit·pipe drain·결과 검증·최종 cgroup 수집까지,
Primary P는 file cache와 자식 Rust/ORT 메모리를 포함한 전체 `memory.peak`다.
종료 뒤 한 번 읽은 P를 phase `memory.current`나 helper/추론 process의 `VmHWM`으로
대체하지 않았다. phase별 file/anon·시각과 두 process의 HWM도 별도로 보존했다.

| 쌍 | 기준 T (s) | 파일별 hint T (s) | 기준 P (MiB) | 파일별 hint P (MiB) | T1/T0 | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 60.163 | 64.354 | 4028.332 | 2766.004 | 1.069656 | 0.686638 |
| 2 | 66.626 | 49.260 | 4028.266 | 2769.734 | 0.739350 | 0.687575 |
| 3 | 87.194 | 88.545 | 4028.148 | 2765.941 | 1.015499 | 0.686653 |

**HOLD·기본 off 유지:** P는 세 쌍 모두 약 31.3% 줄어 0.80 문턱을 충족했지만 첫 T가
1.05를 초과했다. 사전 Pareto 조건의 모든 T/P 비악화도 충족하지 못했다. 기준 T 범위는
27.031초이며, 변경하지 않은 원본 검증도 11.013/10.274/16.578초와
17.417/12.846/16.895초로 변동했다. 복사·사본 검증은
24.759/30.546/46.433초와 25.852/18.871/50.332초, B1 로딩·추론·exit는
14.441/16.429/16.781초와 12.836/9.966/11.031초다. 메모리 감소 관측을 유지하되
시간 차이 전체를 hint의 인과 효과로 환산하지 않는다. 추론 process HWM은 양쪽 모두
약 1.77~1.78GiB로, 이번 관측은 ORT heap/VRAM 감소나 Windows 전체 commit 해결의
증거가 아니다. VRAM peak와 Windows 전체 commit peak는 unknown이다.

8회의 모든 private 해시·소유권·고정 시작 입력/raw output SHA·B1 물리 완료 검사가
통과했고 실제 CUDA 실행 node는 각각 687개다. 각 실행의 명시적 warm3/timed20을
확인했으며 기존 전체 독립 수치 suite를 새 실행으로 보고하지 않는다. 제품 preparation에서
각 실험의 대상 15개 모두 힌트 성공으로 기록했고 힌트 오류·OOM·강제 종료·잔류 process는
0이다. 준비 A0는 약 6GiB peak와 high 이벤트 28,907을 보였으며 비교 6회는 high0이다.
기본/reclaim-only arena 검사는 각각 128개, all-feature는 159개 통과했고 helper 14개는
의도된 ignored 상태다. readonly pin의 bytes/inode/cursor·작은 파일 제외·힌트 오류 보존·
미확인 cleanup 제외 검사와 strict Clippy를 포함한다. `eaa41f3`의 Ubuntu·Windows·
bindings CPU CI도 모두 SUCCESS로 확인했다. 성공한 비활성 재생성 사본만 pin/자식/물리
완료와 정확한 경로를 확인한 뒤 회수했고 원본·원시 로그·명세·분석은 Git 밖 논리 경로
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-snapshot-reclaim-20261006/`에 보존했다.

### 4.7 실제 native runner 대조: file cache 감소, 채택 보류

2026-10-06, 동일 source `9b5064e49cb9cc0359a52dce87459d672aa19597`의 기본 arena와
`experimental-snapshot-reclaim` arena로 실제 V5 preparation → prelaunch → 두 fresh
B1 엔진 → readiness → 대국 → provider/Rules/PGN/clock 검사 → postcheck 경로를
비교했다. 4.6의 독립 재읽기 oracle을 제품 prelaunch/postcheck 인수로 재사용하지 않았다.
엔진 source `ef86138c13b2188f7ee1953eb8bed60b5fec504f`와 binary는 양쪽에 같으며,
BT4·FP32·TF32 off·HistoryFill No·4096 simulation **상한**·visits·120+1·16KiB를
고정했다. buffer reuse·notify·cache·dedup·I/O Binding·CUDA Graph·chunk 변경은 off다.

초기 한 수 메이트 진단은 흑 엔진이 평가를 실행하지 않고 백도 비루트 NN backup이 없어
기존 integration gate에서 거부됐다. GPU/Rules 실패로 해석하지 않고 원본 failed receipt·
PGN·사본을 보존했다. 제품 gate를 완화하지 않았다. 자체 Rust Rules와 독립 python-chess로
전체 분기를 확인한 세 수 메이트를 새 등록했다. FEN의 과거 이력은 `unknown_prefix`이며,
흑·백 fixture는 CPU 규칙으로 확인하고 실제 GPU 실행에는 백 메이트 fixture를 사용했다.
첫 두 선택은 모두 세 수 메이트이며 이후 수는 강제된다. 매 실행의 두 PGN 수순과 네
role/session 순서의 완료·소비 평가 수가 conditioning A0와 같아야 비교에 포함했다.

새 V5 lock은 `f658a21d0bcdd85311815eae20ae5fae4f3e7cf1fc16e6aeb6e5c54a6b08f0a7`,
27개·4,099,789,320bytes다. RTX 4050 Laptop 6GB·WSL Ubuntu·CPU2,
`memory.high=6GiB`·`memory.max=12GiB`·swap0, 실행당 480초·정리30초·전체3600초를
사전 등록했다. 제품 pair 한도900초는 유지하고 외부 소유 감독이 더 짧은480초를 적용했다.
A0/B0 conditioning 뒤 A1/B1/B2/A2/A3/B3를 실행했다. Primary T는 실제 CLI 시작부터
정상 종료·pipe drain·receipt/provider/PGN 검사·최종 cgroup 수집까지다. Primary P는
청구된 원본/private/shared file cache와 두 엔진의 Rust/ORT를 포함한 전체 `memory.peak`다.
일반 성능 실행에서 반복 GPU/PID 조사나 전체 수치 회귀 suite는 실행하지 않았다.

| 쌍 | 기준 T (s) | 파일별 hint T (s) | 기준 P (MiB) | 파일별 hint P (MiB) | T1/T0 | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 126.531 | 131.101 | 4691.426 | 3461.570 | 1.036119 | 0.737850 |
| 2 | 114.162 | 109.336 | 4029.922 | 3384.137 | 0.957723 | 0.839752 |
| 3 | 110.804 | 113.787 | 4030.230 | 3465.633 | 1.026921 | 0.859909 |

**HOLD·기본 off 유지:** T는 모두 1.05 이내지만 두 P 비율이 0.80을 초과했다.
P 감소는 26.2%/16.0%/14.0%로, 독립 준비·추론의 약31.3%를 실제 native 인수로
확대하지 않는다. Pareto 조건도 첫째·셋째 T 악화와 기준 T 범위15.727초 때문에 미충족이다.
변경하지 않은 원본 검증은 기준16.715/16.456/17.454초와 실험22.641/15.838/17.037초다.
copy는 기준42.082/43.802/40.911초와 실험56.508/44.042/48.113초이며,
prelaunch는 기준6.213/3.176/2.899초와 실험6.099/6.408/5.351초,
postcheck는 기준8.707/6.724/5.883초와 실험5.936/5.564/5.878초다.
각 게임 시작→both-ready는16.135~30.609초, first-go→종료는0.158~0.201초였다.
긴 구간은 모델 로딩·검증·복사이며 반복 isready 응답·로그 배출을 주 병목으로 단정하지 않는다.

최대64개의 기존 phase marker를 수집할 때 `memory.current/stat`를 읽은 보조 관측에서,
copy 종료 file cache는 기준3913~4575MiB와 실험약26MiB, postcheck 종료 file cache는
기준3908~4571MiB와 실험약28MiB였다. 두 엔진 ready에서 anon은 양쪽 약1051~1066MiB다.
이 endpoint를 정확한 phase peak나 heap peak로 대체하지 않는다. 관측된 감소는 file cache에
집중하며 ORT heap·VRAM·Windows 전체 commit 해결을 입증하지 않는다. VRAM/Windows
commit peak는 unknown이다. 원본 hint·전역 cache 비우기를 적용하지 않았고 warm 조건·
cgroup cache 청구의 재귀속·시간 편차를 보존한다. conditioning A0/B0의 high 이벤트는
29,949/1,841이며 비교6회는 high0이다.

8회 모두 fresh session4개와 실행당 D 완료16개·탐색 소비16개(루트6/비루트10), private
bytes/inode/readonly·Rules 메이트·흑백 엔진명·실제120+1 시계·process 및 물리 drain을
인수했다. 완료/소비 수에 startup provider 검사나 terminal traversal을 포함하지 않는다.
OOM·max·hint 오류·강제 종료·잔류 process는0이며 마지막 종료 후 GPU 사용 메모리0MiB를
확인했다. 성공한 비활성 private inputs만 소유권·pin·자식·물리 완료를 확인하고 회수했다.
초기 실패 사본과 모든 PGN·명세·로그·분석은 Git 밖 논리 경로
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-native-reclaim-20261006-v2/`와
초기 실패 경로 `pr20-native-reclaim-20261006/`에 보존했다. 이 세 수 진단은 실제 native
수명주기 대조이며 일반 대국의 처리량·기력·Elo 인수가 아니다. 제품 source·Cargo·feature·
workflow·공통 계약 변경 없이 `9b5064e`의 성공한 CPU CI를 별도 근거로 재사용한다.

### 4.8 ONNX 읽기·해시 결합: 정확성 통과, 효과 채택 보류

<a id="onnx-read-hash-ab"></a>

2026-10-06 source `f68aace26be721bd7ed2da689347b53b63f98c95`에서 독립 E 후보
`experimental-onnx-read-hash`를 검사했다. 기존 경로는 ONNX를 bounded Vec에 읽고
전체 SHA-256을 따로 계산한다. 실험은 읽는 동안 동일한 소유 바이트를 해시한다.
파일 크기 기반 예약·limit+1 growth probe·길이/전체 digest·오류 관점·ORT commit 후
직렬화 버퍼 해제를 보존한다. 추가 model-sized buffer를 제거하는 변경은 아니며 기본 off다.

`rz-eval` 기본56→57개, ONNX/contracts/새 feature91→92개와 `rz-uci`199개가 WSL
Rust1.96에서 통과했다. 증가한 한 검사는 별도의 cache 경합 복구 검사다. 같은 source의
[CPU CI](https://github.com/daejunnom/RoveZero/actions/runs/37397210012)는 Windows·Ubuntu·
bindings 세 job 모두 SUCCESS다. 앞선 `f93e75d`의
[Windows CI 실패](https://github.com/daejunnom/RoveZero/actions/runs/37396381591)는 보존한다.
새 읽기 검사와 구분되는 기존 동시 cache 게시의 lock 오류였으며, 게시 완료된 entry만
기존 전체 검증으로 재사용하도록 수정했다. 원래 오류는 bounded stderr 경고로 남기고,
미게시·조회 불가 entry는 원래 실패, 손상 entry는 검증 실패를 유지한다. 이 정확성 수정은
A/B 양쪽 공통이다. Windows 로컬 검사는 컴파일됐지만 Application Control4551로 미실행이다.

새 feature의 독립 CUDA 수치 suite는 LC0 v0.32.1/eigen 원본 protobuf 참조12개와 실제
batch1/2/4/8/16을 통과했다. 최대절대차는 logits `6.50882721e-5`, 합법 policy
`2.80141830e-6`, WDL `1.78813934e-7`이다. 기존 logits atol1e-4/rtol1e-3와
policy/WDL1e-4를 유지하며 native 대조 앞에 별도로 실행했다.

native V5 공통 arena와 같은 source의 기준/변경 UCI binary를 사용했다. 두 lock의
차이는 engine binary identity·표시명이며, 각각27개 snapshot 약4.10GB다. BT4·FP32·
TF32 off·HistoryFill No·B1·visits·4096 simulation 상한·120+1·최대256ply를 고정했다.
CPU2, high6GiB/max12GiB/swap0, 실행480초+정리30초·전체3600초와 세 수 강제 메이트를
등록했다. 64KiB I/O·rolling snapshot reclaim·buffer reuse·notify·cache·dedup·I/O Binding·
CUDA Graph는 off다. 기존3단계 private FD hint는 양쪽 동일하다. A0/B0 conditioning 이후
A1/B1/B2/A2/A3/B3로 실행했다.

T는 CLI→정상 종료·pipe drain·production receipt/provider/PGN 검사·최종 cgroup 수집,
P는 원본/private/shared file cache 청구와 두 엔진을 포함하는 whole cgroup memory.peak다.

| 쌍 | 기준 T(s) | 결합 T(s) | 기준 P(MiB) | 결합 P(MiB) | T1/T0 | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 127.204 | 133.569 | 5229.434 | 4030.148 | 1.050039 | 0.770666 |
| 2 | 110.382 | 119.024 | 4030.035 | 4029.949 | 1.078293 | 0.999979 |
| 3 | 120.525 | 124.609 | 4030.090 | 4029.910 | 1.033889 | 0.999955 |

**HOLD·기본 off:** 첫 T는 실제값에서1.05를 넘으며 반올림해 승인하지 않는다. 둘째 T와
둘째·셋째 P도 문턱을 충족하지 않는다. 모든 T가 증가했고 기준 T 범위16.821초이므로
사전 Pareto 조건도 미충족이다. 비교6회의 원본 검증16.811~19.281초, copy38.715~53.156초,
게임별 ready15.848~28.481초, go0.066~0.252초였다. 읽기·해시 내부 시간을 별도로
분해하지 않았으므로 전체 ready 차이를 제거한 scan의 인과 효과로 환산하지 않는다.
peak에 file cache 청구가 포함되며 ORT heap·VRAM·Windows 전체 commit 절약을 입증하지 않는다.

8회 모두 같은 PGN 수순·role별 완료/소비 수를 확인했다. 실행당 fresh session4개,
D 완료16/탐색 소비16(루트6/비루트10), private bytes/inode/readonly·Rules 메이트·정확한
흑백 표시명·실제120+1 시계·정상 process/물리 drain을 인수했다. OOM·max·강제 종료·잔류
process·hint 오류0, 비교6회 high0이다. conditioning high29,974/21,704는 별도로 남겼다.
마지막 GPU 관측은0MiB이며 VRAM/Windows commit peak는 unknown이다. PGN16판·등록·분석·
로그·CI 실패 자료는 `${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-onnx-read-hash-20261006/`
에 보존했다. 성공한 비활성 private inputs만 동일한 소유/완료 검사를 거쳐 회수했다.
이 수명주기 진단으로 장시간 강도·Elo·기본 batch 승격을 판정하지 않는다.

### 4.9 누적 ledger와 보류 옵션 재대조

<a id="memory-cumulative"></a>
2026-10-06 사용자 지시에 따라 이전·현재 계측을 함께 보존하는
[누적 도구](../../benches/runtime/memory_evidence.py)와
[사용법](../../benches/runtime/README.md)을 추가했다. 원시 결과·실행 명세·ledger SHA를
고정하고, 같은 옵션·모델·작업량·시간 구간·자원 정책·primary peak별로 합산한다.
소스·binary·snapshot bytes·환경 변화는 별도 epoch로 남긴다. 현재 재대조 build source는
`a3a350c26026e691fd53d8c14e6001fcce274f5c`, Rust 실행 코드와 재사용한 기본 UCI/arena는
`f68aace26be721bd7ed2da689347b53b63f98c95`다. 두 SHA 사이 소스·Cargo·feature·workflow가
같다는 재사용 근거를 보존했다. 이전 binary와 current binary를 동일 SHA로 표기하지 않는다.

`--include-ledger`로 이전 ledger를 수정하지 않고 새 ledger를 추가한다. 동일 comparison은
한 번만 세며 ID 충돌·raw SHA 변조·동일 series의 실행 재사용·작업량/출력 불일치·OOM·
취소·강제 종료를 거부한다. 같은 기준을 두 옵션이 공유한 경우 각 비교에 표시하면서
실제 실행 건수에서는 중복을 제거한다. 누적 값은 `sum(T1)/sum(T0)`와
`sum(P1)/sum(P0)`, 쌍별 범위·중앙값·기하평균·epoch별 합계를 함께 남긴다.
**P 합계는 독립 실행 peak 관측값의 합이며 동시에 필요한 메모리나 시스템 peak가 아니다.**
아래 P는 그 합계를 쌍 수로 나눈 평균이다. 버퍼 series는 RSS, 나머지는 cgroup peak다.
준비-only·독립 oracle·native를 서로 합산하지 않는다. 기존 HOLD·실패·conditioning은
유지하며, 누적 평균만으로 기존 채택 문턱이나 실패를 통과 처리하지 않는다.

실제 native는 같은 V5 lock `3c2013f74026da31db81673e5c8efacb7235eac74a3159103c5a36454d265938`,
27개·4,099,796,728bytes·동일 두 B1 engine·BT4 FP32·TF32 off·HistoryFill No·
4096 simulation 상한·visits·120+1·세 수 메이트로 진행했다. 원본·사본 검증·prelaunch·
readiness·provider/Rules/PGN·postcheck·정상 물리 drain을 그대로 포함한다.
기본/rolling/64KiB arena를 별도로 빌드해 두 옵션을 각각 한 변수로 비교했다.
A0/R0/I0 conditioning 뒤 A1/R1/I1/I2/R2/A2/R3/A3/I3를 사전 등록했다.
세 기준을 공유한 6개 비교는 독립적인 12회 실행으로 세지 않는다.
CPU2·high6GiB/max12GiB·swap0, 실행당480초+정리30초·전체3600초다.
호스트에 swap32GiB가 관측됐으나 실험 cgroup의 swap0은 유지했다.
단일 추론 버퍼 재대조는 기존 supervised helper의 전체 시간 정의와 CPU quota를 유지하고,
warm3/timed20·A1/B1/B2/A2/A3/B3·실행당600초+정리30초·전체900초로 고정했다.
CPU PoC 성능 실행이나 cloud 작업은 수행하지 않았다.

| 옵션·구간 | 쌍 / epoch | 기준 T 합계(s) | 변경 T 합계(s) | sum(T1)/sum(T0) | 기준/변경 P 평균(MiB) | sum(P1)/sum(P0) |
|---|---:|---:|---:|---:|---:|---:|
| B1 버퍼 재사용 · 단일 추론 | 6 / 2 | 111.999 | 114.454 | 1.021922 | 1833.660 / 1817.142 | 0.990992 |
| 64KiB I/O · 준비만 (이전) | 3 / 1 | 111.569 | 84.635 | 0.758585 | 4733.392 / 4027.941 | 0.850963 |
| 파일별 hint · 독립 oracle/추론 (이전) | 3 / 1 | 213.983 | 202.159 | 0.944743 | 4028.249 / 2767.227 | 0.686955 |
| 파일별 hint · 실제 native | 6 / 2 | 716.835 | 700.842 | 0.977688 | 4140.361 / 3421.758 | 0.826439 |
| 읽기·해시 결합 · 실제 native (이전) | 3 / 1 | 358.111 | 377.202 | 1.053312 | 4429.853 / 4030.003 | 0.909737 |
| 64KiB I/O · 실제 native (신규) | 3 / 1 | 365.339 | 329.735 | 0.902547 | 4030.197 / 4030.234 | 1.000009 |

**모두 HOLD·기본 off:** native hint의 누적 T는 약2.2% 감소했지만 P 감소는 약17.4%로
20% 문턱에 못 미쳤다. 이번 세 쌍 P는14.3~16.1% 감소했지만 마지막 T는 약8.2% 증가했다.
버퍼 재사용은 이번 epoch에서 T 합계가 줄었으나 이전·현재 누적 T는 약2.2% 증가,
P는 약0.9% 감소이며, 이번 마지막 쌍 T도 약12.3% 증가했다.
64KiB native의 T 합계는 약9.7% 감소했지만 P는 사실상 동일하고 세 번째 T는 약10.3%
증가했다. 이번 기준 T 범위는37.703초다.
새 비교도 사전 all-pair tradeoff/Pareto 조건을 충족하지 않았다. 준비-only의 시간 개선이나
oracle의31.3% P 감소를 native 인수로 확대하지 않는다. 미세한 P 차이를 확정 악화/개선으로
일반화하지 않으며 새 승인 문턱을 사후에 만들지 않는다.

현재 native12회와 버퍼6회는 모두 정상 완료했다. 비교에 사용한 measured15회는 high0,
OOM/max/강제 종료/잔류 process0이며 conditioning high는 별도 보존했다.
Native는 실행마다 fresh session4개, NN 완료16/소비16(루트6/비루트10), 같은 두 PGN
수순을 확인했다. Startup placement와 terminal traversal은 이 평가 수에 포함하지 않는다.
누적24개 비교는 중복 제거 후 실제45회 실행이다. 이전15개 comparison을 재포함한 검사에서
15개를 한 번만 계산했고 공유 기준3개도 실제 건수에서 한 번만 세었다.

이번 버퍼 수치 검사는 독립 LC0 참조12개·batch1/2/4/8/16·가변 B1 32회와 보존 출력 불변을
확인했다. 기존 logits atol1e-4/rtol1e-3, policy/WDL1e-4를 유지했다.
여섯 버퍼 실행의23회 출력 digest는 이전과 동일하다. GPU suite는 별도로 실행했으며
각 performance 호출 앞에 반복하지 않았다. Rust CPU eval98개, arena159개 통과,
helper14개 ignored는 구분한다. 누적 도구의 CPU 검사와 실제 CI는 해당 보고서 SHA로
별도 기록한다. API·receipt·공통 계약·엔진 기본 feature·규약과 기존 별도 문서 WIP는
변경하지 않았다.

현재 cgroup peak에는 source/private/shared file cache가 포함된다. 이벤트 endpoint의
anon/file은 phase peak나 ORT heap peak가 아니며, VRAM·Windows commit peak는 unknown이다.
후속 E 후보는 모델/session 준비의 동시 생존량과 반복 원본/사본 I/O 비용을 구분하여
선택한다. 이번 결과만으로 버퍼 재사용을 큰 모델 메모리 해결책으로 채택하지 않는다.
원시 자료·누적 JSON·명세·24판 진단 PGN은 Git 밖 논리 경로
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-memory-cumulative-20261006/`에 보존했다.
이 세 수 진단은 기력·Elo 평가가 아니다.

### 4.10 CUDA 내부 CPU arena와 session 준비 메모리

<a id="cuda-cpu-arena-ab"></a>

2026-10-06, source `9aa6fdbcadd2176a6b489a367dc030f89eb07e51`에서
`experimental-ort-cpu-arena`를 **기본 off·명시 선택**으로 추가했다.
ORT 1.22는 CUDA에서 CPU node fallback을 금지해도 내부 CPU provider를 생성한다.
[ORT 생성 경로](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/session/inference_session.cc#L1667)와
[ort rc.10 CPU 옵션 구현](https://github.com/pykeio/ort/blob/v2.0.0-rc.10/src/execution_providers/cpu.rs#L39)을 대조했다.
후자의 CPU 설정 호출은 session option의 DisableCpuMemArena이며 명시 CPU EP를 append하지 않는다.
실험은 CUDA의 내부 CPU allocator arena만 끄고 fallback 금지·CUDA 실제 배치·FP32·TF32 off·
물리 완료를 유지한다. 미컴파일 옵션과 CPU provider 선택은 거부한다.

[단일 추론 진입점](../../crates/rz-eval/examples/inference_bench.rs)의
`--b1-cpu-arena=baseline|disabled`로 같은 바이너리의 한 변수를 비교했다.
입력·출력 버퍼 재사용/cache/dedup/I/O Binding/CUDA Graph는 off다.
`load_owned_observed`는 준비 시점만 관측하는 additive C backend Rust API이며
일반 생성자는 no-op observer를 사용한다. 일반 대국의 PID/GPU 반복 조사는 추가하지 않는다.
새 옵션의 backend identity는 분리하고 기존 B1·I/O 옵션의 identity codec은 보존했다.
UCI 대국 옵션·기본 feature·receipt schema·공통 계약 revision은 변경하지 않았다.

별도 GPU 수치 검사 뒤 A1/B1/B2/A2/A3/B3를 사전 등록했다.
RTX4050 6GB·driver610.62·WSL Ubuntu·CPU quota2/기존 전체 affinity,
high6GiB/max12GiB·swap0·pids128·AS128GiB, 실행당300초+정리30초·전체900초다.
BT4 모델/runtime/입력 hash와 FP32·TF32 off·HistoryFill No를 유지했다.
Fresh process/session에서 placement probe와 warm3/timed20을 실행하고 두 arm 모두
출력 digest와 ownership 반환을 측정 구간에 포함했다. 수치 suite는 각 측정 앞에 반복하지 않았다.

T는 기존 supervised helper의 전체 elapsed로 등록 검증·모델/runtime 준비·추론·
출력/물리 완료·보고서/최종 cgroup 수집을 포함한다.
Primary P는 fresh process 전체의 VmHWM이다. 준비 완료 RSS/anon과 cgroup peak는 보조 지표다.
사전 tradeoff all-pair T≤1.05/P≤0.80 및 Pareto 조건을 유지하며, 관측 후 primary를 바꾸지 않는다.

| 쌍 | 기준/변경 T(s) | T 비율 | 기준/변경 P(MiB) | P 비율 |
|---|---:|---:|---:|---:|
| 1 | 19.831 / 17.506 | 0.882754 | 1815.863 / 1818.008 | 1.001181 |
| 2 | 18.925 / 16.197 | 0.855826 | 1818.664 / 1816.191 | 0.998640 |
| 3 | 17.495 / 17.455 | 0.997678 | 1818.555 / 1816.074 | 0.998636 |

**HOLD·기본 off:** T 합계56.252→51.158s(비율0.909437), P 평균1817.694→1816.758MiB
(비율0.999485)다. 준비 완료 RSS는 평균788.896→676.146MiB로112.750MiB(약14.3%) 낮았고,
anon은529.271→416.521MiB였다. 이 endpoint 감소를 전체 peak의20% 감소로 바꾸지 않는다.
첫 쌍 P가 소폭 높고 마지막 T 절약은 기준 T 범위2.336s보다 작아 all-pair Pareto도 미충족이다.
세 쌍의 시간 감소는 이 단일 프로세스·준비/추론 작업의 관측이며 대국 속도 개선으로 승격하지 않는다.

두 arm 모두 asset load 전·직렬화 검증 후·session commit 직후 원본 해제 전·해제 후·
provider ready·warm 후·측정 후·backend drop 후의8개 self status를 남겼다.
모든 실행에서 commit 직후 원본 해제 전까지 VmHWM이 약1.8GiB로 상승했고,
owned 원본 해제 직후 RSS는706.8125MiB 감소했다. 기존 조기 해제가 실제 동작함을 확인했으며
새로 구현한 절약으로 세지 않는다. Commit 내부의 graph/initializer/allocator별 peak는 미분해다.
Endpoint RSS/anon은 live heap·phase peak가 아니며 VRAM/Windows 전체 commit peak는 unknown이다.
보조 cgroup peak는 첫 기준4504.578MiB, 나머지1664~1667MiB였다. 공유 file cache의 청구 시점이
다르므로 그 차이를 CPU arena의 전체 RAM 절약으로 환산하거나 첫 기준을 사후 제외하지 않는다.

6회 모두 정상 종료·완료 NN20/warm3·동일23회 output digest를 확인했다.
OOM/max/high/강제 종료/잔류 process는0이며 backend identity는 arm별 고정·서로 달랐다.
독립 LC0 원본 참조12개·No/Repeat·batch1/2/4/8/16 및 가변 B1 32회/보존 출력 불변을 통과했다.
Logits 최대절대차6.5088272e-5, 합법 policy2.8014183e-6, WDL1.7881393e-7로
기존 허용오차를 유지했다. Linux CPU eval ONNX/새 feature 각각69개·all-feature99개,
UCI CUDA/batch202개·누적 도구18개·fmt/strict Clippy 통과다. 후속 ledger 검사는
양쪽 baseline·뒤바뀐 옵션을 거부하는 사례를 추가해 현재19개가 통과했다.

누적 ledger는 이전24개에 새3개를 추가해 **27개 비교·실제51회**다.
이전6개 series의 전체 누적/epoch 결과가 그대로임을 재대조했다. 공유 기준3개와
기존 실패/conditioning13개는 별도로 유지하고 다른 option/작업량/peak를 합산하지 않는다.
원시 자료·등록·수치 검사·누적 JSON·분석은 Git 밖 논리 경로
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-ort-cpu-arena-20261006/`에 보존한다.
Native 대국·PGN은 이번 단일 추론 연구에서 새로 생성하지 않았다.
후속 큰 peak 후보는 session commit 중 직렬화·graph/initializer의 동시 생존량이며,
CPU arena의 준비 완료 RSS 감소와 분리해 조사한다.

### 4.11 Rust 소유 ORT flatbuffer와 native 종료 실패

<a id="owned-ort-flatbuffer-ab"></a>

아래는 최초 실행과 수정 당시의 기록이다. 수정 후 GPU 재검사와 A/B는 §4.12에 추가한다.

2026-10-06, source `238629a25d4fda0a32200d317d806e2f29909860`에서
`experimental-ort-model`을 **기본 off·별도 연구 진입점**으로 추가했다.
직전 §4.10의 peak가 session commit 직후까지 약1.8GiB로 상승한 것을 따라,
직렬화 모델을 직접 참조해 graph/initializer 복제를 줄일 수 있는지 조사한다.
원본 BT4 graph의822 node 중 입력 전부가 initializer인 node는 없었다.
이 관측만으로 ConstantFolding 비활성화를 큰 절약 후보로 삼지 않았다.
기존 runner도 첫 엔진의 uciok 이후 두 번째 엔진에 uci를 보냈으므로
새 startup 직렬화를 구현한 효과로 세지 않는다.

[ORT 공식 문서](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html#load-ort-format-model-from-an-in-memory-byte-array)와
[고정 ort rc.10 구현](https://github.com/pykeio/ort/blob/v2.0.0-rc.10/src/session/builder/impl_commit.rs#L122)을 대조했다.
`session.use_ort_model_bytes_directly`와 `session.use_ort_model_bytes_for_initializers`는
ORT flatbuffer의 직접 참조 옵션이다. Rust 소유 Vec를 private session owner 안에
유지하고 native session을 먼저 해제한다. Direct 경로의 출력은 Rust 소유 값으로 복사하고
native Value는 함수 안에서 해제한다. I/O Binding·buffer reuse·CUDA Graph는 함께 켜지 않는다.
불확실 CUDA 완료 때 session·모델·입력을 함께 보존한다. CPU provider와
미컴파일 선택·CPU arena 동시 변경도 거부한다. 기존 ONNX 조기 해제 경로는 유지한다.

`ort_export`는 동일 CUDA runtime1.22·Level1·FP32·TF32 off에서 caller가
독점 생성한 새 FD에 파생 ORT/manifest를 기록한다. 원본 gzip/protobuf/export/ONNX와
runtime core/bundle을 검증하고 파생 모델의 hash·크기·변환 조건을 추가로 식별한다.
원본 ONNX 검증은64KiB streaming hash를 사용하며 그 파일의 전체 Vec를 추가로 만들지 않는다.
기존 `maia_check`의 독립 수치 검사를 재사용하고 `inference_bench`에는
`--b1-ort-model=baseline|direct`를 추가했다. B1/warm3/timed20·출력 digest·수명 조건을
유지하며 ledger는 실제 옵션·파생 provenance·retained bytes와 비교 방향을 확인한다.
UCI·기본 feature·receipt schema·공통 계약0.1·기본 대국 모델은 변경하지 않았다.

별도 변환은39.921s·정상 종료로 완료했다. 파생 모델은741,414,656B,
SHA256 `f616299c0f69606741b8390e3581a277ad34d03f27469bf12f9b7147662dd4b3`다.
원본 ONNX741,143,425B보다271,231B 크며 양자화나 가중치 압축을 적용하지 않았다.
변환 성공·placement probe는 새 경로의 GPU 수치/수명 인수를 대신하지 않는다.

**HOLD·GPU 인수 실패:** 이어진 독립 수치 실행은12개 No/Repeat 참조,
batch1/2/4/8/16과 가변 B1 32회를 계산한 뒤 `status=passed` 보고서를 썼지만,
프로세스 종료 중 `malloc(): unsorted double linked list corrupted`와
SIGABRT(exit -6)가 발생했다. 계산된 수치와 보고서 상태는 진단 자료로만 보존한다.
전체 정상 종료가 없으므로 수치 인수·성능 표본으로 인정하지 않는다.
이 단계의 wall52.547s와 cgroup peak4053.750MiB는 성능 A/B 값이 아니다.
CPU quota2/affinity2·high6GiB/max12GiB·swap0·pids128·AS128GiB,
실행300초+정리30초를 유지했다. High/max/OOM/강제 정리/잔류 process는0이다.
이 자료로 RAM/VRAM 부족이나 ORT 자체 결함을 원인으로 확정하지 않는다.
VRAM/Windows commit peak는 unknown이다.

코드에서 별도로 확인한 누락은 수치 검사의 `PhysicalPoll::Ready` 이후
`SingleWorker::try_shutdown`의 join을 기다리지 않고 성공을 기록한 점이다.
수정 source `dd521d79cfc2b929f519770f8b2ceca1ed63e76e`는5초의 유한 shutdown 확인을
추가했다. Run 결과와 session/TLS 파괴를 구분하고 완료 Run 오류에서도 먼저 종료를
확인한다. Contracts 없이 실행할 때에도 직접 backend를 해제한 후 보고서를 기록한다.
Pending destructor를 성공으로 승인하지 않는 CPU 회귀와 오류 전파/수명 검사를 확인했다.
**수정의 GPU 재실행은 미실행이며 이번 heap 오류의 원인과 해결 여부는 미확정**이다.
첫 실패에서 후속 GPU 작업을 중단했고 A/B 사전 등록·성능 실행·새 PGN은 생성하지 않았다.
다음 실험의 선행 조건은 수정 source를 새로 고정한 독립 수치 실행의 정상 물리 종료다.

로컬 CPU는 최초 source의 ONNX/contracts92·ORT/contracts93·all-feature99개,
shutdown 수정의 all-feature/all-target103개·contracts 없는 수치 진입점 컴파일,
strict Clippy/fmt·누적 도구21개를 통과했다. 최초 source의
[CPU CI37411071663](https://github.com/daejunnom/RoveZero/actions/runs/37411071663)는
Ubuntu·Windows·bindings 세 job SUCCESS다. 이후 SHA의 CI와 GPU 미실행을 구분한다.

기존 **27개 비교·실제51회·공유 기준3개**와7개 series/epoch 값은 그대로다.
새 수치 실패를 별도 제외 기록으로 연결해 기존13개에 더한 **14개 제외 기록**을 보존했다.
실패 시간·peak를 이전 성공 A/B의 합계에 더하지 않는다.
원시 모델·등록·종료 영수증·수치/오류 로그·누적 ledger/JSON은 Git 밖
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-ort-zero-copy-20261006/`에 둔다.
기존 비교 기준과 모든 보류 판정, 별도 문서 WIP와 작업 규약을 보존했다.

### 4.12 종료 수정 재검사와 retained-direct ORT A/B

<a id="owned-ort-recheck-ab"></a>

2026-10-06, source `f9ed02a37b3c13ced140f60f3d3dc5c3cfed70b1`의
새 release 바이너리에서 §4.11의 shutdown 수정을 다시 검사했다.
독립 CUDA 수치 검사는23.013s·exit0으로 정상 종료했다. 12개 No/Repeat 참조,
batch1/2/4/8/16·가변 B1 32회·보존 출력·잘못된 입력 거부 뒤 유효 실행을 확인했다.
Logits 최대절대6.508827209e-5·합법 policy2.801418304e-6·WDL1.788139343e-7이며
기존 허용 오차를 유지했다. `backend_shutdown_completed`와 worker `shutdown_joined`를
둘 다 요구했다. High/max/OOM/강제 정리/잔류 process는0이다. 최초 SIGABRT는 제외
기록으로 보존하며, 한 번의 정상 재검사로 heap 오류의 정확한 원인을 확정하지 않는다.

수치·정상 종료 인수 뒤 같은 바이너리의 원본 ONNX(A)와 retained-direct ORT(B)를
`A1 B1 B2 A2 A3 B3` 순서로 세 쌍 비교했다. 파생 모델·manifest와 원본/runtime hash는
§4.11과 동일하며 변환은 다시 실행하지 않았다. BT4 FP32/TF32 off·HistoryFill No·
실제 B1·warm3/timed20·출력 digest·물리 완료와 모든 다른 옵션 off를 고정했다.
CPU quota2·기존 전체 affinity·high6GiB/max12GiB/swap0·pids128·AS128GiB,
개별300초+정리30초·전체900초를 등록했고 실제 여섯 실행은154.935s에 완료했다.

T는 같은 supervisor의 입력·등록 검증부터 모델/runtime 준비, warm/Run, 출력 digest,
native 파괴, 보고서 인수와 cgroup 종료 영수증까지다. 변환·독립 수치는 별도다.
Primary P는 fresh-process VmHWM이며 cgroup/file cache 관측을 대신 쓰지 않는다.

| 쌍 | A T(s) | B T(s) | A P(MiB) | B P(MiB) | T1/T0 | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 25.287 | 24.392 | 1820.316 | 1380.152 | 0.964605 | 0.758194 |
| 2 | 20.699 | 30.817 | 1820.750 | 1380.488 | 1.488832 | 0.758198 |
| 3 | 22.426 | 27.025 | 1912.371 | 1380.418 | 1.205042 | 0.721836 |

**HOLD·기본 off:** T 합계68.412→82.234s(비율1.202033), P 평균1851.146→
1380.353MiB(비율0.745675)다. 모든 쌍의20% peak 감소 문턱은 만족하지만 두 쌍의
시간 증가가5%를 넘는다. Baseline T spread4.588s이며 all-pair Pareto도 미충족이다.
준비 완료 RSS 평균은789.724→1380.311MiB로590.587MiB 늘었다. Direct는
741,414,656B 직렬화 모델을 session 전체 수명에 유지한다. Startup peak 절약과
ready 이후 상주량 증가는 각각 판단해야 하며, 이 자료로 native 두 엔진의 메모리나
대국 성과를 승인하지 않는다. 평균 Run12.253→13.130ms도 작은 표본의 기술 통계다.
VRAM·Windows commit·phase peak·live heap은 unknown이다.

집계 source `9187c10b8efef63a5dec08a43cb41efc0b5f7fed`는 ORT arm에 명시한
비활성 `disable_cuda_cpu_arena=false`를 별도 arena 비교로 오인한 분류 오류만 고쳤다.
Offline 분석 실패 JSON을 보존하고 GPU 실행은 반복하지 않았다. 도구 회귀22개·
런타임 도구 전체34개가 통과했고 [CPU CI37414687874](https://github.com/daejunnom/RoveZero/actions/runs/37414687874)는
Ubuntu·Windows·bindings 세 job SUCCESS다. 집계 SHA와 실제 GPU source SHA를 구분한다.

기존7개 series/epoch 값과14개 제외 기록은 그대로다. 새 독립 series를 더해
**30개 비교·고유57회·공유 기준3개**가 됐다. 원시 등록·수치·각 실행·누적 ledger·분석은
Git 밖 `${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-ort-zero-copy-recheck-20261006/`에 보존한다.
UCI·기본 batch·공통 계약·대국 receipt는 그대로이며 새 대국/PGN은 생성하지 않았다.

### 4.13 Native copy ORT와 반복된 종료 실패

<a id="copied-ort-flatbuffer-ab"></a>

§4.12에서 ready RSS가 상승했으므로 source
`33f1b865ae6eb6bdac0dc108963314928e79e420`에 **별도 복사형 ORT 실험**을 추가했다.
같은 파생 모델과 export runtime을 사용하고 두 직접 참조 옵션을 모두0으로 고정한다.
Native copy가 반환된 뒤 기존 `commit_verified_model`의 성공/오류 버퍼 해제 경계를
재사용한다. Direct 경로의 backing bytes는 계속 session 전체 수명 동안 유지한다.
[공개 config 계약](https://github.com/microsoft/onnxruntime/blob/v1.22.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L102)과
[ort rc.10의 일반 commit](https://github.com/pykeio/ort/blob/v2.0.0-rc.10/src/session/builder/impl_commit.rs#L136)을 따랐다.
[고정 native 구현](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/session/inference_session.cc#L1337)의
초기화 뒤 span 해제를 근거로 direct buffer 수명을 임의 단축하지 않는다.
복사형도 직렬화 중복 때문에 startup peak가 증가할 수 있어 실측 전 개선으로 보지 않는다.

컴파일 feature는 기존 `experimental-ort-model`이며 기본 off다. 독립 수치는 기존 파생
모델·manifest 인자에 `--experimental-ort-copy`를 추가한다. 성능 진입점은
`--b1-copied-ort=baseline|copied`, report kind는 `b1_copied_ort_fixed_work`,
ledger option은 `copied-ort-flatbuffer`다. 원본/직접 참조/복사형의 identity를 구분한다.
집계기는 실제 mode·출처·retained bytes0·다른 옵션 off를 검증하며 이 series를
§4.12와 합산하지 않는다. 포함 ledger의 상대 `derived_manifest` 경로도 해당
ledger 디렉터리 기준으로 정규화해 기존 자료의 위치를 잃지 않게 했다.

**HOLD·GPU 인수 실패:** 복사형의 첫 독립 수치 실행은12개 No/Repeat 참조와
batch1/2/4/8/16·가변 B1 32회를 계산했고 `backend_shutdown_completed=true`·
worker `shutdown_joined=true` 보고서를 쓴 뒤, 프로세스 종료 중 동일
`malloc(): unsorted double linked list corrupted`와 SIGABRT(exit -6)로 실패했다.
전체59.310s·cgroup peak5351.059MiB이며 high/max/OOM·강제 정리·잔류 process는0이다.
Logits 최대절대6.508827209e-5·합법 policy2.801418304e-6·WDL1.788139343e-7은
진단 값이다. 정상 프로세스 종료가 없으므로 수치/수명 성공이나 성능 표본으로 세지 않는다.
Join 수정만으로 native heap 오류가 해결됐다고 판단하지 않는다. 이 실패가 모델
버퍼·ORT·CUDA·worker·전역 종료 중 어디서 시작됐는지는 unknown이며 OOM으로 분류하지 않는다.

첫 실패에서 후속 GPU 작업을 중단했다. 복사형 성능 A/B 사전 등록·실행·대국/PGN은 없다.
실험 코드는 기본 off로 보존하고 다음 GPU 실행 전에 native 종료 경로와 손상 지점을
좁히는 독립 진단을 준비한다. 강제 `_exit`나 buffer leak로 정상 종료를 꾸미지 않는다.
로컬 all-feature/all-target103·ONNX/contracts96개, contracts 없는 ORT examples check,
strict Clippy/fmt·런타임 도구36개가 통과했다. [33f1b86 CPU CI37415577209](https://github.com/daejunnom/RoveZero/actions/runs/37415577209)는
Ubuntu·Windows·bindings 세 job SUCCESS지만 GPU 실패를 대체하지 않는다.

누적 **30개 비교·고유57회·공유 기준3개**와8개 series/epoch 값은 그대로다.
복사형 실패를 제외 기록에 추가해 **15개 제외 기록**을 보존했다. 실패 시간/peak는
성공 비교 합계에 더하지 않는다. 원시 source·binary/입력 pin·수치·종료 로그·실패 ledger는
Git 밖 `${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-ort-copy-20261006/`에 둔다.
규약·두 별도 문서 WIP·UCI·기본 모델·공통 계약·기존 PGN은 유지했다.

### 4.14 Native 종료 호출 스택과 allocator 진단

<a id="ort-native-shutdown-diagnostic"></a>

2026-10-06, §4.13의 **동일 source33f1b86·바이너리·BT4·파생 ORT·runtime**을
GDB에서 한 번 실행해 SIGABRT 호출 스택을 확보했다. 현재 문서 HEAD24037d7은
해당 Rust 소스·의존성·feature·workflow가 같다. Core dump와 외부 symbol 자동
다운로드를 끄고 진단 스크립트·원시 자료를 Git 밖에 두었다. 비용·학습·대국은 없다.

수치·backend/worker 종료 보고 뒤 `exit`의 callback이
`libcudnn_engines_precompiled.so.9`에서 `calloc`을 호출할 때 glibc가 heap 손상을
발견했다. 이는 **오류를 감지한 종료 위치**이며 처음 잘못 쓴 메모리나 잘못 해제한
객체의 위치를 확정하지 않는다. 해당 스택에는 `ReleaseEnv`가 나타나지 않는다.
모델 버퍼·native allocator·라이브러리 전역/TLS의 책임을 아직 확정하지 않았다.

| 독립 진단 | 결과 | 확인한 경계 |
|---|---|---|
| 기존 contracts 포함 바이너리 + GDB | SIGABRT, 정상 exit 아님 | worker join 뒤 cuDNN 종료 callback의 `calloc`에서 감지 |
| contracts 없는 example + GDB | 같은 종료 스택의 SIGABRT | main에서 Run/drop해도 재현; worker 이동은 재현의 필수 조건 아님 |
| 기존 바이너리 + tcache0/perturb165 + GDB | 수치·worker 종료·exit0 | 두 allocator 조건을 바꾼 진단; 성능 인수 아님 |
| 기존 바이너리 + tcache0만 + GDB | 수치·worker 종료·exit0 | 한 allocator 조건의 별도 진단; 원인 해결·기본 설정 채택 아님 |

contracts 없는 빌드는 main의12개 No/Repeat 참조·batch1/2/4/8/16·가변 B1 32회와
잘못된 입력 거부/후속 유효 실행을 유지하되 네 worker 계약 요청은 실행하지 않는다.
따라서 전체 workload가 동일한 성능 비교로 보지 않는다. 네 실행의 logits 최대절대
6.508827209e-5·policy2.801418304e-6·WDL1.788139343e-7은 기존 오차 범위다.
앞의 두 실행은 정상 exit가 없어 GPU 수치/수명 인수 실패다. 뒤의 두 실행은 정상 종료한
진단이지만, 변경된 allocator 환경을 제품이나 기존 GPU 인수로 대체하지 않는다.

`glibc.malloc.tcache_count=0`은 thread cache를 끄며 `glibc.malloc.perturb=165`는
할당/해제 메모리의 내용에 영향을 준다. [glibc 공식 계약](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html)을
참고해 **해당 자식 환경에만** 적용했다. Heap 배치·시점 변화가 잠복 오류를 드러내지
않았을 가능성을 남긴다. 두 정상 진단으로 portable한 수정·메모리/속도 개선·안전한
기본 승격을 주장하지 않는다. Session leak·강제 `_exit`·완료 전 버퍼 해제는 없다.

추가로 정확한 NVIDIA 의존16개만 `dlopen`하고 끝내는 작은 native 검사는 exit0이었다.
이는 모델/session/GPU 연산을 실행하지 않은 경계 검사다. ORT provider 초기화를
생략하고19개를 직접 `dlopen`한 최초 fixture는 CUDA provider의 constructor에서
SIGSEGV가 났다. 이 setup은 제품 초기화 순서와 달라 엔진/cuDNN 종료 오류의 재현으로
취급하지 않는다. 실패·GDB 로그를 보존했다. 짧은 loader 로그는 capture close 후
marker를 재확인했으며, 중간 분류와 후속 확인을 별도 자료로 남겼다.

[ort #609](https://github.com/pykeio/ort/issues/609)와 [수정 #610](https://github.com/pykeio/ort/pull/610)은
계산 뒤 native 종료 heap 오류를 다루지만 rc13/ORT1.27의 global environment 해제
조건이다. 현재 rc10/ORT1.22와 관측 스택이 달라 같은 원인으로 보지 않으며 runtime이나
의존성을 바꾸지 않았다. [고정 rc10 환경 구현](https://github.com/pykeio/ort/blob/v2.0.0-rc.10/src/environment.rs)을
함께 확인했다. 첫 손상 지점과 배포 가능한 수정은 여전히 **unknown·HOLD**다.

각 모델 진단은 CPU quota2·affinity 두 core·high6GiB/max12GiB·swap0·pids128·
AS128GiB, 300초+정리30초와 로그/core 상한을 사용했다. Loader 진단은60초+정리30초다.
High/max/OOM·강제 정리·소유 잔류는0이며, VRAM peak와 Windows commit은 unknown이다.
진단의 시간/peak는 성능 분모·분자에 넣지 않는다. 새로운 성능 A/B·대국은 실행하지 않았다.

집계 source `cfb99f60602a5d34e0706d9abf6d9ec68577d10f`는 명시적
`diagnostic_only=true`/`performance_measurement=false`를 `accepted=true`·exit0일
때에도 거부한다. 성공한 진단은 제외 근거로 보존한다. WSL Python3.12 도구38개가
통과했다. 최초 Windows 기본 Python 실행은 기존 `Path.is_junction`의3.12 요구로
35개 오류가 나 실패로 남겼으며 검사 성공에 포함하지 않는다. Python 요건을 낮추거나
link 검사를 우회하지 않았다. 기본 인터프리터를 실제 조회한 버전은3.10.11이며,
진단 초안의3.11 표기를 정정했다. 실패 분류와3.12 재검사 결과는 같다.
[cfb99f6 CPU CI37418772386](https://github.com/daejunnom/RoveZero/actions/runs/37418772386)는
Ubuntu·Windows·bindings 세 job SUCCESS로 직접 확인했으며 GPU heap 문제의 해결
증거로 쓰지 않는다.

새 ledger로 전부 다시 계산해 기존 **30개 비교·고유57회·공유 기준3개·8개 series/epoch**가
정확히 같음을 확인했다. 진단 collection 한 개만 추가해 **제외16개**다. 원시 source/
binary/helper pin·호출 스택·수치·자원·정상/실패 종료·집계 확인은
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-ort-shutdown-diagnostic-20261006/`에
보존한다. 기존 실패·HOLD를 지우지 않으며 규약과 별도 문서 WIP는 수정하지 않는다.

### 4.15 Free-list 검사와 독립 native 경계

<a id="ort-heap-first-write-diagnostic"></a>

§4.14와 같은 source33f1b86·복사형 바이너리·모델·runtime을 유지하고, 실제 glibc
2.39-0ubuntu8.9의 debug symbol을 사용했다. `malloc_printerr`에서 확인한 오류는
`malloc(): unsorted double linked list corrupted`이며 cuDNN 종료 callback의
9448B 할당 처리에서 감지됐다. [고정 glibc 구현](https://github.com/bminor/glibc/blob/glibc-2.39/malloc/malloc.c)의
free-list 연결 검사에 해당하지만 **첫 잘못된 write/free와 책임 객체는 여전히 unknown**이다.
감지 시점의 stack이나 allocator 옵션으로 원인 수리를 주장하지 않는다.

| 독립 조건 | 실제 결과 | 해석 한계 |
|---|---|---|
| 원래 바이너리의 종료 시 free-list watchpoint | 전체 수치·worker join·exit0 | 관측이 배치/시점을 바꿨을 수 있음; 각 marker에서 처음8개 chunk만 관측 |
| Rust/ORT/모델 없는 cuDNN·cuBLAS handle 생성/파괴·CUDA sync | exit0 | 신경망 Run이나 제품 수명 전체를 검사하지 않음 |
| 위 handle fixture + Memcheck | 첫 cudnnCreate 중 exit97 | WSL driver의 미추적 주소 read와 unhandled ioctl 경고; 원래 heap 손상 재현이나 driver 결함 확정이 아님 |
| Native ORT C API22·복사형 모델·zero B1, GLOBAL / LOCAL | 각각 CUDA Run·명시적 native 해제·exit0 | CUDA node687/CPU node0 확인; 독립 참조·가변 입력·Rust 전체 경로 대조는 없음 |
| 원래 바이너리의16개 bootstrap dlopen을 LOCAL로 변경 | 전체 수치·worker join 뒤 실제 SIGABRT | 심볼 공개 범위만 바꾸는 수정으로 해결되지 않음 |

C API fixture는 [고정 ORT1.22 header](https://github.com/microsoft/onnxruntime/blob/v1.22.0/include/onnxruntime/core/session/onnxruntime_c_api.h)를
사용하고 두 직접 참조 설정0·FP32/TF32 off·CPU fallback 금지를 유지했다. GDB scope
실험은 native C fixture와 구분한다. 제품 loader의 GLOBAL·process lifetime 정책은
바꾸지 않았다. Memcheck의 [공식 검사 범위](https://valgrind.org/docs/manual/mc-manual.html)를
참고하되 WSL 경고를 무시하는 blanket suppression이나 안전 경계 우회는 적용하지 않았다.

총9개 진단 시도 중 setup/기록 실패도 보존했다. 최초 chunk wrapper는 capture를 두 번
닫아 최종 영수증 생성에 실패했으므로 원시 수치/스택만 보존하고 자원·최종 정리 계수는
unknown이다. Scope v1의 override0은 LOCAL 실행으로 세지 않는다. V2는 합법적인
`dlopen(NULL)`에서 debugger 예외로 멈췄으며 원시 helper의 SIGABRT 표기는 잘못된
분류다. 후속 별도 correction에서 setup failure로 정정했다. V3는 NULL을 처리하고
16개 변경·실제 SignalEvent(SIGABRT)를 관측했다. 실패 원본을 덮어쓰지 않았다.

유효한7개 완료 영수증은 high/max/OOM·강제 정리·소유 잔류0이다. 모델 진단은
CPU2/affinity2·high6GiB/max12GiB/swap0·pids128·AS128GiB·300초+정리30초,
handle fixture는180초+정리30초였다. VRAM peak·Windows commit은 unknown이다.
성공한 작은 fixture·관측으로 정상 종료한 실행·setup failure를 모두 성능 자료에서 제외한다.
복사형 GPU 인수와 배포 가능한 수정은 **HOLD**이며 새 성능 A/B·대국은 없다.

기존 **30개 비교·고유57회·공유 기준3개·8개 series/epoch**가 재집계에서 정확히 같았다.
진단 collection 한 개를 추가해 **제외17개**다. 독립 실행 peak 합계를 시스템 동시 peak로
해석하지 않는다. Helper·등록·원시 로그/수치·scope correction·native fixture·누적 확인은
Git 밖 `${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-ort-heap-first-write-20261006/`에
보존한다. 진단 스크립트를 제품 workspace나 반복 대국 준비에 추가하지 않았다.

### 4.16 Raw cache hit의 불변 host head 공유

<a id="raw-cache-shared-heads"></a>

Source `db0c3abecaa7460eb38cd19d141df0be609c2025`에서 `rz-eval/raw_cache.rs`의
hit가 매번 `RawOutput`의 두 Vec를 복제하던 경로를 `Arc<RawOutput>` 공유로 바꿨다.
고정 policy1858+WDL3의 **7444B payload 복사와 두 Vec 할당**을 hit마다 제거한다.
이는 코드의 할당 경계이며 전체 RSS/VRAM 감소 실측이 아니다. 합법 policy 투영·입력
준비·새 물리 평가 실행의 비용은 남아 있다.

Stage는 worker raw를 한 번 독립 복제해 immutable owner에 저장한다. GPU/pooled buffer를
직접 공유하지 않는다. Hit가 owner를 pin한 뒤 cache mutex 밖에서 기존 policy/WDL을
생성하므로 clear/eviction이 진행 중인 hit를 무효화하지 않는다. 마지막 owner가 해제되면
raw payload도 해제된다. Retained-byte charge에 별도 RawOutput descriptor와 Arc
counter 공간을 추가했다. Stage의 Arc 할당·hit의 atomic 참조 비용과 추가 owner 메모리를
절약분과 함께 A/B해야 한다. Cache 밖에서 살아 있는 hit는 진행 중 요청의 임시 수명이다.

새 회귀는 실제 Rules·projection·stage/accept 경로에서 두 hit의 owner 공유, 동시 clear,
policy/WDL 일치와 마지막 owner 해제를 확인한다. WSL의 eval all-feature/all-target
**104개**, raw-cache 최소 feature 회귀**1개**, UCI 실제 experimental pipeline**1개**와
strict Clippy·workspace fmt가 통과했다. 최초 잘못된 `LegalPolicy.values` 검사와
format 실패를 수정한 뒤 재검사했다. [db0c3ab CPU CI37422644706](https://github.com/daejunnom/RoveZero/actions/runs/37422644706)의
Ubuntu·Windows·bindings 세 job도 SUCCESS로 직접 확인했다.

**효과 미측정·HOLD:** 이번 GPU 진단은 이전 source33f1b86이며 이 raw-cache 변경의
독립 GPU 수치나 동일 작업량 메모리/시간 A/B가 아니다. Default cache는 계속 off이고
공통 계약·receipt schema·UCI 옵션·ORT loader/allocator·탐색 정책은 바꾸지 않았다.
기존 수치·계측을 새 source의 성능으로 재사용하지 않는다. 현재104개 검사는 CPU/mock
정확성 증거다. 채택은 §2의 전체 시간·사전 primary peak·의미/수명 문턱을 별도로 따른다.

### 4.17 Raw cache 공유의 CPU 할당 추적·고정 작업 A/B

<a id="raw-cache-shared-heads-cpu-ab"></a>

§4.16의 후속 검증은 baseline `0566fb90b3d2bfe9da9d085b097c9dfc019b7e4e`와
variant `7d6311cde5e16eb07b4fdd5085ea9176ff74b23d`를 같은 독립 harness로 빌드했다.
실제 Rules의 여섯 상태·No history·결정적 raw logits/WDL을 사용하고, 여섯 seed 평가 후
**24,576회 exact raw cache hit**를 수행했다. 신경망·GPU·search/D scheduler는 실행하지
않았다. 다른 합법 수 투영·반복 상태·counter가 포함되며 출력 digest와 원래 수명 검사를
양쪽에서 대조했다. 초기 fixture는 서로 다른 Rules 이력이 같은 No 입력이 되는 사례를
miss로 잘못 기대했다. 제품의 exact hit가 맞았으며 실패를 보존하고 구별되는 counter로
fixture를 수정해 새 digest의 별도 조건으로 등록했다.

제품 allocator는 바꾸지 않았다. Git 밖의 allocation trace는
[System](https://doc.rust-lang.org/std/alloc/struct.System.html)에 그대로 위임하고
[GlobalAlloc의 재진입·unwind 제한](https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html)을
지키는 작은 관측 경계다. 이 관측이 시간을 교란하므로 trace와 **관측 hook이 없는 성능
바이너리**를 따로 빌드했다. 다음 값은 trace의 host 요청 할당이며 native/VRAM 실측이 아니다.

| 24,576 hit의 trace | baseline | 공유 variant |
|---|---:|---:|
| 할당 요청 수 | 245,760 | 196,608 |
| 누적 요청 bytes | 939,622,400 | 756,678,656 |
| 관측 phase의 새 live 할당 peak bytes | 37,616 | 30,408 |
| 여섯 entry의 cache retained charge bytes | 225,096 | 225,240 |

Hit당 **두 할당·7444B 요청** 제거를 관측했다. 누적 **182,943,744B(174.47MiB)**는
할당 요청 총량의 차이이며 RAM/VRAM 절약량이 아니다. 추가 owner charge는 전체144B다.
관측 phase 종료의 잔류는0이고 정상 cache/owner 해제를 확인했다.

성능은 세 쌍을 `A1 B1 B2 A2 A3 B3` 순서로 사전 등록했다. CPU2/affinity2,
high=max512MiB/swap0·pids128·AS2GiB·실행120초+정리30초·전체900초다.
전체 시간은 자식 startup·작업·종료·report 인수·최종 cgroup 영수증까지 포함하고,
build·allocation trace·독립 GPU gate를 제외한다. Primary peak는 전체 cgroup `memory.peak`다.

| 쌍 | A 전체 초 | B 전체 초 | T1/T0 | A peak bytes | B peak bytes | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.675944 | 0.560630 | 0.82940 | 1,335,296 | 1,335,296 | 1.00000 |
| 2 | 0.587666 | 0.686395 | 1.16800 | 1,335,296 | 1,589,248 | 1.19018 |
| 3 | 0.692256 | 0.582547 | 0.84152 | 1,335,296 | 1,593,344 | 1.19325 |

누적 시간비는 **0.93543**, 실행 peak 관측 합계비는 **1.12781**이다. Baseline 시간
spread는0.104591초다. 짧은 실행·환경 편차가 크고 두 번째 쌍이 더 느리며 primary peak도
개선되지 않아 §2의 상충/Pareto 문턱 모두 실패한다. **HOLD·기본 cache off**를 유지한다.
분모가 다른 GPU series와 시간을 합쳐 개선율을 만들지 않고, peak 합계를 동시 사용량으로
해석하지 않는다. 여섯 CPU 실행의 high/max/OOM·강제 정리·소유 잔류는0이다.

`rules_maia_check --experimental-raw-cache`는 실제 Rules→원본 BT4 ONNX FP32 B1→D
fresh 결과와 요청 ID가 다른 두 cache replay의 합법 policy/WDL·물리 실행 없음·예약0을
대조하는 선택적 수치 진입점이다. No/Repeat 전체 검사를 계획했으나 GPU 인수는 실패했다.
최초 잘못된 runtime cache 부모 root는 session 이전에 거부됐다. 별도 수정 조건은 No의
fresh1개와 replay2개를 관측한 뒤 검사 예제의 ID `1→64→65→2` 역행을 실제 계약이 거부했다.
이어 `cudaFreeHost(p)`에서 **CUDA failure4 driver shutting down·SIGABRT**가 발생했다.
이 오류를 앞선 malloc unsorted/free-list 실패의 재현으로 분류하지 않는다. 공통 원인은
unknown이며 원본 ONNX gate로 복사형 ORT 수명 문제를 해결했다고 주장하지 않는다.

관측된 한 case의 policy/WDL 최대 오차는 각각 `5.2154e-7`/`1.1921e-7`이지만 부분 자료다.
실패 GPU 실행의 peak는4,724,609,024B, high/max/OOM·강제 정리·소유 잔류는0이다.
VRAM peak·Windows commit은 unknown이다. 첫 CUDA 실패 뒤 후속 GPU 실행·A/B·대국을
중단했다. Numerical report·drain/예약0으로 native 정상 종료를 대체하지 않는다.

후속 source `c74735dbfddafac8a4897a9c0950f012f4bb0637`은 예제 ID를 연속 세 개씩
배정하고 default stream은 유지했다. 두 stream을 실제 admission adapter로 검사하는
CPU 회귀를 추가했다. Eval all-feature/all-target **105개**, 최소 onnx/contracts 예제1개,
strict example Clippy·workspace fmt가 통과했다. 이는 ID 수정과 CPU 정확성 증거이며
GPU를 다시 실행한 결과가 아니다. 성능 바이너리/기록은 계속7d6311c로 식별한다.
[7d6311c CI37425154376](https://github.com/daejunnom/RoveZero/actions/runs/37425154376)의
Ubuntu·Windows·bindings 세 job SUCCESS와 후속 source 검사 결과를 구별한다.
[c74735d CI37427203372](https://github.com/daejunnom/RoveZero/actions/runs/37427203372)도
같은 세 job의 SUCCESS를 직접 확인했다. 이 workflow는 CPU 검사이며 GPU native 종료
실패를 해결하거나 인수하지 않는다.

기존8개 series의 집계가 정확히 같은지 확인하고 새 CPU series를 추가했다. 누적은
**33개 비교·고유63회·공유 기준3개·9개 series/epoch·제외19개**다. 초기 fixture 실패와
trace/GPU 진단 collection을 제외해 실패·관측을 성능 성공으로 세지 않는다. 등록·바이너리
hash·raw receipts·correction·누적 ledger는 Git 밖
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-raw-cache-ownership-ab-20261006-v2/`에
보존한다. 새 CPU harness/allocator trace를 제품·CI·일반 대국 준비에 추가하지 않았다.

### 4.18 Raw cache gate의 native 종료 수정·긴 CPU A/B

<a id="raw-cache-native-shutdown-long-ab"></a>

§4.17의 GPU 오류 뒤 실제 종료 경계를 대조했다. `rules_maia_check`의 request drain은
예약과 요청이 비었음을 확인하지만 native session·worker thread/TLS 파괴를 기다리지
않았다. Source `b39dc4aa4d34719134456967d2b9b36b8975388a`는 기존
`NativeWorkerOwner::try_shutdown()`을 **수치 검사 예제에 연결**하고 최대5초 안의 join을
별도 영수증으로 남긴다. 성공·실패한 work 모두 같은 정리를 거치며 원래 오류를 보존한다.
종료/진단 회수가 미확정이면 owner를 보존한다. Native 파괴 전 작은 stderr checkpoint를
남겨 C++ abort가 모든 진행 근거를 지우지 않게 했다. 반복 PID/GPU 조사기는 추가하지 않았다.

제품 UCI의 `finish_until`은 이미 이 join을 확인하고 있었다. UCI·backend·allocator·
공통 계약·기본 feature는 이번 수정에서 바꾸지 않았다. 고정
[ORT1.22 pinned allocator](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/providers/cuda/cuda_allocator.cc#L90)의
`cudaFreeHost` 오류를 숨기는 수정도 아니다. 이전 driver-shutdown SIGABRT와 복사형
ORT의 malloc/free-list 손상이 같은 원인인지 아직 확인하지 않았다.

CPU 회귀는 실제 ContractEvaluator/NativeWorkerOwner 경계에서 정상 요청과 물리 실패
요청의 native destructor를 지연했다. 예약0 뒤에도 join은 Pending이며, 정리를 허용한
뒤에만 완료된다. 원래 실패 진단도 보존한다. Eval all-feature/all-target **106개**,
최소 onnx/contracts 예제 **2개**, strict example Clippy·workspace fmt·release gate build가
통과했다. [b39dc4a CI37429555483](https://github.com/daejunnom/RoveZero/actions/runs/37429555483)의
Ubuntu·Windows·bindings 세 job SUCCESS를 직접 확인했다. CPU CI와 다음 실제 GPU 근거는
각각 보존한다.

후속 오류 경로 검토에서는 cache 통계 조회/불일치가 profile 영수증과 owner 보존 전에
반환될 수 있음을 확인했다. 통계 결과를 `Result`로 수집하고 오류를 profile에 함께 기록한
뒤 원래 work·drain·native join 오류부터 반환하도록 보완했다. 이 보완도 수치 예제에만
적용했다. Eval106개·최소 예제2개·strict Clippy/fmt·release build로 CPU 검사했으며,
초기 formatting check 실패와 후속 수정은 별도 로그로 보존한다. 다음 GPU 근거의 실제
실행 source는 계속b39dc4a이며 통계 영수증 보완 뒤 GPU 재실행으로 표시하지 않는다.

수치 gate는 같은 원본 BT4 ONNX·FP32·TF32 off·B1·RTX4050에서 No/Repeat를 실행했다.
**Fresh 요청 출력12개·cache replay24개**, 두 profile의 native join·process exit0을
확인했다. Provider 준비 probe 두 개는 요청 출력12개와 별개다. 입력 최대 오차0,
합법 policy 최대 오차`2.8014183044433594e-6`, WDL 최대 오차`1.7881393432617188e-7`이다.
각 profile의 CUDA placement687node/CPU fallback0을 확인했다. 전체21.285788초,
cgroup peak5,087,748,096B이며 high/max/OOM·강제 정리·소유 잔류는0이다.

실패 work의 정리는 Git 밖에 원본 reference의 사본을 만들고 두 번째 No case의 WDL을
의도적으로 틀린 유효 확률 벡터로 바꿔 별도 검사했다. 원본 참조는 보존했다. 제품 report는
**failed·exit1**, 원래 `numerical mismatch: absolute error=0.831609234213829`를 유지하면서
native join을 완료했다. Validation accepted는 **예상 실패와 정리의 확인**이며 신경망
수치 통과가 아니다. 전체19.866445초, peak4,829,409,280B, high/max/OOM·강제 정리·잔류0이다.
이 두 실행에서 native abort는 재현되지 않았다. VRAM peak·Windows commit peak는 unknown이다.

이 gate는 Rules 입력·합법 policy/WDL·cache 재사용·native 정상/실패 정리의 근거다.
Raw logits와 batch1/2/4/8/16 독립 수치 suite는 별도다. 전체 시간/peak를 이전 실패한
부분 실행과 나누어 개선율로 만들지 않는다. GPU 성능 A/B·대국·Elo 인수는 없다.
**복사형 ORT heap 문제는 별도 HOLD이며 이번에 수정하거나 다시 검사하지 않았다.**

짧은 CPU 표본의 startup 편차를 줄이기 위해 같은 baseline0566fb9/variant7d6311c·여섯
Rules 상태·결정적 raw head에서 **1,572,864 cache hit**의 새 series를 사전 등록했다.
공통 harness의 repeat만4096→262144로 늘렸다. 측정에는 allocator hook·NN·GPU·search/D
없음, 출력 digest·seed6개·모든 작업은 양쪽 동일이다. 이전24,576-hit series와 조건/epoch를
분리한다. Source7d6311c 이후 b39dc4a까지 제품 Rust/Cargo 변경이 수치 gate 예제에만 있음을
확인했지만 성능 source 식별은 계속7d6311c다. §4.17 할당 trace를 새 RAM/VRAM 절약량으로
환산하거나 반복 배수만큼 새 실측으로 확대하지 않는다.

순서는 `A1 B1 B2 A2 A3 B3`, CPU2/affinity2·high=max512MiB/swap0·pids128·AS2GiB·
실행120초+정리30초·전체900초다. Primary peak는 전체 cgroup `memory.peak`이고,
시간은 자식 startup·고정 작업·종료·report·cgroup 영수증까지다.

| 쌍 | A 전체 초 | B 전체 초 | T1/T0 | A peak bytes | B peak bytes | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 34.198197 | 34.795707 | 1.017472 | 1,597,440 | 1,589,248 | 0.994872 |
| 2 | 34.125705 | 33.740120 | 0.988701 | 1,593,344 | 1,335,296 | 0.838046 |
| 3 | 34.217314 | 34.867408 | 1.018999 | 1,523,712 | 1,593,344 | 1.045699 |

누적 시간은**102.541216→103.403235초**, 비율**1.008407**이다. 실행 peak 관측 평균은
**1,571,498.67→1,505,962.67B**, 관측 합계비**0.958297**이다. Baseline 시간 spread는
0.091610초다. 시간은 약0.84% 늘고 평균 관측 peak는 약4.17% 줄었지만, 세 번째 쌍은
peak가 늘고 두 쌍은 느려졌다. All-pair 상충/Pareto 문턱을 모두 충족하지 않아
**HOLD·기본 cache off**다. 여섯 실행은 exit0·같은 출력 digest·high/max/OOM/강제 정리/잔류0이다.
이 작은 CPU fixture의 peak를 실제 GPU 엔진 메모리로 확대하지 않으며 독립 peak 합계를
동시 메모리 사용량으로 해석하지 않는다.

기존9개 series 집계의 완전 일치를 확인하고 긴 CPU3쌍만 추가했다. 누적은
**36개 비교·고유69회·공유 기준3개·10개 series/epoch·제외20개 collection**이다.
정상 GPU/예상 실패 GPU와 setup·compile 수정은 별도 제외 collection으로 보존한다.
Windows worktree Git metadata를 WSL에서 읽은 preflight 실패는 Windows Git의 source 확인과
WSL byte hash로 바로잡았으며 `.git`은 바꾸지 않았다. 초기 CPU 회귀의 타입 오류와 집계
helper의 schema/non-JSON evidence 오류도 correction과 함께 보존했다. 집계기·제품 guard·
원시 측정값은 바꾸지 않았다. 원시 등록·hash·report·receipt·CI·누적 동일성 근거는 Git 밖
`${ARTIFACT_ROOT}/reports/coordinator-integration/pr20-native-cache-shutdown-20261006/`에 둔다.
별도 성능 harness는 제품·CI·일반 대국 준비에 추가하지 않았다.

## 5. 후속 구현·검증 순서

할당/복사 제거 → 저장 밀도 → bounded scratch/arena/pool → 임계 구역/완료 통지 →
allocator 한 변수 비교 → 필요한 GPU 비동기 경계 순서를 초기 제안으로 둔다.
각 단계의 실제 우선순위는 profile로 바꿀 수 있으나, 후보를 한꺼번에 켜서 원인을 잃지 않는다.

| 관측 층위 | 최소 기록 |
|---|---|
| 타입·할당 | 실제 target size_of/align_of, 할당 수·크기·수명, live/retained/capacity, cross-thread free |
| CPU | cycles/cache/branch, copy bytes, allocator 시간, lock 보유/대기, queue 대기 |
| GPU | H2D/D2H·kernel·fence·완료→소비, pinned RAM, used/reserved VRAM, batch histogram |
| 전체 | startup와 warm steady state 분리, request→backup P50/P95/P99, peak RSS/PSS/VRAM의 범위 |
| 탐색 | 완료 방문·fresh NN 입력 수·physical batch·raw hit·terminal backup·discarded 결과 |
| 수명 | 정상/취소/root 교체/late/duplicate/poison/OOM/quarantine에서 자원과 원래 오류 |

같은 방문 수에서 작업당 bytes/time, 같은 시간에서 유효 방문, 같은 메모리 한도에서 가능한
작업량을 분리한다. 최적화 후 방문 수가 늘어 전체 peak가 커지는 것을 숨기지 않는다.
allocator allocated bytes·RSS·PSS·전체 장치 VRAM은 서로 대체 지표가 아니다.
시간 비교에 host 준비·cache 복원·전송·queue·GPU·검증·backup·필수 종료를 포함하고,
startup 비교는 별도 고정 workload로 보고한다. sampling peak는 짧은 spike를 놓칠 수 있으므로
관측 해상도와 unknown을 기록한다.

기존 독립 Rules·인코딩·고정 방문 witness·요청 수명 검사를 우선 사용한다. 단순 문구 수정마다
새 영구 fixture나 validator-on-validator 체계를 추가하지 않는다. 필요한 작은 회귀만 추가하고
Miri/Loom 등은 지원 범위에서 보조로 사용한다. native CUDA 전체의 안전성을 증명한다고 하지 않는다.

후속 PR은 변경 기법 ID, 바뀌는 소유 경계, 기준/변경 SHA·features·flags·모델 hashes·장비,
사전 고정 workload·측정 구간·primary peak, 의미 대조, 시간/peak 원시 요약과 불확실성,
채택/보류 이유, 실패·미실행·rollback 방법을 짧게 남긴다. 같은 workload의 A/B 순서를 교차하고
원시 자료는 저장소 밖에 보존한다. 대국 강도는 [평가 계약](../EVALUATION-PROTOCOL.md)의
동일 시작 상태·흑백 교대·동일 벽시계/자원·holdout으로 별도 확인한다.

## 6. 문서 결정·협업 범위

이 문서 PR은 사용자 직접 요청에 따른 문서 작업이다. 다른 담당의 TASK를 인수하거나 코드
변경·훈련·GPU 비용·merge 권한을 얻지 않는다. 일반 develop 대상 규약의 이번 예외는
**PR #20의 feature/lc0-model-evaluation을 base로 하는 stacked 문서 PR**이다.
이유는 현재 구현을 기준으로 후속 규칙만 검토하기 위해서다. 보완은 base SHA 고정·문서만의
compare·원격 head 확인이며, #20 변경/병합 시 차이를 재검토하고 명시적으로 재대상화한다.
원래 branch·규약·76개 후보와 기존 실행 증거를 보존한다. 이번 한정 예외는 이후 모든 작업의
분기 규약을 변경하지 않는다.

## 7. 출처와 확인 범위

원리는 공식 문서·논문/구현 설명에 근거하며 표의 장단점은 그 메커니즘에서 도출한 조건부
분석이다. 문서의 최신 버전과 프로젝트의 고정 toolchain은 다를 수 있다. API 채택 전에
대상 버전·MSRV·feature·라이선스를 다시 잠근다. 아래 원문 링크가 benchmark 재현을 뜻하지 않는다.

[^vec]: [Rust Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html). Capacity·재할당·clear·Box slice와 가변 저장소 계약.
[^layout]: [Rust type layout](https://doc.rust-lang.org/reference/type-layout.html). repr·padding·정렬·동적 크기 표현의 보장 범위.
[^niche]: [NonZeroU32](https://doc.rust-lang.org/std/num/type.NonZeroU32.html). Option의 크기 보장; 임의 enum에 일반화하지 않는다.
[^indirection]: [Abseil: Reducing memory indirections](https://abseil.io/fast/83). 간접 접근 감소와 실제 workload의 교환 관계.
[^soa]: [Intel data layouts](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-1/layouts.html). 배열 배치·벡터화 참고.
[^false-sharing]: [Linux false sharing](https://docs.kernel.org/kernel-hacking/false-sharing.html). Cache-line 경쟁과 perf c2c 진단.
[^hash]: [Abseil Swiss tables](https://abseil.io/about/design/swisstables), [hashbrown](https://docs.rs/hashbrown/latest/hashbrown/). Flat table의 구조 참고.
[^smallvec]: [SmallVec 1.15.1](https://docs.rs/smallvec/1.15.1/smallvec/). Inline/spill과 객체 크기 증가 가능성.
[^arrayvec]: [ArrayVec](https://docs.rs/arrayvec/latest/arrayvec/). Fixed capacity와 초과 처리.
[^uninit]: [MaybeUninit](https://doc.rust-lang.org/std/mem/union.MaybeUninit.html). 초기화·부분 실패·Drop 안전 조건.
[^cow]: [Cow](https://doc.rust-lang.org/std/borrow/enum.Cow.html). Borrowed/owned·필요 시 복사. Rc/Arc와 다른 표현.
[^arc]: [Arc](https://doc.rust-lang.org/std/sync/struct.Arc.html). 참조계수·Send/Sync·make_mut의 조건.
[^bytes]: [Bytes](https://docs.rs/bytes/latest/bytes/struct.Bytes.html). Shared backing과 view 수명.
[^bump]: [bumpalo](https://docs.rs/bumpalo/latest/bumpalo/). Arena reset과 소멸자·별도 heap의 주의점.
[^slab]: [slab](https://docs.rs/slab/latest/slab/). Index 기반 사전 할당 저장; generation/ABA는 적용 설계의 별도 책임.
[^queues]: [rtrb SPSC](https://docs.rs/rtrb/latest/rtrb/), [Crossbeam ArrayQueue](https://docs.rs/crossbeam-queue/latest/crossbeam_queue/struct.ArrayQueue.html). Producer/consumer와 bounded queue 조건.
[^atomics]: [Rust Ordering](https://doc.rust-lang.org/std/sync/atomic/enum.Ordering.html). 메모리 게시 순서; GPU fence와 다르다.
[^rcu]: [ArcSwap](https://docs.rs/arc-swap/latest/arc_swap/), [crossbeam-epoch](https://docs.rs/crossbeam-epoch/latest/crossbeam_epoch/). 읽기 공유·지연 회수.
[^seqlock]: [Linux sequence counters](https://docs.kernel.org/locking/seqlock.html), [Rust undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html). 재시도로 data race를 정당화하지 않는다.
[^mutex]: [parking_lot Mutex](https://docs.rs/parking_lot/latest/parking_lot/type.Mutex.html). Non-poisoning 등 표준 mutex와의 동작 차이.
[^ort-io]: [ORT I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html). Buffer 위치와 복사 경계.
[^ort-cuda]: [ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). Stream·Graph·동기화·지원 제약.
[^cuda]: [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html). Pinned/mapped memory·전송 overlap·지역성.
[^cuda-pool]: [CUDA memory pools](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html). Stream-ordered 재사용·used/reserved 구분.
[^planning]: [ExecuTorch memory planning](https://docs.pytorch.org/executorch/stable/compiler-memory-planning.html). Tensor liveness 기반 공유 공간의 참고.
[^pin]: [Rust Pin](https://doc.rust-lang.org/std/pin/index.html). Rust 이동 계약은 CUDA pinned page가 아니다.
[^global]: [Rust GlobalAlloc](https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html), [ORT memory tuning](https://onnxruntime.ai/docs/performance/tune-performance/memory.html). Rust/native/device allocator 경계.
[^mimalloc]: [mimalloc](https://github.com/microsoft/mimalloc). Free-list·page·지역성과 purge/retention 정책.
[^jemalloc]: [jemalloc manual](https://jemalloc.net/jemalloc.3.html). Arena·tcache·decay와 통계.
[^tcmalloc]: [TCMalloc design](https://google.github.io/tcmalloc/design.html). Per-CPU/thread cache 구조.
[^mmap]: [memmap2](https://docs.rs/memmap2/latest/memmap2/struct.MmapOptions.html). 파일 기반 mapping의 변경·잘림·수명 안전성.
[^os]: [Linux huge pages](https://docs.kernel.org/admin-guide/mm/transhuge.html), [NUMA policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html), [proc memory](https://docs.kernel.org/filesystems/proc.html). OS 메모리·관측 범위.
[^fadvise]: [Linux posix_fadvise](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html). DONTNEED는 best-effort cache hint이며 부분 page·dirty page의 실제 회수를 보장하지 않는다.
[^pgo]: [Rust PGO](https://doc.rust-lang.org/rustc/profile-guided-optimization.html), [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html). LTO·panic 전략을 구분한다.
[^pr20]: [PR #20](https://github.com/daejunnom/RoveZero/pull/20). 동적 PR 상태 대신 본문 조사 SHA를 함께 사용한다.
[^rz-tree]: [tree.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-search/src/tree.rs). Vec<Node>·Expanded(Vec<Edge>)·pending·선택/backup.
[^rz-prepared]: [rules_projection.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/rules_projection.rs). Prepared-input·history·feature별 경로.
[^rz-cache]: [raw_cache.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/raw_cache.rs). Store·profile·staged/live·정확 입력·출력 공유 후보.
[^rz-output]: [lib.rs](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/lib.rs), [output.rs](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/output.rs). 미검증 RawOutput과 검증된 head, unsafe 제한.
[^rz-worker]: [native_runtime_bridge.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/native_runtime_bridge.rs). Native owner·lease·진단·물리 수명.
[^rz-io]: [onnx_io.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/onnx_io.rs). 고정 Binding·동기 copy/fence 경로.
