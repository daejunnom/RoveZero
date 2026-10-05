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
[^pgo]: [Rust PGO](https://doc.rust-lang.org/rustc/profile-guided-optimization.html), [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html). LTO·panic 전략을 구분한다.
[^pr20]: [PR #20](https://github.com/daejunnom/RoveZero/pull/20). 동적 PR 상태 대신 본문 조사 SHA를 함께 사용한다.
[^rz-tree]: [tree.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-search/src/tree.rs). Vec<Node>·Expanded(Vec<Edge>)·pending·선택/backup.
[^rz-prepared]: [rules_projection.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/rules_projection.rs). Prepared-input·history·feature별 경로.
[^rz-cache]: [raw_cache.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/raw_cache.rs). Store·profile·staged/live·정확 입력·출력 공유 후보.
[^rz-output]: [lib.rs](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/lib.rs), [output.rs](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/output.rs). 미검증 RawOutput과 검증된 head, unsafe 제한.
[^rz-worker]: [native_runtime_bridge.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/native_runtime_bridge.rs). Native owner·lease·진단·물리 수명.
[^rz-io]: [onnx_io.rs @ f442c41](https://github.com/daejunnom/RoveZero/blob/f442c41aa6f1885d4ae06aab874420a4cd8f7062/crates/rz-eval/src/onnx_io.rs). 고정 Binding·동기 copy/fence 경로.
